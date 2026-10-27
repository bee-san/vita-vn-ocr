#!/usr/bin/env python3
"""Model B (vita-vn-ocr #126): the native-PSP training pools.

  modelb_pools.py native --out PUBLIC_DIR --private-out PRIVATE_DIR
      VOCR5's pinned real pools (eval/vocr5_ft_sources.json) cut to their native 480x272 PSP rows (meta kind "psp";
      never a PC-scaled "hires" row or a 2x view): one npz per source pool, rows and meta unchanged. The owner's
      console captures (shared/private) go to PRIVATE_DIR. Every input is checked against eval/vocr5_pools.sha256.
  modelb_pools.py console --pagepool PAGEPOOL --pages PAGES --out DIR [--splits train,guard,hval]
      console crops of the hooked corpus' PSP pages (vocr6-ingest's page pool): every whole-dialogue region (base,
      +4, +8 px) through the C line finder at native size, VOCR_LINES_GRAD then vocr_refine_lines(VOCR_REFINE_DEFAULT)
      (model B's page path), each finder line paired one to one with a verified line of the region (vertical overlap
      >= 0.5 of the smaller height, horizontal overlap; eval/README's matching rule) and kept only when it holds that
      line's whole text (at least COVER = 40% of the width of every non-blank character cell of the line record -
      the hooked corpus' lines.jsonl.zst cell_boxes / cell_chars, under --root - lies inside its columns: a cell is
      the glyph's full-width quad, and 「 or 、。 sit in one half of it, so a correct box can leave up to half a
      punctuation cell out, but never a whole glyph; a line without cells: its columns cover the line's box less a
      quarter of the line's height) and no other line's row (no merged rows), so the line's label is the crop's.
      Crops in the console's view: integer luma, dark text inverted.
      DIR/<split>/<title id>.npz (train2's shard format), DIR/record.json the counts and every file's sha256.
  modelb_pools.py rank --sources GLOB [GLOB ...] --vocab VOCAB --out RANK.json
      class counts in the labels of the given shards and the vocab in decreasing frequency (ties: vocab order;
      the blank first), the two-level head's permutation (train/head2level.py permute) for model B.

Everything read or written here is copyrighted game data or derived crops: it stays under $HERD, never committed;
record.json and RANK.json hold counts and hashes only.
"""
import argparse
import glob
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in ("synth", "eval", "train"):
    if str(ROOT / _p) not in sys.path:
        sys.path.insert(0, str(ROOT / _p))

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
V3 = HERD / "shared/artifacts/vocr5-train/pools/v3"
V3P = HERD / "shared/private/vocr5-train/pools/v3"
NATIVE = {   # source pool -> (directory, private)
    "tt": (V3, False), "tvt": (V3, False), "rd": (V3, False), "rvt": (V3, False),
    "nat_ink": (V3, False), "nat_grad": (V3, False), "nat_ref": (V3, False),
    "dntv_ink": (V3, False), "dntv_grad": (V3, False), "dntv_ref": (V3, False),
    "dnst_ink": (V3, False), "dnst_grad": (V3, False), "dnst_ref": (V3, False),
    "own_ink": (V3P, True), "own_m4": (V3P, True), "own_m8": (V3P, True), "own_grad": (V3P, True),
    "own_ref": (V3P, True),
}
GRAD, REFINE = 0, 0x1F
COVER = 0.4             # share of each character cell's width a console crop must hold
VARIANTS = ("base", "expanded-4", "expanded-8")
FINDER_FLAGS = ("-std=c99", "-O2", "-ffp-contract=off", "-DVOCR_NO_NEON", "-Wall", "-Wextra", "-Werror")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_shard(path):
    z = np.load(path)
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    labels = json.loads(str(z["labels"]))
    meta = json.loads(str(z["meta"])) if "meta" in z.files else [{}] * len(labels)
    return [data[offs[i]:offs[i + 1]].reshape(int(shapes[i][0]), int(shapes[i][1])) for i in range(len(labels))], \
        labels, meta


def write_shard(path, crops, labels, meta):
    """train2's shard format: data (uint8, every crop's rows), offsets, shapes, labels, meta (JSON strings)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    offs = np.zeros(len(crops) + 1, np.int64)
    offs[1:] = np.cumsum([c.size for c in crops])
    data = np.concatenate([np.ascontiguousarray(c, np.uint8).ravel() for c in crops]) if crops else \
        np.zeros(0, np.uint8)
    shapes = np.array([c.shape for c in crops], np.int64).reshape(-1, 2)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, data=data, offsets=offs, shapes=shapes, labels=json.dumps(labels, ensure_ascii=False),
             meta=json.dumps(meta, ensure_ascii=False))
    tmp.replace(path)
    return sha256(path)


def pinned():
    """eval/vocr5_pools.sha256: {file name: sha256}."""
    out = {}
    for ln in (ROOT / "eval" / "vocr5_pools.sha256").read_text().splitlines():
        parts = ln.split()
        if len(parts) >= 2:
            out[Path(parts[-1]).name] = parts[0]
    return out


def cmd_native(a):
    pins = pinned()
    rec = {"kind": "modelb-native-pools", "rule": "VOCR5 pinned real pools, rows with meta kind == 'psp'",
           "files": {}}
    for name, (d, private) in NATIVE.items():
        src = d / f"{name}.npz"
        want = pins.get(src.name)
        got = sha256(src)
        if want and want != got:
            raise SystemExit(f"{src}: sha256 {got} differs from eval/vocr5_pools.sha256")
        crops, labels, meta = load_shard(src)
        keep = [i for i, m in enumerate(meta) if m.get("kind") == "psp" and labels[i]]
        out = Path(a.private_out if private else a.out) / f"{name}_psp.npz"
        sha = write_shard(out, [crops[i] for i in keep], [labels[i] for i in keep], [meta[i] for i in keep])
        rec["files"][name] = {"source": str(src), "source_sha256": got, "pinned": bool(want), "rows": len(labels),
                              "psp_rows": len(keep), "vns": len({meta[i].get("vn") for i in keep}), "out": str(out),
                              "sha256": sha, "private": private}
        print(name, len(labels), "->", len(keep), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "native_record.json").write_text(json.dumps(rec, indent=1) + "\n")


# ---------------------------------------------------------------- console crops of the hooked PSP pages

def build_finder(out_dir):
    drv = Path(out_dir) / "vocr_lines_refine_test"
    drv.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gcc", *FINDER_FLAGS, "-o", str(drv), "runtime/vocr_lines.c", "runtime/vocr_lines_refine.c",
                    "runtime/vocr_lines_refine_test.c"], check=True, cwd=ROOT)
    return drv


def c_find(driver, images, finder=GRAD, flags=REFINE):
    if not images:
        return []
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        for img in images:
            img = np.ascontiguousarray(img, np.uint8)
            h, w = img.shape
            f.write(struct.pack("<6I", w, h, 1, finder, w, flags))
            f.write(img.tobytes())
        path = f.name
    try:
        out = subprocess.run([str(driver), path], check=True, capture_output=True, text=True).stdout.splitlines()
    finally:
        os.unlink(path)
    res = []
    for ln in out:
        v = [int(t) for t in ln.split()]
        if v[0]:
            raise RuntimeError(f"finder rc {v[0]}")
        res.append([tuple(v[5 + 5 * i:10 + 5 * i]) for i in range(v[2])])
    if len(res) != len(images):
        raise RuntimeError("finder driver: result count differs")
    return res


def pair_lines(boxes, lines, others=(), cells=None):
    """One-to-one pairs (finder box index, line index) of eval's matching rule, kept when the finder box holds the
    line's whole text and no other line's row. boxes: [(x, y, w, h, dark)], lines: [(x, y, w, h) of the line's box,
    (x, y, w, h) of its main text]; cells: per line the (x0, x1) of each non-blank character cell, or None (then the
    finder box must cover the main text's columns less a quarter of its height); others: boxes of further text rows
    (other lines of the page) that must not share a finder box. Returns (pairs, reasons Counter)."""
    why = Counter()
    cand = []
    for i, (x, y, w, h, _) in enumerate(boxes):
        for j, (lb, _) in enumerate(lines):
            ov = min(y + h, lb[1] + lb[3]) - max(y, lb[1])
            hov = min(x + w, lb[0] + lb[2]) - max(x, lb[0])
            if ov >= 0.5 * min(h, lb[3]) and hov > 0:
                cand.append((ov, -i, i, j))
    cand.sort(reverse=True)
    used_b, used_l, pairs = set(), set(), []
    for _, _, i, j in cand:
        if i in used_b or j in used_l:
            continue
        used_b.add(i)
        used_l.add(j)
        x, y, w, h, _ = boxes[i]
        mx, my, mw, mh = lines[j][1]
        cl = cells[j] if cells is not None else None
        if cl:
            cut = any(min(c1, x + w) - max(c0, x) < COVER * max(c1 - c0, 1e-6) for c0, c1 in cl)
        else:
            tol = 0.25 * mh
            cut = x > mx + tol or x + w < mx + mw - tol
        if cut:
            why["cuts_text"] += 1
            continue
        rows = [lb for k, (lb, _) in enumerate(lines) if k != j] + list(others)
        if any(min(y + h, b[1] + b[3]) - max(y, b[1]) >= 0.5 * min(h, b[3]) and
               min(x + w, b[0] + b[2]) - max(x, b[0]) > 0 for b in rows):
            why["merged_rows"] += 1
            continue
        pairs.append((i, j))
        why["kept"] += 1
    why["finder_unmatched"] += len(boxes) - len(used_b)
    why["line_unmatched"] += len(lines) - len(used_l)
    return pairs, why


def cmd_console(a):
    import cv2

    from common import to_gray
    driver = build_finder(Path(a.out) / "build")
    splits = a.splits.split(",")
    rec = {"kind": "modelb-console-crops", "finder": "VOCR_LINES_GRAD + vocr_refine_lines(VOCR_REFINE_DEFAULT)",
           "pagepool": str(a.pagepool), "titles": {}, "finder_driver_sha256": sha256(driver)}
    for f in sorted(Path(a.pagepool).glob("*.jsonl")):
        pages = [json.loads(ln) for ln in open(f, encoding="utf-8")]
        pages = [p for p in pages if p.get("platform") == "psp"]
        if not pages:
            continue
        tid = f.stem
        cells = {}                                 # line id -> page (x0, x1) of each non-blank character cell
        folders = [d for d in Path(a.root, "psp").glob(f"{tid} *") if (d / "lines.jsonl.zst").exists()]
        if len(folders) != 1:
            raise SystemExit(f"{tid}: {len(folders)} title folders under {a.root}/psp")
        raw = subprocess.run(["zstd", "-dc", str(folders[0] / "lines.jsonl.zst")], check=True,
                             capture_output=True).stdout.decode("utf-8")
        for ln in raw.splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            cb, cc = r.get("cell_boxes"), r.get("cell_chars")
            if cb and cc and len(cb) == len(cc):
                cells[r["id"]] = [(b[0], b[2]) for b, ch in zip(cb, cc) if ch.strip()]
        out = {s: ([], [], []) for s in splits}
        why = Counter()
        jobs = []                                  # (split, page, region, region image)
        for p in pages:
            if p["split"] not in splits or not p.get("regions") or p.get("untrusted"):
                continue                         # a page whose labels are in doubt gives no crop (audit N8)
            path = Path(a.pages) / p["image"]
            if sha256(path) != p["image_sha256"]:
                raise SystemExit(f"{path}: sha256 differs from the page record")
            grey = to_gray(cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
            for r in p["regions"]:
                if r["variant"] in VARIANTS and r["lines"]:
                    x0, y0, x1, y1 = r["region"]
                    jobs.append((p["split"], p, r, np.ascontiguousarray(grey[y0:y1, x0:x1])))
        found = c_find(driver, [j[3] for j in jobs])
        for (split, p, r, img), boxes in zip(jobs, found):
            by_id = {ln["id"]: ln for ln in p["lines"]}
            x0, y0 = r["region"][:2]
            if any(by_id[i].get("drop") for i in r["lines"]):
                why["region_with_dropped_line"] += 1
                continue

            def rel(b):
                return (b[0] - x0, b[1] - y0, b[2], b[3])
            lines = [(rel(by_id[i]["box"]), rel(by_id[i].get("box_main") or by_id[i]["box"])) for i in r["lines"]]
            others = [rel(ln["box"]) for ln in p["lines"] if ln["id"] not in r["lines"] and not ln["vertical"]]
            cl = [[(c0 - x0, c1 - x0) for c0, c1 in cells[i]] if i in cells else None for i in r["lines"]]
            why["lines_without_cells"] += sum(c is None for c in cl)
            pairs, w = pair_lines(boxes, lines, others, cl)
            why.update(w)
            for i, j in pairs:
                x, y, bw, bh, dark = boxes[i]
                lid = r["lines"][j]
                g = img[y:y + bh, x:x + bw]
                crops, labels, meta = out[split]
                crops.append(np.ascontiguousarray(255 - g if dark else g))
                labels.append(by_id[lid]["text"])
                meta.append({"real": True, "pool": "modelb-console-v1", "id": f"{lid}|{r['variant']}",
                             "title_id": tid, "platform": "psp", "kind": "psp", "target": "psp", "split": split,
                             "page": p["page"], "variant": r["variant"], "crop": "grad+refine",
                             "box": [x + x0, y + y0, bw, bh], "dark": int(dark), "inverted": bool(dark),
                             "line_box": by_id[lid]["box"], "image_sha256": p["image_sha256"]})
        rec["titles"][tid] = {"regions": len(jobs), "reasons": dict(why), "files": {}}
        for split, (crops, labels, meta) in out.items():
            if crops:
                path = Path(a.out) / split / f"{tid}.npz"
                rec["titles"][tid]["files"][split] = {"rows": len(crops), "sha256": write_shard(path, crops, labels,
                                                                                                 meta)}
        print(tid, len(jobs), dict(why), {s: len(v[0]) for s, v in out.items()}, flush=True)
    (Path(a.out) / "record.json").write_text(json.dumps(rec, indent=1) + "\n")


# ---------------------------------------------------------------- the PSP character ranking

def cmd_rank(a):
    from common import load_vocab
    vocab = load_vocab(a.vocab)
    counts = Counter()
    files = []
    for g in a.sources:
        for f in sorted(glob.glob(g)):
            _, labels, _ = load_shard(f)
            for t in labels:
                counts.update(t)
            files.append({"file": f, "sha256": sha256(f), "lines": len(labels)})
    idx = {c: i for i, c in enumerate(vocab)}
    order = sorted(range(len(vocab)), key=lambda i: (-counts.get(vocab[i], 0), i))
    total = sum(counts.values())
    inv = sum(n for c, n in counts.items() if c in idx)
    out = {"kind": "modelb-rank", "vocab": str(a.vocab), "vocab_sha256": sha256(a.vocab), "files": files,
           "characters": total, "in_vocab_characters": inv,
           "order": [vocab[i] for i in order],                       # class i + 1 of the permuted head
           "counts": [int(counts.get(vocab[i], 0)) for i in order],
           "coverage": {str(k): round(sum(counts.get(vocab[i], 0) for i in order[:k]) / max(inv, 1), 6)
                        for k in (255, 503, 1023, 1503, 2047)}}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False) + "\n")
    print(json.dumps({k: out[k] for k in ("characters", "in_vocab_characters", "coverage")}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("native")
    n.add_argument("--out", required=True)
    n.add_argument("--private-out", required=True)
    c = sub.add_parser("console")
    c.add_argument("--pagepool", required=True)
    c.add_argument("--pages", required=True)
    c.add_argument("--root", required=True, help="the hooked corpus' mirror (lines.jsonl.zst cells)")
    c.add_argument("--out", required=True)
    c.add_argument("--splits", default="train,guard,hval")
    r = sub.add_parser("rank")
    r.add_argument("--sources", nargs="+", required=True)
    r.add_argument("--vocab", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    {"native": cmd_native, "console": cmd_console, "rank": cmd_rank}[a.cmd](a)


if __name__ == "__main__":
    main()
