#!/usr/bin/env python3
"""Build the line-finder ground truth manifests (build tool; outputs are committed).

  build_gt.py research  -> manifests/research_a.jsonl, research_b.jsonl
  build_gt.py det SET   -> manifests/SET.candidates.jsonl from detector output
                           (vita_dev, psp_dev; see DET_SETS), for review on sheets
  build_gt.py sheets MANIFEST [--out DIR]
                        -> contact sheets with every GT line box drawn, for checking
  build_gt.py finalize  -> SET.jsonl from SET.candidates.jsonl and the hand review in
                           manifests/review.json (dropped regions, ignored watermark lines)
  build_gt.py fetch [--dest DIR]
                        -> download the screenshots the manifests name (VNDB URL + sha256)

manifests/psp_pool.jsonl lists the 264 PSP screenshots behind psp_*.jsonl: VNDB
screenshots at 480x272 of 90 Japanese-only PSP VNs (every k-th VN by id, up to 3 shots each)
with sexual < 0.3 and violence < 0.5, none from eval's test games.

One JSON row per region:
  id, set, vn, shot (local file name), url, sha256, platform ("psv" or "psp"),
  region [x0, y0, x1, y1] in screenshot pixels (exclusive ends),
  mask [[x0, y0, x1, y1], ...]  rectangles filled with the median colour of the
                                 6 rows above them (press-kit watermarks),
  lines [{box [x0, y0, x1, y1] in screenshot pixels, text or null, ignore bool}],
  tags [...]

Set A is the research campaign's 16 tight regions (bf653e98, cycle 005) and set B
its independent 19-screenshot replication (q5_run_a), both hand-transcribed by
the research agents. Line boxes for A come from B where the screenshot is the same,
otherwise from PP-OCRv5 server detection, and were checked on contact sheets.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAN = HERE / "manifests"
RESEARCH = Path("/local/home/skerraut/herd-vita-ocr/shared/research/scratch/real")
# Where screenshots are looked up: the repo's VNDB download (data/fetch_assets.py vndb) and
# the destination of `build_gt.py fetch`. Both are gitignored; screenshots are never committed.
SHOT_DIRS = [ROOT / "data" / "vndb" / "shots", ROOT / "data" / "linefind_shots"]
UA = "vita-vn-ocr-linefind/0.1 (line-finder eval set; github.com/bee-san/vita-vn-ocr)"


def vndb_url(shot):
    """data/vndb/shots names are <vn>_<screenshot id>.jpg."""
    sid = shot.rsplit("_", 1)[1].split(".")[0]
    return f"https://t.vndb.org/sf/{sid[-2:]}/{sid}.jpg"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_shot(row, shots_dirs):
    for d in shots_dirs:
        p = Path(d) / row["shot"]
        if p.exists():
            break
    else:
        raise FileNotFoundError(row["shot"])
    a = np.asarray(Image.open(p).convert("RGB")).copy()
    for x0, y0, x1, y1 in row.get("mask", []):
        ref = a[max(0, y0 - 6):y0, x0:x1].reshape(-1, 3)
        a[y0:y1, x0:x1] = np.median(ref, axis=0).astype(np.uint8)
    return a


def region_rgb(row, shots_dirs):
    a = load_shot(row, shots_dirs)
    x0, y0, x1, y1 = row["region"]
    return a[y0:y1, x0:x1]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{path}: {len(rows)} regions, {sum(len(r['lines']) for r in rows)} lines")


def read(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def det_index(paths):
    idx = {}
    for p in paths:
        if Path(p).exists():
            for r in read(p):
                idx[r["shot"]] = r
    return idx


def det_boxes(det_row, min_score=0.5):
    """Axis-aligned boxes of the detector's quads."""
    out = []
    for q, s in zip(det_row["quads"], det_row["scores"]):
        if s < min_score:
            continue
        q = np.asarray(q)
        out.append([int(np.floor(q[:, 0].min())), int(np.floor(q[:, 1].min())),
                    int(np.ceil(q[:, 0].max())), int(np.ceil(q[:, 1].max()))])
    return out


def merge_rows(boxes, max_gap_h=3.0):
    """Join detector boxes that sit on the same text row (the detector splits a line
    at wide gaps, e.g. between a speaker name and the dialogue)."""
    boxes = sorted(boxes, key=lambda b: b[0])
    out = []
    for b in boxes:
        for o in out:
            h = min(b[3] - b[1], o[3] - o[1])
            ov = min(b[3], o[3]) - max(b[1], o[1])
            gap = b[0] - o[2]
            if h > 0 and ov >= 0.5 * h and gap <= max_gap_h * h:
                o[:] = [min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3])]
                break
        else:
            out.append(list(b))
    return sorted(out, key=lambda b: b[1])


def research():
    """Sets A and B from the research campaign's hand-transcribed real regions."""
    shots_dir = ROOT / "data" / "vndb" / "shots"
    by_hash = {sha(p): p.name for p in shots_dir.glob("*.jpg")}
    det = det_index([Path("/local/home/skerraut/herd-vita-ocr/shared/artifacts/linefind/det/vita_shots_det.jsonl")])
    b_rows = []
    for it in json.loads((RESEARCH / "q5_run_a" / "gt.json").read_text(encoding="utf-8")):
        shot = by_hash[sha(it["file"])]
        lines = [{"box": list(ln["box"]), "text": ln["text"], "ignore": False,
                  **({"uncertain": ln["uncertain"]} if not ln["confident"] else {})}
                 for ln in it["lines"]]
        n = it["notes"]
        tags = [it["kind"]] + (["outlined"] if "outline" in n.get("edge", "") else []) \
            + (["shadow"] if "shadow" in n.get("edge", "") else []) \
            + (["mincho"] if "Mincho" in n.get("font", "") else [])
        b_rows.append({"id": f"B_{it['tag']}", "set": "B", "vn": it["vn"], "shot": shot,
                       "url": it["url"], "sha256": sha(shots_dir / shot), "platform": "psv",
                       "region": list(it["region"]), "mask": [list(m) for m in it.get("masked", [])],
                       "lines": lines, "tags": tags, "notes": n})
    b_by_shot = {r["shot"]: r for r in b_rows}
    a_rows, todo = [], []
    for it in json.loads((RESEARCH / "gt_real.json").read_text(encoding="utf-8")):
        shot = by_hash[sha(RESEARCH / "shots" / f"{it['shot']}.jpg")]
        x0, y0, x1, y1 = it["region"]
        boxes = None
        if shot in b_by_shot:
            boxes = [ln["box"] for ln in b_by_shot[shot]["lines"]
                     if (ln["box"][1] + ln["box"][3]) / 2 > y0 and (ln["box"][1] + ln["box"][3]) / 2 < y1]
        if (boxes is None or len(boxes) != len(it["lines"])) and shot in det:
            cand = [b for b in det_boxes(det[shot])
                    if x0 <= (b[0] + b[2]) / 2 <= x1 and y0 <= (b[1] + b[3]) / 2 <= y1]
            boxes = merge_rows(cand)
        if boxes is None or len(boxes) != len(it["lines"]):
            todo.append(it["shot"])
            boxes = [None] * len(it["lines"])
        lines = [{"box": None if b is None else [max(b[0], x0), max(b[1], y0), min(b[2], x1), min(b[3], y1)],
                  "text": t, "ignore": False} for b, t in zip(boxes, it["lines"])]
        a_rows.append({"id": f"A_{it['shot']}", "set": "A", "vn": it["shot"].split("_")[0], "shot": shot,
                       "url": vndb_url(shot), "sha256": sha(shots_dir / shot), "platform": "psv",
                       "region": [x0, y0, x1, y1], "mask": [], "lines": lines, "tags": [],
                       "notes": it["notes"]})
    write(MAN / "research_a.jsonl", a_rows)
    write(MAN / "research_b.jsonl", b_rows)
    if todo:
        print("A regions still without line boxes:", todo)


ART = Path("/local/home/skerraut/herd-vita-ocr/shared/artifacts")
DET_SETS = {
    "vita": dict(det=ART / "linefind/det/vita_shots_det.jsonl", shots=ROOT / "data/vndb/shots",
                 platform="psv", size=(960, 544)),
    "psp": dict(det=ART / "linefind/det/psp_dev_det.jsonl", shots=ART / "linefind/psp_dev/shots",
                platform="psp", size=(480, 272)),
}


def test_games():
    """Every VN reserved for eval's test sets (provisional list) plus the hash rule."""
    sys.path.insert(0, str(ROOT / "synth"))
    from common import is_test_vn
    p = ART / "eval" / "test_games.json"
    listed = set(json.loads(p.read_text())["all"]) if p.exists() else set()
    return lambda vn: vn in listed or is_test_vn(vn)


def holdout_vn(vn):
    return int(hashlib.md5(("linefind:" + vn).encode()).hexdigest(), 16) % 3 == 0


def det_set(which):
    """Region candidates with detector line boxes, split by VN into dev / holdout.
    Each screenshot gives a 'tight' region (dialogue block + 4..12 px) and a 'loose' one
    (+16..64 px sideways, as when a whole text box is selected), and every fourth gives a
    text-free 'empty' region from the upper screen (false alarm check)."""
    cfg = DET_SETS[which]
    W, H = cfg["size"]
    s = W / 960
    is_test = test_games()
    out = {"dev": [], "holdout": []}
    per_vn = {}
    for r in read(cfg["det"]):
        vn = r["shot"].split("_")[0]
        if is_test(vn) or r["size"] != [W, H] or per_vn.get(vn, 0) >= 2:
            continue
        rng = np.random.default_rng(int(hashlib.md5(r["shot"].encode()).hexdigest()[:8], 16))
        weak = det_boxes(r, min_score=0.3)
        allb = merge_rows(det_boxes(r))
        # Dialogue line candidates: horizontal, plausible size, lower screen, not a
        # bottom-edge watermark/UI strip.
        cand = [b for b in allb if 10 * s <= b[3] - b[1] <= 72 * s and b[2] - b[0] >= 1.2 * (b[3] - b[1])
                and (b[1] + b[3]) / 2 >= 0.45 * H and not (b[3] > H - 22 * s and b[3] - b[1] < 30 * s)]
        blocks = []
        for b in sorted(cand, key=lambda b: b[1]):
            for blk in blocks:
                u = blk["u"]
                hmax = max(x[3] - x[1] for x in blk["l"])
                xgap = max(u[0] - b[2], b[0] - u[2], 0)
                if b[1] - u[3] <= 1.2 * hmax and xgap <= 3 * hmax:
                    blk["l"].append(b)
                    blk["u"] = [min(u[0], b[0]), min(u[1], b[1]), max(u[2], b[2]), max(u[3], b[3])]
                    break
            else:
                blocks.append({"l": [b], "u": list(b)})
        base = {"vn": vn, "shot": r["shot"], "url": vndb_url(r["shot"]), "sha256": r["sha256"],
                "platform": cfg["platform"], "mask": []}
        split = "holdout" if holdout_vn(vn) else "dev"
        if blocks:
            blk = max(blocks, key=lambda k: sum(area(x) for x in k["l"]))
            long_line = any(x[2] - x[0] >= 6 * (x[3] - x[1]) for x in blk["l"])
            if len(blk["l"]) <= 8 and long_line:
                u = blk["u"]
                med_h = float(np.median([x[3] - x[1] for x in blk["l"]]))
                m = int(rng.integers(4, 13) * max(s, 0.5))
                variants = {
                    "tight": [u[0] - m, u[1] - m, u[2] + m, u[3] + m],
                    "loose": [u[0] - int(rng.integers(16, 65) * s), u[1] - int(rng.integers(8, 33) * s),
                              u[2] + int(rng.integers(16, 65) * s), u[3] + int(rng.integers(8, 41) * s)],
                }
                for name, reg in variants.items():
                    reg = [max(0, reg[0]), max(0, reg[1]), min(W, reg[2]), min(H, reg[3])]
                    # Never cut text: drop the variant if any detected text straddles its edge.
                    if any(area(b) > 0 and 4 < area([max(b[0], reg[0]), max(b[1], reg[1]), min(b[2], reg[2]),
                                                     min(b[3], reg[3])]) < area(b) - 2 * (b[2] - b[0] + b[3] - b[1])
                           for b in weak):
                        continue
                    lines = []
                    for b in allb:
                        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                        if not (reg[0] <= cx < reg[2] and reg[1] <= cy < reg[3]):
                            continue
                        clipped = b[0] < reg[0] - 2 or b[1] < reg[1] - 2 or b[2] > reg[2] + 2 or b[3] > reg[3] + 2
                        small = b[3] - b[1] < 0.6 * med_h or b[3] - b[1] > 1.2 * (b[2] - b[0])
                        in_block = b in blk["l"]
                        bb = [max(b[0], reg[0]), max(b[1], reg[1]), min(b[2], reg[2]), min(b[3], reg[3])]
                        lines.append({"box": bb, "text": None, "ignore": bool(clipped or small or not in_block)})
                    out[split].append({**base, "id": f"{which}_{name}_{r['shot'].split('.')[0]}", "set": which,
                                       "region": reg, "lines": lines, "tags": [name]})
                    if name == "tight":
                        per_vn[vn] = per_vn.get(vn, 0) + 1
        if int(hashlib.md5(r["shot"].encode()).hexdigest(), 16) % 4 == 0:
            for _ in range(20):
                w, h = int(rng.integers(300, 601) * s), int(rng.integers(60, 141) * s)
                x0, y0 = int(rng.integers(0, W - w + 1)), int(rng.integers(0, int(0.45 * H) - h + 1))
                reg = [x0, y0, x0 + w, y0 + h]
                pad = 10 * s
                if all(b[2] < x0 - pad or b[0] > reg[2] + pad or b[3] < y0 - pad or b[1] > reg[3] + pad
                       for b in det_boxes(r, min_score=0.3)):
                    out[split].append({**base, "id": f"{which}_empty_{r['shot'].split('.')[0]}", "set": which,
                                       "region": reg, "lines": [], "tags": ["empty"]})
                    break
    for split, rows in out.items():
        write(MAN / f"{which}_{split}.candidates.jsonl", rows)


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def fetch(dest, shots_dirs):
    """Download every screenshot a manifest names that is not already local, from its VNDB
    URL, at most two requests a second, and keep it only if its sha256 matches."""
    import time

    import requests
    want = {}
    for m in sorted(MAN.glob("*.jsonl")):
        for r in read(m):
            want.setdefault(r["shot"], (r["url"], r["sha256"]))
    dest.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = UA
    got = bad = 0
    for shot, (url, digest) in sorted(want.items()):
        if any((Path(d) / shot).exists() for d in shots_dirs):
            continue
        for attempt in range(4):
            try:
                r = s.get(url, timeout=60)
                if r.status_code == 429:
                    time.sleep(30 * (attempt + 1))
                    continue
                r.raise_for_status()
                break
            except requests.RequestException as e:
                print("retry", url, e, flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            bad += 1
            continue
        time.sleep(0.5)
        if hashlib.sha256(r.content).hexdigest() != digest:
            print("sha256 mismatch, skipped:", url, flush=True)
            bad += 1
            continue
        (dest / shot).write_bytes(r.content)
        got += 1
    print(f"{len(want)} screenshots named, {got} downloaded to {dest}, {bad} failed")
    return 1 if bad else 0


def finalize():
    """Apply the hand review (manifests/review.json) to every SET.candidates.jsonl."""
    review = json.loads((MAN / "review.json").read_text())
    drop, ignore = review["drop"], review["ignore"]
    seen = set()
    for cand in sorted(MAN.glob("*.candidates.jsonl")):
        rows = []
        for r in read(cand):
            seen.add(r["id"])
            if r["id"] in drop:
                continue
            boxes = ignore.get(r["id"], [])
            for ln in r["lines"]:
                if ln["box"] in boxes:
                    ln["ignore"] = True
            rows.append(r)
        write(MAN / cand.name.replace(".candidates", ""), rows)
    stale = sorted((set(drop) | set(ignore)) - seen)
    if stale:
        sys.exit(f"review.json names regions that no candidate set has: {stale}")


def sheets(manifest, out, shots_dirs):
    rows = read(manifest)
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for r in rows:
        rgb = region_rgb(r, shots_dirs)
        im = Image.fromarray(rgb)
        d = ImageDraw.Draw(im)
        x0, y0 = r["region"][:2]
        for ln in r["lines"]:
            if ln["box"]:
                b = ln["box"]
                d.rectangle((b[0] - x0, b[1] - y0, b[2] - x0 - 1, b[3] - y0 - 1),
                            outline=(255, 160, 0) if ln.get("ignore") else (0, 255, 0))
        tiles.append((r["id"], im))
    page, cur, h = 0, [], 0
    for t in tiles + [None]:
        if t is None or (cur and h + t[1].height + 14 > 1250):
            W = min(1280, max(im.width for _, im in cur))
            s = Image.new("RGB", (W, sum(im.height + 14 for _, im in cur)), (40, 40, 40))
            d = ImageDraw.Draw(s)
            y = 0
            for name, im in cur:
                d.text((2, y), name, fill=(255, 255, 0))
                s.paste(im, (0, y + 12))
                y += im.height + 14
            s.save(out / f"{Path(manifest).stem}_{page:02d}.png")
            page, cur, h = page + 1, [], 0
            if t is None:
                break
        cur.append(t)
        h += t[1].height + 14
    print(f"{page} sheets in {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["research", "det", "sheets", "finalize", "fetch"])
    ap.add_argument("manifest", nargs="?")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "gt_sheets")
    ap.add_argument("--shots", action="append", default=[], help="extra screenshot dirs")
    ap.add_argument("--dest", type=Path, default=SHOT_DIRS[-1], help="fetch destination")
    args = ap.parse_args()
    shots = args.shots + [str(d) for d in SHOT_DIRS]
    if args.cmd == "research":
        research()
    elif args.cmd == "det":
        det_set(args.manifest)
    elif args.cmd == "sheets":
        sheets(args.manifest, args.out, shots)
    elif args.cmd == "finalize":
        finalize()
    elif args.cmd == "fetch":
        return fetch(args.dest, shots)


if __name__ == "__main__":
    sys.exit(main())
