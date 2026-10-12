#!/usr/bin/env python3
"""Build the devnative cohort files from the final labels.

  build.py npz [--labels $OUT/labels.jsonl] [--reads ...] [--out $OUT] [--prefix devnative]
               [--manifests devnative/manifests]

For every kept line (decide.py final) two crops are cut from its native screenshot, both as
grey integer luma (synth/common.to_gray) in the console polarity (a crop whose text
common.line_dark calls dark is stored inverted, meta dark/inverted):
  ink   the line's ink box with furigana, plus 4 px (Vita) / 2 px (PSP): teacher's and eval's crop;
  grad  the box the console's GRAD finder returns for the line's dialogue region (devnative/
        lines.py), only where it holds the whole line and no other text (status "ok").
Identical labels are kept at most twice per VN (teacher's cap: names and UI strings repeat).

Outputs, one file per subset and crop, never pooled (the conductor's decision):
  $OUT/<prefix>_<subset>_<ink|grad>.npz    teacher_val's format: data, offsets, shapes, labels
        (JSON list), meta (JSON list; id, vn, session, kind, target, box, dark, inverted, how, pol,
        chars, plus subset, crop, flags, decision provenance), so eval/synth3_ablation.py score
        reads it unchanged
  $OUT/lines_full.jsonl                    every kept line with its label and readings (local only)
  devnative/manifests/lines.jsonl          every kept line: ids, boxes, provenance, no text
  devnative/manifests/dropped.jsonl        every adjudicated line not kept, and why (no text)
  devnative/manifests/stats.json           counts
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from devnative import lines as L  # noqa: E402
from devnative import paths as P  # noqa: E402
from devnative import sheets as S  # noqa: E402
from teacher import geometry as G  # noqa: E402
from teacher import vote  # noqa: E402

PER_VN_CAP = 2
SUBSETS = ("strict", "tv_native", "scene_vn")
CROPS = ("ink", "grad")


def chars_in(line, label, box):
    """MeikiOCR's character boxes relative to the crop ([c, x0, y0, x1, y1, conf]): of the
    reading that matches the label (its own box first, then the view), else of its own box."""
    k = vote.key(label)
    pick = None
    for e in ("meiki", "meikiv"):
        r = (line.get("reads") or {}).get(e)
        if not r:
            continue
        chars = r[1] if e == "meiki" else r[3]
        if vote.key(r[0]) == k and chars:
            pick = (chars, True)
            break
    if pick is None:
        r = (line.get("reads") or {}).get("meiki") or ["", []]
        pick = (r[1], False)
    x0, y0 = box[0], box[1]
    return [[c[0], round(c[1] - x0, 1), round(c[2] - y0, 1), round(c[3] - x0, 1), round(c[4] - y0, 1), c[5]]
            for c in pick[0]], pick[1]


def teacher_val_lines():
    """{shot key: [(teacher_val id, crop box, label)]} for native teacher_val lines."""
    z = np.load(P.ART / "teacher" / "teacher_val.npz")
    out = defaultdict(list)
    for m, lab in zip(json.loads(str(z["meta"])), json.loads(str(z["labels"]))):
        if m["kind"] in ("vita", "psp"):
            out[m["id"].rsplit("_", 1)[0]].append((m["id"], m["box"], lab))
    return out


def tv_overlap(box, cands):
    """The teacher_val line whose crop box best overlaps this one (IoU >= 0.3), or None."""
    best = max(cands, key=lambda c: L.box_iou(box, c[1]), default=None)
    if best is None or L.box_iou(box, best[1]) < 0.3:
        return None
    return best


def styles_of(line, rec, kind):
    tags = set()
    pol = G.polarity(line.get("hint", {}))
    if pol:
        tags.add(pol)
    if line.get("furi"):
        tags.add("furigana")
    if rec.get("name"):
        tags.add("name")
    if line["ink"][3] - line["ink"][1] < 16:
        tags.add("small")
    return sorted(tags)


def _save(path, crops, labels, metas):
    shapes = np.array([c.shape for c in crops], np.int32).reshape(-1, 2)
    offsets = np.concatenate([[0], np.cumsum(shapes[:, 0].astype(np.int64) * shapes[:, 1])]).astype(np.int64)
    data = np.concatenate([c.ravel() for c in crops]) if crops else np.zeros(0, np.uint8)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, data=data.astype(np.uint8), offsets=offsets, shapes=shapes,
                        labels=np.array(json.dumps(labels, ensure_ascii=False)),
                        meta=np.array(json.dumps(metas, ensure_ascii=False)))
    tmp.replace(path)


def cap_repeats(kept):
    """Keep identical labels at most PER_VN_CAP times per VN (first by line id)."""
    seen, out, capped = Counter(), [], []
    for r in sorted(kept, key=lambda r: (int(r["vn"][1:]), r["id"])):
        k = (r["vn"], vote.key(r["label"]))
        if seen[k] >= PER_VN_CAP:
            capped.append(r)
            continue
        seen[k] += 1
        out.append(r)
    return out, capped


def build(labels, shots, out_dir, manifests=P.MANIFESTS, tvl=None, prefix="devnative"):
    """Cut, write the npz files and the manifests. tvl: teacher_val's native lines
    (teacher_val_lines(); loaded when a tv_native line is kept and none are given)."""
    idx = S.line_index(shots)
    kept, capped = cap_repeats([r for r in labels if r["status"] == "kept"])
    dropped = [r for r in labels if r["status"] != "kept"] + [dict(r, status="dropped", why="repeat-cap")
                                                               for r in capped]
    if tvl is None:
        tvl = teacher_val_lines() if any(r["subset"] == "tv_native" for r in kept) else {}
    by_shot = defaultdict(list)
    for r in kept:
        by_shot[r["shot"]].append(r)
    subsets = sorted({r["subset"] for r in kept} | set(SUBSETS if prefix == "devnative" else ()))
    files = {(s, c): ([], [], []) for s in subsets for c in CROPS}
    man, full = [], []
    grad_stats = Counter()
    for sk in sorted(by_shot, key=lambda k: (int(k.split("_")[0][1:]), k)):
        s = shots[sk]
        rgb = np.ascontiguousarray(P.load_screen(s))
        H, W = rgb.shape[:2]
        recs = sorted(by_shot[sk], key=lambda r: r["id"])
        lines = [idx[r["id"]][1] for r in recs]
        grads = L.grad_assign(rgb, lines, s.get("dets", []), s["kind"])
        shot_meta = {"url": s["url"], "release": s.get("release")}
        for r, ln, g in zip(recs, lines, grads):
            grad_stats[f"{s['subset']}:{g['status']}"] += 1
            tv = tv_overlap(L.crop_box(ln, s["kind"], W, H), tvl.get(sk, [])) if s["subset"] == "tv_native" else None
            base = {"real": True, "pool": prefix, "id": r["id"], "vn": s["vn"], "session": s.get("release"),
                    "kind": s["kind"], "target": s["kind"], "how": r["how"], "pol": G.polarity(ln.get("hint", {})),
                    "subset": s["subset"], "flags": s.get("flags", []), "support": r["support"],
                    "reviewer_kind": "assistant", "styles": styles_of(ln, r, s["kind"])}
            if tv:
                base["tv_id"] = tv[0]
            row = {"id": r["id"], "shot": sk, "subset": s["subset"], "vn": s["vn"], "kind": s["kind"],
                   "session": s.get("release"), **shot_meta, "ink": ln["ink"], "text_box": L.text_box(ln),
                   "region": g["region"], "grad": {"box": g["box"], "status": g["status"], "dark": g["dark"]},
                   "styles": base["styles"], "flags": base["flags"], "how": r["how"], "support": r["support"],
                   "readers": r["readers"], "pass1": r["pass1"], "pass2": r["pass2"],
                   "reviewer_kind": "assistant", "chars": len(vote.key(r["label"]))}
            if r.get("resolution"):
                row["resolution"] = r["resolution"]
            if tv:
                row["tv_id"] = tv[0]
                row["tv_label_agrees"] = vote.key(tv[2]) == vote.key(r["label"])
            for crop in CROPS:
                box = L.crop_box(ln, s["kind"], W, H) if crop == "ink" else g["box"]
                if crop == "grad" and g["status"] != "ok":
                    continue
                img, dark, inverted = L.cut(rgb, box)
                ch, ch_ok = chars_in(ln, r["label"], box)
                meta = dict(base, crop=crop, box=box, dark=dark, inverted=inverted, chars=ch, chars_match=ch_ok)
                crops, labs, metas = files[(s["subset"], crop)]
                crops.append(img)
                labs.append(r["label"])
                metas.append(meta)
                row[f"{crop}_box"] = box
                row[f"{crop}_dark"] = dark
            man.append(row)
            full.append(dict(row, label=r["label"], readings=S.readings(ln),
                             tv_label=tv[2] if tv else None))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for (sub, crop), (crops, labs, metas) in files.items():
        name = f"{prefix}_{sub}_{crop}.npz"
        counts[name] = {"lines": len(crops), "vns": len({m["vn"] for m in metas}),
                        "by_kind": dict(Counter(m["kind"] for m in metas)),
                        "chars": sum(len(vote.key(x)) for x in labs)}
        if crops:
            _save(out_dir / name, crops, labs, metas)
            counts[name]["sha256"] = P.file_sha256(out_dir / name)
    (out_dir / "lines_full.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in full),
                                              encoding="utf-8")
    manifests = Path(manifests)
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "lines.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in man),
                                           encoding="utf-8")
    drop_fields = ("id", "shot", "subset", "vn", "kind", "why", "how", "pass1", "pass2", "reviewer_kind")
    (manifests / "dropped.jsonl").write_text(
        "".join(json.dumps({k: r.get(k) for k in drop_fields}, ensure_ascii=False) + "\n"
                for r in sorted(dropped, key=lambda r: r["id"])), encoding="utf-8")
    stats = {"files": counts, "kept": len(man), "dropped": dict(Counter(r["why"] for r in dropped)),
             "by_subset_kind": dict(Counter(f"{r['subset']}/{r['kind']}" for r in man)),
             "vns": {sub: len({r["vn"] for r in man if r["subset"] == sub}) for sub in subsets},
             "screenshots": {sub: len({r["shot"] for r in man if r["subset"] == sub}) for sub in subsets},
             "how": dict(Counter(f"{r['subset']}:{r['how']}" for r in man)),
             "grad": dict(grad_stats), "names": sum("name" in r["styles"] for r in man),
             "tv_overlap": {"lines": sum(1 for r in man if r.get("tv_id")),
                            "label_agrees": sum(1 for r in man if r.get("tv_label_agrees"))}}
    (manifests / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return stats


def cmd_npz(args):
    labels = [json.loads(x) for x in Path(args.labels).read_text(encoding="utf-8").splitlines() if x.strip()]
    shots = S.load_reads(args.reads)
    stats = build(labels, shots, args.out, manifests=args.manifests, prefix=args.prefix)
    print(json.dumps(stats, ensure_ascii=False, indent=1), flush=True)
    print("BUILD_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("npz")
    p.add_argument("--labels", default=str(P.OUT / "labels.jsonl"))
    p.add_argument("--reads", default=str(P.OUT / "reads" / "reads.jsonl"))
    p.add_argument("--out", default=str(P.OUT))
    p.add_argument("--prefix", default="devnative", help="npz file prefix and meta 'pool'")
    p.add_argument("--manifests", default=str(P.MANIFESTS), help="where lines/dropped/stats.json go")
    args = ap.parse_args()
    cmd_npz(args)


if __name__ == "__main__":
    main()
