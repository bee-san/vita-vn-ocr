#!/usr/bin/env python3
"""textdet: checks of decode variants that were not adopted, on held-out canvases (never development data).

  textdet_decode_check.py --blob NAME=BLOB ... --out JSON [--synth 400] [--real 150]

Held-out canvases: synthetic ones rendered with a seed the training shards never use (synth/textdet_data.py's
generators, seed 99), and player-like regions cut from realplus screenshots that pass every exclusion but were left
out of the training pool by its per-VN cap. The frozen decode (synth/textdet_twin.py, "plain") and a variant with
centre-consistent connectivity ("centre", not adopted: it splits more lines than it separates) run on every canvas;
line matching is eval/linefind/metrics.py's (IoU >= 0.5, one to one; ignore areas absorb), against the canvases'
line boxes (main-text ink boxes, unpadded).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "synth", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def held_out_real(n, seed):
    import textdet_data as D
    import textdet_train as TT
    kept, _ = D.real_rows("realplus")
    kept_shots = {r["shot"] for r in kept}
    ids, shas, lf_vns = D._eval_exclusions()
    from common import dev_sample_vns, reserved_sessions, test_set_vns
    ex = (ids, shas, lf_vns, set(test_set_vns()), set(dev_sample_vns()), set(reserved_sessions()))
    rows = {}
    for r in D._jsonl(D.ART / "realplus" / "reads" / "reads.jsonl"):
        if r.get("dets") is not None:
            rows[r["shot"]] = r
    meta = {r["shot"]: r for r in D._jsonl(D.ART / "realplus" / "reads" / "shots.jsonl")}
    for k, r in rows.items():
        for f in ("sha256", "session", "release", "url"):
            r.setdefault(f, (meta.get(k) or {}).get(f))
    pool = sorted(k for k, r in rows.items() if k not in kept_shots and not D._excluded(r, ex))
    rng = np.random.default_rng(seed)
    pick = [rows[pool[i]] for i in rng.choice(len(pool), size=min(n, len(pool)), replace=False)]
    out = []
    for r in pick:
        for g, lines, ignore, _ in D.real_canvases(r, "realplus", rng):
            hz = [ln for ln in lines if not ln[4]]
            if not hz:
                continue
            H, W = g.shape
            reg = TT.group_region(lines, int(rng.integers(1 << 30)), rng, H, W, False)
            x0, y0, x1, y1 = reg
            out.append(crop(g, lines, ignore, x0, y0, x1, y1, "real"))
    return out


def crop(g, lines, ignore, x0, y0, x1, y1, src):
    keep, ign = [], [[b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0] for b in ignore]
    for ln in lines:
        b = [ln[0] - x0, ln[1] - y0, ln[2] - x0, ln[3] - y0]
        inside = b[0] >= 0 and b[1] >= 0 and b[2] <= x1 - x0 and b[3] <= y1 - y0
        if inside:
            keep.append(b + [ln[4]])
        elif min(b[2], x1 - x0) > max(b[0], 0) and min(b[3], y1 - y0) > max(b[1], 0):
            ign.append(b)
    return {"image": np.ascontiguousarray(g[y0:y1, x0:x1]), "lines": keep, "ignore": ign, "src": src}


def held_out_synth(n, seed):
    import textdet_data as D
    D.init_synth()
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        kind = D.choose(rng, D.SYNTH_MIX)
        s = D.SYNTH[kind](rng)
        if s is None:
            continue
        g = s["canvas"]
        out.append({"image": g, "lines": [ln[:4] + [ln[4]] for ln in s["lines"]], "ignore": s["ignore"],
                    "src": f"synth:{kind}"})
    return out


def consistent_components(maps, s, across_only=False):
    """The variant checked (not adopted): neighbouring kernel cells join only when they agree on their line (the
    same orientation sign and predicted centres P = 2*c2 - A1 + A2 within max(8, (H_a + H_b) / 4), H = A1 + A2)."""
    kmask, vert = maps[0] > 0, maps[1] > 0
    h, w = kmask.shape
    rows = np.arange(h, dtype=np.int64)[:, None]
    cols = np.arange(w, dtype=np.int64)[None, :]
    c2 = np.where(vert, 2 * s * cols + s - 1, 2 * s * rows + s - 1)
    P, H = 2 * c2 - maps[2] + maps[3], maps[2] + maps[3]
    parent = {(i, j): (i, j) for i in range(h) for j in range(w) if kmask[i, j]}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def ok(a, b, across):
        if not kmask[b] or vert[a] != vert[b]:
            return False
        if across_only and not across:
            return True                     # along the line: always joined
        return abs(int(P[a]) - int(P[b])) <= max(8, (int(H[a]) + int(H[b])) // 4)

    for (i, j) in list(parent):
        for b, down in (((i, j + 1), False), ((i + 1, j), True)):
            # a vertical step crosses a horizontal line's rows; a horizontal step crosses a vertical line's columns
            across = down != bool(vert[i, j])
            if b[0] < h and b[1] < w and ok((i, j), b, across):
                ra, rb = find((i, j)), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for cell in parent:
        groups.setdefault(find(cell), []).append(cell)
    return [[(i, j, j + 1) for i, j in sorted(cells)] for cells in groups.values()]


def find_lines(model, img, connect):
    import textdet_twin as TW
    if connect == "plain":
        return TW.find_lines(model, img)
    if connect.startswith("thr"):                       # plain connectivity, kernel cells K > t
        t = int(connect[3:])
        maps = TW.run_net(model, img)
        maps = maps.copy()
        maps[0] = np.where(maps[0] > t, maps[0], 0)
        boxes, truncated = TW.decode(model, maps, img.shape[0], img.shape[1])
        return [(x, y, w, h, 0, v) for x, y, w, h, v in boxes], truncated
    saved = TW.components
    maps = TW.run_net(model, img)
    TW.components = lambda kmask: consistent_components(maps, model.stride, connect == "across")
    try:
        boxes, truncated = TW.decode(model, maps, img.shape[0], img.shape[1])
    finally:
        TW.components = saved
    return [(x, y, w, h, 0, v) for x, y, w, h, v in boxes], truncated


def score(cases, model, connect):
    import textdet as TD
    M = TD._linefind("metrics")
    tot = {"tp": 0, "fp": 0, "fn": 0, "merged": 0, "split": 0, "exact": 0, "regions": 0}
    for c in cases:
        lines, _ = find_lines(model, c["image"], connect)
        pred = [(x, y, x + w, y + h) for x, y, w, h, _, _ in lines]
        m = M.match(pred, [tuple(b[:4]) for b in c["lines"]], [tuple(b) for b in c["ignore"]])
        for k in ("tp", "fp", "fn", "merged", "split"):
            tot[k] += m[k]
        tot["exact"] += m["fp"] == 0 and m["fn"] == 0
        tot["regions"] += 1
    p = tot["tp"] / max(1, tot["tp"] + tot["fp"])
    r = tot["tp"] / max(1, tot["tp"] + tot["fn"])
    return {**tot, "precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / max(1e-9, p + r), 4),
            "exact_rate": round(tot["exact"] / max(1, tot["regions"]), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blob", action="append", required=True)
    ap.add_argument("--synth", type=int, default=400)
    ap.add_argument("--real", type=int, default=150)
    ap.add_argument("--out", required=True)
    ap.add_argument("--connect", nargs="+", default=["plain", "centre", "across"])
    args = ap.parse_args()
    import textdet_twin as TW
    sets = {"synth": held_out_synth(args.synth, 99), "real": held_out_real(args.real, 99)}
    res = {"kind": "textdet-decode-check", "cases": {k: len(v) for k, v in sets.items()}, "results": {}}
    for spec in args.blob:
        name, _, blob = spec.partition("=")
        model = TW.Model(Path(blob).read_bytes())
        for conn in args.connect:
            for k, cases in sets.items():
                r = score(cases, model, conn)
                res["results"][f"{name}/{conn}/{k}"] = r
                print(f"{name} {conn:6s} {k:5s} F1 {r['f1']:.3f} P {r['precision']:.3f} R {r['recall']:.3f} "
                      f"exact {r['exact_rate']:.3f} merged {r['merged']} split {r['split']}", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
