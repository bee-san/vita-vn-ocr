#!/usr/bin/env python3
"""Style and text statistics of line-crop shards, for calibrating synth3 against real crops.

  synth3_stats.py measure --name real=A/teacher/teacher_train.npz+A/realdata/real_train.npz \
      --name s2=A/synth2/shards/synth_0000[0-1].npz --name s3='S3/shards/train/synth3_0000[0-1].npz' \
      --out stats.json [--limit 20000]
  synth3_stats.py table stats.json            # markdown tables (docs/synth3.md)

Only training-side real crops may be measured here (teacher_train, real_train): never the
selection set and never eval's test sets. Statistics per source:

  text      label length histogram, script shares (hiragana/katakana/kanji/latin/punct),
            VN punctuation per 1,000 characters, name-like line share, distinct characters,
            the character unigram distribution (for KL divergence between sources)
  geometry  native crop height and width/char pitch per platform target, ink margins (from
            teacher's character boxes for real crops, from the renderer's ink box for synth3)
  look      console-polarity share (meta "dark"), contrast p98 - p2, edge busyness (std of the
            top and bottom two rows) of the grey crop as stored
"""
import argparse
import glob
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval"))
from metrics import norm  # noqa: E402

PUNCT = {"……": "…", "――": "―", "「": "「", "」": "」", "『": "『", "』": "』", "！": "!", "？": "?",
         "、": "、", "。": "。", "～": "~", "♪": "♪", "（": "(", "ー": "ー", "っ": "っ"}
NAME_STOP = set("「」『』。、！？!?…―（）()")


def script(c):
    o = ord(c)
    if 0x3041 <= o <= 0x309F:
        return "hiragana"
    if 0x30A0 <= o <= 0x30FF:
        return "katakana"
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or c == "々":
        return "kanji"
    if c.isascii() and c.isalnum():
        return "latin"
    return "punct"


def expand(spec):
    out = []
    for g in spec.split("+"):
        hit = sorted(glob.glob(g))
        if not hit:
            raise SystemExit(f"no files match {g}")
        out += hit
    return out


def pct(values, qs=(10, 50, 90)):
    if not len(values):
        return None
    return [round(float(v), 2) for v in np.percentile(np.asarray(values, np.float64), qs)]


def hist(values, edges):
    counts = np.histogram(np.asarray(values, np.float64), bins=edges)[0]
    total = max(1, int(counts.sum()))
    return [round(int(c) / total, 4) for c in counts]


LEN_EDGES = [0, 2, 4, 6, 9, 13, 18, 24, 31, 41, 200]
H_EDGES = [0, 12, 14, 16, 18, 20, 23, 26, 29, 32, 36, 41, 48, 200]


def measure(paths, limit, rng):
    rows = []
    for p in paths:
        z = np.load(p)
        labels = json.loads(str(z["labels"]))
        meta = json.loads(str(z["meta"])) if "meta" in z.files else [{}] * len(labels)
        rows += [(p, i, labels[i], meta[i]) for i in range(len(labels)) if labels[i]]
    if limit and len(rows) > limit:
        idx = rng.choice(len(rows), limit, replace=False)
        rows = [rows[i] for i in sorted(idx)]
    by_file = defaultdict(list)
    for r in rows:
        by_file[r[0]].append(r)
    chars, scripts, lengths, namelike = Counter(), Counter(), [], 0
    punct = Counter()
    geo = defaultdict(lambda: defaultdict(list))
    look = defaultdict(list)
    dark = Counter()
    for p, rs in by_file.items():
        z = np.load(p)
        data, offs, shapes = z["data"], z["offsets"], z["shapes"]
        for _, i, lab, m in rs:
            t = norm(lab)
            lengths.append(len(t))
            chars.update(t)
            for c in t:
                scripts[script(c)] += 1
            for k, v in PUNCT.items():
                punct[v] += lab.count(k)
            if len(t) <= 6 and not (set(t) & NAME_STOP):
                namelike += 1
            h, w = (int(v) for v in shapes[i])
            g = data[offs[i]:offs[i + 1]].reshape(h, w)
            tgt = m.get("target") or ("vita" if m.get("platform") == "vita" else "psp" if m.get("platform") else "?")
            if m.get("platform") == "psp2x":
                tgt = "vita"
            geo[tgt]["height"].append(h)
            geo[tgt]["pitch"].append(w / max(1, len(t)) / h)
            ink = None
            if m.get("chars"):                     # teacher: [char, x0, y0, x1, y1, conf] in crop px
                bx = np.array([c[1:5] for c in m["chars"]], np.float64)
                ink = (bx[:, 0].min(), bx[:, 1].min(), bx[:, 2].max(), bx[:, 3].max())
            elif m.get("ink_box"):                 # synth3: [x0, y0, x1, y1] in crop px
                ink = m["ink_box"]
            if ink is not None:
                ih = max(1.0, ink[3] - ink[1])
                geo[tgt]["ink_h_ratio"].append(ih / h)
                geo[tgt]["top"].append(ink[1] / ih)
                geo[tgt]["bottom"].append((h - ink[3]) / ih)
                geo[tgt]["left"].append(ink[0] / ih)
                geo[tgt]["right"].append((w - ink[2]) / ih)
            dark[bool(m.get("dark"))] += 1
            gf = g.astype(np.float32)
            lo, hi = np.percentile(gf, (2, 98))
            look["contrast"].append(hi - lo)
            look["busy"].append(float(np.concatenate([gf[:2].ravel(), gf[-2:].ravel()]).std()))
            look["mean"].append(float(gf.mean()))
    n_chars = max(1, sum(chars.values()))
    out = {
        "lines": len(rows), "chars": n_chars, "distinct_chars": len(chars),
        "length": {"p10_50_90": pct(lengths), "hist_edges": LEN_EDGES, "hist": hist(lengths, LEN_EDGES),
                   "mean": round(float(np.mean(lengths)), 2) if lengths else None},
        "scripts": {k: round(v / n_chars, 4) for k, v in sorted(scripts.items())},
        "punct_per_1000": {k: round(1000 * v / n_chars, 2) for k, v in sorted(punct.items())},
        "namelike_share": round(namelike / max(1, len(rows)), 4),
        "dark_share": round(dark[True] / max(1, len(rows)), 4),
        "look": {k: pct(v) for k, v in look.items()},
        "geometry": {t: {"lines": len(v["height"]), "height_p10_50_90": pct(v["height"]),
                         "height_hist_edges": H_EDGES, "height_hist": hist(v["height"], H_EDGES),
                         **{k: pct(v[k]) for k in ("pitch", "ink_h_ratio", "top", "bottom", "left", "right")
                            if v[k]}}
                     for t, v in sorted(geo.items())},
        "char_freq": dict(chars.most_common()),
    }
    return out


def kl(p, q, eps=1e-6):
    keys = set(p) | set(q)
    sp, sq = sum(p.values()), sum(q.values())
    return sum((p.get(k, 0) / sp) * math.log((p.get(k, 0) / sp + eps) / (q.get(k, 0) / sq + eps))
               for k in keys if p.get(k, 0))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--name", action="append", required=True, help="NAME=GLOB[+GLOB...]")
    m.add_argument("--limit", type=int, default=20000)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", required=True)
    t = sub.add_parser("table")
    t.add_argument("stats")
    args = ap.parse_args()
    if args.cmd == "measure":
        rng = np.random.default_rng(args.seed)
        res = {}
        for spec in args.name:
            name, _, globs = spec.partition("=")
            paths = expand(globs)
            res[name] = measure(paths, args.limit, rng)
            res[name]["files"] = [str(p) for p in paths]
            print(f"{name}: {res[name]['lines']} lines", flush=True)
        if "real" in res:
            for name in res:
                if name != "real":
                    res[name]["kl_real_chars"] = round(kl(res["real"]["char_freq"], res[name]["char_freq"]), 4)
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    res = json.loads(Path(args.stats).read_text(encoding="utf-8"))
    names = list(res)
    print("| source | lines | len p10/50/90 | hira/kata/kanji/latin/punct | name-like | dark | KL(real‖src) |")
    print("|---|---|---|---|---|---|---|")
    for n in names:
        r = res[n]
        s = r["scripts"]
        print(f"| {n} | {r['lines']:,} | {'/'.join(str(int(v)) for v in r['length']['p10_50_90'])} | "
              + "/".join(f"{s.get(k, 0):.2f}" for k in ("hiragana", "katakana", "kanji", "latin", "punct"))
              + f" | {r['namelike_share']:.2f} | {r['dark_share']:.2f} | {r.get('kl_real_chars', '')} |")
    print()
    print("| source | target | height p10/50/90 | ink/crop | top/bottom margin (x ink h) | contrast p50 | busy p50 |")
    print("|---|---|---|---|---|---|---|")
    for n in names:
        for tg, g in res[n]["geometry"].items():
            hp = "/".join(str(int(v)) for v in g["height_p10_50_90"])
            ink = g.get("ink_h_ratio")
            tb = (f"{g['top'][1]:.2f}/{g['bottom'][1]:.2f}" if g.get("top") else "")
            print(f"| {n} | {tg} | {hp} | {ink[1] if ink else ''} | {tb} | {res[n]['look']['contrast'][1]} | "
                  f"{res[n]['look']['busy'][1]} |")


if __name__ == "__main__":
    main()
