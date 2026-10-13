#!/usr/bin/env python3
"""Style and text statistics of line-crop shards, for calibrating synth3 against real crops.

  synth3_stats.py measure --name real=A/teacher/teacher_train.npz+A/realdata/real_train.npz \
      --name s2=A/synth2/shards/synth_0000[0-1].npz --name s3='S3/shards/train/synth3_0000[0-1].npz' \
      --out stats.json [--limit 20000]
  synth3_stats.py table stats.json            # markdown tables (docs/synth3.md, docs/synth3-v3.md)

Only training-side real crops may be measured here (teacher_train, real_train): never the
selection set and never eval's test sets. Rows of a reserved VN are dropped and counted
(eval's test games, synth/common.is_test_vn, and every VN of the exclusion manifests, by default
data/manifests/synth3_exclude_fresh_native.json, synth3_exclude_clannad.json and
synth3_exclude_owner_games.json; add the
development VNs with --exclude ... exclude_dev_vns.txt). Statistics per source:

  text      label length histogram, script shares (hiragana/katakana/kanji/latin/punct),
            VN punctuation per 1,000 characters, name-like line share, distinct characters,
            bracket line shapes (opens with 「/『, closes with 」/』, both), lines holding Latin
            letters or digits, the character unigram distribution (for KL divergence)
  geometry  per platform target and per capture kind (real: vita | psp | hires; synth3:
            vita | psp | psp2x | hires for its area-scaled renders): crop height, width per
            character, and the ink box and margins
            - "top/bottom/left/right": from teacher's MeikiOCR character boxes for real crops
              and from the renderer's ink box for synth3 (version 1-2 calibration; the two
              boxes are not the same measure: MeikiOCR's boxes sit 1-2 px inside the ink at the
              right and bottom, docs/synth3-v3.md)
            - "px": one pixel-ink measure for every source (px_ink(): teacher/geometry.py
              refine()'s rule run on the stored crop)
  look      console-polarity share (meta "dark"), contrast p98 - p2, edge busyness (std of the
            top and bottom two rows) of the grey crop as stored, and its parts: the share of
            those edge pixels that are ink (a neighbour cut by the crop), the std of the other
            edge pixels (box and scene texture), and |mean(top rows) - mean(bottom rows)|
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
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "eval"))
from metrics import norm  # noqa: E402

PUNCT = {"……": "…", "――": "―", "「": "「", "」": "」", "『": "『", "』": "』", "！": "!", "？": "?",
         "、": "、", "。": "。", "～": "~", "♪": "♪", "（": "(", "ー": "ー", "っ": "っ"}
NAME_STOP = set("「」『』。、！？!?…―（）()")
DEFAULT_EXCLUDE = (ROOT / "data" / "manifests" / "synth3_exclude_fresh_native.json",
                   ROOT / "data" / "manifests" / "synth3_exclude_clannad.json",
                   ROOT / "data" / "manifests" / "synth3_exclude_owner_games.json")
KIND_OF_PLATFORM = {"vita": "vita", "psp": "psp", "psp2x": "psp2x", "scaled": "hires"}


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


def px_ink(g):
    """(x0, y0, x1, y1) of a crop's text ink, found the same way for every source: teacher's
    refine() rule (teacher/geometry.py) on the stored grey crop. Background = median of the
    border ring; ink = pixels deviating by more than max(28, 0.4 x their p99); rows with ink in
    at least max(2, 1%) of the columns, split into runs at gaps over 3 rows; the longest run is
    the line, its columns with any ink the line's extent. None when no row qualifies."""
    g = np.asarray(g, np.float32)
    h, w = g.shape
    if h < 6 or w < 6:
        return None
    ring = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    d = np.abs(g - float(np.median(ring)))
    m = d > max(28.0, 0.4 * float(np.percentile(d, 99)))
    rows = np.flatnonzero(m.sum(1) >= max(2, 0.01 * w))
    if not len(rows):
        return None
    runs, s, last = [], rows[0], rows[0]
    for r in rows[1:]:
        if r - last > 3:
            runs.append((s, last))
            s = r
        last = r
    runs.append((s, last))
    r0, r1 = max(runs, key=lambda r: r[1] - r[0])
    cols = np.flatnonzero(m[r0:r1 + 1].any(axis=0))
    return int(cols[0]), int(r0), int(cols[-1]) + 1, int(r1) + 1


def edge_parts(g):
    """Edge busyness and its parts: (std of the top and bottom two rows, share of those pixels
    that are ink, std of the others, |mean of the top two rows - mean of the bottom two|)."""
    g = np.asarray(g, np.float32)
    e = np.concatenate([g[:2].ravel(), g[-2:].ravel()])
    ring = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    bg = float(np.median(ring))
    d = np.abs(g - bg)
    t = max(28.0, 0.4 * float(np.percentile(d, 99)))
    de = np.abs(e - bg)
    rest = e[de <= t]
    return (float(e.std()), float((de > t).mean()), float(rest.std()) if rest.size > 4 else 0.0,
            abs(float(g[:2].mean() - g[-2:].mean())))


def reserved_vns(paths):
    """VN ids no statistic may use: eval's test games plus every exclusion manifest's VNs."""
    out = set(json.loads((ROOT / "eval" / "manifests" / "test_games.json").read_text())["all"])
    for p in paths:
        p = Path(p)
        if p.suffix == ".json":
            out |= set(json.loads(p.read_text(encoding="utf-8"))["vns"])
        else:
            out |= set(p.read_text().split())
    return out


def kind_of(m):
    """Capture kind: real crops carry it (vita | psp | hires); synth3 crops get it from the
    platform (scaled renders are the synthetic counterpart of hires)."""
    if m.get("real") or m.get("teacher"):
        return m.get("kind") or "?"
    return KIND_OF_PLATFORM.get(m.get("platform"), "?")


def measure(paths, limit, rng, reserved=frozenset()):
    from common import is_test_vn
    rows, dropped = [], 0
    for p in paths:
        z = np.load(p)
        labels = json.loads(str(z["labels"]))
        meta = json.loads(str(z["meta"])) if "meta" in z.files else [{}] * len(labels)
        for i in range(len(labels)):
            if not labels[i]:
                continue
            vn = meta[i].get("vn")
            if vn and (vn in reserved or is_test_vn(vn)):
                dropped += 1
                continue
            rows.append((p, i, labels[i], meta[i]))
    if limit and len(rows) > limit:
        idx = rng.choice(len(rows), limit, replace=False)
        rows = [rows[i] for i in sorted(idx)]
    by_file = defaultdict(list)
    for r in rows:
        by_file[r[0]].append(r)
    chars, scripts, lengths, namelike = Counter(), Counter(), [], 0
    punct, shapes_ = Counter(), Counter()
    latin_lines = 0
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
            latin_lines += any(script(c) == "latin" for c in t)
            for k, v in PUNCT.items():
                punct[v] += lab.count(k)
            o, c_ = lab[:1] in ("「", "『"), lab[-1:] in ("」", "』")
            shapes_["opens"] += o
            shapes_["closes"] += c_
            shapes_["both"] += o and c_
            if len(t) <= 6 and not (set(t) & NAME_STOP):
                namelike += 1
            h, w = (int(v) for v in shapes[i])
            g = data[offs[i]:offs[i + 1]].reshape(h, w)
            tgt = m.get("target") or ("vita" if m.get("platform") == "vita" else "psp" if m.get("platform") else "?")
            if m.get("platform") == "psp2x":
                tgt = "vita"
            ink = None
            if m.get("chars"):                     # teacher: [char, x0, y0, x1, y1, conf] in crop px
                bx = np.array([c[1:5] for c in m["chars"]], np.float64)
                ink = (bx[:, 0].min(), bx[:, 1].min(), bx[:, 2].max(), bx[:, 3].max())
            elif m.get("ink_box"):                 # synth3: [x0, y0, x1, y1] in crop px
                ink = m["ink_box"]
            px = px_ink(g)
            for key in (tgt, f"{tgt}:{kind_of(m)}"):
                G = geo[key]
                G["height"].append(h)
                G["pitch"].append(w / max(1, len(t)) / h)
                if ink is not None:
                    ih = max(1.0, ink[3] - ink[1])
                    G["ink_h"].append(ih)
                    G["ink_h_ratio"].append(ih / h)
                    G["top"].append(ink[1] / ih)
                    G["bottom"].append((h - ink[3]) / ih)
                    G["left"].append(ink[0] / ih)
                    G["right"].append((w - ink[2]) / ih)
                if px is not None:
                    ph = max(1, px[3] - px[1])
                    G["px_ink_h"].append(ph)
                    G["px_ink_h_ratio"].append(ph / h)
                    G["px_top"].append(px[1] / ph)
                    G["px_bottom"].append((h - px[3]) / ph)
                    G["px_left"].append(px[0] / ph)
                    G["px_right"].append((w - px[2]) / ph)
            dark[bool(m.get("dark"))] += 1
            gf = g.astype(np.float32)
            lo, hi = np.percentile(gf, (2, 98))
            look["contrast"].append(hi - lo)
            busy, edge_ink, tex, tb = edge_parts(gf)
            look["busy"].append(busy)
            look["edge_ink"].append(edge_ink)
            look["edge_tex"].append(tex)
            look["edge_top_bottom"].append(tb)
            look["mean"].append(float(gf.mean()))
            for key in (tgt, f"{tgt}:{kind_of(m)}"):
                geo[key]["busy"].append(busy)
                geo[key]["edge_ink"].append(edge_ink)
                geo[key]["edge_tex"].append(tex)
    n_chars = max(1, sum(chars.values()))
    n = max(1, len(rows))

    def geometry(v):
        out = {"lines": len(v["height"]), "height_p10_50_90": pct(v["height"]), "height_hist_edges": H_EDGES,
               "height_hist": hist(v["height"], H_EDGES)}
        out.update({k: pct(v[k]) for k in ("pitch", "ink_h", "ink_h_ratio", "top", "bottom", "left", "right")
                    if v[k]})
        if v["px_ink_h"]:
            out["px"] = {k[3:]: pct(v[k]) for k in ("px_ink_h", "px_ink_h_ratio", "px_top", "px_bottom", "px_left",
                                                     "px_right")}
        if v["busy"]:
            out["busy_p10_50_90"] = pct(v["busy"])
            out["edge_ink_mean"] = round(float(np.mean(v["edge_ink"])), 4)
            out["edge_tex_p10_50_90"] = pct(v["edge_tex"])
        return out
    out = {
        "lines": len(rows), "chars": n_chars, "distinct_chars": len(chars), "reserved_rows_dropped": dropped,
        "length": {"p10_50_90": pct(lengths), "hist_edges": LEN_EDGES, "hist": hist(lengths, LEN_EDGES),
                   "mean": round(float(np.mean(lengths)), 2) if lengths else None},
        "scripts": {k: round(v / n_chars, 4) for k, v in sorted(scripts.items())},
        "latin_per_1000": round(1000 * scripts["latin"] / n_chars, 2),
        "latin_line_share": round(latin_lines / n, 4),
        "punct_per_1000": {k: round(1000 * v / n_chars, 2) for k, v in sorted(punct.items())},
        "bracket_lines": {k: round(v / n, 4) for k, v in sorted(shapes_.items())},
        "namelike_share": round(namelike / n, 4),
        "dark_share": round(dark[True] / n, 4),
        "look": {k: pct(v) for k, v in look.items()},
        "edge_ink_mean": round(float(np.mean(look["edge_ink"])), 4) if look["edge_ink"] else None,
        "geometry": {t: geometry(v) for t, v in sorted(geo.items())},
        "char_freq": dict(chars.most_common()),
    }
    return out


def kl(p, q, eps=1e-6):
    keys = set(p) | set(q)
    sp, sq = sum(p.values()), sum(q.values())
    return sum((p.get(k, 0) / sp) * math.log((p.get(k, 0) / sp + eps) / (q.get(k, 0) / sq + eps))
               for k in keys if p.get(k, 0))


def table(res):
    names = list(res)
    print("| source | lines | len p10/50/90 | hira/kata/kanji/latin/punct | Latin / 1,000 | name-like | dark | "
          "KL(real‖src) |")
    print("|---|---|---|---|---|---|---|---|")
    for n in names:
        r = res[n]
        s = r["scripts"]
        print(f"| {n} | {r['lines']:,} | {'/'.join(str(int(v)) for v in r['length']['p10_50_90'])} | "
              + "/".join(f"{s.get(k, 0):.2f}" for k in ("hiragana", "katakana", "kanji", "latin", "punct"))
              + f" | {r.get('latin_per_1000', '')} | {r['namelike_share']:.2f} | {r['dark_share']:.2f} | "
              f"{r.get('kl_real_chars', '')} |")
    print()
    print("| source | 「 / 」 per 1,000 | lines opening / closing / both | busy p50 | edge ink (mean) | "
          "edge texture p50 | top-bottom p50 |")
    print("|---|---|---|---|---|---|---|")
    for n in names:
        r = res[n]
        p, b, lk = r["punct_per_1000"], r.get("bracket_lines", {}), r["look"]
        print(f"| {n} | {p.get('「', 0)} / {p.get('」', 0)} | {b.get('opens', '')} / {b.get('closes', '')} / "
              f"{b.get('both', '')} | {lk['busy'][1]} | {r.get('edge_ink_mean', '')} | "
              f"{(lk.get('edge_tex') or [None, ''])[1]} | {(lk.get('edge_top_bottom') or [None, ''])[1]} |")
    print()
    print("| source | target[:kind] | crops | height p10/50/90 | px ink p10/50/90 | px margin t/b/l/r p50 | "
          "busy p50 | edge ink | edge texture p50 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for n in names:
        for tg, g in res[n]["geometry"].items():
            hp = "/".join(str(int(v)) for v in g["height_p10_50_90"])
            px = g.get("px")
            ih = "/".join(f"{v:g}" for v in px["ink_h"]) if px else ""
            mg = "/".join(f"{px[k][1]:.2f}" for k in ("top", "bottom", "left", "right")) if px else ""
            bz = g.get("busy_p10_50_90", [None, ""])[1]
            tx = g.get("edge_tex_p10_50_90", [None, ""])[1]
            print(f"| {n} | {tg} | {g['lines']:,} | {hp} | {ih} | {mg} | {bz} | {g.get('edge_ink_mean', '')} | {tx} |")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--name", action="append", required=True, help="NAME=GLOB[+GLOB...]")
    m.add_argument("--limit", type=int, default=20000)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--exclude", nargs="*", default=[],
                   help="more exclusion manifests (.json with 'vns') or VN-id lists (.txt) whose rows are dropped "
                        "(always: eval's test games and the default manifests)")
    m.add_argument("--out", required=True)
    t = sub.add_parser("table")
    t.add_argument("stats")
    args = ap.parse_args()
    if args.cmd == "measure":
        rng = np.random.default_rng(args.seed)
        reserved = reserved_vns(list(DEFAULT_EXCLUDE) + list(args.exclude))
        res = {}
        for spec in args.name:
            name, _, globs = spec.partition("=")
            paths = expand(globs)
            res[name] = measure(paths, args.limit, rng, reserved)
            res[name]["files"] = [str(p) for p in paths]
            print(f"{name}: {res[name]['lines']} lines ({res[name]['reserved_rows_dropped']} reserved rows dropped)",
                  flush=True)
        if "real" in res:
            for name in res:
                if name != "real":
                    res[name]["kl_real_chars"] = round(kl(res["real"]["char_freq"], res[name]["char_freq"]), 4)
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    table(json.loads(Path(args.stats).read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
