#!/usr/bin/env python3
"""synth3's controlled ablation: export, C W8I scoring on the development cohort, paired bootstraps.

  synth3_ablation.py export  --ckpt runs/RUN/ckpt_step16000.pt --out RUN_w8.vocr
  synth3_ablation.py score   --blob s2_s1=A_w8.vocr --blob s3_s1=B_w8.vocr ... --dev teacher_val.npz \
                             --out scores.json
  synth3_ablation.py compare --scores scores.json --protocol eval/synth3_ablation_protocol.json --out result.json

The development cohort is teacher's teacher_val (1,651 real lines, 181 VNs; train2's selection
set; no VN, release or screenshot shared with any training pool or with eval's test sets). Each
line is read twice through the deployment C runtime in W8I mode (runtime/pyvocr.py): as stored
(`rv`, the console polarity: dark text inverted) and in screen polarity (`rvn`, the inversion
undone), the two views train2 selects on. CER is eval's (metrics.norm both sides, Levenshtein
over reference characters).

Strata are derived from the cohort's own metadata and labels, never from a model's output:
platform target and capture kind (native Vita, native PSP, PC/console screenshots scaled to
either), polarity, ink height (teacher's character boxes), label length, name-like lines (at
most 6 characters, no dialogue punctuation), lines with dakuten/handakuten/small-kana
characters, rare kanji (outside the 2,000 most frequent Jiten kanji), Latin letters or digits.

The comparison follows the frozen protocol file: paired bootstrap of the CER difference with
VNs resampled as clusters (and lines, for reference), per stratum and per seed, plus per-VN
win/loss counts. It prints the protocol's decision; it does not choose anything else.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "runtime"))
sys.path.insert(0, str(ROOT / "train"))
from metrics import lev, norm  # noqa: E402

DAKUTEN = set("がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポヴ"
              "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮ")
PUNCT = set("「」『』。、！？!?…―（）")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_kanji(c):
    o = ord(c)
    return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or c == "々"


def kanji_ranks():
    p = ROOT / "data" / "manifests" / "fl10_kanji_policy_20261011.json"
    return json.loads(p.read_text(encoding="utf-8"))["ranks"]


def strata(meta, label, ranks):
    """Style tags of one development line, from its metadata and reference only."""
    tags = [f"target:{meta['target']}", f"kind:{meta['kind']}", f"pol:{meta.get('pol')}"]
    chars = meta.get("chars") or []
    if chars:
        ink = max(c[4] for c in chars) - min(c[2] for c in chars)
        tags.append("ink:<14" if ink < 14 else "ink:14-19" if ink < 20 else "ink:20-27" if ink < 28 else "ink:28+")
    t = norm(label)
    n = len(t)
    tags.append("len:1-4" if n <= 4 else "len:5-12" if n <= 12 else "len:13+")
    if n <= 6 and not set(t) & PUNCT:
        tags.append("namelike")
    if set(t) & DAKUTEN:
        tags.append("dakuten-smallkana")
    if any(is_kanji(c) and ranks.get(c, 99999) > 2000 for c in t):
        tags.append("rare-kanji")
    if any(c.isascii() and c.isalnum() for c in t):
        tags.append("latin-digits")
    return tags


def load_dev(path):
    z = np.load(path)
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    labels = json.loads(str(z["labels"]))
    meta = json.loads(str(z["meta"]))
    ranks = kanji_ranks()
    rows = []
    for i, (lab, m) in enumerate(zip(labels, meta)):
        if not lab:
            continue
        h, w = (int(v) for v in shapes[i])
        g = data[offs[i]:offs[i + 1]].reshape(h, w)
        rows.append({"id": m["id"], "vn": m["vn"], "ref": norm(lab), "img": g, "inverted": bool(m.get("inverted")),
                     "tags": strata(m, lab, ranks)})
    return rows


def cmd_export(args):
    import export
    model, vocab, height = export.load_checkpoint(args.ckpt)
    blob, info = export.build_blob(export.model_ops(model, height), vocab, height, True,
                                   source_sha=hashlib.sha256(Path(args.ckpt).read_bytes()).digest())
    Path(args.out).write_bytes(blob)
    print(json.dumps({"ckpt": args.ckpt, "ckpt_sha256": sha256(args.ckpt), "blob": args.out,
                      "blob_sha256": sha256(args.out), "bytes": len(blob)}))


def cmd_score(args):
    from pyvocr import Vocr
    rows = load_dev(args.dev)
    out = {"kind": "synth3-dev-scores", "dev": {"file": str(args.dev), "sha256": sha256(args.dev), "lines": len(rows),
                                                 "chars": sum(len(r["ref"]) for r in rows),
                                                 "vns": len({r["vn"] for r in rows})},
           "lines": [{"id": r["id"], "vn": r["vn"], "ref": r["ref"], "tags": r["tags"]} for r in rows], "models": {}}
    for spec in args.blob:
        name, _, path = spec.partition("=")
        rt = Vocr(path, mode="w8i", max_width=1024)
        preds = {"rv": [], "rvn": []}
        for r in rows:
            g = r["img"]
            preds["rv"].append(norm(rt.text(g)))
            preds["rvn"].append(norm(rt.text(255 - g if r["inverted"] else g)))
        errs = {k: [lev(p, r["ref"]) for p, r in zip(v, rows)] for k, v in preds.items()}
        n = sum(len(r["ref"]) for r in rows)
        out["models"][name] = {"blob": path, "blob_sha256": sha256(path), "blob_bytes": rt.blob_bytes,
                               "arena_bytes": rt.arena_bytes, "pred": preds, "errors": errs,
                               "cer": {k: round(sum(v) / n, 5) for k, v in errs.items()}}
        print(f"{name}: rv {sum(errs['rv']) / n:.4f} rvn {sum(errs['rvn']) / n:.4f}", flush=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


def arm_errors(scores, runs, views):
    """Per-line error counts of an arm: summed over its runs (seeds) and views (rv, rvn)."""
    e = None
    for run in runs:
        for v in views:
            x = np.asarray(scores["models"][run]["errors"][v], np.float64)
            e = x if e is None else e + x
    return e


def bootstrap(ea, eb, chars, groups, n_boot, seed, scale):
    """CER(B) - CER(A) in points, with a 95% interval, resampling `groups` (cluster ids)."""
    d = (eb.sum() - ea.sum()) / (chars.sum() * scale) * 100
    ug, inv = np.unique(groups, return_inverse=True)
    ga = np.bincount(inv, weights=ea, minlength=len(ug))
    gb = np.bincount(inv, weights=eb, minlength=len(ug))
    gc = np.bincount(inv, weights=chars, minlength=len(ug))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ug), size=(n_boot, len(ug)))
    diffs = (gb[idx].sum(1) - ga[idx].sum(1)) / (gc[idx].sum(1) * scale) * 100
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"delta_points": round(float(d), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "cer_a": round(float(ea.sum() / (chars.sum() * scale)), 5),
            "cer_b": round(float(eb.sum() / (chars.sum() * scale)), 5)}


def cmd_compare(args):
    scores = {"models": {}}
    lines = None
    for p in args.scores:
        s = json.loads(Path(p).read_text(encoding="utf-8"))
        if lines is None:
            lines, dev = s["lines"], s["dev"]
        elif [x["id"] for x in s["lines"]] != [x["id"] for x in lines]:
            raise SystemExit("score files hold different development lines")
        scores["models"].update(s["models"])
    proto = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    chars = np.array([len(x["ref"]) for x in lines], np.float64)
    vns = np.array([x["vn"] for x in lines])
    line_ids = np.arange(len(lines))
    views = proto["views"]
    B, seed = proto["bootstrap"]["resamples"], proto["bootstrap"]["seed"]
    res = {"kind": "synth3-ablation-result", "protocol_sha256": sha256(args.protocol), "dev": dev,
           "models": {k: {kk: v[kk] for kk in ("blob_sha256", "blob_bytes", "arena_bytes", "cer")}
                      for k, v in scores["models"].items()},
           "comparisons": {}}
    tags = sorted({t for x in lines for t in x["tags"]})
    for cmp_name, (a, b) in proto["comparisons"].items():
        runs_a, runs_b = proto["arms"][a], proto["arms"][b]
        if not all(r in scores["models"] for r in runs_a + runs_b):
            continue
        out = {"a": a, "b": b, "runs_a": runs_a, "runs_b": runs_b}
        scale = len(runs_a) * len(views)
        if len(runs_b) != len(runs_a):
            raise SystemExit(f"{cmp_name}: arms need the same number of runs")
        ea, eb = arm_errors(scores, runs_a, views), arm_errors(scores, runs_b, views)
        out["all_vn_cluster"] = bootstrap(ea, eb, chars, vns, B, seed, scale)
        out["all_line"] = bootstrap(ea, eb, chars, line_ids, B, seed, scale)
        out["per_view"] = {v: bootstrap(arm_errors(scores, runs_a, [v]), arm_errors(scores, runs_b, [v]), chars, vns,
                                        B, seed, len(runs_a)) for v in views}
        out["per_seed"] = {}
        for ra, rb in zip(runs_a, runs_b):
            out["per_seed"][f"{ra}->{rb}"] = bootstrap(arm_errors(scores, [ra], views), arm_errors(scores, [rb], views),
                                                       chars, vns, B, seed, len(views))
        out["strata"] = {}
        for t in tags:
            m = np.array([t in x["tags"] for x in lines])
            if m.sum() < 5:
                continue
            r = bootstrap(ea[m], eb[m], chars[m], vns[m], B, seed, scale)
            r["lines"] = int(m.sum())
            out["strata"][t] = r
        per_vn = defaultdict(lambda: [0.0, 0.0, 0.0])
        for i, v in enumerate(vns):
            per_vn[v][0] += ea[i]
            per_vn[v][1] += eb[i]
            per_vn[v][2] += chars[i]
        better = sum(1 for x in per_vn.values() if x[1] < x[0])
        worse = sum(1 for x in per_vn.values() if x[1] > x[0])
        regress = sorted(((x[1] - x[0]) / (x[2] * scale) * 100, v, x[2]) for v, x in per_vn.items())[::-1][:10]
        out["per_vn"] = {"vns": len(per_vn), "better": better, "worse": worse, "equal": len(per_vn) - better - worse,
                         "largest_regressions": [{"vn": v, "delta_points": round(d, 2), "chars": int(c)}
                                                 for d, v, c in regress if d > 0]}
        rule = proto["decision"]
        ok_ci = out["all_vn_cluster"]["ci95"][1] < 0
        ok_seeds = all(x["delta_points"] < 0 for x in out["per_seed"].values())
        out["decision"] = {"b_better": bool(ok_ci and ok_seeds), "ci_upper_below_zero": bool(ok_ci),
                           "every_seed_below_zero": bool(ok_seeds), "rule": rule}
        res["comparisons"][cmp_name] = out
        a_all = out["all_vn_cluster"]
        print(f"{cmp_name}: {b} - {a} = {a_all['delta_points']:+.3f} points "
              f"[{a_all['ci95'][0]:+.3f}, {a_all['ci95'][1]:+.3f}] (CER {a_all['cer_a']:.4f} -> {a_all['cer_b']:.4f}); "
              f"VNs better {better} / worse {worse}; decision {out['decision']['b_better']}", flush=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--blob", action="append", required=True)
    s.add_argument("--dev", required=True)
    s.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--scores", nargs="+", required=True)
    c.add_argument("--protocol", required=True)
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    {"export": cmd_export, "score": cmd_score, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
