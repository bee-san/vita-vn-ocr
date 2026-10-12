#!/usr/bin/env python3
"""Hard-example mining for synth3 on training-side real crops (never the selection or test sets).

  synth3_mine.py run --pool A/teacher/teacher_distill_train.npz --pool A/realdata/real_unlabelled.npz \
      --targets A/train2/targets --blob FL10=A/train2/release/FL10_w8.vocr \
      --blob H15=A/train2/release/H15_w8.vocr --out mining.json

The pools are real lines from training-split games that the label rules did not accept, with
every engine's reading in their meta (teacher/, realdata/). A line gets a reference only when
PP-OCRv6 small's reading (train2's teacher targets) agrees, after eval's normalisation, with at
least one other independent reader, and its lowest character confidence is at least --conf.
Each released int8 blob then reads the crop through the C runtime in W8I mode, as stored
(console polarity: dark text inverted). The report says which strata and which characters the
released models get wrong against that consensus. It steers synth3's style and character
weights; it is not an accuracy measurement (the references are machine consensus, and part of
the pool was a pseudo-label source in train2's fine-tune).

Every VN of eval's reserved list and every is_test_vn VN is refused, and so is any VN given with
--forbid-vns (the development cohort): mining never sees a selection or test game.
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "runtime"))
from metrics import align_text, lev, norm  # noqa: E402

from common import is_test_vn  # noqa: E402

READERS = ("meiki", "meikiv", "manga", "screenai", "v5s")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reserved_vns():
    doc = json.loads((ROOT / "eval" / "manifests" / "test_games.json").read_text())
    return set(doc["all"])


def script_of(c):
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


def strata(meta, h, text):
    out = [f"target:{meta.get('target')}", f"kind:{meta.get('kind')}",
           f"pol:{meta.get('pol') or ('dark' if meta.get('dark') else 'light')}"]
    native_h = h
    out.append("h:<14" if native_h < 14 else "h:14-17" if native_h < 18 else "h:18-23" if native_h < 24
               else "h:24-31" if native_h < 32 else "h:32+")
    n = len(text)
    out.append("len:1-4" if n <= 4 else "len:5-12" if n <= 12 else "len:13-24" if n <= 24 else "len:25+")
    if n <= 6 and not any(c in "「」『』。、！？…―" for c in text):
        out.append("namelike")
    return out


def load_pool(path, targets, conf_min, forbid):
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    tp = Path(targets) / (Path(path).stem + ".v6s.npz")
    t = np.load(tp)
    ttext, tconf = json.loads(str(t["text"])), t["conf"]
    rows, skipped = [], Counter()
    for i, m in enumerate(meta):
        vn = m.get("vn", "")
        if vn in forbid or is_test_vn(vn):
            skipped["forbidden_vn"] += 1
            continue
        v6 = norm(ttext[i]) if i < len(ttext) else ""
        if not v6:
            skipped["no_teacher"] += 1
            continue
        if float(tconf[i][0]) < conf_min:
            skipped["low_conf"] += 1
            continue
        reads = m.get("reads") or {}
        agree = [r for r in READERS if r in reads and norm(reads[r]) == v6]
        if not agree:
            skipped["no_agreement"] += 1
            continue
        h, w = (int(v) for v in shapes[i])
        rows.append({"pool": Path(path).stem, "i": i, "id": m.get("id"), "vn": vn, "meta": m,
                     "ref": v6, "agree": agree, "img": data[offs[i]:offs[i + 1]].reshape(h, w)})
    return rows, dict(skipped), {"file": str(path), "sha256": sha256(path), "targets": str(tp),
                                 "targets_sha256": sha256(tp)}


def main_run(args):
    from pyvocr import Vocr
    forbid = reserved_vns() | set(args.forbid_vns or [])
    rows, sources, skipped = [], [], {}
    for p in args.pool:
        r, s, src = load_pool(p, args.targets, args.conf, forbid)
        rows += r
        sources.append(src)
        skipped[Path(p).stem] = s
    print(f"{len(rows)} consensus lines; skipped {skipped}", flush=True)
    models = {}
    for spec in args.blob:
        name, _, path = spec.partition("=")
        models[name] = (Vocr(path, mode="w8i", max_width=1024), sha256(path))
    t0 = time.time()
    for name, (rt, _) in models.items():
        for r in rows:
            r[name] = norm(rt.text(r["img"]))
        print(f"{name}: {len(rows)} lines in {time.time() - t0:.0f}s", flush=True)
    report = {"kind": "synth3-mining", "conf_min": args.conf, "readers": READERS,
              "sources": sources, "skipped": skipped, "lines": len(rows),
              "models": {k: {"sha256": v[1]} for k, v in models.items()}, "by_model": {}}
    for name in models:
        tot_e = tot_n = exact = 0
        st = defaultdict(lambda: [0, 0, 0, 0])      # errors, chars, lines, exact
        char_occ, char_err = Counter(), Counter()
        conf = Counter()
        scripts = defaultdict(lambda: [0, 0])
        for r in rows:
            ref, pred = r["ref"], r[name]
            e = lev(pred, ref)
            tot_e, tot_n, exact = tot_e + e, tot_n + len(ref), exact + (e == 0)
            for s in strata(r["meta"], r["img"].shape[0], ref):
                v = st[s]
                v[0] += e
                v[1] += len(ref)
                v[2] += 1
                v[3] += e == 0
            for g, p in align_text(ref, pred):
                if g is not None:
                    char_occ[g] += 1
                    scripts[script_of(g)][1] += 1
                if g != p:
                    conf[(g or "", p or "")] += 1
                    if g is not None:
                        char_err[g] += 1
                        scripts[script_of(g)][0] += 1
        hard = sorted(((c, char_err[c], char_occ[c]) for c in char_occ if char_occ[c] >= args.min_occ),
                      key=lambda x: (-(x[1] / x[2]), -x[1]))
        report["by_model"][name] = {
            "cer": round(tot_e / max(tot_n, 1), 5), "errors": tot_e, "chars": tot_n,
            "exact": round(exact / max(len(rows), 1), 4),
            "strata": {k: {"errors": v[0], "chars": v[1], "lines": v[2], "exact": v[3],
                           "cer": round(v[0] / max(v[1], 1), 5)} for k, v in sorted(st.items())},
            "scripts": {k: {"errors": v[0], "chars": v[1], "cer": round(v[0] / max(v[1], 1), 5)}
                        for k, v in sorted(scripts.items())},
            "confusions": [{"ref": g, "pred": p, "count": n} for (g, p), n in conf.most_common(args.top)],
            "hard_chars": [{"char": c, "errors": e, "occurrences": n, "rate": round(e / n, 4)}
                           for c, e, n in hard[:args.top]],
            "char_errors": {c: [char_err[c], char_occ[c]] for c in char_occ},
        }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pool", action="append", required=True)
    r.add_argument("--targets", required=True, help="train2 teacher targets dir (<stem>.v6s.npz)")
    r.add_argument("--blob", action="append", required=True, help="NAME=path to an int8 .vocr blob")
    r.add_argument("--conf", type=float, default=0.9)
    r.add_argument("--forbid-vns", nargs="*", help="development VNs that mining must never read")
    r.add_argument("--min-occ", type=int, default=8)
    r.add_argument("--top", type=int, default=300)
    r.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        main_run(args)


if __name__ == "__main__":
    main()
