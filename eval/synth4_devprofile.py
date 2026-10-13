#!/usr/bin/env python3
"""Error profile of the VOCR3 selections on the development cohorts: report only (docs/synth4.md).

  synth4_devprofile.py --select-dir $S3/vocr3/select --model FL10=rp_FL10_C_mix_pre128k_seed1 \
      --model H15=rp_H15_C_mix_seed2 --curriculum CURR.json --fl10-vocab V.txt --out OUT.json

The task asked which characters and styles the current selections get wrong on teacher_val and
devnative. Mining them for synth4's design would bias the synth4 ablation, which scores on those
cohorts (conductor, 2026-10-13 14:45Z), so this runs only after synth4's design and its protocol
were committed (eval/synth4_protocol.json), and reads the selection's existing score files (the
VOCR3 selection's C-runtime W8I reads, console polarity "rv"); it scores nothing itself.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import align_text  # noqa: E402
from synth4_ablation import curriculum_tags, norm  # noqa: E402

FILES = ["teacher_val", "devnative_strict_ink", "devnative_strict_grad", "devnative_tv_native_ink",
         "devnative_tv_native_grad"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    return "other"


def profile(lines, preds, errs, tail, fl10):
    tag_e, tag_n = Counter(), Counter()
    ref_n, err, conf, by_s = Counter(), Counter(), Counter(), defaultdict(lambda: [0, 0])
    for x, p, e in zip(lines, preds, errs):
        tags = x["tags"] + curriculum_tags(x["ref"], tail, fl10)
        for t in tags + ["all"]:
            tag_e[t] += e
            tag_n[t] += len(x["ref"])
        for a, b in align_text(x["ref"], p):
            if a is None:
                continue
            ref_n[a] += 1
            if b != a:
                err[a] += 1
                if b is not None:
                    conf[(a, b)] += 1
    for c, n in ref_n.items():
        by_s[script(c)][0] += err[c]
        by_s[script(c)][1] += n
    return {"strata": {t: {"chars": tag_n[t], "cer": round(tag_e[t] / max(1, tag_n[t]), 5)} for t in sorted(tag_n)},
            "scripts": {k: {"chars": v[1], "err_rate": round(v[0] / max(1, v[1]), 5)} for k, v in by_s.items()},
            "confusions_top": [[a, b, k] for (a, b), k in conf.most_common(30)],
            "char_errors_top": [[c, err[c], ref_n[c]] for c in sorted(err, key=lambda c: -err[c])[:40]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select-dir", required=True)
    ap.add_argument("--model", action="append", required=True, help="SIZE=RUN (SIZE: the select/ subdirectory)")
    ap.add_argument("--curriculum", required=True)
    ap.add_argument("--fl10-vocab", required=True)
    ap.add_argument("--view", default="rv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cur = json.loads(Path(a.curriculum).read_text(encoding="utf-8"))
    tail = {norm(c) for c, d in cur["deficit"].items() if d > 0 and norm(c)}
    fl10 = {norm(c) for c in Path(a.fl10_vocab).read_text(encoding="utf-8").split("\n") if c}
    out = {"kind": "synth4-devprofile", "view": a.view, "models": {}}
    for spec in a.model:
        size, _, run = spec.partition("=")
        res = {}
        for f in FILES:
            p = Path(a.select_dir) / size / f"{f}.json"
            s = json.loads(p.read_text(encoding="utf-8"))
            m = s["models"][run]
            r = profile(s["lines"], m["pred"][a.view], m["errors"][a.view], tail, fl10)
            r.update(file=str(p), sha256=sha256(p), dev=s["dev"], blob_sha256=m["blob_sha256"])
            res[f] = r
        dn = [Path(a.select_dir) / size / f"{f}.json" for f in FILES[1:]]
        lines, preds, errs = [], [], []
        for p in dn:
            s = json.loads(p.read_text(encoding="utf-8"))
            lines += s["lines"]
            preds += s["models"][run]["pred"][a.view]
            errs += s["models"][run]["errors"][a.view]
        res["devnative_pooled"] = profile(lines, preds, errs, tail, fl10)
        out["models"][f"{size}:{run}"] = res
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for k, v in out["models"].items():
        print(k, {f: v[f]["strata"]["all"] for f in ("teacher_val", "devnative_pooled")}, flush=True)


if __name__ == "__main__":
    main()
