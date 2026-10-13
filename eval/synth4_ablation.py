#!/usr/bin/env python3
"""synth4's pre-registered ablation (eval/synth4_protocol.json): the strata it adds and its result.

  synth4_ablation.py tag --scores S.json --curriculum CURR.json --fl10-vocab V.txt --out S_tagged.json
  synth4_ablation.py result --protocol eval/synth4_protocol.json --tv COMPARE_TV.json \
      --dn COMPARE_DN_POOLED.json [--dn-file NAME=COMPARE.json ...] --out eval/synth4_ablation_20261013.json

Scoring and the comparison itself are eval/synth3_ablation.py's (score, compare) and
eval/synth3_study2.py merge's, unchanged. `tag` adds three strata to a score file, fixed by the
protocol and derived from the references only (never from a model's output):

  tail-kanji   the line holds a kanji the curriculum lifts: fewer than K exposures in synth3 v3's
               1,000,000 pinned crops (K = 80 for a class VN text uses, 25 otherwise; the
               curriculum file's deficit > 0)
  tail-any     the line holds any such character
  fl10-unk     the line holds a character outside FL10's 3,500-class vocab (read as <unk> by every
               arm, so no arm can read it exactly)

`result` applies the protocol's two-part decision to the comparisons and writes the result file.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import norm  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_kanji(c):
    o = ord(c)
    return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or c == "々"


def curriculum_tags(ref, tail, fl10):
    """The protocol's extra strata of one reference (normalised as eval scores it)."""
    tags = []
    if any(is_kanji(c) and c in tail for c in ref):
        tags.append("tail-kanji")
    if any(c in tail for c in ref):
        tags.append("tail-any")
    if any(c not in fl10 for c in ref):
        tags.append("fl10-unk")
    return tags


def cmd_tag(args):
    s = json.loads(Path(args.scores).read_text(encoding="utf-8"))
    cur = json.loads(Path(args.curriculum).read_text(encoding="utf-8"))
    tail = {norm(c) for c, d in cur["deficit"].items() if d > 0 and norm(c)}
    fl10 = {norm(c) for c in Path(args.fl10_vocab).read_text(encoding="utf-8").split("\n") if c}
    n = 0
    for x in s["lines"]:
        extra = [t for t in curriculum_tags(x["ref"], tail, fl10) if t not in x["tags"]]
        x["tags"] = x["tags"] + extra
        n += bool(extra)
    s["synth4_tags"] = {"curriculum_sha256": sha256(args.curriculum), "fl10_vocab_sha256": sha256(args.fl10_vocab),
                        "tail_classes": len(tail), "lines_tagged": n}
    Path(args.out).write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
    print(f"{n} of {len(s['lines'])} lines carry a synth4 stratum -> {args.out}", flush=True)


def decide(tv, dn):
    """The protocol's decision from the two comparison results (eval/synth3_ablation.py compare)."""
    a = tv["all_vn_cluster"]
    seeds = [x["delta_points"] for x in tv["per_seed"].values()]
    tv_better = a["ci95"][1] < 0 and all(d < 0 for d in seeds)
    d = dn["all_vn_cluster"]
    dn_not_worse = not d["ci95"][0] > 0
    dn_better = d["ci95"][1] < 0
    return {"teacher_val_better": bool(tv_better), "devnative_not_worse": bool(dn_not_worse),
            "devnative_better": bool(dn_better), "synth4_better": bool(tv_better and dn_not_worse),
            "synth4_better_on_both": bool(tv_better and dn_not_worse and dn_better)}


def cmd_result(args):
    proto = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    tvr = json.loads(Path(args.tv).read_text(encoding="utf-8"))
    dnr = json.loads(Path(args.dn).read_text(encoding="utf-8"))
    name = next(iter(proto["comparisons"]))
    tv, dn = tvr["comparisons"][name], dnr["comparisons"][name]
    out = {"kind": "synth4-ablation-result", "protocol": {"file": args.protocol, "sha256": sha256(args.protocol)},
           "comparison": name, "decision": decide(tv, dn), "rule": proto["decision"],
           "teacher_val": {"dev": tvr["dev"], "models": tvr["models"], "result": tv},
           "devnative_pooled": {"dev": dnr["dev"], "models": dnr["models"], "result": dn},
           "devnative_files": {}}
    for spec in args.dn_file:
        k, _, p = spec.partition("=")
        r = json.loads(Path(p).read_text(encoding="utf-8"))
        out["devnative_files"][k] = {"dev": r["dev"], "result": r["comparisons"].get(name)}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    d = out["decision"]
    print(f"{name}: teacher_val {tv['all_vn_cluster']['delta_points']:+.3f} {tv['all_vn_cluster']['ci95']}, "
          f"devnative {dn['all_vn_cluster']['delta_points']:+.3f} {dn['all_vn_cluster']['ci95']} -> {d}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tag")
    t.add_argument("--scores", required=True)
    t.add_argument("--curriculum", required=True)
    t.add_argument("--fl10-vocab", required=True)
    t.add_argument("--out", required=True)
    r = sub.add_parser("result")
    r.add_argument("--protocol", required=True)
    r.add_argument("--tv", required=True)
    r.add_argument("--dn", required=True)
    r.add_argument("--dn-file", action="append", default=[])
    r.add_argument("--out", required=True)
    args = ap.parse_args()
    {"tag": cmd_tag, "result": cmd_result}[args.cmd](args)


if __name__ == "__main__":
    main()
