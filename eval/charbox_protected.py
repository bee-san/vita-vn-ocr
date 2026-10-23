#!/usr/bin/env python3
"""The protected check of the per-character boxes (docs/char-boxes.md, eval/charboxes_protocol.json "protected"):
vita-test and psp-test read ONCE, after the development results were committed, by the method of record only.

  teacher/.venv/bin/python eval/charbox_protected.py run --private DIR --out eval/charboxes_protected_<date>.json

The units are eval's hand-checked lines (not uncertain) as `eval/sets.py` cuts them: the line's box (main text
plus its furigana) and eval's region; the frames from $VOCR_EVAL_DATA (default $HERD/shared/artifacts/eval),
sha256-checked. Reference boxes as on development (MeikiOCR on each line's view, the aligned equal characters,
lines within MAX_GT_CER). Lines: fixed (box + pad, 4 px Vita / 2 px PSP) and finder (GRAD + refine on eval's
region). Model VOCR5-N, arm cal + INK, eval/charboxes_cal.json. The attempt record in DIR (private: it holds the
test texts' alignment records) refuses a second run; the public JSON holds counts and rates only.
"""
import argparse
import datetime
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "synth"), str(ROOT / "runtime"), str(ROOT)]
import charbox_eval as CE  # noqa: E402
import charbox_sets as CS  # noqa: E402

EVAL_DATA = Path(os.environ.get("VOCR_EVAL_DATA") or CE.ART / "eval")
SETS = {"vita-test": ("vita", 4), "psp-test": ("psp", 2)}


def units():
    from fetch import shot_path
    out = []
    for name, (part, pad) in SETS.items():
        for r in CS.jsonl(ROOT / "eval" / "manifests" / f"{name}.jsonl"):
            if r.get("uncertain") or not r.get("text"):
                continue
            p = shot_path(EVAL_DATA, part, r["vn"], r["url"])
            cls = "name" if "name" in (r.get("styles") or []) else "dialogue"
            frame = {"id": Path(p).stem, "url": r["url"], "sha256": r["sha256"], "path": str(p), "scale": None,
                     "dims": r["dims"]}
            out.append({"uid": f"{name}:{r['id']}", "src": name, "vn": r["vn"], "part": part, "frame": frame,
                        "box": [int(v) for v in r["box"]], "text": r["text"], "cls": cls, "vertical": False,
                        "region": [int(v) for v in r["region"]], "pad": pad})
    return out


def cmd_run(args):
    priv = Path(args.private)
    rec_path = priv / "attempt.json"
    if rec_path.exists():
        raise SystemExit(f"{rec_path} exists: the protected sets were read once already; refusing a second run")
    proto = json.loads((ROOT / "eval" / "charboxes_protocol.json").read_text(encoding="utf-8"))
    priv.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rec_path.write_text(json.dumps({"started": started, "state": "running"}) + "\n", encoding="utf-8")
    us = units()
    up = priv / "units.jsonl"
    with open(up, "w", encoding="utf-8") as f:
        for u in us:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
    mp = priv / "meiki.jsonl"
    sys.argv = ["charbox_sets.py", "meiki", "--units", str(up), "--out", str(mp)]
    CS.cmd_meiki(argparse.Namespace(units=str(up), out=str(mp)))
    sp = priv / "scores.jsonl"
    CE.ARMS = {CE.RECORD_ARM: CE.ARMS[CE.RECORD_ARM]}         # the method of record only
    CE.cmd_score(argparse.Namespace(units=str(up), meiki=str(mp), hand=None, cal=str(CE.CAL_FILE), tiers=["N"],
                                    sources=["fixed", "finder"], out=str(sp)))
    recs = CS.jsonl(sp)
    tab = CE.summarise(recs, lambda r: (r["src"], r["lines"], r["tier"], r["arm"], r["cls"]), boot=True)
    gt = Counter()
    um = {u["uid"]: u for u in us}
    for r in CS.jsonl(mp):
        b, _ = CS.reference_boxes(um[r["uid"]], r["meiki_text"], r["meiki_chars"])
        gt[(um[r["uid"]]["src"], "ref_chars")] += len(b)
        gt[(um[r["uid"]]["src"], "with_box")] += sum(x is not None for x in b)
    doc = {"kind": "charboxes-protected", "issue": 114, "read": "once", "started": started,
           "finished": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "protocol_pins": proto["code_sha256"], "amendments": len(proto.get("amendments", [])),
           "model": "N", "arm": CE.RECORD_ARM,
           "units": {k: sum(1 for u in us if u["src"] == k) for k in SETS},
           "reference_boxes": {f"{k[0]}|{k[1]}": v for k, v in sorted(gt.items())},
           "tables": {"|".join(map(str, k)): v for k, v in sorted(tab.items())},
           "private": {"units_sha256": CS.sha256(up), "meiki_sha256": CS.sha256(mp), "scores_sha256": CS.sha256(sp)}}
    Path(args.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    rec_path.write_text(json.dumps({"started": started, "finished": doc["finished"], "state": "done",
                                    "public": str(args.out)}) + "\n", encoding="utf-8")
    for k, v in sorted(tab.items()):
        print(k, v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--private", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args()
    cmd_run(args)


if __name__ == "__main__":
    main()
