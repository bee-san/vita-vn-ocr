#!/usr/bin/env python3
"""eval/charboxes_protocol.json: the pre-registered protocol of the per-character screen boxes (issue #114,
docs/char-boxes.md), written once before any development score, then only amended in place by an appended
amendment.

  .venv-ci/bin/python eval/charbox_protocol.py write --units U --meiki M --hand H --train-check T
  .venv-ci/bin/python eval/charbox_protocol.py check           # the committed pins against the checkout
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
PROTOCOL = ROOT / "eval" / "charboxes_protocol.json"
CODE = ["eval/charbox.py", "eval/charbox_sets.py", "eval/charbox_eval.py", "eval/charbox_hand.py",
        "eval/charboxes_cal.json"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_write(args):
    import charbox as CB
    import charbox_eval as CE
    import charbox_sets as CS
    cal = json.loads((ROOT / "eval" / "charboxes_cal.json").read_text(encoding="utf-8"))
    tc = json.loads(Path(args.train_check).read_text(encoding="utf-8"))
    doc = {
        "kind": "charboxes-protocol", "schema": 1, "issue": 114, "doc": "docs/char-boxes.md",
        "written_before": "any development, synthetic-development or protected score of any arm",
        "code_sha256": {p: sha256(ROOT / p) for p in CODE},
        "models": {t: {"file": str(Path(p).name), "sha256": sha256(p)} for t, p in CE.BLOBS.items()},
        "spotters": {k: {"file": str(Path(p).name), "sha256": sha256(p)} for k, p in CE.SPOTTERS.items()},
        "calibration": {t: m["cal"] for t, m in cal["models"].items()},
        "calibration_rule": cal["rule"],
        "groups": CB.GROUP_NAMES,
        "method_of_record": {"arm": CE.RECORD_ARM, "flags": "INK (4) with the model's calibration",
                             "chosen_on": "training-side lines only (eval/charbox_eval.py train-check, the lines "
                                          "after the calibration's in its seeded order); exploration before it "
                                          "on other training-side lines (issue #114)",
                             "train_check": {t: {k: v for k, v in m["hit"].items() if k.endswith(("|vita", "|psp"))}
                                             for t, m in tc["models"].items()}},
        "arms": {k: {"flags": v[0], "cal": v[1]} for k, v in CE.ARMS.items()},
        "constants": {"Q": CB.Q, "stride": CB.STRIDE, "pitch_min": CB.PITCH_MIN, "pitch_tol": CB.PITCH_TOL,
                      "ink_frac": CB.INK_FRAC, "max_chars": CB.MAX_CHARS},
        "sets": {
            "units": {"file": "$HERD/shared/artifacts/overlay-boxes/sets/units_dev.jsonl", "sha256": sha256(args.units),
                      "rule": "eval/charbox_sets.py units: textspot-dev references (dev_sample VNs), devnative's "
                              "dev_sample lines (+ every PSP line on its frame at 2x, cv2.INTER_LINEAR: adrenaline), "
                              "the vertical study's development columns; no test VN"},
            "meiki": {"file": "$HERD/shared/artifacts/overlay-boxes/sets/meiki_dev.jsonl", "sha256": sha256(args.meiki),
                      "rule": "MeikiOCR (teacher/engines.Meiki, pinned files) on each unit's view; reference box = "
                              "the aligned equal MeikiOCR character, lines within MAX_GT_CER "
                              f"({CS.MAX_GT_CER}; columns {CS.MAX_GT_CER_V})"},
            "hand": {"file": "$HERD/shared/artifacts/overlay-boxes/hand/hand_dev.jsonl", "sha256": sha256(args.hand),
                     "rule": "eval/charbox_hand.py sheets (seed 20261023, 48 Vita + 48 PSP native dialogue lines), "
                             "every MeikiOCR box checked by eye before any score (reviewer_kind assistant); the "
                             "hand-checked subset scores only the accepted boxes"},
            "synthetic": "report only: the synth4 renderer's development split with exact glyph boxes (helper "
                         "overlay-boxes-synth), when it is ready; never a gate",
        },
        "line_sources": {"fixed": "the unit's ink box plus the console pad (4 px; PSP 2), dark text with VOCR_INVERT",
                         "finder": "GRAD + refine (DEFAULT) on the unit's region; columns: vertical_twin.find_columns "
                                   "variant 1; one-to-one matching (eval's rule)",
                         "ts1": "textspot v0.3.3 det+refine on the whole textspot-dev frame",
                         "ts3": "textspot3 v0.3.4 det+refine on the whole textspot-dev frame"},
        "metrics": {"cursor_hit": "estimated centre inside the reference box along the reading direction and, "
                                  "across it, inside the reference line's ink box plus the pad; over aligned "
                                  "pairs (hit_all: every reference character with a box, missing reads as misses)",
                    "iou": "1-D along the reading direction, the cell and the ink box against the reference box",
                    "ci": "VN-cluster bootstrap of the hit ratio, 10,000 resamples, numpy seed 0"},
        "bar": {"model": "N", "arm": CE.RECORD_ARM, "class": "dialogue (horizontal)", "reference": "MeikiOCR",
                "parts": ["vita", "psp", "adrenaline"], "line_sources": ["fixed", "finder"],
                "rule": "cursor-hit >= 0.970 in every one of the six (part, line source) cells",
                "pass": "the C module runtime/vocr_charbox.c with twin parity, then Rubi's overlay mode",
                "fail": "the other arms, then a report of the model change needed; no runtime change"},
        "report_only": ["VOCR5-S/L", "VOCR5-V on columns", "ts1/ts3 lines", "names, choices, UI, scene text",
                        "PC scaled frames", "hit_all", "IoU", "the hand-checked subset", "the synthetic set"],
        "protected": "vita-test and psp-test, fixed and finder lines, VOCR5-N, cal+ink, reference boxes made the "
                     "same way; read once at the end as a check, after the development results are committed; "
                     "never used to choose anything",
    }
    PROTOCOL.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"-> {PROTOCOL}")


def pins(doc):
    """The code pins in force: the protocol's, as the last amendment that re-pinned them left them."""
    out = dict(doc["code_sha256"])
    for a in doc.get("amendments", []):
        out.update(a.get("code_sha256", {}))
    return out


def check():
    """[(file, pinned, actual)] for every committed file whose sha256 differs from the pins in force."""
    doc = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    bad = []
    for p, want in pins(doc).items():
        got = sha256(ROOT / p)
        if got != want:
            bad.append((p, want, got))
    return bad


def cmd_amend(args):
    """Append an amendment: what changed, why, and the current pins of the protocol's files plus any new file."""
    import datetime
    doc = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    files = list(dict.fromkeys(list(pins(doc)) + list(args.add or [])))
    am = {"n": len(doc.get("amendments", [])) + 1,
          "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
          "what": args.what, "why": args.why, "code_sha256": {p: sha256(ROOT / p) for p in files}}
    doc.setdefault("amendments", []).append(am)
    PROTOCOL.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"amendment {am['n']} -> {PROTOCOL}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("write")
    a.add_argument("--units", required=True)
    a.add_argument("--meiki", required=True)
    a.add_argument("--hand", required=True)
    a.add_argument("--train-check", required=True)
    a = sub.add_parser("amend")
    a.add_argument("--what", required=True)
    a.add_argument("--why", required=True)
    a.add_argument("--add", nargs="*", help="files to pin from now on as well")
    sub.add_parser("check")
    args = ap.parse_args()
    if args.cmd == "write":
        cmd_write(args)
    elif args.cmd == "amend":
        cmd_amend(args)
    else:
        bad = check()
        for b in bad:
            print("changed since the protocol:", *b)
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
