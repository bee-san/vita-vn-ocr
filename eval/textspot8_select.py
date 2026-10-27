#!/usr/bin/env python3
"""textspot8's selection (eval/textspot8_protocol.json "rule"): eval/textspot6_select.py, unchanged, with textspot8's
decode order (amendments 3 and 4: crop, crop+refine, ctx, det, det+refine, det+D, F, F+D) for its cost key.

  textspot8_select.py --table TABLE.json --out eval/textspot8_dev_<date>.json     (TABLE: eval/textspot8_table.py)

A decode without a parity record (F, F+D: report only, ineligible by their cost) counts as not C == twin.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import textspot6_select as S6  # noqa: E402

S6.DECODES = ("crop", "crop+refine", "ctx", "det", "det+refine", "det+D", "F", "F+D")

if __name__ == "__main__":
    sys.exit(S6.main())
