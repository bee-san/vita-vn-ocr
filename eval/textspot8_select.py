#!/usr/bin/env python3
"""textspot8's selection (eval/textspot8_protocol.json "rule"): eval/textspot6_select.py, unchanged, with textspot8's
decode order (amendments 3 and 4: crop, crop+refine, ctx, det, det+refine, det+D, F, F+D) for its cost key, then
amendment 5's Rubi-default test written beside textspot6's.

  textspot8_select.py --table TABLE.json --out eval/textspot8_dev_<date>.json     (TABLE: eval/textspot8_table.py)

A decode without a parity record (F, F+D: report only, ineligible by their cost) counts as not C == twin.

Amendment 5 (Rubi's default, dev / hval part; the confirmation part is added by the confirmation read): against
rubi,fix=1 on the same units, hval resolved below 0 and unseen dev not resolved above 0. textspot6's own clause
(unseen dev resolved below 0, hval not resolved above 0) stays in the record as "rubi_default" for comparison; the
amendment's result is "rubi_default_t8".
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import textspot6_select as S6  # noqa: E402

S6.DECODES = ("crop", "crop+refine", "ctx", "det", "det+refine", "det+D", "F", "F+D")


def default_t8(vs):
    """amendment 5's dev / hval test against rubi,fix=1: hval resolved below 0, unseen dev not resolved above 0."""
    return S6.resolved_below(vs["hval"]) and not S6.resolved_above(vs["dev"])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    rc = S6.main(argv)
    out = Path(argv[argv.index("--out") + 1])
    rec = json.loads(out.read_text(encoding="utf-8"))
    for c in rec.get("candidates", []):
        if "vs_rubi_next" in c:
            c["rubi_default_t8_dev_hval"] = default_t8(c["vs_rubi_next"])
    pick = rec.get("pick")
    if pick:
        c = next(x for x in rec["candidates"] if x["arm"] == pick["arm"] and x["decode"] == pick["decode"])
        rec["rubi_default_t8"] = {"rule": "amendment 5 (dev / hval part): against rubi,fix=1 hval resolved below 0 "
                                          "and unseen dev not resolved above 0; the confirmation part follows",
                                  "dev_hval_pass": c.get("rubi_default_t8_dev_hval"),
                                  "vs_rubi_next": c.get("vs_rubi_next")}
    rec["kind"] = "textspot8-selection"
    rec["textspot8_protocol_sha256"] = S6.sha256_file(ROOT / "eval" / "textspot8_protocol.json")
    out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main())
