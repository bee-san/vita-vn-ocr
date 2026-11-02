"""textspot-desktop (#173): the PC test table's breakdown columns, re-scored from the stored reads.

The docs' PC test table gives each pipeline's dialogue CER on all units, without the flagged VNs (the set's family and
release-partner VNs, docs/desktop-data.md) and by the frame's width. This script recomputes those columns from the
stored result files of the one PC test read (eval/textspot_desktop_final.json lists them) and the manifest's metadata:
each unit's VN, frame width and flags. It runs no detector and reads no frame or text.

  tsdesk_pctest_breakdown.py --res DIR --out eval/textspot_desktop_pctest_breakdown.json
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "eval" / "manifests" / "desktop-pc-test-v1.jsonl"
# the stored reads (eval/textspot_desktop_final.json's pctest keys): whole frames, then frames fit into 960 x 544
PIPELINES = {"this detector": ("pctest_cand_long960", "pctest_l544_cand"), "textspot8": ("pctest_T8", "pctest_l544_T8"),
             "R (MeikiOCR's detector + L)": ("pctest_R", "pctest_l544_R"), "exact boxes": ("pctest_exact", None),
             "MeikiOCR 0.3.5 itself": ("pctest_meikiocr", "pctest_l544_meikiocr"), "mida + L": ("pctest_mida", None)}
WIDTHS = (("<=800", 0, 800), ("801-1279", 801, 1279), ("1280-1919", 1280, 1919), (">=1920", 1920, 1 << 30))


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def meta():
    """{unit: (vn, frame width)} and the flagged VNs, from the manifest (no text is read into the output)."""
    units, flagged = {}, set()
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        units.setdefault(r["unit"], (r["vn"], int(r["dims"][0])))
        if r.get("family_of") or r.get("release_partner_of"):
            flagged.add(r["vn"])
    return units, flagged


def cer(us):
    e, c = sum(u["errors"] for u in us), sum(u["chars"] for u in us)
    return {"cer": round(100.0 * e / c, 3) if c else None, "errors": e, "chars": c, "units": len(us)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", required=True, help="the directory of the stored PC test reads (res/pctest)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    units, flagged = meta()
    out = {"kind": "textspot-desktop-pctest-breakdown", "issue": 173, "manifest_sha256": sha256(MANIFEST),
           "flagged_vns": len(flagged), "widths": [w[0] for w in WIDTHS],
           "how": "CER = 100 x errors / reference characters over the units of each column; the units' counts are "
                  "the stored reads' own (no detector run, no frame read)", "pipelines": {}}
    for name, (whole, fit) in PIPELINES.items():
        p = Path(a.res) / f"{whole}.json"
        us = json.loads(p.read_text())["units"]
        missing = [u["uid"] for u in us if u["uid"] not in units]
        if missing:
            raise SystemExit(f"{p}: {len(missing)} units not in the manifest")
        row = {"file": p.name, "file_sha256": sha256(p), "all": cer(us),
               "without_flagged": cer([u for u in us if units[u["uid"]][0] not in flagged])}
        for label, lo, hi in WIDTHS:
            row[label] = cer([u for u in us if lo <= units[u["uid"]][1] <= hi])
        if fit:
            q = Path(a.res) / f"{fit}.json"
            row["fit_960x544"] = dict(cer(json.loads(q.read_text())["units"]), file=q.name, file_sha256=sha256(q))
        out["pipelines"][name] = row
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, row in out["pipelines"].items():
        print(f"{name:30s}", row["all"]["cer"], row["without_flagged"]["cer"],
              [row[w[0]]["cer"] for w in WIDTHS], (row.get("fit_960x544") or {}).get("cer"))


if __name__ == "__main__":
    main()
