"""The screening table of #169 (hval dialogue CER per arm and policy, all titles / without PCSG00854) from the
read results score_run.sh writes ($HERD/shared/artifacts/vocr-desktop/res/hval_*.json, counts only).

    python3 screen_table.py S1_d48 S2_d64 ...
"""
import json
import os
import sys

A = os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr") + "/shared/artifacts/vocr-desktop/res"
POL = ["fit960", "fit800", "fit640", "fit480", "long960", "long800", "long640"]


def cer(name, drop=None, view=None):
    path = f"{A}/hval_{name}.json"
    if not os.path.exists(path):
        return None
    errors = chars = 0
    for u in json.load(open(path))["units"]:
        if drop and u["cluster"] == drop:
            continue
        if view and u["view"] != view:
            continue
        errors += u["errors"]
        chars += u["chars"]
    return 100 * errors / chars


def main(arms):
    print("baselines", {n: (round(cer(n), 3), round(cer(n, "PCSG00854"), 3)) for n in ("meiki4_L_fast", "ts8_L_fast")})
    for arm in arms:
        for pad in ("", "_frame"):
            row, vals, vals2 = [], [], []
            for p in POL:
                n = f"{arm}_{p}{pad}_L_fast"
                a, b = cer(n), cer(n, "PCSG00854")
                row.append(f"{p}:{a:.2f}/{b:.2f}" if a is not None else f"{p}:-")
                if a is not None:
                    vals.append(a)
                    vals2.append(b)
            if len(vals) == 7:
                m = f"mean7 {sum(vals) / 7:.3f} / x854 {sum(vals2) / 7:.3f}"
            else:
                m = f"({len(vals)} of 7)"
            print(f"{arm}{pad or '_pct'}: " + " ".join(row) + "  " + m)


if __name__ == "__main__":
    main(sys.argv[1:])
