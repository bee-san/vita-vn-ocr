#!/usr/bin/env python3
"""Check a vocr3_report.py run of the three released blobs against their published numbers.

  vocr3_verify_released.py RUN_DIR --out RESULT.json

RUN_DIR holds shared.json (spec eval/vocr3_harness_released.json, with the stored bar arm), a9.json and
overlay.json. Every published figure below names its source; RESULT.json lists each check with the
published and reproduced values. This is the harness's only verification: it never scores any other
model on a protected set.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import vocr3_report as R  # noqa: E402

SETS = ("vita-test", "psp-test")
TRAIN2 = "docs/train2.md"
VOCR2_DOC = "docs/vocr2-qualification.md + eval/vocr2_candidate_20261012.json"
OVERLAY = ("vita-jp-overlay docs/benchmarks/vocr-release-20261011.json (benchmark_local_ocr.py, FL10 and H15 "
           "v0.2.0)")
RESOURCES = {"VOCR1": {"sha256": "500e0b842e5f0cb326f55e7d991a52ab66dd24977ccc89bd188228aedc825e84",
                       "blob_bytes": 657664, "arena_w8f": 118432, "arena_w8i": 124832, "classes": 3501},
             "VOCR_H15": {"sha256": "316ebc377ec34520440991c5085199d8e74031f01a867938be90a60c1e3f03cd",
                          "blob_bytes": 1155968, "arena_w8f": 173472, "arena_w8i": 182944, "classes": 5355},
             "VOCR2": {"sha256": "d3adc52e5cf62b548af04c5ab07a2f57cf0419386bbdf49ba19c9703be6e1766",
                       "blob_bytes": 657664, "arena_w8i": 124832, "classes": 3501}}
# train2.md release table: errors/chars (CER) and exact lines, native polarity, int8 through the C runtime
NATIVE = {"VOCR1": {"vita-test": (59, 2564, "77.7%"), "psp-test": (130, 2510, "69.8%")},
          "VOCR_H15": {"vita-test": (40, 2564, "83.5%"), "psp-test": (66, 2510, "82.4%")},
          "bar": {"vita-test": (59, 2564, "79.1%"), "psp-test": (134, 2510, "66.1%")}}
# train2.md per-style tables: CER % for (bar, H15, FL10)
STYLES = {
    "vita-test": {"bright-on-dark": (2.48, 1.79, 2.73), "coloured": (5.13, 6.67, 6.15),
                  "coloured-name": (9.52, 14.29, 14.29), "dark-on-light": (1.64, 0.73, 0.73),
                  "furigana": (3.12, 0.00, 4.69), "mincho": (1.33, 0.74, 1.48), "name": (4.07, 5.23, 6.98),
                  "outlined": (3.03, 2.12, 2.12)},
    "psp-test": {"bright-on-dark": (5.18, 2.98, 5.39), "coloured": (7.48, 3.74, 5.61),
                 "coloured-name": (8.33, 4.17, 0.00), "dark-on-light": (5.54, 2.18, 4.90),
                 "furigana": (9.71, 14.08, 23.30), "mincho": (4.98, 1.93, 4.57), "name": (5.74, 9.09, 16.75),
                 "no-box": (5.08, 6.78, 13.56), "outlined": (5.00, 2.97, 4.32), "small": (5.45, 1.54, 3.19)}}
# train2.md: CAND - bar, points, 95% paired bootstrap over lines (10,000 resamples, seed 0)
BOOT = [("VOCR_H15:bar", "vita-test", "all", (-0.74, -1.44, -0.04), 2),
        ("VOCR_H15:bar", "psp-test", "all", (-2.71, -4.05, -1.36), 2),
        ("VOCR1:bar", "psp-test", "all", (-0.16, -1.66, 1.42), 2),
        ("VOCR_H15:bar", "psp-test", "style:name", (3.35, -2.36, 10.65), 2),
        ("VOCR1:bar", "psp-test", "style:name", (11.0, 4.3, 18.3), 1),
        ("VOCR1:bar", "psp-test", "style:furigana", (13.6, 1.0, 25.9), 1)]
# docs/vocr2-qualification.md fixed-line table (console polarity; the shared sets, CLANNAD captures excluded)
CONSOLE = {"VOCR1": {"vita-test": (61, 15), "psp-test": (133, 30)},
           "VOCR2": {"vita-test": (59, 13), "psp-test": (119, 25)}}
# vocr-release-20261011.json in vita-jp-overlay at the cloned commit
OVERLAY_PUB = {
    "VOCR1": {"vita-test": dict(regions=84, refused=0, failed=0, concat_errors=217, concat_chars=2506, lines_read=207,
                                workspace_peak_max=914000, workspace_peak_median=837056),
              "psp-test": dict(regions=93, refused=0, failed=0, concat_errors=443, concat_chars=2510, lines_read=233,
                               workspace_peak_max=827216, workspace_peak_median=805440)},
    "VOCR_H15": {"vita-test": dict(regions=84, refused=0, failed=0, concat_errors=170, concat_chars=2506,
                                   lines_read=207, workspace_peak_max=1470416, workspace_peak_median=1393472),
                 "psp-test": dict(regions=93, refused=0, failed=0, concat_errors=369, concat_chars=2510,
                                  lines_read=233, workspace_peak_max=1383632, workspace_peak_median=1361856)}}
# train2.md (ms per 640-px line at 444 MHz) and the train2 review's vita/a9_cycles.py reports (cycles/frame)
A9 = {"VOCR1": {"w8i": ((94, 178), (521887, 986169)), "w8f": ((157, 194), (871347, 1077864))},
      "VOCR_H15": {"w8i": ((166, 302), (920390, 1678821)), "w8f": ((294, 365), (1633471, 2028094))}}
A9_FRAMES = 1828


def check(checks, name, published, reproduced, source, match=None, note=None):
    ok = (published == reproduced) if match is None else match
    entry = {"check": name, "published": published, "reproduced": reproduced, "match": bool(ok), "source": source}
    if note:
        entry["note"] = note
    checks.append(entry)


def verify(run_dir):
    run_dir = Path(run_dir)
    sh = json.loads((run_dir / "shared.json").read_text())
    checks = []
    for arm, want in RESOURCES.items():
        got = {k: sh["resources"][arm][k] for k in want}
        check(checks, f"resources/{arm}", want, got, "plan appendix A; docs/train2.md; docs/vocr2-qualification.md")
    native = sh["fixed_line"]["native"]
    for arm, by_set in NATIVE.items():
        for s, (e, n, exact) in by_set.items():
            t = native[arm]["summary"][s]["all"]
            check(checks, f"native/{arm}/{s}", [e, n, exact], [t["errors"], t["chars"],
                                                               f"{100 * t['exact'] / t['lines']:.1f}%"], TRAIN2)
    for s, styles in STYLES.items():
        for tag, cers in styles.items():
            got = tuple(round(100 * native[a]["summary"][s]["styles"][tag]["errors"] /
                              native[a]["summary"][s]["styles"][tag]["chars"], 2) for a in ("bar", "VOCR_H15", "VOCR1"))
            check(checks, f"native-style/{s}/{tag} (bar, H15, FL10 CER %)", list(cers), list(got), TRAIN2)
    for key, s, stratum, want, digits in BOOT:
        e = sh["pairs"][key]["native"][s][stratum]["full"]["lines"]
        got = [round(e["delta"], digits), round(e["ci95"][0], digits), round(e["ci95"][1], digits)]
        check(checks, f"bootstrap/{key}/{s}/{stratum}", list(want), got, TRAIN2)
    tie = [native[a]["summary"]["vita-test"]["all"]["errors"] for a in ("VOCR1", "bar")]
    check(checks, "native/VOCR1 ties the bar on vita-test (errors)", [59, 59], tie, TRAIN2)
    console = sh["fixed_line"]["console"]
    for arm, by_set in CONSOLE.items():
        for s, (e, h) in by_set.items():
            t = console[arm]["summary"][s]["all"]
            check(checks, f"console/{arm}/{s} (full, hiragana errors)", [e, h], [t["errors"], t["hira_errors"]],
                  VOCR2_DOC)
    gate = sh["gates"]["VOCR2:VOCR1"]
    check(checks, "gate/VOCR2:VOCR1 fixed-line gate passes", True, gate["fixed_line"]["passed"], VOCR2_DOC)
    hb = json.loads((ROOT / "eval/hiragana_baseline_20261011.json").read_text())
    for s in SETS:
        mine = console["VOCR1"]["summary"][s]
        pub = hb["sets"][s]
        a = mine["all"]
        got = {"all": [a["errors"], a["chars"], a["hira_errors"], a["hira_chars"]],
               **{f"style:{t}": [mine["styles"][t]["lines"], mine["styles"][t]["chars"], mine["styles"][t]["errors"]]
                  for t in pub["styles"]},
               **{f"game:{g}": [mine["games"][g]["lines"], mine["games"][g]["chars"], mine["games"][g]["errors"]]
                  for g in pub["games"]}}
        want = {"all": [pub["all"]["errors"], pub["all"]["chars"], pub["hiragana"]["errors"], pub["hiragana"]["chars"]],
                **{f"style:{t}": [v["lines"], v["chars"], v["errors"]] for t, v in pub["styles"].items()},
                **{f"game:{g}": [v["lines"], v["chars"], v["errors"]] for g, v in pub["games"].items()}}
        check(checks, f"console/VOCR1/{s} per style and per VN (lines, chars, errors)", want, got,
              "eval/hiragana_baseline_20261011.json")
    v2 = json.loads((ROOT / "eval/vocr2_candidate_20261012.json").read_text())["whole_dialogue_test"]
    tot = sh["dialogue"]["totals"]
    for variant in R.VARIANTS:
        for side, arm in (("baseline", "VOCR1"), ("candidate", "VOCR2")):
            pub = v2["variants"][variant]["by_side"][side]["psp"]
            mine = tot[arm][variant]["psp"]
            keys = [(sc, t) for sc in R.SCOPES for t in R.SCRIPTS]
            check(checks, f"dialogue/psp/{variant}/{arm} (all/body x full/hira errors/chars, detections)",
                  [[pub[sc][t]["errors"], pub[sc][t]["chars"]] for sc, t in keys] + [pub["detections"]],
                  [[mine[sc][t]["errors"], mine[sc][t]["chars"]] for sc, t in keys] + [mine["detections"]], VOCR2_DOC)
    guard = gate["dialogue"]
    pub_cells = sorted([c["variant"], c["group"], c["scope"], c["metric"], c["baseline_errors"], c["candidate_errors"],
                        c["chars"]] for c in v2["failed_guard_cells"])
    got_cells = sorted([c["variant"], c["group"], c["scope"], c["metric"], c["baseline_errors"], c["candidate_errors"],
                        c["chars"]] for c in guard["failed_cells"])
    check(checks, "gate/VOCR2:VOCR1 whole-dialogue guard fails on the same title cells", pub_cells, got_cells,
          VOCR2_DOC, note="the published guard also covers the private CLANNAD captures; none of its 19 failed "
                          "cells is CLANNAD")
    check(checks, "gate/VOCR2:VOCR1 gain titles per variant",
          {v: sorted(x) for v, x in v2["decision"]["gain_vns_by_variant"].items()},
          {v: sorted(x.split(":")[1] for x in t) for v, t in guard["gain_titles"].items()}, VOCR2_DOC)
    if (run_dir / "overlay.json").exists():
        ov = json.loads((run_dir / "overlay.json").read_text())
        for arm, by_set in OVERLAY_PUB.items():
            for s, want in by_set.items():
                got = {k: ov["arms"][arm]["sets"][s][k] for k in want}
                check(checks, f"overlay/{arm}/{s}", want, got, OVERLAY)
        for s in SETS:
            plat = {"vita-test": "vita", "psp-test": "psp"}[s]
            for arm in ("VOCR1", "VOCR_H15", "VOCR2"):
                o = ov["arms"][arm]["sets"][s]
                h = tot[arm]["base"][plat]["all"]["full"]
                check(checks, f"overlay-vs-harness/{arm}/{s} (concat errors/chars = dialogue base all-text)",
                      [o["concat_errors"], o["concat_chars"]], [h["errors"], h["chars"]],
                      "this run: overlay vjo-ocr and the harness's whole-dialogue mode")
    if (run_dir / "a9.json").exists():
        a9 = json.loads((run_dir / "a9.json").read_text())
        for arm, modes in A9.items():
            for mode, ((ms_lo, ms_hi), (c_lo, c_hi)) in modes.items():
                r = a9["arms"][arm]["modes"][mode]
                lo, hi = r["cycles_per_frame"]["444"]
                check(checks, f"a9/{arm}/{mode} frames", A9_FRAMES, r["frames"], "train2 review a9 meta.json")
                check(checks, f"a9/{arm}/{mode} low bound, cycles/frame", c_lo, lo, "train2 review a9 report.json")
                check(checks, f"a9/{arm}/{mode} low bound, ms (640 px, 444 MHz)", ms_lo,
                      round(r["ms_640px_line"]["444"][0]), TRAIN2)
                rel = (hi - c_hi) / c_hi
                check(checks, f"a9/{arm}/{mode} high bound, cycles/frame", c_hi, hi, "train2 review a9 report.json",
                      match=False if hi != c_hi else True,
                      note=f"{100 * rel:+.2f}%: the high bound charges every simulated L1 miss, whose count depends "
                           "on the guest memory layout (the review's run had another executable path and environment); "
                           "docs/vocr3-harness.md")
                check(checks, f"a9/{arm}/{mode} high bound, ms (640 px, 444 MHz)", ms_hi,
                      round(r["ms_640px_line"]["444"][1]), TRAIN2, note="layout-dependent, as above")
    summary = {"checks": len(checks), "matched": sum(c["match"] for c in checks),
               "not_matched": [c["check"] for c in checks if not c["match"]]}
    return {"schema": 1, "kind": "vocr3-harness-released-verification", "run": {
        "shared_sha256": R.sha256(run_dir / "shared.json"),
        "a9_sha256": R.sha256(run_dir / "a9.json") if (run_dir / "a9.json").exists() else None,
        "overlay_sha256": R.sha256(run_dir / "overlay.json") if (run_dir / "overlay.json").exists() else None,
        "git": sh["environment"]["git"], "code_sha256": sh["code_sha256"],
        "native_library_sha256": sh["native"]["library"]["sha256"], "compiler": sh["native"]["compiler"]["version"]},
        "summary": summary, "checks": checks}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = verify(args.run_dir)
    R.write_new(args.out, result)
    s = result["summary"]
    print(f"{s['matched']}/{s['checks']} checks match; not matched: {s['not_matched']}")


if __name__ == "__main__":
    main()
