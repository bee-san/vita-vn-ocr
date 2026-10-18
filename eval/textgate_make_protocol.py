#!/usr/bin/env python3
"""Write eval/textgate_protocol.json, the pre-registered protocol of the text-presence gate (docs/textgate.md,
issue #100). Run once, before any development score; the file records the sha256 of every input it names.

  textgate_make_protocol.py --out eval/textgate_protocol.json
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth", ROOT / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
T = ART / "textdet"

# #97's trained detectors read as presence gates: every run under 4,100 B (much smaller than s8_c24's 5,700 B)
R97 = ["s4_c4_seed1", "s4_c4_seed2", "s4p_c4_seed1", "s4_c6_seed1", "s4_c8_seed1", "s4_c8_seed2", "s8_c8_seed1",
       "s8_c8_seed2", "s8_c8_seed3", "s4_c8n5_seed1", "s4_c8n5_seed3", "s4_c12_seed1", "s4p_c8_seed1",
       "s4_c12n4_seed3", "s4_c16_seed1", "s4_c16_seed2", "s8_c16_seed1", "s8_c16_seed2", "s8_c16_seed3",
       "s4_c16n4_seed1", "s4_c16n4_seed3", "s8_c16n4_seed3"]
NEW_RUNS = [("p8_c4", 1), ("p8_c6", 1), ("p8_c6", 2), ("p8_c8", 1), ("p8_c8", 2), ("p8_c12", 1), ("p4_c4", 1),
            ("p4_c8", 1)]
SOURCES = ("eval/textgate.py", "eval/textgate_data.py", "train/textgate_train.py", "train/textdet_train.py",
           "train/textdet_model.py", "train/textdet_export.py", "synth/textdet_twin.py", "synth/textdet_data.py",
           "eval/textdet.py", "eval/finder_tighten.py", "synth/lines_refine.py", "synth/common.py",
           "runtime/vocr_lines.c", "runtime/vocr_lines.h", "runtime/vocr_lines_rows.c", "runtime/vocr_lines_rows.h",
           "runtime/vocr_lines_refine.c", "runtime/vocr_lines_refine.h", "runtime/vocr_textdet.c",
           "runtime/vocr_textdet.h", "eval/manifests/reserved.json", "data/manifests/textdet_shards_v1.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shape_bytes(shape):
    """The VTDT blob bytes of a shape (a zero network packed: the layout does not depend on the values)."""
    import numpy as np

    import textdet_export as EX
    import textdet_model as TM
    q = []
    for L in TM.layers(shape):
        cin = 1 if L["kind"] == "dw" else L["cin"]
        q.append({**L, "wq": np.zeros((L["cout"], cin, L["k"], L["k"]), np.int8), "bq": np.zeros(L["cout"], np.int64),
                  "m": np.full(L["cout"], 1 << 30, np.int64), "sh": np.full(L["cout"], 31, np.int64)})
    return len(EX.pack(shape, q))


def macs(shape, w, h):
    import textdet_model as TM
    return TM.cost(shape, w, h)["macs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "eval" / "textgate_protocol.json"))
    args = ap.parse_args()
    import textgate_train as GT
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    cal = json.loads((HERD / "shared" / "private" / "textgate" / "cohorts" / "cohorts.json").read_text())
    synth_man = ROOT / "data" / "manifests" / "textgate_shards_v1.json"
    base = {m: sha256(T / "baseline" / "wd" / m / f"units_{m}.json") for m in ("VOCR5-S", "VOCR5-N", "VOCR5-L")}
    td = json.loads((ROOT / "eval" / "textdet_protocol.json").read_text())
    doc = {
        "kind": "textgate-protocol", "issue": 100, "schema": 1,
        "frozen": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "written_at_head": head,
        "rule_of_this_file": "Frozen before any development score of any gate. Changes only by a dated amendment "
                             "appended to 'amendments', made before the scores it could affect.",
        "owner": "bee-san 2026-10-18 06:50Z: do all four workstreams, PRs and issues, merge automatically; earlier: "
                 "'we need it to run on a ps vita, so smaller == better'",
        "problem": "GRAD and GRAD + refine return no line for 5 of the 66 development empty regions (0.076); #97's "
                   "s8_c24 detector as a gate cost +0.02-0.03 points for 0.92, but Rubi #209 declined its 8,360 B "
                   "of worker code and 5,700 B of weights",
        "sources_sha256": {f: sha256(ROOT / f) for f in SOURCES if (ROOT / f).exists()},
        "recognizers": td["recognizers"],
        "finder": {"baseline": "GRAD + refine: runtime/vocr_lines.c VOCR_LINES_GRAD, then runtime/vocr_lines_refine.c "
                               "VOCR_REFINE_DEFAULT (0x1F); the C driver eval/finder_tighten.build, checked against "
                               "the twins on every case",
                   "gate": "the gate decides per region; 'no text' -> the finder returns no line (nothing is "
                           "recognized), 'text' -> GRAD + refine unchanged"},
        "gate": {
            "learned": "an int8 network in #97's VTDT layout (synth/textdet_twin.py); K = the head's first map, "
                       "textdet_twin.run_net(model, grey)[0] on the region's grey (synth/common.to_gray)",
            "modes": {"region": "score = the k-th largest K code of the region's map (the smallest when fewer cells)",
                      "boxes": "score = the k-th largest K code over the cells whose s x s block meets a GRAD box "
                               "(-129 when no cell does)"},
            "decision": "text iff score >= t",
            "k_set": [1, 2, 4], "targets": [0.998, 0.999],
            "calibration": "per candidate, mode and target: for each k, t = the largest code keeping at least the "
                           "target share of the cal_dlg cases GRAD + refine returns a line for (the (q+1)-th smallest "
                           "score, q = floor((1 - target) n)); "
                           "k = the one whose (k, t) leaves the most cal_empty regions with no line (gate says no "
                           "or GRAD + refine returns none), the smaller k on a tie",
            "screen": "a candidate operating point goes to development only if that cal_empty rate is >= 0.85",
            "rules": {"blank": "Rubi's blank-line check (core/vocr_ocr.c vjo_vocr_ink, twin eval/textgate."
                               "rubi_ink_edges): text iff a GRAD + refine line's crop has >= 50 edges per 10,000 "
                               "pixels; t fixed at 1 (calibrated like the others, it has one score level)",
                      "rule8": "the longest horizontal run of 8 x 8 blocks with >= 16 edge pixels (|dx| or |dy| >= "
                               "24) and contrast >= 48; text iff the run is >= t blocks, t calibrated"}},
        "candidates": {
            "new_presence_runs": [{"shape": s, "seed": seed, "run": f"{s}_seed{seed}",
                                   "blob_bytes": shape_bytes(GT.SHAPES[s]),
                                   "macs_900x150": macs(GT.SHAPES[s], 900, 150),
                                   "macs_960x544": macs(GT.SHAPES[s], 960, 544)} for s, seed in NEW_RUNS],
            "shapes": GT.SHAPES,
            "r97_runs": [{"run": r, "blob_sha256": sha256(next(p for p in (T / "sweep" / r / "final.vtd",
                                                                           T / "sweep_r3" / r / "final.vtd")
                                                               if p.exists())),
                          "blob_bytes": os.path.getsize(next(p for p in (T / "sweep" / r / "final.vtd",
                                                                         T / "sweep_r3" / r / "final.vtd")
                                                             if p.exists()))} for r in R97],
            "rules": ["blank", "rule8"],
            "report_only": {"s8_c24_seed4": "#97's report candidate (5,700 B): as a presence gate (calibrated like "
                                            "the others) and as #97's own gate (its decode returns no line)",
                            "sha256": sha256(T / "sweep_r3" / "s8_c24_seed4" / "final.vtd")}},
        "training": {"recipe": "train/textgate_train.py: #97's windows, masks, optimiser and QAT; 24,000 steps, "
                               "16 windows of at most 192 x 512, QAT from step 19,200; half real (realplus 0.35, "
                               "teacher 0.35, owner 0.25, devnative 0.05 of the half) and half clean synthetic; "
                               "presence targets and loss as the trainer's docstring; the last step is the model",
                     "real_shards": "data/manifests/textdet_shards_v1.json's realplus, teacher, devnative and owner "
                                    "shards, without the calibration split's VNs; #97's synthetic shards are not used",
                     "synth_shards": "data/manifests/textgate_shards_v1.json (eval/textgate_data.py synth: #97's "
                                     "generator, synth3's scene pool without any eval/linefind VN or screenshot, "
                                     "eval's screenshots and v14018)",
                     "synth_manifest_sha256": sha256(synth_man) if synth_man.exists() else None,
                     "threads": 5, "heavy": "HEAVY_SLOTS=12 $HERD/heavy.sh"},
        "calibration_data": {"builder": "eval/textgate_data.py cal", "split": "sha256('textgate-split:' + vn)[:8] "
                                                                              "% 5 == 0 (never trained on)",
                             "cohorts": cal},
        "development": {
            "wd": "VOCR5's development whole dialogue, as #97 (eval/textdet.wd_units: tv_dlg plus the dn_dlg frames "
                  "tv_dlg does not hold, base / +4 / +8, pooled; 1,515 cases, 1,161 pooled units, 93 VNs); the "
                  "baseline units are #97's frozen GRAD + refine units",
            "baseline_units_sha256": base,
            "cohorts_sha256": {n: sha256(ART / "vocr5-train" / "cohorts" / f"{n}.pkl") for n in ("tv_dlg", "dn_dlg")},
            "scoring": "a case the gate passes keeps its GRAD + refine unit; a case it stops reads nothing (every "
                       "character a deletion). Intervals: eval/textdet.vn_diff, 10,000 VN-cluster resamples, seed 0",
            "native_vita": "eval/textdet.native_vita_shots (150 units, 9 VNs)",
            "empties": "eval/linefind's empty regions of vita_dev + psp_dev, a reserved test VN's left out: 66; the "
                       "rate = the share with no line (the gate says no, or GRAD + refine returns none); reported "
                       "by whether the screenshot is in synth3's default scene pool (#97's caveat: 54 in, 12 out)",
            "report_only": "the holdout empties (vita_holdout + psp_holdout, 25); per platform and variant"},
        "selection_rule": {
            "wd_cost_cap_points": 0.10, "empty_rate_min": 0.85,
            "eligible": "a learned gate's C runtime (runtime/vocr_textgate.c) gives the twin's score on every "
                        "development case and empty region; a rule needs no C to be scored",
            "qualifies": "for each of VOCR5-S, -N and -L: gate - refine on wd (points, variants pooled) has a VN-"
                         "cluster 95% interval whose lower end is <= 0 and a point estimate <= +0.10; on native_vita "
                         "the same interval's lower end is <= 0; and the development empty rate is >= 0.85",
            "size_order": "total bytes = int8 blob bytes + the VitaSDK gcc 15.2 -Os .text bytes of the gate's code "
                          "for its mode (linked with --gc-sections from the gate's entry points; vita/"
                          "textgate_size.py), then blob bytes, then region before boxes, then the higher target, "
                          "then the name",
            "winner": "the first qualifying candidate in size order",
            "no_winner": "negative verdict; GRAD + refine stays as it is; the protected sets are read once for the "
                         "candidate with the most development empties returning no line among those meeting the wd "
                         "conditions (labelled not promoted)"},
        "protected": {"when": "once, after the selection record is committed and pushed",
                      "tests": "vita-test + psp-test whole dialogue (eval's regions, base / +4 / +8; #97's GRAD + "
                               "refine units for VOCR5-S/N/L, with the gate's decisions on the same regions; "
                               "attempt record)",
                      "clannad": "CLANNAD's long run: the gate on each page's ROI; GRAD + refine reads of every "
                                 "reference line (eval/textdet_protected's mapping), dialogue CER with and without "
                                 "the gate; lines returned on the pages the references hold no text for"},
        "estimates": "console time from the MAC rate of VOCR3-H15 (1,564,992 MACs per CTC frame at 5.0-5.2 ms, "
                     "333 MHz assumed) and, where measured, vita/a9_cycles.py; every console figure an estimate",
        "pre_study": {
            "note": "training-side only (calibration cohorts), before this protocol: block-statistics designs "
                    "(per 8 x 8 / 16 x 16 block: contrast, mean |dx|, mean |dy|, edge share; tiny CNNs of 0.3-5 K "
                    "parameters on them, trained densely on #97's real canvases) kept at most 0.47 of cal_empty "
                    "at 0.998 recall of cal_dlg (stride 8, two 3 x 3 x 16 layers); the edge-run rule's run length "
                    "separated nothing at that recall. #97's detectors read as presence gates: 0.17-0.81 (s4_c4 "
                    "to s8_c8, 1,004-1,884 B), 0.80-0.91 (s4_c8n5 to s8_c16, 2,228-3,276 B), 0.95 (s8_c24_seed1). "
                    "The block-statistics CNNs are therefore not carried into the sweep; the rules are, as the "
                    "baseline the networks must beat",
            "cal_reference": "#97's s8_c24_seed4 with its own decode on the calibration cohorts: 14 of 11,103 "
                             "cal_dlg cases with no line (0.9987 kept), 0.904 of cal_empty with no line (its "
                             "development figures: 3 of 1,515 cases and 0.92)"},
        "amendments": []}
    Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.out, {r["run"]: r["blob_bytes"] for r in doc["candidates"]["new_presence_runs"]})


if __name__ == "__main__":
    main()
