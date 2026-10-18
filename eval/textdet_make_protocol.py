#!/usr/bin/env python3
"""Writes eval/textdet_protocol.json (issue #96) from the frozen baseline and the sources' hashes.

  textdet_make_protocol.py --baseline eval/textdet_baseline_20261017.json --out eval/textdet_protocol.json

Run once at freezing; amendments are appended by hand (each with its UTC time and reason) and never rewrite an
earlier field.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
import textdet_model as TM  # noqa: E402

FROZEN_SOURCES = ("runtime/vocr_lines.c", "runtime/vocr_lines.h", "runtime/vocr_lines_rows.c",
                  "runtime/vocr_lines_rows.h", "runtime/vocr_lines_refine.c", "runtime/vocr_lines_refine.h",
                  "synth/common.py", "synth/lines_refine.py", "eval/finder_tighten.py", "eval/textdet.py",
                  "train/general_vn_textbox_guard.py", "runtime/vocr.c", "runtime/vocr.h",
                  "eval/manifests/reserved.json", "eval/manifests/vita-test.jsonl", "eval/manifests/psp-test.jsonl",
                  "eval/linefind/manifests/vita_dev.jsonl", "eval/linefind/manifests/psp_dev.jsonl",
                  "eval/linefind/manifests/vita_holdout.jsonl", "eval/linefind/manifests/psp_holdout.jsonl",
                  "train/textdet_model.py")
RECOGNIZERS = {
    "VOCR5-S": {"release": "v0.3.0", "asset": "VOCR5_S_w8.vocr", "run": "v5f_S_seed2", "blob_bytes": 819792,
                "sha256": "f175bfcb8cec65d69e85dd5f2cf707c9a711ada7203c0d1099cd2d0142978b8d"},
    "VOCR5-N": {"release": "v0.3.0", "asset": "VOCR5_N_w8.vocr", "run": "v5f_N_seed3", "blob_bytes": 1185632,
                "sha256": "20be179def8033171435ba62388657b3d3c932b365093737948c636e025e3bdc"},
    "VOCR5-L": {"release": "v0.3.0", "asset": "VOCR5_L_w8.vocr", "run": "v5f_L_seed2", "blob_bytes": 2473696,
                "sha256": "50413b37dbd85b2acb0908967bdeaf7ca67803850e0627a7b7cc3d274edc0ba3"},
}
SEED2 = ("s4_c4", "s4_c8", "s4_c16", "s8_c8", "s8_c16", "s8_c32")
H15_MACS_PER_FRAME = 1564992


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--frozen", required=True, help="UTC time of freezing")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    wd = base["wd"]
    runs = [f"{name}_seed1" for name in TM.SHAPES] + [f"{name}_seed2" for name in SEED2]
    doc = {
        "kind": "textdet-protocol", "issue": "https://github.com/bee-san/vita-vn-ocr/issues/96", "schema": 1,
        "frozen": args.frozen,
        "rule_of_this_file": "Frozen before any detector training run (including speed probes). Amendments are "
                             "appended to 'amendments' with their UTC time and reason, before the score they could "
                             "affect is computed; no field above them is rewritten.",
        "owner_steering": [
            "2026-10-17 21:15Z: complete issue #96.",
            "2026-10-17 21:16Z: we need it to run on a ps vita, so smaller == better; play around with sizings."],
        "sources_sha256": {f: sha256(ROOT / f) for f in FROZEN_SOURCES},
        "recognizers": RECOGNIZERS,
        "finders": {
            "grad": "runtime/vocr_lines.c VOCR_LINES_GRAD (frozen; bit-exact twin synth/common.find_lines_ex)",
            "refine": "GRAD then runtime/vocr_lines_refine.c with VOCR_REFINE_DEFAULT = 0x1F "
                      "(twin synth/lines_refine.py)",
            "det": "a detector candidate: the int8 blob through runtime/vocr_textdet.c (C) and synth/textdet_twin.py "
                   "(its bit-exact Python twin); every development region is run through both and must agree box for "
                   "box, or the candidate is ineligible"},
        "development": {
            "wd": {
                "what": "VOCR5's development whole dialogue (eval/vocr5_select.load_wd): finder-tighten's tv_dlg and "
                        "dn_dlg regions of the reserved.json dev_sample VNs, tv_dlg plus the dn_dlg frames tv_dlg does "
                        "not hold, base / +4 / +8 px; every crop a finder returns (vertical-flagged boxes excepted) is "
                        "read by the VOCR5 model in W8I through the C runtime with VOCR_INVERT from the box's dark "
                        "flag, joined in the finder's order and scored by transcript_score (all/full): dialogue CER, "
                        "the regions hold no speaker name",
                "cohorts": wd["cohorts"], "units": wd["cer"]["VOCR5-N"]["grad"]["units"],
                "vns": wd["cer"]["VOCR5-N"]["grad"]["vns"],
                "primary": "all three variants pooled",
                "bootstrap": "VN clusters, 10,000 resamples, numpy default_rng(0), sorted VN ids, linear 2.5/97.5 "
                             "percentiles (eval/textdet.vn_diff; one pair = eval/vocr3_report.cluster_bootstrap)"},
            "native_vita": {
                "what": "the CLANNAD-dev-style subset: the wd units whose frame is a native PS Vita screenshot "
                        "(teacher_val frames whose teacher read is of kind 'vita', devnative's Vita frames): 960x544 "
                        "native Vita text, where finder-tighten's refinement was level on development and worse on "
                        "CLANNAD (docs/finder-tighten.md)",
                "units": 150, "base_regions": 50, "chars": 3738,
                "vns": ["v10383", "v16128", "v16770", "v18161", "v18163", "v18432", "v18643", "v18857", "v21004"]},
            "empty": {
                "what": "eval/linefind's 'empty' regions (text-free regions from the upper screen) of vita_dev and "
                        "psp_dev, regions of a reserved test VN left out (vita_dev v36, psp_dev v1267 / v4060): 66 "
                        "regions; empty rate = the share for which the finder returns no line",
                "regions": base["linefind_pooled"]["development"]["grad"]["empty"]["empty_regions"]},
            "linefind": "line recall / precision / F1 (IoU >= 0.5), region exact, merged / split / cut per set and "
                        "tag on vita_dev, psp_dev (development) and vita_holdout, psp_holdout (report only); never "
                        "research_a/b",
            "report_only": [
                "the owner's CHAOS;CHILD (v14018) native Vita whole dialogue (finder-tighten's own_dlg Vita units): "
                "the detector never trains on v14018, the VOCR5 models did (in-sample for the recognizers), so it is "
                "a secondary native-Vita check that decides nothing",
                "the dn pairs of the dev_sample VNs: the detector's box for the line against the hand-cut ink crop",
                "gate arms: GRAD and GRAD + refine with every line dropped when the detector returns no text for the "
                "region (an empty-region gate in front of the hand-written finder)"]},
        "baseline": {
            "computed": wd.get("computed", base["computed"]),
            "file": "eval/textdet_baseline_20261017.json",
            "wd_dialogue_cer_all_variants": {a: {m: wd["cer"][a][m]["all"]["cer"] for m in ("grad", "refine")}
                                             for a in wd["cer"]},
            "wd_native_vita_cer": {a: {m: wd["cer"][a][m]["native_vita"]["cer"] for m in ("grad", "refine")}
                                   for a in wd["cer"]},
            "empty_rate_dev": {f: base["linefind_pooled"]["development"][f]["empty"]["empty_rate"]
                               for f in ("grad", "refine")},
            "note": "the wd units equal VOCR5's stored selection units for the three selected runs, unit for unit"},
        "detector": {
            "family": "train/textdet_model.py: a stem (3x3/2), a downsampling stage to output stride 4 or 8, 3-5 "
                      "blocks (separable: 3x3 depthwise with dilation + 1x1; or plain 3x3 with dilation) and a 1x1 "
                      "head with 4 maps per cell: K (text kernel), O (vertical), A1 / A2 (distances to the line's near "
                      "and far edges across the line). BatchNorm folded; int8 weights, int8 activations, int32 "
                      "accumulators, fixed-point requantisation: integer arithmetic only",
            "targets": {
                "box": "a line's main-text ink box (fill, outline and shadow; furigana, emphasis dots, cursors and "
                       "icons excluded) on its long axis and across it",
                "kernel": "cells whose centre lies within max(h/4, s/2) of the line's centre line across it (h = ink "
                          "size across the line, s = stride) and whose span meets the ink box along it",
                "A1_A2": "for kernel cells, the distances in px from the cell centre to the ink box's near and far "
                         "edges across the line (int8 code = 2 x px, clamped to 0..127)",
                "O": "1 on the kernel cells of vertical lines, 0 on horizontal ones (loss on kernel cells only)",
                "ignore": "areas the generator marks unlabelled (unknown or unreadable text) carry no loss"},
            "decode": {
                "kernel": "cells with K code > 0, 4-connected components",
                "drop": "components of fewer than 2 cells (stride 4) or 1 cell (stride 8); boxes under 4 px across",
                "orientation": "vertical iff the sum of the component's O codes is > 0",
                "across": "with c2 = 2 x a cell's centre across the line (2*s*i + s - 1) and n cells: near edge = "
                          "floor(sum(c2 - A1) / (2n)), far edge = ceil(sum(c2 + A2) / (2n)), integers only",
                "along": "the component's first cell x stride to (last cell + 1) x stride",
                "pad": "max(2, (h * 18 + 50) / 100) px on every side (h = far - near), then clipped to the region",
                "order": "horizontal lines top to bottom (a line whose rows overlap the previous one's by half the "
                         "smaller height goes left to right), then vertical lines right to left",
                "cap": "16 lines (VocrLineSet), the rest dropped and truncated set",
                "dark": "vocr_line_dark on the padded box (the GRAD finder's polarity rule, unchanged)",
                "text_present": "at least one line returned",
                "vertical_boxes": "flagged, never handed to the horizontal recognizers (not read in wd, protected or "
                                  "CLANNAD scoring)"},
            "constants_rule": "the decode constants above are fixed here; they may change only by an amendment made "
                              "before any candidate's wd score exists"},
        "sweep": {
            "shapes": {name: {**s, **{k: v for k, v in TM.cost(s, 900, 150).items() if k in ("macs", "params")},
                              "macs_960x544": TM.cost(s, 960, 544)["macs"],
                              "macs_480x272": TM.cost(s, 480, 272)["macs"],
                              "macs_450x80": TM.cost(s, 450, 80)["macs"], "receptive_field": TM.receptive_field(s)}
                       for name, s in TM.SHAPES.items()},
            "runs": runs,
            "seeds": "seed 1 for every shape, seed 2 for " + ", ".join(SEED2),
            "outside_the_rule": "an int4-weight study (fake-quantised 4-bit weights of 2-3 trained shapes, report "
                                "only: the runtime runs int8) and the strip/row-buffer arena table (the runtime "
                                "streams rows exactly, so a strip height changes no output)",
            "shared_trunk": "considered and not swept: the recognizers read 32-px line crops resized after detection, "
                            "so no layer of theirs can run on the region before the lines are known"},
        "training": {
            "where": "train/textdet_train.py through HEAVY_SLOTS=12 $HERD/heavy.sh, 5 torch threads per run",
            "steps": 30000, "batch": 16, "window": [192, 512],
            "optimizer": "AdamW, lr 4e-3, 500 warm-up steps, cosine to 1e-4, weight decay 1e-4 (not on BN / bias)",
            "loss": "K: BCE + dice over valid cells; O: BCE on kernel cells x 0.5; A1/A2: smooth L1 on kernel cells "
                    "(px / 8)",
            "qat": "int8 fake quantisation (weights per output channel, activations per tensor) from step 24,000",
            "mix": "synthetic 0.5, real 0.5 per batch",
            "selection_inside_a_run": "none: the last step is the run's model (no checkpoint is picked on any score)"},
        "data": {
            "generator": "synth/textdet_data.py: pinned shards (sha256 manifest committed before the first "
                         "training run) of grey canvases with line boxes (main-text ink box, orientation, role) and "
                         "ignore areas",
            "synthetic": "synth3 version 3's region renderer (its ink, fill and ruby masks give exact boxes) in "
                         "canvases of several sizes (the region, loose crops, whole screens on text-free "
                         "backgrounds), plus vertical-text renders and empty negatives (boxes, scenes and screens "
                         "with no text)",
            "real": "MeikiOCR line boxes refined to ink and its detector boxes (with their vertical flag) from "
                    "realplus, teacher, devnative (training part) and the owner's frames, cut into regions like a "
                    "player's plus text-free regions",
            "excluded": ["every reserved.json 'test' VN (vita-test, psp-test, CLANNAD family and their release "
                         "partners) and 'dev_sample' VN, and every 'sessions' release",
                         "every screenshot of eval's sets (vita-test, psp-test), of eval/linefind's six manifests and "
                         "every VN of vita_dev, vita_holdout, psp_dev, psp_holdout",
                         "every frame of the wd cohorts (by shot id) and of finder-tighten's own_dlg Vita units (the "
                         "owner's CHAOS;CHILD v14018)",
                         "synthetic scenes and names: synth3's own exclusions (CLANNAD manifest, test VNs)"],
            "check": "eval/tools/reserve_check.py on every real shard, recorded"},
        "promotion_rule": {
            "eligible": "int8 blob < 100,000 B; arena at 960x544 <= 131,072 B (the C runtime's exact function); "
                        "<= 30,000,000 MACs on a 900x150 region; C == twin on every development region",
            "size_order": "blob bytes ascending, then MACs on a 900x150 region, then run name",
            "qualifies": "for each of VOCR5-S, -N, -L: the wd CER difference det - GRAD and det - refine (points, all "
                         "variants pooled) has a VN-cluster 95% interval entirely below 0; on the native_vita subset "
                         "no such difference (det - GRAD, det - refine, any model) has an interval entirely above 0; "
                         "and the development empty rate is >= 0.75",
            "winner": "W = the smallest qualifying candidate; then repeatedly, among the qualifying candidates larger "
                      "than W, the smallest whose mean over the three models of its wd CER difference to W has a "
                      "VN-cluster interval entirely below 0 becomes W; the final W is selected",
            "no_winner": "if no candidate qualifies the verdict is negative: GRAD (+ refine where the overlay uses it) "
                         "stays, and the protected sets are read once for the eligible candidate with the lowest mean "
                         "wd CER over the three models, labelled not promoted"},
        "fine_tune": {
            "trigger": "no candidate qualifies, and an eligible candidate passes the native_vita and empty conditions "
                       "with a wd CER below GRAD's and refine's (point estimates) for every model",
            "recipe": "for each tier, the released checkpoint fine-tuned 3,000 steps with VOCR5's arm F recipe, the "
                      "GRAD-crop and refined-crop sources replaced by that candidate's crops of the same frames; then "
                      "the rule re-applied with the fine-tuned models in every arm (GRAD, refine and det)",
            "otherwise": "not run"},
        "protected": {
            "when": "once, after the selection (or the no-winner record) is committed and pushed; attempt records are "
                    "created exclusively and refuse a second run",
            "sets": [
                "vita-test + psp-test whole dialogue: eval's regions (an uncertain line leaves its region out), base "
                "/ +4 / +8, GRAD, refine and det with VOCR5-S/N/L, body and all text, VN-cluster intervals",
                "vita-test + psp-test detection: the same regions against eval's line boxes (line R / P / F1, region "
                "exact), GRAD, refine and det",
                "CLANNAD long run (owner-drive-ingest's frozen references, refs_sha256 35411688...): each page's "
                "ROI; a reference line reads the finder lines whose rows lie mostly in its GRAD box (most of a "
                "line's height inside the box), joined in order; whole-page bootstraps; plus the number of lines "
                "returned on the "
                "pages and boxes the references hold no text for"],
            "fixed_lines": "unchanged (no finder)"},
        "estimates": {
            "label": "every console time and memory figure is a host estimate until measured on a console",
            "mac_rate": f"VOCR3-H15 ran {H15_MACS_PER_FRAME} MACs per CTC frame at 5.0-5.2 ms on one A9 core "
                        "(v0.8.0-test.1, 333 MHz assumed): 0.301-0.313 GMAC/s at 333 MHz, x444/333 at 444 MHz",
            "cycle_model": "vita/a9_cycles.py on the Vita compiler's code of runtime/vocr_textdet.c under plugin "
                           "qemu-arm, with ocr-speed's calibration factor (x1.0007 of the high bound at 333 MHz)",
            "regions": {"vita_box": [900, 150], "psp_box": [450, 80], "vita_screen": [960, 544],
                        "psp_screen": [480, 272]}},
        "amendments": []}
    Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(runs)} runs, {len(TM.SHAPES)} shapes")


if __name__ == "__main__":
    main()
