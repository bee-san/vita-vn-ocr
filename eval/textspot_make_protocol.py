#!/usr/bin/env python3
"""Write eval/textspot_protocol.json, the frozen protocol of the full-frame text spotter (issue #104,
docs/textspot.md), before any textspot training run and before any arm is scored on the textspot frames.

  textspot_make_protocol.py --blob NAME=PATH ... --out eval/textspot_protocol.json

Every --blob is checked against the sha256 of its release asset (RELEASE_SHA256 below). The frozen sources' sha256
(SOURCES) are recorded; tests/test_textspot.py fails when one of them changes without an amendment re-pinning it.
Amendments are appended by hand-written dated entries ("amendments"), never by editing what is above them.
"""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
import textdet_model as TM  # noqa: E402

SOURCES = [
    "synth/textspot_twin.py", "eval/textspot_metrics.py", "eval/metrics.py", "synth/textdet_twin.py",
    "synth/common.py", "synth/lines_refine.py", "synth/vertical_twin.py", "eval/textgate.py",
    "runtime/vocr_lines.c", "runtime/vocr_lines.h", "runtime/vocr_lines_rows.c", "runtime/vocr_lines_rows.h",
    "runtime/vocr_lines_refine.c", "runtime/vocr_lines_refine.h", "runtime/vocr_textdet.c", "runtime/vocr_textdet.h",
    "runtime/vocr_textgate.c", "runtime/vocr_textgate.h", "runtime/vocr_vertical.c", "runtime/vocr_vertical.h",
    "runtime/vocr.c", "runtime/vocr.h", "train/textdet_model.py", "train/textdet_export.py",
]
RELEASE_SHA256 = {
    "VOCR5-S": ("v0.3.0/VOCR5_S_w8.vocr", "f175bfcb8cec65d69e85dd5f2cf707c9a711ada7203c0d1099cd2d0142978b8d"),
    "VOCR5-N": ("v0.3.0/VOCR5_N_w8.vocr", "20be179def8033171435ba62388657b3d3c932b365093737948c636e025e3bdc"),
    "VOCR5-L": ("v0.3.0/VOCR5_L_w8.vocr", "50413b37dbd85b2acb0908967bdeaf7ca67803850e0627a7b7cc3d274edc0ba3"),
    "VOCR5-V": ("v0.3.2-vertical.1/VOCR5_V_w8.vocr",
                "ccd20c63bb2620f054538b456904bdf19b74d483d3da8de865009e282bb04172"),
    "s8_c24": ("v0.3.1-textdet.1/textdet_s8_c24_w8.vtd",
               "8a13dbddbc7d6305ccfaf8164ad1c4d9030437fde76eeedca68f24ac5cc47179"),
    "p8_c8": ("v0.3.2-textgate.1/textgate_p8_c8_w8.vtd",
              "a4e73ad78451c75a19f88a479fc0108daac85beaa88d875b73212ed3c09d85d1"),
}
HALF_ABOVE = 600

# Family B: new full-frame runs (train/textspot_train.py). Name: <h2|f1>_<#97 shape>_seed<n>; h2 = half_above 600
# (a frame wider than 600 px is halved: Vita frames read at the PSP's scale), f1 = half_above 0 (full resolution).
FAMILY_B = {
    "h2": ["s4_c8", "s4_c12n4", "s4_c16n4", "s4_c24n4", "s8_c16", "s8_c24", "s8_c32"],
    "f1": ["s8_c8", "s8_c16", "s8_c16n4", "s4_c6"],
}
SEEDS_B = [1, 2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shape_costs(name):
    s = TM.SHAPES[name]
    c = TM.cost(s, 960, 544)
    return {"weights_int8": c["weights_int8"], "params": c["params"], "receptive_field": TM.receptive_field(s),
            "macs": {"960x544": c["macs"], "480x272": TM.cost(s, 480, 272)["macs"]}}


def protocol(blobs):
    shapes = sorted({s for v in FAMILY_B.values() for s in v})
    return {
        "kind": "vita-vn-ocr-textspot-protocol",
        "schema": 1,
        "issue": "https://github.com/bee-san/vita-vn-ocr/issues/104",
        "frozen_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frozen_before": "every textspot training run and every score of any arm (baseline or candidate) on the "
                         "textspot development or test frames",
        "sources_sha256": {p: sha256(ROOT / p) for p in SOURCES},
        "recognizers": {k: {"asset": RELEASE_SHA256[k][0], "sha256": blobs[k]} for k in
                        ("VOCR5-S", "VOCR5-N", "VOCR5-L", "VOCR5-V")},
        "reading": {
            "rule_models": {"horizontal": "VOCR5-N", "vertical": "VOCR5-V"},
            "report_models": {"horizontal": ["VOCR5-S", "VOCR5-L"]},
            "how": "every detection's box (frame pixels, as the arm returns it) is cut from the frame, turned to grey "
                   "((77 R + 150 G + 29 B) >> 8) and read by the C runtime in W8I with greedy CTC decoding, with "
                   "VOCR_INVERT when the box's dark flag (vocr_line_dark on the frame) is 1; a vertical box is turned "
                   "90 degrees anticlockwise first (vocr_column_rotate) and read by VOCR5-V",
        },
        "baselines": {
            "grad": {"what": "vocr_find_lines(frame, VOCR_LINES_GRAD): the whole frame as one region", "boxes": "h"},
            "grad_refine": {"what": "grad, then vocr_lines_refine with VOCR_REFINE_DEFAULT on the frame", "boxes": "h"},
            "grad_refine_gate": {"what": "grad_refine behind #101's gate in boxes mode on the frame (k 4, t -8): no "
                                         "box when the gate says no text",
                                 "gate": {"asset": RELEASE_SHA256["p8_c8"][0], "sha256": blobs["p8_c8"],
                                          "mode": "boxes", "k": 4, "t": -8}},
            "s8c24_textdet": {"what": "#97's detector through its own finder (synth/textdet_twin.find_lines, "
                                      "vocr_textdet_find) on the frame: horizontal and vertical lines, at most 16",
                              "blob": {"asset": RELEASE_SHA256["s8_c24"][0], "sha256": blobs["s8_c24"]}},
            "s8c24_textdet_refine": {"what": "s8c24_textdet with its horizontal lines through VOCR_REFINE_DEFAULT on "
                                             "the frame (#97's det+refine)"},
            "s8c24_spot": {"what": "#97's s8_c24 blob through the textspot decode at full resolution "
                                   "(pack(blob, half_above 0, conf_min 1)): the #97 detector run full-frame"},
            "columns": {"what": "#103's column finder (vocr_find_columns, VOCR_COLUMNS_VALLEYS) on the frame: "
                                "vertical boxes only", "boxes": "v"},
            "grad_refine_columns": {"what": "grad_refine's boxes plus every column of `columns` of which less than "
                                            "half the area lies inside one grad_refine box"},
            "blocks": "every baseline's boxes go through textspot_twin.group_and_order (blocks, order, pick) too",
        },
        "sets": {
            "dev": {
                "role": "the only data the selection reads",
                "vns": "eval/manifests/reserved.json dev_sample (and its release partners)",
                "sources": "the local copies of their VNDB screenshots (realdata and devnative shots), deduplicated "
                           "by VNDB screenshot id and sha256; content flags sexual < 0.3 and violence < 0.5 when known",
                "order": "a frame's rank is sha256('textspot-dev:' + shot id), a VN's sha256('textspot-dev-vn:' + vn)",
                "composition": {
                    "vita": "native 960x544 PS Vita screenshots, up to 5 per VN by rank",
                    "psp": "native 480x272 PSP screenshots, up to 2 per VN by rank",
                    "adrenaline": "for the first 20 PSP VNs by VN rank, the next PSP screenshot by rank not used "
                                  "above, scaled 2x (cv2.resize INTER_LINEAR) to 960x544",
                    "pc": "for the first 40 VNs with PC (hires) screenshots by VN rank, the first by rank, scaled "
                          "to fit 960x544 with its aspect (cv2.INTER_AREA), as #97's real canvases",
                    "text_free": "up to 25 frames with no text at all by eye, drawn by rank from the dev VNs' "
                                 "remaining screenshots where MeikiOCR detects at most one box",
                    "dropped": "a frame that is not a game screen (box art, photo, a mostly unreadable thumbnail) "
                               "is dropped by the reviewer and the next by rank is taken",
                },
                "pinned_by": "an amendment with the manifest's sha256, before any arm is scored on it",
            },
            "test": {
                "role": "read once, after the selection is committed",
                "frames": "every vita-test (74) and psp-test (83) screenshot as a full frame: eval's hand-checked "
                          "lines are its dialogue (lines tagged name: class name; uncertain lines: ignore) and every "
                          "other text instance is labelled as the dev set is; plus up to 25 text-free screenshots "
                          "of the 28 test games outside eval's screenshots; report only: psp-test scaled 2x "
                          "(Adrenaline)",
                "pinned_by": "an amendment with the manifest's sha256, before the selection commit",
                "rule": "no model of this work runs on a test frame before the conductor's single read",
            },
            "protected": "the CLANNAD long run (owner-drive-ingest's frozen references): whole frames in, each "
                         "reference line reads the detections lying mostly inside its GRAD box (at least half of a "
                         "detection's area in the box it overlaps most), dialogue CER with whole-page intervals; "
                         "the boxes returned on pages without reference text are counted. Once, after the selection "
                         "commit, with the test set",
        },
        "labels": {
            "instance": "one line of horizontal text (characters on one baseline; a gap wider than about 1.5 "
                        "character widths or a separate UI element starts a new one) or one vertical column",
            "box": "the main-text ink box in frame pixels (fill, outline and shadow), refined to ink from an "
                   "approved proposal; furigana, emphasis dots, page cursors, icons and button glyphs are not part "
                   "of it",
            "classes": {"dialogue": "story text: ADV and NVL dialogue and narration, monologue, balloon text",
                        "name": "speaker name plates", "choice": "in-story choice options",
                        "ui": "menus, buttons, system and HUD labels, dates, chapter titles, save slots, ...",
                        "scene": "text that is part of the picture: signs, posters, letters, screens, ..."},
            "ignore": "furigana (an ignore area over the ruby), logos and stylised art text, text under 6 px tall "
                      "in frame pixels or otherwise unreadable, watermarks and capture overlays, scripts other than "
                      "Japanese and Latin, rotated or curved text, and any instance whose text two looks could not "
                      "agree on",
            "text": "eval's convention (main text only; full-width ！？～; … for ellipses; eval/metrics.norm), "
                    "Latin as written",
            "review": "proposals: MeikiOCR's detector boxes (with its vertical flag) refined to ink, read by MeikiOCR "
                      "and PP-OCRv6 small (turned 90 degrees for a vertical box); pass 1 decides each frame's boxes "
                      "(keep, drop, ignore, split, merge, class, orientation, and missed text as a rough rectangle "
                      "refined to ink) and every kept box's text from numbered candidates; pass 2, a separate "
                      "session, rechecks the boxes and reads the text blind (candidates shuffled and unattributed); "
                      "a disagreement goes to a third look; only text two looks agree on is kept. Reviewer kind "
                      "'assistant' (Claude); no Codex, no manga-ocr",
        },
        "metrics": {
            "module": "eval/textspot_metrics.py (frozen above)",
            "detection": "P / R / F1 at IoU >= 0.5 one to one against the references' ink boxes padded by the "
                         "spotter's pad rule; recall per class; ink tightness (mean IoU, mean ink coverage of matched "
                         "pairs); detections mostly inside an ignore area left out",
            "end_to_end": "frame units: every reference's Levenshtein errors of its assigned detections' joined "
                          "readings, plus every character read from a detection assigned to no reference; all text "
                          "and dialogue (classes dialogue and choice, no junk); CER pooled over frames",
            "text_free": "the share of text-free frames with any box, and boxes per text-free frame (report)",
            "pick": "report only: on frames with dialogue, the picked block's box against the union of the "
                    "padded dialogue references (IoU >= 0.5), and the CER of the picked block's readings joined in "
                    "its order against the dialogue references joined top to bottom",
            "intervals": "10,000 VN-cluster bootstrap resamples, seed 0 (textspot_metrics.vn_diff, f1_diff)",
            "per_platform": "report: vita, psp, adrenaline, pc and text-free rows",
        },
        "candidates": {
            "family_a": {"what": "#97's 34 trained runs (rounds 1 and 3; their final.vtd) through the textspot "
                                 "decode, each as pack(blob, half_above, 1) with half_above 0 and 600; no training"},
            "family_b": {
                "what": "new runs trained for whole frames (train/textspot_train.py); name <scale>_<shape>_seed<n>",
                "scales": {"h2": {"half_above": HALF_ABOVE}, "f1": {"half_above": 0}},
                "shapes": FAMILY_B, "seeds": SEEDS_B,
                "shape_costs": {s: shape_costs(s) for s in shapes},
                "recipe": "#97's network, targets (K kernel band, O, A1 / A2) and losses on windows cut from whole "
                          "frames at the network's input scale (a frame wider than half_above halved by "
                          "textspot_twin.halve first); 30,000 steps of 16 windows of 256 x 512; AdamW 4e-3, cosine; "
                          "int8 fake quantisation from step 24,000; half synthetic full frames and half #97's real "
                          "whole-screenshot canvases per batch; the last step is the run's model; exported by "
                          "train/textdet_export.py and wrapped by textspot_twin.pack(half_above, conf_min 1). The "
                          "data pin and the remaining details are fixed by an amendment before the first run",
                "data": "synthetic whole frames (Vita, PSP, Adrenaline 2x, PC at the Vita size, text-free) with exact "
                        "ink boxes over textgate's clean scene pool; #97's real canvases (realplus, teacher, "
                        "devnative, owner); excluded: every reserved.json test and dev_sample VN and test session, "
                        "every eval and eval/linefind screenshot and linefind VN, the owner's CHAOS;CHILD (v14018)",
            },
            "decodes": {"det": "the textspot decode", "det+refine": "the same, then its horizontal boxes through "
                                                                     "VOCR_REFINE_DEFAULT on the frame"},
        },
        "rule": {
            "arena_max": 131072,
            "ms_max_333_960": 150,
            "eligible": "arena at 960x544 <= 131,072 B (the runtime's exact function; for det+refine the larger of "
                        "the spotter's and the refinement's); estimated time per 960x544 frame on one A9 core at "
                        "333 MHz <= 150 ms (the A9 cycle model's high bound for the shape and scale on the runtime, "
                        "plus 1.4 ms per horizontal box at the arm's median box count on the dev Vita frames for "
                        "det+refine); the C runtime gives the twin's boxes on every dev frame",
            "size_order": "VTSP blob bytes, then MACs per 960x544 frame, then det before det+refine",
            "qualifies": "against every baseline: dev F1 difference with a VN-cluster 95% interval entirely above 0, "
                         "dev end-to-end CER (all text) difference with an interval entirely below 0, and the "
                         "dialogue CER difference not resolved above 0",
            "winner": "the smallest qualifying candidate W; a larger qualifying candidate replaces W only when its "
                      "all-text CER minus W's has an interval entirely below 0 (the smallest such, repeated)",
            "no_winner": "the verdict is negative; the eligible candidate with the lowest dev all-text CER is the "
                         "report candidate, read once on the test and protected sets and labelled not promoted",
            "code": "eval/textspot_metrics.apply_rule",
        },
        "estimates": {
            "a9": "vita/a9_cycles.py's model on the Vita compiler's code under the plugin qemu-arm, as #97 "
                  "(eval/textdet_a9.py), on a pinned procedural 960x544 and 480x272 frame; high bound x 1.0007 at "
                  "333 MHz, x 333/444 at 444 MHz",
            "mac_rate": "VOCR3-H15's console rate: 1,564,992 MACs per CTC frame at 5.0-5.2 ms at 333 MHz (report)",
            "label": "every console time and memory figure is a host estimate; nothing runs on a console",
        },
        "amendments": [],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blob", action="append", default=[], help="NAME=PATH, NAME in " + ", ".join(RELEASE_SHA256))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    blobs = {}
    for spec in a.blob:
        name, _, path = spec.partition("=")
        digest = sha256(path)
        if RELEASE_SHA256[name][1] != digest:
            raise SystemExit(f"{name}: {path} is not the release asset {RELEASE_SHA256[name][0]}")
        blobs[name] = digest
    missing = set(RELEASE_SHA256) - set(blobs)
    if missing:
        raise SystemExit(f"missing --blob for {sorted(missing)}")
    Path(a.out).write_text(json.dumps(protocol(blobs), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(a.out)


if __name__ == "__main__":
    main()
