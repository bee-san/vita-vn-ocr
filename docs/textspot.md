# textspot: full-screen text spotting (issue #104)

Every finder so far works inside a region the player draws first: GRAD and its refinement, #97's learned detector,
#101's text-presence gate and #103's column finder. textspot takes a **whole frame** and returns **every text
instance** on it, without a drawn region:
- the frame is a PS Vita 960×544 screen, a PSP 480×272 screen, Adrenaline's PSP picture at 2× or a PC screenshot
  scaled to the Vita;
- the instances are dialogue, speaker names, menus, choices, UI labels, signs in a CG, vertical columns and manga
  balloons;
- each instance comes back as a box in screen coordinates, with an orientation, a confidence, a block and a reading
  order;
- one block is picked as the likely dialogue.

Each box can then be read with VOCR5 (horizontal) or VOCR5-V (vertical). In Rubi this is a "scan screen" mode:
- every text box is highlighted;
- the cursor or a touch picks one, and its text goes to the existing popup and lookup;
- the cursor starts on the dialogue block, so the player never draws a region.

This document starts as the pre-registered protocol, frozen before any training run and any score
(`eval/textspot_protocol.json`, written by `eval/textspot_make_protocol.py`). Results are added below it as they are
measured. Nothing above "Results" changes except through a dated amendment, listed in the protocol's
`amendments`.

## Protocol

### The spotter

`synth/textspot_twin.py` is the specification and the bit-exact twin of the C runtime (`runtime/vocr_textspot.c`, to
be written).

- **The network** is #97's int8 detector, unchanged: the VTDT layout, integer arithmetic only and four maps per
  output cell (`synth/textdet_twin.py`):
  - K: the text kernel;
  - O: vertical text;
  - A1 and A2: the line's near and far edges across it.

  A **VTSP** blob wraps a VTDT blob in a 32-byte header with two decode parameters:
  - `half_above`: a frame wider than this is halved (an exact 2×2 mean) before the network. With 600, a Vita frame is
    read at the PSP's scale and a PSP frame as it is;
  - `conf_min`: 1, so every candidate is kept.
- **The frame decode** (integers only):
  - kernel cells K > 0, in 4-connected components;
  - per component:
    - a confidence (the mean K code, 1..127);
    - an orientation (ΣO > 0);
    - the edges across the line from A1 and A2, scaled to frame pixels;
    - the extent along the line from its cells;
    - #97's pad: max(2, 18% of the size across) on every side, clipped;
  - at most 64 boxes, the most confident first (truncated when more).
- **Blocks and reading order:**
  - Two boxes of one orientation join a block when all three hold:
    - they overlap along the line;
    - the gap across is at most the larger one's size across;
    - the two sizes across differ by at most 3/2.
  - Inside a block, horizontal lines run top to bottom (side by side ones left to right) and vertical columns right
    to left.
  - Blocks run in rows, top to bottom, left to right; a row of vertical blocks runs right to left.
- **Pick:** the block with the largest text area, doubled in the lower half of the frame, is the likely dialogue.
- **Dark flag:** `vocr_line_dark` on the frame, as every finder's.

The same block, order and pick rules run on every baseline's boxes (`group_and_order`), so the pick can be compared.

### Arms

**Reading.** Every box is cut from the frame and read by the C runtime in W8I with greedy decoding:
- with `VOCR_INVERT` when its dark flag is 1;
- a vertical box is turned 90° anticlockwise and read by VOCR5-V;
- the rule reads horizontal boxes with VOCR5-N (the default tier); VOCR5-S and VOCR5-L are report only.

The protocol pins the four recognizers' sha256.

**Baselines** (every one must be beaten):

| name | what |
|---|---|
| `grad` | GRAD with the whole frame as one region |
| `grad_refine` | GRAD, then `VOCR_REFINE_DEFAULT` |
| `grad_refine_gate` | `grad_refine` behind #101's `p8_c8` gate (boxes mode, k 4, t −8): no box when the gate says no text |
| `s8c24_textdet` | #97's `s8_c24` through its own finder (at most 16 lines, horizontal and vertical) |
| `s8c24_textdet_refine` | the same with its horizontal lines refined (#97's det+refine) |
| `s8c24_spot` | `s8_c24` through the textspot decode at full resolution: #97's detector run full-frame |
| `columns` | #103's column finder (`VOCR_COLUMNS_VALLEYS`): vertical boxes only |
| `grad_refine_columns` | `grad_refine` plus each column of which less than half lies inside one of its boxes |

**Candidates**, all as VTSP blobs. Each is scored as `det` (the textspot decode) and `det+refine` (its horizontal
boxes through `VOCR_REFINE_DEFAULT`).
- **Family A**: #97's 34 trained runs, at `half_above` 0 and 600, with no training.
- **Family B**: new runs trained for whole frames, seeds 1 and 2, `<scale>_<shape>_seed<n>`:
  - `h2` (`half_above` 600): `s4_c8`, `s4_c12n4`, `s4_c16n4`, `s4_c24n4`, `s8_c16`, `s8_c24`, `s8_c32`;
  - `f1` (full resolution): `s8_c8`, `s8_c16`, `s8_c16n4`, `s4_c6`.

  This is the stride and downsampling study. At `h2`, stride 4 is 8 frame pixels per cell on a Vita frame and the
  network costs what it costs on a PSP frame, so wider trunks fit the time limit. At `f1`, only the cheapest shapes
  fit.

  The recipe is #97's (targets, losses, QAT, 30,000 steps of 16 windows, AdamW 4e-3 cosine), on windows of 256×512
  cut from whole frames at the network's input scale:
  - half of each batch is synthetic whole frames: Vita, PSP, Adrenaline 2×, PC at the Vita size and text-free frames,
    with exact ink boxes, over textgate's clean scene pool;
  - half is #97's real whole-screenshot canvases: MeikiOCR's boxes refined to ink, with the vertical flag, from
    realplus, the teacher, devnative and the owner's frames.

  The data pin is an amendment made before the first run.

**Excluded from all training:**
- every `reserved.json` test and dev_sample VN, and every test session;
- every eval and eval/linefind screenshot, and every linefind VN;
- CHAOS;CHILD (v14018).

### Data

**Development** (the only data the rule reads), from the `dev_sample` VNs' local VNDB screenshots. A frame's rank is
sha256(`textspot-dev:` + shot id), a VN's is sha256(`textspot-dev-vn:` + vn).

| part | frames |
|---|---|
| vita | native 960×544, up to 5 per VN |
| psp | native 480×272, up to 2 per VN |
| adrenaline | for the first 20 PSP VNs, the next PSP screenshot, scaled 2× bilinear to 960×544 |
| pc | for the first 40 VNs with PC screenshots, one, scaled to fit 960×544 with its aspect |
| text-free | up to 25 frames with no text by eye |

**Test** (read once, after the selection is committed):
- every vita-test (74) and psp-test (83) screenshot as a full frame. Their hand-checked eval lines are the dialogue
  (lines tagged `name`: class name; uncertain lines: ignore), and every other text instance is labelled as the
  development set is;
- up to 25 text-free screenshots of the 28 test games;
- report only: psp-test at 2× (Adrenaline).

No model of this work runs on a test frame before that single read.

**CLANNAD** (once, with the test set): the frozen long-run references, with whole frames in. Each reference line
reads the detections that lie mostly inside its GRAD box. The result is dialogue CER with whole-page intervals, plus
the boxes returned on pages without reference text.

**Labels:**
- An instance is one horizontal line, or one vertical column, of one of five classes: `dialogue`, `name`, `choice`,
  `ui` or `scene`.
- Its box is the main-text ink box: fill, outline and shadow; furigana, cursors and icons are left out.
- Ignore areas, which are neither found nor missed:
  - furigana;
  - logos and stylised text;
  - text under 6 px tall;
  - watermarks;
  - other scripts;
  - rotated text;
  - text two looks could not agree on.
- Text follows eval's convention, with Latin as written.
- Review:
  - Proposals: MeikiOCR's boxes refined to ink, with candidate readings from MeikiOCR and PP-OCRv6 small.
  - Pass 1 decides the boxes, the classes, the orientations and any missed text, then the text.
  - Pass 2, in a separate session, rechecks the boxes and reads the text blind; disagreements get a third look.
  - Reviewer: Claude; no Codex.

The development and test manifests are pinned by sha256 in amendments: the development one before any arm is scored
on it, the test one before the selection commit.

### Metrics (`eval/textspot_metrics.py`, frozen)

- **Detection:** P / R / F1 at IoU ≥ 0.5, one to one, against the references' ink boxes padded by the spotter's pad
  rule.
  - Recall is given per class.
  - Ink tightness: the mean IoU and mean ink coverage of matched pairs.
  - A detection with at least half its area inside an ignore area is left out.
- **End to end:**
  - A detection goes to the reference whose padded box holds the largest part of it, if that part is at least half
    of the detection.
  - A reference's detections are joined along it, and the reference scores Levenshtein(norm(text), norm(reading)).
  - A detection that goes nowhere adds every character it reads.
  - Two figures: all-text CER, and dialogue CER (classes dialogue and choice, without the unassigned readings).
- **Text-free frames:** the share with any box, and boxes per frame. **Pick** (report only): the picked block against
  the dialogue references (IoU ≥ 0.5, and the CER of its joined readings).
- **Intervals:** 10,000 VN-cluster bootstrap resamples, seed 0.
- Per-platform rows (vita, psp, adrenaline, pc, text-free) are reported.

### The rule (pre-registered; `textspot_metrics.apply_rule`)

- **Eligible:**
  - arena at 960×544 ≤ **131,072 B** (exact; for det+refine, the larger of the two arenas);
  - estimated time per 960×544 frame on one A9 core at 333 MHz ≤ **150 ms**. This is the A9 cycle model's high
    bound on the runtime's code for the shape and scale, plus 1.4 ms per refined line for det+refine;
  - the C runtime gives the twin's boxes on every development frame.
- **Size order:** VTSP bytes, then MACs per 960×544 frame, then det before det+refine.
- **Qualifies:** against **every** baseline, all three hold:
  - the development F1 difference has a VN-cluster 95% interval entirely above 0;
  - the all-text end-to-end CER difference has an interval entirely below 0;
  - the dialogue CER difference is not resolved above 0.
- **Winner:** the smallest qualifying candidate. A larger qualifying candidate replaces it only when its all-text CER
  is resolved below the winner's; this repeats until none is left.
- **No winner:** the verdict is negative. The eligible candidate with the lowest development all-text CER is read
  once on the test and protected sets, labelled not promoted.

### Estimates

- **Binding:** the A9 cycle model, as #97 used it: the Vita compiler's code under the plugin qemu-arm and
  `vita/a9_cycles.py`, on a pinned procedural frame at 960×544 and 480×272.
- **Report only:** the MAC rate of VOCR3-H15 on the console (1,564,992 MACs per CTC frame at 5.0-5.2 ms).
- Times are given at 333 and 444 MHz.
- Sizes are exact: VTSP bytes, the arena, and VitaSDK `-Os` `.text`.
- Every console figure is a host estimate: nothing runs on a console (hardware work is paused).

### Amendments

Each amendment is dated in `eval/textspot_protocol.json` and precedes the runs and scores it names.

1. **Data and recipe** (2026-10-19T11:33Z, before every family B run and every score).
   - **Synthetic half:** `data/manifests/textspot_shards_v1.json`, 80,000 whole frames:
     - by kind: vita 27,828, psp 20,063, adrenaline 11,992, pc 8,047, text-free 12,070;
     - 501,887 lines: dialogue 114,822, name 38,831, choice 23,908, ui 297,494, scene 26,832; 35,220 of them vertical.

     The QA saw unlabelled scene text twice in 60+ frames; it stays.
   - **Real half:** #97's 28,602 whole-screenshot canvases, minus 337 the protocol excludes and #97 did not:
     - screenshots of #103's vertical sets and of eval/linefind's candidate lists and `psp_pool`;
     - every canvas of the 23 VNs of those lists.
   - **Recipe:**
     - per-window synthetic draw p 0.5 (#97's);
     - 256×512 windows at uniform offsets;
     - the exact halving for `h2`;
     - every class a positive;
     - #97's round-1 targets, losses and schedule.
   - **Eligibility basis:** the runtime (`runtime/vocr_textspot.c`) and its A9 / arena table
     (`eval/textspot_a9_20261019.json`). The arena is `vocr_textspot_arena_bytes(m, 960, 544)` (any frame up to
     960×544). The time is the default call on a grey frame. All 11 family B shapes are eligible: 44.8-136.4 ms and
     ≤ 80,799 B.

2. **The development set and the harness** (before any score on a development frame).
   - `eval/manifests/textspot-dev.jsonl`: 188 frames of 81 dev_sample VNs.
     - Parts: vita 39, psp 64, adrenaline 20, pc 40, text-free 25.
     - 885 instances (ui 519, dialogue 249, name 96, choice 15, scene 6; 41 vertical) with 6,770 characters
       (dialogue measure: 3,987), and 196 ignore areas.
     - Review agreement: pass 2 approved 96.7% of the boxes and agreed on 97.7% of the texts.
   - The harness (`eval/textspot.py`, `eval/textspot_protected.py`) is pinned. The frozen baseline is committed
     before any candidate is scored.

3. **The test set** (before the selection commit; no model of this work has run on a test frame).
   - `eval/manifests/textspot-test.jsonl`: 259 frames of the 28 test VNs. Every vita-test (74) and psp-test (83)
     screenshot is a whole frame; there are 19 text-free frames, and psp-test at 2× is report only (83).
   - Scored: 176 frames, 714 instances (dialogue 312, ui 229, name 122, choice 31, scene 20; 16 vertical) with 6,229
     characters (dialogue measure: 4,707), and 235 ignore areas.
   - Eval's lines: 450 of the 451 certain lines are kept with eval's text and class. The last (【全員】) is an undecided
     ignore area.
   - Review agreement: 97.05% of the boxes and 97.72% of the texts.
   - The set tool is re-pinned for its test-assembly fix. The dev manifest re-assembles byte for byte.

4. **Family C: the fastest and smallest** (owner, 15:32Z: "continue working on the fastest, smallest japanese text
   detector"). Registered before any of its runs and scores.
   - **The arms:** four more of #97's shapes at `h2`, seeds 1 and 2 (8 runs):
     - `s4_c8n5`: 2,260 B, est. 64.3 ms;
     - `s4_c12`: 2,308 B, 69.5 ms;
     - `s4p_c8`: 2,732 B, 71.9 ms;
     - `s4_c16`: 3,140 B, 84.1 ms.
   - **Recipe and scoring:** the recipe, data, trainer and sweep script are amendment 1's. Scoring goes through the
     pinned harness as `det:` / `detref:` arms.
   - **The rule is unchanged:** one selection over families A, B and C. The speed/size Pareto front is reported
     beside it.
   - **Disclosure:** the four shapes were chosen after family B's first 20 runs had been scored on the development
     frames. They fill the gap where the point estimates put the break: `h2_s4_c8` (1,572 B) reads dialogue at
     7.0-9.8% CER, while `h2_s4_c12n4` and `h2_s4_c16n4` are 2.8-3.9 KB. Family C's development intervals are
     therefore optimistic; the test set and CLANNAD are the check.
   - **Not added:**
     - fewer channels at `h2`: no time saved (`s4_c6` 47.1 ms, `s4_c8` 46.7 ms);
     - a coarser resolution: stride 8 at `h2` is already in family B and reads dialogue at 9-13%;
     - a quarter-scale input: it would need a new decode;
     - strip tiling: it changes no output (the runtime streams rows), so it is reported, not swept.

## Results

(none yet)
