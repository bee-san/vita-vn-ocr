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

Every number here is either a host measurement through the C runtimes or a host estimate of console time,
labelled as such:
- the host measurements cover the spotter, the finders, the refinement and the VOCR5 recognizers in W8I;
- nothing of this work has run on a console.

"N" means horizontal boxes read by VOCR5-N and vertical ones by VOCR5-V; S and L are VOCR5-S and VOCR5-L (report
only).

### Answer first

- **The rule has a winner: `B_h2_s4_c24n4_seed2` with det+refine.**
  - Its blob, `textspot_s4_c24n4_w8.vtsp`, is **5,732 B** (int8 weights, sha256 `85250691…`).
  - It reads a Vita frame at half size.
  - Arena: 80,799 B for any frame up to 960×544.
  - Estimated time on one Cortex-A9 core: **139 ms per 960×544 frame at 333 MHz**, 104 ms at 444 MHz. That is the
    A9 cycle model's high bound (131 ms for the spotter) plus the refinement of 6 lines (8 ms).
- **Development** (188 frames, 81 VNs): it beats all eight baselines as the rule requires: F1 and all-text CER
  resolved, dialogue CER not resolved worse.
  - Against #97's detector run on whole frames: F1 0.542 against 0.470, all-text CER 32.04% against 40.04%,
    dialogue CER 3.01% against 3.84% (not resolved).
  - Against GRAD + refine on the whole frame: F1 0.186 and 86.20% / 43.49% CER.
- **The fastest and smallest candidate that clears the bar** (the owner's ranking keys) is `C_h2_s4p_c8_seed1` with
  det+refine:
  - **2,732 B**, est. **80 ms** per Vita frame at 333 MHz (60 ms at 444), arena 72,391 B (the refinement's);
  - development F1 0.533, CER 34.92% / 3.64%.

  The registered rule replaces it by the winner: the winner's all-text CER is resolved 2.88 points lower (VN-cluster
  interval −5.52 to −0.38). Dialogue CER (−0.63, −1.92 to +0.60) and F1 (+0.009) are not resolved. The test sets
  were read for the winner only, so the smaller one's test numbers are unknown. Its blob is published beside the
  winner's for A/B tests, labelled so.
- **Family C changed the winner** (found by the audit, disclosed here). Without amendment 4's family C, which was
  chosen after family B's development scores, the same rule selects `B_h2_s4_c16n4_seed2` det+refine: 3,892 B, est.
  104.6 ms. Adding the smaller family C qualifier moved the replacement chain's start, and the chain then jumped
  past `s4_c16n4` to the larger, slower `s4_c24n4`. The two are not resolved apart on any development metric:
  `s4_c24n4` minus `s4_c16n4` is F1 +0.021 (−0.011 to +0.051), all-text CER −1.23 (−5.03 to +2.51) and dialogue
  +0.50 (−1.14 to +2.21). The protected sets were read for `s4_c24n4` only; reading them for `s4_c16n4` would be a
  second protected read, so the remedy is this disclosure, not a re-selection.
- **vita-test + psp-test as whole frames, read once** (176 frames, 28 VNs): F1 **0.694**, all-text CER **20.02%**,
  dialogue CER **3.00%**.
  - Best baselines: #97's detector 0.572-0.580 / 28.2-30.7% / 5.5-8.9%; GRAD + refine 0.266 / 79.03% / 43.70%.
  - Every comparison is resolved for the winner except the dialogue CER against `s8c24_spot` with VOCR5-S (+0.02)
    and VOCR5-L (+0.06), whose intervals reach above 0.
  - Recall: 96% of the dialogue lines, 93% of the names, 84% of the choices, 48% of the UI labels and 45% of the
    scene text.
- **CLANNAD, whole frames, read once** (2,052 dialogue lines on 1,867 pages): dialogue CER **1.14%** (S 1.44%, L
  0.94%), against #97's detector 1.29-1.44% (−0.15 to −0.30 points, resolved) and GRAD + refine on the whole frame
  16.64%.
  - With speaker names it is worse than #97's detector without refinement (+0.45 points, resolved): it finds fewer
    name plates.
  - For scale, though not a paired comparison: GRAD + refine on the hand-drawn dialogue region reads the same lines
    at 0.92% (#97's record, VOCR5-N).
- **What it does not do well:**
  - UI labels: about half are found on the test frames (a third on development), and their errors make up most
    of the all-text CER (20-32%).
  - Text-free frames: 6 of 19 test frames get a box (#97's detector: 3).
  - The dialogue pick: it lands on the dialogue block on 62% of the test frames with dialogue (68% on development;
    report only).

### Development: the selection (`eval/textspot_dev_20261019.json`)

- **The arms:**
  - family A: #97's 34 runs at two scales, two decodes each (136 arms);
  - family B: 22 new runs, two decodes each (44 arms);
  - family C: 8 runs, two decodes each (16 arms).
- **Eligibility:** 147 of the 196 arms are eligible. 49 are over 150 ms (the full-resolution arms of the bigger
  shapes); none is over the arena limit. The C runtime equals the twin on every development frame for every arm.
- **Qualifying** (beat all eight baselines), seven arms in the rule's size order:

| candidate | VTSP B | est. ms @333 / @444 | arena B | F1 | CER all | dialogue | text-free frames with a box |
|---|---:|---:|---:|---:|---:|---:|---:|
| `C_h2_s4p_c8_seed1` det+refine | 2,732 | 80.3 / 60.2 | 72,391 | 0.533 | 34.92 | 3.64 | 4 / 25 |
| `C_h2_s4_c16_seed2` det+refine | 3,140 | 92.5 / 69.4 | 72,391 | 0.513 | 35.26 | 4.19 | |
| `B_h2_s4_c16n4_seed2` det | 3,892 | 97.6 / 73.2 | 73,855 | 0.522 | 36.12 | 3.39 | |
| `B_h2_s4_c16n4_seed2` det+refine | 3,892 | 104.6 / 78.5 | 73,855 | 0.520 | 33.26 | 2.51 | 4 / 25 |
| `B_h2_s4_c24n4_seed2` det | 5,732 | 130.7 / 98.0 | 80,799 | 0.553 | 33.16 | 2.86 | |
| `B_h2_s4_c24n4_seed1` det+refine | 5,732 | 139.1 / 104.3 | 80,799 | 0.534 | 34.59 | 4.19 | |
| **`B_h2_s4_c24n4_seed2` det+refine (winner)** | **5,732** | **139.1 / 104.3** | **80,799** | **0.542** | **32.04** | **3.01** | 5 / 25 |

The rule's steps:
1. The smallest qualifying candidate is `C_h2_s4p_c8_seed1` det+refine.
2. The next qualifying candidates in size order, up to `B_h2_s4_c24n4_seed1` det+refine, are not resolved better on
   all-text CER.
3. `B_h2_s4_c24n4_seed2` det+refine is resolved better (−2.88, −5.52 to −0.38), so it replaces it.
4. No qualifying candidate is larger, so it is the winner.

Without family C (the candidate set before amendment 4) the same rule gives another winner:
1. The smallest qualifying candidate is `B_h2_s4_c16n4_seed2` det (36.12%).
2. Its det+refine (33.26%) replaces it: −2.85 (−5.05 to −1.09).
3. No larger qualifier is then resolved better (`B_h2_s4_c24n4_seed2` det+refine: −1.23, −5.03 to +2.51), so the
   winner would be **`B_h2_s4_c16n4_seed2` det+refine**: 3,892 B, est. 104.6 ms, arena 73,855 B, F1 0.520, CER 33.26%
   / 2.51%.

With family C the chain starts at `C_h2_s4p_c8_seed1` det+refine (34.92%), which `s4_c16n4` det+refine is not
resolved better than (−1.65, −5.73 to +2.42), so it passes to `s4_c24n4`. So the extension, chosen after family B's
development scores, made the selection 47% larger and 33% slower, the opposite of the owner's "fastest, smallest".
The two candidates are not resolved apart on any development metric (Answer first).

The winner against each baseline (development, N; VN-cluster 95% intervals; "beats" is all three conditions):

| minus | F1 Δ | CER all Δ, points | dialogue CER Δ, points | beats |
|---|---:|---:|---:|---|
| `grad` | +0.354 (+0.300 to +0.406) | −54.73 (−64.10 to −45.18) | −41.61 (−51.46 to −31.75) | yes |
| `grad_refine` | +0.355 (+0.302 to +0.405) | −54.17 (−63.69 to −44.66) | −40.48 (−50.58 to −30.41) | yes |
| `grad_refine_gate` | +0.343 (+0.289 to +0.393) | −53.62 (−63.11 to −44.16) | −40.48 (−50.58 to −30.41) | yes |
| `s8c24_textdet` | +0.071 (+0.027 to +0.111) | −11.60 (−16.20 to −7.19) | −1.96 (−5.31 to +0.66) | yes |
| `s8c24_textdet_refine` | +0.079 (+0.040 to +0.113) | −8.01 (−11.68 to −4.31) | −0.83 (−3.38 to +1.24) | yes |
| `s8c24_spot` | +0.076 (+0.029 to +0.119) | −13.47 (−18.81 to −8.32) | −1.96 (−5.31 to +0.66) | yes |
| `columns` | +0.534 (+0.474 to +0.591) | −131.77 (−147.95 to −116.37) | −94.48 (−96.22 to −92.28) | yes |
| `grad_refine_columns` | +0.410 (+0.356 to +0.460) | −88.71 (−100.73 to −76.94) | −40.28 (−50.33 to −30.33) | yes |

By part (development, N):

| part | frames | references | winner F1 / CER all / dialogue | `s8c24_textdet_refine` |
|---|---:|---:|---|---|
| vita | 39 | 232 | 0.623 / 18.76 / 1.50 | 0.459 / 28.66 / 1.86 |
| psp | 64 | 258 | 0.601 / 20.20 / 4.06 | 0.551 / 31.15 / 7.03 |
| adrenaline | 20 | 68 | 0.652 / 17.95 / 1.87 | 0.582 / 21.81 / 5.61 |
| pc | 40 | 327 | 0.385 / 58.16 / 3.71 | 0.361 / 62.69 / 1.05 |
| text-free | 25 | 0 | 5 frames with a box (0.40 per frame) | 5 (0.44) |

**Recall by class** (development, winner): dialogue 0.944, name 0.688, choice 0.867, ui 0.324, scene 0.500. #97's
detector (`s8c24_textdet_refine`): 0.940, 0.667, 0.867, 0.256, 0.500.

**Tightness:** mean IoU 0.802 and ink coverage 0.963 over the matched pairs (#97's detector with refine: mean IoU
0.801). On the PC frames, whose references are mostly small UI labels, both have their lowest F1 (0.385 and 0.361).

### The speed / size Pareto front (development)

This answers the owner's 15:32Z ask: the fastest and smallest detector that clears the bar. Of the 147 eligible
candidates, 41 are dominated by no other on VTSP bytes, est. ms, F1, all-text CER and dialogue CER. The main
points, in bytes:

| candidate | VTSP B | est. ms @333 / @444 | F1 | CER all | dialogue | qualifies |
|---|---:|---:|---:|---:|---:|---|
| `A_s4_c4_seed1_h600` det | 1,036 | 41.7 / 31.2 | 0.231 | 73.69 | 35.92 | no |
| `A_s4_c4_seed2_h600` det+refine | 1,036 | 55.7 / 41.7 | 0.305 | 46.57 | 11.69 | no |
| `B_h2_s4_c8_seed1` det | 1,572 | 46.7 / 35.0 | 0.435 | 43.60 | 8.28 | no |
| `B_h2_s4_c8_seed2` det+refine | 1,572 | 57.9 / 43.4 | 0.462 | 39.31 | 7.65 | no |
| `A_s8_c8_seed1_h600` det | 1,916 | 31.3 / 23.5 | 0.339 | 59.69 | 27.61 | no |
| `C_h2_s4_c8n5_seed2` det+refine | 2,260 | 72.7 / 54.6 | 0.475 | 38.55 | 4.04 | no |
| `C_h2_s4_c12_seed1` det+refine | 2,308 | 79.3 / 59.4 | 0.480 | 37.93 | 4.41 | no |
| **`C_h2_s4p_c8_seed1` det+refine** | **2,732** | **80.3 / 60.2** | **0.533** | **34.92** | **3.64** | **yes (smallest)** |
| `B_h2_s8_c16_seed2` det | 3,308 | 44.8 / 33.6 | 0.481 | 51.64 | 11.41 | no |
| `B_h2_s4_c16n4_seed2` det+refine | 3,892 | 104.6 / 78.5 | 0.520 | 33.26 | 2.51 | yes |
| `B_h2_s4_c24n4_seed2` det | 5,732 | 130.7 / 98.0 | 0.553 | 33.16 | 2.86 | yes |
| **`B_h2_s4_c24n4_seed2` det+refine** | **5,732** | **139.1 / 104.3** | **0.542** | **32.04** | **3.01** | **winner** |

What the front shows:
- **No candidate under 2.7 KB clears the bar.** At 2.3 KB (`s4_c8n5`, `s4_c12`) the best reach F1 0.48-0.50 and
  dialogue CER 4.0-4.7%. Each misses the resolved F1 or all-text CER margin over each of #97's three arms, and one
  `s4_c12` seed is resolved worse on dialogue. Under 2 KB, dialogue CER is 7-36% and F1 under 0.47.
- **Stride 8 at half scale** (16 frame pixels per cell) is the fastest way to a good F1 (0.46-0.52 at 45-82 ms),
  but it reads dialogue at 8.5-13% CER.
- **The cheapest shape that clears the bar is a plain 3×3 network at stride 4** (`s4p_c8`): 2.7 KB, 80 ms. Its
  second seed does not clear the bar (F1 0.468, dialogue 5.14%). Seeds of one shape differ by up to 6.7 points of
  dialogue CER (`C_h2_s4_c12` det: 5.19% and 11.91%).
- **Strip tiling** changes no output (the runtime streams rows). The arena is the frame's width alone: the winner's
  own 960×544 need is 67,583 B, and 80,799 B for any frame up to that size.

The full candidate table, every comparison and the per-part rows are in `eval/textspot_dev_20261019.json` (tables:
`eval/textspot_tables.py`).

### The protected sets, once (`eval/textspot_protected_20261019.json`, `eval/textspot_clannad_20261019.json`)

After the selection was committed and pushed (`085a741`, 2026-10-19T18:46:03Z; tag `textspot-selection-085a741`),
the winner was read once on each set. The attempt records are dated 18:46:45Z (test set) and 18:46:50Z (CLANNAD).
The C runtime equals the twin on every protected frame.

**vita-test + psp-test as whole frames** (176 scored frames of 28 VNs: vita 74, psp 83, text-free 19; 714
instances, 6,229 characters, dialogue 4,707):

| arm | P | R | F1 | CER all (N) | dialogue (N) | dialogue S / L | text-free frames with a box |
|---|---:|---:|---:|---:|---:|---:|---:|
| **winner** | **0.625** | **0.780** | **0.694** | **20.02** | **3.00** | 3.38 / 3.00 | 6 / 19 |
| `s8c24_spot` | 0.501 | 0.689 | 0.580 | 28.22 | 5.54 | 5.84 / 5.18 | 3 / 19 |
| `s8c24_textdet` | 0.502 | 0.665 | 0.572 | 30.68 | 8.88 | 9.24 / 8.60 | 3 / 19 |
| `s8c24_textdet_refine` | 0.498 | 0.668 | 0.571 | 29.03 | 8.54 | 9.07 / 8.24 | 3 / 19 |
| `grad_refine_gate` | 0.263 | 0.286 | 0.274 | 78.62 | 43.70 | 43.57 / 43.30 | 7 / 19 |
| `grad_refine` | 0.249 | 0.286 | 0.266 | 79.03 | 43.70 | 43.57 / 43.30 | 19 / 19 |
| `grad` | 0.250 | 0.280 | 0.264 | 80.78 | 45.04 | 45.34 / 44.81 | 19 / 19 |
| `grad_refine_columns` | 0.136 | 0.287 | 0.185 | 111.46 | 44.02 | 43.89 / 43.62 | 19 / 19 |
| `columns` | 0.002 | 0.004 | 0.003 | 158.85 | 97.17 | 97.17 / 97.17 | 18 / 19 |

Against the three #97 arms (VOCR5-N):
- F1 +0.114 to +0.124 (every interval above +0.06);
- all-text CER −8.2 to −10.7 points (intervals below −3.5);
- dialogue CER −2.55 (−5.51 to −0.25) against `s8c24_spot` and −5.5 to −5.9 against the other two (resolved).

With VOCR5-S and VOCR5-L the dialogue interval against `s8c24_spot` reaches +0.02 and +0.06. By part (N):
- vita-test: F1 0.692, CER 20.95% / 1.83%;
- psp-test: F1 0.705, CER 18.99% / 4.21%;
- report only, psp-test at 2× (Adrenaline): 0.633 / 20.85% / 6.98%, against #97's detector 0.587 / 26.19% /
  15.23%.

Recall: dialogue 0.955, name 0.926, choice 0.839, ui 0.485, scene 0.450. Tightness: mean IoU 0.821, ink coverage
0.969. Dialogue pick hit: 0.618.

**CLANNAD long run** (owner-drive-ingest's frozen references, `refs_sha256` `35411688…`; whole frames in):
- A reference line reads the detections that lie mostly inside its GRAD box: at least half of a detection's area in
  the box it overlaps most.
- The pages are all 2,177 of the run's; 3,048 reference lines; whole-page intervals.

| model | winner | `s8c24_textdet` | `s8c24_textdet_refine` | `s8c24_spot` | `grad_refine` (whole frame) | winner − `s8c24_textdet_refine` |
|---|---:|---:|---:|---:|---:|---|
| VOCR5-N | **1.14** | 1.41 | 1.44 | 1.29 | 16.64 | −0.30 (−0.54 to −0.10) |
| VOCR5-S | 1.44 | 1.55 | 1.83 | 1.43 | 16.91 | −0.39 (−0.65 to −0.20) |
| VOCR5-L | 0.94 | 1.14 | 1.30 | 1.03 | 16.35 | −0.36 (−0.60 to −0.17) |

- **Dialogue** (2,052 lines), N:
  - −0.26 points against `s8c24_textdet` (−0.52 to −0.06);
  - −0.15 against `s8c24_spot` (−0.30 to −0.00).
- **All lines with speaker names** (3,048), N: worse than `s8c24_textdet` (+0.45, +0.21 to +0.68) and `s8c24_spot`
  (+0.56), better than `s8c24_textdet_refine` (−0.62).
- **The 133 pages without reference text:** the winner returns 790 boxes there, #97's detector 877-952 and GRAD +
  refine 471. Whether those pages show text was not checked.
- **Training disclosures:**
  - The VOCR5 models and the spotter's real half include the owner's planetarian PSP frames (Key / Prototype, like
    CLANNAD's Vita port).
  - No CLANNAD-family frame is in any training pool.

### Console time and memory (host estimates)

| winner `s4_c24n4`, h2, det+refine | 960×544 frame | 480×272 frame |
|---|---:|---:|
| MACs (network) | 28.5 M | 28.5 M (a PSP frame is read as it is) |
| A9 cycle model, high bound × 1.0007, @333 / @444 MHz | 130.7 / 98.0 ms | 121.2 / 90.9 ms |
| + refinement (1.4 ms × 6 lines @333) | +8.4 ms | +8.4 ms |
| MAC-rate estimate @333 MHz (VOCR3-H15's console rate, report) | 90.9-94.6 ms | the same |
| arena (any frame up to 960×544; the refinement's 72,391 B fits in it) | 80,799 B | 66,623 B |

- **Code** (VitaSDK gcc 15.2, Thumb-2, the spotter's own sections, `--gc-sections`):
  - 11,440 B at -O2 with NEON (the measured code);
  - 9,466 B at -Os;
  - 8,058 B for -Os scalar;
  - psp-gcc mips32: 15,786 B at -O2, 12,890 B at -Os.

  Stack peak: 2,880 B on the Vita.
- **Weights:** 5,732 B. For the smaller candidate, 2,732 B, arena 37,767 B (the refinement's 72,391 B with
  det+refine), est. 71.9 + 8.4 ms.
- **In Rubi's game worker** (its draft PR, measured there): the worker module grows from 98,304 to 110,592 B of
  pages, and a scan takes two transient blocks within the job's existing items.
- **Every time above is an estimate:** nothing of the spotter has run on a console.

### Measured and estimated

**Measured on the host:**
- every P / R / F1, CER, interval, recall and text-free count above (the C runtimes, the frozen metrics);
- blob, arena and code sizes and stack frames (exact; compiler output).

**Estimated:**
- every console time: the A9 cycle model on the Vita compiler's code under qemu, the binding method, and the MAC
  rate (report);
- the console memory beyond what Rubi's build measures.

### Limits

- **The development sample** is 81 VNs, and the candidate set was extended once after family B's development
  scores (amendment 4). That extension changed the winner: without it the rule selects `s4_c16n4` (3,892 B, est.
  105 ms), not resolved apart from `s4_c24n4` on development ("The rule's steps"). Family C's development intervals
  are optimistic, and so is the rule's choice among seeds: seeds of one shape differ by up to 6.7 points of dialogue
  CER. The test set and CLANNAD, read once, agree with the development verdict for the winner.
- **UI text is the weak class:** about half is found on the test frames, a third on the development frames. Its
  readings (VOCR5 was trained on dialogue) dominate the all-text CER.
- **Labels:** the training labels of the real half are MeikiOCR's boxes refined to ink, not hand-checked. The
  synthetic frames hold a little unlabelled scene text. In the hand-checked sets, copyright lines are not labelled
  one way: 8 of them, in 5 development frames, are class `ui`, while others are `watermark` ignore areas. One name
  keeps a decorative `◆` (`v7321_57909`).
- **The protocol's decode text** (above, frozen) leaves out two drops that the twin makes: components of fewer
  cells than the blob's `min_cells`, and boxes under 4 f px across (f = 2 for a halved frame). The twin is the
  specification and the C runtime matches it.
- **Vertical text:** the spotter flags vertical boxes (42 on the test frames, 74 on development), but the scored sets
  hold few vertical instances (16 and 41). #103's column finder remains the measured vertical path.
- **Not compared:** the hand-drawn-region pipeline (GRAD + refine on a region the player draws) is not an arm here,
  since it needs a human. On CLANNAD it reads dialogue at 0.92% (VOCR5-N, #97's record). The whole-frame spotter's
  1.14% is not a paired comparison with it.
- **Nothing has run on a console.**

### Reproduce

```sh
H=$HERD; T=$H/shared/artifacts/textspot; P=$H/shared/private/textspot
.venv/bin/python synth/textspot_data.py synth --out $T/shards/synth_frames ...      # pinned by data/manifests/textspot_shards_v1.json
eval/reproducibility/textspot-20261019/sweep.sh $T/runs                              # family B (22 runs); $T/runs_c with the 8 names: family C
.venv/bin/python eval/textspot.py run --set dev --arms baselines --cbin auto --out $T/baseline/units
.venv/bin/python eval/textspot.py baseline --units $T/baseline/units --out eval/textspot_baseline_20261019.json
.venv/bin/python eval/textspot.py run --set dev --arms famA,famB,det:C_<run>=...,detref:C_<run>=... --cbin $T/eval/cbuild --out $T/eval/famX
.venv/bin/python eval/textspot.py select --baseline eval/textspot_baseline_20261019.json --units $T/eval/famA \
    --units $T/eval/famB --units $T/eval/famC --a9 eval/textspot_a9_20261019.json --cbin $T/eval/cbuild \
    --out eval/textspot_dev_20261019.json                                           # the rule; then compact it (2.8 MB indented, over the repo's 2 MiB):
python3 -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); open(p,"w").write(json.dumps(d,separators=(",",":"),ensure_ascii=False)+"\n")' eval/textspot_dev_20261019.json
.venv/bin/python eval/textspot_tables.py --baseline eval/textspot_baseline_20261019.json --dev eval/textspot_dev_20261019.json \
    --units $T/eval/famA --units $T/eval/famB --units $T/eval/famC --out tables.md
.venv/bin/python eval/textspot_protected.py tests|clannad --cand B:h2_s4_c24n4_seed2 --decode det+refine ...   # once (attempt records refuse a second run)
```
