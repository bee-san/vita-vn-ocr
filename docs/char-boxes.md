# Per-character screen boxes from the CTC frames (issue #114)

The owner (2026-10-23): "if we found a way to get true bounding boxes for vn text, could we make a overlay like gsm
over the text itself? what data would we need? does our existing text detector (line spot) work like this? what
would we need to do to make it work?"

A GSM-style overlay draws on the game's own text: the word under the cursor is highlighted where it is on the
screen, a tap on a character looks it up, and the popup sits beside the word. That needs a box for every
**character**. textspot (#104) and textspot3 (#108) return **line** boxes, the GRAD + refine finder knows where each
line crop is in the drawn region, and the native hooks give text without positions. Nothing gives characters.

This document is the pre-registered protocol (`eval/charboxes_protocol.json`, committed and tagged before any
development score), then the results.

## The method: positions from the recognizer's own frames

The recognizer already knows where each character is. `vocr_run()` resizes a line crop of w × h px to height 32
and nw = `vocr_resized_width(w, h)` columns (OpenCV INTER_LINEAR), pads it to a multiple of 8 and emits one CTC
frame per 8 resized columns. Greedy decoding emits a character at the first frame of each run of its class (after a
blank or another class), so every character has a run of frames [f0, f1]. Frame f covers resized columns
[8f, 8f + 8), and INTER_LINEAR maps a resized coordinate u to the crop coordinate u · w / nw (continuous
coordinates: pixel i spans [i, i + 1)). The crop's place on the screen is known (the finder's or the spotter's box),
so each character gets a screen position with no new model and no new network computation.

`eval/charbox.py` (the twin, integer arithmetic in 1/16 px; `runtime/vocr_charbox.c` will compute the same numbers)
gives every greedy character:

- **centre**: the centre of the first frame of its run, 8 f0 + 4 resized px, plus the model's calibration, mapped to
  the crop and clipped to it;
- **cell**: the line cut between neighbouring centres (midpoints); the first and last cells reach half a pitch (the
  median spacing of the centres) beyond their centres. The cells tile the line, so a point on the line picks one
  character (which cells to hit-test a touch with: "Hit-test", below);
- **ink box**: the cell, or with INK the columns of the cell that hold ink.

Refinements (flags):

- **calibration** (`cal`): where a model's CTC fires is a property of the model. Per model, a shift for every
  character and an offset for each of seven groups of characters that sit at one side of their cell or are narrow
  (`close` 、。」』）…, `open` 「『（…, `dots` …‥・, `small` small kana, `long` ー―～, `mark` ！？♪…, `half`
  half-width), in resized px. They are medians of MeikiOCR's box centre minus the frame centre, on **training-side**
  lines the model reads exactly as MeikiOCR does (`eval/charbox_eval.py calibrate`, `eval/charboxes_cal.json`).
  `mark` got **no** calibration: teacher_train's MeikiOCR characters write ASCII `!` `?` (and `~` `(` `)`) where
  VOCR5 writes ！ ？ (～ （ ）), and the exact-read filter compares the raw strings, so every line holding one was left
  out of the calibration and of train-check (286 / 289 / 305 of the 3,000 calibration lines for S / N / L for ！？
  alone): `mark` has 0 training characters and offset 0 for every model (found by the independent audit, F2);
- **INK**: the cuts between characters move to the emptiest run of columns between their centres (an ink-column
  gap; the ink profile is each crop column's vertical gradient energy, Σ |g(y + 1) − g(y)|, which needs no polarity),
  each ink box shrinks to its cell's columns above a quarter of the line's peak, and the centre moves to the middle
  of that ink;
- **PITCH**: on a line of at least 4 characters whose spacings are all within 25% of their median (a fixed font
  pitch), the centres move to the least-squares grid a + p k before the cells are cut.

Vertical text: VOCR5-V reads a column turned 90° anticlockwise (`vertical_twin.rotate_ccw`: the column's top is the
strip's left), so the same positions along the strip are positions down the column.

### What was decided before this protocol, and on what

The evidence of record is on **training-side** data only: teacher_train's crops with MeikiOCR's character boxes,
lines VOCR5 reads exactly as MeikiOCR, with no line of a dev_sample or test VN (`excluded_vns()`; CLANNAD and its
family are test-reserved) and no screenshot shared with a development unit. An informal exploration before it
(scripts outside the repo; issue #114's first numbers) chose the arms. **It sampled teacher_train without the
reservation filter**: 45 of its 7,261 lines were on 8 test-reserved VNs, the exposure `docs/reservations.md` lists
for teacher_train (v1267 12 lines, Little Busters! 9, Kanon 8, v11706 5, CLANNAD Side Stories 5, AIR 3, CLANNAD 2,
Tomoyo After 1; v1267 and v11706 share a release with psp-test games), and VOCR5-N read 37 of them as MeikiOCR did, so
they entered its statistics. None is a vita-test or psp-test screenshot. The calibration (`calibrate`, the first
3,000 lines of a seeded order) and then `train-check` (the next 1,500 lines, `eval/charboxes_protocol.json`
"method_of_record") exclude them, and give the same choice (cal + INK 99.3-99.5%; cal or INK alone at most 99.2%;
PITCH adds nothing). Explorations from now on sample `pools/teacher_train_train.npz` or apply `excluded_vns()` (found by the
independent audit, F1). Centre inside MeikiOCR's box (lines with ！？～（） left out, as above):

| arm | VOCR5-S Vita / PSP | VOCR5-N Vita / PSP | VOCR5-L Vita / PSP | N: 、。」 etc. (close) | N: cell IoU |
|---|---:|---:|---:|---:|---:|
| raw frame centres | 87.7 / 90.4% | 97.0 / 97.4% | 96.8 / 97.3% | 59.7% | 0.81 |
| cal | 98.7 / 98.8% | 98.7 / 98.8% | 98.6 / 99.0% | 82.9% | 0.81 |
| INK | 96.3 / 97.5% | 98.9 / 99.2% | 98.9 / 99.2% | 86.4% | 0.79 |
| **cal + INK** | **99.5 / 99.3%** | **99.5 / 99.5%** | **99.4 / 99.5%** | **93.9%** | 0.79 |
| cal + PITCH + INK | 99.5 / 99.3% | 99.5 / 99.5% | 99.4 / 99.5% | 93.8% | 0.79 |

(1,202 / 1,252 / 1,280 lines of 11,961-13,045 characters.) VOCR5-S fires about 0.21 crop heights late, which the
calibration's shift removes; PITCH adds nothing on these lines. So the **method of record is cal + INK**, chosen
before any development line was read; every other arm is reported beside it. The calibration
(`eval/charboxes_cal.json`, resized px × 16, [shift, other, close, open, dots, small, long, mark, half]): S
[-110, 0, 32, 72, 39, -11, -1, 0, 18], N [0, 0, -45, -32, 48, -23, -4, 0, -6], L [-36, 0, -12, 4, 38, -16, -10, 0, 8],
V [-104, 0, ...] (VOCR5-V: 619 characters of 173 training columns, too few for any group).

## Development sets and reference boxes

Only development data: every horizontal unit is on a `reserved.json` dev_sample VN; no unit is on a test VN, and no
frame of vita-test, psp-test or CLANNAD is read (`eval/charbox_sets.py units` checks it).

| source | what | parts |
|---|---|---|
| textspot-dev | every labelled, non-ignored reference of `eval/manifests/textspot-dev.jsonl`'s 188 frames (dialogue, names, choices, UI, scene text; horizontal and vertical); a region per frame around its horizontal dialogue (+12 px, PSP +6) | vita, psp, adrenaline (PSP frames at 2x), pc (scaled to fit 960×544) |
| devnative | devnative's hand-checked lines of dev_sample VNs (`lines_full.jsonl`), the line's dialogue region; every PSP line again on its frame at 2x (cv2.INTER_LINEAR, Adrenaline's picture) | vita, psp, adrenaline |
| vertical-dev | the vertical study's development columns (#99's own split, never read by VOCR5-V's training; VN pages and the manga cohort) with their regions | vita (manga), psp, pc |

**Reference boxes.** MeikiOCR (`teacher/engines.Meiki`, the teacher's pinned files) reads each unit's view on its
frame: the ink box plus the console pad across the line and half a line height along it; a column with MeikiOCR's
vertical recognizer. A reference character's box is the MeikiOCR character it is aligned to (Levenshtein on
`metrics.norm` of each character) when the two are equal, on a line MeikiOCR reads within 20% CER of the reference
(60% for a column: MeikiOCR's vertical recognizer reads PSP and scaled PC columns poorly, #99). Horizontal dialogue:
97-99% of the reference characters get a box. The column threshold was raised from 20% to 60% after MeikiOCR had
read the development columns and before any score; the stored `meiki_dev.jsonl` keeps `ref_boxes` made at 20% for
296 vertical rows, which nothing reads (the scorer makes the boxes again from MeikiOCR's characters).

**Hand-checked set.** 96 dialogue lines (48 Vita native, 48 PSP native, seed 20261023; `eval/charbox_hand.py`) with
MeikiOCR's boxes drawn on contact sheets, checked by eye (Claude, reviewer_kind assistant) before any score: every
box of all 96 lines sits on its glyph. Its score is reported as the subset whose every box was checked. Seven of
the 96 are the same line twice (next paragraph): 89 distinct lines.

**Overlap between the sources** (found by the independent audit, F4). textspot-dev and devnative take lines from the
same 82 screenshots: 169 lines (127 of them dialogue) are in both with the same frame, scale and box, and the
counts below count them twice. VOCR5-N dialogue characters on fixed lines: Vita 2,612 (1,733-1,771 distinct,
depending on which copy is kept), PSP 5,175 (4,302-4,355), Adrenaline 4,284 (4,065-4,073). Deduplicated, every bar
rate moves by at most 0.24 points: fixed 99.0 / 99.3 / 99.4%, finder 99.0 / 97.95-98.0 / 99.2% (PSP's VN interval
95.9-99.2%), all at least 97.0%. On the spotters' lines the two copies of a shared line compete for one detection,
so one of them is always recorded as missed (112 and 125 copies for textspot and textspot3).

**Synthetic exact boxes** (helper overlay-boxes-synth): the synth4 renderer's development split (held-out text and
scenes) at Vita, PSP and psp2x (Adrenaline), each glyph's exact ink box and advance cell through every geometric step
of the render. Report only: the method against exact boxes, and MeikiOCR's own box accuracy against them.

## Line sources, models, arms

- **fixed**: the unit's ink box plus the console pad (4 px; PSP 2), clipped; dark text read with `VOCR_INVERT`
  (`common.line_dark`), as the console reads it.
- **finder**: GRAD + refine (`common.find_lines_ex(grad)`, `lines_refine.refine_lines(DEFAULT)`) on the unit's
  region, Rubi's default path; columns: the column finder (`vertical_twin.find_columns(..., 1)`). A finder box is
  matched one-to-one to the region's references (eval's rule: across-line overlap at least half the smaller extent,
  any overlap along the line, largest first). An unmatched reference has no read.
- **ts1 / ts3**: the released spotters (v0.3.3 `textspot_s4_c24n4_w8.vtsp`, v0.3.4 `textspot3_t64n8c16_kd_w8.vtsp`)
  on the whole textspot-dev frame, det+refine (horizontal boxes through the refinement, as Rubi's scan screen),
  matched to the frame's references the same way.
- Models: the released VOCR5-S/N/L (v0.3.0) for horizontal lines and VOCR5-V (v0.3.2-vertical.1) for columns,
  through the C runtime in W8I. Arms: raw, cal, ink, cal+ink (record), cal+pitch+ink.

## Metrics

Per reference character with a reference box, aligned (Levenshtein on `metrics.norm` keys) to a character of the
read:

- **cursor-hit**: the estimated centre lies inside the reference box along the reading direction, and across it
  within the reference line's ink box plus the pad. The rate is over aligned pairs (the read's character may be a
  misreading: its position is what is scored); `hit_all` also counts reference characters the read has no character
  for (a missed line, a deleted character) as misses. As scored, a line that a finder or a spotter does not return is
  recorded apart (tier `-`) and the reported `hit_all` leaves it out: it counts the deleted characters of the lines
  returned (found by the independent audit, F3; the missed lines are given with the results).
- The metric measures where the highlight goes (the character's centre). Whether a point on the glyph falls in that
  character's **cell**, a touch's hit-test, was **not** part of the protocol ("Hit-test", below, report only).
- **IoU** along the reading direction of the estimated cell and of the ink box with the reference box (and 2-D on
  the synthetic set).
- Per platform (Vita native, PSP native, Adrenaline 2x; PC scaled as a column), class (dialogue, name, choice, UI),
  line source and model. 95% intervals: VN-cluster bootstrap of the hit ratio, 10,000 resamples, seed 0.

## Pre-registered bar

Phase 1 passes when, for **VOCR5-N** with **cal + INK** on horizontal **dialogue** development characters with
MeikiOCR reference boxes, the cursor-hit rate is **at least 97.0% on each of Vita native, PSP native and Adrenaline
2x**, for **fixed lines and for finder lines** (six numbers, every one ≥ 97.0%).

If it passes: the C module (`runtime/vocr_charbox.c`), twin parity, tests, this document's results, then Rubi's
overlay mode. If it does not: the other arms, then a report of the model change that would be needed (e.g. a small
alignment head), and no runtime change.

Everything else is reported, not gated: VOCR5-S/L, VOCR5-V on columns, the spotters' lines, names, choices, UI, PC,
`hit_all`, IoU, the hand-checked subset and the synthetic set. Protected sets (vita-test and psp-test, fixed and
finder lines, VOCR5-N, cal + INK, reference boxes made the same way) are read once at the end as a check, after the
development results are committed.

## Results (development; measured)

Scored 2026-10-23 06:48:20-06:50:31Z at the protocol's tag, `eval/charbox_eval.py score` and `report --boot`
(`eval/charboxes_dev_20261023.json`: every table below, counts and rates only). Host, the C runtime in W8I.

**The bar is met.** VOCR5-N, cal + INK, horizontal dialogue, cursor-hit with VN-cluster 95% intervals:

| lines | Vita native | PSP native | Adrenaline 2x | PC scaled (report) |
|---|---:|---:|---:|---:|
| fixed | **98.9%** (98.1-99.6) | **99.3%** (98.8-99.7) | **99.4%** (99.1-99.7) | 98.5% (97.6-99.4) |
| finder (GRAD + refine) | **98.9%** (98.0-99.6) | **98.2%** (96.4-99.2) | **99.2%** (98.6-99.6) | 97.8% (96.8-98.7) |
| characters (pairs), VNs | 2,612 / 2,556, 10 | 5,175 / 4,808, 30 | 4,284 / 4,185, 29 | 946 / 891, 33 |

Every one of the six cells is at least 97.0% (the smallest, PSP finder lines, 98.2%). Vita native rests on 10 VNs
(its interval is the widest). `hit_all` (a reference character the read has no character for, within the lines a
source returns, counts as a miss): fixed 98.9 / 99.0 / 99.1%, finder 98.1 / 97.5 / 97.8%. The finder returns no line
for 35 / 349 / 52 reference characters (PC 14; fixed lines miss none); with those counted as misses too, the finder's
`hit_all` is 96.8 / 91.0 / 96.6% (PC 92.1%).

Every model and line source (cal + INK; Vita / PSP / Adrenaline / PC):

| lines | VOCR5-S | VOCR5-N | VOCR5-L |
|---|---|---|---|
| fixed | 98.9 / 99.3 / 99.4 / 98.8% | 98.9 / 99.3 / 99.4 / 98.5% | 99.1 / 99.3 / 99.5 / 98.4% |
| finder | 98.9 / 98.3 / 99.0 / 98.5% | 98.9 / 98.2 / 99.2 / 97.8% | 99.2 / 98.6 / 99.1 / 98.3% |
| textspot (`ts1`) | 98.6 / 98.3 / 99.0 / 97.9% | 98.8 / 98.0 / 99.0 / 98.3% | 98.9 / 98.0 / 99.0 / 98.0% |
| textspot3 (`ts3`) | 98.6 / 99.0 / 99.5 / 98.0% | 98.9 / 99.2 / 99.3 / 98.0% | 99.2 / 99.0 / 99.5 / 98.2% |

The spotters' lines cover textspot-dev's frames only (1,134-1,345 Vita and PSP characters, 394-406 Adrenaline);
within the lines they return, textspot's reads lack more reference characters than textspot3's (`hit_all` on PSP
82.4% against 97.9%), which the position does not see: a line that is found is placed as well as a fixed one. Their
whole-line misses are not comparable as recorded (the shared lines, above).

What each refinement buys (VOCR5-N, fixed lines; finder lines alike):

| arm | Vita | PSP | Adrenaline | PC |
|---|---:|---:|---:|---:|
| raw frame centres | 95.6% | 96.2% | 96.8% | 94.3% |
| cal | 98.3% | 98.7% | 98.9% | 96.8% |
| INK | 97.5% | 98.6% | 98.7% | 97.5% |
| **cal + INK** | **98.9%** | **99.3%** | **99.4%** | **98.5%** |
| cal + PITCH + INK | 98.9% | 99.3% | 99.4% | 98.5% |

VOCR5-S's raw centres are at 88.4-90.4% (it fires late); its calibration brings it to VOCR5-N's level. PITCH changes
nothing, as on the training lines: the frame quantisation it would average out is already within a glyph.

**Where the misses are** (VOCR5-N, cal + INK, fixed lines, Vita / PSP / Adrenaline): kana and kanji (group `other`)
99.6 / 99.95 / 100%, small kana 100%; `close` (、。」』）…) 96.1 / 94.6 / 94.3%; `mark` (！？…) 76.5 / 89.4 / 96.7%
(34, 104 and 90 characters); `dots` 94.2 / 97.8 / 96.2%. The narrow punctuation at one side of its cell is what is
left; a word lookup starts on kana or kanji. Part of the `mark` misses is its missing calibration: with ！？ compared
as full-width on the training side, the offsets would be S +34, N −42, L +1 (the other groups move by at most 8), and
VOCR5-N's deduplicated development `mark` 85.0 / 93.3 / 100% instead of 75.0 / 89.9 / 96.3% (20, 89 and 82
characters; the independent audit's what-if, report only; adopting it would be a registered amendment).

**Hand-checked subset** (the 96 lines whose every MeikiOCR box was checked by eye), VOCR5-N, cal + INK: fixed
98.8% (97.5-99.8) Vita, 99.2% (98.3-99.9) PSP; finder 98.9% (97.9-99.8), 98.7% (97.4-99.6) (721 / 663 and
710 / 595 characters). The same as the whole set, within its intervals.

**Other classes** (VOCR5-N, fixed, cal + INK): names 99.4-100% (797 characters), choices 98.7-100% (118), UI 95.6%
(Vita), 94.7% (PSP), 98.2% (Adrenaline), 86.3% (PC: small labels scaled down).

**Vertical text** (VOCR5-V, cal + INK, columns; development columns of the vertical study): fixed columns 97.6% on
the manga cohort (Vita views, 1,188 characters of one book: no interval), 98.5% (96.3-99.6) on PSP VN columns, 97.1%
(95.5-98.6) on scaled PC pages; the column finder's columns 97.6%, 98.9% (97.1-100), 95.7% (91.7-97.9). Raw frame
centres: 90.0, 83.5, 83.9%: VOCR5-V fires about 0.8 frames late, which its calibration removes.

**Box IoU** along the line against MeikiOCR's boxes (VOCR5-N, cal + INK): cells 0.74-0.76 on fixed Vita, PSP and
Adrenaline lines, 0.73-0.76 over every line source and PC (raw cells 0.76-0.80); ink boxes 0.70-0.74 (PC 0.69-0.70).
INK moves the cuts into the gaps between glyphs, where MeikiOCR's boxes of kana and kanji often reach; the centres
move onto the glyphs. The synthetic set's exact boxes measure the ink boxes without MeikiOCR's convention (next
section).

**Hit-test** (report only; not part of the protocol; the independent audit's measurement, F5). A tap at the middle of
MeikiOCR's box of a character, VOCR5-N, fixed dialogue lines, deduplicated, 11,145 taps over Vita / PSP / Adrenaline
/ PC; a tap succeeds when it falls in the cell of the character aligned to that reference character:

| cells of | all taps | ellipses (……) | kana and kanji | wrong cell or none |
|---|---:|---:|---:|---:|
| raw frame centres | 99.58-99.88% | | | |
| cal | 99.58-99.89% | 96.8-100% | 100% | 16 (0.14%): 15 line-end punctuation past the last cell |
| cal + INK | 98.84-99.47% | 83.6-91.0% | 99.73-100% | 84 (0.75%): 23 the other … of a ……, the same 15, 46 another character |

INK cuts at the emptiest columns between two centres; inside a …… those are the gaps between its own dots, so a
cell can hold half the glyph (`dn:v16770_154901_1`: MeikiOCR's box of the first … 401-425, its cal + INK cell
403.5-412.5, its cal cell 401.5-424.6). Of the 46 cal + INK taps that pick another character, 34 pick a kana or
kanji, and 19 are taps on a kana or kanji (0.21% of 9,116); with cal's cells, none. So the record arm places the
centres best, and a find without INK (cal's cells) hit-tests best ("What Rubi needs").

## Synthetic exact boxes (report only; measured)

The helper's renders (overlay-boxes-synth; `eval/charbox_synth.py`, 3,000 development-split regions per platform from
synth4 as VOCR5 was rendered, every glyph's exact ink box and advance cell; the pickles stay local, their sha256 are
in `eval/charboxes_synth_20261023.json`), read by `eval/charbox_eval.py synth`: every clean line (a gt crop, every
glyph's ink inside it: 6,778 / 6,887 / 6,834 lines), each model in W8I, every arm; per glyph of an equal aligned
character, the centre inside the exact ink along the line and inside the exact cell, the 1-D IoU of the method's
cell and ink box with the exact ones, and the centre's offset from the ink centre (in line ink heights; negative is
left). Nothing was chosen on it: the method of record was fixed before these renders were read.

Dialogue lines (dialogue, narration, name + dialogue), cal + INK against raw frame centres:

| VOCR5 | platform | glyphs | inside exact ink | inside exact cell | cell IoU | ink IoU | offset (ink h) | raw: ink / cell / offset |
|---|---|---:|---:|---:|---:|---:|---:|---|
| N | Vita | 51,372 | 98.20% | 99.56% | 0.827 | 0.758 | −0.018 | 97.09% / 99.89% / −0.049 |
| N | PSP | 44,162 | 98.37% | 99.55% | 0.816 | 0.781 | −0.017 | 97.40% / 99.84% / −0.048 |
| N | psp2x | 45,345 | 98.21% | 99.57% | 0.820 | 0.785 | −0.016 | 97.42% / 99.83% / −0.051 |
| S | Vita / PSP / psp2x | | 98.33 / 98.44 / 98.12% | 99.36 / 99.38 / 99.35% | 0.82-0.83 | 0.76-0.79 | −0.014 to −0.016 | 88.9-90.2% / 97.0-97.8% / +0.21 |
| L | Vita / PSP / psp2x | | 98.04 / 98.30 / 98.10% | 99.58 / 99.60 / 99.56% | 0.82-0.83 | 0.76-0.78 | −0.018 to −0.020 | 96.9-97.4% / 99.86-99.92% / +0.03 |

- Every tier lands 98.0-98.4% of its centres on the exact glyph ink and 99.35-99.60% in the exact cell. Names, choices
  and UI lines are higher (VOCR5-N 98.8-99.5% ink).
- MeikiOCR's boxes, the development reference, against the same exact boxes (the helper's measurement, on the lines
  MeikiOCR reads exactly): centre inside the exact cell 99.99% on all three platforms, inside the glyph's ink along
  the line 99.0-99.3%, inside the 2-D ink box 94.6-94.7% (78-83% of those misses are 、 or 。: its boxes span the
  crop's full height, so the cursor-hit test's across-line check uses the line's ink box instead).
- The calibration is fitted to MeikiOCR's boxes, so it inherits their left bias: the helper measured MeikiOCR's
  centres 0.05-0.06 ink heights left of the exact ink centre, and the calibrated centres sit 0.014-0.020 left of
  it. For VOCR5-N and L the raw frame centres are already in the exact cell more often (99.8-99.9% against
  99.6%) and cal + INK gains 0.8-1.1 points on the ink; for VOCR5-S, which fires about 0.2 ink heights right,
  the calibration is what puts its centres on the glyphs (88.9-90.2% → 98.1-98.4% ink).
- The misses are narrow punctuation, as on development: VOCR5-N cal + INK, ink / cell, Vita / PSP / psp2x: an
  opening bracket 81.9 / 84.7 / 85.4% / 98.7-99.2%, closing marks (、。」) 91.5-93.8% / 97.3-98.4%, ！？ 94.4-96.5% /
  99.4-99.8%, dots 94.6-95.7% / 97.0-98.2%; small kana 98.2-99.3% / 99.7-99.9%; the other characters (kana, kanji)
  99.4-99.6% / 99.8-99.9%.
- PITCH changes nothing here either (within 0.08 points of cal + INK everywhere).

## Protected check (read once; measured)

vita-test and psp-test, read once on 2026-10-23 07:07:24-07:07:54Z after the development results were committed
(`eval/charbox_protected.py`, which refuses a second run; `eval/charboxes_protected_20261023.json`, counts and rates
only). VOCR5-N, cal + INK, eval's hand-checked lines (eval's box and region), MeikiOCR's reference boxes made as on
development (2,498 of vita-test's 2,564 reference characters get one, 2,262 of psp-test's 2,510):

| set, dialogue | fixed lines | finder lines (GRAD + refine on eval's region) |
|---|---:|---:|
| vita-test (16 VNs) | 98.9% (98.2-99.4), 2,349 characters | 98.8% (98.1-99.4), 2,280 |
| psp-test (14 VNs) | 99.5% (99.0-99.9), 2,135 | 98.5% (97.0-99.5), 2,042 |

Names: 100% / 96.6% (vita-test fixed / finder), 98.4% / 91.3% (psp-test). The protected sets agree with
development; nothing was chosen on them. The JSON's `protocol_pins` are the tag's; the code that ran had amendment 1's
pins (`eval/charbox.py` `f05fc14a…`, `eval/charbox_eval.py` `9e013d43…`; `"amendments": 1`), which changed no scored
record.

## The C module

`runtime/vocr_charbox.c` / `.h` (runtime/README.md "Per-character positions" has the API, sizes and checks):
`vocr_charbox_find()` gives every greedy character its class, frames, group, centre, cell and ink box from
`vocr_extra.best` (or the top-K's top-1) and the crop's column profile (`vocr_charbox_profile()`), in 1/16 crop
px; `vocr_charbox_align()` maps another text's characters (the beam + LM text, the normalised text, a hook's line)
onto them. It is a new file: `vocr.c` and every other runtime file are unchanged, so a caller that does not call it
gets exactly what it got before. No allocation, bounded by `VOCR_CHARBOX_MAX` (256) characters a line, integer
arithmetic; 3,186 B of Thumb-2 at `-Os` (5,032 B at `-O2`); est. 1.6 ms per 900 × 36 line with INK at 333 MHz
(instruction count under qemu-arm, one instruction a cycle; `vita/a9_cycles.py`'s Cortex-A9 model, which prices
`vocr.c`, was not run on it; recognizing such a line takes roughly 0.3-0.5 s).
Parity with the twin: random lines and text pairs on gcc and clang (ASan + UBSan), real development lines of every
model, and Cortex-A9 builds under qemu-arm, every number equal.

## What Rubi needs

- In the game worker, per recognized line: `vocr_extra.best` (or the top-K it already keeps for the beam + LM) and
  the model's calibration (18 B, `eval/charboxes_cal.json`), then `vocr_charbox_find()`; the boxes plus the line's
  crop rectangle give screen boxes. A frame shown at 2x (a PSP game in Adrenaline) is just another crop size: the
  boxes are in the frame's pixels.
- Which boxes for what: a touch's hit-test and a highlight that spans a word take the **cells of a find without
  INK** (cal only: 0.14% of taps pick a wrong cell or none, against 0.75% with INK's cells; "Hit-test" above), which
  needs no crop and costs about 10-27 K instructions a line. INK's centres and ink boxes (with the crop's profile,
  4 B a column, about 0.5 M instructions a line) are the record arm's, for marking the glyph itself.
- The beam + LM text (Rubi's default) and the normalised text: `vocr_charbox_align()` from the greedy characters;
  an inserted character takes its neighbours' cells. A native hook's line: the same alignment against the OCR'd
  characters of the job, when OCR ran.
- What the Shell needs per character is small: the line, and the cell's and the ink's start and end along it (the
  line's rows are shared).

## Limits

- The reference is MeikiOCR's character boxes: checked by eye on 96 lines, not on all; its boxes of narrow
  punctuation are tight while its kana and kanji boxes reach into the gaps, which is why the cells' IoU is a
  moderate 0.73-0.76 and the record arm's misses are punctuation.
- Vita native development data is 10 VNs; the protected check (16 VNs) agrees.
- The `mark` group (！？♪…) has no calibration (the method, above): its centres are the frame centres plus the model's shift,
  and it is the worst group on development (VOCR5-N 76.5 / 89.4 / 96.7%).
- INK's cells hit-test worse than cal's (the Hit-test table): inside a …… a cell can hold one dot.
- Adrenaline 2x is modelled as an `INTER_LINEAR` 2x copy of the PSP frame only; Adrenaline can also show the PSP
  picture nearest-neighbour (`docs/finder-tighten.md`), which these sets do not hold (the synthetic psp2x set does:
  it renders bilinear, nearest and 1.995x upscales).
- Vertical columns are calibrated on 619 characters of 173 training columns (no group offsets); MeikiOCR's vertical
  recognizer misreads many columns, so the vertical reference covers 27-81% of the characters (scaled PC pages the least).
- Host measurements only; nothing of it has run on a console. The time is an estimate.
