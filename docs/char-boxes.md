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
  median spacing of the centres) beyond their centres. The cells tile the line: a cursor or a touch picks the cell
  that holds the point;
- **ink box**: the cell, or with INK the columns of the cell that hold ink.

Refinements (flags):

- **calibration** (`cal`): where a model's CTC fires is a property of the model. Per model, a shift for every
  character and an offset for each of seven groups of characters that sit at one side of their cell or are narrow
  (`close` 、。」』）…, `open` 「『（…, `dots` …‥・, `small` small kana, `long` ー―～, `mark` ！？♪…, `half`
  half-width), in resized px. They are medians of MeikiOCR's box centre minus the frame centre, on **training-side**
  lines the model reads exactly as MeikiOCR does (`eval/charbox_eval.py calibrate`, `eval/charboxes_cal.json`);
- **INK**: the cuts between characters move to the emptiest run of columns between their centres (an ink-column
  gap; the ink profile is each crop column's vertical gradient energy, Σ |g(y + 1) − g(y)|, which needs no polarity),
  each ink box shrinks to its cell's columns above a quarter of the line's peak, and the centre moves to the middle
  of that ink;
- **PITCH**: on a line of at least 4 characters whose spacings are all within 25% of their median (a fixed font
  pitch), the centres move to the least-squares grid a + p k before the cells are cut.

Vertical text: VOCR5-V reads a column turned 90° anticlockwise (`vertical_twin.rotate_ccw`: the column's top is the
strip's left), so the same positions along the strip are positions down the column.

### What was decided before this protocol, and on what

All on **training-side** data only: teacher_train's crops with MeikiOCR's character boxes, lines VOCR5 reads exactly
as MeikiOCR, no dev_sample, test or CLANNAD VN. An exploration on other training lines (issue #114) chose the arms;
the calibration (`calibrate`, the first 3,000 lines of a seeded order) and then `train-check` (the next 1,500 lines,
`eval/charboxes_protocol.json` "method_of_record") measured them. Centre inside MeikiOCR's box:

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
97-99% of the reference characters get a box.

**Hand-checked set.** 96 dialogue lines (48 Vita native, 48 PSP native, seed 20261023; `eval/charbox_hand.py`) with
MeikiOCR's boxes drawn on contact sheets, checked by eye (Claude, reviewer_kind assistant) before any score: every
box of all 96 lines sits on its glyph. Its score is reported as the subset whose every box was checked.

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
  for (a missed line, a deleted character) as misses.
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
