# CLANNAD: v1, v2, v3 and Google Lens against one set of references

On the 3,048 CLANNAD lines that have a reference (2,044 screens, 35,694 reference characters), Google Lens
has 1.47% CER and reads 97.5% of lines exactly. The best of our recognizers is VOCR3_W20 at 2.34% CER
[2.13, 2.55] and 80.8% exact. Next come F20 (2.75%), v0.2.0 H15 (2.82%), VOCR3_FL10 and VOCR3_H15 (3.24%
each), VOCR2 (4.14%), FL10 (4.56%, the long run's model) and v1 (13.67%).

- **Dialogue.** The three v3 models' CER is below Lens's (1.17-1.46% against 1.53%), but the intervals
  overlap. Lens still reads 96.5% of dialogue lines exactly, against VOCR3_W20's 85.3%: all 526 of its errors
  sit on 77 lines, and 10 of those lines hold 346. Our errors are spread over 584 lines (VOCR3_W20) or more.
- **Names.** Our models lose most on speaker names. The finder's name-plate crop takes in scene art, and for
  most of our models (all but F20, at 46.5%) over half their name errors are characters beyond the
  reference's length.
- **Crops with no text.** The v3 models write more than v0.2.0 on them.

VOCR3_W20 beats both v0.2.0 H15 and F20, and VOCR3_FL10 beats FL10. VOCR3_H15 has a higher CER than v0.2.0
H15 (+0.42 points [+0.20, +0.63]) because of names. On dialogue it is lower.

The references are Claude reads, not human ones. The cached FL10's disagreement with Lens on all 4,271 lines
reproduces the published 8.60%, and against the references FL10's CER is 4.56%.

This is a report on a protected test title. Nothing here chose, tuned, calibrated, qualified or dropped a
model, threshold, finder or recipe, and the references and crops never enter any training, development or
calibration pool. The protocol (`eval/clannad_refs_protocol.json`) was committed as `991d71c` and pushed at
2026-10-13T20:34:43Z (GitHub's activity log for `feat/owner-drive-ingest`), before any arm read a CLANNAD
crop; the arm outputs were written at 20:56:04Z. Its amendments were committed at 22:24:36Z, before the
references were frozen at 03:04:39Z on 2026-10-14. The branch was rebased onto main later, which changed
the commit ids but not the author dates; `991d71c` itself is still on GitHub.

## What was compared

**Lines.** The owner's Google Drive drop of 2026-10-13 holds the CLANNAD long run (`vita/CLANNAD/training-data`),
with the pins that `eval/vocr3_clannad.py` publishes (lock `14ba9297…`, captures `b6cd582c…`, comparisons
`f2754897…`). Every line of every successful page is read: 4,271 lines on 2,177 pages, of which the 3,048
with a reference are scored. The 27 pages that failed in the long run are left out for every system and
were never retried. Each line is the stored
processed line PNG: the C GRAD finder's crop of the dialogue box, grey, with dark text inverted once. These
are the pixels Google Lens and the cached FL10 read in the long run.

**Systems.** Every arm runs through the C runtime in deployment W8I (`runtime/vocr.c` built now as the scalar
library, `-std=c99 -O2 -ffp-contract=off -DVOCR_NO_NEON`, max width 1024) on the same stored crop, as
`eval/vocr3_clannad.py`'s paired-line mode does. The finder column is each blob's deployment finder; here
every arm reads the GRAD finder's crops, so VOCR2's numbers are recognition only, without its own textbox
finder.

| system | blob | sha256 | finder | int8 bytes | W8I arena | what it is |
|---|---|---|---|---:|---:|---|
| v1 | `v1_w8.vocr` | `23da2a2a…` | grad | 859,904 | 124,832 | v0.1.0 baseline, synthetic training only |
| VOCR1 | `FL10_w8.vocr` | `500e0b84…` | grad | 657,664 | 124,832 | v0.2.0 FL10, the long run's model |
| VOCR_H15 | `H15_w8.vocr` | `316ebc37…` | grad | 1,155,968 | 182,944 | v0.2.0 H15, vita-jp-overlay's default |
| F20 | `F20_w8.vocr` | `f8c1abe4…` | grad | 1,558,016 | 241,056 | v0.2.0 F20 |
| VOCR2 | `VOCR2_FL10_w8.vocr` | `d3adc52e…` | textbox | 657,664 | 124,832 | VOCR2_FL10 (v0.2.3-vocr2.1) |
| VOCR3_FL10 | `VOCR3_FL10_w8.vocr` | `87fb4d0c…` | grad | 657,664 | 124,832 | vocr3 selection FL10 |
| VOCR3_H15 | `VOCR3_H15_w8.vocr` | `cb3c61a8…` | grad | 1,155,968 | 182,944 | vocr3 selection H15 |
| VOCR3_W20 | `VOCR3_W20_w8.vocr` | `2cedeb01…` | grad | 1,558,016 | 241,056 | vocr3 selection W20 |

Lens is the cached Google Lens reading (owocr 1.26.7) of each line from the long run, one request per line,
nothing re-requested. "cached" is the FL10 output the long run stored; VOCR1 run now gives the same text on
all 4,271 lines.

## References

The drop has no human references, so the references are Claude Opus 5.5 reads of contact sheets: each
framed crop in its native frame's context, at most 1280 px per sheet, the screen outside the frame dimmed
(on every sheet but R1's b11 and b12). No person checked them.

- **R1**, every line once. The candidates are the distinct plausible readings of two local labelling engines
  on the same crop (MeikiOCR's recognizer and PP-OCRv6 small, pinned, offline), never Lens or any compared
  system. A read picks a candidate, types the text, marks the line unsure, or skips it (no text, more than
  one line, a glyph cut by the frame, UI, or anything else unreadable).
- **R2**, every line again, by another session. The candidates are R1's text, both engines, Lens and every
  arm, shuffled per line, with no sources and no counts.
- **R3**, a third read where R1 and R2 do not give the same agreement key (`teacher/vote.key`, the scoring
  normalisation), with R2's decision added to the candidates.

A line's reference is the text two reads give with one key. A line that two reads skip or mark unsure, or on
which no two reads agree, is excluded for every system and counted.

The reads, in numbers:

- **R1** (4,271 lines): picked 2,772, typed 314, skipped 1,153 (904 no text, 225 cut, 16 multi, 7 UI, 1 bad)
  and marked 32 unsure.
- **R2** (4,271 lines): picked 3,039, typed 10, skipped 1,189 and marked 33 unsure. On nearly every text line
  it picked one of its candidates, which included R1's reading.
- **Agreement.** On the 3,025 lines both rounds read as text, they gave the same key on 3,006 (99.4%).
- **R3** (104 lines) gave 42 references; two of its three reads were non-text on 52 lines, and no two reads
  agreed on 10.

| outcome | lines |
|---|---:|
| reference, R1 = R2 | 3,006 |
| reference, two of three | 42 |
| excluded: R1 and R2 both found no text | 904 |
| excluded: R1 and R2 both found a glyph cut by the frame | 213 |
| excluded: one cut, one unsure | 12 |
| excluded: R1 and R2 both more than one line | 13 |
| excluded: R1 and R2 both unsure | 13 |
| excluded: other pairs of non-text reads (UI, bad, mixed) | 6 |
| excluded: two of three non-text | 52 |
| excluded: no two reads agree | 10 |
| **scored / excluded** | **3,048 / 1,223** |

Of the 3,048 scored lines, 996 are speaker names (2,209 reference characters) and 24 are inverted (dark text).

Every scored line whose reference's scoring key differs from Lens's reading is flagged `lens_disagree`: 76 of
the 3,048 (2.5%; 64 settled by R1 = R2 and 12 by R3; 5 are names). R1 never saw Lens, and on none of the 76
did it
give Lens's reading. The flagged lines are listed privately for the owner (`clannad-compare/lens_disagree.jsonl`
under `shared/private/`), never here.

The references were frozen at 2026-10-14T03:04:39Z, before any score was computed (`refs.freeze.json`:
refs sha256 `35411688749b2968…6212b`, with the sha256 of every round, every decision file and the protocol,
`3b86ca54…`). The score then ran once.

**Amendments.** All were made before the freeze and are recorded in the protocol file:

- the R1 sheets of b01-b10 were re-rendered with the context dimmed before any reviewer opened them (b11 and
  b12 keep the first render);
- R2 was built per finished R1 batch rather than after all of R1;
- the R1 reader of b11 and b12 corrected 21 of its own reads on geometry measured on the crops, each before
  that batch's R2 existed;
- one deviation: while measuring geometry, that reader saw the cached FL10 reading of 5 lines; 4 were read
  afterwards (they follow the image, one against that reading). All 5 ended as references on which R2,
  reading independently, agreed with R1;
- later blobs (such as VOCR4) can be scored against the frozen references in separate files; they are not
  part of this protocol.

## Results

The score ran once on the frozen references; the public result is `eval/clannad_v1_v2_v3_lens_20261014.json`
(counts, rates, intervals and hashes only). All 3,048 scored lines: 2,044 screens and 35,694 reference
characters. The intervals are 95%, from 10,000 bootstrap resamples of whole screens. `cached`, the long run's
stored FL10 output, equals VOCR1 on every line.

| system | errors | CER % [95% CI] | exact % [95% CI] | hiragana errors / chars |
|---|---:|---|---|---:|
| v1 | 4,881 | 13.67 [12.69, 14.70] | 58.1 [56.4, 59.8] | 1,303 / 20,120 |
| VOCR1 (FL10) | 1,627 | 4.56 [4.29, 4.83] | 63.3 [61.5, 64.9] | 736 / 20,120 |
| VOCR_H15 | 1,006 | 2.82 [2.61, 3.04] | 75.5 [74.0, 76.9] | 474 / 20,120 |
| F20 | 981 | 2.75 [2.56, 2.94] | 74.7 [73.1, 76.2] | 551 / 20,120 |
| VOCR2 | 1,479 | 4.14 [3.89, 4.41] | 66.2 [64.6, 67.9] | 652 / 20,120 |
| VOCR3_FL10 | 1,155 | 3.24 [2.96, 3.51] | 75.1 [73.6, 76.6] | 477 / 20,120 |
| VOCR3_H15 | 1,155 | 3.24 [2.96, 3.52] | 77.4 [75.9, 78.8] | 375 / 20,120 |
| VOCR3_W20 | 834 | 2.34 [2.13, 2.55] | 80.8 [79.4, 82.2] | 299 / 20,120 |
| Lens | 526 | 1.47 [0.88, 2.14] | 97.5 [96.9, 98.0] | 105 / 20,120 |

The protocol's pairs, candidate minus baseline in points, from the paired screen bootstrap (a negative CER
delta is better):

| candidate : baseline | CER delta [95% CI] | exact delta [95% CI] |
|---|---|---|
| VOCR1 : v1 | -9.12 [-10.01, -8.26] | +5.2 [+3.4, +6.8] |
| VOCR2 : VOCR1 | -0.41 [-0.51, -0.32] | +3.0 [+2.2, +3.8] |
| VOCR_H15 : VOCR1 | -1.74 [-1.94, -1.54] | +12.2 [+10.7, +13.8] |
| F20 : VOCR_H15 | -0.07 [-0.23, +0.08] | -0.8 [-2.1, +0.5] |
| VOCR3_FL10 : VOCR1 | -1.32 [-1.56, -1.08] | +11.8 [+10.4, +13.4] |
| VOCR3_FL10 : VOCR2 | -0.91 [-1.13, -0.69] | +8.9 [+7.4, +10.3] |
| VOCR3_H15 : VOCR_H15 | +0.42 [+0.20, +0.63] | +1.9 [+0.5, +3.4] |
| VOCR3_H15 : VOCR3_FL10 | +0.00 [-0.20, +0.20] | +2.3 [+1.0, +3.6] |
| VOCR3_W20 : VOCR_H15 | -0.48 [-0.66, -0.30] | +5.4 [+3.9, +6.8] |
| VOCR3_W20 : F20 | -0.41 [-0.59, -0.23] | +6.2 [+4.7, +7.6] |
| VOCR3_W20 : VOCR3_H15 | -0.90 [-1.11, -0.70] | +3.5 [+2.1, +4.9] |
| v1 : Lens | +12.20 [+11.09, +13.31] | -39.4 [-41.1, -37.6] |
| VOCR1 : Lens | +3.08 [+2.41, +3.71] | -34.2 [-35.9, -32.5] |
| VOCR_H15 : Lens | +1.34 [+0.67, +1.95] | -22.0 [-23.5, -20.5] |
| F20 : Lens | +1.27 [+0.60, +1.88] | -22.8 [-24.4, -21.3] |
| VOCR2 : Lens | +2.67 [+2.00, +3.30] | -31.2 [-32.9, -29.6] |
| VOCR3_FL10 : Lens | +1.76 [+1.08, +2.38] | -22.4 [-23.9, -20.9] |
| VOCR3_H15 : Lens | +1.76 [+1.08, +2.39] | -20.1 [-21.5, -18.7] |
| VOCR3_W20 : Lens | +0.86 [+0.20, +1.47] | -16.6 [-18.0, -15.2] |

Each v3 selection is at least as good as its v0.2.0 counterpart, with one exception: VOCR3_H15's CER is
higher than v0.2.0 H15's, though it reads more lines exactly. F20 and H15 are tied. Every recognizer is
behind Lens.

**Names and dialogue.** Each subset has its own screen bootstrap. There are 996 name lines (2,209
characters) and 2,052 dialogue lines (33,485 characters).

| system | names: CER % [95% CI] | names: exact % | dialogue: CER % [95% CI] | dialogue: exact % |
|---|---|---:|---|---:|
| v1 | 169.81 [155.70, 184.35] | 50.7 | 3.37 [3.15, 3.62] | 61.7 |
| VOCR1 (FL10) | 26.66 [23.98, 29.43] | 62.4 | 3.10 [2.89, 3.32] | 63.6 |
| VOCR_H15 | 16.89 [14.85, 19.06] | 73.3 | 1.89 [1.72, 2.07] | 76.5 |
| F20 | 12.95 [11.39, 14.57] | 75.0 | 2.08 [1.90, 2.26] | 74.5 |
| VOCR2 | 25.53 [22.93, 28.21] | 63.8 | 2.73 [2.53, 2.94] | 67.4 |
| VOCR3_FL10 | 30.19 [26.84, 33.69] | 62.7 | 1.46 [1.30, 1.61] | 81.1 |
| VOCR3_H15 | 33.00 [29.39, 36.69] | 64.6 | 1.27 [1.13, 1.42] | 83.6 |
| VOCR3_W20 | 20.01 [17.60, 22.54] | 71.7 | 1.17 [1.03, 1.31] | 85.3 |
| Lens | 0.68 [0.13, 1.44] | 99.4 | 1.53 [0.90, 2.25] | 96.5 |

In this game the GRAD finder's crop of a name plate is often much wider than the name. Of the 996 name
crops, 449 are over 500 px wide, up to 896 px (the whole dialogue-box region), where a tight name crop is
about 70-100 px. Those wide crops take in the scene art beside the plate, and our recognizers write
characters for the art. On name lines, the share of each model's errors that are characters beyond the
reference's length (a lower bound on insertions) runs from 46.5% (F20) to 89.5% (VOCR3_FL10); Lens reads
only the name.

On dialogue the three v3 models have the lowest CER point estimates of all systems, Lens included, though
their intervals overlap Lens's. Lens still reads far more lines exactly. Its errors are few and
concentrated: 77 lines hold all 526 of them, and 10 lines hold 346. Our errors are spread thin: among the
v2/v3 recognizers compared with Lens, over 584 lines (VOCR3_W20) to 1,120 (FL10); v1, far behind, has
errors on 1,277. Only 24 scored lines are inverted (92 characters), too few to report.

**Crops with no text.** Two reads found 904 crops empty (scene art, box edges): line-finder false positives.
They are not in CER. This is what each system writes on them:

| system | crops with any output | characters written |
|---|---:|---:|
| v1 | 891 | 7,170 |
| VOCR1 (FL10) | 350 | 550 |
| VOCR_H15 | 296 | 370 |
| F20 | 284 | 335 |
| VOCR2 | 377 | 604 |
| VOCR3_FL10 | 637 | 1,625 |
| VOCR3_H15 | 552 | 1,073 |
| VOCR3_W20 | 450 | 1,033 |
| Lens | 19 | 20 |

The v3 selections write on more empty crops than their v0.2.0 counterparts. VOCR3_FL10 writes on 637 of the
904; FL10 writes on 350.

**The Lens proxy.** The long run measured each system's edit disagreement with Lens on the same pixels. This
is never CER.

| system | all 4,271 lines (36,915 Lens characters) | the 3,048 scored lines | CER against the references |
|---|---:|---:|---:|
| v1 | 13,523 (36.63%) | 5,273 (14.65%) | 13.67% |
| VOCR1 (FL10) | 3,176 (8.60%) | 2,045 (5.68%) | 4.56% |
| VOCR_H15 | 2,457 (6.66%) | 1,414 (3.93%) | 2.82% |
| F20 | 2,293 (6.21%) | 1,409 (3.92%) | 2.75% |
| VOCR2 | 3,069 (8.31%) | 1,891 (5.25%) | 4.14% |
| VOCR3_FL10 | 3,837 (10.39%) | 1,573 (4.37%) | 3.24% |
| VOCR3_H15 | 3,171 (8.59%) | 1,588 (4.41%) | 3.24% |
| VOCR3_W20 | 2,929 (7.93%) | 1,269 (3.53%) | 2.34% |

On all lines the cached FL10 reproduces the published 3,176 / 36,915 (8.60%,
`docs/clannad-longrun-results-20261012.md`), and VOCR2 reproduces the published 3,069 (8.31%,
`docs/clannad-vocr2-comparison-20261012.md`). The proxy overstates every system's error against the
references.

On all lines the proxy also ranks VOCR3_FL10 behind FL10 (10.39% against 8.60%), where the references rank
it ahead (3.24% against 4.56%). The empty crops make the difference: VOCR3_FL10 writes 1,625 characters on
them, and FL10 writes 550.

## Limits

- The references are an assistant's reads with a second and, where needed, a third read: no person checked
  them. R2 saw R1's reading, Lens and every arm among its unattributed candidates, so R1 and R2 are not fully
  independent, and a reference can lean towards a reading that R1 or some system produced. Where every
  candidate was wrong, a read had to type the text; R2 typed 10 of 4,271.
- Lens is one cached automated reading per line, not a reference.
- One title, one long run, the GRAD finder's crops only: these numbers do not say how the systems' own
  finders do, and say nothing about other games. The shared hand-checked benchmark (`eval/`) remains the
  qualification set.

## planetarian (owner 2026-10-13 20:47Z)

The owner made planetarian (v34, all releases) training data, allowed even where it overlaps an eval set, and
asked for any of its lines in teacher_val, devnative, vita-test or psp-test to be removed in a new manifest
version. It has no rows in any of them (by VN and by release session), nor in teacher_distill_val or
real_val, so no manifest changed and every published score is unchanged (old = new). Its 202 frames are in
`owner-games/psp-planetarian-v34/` for training (`docs/owner-games.md`). Its PSP port and CLANNAD's Vita port
are both Key/Prototype releases, and VNDB lists a character relation between planetarian and v5698 (CLANNAD:
Lost "Winter", on the CLANNAD family list): a CLANNAD score of any model trained on planetarian must disclose
that exposure. The systems above were trained before the drop arrived, and none of the teacher, realdata,
realplus (v1, v2) or devnative pools holds a planetarian row.

## Reproduce

```sh
W=$HERD/shared/private/owner-drive-20261013/clannad-compare
S=$HERD/shared/private/owner-drive-20261013/vn-ocr-dataset-20261013/vita/CLANNAD/training-data
.venv-ci/bin/python eval/clannad_refs.py lines   --session $S --work $W
teacher/.venv/bin/python eval/clannad_refs.py engines --work $W        # offline (unshare -rn)
.venv-ci/bin/python eval/clannad_refs.py arms    --work $W --blobs $A/owner-drive-ingest/blobs
.venv-ci/bin/python eval/clannad_refs.py sheets  --work $W --round r1 --out $W/r1 --lines-per-batch 380
.venv-ci/bin/python eval/clannad_refs.py sheets  --work $W --round r2 --batch $W/r1/bNN --out $W/r2/bNN
.venv-ci/bin/python eval/clannad_refs.py collect --work $W --round r1 --batches $W/r1   # then r2
.venv-ci/bin/python eval/clannad_refs.py sheets  --work $W --round r3 --out $W/r3        # then collect r3
.venv-ci/bin/python eval/clannad_refs.py refs    --work $W
.venv-ci/bin/python eval/clannad_refs.py score   --work $W --out eval/clannad_v1_v2_v3_lens_20261014.json   # once
```

A later blob is scored against the same frozen references without touching `$W`:
`arms --work $W --blobs DIR --extra NAME=FILE#SHA256 --suffix S --into DIR2`, then
`score --work $W --suffix S --into DIR2 --pair NAME:VOCR3_H15 --out OUT.json` (once per suffix).
