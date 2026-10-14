# CLANNAD: v1, v2, v3, MeikiOCR and Google Lens against one set of references

On the 3,048 CLANNAD lines that have a reference (2,044 screens, 35,694 reference characters), Google Lens
has 1.47% CER and reads 97.5% of lines exactly. The best of our recognizers is VOCR3_W20 at 2.34% CER
[2.13, 2.55] and 80.8% exact. Next come F20 (2.75%), v0.2.0 H15 (2.82%), VOCR3_FL10 and VOCR3_H15 (3.24%
each), VOCR2 (4.14%), FL10 (4.56%, the long run's model) and v1 (13.67%).

MeikiOCR (rtr46/meikiocr 0.3.5, a desktop model run locally on the CPU) was added on 2026-10-14 (owner
request), in two arms kept apart. On the same stored line crops its recognizer reads **dialogue** at 0.78%
CER [0.65, 0.93] — below every system here, Lens included — but its overall line CER is 14.21%, because on
the finder's wide name-plate crops it reads the scene art and writes far past the name (names 217.75% CER;
94.9% of those errors are characters beyond the name's length, 4,690 of 4,810 on crops over 500 px). On the
full screens its own detector isolates most of those name plates from the scene art, and whole dialogue per
screen comes to
7.54% CER [6.93, 8.19], between VOCR3_W20 (4.86%) and VOCR2/VOCR3_FL10. MeikiOCR is **not a Vita candidate**:
its three fp32 ONNX models are 46 MB, and even the project's int8 MNN conversion of the recognizer alone is
5.4 MB (notes/27 measured about 8 s per line on the console). It also helped build the references (its
recognizer was one of R1's two engines), so arm (a)'s numbers are entangled with them; see "MeikiOCR" below.


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
| MeikiOCR | 5,072 | 14.21 [12.95, 15.52] | 78.1 [76.7, 79.6] | 249 / 20,120 |
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

MeikiOCR's two pairs (arm (a), the stored line crops; not protocol pairs, scored once against the frozen
references under `eval/clannad_meiki_protocol.json`):

| candidate : baseline | CER delta [95% CI] | exact delta [95% CI] |
|---|---|---|
| MeikiOCR : VOCR3_W20 | +11.87 [+10.72, +13.08] | -2.7 [-4.2, -1.2] |
| MeikiOCR : Lens | +12.74 [+11.41, +14.11] | -19.3 [-20.8, -17.8] |

Both deltas are driven entirely by the name plates: on dialogue alone MeikiOCR is slightly *ahead* of both
(a post-hoc paired page bootstrap, descriptive only: MeikiOCR − VOCR3_W20 −0.39 [−0.53, −0.25],
MeikiOCR − Lens −0.74 [−1.47, −0.11]).

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
| MeikiOCR | 217.75 [199.68, 236.89] | 51.6 | 0.78 [0.65, 0.93] | 91.0 |
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
| MeikiOCR | 904 | 10,757 |
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

## MeikiOCR (owner 2026-10-14)

The owner asked to add [MeikiOCR](https://github.com/rtr46/meikiocr) (rtr46; `meikiocr` 0.3.5, PyPI wheel =
GitHub commit `52aa607`) to this comparison. It is a desktop Japanese-VN OCR: a text detector plus a
horizontal and a vertical recognizer, all ONNX, run here locally on the CPU — never a hosted API. It ran
with the package's own defaults (`run_ocr` / `run_recognition`, `det_threshold` 0.5, `rec_threshold` 0.1,
`punct_conf_factor` 1.0, one thread), offline (`unshare -rn`), on the same pinned model files the teacher
labeller uses (`teacher/engines.PINS`; sha256-checked, reused). The protocol
(`eval/clannad_meiki_protocol.json`) was committed as `6494471` and pushed at 2026-10-14T08:24:08Z (GitHub's
activity log for `feat/meiki-clannad`) before MeikiOCR read any CLANNAD crop; both arms wrote their outputs
afterwards and each score ran once against the same frozen references (`refs.freeze.json`,
`35411688…`). The public results are `eval/clannad_meiki_lines_20261014.json`,
`eval/clannad_meiki_screens_20261014.json` and `eval/clannad_meiki_resources_20261014.json` (counts, rates,
intervals, hashes, sizes and host times only).

**Entanglement with the references.** MeikiOCR's recognizer is one of the two engines (with PP-OCRv6 small)
whose readings were R1's candidates, and its reading was among R2's and R3's unattributed candidates. So the
references lean towards it: of the 3,048 scored lines, 2,382 have a reference key equal to R1's MeikiOCR
reading (2,263 equal PP-OCRv6 small's); on dialogue, 1,868 of 2,052. Arm (a)'s dialogue numbers below are
therefore optimistic for MeikiOCR against every system that was not an R1 candidate, and arm (b) uses the
same recognizer. Nothing here corrects for this; it is why MeikiOCR is reported as an added arm, not folded
into the ranking.

### Arm (a): recognition on the stored line crops

The three tables above include MeikiOCR. On the same GRAD-finder crops as every other arm it reads
**dialogue** (the 2,052 not-name lines, 33,485 characters) at **0.78% CER [0.65, 0.93]** and 91.0% exact —
the lowest dialogue CER of any system, Lens's 1.53% included, though the entanglement above inflates it. Its
**overall** line CER is 14.21%, worse than every v2/v3 model, entirely because of **names**: 996 name lines,
217.75% CER, 4,810 errors, of which 94.9% are characters beyond the name's length and 4,690 of 4,810 fall on
the 449 name crops wider than 500 px. MeikiOCR has no line-level box of its own here — it is handed the whole
wide crop — so it reads the name *and* the scene art beside it. On the 547 name crops at most 500 px wide it
is close (120 errors against VOCR3_W20's 24, 514 of 547 exact). It writes on all 904 empty (no-text) crops,
10,757 characters, 10,340 of them Latin/digits — garbage from scene art, the most of any system. 78.1% of
all lines are exact (second only to Lens): the name damage is concentrated, not spread. The vertical
recognizer was used on 26 crops (all names, 0 errors). Only 24 lines are inverted (dark), too few to read
into (51.09% on 92 characters). Its edit disagreement with Lens (the long run's proxy) is 47.33% on all
4,271 lines and 15.34% on the scored lines — the empty-crop and name garbage, not a CER.

### Arm (b): MeikiOCR's own detection and recognition on the full screens

Kept apart from (a): MeikiOCR's own detector finds the lines on each stored 960×544 frame, and its
recognizer reads them. A detection is scored only if its box centre lies in the long-run ROI
(`[32, 360, 896, 184]`, the region the GRAD finder scanned and the only region the references cover);
detections elsewhere are counted, never scored. Each screen's kept texts are joined in reading order and
scored against the screen's reference lines (`general_vn_textbox_guard.transcript_score`), one bootstrap over
whole screens. The screens whose every crop is a reference or a two-read empty crop are scored (1,837 of
2,177; 91 are all-empty, 249 are left out whole because a crop was cut, multi-line, UI or unsettled).

| system | screen CER % [95% CI] | dialogue CER % [95% CI] | exact screens % [95% CI] |
|---|---|---|---:|
| MeikiOCR_screen (its own det + rec) | 7.54 [6.93, 8.19] | 8.06 [7.40, 8.76] | 43.0 [40.8, 45.3] |
| VOCR3_W20 (GRAD crops) | 4.86 [4.51, 5.23] | 4.86 [4.49, 5.24] | 58.6 [56.5, 60.8] |
| VOCR_H15 (GRAD crops) | 3.57 [3.33, 3.82] | 3.23 [2.99, 3.49] | 58.0 [55.7, 60.2] |
| Lens (GRAD crops) | 1.65 [1.00, 2.38] | 1.76 [1.07, 2.54] | 95.4 [94.4, 96.3] |
| MeikiOCR (arm a, GRAD crops) | 40.52 [38.26, 42.86] | 42.52 [40.14, 44.97] | 41.0 [38.7, 43.2] |

(The GRAD-crop systems here read every crop of a page, names included, so their screen CER is above their
line CER; MeikiOCR arm (a) on whole screens is dominated by the name plates, as in (a).) Pairs, candidate
minus baseline, paired screen bootstrap:

| pair | screen CER delta [95% CI] | dialogue CER delta [95% CI] | exact delta [95% CI] |
|---|---|---|---|
| MeikiOCR_screen : VOCR3_W20 | +2.69 [+2.10, +3.29] | +3.21 [+2.57, +3.86] | -15.6 [-18.4, -12.8] |
| MeikiOCR_screen : Lens | +5.90 [+4.97, +6.80] | +6.30 [+5.30, +7.27] | -52.4 [-54.8, -49.9] |
| MeikiOCR_screen : MeikiOCR (arm a) | -32.97 [-35.15, -30.91] | -34.45 [-36.73, -32.29] | +2.0 [-0.7, +4.8] |

So MeikiOCR's own detector is far better than feeding it the wide GRAD crops (−32.97 points): it mostly
isolates the name plate from the scene art. On whole dialogue per screen it lands between VOCR3_W20 and Lens,
+2.69 points of screen CER behind VOCR3_W20. **This comparison favours the GRAD-crop systems.** The
references only cover text inside the GRAD finder's crops, so dialogue the finder missed or cut counts
against `MeikiOCR_screen` as insertions and against no GRAD-crop system, and the 249 pages with a cut or
multi-line crop are dropped whole. Of MeikiOCR's 3,332 kept detections on scored screens, 471 (1,478
characters) overlap no GRAD crop: mostly name plates the references did not score as a line at that position,
plus some dialogue the GRAD finder marked empty and MeikiOCR actually read (a sample was inspected privately
and deleted). As a diagnostic only, dropping every detection that overlaps no GRAD crop would put
`MeikiOCR_screen` at 3.52% — i.e. much of its gap to VOCR3_W20 is these extra detections, not misreads.

### Size and host speed

MeikiOCR is a desktop model and **not a Vita candidate**. The three fp32 ONNX files are the detector
14,503,825 B (3,547,383 parameters), the horizontal recognizer 18,593,254 B (4,567,437) and the vertical
recognizer 12,872,961 B (3,151,989): about 46 MB in all, against VOCR3_W20's 1,558,016 B int8 blob with a
241,056 B arena. Arm (a) loads the recognizer (and the vertical recognizer); arm (b) loads all three. The
project's own int8 MNN conversion of the recognizer alone is still 5,392,856 B (`ports/meiki/`), larger than
the whole Vita budget; on a PCH-1001 inside vita-jp-overlay it took about 8 s per line and was removed
(`notes/27-Meiki-hardware-and-project-lessons.md`, `docs/vita-memory-budget.md`).

Host speed on this box (AMD EPYC 9R14, one thread, the same seeded sample of 200 lines and 50 screens, median
crop width 433 px): MeikiOCR's recognizer is 20.6 ms per line (model load 577 ms), its detector 117 ms per
screen and `run_ocr` (detect + recognise) 211 ms per screen. The VOCR arms, built as the comparison's scalar
C runtime (`-DVOCR_NO_NEON`, W8I), are 41.9 ms per line (VOCR3_W20, load 14.5 ms) and 28.3 ms (VOCR_H15).
These host numbers are **not comparable across the two**: onnxruntime uses vectorised CPU kernels, while the
VOCR build is the deliberately unvectorised scalar twin of the console code. For the console, VOCR3_W20 is an
estimated 314-554 ms per 640-px line at 333 MHz on one Cortex-A9 (`vita/a9_cycles.py`, `docs/vocr3.md`);
MeikiOCR has no console-sized build.

## Limits

- The references are an assistant's reads with a second and, where needed, a third read: no person checked
  them. R2 saw R1's reading, Lens and every arm among its unattributed candidates, so R1 and R2 are not fully
  independent, and a reference can lean towards a reading that R1 or some system produced. Where every
  candidate was wrong, a read had to type the text; R2 typed 10 of 4,271.
- Lens is one cached automated reading per line, not a reference.
- MeikiOCR (added 2026-10-14) helped build the references (its recognizer was one of R1's two candidate
  engines), so arm (a)'s dialogue numbers are entangled with them and favour MeikiOCR; and its arm (b),
  scored against GRAD-crop-only references, is favoured against the GRAD-crop systems (missed or cut dialogue
  counts as insertions only against it). It is a 46 MB desktop model, reported for context, never a Vita
  candidate. See "MeikiOCR" above.
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

MeikiOCR (added 2026-10-14; `eval/clannad_meiki_protocol.json`, `eval/clannad_meiki.py`), with its own
hash-pinned env `.venv-meiki` (`eval/clannad_meiki_requirements.lock`) and `P=$HERD/shared/private/meiki-clannad`
(chmod 700). Both arms run offline through `heavy.sh`; each score runs once:

```sh
.venv-meiki/bin/python eval/clannad_meiki.py verify  --into $P
unshare -rn env HF_HUB_OFFLINE=1 .venv-meiki/bin/python eval/clannad_meiki.py lines   --work $W --into $P   # arm (a)
unshare -rn env HF_HUB_OFFLINE=1 .venv-meiki/bin/python eval/clannad_meiki.py screens --work $W --into $P   # arm (b)
.venv-ci/bin/python eval/clannad_refs.py score --work $W --suffix meiki --into $P \
    --pair MeikiOCR:VOCR3_W20 --pair MeikiOCR:Lens --out eval/clannad_meiki_lines_20261014.json      # once
.venv-ci/bin/python eval/clannad_meiki.py score-screens --work $W --into $P \
    --out eval/clannad_meiki_screens_20261014.json                                                   # once
.venv-meiki/bin/python eval/clannad_meiki.py resources --into $P                                     # model sizes
unshare -rn env HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 .venv-meiki/bin/python eval/clannad_meiki.py speed --engine meiki --work $W --into $P
.venv-ci/bin/python eval/clannad_meiki.py speed --engine vocr --work $W --into $P --blobs $A/owner-drive-ingest/blobs
.venv-ci/bin/python eval/clannad_meiki.py report --into $P --out eval/clannad_meiki_resources_20261014.json
```
