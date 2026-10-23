# headgate: skipping the output head on blank CTC frames (issue #117, item 1)

About three CTC frames in five read as the blank, and the output head (96 → 4,987 or 5,631 classes) is 15-40% of a
frame's time. A 96 → 16 → 1 gate on the head's input now says "blank" before the head runs, and the runtime skips the
head there. It is fit on each VOCR5 model's own frozen frames; nothing is retrained.

**Result.** The pre-registered rule (`eval/headgate_protocol.json`, frozen before any gate was trained) adopts the
gate for all three tiers.

- **Skip rate.** The gate skips 61.4% / 61.0% / 57.9% of the frames of the finder-cut development lines (S / N / L).
  It skips 0.16-0.18% of the frames whose argmax is not the blank.
- **Accuracy.** No decode is resolved worse on the development score: greedy, Rubi's beam + 1 MB LM with the counter
  re-decode, or that with Rubi's confidence cascade. Every 95% upper bound is at most +0.050 points, the margin. Normal
  and Large pass with room; Small passes at the rule's edges (its greedy lower bound is exactly 0.000 and its cascade
  upper bound +0.050). These are the corrected numbers ("Development", "Correction").
- **Time saved (estimates).** About −122 / −137 / −163 ms per nominal 3-line box at 333 MHz on one Cortex-A9 core,
  without prefetch credit (−19.5% / −14.9% / −7.9%). On a real Vita box (89 frames) that is −45 / −51 / −60 ms.
- **Bytes.** Each blob grows by 1,792 B (`VOCR5_{S,N,L}_w8g.vocr`), the W8I arena by 48 B, and `vocr.o` by
  1,940 B (VitaSDK `-O2`, NEON).

Nothing ran on a console: every millisecond is an estimate from qemu instruction counts, calibrated to VOCR3-H15's
measured 5.0-5.2 ms per CTC frame.

The protected sets were read once, after this verdict was pushed ("Protected sets"). Greedy text is unchanged on every
vita-test and psp-test dialogue line. Dialogue CER changes by −0.043 to +0.006 points on them and on CLANNAD, with
0-3 lines changed per tier.

## How it works

**The gate.** Per tier, a 96 → 16 (ReLU) → 1 network on the head's input, as the W8I head sees it:

- **Layer 1:** int8, one scale per hidden unit, through the head's own integer kernel (`mm_i16`: exact int32 sums,
  then float).
- **Layer 2:** float32, summed in a fixed order.
- **Decision:** a frame whose score is above the gate's threshold *t* skips the head.

It costs 1,552 MACs a frame, against 478,752 (S) or 540,576 (N, L) for the head.

**A skipped frame.** Its argmax is 0, its top-K is `[blank logit, -inf, ...]` with classes 0, and it never emits a
character. Its blank logit is the head's own class-0 logit: the head's first 8-class block on the same int16 input,
bit for bit. The other frames' logits, top-K and confidence are exactly the ungated run's. The text is the CTC collapse
of the result: a skipped frame between two frames of one class splits them, as a blank does.

**Batching.** The gate runs once per tile, between the frame row's last layer and the head. A skipped frame leaves
the accumulator, so the head's batches hold only the frames it runs on. They stay full: 16 kept frames per pass
over the head's weights. Each kept frame's line index is kept (2 B per accumulated frame: the 48 B of arena), so the
top-K lands in the right rows and the CTC collapse sees the skipped frames. The output does not depend on the tile or
batch size (tested at 1/1 to 128/128).

**Storage.** The gate is a `VOCG` section after the unchanged version-1 blob, at its 16-byte aligned end. It has its
own version and CRC-32, and it is bound to the blob's byte count and CRC (`train/headgate.py` documents and writes
the layout).

- A runtime without gate support reads a gated blob as the plain version-1 blob. A blob without the section runs as
  before.
- `vocr_opts.head_gate` is `VOCR_HEAD_GATE_OFF` by default: today's output, arena and code path. The gate-off
  instruction count on the Vita build is 0.13-0.34% below main's (the top-K rows' offsets are now computed once per
  batch, not per class block).
- `VOCR_HEAD_GATE_ON` needs W8I and a gated blob, else `VOCR_E_MODE`. `VOCR_HEAD_GATE_SCORE` computes the scores
  only. A run that asks for the logits, or sets `vocr_extra.ungated`, skips nothing.
- API and checks: [runtime/README.md](../runtime/README.md), "Skipping the output head on blank frames".

**Training (the protocol's "training").** The data are the console-cut crops of VOCR5's own fine-tune pools, as
stored, in console polarity:

- **Rows:** 11,245 GRAD crops and their refined copies of owner, native and devnative training lines, 232 VNs.
  None is a test or `dev_sample` VN (asserted).
- **Features:** the head's input (`vocr_extra.head_in`) from the C runtime in W8I.
- **Label:** whether the W8I argmax is the blank.
- **Split by VN:** sha1(vn) % 5 = 0 calibrates, 55 VNs and 154,263 frames. The other 177 VNs (390,374 frames) fit.
- **Fit:** one per tier. Standardised inputs, BCE, Adam 3e-3, cosine schedule over 12 epochs, batch 4,096, seed 0. The
  standardisation is folded into layer 1, which is then quantised to int8.
- **Threshold:** on the exact int8 scores (the twin, equal to the C runtime bit for bit on every calibration line). *t*
  is the (⌊εn⌋ + 1)-th largest score of the n calibration frames whose argmax is not the blank, with ε = 1e-3.

| tier | calibration AUC (int8) | *t* | calibration frames skipped | non-blank frames skipped | blank frames skipped | fallback ε = 1e-4: *t*, skipped |
|---|---:|---:|---:|---:|---:|---:|
| S | 0.99958 | 3.4814 | 61.3% | 54 of 54,038 (0.100%) | 94.3% | 6.0421, 51.5% |
| N | 0.99954 | 3.8066 | 60.2% | 54 of 54,039 (0.100%) | 92.7% | 5.9600, 50.2% |
| L | 0.99922 | 3.9438 | 57.6% | 55 of 55,030 (0.100%) | 89.5% | 5.9148, 47.3% |

The fallback blobs (`*_w8g_eps0.0001.vocr`) were written with the same fit, as the protocol pre-registers. No tier
needed them, so none was scored. The records are `eval/headgate_fit_20261023.json`.

## Development (the pre-registered rule)

**Data.** VOCR5's development cohorts exactly, in the C runtime in W8I, gated against today (`eval/headgate_eval.py`):

- **tv:** teacher_val's development lines.
- **dn:** devnative's four development files.
- **wd:** the whole-dialogue regions through the C GRAD finder and the refinement at three region sizes.
- tv and dn are read in both views (console polarity as stored, and screen polarity).

**Score and rule.** The development score is VOCR5's: 0.25 tv + 0.25 dn + 0.5 wd, dialogue only. The difference
gated − today has `eval/vocr5_select.py`'s VN-cluster bootstrap. The rule adopts a tier iff, for each decode:

- (a) neither the score interval nor any component's lies entirely above 0;
- (b) the score's upper bound is at most +0.05 points;
- (c) the estimated saving is at least 5% of the nominal box.

**Decodes:**

- **greedy:** `vocr_run`'s text.
- **rubi:** Rubi's line path: the counter re-decode {6000, 4000}, the beam (8, top-8) with the 1 MB LM (α 0.5, β 3.5,
  gate 8, end of line, bracket 8), a line read as empty staying empty, then the counter text rule.
- **rubi + cascade:** a line whose confidence is under 0.700 is read again by the next tier up, gated too.

**Checks.** Every tenth read was checked two ways (824 per tier):

- the gated blob with the gate off equals today bit for bit;
- the gated read equals `train/headgate.simulate` applied to today's read.

| tier | decode | dev score today → gated | gated − today, points [95% CI] | tv | dn | wd | rows changed (fixed) / lines (wd) |
|---|---|---:|---:|---:|---:|---:|---:|
| S | greedy | 3.7821 → 3.7980 | +0.016 [0.000, +0.033] | 1.363 → 1.387% | 1.110 → 1.110% | 6.328 → 6.347% | 10 / 31 |
| S | rubi | 3.4149 → 3.4068 | −0.008 [−0.063, +0.035] | 0.819 → 0.825% | 0.978 → 0.924% | 5.931 → 5.939% | 17 / 50 |
| S | rubi + cascade | 3.0831 → 3.0964 | +0.013 [−0.022, +0.050] | 0.694 → 0.713% | 0.762 → 0.750% | 5.438 → 5.461% | 21 / 67 |
| N | greedy | 3.3420 → 3.3571 | +0.015 [−0.001, +0.037] | 0.737 → 0.775% | 0.984 → 0.984% | 5.823 → 5.835% | 9 / 26 |
| N | rubi | 3.0839 → 3.0746 | −0.009 [−0.040, +0.023] | 0.675 → 0.650% | 0.672 → 0.648% | 5.494 → 5.500% | 16 / 69 |
| N | rubi + cascade | 2.8777 → 2.8659 | −0.012 [−0.032, +0.008] | 0.506 → 0.494% | 0.600 → 0.600% | 5.202 → 5.185% | 8 / 39 |
| L | greedy | 2.9589 → 2.9634 | +0.005 [−0.003, +0.014] | 0.494 → 0.506% | 0.678 → 0.672% | 5.332 → 5.338% | 3 / 13 |
| L | rubi | 2.8337 → 2.8320 | −0.002 [−0.017, +0.012] | 0.462 → 0.475% | 0.576 → 0.576% | 5.148 → 5.138% | 4 / 29 |
| L | rubi + cascade | 2.8337 → 2.8320 | −0.002 [−0.017, +0.012] | 0.462 → 0.475% | 0.576 → 0.576% | 5.148 → 5.138% | 4 / 29 |

Large has no larger tier, so its cascade is its rubi. Every row meets (a) and (b). (c) is met by all three tiers
(19.5% / 14.9% / 7.9% of the nominal box), so the verdict is **adopted** for S, N and L. Normal and Large have room on
both conditions. Small passes at the rule's edges:

- (a), greedy: the score interval's lower bound is exactly 0.000 (2.5% of the replicates fall below 0, 0.3% at 0), so
  "not entirely above 0" holds with no slack;
- (b), cascade: the upper bound is +0.050, the margin itself (+0.0504 before `vocr5_select.compare` rounds the interval
  to 3 decimals; the rule reads the rounded interval, as VOCR5's own rule does).

**Correction (after audit-head-gate, 2026-10-23).** The first record of this verdict (`a95994f`) keyed each fixed read
by (cohort, file, line id, view). teacher_val holds 129 PC lines twice, scaled once to Vita size and once to PSP size,
with the same id and different crops, so both rows of each pair were scored with the PSP-scaled crop's reading and
tv's component was not VOCR5's. The score now keys each read by its row (`eval/headgate_eval.py` `row_keys`), and
today's greedy scores equal VOCR5's recorded scores of the same blobs to the last digit (3.7821 / 3.3420 / 2.9589;
`tests/test_headgate.py` checks it). The corrected record was re-scored from the stored reads, after the protected
read; the key is its only change, and the rule's code, the reads, the fits and the protected read are unchanged. The
first record gave every upper bound at most +0.046 points and the same verdict (S greedy +0.013 [−0.002, +0.029], S
cascade +0.007 [−0.030, +0.046], N greedy +0.018 [−0.001, +0.046]); its dn, wd, skip-rate, LM-candidate and counter
numbers are the corrected record's. The column of changed fixed lines now counts rows (two reads each), not reads.

**Names** (secondary). Name CER moves by −0.31 to +0.16 points across tiers, decodes and the tv / dn name lines
(`eval/headgate_dev_20261023.json` `names_cer_pct`).

**Disclosure.** text-stats-gate chose this gate's design (MLP, blank-argmax target, ε = 1e-3) after seeing tv and dn
with a float gate, so tv and dn here are a confirmation on the same data. wd, the cascade, the counter set, the int8
gate and the runtime had not been scored by anyone.

**Skip rates** (gated arm, share of frames whose head was skipped):

| tier | all reads | dialogue | finder-cut lines (wd) | wd GRAD only | wd refined | dn GRAD crops | Vita / PSP / PC fixed dialogue | names | non-blank frames skipped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S | 61.9% | 61.6% | 61.4% | 57.3% | 64.9% | 57.3% | 63.5 / 62.2 / 59.5% | 67.2% | 0.18% |
| N | 61.4% | 61.1% | 61.0% | 57.1% | 64.2% | 56.8% | 62.6 / 61.6 / 59.1% | 68.4% | 0.16% |
| L | 58.5% | 58.2% | 57.9% | 54.4% | 60.9% | 53.9% | 60.8 / 58.6 / 57.1% | 64.0% | 0.17% |

The gate also trims the LM decoder's work. Its candidate rule leaves no non-blank candidate on more frames:
blank-only frames go from 53.6% / 51.7% / 49.4% to 62.6% / 62.5% / 60.2%. Non-blank candidates per frame drop from
0.84 / 0.80 / 0.76 to 0.67 / 0.61 / 0.59. These are counts only; the decoder's time was not measured.

## Counters: the re-decode needs an ungated read

Rubi's counter re-decode (#109, `vocr_post_redecode`) looks at the top-K of up to four blank frames beside a numeral
it suspects, and of the blank frames before a line's first character, for a cut 「. A skipped frame offers only the
blank there.

The development cohorts hold no counter context in any gated read (0 per tier), so the table above is the same with
or without a remedy. The protocol's check is #109's synthetic counter-context set: 3,969 crops of development-pool
texts, many cut into the line (`eval/kanum_dev.py`). There, gated reads do lose re-decode edits.

The remedy, pre-registered for this case, is to read a line again without the gate (`vocr_extra.ungated = 1`) when
the re-decode finds a counter context in its gated top-K. That re-read recovers the re-decode's edits wherever the
gated read still shows the context (655 / 645 / 613 lines of the set). Lines whose context the gate itself removed stay
worse:

| tier | decode | CER today → gated (all 3,969 crops) | lines worse / better | cut +3 px (the finder's margin) | cut 0 | cut −2 px | cut −4 px | cut −6 px | cut −8 px | cut −10 px |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S | greedy | 4.81 → 4.82% | 5 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1 / 0 | 0 / 0 | 4 / 0 | 0 / 0 |
| S | rubi | 3.45 → 3.47% | 14 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 2 / 0 | 5 / 0 | 7 / 0 | 0 / 0 |
| S | rubi + re-read | 3.45 → 3.47% | 14 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 2 / 0 | 5 / 0 | 7 / 0 | 0 / 0 |
| N | greedy | 5.18 → 5.23% | 27 / 3 | 0 / 0 | 0 / 2 | 6 / 0 | 7 / 0 | 8 / 1 | 6 / 0 | 0 / 0 |
| N | rubi | 3.60 → 3.78% | 96 / 5 | 0 / 0 | 0 / 2 | 9 / 0 | 32 / 0 | 24 / 2 | 25 / 1 | 6 / 0 |
| N | rubi + re-read | 3.60 → 3.74% | 76 / 5 | 0 / 0 | 0 / 2 | 5 / 0 | 19 / 0 | 21 / 2 | 25 / 1 | 6 / 0 |
| L | greedy | 4.29 → 4.30% | 5 / 1 | 0 / 0 | 1 / 0 | 1 / 1 | 0 / 0 | 2 / 0 | 1 / 0 | 0 / 0 |
| L | rubi | 2.98 → 3.03% | 32 / 4 | 0 / 0 | 2 / 1 | 0 / 2 | 7 / 1 | 10 / 0 | 10 / 0 | 3 / 0 |
| L | rubi + re-read | 2.98 → 3.00% | 18 / 4 | 0 / 0 | 2 / 1 | 0 / 2 | 3 / 1 | 6 / 0 | 6 / 0 | 1 / 0 |

(Lines worse / better, gated against today; `eval/headgate_counters_20261023.json`.)

On real text the re-read costs one more recognizer run for a line with a counter context. There were 0 such lines in
8,233 development reads per tier, and the re-read changes no line of the protected sets.

The losses that remain are on lines cut 2-10 px into their first character, a region drawn into the text: the gate
skips a few low-confidence frames at the cut. Lines with the finder's 3 px margin lose nothing, and lines cut at the
ink edge lose at most 2. That is the case #109's edge continuation addresses, and it is not adopted as a default
there either. A caller that expects such regions can keep the gate off.

## Time (estimates)

**Measurement.** `vita/a9_layers.py` counted per-phase instructions under qemu-arm of the VitaSDK `-O2` NEON build,
with its cache model (`eval/reproducibility/headgate-20261023/a9.sh`):

- **Lines:** ocr-speed's 24 development lines of 60-100 frames (1,609 frames), top-8.
- **Configurations, per tier:** main's runtime, and this one with the gate off, score-only and on.
- **Basis:** without prefetch credit (the safe basis the tiers are sized on) and with it.
- **Calibration:** ×1.000676 to VOCR3-H15's measured console rate, at an assumed 333 MHz. 444 MHz is by the clock
  ratio.

**Projection.** The gate skips 54.8% / 57.1% / 48.9% of these lines' frames. The saving is projected to the
development share on finder-cut lines (61.4% / 61.0% / 57.9%) as:

- the head's measured cost × the share skipped, scaled by the measured batch efficiency (1.03-1.05: a gated batch of
  16 kept frames costs a few percent more per frame than an ungated one);
- minus the gate's measured cost (0.005-0.009 ms a frame);
- minus each skipped frame's blank logit (0.004-0.007 ms).

| tier | ms per frame: main / gate off / gate on (these lines) | head: off → on | gate | est. saving per frame at the dev share | nominal 3-line box (240 frames + 0.1 s) at 333 MHz | at 444 MHz | real box: Vita 89 / PSP 93 / Adrenaline 95 frames | with prefetch credit, nominal box |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S | 2.197 / 2.192 / 1.743 | 0.867 → 0.410 | 0.007 | 0.510 ms | 626 → 504 ms (−122.3 ms, −19.5%) | −91.7 ms | −45.4 / −47.4 / −48.4 ms | −110.3 ms |
| N | 3.413 / 3.407 / 2.876 | 0.978 → 0.439 | 0.007 | 0.572 ms | 918 → 780 ms (−137.2 ms, −14.9%) | −102.9 ms | −50.9 / −53.1 / −54.3 ms | −123.4 ms |
| L | 8.222 / 8.217 / 7.309 | 1.225 → 0.649 | 0.012 | 0.678 ms | 2,072 → 1,909 ms (−162.7 ms, −7.9%) | −122.0 ms | −60.3 / −63.1 / −64.4 ms | −119.0 ms |

text-stats-gate's float gate was estimated at −116 / −123 / −151 ms. Its estimate assumed 54.5% / 51.9% / 50.8% of
frames skipped; the int8 gate skips more.

In instructions per frame (gate off → on, these lines' skip shares):

- **Head:** 239,827 → 111,764 (S), 270,807 → 120,138 (N), 270,878 → 143,394 (L).
- **Gate:** 1,573-1,626.
- **Main against gate off:** the gate-off build is 1,782-2,011 instructions a frame below main (0.34 / 0.25 / 0.13%).

Large's measured saving on these lines (0.908 ms a frame at their 48.9% skip share) is larger than its projection at
the development share (0.678 ms at 57.9%). With the gate on, Large's other phases also get cheaper in the cache model
(6.992 → 6.648 ms a frame: the head's weights are streamed less often and evict less), and the projection counts only
the head and the gate. Small's and Normal's other phases do not move. Large's estimate is therefore conservative.

The measurements are of the committed runtime (`vocr.c` `ad1bfa89…`, recorded in `eval/headgate_a9_20261023.json`),
re-measured after audit-head-gate's fixes; the first ones (`3f05dee2…`) gave the same savings at the reported
precision.

In a real job the recognizer is about a fifth of the latency (text-stats). The saving is per CTC frame, so a real box
gains about 0.37 of the nominal figure. Nothing of this has run on a console.

## Sizes

| | S | N | L |
|---|---:|---:|---:|
| blob: today → gated (+1,792 B section) | 819,792 → 821,584 B | 1,185,632 → 1,187,424 B | 2,473,696 → 2,475,488 B |
| W8I arena, default tile 4 / batch 16: off → on | 124,832 → 124,880 B | 182,944 → 182,992 B | 261,024 → 261,072 B |

- **Arena.** With the gate on it grows by `al16(2 × (batch + tile − 1))` bytes. The gate's scratch (a tile of
  frames' int16 input, scales and hidden units) fits inside the head's.
- **Code.** `vocr.o` grows from 18,219 to 20,159 B (VitaSDK gcc 15.2, Cortex-A9 NEON, `-O2`, `make -C runtime size`).
  The scalar build grows from 13,403 to 15,159 B, and psp-gcc `-O2 -G0` (`make -C psp size`) from 19,795 to 22,959 B.
- **Stack.** The deepest `vocr_run` → `run_head` → `mm_i16` chain grows from 2,392 to 2,528 B (`-fstack-usage`).
- **Rubi's weights block** (page-rounded). Large's crosses a page: 604 → 605 pages, +4,096 B. Small and Normal stay
  at 201 and 290 pages.

## Protected sets

Read once, after the verdict was pushed (`a95994f`, 2026-10-23T16:00:57Z; attempt records from
2026-10-23T16:04:24Z), by `eval/headgate_protected.py`. This is report only: it decides and tunes nothing. The public
records hold counts, rates and hashes (`eval/headgate_protected_20261023.json`, `eval/headgate_clannad_20261023.json`).

**vita-test + psp-test** fixed lines: 336 dialogue lines in console polarity, pooled. Dialogue CER today → gated, with
lines changed and gated − today in points [95% VN-cluster CI]:

| tier | greedy | rubi | rubi + cascade | frames skipped |
|---|---:|---:|---:|---:|
| S | 1.598 → 1.598% (0 lines) | 1.151 → 1.151% (0 lines) | 0.810 → 0.788% (1 line; −0.021 [−0.069, 0.000]) | 62.5% |
| N | 1.129 → 1.129% (0 lines) | 0.725 → 0.682% (2 lines; −0.043 [−0.110, 0.000]) | 0.618 → 0.575% (2 lines; −0.043 [−0.110, 0.000]) | 61.4% |
| L | 0.788 → 0.788% (0 lines) | 0.533 → 0.511% (1 line; −0.021 [−0.069, 0.000]) | 0.533 → 0.511% (1 line; −0.021 [−0.069, 0.000]) | 58.7% |

**CLANNAD**: 3,048 reference lines on 2,177 pages, of which 2,052 are dialogue, each read through GRAD + refine inside
its GRAD box. Dialogue CER today → gated, with lines changed and gated − today in points [95% page CI]:

| tier | greedy | rubi | rubi + cascade | frames skipped |
|---|---:|---:|---:|---:|
| S | 1.433 → 1.431% (1 line; −0.003 [−0.009, 0.000]) | 0.959 → 0.956% (3 lines; −0.003 [−0.012, +0.006]) | 0.839 → 0.842% (3 lines; +0.003 [−0.006, +0.015]) | 65.7% |
| N | 0.923 → 0.923% (0 lines) | 0.764 → 0.767% (1 line; +0.003 [0.000, +0.009]) | 0.654 → 0.660% (2 lines; +0.006 [0.000, +0.015]) | 65.1% |
| L | 0.630 → 0.630% (0 lines) | 0.546 → 0.549% (1 line; +0.003 [0.000, +0.009]) | 0.546 → 0.549% (1 line; +0.003 [0.000, +0.009]) | 60.7% |

Greedy text is unchanged on every vita-test and psp-test dialogue line for all three tiers, and on every CLANNAD
dialogue line for N and L (one line changes for S). Rubi's path changes 0-3 lines per set and tier. Every interval
lies within ±0.15 points.

Over all CLANNAD lines (names included, a secondary column) the gate lowers CER:

| tier | greedy | rubi | rubi + cascade |
|---|---:|---:|---:|
| S | 2.45 → 2.43% | 2.72 → 2.68% | 1.98 → 1.68% |
| N | 1.46 → 1.36% | 2.38 → 1.80% | 2.11 → 1.70% |
| L | 0.90 → 0.90% | 2.07 → 1.80% | 2.07 → 1.80% |

On CLANNAD's name lines Rubi's beam + LM scores worse than greedy, and gated blank frames leave it fewer non-blank
candidates. That is the measured effect; its cause was not analysed further.

Nothing above changes the verdict, which was fixed on development data.

## Checks

**C runtime** (`runtime/tests/test_vocr.c`; gcc and clang with ASan and UBSan, and Cortex-A9 NEON and scalar builds
under qemu-arm):

- the section's validation: version, flags, the head's K, the hidden size, bounds, alignment, a NaN threshold, the
  binding to the base blob, and the CRC;
- gate off equals the blob without the section, arena included;
- score-only scores equal an independent reference of the arithmetic;
- the gated run equals the reference's skips, at six tilings and K = 0 / 1 / 3;
- *t* = +∞ skips nothing and *t* = −∞ skips every frame;
- `ungated` and asking for the logits each give today's read;
- all of it again on a blob whose head follows the stem directly (no frame-row layer between them, so the front's
  last layer appends the frames): gate off gives main's known answer, gate on the reference's skips
  (audit-head-gate F1: before the fix that blob failed with the gate off and wrote out of bounds with it on; no
  released model has that shape).

**Twin** (`tests/test_headgate.py`): on a random-init model, the C runtime equals `train/headgate.py` bit for bit.
This covers gate scores, skipped frames, top-K, blank logits, best, text, confidence and character count, at six
tilings; and `plan_arena` equals `vocr_arena_bytes` with the gate off, on and score-only.

**Real data.** On the training side, the C scores equal the twin's on all 3,108 calibration lines per tier. On
development, 824 reads per tier were checked against today and the simulation (above).

**Builds.** The Vita test app (`vita/scripts/build.sh`) and the PSP EBOOT build with the change.

## Reproduce

```sh
H=$HERD/shared/artifacts; R=$H/vocr5-train/release/v0.3.0; B=$H/head-gate/blobs
P=$HERD/shared/private/head-gate
for T in S N L; do
  HEAVY_SLOTS=12 $HERD/heavy.sh .venv/bin/python train/headgate.py features --tier $T \
    --blob $R/VOCR5_${T}_w8.vocr --out $P/features
  HEAVY_SLOTS=12 $HERD/heavy.sh .venv/bin/python train/headgate.py fit --tier $T \
    --blob $R/VOCR5_${T}_w8.vocr --features $P/features --out $B --record $B/fit_$T.json
  HEAVY_SLOTS=12 $HERD/heavy.sh .venv/bin/python eval/headgate_eval.py cache --tier $T \
    --base $R/VOCR5_${T}_w8.vocr --gated $B/VOCR5_${T}_w8g.vocr --out $P/dev
done
for T in S N L; do for C in off on score; do for D in 0 1; do        # and main's runtime (a9 "main")
  HEAVY_SLOTS=12 $HERD/heavy.sh eval/reproducibility/headgate-20261023/a9.sh $T $C $D $H/head-gate/a9
done; done; done
.venv/bin/python eval/headgate_eval.py a9 --dir $H/head-gate/a9 --blobs $B --cache $P/dev \
  --out eval/headgate_a9_20261023.json --savings $H/head-gate/savings.json
HEAVY_SLOTS=12 $HERD/heavy.sh .venv/bin/python eval/headgate_eval.py score --cache $P/dev \
  --blob S=$R/VOCR5_S_w8.vocr --blob N=$R/VOCR5_N_w8.vocr --blob L=$R/VOCR5_L_w8.vocr \
  --gated-blob S=$B/VOCR5_S_w8g.vocr --gated-blob N=$B/VOCR5_N_w8g.vocr --gated-blob L=$B/VOCR5_L_w8g.vocr \
  --lm $R/vocr_lm_o3_1000k.vlm --savings $H/head-gate/savings.json --synth $H/kanum-fix/synth/synth.npz \
  --out eval/headgate_dev_20261023.json
```

## Limits

- **Estimates.** Every console time is an estimate: qemu instruction counts with a cache model, calibrated on one
  model (H15) in one game at an assumed 333 MHz. The gated runtime has not run on a console. Rubi's draft PR, to be
  opened after this merge, will pin these blobs, turn the gate on in its jobs with the counter re-read, and list the
  console test.
- **Training data.** The gate's training frames are 91% the owner's captures, and the frozen model's frames there may
  be more confident than on unseen games. The development check is the guard.
- **Counter set.** The synthetic counter set is a stress test of cut lines, not real text. The re-read remedy is
  needed only where a counter context occurs.
