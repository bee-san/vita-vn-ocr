# vocr6-psp: model B, a recognizer that runs on the PSP

Issue [#126](https://github.com/bee-san/vita-vn-ocr/issues/126) is track T5 of VOCR6 ([#125](https://github.com/bee-san/vita-vn-ocr/issues/125); `plans/vocr6-plan.md` §1b B). Model B was to read PSP visual-novel dialogue on a PSP at native 480×272. It targets the 64 MB models (PSP-2000, PSP-3000, PSP Go); the PSP-1000 is not supported (owner, 2026-10-23).

Every console time here is an estimate. Nothing of model B has run on a PSP console. PPSSPPHeadless 1.20.4 gives emulated time, and the console factors are `psp/README.md`'s.

## Verdict: no model B is released

Under the PSP's time budget, no recognizer reached VOCR5-S's accuracy on PSP games it was not trained on, so the pre-registered rule releases nothing. The budget is an est. ≤ 1 s per nominal 3-line box at 333 MHz on the scalar build; the comparison is VOCR5-S on the same native crops.

- **The fast shapes are worse.** The best arm the feasibility rule calls time-feasible is B24: VOCR5-S's layers at depth 0 and input height 24.
  - The rule estimates 0.93 s per box. On its own development frame mix, measured in PPSSPP, it is 1.21 s, so it is in fact over the budget.
  - On development whole dialogue it is resolved worse than VOCR5-S by **+3.93 points** [+2.63, +5.59] (10.42% against 6.49%), and on development fixed lines by +3.40 [+2.35, +4.75].
  - Every other arm the rule calls feasible is worse still.
- **VOCR5-S's own shape is accurate but slow.** The control S32 is VOCR5-S fine-tuned on the same PSP data at its own shape.
  - It ties VOCR5-S on unseen PSP games: whole dialogue +0.04 [−0.24, +0.31], fixed lines −0.02 [−0.23, +0.16].
  - It is resolved better on the 15 hooked titles' hval: page path −0.73 [−1.32, −0.29], lines −0.75 [−1.43, −0.29].
  - It needs an est. 2.05 s per nominal box on the scalar build (1.40 s with the VFPU), with the head gate and two-level head. It is not time-feasible.
- **What costs accuracy.** Lowering the input height costs the most: from 32 to 24 rows +2.1 points of whole dialogue on top of depth 0 (B32 → B24), and to 18-20 rows +4.6. Dropping the stride-1 DS blocks (depth 0) costs +1.9 at height 32 (B32).
- **The runs stopped early.** Stage 1 stopped after every arm's first seed (protocol amendment 4). No later stage could plausibly close a 3-4 point gap to a target that VOCR5-S's own shape only ties. Stopping can only lead to no release.

**What ships instead:**

- **A runtime change.** The head gate now runs in every mode, and the two-level head in W8F, the PSP's fastest mode. Both are bit-exact with their twins and off by default.
- **The PSP test app's new modes.** Head gate and two-level options, a page path (the line finder, its refinement, then every line), a memory-partition probe and a 64 MB build.
- **The tools.** The native-PSP pools and scorer, the PPSSPP sweep, verification and budget tools, and a console test bundle that holds no game data.
- **The measurements**, below.
- **A console-test pre-release** with two candidate blobs (S32 and B24, gated) and the bundle, so a PSP can measure what PPSSPP cannot. It is not a model release. RESULTS_TBD (issue link)

## Budget and rule (pre-registered: `eval/vocr6_psp_protocol.json`)

- **Memory: at most 4,000,000 B in all.** That covers:
  - the int8 blob with its gate section and VOC2 trailer;
  - the W8F arena at the shipped tile;
  - the code of `vocr.o`, `vocr_lines.o` and `vocr_lines_refine.o` (psp-size);
  - the worker stack;
  - the line finder's arena for a whole 480×272 screen (25,355 B).

  It is an estimate of a real PSP until a 64 MB PSP measures what is free next to a game.
- **Speed: est. at most 1.0 s per nominal PSP box at 333 MHz, scalar build.**
  - The nominal box is 3 lines of 320 input columns at the median native PSP finder crop height, 19 px (`reports/text-height.md`, refine on). That is 3 × ceil(round(320·H/19)/8) CTC frames: 204 at input height 32, 153 at 24, 129 at 20, 114 at 18.
  - The estimate is 1.45 × the PPSSPP-emulated time of the box's recognition plus the finder on a 480×90 region. 1.45 is the upper end of `psp/README.md`'s console range for scalar W8F.
  - The nominal box is long. Real PSP boxes are 93 frames at height 32 on average (`reports/text-stats.md`: owner captures through Rubi's finder; median 75, p90 166).
- **Accuracy.** Release rule (a): on the development sample's native PSP whole-dialogue regions (GRAD + refine at native size), the mean over 3 final seeds of model B − VOCR5-S has a paired VN-cluster 95% interval entirely below 0. Rule (b): native PSP fixed lines are not resolved worse. Rule (c): the budgets. Rule (d): bit-exact in PPSSPP.
- **Amendments,** each committed before the runs it concerns were scored:
  1. The released (gated) form is the one scored, tiling is free, and a synth6 arm runs when synth6 is ready (it was not).
  2. A diagnostic control, S32.
  3. The arms S18 / S18c1, and successive halving.
  4. The futility stop.

## The data (native 480×272 only, never the 2× view)

| source | what | rows | train2 weight |
|---|---|---:|---:|
| hk | the hooked corpus' verified PSP lines, names and furigana-cut lines (vocr6-ingest pools v1, split `train`: hval and guard pages out), ink crops | 26,638 | 0.15 (fa 0.3) |
| hc | console crops of the hooked PSP train pages (`data/modelb_pools.py console`) | 82,162 | 0.15 |
| npi | VOCR5's pinned real pools cut to their native PSP rows (meta `kind == "psp"`), ink crops: tt, tvt, rd, rvt, nat, dntv, dnst, own | 16,703 | 0.14 (fa 0.3) |
| npg | the same pools' native PSP GRAD crops and refined copies | 7,204 | 0.06 |
| nv, s2, s3 | VOCR5-S's synthetic recipe C (nvsynth, synth2, synth3 v2) | streamed | 0.25 / 0.125 / 0.125 |

- **How the console crops are cut.** Every whole-dialogue region of a hooked PSP train page (base, +4 and +8 px) goes through the C line finder at native size (VOCR_LINES_GRAD + vocr_refine_lines). Each finder line is paired with one verified line of the region. A crop is kept only if it holds at least 40% of every non-blank character cell of that line and no other line's row.
- **Why 40% of a cell.** A cell is the glyph's full-width quad, and 「 or 、。 sit in one half of it.
- **The visual checks** (local contact sheets, assistant):
  - 25 random kept crops: every label matches its crop;
  - 16 crops a centre-of-cell rule rejected: 14 lose only half a punctuation cell, 2 are real cuts;
  - 25 crops the 40% rule keeps and the centre rule rejected: every label matches.
- **The hooked titles.** 15 PSP titles with verified lines; Tokimeki Memorial 4 has none. None is a psp-test, vita-test or development-sample VN (`eval/manifests/reserved.json`).
- **Development (unseen games).** The native PSP rows of VOCR5's development cohorts:
  - teacher_val_dev's PSP lines and devnative's four development files' PSP lines, in both views (1,276 dialogue rows, 31 VNs);
  - the whole-dialogue PSP regions of `tv_dlg` / `dn_dlg` (627 units, 55 VNs).
- **hval (screening only).** The hooked PSP titles' hval pages: 1,096 lines and 2,403 regions, 15 titles.

## Speed on the PSP (PPSSPP, emulated)

All shapes are random-init blobs, timed on 3 lines of 19×320 px (`psp/modelb_sweep.py`; `sweep{1..5}/sweep.json`). The kernels' work does not depend on the weights.

| shape | MACs / frame | int8 blob B | scalar W8F (tile 16) ms / frame | VFPU W8F ms / frame | scalar W8I (tile 4) ms / frame | frames per nominal box | scalar W8F s per box (emulated) |
|---|---:|---:|---:|---:|---:|---:|---:|
| VOCR5-S (height 32) | 967,392 | 819,792 | 11.26 | 7.19 | 15.86 | 204 | 2.30 |
| VOCR5-S at height 24 | 908,448 | 819,792 | 10.85 | 7.14 | 14.75 | 153 | 1.66 |
| VOCR5-S at height 24, stride 12 | 957,408 | 819,792 | 12.13 | 8.31 | 15.75 | 102 | 1.24 |
| VOCR5-S at height 24, depth 0 | 745,696 | 756,432 | 9.05 | 6.01 | 12.06 | 153 | 1.39 |
| width 0.5 at height 24 | 599,904 | 629,712 | 7.37 | 4.94 | 9.95 | 153 | 1.13 |

- **W8I is the slowest mode on the PSP.** Allegrex does every integer multiply through HI/LO.
- **The output head is about half the time.** It costs 1.15 µs per class per frame in scalar W8F, so 4,987 classes are about 5.7 ms per frame.

With the head gate and the two-level head (W8F, scalar, tile 16, head batch 64), per frame:

| arm shape | backbone + gate | + level 1 and escape (per kept frame) | + rare rows (per fired frame) | est. console s per nominal box |
|---|---:|---:|---:|---:|
| S32 (VOCR5-S shape) | 5.76 | 0.56 | 4.48 | 1.90 (VFPU: 0.86 s emulated) |
| B32 (depth 0, height 32) | 3.66 | 0.56 | 4.48 | 1.30 |
| B24 (depth 0, height 24) | 3.32 | 0.57 | 4.73 | 0.93 |
| B24c1 (B24, one temporal block) | 2.86 | 0.57 | 4.74 | 0.81 |
| B20 (depth 0, height 20) | 3.18 | 0.58 | 4.78 | 0.75 |
| S18 (VOCR5-S at height 18) | 4.82 | 0.56 | 4.59 | 0.95 |
| S18c1 | 4.38 | 0.56 | 4.59 | 0.88 |

The estimate column is protocol's feasibility rule: the gate skips 0.9 × the arm's development blank share, the escape fires on 20% of the kept frames, the finder costs 0.562 emulated µs per pixel, and the console factor is 1.45×.

## The runtime change (commit 889bd10)

- **The head gate in every mode.** Its own arithmetic is W8I's (int16 head input, int8 layer 1) in every mode. A skipped frame's blank logit is the run's own head arithmetic, computed only when the top-K is kept.
- **The two-level head in W8F.** Level 1 and level 2 are W8F's head rows (dequant8 + mm_f32, as `run_head`); the escape is W8I's. The fired frames' float inputs are packed to the front of the accumulator.
- **W8I runs are unchanged** bit for bit, and both options stay off by default.
- **Arena.** A W8F two-level head needs its escape's int16 input in the head's scratch, and a W8F gate a dequantised first head block (`train/export.plan_arena` mirrors both).
- **Tests.** `tests/test_gate_w8f.py` (19 tests) checks the C runtime against `train/headgate.py`'s and `train/head2level.py`'s twins bit for bit in W8F and F32, at several tilings: gate scores, gated runs, the two-level head, both together, and the arena mirror. `tests/test_headgate.py` and `tests/test_head2level.py` were updated for the wider modes.
- **On the PSP.** In PPSSPP the scalar and VFPU builds return the host's text and logits bit for bit with the W8F gate and two-level head (`smoke_w8f`).

## Results (stage 1, seed 1 of every arm; `eval/vocr6_psp_stage1_20261026.json`)

W8F int8 through the C runtime, greedy, no gate. Each arm is one fine-tune of 10,000 steps from VOCR5-S (`eval/vocr6_psp_run.py`).

| arm | est. s / box (feasibility rule) | SC | dev whole dialogue (GRAD + refine) | dev fixed lines | hval page path | hval lines |
|---|---:|---:|---:|---:|---:|---:|
| VOCR5-S (comparator) | about 3.3 (plain) | 5.73 | 6.49% | 1.17% | 7.64% | 1.26% |
| S32 (control, VOCR5-S's shape) | 1.90 (gated) | 5.37 | 6.52% | 1.15% | 6.91% | 0.52% |
| B32 (depth 0, height 32) | 1.30 | 6.69 | 8.36% | 2.65% | 7.89% | 1.01% |
| **B24** (depth 0, height 24) | **0.93** | 8.69 | 10.42% | 4.58% | 9.88% | 1.73% |
| B24c1 | 0.81 | 9.77 | 11.83% | 6.06% | 10.59% | 2.51% |
| B20 | 0.75 | 11.26 | 12.94% | 7.35% | 12.37% | 3.50% |
| S18 | 0.95 | 11.23 | 12.99% | 7.22% | 12.36% | 3.50% |
| S18c1 | 0.88 | 11.74 | 13.23% | 8.13% | 12.81% | — |

Arm − VOCR5-S, paired 95% intervals (VN clusters on development, title clusters on hval):

| arm | dev whole dialogue | dev fixed lines | hval page path | hval lines |
|---|---|---|---|---|
| S32 | +0.04 [−0.24, +0.31] | −0.02 [−0.23, +0.16] | −0.73 [−1.32, −0.29] | −0.75 [−1.43, −0.29] |
| B32 | +1.87 [+1.14, +2.75] | +1.48 [+0.89, +2.25] | +0.25 [−0.18, +0.79] | −0.26 [−0.76, +0.12] |
| B24 | +3.93 [+2.63, +5.59] | +3.40 [+2.35, +4.75] | +2.24 [+0.92, +3.97] | +0.47 [+0.03, +0.93] |

- **The data helps the titles it comes from.** On the 15 hooked titles, S32 halves VOCR5-S's line errors. It leaves PSP games outside the corpus where VOCR5-S already was.
- **The PSP's speed costs depth and input height,** and both cost far more on unseen games than the data gains on seen ones.

### The head on the two console-test candidates (`eval/vocr6_psp_head.py`; protocol decisions.head)

Each blob's head is permuted into the PSP training labels' frequency order (`data/modelb_pools.py rank`); its 503 most frequent classes cover 93.2% of those labels' characters.

- **The fits.** The gate (96 → 16 → 1) and the escape (98 → 16 → 1) are fit in W8F on 12,000 training-side console crops: the hooked PSP train pages' console crops, and VOCR5's native PSP GRAD and refined crops. Calibration uses a VN/title-hash split.
- **The thresholds.** The gate is set at eps 1e-3 of the non-blank calibration frames; the escape at recall 0.999 of the calibration positives.
- **The twins.** The C runtime equals the twins bit for bit on 400 calibration lines: gate scores, the two-level head, and both together.

| candidate | gated share (dev whole dialogue) | escape share of kept frames | gate + two-level − plain: dev whole dialogue | dev fixed | hval page path | head rule (upper bounds ≤ +0.05) |
|---|---:|---:|---|---|---|---|
| S32 | 64.7% | 27.7% | +0.03 [−0.03, +0.13] | 0.00 [0.00, 0.00] | +0.002 | not kept (the gate alone: +0.007 [−0.03, +0.06], not kept) |
| B24 | 45.5% | 47.2% | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.005 | kept |

### PPSSPP: exactness and time on the real crops (`psp/modelb_verify.py`; `eval/vocr6_psp_verify_20261026.json`)

These runs use the gated two-level blobs in W8F (tile 16, head batch 64). The scalar and VFPU builds read every unique crop of the development and hval plans as fixed lines, and every region of both through the app's finder mode (GRAD + refine, then every line). Each is checked against the host's text and the CRC of its logits.

| candidate | set | items | scalar: same as host | VFPU: same as host | scalar µs / frame | VFPU µs / frame |
|---|---|---:|---|---|---:|---:|
| B24 | dev lines | 1,460 | 1,460 / 1,460 | 1,460 / 1,460 | 5,463 | 3,905 |
| B24 | hval lines | 3,205 | 3,205 / 3,205 | 3,205 / 3,205 | 5,138 | 3,619 |
| B24 | dev regions (page path) | 627 | 627 / 627 | 627 / 627 | 5,218 | 3,689 |
| B24 | hval regions (page path) | 2,403 | 2,403 / 2,403 | 2,403 / 2,403 | 5,118 | 3,619 |
| S32 | dev lines | 1,460 | 1,460 / 1,460 | 1,460 / 1,460 | 6,892 | 4,734 |
| S32 | dev regions (page path) | 627 | 627 / 627 | 627 / 627 | 6,742 | 4,588 |

The budget, estimated on the development whole-dialogue crops' own frame mix (`psp/modelb_verify.py budget`):

| candidate | build | µs / frame (dev whole dialogue) | nominal box, emulated | est. console s per nominal box (× 1.45) | memory in all |
|---|---|---:|---:|---:|---:|
| B24 | scalar | 5,305 | 0.837 s | **1.21** (over the 1.0 s budget) | 1,037,838 B |
| B24 | VFPU | 3,741 | 0.597 s | 0.87 (VFPU unverified on a console) | 1,038,374 B |
| S32 | scalar | 6,795 | 1.411 s | **2.05** | 1,287,278 B |
| S32 | VFPU | 4,620 | 0.968 s | 1.40 (VFPU unverified on a console) | 1,287,814 B |

- **B24 is not time-feasible either.** The feasibility rule assumed the escape fires on 20% of the kept frames; on B24's development crops it fires on 47%. Its scalar estimate is 1.21 s, not 0.93 s.
- **Memory is far inside the budget.** B24 needs 1,037,838 B in all: the blob with its gate and trailer 760,208, arena 192,448, code 55,731 (vocr.o 28,743, vocr_lines.o 19,084, vocr_lines_refine.o 7,904), stack 4,096 (3,744 used) and the finder's arena 25,355. The VFPU build's vocr.o is 29,279 B.

## Reproduce

```sh
# pools, plans, the comparator (artifacts under $HERD/shared/artifacts/vocr6-psp)
data/modelb_pools.py native --out A/pools/native --private-out P/pools/native
data/modelb_pools.py console --pagepool INGEST/pagepool/v1 --pages INGEST/pages --root HOOKED --out A/pools/console
eval/modelb_eval.py dev-plan --out A/eval/dev_plan.pkl
eval/modelb_eval.py hval-plan --pools INGEST/pools/v1 --pagepool INGEST/pagepool/v1 --pages INGEST/pages --out A/eval/hval_plan.pkl
eval/modelb_eval.py read --plan A/eval/dev_plan.pkl --blob VOCR5_S_w8.vocr --mode w8f --out A/eval/dev_VOCR5S_w8f.json
# runs (one heavy slot each), the comparisons, the head
eval/vocr6_psp_run.py B24 1                 # train2 + export + dev/hval reads + experiments.jsonl
eval/modelb_eval.py compare --cand A/runs/mb_B24_seed1/dev.json --base A/eval/dev_VOCR5S_w8f.json --out CMP.json
eval/vocr6_psp_head.py A/runs/mb_B24_seed1  # gate + two-level fit (W8F), reads, the head rule
# PSP
make -C psp [VFPU=1] [LARGEMEM=1]; psp/modelb_sweep.py ...; psp/modelb_verify.py run ...; psp/modelb_bundle.py ...
```
