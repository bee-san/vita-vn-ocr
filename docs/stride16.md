# stride16: the recognizers retrained at a coarser CTC frame step

Issue [#118](https://github.com/bee-san/vita-vn-ocr/issues/118), item 3 of [#117](https://github.com/bee-san/vita-vn-ocr/issues/117). Owner (2026-10-23 13:34Z): "retrain for a coarser frame step".

VOCR5's recognizers emit one CTC frame per 8 input columns. VN dialogue has about 3 frames per character at that step (median 3.14 on Vita, 2.88 on PSP; #117), and CTC needs about 1. stride16 retrains each tier with one frame per 16 columns (Normal also at 12). The recipe, data and vocabularies are VOCR5's. A tier's coarser model is adopted only if it reads development dialogue no worse than VOCR5's.

Every console time below is an estimate. Nothing of stride16 has run on a console.

**Verdict: no tier adopts a coarser frame step. VOCR5-S, VOCR5-N and VOCR5-L stay.**

- **Development.** Every arm is resolved worse than its incumbent on the development score (½ fixed-line, ½ whole-dialogue dialogue CER; mean of 2 seeds, VN-cluster 95% CI):

  | arm | Δ dev score | whole dialogue Δ |
  |---|---|---|
  | S16 | +1.58 [+1.24, +2.00] | +2.12 |
  | N16 | +1.34 [+0.99, +1.73] | +1.82 |
  | L16 | +1.18 [+0.83, +1.60] | +1.75 |
  | N12 | +0.40 [+0.27, +0.54] | +0.55 [+0.32, +0.80] |

- **Not only the lines short of frames.** On rows with frames to spare (≥ 1.25 per character + repeat), the stride-16 models still lose 0.2-0.8 points. PSP loses more than Vita.
- **The protected sets, read once afterwards, agree** (report only).
  - vita-test + psp-test dialogue: S16 +1.02, N16 +0.66, L16 +0.64 points, all VN intervals above 0; N12 +0.21 (−0.27 to +0.62).
  - CLANNAD dialogue: +0.13 to +0.27 points.
  - The incumbents reproduce VOCR5's locked numbers exactly, so the changed runtime reads every stride-8 model bit for bit as before.
- **The saving would have been real** (estimates, nominal 3-line box, 333 MHz, #85 without prefetch credit):

  | tier | s per box | saving |
  |---|---|---:|
  | S | 0.627 → 0.495 | −132 ms |
  | N | 0.919 → 0.741 (N12 0.783) | −178 ms (N12 −136 ms) |
  | L | 2.073 → 1.598 | −475 ms |

  The weights are the same and the arenas are smaller, but the accuracy cost is resolved at every tier, so nothing ships.
- **What stays.** The code that trains, exports and runs a stride-12 or stride-16 recognizer stays:
  - `train2.py --stride`;
  - the stride-aware export;
  - op stride 3 and a stride-aware default tile in `runtime/vocr.c`, where stride-8 blobs run bit for bit as before.
- **Nothing to publish or pin.** There is no pre-release and no Rubi pin. head-gate (#120, merged) remains the speed path for VOCR5, and its gate composes with no stride16 model because none was adopted.

## Protocol (frozen before any run: `eval/stride16_protocol.json`)

### Arms

Each arm is VOCR5's tier shape with one downsampling stride changed (`train/model.py` `stage_strides`). No parameter changes, so a stride-16 model has a stride-8 model's `state_dict` and its exact weight bytes.

| arm | tier | stride | change | vocab | recipe | pretrain steps | seeds |
|---|---|---:|---|---|---|---:|---|
| S16 | Small | 16 | `ds22`: the 2-row stage's DS block at stride (2, 2) | `data/vocab_4987_proof.txt` | C | 216,000 | 1, 2 |
| N16 | Normal | 16 | `ds22` | `data/vocab_vn_5631.txt` | M | 150,000 | 1, 2 |
| L16 | Large | 16 | `ds22` (late-wide `--chans 64,128,256,480,576`) | `data/vocab_vn_5631.txt` | M | 108,000 | 1, 2 |
| N12 | Normal | 12 | `ds3w3`: the 4-row stage's DS block at stride (2, 3); width clamp 1,020 px | `data/vocab_vn_5631.txt` | M | 150,000 | 1, 2 |

The incumbents are VOCR5-S, VOCR5-N and VOCR5-L (release v0.3.0; runs `v5f_S_seed2`, `v5f_N_seed3`, `v5f_L_seed2`).

**V16** (VOCR5-V at stride 16) runs only if S16 is adopted, because VOCR5-V started from VOCR5-S's selection. It uses VOCR5-V's recipe (`eval/vertical_protocol.json`) with `--stride 16`, starts from the selected S16 run, and has seeds 1 and 2. It replaces VOCR5-V only if the mean over seeds of V16 − VOCR5-V is not resolved worse on vertical development VN, on manga, or on the columns short of frames.

### Recipe

VOCR5's recipe (`eval/vocr5_protocol.json`), unchanged except for `--stride`:

- **Pretrain from scratch:**
  - batch 96, lr 2e-3, warmup 1,000, cosine to 0.05, wd 0.01, EMA 0.999;
  - synthetic recipe C for Small, M for Normal and Large;
  - the arm's steps.
- **Arm F fine-tune.** VOCR5 adopted this arm at every tier:
  - 15,000 steps at lr 1.2e-3;
  - the synthetic half at half weight;
  - the real half from VOCR5's pinned pools (`eval/vocr5_pools.sha256`, `eval/vocr5_ft_sources.json`), with finder augmentation on the ink sources and the refined copies of every GRAD source.
- **The run** is the final EMA checkpoint.
- **Arm P** is not run, because VOCR5 resolved F better at every tier.

`--stride` sets four things:

- the CTC input lengths (`ceil(w / s)`);
- the filter that drops a training line whose characters plus adjacent repeats exceed its frames (`train/train.py` `ctc_ok`, the same rule as at stride 8);
- the augmentation's minimum width;
- the width clamp: 1,024, or 1,020 at stride 12.

Runs go through `eval/stride16_run.sh TIER STRIDE pre|F SEED` from a detached runner worktree (tag `stride16-runner-<sha>`) and `HEAVY_SLOTS=12 heavy.sh`, at most 8 at once. They share the box with head-gate and head-2level.

**Speed probe** (1,000 steps per configuration, 8 at once, not runs: no checkpoint kept or scored). At VOCR5's step counts the pretrains take an estimated 14-16 h, against VOCR5's 22.5 h, so VOCR5's lengths are kept:

| arm | lines/s (steps 400-1,000) | s per step | pretrain hours at the protocol's steps |
|---|---:|---:|---:|
| S16 | 398-404 | 0.24 | 14.3-14.5 |
| N16 | 266-268 | 0.36 | 14.9-15.0 |
| N12 | 249-251 | 0.38-0.39 | 15.9-16.1 |
| L16 | 189-190 | 0.51 | 15.2 |

VOCR5's probe for comparison: 254 / 179 / 138 lines/s (S / N / L).

### Code

- **`train/model.py`.** `TinyVNOCR(stride=8|12|16)`. Checkpoints record `stride` and `max_width`; a checkpoint without them is stride 8.
- **`train/train2.py`.**
  - `--stride`.
  - `set_geometry()` sets the module's `STRIDE` / `MAX_W`.
  - Teacher frame targets (`kd=`) are refused at strides other than 8.
- **`train/export.py`.**
  - The width clamp defaults to the largest multiple of the stride up to 1,024 (`default_max_width`), and a clamp that is not a multiple of the stride is refused.
  - `plan_arena` mirrors the runtime's new default tile.
- **`runtime/vocr.c`.**
  - Op width strides up to 3 (stride 12).
  - The default tile (`vocr_opts.tile_frames = 0`) is still 4 frames at stride 8. For a stride over 8 it is the whole frames nearest 32 input columns: 3 at stride 12, 2 at stride 16. Rubi's zeroed options therefore give a coarser model a smaller arena, where a 4-frame tile would grow it by 72-144 KB.
  - Every stride-8 blob runs bit for bit as before.
- **Tests.**
  - `tests/test_stride16.py`.
  - `make check` also runs `runtime/tests/parity.py --fixture --stride 16` and `--stride 12`. Each checks bit-exact logits against `vocr_ref`, the PyTorch tolerance, tile invariance, `VOCR_INVERT`, and that the arena equals `plan_arena`.

### Selection (development sample only)

**Cohorts and score.** These are VOCR5's, unchanged:

- **tv:** `teacher_val_dev.npz`, 959 lines, 101 VNs.
- **dn:** devnative's four development files, 765 crops, 34 VNs.
- **wd:** the whole-dialogue regions of `tv_dlg` / `dn_dlg` at base, +4 and +8 px, read through the C GRAD finder with and without the refinement.
- **Score:** 0.25 tv + 0.25 dn + 0.5 wd, in dialogue CER points.
- **Reading:** the C runtime in W8I. Names are a secondary column.

**The CTC floor** (`eval/stride16_floor.py`, model-independent: no recognizer read anything; its output, ids and frame counts only, is `eval/stride16_floor_dev_20261023.json`). Per development row:

- **NR** is the reference's characters plus adjacent repeated characters.
- **T_s** is the row's frames at stride s as the runtime cuts them:
  - for a fixed line, `ceil(q / s)`, with q the runtime's resized width of the stored crop;
  - for a whole-dialogue unit, the sum over the C finder's lines.
- A row is **below the floor** when T_s < NR: no model can read it exactly.
- A row is **tight** when T_s < 1.25 NR.

| rows (dialogue and names) | stride 8: below / tight | stride 12 | stride 16 |
|---|---|---|---|
| fixed lines, PSP (1,081) | 0 / 0 | 1 / 4 | 4 / 62 |
| fixed lines, Vita (643) | 0 / 0 | 0 / 0 | 0 / 31 |
| whole-dialogue units, GRAD (1,515) | 20 / 24 | 31 / 90 | 101 / 362 |
| whole-dialogue units, GRAD + refine (1,515) | 16 / 16 | 21 / 35 | 39 / 112 |

**Amendment 1** (before any candidate score): the file first keyed the fixed rows by line id. teacher_val_dev holds 129 ids that two different crops share, so those rows got another crop's frames. The file now keys a row by its position in the scorer's row order, with the id as a check. That raised the Vita fixed rows from 514 to 643 and the stride-16 tight rows from 19 to 31; nothing else changed. head-gate's audit found the same collision in its scorer.

A unit's ratio is a proxy: a region is read line by line, but its reference is the whole region's. A unit whose finder misses or merges a line also looks short of frames, which is why 36 units are below the floor already at stride 8.

The refinement, which tightens the GRAD crop's rows, leaves a third as many tight units as GRAD alone at stride 16.

**Rule.** Candidate − incumbent is the mean over the arm's two seeds. Every interval comes from a paired, stratified VN-cluster bootstrap with 10,000 resamples:

- tv, dn and wd use seeds 0, 1 and 2, as in VOCR5;
- the fixed-line tight stratum (tv's and dn's tight dialogue lines pooled) uses seed 3;
- the whole-dialogue tight stratum uses seed 4.

An arm is adopted only if all of these hold:

- **(a) Whole dialogue:** the wd interval does not lie entirely above 0 (not resolved worse).
- **(b) Fixed lines:** neither the tv nor the dn interval lies entirely above 0.
- **(c) The lines below the CTC floor are handled:**
  - neither tight stratum's interval, at the arm's stride, lies entirely above 0;
  - a stratum of fewer than 5 VN clusters cannot be resolved; it passes and is flagged;
  - the floor rows' counts and CERs are reported.
- **(d) Limits:** both runs meet the tier's limits:
  - weights equal to the incumbent's bytes;
  - a W8I arena at the default tile no larger than the incumbent's;
  - a box estimate (#85 without prefetch credit, 333 and 444 MHz) under the tier's ceiling: Small < 1 s, Normal < 2 s, Large < 3 s.

Normal decides N16 first. N12 is decided only if N16 is not adopted, but both are trained and reported.

The adopted arm's run with the lowest development score is selected; ties at 5 decimals go to the lower seed. Otherwise VOCR5 stays.

Each decision and each selection is committed before the locked report.

### Estimates of the candidate shapes

Method (VOCR5's):

- **Measurement.** `vita/a9_layers.py` (ocr-speed 8d21908) on random-init blobs of the exact shapes, over ocr-speed's 24 development lines of 60-100 stride-8 frames, with `--topk 8`.
- **Calibration.** Scaled by 1.000676 to VOCR3-H15's console rate: 4,941.7 µs per frame at an assumed 333 MHz, the owner's 5.0-5.2 ms per CTC frame.
- **Clocks.** 444 MHz comes from a9_layers' own analysis at that clock.
- **Box.** A nominal box is 240 stride-8 frames plus 0.1 s for the finder. A real box is 89 stride-8 frames (the owner's Vita captures through Rubi's finder) and counts recognition only.
- **Stride 12** runs on copies of the same runtimes with op strides up to 3.
- **Tile:** the runtime's default.

| model | weights | W8I arena (default tile) | box s, #85 no prefetch credit, 333 / 444 MHz | box s, main before #85, 333 / 444 | real box ms (333, no PLD) |
|---|---:|---:|---|---|---:|
| VOCR5-S | 819,792 | 124,832 | 0.631 / 0.499 | 0.940 / 0.730 | 197 |
| **S16** | 819,792 | 116,384 | **0.497 / 0.398** | 0.747 / 0.586 | 147 |
| VOCR5-N | 1,185,632 | 182,944 | 0.921 / 0.716 | 1.339 / 1.029 | 305 |
| **N16** | 1,185,632 | 170,656 | **0.743 / 0.583** | 1.123 / 0.867 | 239 |
| **N12** (tile 3) | 1,185,632 | 176,832 | **0.786 / 0.615** | 1.265 / 0.974 | 254 |
| VOCR5-L | 2,473,696 | 261,024 | 2.077 / 1.627 | 2.680 / 2.079 | 733 |
| **L16** | 2,473,696 | 239,264 | **1.600 / 1.251** | 2.377 / 1.834 | 556 |

At its tile of 2 frames, N12 would be 0.824 / 0.643 s with a 133,728 B arena. That is why the default tile at stride 12 is 3.

The protocol's `estimates.table` adds #85 with prefetch credit. `$HERD/shared/artifacts/stride16/est/` holds every measurement directory and `price.py`.

### Locked report (once, after the selections)

Read once:

- **Fixed lines:** vita-test and psp-test through the scalar C W8I runtime, console polarity primary, dialogue primary.
- **Whole dialogue:** the same sets' regions at base, +4 and +8 px.
- **CLANNAD:** the frozen references, read once through `eval/clannad_refs.py`'s extra-arms path.

Arms and pairs:

- **Arms:** the incumbents, and each arm's best run by development score. Arms that were not adopted are report only.
- **Pairs:** candidate − incumbent per tier.
- **Deployed decode:** beam 8 + the 1 MB LM, report only.

The report decides and tunes nothing.

If head-gate's gate (#117 item 1) is on main by then, it is trained on each adopted model with head-gate's own recipe, and the combination is reported.

### Hygiene

- Nothing is chosen, tuned or dropped on vita-test, psp-test or CLANNAD.
- The training data is VOCR5's pinned data. No dev-sample or test VN trains.
- In-run evaluations are logged only, and the final checkpoint is the run.
- Not stride16 runs:
  - two 20-step `train2` smokes (stride 16 and 12, random init, 13:50-13:58Z, deleted; their in-run read of 50 `teacher_val_dev` lines was logged and not used);
  - the 1,000-step speed probes, whose checkpoints were deleted and which had no development evaluation.

## Results

### Runs

All 16 runs finished; none failed. They were 8 pretrains (2026-10-23 14:43Z to 2026-10-24 08:20Z) and 8 arm-F fine-tunes (06:22-09:54Z), every one through `heavy.sh` from the runner `stride16-runner-eeb91d1`.

The two Small pretrains were stopped right after their step-24,000 checkpoints (16:33 and 16:40Z), to give heavy slots back to head-gate and head-2level. They resumed from `ckpt_last.pt` at 18:00Z. The protocol allows this ("fewer [runs at once] if head-gate or head-2level need slots"; a pretrain resumes from `ckpt_last.pt` after a checkpoint, and train2's resume reseeds the batch stream).

**The CTC floor at training time.** `train2` drops a line whose label does not fit its frames. On the fine-tune's real sources it dropped:

| stride | dropped lines | share | where |
|---:|---:|---:|---|
| 8 (VOCR5's logs) | 2 of 442,351 | | |
| 12 | 249 | 0.06% | |
| 16 | 7,309 | 1.65% | realplus v3's GRAD crops and their refined copies 5,460; the owner's ink, m4 and m8 crops 1,245 |

On a sample of the synthetic shards, stride 16 drops 0.6% (nvsynth) to 3.1% (synth3 v2) of the lines, and stride 12 at most 0.19%.

### Development (selection)

Mean over the two seeds of candidate − incumbent, in dialogue CER points. Every interval is a stratified VN-cluster bootstrap (10,000 resamples). The decision files are `eval/stride16_decision_{S16,N16,N12,L16}_20261024.json`, each committed before its tier's selection.

| arm | dev score Δ [95% CI] | tv Δ | dn Δ | wd Δ | fixed-line tight rows Δ (rows, VNs) | whole-dialogue tight rows Δ (rows, VNs) | adopted |
|---|---|---|---|---|---|---|---|
| S16 | **+1.583 [+1.241, +1.997]** | +0.766 [+0.331, +1.217] | +1.323 [+0.841, +1.938] | +2.122 [+1.520, +2.839] | +5.70 [+3.79, +7.75] (87, 37) | +8.83 [+6.77, +11.30] (398, 52) | no |
| N16 | **+1.336 [+0.994, +1.728]** | +0.859 [+0.494, +1.264] | +0.837 [+0.421, +1.412] | +1.824 [+1.210, +2.538] | +4.21 [+2.41, +6.26] (87, 37) | +8.44 [+6.57, +10.32] (398, 52) | no |
| N12 | **+0.397 [+0.267, +0.539]** | +0.306 [+0.153, +0.467] | +0.186 [−0.003, +0.401] | +0.547 [+0.319, +0.802] | +2.44 (4, 4: not resolvable, flagged) | +2.93 [+0.93, +6.24] (110, 21) | no |
| L16 | **+1.181 [+0.832, +1.596]** | +0.559 [+0.199, +1.034] | +0.669 [+0.272, +1.236] | +1.747 [+1.116, +2.511] | +3.99 [+2.13, +6.05] (87, 37) | +8.45 [+6.37, +10.42] (398, 52) | no |

- **S16, N16 and L16** fail every part of the rule except the limits: whole dialogue, both fixed-line components and both tight strata are resolved worse.
- **N12** (decided because N16 failed) fails on whole dialogue, on tv and on the whole-dialogue tight stratum. Its dn interval just reaches 0.
- **The selections** (`eval/stride16_selection_{Small,Normal,Large}_20261024.json`) keep VOCR5-S, VOCR5-N and VOCR5-L.

Every run (`eval/stride16_scores_{Small,Normal,Large}_20261024.json`; dialogue CER, names a secondary column):

| tier | run | stride | dev score | tv | dn | wd | names tv / dn | weights | W8I arena (default tile) |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| Small | VOCR5-S (incumbent) | 8 | 3.782 | 1.36% | 1.11% | 6.33% | 5.86% / 10.25% | 819,792 | 124,832 |
| Small | `s16f_S_seed1` | 16 | 5.356 | 2.19% | 2.47% | 8.38% | 7.84% / 14.08% | 819,792 | 116,384 |
| Small | `s16f_S_seed2` | 16 | 5.375 | 2.06% | 2.40% | 8.52% | 7.73% / 15.10% | 819,792 | 116,384 |
| Normal | VOCR5-N (incumbent) | 8 | 3.342 | 0.74% | 0.98% | 5.82% | 4.17% / 7.20% | 1,185,632 | 182,944 |
| Normal | `s12f_N_seed2` | 12 | 3.721 | 1.14% | 1.12% | 6.31% | 4.45% / 8.61% | 1,185,632 | 176,832 |
| Normal | `s12f_N_seed1` | 12 | 3.757 | 0.95% | 1.22% | 6.43% | 5.47% / 9.15% | 1,185,632 | 176,832 |
| Normal | `s16f_N_seed2` | 16 | 4.634 | 1.56% | 1.81% | 7.58% | 5.83% / 11.19% | 1,185,632 | 170,656 |
| Normal | `s16f_N_seed1` | 16 | 4.722 | 1.64% | 1.83% | 7.71% | 4.94% / 9.55% | 1,185,632 | 170,656 |
| Large | VOCR5-L (incumbent) | 8 | 2.959 | 0.49% | 0.68% | 5.33% | 3.71% / 5.48% | 2,473,696 | 261,024 |
| Large | `s16f_L_seed2` | 16 | 4.116 | 1.14% | 1.30% | 7.01% | 4.31% / 8.22% | 2,473,696 | 239,264 |
| Large | `s16f_L_seed1` | 16 | 4.164 | 0.96% | 1.39% | 7.15% | 3.85% / 8.61% | 2,473,696 | 239,264 |

The incumbents' scores, read by this round's runner, equal VOCR5's records to the fifth decimal (3.78209 / 3.34196 / 2.95890).

**Where the loss sits** (report only, the same rows; incumbent → arm, mean of the seeds):

| arm | component | rows with room (≥ 1.25 frames per character + repeat) | tight rows | PSP Δ | Vita Δ |
|---|---|---|---|---:|---:|
| S16 | tv / dn / wd | 1.07 → 1.56% / 0.85 → 1.61% / 1.95 → 2.72% | 4.85 → 8.90% / 3.78 → 10.86% / 28.00 → 36.83% | +0.89 / +1.59 / +2.73 | +0.65 / +0.36 / +1.42 |
| N16 | tv / dn / wd | 0.49 → 1.24% / 0.72 → 1.07% / 1.77 → 2.26% | 3.64 → 5.87% / 3.64 → 9.51% / 25.89 → 34.33% | +1.09 / +1.06 / +2.46 | +0.65 / +0.04 / +1.10 |
| L16 | tv / dn / wd | 0.30 → 0.72% / 0.52 → 0.72% / 1.61 → 2.00% | 2.83 → 5.06% / 2.29 → 7.76% / 23.78 → 32.22% | +0.76 / +0.86 / +2.39 | +0.38 / +0.00 / +1.01 |
| N12 | tv / dn / wd | 0.74 → 1.04% / 0.82 → 1.00% / 2.28 → 2.69% | (12-stride tight: 0 / 4 / 110 rows) | +0.53 / +0.27 / +0.72 | +0.11 / −0.11 / +0.35 |

The rows short of frames lose the most: 2-9 points on 8-17% of the dialogue rows. But the rows with room lose 0.2-0.8 points as well, so a frame budget is not the whole story. Half the frame row also halves the temporal resolution and context the frame row and the head see. PSP text, smaller on screen, loses about twice or more what Vita's does.

### Locked report (once, after the selections; report only)

The three selections were committed and pushed (last `b5ade5a`, 09:56:58Z) before the report ran from the runner `stride16-report-b5ade5a` (2026-10-24 09:57-10:20Z).

- **Fixed lines** (vita-test + psp-test) and **whole dialogue:** `eval/reproducibility/stride16-20261023/report.sh` through `eval/vocr3_report.py`, the scalar C W8I runtime.
- **CLANNAD:** `eval/clannad_refs.py`'s extra-arms path, one attempt, suffix `stride16`.
- **Arms:** the incumbents, and each arm's best run by development score (`s16f_S_seed1`, `s16f_N_seed2`, `s12f_N_seed2`, `s16f_L_seed2`).
- **Records:** `eval/stride16_report_20261024.json` (with `docs/stride16-report-20261024.md`), `eval/stride16_dialogue_20261024.json`, `eval/stride16_clannad_20261024.json`, `eval/stride16_clannad_dialogue_20261024.json` and `eval/stride16_floor_protected_20261024.json`.

**Fixed lines, dialogue, console polarity** (pooled, 4,693 characters; pairs: line / VN-cluster 95% CI):

| model | vita-test | psp-test | pooled | with beam + 1 MB LM | arm − incumbent, pooled |
|---|---|---|---|---|---|
| VOCR5-S | 1.00% | 2.22% | **1.60%** | 1.15% | |
| S16 | 2.09% | 3.17% | **2.62%** | 2.07% | +1.02 (+0.49 to +1.61; VN +0.30 to +1.92) |
| VOCR5-N | 0.96% | 1.30% | **1.13%** | 0.72% | |
| N16 | 1.55% | 2.04% | **1.79%** | 1.41% | +0.66 (+0.23 to +1.15; VN +0.15 to +1.29) |
| N12 | 1.09% | 1.61% | **1.34%** | 1.04% | +0.21 (−0.11 to +0.56; VN −0.27 to +0.62) |
| VOCR5-L | 0.59% | 1.00% | **0.79%** | 0.53% | |
| L16 | 1.25% | 1.61% | **1.43%** | 1.07% | +0.64 (+0.22 to +1.11; VN +0.04 to +1.35) |

**Whole dialogue** (finder-cut regions, the C GRAD finder, dialogue text, pooled; VN-cluster 95% CI):

| model | base | +4 px | +8 px | arm − incumbent at base |
|---|---|---|---|---|
| VOCR5-S | 8.31% | 11.28% | 12.00% | |
| S16 | 10.18% | 13.42% | 14.02% | +1.88 (+0.81 to +3.14) |
| VOCR5-N | 7.98% | 10.68% | 12.02% | |
| N16 | 9.39% | 12.75% | 13.18% | +1.40 (+0.61 to +2.45) |
| N12 | 8.20% | 11.48% | 12.06% | +0.22 (−0.24 to +0.69) |
| VOCR5-L | 7.08% | 10.03% | 11.15% | |
| L16 | 8.65% | 11.78% | 12.47% | +1.57 (+0.79 to +2.52) |

**CLANNAD** (owner-drive-ingest's frozen references; page bootstrap):

| model | dialogue CER (2,052 lines) [95% CI] | dialogue: arm − incumbent | all lines (3,048) | all: arm − incumbent |
|---|---|---|---|---|
| VOCR5-S | 1.38% [1.23, 1.54] | | 2.07% | |
| S16 | 1.52% [1.35, 1.69] | +0.13 [−0.01, +0.28] | 2.15% | +0.08 [−0.12, +0.29] |
| VOCR5-N | 0.96% [0.84, 1.08] | | 1.45% | |
| N16 | 1.23% [1.10, 1.37] | +0.27 [+0.14, +0.40] | 1.78% | +0.32 [+0.16, +0.50] |
| N12 | 1.09% [0.96, 1.23] | +0.13 [+0.01, +0.25] | 1.53% | +0.08 [−0.06, +0.22] |
| VOCR5-L | 0.73% [0.63, 0.85] | | 0.98% | |
| L16 | 0.88% [0.77, 1.00] | +0.15 [+0.04, +0.25] | 1.46% | +0.48 [+0.34, +0.61] |

Notes on the locked report:

- **The incumbents reproduce VOCR5's locked report exactly** on every set: fixed lines 1.60 / 1.13 / 0.79% (and 1.15 / 0.72 / 0.53% with the LM), whole dialogue 8.31 / 7.98 / 7.08%, CLANNAD dialogue 1.38 / 0.96 / 0.73%. The runtime on this branch carries stride16's and head-gate's changes, so this is also an end-to-end check that both leave stride-8 models unchanged.
- **The frame budget of the protected lines** (report only, `eval/stride16_floor_protected_20261024.json`):
  - At stride 16, 31 of 451 protected lines are tight and 4 are below the floor.
  - On their 28 dialogue lines (352 characters): VOCR5-S 7.4% → S16 17.9%, VOCR5-N 5.4% → N16 12.5% (N12 8.5%), VOCR5-L 4.3% → L16 10.8%.
- **Disclosures.** VOCR5's disclosures apply unchanged, because the training data is VOCR5's: planetarian's PSP port among the owner's captures, CLANNAD's cast names in synth3 v2 (Small), and the reserved speaker names in the synthetic renders. The 1 MB LM's word list includes CLANNAD's and other test titles' scripts.

### Estimates (A9, per nominal 3-line box)

Measured with `vita/a9_layers.py` on the trained blobs above (`eval/stride16_estimates_20261024.json`, `trained_blobs`). The default tile is 2 at stride 16, 3 at 12 and 4 for the incumbents. They agree with the random-init shape estimates of the protocol to within 0.004 s.

| model | #85, no prefetch credit: 333 / 444 MHz | #85 with prefetch credit, 333 | main before #85: 333 / 444 | real box (89 frames), 333, no prefetch credit |
|---|---|---|---|---|
| VOCR5-S | 0.627 / 0.495 s | 0.564 | 0.938 / 0.729 | 195 ms |
| S16 | 0.495 / 0.396 s (−132 / −99 ms) | 0.442 | 0.747 / 0.585 | 147 ms (−49) |
| VOCR5-N | 0.919 / 0.714 s | 0.805 | 1.338 / 1.029 | 304 ms |
| N16 | 0.741 / 0.581 s (−178 / −133 ms) | 0.639 | 1.122 / 0.867 | 238 ms (−66) |
| N12 | 0.783 / 0.613 s (−136 / −101 ms) | 0.687 | 1.264 / 0.973 | 253 ms (−50) |
| VOCR5-L | 2.073 / 1.624 s | 1.461 | 2.678 / 2.078 | 732 ms |
| L16 | 1.598 / 1.249 s (−475 / −375 ms) | 1.114 | 2.377 / 1.834 | 555 ms (−176) |

Sizes:

- **Weights** are unchanged.
- **The W8I arena at the default tile is smaller** than the incumbent's (exact): Small 124,832 → 116,384 B; Normal 182,944 → 170,656 B (N12 176,832); Large 261,024 → 239,264 B.

None of these times was measured on a console.

### Head gate, V16, release

- **Head gate.** head-gate (#120) merged before the selections. The protocol trains a gate only for an adopted model, and none was adopted, so nothing was composed. The gate's own estimate for VOCR5 stands: −122 / −137 / −163 ms per nominal box, `docs/headgate.md`.
- **V16** was not run, because S16 was not adopted.
- **Release.** No model is published and no Rubi pin changes. The stride-12 and stride-16 blobs stay under `$HERD/shared/artifacts/stride16/blobs` (sha256 in the decision and score files).

### Timeline (UTC) and hygiene

| when | what |
|---|---|
| 2026-10-23 13:50-13:58Z | two 20-step `train2` smokes (stride 16 and 12, random init, `/tmp`, deleted) |
| 13:58-14:08Z | the 1,000-step speed probes (8 at once; checkpoints deleted, no development evaluation) |
| 14:42:49Z | protocol commit (`eeb91d1` on the original history), tag `stride16-runner-eeb91d1` |
| 14:43:05-14:45:25Z | the 8 pretrains start |
| 14:5xZ-15:08Z | the incumbents scored on development data (they equal VOCR5's records) |
| 16:33-18:00Z | Small pretrains paused at step 24,000 to share slots, then resumed |
| 19:44Z | amendment 1 (rows keyed by position; before any candidate score), tag `stride16-runner-2813e7f`, the scoring runner |
| 21:51Z | the branch rebased onto main `868bc91` (head-gate, #120) |
| 2026-10-24 06:21-08:20Z | the pretrains end |
| 06:22-09:54Z | the fine-tunes |
| 08:25-09:57Z | the development scores, decisions and selections, each committed in order |
| 09:57-10:20Z | the locked report and CLANNAD, once |

The branch was rebased twice: onto `868bc91` (head-gate, #120) on 2026-10-23 at 21:51Z, and onto `fb63532` (head-2level, #122) on 2026-10-24 after the locked report. The original commits are kept at the tags `stride16-history-20261023a` and `stride16-history-20261024a`, and the runner tags `stride16-runner-eeb91d1`, `stride16-runner-2813e7f` and `stride16-report-b5ade5a` hold the trees the runs, scores and report used. Author dates keep the times above.

The first rebase replaced this branch's first re-pin of the shared runtime files. head-gate makes the textspot pin tests skip `runtime/vocr.c` and `vocr.h`, and re-pins them in textspot3's amendment 8; head-2level's amendment 9 does the same for its own change; this branch's amendment 10 re-pins the files with stride16's changes on top. The second rebase changes no file a score read: head-2level's runtime additions are off by default (`vocr_opts.two_level = 0`), and the scoring and report runners are the tagged trees.

Hygiene:

- No arm, run, step count, checkpoint, threshold or pool was chosen, tuned or dropped on vita-test, psp-test or CLANNAD.
- The in-run development evaluations were logged only, and every run is its final checkpoint.
- The owner's frames and every CLANNAD reading stay under `shared/private`. Committed files hold counts, ids and hashes only.

## Reproduce

```sh
# runs:      eval/reproducibility/stride16-20261023/queue.sh + feeder.sh (jobs: eval/stride16_run.sh TIER STRIDE pre|F SEED)
# floor:     eval/stride16_floor.py build --out FLOOR.json    (model-independent)
# select:    eval/reproducibility/stride16-20261023/select_tier.sh TIER export|score|scores|decide|select
# report:    eval/reproducibility/stride16-20261023/report.sh spec|shared|render|lm|dialogue|floor|clannad
# CLANNAD dialogue pairs: eval/stride16_clannad_dialogue.py --work WORK --private score.stride16.private.jsonl --pair ...
```

Artifacts: `$HERD/shared/artifacts/stride16/` (runs, blobs, select, report, est, floor, logs, NOTES.md) and `$HERD/shared/private/stride16/clannad/` (private).
