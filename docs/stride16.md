# stride16: the recognizers retrained at a coarser CTC frame step

Issue [#118](https://github.com/bee-san/vita-vn-ocr/issues/118), item 3 of [#117](https://github.com/bee-san/vita-vn-ocr/issues/117). Owner (2026-10-23 13:34Z): "retrain for a coarser frame step".

VOCR5's recognizers emit one CTC frame per 8 input columns. VN dialogue has about 3 frames per character at that step (median 3.14 on Vita, 2.88 on PSP; #117), and CTC needs about 1. stride16 retrains each tier with one frame per 16 columns (Normal also at 12). The recipe, data and vocabularies are VOCR5's. A tier's coarser model is adopted only if it reads development dialogue no worse than VOCR5's.

Every console time below is an estimate. Nothing of stride16 has run on a console.

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
| fixed lines, Vita (514) | 0 / 0 | 0 / 0 | 0 / 19 |
| whole-dialogue units, GRAD (1,515) | 20 / 24 | 31 / 90 | 101 / 362 |
| whole-dialogue units, GRAD + refine (1,515) | 16 / 16 | 21 / 35 | 39 / 112 |

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

To come: the runs, the decisions, the selections and the locked report.
