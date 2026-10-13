# VOCR3: console-sized recognizers from the synth3 studies

Three recognizers were selected on development data only and then scored once on the protected test
sets through the plain-C W8I runtime. **All three beat the model they would replace on the hand-checked
lines, by margins that are resolved on the pooled full-text intervals** (the psp-test intervals of the
two larger sizes and their pooled hiragana intervals include 0): VOCR3-FL10 (657,664 B) 2.13% pooled CER against VOCR1's 3.82% (−1.69
points, VN-cluster 95% CI −2.52 to −0.96) and VOCR2's 3.51%; VOCR3-H15 (1,155,968 B) 1.73% against
VOCR_H15's 2.21% (−0.47, CI −0.89 to −0.10), the model users run; VOCR3-W20 (1,558,016 B) 1.62% against
VOCR_H15 (−0.59, CI −1.14 to −0.12) and F20 (−0.57). The 657 KB VOCR3-FL10 makes fewer errors than the released
1.16 MB H15 on these lines in console polarity (108 against 112; a point estimate without an interval,
since the pair is not in the protocol, and reversed in native polarity, 109 against 106, and on whole
dialogue). **None qualifies under the repo's gates:** every candidate passes the fixed-line gate
and fails VOCR2's whole-dialogue guard, which forbids any per-title regression; on whole dialogue
VOCR3-FL10 is still clearly better than VOCR1 (−1.44 points, CI −2.31 to −0.59), VOCR3-W20 is level with
VOCR_H15 at the base region (−0.26, CI −0.97 to +0.40) and fails the guard's console-balanced hiragana
condition at +8 px, and VOCR3-H15 is slightly behind it (+0.30, CI −0.49 to +1.06), so the fixed-line
gains do not all carry over to finder-cut regions. What made the difference, by measured size: at FL10, four times longer pretraining (the winner's
128k-step pretrain is −0.85 development points below the same arm and seed at 32k steps), then the
study-2 recipe over the released FL10's (about −0.7 points on teacher_val; most likely its frame KD
and/or best-checkpoint init, not separated here), then the realplus pool in the fine-tune (adopted at
every size, −0.18 to −0.23 points on development data). Nothing has run on a console; the A9 times are estimates.

Machine-readable: the protocol `eval/vocr3_protocol.json` (frozen before any candidate was scored, with
its amendments), the selections `eval/vocr3_selection_<SIZE>_20261013.json`, the realplus decisions
`eval/vocr3_realplus_<SIZE>_20261013.json`, and the once-only locked report
`eval/vocr3_report_20261013.json`. Every number below is a host evaluation of real native screenshots
through the plain-C W8I runtime, or a development-cohort score; nothing has run on a console.

## What VOCR3 is

Three recognizers, one per size class, trained with train2's release recipe (hard labels, no frame KD)
on the VN-style synthetic data study 2 recommends for the size (`docs/synth3.md`), and compared head to
head with the released model each would replace:

| size | topology | vocab | int8 blob | W8I arena | replaces |
|---|---|---|---|---|---|
| FL10 | width 1.0, bottleneck 96, depth 1, ctx 2, ctx_k 5, height 32 (FL10's exactly) | FL10's 3,500 characters (`7f1cb2eb…`) | 657,664 B | 124,832 B | VOCR1 (`FL10_w8.vocr`, v0.2.0, `500e0b84…`); VOCR2 (`VOCR2_FL10_w8.vocr`, v0.2.3-vocr2.1, `d3adc52e…`) is the opt-in trial, finder-matched arm reported |
| H15 | width 1.5, same blocks | `data/vocab.txt`, 5,354 + blank (`ae09932d…`) | 1,155,968 B | 182,944 B | VOCR_H15 (`H15_w8.vocr`, v0.2.0, `316ebc37…`): identical to vita-jp-overlay main's default (`core/vocr_ocr.c`), the model users run |
| W20 class | width 2.0 (train2's F20 shape), and the late-wide LW shape `--chans 48,96,192,320,384` (amendment 6) | `data/vocab.txt` | 1,558,016 B (W20) / 1,473,280 B (LW) | 241,056 B / 192,928 B | VOCR_H15, if it fits the Vita budget and qualifies; F20 (v0.2.0, `f8c1abe4…`) is a same-size reference |

Budget (`docs/vita-memory-budget.md`, #53 and #60): int8 weights ≤ 1,600,000 B, W8I arena ≤ 262,144 B,
an estimated ≤ 550 ms per 640-px line on one 333 MHz Cortex-A9 (`vita/budget_sweep.py check`; the ms
figure is a host estimate). A candidate outside any limit is not selectable; none was. #60 kept that
ceiling and added tiers: always safe (≤ 900,000 B, ≤ 131,072 B, ≤ 300 ms: the FL10 class) and
safe-with-preflight (the ceiling: the H15 and W20 classes, which need the overlay's free-memory check
and an always-safe fallback; the cost is speed, not safety). Everything about the recognizer must live in
the game process: SceShell has tens of KiB of headroom (`docs/vita-platform-memory.md`;
vita-jp-overlay #68 moved the overlay's worker out of SceShell for that reason).

## Recipe and candidates

- Pretrain from scratch: nvsynth's every-line set (`docs/nvsynth.md`, rebuilt here) 0.5 + the arm's
  VN-style synthetic data 0.5 (A_s2: synth2; B_s3: synth3 v2; C_mix: 0.25 + 0.25); FL10 and the W20
  class 32,000 steps, H15 40,000; batch 96, lr 2e-3, EMA 0.999.
- Fine-tune 15,000 steps from the pretrain's final checkpoint: nv 0.25 + synthetic 0.25 + teacher_train
  0.275 + real_train 0.125 + PP-OCRv6 pseudo-labels 0.1, kd=0, on the copies of the real pools without
  CLANNAD (v4) and its VNDB family (`eval/synth3_study2_pools.sha256`; study 2's amendment 2, VOCR3's
  amendment 1).
- The realplus fine-tune (amendment 4, "rp" runs): the same fine-tune with the real half re-split as
  teacher_train 0.2 + real_train 0.1 + realplus 0.2 and no pseudo-labels, for every candidate pretrain.
  realplus is the 43k-line real training pool (`docs/realplus.md`, #57), without CLANNAD's relations
  and fan works (amendment 5; `eval/vocr3_realplus.sha256`).
- The final EMA checkpoint of each fine-tune is the candidate; no checkpoint selection inside a run.
- Candidates per size (`candidates` in the protocol): FL10, study 2's six runs plus the pretrain-length
  curve (64k and 128k pretrain steps); H15, study 2's four runs plus `ft_H15_B_s3_seed1` (amendment 2);
  W20 class, `ft_W20_{C_mix,A_s2,B_s3}_seed1` (amendment 1) and `ft_LW_<ARM>_seed1` (amendment 6).
- Variants (optional, each only if its precondition holds, each entering only through the written
  ablation rule): realplus (paired decision per size; adopted for every size, below) and synth3 v3 (only
  when the size's recommended arm uses synth3). Study 2 recommended synth2 for FL10 and H15 (no measured
  difference) and the W20 class has no study-2 decision, so no synth3-v3 variant was trained; study 1b's
  result (#56: v3 better than v2 on teacher_val, but worse on devnative's native-Vita ink crops) is a
  follow-up for the next round.

## Selection (development data only)

- Cohorts: teacher_val (1,651 lines, 181 VNs, `ba46539c…`; views rv and rvn) and devnative, the native
  Vita/PSP development cohort (`docs/devnative.md`, #52), version v3 (14ae22d), the version announced in
  devnative.status when the first VOCR3 selection score was computed (amendment 3): strict and tv_native
  subsets, ink and GRAD crops pooled, both views. Amendment 8 (below) removes the owner's Memories Off
  series from both before any selection score.
- Score: CER over teacher_val in both views, averaged with devnative's; the candidate with the lowest
  score wins (ties: shorter pretrain, then lower seed). Every blob passes the budget check first.
- realplus decision (amendment 4): rp runs replace the base candidates of a size iff, over its
  decision pairs on teacher_val, the VN-cluster 95% interval of CER(rp) − CER(base) is entirely below 0,
  every pair improves, and devnative's interval is not entirely above 0.

Both cohorts are the copies without the owner's Memories Off series (amendment 8; see Disclosure):
teacher_val 1,643 lines, devnative v3 986 crops (strict 34 ink + 23 GRAD, tv_native 509 ink + 420 GRAD),
commit 14ae22d. Every candidate's blob passed the budget check (`vita/budget_sweep.py check`, main
fab5d3f+: blob bytes, W8I arena, estimated ms per 640-px line at 333 MHz, high bound; the tier column is
`docs/vita-memory-budget.md`'s: FL10 blobs are in the always-safe tier, the H15 and W20 classes in the
preflight tier, which needs the overlay's free-memory check and an always-safe fallback).

### FL10: realplus adopted; the 128k-step pretrain wins

realplus decision over the six study-2 pretrains: CER(rp) − CER(base fine-tune) on teacher_val −0.23
points (VN-cluster 95% CI −0.34 to −0.12), every pair below 0 (−0.24, −0.19, −0.30, −0.15, −0.36, −0.11);
devnative +0.04 (CI −0.10 to +0.19), not entirely above 0: adopted, so the candidates are the eight rp
runs. `eval/vocr3_realplus_FL10_20261013.json`, `eval/vocr3_selection_FL10_20261013.json`.

| run | teacher_val CER | devnative v3 CER | development score | blob B / W8I arena B / est. ms high (tier) |
|---|---|---|---|---|
| rp_FL10_C_mix_pre128k_seed1 **(selected)** | 2.26% | 2.34% | 2.30% | 657,664 / 124,832 / 235 (always) |
| ft_FL10_C_mix_pre128k_seed1 | 2.61% | 2.49% | 2.55% | 657,664 / 124,832 / 235 (always) |
| rp_FL10_C_mix_pre64k_seed1 | 2.67% | 3.01% | 2.84% | 657,664 / 124,832 / 235 (always) |
| ft_FL10_C_mix_pre64k_seed1 | 2.82% | 2.87% | 2.84% | 657,664 / 124,832 / 235 (always) |
| rp_FL10_C_mix_seed1 | 3.16% | 3.14% | 3.15% | 657,664 / 124,832 / 235 (always) |
| rp_FL10_C_mix_seed2 | 3.06% | 3.25% | 3.16% | 657,664 / 124,832 / 235 (always) |
| ft_FL10_C_mix_seed3 | 3.30% | 3.28% | 3.29% | 657,664 / 124,832 / 235 (always) |
| rp_FL10_C_mix_seed3 | 3.18% | 3.43% | 3.31% | 657,664 / 124,832 / 235 (always) |
| ft_FL10_A_s2_seed2 | 3.32% | 3.30% | 3.31% | 657,664 / 124,832 / 235 (always) |
| ft_FL10_C_mix_seed1 | 3.31% | 3.35% | 3.33% | 657,664 / 124,832 / 235 (always) |
| rp_FL10_A_s2_seed2 | 3.13% | 3.55% | 3.34% | 657,664 / 124,832 / 235 (always) |
| rp_FL10_A_s2_seed3 | 3.11% | 3.60% | 3.36% | 657,664 / 124,832 / 235 (always) |
| ft_FL10_C_mix_seed2 | 3.43% | 3.32% | 3.37% | 657,664 / 124,832 / 235 (always) |
| ft_FL10_A_s2_seed3 | 3.41% | 3.39% | 3.40% | 657,664 / 124,832 / 235 (always) |
| rp_FL10_A_s2_seed1 | 3.28% | 3.54% | 3.41% | 657,664 / 124,832 / 235 (always) |
| ft_FL10_A_s2_seed1 | 3.52% | 3.61% | 3.56% | 657,664 / 124,832 / 235 (always) |

**Selected: `rp_FL10_C_mix_pre128k_seed1`** (blob sha256 `87fb4d0c…`), development score 2.30%
(teacher_val 2.26%, devnative 2.34%). The 128,000-step pretrain is study 2's step-curve run (one seed):
pretrain length moved FL10 by more than any data arm (the pre64k rp is next, 2.84%; the 32k-pretrain rp
runs are at 3.15-3.41%). The study-2 recipe at 32k steps already puts every FL10 candidate far below
the released FL10, 4.08% on teacher_val scored the same way (C W8I; `docs/synth3.md`), most likely
because of the release recipe's frame KD and/or best-checkpoint init.

### H15: realplus adopted; `rp_H15_C_mix_seed2`

realplus decision over the four study-2 pretrains: teacher_val −0.19 (CI −0.29 to −0.10), every pair
below 0 (−0.20, −0.26, −0.19, −0.12); devnative +0.03 (CI −0.12 to +0.18): adopted; the candidates are
the five rp runs (study 2's four pretrains and amendment 2's synth3-only pretrain).
`eval/vocr3_realplus_H15_20261013.json`, `eval/vocr3_selection_H15_20261013.json`.

| run | teacher_val CER | devnative v3 CER | development score | blob B / W8I arena B / est. ms high (tier) |
|---|---|---|---|---|
| rp_H15_C_mix_seed2 **(selected)** | 2.06% | 2.15% | 2.11% | 1,155,968 / 182,944 / 398 (preflight) |
| rp_H15_A_s2_seed2 | 2.07% | 2.18% | 2.13% | 1,155,968 / 182,944 / 398 (preflight) |
| rp_H15_C_mix_seed1 | 2.10% | 2.24% | 2.17% | 1,155,968 / 182,944 / 398 (preflight) |
| ft_H15_C_mix_seed2 | 2.19% | 2.18% | 2.19% | 1,155,968 / 182,944 / 398 (preflight) |
| ft_H15_A_s2_seed2 | 2.33% | 2.13% | 2.23% | 1,155,968 / 182,944 / 398 (preflight) |
| ft_H15_C_mix_seed1 | 2.29% | 2.21% | 2.25% | 1,155,968 / 182,944 / 398 (preflight) |
| rp_H15_A_s2_seed1 | 2.23% | 2.42% | 2.33% | 1,155,968 / 182,944 / 398 (preflight) |
| ft_H15_A_s2_seed1 | 2.43% | 2.36% | 2.39% | 1,155,968 / 182,944 / 398 (preflight) |
| ft_H15_B_s3_seed1 | 2.31% | 2.72% | 2.52% | 1,155,968 / 182,944 / 398 (preflight) |
| rp_H15_B_s3_seed1 | 2.22% | 3.31% | 2.76% | 1,155,968 / 182,944 / 398 (preflight) |

**Selected: `rp_H15_C_mix_seed2`** (blob sha256 `cb3c61a8…`), development score 2.11% (teacher_val
2.06%, devnative 2.15%); the A_s2 seed-2 rp run is 0.02 points behind, the synth3-only pretrain's rp run
last (its devnative 3.31% is the worst of the class, while its teacher_val is ordinary: synth3 alone
transfers worse to native frames than the mix or synth2). The released H15 is at 2.28% on teacher_val scored the same
way (C W8I; `docs/synth3.md`).

### W20 class: realplus adopted; `rp_W20_C_mix_seed1`; the late-wide LW shape loses

realplus decision over the three W20 pretrains: teacher_val −0.18 (CI −0.27 to −0.08), every pair below 0
(−0.14, −0.26, −0.14); devnative +0.14 (CI −0.09 to +0.39): adopted; the candidates are the four rp runs
(the three W20 arms and amendment 6's LW shape). `eval/vocr3_realplus_W20_20261013.json`,
`eval/vocr3_selection_W20_20261013.json`.

| run | teacher_val CER | devnative v3 CER | development score | blob B / W8I arena B / est. ms high (tier) |
|---|---|---|---|---|
| rp_W20_C_mix_seed1 **(selected)** | 1.74% | 1.88% | 1.81% | 1,558,016 / 241,056 / 548 (preflight) |
| rp_W20_A_s2_seed1 | 1.82% | 1.97% | 1.90% | 1,558,016 / 241,056 / 548 (preflight) |
| ft_W20_C_mix_seed1 | 1.87% | 1.95% | 1.91% | 1,558,016 / 241,056 / 548 (preflight) |
| ft_W20_A_s2_seed1 | 2.08% | 1.84% | 1.96% | 1,558,016 / 241,056 / 548 (preflight) |
| rp_LW_A_s2_seed1 | 2.13% | 2.36% | 2.24% | 1,473,280 / 192,928 / 468 (preflight) |
| ft_LW_A_s2_seed1 | 2.32% | 2.22% | 2.27% | 1,473,280 / 192,928 / 468 (preflight) |
| ft_W20_B_s3_seed1 | 1.96% | 2.60% | 2.28% | 1,558,016 / 241,056 / 548 (preflight) |
| rp_W20_B_s3_seed1 | 1.82% | 2.97% | 2.39% | 1,558,016 / 241,056 / 548 (preflight) |

**Selected: `rp_W20_C_mix_seed1`** (blob sha256 `2cedeb01…`), development score 1.81% (teacher_val
1.74%, devnative 1.88%). The LW shape (`--chans 48,96,192,320,384`, 1,473,280 B, est. 468 ms), which
arch-research's 5k-step proxies put level with F20 (#58), is 0.4 points behind the width-2.0 runs after
the full recipe (2.24% against 1.81-1.90%): the proxy did not predict the full-length ordering. The
synth3-only pretrain is again last on devnative (2.60-2.97%). Amendment 7's condition held: main after
#58 (fab5d3f) has `train/model.py` c21b9d85, `train/train2.py` cf72081e, `train/export.py` 421376ae and
`eval/backends.py` b42d0591, byte-identical to the audited PR head a4b6202 and to the LW runner commit
704ed0e (tag `vocr3-runner-lw-704ed0e`), so the LW runs are valid candidates. The W20 blobs sit at the
budget's edge: `vita/budget_sweep.py check` (the protocol's selectability tool, 2,133 synth-val frames)
estimates 311-548 ms per 640-px line at 333 MHz, inside the 550 ms ceiling, while the locked report's own
A9 estimate of the same blob (1,828 frames) is 314-554 ms, over it; the ceiling lies inside the spread of
the two estimators, and so does F20's. The selection stands by the protocol's tool; a 550 ms ceiling
should not be read as met with margin.

## Locked report (once, after every selection was committed)

`eval/vocr3_report.py` (#51, verified by reproducing the released blobs' published numbers) ran once,
16:21-16:49Z on 2026-10-13, after the last selection commit (87641eb at 16:19Z; 37a8eb9 on this branch
after the rebase). A first start at 16:19:33Z aborted in the same second at spec validation
(`vocr3_make_spec.py` had written a `selected_runs` field the harness rejects) and wrote nothing. The
`shared` step (the test-set scoring) then ran from feat/vocr3 at tag `vocr3-report-runner-9eadb90`
with the spec `eval/vocr3_report_spec_20261013.json` built by `eval/vocr3_make_spec.py` from the three
committed selections; the `a9`, `overlay` and `render` steps ran from tag `vocr3-report-runner-f978800`
(the same tree with a reformatted `vocr3_make_spec.py` and a stronger test; `eval/vocr3_report.py` is the
same blob `f7659ec7` in both and on this branch), which the branch's rebase later replaced by 19139f2. Scalar C W8I runtime (gcc 11.5),
vita-test (206 lines, 16 games) and psp-test (245 lines, 14 games): fixed lines (console polarity primary,
native also; per set, per style, per VN, pooled; full text and hiragana) and whole dialogue (each
manifest region at its base box and expanded by 4 and 8 native px, the arm's own C finder, crops joined
in finder order). Paired bootstrap 95% intervals, 10,000 resamples, seed 0, over lines and over VN
clusters (strata with at least 5 VNs). Output: `eval/vocr3_report_20261013.json` (integer counts and
hashes; the per-line records stay in the private part) and the generated `docs/vocr3-report-20261013.md`
(every table). The fresh native textbox test and the CLANNAD long run need private inputs that are not on
this machine; they are not in this report.

### Fixed lines (the hand-checked crops; console polarity unless stated)

| arm | vita-test CER | psp-test CER | pooled CER | pooled hiragana CER | pooled exact lines | pooled CER, native polarity | blob B / W8I arena B | est. ms per 640-px line (W8I, A9 at 333 MHz) |
|---|---|---|---|---|---|---|---|---|
| VOCR1 | 2.38% (61/2564) | 5.30% (133/2510) | 3.82% (194/5074) | 1.61% (45/2799) | 71.6% | 3.72% (189/5074) | 657,664 / 124,832 | 125-239 |
| VOCR2 | 2.30% (59/2564) | 4.74% (119/2510) | 3.51% (178/5074) | 1.36% (38/2799) | 73.6% | 3.49% (177/5074) | 657,664 / 124,832 | 125-239 |
| VOCR_H15 | 1.56% (40/2564) | 2.87% (72/2510) | 2.21% (112/5074) | 0.75% (21/2799) | 82.5% | 2.09% (106/5074) | 1,155,968 / 182,944 | 221-403 |
| F20 | 1.72% (44/2564) | 2.67% (67/2510) | 2.19% (111/5074) | 1.21% (34/2799) | 82.3% | 2.29% (116/5074) | 1,558,016 / 241,056 | 314-554 |
| VOCR3-FL10 | 1.44% (37/2564) | 2.83% (71/2510) | 2.13% (108/5074) | 0.61% (17/2799) | 84.0% | 2.15% (109/5074) | 657,664 / 124,832 | 125-239 |
| VOCR3-H15 | 1.09% (28/2564) | 2.39% (60/2510) | 1.73% (88/5074) | 0.61% (17/2799) | 85.6% | 1.73% (88/5074) | 1,155,968 / 182,944 | 221-403 |
| VOCR3-W20 | 1.05% (27/2564) | 2.19% (55/2510) | 1.62% (82/5074) | 0.57% (16/2799) | 86.9% | 1.58% (80/5074) | 1,558,016 / 241,056 | 314-554 |

| pair (console polarity) | vita-test, full text | psp-test, full text | pooled, full text | pooled, hiragana |
|---|---|---|---|---|
| VOCR3-FL10 − VOCR1 | −0.94 (−1.77 to −0.27) | −2.47 (−3.80 to −1.25) | −1.69 (−2.52 to −0.96) | −1.00 (−1.74 to −0.46) |
| VOCR3-FL10 − VOCR2 | −0.86 (−1.60 to −0.26) | −1.91 (−3.04 to −0.85) | −1.38 (−2.08 to −0.75) | −0.75 (−1.33 to −0.30) |
| VOCR3-FL10-C3 − VOCR2 | −0.86 (−1.60 to −0.26) | −1.91 (−3.04 to −0.85) | −1.38 (−2.08 to −0.75) | −0.75 (−1.33 to −0.30) |
| VOCR3-H15 − VOCR_H15 | −0.47 (−0.95 to −0.08) | −0.48 (−1.33 to +0.18) | −0.47 (−0.89 to −0.10) | −0.14 (−0.50 to +0.16) |
| VOCR3-H15 − VOCR1 | −1.29 (−2.07 to −0.68) | −2.91 (−4.84 to −1.42) | −2.09 (−3.05 to −1.29) | −1.00 (−1.69 to −0.44) |
| VOCR3-W20 − VOCR_H15 | −0.51 (−1.31 to −0.03) | −0.68 (−1.42 to +0.12) | −0.59 (−1.14 to −0.12) | −0.18 (−0.49 to +0.07) |
| VOCR3-W20 − F20 | −0.66 (−1.40 to −0.16) | −0.48 (−1.21 to +0.32) | −0.57 (−1.08 to −0.09) | −0.64 (−1.22 to −0.22) |
| VOCR3-W20 − VOCR3-H15 | −0.04 (−0.50 to +0.29) | −0.20 (−0.85 to +0.57) | −0.12 (−0.51 to +0.27) | −0.04 (−0.41 to +0.40) |

Per style (`docs/vocr3-report-20261013.md`): the gains are largest where the released models are
weakest. On psp-test, names go from 17.7% (VOCR1) / 9.1% (VOCR_H15) to 11.5% (VOCR3-FL10), 9.1%
(VOCR3-H15) and 7.7% (VOCR3-W20); furigana lines from 23.3% / 14.6% to 16.5% / 10.2% / 11.2%; dark-on-light
text from 5.4% / 2.6% to 2.4% / 2.6% / 2.3%; small text from 3.6% / 1.6% to 1.4% / 2.4% / 1.4%. On
vita-test, coloured text goes from 6.7% (VOCR1 and VOCR_H15) to 4.1% (VOCR3-H15) and 2.6% (VOCR3-W20),
names from 7.0% / 5.2% to 2.3-2.9% for the two larger VOCR3 sizes. VOCR3-H15 is behind VOCR_H15 on
psp-test's small text (2.4% against 1.6%, 23 against 16 errors), on psp-test mincho by one error (24
against 23) and on vita-test's three furigana lines (2 errors against 0); no other style of the H15 size
regresses on full text, and three psp-test styles regress by one or two hiragana errors (dark-on-light
9 against 7, mincho 7 against 5, small 9 against 8).

### Whole dialogue (finder-cut regions, joined; all text)

| arm | base region: all-text errors (CER) | expanded +4 px | expanded +8 px |
|---|---|---|---|
| VOCR1 | 660/5016 (13.16%); hiragana 217/2770 | 816/5016 (16.27%); hiragana 280/2770 | 866/5016 (17.26%); hiragana 296/2770 |
| VOCR2 | 629/5016 (12.54%); hiragana 206/2770 | 793/5016 (15.81%); hiragana 270/2770 | 845/5016 (16.85%); hiragana 288/2770 |
| VOCR_H15 | 539/5016 (10.75%); hiragana 177/2770 | 694/5016 (13.84%); hiragana 241/2770 | 738/5016 (14.71%); hiragana 249/2770 |
| F20 | 530/5016 (10.57%); hiragana 183/2770 | 688/5016 (13.72%); hiragana 248/2770 | 717/5016 (14.29%); hiragana 254/2770 |
| VOCR3-FL10 | 588/5016 (11.72%); hiragana 187/2770 | 728/5016 (14.51%); hiragana 249/2770 | 791/5016 (15.77%); hiragana 261/2770 |
| VOCR3-H15 | 554/5016 (11.04%); hiragana 177/2770 | 713/5016 (14.21%); hiragana 238/2770 | 761/5016 (15.17%); hiragana 249/2770 |
| VOCR3-W20 | 526/5016 (10.49%); hiragana 175/2770 | 693/5016 (13.82%); hiragana 237/2770 | 743/5016 (14.81%); hiragana 258/2770 |

| pair, all text, pooled (points; VN-cluster 95% CI) | base | expanded +4 | expanded +8 |
|---|---|---|---|
| VOCR3-FL10 − VOCR1 | −1.44 (−2.31 to −0.59) | −1.75 (−2.77 to −0.73) | −1.50 (−2.62 to −0.34) |
| VOCR3-H15 − VOCR_H15 | +0.30 (−0.49 to +1.06) | +0.38 (−0.37 to +1.11) | +0.46 (−0.28 to +1.22) |
| VOCR3-W20 − VOCR_H15 | −0.26 (−0.97 to +0.40) | −0.02 (−0.72 to +0.67) | +0.10 (−0.55 to +0.67) |
| VOCR3-W20 − F20 | −0.08 (−0.81 to +0.62) | +0.10 (−0.60 to +0.84) | +0.52 (−0.20 to +1.27) |
| VOCR3-FL10-C3 − VOCR2 (finder-matched: both C3) | −0.82 (−1.66 to +0.02) | −1.30 (−2.28 to −0.31) | −1.06 (−2.13 to +0.08) |
| VOCR3-W20 − VOCR3-H15 | −0.56 (−0.99 to −0.09) | −0.40 (−0.99 to +0.19) | −0.36 (−0.93 to +0.21) |

On the fixed lines the finder-matched VOCR3-FL10-C3 row equals the VOCR3-FL10 row by construction (same
blob; fixed lines use no finder); on whole dialogue the C3 finder separates them. The fixed-line gains
carry over for the FL10 size (VOCR3-FL10 beats VOCR1 on whole dialogue by −1.44 to −1.75 points,
resolved) but not for the H15 size: VOCR3-H15 makes 554 all-text errors on the base regions
against VOCR_H15's 539 (+0.30 points, CI −0.49 to +1.06; equal hiragana errors, 177), and is behind on
both expansions; VOCR3-W20 is level with VOCR_H15 (526 against 539 at base, 743 against 738 at +8 px).
The whole-dialogue scoring reads every crop the GRAD finder returns in a region, name plates and
partial lines included, and the selection cohorts are hand-cut lines, so a model can gain on the lines
and lose on the finder's crops; which crops VOCR3-H15 reads worse is not analysed here (the per-title
cells are in the report JSON). This is the open question for the next round, and the reason a
whole-dialogue development cohort is needed before a candidate can be selected for the overlay's actual
reading path.

### Size, speed and the overlay cross-check

Sizes and arenas are exact (table above). The Cortex-A9 times are `vita/a9_cycles.py` estimates on 40
synth-val lines (host estimate, never a console measurement): VOCR3-FL10 125-239 ms per 640-px line at
333 MHz in W8I, VOCR3-H15 221-403 ms, VOCR3-W20 314-554 ms, identical to the released models of the same
shape because the kernels have no data-dependent branches. The overlay host cross-check
(vita-jp-overlay `1a118ef`'s `tools/benchmark_local_ocr.py` on a read-only clone, runtime pin `49ff6f4d`)
reads every region of both sets with each blob: its all-text errors equal this harness's base-region
whole-dialogue errors for every arm (12 of 16 cells equal exactly; the other 4 are the VOCR3-FL10 blob,
which the overlay table ran twice because two arms share it, giving exactly twice the errors over twice
the characters: 388/5012 and 788/5020 against 194/2506 and 394/2510). The overlay's workspace peak, as the
host benchmark measures what the overlay would allocate in the game process, was 0.9 MiB for the FL10
blobs, 1.4 MiB for H15 and 1.8 MiB for W20.

## Qualification

VOCR2's gates against the model a candidate would replace: the fixed-line gate (primary polarity: no set's
full-text or hiragana errors increase, pooled hiragana errors decrease) and the held-out whole-dialogue
guard (no title/variant regression; every variant improves console-balanced hiragana in both scopes and
gains on at least two titles).

| gate | fixed-line gate (console) | whole-dialogue guard | qualifies |
|---|---|---|---|
| VOCR3-FL10 vs VOCR1 | pass: vita-test 61→37 errors (hiragana 15→10), psp-test 133→71 (30→7), pooled hiragana 45→17 | FAIL: 69 of 360 title checks regress (13 titles); titles gained: base 16, +4 14, +8 14 | no |
| VOCR3-H15 vs VOCR_H15 | pass: vita-test 40→28 errors (hiragana 8→5), psp-test 72→60 (13→12), pooled hiragana 21→17 | FAIL: 124 of 360 title checks regress (19 titles); titles gained: base 7, +4 7, +8 8 | no |
| VOCR3-W20 vs VOCR_H15 | pass: vita-test 40→27 errors (hiragana 8→7), psp-test 72→55 (13→9), pooled hiragana 21→16 | FAIL: 126 of 360 title checks regress (24 titles); titles gained: base 9, +4 8, +8 6 | no |

No candidate qualifies: each passes the fixed-line gate and fails the whole-dialogue guard, as VOCR2 did
(`docs/vocr2-qualification.md`). The guard is strict by design (any of 30 titles × 3 variants × 2 scopes
× 2 metrics regressing by one error fails it), and the three candidates regress 69-126 of its 360 cells.
VOCR3-W20 also fails the guard's second condition: at +8 px its console-balanced hiragana gets worse than
VOCR_H15's (0.0974 → 0.1003 all text, 0.0952 → 0.0989 body), so it is not "level" there. For VOCR3-FL10
the regressions sit beside a resolved overall gain; for VOCR3-H15 they go with a small overall loss. The
released defaults therefore stay. VOCR3-FL10 is the candidate for an opt-in trial (it beats VOCR1 on
whole dialogue by a resolved margin and VOCR2 by −0.82 to −1.30 points); VOCR3-W20 would need the guard's
hiragana condition and its A9 estimate looked at first; the owner decides. Weights are not committed; they are at
`$HERD/shared/artifacts/synth3/vocr3/report/blobs/` (sha256s in `eval/vocr3_report_spec_20261013.json`)
for a release asset if the owner wants one.

## Hygiene

- No candidate, variant, recipe, checkpoint or threshold was chosen, tuned or dropped on vita-test,
  psp-test, the fresh native textbox test or CLANNAD; the locked report ran once and could not change the
  selection (its outputs are dated after the selection commits). Before it, study 2's own protocol had
  scored the base fine-tunes (the `ft_` candidates, not the rp runs or any selected model) on vita-test and
  psp-test as reported-only test reports: FL10 at 23:48Z on 2026-10-12, H15 at 08:04Z, the step-curve runs
  at 13:35Z (after the FL10 selection was pushed at 13:33Z); `docs/synth3.md`. No rp blob was scored on a
  protected set outside the locked report.
- Every training pool and selection cohort excludes the test VNs and families of `eval/`, CLANNAD and
  its relations, the fresh native textbox test's titles and relations, and every eval screenshot
  (`docs/realplus.md`, `docs/devnative.md`, `eval/synth3_study2_pools.sha256`, `eval/vocr3_realplus.sha256`);
  the owner's titles reserved on 2026-10-13 are in the pinned training pools (Disclosure below).
- Each amendment was committed before the selection it governs: amendments 1 and 2 before any study-2
  score, amendments 3-5 after study 2's FL10 scores (teacher_val and devnative at 23:19-23:25Z, its
  reported-only test report at 23:48Z) but before any VOCR3 selection score, and none of them picks among
  FL10 candidates by score (3 fixes the devnative version mechanically, 4 applies to every pretrain, 5
  removes CLANNAD fan works); amendments 2, 4 and 6 list what was known, amendment 3 does not; amendment 7
  before any W20-class score; amendment 8 before the first selection score.

## Disclosure: the owner's reserved titles

On 2026-10-13 the owner reserved, as fresh native console test sets, the Memories Off series (10:51Z;
45 VNDB ids, `data/manifests/synth3_exclude_owner_games.json` as of #62, commit c5e2ef0) and Aiyoku no
Eustia and 428 with their relations (13:10Z; 5 more ids, 50 in the manifest #63 rebuilt). Every pretrain
and most fine-tunes had started before the first rule, but four fine-tunes started after it, from jobs
queued before the rules (queue6 at 02:17Z, the LW waiter at 09:12Z) and not stopped: the step-curve
fine-tune `ft_FL10_C_mix_pre128k_seed1` (12:06-13:02Z), **the selected VOCR3-FL10 `rp_FL10_C_mix_pre128k_seed1`
(12:08-13:08Z)**, and the LW pair `ft_LW_A_s2_seed1` and `rp_LW_A_s2_seed1` (14:06/14:08-15:50/15:54Z, after
both rules); four rp runs that were training at 10:51Z ended between 10:54Z and 12:05Z, and the 128k and
LW pretrains ran until 12:05Z and 14:05Z on synthetic data whose scene lists and name pools hold series
material (`docs/owner-games.md`). Amendment 8's reason, written at 11:16Z, said "after every candidate
run had started": that was wrong when written (erratum, 2026-10-13 19:00Z). The pinned fine-tune pools
hold rows of those titles: Memories Off series, teacher_train 66, real_train 70, teacher_distill_train 97,
real_unlabelled 86, realplus 8 ink + 6 GRAD (v23725); 428, 14 / 14 / 2 / 3; Eustia, 2 / 4 / 2 / 2;
街, 41 / 23 / 21 / 36 (ids in `$HERD/shared/artifacts/native-frames/owner_games_rows.json`). Every
candidate trained on the same pinned copies, so the comparisons here are unaffected, but any test on
the owner's captures of those titles must carry this caveat for every model trained on these pools (the
released v0.2.0 models included). The selection cohorts were cleaned before any selection score
(amendment 8: teacher_val's 8 lines and devnative v3's 14 crops of v1331; they held nothing of the
13:10Z titles). The native-frames pool (#66) excludes the titles and realplus v3 is being built to exclude them.

## Limits

- One machine and one seed for most candidates (the W20 class has one seed per arm; the FL10 winner's
  128k pretrain is a single seed): the development ranking inside a class is noisy at the 0.02-0.1
  point level; only the realplus decisions and the pretrain-length gain are resolved by intervals.
- teacher_val is 64% scaled PC screenshots; devnative v3 is 986 native crops from 43 VNs after amendment 8, assistant-
  reviewed. realplus's labels are engine-plus-assistant agreed, blind precision 495/500 (#57).
- Cortex-A9 times are host estimates from a cycle model; no console has run any of these models. vita-jp-
  overlay #68 (the fix for its issue #59) moved the overlay's worker out of SceShell; the preflight tier's free-memory check and
  fallback are the overlay's job, not this repo's.
- The fresh native textbox test and the CLANNAD long run need private inputs that are not on this
  machine; the locked report covers vita-test and psp-test only.
- The arch candidates EW (`--chans 64,128,256,240,288`) and the training-only `--rep 2`, both positive
  in 5k-step proxies (#58), and the FL10-class LWa4 shape were not trained in this round for lack of
  slot-hours; `--guide nrtr` was dropped by its proxy (+3 points worse).

## Reproduce

```sh
# runs (HEAVY_SLOTS=10 $HERD/heavy.sh ...): eval/vocr3_run.sh ARCH ARM SEED [PRE]; eval/vocr3_rp_run.sh ARCH ARM SEED [PRE]
# export and development scores: eval/synth3_ablation.py export / score (teacher_val, devnative's npz files)
# budget: vita/budget_sweep.py check --blob ... --out $S3/vocr3/budget
# selection: eval/vocr3_select.py realplus / base; spec: eval/vocr3_make_spec.py; report: eval/vocr3_report.py shared / a9 / overlay / render
```
Provenance, as run (2026-10-13, 64-core EPYC 9R14 box, 5 threads per run through heavy.sh):
- Candidate fine-tunes: study 2's runs (`eval/synth3_study2_run.sh`, feat/synth3-study2) and
  `eval/vocr3_run.sh` from detached worktrees at feat/vocr3 commits (tags `vocr3-runner-c464aa8`,
  `vocr3-runner-578821d`; the LW run from `vocr3-runner-lw-704ed0e` = feat/vocr3 e18c9a9 + PR #58's head
  a4b6202); rp runs from `vocr3-runner-578821d` through `study2_queue6.sh` and by hand for H15 B_s3 and
  LW; every run's `args.json` records its sources.
- Selection: the conductor's driver `eval/reproducibility/vocr3-20261013/select_size.sh` (export with
  `eval/synth3_ablation.py export`, scores with `eval/synth3_ablation.py score` on the amendment-8
  cohorts, `budget_check.sh` = `vita/budget_sweep.py check` from main fab5d3f, `eval/vocr3_select.py
  realplus` then `base`), outputs under `$HERD/shared/artifacts/synth3/vocr3/select/<SIZE>/`; every
  candidate's development scores (the tables above) are in `eval/vocr3_dev_scores_20261013.json`
  (`dev_scores.py`, reported only); the first H15 pass read the unfiltered cohorts by
  mistake, was superseded by the amendment-8 pass (commit d8ba6e8, 2994275 on this branch after the rebase) and is kept
  under `H15_unfiltered_mistake`.
- Report: `eval/reproducibility/vocr3-20261013/report.sh`: `eval/vocr3_make_spec.py`, then
  `eval/vocr3_report.py shared | a9 | overlay | render` with the arguments in `docs/vocr3-harness.md` (VitaSDK, qemu-arm plugins, Arm GNU toolchain 14.2, the overlay
  clone and CMake/Ninja of the harness verification); outputs under
  `$HERD/shared/artifacts/synth3/vocr3/report/out/`.
- The pre-rebase history of feat/vocr3 (amendments 1-7 with their original commit times) is at tag
  `vocr3-history-20261013a`.
