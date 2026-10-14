# VOCR4: the console-sized recognizers retrained with the new real data

VOCR4 trained VOCR3's three recognizers again (FL10, H15, W20) with the real data that arrived after VOCR3
and with synth3 version 3. It selected on development data only, under a protocol frozen before any run,
then scored the protected sets once through the plain-C W8I runtime. Every number below is a host
evaluation or a host estimate; nothing has run on a console.

**Results.**

- **W20 (the Large tier) improves.** synth3 version 3 in the pretrain was adopted at W20 (development score
  −0.137 points, stratified VN-cluster 95% CI −0.257 to −0.022, both seed pairs better).
  - Fixed lines (vita-test and psp-test pooled, console polarity): **1.40% CER** (71/5074) against VOCR3-W20's
    1.62% (−0.22 points, VN-cluster CI −0.53 to +0.09: better, not resolved).
  - Against the released models: VOCR_H15 −0.81 (CI −1.31 to −0.39), F20 −0.79 (CI −1.29 to −0.35). It is the
    lowest of every model measured here.
  - CLANNAD long run (3,048 lines with Claude references): **1.81%** against VOCR3-W20's 2.34% (−0.53, page CI
    −0.71 to −0.35), level with the cached Google Lens readings (1.47%; +0.33, CI −0.34 to +0.94).
- **FL10 and H15 (Small, Normal) stay on VOCR3's recipe.** Neither the new real pools nor synth3 v3 passed the
  decision rule at these sizes, so both selections are control runs: VOCR3's fine-tune on VOCR4's
  more strictly cleaned pools.
  - Fixed lines: FL10 **2.15%** (VOCR3-FL10 2.13%, +0.02, CI −0.18 to +0.26) and H15 **1.83%** (VOCR3-H15
    1.73%, +0.10, CI −0.11 to +0.34), level within the intervals.
  - CLANNAD: FL10 3.05% (−0.19 against VOCR3-FL10, CI −0.32 to −0.06) and H15 2.79% (−0.45, CI −0.61 to −0.29).
  - The 657 KB FL10 is level with the released 1.16 MB H15 on the fixed lines (−0.06, CI −0.37 to +0.26).
- **The new real data gave no resolved gain at any size.** Development score with realplus v3i, native pool v1
  and the owner's frames: FL10 −0.036 points, H15 −0.007, W20 +0.001. It trades the scaled PC screenshots for
  native console text. At H15, teacher_val got worse (+0.12, CI +0.01 to +0.23) and devnative better (−0.13,
  CI −0.31 to +0.03). At FL10 and W20, devnative's native Vita lines got better (−0.19 and −0.26, both resolved;
  report-only strata).
- **None qualifies under the repo's gates.** Each fails VOCR2's whole-dialogue guard, as VOCR3's did, and
  VOCR4-H15 also fails the fixed-line gate's hiragana condition (pooled 21 → 22 errors). The released defaults
  stay. On whole dialogue, VOCR4-W20 is level with VOCR_H15 (−0.24 points at the base region, CI −0.92 to
  +0.47).

All three fit the owner's final tiers (2026-10-14 06:21Z to 07:33Z; ceilings in model-tiers' PR #74, all
estimates). Each tier's limit is per typical full textbox (3 lines of about 640 px, one A9 core at 444 MHz):

| tier | ceiling | VOCR4 model | int8 weights | W8I arena | est. ms per 640-px line, 444 / 333 MHz | est. per 3-line box, 444 MHz |
|---|---|---|---:|---:|---|---|
| Small | < 1 MB, ≤ 1 s per box | VOCR4-FL10 | 657,664 B | 124,832 B | 94-179 / 125-239 | 0.28-0.54 s |
| Normal | ≤ 4 MB, ≤ 2 s per box | VOCR4-H15 | 1,155,968 B | 182,944 B | 166-302 / 221-403 | 0.50-0.91 s |
| Large | ≤ 9 MB, ≤ 3 s per box | VOCR4-W20 | 1,558,016 B | 241,056 B | 235-416 / 314-554 | 0.71-1.25 s |

They use under half of each tier's time. The section "The final round" below has the larger models the tiers
allow.

Machine-readable outputs:

- `eval/vocr4_protocol.json`: the protocol, with its one amendment.
- Pins: `eval/vocr4_inputs.sha256`, `eval/vocr4_pools_c.sha256`, `eval/vocr4_pools_a.sha256` and
  `eval/vocr4_pools_20261014.json`.
- Decisions: `eval/vocr4_{real,synth}_<SIZE>_20261014.json`.
- Selections: `eval/vocr4_selection_<SIZE>_20261014.json`.
- Report-only strata: `eval/vocr4_strata_*_20261014.json`.
- The locked report: `eval/vocr4_report_20261014.json` and `docs/vocr4-report-20261014.md`, with every table.
- The CLANNAD score: `eval/vocr4_clannad_20261014.json`.
- Weights: the pre-release `v0.2.5-vocr4.1` (`VOCR4_FL10_w8.vocr`, `VOCR4_H15_w8.vocr`, `VOCR4_W20_w8.vocr`; sha256 in the
  selections and the report spec). No released default changes.

## What was new

| data | what | used as |
|---|---|---|
| realplus v3 interim (`rp3i`) | realplus v2 plus the lines realplus-v3 had decided by 21:36Z on 2026-10-13 (Claude and Codex Luna agree): 48,479 lines of 9,499 VNs (v2 + 5,310), 79,644 ink and 58,735 GRAD crops; new lines' precision not measured yet | arm A's realplus |
| native pool v1 (`nat`, PR #66) | reviewed native Vita/PSP lines from VNDB screenshots: 542 ink + 402 GRAD crops of 185 VNs after deduplication | arm A |
| the owner's frames (`own`, PR #72) | the owner's own console captures, labelled by native-frames (Claude plus engine agreement, no Codex; blind precision 160/160): 594 ink + 540 GRAD crops, 595 lines: planetarian (v34, PSP) 323, CHAOS;CHILD (v14018, Vita) 227, Sharin no Kuni (v57, PSP) 45. Private: never committed | arm A |
| synth3 version 3 | study 1b's VN-style synthetic data (#56), which VOCR3 never trained on (`data/manifests/synth3_shards_v3.json`) | arm M's pretrain and fine-tune |

synth4 (#69) measured no gain over synth3 version 3, so there was no synth4 arm.

## Arms and runs

Each size had three arms of two seeds each, 18 runs in all plus six pretrains:

- **C, the control.** VOCR3's realplus fine-tune (`eval/vocr3_rp_run.sh`) on VOCR4's cleaned pools, from the
  pretrain of VOCR3's selection of that size (FL10 128,000 steps, H15 40,000, W20 32,000; reused and pinned).
  Fine-tune mix: nvsynth 0.25, synth2 0.125, synth3 v2 0.125, teacher_train 0.2, real_train 0.1, realplus v2 0.2.
- **A, the new real pools.** The same, with the real half split as teacher_train 0.175, real_train 0.1, realplus
  v3i 0.175, native pool v1 0.025 and owner frames 0.025.
- **M, synth3 version 3.** A new pretrain from scratch with synth3 v3 in place of v2 (nvsynth 0.5, synth2 0.25,
  synth3 v3 0.25; VOCR3's steps and schedule), then A's fine-tune with synth3 v3 in place of v2.

Every fine-tune is VOCR3's: 15,000 steps, batch 96, lr 1.2e-3, EMA 0.999, hard labels; the final EMA checkpoint
is the run. C and A start from one pretrain, so their seed varies the fine-tune only. Every run went through
`heavy.sh` at 5 threads, at most 8 at once, from detached worktrees: tag `vocr4-runner-1d8110c` (the C runs and
the M pretrains) and `vocr4-runner-d67ae32` (the A runs and the M fine-tunes). None failed or was restarted.

## Pools and hygiene

`eval/vocr4_pools.py` builds every training pool from one exclusion list of 10,487 VNs, computed on the
2026-10-13 VNDB dump. The list holds:

- every VN that `realplus/shots.reserved_vns()` reserves for a protected or development reason: CLANNAD with its
  family and relations; the owner's test series with their relations and release partners; eval's test games and
  `is_test_vn`; the fresh native titles; the development VNs;
- every VN that shares a VNDB release or a family relation with a CLANNAD, owner-test, test-set or fresh-native VN;
- the owner's 50 test VNs and every VN of teacher_val and devnative.

planetarian (v34) is the one reserved VN allowed, by the owner's rule of 20:47Z. VNs linked only to development
or hash-held-out VNs stay, as in the pools every released model trained on. This rule is stricter than VOCR3's
pools, which predate the owner's reservations and the release-partner rule:

| pool | rows before | after | dropped |
|---|---:|---:|---|
| teacher_train | 12,100 | 11,153 | 695 linked to a test game by a shared release or family relation, 146 owner test set, 68 development VNs, 38 sharing a release with CLANNAD |
| real_train | 9,795 | 9,005 | 570 test-game links, 137 owner test set, 51 development, 32 CLANNAD release partners |
| realplus v2, ink / GRAD | 71,153 / 53,566 | 71,142 / 53,558 | owner test set |
| realplus v3i, ink / GRAD | 79,644 / 58,735 | the same | none (realplus-v3's own hygiene) |
| native pool v1, ink / GRAD | 557 / 411 | 542 / 402 | 15 / 9 lines teacher_train or real_train already hold |
| owner frames, ink / GRAD | 594 / 540 | the same | none |

The new pools are deduplicated against teacher_train, real_train and each other, in the order native pool, owner
frames, realplus. A line goes when one of those already holds the same screenshot's line with the same label.
That is the protocol's **amendment 1** (commit `7e710ae`, 00:28Z, before any A pool was pinned or any run
scored). The frozen rule dropped every line of a screenshot those pools hold, which would have cut native pool
v1 from 557 lines to 101, although its other 456 are lines those pools' labelling left out, not duplicates. Line
ids could not serve as a key either, since each pipeline numbers a screenshot's lines its own way (58 of 59
shared ids carry different text). `eval/vocr4_pools.py check` printed CHECK_OK for all ten pools: no excluded VN,
no development VN or screenshot, no test-set screenshot or capture session. The control's pools were pinned with
the protocol; the new pools at 01:27Z (`eval/vocr4_pools_a.sha256`), before the first A run started (01:28:54Z)
and before the first score (01:29:02Z).

The cleaning has a measurable cost at FL10. The control's two FL10 runs score 2.48% and 2.51% on development data,
against 2.30% for VOCR3-FL10, which used the same pretrain on the uncleaned pools. At H15 and W20 the control
runs are 0.03 to 0.13 points behind VOCR3's. One seed of VOCR3 against two here cannot separate the pools'
effect from seed noise.

## The planetarian rule

The owner (2026-10-13 20:47Z) made planetarian training data and asked that its lines leave teacher_val,
devnative, vita-test and psp-test, in a new manifest version, with old and new scores side by side. There are
none. `eval/vocr4_planetarian_check_20261013.json` counts 0 rows of v34 or its VNDB relations (v27747, v5698),
by VN id and by any of their 84 releases, in:

- teacher_val and devnative v3, both the originals and the copies VOCR3 selected on;
- vita-test and psp-test.

So the planetarian-free versions of these sets are the current ones byte for byte, and every development and
test score in this document is both the old and the new score.

## Selection (development data only)

The cohorts were VOCR3's copies of teacher_val (1,643 lines) and devnative v3 (986 crops), pinned by sha256.
Scores came from `eval/synth3_ablation.py score`, the C runtime in W8I mode in both polarity views. A run's
development score is the mean of its teacher_val CER and its devnative CER. The decisions are paired over the
two seeds, with a stratified VN-cluster bootstrap: 10,000 resamples, with teacher_val's and devnative's VNs
resampled independently. A candidate arm is adopted only if all three of these hold:

- the interval of the development-score difference lies entirely below 0;
- both seed pairs improve;
- neither cohort's own interval lies entirely above 0.

| size | decision | development score | teacher_val | devnative | pairs (seed 1, seed 2) | adopted |
|---|---|---|---|---|---|---|
| FL10 | C → A | −0.036 (−0.121 to +0.048) | −0.029 (−0.116 to +0.057) | −0.044 (−0.188 to +0.102) | +0.037, −0.110 | no |
| FL10 | C → M | +0.043 (−0.118 to +0.194) | +0.070 (−0.078 to +0.226) | +0.015 (−0.270 to +0.269) | +0.043, +0.043 | no |
| H15 | C → A | −0.007 (−0.107 to +0.087) | +0.119 (+0.014 to +0.225) | −0.133 (−0.305 to +0.026) | +0.003, −0.017 | no |
| H15 | C → M | −0.017 (−0.155 to +0.112) | +0.012 (−0.115 to +0.139) | −0.046 (−0.295 to +0.175) | +0.173, −0.207 | no |
| W20 | C → A | +0.001 (−0.081 to +0.078) | +0.038 (−0.051 to +0.128) | −0.037 (−0.176 to +0.092) | −0.051, +0.052 | no |
| W20 | C → M | −0.137 (−0.257 to −0.022) | −0.104 (−0.260 to +0.042) | −0.170 (−0.362 to +0.007) | −0.152, −0.121 | **yes** |

The synth decision compares the real decision's arm, here C at every size, with M. M carries both synth3 v3 and
the new pools, but A alone is level with C at W20, so the W20 gain is most likely synth3 v3's. These report-only
per-tag strata (`eval/vocr4_strata_*`) decide nothing:

- C → A: native Vita lines on devnative improve at FL10 (−0.19, CI −0.28 to −0.03) and W20 (−0.26, −0.58 to −0.08);
  teacher_val's scaled PC lines get worse at H15 (+0.20, +0.06 to +0.34).
- C → M at H15: devnative's native Vita lines get worse (+0.51, +0.13 to +0.91). Study 1b saw the same direction
  at FL10.

Selected runs (every run listed, the candidates are the final arm's two runs; every blob passes
`vita/budget_sweep.py check`):

| size | selected | arm | teacher_val | devnative | development score | VOCR3's selection (same cohorts) |
|---|---|---|---|---|---|---|
| FL10 | `v4c_FL10_seed1` (`d7c4c08a…`) | C | 2.36% | 2.59% | 2.48% | 2.30% |
| H15 | `v4c_H15_seed1` (`d380d1a4…`) | C | 2.07% | 2.21% | 2.14% | 2.11% |
| W20 | `v4m_W20_seed2` (`2745e696…`) | M | 1.67% | 1.76% | 1.72% | 1.81% |

| run | arm | FL10 score | H15 score | W20 score |
|---|---|---|---|---|
| seed 1 | C | **2.48%** | **2.14%** | 1.91% |
| seed 2 | C | 2.51% | 2.24% | 1.84% |
| seed 1 | A | 2.51% | 2.15% | 1.86% |
| seed 2 | A | 2.40% | 2.22% | 1.89% |
| seed 1 | M | 2.52% | 2.32% | 1.76% |
| seed 2 | M | 2.56% | 2.03% | **1.72%** |

The lowest development scores of FL10 and H15 are runs of arms the decisions did not adopt (A seed 2 at 2.40%,
M seed 2 at 2.03%). Their other seed is worse than the control, and the protocol picks only within the adopted
arm, so these runs were never candidates.

## Locked report (once, after the three selections were committed)

`eval/vocr3_report.py` ran once:

- `shared` (the test-set scoring), 09:59-10:08Z on 2026-10-14, from the commit holding the three selections
  (`a13a31e`, tag `vocr4-report-runner-a13a31e`, committed and pushed 09:59Z);
- then `a9`, `overlay` and `render`.

The spec (`eval/vocr4_report_spec_20261014.json`, from `eval/vocr4_make_spec.py`) has the protocol's ten arms,
nine pairs and three gates. It ran the scalar C W8I runtime (gcc 11.5) on vita-test (206 lines, 16 games) and
psp-test (245 lines, 14 games), with paired bootstrap 95% intervals (10,000 resamples, seed 0) over lines and
VN clusters.

### Fixed lines (console polarity unless stated)

| arm | vita-test CER | psp-test CER | pooled CER | pooled hiragana CER | pooled exact | pooled CER, native polarity |
|---|---|---|---|---|---|---|
| VOCR1 | 2.38% (61/2564) | 5.30% (133/2510) | 3.82% (194/5074) | 1.61% (45/2799) | 71.6% | 3.72% |
| VOCR2 | 2.30% (59/2564) | 4.74% (119/2510) | 3.51% (178/5074) | 1.36% (38/2799) | 73.6% | 3.49% |
| VOCR_H15 | 1.56% (40/2564) | 2.87% (72/2510) | 2.21% (112/5074) | 0.75% (21/2799) | 82.5% | 2.09% |
| F20 | 1.72% (44/2564) | 2.67% (67/2510) | 2.19% (111/5074) | 1.21% (34/2799) | 82.3% | 2.29% |
| VOCR3-FL10 | 1.44% (37/2564) | 2.83% (71/2510) | 2.13% (108/5074) | 0.61% (17/2799) | 84.0% | 2.15% |
| VOCR3-H15 | 1.09% (28/2564) | 2.39% (60/2510) | 1.73% (88/5074) | 0.61% (17/2799) | 85.6% | 1.73% |
| VOCR3-W20 | 1.05% (27/2564) | 2.19% (55/2510) | 1.62% (82/5074) | 0.57% (16/2799) | 86.9% | 1.58% |
| **VOCR4-FL10** | 1.44% (37/2564) | 2.87% (72/2510) | **2.15%** (109/5074) | 0.57% (16/2799) | 83.4% | 2.11% |
| **VOCR4-H15** | 1.05% (27/2564) | 2.63% (66/2510) | **1.83%** (93/5074) | 0.79% (22/2799) | 85.6% | 1.83% |
| **VOCR4-W20** | 0.82% (21/2564) | 1.99% (50/2510) | **1.40%** (71/5074) | 0.46% (13/2799) | 88.7% | 1.46% |

| pair (console polarity; points, VN-cluster 95% CI) | vita-test | psp-test | pooled | pooled hiragana |
|---|---|---|---|---|
| VOCR4-FL10 − VOCR3-FL10 | +0.00 (−0.20 to +0.23) | +0.04 (−0.30 to +0.44) | +0.02 (−0.18 to +0.26) | −0.04 (−0.30 to +0.21) |
| VOCR4-FL10 − VOCR1 | −0.94 (−1.74 to −0.24) | −2.43 (−3.58 to −1.40) | −1.68 (−2.42 to −1.02) | −1.04 (−1.65 to −0.55) |
| VOCR4-FL10 − VOCR_H15 | −0.12 (−0.52 to +0.29) | +0.00 (−0.49 to +0.52) | −0.06 (−0.37 to +0.26) | −0.18 (−0.42 to +0.00) |
| VOCR4-H15 − VOCR3-H15 | −0.04 (−0.31 to +0.25) | +0.24 (−0.08 to +0.73) | +0.10 (−0.11 to +0.34) | +0.18 (−0.08 to +0.48) |
| VOCR4-H15 − VOCR_H15 | −0.51 (−1.08 to −0.08) | −0.24 (−0.74 to +0.20) | −0.37 (−0.74 to −0.05) | +0.04 (−0.28 to +0.34) |
| VOCR4-W20 − VOCR3-W20 | −0.23 (−0.60 to +0.15) | −0.20 (−0.75 to +0.30) | −0.22 (−0.53 to +0.09) | −0.11 (−0.43 to +0.15) |
| VOCR4-W20 − VOCR_H15 | −0.74 (−1.71 to −0.04) | −0.88 (−1.29 to −0.48) | −0.81 (−1.31 to −0.39) | −0.29 (−0.59 to −0.04) |
| VOCR4-W20 − F20 | −0.90 (−1.76 to −0.26) | −0.68 (−1.25 to −0.11) | −0.79 (−1.29 to −0.35) | −0.75 (−1.37 to −0.29) |
| VOCR4-W20 − VOCR4-H15 | −0.23 (−0.68 to +0.12) | −0.64 (−0.92 to −0.36) | −0.43 (−0.71 to −0.19) | −0.32 (−0.71 to +0.00) |

Per style (`docs/vocr4-report-20261014.md`, every row). VOCR4-W20 is the lowest or tied lowest on most styles:

| style (set) | VOCR4-W20 | VOCR3-W20 | VOCR_H15 |
|---|---|---|---|
| outlined (vita-test) | 0.51% | 1.01% | 2.12% |
| coloured (vita-test) | 2.05% | 2.56% | 6.67% |
| names (vita-test) | 1.16% | 2.91% | 5.23% |
| names (psp-test) | 6.70% | 7.66% | 9.09% |
| dark-on-light (psp-test) | 1.82% | 2.27% | 2.63% |

VOCR4-W20 is behind VOCR3-W20 on psp-test's text without a box (3 errors against 1, 59 characters). VOCR4-H15
is behind VOCR3-H15 on psp-test's names (21 errors against 19) and furigana (26 against 21). VOCR4-FL10 is
behind VOCR3-FL10 on psp-test's coloured (9 against 5) and small text (20 against 14), and ahead on names (19
against 24).

### Whole dialogue (finder-cut regions, joined; all text, pooled)

| arm | base region | expanded +4 px | expanded +8 px |
|---|---|---|---|
| VOCR1 | 660/5016 (13.16%) | 816/5016 (16.27%) | 866/5016 (17.26%) |
| VOCR_H15 | 539/5016 (10.75%) | 694/5016 (13.84%) | 738/5016 (14.71%) |
| F20 | 530/5016 (10.57%) | 688/5016 (13.72%) | 717/5016 (14.29%) |
| VOCR3-FL10 | 588/5016 (11.72%) | 728/5016 (14.51%) | 791/5016 (15.77%) |
| VOCR3-H15 | 554/5016 (11.04%) | 713/5016 (14.21%) | 761/5016 (15.17%) |
| VOCR3-W20 | 526/5016 (10.49%) | 693/5016 (13.82%) | 743/5016 (14.81%) |
| VOCR4-FL10 | 582/5016 (11.60%) | 746/5016 (14.87%) | 804/5016 (16.03%) |
| VOCR4-H15 | 547/5016 (10.91%) | 704/5016 (14.04%) | 772/5016 (15.39%) |
| VOCR4-W20 | 527/5016 (10.51%) | 678/5016 (13.52%) | 751/5016 (14.97%) |

| pair, all text, pooled (points; VN-cluster 95% CI) | base | expanded +4 | expanded +8 |
|---|---|---|---|
| VOCR4-FL10 − VOCR1 | −1.56 (−2.40 to −0.78) | −1.40 (−2.39 to −0.40) | −1.24 (−2.24 to −0.26) |
| VOCR4-FL10 − VOCR3-FL10 | −0.12 (−0.48 to +0.22) | +0.36 (+0.00 to +0.73) | +0.26 (−0.19 to +0.70) |
| VOCR4-H15 − VOCR_H15 | +0.16 (−0.42 to +0.74) | +0.20 (−0.36 to +0.76) | +0.68 (−0.08 to +1.51) |
| VOCR4-W20 − VOCR_H15 | −0.24 (−0.92 to +0.47) | −0.32 (−0.95 to +0.32) | +0.26 (−0.51 to +1.05) |
| VOCR4-W20 − VOCR3-W20 | +0.02 (−0.55 to +0.61) | −0.30 (−0.76 to +0.16) | +0.16 (−0.44 to +0.82) |

As in VOCR3, the fixed-line gains do not carry over to the finder-cut regions. On whole dialogue every model
from H15 up is level with VOCR_H15. The selection cohorts are hand-cut lines, so this gap remains the open
question; a whole-dialogue development cohort is needed to select for the overlay's actual reading path.

### Gates

| gate | fixed-line gate (console) | whole-dialogue guard | qualifies |
|---|---|---|---|
| VOCR4-FL10 vs VOCR1 | pass: vita-test 61→37 (hiragana 15→6), psp-test 133→72 (30→10), pooled hiragana 45→16 | FAIL: 77 of 360 title checks regress | no |
| VOCR4-H15 vs VOCR_H15 | FAIL: psp-test hiragana 13→15, pooled hiragana 21→22 (vita-test 40→27, psp-test 72→66) | FAIL: 129 of 360 | no |
| VOCR4-W20 vs VOCR_H15 | pass: vita-test 40→21 (8→4), psp-test 72→50 (13→9), pooled hiragana 21→13 | FAIL: 114 of 360 | no |

No candidate qualifies, so the released defaults stay. The guard is strict by design: any of 30 titles × 3
variants × 2 scopes × 2 metrics regressing by one error fails it.

### Size, speed and the overlay cross-check

Sizes and arenas are exact. The A9 times are `vita/a9_cycles.py` host estimates on 40 synth-val lines (W8I), not
measurements. They are identical to VOCR3's and the released models' of the same shape, because the kernels
have no data-dependent branches:

| model | weights | arena | est. ms per 640-px line at 444 MHz | at 333 MHz |
|---|---:|---:|---|---|
| FL10 | 657,664 B | 124,832 B | 94-179 | 125-239 |
| H15 | 1,155,968 B | 182,944 B | 166-302 | 221-403 |
| W20 | 1,558,016 B | 241,056 B | 235-416 | 314-554 |

The overlay host cross-check (vita-jp-overlay `1a118ef`'s `tools/benchmark_local_ocr.py`, read-only) gives the same
all-text errors as this harness's base-region whole dialogue for every arm and set (20 of 20 cells equal). Its
workspace peak was 0.87 MiB for FL10, 1.40 MiB for H15 and 1.84 MiB for W20 on vita-test (0.79, 1.32 and 1.76
MiB on psp-test).

## CLANNAD (once, against owner-drive-ingest's frozen references)

owner-drive-ingest built and froze its references first: Claude references read twice, 3,048 scored lines of
the long run's 4,271 paired crops, `refs_sha256` `35411688…`, frozen 03:04Z on 2026-10-14, protocol
`eval/clannad_refs_protocol.json`. Only after that were the three VOCR4 selections read, once, through its
extra-arms path (`eval/clannad_refs.py arms --extra ... --suffix vocr4`: the same stored crops, the C runtime in
W8I mode, invert=False). They were scored once (`score --suffix vocr4`, 10:35-10:36Z), beside its other arms and
the cached Google Lens readings. Intervals come from 10,000 whole-page bootstraps over 2,044 pages. The units are
the long run's processed GRAD-finder crops, not the hand-cut test lines, so these numbers are not comparable to
the tables above.

| arm | CER [95% CI] | exact lines | hiragana errors |
|---|---|---|---|
| VOCR1 | 4.56% [4.29, 4.83] | 63.3% | 736 / 20,120 |
| VOCR2 | 4.14% [3.89, 4.41] | 66.2% | 652 / 20,120 |
| VOCR_H15 | 2.82% [2.61, 3.04] | 75.5% | 474 / 20,120 |
| F20 | 2.75% [2.56, 2.94] | 74.7% | 551 / 20,120 |
| VOCR3-FL10 | 3.24% [2.96, 3.51] | 75.1% | 477 / 20,120 |
| VOCR3-H15 | 3.24% [2.96, 3.52] | 77.4% | 375 / 20,120 |
| VOCR3-W20 | 2.34% [2.13, 2.55] | 80.8% | 299 / 20,120 |
| **VOCR4-FL10** | 3.05% [2.79, 3.32] | 76.0% | 456 / 20,120 |
| **VOCR4-H15** | 2.79% [2.52, 3.06] | 79.6% | 452 / 20,120 |
| **VOCR4-W20** | 1.81% [1.62, 2.00] | 84.9% | 241 / 20,120 |
| Google Lens (cached) | 1.47% [0.88, 2.14] | 97.5% | 105 / 20,120 |

| pair | CER, points [95% CI] | exact lines, points [95% CI] |
|---|---|---|
| VOCR4-FL10 − VOCR3-FL10 | −0.19 [−0.32, −0.06] | +1.0 [−0.1, +2.0] |
| VOCR4-H15 − VOCR3-H15 | −0.45 [−0.61, −0.29] | +2.3 [+1.1, +3.4] |
| VOCR4-W20 − VOCR3-W20 | −0.53 [−0.71, −0.35] | +4.1 [+2.8, +5.4] |
| VOCR4-FL10 − VOCR1 | −1.51 [−1.74, −1.27] | +12.8 [+11.2, +14.4] |
| VOCR4-H15 − VOCR_H15 | −0.03 [−0.24, +0.18] | +4.2 [+2.7, +5.7] |
| VOCR4-W20 − VOCR_H15 | −1.01 [−1.21, −0.82] | +9.5 [+7.9, +11.0] |
| VOCR4-FL10 − Lens | +1.57 [+0.89, +2.20] | −21.4 [−22.9, −20.0] |
| VOCR4-H15 − Lens | +1.31 [+0.63, +1.94] | −17.8 [−19.3, −16.4] |
| VOCR4-W20 − Lens | +0.33 [−0.34, +0.94] | −12.5 [−13.9, −11.3] |

**Disclosure.** VOCR4-W20 (arm M) trained on the owner's planetarian PSP frames. planetarian's PSP release and
CLANNAD's Vita release are both Key / Prototype ports, so that model's CLANNAD number carries this exposure.
VOCR4-FL10 and VOCR4-H15 (arm C) never read the owner's frames. VOCR4-W20 is the only recognizer here below 2%
CER on these references; Lens's lead in exact lines (97.5% against 84.9%) is still large.

## Hygiene and timeline (2026-10-13 to 14, UTC)

| when | what |
|---|---|
| 22:16:58Z | protocol committed and pushed (`1d8110c`, tag `vocr4-runner-1d8110c`) |
| 22:20:30Z | first run (the M pretrains, the C fine-tunes) |
| 00:28:40Z | amendment 1 (`7e710ae`): new pools deduplicated by line; from the build's dry-run counts only |
| 01:27:12Z | the new pools pinned (`d67ae32`, tag `vocr4-runner-d67ae32`) |
| 01:28:54Z / 01:29:02Z | first A run / first score file |
| 02:47Z, 04:57Z, 06:35Z, about 09:00Z, 09:56Z | decisions: FL10 real, H15 real, H15 synth, W20 real and synth, FL10 synth |
| 06:38Z, 09:06Z, 09:59Z | selections committed: H15, W20, FL10 (pushed 09:59Z) |
| 09:59:41Z to 10:29Z | locked report (spec, shared, a9, overlay, render); CLANNAD 10:35-10:36Z |

- The branch was rebased onto main after these commits, so their hashes changed. The original commits are
  kept at tags `vocr4-runner-1d8110c`, `vocr4-runner-d67ae32`, `vocr4-history-20261014a` and
  `vocr4-report-runner-a13a31e`, and the author dates are the times above.
- No arm, run, weight, checkpoint, threshold or pool was chosen, tuned or dropped on vita-test, psp-test, the fresh
  native textbox test or CLANNAD. Each was scored once, after every selection was committed. train2's in-run
  evaluations (teacher_val, logged) selected nothing and were not read before the new pools were pinned.
- The owner's frames and every CLANNAD reading stay local, under `shared/private`. The committed files hold counts,
  ids and hashes only.
- The reused pretrains and synth3's shards predate the owner's reservations of 10:51Z and 13:10Z and hold some
  material of those titles (`docs/owner-games.md`, `docs/vocr3.md`). Any test on the owner's captures of Memories
  Off, Aiyoku no Eustia or 428 must carry that caveat for every VOCR4 model.

## Limits

- Two seeds per arm. C and A share one pretrain per size, so their seed covers fine-tune variance only. M's
  pretrain seeds vary, which widens its intervals. Only W20's synth decision is resolved.
- The new real data is small next to the existing pools: 5,310 new realplus lines and about 2,100 native crops,
  against 145,000 crops. Its weights (0.05 for the native data) were set once, before any score; no other split
  was tried.
- teacher_val is 64% scaled PC screenshots; devnative is 986 native crops from 43 VNs. The development score
  weighs them equally.
- Whole dialogue is not a selection cohort, and the gains do not carry over to it (above).
- Nothing ran on a console. Every time is a host estimate.

## The final round: three tiers within the owner's ceilings

The owner's final deliverable is three models: Small (< 1 MB weights, ≤ 1 s per 3-line box), Normal (≤ 4 MB,
≤ 2 s) and Large (≤ 9 MB, ≤ 3 s). Each box is about 640 px a line, one A9 core at 444 MHz, estimated.
model-tiers (#74) maps VOCR4 onto them as Small = FL10, Normal = H15, Large = W20, and proposes one larger shape
per tier, with exact export sizes:

| tier | shape | int8 weights | est. per box, 444 MHz | what it adds |
|---|---|---:|---|---|
| Small | `s_w100v4987`: FL10 with a 4,987-class VN-ordered head | 819,792 B | 0.32-0.61 s | 22× fewer missing VN characters than FL10's 3,500 classes (#65) |
| Normal | `n_w250v5631`: width 2.5 with a 5,631-class head | 2,095,712 B | 0.96-1.82 s | width, every VN-list kanji |
| Large | `l_w300v7501b192`: width 3.0, bottleneck 192, 7,501 classes, 16-frame tiles | 3,689,840 B | 1.5-2.9 s | width and a parameter-heavy head; plain width ≥ 3.25 would be compute-bound |

VOCR4's evidence for that round:

- Use synth3 version 3 in the pretrain at the larger sizes: it was the only resolved gain here, at W20.
- Keep VOCR3's real mix, with the new native data as a test arm only: no size showed a resolved gain from it,
  and at H15 it cost scaled-PC accuracy.
- A new vocabulary needs synthetic data rendered with it and a real-pool relabel (the pools were filtered to
  H15's vocab). vocab-expand's study decides that.
- Each shape needs its own pre-registered protocol and new pretrains; the larger ones are roughly 13 to 40 h
  per run on this box at 5 threads.

## Reproduce

```sh
# pools: eval/vocr4_pools.py exclusions | build (--tag c, then --tag a with --new nat_* own_* rp3i_* --ref tt --ref rd) | check
# runs:  eval/reproducibility/vocr4-20261013/queue.sh (jobs.txt: eval/vocr4_run.sh SIZE ARM SEED via heavy.sh) + feeder.sh
# selection: eval/reproducibility/vocr4-20261013/select_size.sh SIZE export|score|real|synth|budget|select|strata X Y
# report: eval/reproducibility/vocr4-20261013/report.sh spec|shared|a9|overlay|render|clannad
# tables: eval/reproducibility/vocr4-20261013/doc_tables.py
```

Artifacts: `$HERD/shared/artifacts/vocr4-train/` (runs, blobs, pools, select, report, logs), and
`$HERD/shared/private/vocr4-train/`, which holds the owner-frame pool copies and the CLANNAD readings and is
private.
