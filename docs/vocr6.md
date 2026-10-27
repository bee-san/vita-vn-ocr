# VOCR6: the recognizers retrained on the owner's hooked PS Vita and PSP corpus

Issue #125, track T1 of the VOCR6 plan. VOCR6 keeps VOCR5's shapes for Small, Normal (the default) and Large, so each
tier's weights, arena and compute per frame stay VOCR5's; only the speed sections appended to the blobs differ (below).
What changes is the data: each tier's VOCR5 model is fine-tuned again on the owner's hooked corpus (exact game text on
native frames, docs/vocr6-data.md) and on the finder's own crops of those pages. Since the follow-up (pre-release
v0.4.1-vocr6.2), every tier ships as VOCR6; v0.4.0-vocr6.1 shipped Normal and Large, with Small still VOCR5-S. Every
number here was measured on the host. Console times are estimates.
**Nothing of VOCR6 has run on a console.**

**Summary.**
- **Every tier is VOCR6 in v0.4.1-vocr6.2** (the follow-up, below): Small from stage 4d, Normal and Large from stage
  4c (the shipped recipe trained 3x longer). Each passed its registered rule on Rubi's shipped path.
  - **What ships:** `VOCR6.2_S_w8.vocr` (819,792 B, plain), `VOCR6.2_N_w8g2.vocr` (1,189,408 B, head gate + two-level
    head) and `VOCR6.2_L_w8_2l.vocr` (2,475,680 B, two-level head).
  - **Est. 0.63 / 0.77 / 1.88 s per 3-line box** on one 333 MHz A9 core. Small lost VOCR5-S's head gate (+0.11 s);
    Normal is 15 ms faster and Large 8 ms slower than v0.4.0-vocr6.1's.
  - **The follow-up's final test, read once:** CLANNAD lines 0.47 / 0.33 / 0.31% (VOCR5: 0.73 / 0.66 / 0.60%, all
    resolved better); vita-test + psp-test fixed lines 0.58 / 0.36 / 0.40% (VOCR5: 0.68 / 0.49 / 0.47%, level). Against
    v0.4.0-vocr6.1, CLANNAD's pages read worse at Normal and Large (+0.33, +0.27 points, resolved).
  - **Disclosed:** Small's adopted arm was the third recipe tried at Small, and its development score is worse in point
    (+0.064, not resolved). In the set as it ships it reads +0.041 [-0.016, +0.099] against VOCR5-S.
- **v0.4.0-vocr6.1 (stage 4a/4b): Normal and Large VOCR6, Small VOCR5-S.** Each tier was adopted over the released
  VOCR5 tier only if it read the hooked games' held-out pages better, with every seed pair below 0, and other games
  no worse. That was checked on Rubi's shipped path, with three seeds.
- **Normal and Large on the hooked games' pages** (hval, held out): 1.008 → 0.936% and 0.990 → 0.917% dialogue CER,
  both resolved better. On other games they are level: -0.038 and +0.026 points on VOCR5's development score.
- **Small** read the hooked pages better too (-0.134), but VOCR5's development score on other games a little worse
  (+0.070 [+0.004, +0.138], resolved; no part of it resolved alone), so stage 4a did not adopt it. Stage 4d then did.
- **What v0.4.0-vocr6.1 ships:** `VOCR6_N_w8_2l.vocr` (1,187,616 B) and `VOCR6_L_w8_2l.vocr`
  (2,475,680 B). They have VOCR5's shapes, with #122's two-level head appended; the head gate was rejected on
  VOCR6's logits. Est. 0.79 s and 1.87 s per 3-line box on one 333 MHz A9 core, about 30 and 80 ms more than VOCR5
  as Rubi runs it today. W8I arena 184,400 B and 262,480 B.
- **Its final test, read once:**
  - CLANNAD lines: Normal 0.66 → 0.44%, Large 0.60 → 0.36%, resolved better. CLANNAD's glyph raster is in the
    training data, so these are not unseen-game numbers.
  - vita-test + psp-test: level.
  - The hooked test title: level or better.
- **The owner's < 1% target** (page path): met on the hooked games' held-out pages by Normal and Large (0.94% /
  0.92%; Small 1.05%). Not met on other games: whole dialogue 2.47 / 2.37 / 2.39% for the follow-up's set (v0.4.0's
  4a means 2.43% / 2.46%), about 80% of it the line finder's. The spotter tracks (#128, #149) target that.


## What was decided, and how

Every comparison was registered in `eval/vocr6_protocol.json` before its runs were scored. The amendments are listed in
the protocol with their times, and each one also has a `preregistration` line in the fleet's experiment log. The rule
was the same at every stage (`decision.adopt`):

- **Better on hval.** The interval of mean(Y) - mean(X) on hval page-path dialogue CER lies below 0, and every seed
  pair's point difference is below 0. hval is a held-out block of each train title's pages (below).
- **Not worse on other games.** The interval of the development score does not lie above 0. That score is VOCR5's:
  0.25 teacher_val + 0.25 devnative + 0.5 whole dialogue, all on games outside the hooked corpus, in the console view.
- **Intervals** are 95% bootstrap intervals with 10,000 resamples, clustered by title (hval) or by VN (dev).

**The shipped form decides.** Each candidate was exported to the int8 W8I blob and read through the C runtime, with
Rubi's per-line decode (greedy plus top-8, the counter re-decode, beam 8 with a 1 MB LM, textnorm and the non-Japanese
filter). Stages 1-3 used ctclm.2's 1 MB LM at 500 / 3500 / EOS (Rubi's setting then); stage 4 used lm6's game-text 1 MB
LM at 600 / 2500 / BOS (Rubi's now). Stages 1-3 read without the cascade. Stage 4 read Rubi's whole shipped path: the COLGAP finder
default with its short-line probe (#135), the 1 MB game-text LM at alpha 600 / beta 2500 / BOS (#137), and the cascade
S -> N -> L. The cascade is read without Rubi's per-job time ceiling (`ocr_cascade_ms`, 2,000 ms by default), as #119
read it: every line whose greedy confidence is under 0.7 is re-read by the next tier, so the numbers show the
cascade's full effect. X and Y always use the same rule. Each tier's 4a runs cascade into the same seed's VOCR6 tier
above. Where an upper tier is not adopted, the tier below is also reported cascading into the upper tier as shipped.

**hval page** is the decision metric. It covers every hval page's whole-dialogue regions (base, +4 and +8 px), cut by
Rubi's line finder in the console view (PSP pages upscaled 2x). Merged lines, missed lines and the finder's imperfect
crops all count as errors. Two kinds of unit are left out:

- units where a finder box covers no reference line (reported with them as `hval_page_all`; 113 of 6,327 units on plan v2);
- units holding a line that also occurs in a train page of the same title (354 units; reported as `hval_page_seen`).

## Data

- **The hooked corpus** (docs/vocr6-data.md; vocr6-ingest's pools v1, manifest `ba95ff2d6c31`). It has 49 titles
  (33 Vita, 16 PSP), 55,493 native pages and 81,977 distinct dialogue lines (1,293,093 characters). Each label is the
  game's own hooked text. A label audit by eye found 0 wrong lines in the 318 that remain after cleaning.
- **Splits** (pages; `splits_v1.json`, `417df3d784cb`):
  - train: 50,272.
  - hval: 2,193 (3.95%), contiguous blocks of each train title with a 15-page guard band either side (guard: 2,338).
    hval is for screening only: the shipped models train on it (stage 4b).
  - test: 690 pages (1.24%), the one test title, Dance with Devils (PCSG00807).
  - So the shipped models hold out the test title and CLANNAD only, which is within AGENTS.md's 10% budget.
- **Console crops (crop-in-the-loop).** vocr6-finder's line finder ran on the train pages as Rubi runs it. Its crops
  were matched to the verified lines and labelled with the hooked text (data/console_crops.py, #136). There are two
  kinds, whole-line crops and jittered boxes. The screening arms used crops v1 (`4adb00c9`, finder fix 5): 98,324
  whole-line crops and 304,524 jittered boxes from train pages. The shipped models use crops v2 (`45cbddaa`), cut with
  the COLGAP default Rubi ships (#142), from train, hval and guard pages.
- **VOCR5's real pools**, as pinned by `eval/vocr5_pools.sha256`: teacher-labelled VNDB screenshots, the realplus
  pools, the owner's earlier console captures and the native cohort. The screening copies leave out the few rows whose
  label equals an hval line of the same title (`eval/vocr6_guard.py`: 75 ink / 62 GRAD rows of the owner pool, 1
  teacher row); the shipped models use the pools as pinned.
- **Synthetic replay.** VOCR5's synthetic half is kept unchanged: NVIDIA's OCR-Synthetic-Multilingual crops 0.225,
  synth2 0.1125, synth3 v3 0.1125 and synth4 0.05.

## The recipe

The shipped recipe, called R3, is A15 + S2cil.

- **Init:** each tier's own VOCR5 fine-tuned checkpoint, `v5f_{S,N,L}_seed{1,2,3}/ckpt_step15000.pt`. The released
  VOCR5 blobs are v5f_S_seed2, v5f_N_seed3 and v5f_L_seed2.
- **Schedule:** 15,000 steps, batch 96, AdamW, lr 1.2e-3 with 500 warmup steps, cosine decay to 5%, weight decay 0.01,
  EMA 0.999. The model is the final step: no checkpoint selection.
- **Mix** `r3all`, which equals `m1_cil`:

  | source | weight | what |
  |---|---:|---|
  | nv, s2, s3v3, s4 | 0.225, 0.1125, 0.1125, 0.05 | synthetic half, as VOCR5-N's fine-tune |
  | tt, rd, rpi, rpg, owi, owg, nti, ntg | 0.25 in total | VOCR5's real pools at half their VOCR5 weights |
  | hk | 0.15 | hooked lines and names of the train pages, console polarity |
  | cc | 0.06 | console crops: whole lines |
  | ccj | 0.04 | console crops: jittered boxes |

- **Shipped blobs (4b)** retrain seed 1 of each tier on `r3all_fold`. This is the same recipe with hval folded back in:
  the hooked train + guard + hval pages, console crops v2 of the same pages, and VOCR5's real pools as pinned.

## Screening (Normal tier, two seeds per arm)

All rows are read on hval plan v2 (Rubi's finder with the short-line probe, finder fix 5), through Rubi's per-line
decode, with no cascade. CER is in percent; differences are in CER points with 95% intervals.

### Stage 1: initialisation (2026-10-26)

| arm | what | hval page | hval clean | dev score | Δ hval page | Δ dev | decision |
|---|---|---:|---:|---:|---|---|---|
| VOCR5-N | released (v5f_N_seed3) | 1.209 | 0.564 | 1.990 | | | incumbent |
| A15 | fine-tune VOCR5-N's checkpoint, 15k steps | 1.076 | 0.391 | 2.061 | -0.133 [-0.249, -0.031] vs VOCR5-N | +0.070 [-0.021, +0.176] | **adopted** |
| B15 | VOCR5-N's 150k synthetic pretrain, then fine-tune | 1.069 | 0.416 | 2.161 | -0.009 [-0.033, +0.017] vs A15 (exposed units out) | +0.101 [-0.040, +0.285] | not (hval page not resolved better; clean +0.025 resolved worse) |

- **C (fresh mixed training) was stopped (amendment 4).** At the box's loaded rate of 0.93-1.0 steps/s per slot, C50
  takes ~15 h and C150 ~45 h per run, so its rung could not change this round's decision. The A45 and B45 rungs were
  not run either; the long budget moved to stage 3 (S3long).

### Stages 2 and 3: data and capacity, one factor at a time from A15

| arm | what | hval page | hval clean | dev score (wd) | Δ hval page vs A15 | Δ dev | slot-h | decision |
|---|---|---:|---:|---:|---|---|---:|---|
| A15 | base | 1.076 | 0.391 | 2.061 (3.40%) | | | 6.7 | |
| S2cil | console crops at 0.10 of the hooked 0.25 | 1.032 | 0.407 | 2.034 (3.36%) | -0.044 [-0.073, -0.017] | -0.027 [-0.111, +0.071] | 5.8 | **adopted** |
| S2t5 | per-title temperature 0.5 | 1.085 | 0.406 | 2.031 (3.39%) | +0.009 [-0.020, +0.035] | -0.030 [-0.086, +0.029] | 6.5 | not |
| S2x2 | PSP lines in the Adrenaline 2x view | 1.094 | 0.409 | 2.003 (3.35%) | +0.018 [-0.004, +0.042] | -0.057 [-0.125, +0.016] | 6.6 | not |
| S2s6 | synth6 v3 (game fonts) at 0.15 | 1.047 | 0.370 | 1.939 (3.26%) | -0.029 [-0.060, +0.001] | -0.121 [-0.261, +0.068] | 6.6 | not |
| S2r70 | real share 70% | 1.058 | 0.380 | 2.014 (3.34%) | -0.018 [-0.042, +0.005] | -0.047 [-0.119, +0.019] | 6.6 | not |
| S2r30 | real share 30% | 1.111 | 0.411 | 2.038 (3.35%) | +0.035 [-0.006, +0.079] | -0.022 [-0.075, +0.030] | 5.7 | not |
| S2fa2 | finder augmentation x2 | 1.091 (seed 1) | 0.417 | 2.069 (3.48%) | +0.003 (seed-1 pair) | -0.024 (seed-1 pair) | 3.2 | not (the every-pair condition failed; seed 3 not run) |
| S3attn | one global attention layer (#130) | 1.059 | 0.398 | 2.054 (3.47%) | -0.017 [-0.049, +0.014] | -0.006 [-0.088, +0.085] | 6.5 | not |
| S3rep | RepCBR branches, training only | 1.097 | 0.403 | 2.001 (3.28%) | +0.021 [-0.005, +0.048] | -0.059 [-0.134, +0.011] | 9.5 | not |
| S3long | A15 at 45,000 steps | 1.033 | 0.373 | 2.028 (3.40%) | -0.043 [-0.077, -0.014] | -0.033 [-0.116, +0.065] | 19.7 | **adopted** (decides stage 4c only) |

- **Not run.** Names in or out: names are 0.9% of rows, and the owner crops them out. Vertical re-flow: one title,
  and no vertical model this round. Width/depth: a wider model needs a fresh 150k-step pretrain per seed, outside this
  round's CPU budget. QAT: int8 costs about 0 (#119). S3gtc (an auxiliary NRTR loss at weight 0.1) was dropped for
  compute; its prior at weight 1.0 was +3.08 at 5k steps.
- **S2s6** (synth6 v3, the harvested game fonts) has the biggest dev gain, -0.121 points, but its hval interval
  reaches +0.001, so it is not adopted.
- **S3long** (3x the steps) is adopted over A15 on both seeds: hval page -0.043 [-0.077, -0.014], seed pairs -0.054
  and -0.032, dev -0.033 [-0.116, +0.065]. It decides stage 4c only (amendment 6): a 45k-step retrain of the
  shipped tiers, which replaces a shipped blob only if it is not resolved worse on dev than that blob.

## Stage 4: the shipped tiers of v0.4.0-vocr6.1 (2026-10-27)

**What ran** (protocol stage 4, amendments 1-6; tag `vocr6-runner-00b8288`):
- **4a:** R3 at each tier, three seeds, hval held out:
  - Small and Large: seeds 1-3, each from the tier's own VOCR5 fine-tuned checkpoint (Small's seeds 2 and 3 both
    start from `v5f_S_seed2`, the released one, with training seeds 2 and 3);
  - Normal: S2cil's two runs (seeds 1 and 3) and a new seed 2.
- **4b:** seed 1 of each tier, retrained on the same recipe with hval folded back in. These are the candidate
  shipped blobs.

**How it was read.** Every blob went through Rubi's shipped path:
- **The finder:** hval plan v3 and dev plan v2, both cut by Rubi's finder with the COLGAP default and the short-line
  probe.
- **The LM:** the 1 MB game-text LM at alpha 600 / beta 2500 / BOS. For hval it is the screening build h64_1000k,
  which holds no hval or guard text; for dev it is the build Rubi ships.
- **The cascade** into the same seed's VOCR6 tier above. X is the released VOCR5 tier, cascading into VOCR5's.

Each tier was adopted over its released VOCR5 tier under `decision.adopt`, with three seed pairs.

| tier | model | hval page | hval clean | dev score | dev wd | tv | dn |
|---|---|---:|---:|---:|---:|---:|---:|
| S | VOCR5-S (released) | 1.177 | 0.523 | 1.674 | 2.704 | 0.726 | 0.561 |
| S | v6f4_S_seed1 | 1.022 | 0.382 | 1.741 | 2.732 | 0.802 | 0.698 |
| S | v6f4_S_seed2 | 1.051 | 0.417 | 1.738 | 2.744 | 0.764 | 0.698 |
| S | v6f4_S_seed3 | 1.055 | 0.366 | 1.752 | 2.748 | 0.854 | 0.661 |
| S | **R3 mean of 3** | **1.043** | **0.389** | **1.744** | **2.741** | **0.807** | **0.686** |
| N | VOCR5-N (released) | 1.008 | 0.404 | 1.482 | 2.484 | 0.548 | 0.411 |
| N | v6s2cil_N_seed1 | 0.918 | 0.321 | 1.464 | 2.448 | 0.484 | 0.474 |
| N | v6f4_N_seed2 | 0.946 | 0.331 | 1.444 | 2.441 | 0.459 | 0.436 |
| N | v6s2cil_N_seed3 | 0.946 | 0.321 | 1.422 | 2.397 | 0.433 | 0.461 |
| N | **R3 mean of 3** | **0.936** | **0.325** | **1.443** | **2.429** | **0.459** | **0.457** |
| L | VOCR5-L (released) | 0.990 | 0.390 | 1.439 | 2.405 | 0.535 | 0.411 |
| L | v6f4_L_seed1 | 0.894 | 0.288 | 1.471 | 2.488 | 0.420 | 0.486 |
| L | v6f4_L_seed2 | 0.922 | 0.314 | 1.458 | 2.425 | 0.522 | 0.461 |
| L | v6f4_L_seed3 | 0.937 | 0.296 | 1.466 | 2.472 | 0.459 | 0.461 |
| L | **R3 mean of 3** | **0.917** | **0.299** | **1.465** | **2.462** | **0.467** | **0.470** |

| tier | Δ hval page (R3 − VOCR5) | Δ hval clean | Δ dev score | Δ dev wd | seed pairs (hval page) | adopted |
|---|---|---|---|---|---|---|
| S | -0.134 [-0.198, -0.079] | -0.135 [-0.200, -0.078] | +0.070 [+0.004, +0.138] | +0.037 [-0.067, +0.139] | -0.155, -0.126, -0.122 | no |
| N | -0.072 [-0.111, -0.038] | -0.079 [-0.134, -0.034] | -0.038 [-0.125, +0.048] | -0.055 [-0.207, +0.091] | -0.090, -0.062, -0.062 | **yes** |
| L | -0.073 [-0.118, -0.036] | -0.091 [-0.154, -0.039] | +0.026 [-0.049, +0.104] | +0.056 [-0.065, +0.182] | -0.096, -0.068, -0.054 | **yes** |

dev score = 0.25 tv + 0.25 dn + 0.5 wd (other games: teacher_val and devnative fixed lines, whole dialogue).

- **Normal and Large are adopted. Small is not.**
  - The recipe reads the hooked titles' pages better at every tier: 0.07-0.13 points of page-path CER, every seed
    pair below 0.
  - Small's dev interval lies above 0 (+0.070 [+0.004, +0.138]), so the rule keeps VOCR5-S (until stage 4d, below).
  - Small loses a little on every part of the development score, none resolved alone: teacher_val +0.081, devnative
    +0.125, whole dialogue +0.037.
  - The likely cause: the recipe gives VOCR5's real pools half their VOCR5 weight to make room for the hooked data,
    and Small has the least capacity to keep both.
- **The 4b check** (the 4b blob against its 4a sibling: dev not resolved worse; hval is in 4b's training, so its hval
  number is reported only):

| tier | 4b (hval folded in) vs its 4a sibling | Δ dev score | Δ dev wd | hval page, 4b (in training) | check |
|---|---|---|---|---:|---|
| N | v6f4h_N_seed1 vs v6s2cil_N_seed1 | -0.013 [-0.093, +0.074] | -0.031 [-0.173, +0.113] | 0.893 | passes |
| L | v6f4h_L_seed1 vs v6f4_L_seed1 | -0.029 [-0.106, +0.051] | -0.114 [-0.243, +0.026] | 0.883 | passes |

  Both pass, so the 4b blobs ship. Folding hval in did not lower the hval page CER much: 0.918 → 0.893 (Normal),
  0.894 → 0.883 (Large). Most of that error is the finder's, and more lines do not fix the finder.
- **What ships together:** Small VOCR5-S → Normal VOCR6 → Large VOCR6, Rubi's cascade S → N → L.
  - With VOCR6-N above it, VOCR5-S reads dev at 1.667 (1.674 with VOCR5-N above it).
  - VOCR6-N (4b) cascading into VOCR6-L (4b) reads dev at 1.451 (VOCR5-N into VOCR5-L: 1.482).
  - VOCR6-L (4b) reads dev at 1.442 (VOCR5-L: 1.439).

**The gap to the owner's < 1% target** (dialogue CER through Rubi's full page path):
- **The hooked games' pages** (hval, held out, 4a means): Normal 0.936%, Large 0.917%, so both are under 1%. About a
  third of it is the recognizer: 0.325% / 0.299% on clean lines. The rest is the finder: merged and missed lines,
  and cut crops.
- **Other games** (#119's whole-dialogue development units, 4a means): Normal 2.43%, Large 2.46% (VOCR5: 2.48% /
  2.41%; the shipped 4b blobs: 2.42% / 2.37%). Their
  fixed lines read at 0.46-0.47%, so about 80% of the whole-dialogue error is the finder's.
- So the remaining gap is the finder. textspot7 and textspot8 (#128, #149) target it.

### Speed: the head gate and the two-level head (vocr6-final-gate, #146; v0.4.0-vocr6.1)

Both are speed-only sections, fit on each shipped blob's own outputs under #120's and #122's pre-registered rules
(`eval/vocr6_gate_protocol.json`). Times are estimates from ocr-speed's method: `vita/a9_layers.py` under qemu,
calibrated to VOCR3-H15's measured console rate, one Cortex-A9 core, no prefetch credit, a nominal 3-line box.

| tier | shipped blob | bytes | W8I arena @1,024 px | head gate | two-level head | est. s per box, 333 / 444 MHz (plain → shipped) |
|---|---|---:|---:|---|---|---|
| Small (VOCR5-S, kept) | `VOCR5_S_w8g.vocr` (v0.3.5-headgate.1) | 821,584 | 124,880 | yes | no | 0.63 / 0.50 → **0.52 / 0.41** |
| Normal | `VOCR6_N_w8_2l.vocr` | 1,187,616 | 184,400 | **no** | yes (K 500, r96) | 0.92 / 0.71 → **0.79 / 0.62** |
| Large | `VOCR6_L_w8_2l.vocr` | 2,475,680 | 262,480 | **no** | yes (K 500, r96) | 2.07 / 1.62 → **1.87 / 1.46** |

- **The head gate is rejected for Normal and Large.** On VOCR6's sharper logits, #120's ε = 1e-3 gate makes the
  greedy development score resolved worse: Normal +0.021 [+0.004, +0.042], Large +0.047 [+0.003, +0.102]. The
  ε = 1e-4 fallback stays behind the switch. Normal fails the cascade margin. Large's saving on development
  whole dialogue is 4.98% of a nominal box, below the 5% bar.
- **The two-level head is adopted for both.**
  - It is not resolved worse on the development rule. On Rubi's shipped path (vocr6-lead's kit), hval page is
    +0.005 points for both tiers ([-0.004, +0.016] for Normal), and dev is +0.003 (Normal) and 0.000 (Large).
  - It saves an est. 129 ms (Normal) and 202 ms (Large) per nominal box at 333 MHz.
- **Against VOCR5 as Rubi ships it today**, an estimate derived from #120's and #122's published savings: VOCR5-N
  and VOCR5-L run with both sections, at about 0.76 s and 1.79 s per box at 333 MHz. VOCR6's tiers are about 30 ms
  (Normal) and 80 ms (Large) slower per box, still well inside the tiers' ceilings (Normal ~1 s, Large < 3 s).
- Rubi runs a blob without a gate section on its two-level head alone under `ocr_head_gate = on` (the default).
- Weights and arena are VOCR5's tiers' plus the sections: +1,984 B and +1,456 B arena for the two-level trailer.


### LM settings on VOCR6's logits (vocr6-final-lm; v0.4.0-vocr6.1)

Rubi's decoder settings were chosen on VOCR5's logits. They were re-checked on the shipped Normal and Large under a
pre-registered rule (`eval/vocr6_final_lm_protocol.json`): keep the shipped settings unless a grid setting (2-fold
CV, as #127) or a counter threshold is resolved better on the objective and not resolved worse on any guard,
#109's synthetic counter set included.
- **The default 1 MB LM:** the settings are kept for both tiers.
- **Large:** kept at all four LM tiers.
- **Normal at the 2 MB tier** (ctclm.2 2000k, a user option) changes to alpha 400 / beta 1500 / BOS + EOS / gate
  8000. Its CV objective is -0.260 (-0.800 to -0.006). Most of that is one junk line of PCSG00854; without it the
  gain is -0.027 (-0.058 to -0.000).
- **The counter re-decode thresholds** stay at 6 / 4 nats.
- **On #109's counter set,** the 1 MB LM keeps the counter in 3,322 (Normal) and 3,326 (Large) of 3,969 crops,
  against 3,064 for VOCR5-N with its cascade.


### Final test of v0.4.0-vocr6.1 (read once, report only)

vocr6-final-test (PR #140: its protocol `eval/vocr6_final_test_protocol.json`, driver and records) read
the protected sets once, after every choice above. Nothing below changed a choice. The shipped blobs were read on
two paths:
- **(b) Rubi's shipped line path:** Rubi main 892cfb9; the lm6 1 MB LM at 600 / 2500 / BOS; the COLGAP finder;
  the cascade S -> N -> L; textnorm and the line filter;
- **(a) VOCR5's locked path,** to compare with VOCR5's published numbers. VOCR5 reproduces its records in 29 of 29
  checks.

Small is VOCR5-S in both columns. On path (b) it differs only through its cascade into Normal.

| set (path b, dialogue CER) | tier | VOCR5 | VOCR6 | VOCR6 − VOCR5 |
|---|---|---:|---:|---|
| CLANNAD, 2,052 lines | Small | 0.73% | 0.68% | -0.05 [-0.12, +0.02] |
| | Normal | 0.66% | **0.44%** | **-0.22 [-0.30, -0.15]** |
| | Large | 0.60% | **0.36%** | **-0.24 [-0.31, -0.17]** |
| CLANNAD, 1,837 pages through Rubi's finder | Small | 5.84% | 6.16% | **+0.32 [+0.14, +0.50]** |
| | Normal | 5.94% | **5.47%** | **-0.46 [-0.63, -0.30]** |
| | Large | 6.19% | **5.40%** | **-0.78 [-0.97, -0.60]** |
| vita-test + psp-test, 336 fixed lines | Small | 0.68% | 0.81% | +0.13 [-0.09, +0.41] |
| | Normal | 0.49% | 0.49% | +0.00 [-0.20, +0.18] |
| | Large | 0.47% | 0.43% | -0.04 [-0.19, +0.10] |
| vita-test + psp-test, 175 whole-dialogue regions | Small | 3.04% | 3.26% | +0.22 [-0.11, +0.61] |
| | Normal | 2.83% | 2.63% | -0.19 [-0.51, +0.11] |
| | Large | 2.72% | 2.52% | -0.19 [-0.55, +0.15] |
| Dance with Devils (PCSG00807), 1,152 clean lines | Small | 0.27% | 0.24% | -0.03 [-0.07, +0.01] |
| | Normal | 0.26% | 0.21% | **-0.05 [-0.10, -0.01]** |
| | Large | 0.22% | 0.21% | -0.01 [-0.06, +0.02] |
| Dance with Devils, 685 page regions | Small / Normal / Large | 0.85 / 0.84 / 0.81% | 0.84 / 0.80 / 0.80% | every interval includes 0 |

Intervals: games (vita-test, psp-test) or pages (CLANNAD, Dance with Devils), paired.

- **On path (a)** (greedy, VOCR5's published path), Normal and Large gain more:
  - CLANNAD 0.96 → 0.50% and 0.73 → 0.31%;
  - vita-test + psp-test fixed lines 1.13 → 0.83% (-0.30 [-0.73, +0.02]) and 0.79 → 0.53% (-0.26 [-0.49, -0.07]);
  - their whole dialogue 7.98 → 7.21% (Normal, -0.78 [-1.15, -0.38]) and 7.08 → 6.86% (Large).

  With beam + ctclm.2's 1 MB LM, the fixed lines are level: 0.72 → 0.70% and 0.53 → 0.49%.
- **Vita and PSP apart** (path b, fixed lines): vita-test is better in point (Normal -0.21 [-0.56, +0.05], Large
  -0.17 [-0.40, +0.00]; neither resolved). psp-test does not (Normal +0.22 [+0.00, +0.41], Large +0.09 [-0.12, +0.30]). On path (a)
  with the LM, psp-test's Normal is resolved worse: +0.30 [+0.12, +0.53].
- **Small's cascade into VOCR6-N costs on CLANNAD's pages** (+0.32, resolved), though not on the development sets
  (dev 1.674 → 1.667).
  - The decomposition places the loss after the cascade: before textnorm and the line filter, the two read 6.46%
    and 6.51%.
  - So VOCR6-N's readings of the low-confidence boxes that Small hands it (many of them junk boxes) more often hold a
    Japanese character, which the line filter keeps.
  - Speaker names show it most: CLANNAD names +3.76 points for Small, while Normal and Large read names better
    (-2.54 / -3.67).
  - **A follow-up to check on development data:** only take the upper tier's text when it is more confident than the
    lower tier.
- **Seen flags** (every number above carries them):
  - every VOCR6 model has seen CLANNAD's glyph raster (two hooked train titles);
  - Little Busters! and planetarian share the Key 10th Memorial BOX with CLANNAD, and the lm6 LMs hold their text;
  - Collar x Malice shares releases with vita-test's Collar x Malice -Unlimited-: VOCR6 trained on it, and its text
    is in the lm6 LMs;
  - the LMs' Jiten word list includes the scripts of CLANNAD, of 7 of vita-test's 16 games and of 1 of psp-test's 14.

  So **CLANNAD's gain is not an unseen-game result.** The development sets are the unseen-game evidence, and there
  VOCR6 is level.


## The follow-up: stages 4c and 4d (v0.4.1-vocr6.2)

After v0.4.0-vocr6.1 shipped, two registered stages decided a second set. Every tier changes:
- **Normal and Large** come from stage 4c: the shipped recipe trained 3x longer (45,000 steps).
- **Small** becomes VOCR6 too, from stage 4d: the recipe at half the learning rate.

The rules are the protocol's (`eval/vocr6_protocol.json` stages.4 amendments 6 and 7, with audit-vocr6-lead-h H1's
open cases in amendment 6). vocr6-lead made every decision (experiment log, 2026-10-27 15:25-18:24Z). vocr6-followup
then ran the gate fits and the final-test read below. Nothing in this section is held out on hval: every blob that
ships trained on hval too.

### Stage 4c: Normal and Large at 45,000 steps

**What ran** (amendment 6; tag `vocr6-runner-a6430de`): F4Nc_s1 and F4Lc_s1. Each is 4b's recipe (mix `r3all_fold`,
hval folded in, lr 1.2e-3, warmup 500, init the tier's VOCR5 seed-1 checkpoint) at 45,000 steps instead of 15,000.
The runs are `v6f4c_N_seed1` (10.3 slot-hours) and `v6f4c_L_seed1` (13.7 slot-hours). Small's 4c run was stopped
unscored when Small was not adopted in 4a (H1: 4c is used only where 4b ships).

**The rule.** A 4c blob replaces the shipped 4b blob iff:
- (i) S3long is adopted over A15 on stage 3's held-out pair. It is: hval page -0.043 [-0.077, -0.014] on both seeds
  (stage 3, above).
- (ii) the 4c blob is not resolved worse than its 4b blob on the shipped-path development score (one pair).

H1 fixed the composition: 4c-N is read cascading into 4c-L, against 4b-N into 4b-L. The hval comparison is reported
only, since both blobs trained on hval. H1's cases were logged at 07:56Z, before any 4c score; they enter the protocol
file (`stages.4.amendments[5].open_cases`) with this follow-up's records.

| tier | 4c (45,000 steps, hval in) vs the shipped 4b | Δ dev score | Δ tv | Δ dn | Δ dev wd | Δ hval page (both in training) | rule (ii) |
|---|---|---|---|---|---|---|---|
| N | v6f4c_N_seed1 (into 4c-L) vs v6f4h_N_seed1 (into 4b-L) | -0.063 [-0.148, +0.020] | -0.064 [-0.191, +0.048] | -0.100 [-0.324, +0.080] | -0.043 [-0.166, +0.077] | -0.014 [-0.055, +0.023] | passes |
| L | v6f4c_L_seed1 vs v6f4h_L_seed1 | -0.046 [-0.132, +0.034] | -0.115 [-0.269, +0.014] | -0.100 [-0.267, +0.060] | +0.016 [-0.114, +0.137] | -0.016 [-0.057, +0.025] | passes |

Both pass, so 4c's blobs replace 4b's. Their development scores are lower in point estimate (-0.06 / -0.05), but not
resolved: these are improvements within the noise.

### Stage 4d: Small

**Why.** 4a's Small (R3-S) read the hooked pages better but VOCR5's development score worse (+0.070 [+0.004,
+0.138]). The hypothesis (amendment 7): R3 halves VOCR5's real pools to make room for the hooked data, and Small, the
smallest model, loses some of what they taught it.

**What ran** (amendment 7; tags `vocr6-runner-64dd023` and `vocr6-runner-49854df` for the fill): two arms, three
seeds each, otherwise as 4a's Small.
- **4d-r** (`v6f4dr_S`) puts VOCR5's real pools back at full weight (mix `r3all_v5r`).
- **4d-l** (`v6f4dl_S`) keeps R3 at half the learning rate (6e-4).

Each arm was compared with the released VOCR5-S under `decision.adopt`, on stage 4's shipped path, with Small
cascading into the same seed's VOCR6-N.

| arm | model | hval page | hval clean | dev score | dev wd | tv | dn |
|---|---|---:|---:|---:|---:|---:|---:|
| - | VOCR5-S (released) | 1.177 | 0.523 | 1.674 | 2.704 | 0.726 | 0.561 |
| 4d-r | v6f4dr_S_seed1 | 1.009 | 0.398 | 1.787 | 2.818 | 0.739 | 0.773 |
| 4d-r | v6f4dr_S_seed2 | 1.059 | 0.421 | 1.752 | 2.767 | 0.777 | 0.698 |
| 4d-r | v6f4dr_S_seed3 | 1.065 | 0.378 | 1.710 | 2.649 | 0.879 | 0.661 |
| 4d-r | **mean of 3** | **1.044** | **0.399** | **1.750** | **2.745** | **0.798** | **0.711** |
| 4d-l | v6f4dl_S_seed1 | 1.028 | 0.390 | 1.729 | 2.752 | 0.764 | 0.648 |
| 4d-l | v6f4dl_S_seed2 | 1.059 | 0.406 | 1.720 | 2.696 | 0.841 | 0.648 |
| 4d-l | v6f4dl_S_seed3 | 1.060 | 0.372 | 1.764 | 2.783 | 0.802 | 0.686 |
| 4d-l | **mean of 3** | **1.049** | **0.389** | **1.738** | **2.744** | **0.803** | **0.661** |

| arm | Δ hval page (4d − VOCR5-S) | Δ hval clean | Δ dev score | Δ dev wd | seed pairs (hval page) | seed pairs (dev) | adopted |
|---|---|---|---|---|---|---|---|
| 4d-r | -0.133 [-0.188, -0.085] | -0.124 [-0.186, -0.072] | +0.076 [+0.005, +0.149] | +0.041 [-0.067, +0.148] | -0.169, -0.118, -0.112 | +0.113, +0.079, +0.036 | no |
| 4d-l | -0.128 [-0.189, -0.074] | -0.134 [-0.196, -0.081] | +0.064 [-0.009, +0.138] | +0.039 [-0.078, +0.151] | -0.149, -0.118, -0.117 | +0.055, +0.047, +0.090 | **yes** |

- **4d-r is not adopted:** its dev interval lies above 0.
- **4d-l is adopted:** hval page below 0 with every pair below 0, and its dev interval includes 0 ([-0.009,
  +0.138]), so the rule calls it not worse. **Disclosed:**
  - its dev point estimate is worse (+0.064, every seed pair above 0);
  - it was the second Small arm of 4d, after 4d-r failed, and the third recipe tried at Small (after 4a's R3-S).
    Amendment 7 registered both 4d arms together before any 4d run, but with more arms tried, a "not resolved worse"
    pass by chance is likelier than with one.
- **The fold-in** (the fill, registered after these decisions and before its run): F4Sdlh_s1 retrains 4d-l's seed 1
  with hval folded in (`r3all_fold` at 6e-4, 15,000 steps, 1.9 slot-hours; run `v6f4dlh_S_seed1`). It ships iff its
  dev score is not resolved worse than 4d-l's seed 1.

| check | compared | Δ dev score | Δ tv | Δ dn | Δ dev wd | Δ hval page | result |
|---|---|---|---|---|---|---|---|
| 4d-l's fold-in vs its 4a sibling | v6f4dlh_S_seed1 (into 4b-N) vs v6f4dl_S_seed1 (into v6s2cil_N_seed1) | -0.041 [-0.137, +0.049] | -0.089 [-0.207, +0.036] | -0.075 [-0.207, +0.029] | +0.000 [-0.175, +0.154] | -0.004 [-0.061, +0.043] (hval in the fold-in's training) | passes |
| Small in the follow-up's cascade | v6f4dlh_S_seed1 vs VOCR5-S, both into 4c-N | +0.041 [-0.016, +0.099] | +0.000 [-0.067, +0.074] | +0.012 [-0.105, +0.115] | +0.075 [-0.015, +0.171] | -0.064 [-0.117, +0.006] (hval in the new Small's training) | report only |

- The fold-in passes, so `v6f4dlh_S_seed1` replaces VOCR5-S. The check compares two cascades (into the 4b-N that
  shipped then, against into v6s2cil_N_seed1), as the fill registered.
- **Disclosed (report only):** in the set as it ships (both Smalls cascading into 4c-N), the new Small reads dev
  +0.041 [-0.016, +0.099] against VOCR5-S. Not resolved, but worse in point.
  - It is worse in point in either cascade: +0.021 into 4b-N, +0.041 into 4c-N. The difference is whole dialogue's
    (+0.067 / +0.075 points there; +17 / +19 errors, spread over VNs, the largest v2817 with +7). teacher_val and
    devnative move little (into 4c-N: +0.000 and +0.012; into 4b-N: -0.025 and -0.025).
  - Moving both Smalls' cascade from 4b-N to 4c-N lowers both scores by 0.14-0.16 points (VOCR5-S 1.667 → 1.509,
    74 fewer whole-dialogue errors; the new Small 1.688 → 1.549, 72 fewer). 33 of VOCR5-S's 74 are one VN's junk-box
    text (v21004), the display-rule effect the final test saw on CLANNAD's pages (above, in v0.4.0-vocr6.1's final
    test). That swing is the same for both Smalls, so it does not explain the difference between them.

**The development score of each set as it ships** (Rubi's cascade S -> N -> L; hval is in the training of every VOCR6
blob here):

| tier | VOCR5 (released) | v0.4.0-vocr6.1 | v0.4.1-vocr6.2 (the follow-up) |
|---|---|---|---|
| Small | VOCR5-S into VOCR5-N: 1.674 | VOCR5-S into 4b-N: 1.667 | v6f4dlh_S_seed1 into 4c-N: **1.549** (VOCR5-S into 4c-N: 1.509) |
| Normal | VOCR5-N into VOCR5-L: 1.482 | 4b-N into 4b-L: 1.451 | 4c-N into 4c-L: **1.389** |
| Large | VOCR5-L: 1.439 | 4b-L: 1.442 | 4c-L: **1.396** |

**The gap to the owner's < 1% target** (dialogue CER through Rubi's full page path):
- **Other games:** whole dialogue on #119's development units reads 2.47 / 2.37 / 2.39% (S / N / L, the set as it
  ships), against VOCR5's 2.70 / 2.48 / 2.41%. The same tiers read those games' fixed lines at 0.39-0.68%, so about
  74-83% of the remaining whole-dialogue error is the line finder's (1 - the mean of teacher_val's and devnative's
  fixed-line CER over whole dialogue, per tier). Rubi's new default finder, the textspot8 spotter
  (#149), targets it; the GRAD path measured here is not Rubi's default any more.
- **The hooked games' pages:** the held-out evidence is the recipes' seeds above: Normal 0.936% and Large 0.917%,
  under 1%; Small 1.049%. The shipped blobs' own hval reads (0.98 / 0.88 / 0.87%) are not held out.

### Speed: the follow-up's gate fits

vocr6-followup ran vocr6-final-gate's driver (main's `eval/vocr6_gate.sh`, #146) on the three blobs, in the
protocol's order L, N, S. Each tier's VOCR6-path arm cascades into the next tier as it ships. The settings are those
of vocr6-final-gate's final run: the VOCR6 path reads hval with the screening LM h64_1000k and dev with the shipped
lm6 1000k, at alpha 600 / beta 2500 / BOS. The rules are #120's and #122's, as `eval/vocr6_gate_protocol.json`
freezes them. The fits' steps took 2.29 slot-hours of compute; the three jobs held 2.56 slot-hours. Times are
estimates, by the same method as above.

| tier | shipped blob (gate driver's name → release name) | bytes | sha256 | W8I arena @1,024 px | head gate | two-level head | est. s per box, 333 / 444 MHz (plain → shipped) |
|---|---|---:|---|---:|---|---|---|
| Small | `VOCR6_S_w8.vocr` → `VOCR6.2_S_w8.vocr` (plain) | 819,792 | `864bbd196d5bc5d3…` | 124,832 | **no** | - | 0.63 / 0.50 (no change) |
| Normal | `VOCR6_N_w8g2.vocr` → `VOCR6.2_N_w8g2.vocr` | 1,189,408 | `107cf084cc0c1a3a…` | 184,448 | **yes, ε = 1e-4** | yes (K 500, r96) | 0.92 / 0.71 → **0.77 / 0.61** |
| Large | `VOCR6_L_w8_2l.vocr` → `VOCR6.2_L_w8_2l.vocr` | 2,475,680 | `65310013e8f99088…` | 262,480 | **no** | yes (K 500, r96) | 2.07 / 1.62 → **1.88 / 1.47** |

Development score, gated − ungated, in points [95% VN-cluster interval]; the rule's margin is +0.05:
- **Small: no gate.**
  - The ε = 1e-3 gate is not resolved worse on any decode and saves 18-19% of a box. But its upper bounds exceed the
    +0.05 margin: rubi +0.019 [-0.063, +0.125] and rubi_cascade +0.029 [-0.016, +0.074]. So it stays behind the
    switch.
  - The ε = 1e-4 fallback is rejected: on the cascade decode its whole-dialogue component is resolved worse, +0.051
    [+0.006, +0.101].
  - So Small ships plain. At est. 0.63 s per box it is about 0.11 s slower than VOCR5-S as Rubi runs it today (gated:
    0.52 s), still well inside Small's < 1 s ceiling.
- **Normal: the ε = 1e-4 gate and the two-level head.**
  - The ε = 1e-3 gate is rejected: greedy +0.026 [+0.006, +0.052], resolved worse.
  - The ε = 1e-4 fallback passes every rule and the VOCR6-path check: greedy +0.001 [+0.000, +0.003], rubi_cascade
    +0.008 [-0.006, +0.022]. It saves 9-12% of a box.
  - The two-level head on top of it passes too: against the ε = 1e-4 gate, rubi +0.004 [+0.000, +0.010]; on the
    VOCR6 path, dev +0.003 [+0.000, +0.009].
  - So VOCR6-N's follow-up blob runs at est. 0.77 s per box (v0.4.0's VOCR6-N: 0.79 s).
- **Large: the two-level head alone, as in v0.4.0.**
  - The ε = 1e-3 gate is rejected (greedy +0.027 [+0.006, +0.051]; on the VOCR6 path dev +0.041 [+0.001, +0.087]).
  - The ε = 1e-4 fallback saves only 2.9-4.0% of a box, under the 5% bar.
  - The two-level head is adopted: on the VOCR6 path, hval page +0.004 [+0.000, +0.009] and dev 0.000. It saves an
    est. 194 ms per box. VOCR6-L runs at est. 1.88 s per box (v0.4.0's: 1.87 s).
- **The release names** add `.2` to the family: `VOCR6.2_*`. One file name then always means one set of bytes;
  v0.4.0-vocr6.1's `VOCR6_N_w8_2l.vocr` and `VOCR6_L_w8_2l.vocr` keep theirs.
- **The record.** The driver's `result.json` records `repo_head` 0d3dffe (reachable from the tag
  `vocr6-followup-read-27e0126`): main 2e63b74 plus feat/vocr6-s4d's two protocol commits, which the clone was
  switched to during the run. Every file the driver reads is main's, its protocol sha256 `491e3865…` included.

### The follow-up's final test (read once, report only)

The protected sets were read once for the new set, after every choice above and after its gate fits, with
vocr6-final-test's harness (amendment 9 of `eval/vocr6_final_test_protocol.json`, committed before the read).
Nothing below changed a choice.
- **The set, as it ships:** VOCR6-S (plain) → VOCR6-N (head gate + two-level head) → VOCR6-L (two-level head),
  VOCR5 beside it.
- **The paths are the earlier read's.** Path (b) reads Rubi at 892cfb9, the Rubi main of v0.4.0-vocr6.1's read.
  The report calls it "Rubi main", as it was then. The LM is lm6's 1 MB at 600 / 2500 / BOS, with the COLGAP
  finder, the cascade S -> N -> L, textnorm and the line filter.
- **A check that the two reads share one path:** VOCR5 reads every unit of every headline measure exactly as in the
  earlier read (0 units differ, `compare.json`). Its path (a) reproduces the published records in 29 of 29 checks.
- **Rubi main has moved since.** Its default finder is now the textspot8 spotter (#396), which this harness does not
  model. vocr6-spotter2 reports the spotter's own test read (`eval/textspot8_test_20261027.json`).

Every number is dialogue CER on path (b), with paired intervals (games for vita-test / psp-test, pages for CLANNAD
and Dance with Devils). Records: `eval/vocr6_final_test_followup_20261027.json` and `..._compare_20261027.json`.

| set (path b, dialogue CER) | tier | VOCR5 | v0.4.0-vocr6.1 | **v0.4.1-vocr6.2** | v0.4.1 − VOCR5 | v0.4.1 − v0.4.0 |
|---|---|---:|---:|---:|---|---|
| CLANNAD, 2,052 lines | Small | 0.73% | 0.68% | 0.47% | **-0.26 [-0.35, -0.18]** | **-0.21 [-0.30, -0.12]** |
|  | Normal | 0.66% | 0.44% | 0.33% | **-0.33 [-0.41, -0.24]** | **-0.10 [-0.17, -0.04]** |
|  | Large | 0.60% | 0.36% | 0.31% | **-0.29 [-0.37, -0.21]** | -0.05 [-0.10, +0.01] |
| CLANNAD, 1,837 pages through Rubi's finder | Small | 5.84% | 6.16% | 5.75% | -0.09 [-0.28, +0.10] | **-0.41 [-0.61, -0.21]** |
|  | Normal | 5.94% | 5.47% | 5.81% | -0.13 [-0.31, +0.05] | **+0.33 [+0.19, +0.48]** |
|  | Large | 6.19% | 5.40% | 5.67% | **-0.52 [-0.71, -0.32]** | **+0.27 [+0.13, +0.41]** |
| vita-test + psp-test, 336 fixed lines | Small | 0.68% | 0.81% | 0.58% | -0.11 [-0.33, +0.11] | **-0.23 [-0.46, -0.04]** |
|  | Normal | 0.49% | 0.49% | 0.36% | -0.13 [-0.33, +0.04] | -0.13 [-0.26, +0.00] |
|  | Large | 0.47% | 0.43% | 0.40% | -0.06 [-0.18, +0.04] | -0.02 [-0.14, +0.11] |
| vita-test + psp-test, 175 whole-dialogue regions | Small | 3.04% | 3.26% | 3.02% | -0.02 [-0.45, +0.45] | -0.24 [-0.58, +0.06] |
|  | Normal | 2.83% | 2.63% | 2.70% | -0.13 [-0.55, +0.27] | +0.06 [-0.33, +0.48] |
|  | Large | 2.72% | 2.52% | 2.68% | -0.04 [-0.47, +0.39] | +0.15 [-0.17, +0.52] |
| Dance with Devils (PCSG00807), 1,152 clean lines | Small | 0.27% | 0.24% | 0.21% | **-0.06 [-0.11, -0.01]** | -0.03 [-0.06, +0.00] |
|  | Normal | 0.26% | 0.21% | 0.19% | **-0.06 [-0.12, -0.02]** | -0.01 [-0.04, +0.00] |
|  | Large | 0.22% | 0.21% | 0.19% | **-0.04 [-0.08, -0.01]** | -0.02 [-0.05, +0.00] |
| Dance with Devils, 685 page regions | Small / Normal / Large | 0.85 / 0.84 / 0.81% | 0.84 / 0.80 / 0.80% | 0.85 / 0.80 / 0.79% | every interval includes 0 | every interval includes 0 |

In bold: resolved (the interval excludes 0).

- **Small** is now better than VOCR5-S or level on every set. The CLANNAD-pages loss of v0.4.0's composition
  (VOCR5-S cascading into VOCR6-N: +0.32, resolved) is gone: -0.09 against VOCR5, -0.41 against v0.4.0's Small.
- **Normal and Large read CLANNAD's lines better** than v0.4.0's (-0.10, resolved, and -0.05) **but its pages worse** (+0.33 and
  +0.27, both resolved). Against VOCR5 the pages are still better or level (-0.13, and -0.52 resolved).
  - Large has no cascade, and its loss sits in the counter + LM stage: its greedy pages read level with v0.4.0's
    (6.07% / 6.10%), but the counter and LM gain less (6.03% against 5.78%).
  - Normal's gap, from the error counts (30,436 reference characters), is +0.34 points greedy (8.29% / 7.94%), +0.07
    after the counter and LM, +0.24 after the cascade and +0.33 as shown: it comes from its greedy reads and from the
    cascade.
  - Lines and pages differ in their crops. The pages are the finder's crops of whole ROIs, junk boxes included; the
    lines are the frozen line crops.
- **vita-test + psp-test:** level against VOCR5 at every tier on path (b). vita-test is better (Small -0.25
  [-0.55, -0.03], Normal -0.33 [-0.66, -0.11], both resolved); psp-test is not (+0.04 / +0.09 / -0.04).
  - On path (a), with ctclm.2's LM, psp-test's Normal is resolved worse: +0.26 [+0.04, +0.51] (v0.4.0: +0.30).
  - On path (a) greedy, whole dialogue gains: Normal 7.98 → 7.34% (-0.65 [-1.14, -0.16]), Large 7.08 → 6.58% (-0.50
    [-0.88, -0.11]).
- **Seen flags,** on every number above as in the earlier read:
  - every VOCR6 model has seen CLANNAD's glyph raster (two hooked train titles);
  - Little Busters! and planetarian share the Key 10th Memorial BOX with CLANNAD, and the lm6 LMs hold their text;
  - Collar x Malice shares releases with vita-test's Collar x Malice -Unlimited-: VOCR6 trained on it, and its text
    is in the lm6 LMs;
  - the LMs' Jiten word list includes the scripts of CLANNAD, of 7 of vita-test's 16 games and of 1 of psp-test's 14.

  So **CLANNAD's gain is not an unseen-game result.** The development sets are the unseen-game evidence. There the
  follow-up's development scores are 1.389 (Normal) and 1.396 (Large), against VOCR5's 1.482 and 1.439 (point
  estimates; the registered comparisons are against 4b, above, and neither is resolved). Small's set reads 1.549,
  and +0.041 [-0.016, +0.099] against VOCR5-S in the same cascade.

## Reproduce

```sh
# a run (the protocol's arms; checkpoints, blob and the plan-v2 read under $HERD/shared/artifacts/<label>/)
git checkout vocr6-runner-00b8288        # stage 4 (4c: vocr6-runner-a6430de; 4d: vocr6-runner-64dd023, its fold-in -49854df)
eval/vocr6_run.py command --stage 4 --arm F4Nh_s1 --seed 1 --label <label>    # prints the train2 command
eval/vocr6_run.py run     --stage 4 --arm F4Nh_s1 --seed 1 --label <label>    # checks every pin, trains, exports, scores
# the shipped path: dumps per plan, then the score with the cascade into the next tier's dumps
eval/vocr6_score.py read  --plan hval_plan_v3.pkl --blob B --lm h64_1000k.vlm --lm-opts alpha=600,beta=2500,flags=1 --out H.json
eval/vocr6_score.py read  --plan dev_plan_v2.pkl  --blob B --lm vocr_lm6_o3_1000k.vlm --lm-opts alpha=600,beta=2500,flags=1 --out D.json
eval/vocr6_score.py score --name N --blob B --rubi-norm rubi_norm --hval-plan hval_plan_v3.pkl --hval-dump H.json \
    --dev-plan dev_plan_v2.pkl --dev-units units.pkl --dev-dump D.json --exclude-vns v12188 v18432 \
    --hval-cascade-dump H_up.json --dev-cascade-dump D_up.json --out score_N.json
# a decision (decision.adopt), the records and this document's stage-4 tables
eval/vocr6_select.py compare --scores DIR --x VOCR5-N,VOCR5-N,VOCR5-N --y v6s2cil_N_seed1,v6f4_N_seed2,v6s2cil_N_seed3
eval/vocr6_report.py records && eval/vocr6_report.py tables
# the follow-up's gate fits (main's driver, vocr6-final-gate's final settings), in the order L, N, S, into one OUT
export VOCR6_GATE_ARGS="--path-lm-hval h64_1000k.vlm --path-lm-dev vocr_lm6_o3_1000k.vlm --path-lm-opts alpha=600,beta=2500,flags=1"
eval/vocr6_gate.sh L v6f4c_L_seed1_w8.vocr - OUT
eval/vocr6_gate.sh N v6f4c_N_seed1_w8.vocr v6f4c_L_seed1_w8.vocr OUT
eval/vocr6_gate.sh S v6f4dlh_S_seed1_w8.vocr v6f4c_N_seed1_w8.vocr OUT
# its final read: once per registered finalist set (eval/vocr6_final_test_protocol.json amendment 9)
eval/vocr6_final_test.py stage --s S.vocr --n N.vocr --l L.vocr --out MODELS_DIR
eval/vocr6_final_test.py runtimecheck --rubi RUBI_AT_892cfb9 --models MODELS_DIR --work W --out RUNTIMECHECK.json
eval/reproducibility/vocr6-final/report.sh MODELS_DIR OUT     # refuses a set already read or not registered
```

- The plans and LMs are pinned by sha256 in `eval/vocr6_protocol.json` (`scoring`, `stages.4.plans`,
  `stages.4.amendments[2]`).
- The head-gate and two-level fits are vocr6-final-gate's driver (PR #146).
- The final test is vocr6-final-test's (PR #140); the follow-up's read is its amendment 9 (vocr6-followup).
- The LM check is vocr6-final-lm's (PR #151).

## Compute

CPU only: one 64-core host with no GPU and 12 heavy slots of 5 threads.
- **Per run, at the loaded rate:** a Normal fine-tune of 15k steps takes 2.9-3.5 slot-hours (4.7 with S3rep's
  training-only branches), a Large one about 4.5-4.7.
- **Stage 1:** 13.4 slot-hours for A15 and B15, plus 2.0 for the stopped C50 runs.
- **Stages 2-3:** 76.8 slot-hours over 19 runs, plus 0.5 for a stopped S3long segment.
- **The follow-up:** stage 4c 24.0 slot-hours (Normal 10.3, Large 13.7, 45k steps each), plus 0.6 for Small's run,
  stopped unscored (06:51-07:30Z); stage 4d 19.3 (six runs
  17.4, the fold-in 1.9); the gate fits 2.3 of compute (2.56 slot-hours held); the final read 0.8
  (one slot, 47 minutes); its runtime check 0.3.

## Limits

- **Nothing has run on a console.** Console times and memory are estimates (ocr-speed's method, calibrated to
  VOCR3-H15's measured 5.0-5.2 ms per CTC frame).
- **hval comes from the training titles** (46 of the 48 have hval pages; the test title has none). It measures how
  well VOCR6 reads the hooked games. The development score measures other games. Most of VOCR6's gain is on the
  hooked titles.
- **The finder carries most of the remaining error** (#119: 88% of VOCR5-N's whole-dialogue error on development).
  Recognizer arms move the development score by 0.03-0.12 points. The learned spotter (textspot6) was not adopted.
- **Seen flags** apply to every test-set number:
  - Collar x Malice (PCSG00866) shares releases with vita-test's Collar x Malice -Unlimited-.
  - Little Busters! and planetarian share the Key 10th Memorial BOX with CLANNAD.
  - 38 of the 49 hooked titles' VNs were in VOCR5's own fine-tune pools.
  - Two hooked train titles draw dialogue in CLANNAD's glyph raster, re-quantised (docs/synth6.md): PCSG00620
    Grisaia Spinout and PCSG00007 Little Busters!. With hval and guard pages folded in, the shipped models have seen
    that raster for 1,234 characters. synth6 is not in the shipped recipe.
