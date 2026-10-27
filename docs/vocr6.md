# VOCR6: the recognizers retrained on the owner's hooked PS Vita and PSP corpus

Issue #125, track T1 of the VOCR6 plan. VOCR6 keeps VOCR5's shapes for Small, Normal (the default) and Large, so each
tier's weights, arena and compute per frame stay VOCR5's; only the speed sections appended to the blobs differ (below).
What changes is the data: each tier's VOCR5 model is fine-tuned again on the owner's hooked corpus (exact game text on
native frames, docs/vocr6-data.md) and on the finder's own crops of those pages. Normal and Large ship as VOCR6; Small
stays VOCR5-S. Every number here was measured on the host. Console times are estimates.
**Nothing of VOCR6 has run on a console.**

**Summary.**
- **Normal and Large are VOCR6. Small stays VOCR5-S.** Each tier was adopted over the released VOCR5 tier only if it
  read the hooked games' held-out pages better, with every seed pair below 0, and other games no worse. That was
  checked on Rubi's shipped path, with three seeds.
- **Normal and Large on the hooked games' pages** (hval, held out): 1.008 → 0.936% and 0.990 → 0.917% dialogue CER,
  both resolved better. On other games they are level: -0.038 and +0.026 points on VOCR5's development score.
- **Small** read the hooked pages better too (-0.134), but other games' fixed lines a little worse (+0.070,
  resolved), so it was not adopted.
- **What ships** (pre-release v0.4.0-vocr6.1): `VOCR6_N_w8_2l.vocr` (1,187,616 B) and `VOCR6_L_w8_2l.vocr`
  (2,475,680 B). They have VOCR5's shapes, with #122's two-level head appended; the head gate was rejected on
  VOCR6's logits. Est. 0.79 s and 1.87 s per 3-line box on one 333 MHz A9 core, about 30 and 80 ms more than VOCR5
  as Rubi runs it today. W8I arena 184,400 B and 262,480 B.
- **The final test, read once:**
  - CLANNAD lines: Normal 0.66 → 0.44%, Large 0.60 → 0.36%, resolved better. CLANNAD's glyph raster is in the
    training data, so these are not unseen-game numbers.
  - vita-test + psp-test: level.
  - The hooked test title: level or better.
- **The owner's < 1% target** (page path): met on the hooked games' pages (0.94% / 0.92%). Not met on other games:
  whole dialogue 2.43% / 2.46% (4a means), about 80% of it the line finder's. The spotter tracks (#128, #149) target that.


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
Rubi's per-line decode (greedy plus top-8, the counter re-decode, beam 8 with the 1 MB LM, textnorm and the
non-Japanese filter). Stages 1-3 read without the cascade. Stage 4 read Rubi's whole shipped path: the COLGAP finder
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
| S3long | A15 at 45,000 steps | 1.034 (seed 1) | 0.370 | 2.046 (3.47%) | -0.054 [-0.089, -0.021] (seed 1) | -0.047 [-0.195, +0.138] (seed 1) | 9.3 | seed 3 running (ends ~14Z); decides stage 4c only |

- **Not run.** Names in or out: names are 0.9% of rows, and the owner crops them out. Vertical re-flow: one title,
  and no vertical model this round. Width/depth: a wider model needs a fresh 150k-step pretrain per seed, outside this
  round's CPU budget. QAT: int8 costs about 0 (#119). S3gtc (an auxiliary NRTR loss at weight 0.1) was dropped for
  compute; its prior at weight 1.0 was +3.08 at 5k steps.
- **S2s6** (synth6 v3, the harvested game fonts) has the biggest dev gain, -0.121 points, but its hval interval
  reaches +0.001, so it is not adopted.
- **S3long** decides stage 4c only (amendment 6): a 45k-step retrain of the shipped tiers. It replaces a shipped blob
  only if S3long's two held-out seeds pass the rule and the 4c blob is not resolved worse on dev.

## Stage 4: the shipped tiers (2026-10-27)

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
  - Small's dev interval lies above 0 (+0.070 [+0.004, +0.138]), so the rule keeps VOCR5-S.
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

### Speed: the head gate and the two-level head (vocr6-final-gate, #146)

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


### LM settings on VOCR6's logits (vocr6-final-lm)

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


### Final test (read once, report only)

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
- **Vita and PSP apart** (path b, fixed lines): vita-test improves (Normal -0.21 [-0.56, +0.05], Large
  -0.17 [-0.40, +0.00]). psp-test does not (Normal +0.22 [+0.00, +0.41], Large +0.09 [-0.12, +0.30]). On path (a)
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
  - the LMs' Jiten word list includes CLANNAD's script.

  So **CLANNAD's gain is not an unseen-game result.** The development sets are the unseen-game evidence, and there
  VOCR6 is level.


### Stage 4c (in progress)

Stage 4c retrains the shipped recipe at 45,000 steps, with hval folded in, for Normal and Large (amendment 6;
Small's run was stopped when Small was not adopted). Its blobs replace the shipped ones only if both hold:
- S3long is adopted over A15 on stage 3's held-out pair (seeds 1 and 3);
- the 4c blob is not resolved worse than the shipped 4b blob on the development score.

If they pass, they come as a follow-up release, with their own gate fits and their own final-test read.



## Reproduce

```sh
# a run (the protocol's arms; checkpoints, blob and the plan-v2 read under $HERD/shared/artifacts/<label>/)
git checkout vocr6-runner-00b8288        # stage 4 (4c: vocr6-runner-a6430de)
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
```

- The plans and LMs are pinned by sha256 in `eval/vocr6_protocol.json` (`scoring`, `stages.4.plans`,
  `stages.4.amendments[2]`).
- The head-gate and two-level fits are vocr6-final-gate's driver (PR #146).
- The final test is vocr6-final-test's (PR #140).
- The LM check is vocr6-final-lm's (PR #151).

## Compute

CPU only: one 64-core host with no GPU and 12 heavy slots of 5 threads. A Normal fine-tune of 15k steps takes 2.9-3.4
slot-hours at the loaded rate, a Large one about 4.5. Stage 1 used 13.4 slot-hours for A15 and B15 plus 2.0 for the
stopped C50 runs. Stages 2-3 used 66.3 slot-hours before S3long's seed 3.

## Limits

- **Nothing has run on a console.** Console times and memory are estimates (ocr-speed's method, calibrated to
  VOCR3-H15's measured 5.0-5.2 ms per CTC frame).
- **hval comes from the same 49 titles as the training pages.** It measures how well VOCR6 reads the hooked games. The
  development score measures other games. Most of VOCR6's gain is on the hooked titles.
- **The finder carries most of the remaining error** (#119: 88% of VOCR5-N's whole-dialogue error on development).
  Recognizer arms move the development score by 0.03-0.12 points. The learned spotter (textspot6) was not adopted.
- **Seen flags** apply to every test-set number:
  - Collar x Malice (PCSG00866) shares releases with vita-test's Collar x Malice -Unlimited-.
  - Little Busters! and planetarian share the Key 10th Memorial BOX with CLANNAD.
  - 38 of the 49 hooked titles' VNs were in VOCR5's own fine-tune pools.
  - Two hooked train titles draw dialogue in CLANNAD's glyph raster, re-quantised (docs/synth6.md): PCSG00620
    Grisaia Spinout and PCSG00007 Little Busters!. With hval and guard pages folded in, the shipped models have seen
    that raster for 1,234 characters. synth6 is not in the shipped recipe.
