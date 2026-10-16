# VOCR5: the final round, three recognizers (Small, Normal, Large)

VOCR5 trained one recognizer per tier of the owner's final ceilings, on everything the 2026-10-14 13:25Z policy
allows. It used vocab-expand's tier vocabularies and model-tiers' shapes as far as the final sizing allowed, and it
taught the models to read what the console's line finder hands them. Selection used the fixed development sample
only, with dialogue CER primary. The protected sets (vita-test, psp-test, CLANNAD) were scored once, after the three
selections were committed, through the plain-C W8I runtime.

Every number below is a host evaluation or a host estimate. Nothing of VOCR5 has run on a console.

**Results.**

- **Each tier's VOCR5 model replaced its VOCR4 incumbent under the pre-registered paired rule** on the development
  score (½ fixed-line dialogue CER, ½ whole-dialogue CER):
  - Small: −0.52 points against VOCR4-FL10 (VN-cluster 95% CI −0.70 to −0.36);
  - Normal: −0.20 against VOCR4-W20 (−0.38 to −0.05);
  - Large: −0.60 against VOCR4-W20 (−0.90 to −0.37).

  At every tier the finder-robust fine-tune (arm F) beat the plain one (arm P): −0.13 to −0.15 points, every
  interval below 0. The gain is all whole dialogue (−0.25 to −0.28 points); the hand-cut fixed lines are level.
- **The released models** (full release [v0.3.0](https://github.com/bee-san/vita-vn-ocr/releases/tag/v0.3.0)):
  `VOCR5_S_w8.vocr` (819,792 B), `VOCR5_N_w8.vocr` (1,185,632 B, the overlay's default) and `VOCR5_L_w8.vocr`
  (2,473,696 B).
- **Fixed lines, vita-test + psp-test, dialogue, console polarity:**
  - Small **1.60%**, Normal **1.13%**, Large **0.79%**;
  - against VOCR4-W20 1.17% and the released v0.2.0 H15 1.79%;
  - Large − VOCR4-W20 −0.38 points (VN CI −0.98 to +0.00); Normal − v0.2.0 H15 −0.66 (−1.19 to −0.21);
  - with the overlay's beam + 1 MB LM decode: 1.15%, 0.72% and **0.53%**.
- **Whole dialogue** (finder-cut regions, base, dialogue text): 8.31%, 7.98% and **7.08%**, against VOCR4-W20's
  8.72%. Large − VOCR4-W20 is −1.64 points (VN CI −2.33 to −0.97); Normal − VOCR4-W20 −0.73 (−1.29 to −0.16). This is
  the first round in which the gains carry over to the overlay's own reading path.
- **CLANNAD** (2,052 dialogue lines, frozen Claude references): 1.38%, 0.96% and **0.73%** dialogue CER.
  - Google Lens: 1.53% (exact lines 96.5% against VOCR5-L's 89.9%).
  - VOCR4-W20: 0.89%.
  - All 3,048 lines: VOCR5-L 0.98%, Lens 1.47%; L − Lens −0.50 points, page CI −1.15 to +0.09.
- **Gates.** VOCR2's fixed-line gate passes for all three. Its strict whole-dialogue guard fails for all three (36, 68
  and 43 of 360 title checks regress), as it has for every model. The owner made VOCR5 the default regardless (13:26Z).

Machine-readable outputs:

- The protocol: `eval/vocr5_protocol.json` with three amendments.
- Pins: the render `data/manifests/synth4_shards_v5631.json` and the fine-tune pools `eval/vocr5_pools.sha256` +
  `eval/vocr5_ft_sources.json`.
- Records: `eval/vocr5_render_s4v5631_20261015.json`, `eval/vocr5_pools_20261016.json`,
  `eval/vocr5_refine_20261015.json`.
- Decisions and selections: `eval/vocr5_{arm,replace,selection,scores}_<Tier>_20261016.json`.
- The locked report: `eval/vocr5_report_20261016.json` and `docs/vocr5-report-20261016.md`.
  - The JSON is the report's `report_summary.json` written compactly (no indentation). It has the same content, and
    the compact form keeps it under the repository's 2 MiB file limit.
  - The dialogue-primary tables with the Lens and LM arms, and per-style rows for every arm, are in
    `eval/vocr5_dialogue_20261016.json`: `eval/vocr5_dialogue.py tables` on the report's stored per-line records.
- Lens: `eval/vocr5_lens_20261016.json`.
- CLANNAD: `eval/vocr5_clannad_20261016.json` and `eval/vocr5_clannad_lens_view_20261016.json`.

## The tiers

The sizing follows the owner's caps (18:09Z) and the conductor's final notes (10:07Z / 10:21Z; 2026-10-16 07:52Z).
Speed is per typical full textbox: 3 lines of about 640 px, which is 240 CTC frames plus 0.1 s for the finder, on
one Cortex-A9 core. At freezing, Normal's ceiling was 1.3 MB, because #85 and overlay #93 were open and test.3 had
not confirmed the 444 MHz boost. Amendment 3 (2026-10-16 08:23Z, before any candidate was scored) raised it to the
owner's 1.6 MB cap once test.3 confirmed the boost, and made the already-trained VOCR4-W20 Normal's incumbent. A
1.6 MB VOCR5 shape would have needed a new pretrain, about a day.

The 10:21Z rule allowed the 1.6 MB Normal only if test.3 confirmed the boost **before VOCR5 training started**.
test.3 came on 2026-10-16, after training had started. The conductor waived that condition (2026-10-16 07:52Z)
because the trained Normal shape stayed the same: the amendment changed only the incumbent Normal had to beat. That
raised the bar, since VOCR4-W20's development score (3.580) is better than VOCR4-H15's (3.959).

| tier | shape | vocab | int8 weights | W8I arena | MACs/frame | incumbent | selected run |
|---|---|---|---:|---:|---:|---|---|
| Small (< 1 MB, < 1 s) | `s_w100v4987`: FL10's topology | `data/vocab_4987_proof.txt` (4,987) | 819,792 B | 124,832 B | 967,392 | VOCR4-FL10 (657,664 B) | `v5f_S_seed2` |
| Normal (≤ 1.6 MB since amendment 3; about 1 s per box) | `n_w150v5631`: H15's topology | `data/vocab_vn_5631.txt` (5,631) | 1,185,632 B | 182,944 B | 1,591,488 | VOCR4-W20 (1,558,016 B; VOCR4-H15 until amendment 3) | `v5f_N_seed3` |
| Large (≤ 2.5 MB, < 3 s) | `l_lw300v5631`: late-wide 2.0 → 3.0 | `data/vocab_vn_5631.txt` (5,631) | 2,473,696 B | 261,024 B | 3,402,720 | VOCR4-W20 (1,558,016 B) | `v5f_L_seed2` |

Estimated seconds per 3-line box. The method is ocr-speed's: `vita/a9_layers.py` at #85's head, top-8, scaled to
VOCR3-H15's console rate of 4,941.7 µs per frame at an assumed 333 MHz. It reproduces `ceilings4.json` to the third
decimal. All of these are estimates:

| model | main's runtime (before #85), 333 / 444 MHz | #85 (merged), 333 / 444 | #85 without prefetch credit, 333 / 444 |
|---|---|---|---|
| VOCR5-S | 0.94 / 0.73 | 0.57 / 0.45 | 0.63 / 0.50 |
| VOCR5-N | 1.34 / 1.03 | 0.81 / 0.63 | 0.92 / 0.72 |
| VOCR5-L | 2.68 / 2.08 | 1.47 / 1.13 | 2.08 / 1.63 |
| VOCR4-W20 (Normal's incumbent) | 1.77 / 1.35 | 1.08 / 0.84 | 1.26 / 0.97 |

VOCR4-W20 would have cost an estimated 0.20-0.43 s more per box than VOCR5-N: 0.25 s at 444 MHz and 0.34 s at 333
MHz without prefetch credit. The locked report's own `vita/a9_cycles.py` estimate is 87-130 / 136-203 / 260-506 ms
per 640-px line at 444 MHz. It is uncalibrated, on 40 synth-val lines, and in `docs/vocr5-report-20261016.md`.

Why these shapes:

- **Small.** FL10's topology with vocab-expand's 4,987-class proof vocab. That vocab is a subset of
  `data/vocab.txt`, so every class has real and synthetic data without a render.
- **Normal.** None of #74's Normal shapes fit the 1.3 MB of freezing. The H15 topology with the 5,631-class VN head
  keeps a box near 1 s at 333 MHz on #85's runtime.
- **Large.** The late-wide `n_lw300v5631` is the parameter-heavy, compute-light kind the owner asked for: 2.47 MB of
  weights for 3.40 M MACs per frame. With the 6,459-class vocab it would weigh 2,564,368 B, over the 2.5 MB cap.

## Data

- **Synthetic.**
  - Small: the C mix (nvsynth 0.5, synth2 0.25, synth3 v2 0.25), the recipe FL10 won with in VOCR3 and VOCR4.
  - Normal and Large: the M mix (synth3 v3, VOCR4's only resolved gain) with a tenth from **s4v5631**. That is a
    synth4 render with `data/vocab_vn_5631.txt` (`synth4.py --vocab`, the context index remapped by character,
    the curriculum rebuilt): 1,000,000 crops, seed 20261015, 0 render errors.
  - Every label of the render is in the vocab; 0 hold the `<unk>` entry.
  - The 610 kanji new against `data/vocab.txt` got 50-127 exposures (median 67). The curriculum's K = 80 was not
    reached for 563 of them, as synth4's own render left classes below K.
  - The first 1,000 crops of shards 0 and 57 re-render byte for byte.
  - **Speaker names of 12 reserved release-partner VNs** (found by audit-vocr5, F1). The renders draw speaker names
    from synth3's name pool, built on 2026-10-12 under that day's reservation, and filtered at render time only by
    the CLANNAD exclusion manifest. That leaves the names of 128 characters of 12 VNs that `reserved.json` reserves:
    7 test-shared-release VNs and 5 dev-sample-shared-release VNs, 190 of the pool's 378,143 rows.
    - 506 of s4v5631's 1,000,000 crops show one of these 109 full names, drawn from the VNDB pool. The inherited
      synth3 v2 and v3 shards hold 588 and 604 such crops.
    - None of the 109 names occurs in a vita-test, psp-test, teacher_val-dev or devnative-dev line.
    - So the protocol's "no material of a reserved.json VN" holds for scenes and real data, not for synthetic
      speaker names. The render record's first hygiene statement said otherwise; it was corrected
      (`eval/vocr5_render_s4v5631_20261015.json` `hygiene`).
  - nvsynth was not cut again.
- **Real** (`eval/vocr5_pools_20261016.json`: CHECK_OK, and `eval/tools/reserve_check.py --require-hashes --pixels`
  RESERVE_OK on every file):

| source | crops | what |
|---|---:|---|
| tt + tvt | 12,721 | teacher_train and teacher_val's train part (the re-carve of #84) |
| rd + rvt | 10,421 | real_train and real_val's train part |
| rp3 ink / GRAD / refined | 170,387 / 116,994 / 99,560 | realplus v3, the final Stage A pool (101,350 lines, 14,461 VNs; audit-realplus-v3 HYGIENE_OK) |
| owner ink + m4 + m8 / GRAD / refined | 20,457 / 5,624 / 4,656 | owner pools v1 + v2 merged (private; 73 VNs) |
| native + devnative ink / GRAD / refined | 566 / 519 / 446 | native pool v1 and devnative's train parts |

Two rules shaped the real data:

- **Exclusion.** A row is dropped on any `reserved.json` VN (test or dev_sample), on a development or test
  screenshot, and on a test session; nothing met these. The old hash splits train.
- **Deduplication.** Rows are deduplicated by screenshot, label and crop style. It dropped 102 of devnative's 121
  tv_native ink crops, which repeat teacher_val's own lines, and 15 + 1 others.

## Arms and runs

- **Pretrains.** Small 216,000 steps, Normal 150,000 and Large 108,000 (VOCR4: 128,000 / 40,000 / 32,000). There
  were 2, 3 and 3 seeds, 8 at once: started 2026-10-15 12:42:28-13:43:39Z, ended 2026-10-16 12:29-13:31Z, about
  24 h each.
- **Fine-tunes.** VOCR4's recipe, 15,000 steps, two arms per pretrain:
  - **P**: every pool, real half 0.5 (tt + tvt 0.15, rd + rvt 0.075, rp3 0.175, own 0.075, dn + nat 0.025, each
    split into ink and GRAD by rows).
  - **F**: P plus finder-tighten's robustness work. `fa=0.3` on the real ink sources and 0.15 on nvsynth and synth2
    adds the finder's extra rows (box borders, neighbour lines, furigana, scenery) and line-end clutter (cursors,
    portrait edges), with the label unchanged (`train/finder_aug.py`). Each GRAD source also reads its refined
    copies.
  - **Refined copies.** Made by the merged refinement and kept only where it changed the rows alone (amendment 1:
    of the column-cut copies the helper read, about 38 of 40 had lost glyphs under an unchanged label).
- **Runs.** The 16 fine-tunes ran 2026-10-16 12:42:14-19:19:07Z. All 24 runs (8 pretrains, 16 fine-tunes) span
  2026-10-15 12:42Z to 2026-10-16 19:19Z; none failed or was restarted.

## Selection (development sample only)

The cohorts are the fixed development sample (`reserved.json` dev_sample):

- tv: teacher_val's dev part (959 lines, 101 VNs; 426 name-like lines are a secondary column);
- dn: devnative's four dev files (765 crops; 200 name lines secondary);
- wd: the whole-dialogue regions of finder-tighten's tv_dlg / dn_dlg on those VNs, read by the C GRAD finder with
  and without the refinement, at base, +4 and +8 px.

Score = 0.25 tv + 0.25 dn + 0.5 wd (dialogue CER points). The decisions use stratified VN-cluster bootstraps
(10,000 resamples). tv's, dn's and wd's VNs are resampled independently, with numpy seeds 0, 1 and 2
(`eval/vocr5_select.py`; the protocol's "seed 0" names only the first). The audit re-decided every tier with three
other seed sets and no decision changed. A candidate is adopted when the interval of the mean seed-pair difference
lies below 0, every pair is below 0 and no component's interval lies above 0.

| tier | decision | score (95% CI) | tv | dn | wd | pairs | adopted |
|---|---|---|---|---|---|---|---|
| Small | P → F | −0.134 (−0.230 to −0.047) | +0.066 (−0.085 to +0.219) | −0.105 (−0.296 to +0.050) | −0.248 (−0.398 to −0.116) | −0.138, −0.129 | **yes** |
| Small | VOCR4-FL10 → F | −0.521 (−0.698 to −0.362) | −0.203 (−0.444 to +0.028) | −0.351 (−0.702 to −0.070) | −0.765 (−1.055 to −0.509) | −0.497, −0.545 | **yes** |
| Normal | P → F | −0.134 (−0.225 to −0.052) | +0.033 (−0.067 to +0.134) | −0.020 (−0.135 to +0.083) | −0.275 (−0.441 to −0.127) | −0.138, −0.108, −0.157 | **yes** |
| Normal | VOCR4-W20 → F | −0.200 (−0.381 to −0.053) | +0.121 (−0.032 to +0.265) | +0.010 (−0.175 to +0.173) | −0.465 (−0.814 to −0.197) | −0.226, −0.135, −0.239 | **yes** |
| Large | P → F | −0.151 (−0.254 to −0.065) | +0.017 (−0.042 to +0.078) | −0.082 (−0.206 to +0.023) | −0.269 (−0.459 to −0.105) | −0.128, −0.211, −0.113 | **yes** |
| Large | VOCR4-W20 → F | −0.599 (−0.901 to −0.374) | −0.277 (−0.469 to −0.111) | −0.318 (−0.624 to −0.094) | −0.901 (−1.474 to −0.479) | −0.572, −0.622, −0.604 | **yes** |

Every run (development score; tv / dn / wd dialogue CER):

| tier | run | score | tv | dn | wd |
|---|---|---:|---:|---:|---:|
| Small | **`v5f_S_seed2`** | 3.782 | 1.36% | 1.11% | 6.33% |
| Small | `v5f_S_seed1` | 3.829 | 1.14% | 1.24% | 6.47% |
| Small | `v5p_S_seed2` / `v5p_S_seed1` | 3.911 / 3.968 | 1.23% / 1.15% | 1.26% / 1.30% | 6.58% / 6.71% |
| Small | VOCR4-FL10 (incumbent) | 4.327 | 1.46% | 1.52% | 7.16% |
| Normal | **`v5f_N_seed3`** | 3.342 | 0.74% | 0.98% | 5.82% |
| Normal | `v5f_N_seed1` / `v5f_N_seed2` | 3.354 / 3.445 | 0.99% / 1.01% | 0.92% / 0.95% | 5.75% / 5.91% |
| Normal | `v5p_N_seed1` / `seed3` / `seed2` | 3.492 / 3.499 / 3.553 | 0.88% / 0.88% / 0.89% | 1.04% / 0.92% / 0.96% | 6.03% / 6.10% / 6.18% |
| Normal | VOCR4-W20 (incumbent) | 3.580 | 0.79% | 0.94% | 6.29% |
| Large | **`v5f_L_seed2`** | 2.959 | 0.49% | 0.68% | 5.33% |
| Large | `v5f_L_seed3` / `v5f_L_seed1` | 2.977 / 3.008 | 0.53% / 0.53% | 0.58% / 0.62% | 5.40% / 5.44% |
| Large | `v5p_L_seed3` / `seed1` / `seed2` | 3.090 / 3.136 / 3.170 | 0.50% / 0.43% / 0.57% | 0.67% / 0.76% / 0.70% | 5.60% / 5.68% / 5.71% |

The former Normal incumbent VOCR4-H15 scores 3.959 on the same cohorts (information only).

## Locked report (once, after the three selections)

Timing (2026-10-16, report runner `7ee5241`):

- `eval/vocr3_report.py` shared, a9, overlay and render, 19:34-20:13Z;
- the Lens reading (amendment 2), 19:47-19:58Z, all 451 lines read on the first attempt;
- the deployed decode, 20:30-20:33Z, from `a5ed2d9` after a fix of its invocation (`vocr_cli --stdin`).

The scalar C W8I runtime (gcc 11.5) read vita-test (206 lines, 16 games) and psp-test (245 lines, 14 games). The
intervals are paired bootstraps (10,000 resamples, seed 0) over lines and VN clusters.

### Fixed lines (console polarity; dialogue = lines without the 'name' style)

| model | vita-test | psp-test | pooled dialogue | pooled exact | all text, pooled | names |
|---|---|---|---|---|---|---|
| VOCR1 (v0.2.0 FL10) | 2.05% (49/2392) | 4.17% (96/2301) | **3.09%** (145/4693) | 72.0% | 3.82% (194/5074) | 12.86% |
| v0.2.0 H15 | 1.30% (31/2392) | 2.30% (53/2301) | **1.79%** (84/4693) | 82.4% | 2.21% (112/5074) | 7.35% |
| F20 | 1.38% (33/2392) | 2.22% (51/2301) | **1.79%** (84/4693) | 82.1% | 2.19% (111/5074) | 7.09% |
| VOCR3-W20 | 0.92% (22/2392) | 1.69% (39/2301) | **1.30%** (61/4693) | 87.2% | 1.62% (82/5074) | 5.51% |
| VOCR4-FL10 | 1.25% (30/2392) | 2.30% (53/2301) | **1.77%** (83/4693) | 83.6% | 2.15% (109/5074) | 6.82% |
| VOCR4-H15 | 1.00% (24/2392) | 1.96% (45/2301) | **1.47%** (69/4693) | 86.0% | 1.83% (93/5074) | 6.30% |
| VOCR4-W20 | 0.79% (19/2392) | 1.56% (36/2301) | **1.17%** (55/4693) | 88.4% | 1.40% (71/5074) | 4.20% |
| **VOCR5-S** | 1.00% (24/2392) | 2.22% (51/2301) | **1.60%** (75/4693) | 86.0% | 1.83% (93/5074) | 4.72% |
| **VOCR5-N** | 0.96% (23/2392) | 1.30% (30/2301) | **1.13%** (53/4693) | 88.4% | 1.30% (66/5074) | 3.41% |
| **VOCR5-L** | 0.59% (14/2392) | 1.00% (23/2301) | **0.79%** (37/4693) | 90.8% | 0.93% (47/5074) | 2.62% |

| pair (dialogue, points; VN-cluster 95% CI) | vita-test | psp-test | pooled |
|---|---|---|---|
| VOCR5-S − VOCR4-FL10 | −0.25 (−0.87 to +0.18) | −0.09 (−0.75 to +0.59) | −0.17 (−0.59 to +0.24) |
| VOCR5-S − VOCR1 | −1.05 (−1.65 to −0.55) | −1.96 (−3.30 to −0.81) | −1.49 (−2.19 to −0.88) |
| VOCR5-N − VOCR4-W20 | +0.17 (−0.38 to +0.87) | −0.26 (−0.93 to +0.22) | −0.04 (−0.43 to +0.37) |
| VOCR5-N − VOCR4-H15 | −0.04 (−0.55 to +0.55) | −0.65 (−1.33 to −0.15) | −0.34 (−0.73 to +0.05) |
| VOCR5-N − v0.2.0 H15 | −0.33 (−0.99 to +0.23) | −1.00 (−1.97 to −0.29) | −0.66 (−1.19 to −0.21) |
| VOCR5-N − VOCR5-S | −0.04 (−0.38 to +0.33) | −0.91 (−1.78 to −0.17) | −0.47 (−0.91 to −0.07) |
| VOCR5-L − VOCR4-W20 | −0.21 (−0.48 to +0.00) | −0.56 (−1.82 to +0.16) | −0.38 (−0.98 to +0.00) |
| VOCR5-L − VOCR5-N | −0.38 (−1.04 to +0.09) | −0.30 (−1.01 to +0.25) | −0.34 (−0.82 to +0.06) |
| VOCR5-L − F20 | −0.79 (−1.44 to −0.31) | −1.22 (−2.17 to −0.49) | −1.00 (−1.53 to −0.56) |

Native polarity, every style row and the all-text pairs are in `docs/vocr5-report-20261016.md`. On all text, pooled,
VOCR5-L − VOCR4-W20 is −0.47 points (line CI −0.82 to −0.16, VN CI −1.03 to −0.09), and VOCR5-N − VOCR4-W20 is −0.10
(line CI −0.41 to +0.21). VOCR5-N reads psp-test better than VOCR4-W20 and vita-test slightly worse; neither
difference is resolved.

**The deployed decode** (report only): beam 8 plus the overlay's default 1 MB character LM (`vocr_lm_o3_1000k.vlm`,
v0.2.6-ctclm.2, its 1MB settings). It lowers pooled dialogue CER to:

| model | greedy | beam + 1 MB LM | change (VN CI) |
|---|---|---|---|
| VOCR5-S | 1.60% | 1.15% (54/4693) | −0.45 (−0.89 to −0.03) |
| VOCR5-N | 1.13% | 0.72% (34/4693) | −0.40 (−0.86 to −0.04) |
| VOCR5-L | 0.79% | 0.53% (25/4693) | −0.26 (−0.56 to −0.04) |
| VOCR4-W20 | 1.17% | 0.92% (43/4693) | −0.26 (−0.59 to +0.08) |

The LM's Jiten visual-novel list includes CLANNAD's script and those of other titles (its names were removed), so
the decoded numbers carry that disclosure.

**Google Lens** (report only, amendment 2) read the same line crops: 5.58% dialogue CER pooled (262/4693; vita-test
3.72%, psp-test 7.52%) with 83.3% of lines exact. VOCR5 − Lens on dialogue, native polarity:

| pair | vita-test | psp-test | pooled |
|---|---|---|---|
| VOCR5-S − Lens | −2.72 (VN −6.03 to −0.54) | −5.35 (−10.37 to −1.39) | −4.01 (−7.48 to −1.38) |
| VOCR5-N − Lens | −2.72 (−5.91 to −0.54) | −6.13 (−11.83 to −1.95) | −4.39 (−8.13 to −1.61) |
| VOCR5-L − Lens | −3.18 (−6.45 to −1.09) | −6.61 (−12.74 to −2.24) | −4.86 (−8.87 to −2.02) |

The crops include the furigana sitting above a line, which the references leave out by convention and Lens
transcribes. On psp-test's 20 furigana lines that is 116 of Lens's 227 errors (all text), and on vita-test's 3 it is
30 of 89 (`eval/vocr5_dialogue_20261016.json`, `arms.native.Lens.styles.furigana`). This is not how Lens is
normally used; on CLANNAD's GRAD line crops it scores 1.53%.

### Whole dialogue (finder-cut regions, each model's C finder; dialogue text, pooled)

| model | base | +4 px | +8 px |
|---|---|---|---|
| VOCR1 | 10.77% (499/4635) | 13.72% | 14.26% |
| v0.2.0 H15 | 8.80% (408/4635) | 11.37% | 11.93% |
| F20 | 8.54% (396/4635) | 11.41% | 11.91% |
| VOCR4-FL10 | 9.54% (442/4635) | 12.64% | 13.33% |
| VOCR4-H15 | 8.91% (413/4635) | 11.93% | 12.86% |
| VOCR4-W20 | 8.72% (404/4635) | 11.54% | 12.62% |
| **VOCR5-S** | 8.31% (385/4635) | 11.28% | 12.00% |
| **VOCR5-N** | 7.98% (370/4635) | 10.68% | 12.02% |
| **VOCR5-L** | 7.08% (328/4635) | 10.03% | 11.15% |

| pair (points; VN-cluster 95% CI) | base | +4 px | +8 px |
|---|---|---|---|
| VOCR5-S − VOCR4-FL10 | −1.23 (−1.85 to −0.65) | −1.36 (−1.93 to −0.83) | −1.34 (−2.22 to −0.50) |
| VOCR5-N − VOCR4-W20 | −0.73 (−1.29 to −0.16) | −0.86 (−1.49 to −0.32) | −0.60 (−1.20 to −0.02) |
| VOCR5-N − VOCR4-H15 | −0.93 (−1.57 to −0.31) | −1.25 (−1.88 to −0.71) | −0.84 (−1.48 to −0.18) |
| VOCR5-N − v0.2.0 H15 | −0.82 (−1.38 to −0.38) | −0.69 (−1.35 to −0.10) | +0.09 (−0.68 to +0.94) |
| VOCR5-L − VOCR4-W20 | −1.64 (−2.33 to −0.97) | −1.51 (−2.12 to −0.95) | −1.47 (−2.40 to −0.50) |
| VOCR5-L − VOCR5-N | −0.91 (−1.46 to −0.40) | −0.65 (−1.05 to −0.26) | −0.86 (−1.44 to −0.28) |

In VOCR3 and VOCR4 the fixed-line gains did not carry over to whole dialogue. VOCR5's do: every VOCR5-versus-VOCR4
interval is below zero at every variant, which is what arm F was for. The overlay host cross-check (vita-jp-overlay
`1a118ef`'s `tools/benchmark_local_ocr.py`) gives the same all-text errors as the base whole dialogue in all 26 cells.
Its workspace peak on vita-test is 1.03, 1.43 and 2.73 MiB for VOCR5-S, -N and -L (1,076,128 / 1,500,080 / 2,866,224 B).

### Gates

| gate | fixed-line gate (console) | whole-dialogue guard |
|---|---|---|
| VOCR5-S vs VOCR1 | pass: vita-test 61 → 28 (hiragana 15 → 8), psp-test 133 → 65 (30 → 14) | FAIL: 36 of 360 title checks regress |
| VOCR5-N vs v0.2.0 H15 | pass: vita-test 40 → 24 (8 → 7), psp-test 72 → 42 (13 → 9) | FAIL: 68 of 360 |
| VOCR5-L vs v0.2.0 H15 | pass: vita-test 40 → 16 (8 → 6), psp-test 72 → 31 (13 → 6) | FAIL: 43 of 360 |

The guard fails a candidate when any of 30 titles × 3 variants × 2 scopes × 2 metrics regresses by one error.
VOCR5's whole-dialogue gains are resolved overall, but a few titles still lose a character or two. By the owner's
instruction (13:26Z) VOCR5 becomes the overlay's default anyway.

## CLANNAD (once, against owner-drive-ingest's frozen references)

The three VOCR5 selections were read once through `eval/clannad_refs.py`'s extra-arms path (suffix `vocr5`, one
attempt, 20:14-20:28Z), beside VOCR4's three. The references are the frozen Claude references (3,048 scored lines,
`refs_sha256` `35411688…`). The units are the long run's GRAD line crops. Intervals come from 10,000 whole-page
bootstraps.

| model | dialogue CER [95% CI] (2,052 lines) | dialogue exact | all lines CER [95% CI] (3,048) | all exact |
|---|---|---|---|---|
| v0.2.0 H15 | 1.89% [1.72, 2.07] | 76.5% | 2.82% [2.61, 3.04] | 75.5% |
| VOCR4-FL10 | 1.49% [1.34, 1.65] | 80.6% | 3.05% [2.79, 3.32] | 76.0% |
| VOCR4-H15 | 1.43% [1.28, 1.58] | 81.9% | 2.79% [2.52, 3.06] | 79.6% |
| VOCR4-W20 | 0.89% [0.77, 1.01] | 88.2% | 1.81% [1.62, 2.00] | 84.9% |
| **VOCR5-S** | 1.38% [1.23, 1.54] | 82.5% | 2.07% [1.87, 2.27] | 83.0% |
| **VOCR5-N** | 0.96% [0.84, 1.08] | 87.4% | 1.45% [1.30, 1.61] | 87.0% |
| **VOCR5-L** | **0.73%** [0.63, 0.85] | 89.9% | **0.98%** [0.85, 1.11] | 90.9% |
| Google Lens (cached) | 1.53% [0.90, 2.25] | 96.5% | 1.47% [0.88, 2.14] | 97.5% |

| pair (all lines) | CER, points [95% CI] | exact lines, points [95% CI] |
|---|---|---|
| VOCR5-S − VOCR4-FL10 | −0.98 [−1.17, −0.79] | +6.9 [+5.4, +8.4] |
| VOCR5-N − VOCR4-W20 | −0.36 [−0.53, −0.18] | +2.1 [+0.8, +3.4] |
| VOCR5-N − VOCR4-H15 | −1.34 [−1.55, −1.13] | +7.4 [+6.0, +8.8] |
| VOCR5-L − VOCR4-W20 | −0.83 [−1.00, −0.66] | +6.0 [+4.7, +7.2] |
| VOCR5-L − VOCR5-N | −0.47 [−0.61, −0.34] | +3.9 [+2.7, +5.0] |
| VOCR5-S − Lens | +0.59 [−0.07, +1.20] | −14.5 [−15.9, −13.1] |
| VOCR5-N − Lens | −0.02 [−0.68, +0.58] | −10.4 [−11.7, −9.2] |
| VOCR5-L − Lens | −0.50 [−1.15, +0.09] | −6.6 [−7.7, −5.5] |

**On dialogue, VOCR5-N is slightly behind VOCR4-W20** (0.96% against 0.89%; their intervals overlap). On all
lines it is ahead (−0.36 points) because it reads speaker names far better (8.87% against 15.71%).

**With Lens's reading as the reference** (issue #90's view, `eval/vocr5_clannad_lens_view_20261016.json`; this is
agreement with Lens, not accuracy): on the dialogue lines VOCR5-L, -N and -S differ from Lens by 2.07%, 2.24% and
2.63% of its characters, and read 88.3%, 86.2% and 81.1% of lines exactly as Lens does.

**Disclosures.**

- **planetarian.** Every VOCR5 model trained on the owner's captures, which hold planetarian's PSP port (v34), a
  Key / Prototype release like CLANNAD's Vita port.
- **CLANNAD's cast names (Small).** VOCR5-S also trained on synth3 v2 (pretrain 0.25, fine-tune 0.125). Its renders
  hold CLANNAD's main cast names, which VNDB lists again under Tomoyo After and Hikari Mimamoru Sakamichi de; 7 crops
  show a full cast name (`docs/synth3.md`, Limits). synth3 v3, which Normal and Large read, excludes them.

## Hygiene and timeline (UTC)

| when | what |
|---|---|
| 2026-10-15 12:24-12:50Z | 1,000-step speed probes of the three shapes (`$V5/probe`, not runs; outputs deleted) |
| 12:42:04Z | protocol committed and pushed (tag `vocr5-runner-10b3160`); 12:42:28Z the Small pretrains start |
| 13:32Z | the s4v5631 render pinned (tag `vocr5-runner-f7a6924`); the Normal and Large pretrains start |
| 14:27Z | amendment 1 (rows-only refined copies) |
| 16:42Z | amendment 2 (Lens report-only arm) |
| 2026-10-16 03:55Z | the fine-tune pools pinned (tag `vocr5-runner-15f244b`, realplus v3 final) |
| 08:23Z | amendment 3 (Normal ≤ 1.6 MB, incumbent VOCR4-W20); scoring runner `vocr5-runner-aa3df3b` |
| 12:29-13:31Z | pretrains end; 12:42-19:19Z fine-tunes |
| 18:45-19:34Z | decisions, each committed before its tier's selection |
| 19:34-20:33Z | locked report, Lens, CLANNAD, deployed decode; 20:38Z release v0.3.0 |

- **Branch history.** The branch was rebased onto main twice, and author dates keep the times above:
  - onto #85, #87 and #88 (commits dated 2026-10-16 02:25Z, force-pushed 02:34Z), with the originals at tag
    `vocr5-history-20261016a` and the runner tags;
  - onto #89 (force-pushed 21:01:54Z), with the originals at tag `vocr5-history-20261016b` (`cbe8497`). That tag
    also holds `a5ed2d9`, the deployed-decode step's commit.
- **Code timing.** The `--rows-only` option of `eval/vocr5_refine.py` was written before amendment 1 decided to use
  it (vocr5-aug asked first).
- **Incumbent scores.** The incumbents' development scores existed from 04:10Z. Amendment 3 was decided by the sizing
  rule, before any candidate was scored.
- **Protected sets.** Nothing was chosen, tuned or dropped on vita-test, psp-test or CLANNAD. Each was read once, after
  the three selections were committed. The in-run evaluations (teacher_val's dev part, logged) selected nothing.
- **Smoke and pipeline runs.** None of these is a VOCR5 run or reached a selection:
  - `$V5/scratch/seltest/run.sh` (2026-10-15 13:05Z) scored VOCR3-FL10 on the development cohorts to test the
    scoring pipeline (score 4.28);
  - `$V5/scratch/smoke/run2.sh` (14:53-14:58Z) ran two 30-step fine-tunes (S F, N F) from early pretrain checkpoints
    on draft sources, then deleted them; their in-run teacher_val-dev evaluation was logged and not read;
  - the helper vocr5-aug's 40-step `fa=0.3` train2 smoke run (14:39Z).
- **Reserved speaker names in the synthetic data.** As above (Data, F1): 128 characters of 12 reserved release-partner
  VNs. None of their names occurs in a vita-test, psp-test or development line. Two short ones occur in 8 of CLANNAD's
  3,048 reference lines (2 of them speaker plates). Both are also common in ordinary text and are not CLANNAD names
  (audit-vocr5-b, b1).
- **Private data.** The owner's frames, the Lens readings and every CLANNAD reading stay under `shared/private`.
  Committed files hold counts, ids and hashes only, and no hostname (#88).

## Limits

- Two or three seeds per arm. A seed pair shares its pretrain, so P → F measures the fine-tune only.
- Nothing of VOCR5 ran on a console. Every time is a host estimate. The #85 runtime is merged, but its speed above
  the 2 MiB L2 (Large) is unconfirmed on the console (audit-ocr-speed-1 F5).
- The development sample is 116 VNs. teacher_val is mostly scaled PC screenshots. Names are a secondary column only.
- 563 of the new vocab's 610 kanji have under 80 synthetic exposures (50-127, median 67) and few real ones, so their
  recall is unmeasured beyond the dev sets.
- The Lens arm read line crops including furigana. It is not a measure of Lens on full screens.

## Reproduce

```sh
# pools:   $V5/build_pools.sh TAG RP3_INK RP3_GRAD RP3_REF; $V5/check_pools.sh TAG; eval/vocr5_pools.py sources|pin
# render:  synth/synth4.py curriculum|shards --vocab data/vocab_vn_5631.txt ... (eval/vocr5_render_s4v5631_20261015.json)
# runs:    eval/reproducibility/vocr5-20261015/queue.sh + feeder.sh (jobs.txt: eval/vocr5_run.sh TIER pre|P|F SEED)
# select:  eval/reproducibility/vocr5-20261015/select_tier.sh TIER export|score|arm|replace|scores|select
# report:  eval/reproducibility/vocr5-20261015/report.sh spec|shared|a9|overlay|render|lens|dialogue|lm|clannad
# issue 90: eval/vocr5_clannad_lens.py --private <attempt's score.vocr5.private.jsonl> --refs <WORK>/refs.jsonl
```

Artifacts: `$HERD/shared/artifacts/vocr5-train/` (runs, blobs, pools, aug, render, select, report, logs) and
`$HERD/shared/private/vocr5-train/` (the owner-pool copies, the Lens readings, the CLANNAD attempt: private).
