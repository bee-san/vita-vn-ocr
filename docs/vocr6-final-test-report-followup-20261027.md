# VOCR6 final test (read once)

Protocol `eval/vocr6_final_test_protocol.json` (sha256 `dc8f3ae38489cd37…`), repo `27e0126`, read 2026-10-27T20:03:42Z. Dialogue CER (names a secondary column), 95% intervals: game (VN) bootstrap for vita-test / psp-test, page bootstrap for CLANNAD and the test title; VOCR6 - VOCR5 paired over the same units. Host measurements; nothing ran on a console.

| tier | shipped blob (the VOCR6 column) | sha256 | bytes | W8I arena (VOCR6 / VOCR5) | head gate | two-level | is a VOCR5 blob |
|---|---|---|---:|---|---|---|---|
| S | `v6f4dlh_S_seed1_w8.vocr` | `864bbd196d5bc5d3…` | 819,792 | 124,832 / 124,832 B | False | False | no |
| N | `VOCR6_N_w8g2.vocr` | `107cf084cc0c1a3a…` | 1,189,408 | 182,944 / 182,944 B | True | True | no |
| L | `VOCR6_L_w8_2l.vocr` | `65310013e8f99088…` | 2,475,680 | 261,024 / 261,024 B | False | True | no |

VOCR5 on path (a) against its published records, in this read: 29 of 29 checks equal (all equal).

Path (b) is Rubi main `892cfb9`'s shipped line path: LM `vocr_lm6_o3_1000k.vlm` (lm6_1000k; alpha 600, beta 2500, flags 1), cascade under 700, finder TEXTBOX_RECOVERY + short-line probe + fix 1 (COLGAP; source checkout `27e0126`), refinement on PSP, textnorm `3fa190177bb2…`, the non-Japanese line filter; PSP in the 2x console view.

Rubi's ocr_vertical = auto (API 47): regions whose rows look like vertical text are read as columns by VOCR5-V (both families). The check runs on the frozen finder's rows for a region at least as tall as wide and on the fixed finder's rows otherwise, as Rubi's recognize does. Regions or pages read as columns: vita-test 0 of 246, psp-test 3 of 279, clannad 0 of 1837, pcsg00807 0 of 2055.

## Headline

### Path (b): Rubi's line path (what of it Rubi main ships: above)

**CLANNAD, 2,052 dialogue lines (the frozen line crops)** (seen flags: all_models, vocr6_training, clannad, clannad_glyph_raster, lm_lm6)

| tier | VOCR5 | VOCR6 | VOCR6 - VOCR5 (points) |
|---|---|---|---|
| S | 0.73% [0.61, 0.87] | 0.47% [0.36, 0.59] | -0.26 [-0.35, -0.18] |
| N | 0.66% [0.54, 0.79] | 0.33% [0.23, 0.45] | -0.33 [-0.41, -0.24] |
| L | 0.60% [0.48, 0.73] | 0.31% [0.21, 0.42] | -0.29 [-0.37, -0.21] |

Line-filter floor: 4 reference line(s), 26 characters (0.08 points) hold no Japanese character; Rubi's non-Japanese line filter deletes them when they are read right (both families alike).

**CLANNAD, 1,837 pages (Rubi's finder on the ROI; pages with a complete reference)** (seen flags: all_models, vocr6_training, clannad, clannad_glyph_raster, lm_lm6)

| tier | VOCR5 | VOCR6 | VOCR6 - VOCR5 (points) |
|---|---|---|---|
| S | 5.84% [5.23, 6.50] | 5.75% [5.13, 6.42] | -0.09 [-0.28, +0.10] |
| N | 5.94% [5.32, 6.61] | 5.81% [5.19, 6.47] | -0.13 [-0.31, +0.05] |
| L | 6.19% [5.56, 6.88] | 5.67% [5.06, 6.34] | -0.52 [-0.71, -0.32] |

Line-filter floor: 4 reference line(s), 26 characters (0.09 points, an upper bound) hold no Japanese character; Rubi's non-Japanese line filter deletes them when they are read right (both families alike).

**vita-test + psp-test fixed lines, 336 dialogue lines** (seen flags: all_models, vocr6_training, vita-test, lm_lm6)

| tier | VOCR5 | VOCR6 | VOCR6 - VOCR5 (points) |
|---|---|---|---|
| S | 0.68% [0.38, 1.06] | 0.58% [0.20, 1.08] | -0.11 [-0.33, +0.11] |
| N | 0.49% [0.21, 0.84] | 0.36% [0.13, 0.66] | -0.13 [-0.33, +0.04] |
| L | 0.47% [0.21, 0.79] | 0.40% [0.18, 0.69] | -0.06 [-0.18, +0.04] |

Line-filter floor: 0 reference line(s), 0 characters (0.00 points) hold no Japanese character; Rubi's non-Japanese line filter deletes them when they are read right (both families alike).

**vita-test + psp-test whole dialogue, 175 base regions** (seen flags: all_models, vocr6_training, vita-test, lm_lm6)

| tier | VOCR5 | VOCR6 | VOCR6 - VOCR5 (points) |
|---|---|---|---|
| S | 3.04% [1.84, 4.72] | 3.02% [1.74, 4.72] | -0.02 [-0.45, +0.45] |
| N | 2.83% [1.74, 4.29] | 2.70% [1.48, 4.36] | -0.13 [-0.55, +0.27] |
| L | 2.72% [1.65, 4.16] | 2.68% [1.49, 4.29] | -0.04 [-0.47, +0.39] |

Line-filter floor: 0 reference line(s), 0 characters (0.00 points, an upper bound) hold no Japanese character; Rubi's non-Japanese line filter deletes them when they are read right (both families alike).

**PCSG00807 clean lines (1,152)** (seen flags: all_models, vocr6_training, pcsg00807, lm_lm6)

| tier | VOCR5 | VOCR6 | VOCR6 - VOCR5 (points) |
|---|---|---|---|
| S | 0.27% [0.12, 0.44] | 0.21% [0.07, 0.38] | -0.06 [-0.11, -0.01] |
| N | 0.26% [0.12, 0.43] | 0.19% [0.06, 0.36] | -0.06 [-0.12, -0.02] |
| L | 0.22% [0.08, 0.40] | 0.19% [0.06, 0.35] | -0.04 [-0.08, -0.01] |

Line-filter floor: 6 reference line(s), 26 characters (0.19 points) hold no Japanese character; Rubi's non-Japanese line filter deletes them when they are read right (both families alike).

**PCSG00807 page path, 685 base regions** (seen flags: all_models, vocr6_training, pcsg00807, lm_lm6)

| tier | VOCR5 | VOCR6 | VOCR6 - VOCR5 (points) |
|---|---|---|---|
| S | 0.85% [0.56, 1.19] | 0.85% [0.55, 1.20] | +0.00 [-0.06, +0.07] |
| N | 0.84% [0.54, 1.19] | 0.80% [0.51, 1.14] | -0.04 [-0.08, +0.01] |
| L | 0.81% [0.51, 1.15] | 0.79% [0.49, 1.13] | -0.02 [-0.05, +0.00] |

Line-filter floor: 17 reference line(s), 75 characters (0.54 points, an upper bound) hold no Japanese character; Rubi's non-Japanese line filter deletes them when they are read right (both families alike).

### Path (a): VOCR5's locked path

**CLANNAD dialogue lines (greedy)** (seen flags: all_models, vocr6_training, clannad, clannad_glyph_raster)

| tier | VOCR5 | VOCR6 | VOCR6 - VOCR5 (points) |
|---|---|---|---|
| S | 1.38% [1.23, 1.54] | 0.72% [0.62, 0.83] | -0.66 [-0.78, -0.54] |
| N | 0.96% [0.84, 1.08] | 0.40% [0.32, 0.49] | -0.56 [-0.66, -0.45] |
| L | 0.73% [0.63, 0.85] | 0.26% [0.19, 0.33] | -0.48 [-0.57, -0.39] |

**vita-test + psp-test fixed lines, dialogue (greedy, console polarity)** (seen flags: all_models, vocr6_training, vita-test)

| tier | VOCR5 | VOCR6 | VOCR6 - VOCR5 (points) |
|---|---|---|---|
| S | 1.60% [0.84, 2.63] | 1.30% [0.78, 1.96] | -0.30 [-0.77, +0.09] |
| N | 1.13% [0.55, 1.90] | 0.81% [0.38, 1.34] | -0.32 [-0.69, -0.04] |
| L | 0.79% [0.44, 1.23] | 0.55% [0.31, 0.83] | -0.23 [-0.47, -0.04] |

**the same with beam 8 + ctclm.2's 1 MB LM** (seen flags: all_models, vocr6_training, vita-test, lm_ctclm2)

| tier | VOCR5 | VOCR6 | VOCR6 - VOCR5 (points) |
|---|---|---|---|
| S | 1.15% [0.62, 1.87] | 0.83% [0.40, 1.40] | -0.32 [-0.64, -0.04] |
| N | 0.72% [0.31, 1.27] | 0.68% [0.31, 1.18] | -0.04 [-0.27, +0.18] |
| L | 0.53% [0.27, 0.85] | 0.49% [0.25, 0.79] | -0.04 [-0.17, +0.07] |

**vita-test + psp-test whole dialogue (base, GRAD)** (seen flags: all_models, vocr6_training, vita-test)

| tier | VOCR5 | VOCR6 | VOCR6 - VOCR5 (points) |
|---|---|---|---|
| S | 8.31% [4.92, 12.69] | 8.24% [4.77, 12.77] | -0.06 [-0.55, +0.48] |
| N | 7.98% [4.74, 12.23] | 7.34% [4.16, 11.58] | -0.65 [-1.14, -0.16] |
| L | 7.08% [3.97, 11.16] | 6.58% [3.41, 10.83] | -0.50 [-0.88, -0.11] |

**PCSG00807 clean lines (rubi_cascade)** (seen flags: all_models, vocr6_training, pcsg00807, lm_ctclm2)

| tier | VOCR5 | VOCR6 | VOCR6 - VOCR5 (points) |
|---|---|---|---|
| S | 0.14% [0.07, 0.24] | 0.09% [0.04, 0.16] | -0.05 [-0.10, +0.00] |
| N | 0.12% [0.06, 0.20] | 0.05% [0.01, 0.10] | -0.07 [-0.13, -0.02] |
| L | 0.07% [0.02, 0.13] | 0.04% [0.01, 0.08] | -0.04 [-0.09, +0.01] |

**PCSG00807 page GRAD + refine (base, rubi_cascade)** (seen flags: all_models, vocr6_training, pcsg00807, lm_ctclm2)

| tier | VOCR5 | VOCR6 | VOCR6 - VOCR5 (points) |
|---|---|---|---|
| S | 1.66% [1.25, 2.12] | 1.64% [1.21, 2.10] | -0.03 [-0.11, +0.04] |
| N | 1.71% [1.28, 2.18] | 1.64% [1.21, 2.11] | -0.07 [-0.14, -0.01] |
| L | 1.67% [1.24, 2.15] | 1.65% [1.23, 2.12] | -0.02 [-0.08, +0.04] |

## Vita and PSP apart

Seen flags: all_models, vocr6_training, vita-test; the decoded rows also lm_lm6 (path b) or lm_ctclm2 (path a's + LM).

| measure | tier | vita-test VOCR5 | vita-test VOCR6 | vita-test Δ | psp-test VOCR5 | psp-test VOCR6 | psp-test Δ |
|---|---|---|---|---|---|---|---|
| (b) fixed lines | S | 0.67% [0.32, 1.19] | 0.42% [0.11, 0.94] | -0.25 [-0.55, -0.03] | 0.70% [0.25, 1.31] | 0.74% [0.12, 1.67] | +0.04 [-0.28, +0.43] |
| (b) fixed lines | N | 0.50% [0.19, 0.96] | 0.17% [0.00, 0.49] | -0.33 [-0.66, -0.11] | 0.48% [0.08, 1.02] | 0.56% [0.19, 1.09] | +0.09 [-0.19, +0.28] |
| (b) fixed lines | L | 0.42% [0.13, 0.82] | 0.33% [0.08, 0.71] | -0.08 [-0.23, +0.00] | 0.52% [0.12, 1.09] | 0.48% [0.14, 0.94] | -0.04 [-0.24, +0.14] |
| (b) whole dialogue base | S | 4.07% [1.99, 7.06] | 3.90% [1.60, 6.98] | -0.17 [-0.86, +0.63] | 2.00% [1.07, 3.01] | 2.13% [1.10, 3.30] | +0.13 [-0.40, +0.69] |
| (b) whole dialogue base | N | 3.47% [1.57, 6.09] | 3.26% [1.15, 6.28] | -0.21 [-1.00, +0.41] | 2.17% [1.25, 3.17] | 2.13% [1.16, 3.19] | -0.04 [-0.42, +0.45] |
| (b) whole dialogue base | L | 3.56% [1.70, 6.13] | 3.21% [1.09, 6.17] | -0.34 [-0.98, +0.21] | 1.87% [0.93, 2.88] | 2.13% [1.20, 3.12] | +0.26 [-0.28, +0.96] |
| (a) fixed lines greedy | S | 1.00% [0.48, 1.80] | 0.92% [0.45, 1.63] | -0.08 [-0.46, +0.26] | 2.22% [0.84, 4.16] | 1.69% [0.82, 2.90] | -0.52 [-1.39, +0.18] |
| (a) fixed lines greedy | N | 0.96% [0.37, 1.97] | 0.46% [0.13, 0.95] | -0.50 [-1.18, -0.08] | 1.30% [0.41, 2.53] | 1.17% [0.40, 2.17] | -0.13 [-0.47, +0.22] |
| (a) fixed lines greedy | L | 0.59% [0.23, 1.09] | 0.42% [0.12, 0.85] | -0.17 [-0.41, +0.00] | 1.00% [0.40, 1.70] | 0.70% [0.35, 1.10] | -0.30 [-0.70, +0.05] |
| (a) fixed lines + LM | S | 1.05% [0.52, 1.82] | 0.67% [0.21, 1.44] | -0.38 [-0.71, -0.10] | 1.26% [0.40, 2.62] | 1.00% [0.31, 1.91] | -0.26 [-0.87, +0.22] |
| (a) fixed lines + LM | N | 0.79% [0.34, 1.47] | 0.46% [0.12, 1.03] | -0.33 [-0.70, -0.04] | 0.65% [0.09, 1.55] | 0.91% [0.32, 1.81] | +0.26 [+0.04, +0.51] |
| (a) fixed lines + LM | L | 0.54% [0.22, 0.99] | 0.38% [0.13, 0.74] | -0.17 [-0.38, +0.00] | 0.52% [0.12, 1.05] | 0.61% [0.22, 1.12] | +0.09 [+0.00, +0.22] |
| (a) whole dialogue base | S | 5.57% [3.29, 8.44] | 5.36% [3.02, 8.25] | -0.21 [-1.06, +0.68] | 11.08% [5.03, 19.24] | 11.17% [4.99, 19.58] | +0.09 [-0.50, +0.68] |
| (a) whole dialogue base | N | 5.31% [3.05, 8.03] | 4.58% [2.38, 7.37] | -0.73 [-1.49, +0.05] | 10.69% [4.86, 18.55] | 10.13% [4.55, 17.90] | -0.56 [-1.26, +0.05] |
| (a) whole dialogue base | L | 4.28% [2.21, 6.85] | 3.98% [1.61, 6.89] | -0.30 [-0.82, +0.20] | 9.91% [4.41, 17.54] | 9.21% [3.71, 17.02] | -0.70 [-1.31, -0.13] |

## Names (secondary)

Seen flags as in the headline tables of each set.

| measure | tier | VOCR5 | VOCR6 | Δ |
|---|---|---|---|---|
| (b) vita+psp fixed lines | S | 3.67% [0.85, 7.01] | 4.72% [1.71, 8.23] | +1.05 [+0.00, +2.29] |
| (b) vita+psp fixed lines | N | 3.67% [0.84, 7.40] | 2.62% [0.52, 5.73] | -1.05 [-3.16, +0.59] |
| (b) vita+psp fixed lines | L | 3.41% [0.56, 7.07] | 2.62% [0.53, 5.71] | -0.79 [-2.30, +0.30] |
| (b) vita+psp whole dialogue base | S | 15.75% [9.35, 22.45] | 14.44% [8.19, 21.10] | -1.31 [-3.26, +0.56] |
| (b) vita+psp whole dialogue base | N | 14.96% [8.62, 21.73] | 13.12% [6.99, 19.81] | -1.84 [-3.58, -0.22] |
| (b) vita+psp whole dialogue base | L | 14.70% [8.27, 21.57] | 13.65% [7.49, 20.34] | -1.05 [-2.73, +0.51] |
| (b) CLANNAD lines | S | 6.02% [4.89, 7.21] | 6.47% [5.27, 7.74] | +0.45 [-0.64, +1.52] |
| (b) CLANNAD lines | N | 7.56% [6.29, 8.91] | 4.62% [3.57, 5.75] | -2.94 [-4.46, -1.41] |
| (b) CLANNAD lines | L | 8.56% [7.23, 9.95] | 3.44% [2.57, 4.39] | -5.12 [-6.44, -3.80] |
| (a) vita+psp fixed | S | 4.72% [1.30, 9.04] | 5.51% [1.84, 9.97] | +0.79 [-0.53, +2.20] |
| (a) vita+psp fixed | N | 3.41% [0.74, 7.37] | 1.84% [0.49, 3.62] | -1.57 [-3.81, +0.00] |
| (a) vita+psp fixed | L | 2.62% [0.52, 5.48] | 1.31% [0.29, 2.49] | -1.31 [-3.34, +0.00] |
| (a) vita+psp whole dialogue base | S | 29.92% [20.81, 40.46] | 31.23% [22.25, 41.53] | +1.31 [-0.33, +3.12] |
| (a) vita+psp whole dialogue base | N | 26.25% [18.26, 34.41] | 25.46% [18.10, 33.01] | -0.79 [-2.58, +1.04] |
| (a) vita+psp whole dialogue base | L | 24.41% [17.66, 31.21] | 23.88% [16.84, 31.10] | -0.52 [-3.24, +2.22] |
| (a) CLANNAD lines | S | 12.45% [10.46, 14.54] | 8.24% [6.65, 9.87] | -4.21 [-6.33, -2.18] |
| (a) CLANNAD lines | N | 8.87% [7.27, 10.54] | 11.23% [9.54, 13.01] | +2.35 [+0.27, +4.49] |
| (a) CLANNAD lines | L | 4.66% [3.49, 5.90] | 1.45% [0.94, 2.04] | -3.21 [-4.34, -2.14] |

## Path (b) decomposition (dialogue CER)

Seen flags as in the headline tables; the decoded stages also lm_lm6. The shipped stage includes the line filter's floor (headline tables).

| measure | family | tier | greedy | + counter / LM (tier alone) | + cascade | shipped (textnorm, filter) |
|---|---|---|---|---|---|---|
| CLANNAD lines | VOCR5 | S | 1.38% | 0.77% | 0.65% | 0.73% |
| CLANNAD lines | VOCR5 | N | 0.96% | 0.59% | 0.58% | 0.66% |
| CLANNAD lines | VOCR5 | L | 0.75% | 0.52% | 0.52% | 0.60% |
| CLANNAD lines | VOCR6 | S | 0.72% | 0.56% | 0.42% | 0.47% |
| CLANNAD lines | VOCR6 | N | 0.41% | 0.32% | 0.25% | 0.33% |
| CLANNAD lines | VOCR6 | L | 0.26% | 0.23% | 0.23% | 0.31% |
| CLANNAD pages | VOCR5 | S | 9.82% | 7.74% | 6.46% | 5.84% |
| CLANNAD pages | VOCR5 | N | 7.31% | 6.54% | 7.23% | 5.94% |
| CLANNAD pages | VOCR5 | L | 7.54% | 7.50% | 7.50% | 6.19% |
| CLANNAD pages | VOCR6 | S | 8.15% | 7.07% | 6.10% | 5.75% |
| CLANNAD pages | VOCR6 | N | 8.29% | 6.69% | 6.08% | 5.81% |
| CLANNAD pages | VOCR6 | L | 6.07% | 6.03% | 6.03% | 5.67% |
| vita+psp fixed lines | VOCR5 | S | 1.53% | 1.07% | 0.68% | 0.68% |
| vita+psp fixed lines | VOCR5 | N | 1.07% | 0.66% | 0.49% | 0.49% |
| vita+psp fixed lines | VOCR5 | L | 0.77% | 0.47% | 0.47% | 0.47% |
| vita+psp fixed lines | VOCR6 | S | 1.19% | 0.77% | 0.58% | 0.58% |
| vita+psp fixed lines | VOCR6 | N | 0.81% | 0.53% | 0.36% | 0.36% |
| vita+psp fixed lines | VOCR6 | L | 0.51% | 0.40% | 0.40% | 0.40% |
| vita+psp whole dialogue base | VOCR5 | S | 4.21% | 3.71% | 3.19% | 3.04% |
| vita+psp whole dialogue base | VOCR5 | N | 3.80% | 3.28% | 2.98% | 2.83% |
| vita+psp whole dialogue base | VOCR5 | L | 3.15% | 2.83% | 2.83% | 2.72% |
| vita+psp whole dialogue base | VOCR6 | S | 3.80% | 3.69% | 3.34% | 3.02% |
| vita+psp whole dialogue base | VOCR6 | N | 3.47% | 3.37% | 2.72% | 2.70% |
| vita+psp whole dialogue base | VOCR6 | L | 2.83% | 2.70% | 2.70% | 2.68% |
| PCSG00807 clean | VOCR5 | S | 0.27% | 0.29% | 0.19% | 0.27% |
| PCSG00807 clean | VOCR5 | N | 0.17% | 0.12% | 0.12% | 0.26% |
| PCSG00807 clean | VOCR5 | L | 0.11% | 0.06% | 0.06% | 0.22% |
| PCSG00807 clean | VOCR6 | S | 0.12% | 0.14% | 0.06% | 0.21% |
| PCSG00807 clean | VOCR6 | N | 0.09% | 0.03% | 0.04% | 0.19% |
| PCSG00807 clean | VOCR6 | L | 0.01% | 0.04% | 0.04% | 0.19% |
| PCSG00807 page base | VOCR5 | S | 0.94% | 0.90% | 0.85% | 0.85% |
| PCSG00807 page base | VOCR5 | N | 0.93% | 0.87% | 0.84% | 0.84% |
| PCSG00807 page base | VOCR5 | L | 0.82% | 0.81% | 0.81% | 0.81% |
| PCSG00807 page base | VOCR6 | S | 0.87% | 0.86% | 0.85% | 0.85% |
| PCSG00807 page base | VOCR6 | N | 0.81% | 0.82% | 0.80% | 0.80% |
| PCSG00807 page base | VOCR6 | L | 0.80% | 0.79% | 0.79% | 0.79% |
| PCSG00807 oracle boxes base | VOCR5 | S | 0.56% | 0.63% | 0.33% | 0.63% |
| PCSG00807 oracle boxes base | VOCR5 | N | 0.32% | 0.22% | 0.22% | 0.62% |
| PCSG00807 oracle boxes base | VOCR5 | L | 0.24% | 0.09% | 0.09% | 0.57% |
| PCSG00807 oracle boxes base | VOCR6 | S | 0.25% | 0.33% | 0.08% | 0.56% |
| PCSG00807 oracle boxes base | VOCR6 | N | 0.14% | 0.04% | 0.14% | 0.55% |
| PCSG00807 oracle boxes base | VOCR6 | L | 0.06% | 0.13% | 0.13% | 0.54% |

PCSG00807's base regions through Rubi's finder: 1,167 reference lines: 1,139 whole, 5 cut, 23 missed; 1,145 boxes, 0 merged, 0 junk.

## Per title (vita-test, psp-test; dialogue)

Intervals per title are in report.json. 'seen': the title's own flag (seen_flags.vita-test); every title also carries all_models and vocr6_training.

**(b) vita psp fixed lines, shipped**

| set | VN | title | chars | 5S | 6S | 5N | 6N | 5L | 6L | seen |
|---|---|---|---:|---|---|---|---|---|---|---|
| psp-test | v11095 | Kimikare | 197 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |  |
| psp-test | v11504 | Shiratsuyu no Kai | 239 | 0.42% | 0.42% | 0.00% | 0.42% | 0.00% | 0.42% |  |
| psp-test | v11708 | Tanteibu: The Detective Club -Haibu to Kaiga to Bakudan to- | 91 | 2.20% | 2.20% | 3.30% | 1.10% | 2.20% | 1.10% |  |
| psp-test | v12155 | -8 | 144 | 0.69% | 0.00% | 0.00% | 0.00% | 0.00% | 0.69% |  |
| psp-test | v12865 | Geten no Hana Yume Akari | 144 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |  |
| psp-test | v12940 | Chronostacia | 159 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |  |
| psp-test | v15910 | Meiji Tokyo Renka Twilight Kiss | 269 | 0.37% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |  |
| psp-test | v3396 | S.Y.K ~Renshouden~ | 221 | 0.90% | 0.00% | 0.45% | 0.45% | 0.45% | 0.00% |  |
| psp-test | v3542 | Hoshigari Empusa! | 108 | 2.78% | 3.70% | 2.78% | 2.78% | 2.78% | 2.78% |  |
| psp-test | v4147 | Juujigen Rippoutai Cipher ~Aoki Tsuki no Suitei~ | 71 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |  |
| psp-test | v4671 | Okashi na Shima no Peter Pan ~Sweet Never Land~ | 208 | 0.00% | 0.48% | 0.00% | 0.48% | 0.00% | 0.00% |  |
| psp-test | v7634 | Hanayaka Nari, Waga Ichizoku | 167 | 0.00% | 0.00% | 0.00% | 0.60% | 0.60% | 0.60% |  |
| vita-test | v10856 | Time Travelers | 231 | 2.60% | 3.90% | 2.16% | 2.16% | 2.16% | 1.73% |  |
| vita-test | v13072 | Kin'iro no Corda 3 AnotherSky | 128 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |  |
| vita-test | v16460 | Sweet Clown ~Gozen San-ji no Okashi na Doukeshi~ | 349 | 0.29% | 0.29% | 0.00% | 0.00% | 0.00% | 0.00% |  |
| vita-test | v16566 | Saenai Heroine no Sodatekata -blessing flowers- | 219 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |  |
| vita-test | v17286 | BELIEVER! | 177 | 0.56% | 0.00% | 0.56% | 0.00% | 0.56% | 0.56% |  |
| vita-test | v17750 | Kenka Banchou Otome | 141 | 0.71% | 0.00% | 1.42% | 0.00% | 0.00% | 0.00% |  |
| vita-test | v17829 | Naderebo! | 117 | 3.42% | 4.27% | 2.56% | 2.56% | 2.56% | 2.56% |  |
| vita-test | v18030 | Kin'iro no Corda 4 | 104 | 0.96% | 0.96% | 0.96% | 0.00% | 0.96% | 0.96% |  |
| vita-test | v19339 | Reco Love Gold Beach | 82 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |  |
| vita-test | v19850 | VARIABLE BARRICADE | 86 | 2.33% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |  |
| vita-test | v20326 | Utsusemi no Meguri | 168 | 0.60% | 0.60% | 0.00% | 0.00% | 0.00% | 0.00% |  |
| vita-test | v21782 | Piofiore no Banshou | 129 | 1.55% | 0.78% | 0.00% | 0.00% | 0.78% | 0.00% |  |
| vita-test | v21789 | Collar x Malice -Unlimited- | 362 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | yes |
| vita-test | v22122 | Gunka o Haita Neko | 114 | 0.88% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |  |
| vita-test | v3820 | Akai Suna Ochiru Tsuki | 114 | 1.75% | 0.88% | 1.75% | 0.88% | 1.75% | 0.88% |  |
| vita-test | v7721 | Utawarerumono: Itsuwari no Kamen | 154 | 0.00% | 0.00% | 1.30% | 0.00% | 1.30% | 1.30% |  |

**(b) vita psp whole dialogue, shipped**

| set | VN | title | chars | 5S | 6S | 5N | 6N | 5L | 6L | seen |
|---|---|---|---:|---|---|---|---|---|---|---|
| psp-test | v11095 | Kimikare | 197 | 4.06% | 4.06% | 4.57% | 4.06% | 4.57% | 4.06% |  |
| psp-test | v11504 | Shiratsuyu no Kai | 239 | 1.26% | 0.42% | 1.67% | 1.26% | 1.67% | 1.67% |  |
| psp-test | v11708 | Tanteibu: The Detective Club -Haibu to Kaiga to Bakudan to- | 91 | 5.49% | 2.20% | 2.20% | 2.20% | 2.20% | 2.20% |  |
| psp-test | v12155 | -8 | 144 | 3.47% | 4.86% | 4.17% | 4.86% | 2.78% | 4.86% |  |
| psp-test | v12865 | Geten no Hana Yume Akari | 144 | 0.00% | 2.08% | 1.39% | 2.08% | 0.00% | 2.08% |  |
| psp-test | v12940 | Chronostacia | 159 | 0.63% | 1.26% | 0.63% | 3.14% | 0.00% | 3.14% |  |
| psp-test | v15910 | Meiji Tokyo Renka Twilight Kiss | 269 | 0.37% | 0.00% | 0.74% | 0.00% | 0.74% | 0.00% |  |
| psp-test | v3396 | S.Y.K ~Renshouden~ | 221 | 0.90% | 0.45% | 0.45% | 0.00% | 0.00% | 0.00% |  |
| psp-test | v3542 | Hoshigari Empusa! | 108 | 5.56% | 5.56% | 6.48% | 5.56% | 5.56% | 5.56% |  |
| psp-test | v4147 | Juujigen Rippoutai Cipher ~Aoki Tsuki no Suitei~ | 71 | 0.00% | 1.41% | 0.00% | 0.00% | 0.00% | 1.41% |  |
| psp-test | v4671 | Okashi na Shima no Peter Pan ~Sweet Never Land~ | 208 | 3.37% | 4.81% | 3.37% | 3.37% | 3.37% | 2.88% |  |
| psp-test | v7634 | Hanayaka Nari, Waga Ichizoku | 167 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |  |
| vita-test | v10856 | Time Travelers | 231 | 3.46% | 3.46% | 3.46% | 3.46% | 3.46% | 3.03% |  |
| vita-test | v13072 | Kin'iro no Corda 3 AnotherSky | 128 | 0.00% | 0.00% | 0.78% | 0.00% | 0.78% | 0.00% |  |
| vita-test | v16460 | Sweet Clown ~Gozen San-ji no Okashi na Doukeshi~ | 349 | 3.44% | 2.58% | 2.87% | 2.87% | 2.87% | 3.15% |  |
| vita-test | v16566 | Saenai Heroine no Sodatekata -blessing flowers- | 219 | 16.44% | 15.53% | 14.16% | 15.98% | 14.16% | 15.53% |  |
| vita-test | v17286 | BELIEVER! | 177 | 1.69% | 2.82% | 0.56% | 0.56% | 2.82% | 0.56% |  |
| vita-test | v17750 | Kenka Banchou Otome | 141 | 8.51% | 12.77% | 6.38% | 7.09% | 6.38% | 7.09% |  |
| vita-test | v17829 | Naderebo! | 117 | 4.27% | 3.42% | 5.13% | 1.71% | 5.13% | 1.71% |  |
| vita-test | v18030 | Kin'iro no Corda 4 | 104 | 1.92% | 0.96% | 1.92% | 0.96% | 1.92% | 1.92% |  |
| vita-test | v19339 | Reco Love Gold Beach | 62 | 4.84% | 4.84% | 4.84% | 4.84% | 4.84% | 4.84% |  |
| vita-test | v19850 | VARIABLE BARRICADE | 86 | 3.49% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |  |
| vita-test | v20326 | Utsusemi no Meguri | 168 | 1.79% | 0.60% | 0.60% | 0.60% | 0.60% | 0.00% |  |
| vita-test | v21782 | Piofiore no Banshou | 129 | 3.10% | 4.65% | 5.43% | 1.55% | 3.88% | 2.33% |  |
| vita-test | v21789 | Collar x Malice -Unlimited- | 324 | 0.93% | 0.93% | 0.62% | 0.93% | 0.62% | 0.00% | yes |
| vita-test | v22122 | Gunka o Haita Neko | 114 | 1.75% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |  |
| vita-test | v3820 | Akai Suna Ochiru Tsuki | 114 | 6.14% | 6.14% | 6.14% | 7.02% | 6.14% | 6.14% |  |
| vita-test | v7721 | Utawarerumono: Itsuwari no Kamen | 154 | 0.00% | 0.00% | 1.30% | 0.00% | 1.30% | 1.30% |  |

**(a) vita psp fixed lines, greedy**

| set | VN | title | chars | 5S | 6S | 5N | 6N | 5L | 6L | seen |
|---|---|---|---:|---|---|---|---|---|---|---|
| psp-test | v11095 | Kimikare | 197 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |  |
| psp-test | v11504 | Shiratsuyu no Kai | 239 | 1.67% | 2.09% | 0.00% | 0.00% | 1.26% | 0.84% |  |
| psp-test | v11708 | Tanteibu: The Detective Club -Haibu to Kaiga to Bakudan to- | 91 | 6.59% | 3.30% | 4.40% | 4.40% | 1.10% | 1.10% |  |
| psp-test | v12155 | -8 | 144 | 2.78% | 0.69% | 0.69% | 0.00% | 0.69% | 0.69% |  |
| psp-test | v12865 | Geten no Hana Yume Akari | 144 | 0.00% | 0.69% | 0.00% | 1.39% | 0.69% | 0.00% |  |
| psp-test | v12940 | Chronostacia | 159 | 0.00% | 0.63% | 0.00% | 0.00% | 0.00% | 0.00% |  |
| psp-test | v15910 | Meiji Tokyo Renka Twilight Kiss | 269 | 1.86% | 0.74% | 0.37% | 0.00% | 0.00% | 0.37% |  |
| psp-test | v3396 | S.Y.K ~Renshouden~ | 221 | 3.17% | 2.71% | 3.17% | 2.71% | 3.17% | 1.36% |  |
| psp-test | v3542 | Hoshigari Empusa! | 108 | 2.78% | 3.70% | 1.85% | 2.78% | 1.85% | 1.85% |  |
| psp-test | v4147 | Juujigen Rippoutai Cipher ~Aoki Tsuki no Suitei~ | 71 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |  |
| psp-test | v4671 | Okashi na Shima no Peter Pan ~Sweet Never Land~ | 208 | 0.00% | 0.48% | 0.48% | 0.48% | 0.48% | 0.00% |  |
| psp-test | v7634 | Hanayaka Nari, Waga Ichizoku | 167 | 0.60% | 0.60% | 0.60% | 0.60% | 0.00% | 0.60% |  |
| vita-test | v10856 | Time Travelers | 231 | 9.09% | 5.63% | 5.63% | 3.90% | 2.60% | 1.73% |  |
| vita-test | v13072 | Kin'iro no Corda 3 AnotherSky | 128 | 0.00% | 0.78% | 0.78% | 0.78% | 0.78% | 0.78% |  |
| vita-test | v16460 | Sweet Clown ~Gozen San-ji no Okashi na Doukeshi~ | 349 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |  |
| vita-test | v16566 | Saenai Heroine no Sodatekata -blessing flowers- | 219 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |  |
| vita-test | v17286 | BELIEVER! | 177 | 0.56% | 1.13% | 0.56% | 0.00% | 0.56% | 0.56% |  |
| vita-test | v17750 | Kenka Banchou Otome | 141 | 0.71% | 0.71% | 1.42% | 0.00% | 0.00% | 0.00% |  |
| vita-test | v17829 | Naderebo! | 117 | 3.42% | 5.13% | 3.42% | 3.42% | 2.56% | 2.56% |  |
| vita-test | v18030 | Kin'iro no Corda 4 | 104 | 0.96% | 0.96% | 0.96% | 0.00% | 0.96% | 0.00% |  |
| vita-test | v19339 | Reco Love Gold Beach | 82 | 1.22% | 1.22% | 0.00% | 0.00% | 1.22% | 0.00% |  |
| vita-test | v19850 | VARIABLE BARRICADE | 86 | 2.33% | 1.16% | 0.00% | 1.16% | 1.16% | 0.00% |  |
| vita-test | v20326 | Utsusemi no Meguri | 168 | 1.19% | 0.60% | 0.60% | 0.60% | 0.00% | 0.60% |  |
| vita-test | v21782 | Piofiore no Banshou | 129 | 5.43% | 3.10% | 6.98% | 2.33% | 2.33% | 2.33% |  |
| vita-test | v21789 | Collar x Malice -Unlimited- | 362 | 0.83% | 0.55% | 0.28% | 0.28% | 0.00% | 0.00% | yes |
| vita-test | v22122 | Gunka o Haita Neko | 114 | 0.00% | 0.00% | 0.88% | 0.00% | 0.00% | 0.00% |  |
| vita-test | v3820 | Akai Suna Ochiru Tsuki | 114 | 0.88% | 0.88% | 0.88% | 0.88% | 1.75% | 0.88% |  |
| vita-test | v7721 | Utawarerumono: Itsuwari no Kamen | 154 | 0.65% | 1.30% | 0.65% | 0.00% | 1.30% | 0.65% |  |

**(a) vita psp whole dialogue, greedy**

| set | VN | title | chars | 5S | 6S | 5N | 6N | 5L | 6L | seen |
|---|---|---|---:|---|---|---|---|---|---|---|
| psp-test | v11095 | Kimikare | 197 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |  |
| psp-test | v11504 | Shiratsuyu no Kai | 239 | 1.67% | 2.51% | 2.09% | 0.84% | 1.67% | 1.26% |  |
| psp-test | v11708 | Tanteibu: The Detective Club -Haibu to Kaiga to Bakudan to- | 91 | 3.30% | 3.30% | 1.10% | 1.10% | 1.10% | 1.10% |  |
| psp-test | v12155 | -8 | 144 | 27.08% | 25.00% | 23.61% | 22.92% | 24.31% | 24.31% |  |
| psp-test | v12865 | Geten no Hana Yume Akari | 144 | 17.36% | 19.44% | 18.75% | 18.06% | 19.44% | 16.67% |  |
| psp-test | v12940 | Chronostacia | 159 | 49.69% | 51.57% | 49.06% | 49.06% | 47.17% | 48.43% |  |
| psp-test | v15910 | Meiji Tokyo Renka Twilight Kiss | 269 | 1.12% | 0.37% | 1.12% | 1.49% | 1.12% | 0.37% |  |
| psp-test | v3396 | S.Y.K ~Renshouden~ | 221 | 9.05% | 9.50% | 9.05% | 7.69% | 6.33% | 6.79% |  |
| psp-test | v3542 | Hoshigari Empusa! | 108 | 15.74% | 16.67% | 16.67% | 14.81% | 14.81% | 12.96% |  |
| psp-test | v4147 | Juujigen Rippoutai Cipher ~Aoki Tsuki no Suitei~ | 71 | 0.00% | 1.41% | 0.00% | 0.00% | 0.00% | 0.00% |  |
| psp-test | v4671 | Okashi na Shima no Peter Pan ~Sweet Never Land~ | 208 | 3.85% | 2.88% | 2.88% | 4.33% | 4.33% | 2.40% |  |
| psp-test | v7634 | Hanayaka Nari, Waga Ichizoku | 167 | 0.60% | 0.60% | 2.40% | 1.80% | 0.60% | 0.60% |  |
| vita-test | v10856 | Time Travelers | 231 | 18.18% | 17.32% | 16.45% | 14.29% | 12.99% | 11.69% |  |
| vita-test | v13072 | Kin'iro no Corda 3 AnotherSky | 128 | 12.50% | 10.94% | 9.38% | 9.38% | 9.38% | 7.03% |  |
| vita-test | v16460 | Sweet Clown ~Gozen San-ji no Okashi na Doukeshi~ | 349 | 3.72% | 3.15% | 3.44% | 2.87% | 3.15% | 2.87% |  |
| vita-test | v16566 | Saenai Heroine no Sodatekata -blessing flowers- | 219 | 10.50% | 11.87% | 11.42% | 10.05% | 9.13% | 10.05% |  |
| vita-test | v17286 | BELIEVER! | 177 | 1.13% | 5.08% | 1.13% | 2.26% | 0.56% | 0.00% |  |
| vita-test | v17750 | Kenka Banchou Otome | 141 | 4.96% | 3.55% | 2.13% | 4.26% | 1.42% | 0.71% |  |
| vita-test | v17829 | Naderebo! | 117 | 6.84% | 2.56% | 7.69% | 2.56% | 2.56% | 3.42% |  |
| vita-test | v18030 | Kin'iro no Corda 4 | 104 | 0.96% | 0.96% | 0.96% | 0.00% | 0.96% | 0.96% |  |
| vita-test | v19339 | Reco Love Gold Beach | 62 | 4.84% | 4.84% | 6.45% | 6.45% | 6.45% | 4.84% |  |
| vita-test | v19850 | VARIABLE BARRICADE | 86 | 2.33% | 1.16% | 1.16% | 1.16% | 1.16% | 0.00% |  |
| vita-test | v20326 | Utsusemi no Meguri | 168 | 7.74% | 7.74% | 7.74% | 7.14% | 7.14% | 7.14% |  |
| vita-test | v21782 | Piofiore no Banshou | 129 | 19.38% | 18.60% | 16.28% | 15.50% | 14.73% | 15.50% |  |
| vita-test | v21789 | Collar x Malice -Unlimited- | 324 | 3.09% | 2.47% | 3.40% | 1.23% | 1.85% | 0.00% | yes |
| vita-test | v22122 | Gunka o Haita Neko | 114 | 0.88% | 0.88% | 1.75% | 0.00% | 0.00% | 0.00% |  |
| vita-test | v3820 | Akai Suna Ochiru Tsuki | 114 | 16.67% | 15.79% | 16.67% | 16.67% | 15.79% | 15.79% |  |
| vita-test | v7721 | Utawarerumono: Itsuwari no Kamen | 154 | 0.65% | 1.30% | 0.65% | 0.65% | 1.30% | 1.30% |  |

## Seen flags and disclosures

- **all_models**: every VOCR5 and VOCR6 model trained on the owner's console captures, which include planetarian's PSP port (v34), a Key / Prototype release like CLANNAD's Vita port
- **vocr6_training**: 38 of the 49 hooked titles' VNs were in VOCR5's own fine-tune pools (data/manifests/vocr6_hooked/overlap_v1.json seen_by_vocr5); VOCR6 trained on every hooked train title
- **vita-test**: v21789: Collar x Malice -Unlimited- shares four releases (r57056, r64104, r69311, r69312) with the hooked train title Collar x Malice (PCSG00866, v18160): VOCR6 trained on it, and its text (825 lines) is in the lm6 LMs
- **clannad**: Little Busters! Converted Edition (PCSG00007, v5) and planetarian (ULJM05383, v34) share the Key 10th Memorial BOX (r3748) with CLANNAD (v4): VOCR6 trained on both and their text is in the lm6 LMs; planetarian's PSP port is in the owner captures every model trained on
- **pcsg00807**: the test title's VN v18432 is a VOCR5 development-sample VN (4 dev lines; VOCR6's dev comparisons leave it out); no VOCR6 run trained on the title (test split) and the lm6 LMs hold no test-title text
- **lm_ctclm2**: ctclm.2's 1 MB LM: its Jiten word list includes the scripts of CLANNAD, 7 of vita-test's 16 games and 1 of psp-test's 14 (their names removed; README, docs/ctc-lm-decode.md)
- **lm_lm6**: the lm6 LMs add the hooked train titles' text (docs/vocr6-lm.md on #137): Little Busters!, planetarian, Collar x Malice among them; the Jiten caveat of ctclm.2 still holds
- **vocr5_dev_sample**: PCSG00576 (v12188) trains VOCR6 and is a development-sample VN (no protected set)
- **clannad_glyph_raster**: vocr6-lead's stage 4 final_test (eval/vocr6_protocol.json, feat/vocr6-t1-stages 7bdc052; vocr6-synth6's finding): two hooked train titles draw dialogue in a font whose 32 px glyphs are CLANNAD's own raster re-quantised, PCSG00620 Grisaia Spinout (1,696 lines) and PCSG00007 Little Busters! (878 lines): 39,447 glyphs of 1,201 characters; synth6 v3 draws it too (27 fonts, 4.06% of Vita crops). Every VOCR6 model trained on the hooked lines has seen CLANNAD's glyph raster: disclosed with every CLANNAD number

