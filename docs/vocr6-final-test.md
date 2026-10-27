# VOCR6 final test (stage 4, read once)

Issue #125, stage 4's `final_test` (`eval/vocr6_protocol.json` on feat/vocr6-t1-stages). The blobs chosen to ship are
read once, after every VOCR6 choice, with VOCR5 beside them, on the protected sets: vita-test, psp-test, CLANNAD's
frozen references and the hooked test title PCSG00807 (Dance with Devils). The read reports; it chooses nothing.
Every number is a host measurement. Nothing ran on a console.

- **Protocol:** `eval/vocr6_final_test_protocol.json`, committed before this harness read anything (d02f0ae). Eight
  amendments followed, each before any read of the path it touches:
  - 91cd103: robustness;
  - 51dd0a6: path (b)'s LM;
  - f289ea5: the check of path (b) against Rubi's own code;
  - 841f293: the CLANNAD glyph-raster disclosure;
  - 8948d52: audit-vocr6-final-test's fixes (one final read under any OUT name, Rubi's job defaults checked,
    the line filter's floor beside path (b)'s headline);
  - 7462538: Rubi #335's vertical text (API 47, `ocr_vertical = auto` by default), modelled and checked
    against Rubi's own code, and the audit's optional nits;
  - 5c9c23c: Rubi #330 (API 50), which ships #129's COLGAP fix by default: re-pinned and re-checked; Rubi's
    pinned runtime files must equal this checkout's;
  - 05a23e1: the audit's optional nits N11-N16.

  The hashes of the first four are those of the original commits (tag `vocr6-final-test-history-20261026a`); the
  branch was rebased onto main after them.
- **Driver:** `eval/reproducibility/vocr6-final/report.sh MODELS_DIR OUT`: the S/N/L blobs in, every table out.
  `MODELS_DIR` is written by `eval/vocr6_final_test.py stage` from vocr6-lead's `FINAL READ` message.

## Two paths

**(a) VOCR5's locked path, as published.** The tools that produced VOCR5's locked report, unchanged:

- vita-test and psp-test fixed lines: `eval/vocr3_report.py shared` (the scalar C W8I runtime, greedy, console and
  native polarity), then `eval/vocr5_dialogue.py tables`. The same lines with beam 8 + ctclm.2's 1 MB LM through
  `runtime/build/vocr_cli`.
- Whole dialogue: `vocr3_report`'s regions at base, +4 and +8 px through the C GRAD finder.
- CLANNAD: `eval/clannad_refs.py`'s extra-arms path, with the dialogue pairs from `eval/stride16_clannad_dialogue.py`.
- PCSG00807: `eval/vocr6_baseline.py` (docs/vocr6-data.md's baseline: clean crops, GRAD + refine, textspot3).

**(b) Rubi's shipped path at the final read.** Each line is read the way Rubi's `core/vocr_ocr.c` reads it:

- Rubi's line path: TEXTBOX_RECOVERY, the short-line probe and the COLGAP fix (#129). The refinement runs on PSP titles
  only. PSP frames are upscaled 2x, the view Rubi has on a Vita.
- The recognizer: W8I with the blob's own head gate and two-level head, the counter re-decode, and the ungated re-read
  where Rubi does it.
- Decoding: beam 8 with the shipped 1 MB game-text LM (v0.3.7-lm6.1: alpha 600, beta 2500, BOS).
- The cascade S -> N -> L under 700, then Rubi's `core/textnorm.c` and the non-Japanese line filter.
- Rubi's `ocr_vertical = auto` (API 47, Rubi #335). Before a region's rows are read, Rubi checks the finder's rows.
  A region they call vertical text is read as columns by the vertical model VOCR5-V, for both families (VOCR6 has no
  vertical model). The report counts such regions per set.
- `vocr6_final_test.py rubi-path` resolves the Rubi side from a checkout of Rubi main taken at the read. It stops if
  Rubi's defaults leave the protocol.

Each set is read in four ways:

- the fixed lines;
- the whole-dialogue regions;
- CLANNAD's line crops and its pages with a complete reference (the ROI through the finder);
- PCSG00807's clean lines and its page units, also cut by their reference boxes to decompose the error.

Both paths report dialogue CER first and names as a secondary column. Intervals are 95% bootstraps: over games (VN
clusters) for vita-test and psp-test, over pages for CLANNAD and PCSG00807. VOCR6 − VOCR5 is paired over the same units.

## Validation on VOCR5 (step 2, 2026-10-26 19:03-19:23Z)

`PATHS=a report.sh - OUT` read VOCR5 alone on path (a). `vocr6_final_test.py validate` compared the outputs with the
published records: **29 of 29 checks equal** (`eval/vocr6_final_test_validation_20261026.json`):

| set | VOCR5-S / N / L, dialogue errors | published |
|---|---|---|
| vita-test + psp-test fixed lines, greedy (4,693 chars) | 75 / 53 / 37 (1.60 / 1.13 / 0.79%) | the same, `eval/vocr5_report_20261016.json` |
| the same, beam + 1 MB LM | 54 / 34 / 25 (1.15 / 0.72 / 0.53%) | the same, `eval/vocr5_dialogue_20261016.json` |
| whole dialogue, base (4,635 chars) | 385 / 370 / 328 (8.31 / 7.98 / 7.08%) | the same; +4 and +8 px equal too |
| CLANNAD, 2,052 dialogue lines (33,485 chars) | 463 / 322 / 246 (1.38 / 0.96 / 0.73%) | the same, `eval/vocr5_clannad_20261016.json`; Lens equal |
| PCSG00807, every tier, decode and measure | e.g. Normal rubi_cascade clean 17 / 13,911, GRAD + refine 238 / 13,941 | the same, `eval/vocr6_baseline_20261026.json` |

What else matched:

- Native polarity, every style row and every per-VN tally.
- The blobs' bytes and arenas.
- The intervals.

The finder driver of the PCSG00807 path is identical to the baseline's. Its scalar library is not, because #130 changed
`runtime/vocr.c` after the baseline ran. That library is only hashed, not used by any read, and the scores are equal;
the file records both hashes.

## Path (b) checked on development data

The shipped path was checked on development data only: the hooked hval block of the train titles
(`eval/vocr6_final_test_selftest_20261026.json`).

- **The finder:** its build matches vocr6-lead's hval plan v2 box for box and dark flag for dark flag (6,327 of 6,327
  units, fix 5, probe 1).
- **The line read with VOCR5's plain blobs:** 3,000 sampled crops against vocr6-lead's `eval/vocr6_score.py` dumps.
  Greedy text, beam text and confidence are equal for N and L, and the shipped text after N's cascade to L, textnorm and
  the filter is equal for all 3,000.
- **The line read with the released gated blobs:** 3,000 crops per tier against `eval/vocr6_baseline.py`'s reads.
  Greedy text, the shown text and confidence are equal for S, N and L. None of those sampled crops was read again
  without the gate; rubicheck below covers the re-read.

**Against Rubi's own code** (`vocr6_final_test.py rubicheck`, `eval/vocr6_final_test_rubicheck_20261026.json`).

- Rubi's host tool `vjo-ocr` (host/ocr_tool.c: `vjo_vocr_ocr_policy`, the game worker's job) was built from Rubi main
  `cbaa4e3`. It read 331 development regions of the hooked train titles: 6 hval regions a title, plus 55 regions whose
  verified text holds a counter, the cases of the re-decode and the re-read. It used the models and LM that checkout
  pins (VOCR5's gated blobs, ctclm.2's 1 MB LM) and its finder (fix 0).
- Path (b)'s Python model gives the same result at every tier: every box and dark flag, every line's text and greedy
  text, and every job's text are equal (S / N / L: 543 lines each). The equal lines include 117 / 102 lines the cascade
  read again and 3 / 9 / 6 re-reads without the head gate.
- `rubi-path` pins the sha256 of the Rubi functions the model stands for (`paths.b.rubi_functions`: `finish_line`,
  `read_again`, `read_full`, `recognize`, `request_policy`, the probe, the line filter and more). It also checks Rubi's
  job defaults (`paths.b.job_defaults`): the finder mode, text gate off, blank lines kept, the line filter on, the
  cascade, and the line-fix default. The final read stops if Rubi main differs; a change needs review, rubicheck again
  and an amendment.
- Rubi main now ships both the pinned LM (#306) and the COLGAP fix (#330), so path (b) is Rubi main's shipped line
  path. A further change of a pinned function or default stops the read for review, a recheck and an amendment.

**Not modelled.**

- Rubi's memory fallbacks: a game without the memory for the LM or the cascade runs without them.
- Trusted hooks that answer without OCR (Rubi API 43).
- The cascade's time ceiling.

On CLANNAD's pages, a reading on an empty box is never charged: the 133 pages without a reference line and the 207 with
an incomplete reference are left out.

**The line filter's floor.** Some reference lines hold no hiragana, katakana or CJK character. Rubi's non-Japanese line
filter deletes them when they are read right. The report prints this floor beside every path (b) measure, computed
from the references alone. It is the same for VOCR5 and VOCR6. On CLANNAD it is 26 of 33,485 dialogue characters; on
PCSG00807's clean lines 26 of 13,911 (audit-vocr6-final-test's count).

**Against Rubi's whole job** (`vocr6_final_test.py pagecheck`, `eval/vocr6_final_test_pagecheck_20261027.json`).
Rubi main moved twice during the wait:

- `b6c715c` added vertical text (#335), which changed `recognize`;
- `0994afc` (#330, API 50) ships #129's COLGAP fix by default (`ocr_line_fix = on`), with a tall region's auto check
  made on the frozen finder's rows.

Rubi's host tool `vjo-page-ocr` runs the game worker's job on the defaults of `config.ini`, with `--vertical auto`
and `--line-fix on`, through textnorm and the line filter. Built from `0994afc`, it read development regions of the
train titles: horizontal ones, counter ones, regions drawn around the columns of the vertical title PCSG00653, and
those regions widened toward square, which Rubi checks on the frozen finder's rows. Path (b)'s model was compared with
it at every tier on four things: the switch to columns, the boxes, the line texts, and the text shown.

All 379 regions are equal at S, N and L, 943 lines per tier:

- 184 horizontal and 55 with a counter, read as rows;
- 60 vertical, read as columns by both;
- 80 widened tall ones, 54 of them read as columns by both.

The first 299 regions were equal against `b6c715c` at fix 0 too. Every runtime file Rubi pins (18) equals this
checkout's, so path (b) reads with Rubi's runtime code.

`vocr6_final_test.py dryrun` then runs the final read's whole path (b) and its report on stand-ins of the four sets,
built from the same development data.

## Results (the final read, 2026-10-27 10:05-11:00Z)

vocr6-lead's `FINAL READ` named the blobs that ship:

| tier | blob | sha256 | what it is |
|---|---|---|---|
| S | `VOCR5_S_w8g.vocr` | `76a1bc5d…` | VOCR5-S with its head gate: stage 4 did not adopt VOCR6 Small |
| N | `VOCR6_N_w8_2l.vocr` | `929652e3…` | VOCR6 Normal with the two-level head, no gate |
| L | `VOCR6_L_w8_2l.vocr` | `b051fd4f…` | VOCR6 Large with the two-level head, no gate |

How the read ran:

- One attempt, at repo `4cb9d46`, with VOCR5 beside the blobs on both paths.
- Path (b) is Rubi main `892cfb9`'s shipped line path: lm6 1000k at 600 / 2500 / BOS, COLGAP, `ocr_vertical = auto`,
  the cascade S → N → L, textnorm and the line filter. `892cfb9` passed every rubi-path pin.
- VOCR5's path (a) again equals its published records, 29 of 29 checks.
- No read errors and no finder errors.

Path (a) reads the shipped blobs as given, with their heads off in vocr3_report, vocr_cli and clannad_refs. vocr6-lead
also named the plain blobs (`v6f4h_N_seed1_w8.vocr`, `v6f4h_L_seed1_w8.vocr`, and v0.3.0's `VOCR5_S_w8.vocr`). On
development data the two read the same: `eval/vocr6_final_test.py plaincheck` on 3,000 crops of vocr6-lead's hval
plan v1 (153,949 frames) finds every tier equal on every crop: the greedy text, conf_min, the top-8 and blank
logits, and the top-8 class texts in order (`eval/vocr6_final_test_plaincheck_20261027.json`). That is expected: the
two-level blob only permutes the head's rows, and the gated one only adds a section.

Records:

- `eval/vocr6_final_test_20261027.json`: every measure, with counts, rates and intervals.
- `docs/vocr6-final-test-report-20261027.md`: the full tables (per title, Vita and PSP apart, names, path (b)'s
  stages). Its blob table's "W8I arena" column is the arena with the heads off, as path (a) reads the blobs: 124,832 /
  182,944 / 261,024 B for S / N / L. As Rubi runs the shipped blobs, with their heads on, they need 124,880 / 184,400 /
  262,480 B (`eval/vocr6_gate_final_20261027.json`, `estimate.final.arena_w8i_1024`; audit N18).

Dialogue CER, VOCR5 → VOCR6, and VOCR6 − VOCR5 in points [paired 95% interval]:

| set | S | N | L |
|---|---|---|---|
| **path (b)**: CLANNAD, 2,052 lines | 0.73 → 0.68% (−0.05 [−0.12, +0.02]) | 0.66 → 0.44% (**−0.22** [−0.30, −0.15]) | 0.60 → 0.36% (**−0.24** [−0.31, −0.17]) |
| CLANNAD, 1,837 pages (finder) | 5.84 → 6.16% (**+0.32** [+0.14, +0.50]) | 5.94 → 5.47% (**−0.46** [−0.63, −0.30]) | 6.19 → 5.40% (**−0.78** [−0.97, −0.60]) |
| vita + psp-test fixed lines, 336 | 0.68 → 0.81% (+0.13 [−0.09, +0.41]) | 0.49 → 0.49% (+0.00 [−0.20, +0.18]) | 0.47 → 0.43% (−0.04 [−0.19, +0.10]) |
| vita + psp-test whole dialogue, 175 regions | 3.04 → 3.26% (+0.22 [−0.11, +0.61]) | 2.83 → 2.63% (−0.19 [−0.51, +0.11]) | 2.72 → 2.52% (−0.19 [−0.55, +0.15]) |
| PCSG00807 clean lines, 1,152 | 0.27 → 0.24% (−0.03 [−0.07, +0.01]) | 0.26 → 0.21% (**−0.05** [−0.10, −0.01]) | 0.22 → 0.21% (−0.01 [−0.06, +0.02]) |
| PCSG00807 page path, 685 regions | 0.85 → 0.84% (−0.01 [−0.05, +0.02]) | 0.84 → 0.80% (−0.04 [−0.08, +0.01]) | 0.81 → 0.80% (−0.01 [−0.04, +0.02]) |
| **path (a)**: CLANNAD lines, greedy | 1.38 → 1.38% | 0.96 → 0.50% (**−0.46** [−0.57, −0.35]) | 0.73 → 0.31% (**−0.42** [−0.51, −0.34]) |
| vita + psp-test fixed lines, greedy | 1.60 → 1.60% | 1.13 → 0.83% (−0.30 [−0.73, +0.02]) | 0.79 → 0.53% (**−0.26** [−0.49, −0.07]) |
| the same, beam + ctclm.2's 1 MB LM | 1.15 → 1.15% | 0.72 → 0.70% (−0.02 [−0.28, +0.21]) | 0.53 → 0.49% (−0.04 [−0.20, +0.11]) |
| vita + psp-test whole dialogue (GRAD) | 8.31 → 8.31% | 7.98 → 7.21% (**−0.78** [−1.15, −0.38]) | 7.08 → 6.86% (−0.22 [−0.52, +0.11]) |
| PCSG00807 clean lines (Rubi cascade) | 0.14 → 0.17% (+0.02 [−0.03, +0.07]) | 0.12 → 0.08% (−0.04 [−0.11, +0.01]) | 0.07 → 0.06% (−0.01 [−0.07, +0.05]) |
| PCSG00807 page GRAD + refine | 1.66 → 1.65% (−0.01 [−0.07, +0.04]) | 1.71 → 1.68% (−0.03 [−0.10, +0.04]) | 1.67 → 1.66% (−0.01 [−0.07, +0.05]) |

Bold marks an interval that excludes 0. The intervals are a game bootstrap for vita-test and psp-test and a page
bootstrap for CLANNAD and PCSG00807.

What the read shows:

- **Normal and Large on CLANNAD.** VOCR6 is resolved better on both paths. On Rubi's path, line errors fall by a third
  for Normal (0.66 → 0.44%) and by two fifths for Large (0.60 → 0.36%), and page errors fall too. Every VOCR6 model has seen CLANNAD's glyph raster
  (the seen flags below), so this is not a measure on an unseen font.
- **Normal and Large on vita-test and psp-test.** On Rubi's path they are level with VOCR5: every interval includes 0.
  Greedy (path (a)), Large's fixed lines and Normal's whole dialogue are resolved better. With an LM most of that gain
  goes, on both paths.
- **Vita and PSP apart** (path (b), fixed lines):
  - vita-test: N −0.21 [−0.56, +0.05], L −0.17 [−0.40, +0.00];
  - psp-test: N +0.22 [+0.00, +0.41], L +0.09 [−0.12, +0.30].

  On path (a) with the LM, psp-test's Normal is resolved worse (+0.30 [+0.12, +0.53]).
- **PCSG00807.** Every tier is within about 0.05 points of VOCR5 (at most 0.05 at two decimals). On path (b), Normal's
  clean lines are resolved better (−0.05 [−0.10, −0.01]).
- **Small.** The shipped Small is VOCR5-S, so path (a) reads the same weights in both columns, apart from PCSG00807's
  Rubi-cascade decode. On path (b) its column differs only through the cascade, which now goes to VOCR6 Normal. That
  makes CLANNAD's pages worse (+0.32 [+0.14, +0.50]); most of the difference arises after the cascade, in textnorm and
  the line filter (the report's stage table). Elsewhere, Small's dialogue is within the noise. Its speaker names, a
  secondary column, are another matter (the report's Names table). On path (b), CLANNAD's names are resolved worse:
  6.02 → 9.78%, +3.76 [+2.54, +5.07]. vita-test + psp-test's fixed name lines are +1.05 [+0.00, +2.38], an interval
  whose lower bound is 0.

## The follow-up's read (v0.4.1-vocr6.2, 2026-10-27 20:03-20:50Z)

The follow-up (vocr6-lead's stages 4c and 4d, `docs/vocr6.md`) is a new finalist set, so its protected sets are read
once for it. **Amendment 9** (committed before the read, after an independent pre-read review) made that possible:
- **One read per registered finalist set.** The protocol's `finalist_sets` lists v0.4.0-vocr6.1 (read above) and
  v0.4.1-vocr6.2. report.sh refuses a set that is not registered and a set read before, under any OUT name.
- **The earlier read's path.** The follow-up reads path (b) at the earlier read's Rubi head, `892cfb9`. Rubi main's
  later changes to the pinned functions (#373, #358, #379) were reviewed, not re-pinned: at Rubi's defaults they
  leave the line read alone. Rubi main's spotter default (#396) is not modelled.
- **The runtime.** Rubi pins vita-vn-ocr `f8a1856`'s `runtime/vocr.c` and `vocr.h`, while this checkout holds #138's,
  which adds W8F paths. `vocr6_final_test.py runtimecheck` read 4,000 development crops (vocr6-lead's hval and dev
  plans) through path (b)'s line read with both builds, for all 7 blobs path (b) reads, and found every crop equal
  (`eval/vocr6_final_test_runtimecheck_20261027.json`). rubi-path accepts exactly those two files' pairs, with that
  record.
- **The comparison.** `vocr6_final_test.py compare` pairs this read with the earlier one unit by unit. VOCR5 read
  every unit of every headline measure exactly as before (0 units differ), so the two reads share one path.

| tier | blob | sha256 | what it is |
|---|---|---|---|
| S | `VOCR6.2_S_w8.vocr` (gate driver: `VOCR6_S_w8.vocr`) | `864bbd19…` | VOCR6 Small (stage 4d), plain: the gate rule kept no gate |
| N | `VOCR6.2_N_w8g2.vocr` (`VOCR6_N_w8g2.vocr`) | `107cf084…` | VOCR6 Normal (stage 4c), head gate at ε = 1e-4 + two-level head |
| L | `VOCR6.2_L_w8_2l.vocr` (`VOCR6_L_w8_2l.vocr`) | `65310013…` | VOCR6 Large (stage 4c), two-level head |

One attempt, at repo `27e0126`, with VOCR5 beside the blobs on both paths. VOCR5's path (a) equals its published
records, 29 of 29 checks. There were no read errors and no finder errors.

Records:
- `eval/vocr6_final_test_followup_20261027.json`: every measure;
- `eval/vocr6_final_test_followup_compare_20261027.json`: the pairing with the earlier read;
- the full tables: `docs/vocr6-final-test-report-followup-20261027.md` and `docs/vocr6-final-test-compare-followup-20261027.md`.

Path (b), dialogue CER, VOCR5 → v0.4.1, with v0.4.1 − VOCR5 and v0.4.1 − v0.4.0 in points [paired 95% interval]:

| set | S | N | L |
|---|---|---|---|
| CLANNAD, 2,052 lines | 0.73 → 0.47% (**−0.26** [−0.35, −0.18]; **−0.21** [−0.30, −0.12]) | 0.66 → 0.33% (**−0.33** [−0.41, −0.24]; **−0.10** [−0.17, −0.04]) | 0.60 → 0.31% (**−0.29** [−0.37, −0.21]; −0.05 [−0.10, +0.01]) |
| CLANNAD, 1,837 pages (finder) | 5.84 → 5.75% (−0.09 [−0.28, +0.10]; **−0.41** [−0.61, −0.21]) | 5.94 → 5.81% (−0.13 [−0.31, +0.05]; **+0.33** [+0.19, +0.48]) | 6.19 → 5.67% (**−0.52** [−0.71, −0.32]; **+0.27** [+0.13, +0.41]) |
| vita + psp-test fixed lines, 336 | 0.68 → 0.58% (−0.11 [−0.33, +0.11]; **−0.23** [−0.46, −0.04]) | 0.49 → 0.36% (−0.13 [−0.33, +0.04]; −0.13 [−0.26, +0.00]) | 0.47 → 0.40% (−0.06 [−0.18, +0.04]; −0.02 [−0.14, +0.11]) |
| vita + psp-test whole dialogue, 175 regions | 3.04 → 3.02% (−0.02 [−0.45, +0.45]; −0.24 [−0.58, +0.06]) | 2.83 → 2.70% (−0.13 [−0.55, +0.27]; +0.06 [−0.33, +0.48]) | 2.72 → 2.68% (−0.04 [−0.47, +0.39]; +0.15 [−0.17, +0.52]) |
| PCSG00807 clean lines, 1,152 | 0.27 → 0.21% (**−0.06** [−0.11, −0.01]; −0.03 [−0.06, +0.00]) | 0.26 → 0.19% (**−0.06** [−0.12, −0.02]; −0.01 [−0.04, +0.00]) | 0.22 → 0.19% (**−0.04** [−0.08, −0.01]; −0.02 [−0.05, +0.00]) |
| PCSG00807 page path, 685 regions | 0.85 → 0.85% (+0.00 [−0.06, +0.07]; +0.01 [−0.04, +0.07]) | 0.84 → 0.80% (−0.04 [−0.08, +0.01]; +0.00 [−0.03, +0.03]) | 0.81 → 0.79% (−0.02 [−0.05, +0.00]; −0.01 [−0.04, +0.00]) |

What the read shows:
- **Small (now VOCR6)** is better than VOCR5-S or level everywhere. v0.4.0's Small lost on CLANNAD's pages through its
  cascade (+0.32); the follow-up's Small does not (−0.09).
- **Normal and Large** read CLANNAD's lines better than v0.4.0's VOCR6 but its pages worse (+0.33 / +0.27, resolved),
  though still better than VOCR5 or level on the pages. The stage table places the page loss before the cascade: Large's
  greedy pages are level with v0.4.0's (6.07 / 6.10%), and its counter + LM stage gains less (6.03 / 5.78%).
- **vita-test + psp-test** are level against VOCR5 on path (b) at every tier. vita-test is better (Small −0.25, Normal
  −0.33, resolved); psp-test is not. On path (a) with ctclm.2's LM, psp-test's Normal is resolved worse: +0.26 [+0.04,
  +0.51].
- **Names** (secondary): path (b)'s CLANNAD names improve for Normal and Large (−2.94, −5.12) and are level for Small
  (+0.45 [−0.64, +1.52]). On path (a), greedy, CLANNAD's Normal names are resolved worse: +2.35 [+0.27, +4.49].

## Seen flags (every table carries them)

- Every VOCR5 and VOCR6 model trained on the owner's console captures, which include planetarian's PSP port.
- 38 of the 49 hooked titles' VNs were in VOCR5's fine-tune pools.
- VOCR6 trained on the hooked train titles, and their text is in the lm6 LM. Three of them relate to protected sets:
  - Collar x Malice (PCSG00866) shares releases with vita-test's Collar x Malice -Unlimited- (v21789);
  - Little Busters! (PCSG00007) and planetarian (ULJM05383) share the Key 10th Memorial BOX with CLANNAD.
- CLANNAD's glyph raster: two hooked train titles, PCSG00620 Grisaia Spinout and PCSG00007 Little Busters!, draw
  dialogue in a font whose 32 px glyphs are CLANNAD's own raster re-quantised (39,447 glyphs of 1,201 characters), and
  synth6 v3 draws it too. So every VOCR6 model has seen CLANNAD's glyph raster (vocr6-lead's stage-4 seen flag,
  vocr6-synth6's finding).
- ctclm.2's Jiten word list includes CLANNAD's script and those of 7 vita-test and 1 psp-test games. lm6, which path
  (b) uses for both families, keeps the same Jiten word lists, so the caveat holds for path (b) too.
- PCSG00807 (v18432) is a VOCR5 development-sample VN. No VOCR6 run trained on it, and no LM holds its text.

## Reproduce

```sh
PATHS=a eval/reproducibility/vocr6-final/report.sh - $A/vocr6-final-test/validation-<date>  # VOCR5, path (a)
.venv-ci/bin/python eval/vocr6_final_test.py stage --s S.vocr --n N.vocr --l L.vocr --out MODELS_DIR
eval/reproducibility/vocr6-final/report.sh MODELS_DIR $A/vocr6-final-test/final-<date>           # once
# a later finalist set (amendment 9): registered in the protocol first, with its runtime check
.venv-ci/bin/python eval/vocr6_final_test.py runtimecheck --rubi RUBI_AT_ITS_HEAD --models MODELS_DIR --work W \
    --out eval/vocr6_final_test_runtimecheck_<date>.json
eval/reproducibility/vocr6-final/report.sh MODELS_DIR $A/<label>/final-<name>                   # once per set
```

Run each through `$HERD/heavy.sh` in one slot. A final read refuses a finalist set that the protocol does not register
or that has a final attempt record already (amendment 9; before it, any second final read). `RESUME=1` continues a
stopped read of its own set; report.sh's header says what each step needs first. Readings, plans and per-unit rows stay in
`$HERD/shared/private/vocr6-final-test/`. The repository holds counts, rates, intervals and hashes only.
