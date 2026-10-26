# VOCR6 final test (stage 4, read once)

Issue #125, stage 4's `final_test` (`eval/vocr6_protocol.json` on feat/vocr6-t1-stages). The blobs chosen to ship are
read once, after every VOCR6 choice, with VOCR5 beside them, on the protected sets: vita-test, psp-test, CLANNAD's
frozen references and the hooked test title PCSG00807 (Dance with Devils). The read reports; it chooses nothing.
Every number is a host measurement. Nothing ran on a console.

- **Protocol:** `eval/vocr6_final_test_protocol.json`, committed before this harness read anything (d02f0ae), with three
  amendments made before any read of the path they touch: robustness (91cd103), path (b)'s LM (51dd0a6), and the check
  of path (b) against Rubi's own code (f289ea5). The hashes are those of the original commits (tag
  `vocr6-final-test-history-20261026a`); the branch was rebased onto main after them.
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
`runtime/vocr.c` after the baseline ran; the scores are equal anyway, and the file records both hashes.

## Path (b) checked on development data

The shipped path was checked on development data only: the hooked hval block of the train titles
(`eval/vocr6_final_test_selftest_20261026.json`).

- **The finder:** its build matches vocr6-lead's hval plan v2 box for box and dark flag for dark flag (6,327 of 6,327
  units, fix 5, probe 1).
- **The line read with VOCR5's plain blobs:** 3,000 sampled crops against vocr6-lead's `eval/vocr6_score.py` dumps.
  Greedy text, beam text and confidence are equal for N and L, and the shipped text after N's cascade to L, textnorm and
  the filter is equal for all 3,000.
- **The line read with the released gated blobs:** 3,000 crops per tier against `eval/vocr6_baseline.py`'s reads.
  Greedy text, re-read text and confidence are equal for S, N and L.

**Against Rubi's own code** (`vocr6_final_test.py rubicheck`, `eval/vocr6_final_test_rubicheck_20261026.json`).

- Rubi's host tool `vjo-ocr` (host/ocr_tool.c: `vjo_vocr_ocr_policy`, the game worker's job) was built from Rubi main
  `cbaa4e3`. It read 331 development regions of the hooked train titles: 6 hval regions a title, plus 55 regions whose
  verified text holds a counter, the cases of the re-decode and the re-read. It used the models and LM that checkout
  pins (VOCR5's gated blobs, ctclm.2's 1 MB LM) and its finder (fix 0).
- Path (b)'s Python model gives the same result at every tier: every box and dark flag, every line's text and greedy
  text, and every job's text are equal (S / N / L: 543 lines each). The equal lines include 117 / 102 lines the cascade
  read again and 3 / 9 / 6 re-reads without the head gate.
- `rubi-path` pins the sha256 of the Rubi functions the model stands for (`paths.b.rubi_functions`: `finish_line`,
  `read_again`, `read_full`, the probe, the line filter and more). The final read stops if Rubi main's differ.

`vocr6_final_test.py dryrun` then runs the final read's whole path (b) and its report on stand-ins of the four sets,
built from the same development data.

## Results

Filled in by the final read: `eval/vocr6_final_test_<date>.json` and the tables below.

## Seen flags (every table carries them)

- Every VOCR5 and VOCR6 model trained on the owner's console captures, which include planetarian's PSP port.
- 38 of the 49 hooked titles' VNs were in VOCR5's fine-tune pools.
- VOCR6 trained on the hooked train titles, and their text is in the lm6 LM. Three of them relate to protected sets:
  - Collar x Malice (PCSG00866) shares releases with vita-test's Collar x Malice -Unlimited- (v21789);
  - Little Busters! (PCSG00007) and planetarian (ULJM05383) share the Key 10th Memorial BOX with CLANNAD.
- ctclm.2's Jiten word list includes CLANNAD's script and those of 7 vita-test and 1 psp-test games.
- PCSG00807 (v18432) is a VOCR5 development-sample VN. No VOCR6 run trained on it, and no LM holds its text.

## Reproduce

```sh
PATHS=a eval/reproducibility/vocr6-final/report.sh - $A/vocr6-final-test/validation-<date>  # VOCR5, path (a)
.venv-ci/bin/python eval/vocr6_final_test.py stage --s S.vocr --n N.vocr --l L.vocr --out MODELS_DIR
eval/reproducibility/vocr6-final/report.sh MODELS_DIR $A/vocr6-final-test/final-<date>           # once
```

Run each through `$HERD/heavy.sh` in one slot. Readings, plans and per-unit rows stay in
`$HERD/shared/private/vocr6-final-test/`. The repository holds counts, rates, intervals and hashes only.
