# VOCR6 final test (stage 4, read once)

Issue #125, stage 4's `final_test` (`eval/vocr6_protocol.json` on feat/vocr6-t1-stages). The blobs chosen to ship are
read once, after every VOCR6 choice, with VOCR5 beside them, on the protected sets: vita-test, psp-test, CLANNAD's
frozen references and the hooked test title PCSG00807 (Dance with Devils). The read reports; it chooses nothing.
Every number is a host measurement. Nothing ran on a console.

- **Protocol:** `eval/vocr6_final_test_protocol.json`, committed before this harness read anything (d02f0ae). Five
  amendments followed, each before any read of the path it touches:
  - 91cd103: robustness;
  - 51dd0a6: path (b)'s LM;
  - f289ea5: the check of path (b) against Rubi's own code;
  - 841f293: the CLANNAD glyph-raster disclosure;
  - 8948d52: audit-vocr6-final-test's fixes (one final read under any OUT name, Rubi's job defaults checked,
    the line filter's floor beside path (b)'s headline);
  - amendment 6: Rubi #335's vertical text (API 47, `ocr_vertical = auto` by default), modelled and checked
    against Rubi's own code, and the audit's optional nits. The hashes are those of the original commits (tag
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
- Rubi #306 merged during the wait (Rubi main `796168e`). Rubi main now ships the pinned LM itself, and its line-read
  functions are unchanged. Rubi #330 (the COLGAP fix in Rubi, API 44) is open. It changes `recognize` and
  `request_policy`, so a merge before the read stops it: rubicheck runs again with `--fix 1`, then an amendment.

**Not modelled.**

- Rubi's memory fallbacks: a game without the memory for the LM or the cascade runs without them.
- Trusted hooks that answer without OCR (Rubi API 43).
- The cascade's time ceiling.

On CLANNAD's pages, a reading on an empty box is never charged: the 133 pages without a reference line and the 207 with
an incomplete reference are left out.

**The line filter's floor.** Some reference lines hold no hiragana, katakana or CJK character. Rubi's non-Japanese line
filter deletes them however well they are read. The report prints this floor beside every path (b) measure, computed
from the references alone. It is the same for VOCR5 and VOCR6. On CLANNAD it is 26 of 33,485 dialogue characters; on
PCSG00807's clean lines 26 of 13,911 (audit-vocr6-final-test's count).

**Against Rubi's whole job** (`vocr6_final_test.py pagecheck`, `eval/vocr6_final_test_pagecheck_20261026.json`).
Rubi main `b6c715c` added vertical text (#335), which changed `recognize`. Its host tool `vjo-page-ocr` runs the game
worker's job on the defaults of `config.ini`, with `--vertical auto`, through textnorm and the line filter. It read
development regions of the train titles: horizontal ones, counter ones, and regions drawn around the columns of the
vertical title PCSG00653. Path (b)'s model was compared with it at every tier on four things: the switch to columns,
the boxes, the line texts, and the text shown.

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
- CLANNAD's glyph raster: two hooked train titles, PCSG00620 Grisaia Spinout and PCSG00007 Little Busters!, draw
  dialogue in a font whose 32 px glyphs are CLANNAD's own raster re-quantised (39,447 glyphs of 1,201 characters), and
  synth6 v3 draws it too. So every VOCR6 model has seen CLANNAD's glyph raster (vocr6-lead's stage-4 seen flag,
  vocr6-synth6's finding).
- ctclm.2's Jiten word list includes CLANNAD's script and those of 7 vita-test and 1 psp-test games.
- PCSG00807 (v18432) is a VOCR5 development-sample VN. No VOCR6 run trained on it, and no LM holds its text.

## Reproduce

```sh
PATHS=a eval/reproducibility/vocr6-final/report.sh - $A/vocr6-final-test/validation-<date>  # VOCR5, path (a)
.venv-ci/bin/python eval/vocr6_final_test.py stage --s S.vocr --n N.vocr --l L.vocr --out MODELS_DIR
eval/reproducibility/vocr6-final/report.sh MODELS_DIR $A/vocr6-final-test/final-<date>           # once
```

Run each through `$HERD/heavy.sh` in one slot. A final read refuses to start while another final attempt record exists.
`RESUME=1` continues a stopped read; report.sh's header says what each step needs first. Readings, plans and per-unit rows stay in
`$HERD/shared/private/vocr6-final-test/`. The repository holds counts, rates, intervals and hashes only.
