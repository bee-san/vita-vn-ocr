# finder-tighten: refining the console finder's line boxes (`runtime/vocr_lines_refine.c`)

The console reads a dialogue line from the box the GRAD finder (`runtime/vocr_lines.c`) returns. On the 443
devnative lines that have both crops, the same VOCR3-W20 reads the GRAD crop at **3.10% CER** and the hand-cut
ink crop at **0.51%** (`docs/model-tiers.md` §2.2). This change adds a post-finder refinement that cuts each GRAD
box back towards its text. It needs no retraining and works with every released recognizer. It is a separate
call behind a flag: flags 0 return the finder's lines bit for bit, so the overlay can A/B it.

Every number below is a host evaluation through the plain-C finder, the C refinement and the C runtime in W8I,
or a host estimate of console time. Nothing ran on a console.

**Results.**

- **Development (selection data), dialogue lines of the 443 pairs** (dn pairs): GRAD → refined closes **38-69%
  of the gap** to the ink crop across the six models (mean 56%); VOCR3-W20 1.53% → 0.71% (ink 0.35%). With names
  included, the gap the model-tiers table quotes (3.10% → 0.51%) closes by 18-36% (VOCR3-W20 → 2.22%): speaker
  names stay at 23-31% CER, and names are a secondary column by the owner's rule of 13:57Z.
- **Development whole dialogue** (teacher_val, devnative and the owner's frames as the dialogue box a player
  draws, at the base region and grown by 4 and 8 px): **−1.44 to −1.67 points** pooled over the variants
  (VOCR3-W20 5.40% → 3.94%), every VN-cluster 95% interval below zero, every model and cohort better.
- **The protected sets, once, report only** (after the rule was frozen and pushed, `1747eb4`):
  - vita-test + psp-test whole dialogue, dialogue (body) text: **−1.13 to −1.28 points** (VN intervals include
    0). The gain is all PSP (−2.19 to −2.90); Vita is level or worse (−0.13 to +0.50, intervals include 0).
  - **CLANNAD long run: worse for five of six models**, +0.07 to +0.43 points (VOCR4-W20 1.81% → 2.24%, page
    interval +0.31 to +0.56); VOCR3-W20 −0.06 (interval includes 0).
- **Cost**: about 0.5-1.1 ms per finder line on one 444 MHz Cortex-A9 core (estimate), under 1% of recognizing
  a 640-px line; no new memory (its arena fits inside the finder's); one extra read of each finder box.

So the refinement is a clear gain on PSP-sized text and on the scaled and native development frames, and **not
a gain on 960×544 Vita text in the protected sets**. The default the overlay ships should follow that (below).

## 1. What the GRAD crop gets wrong

On the 443 pairs (devnative v3, the VOCR3/VOCR4 select-pool copies) the GRAD box is taller and wider than the
ink crop: median height 23 px against 18 for dialogue lines, and it reaches past the ink by up to 22 rows and
181 columns. Of the 136 errors it adds over the ink crop (VOCR3-W20), 58 are on the 333 dialogue lines and 78 on
the 110 name lines. Swapping one axis at a time (VOCR3-W20, dialogue lines; scratch experiment):

| crop | errors / 4,912 chars |
|---|---|
| ink box | 17 |
| ink box with GRAD's polarity flag | 18 |
| GRAD's columns, ink rows | 30 |
| ink columns, GRAD's rows | 71 |
| GRAD box | 75 |

**The extra rows carry nearly all of it.** Reading the worst lines on contact sheets (two sheets, development
only) shows what the rows hold: the dialogue box's border and its inner rule, the edge of the next or previous
line, furigana, scenery behind a translucent box, and two text rows that GRAD merged into one box. The columns
add a few errors through a portrait or an ornament at the line's start and an icon or "next page" cursor at its
end.

## 2. The method

`vocr_refine_lines(im, flags, arena, bytes, &set, &rows)` (and `vocr_refine_lines_rows` over the finder's row
callback) refines the finder's `VocrLineSet` in place. For each box, using only the box's own pixels:

1. **Statistics from the middle rows.** Background = median grey of rows [h/4, h − h/4); `d = max(16,
   (p98 − p2)/4)`, `vocr_line_dark`'s rule on the rows the text fills, so a portrait or scenery at the box's top
   and bottom cannot become the background.
2. **Polarity of the ink mask.** The line's dark flag, unless the middle rows hold over 4× as many pixels
   deviating the other way. A pixel is ink when it deviates from the background by more than `d` that way.
3. **FRAMES**: a row whose longest run of ink covers half the box width or more is a frame line (border, rule,
   stripe) and is cleared.
4. **Bands**: runs of rows holding at least 2 ink pixels in at least 2 runs (a text row crosses several strokes;
   a lone run is a border or the edge of a shape). The main band is the heaviest.
5. **SPLIT**: every other band at least 80% as tall as the main band, at least 6 rows, a quarter of its ink, and
   whose glyphs (the upper median of its column clusters' heights) are at least 70% as tall as the main band's
   becomes a line of its own. Furigana and stray marks stay with their line.
6. **TIGHTEN**: each line's rows become its band plus 16% of the band's height above and below, never past the
   midpoint to the next band.
7. **CURSOR**: a trailing cluster beyond a gap of 1.5 band heights, at most 1.5 band heights wide, that is a speck
   (a quarter of the band height or less) or a solid blob (60% filled, at least 0.6× as wide as tall) is cut off.
8. **PORTRAIT**: the part at either end beyond a gap of one band height is cut off when its ink runs on
   vertically for 1.9 band heights, at least half a band height both above and below the band (an object
   crossing the line, not scenery above text), or when it is 1.5 band heights wide and 60% filled.
   Cuts keep half a band height of margin and never pass the middle of the gap.

A box without a text band is kept as it is; boxes never grow, keep their dark flag and stay in order; split
lines take their box's place (only while the set stays within 16 lines). No line is dropped, so the overlay's
own blank-line rejection does not overlap with it. `VOCR_REFINE_DEFAULT` is all five steps.

**Bit-exact twin**: `synth/lines_refine.py` (`refine_lines(img, boxes, dark, flags)`, which also returns each
line's finder line). `tests/lines_refine_parity.py` runs the C driver against it on procedural regions (both
the in-memory and the row-callback forms) and the development harness checks it on every development region it
scores (4,690 regions, both flag sets): no difference.

### API, memory and the overlay

```c
size_t need = vocr_lines_arena_bytes(w, h);                 /* the finder's arena serves both */
vocr_find_lines_rows(&src, VOCR_LINES_GRAD, arena, need, &set);
vocr_refine_lines_rows(&src, VOCR_REFINE_DEFAULT, arena, need, &set, &rows);   /* 0 = off: set unchanged */
```

- `vocr_refine_arena_bytes(w, h)` ≤ `vocr_lines_arena_bytes(w, h)` for every size (tested on a grid up to
  6000 px), so no arena grows. No heap, no recursion. Stack: 1,704 B in `refine` (arm-vita-eabi-gcc -O2,
  `-fstack-usage`), after the finder has returned.
- Code: 4,496 B of Thumb-2 text at -O2 (3,312 B at -Os); external symbols `memset`, `memcpy` only.
- **Rows read** (the overlay's strip cost): each finder box is read once, top to bottom, boxes in set order,
  when the arena also holds the box's grey pixels (mask + w·h bytes); otherwise twice (its middle half, then
  every row: 1.5× its height). With the overlay's job scratch (the larger of the finder's and the recognizer's
  arenas: 124,832 B for FL10) every development box was read once: 26.2 rows per finder box on average. With
  the finder's arena alone, 22% of them fit one pass (37.2 rows per box).
- Return codes: `VOCR_LINES_E_ARG` (bad image, unknown flags, a line outside the image),
  `VOCR_LINES_E_ARENA`, `VOCR_LINES_E_SOURCE` (a row could not be read). On any error the set is unchanged.
- Integer ranges: row counts are 16-bit and band masses 32-bit (`unsigned long` on the Vita), so the C equals
  the twin for boxes up to 65,535 px wide and about 42 million pixels (any console region); a larger box could
  wrap a comparison (never a memory access).

## 3. Development data

`eval/finder_tighten.py cohorts` builds five private cohorts (crops of copyrighted frames, kept under
`$HERD/shared/artifacts/finder-tighten/cohorts`; sha256 in `eval/finder_tighten_protocol.json`). Under the
owner's policy of 13:25Z everything outside CLANNAD, vita-test and psp-test is development or training data.

| cohort | what | size |
|---|---|---|
| dn_pairs | devnative v3 lines with a clean GRAD crop and an ink crop; the region devnative cut the GRAD crop from | 443 lines (333 dialogue), 43 VNs |
| own_pairs | the same for the owner pool v2 (the owner's PSP/Vita captures) | 539 lines (538 dialogue), 3 VNs |
| tv_dlg | teacher_val as whole dialogue | 601 regions, 865 lines, 168 VNs |
| dn_dlg | devnative v3 as whole dialogue | 215 regions, 388 lines, 43 VNs |
| own_dlg | the owner pool as whole dialogue | 420 regions, 629 lines, 3 VNs |

A whole-dialogue region is a frame's labelled non-name lines grouped with eval's rule (gap ≤ 1.6 line heights),
their union plus 12 px (Vita) / 6 px (PSP), at three sizes (+0, +4, +8 px, the VOCR3 harness's variants). Every
other detected text box (speaker names, unlabelled or rejected lines, UI) is kept out of the region, as a player
crops a name plate out; a group another box overlaps is dropped. The pooled whole-dialogue figure is tv_dlg +
own_dlg + the dn_dlg frames teacher_val does not hold (183 of its 645 cases): devnative's tv_native subset is
teacher_val's own native screenshots, never pooled twice.

The owner pool is VOCR4-W20's training data (arm M fine-tunes on it), so its own_pairs and own_dlg rows are
in-sample for that model; the comparison is paired (the same model on both crops).

### Selection

Over a hundred configurations of ten prototypes were compared on the five cohorts, most with VOCR3-W20 and
VOCR4-FL10 and about fifty with all six models (scratch scripts, not committed; per-unit results under
`$HERD/shared/artifacts/finder-tighten/scratch`; the protocol's one-line summary of this, "about 80 ... with all
six arms", is loose). The search was sequential: a step was kept when it lowered the six models' mean
pairs-dialogue CER without raising their mean whole-dialogue CER at any variant, or fixed a failure seen on a
contact sheet. The flags, measured the same way (mean over the six models; whole dialogue pooled as above):

| flags | dn pairs dialogue | whole dialogue base | +4 | +8 | all variants |
|---|---:|---:|---:|---:|---:|
| GRAD (refinement off) | 1.72 | 4.74 | 6.03 | 6.69 | 5.82 |
| TIGHTEN | 1.27 | 4.06 | 4.82 | 5.42 | 4.77 |
| TIGHTEN + FRAMES | 1.17 | 4.00 | 4.81 | 5.42 | 4.74 |
| TIGHTEN + SPLIT | 1.20 | 3.50 | 4.29 | 4.88 | 4.22 |
| TIGHTEN + FRAMES + SPLIT | 1.11 | 3.49 | 4.33 | 4.92 | 4.25 |
| + CURSOR | 1.03 | 3.49 | 4.33 | 4.92 | 4.25 |
| + PORTRAIT, one-sided test (not frozen) | 0.98 | 3.49 | 4.32 | 4.92 | 4.24 |
| **all five, two-sided portrait test (frozen default)** | **0.99** | **3.49** | **4.32** | **4.90** | **4.24** |
| (ink crop, pairs only) | 0.45 | | | | |

TIGHTEN carries most of the gain, SPLIT most of the rest on whole dialogue, CURSOR and PORTRAIT help the pairs.
The one-sided portrait test cut a real text group whose columns met scenery above the box (a devnative frame
at +8 px); requiring ink both above and below the band fixed it, for 0.01 points on the pairs. The frozen row
equals the C-path report below exactly.

## 4. Development results (C path, W8I; `eval/finder_tighten_dev_20261014.json`)

Δ = refined − GRAD in points; 95% intervals from 10,000 paired bootstraps over VN clusters (eval/vocr3_report's).

| model | dn pairs dialogue: GRAD → refined (ink) | Δ (VN CI) | gap closed | whole dialogue, all variants | Δ (VN CI) | names (dn pairs) |
|---|---|---|---:|---|---|---|
| VOCR3-FL10 | 1.71 → 1.26 (0.53) | −0.45 (−1.04 to +0.02) | 38% | 6.24 → 4.58 | −1.66 (−3.74 to −0.63) | 33.1 → 31.1 |
| VOCR3-H15 | 1.93 → 1.16 (0.41) | −0.77 (−1.48 to −0.25) | 51% | 5.84 → 4.22 | −1.62 (−3.57 to −0.64) | 27.4 → 26.3 |
| VOCR3-W20 | 1.53 → 0.71 (0.35) | −0.81 (−1.55 to −0.31) | 69% | 5.40 → 3.94 | −1.47 (−3.36 to −0.51) | 24.9 → 23.2 |
| VOCR4-FL10 | 2.02 → 1.10 (0.69) | −0.92 (−1.65 to −0.35) | 69% | 6.22 → 4.54 | −1.67 (−3.72 to −0.65) | 33.9 → 29.4 |
| VOCR4-H15 | 1.89 → 0.94 (0.35) | −0.96 (−1.67 to −0.43) | 62% | 5.86 → 4.21 | −1.66 (−3.69 to −0.63) | 29.9 → 28.5 |
| VOCR4-W20 | 1.22 → 0.79 (0.37) | −0.43 (−1.01 to +0.00) | 50% | 5.37 → 3.93 | −1.44 (−3.33 to −0.50) | 26.6 → 23.4 |

Whole dialogue per variant (pooled): base 4.38-5.05% → 3.22-3.80%, +4 5.54-6.52% → 3.98-4.72%, +8 6.17-7.16% →
4.56-5.29%. Per cohort, all variants: tv_dlg −2.32 to −2.68 points, dn_dlg −0.61 to −1.07, own_dlg −0.28 to −0.47.
own_pairs barely moves (GRAD 0.02-0.38% against ink 0.01-0.43%): on the owner's lossless frames the GRAD crop
already reads as well as the hand-cut one. The refinement returns under 1% more lines (splits) on whole
dialogue (tv_dlg 2,556 → 2,573 crops).

By platform, the development gain is mostly PSP: whole-dialogue Vita frames improve on teacher_val (scaled and
native, e.g. VOCR3-W20 4.80% → 4.16%) and are level on devnative's native Vita frames (2.45-2.92% → 2.53-2.92%)
and the owner's CHAOS;CHILD (0.24-0.61% → 0.17-0.65%); devnative's Vita pairs are level or better except
VOCR4-W20 (0.26% → 0.43%, 2 errors more on 1,153 characters).

## 5. The protected sets: once, report only

After the rule and the harness were frozen and pushed (`1747eb4`, 2026-10-14T16:02:28Z) the protected sets were
read once (`protected.attempt.json` 16:14:52Z, `clannad.attempt.json` 16:22:02Z, created exclusively). Nothing
was changed after them.

**vita-test + psp-test, whole dialogue** (`eval/finder_tighten_protected_20261014.json`): eval's 175 regions
(82 Vita, 93 PSP; 2 Vita regions with an uncertain line left out whole) at base / +4 / +8, GRAD against GRAD +
refinement. Body (dialogue) text is the primary scope; these regions include their name plates, which count in
"all text". Fixed lines use no finder and do not change.

| model | body, all variants | Δ (VN CI) | Vita body | PSP body | all text Δ (VN CI) |
|---|---|---|---|---|---|
| VOCR3-FL10 | 11.69 → 10.56 | −1.13 (−2.91 to +0.54) | 7.80 → 7.81 | 15.64 → 13.36 | −1.74 (−3.51 to −0.12) |
| VOCR3-H15 | 11.34 → 10.09 | −1.25 (−3.05 to +0.39) | 7.18 → 7.38 | 15.56 → 12.84 | −1.67 (−3.47 to −0.03) |
| VOCR3-W20 | 11.00 → 9.72 | −1.28 (−3.14 to +0.44) | 7.01 → 7.33 | 15.05 → 12.15 | −1.57 (−3.37 to +0.09) |
| VOCR4-FL10 | 11.84 → 10.65 | −1.19 (−2.91 to +0.38) | 7.78 → 7.81 | 15.95 → 13.53 | −1.81 (−3.51 to −0.29) |
| VOCR4-H15 | 11.23 → 10.08 | −1.15 (−2.74 to +0.30) | 7.47 → 7.34 | 15.05 → 12.86 | −1.64 (−3.20 to −0.23) |
| VOCR4-W20 | 10.96 → 9.79 | −1.17 (−2.95 to +0.44) | 6.74 → 7.24 | 15.24 → 12.37 | −1.54 (−3.33 to +0.08) |

**CLANNAD long run** (`eval/finder_tighten_clannad_20261014.json`): owner-drive-ingest's frozen Claude references
(3,048 scored lines on 2,044 pages, `refs_sha256` `35411688…`). Each page's ROI was cut from its native frame and
run through the C finder, which reproduces every scored line's box; the GRAD reading is that box (the numbers equal
the published VOCR4 CLANNAD scores exactly), the refined reading is every refined line from it, joined.
Whole-page bootstraps, clannad_refs' scoring.

| model | GRAD | refined | Δ (page CI) |
|---|---:|---:|---|
| VOCR3-FL10 | 3.24% | 3.43% | +0.20 (+0.07 to +0.33) |
| VOCR3-H15 | 3.24% | 3.37% | +0.13 (+0.01 to +0.26) |
| VOCR3-W20 | 2.34% | 2.28% | −0.06 (−0.19 to +0.07) |
| VOCR4-FL10 | 3.05% | 3.23% | +0.18 (+0.06 to +0.30) |
| VOCR4-H15 | 2.79% | 2.86% | +0.07 (−0.06 to +0.21) |
| VOCR4-W20 | 1.81% | 2.24% | +0.43 (+0.31 to +0.56) |

VOCR4-W20 trained on the owner's planetarian PSP frames (Key/Prototype, like CLANNAD's Vita port); its CLANNAD
numbers carry that exposure.

**Reading the two together.** PSP text (12-20 px ink in 480×272 frames, often in boxes GRAD grows into the frame
and the next line) gains the most, on development data and on psp-test alike. On 960×544 Vita text the GRAD crop
is already close to the text and the recognizers were fine-tuned on GRAD crops too (the real pools hold GRAD
crops), so cutting it back moves them off what they learned; the development cohorts held too few native Vita
frames of CLANNAD's kind to show it. **Recommendation for the overlay** (made after seeing the protected sets, so
it is a deployment caution, not a selection): apply `VOCR_REFINE_DEFAULT` where the text is PSP-sized (PSP titles,
including under Adrenaline) and keep flags 0 for native Vita titles until a Vita development cohort shows a gain,
with A/B logging in the test build. The refinement and its default are unchanged.

## 6. Cost

`eval/finder_tighten.py a9` (`eval/finder_tighten_a9_20261014.json`): the Vita compiler's machine code
(arm-vita-eabi-gcc, Thumb-2, -O2) for the finder and the refinement, linked into the host driver, under the
plugin-enabled qemu-arm with the Vita's cache sizes, on the 1,236 base whole-dialogue regions; run with flags 0
and with the default, the difference is the refinement (99.99% of it in the refinement's own functions and
`memset`/`memcpy`). Cortex-A9 costs as in `vita/a9_cycles.py`:

- 471,480 instructions per finder line; **est. 0.53-1.07 ms per line at 444 MHz, 0.71-1.42 ms at 333 MHz**
  (the driver gave the refinement only the finder's arena, so 78% of the boxes were read twice; the overlay's
  scratch makes them all one pass, which only removes work). Most of it is the mask (`mask_row`), the grey
  conversion and the band columns.
- Against recognition (`docs/vocr4.md`): FL10 94-179 ms, W20 235-416 ms per 640-px line at 444 MHz, so under 1%.
- Host: 36.7 µs per region on the 1,236 regions (Xeon, gcc -O2, best of 20, three repeats within 0.3%).

### Changed after the protected run (test code only)

A longer self-test (5,000 regions) found that one assertion in `runtime/vocr_lines_refine_test.c` compared the
rows read by two runs on arenas of different sizes; whether a box is read once or twice depends on the arena, so
the counts can legitimately differ (the lines were identical). The assertion now compares the two forms on the
same arena bytes, the self-test's random cases no longer depend on the compiler's argument evaluation order, and
`make check-lines` runs 1,000 self-test regions instead of 200. The driver's batch mode, which produced every
result here, and the refinement itself (`vocr_lines_refine.c/.h`, `synth/lines_refine.py`: their sha256 are the
protocol's, `tests/test_lines_refine.py` checks it) are unchanged; the protocol's sha256 of the driver is the
pre-fix file.

## 7. Limits

- Development labels are engine-plus-assistant agreed, not hand-checked; the protected references are Claude's.
- The remaining development gap on dialogue lines is mostly at the line's ends (portraits, ornaments, scenery
  wider than the cursor and portrait tests allow) and one striped background the ink mask cannot separate.
- Speaker names are not fixed (23-31% CER on GRAD and refined crops alike): GRAD's name boxes often hold the
  plate's decoration, and the owner's 13:57Z rule makes names secondary.
- Vita: see §5. The development cohorts are PSP-heavy (devnative: 108 Vita lines of 551).
- Every console time is a qemu-based estimate; nothing ran on a console.

## 8. For VOCR5

Appended to the fleet's `prompts/vocr5-train.md`: augment line crops the way the console's finder cuts them, so the
model is robust to what is left: the box border or inner rule above or below the text, the edges of the
neighbouring lines and furigana fragments, scenery rows, boxes from ink + 4 px up to about 1.6× the ink height;
a trailing cursor or icon 1-2 line heights after the last glyph, and portrait or ornament edges at either end.
Keep the real pools' GRAD crops and add their refined copies (the twin on the same regions), and train on tight and
loose crops of native Vita text, since tightening hurt CLANNAD.

## 9. Reproduce

```sh
A=$HERD/shared/artifacts; F=$A/finder-tighten; C=$F/cohorts
.venv-ci/bin/python eval/finder_tighten.py cohorts --out $C                       # private cohorts (sha256 in the protocol)
make -C runtime check-lines PYTHON=$PWD/.venv-ci/bin/python                     # selftest + parity under ASan/UBSan
.venv-ci/bin/python tests/lines_refine_parity.py --bin runtime/build/lines/vocr_lines_refine_test \
    --cohort $C/dn_pairs.pkl --cohort $C/own_pairs.pkl --cohort $C/tv_dlg.pkl --cohort $C/dn_dlg.pkl --cohort $C/own_dlg.pkl
for m in VOCR3-FL10 VOCR3-H15 VOCR3-W20 VOCR4-FL10 VOCR4-H15 VOCR4-W20; do                # one heavy.sh slot each
  HEAVY_SLOTS=12 $HERD/heavy.sh .venv-ci/bin/python eval/finder_tighten.py score --cohorts $C --arm $m=BLOB --out $F/dev/$m
done
.venv-ci/bin/python eval/finder_tighten.py report --scores $F/dev/* --out eval/finder_tighten_dev_20261014.json
# the protected sets ran once (attempt records refuse a second run):
#   eval/finder_tighten.py protected --data $A/eval --arm ... --out $F/protected
#   eval/finder_tighten.py clannad --work $HERD/shared/private/owner-drive-20261013/clannad-compare --arm ... --out $F/clannad
VITASDK=$A/vita/vitasdk ARM_CC=... .venv-ci/bin/python eval/finder_tighten.py a9 --cohorts $C \
    --cohort tv_dlg --cohort dn_dlg --cohort own_dlg --qemu $A/vita/qemu --out $F/a9
```

The blobs are the VOCR3 and VOCR4 release assets (sha256 in `eval/finder_tighten_protocol.json`).
