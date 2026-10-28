# vita-vn-ocr

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/charts/vocr-vs-lens-dark.svg">
  <img src="docs/charts/vocr-vs-lens-light.svg" width="100%" alt="Bar chart of dialogue character error rate, lower is better; every reader read the same line boxes. CLANNAD, PS Vita, 2,052 lines: VOCR6 Small 0.72%, VOCR6 Normal 0.40%, VOCR6 Large 0.26%, VOCR5 Small 1.38%, VOCR5 Normal 0.96%, VOCR5 Large 0.73%, Google Lens 1.53%, MeikiOCR 0.78%, PP-OCRv6 small 2.10%, PP-OCRv5 mobile int8 3.27%, manga-ocr 2.06%. vita-test, PS Vita, 153 lines: VOCR6 Small 0.92%, VOCR6 Normal 0.46%, VOCR6 Large 0.42%, VOCR5 Small 1.00%, VOCR5 Normal 0.96%, VOCR5 Large 0.59%, Google Lens 3.72%, MeikiOCR 1.38%, PP-OCRv6 small 1.34%, PP-OCRv5 mobile int8 2.34%, manga-ocr 7.94%. psp-test, PSP, 183 lines: VOCR6 Small 1.69%, VOCR6 Normal 1.17%, VOCR6 Large 0.70%, VOCR5 Small 2.22%, VOCR5 Normal 1.30%, VOCR5 Large 1.00%, Google Lens 7.52%, MeikiOCR 2.22%, PP-OCRv6 small 1.09%, PP-OCRv5 mobile int8 5.52%, manga-ocr 12.39%. Dance with Devils, PS Vita, 1,152 lines: VOCR6 Small 0.12%, VOCR6 Normal 0.09%, VOCR6 Large 0.01%, VOCR5 Small 0.27%, VOCR5 Normal 0.17%, VOCR5 Large 0.11%, Google Lens not run, MeikiOCR 0.89%, PP-OCRv6 small 0.75%, PP-OCRv5 mobile int8 2.06%, manga-ocr 5.07%. Lens's vita-test and psp-test figures include the furigana it reads. VOCR6 is built to run offline in the game's memory on the console in 0.8 to 2.5 MB of int8 weights and has not run on a console yet; Google Lens is a cloud service that needs an internet connection.">
</picture>

**VOCR6 reads visual-novel dialogue better than Google Lens on every test set Lens read, and VOCR6 Large makes the
fewest errors of every OCR engine we measured, on all four test sets. It needs no internet: it is built to run on the
PS Vita itself, in 0.8 to 2.5 MB of int8 weights.**

A tiny Japanese OCR made only for visual-novel dialogue on PS Vita and PSP. It is meant to run on the console itself, inside [Rubi](https://github.com/bee-san/Rubi_for_vita) (Rubi for Vita, named vita-jp-overlay before 2026-10-17), in as little memory as possible.

General OCR models are too big for this. The best general option, a trimmed int8 PP-OCRv5 mobile, still peaks at about 13 MiB (see `docs/research-findings.md`). VN dialogue is a much narrower problem: one or a few horizontal lines in a dialogue box, a handful of game fonts, and a fixed screen size (960×544 on Vita, 480×272 on PSP). A recognizer trained only for that case can be an order of magnitude smaller.

**Release [v0.5.0](https://github.com/bee-san/vita-vn-ocr/releases/tag/v0.5.0) collects the models:** the three
VOCR6 recognizers Rubi runs, its dialogue-line spotter, the character LMs, every earlier model from VOCR3 on
(ctclm.1's superseded LMs aside) and VOCR2's trial blob, byte for byte as their own releases published them, with one
`SHA256SUMS`. [`docs/models.md`](docs/models.md) lists every file with its size, sha256, role, numbers, status (which
ones Rubi uses) and licence, and what stays on the older releases.

## VOCR6 against other OCRs

Every reader read the same lines, cut from the same boxes. Every number is dialogue CER: the character error rate on
the lines without a speaker name, measured on the host; lower is better. VOCR's rows are the recognizer alone
(greedy, int8 through the C runtime, no language model); the other engines run their own decoders.

| reader | weights | est. time per 3-line text box, one PS Vita core at 444 / 333 MHz | CLANNAD | vita-test | psp-test | Dance with Devils |
|---|---|---|---:|---:|---:|---:|
| **VOCR6 Small** [`VOCR6.2_S_w8.vocr`](https://github.com/bee-san/vita-vn-ocr/releases/download/v0.5.0/VOCR6.2_S_w8.vocr) | 819,792 B int8 | 0.50 / 0.63 s | 0.72% | 0.92% | 1.69% | 0.12% |
| **VOCR6 Normal** (the default) [`VOCR6.2_N_w8g2.vocr`](https://github.com/bee-san/vita-vn-ocr/releases/download/v0.5.0/VOCR6.2_N_w8g2.vocr) | 1,189,408 B int8 | 0.61 / 0.77 s | 0.40% | 0.46% | 1.17% | 0.09% |
| **VOCR6 Large** [`VOCR6.2_L_w8_2l.vocr`](https://github.com/bee-san/vita-vn-ocr/releases/download/v0.5.0/VOCR6.2_L_w8_2l.vocr) | 2,475,680 B int8 | 1.47 / 1.88 s | **0.26%** | **0.42%** | **0.70%** | **0.01%** |
| VOCR5 Small `VOCR5_S_w8.vocr` | 819,792 B int8 | 0.50 / 0.63 s | 1.38% | 1.00% | 2.22% | 0.27% |
| VOCR5 Normal `VOCR5_N_w8.vocr` | 1,185,632 B int8 | 0.72 / 0.92 s | 0.96% | 0.96% | 1.30% | 0.17% |
| VOCR5 Large `VOCR5_L_w8.vocr` | 2,473,696 B int8 | 1.63 / 2.08 s | 0.73% | 0.59% | 1.00% | 0.11% |
| Google Lens | a cloud service | needs an internet connection | 1.53% | 3.72%\* | 7.52%\* | not run |
| MeikiOCR (meikiocr 0.3.5) | 18,593,254 B fp32 ONNX | not timed for a console | 0.78% | 1.38% | 2.22% | 0.89% |
| PP-OCRv6 small | 21,234,383 B fp32 ONNX | not timed for a console | 2.10% | 1.34% | 1.09% | 0.75% |
| PP-OCRv5 mobile, trimmed int8 | 2,749,540 B int8 (ncnn) | not estimated in int8; in fp32 about 5 s per 640-px line at 444 MHz (`docs/research-findings.md`, an estimate) | 3.27% | 2.34% | 5.52% | 2.06% |
| manga-ocr | 444,023,808 B fp32 | not timed for a console | 2.06% | 7.94% | 12.39% | 5.07% |

As Rubi runs them (beam + the 1 MB LM, the cascade S → N → L, Rubi's text filter), on the same lines:

| | CLANNAD | vita-test | psp-test | Dance with Devils |
|---|---:|---:|---:|---:|
| VOCR6 Small / Normal / Large | 0.47 / 0.33 / 0.31% | 0.42 / 0.17 / 0.33% | 0.74 / 0.56 / 0.48% | 0.21 / 0.19 / 0.19% |
| VOCR5 Small / Normal / Large | 0.73 / 0.66 / 0.60% | 0.67 / 0.50 / 0.42% | 0.70 / 0.48 / 0.52% | 0.27 / 0.26 / 0.22% |

- **Rubi's text filter** drops lines with no kana or kanji: on Dance with Devils that alone costs 0.19 points (6 lines,
  26 reference characters), on CLANNAD 0.08 (4 lines, 26 reference characters). It is why Rubi's path reads Dance
  with Devils, and CLANNAD at Large, higher than the recognizer alone; at Small and Normal the LM and the cascade gain
  more on CLANNAD than the filter costs.
- **The sets.**
  - CLANNAD: 2,052 dialogue lines of a PS Vita playthrough in Vita3K, references read by Claude and frozen before any
    score.
  - vita-test and psp-test: 153 and 183 hand-checked dialogue lines of 16 and 14 games, from VNDB screenshots.
  - Dance with Devils (PCSG00807): the owner's hooked test game, 1,152 lines, with the game's own text as the
    reference.
- **The crops.** On vita-test and psp-test VOCR reads the crops in the console's polarity (dark text inverted), as
  Rubi does, and the other engines read the colour crops `eval/run.py` cuts, in screen colours. On CLANNAD and Dance
  with Devils every reader read the same stored grey crops. Lens never reads private game data, so it did not read
  Dance with Devils.
- **The blobs.** On CLANNAD, vita-test and psp-test the VOCR blobs read with their head sections off, as VOCR5's
  published path reads them; on Dance with Devils with them on, as Rubi runs them (VOCR5 as its v0.3.6-head2level.1
  blobs, the same weights).
- **Against Google Lens** (95% intervals of the paired difference, clustered by game or page): every VOCR6 model reads
  CLANNAD, vita-test and psp-test better, with every interval below 0.
  - Small −0.80 / −2.80 / −5.82 points, Normal −1.12 / −3.26 / −6.35, Large −1.27 / −3.30 / −6.82.
  - On CLANNAD, VOCR6 Large also reads as many lines exactly as Lens (96.5%).
  - \* Part of Lens's vita-test and psp-test error is the furigana above a line, which Lens transcribes and the
    references leave out: 116 of its 227 psp-test errors and 30 of its 89 vita-test errors (names included) fall on
    the 20 and 3 lines with furigana.
- **Speaker names** (outside the dialogue metric: Rubi's users crop the name plate out) go either way. Lens reads
  CLANNAD's at 0.68% and vita-test's at 0.00%, against VOCR6 Small / Normal / Large's 8.24 / 11.23 / 1.45% and
  3.49 / 0.58 / 1.16%; but psp-test's at 25.84%, against 7.18 / 2.87 / 1.44%.
- **Against the other engines:**
  - **VOCR6 Large** has the lowest CER of every reader on all four sets. Its intervals exclude 0 against every
    engine on every set, except PP-OCRv6 small on psp-test (−0.39 [−0.93, +0.07], level).
  - **VOCR6 Normal**, the default, reads lower than every other engine on CLANNAD, vita-test and Dance with Devils,
    each interval below 0. On psp-test PP-OCRv6 small reads 1.09% against its 1.17% (+0.09 [−0.67, +1.03], level).
  - **VOCR6 Small** reads lower than every other engine in point on every set but psp-test, where PP-OCRv6 small
    reads 1.09% against its 1.69% (+0.61 [−0.35, +1.77], not resolved). Its intervals are below 0 against every
    engine on Dance with Devils. Four of its leads are level: MeikiOCR on CLANNAD (−0.06 [−0.19, +0.07]), on
    vita-test (−0.46 [−1.09, +0.10]) and on psp-test (−0.52 [−1.70, +0.26]), and PP-OCRv6 small on vita-test (−0.42
    [−0.83, +0.04]).
  - MeikiOCR, PP-OCRv6 small and manga-ocr ran on the host from 18.6 to 444 MB of fp32 weights; none of them was
    timed for a console.
- **Against VOCR5:** VOCR6 is lower at every tier on every set, recognizer alone.
  - Resolved on CLANNAD (−0.66 / −0.56 / −0.48 points) and on Dance with Devils (−0.16 / −0.09 / −0.09), and for
    Normal on vita-test (−0.50 [−1.18, −0.08]).
  - PSP is the exception: as Rubi runs them, VOCR6 Small and Normal read psp-test worse in point (Small 0.74%
    against 0.70%, Normal 0.56% against 0.48%); Large reads it at 0.48% against 0.52%.
- **What is not unseen** (VOCR6's training data; every number above carries these):
  - CLANNAD is not an unseen game for VOCR6. Two hooked training games (Grisaia Spinout, Little Busters!) draw
    dialogue in CLANNAD's own glyph raster, and Little Busters! and planetarian share the Key 10th Memorial BOX
    with CLANNAD. Every VOCR5 and VOCR6 model also trained on the owner's console captures, planetarian's PSP port
    among them.
  - VOCR5-S, and VOCR6-S through its checkpoint, also trained on synth3 v2, whose renders show CLANNAD's main cast
    names in 7 crops (speaker names: the dialogue metric leaves them out; `docs/vocr5.md`).
  - The synthetic renders under every VOCR5 and VOCR6 model (synth3 v2 and v3, synth4; VOCR6 replays synth3 v3 and
    synth4 too) draw speaker names from a pool that holds the names of 128 characters of 12 VNs sharing a release
    with a test or development-sample VN (about 0.05-0.06% of the crops). None occurs in a vita-test or psp-test line;
    two short ones, common words and not CLANNAD names, occur in 8 of CLANNAD's 3,048 reference lines
    (`docs/vocr5.md`).
  - VOCR6 trained on Collar x Malice (PCSG00866), which shares releases with vita-test's Collar x Malice
    -Unlimited-. 38 of the 49 hooked titles' VNs were already in VOCR5's fine-tune pools.
  - No VOCR6 run trained on Dance with Devils, and the LMs' game text holds none of it. Its VN gave 4 lines to
    VOCR5's development sample.
  - Rubi's path: the 1 MB LM carries character 3-gram counts of the hooked train titles' text (Little Busters!,
    planetarian and Collar x Malice among them), and its Jiten word list counts the scripts of CLANNAD, of 7 of
    vita-test's 16 games and of 1 of psp-test's 14.
- **Whose readings the references were read against.** CLANNAD's first round had MeikiOCR's and PP-OCRv6 small's
  readings as its candidates, and the later rounds Lens's and the earlier VOCR models' among others, unattributed.
  vita-test's and psp-test's lines were read against PP-OCRv5 server, PP-OCRv6 small and manga-ocr. Dance with
  Devils' are the game's own text. This favours those engines, not VOCR6.
- **Times and sizes.** VOCR's times are estimates for one Cortex-A9 core without prefetch credit, calibrated to
  VOCR3-H15's measured console rate: VOCR6's as Rubi runs its blobs (with their head sections), VOCR5's for its plain
  blobs. **Nothing of VOCR6 has run on a console.** The other engines' sizes are their weight files (manga-ocr's: 4 B
  per parameter); they were not timed for a console.
- **Records.** The engines' read: `eval/other_readers_20261028.json`, registered in `eval/other_readers_protocol.json`
  before it. Its scorer first recounts VOCR6's, VOCR5's, Lens's and MeikiOCR's committed numbers from their reads'
  private per-line records (28 of 28 equal). VOCR6's and VOCR5's numbers are their final read, done once after every
  choice: `eval/vocr6_final_test_followup_20261027.json`. Nothing was tuned on these sets.

### The whole page

Rubi finds the dialogue lines first. Through its finders, with VOCR5-N and the 1 MB LM reading every line
(`eval/textspot8_test_20261027.json`, read once):

| finder | weights; est. time at 333 MHz | Dance with Devils, 2,055 page regions | CLANNAD, 2,052 dialogue lines in their page regions | vita-test + psp-test, 176 whole frames |
|---|---|---:|---:|---:|
| **textspot8 spotter** (Rubi's default) | 52,176 B int8; 82 ms per 900×150 region, at most 264 ms per frame | **0.97%** | 0.97% | **1.91%** |
| Rubi's previous finder (GRAD with its COLGAP fix) | no weights | 0.98% | 0.95% | |
| GRAD + refine | no weights | 1.24% | **0.82%** | |
| textspot3's whole-frame spotter | 51,008 B int8; 268 ms per frame | 1.48% | 0.85% | 2.29% |

- VOCR6 has not been read through the spotter on these sets. On unseen development games the spotter reads 0.99% with
  v0.4.0-vocr6.1's VOCR6-N → L as the recognizer (1.03% with VOCR5-N).
- Through Rubi's previous finder, as Rubi runs them, VOCR6 Small / Normal / Large read Dance with Devils' 2,055 page
  regions at 0.95 / 0.91 / 0.89% (VOCR5: 0.96 / 0.94 / 0.92%; `eval/vocr6_final_test_followup_20261027.json`).
- The line finder made most of the remaining error: through Rubi's previous finder, about 74-83% of other games'
  whole-dialogue error (`docs/vocr6.md`). The spotter was built for it.

## VOCR6 (release v0.5.0; first published as pre-release v0.4.1-vocr6.2)

**VOCR6-S, VOCR6-N and VOCR6-L are Rubi's Small, Normal (the default) and Large model sizes.** The first VOCR6
release, v0.4.0-vocr6.1, replaced Normal and Large; v0.4.1-vocr6.2 replaced all three.

VOCR6 keeps VOCR5's shapes. Each tier is fine-tuned from its VOCR5 checkpoint on VOCR5's data plus two new sources:
- the owner's hooked PS Vita and PSP corpus: 48 training games, exact game text on native frames (a 49th, Dance with
  Devils, is the test title);
- the line finder's own crops of those pages.

A recipe was adopted only if, on Rubi's shipped path and with three seeds, it read the hooked games' held-out pages
better and other games no worse. Normal and Large were then trained 3x longer, which kept other games no worse
(stage 4c). Small needed another recipe, half the learning rate (stage 4d): the third tried at Small, after 4a's
recipe and 4d's other arm failed.
All numbers are measured on the host (`docs/vocr6.md`). Console times are estimates. **Nothing of VOCR6 has run on a
console.**

| model | tier | int8 weights | arena (W8I) | hooked games' pages, held out: VOCR5 → VOCR6 | other games' dev score: VOCR5 → VOCR6 | CLANNAD lines, Rubi's path\* | vita-test + psp-test fixed lines, Rubi's path | est. time per 3-line box, 333 / 444 MHz |
|---|---|---:|---:|---|---|---|---|---|
| [`VOCR6.2_S_w8.vocr`](https://github.com/bee-san/vita-vn-ocr/releases/download/v0.5.0/VOCR6.2_S_w8.vocr) | Small | 819,792 B | 124,832 B | 1.177 → 1.049% | 1.674 → 1.549 | 0.73 → 0.47% | 0.68 → 0.58% | 0.63 / 0.50 s |
| [`VOCR6.2_N_w8g2.vocr`](https://github.com/bee-san/vita-vn-ocr/releases/download/v0.5.0/VOCR6.2_N_w8g2.vocr) | Normal (the default) | 1,189,408 B | 184,448 B | 1.008 → 0.936% | 1.482 → 1.389 | 0.66 → 0.33% | 0.49 → 0.36% | 0.77 / 0.61 s |
| [`VOCR6.2_L_w8_2l.vocr`](https://github.com/bee-san/vita-vn-ocr/releases/download/v0.5.0/VOCR6.2_L_w8_2l.vocr) | Large | 2,475,680 B | 262,480 B | 0.990 → 0.917% | 1.439 → 1.396 | 0.60 → 0.31% | 0.47 → 0.40% | 1.88 / 1.47 s |

- **The columns:**
  - "hooked games' pages" is dialogue CER through Rubi's finder, LM and cascade on hval pages (held out), as the
    mean of three seeds of the recipe: stage 4a's for Normal and Large, stage 4d's for Small. The shipped blobs then
    trained on hval too. The longer schedule's own held-out evidence is stage 3's S3long pair (Normal, -0.043
    [-0.077, -0.014]);
  - the dev score is VOCR5's development score on other games, for the set as it ships (Rubi's cascade S → N → L).
    VOCR5-S in the same cascade reads 1.509, so Small is +0.041 [-0.016, +0.099] against it (not resolved; Small's
    adopted recipe was the third one tried);
  - CLANNAD and vita-test + psp-test were read once for this set, after every choice;
  - "Rubi's path" is the final reads' model of Rubi's line path: the GRAD finder with its COLGAP fix, Rubi 892cfb9.
    Rubi's default finder is now the textspot8 spotter (#149, Rubi #396), whose own test read is in
    `eval/textspot8_test_20261027.json`.
- **\* The CLANNAD exposure.** Two hooked training games draw dialogue in CLANNAD's own glyph raster, so CLANNAD's
  gain is not an unseen-game result (`docs/synth6.md`, `docs/vocr6.md`). The other seen flags are in the notes above.
- **Against v0.4.0-vocr6.1:** CLANNAD's lines read better (Small -0.21 and Normal -0.10, resolved; Large -0.05
  [-0.10, +0.01]), but its pages through Rubi's finder read worse at Normal and Large (+0.33 / +0.27, resolved; still
  better than VOCR5 or level). vita-test + psp-test: Small better (-0.23 [-0.46, -0.04], resolved), Normal and Large
  level.
- **PSP alone** does not improve: psp-test's fixed lines on Rubi's path +0.04 / +0.09 / -0.04, and Normal +0.26
  [+0.04, +0.51] (resolved worse) on VOCR5's locked path with beam 8 + ctclm.2's 1 MB LM. vita-test improves (Small
  -0.25, Normal -0.33, resolved).
- **The blobs:** Normal carries #120's head gate (at ε = 1e-4) and #122's two-level head, Large the two-level head.
  Small has neither: the gate rule kept no gate for it, so it is est. 0.11 s slower than VOCR5-S as Rubi runs it
  (gated). Plain blobs (`VOCR6.2_N_w8.vocr`, `VOCR6.2_L_w8.vocr`) are in the same release.

## VOCR5 (release [v0.3.0](https://github.com/bee-san/vita-vn-ocr/releases/tag/v0.3.0))

VOCR5 was the round before: `VOCR5_S_w8.vocr`, `VOCR5_N_w8.vocr` and `VOCR5_L_w8.vocr` (819,792 / 1,185,632 /
2,473,696 B of int8 weights, W8I arenas of 124,832 / 182,944 / 261,024 B), each chosen over its VOCR4 predecessor
under a pre-registered rule on development data, with model cards on v0.3.0. Rubi 0.9.8 ran them with their head
sections (v0.3.6-head2level.1); Rubi 0.10.0 runs VOCR6. They are in v0.5.0 too; their numbers are in the tables above, and the details in
`docs/vocr5.md`, the model cards and v0.3.0's release notes.

The chart is drawn by `docs/charts/vocr_vs_lens.py` from `docs/charts/vocr-vs-lens.json`, which it builds from
`eval/other_readers_20261028.json`, `eval/vocr6_results_20261027.json`, `eval/vocr6_final_test_followup_20261027.json`,
`eval/vocr5_report_20261016.json`, `eval/vocr5_protocol.json` and `eval/vocr5_dialogue_20261016.json`.
`tests/test_vocr_vs_lens_chart.py` recomputes the chart's numbers, its row notes and footnote figures, and this
README's alt text, the comparison and Rubi's-path tables, their notes and the whole-page table from those files and
`eval/textspot8_test_20261027.json`; it checks the SVGs' text against the JSON and the four images against the hashes
the render recorded.

## Status

The table records each component and each earlier round. VOCR6's follow-up (v0.4.1-vocr6.2) replaces all three VOCR5 tiers; v0.4.0-vocr6.1 replaced Normal and Large, and the VOCR5 models replace the earlier rounds' recommended models.

| | |
|---|---|
| **v0.5.0 release (every model so far)** | VOCR2's trial blob and the models of every release from VOCR3 on in one release, byte for byte as published, with one `SHA256SUMS` (86 files from 20 releases, plus `SHA256SUMS` and `v0.5.0-assets.json`; [`docs/models.md`](docs/models.md)): VOCR6-S / N / L and the plain Normal and Large blobs, v0.4.0-vocr6.1's VOCR6-N and -L, the textspot8 spotter and the earlier spotters, the four LM tiers and ctclm.2's replaced ones, VOCR5 with and without its head sections, VOCR5-V, the text-presence gate, the text detector, model B's PSP candidates, VOCR4, VOCR3 and VOCR2's trial blob. Each file is marked used by Rubi (Rubi 0.10.0's pins), previous or research / not selected, with its attribution file. The FL10 research snapshots before VOCR3 (639 MB together) and MeikiOCR's converted recognizer stay on their pre-releases. Report only: the other OCR engines read on the same lines (`eval/other_readers_20261028.json`; the README's chart and table) |
| Model | `train/model.py`: a CTC line recognizer (stem plus depthwise-separable blocks, factorised head), grey input at height 32. VOCR5 (v0.3.0) ships three sizes: Small (width 1.0, 0.73M parameters in training, 0.72M in the exported blob after BatchNorm folding, 4,987 classes), Normal, the default (width 1.5, 1.07M, 5,631 classes) and Large (late-wide, width 2.0 early and 3.0 late, 2.32M, 5,631 classes). v0.2.0's H15 was width 1.5, 1.04M parameters, 5,355 classes |
| **v0.4.1-vocr6.2 (VOCR6's follow-up, pre-release)** | VOCR6-S, VOCR6-N and VOCR6-L, all three of Rubi's tiers (`docs/vocr6.md`, "The follow-up", #125). Normal and Large: v0.4.0's recipe trained 3x longer (stage 4c), not resolved worse on other games (dev -0.063 / -0.046 points against v0.4.0's). Small: the recipe at half the learning rate (stage 4d, the third recipe tried at Small), hooked games' held-out pages 1.177 → 1.049% (resolved better), other games +0.064 (not resolved). Read once: CLANNAD lines 0.73 / 0.66 / 0.60 → 0.47 / 0.33 / 0.31% (resolved better; the same seen flags as v0.4.0), vita-test + psp-test level (psp-test not better), CLANNAD pages worse than v0.4.0's at Normal and Large. Normal with the head gate (ε = 1e-4) and the two-level head, Large with the two-level head, Small plain: est. 0.63 / 0.77 / 1.88 s per 3-line box at 333 MHz. Nothing of it has run on a console |
| **v0.4.0-vocr6.1 (VOCR6, pre-release)** | VOCR6-N and VOCR6-L, Rubi's Normal and Large tiers; Small stays VOCR5-S. Fine-tuned from VOCR5's checkpoints on the owner's hooked corpus and the finder's crops of its pages (`docs/vocr6.md`, #125). On Rubi's shipped path: hooked games' held-out pages 1.008 → 0.936% (N) and 0.990 → 0.917% (L), both resolved better; other games level (dev -0.038 / +0.026 points). Read once: CLANNAD lines 0.66 → 0.44% / 0.60 → 0.36% (seen flags: CLANNAD's glyph raster, the Key box and Collar x Malice games and 38 of 49 hooked VNs in VOCR5's pools; the lm6 LM's text and Jiten lists), vita-test + psp-test level (psp-test's Normal not better). Two-level head, no gate: est. 0.79 / 1.87 s per 3-line box at 333 MHz. Nothing of it has run on a console |
| **v0.3.0 release (VOCR5, the final round)** | Three int8 recognizers, one per tier of Rubi's model sizes: **Small** `VOCR5_S_w8.vocr` (819,792 B, FL10's topology, 4,987 classes), **Normal** (the default) `VOCR5_N_w8.vocr` (1,185,632 B, H15's topology, 5,631 classes) and **Large** `VOCR5_L_w8.vocr` (2,473,696 B, late-wide, 5,631 classes). Each replaced its VOCR4 predecessor under a pre-registered rule on development data. vita-test + psp-test dialogue CER (fixed lines, console polarity): **1.60% / 1.13% / 0.79%** (0.53% for Large with the overlay's beam + 1 MB LM); whole dialogue 8.31% / 7.98% / 7.08%; CLANNAD dialogue 1.38% / 0.96% / **0.73%** (Google Lens 1.53%). Estimated 0.63 / 0.92 / 2.08 s per 3-line box at 333 MHz on one A9 core without prefetch credit (#85's runtime; estimates, nothing of VOCR5 has run on a console). `docs/vocr5.md` |
| v0.2.0 release (train2) | Pretrained on NVIDIA OCR-Synthetic-Multilingual-v1 Japanese crops (nvsynth) mixed with VN-style synthetic boxes (synth2), fine-tuned on teacher-labelled real crops. **H15** (recommended, width 1.5): real Vita screenshots (`eval/`, vita-test) **1.56% CER, 83.5% exact**; real PSP (psp-test) **2.63% CER, 82.4% exact**, int8 through the C runtime. That beats trimmed int8 PP-OCRv5 mobile (2.30% / 5.34%) on both sets, at 1.16 MB of int8 weights and a 183 KB arena. Not every style is at or below that model's yet: H15 is above it on names, coloured text, PSP furigana and text without a box (4-62 lines each, mostly within noise). **F20** (width 2.0): 1.79% / 2.79%. **FL10** (PSP-lean, width 1.0, 3,500-char vocab, 658 KB): 2.30% / 5.18%, so it ties the bar on vita-test. An estimated 166-302 ms per 640-px line for H15 (w8i, one 444 MHz A9 core; about 220-400 ms at the 333 MHz games run at by default). H15 has not run on a console; FL10 has (see Hardware). Distillation from PP-OCRv6 small was tried and did not help. Ablation, per-style tables and the selection rule are in `docs/train2.md` |
| v1 baseline | Trained on synthetic dialogue boxes only. Synthetic validation: 0.34% CER, 95.6% of lines exact (`docs/v1-metrics.jsonl`). Real Vita screenshots (`eval/`, vita-test, 206 lines from 16 games): 23.75% CER, 52.4% exact, 81.8% CER on its 43 dark-on-light lines. Real PSP screenshots (psp-test, 245 lines from 14 games): 46.45% CER, 30.2% exact, 86.8% CER on its 110 dark-on-light lines (`docs/results.md`). |
| Runtime | `runtime/`: plain-C99 recognizer for Vita (NEON) and PSP (scalar), no malloc, 116-125 KB arena at any line width; int8 weights 0.86 MB (`runtime/README.md`) |
| Finder fixes | `runtime/vocr_lines_fix.c` (`docs/finder-fix.md`, #129): the GRAD line finder with three fixes behind flags, tuned on train pages of the owner's hooked corpus (#125) and measured through Rubi's page path (`eval/pagepath.py`, #131). `VOCR_FIX_DEFAULT` is COLGAP (a line's parts across a full-width space stay one line). VOCR5-N whole-dialogue CER with Rubi's decode: hval (other pages of 46 train titles) **3.875% → 3.292%** (−0.583 points [−0.926, −0.309]), development (93 other VNs) **3.036% → 2.340%** (−0.696 [−1.262, −0.249]), and about as much on regions drawn tight; the test title, read for report only, 1.241% → 0.978%. JUNK (frame strips and ornaments dropped) is opt-in: −1.395 more on hval regions drawn with a margin, but it deletes text lines in regions drawn tight (#135's audit). VALLEY (touching rows split) is opt-in too. +1,256 B of Thumb-2 at -O2 as Rubi links it, no arena change and no measurable finder time; fix 0 is the frozen finder bit for bit, and nothing of it has run on a console |
| Text detector | `runtime/vocr_textdet.c` (finder mode `VOCR_LINES_TEXTDET`, off by default; `docs/textdet.md`): a learned line detector for whole dialogue boxes, 5.7 KB of int8 weights, est. 53 ms per 900×150 box on one 333 MHz A9 core. With the refinement it is resolved better than GRAD on development and on vita-test + psp-test whole dialogue (VOCR5-N 5.99% against GRAD's 10.23%; psp-test 5.58% against GRAD + refine's 11.53%) and returns no line for 92% of empty development regions (GRAD 8%; a caveat on how many of them its synthetic training scenes share is in the docs). It does not meet the pre-registered rule (not resolved better than GRAD + refine on development), and on CLANNAD it is worse as scored, so it does not become the default |
| Text-presence gate | `runtime/vocr_textgate.c` (`docs/textgate.md`, #100): a yes/no per region in front of GRAD + refine, so an empty region (a hidden text box, a CG screen) returns no line. The winner of a pre-registered size sweep, `p8_c8`: **1,540 B of int8 weights and 2,072 B of code** (VitaSDK `-Os`; #97's detector: 5,700 + 7,250 B). On development it leaves 59 of 66 empty regions with no line (GRAD + refine: 5) at +0.06 / +0.06 / +0.07 points of whole-dialogue CER (VOCR5-S / N / L, not resolved worse); on vita-test + psp-test and CLANNAD, read once, the CER is unchanged, and it removes 48 of the 162 lines GRAD + refine return on CLANNAD's 133 pages without reference text. Est. 67-135 ms per 900×150 box on one 333 MHz A9 core (scalar code; estimates). Behind a flag: off, the finder is bit for bit as before |
| Stroke split | `runtime/vocr_lines_split.c` (`VOCR_SPLIT_STROKE`, off by default; `docs/translucent-box.md`, issue #98): a non-learned step between the line finder and the refinement for text over translucent boxes and busy scenes, where GRAD merges lines; no weights, 2,432 B of Thumb-2 code at -Os. The regions a frozen classifier tags (a background as contrasty as the strokes; by eye about half of them scenes or textures) are 13% of the development dialogue regions and hold 53-54% of GRAD + refine's development dialogue errors, 71-74% of those in two VNs. On them it halves the CER (VOCR5-N 20.99% → 10.08%; all of development 4.78% → 3.45%), with no resolved change elsewhere, but the gain rests on those 2 VNs, where its idea was first checked (no VN interval excludes 0), and it is est. 79 ms per 900×150 box at 333 MHz, so its pre-registered rule does not adopt it. vita-test + psp-test whole dialogue 9.16% → 7.88% (VOCR5-N; psp-test −2.5 points, intervals include 0); its few false splits cost a little on CLANNAD (+0.04 points) |
| Counters and cut lines | `runtime/vocr_post.c` and `runtime/vocr_lines_widen.c` (`docs/counters.md`, #109), for CLANNAD's 「九ヶ月です」, which the owner's Vita read as ルッ月です」. A text rule turns ッ into ヶ before 月 and 年 (and before 国 and 条 after a numeral or 何数幾, and 所 after one of those other than 一). ッ月 and ッ年 occur nowhere in 700 M characters of training-side text; the rule changes no development line. A counter re-decode over the per-frame top-K turns a misread numeral back (ー→一, ニ→二, 可→何) and puts back a cut 「: 0 real development lines change, and the synthetic counter set's CER goes from 4.29% to 3.05% (VOCR5-L, greedy) with 0 lines worse. 3,964 B of Thumb-2 at -O2, no arena. An edge continuation for lines that a tight region cuts gives −1.5 to −1.8 points there, but +0.01 to +0.04 (unresolved) on regions with a margin, so the pre-registered rule does not adopt it: off by default |
| Vertical text | `docs/vertical.md` (#99): the column finder `runtime/vocr_vertical.c` (GRAD on the transpose, columns right to left; no weights, 4,312 B of Thumb-2, arena 83,619 B at 960×544) and **VOCR5-V** (`VOCR5_V_w8.vocr`, release v0.3.2-vertical.1, VOCR5-S's shape, 819,792 B), which reads a column turned 90°. Test, read once: vertical VN dialogue 7.17% CER on fixed columns and 11.29% end to end; manga (*Black Jack ni Yoroshiku* in NOBORU's fit-to-width view) 0.86% / 2.45%. The best code-only arm (VOCR5-N on the column cut into characters) reads test VN at 33.71%. For Rubi's vertical mode; nothing of it has run on a console. **VOCR6-V** (`VOCR6_V_w8.vocr`, pre-release v0.4.2-vertical6.1, #159, `docs/vocr6v.md`) replaces VOCR5-V at the same size and arena: VOCR5-V fine-tuned on the owner's hooked vertical game (Yunohana SpRING!: its columns and the column finder's crops of its pages) and a new synthetic render. Through Rubi's own job (`ocr_vertical = on`, tight regions; three seeds of the recipe, pre-registered rule) the game's held-out pages read 7.92% → 4.64% (−3.28 [−4.48, −2.14]), and #103's development regions of other games VN 17.58% → 16.13% (−1.45 [−2.25, −0.58]) and manga 4.18% → 3.69% (−0.49 [−1.14, −0.06]); the released blob, the recipe retrained with those held-out pages folded in, reads development VN at 16.13% and manga at 3.91%; #103's test set, read once, is level (VN 7.26% → 7.05%, −0.20 [−0.69, +0.24]). The owner's Manga109 download is the dataset's super-resolution version (the 109 volume covers, no interior pages, no annotations); no Manga109 material is in the model (the covers' arm was not adopted) |
| Per-character boxes | `runtime/vocr_charbox.c` (`docs/char-boxes.md`, #114), for an overlay drawn on the game's own text (GSM-style: the word under the cursor highlighted where it is on the screen, a tap on a character looks it up): every character the recognizer emits gets a centre, a cell and an ink box from the CTC frame where it fires, mapped through the line's resize, with a per-model calibration of where the CTC fires and the cuts snapped to the gaps between glyphs. No new model, no change to `vocr.c`: 3,186 B of Thumb-2 at -Os, est. 1.6 ms per line at 333 MHz. Pre-registered bar met on development (VOCR5-N, horizontal dialogue, the centre inside MeikiOCR's box of the character): 98.9 / 99.3 / 99.4% on Vita native / PSP native / Adrenaline 2x fixed lines and 98.9 / 98.2 / 99.2% on the finder's lines; VOCR5-S/L and the spotters' lines the same; vita-test / psp-test, read once: 98.9 / 99.5% (fixed), 98.8 / 98.5% (finder). The misses are narrow punctuation; kana and kanji 99.6-100%. Nothing of it has run on a console |
| Head gate | `vocr_opts.head_gate` (`docs/headgate.md`, #117): a 96 → 16 → 1 gate on the output head's input, appended to each blob as a 1,792 B section (`VOCR5_{S,N,L}_w8g.vocr`), skips the head (96 → 4,987 / 5,631 classes) on the frames it calls blank: 61.4 / 61.0 / 57.9% of the finder-cut development frames (S / N / L). The pre-registered rule adopts it for all three tiers: the development score is not resolved worse greedy, with Rubi's beam + LM or with the cascade (every upper bound ≤ +0.050 points; Small passes at the rule's edges, its greedy lower bound exactly 0.000 and its cascade upper bound +0.050). Est. −122 / −137 / −163 ms per nominal 3-line box at 333 MHz (−19.5 / −14.9 / −7.9%; −45 / −51 / −60 ms on a real 89-frame Vita box); arena +48 B, `vocr.o` +1,940 B. vita-test + psp-test and CLANNAD, read once: greedy dialogue text unchanged on vita-test and psp-test, dialogue CER −0.043 to +0.006 points, 0-3 lines changed per tier. Off by default (today's run, bit for bit); nothing of it has run on a console |
| Two-level head | `vocr_opts.two_level` (`docs/head2level.md`, #117): the output head's rows in frequency order, the 504 most frequent (blank included) computed on every frame the gate keeps, the other 5,127 only where a 98 → 16 → 1 escape fires (21% / 13% of the kept frames on finder-cut development lines, N / L); a 1,984 B trailer after the gate, no weight changes. The pre-registered rule adopts it for Normal and Large on top of the head gate (Small keeps the gate alone; a 48-wide rare block is rejected): est. −32 / −66 ms per nominal 3-line box at 333 MHz beyond the gate as measured on 24 development lines (−62 / −94 ms projected to development lines), −160 / −284 ms with the gate against today; development score against the gate +0.000 / +0.006 points; arena +1,456 B, `vocr.o` +3,100 B. vita-test + psp-test, read once: no line changed beyond the gate's; CLANNAD dialogue: N one line beyond the gate's (on the cascade), L 3-5 lines changed where the gate alone changes 0-1 (+0.003 to +0.009 points beyond it). Off by default (today's run, bit for bit); nothing of it has run on a console |
| Coarser frame step | `docs/stride16.md` (#118, #117 item 3): every tier retrained with VOCR5's recipe at one CTC frame per 16 input columns (`train2.py --stride 16`, the 2-row stage's DS block at stride 2 × 2), Normal also at 12, 2 seeds each, under a pre-registered not-resolved-worse rule. **Not adopted at any tier**: the development score is resolved worse by +1.58 / +1.34 / +1.18 points (S16 / N16 / L16) and +0.40 for N12, rows with frames to spare included; vita-test + psp-test dialogue, read once afterwards, +1.02 / +0.66 / +0.64 (N12 +0.21, not resolved). The saving would have been est. −132 / −178 / −475 ms per nominal 3-line box at 333 MHz (N12 −136) at the same weights and a smaller arena. VOCR5-S/N/L stay; the runtime keeps op stride 3 and a stride-aware default tile and refuses an op narrower than its stride, every stride-8 blob bit for bit as before |
| Global attention | `docs/vocr6-attn.md` (#125, VOCR6 stage 3): one light global self-attention layer over the line's frames, between the bottleneck and the head (`train2.py --attn 1`; op type 5 in the blob; the C runtime in f32 / W8F / W8I, bit-exact to its numpy twin, scalar = NEON), behind a flag: without it every model, blob and run is bit for bit as before. +37,153 parameters and +40,000 B of int8 weights per tier, W8I arena +49,152 B at 1,024 px, est. +35 ms per nominal 3-line box for Normal at 333 MHz on top of the head gate and two-level head (+28 ms for Small). Not adopted: VOCR6's stage-3 arm decides; nothing of it has run on a console |
| PSP recognizer (model B) | `docs/vocr6-psp.md` (#126, VOCR6 track T5): a recognizer to run on a 64 MB PSP at native 480×272, within an est. 1 s per 3-line box at 333 MHz on the scalar build. **None is released.** On development PSP games outside the training data, the best arm the pre-registered rule calls time-feasible (VOCR5-S's layers at depth 0, input height 24) reads whole dialogue +3.93 points [+2.63, +5.59] worse than VOCR5-S, and in PPSSPP it takes 1.21 s by the same estimate. VOCR5-S's own shape, fine-tuned on the hooked PSP titles, ties VOCR5-S on those games and is resolved better on the hooked titles (hval page path −0.73 points). But it takes an est. 3.24 s per box as scored, and 2.05 s (1.42 s with the VFPU) with a gated head that the head rule did not keep. What ships: the head gate in every mode and the two-level head in W8F (bit-exact, off by default), the PSP test app's page path, memory probe and 64 MB build, and a console-test pre-release of both candidates with no game data. Nothing of it has run on a PSP |
| Full-screen text spotting | `runtime/vocr_textspot.c` (`docs/textspot.md`, #104): finds every text box on a whole frame (dialogue, names, choices, UI labels, signs, vertical columns) with its orientation, a confidence, blocks, a reading order and the likely dialogue block, with no region drawn, for Rubi's opt-in scan-screen mode. The winner of a pre-registered sweep, extended once after development scores (without that extension the same rule selects `s4_c16n4`, 3.9 KB, est. 105 ms, not resolved apart from it on development), `s4_c24n4` read at half size: **5,732 B of int8 weights**, arena 80,799 B, est. 139 ms per 960×544 frame on one 333 MHz A9 core (104 ms at 444; estimates). vita-test + psp-test as whole frames, read once: box F1 0.694 and dialogue CER 3.00% (VOCR5-N), against 0.571-0.580 and 5.54-8.88% for #97's detector run on the whole frame and 0.266 / 43.70% for GRAD + refine; CLANNAD whole frames 1.14% (#97's detector 1.29-1.44%). Half the UI labels are missed. The smallest candidate that also clears the bar, `s4p_c8` (2,732 B, est. 80 ms), is published beside it; nothing of it has run on a console. textspot2 (#106, `docs/textspot2.md`) tried bigger, meiki-relabelled spotters (up to 18.6 KB, est. <= 144 ms) under a pre-registered rule and selected none: on development they read all text at 34-44% CER against its 32.04% (the best of its recipe's three seeds, whose median reads 34.59%). A stride-8 trunk with a sub-pixel head read development dialogue at 1.13% against its 3.01%, at est. 80 ms (not selected; no test-set check). textspot3 (#108, `docs/textspot3.md`) scaled that design up to 500 KB under a pre-registered dialogue-first rule and selected `kd_t64n8c16` (8 blocks of 64 channels, distilled from a 447 KB multi-scale model): **51,008 B**, arena 240,991 B, est. 268 ms per frame at 333 MHz (201 at 444; estimates), runtime `runtime/vocr_textspot2.c`. vita-test + psp-test as whole frames, read once: F1 0.776 and dialogue CER 2.29% against the textspot winner's 0.694 / 3.00% (both resolved), all-text CER level (20.57% against 20.02%); CLANNAD whole frames 0.99% against 1.14%. The 117-447 KB multi-scale models qualified but were not resolved better on all-text CER, so the smaller one is the pick; it reads Vita frames' UI worse than the winner (vita-test all-text 23.4% against 21.0%). textspot6 (#128, `docs/textspot6.md`) trained tagged spotters of 52-448 KB on the owner's hooked pages under a pre-registered page-path rule, 3 seeds per arm, and **selected none**. Their representatives read unseen development dialogue at 1.39-2.07% CER through Rubi's page path, against GRAD + refine's 3.08%, and are resolved better on the hooked titles' hval (1.30-1.69% against 3.88%). But no unseen-dev interval excludes 0: 40% of GRAD + refine's errors there are in one VN. Read inside a region drawn tight to the text, the crop decode fails (9.15% against `rubi,fix=1`'s 2.16%). The best of them is published as the research pre-release v0.3.8-textspot6.1 (`qS4`, 118,096 B, est. 199 ms per frame at 333 MHz); Rubi keeps GRAD + refine. textspot7 (#128, `docs/textspot7.md`) trained nothing new. It added to textspot6's spotters a per-region GRAD fallback (F) and a post-pass on the boxes (D: join a row's fragments, grow each box over the ink at its ends), under textspot6's rule plus a fresh confirmation set of 41 unseen VNs frozen before its read. F does not fit the 300 ms budget (GRAD on the largest region is est. 122 ms on top of the spotter). D qualifies for three arms, and the cheapest, `base_t64n8c16` read with D, is the pick: unseen dev 1.18% against GRAD + refine's 3.08% (−1.91 [−4.55, −0.12]), hval 1.00%, confirmation set 1.92% against `rubi,fix=1`'s 4.29% (−2.37 [−5.49, −0.18]). It is not resolved better than `rubi,fix=1` on unseen dev (−1.20 [−3.83, +0.47]), so by the rule it is not Rubi's default, and Rubi takes vocr6-spotter2's textspot8 spotter (#149) instead, so this pick is not pinned in Rubi. Retrained on train + guard + hval, it is the pre-release v0.3.9-textspot7.1 (52,176 B, est. 264 + ≤ 32 ms per frame at 333 MHz); nothing of it has run on a console |
| Dialogue-line spotter (Rubi's default finder) | `docs/textspot8.md` (#149): textspot6's 52 KB T trunk (`runtime/vocr_textspot6.c`, VTSP version 4) retrained with the line ends weighted, windows cut tight to the text, synthetic spinners and page cursors as background and a wider zoom, on the hooked pages, textspot's synthetic frames and MeikiOCR's boxes on 52,785 VNDB and owner screenshots (the distillation set). Read on the player's region alone (the crop decode; est. 82 ms for a 900×150 region and at most 264 ms for a whole 960×544 frame at 333 MHz, estimates), it reads #119's unseen-game dev dialogue at **0.99%** page-path CER (VOCR5-N + the 1 MB LM; GRAD + refine 3.08%, Rubi's finder `rubi,fix=1` 2.38%, textspot6 1.39%, MeikiOCR's detector itself 0.91%), hval 1.31%, and vocr6-spotter3's fresh confirmation set of 41 VNs at 1.86% (`rubi,fix=1` 4.29%: −2.43 [−5.69, −0.30]); with VOCR6-N → L as the recognizer, unseen dev **0.95%**. Its pre-registered rule is met (textspot6's, plus the confirmation guard and vocr6-common's default test against `rubi,fix=1`), so it is Rubi's default line finder since bee-san/Rubi_for_vita#396, GRAD the fallback and an off-switch. The released blob (pre-release v0.3.10-textspot8.1, 52,176 B: the pick retrained on the 48 training titles' train, guard and hval pages, as the rule says) reads unseen dev at 1.03% (0.99% with VOCR6-N → L) and, read once, the hooked test title Dance with Devils at 0.97% (GRAD + refine 1.24%: −0.28 [−0.58, −0.005]; `rubi,fix=1` 0.98%) and CLANNAD's regions at 0.97% (GRAD + refine 0.82%, `rubi,fix=1` 0.95%, both level; with the speaker names 1.17% against 2.33% and 2.83%; two training titles draw dialogue in CLANNAD's glyph raster, so CLANNAD is not an unseen-raster result). It still loses on side-by-side choice buttons; whole-frame UI text is no better than textspot6's. Nothing of it has run on a console |
| Architecture sweep | `docs/arch-sweep.md`: width, depth, temporal context, head bottleneck, vocab, input height and int8 on the v1 recipe, 5,000-step runs (plus a 15,000-step width pair) scored on vita-test and psp-test. Recommendation for train2: width 2.0 (1.42M parameters, 1.56 MB int8, 241 KB W8I arena; the later cycle model estimates 234-411 ms per 640-px line on one 444 MHz A9 core, against the sweep's own MAC-rate estimate of 0.59-0.74 s); width 1.5 is the lighter fallback and v1's width 1.0 the PSP-lean variant. Post-training int8 is enough; no QAT needed |
| Vita | `vita/`: VitaSDK build of the runtime plus a test app (VPK). In the Vita3K emulator it reproduces the host bit for bit on 728 test cases (vita-test, psp-test, synth-val) in w8i, and on 52 of them in all three modes. Estimated 113-212 ms per 640-px line (w8i, one Cortex-A9 core at 444 MHz) from the Vita build's instruction mix (`vita/README.md`) |
| Console budget | `docs/vita-memory-budget.md`: three tiers for a recognizer in the game's own process. Always safe: int8 weights ≤ 900 KB, W8I arena ≤ 128 KiB, an estimated ≤ 300 ms per 640-px line on one 333 MHz A9 core (FL10's class; at most about 1.5 MiB of the game's memory at a job's peak). Safe with the overlay's free-memory preflight and an always-safe fallback: weights ≤ 1.6 MB, arena ≤ 256 KiB, ≤ 550 ms (H15, F20). Never: beyond that, and nothing of the recognizer in SceShell at any size (its 50 MiB partition runs within tens of KiB of empty beside a game). Inside a game, memory is not the limit: a ceiling job needs about 2.0-2.4 MB against 56-87 MiB seen free, and the preflight refuses what does not fit. Speed and the shared 2 MiB L2 bind first; the game's share of the L2 and the cost on the system core are unmeasured. It also lists why vita-jp-overlay's OCR attempts failed: never the recognizer's size; the blocking failures were integration bugs and the Shell plugin's own footprint |
| Platform memory | `docs/vita-platform-memory.md`: which PS Vita memory pool the recognizer runs in (the game's 256 MiB budget, never SceShell's 50 MiB), what that means for model size, and the corrections to the budget doc. The full cited reference is vita-jp-overlay's `docs/vita-memory.md` |
| Hardware | FL10 has run on a PCH-1001 (firmware 3.65) inside the overlay (now Rubi), in CLANNAD: model load 213-359 ms and recognition 36-145 ms per job over 14 logged jobs of varying dialogue, at an unknown clock, with no hardware CER (`notes/26`). Neither H15 nor `vita/`'s test app has run on a console, and nothing has run on a PSP; `vita/README.md` has the steps for a PCH-1000. The PSP build (`psp/`) gives the host runtime's exact output in PPSSPP, at 0.38 s emulated for the median psp-test line (`psp/README.md`) |

## Layout

```
setup_env.sh, requirements.*   Python env (uv; CPU torch)
data/fetch_assets.py           OFL fonts, Aozora/Wikipedia text, VNDB screenshot list
data/label_real.py             first real-line pseudo-labeller (PP-OCRv5 server + manga-ocr agreement)
realdata/                      real training lines from Vita, PSP and scaled PC screenshots (4-reader vote)
devnative/                     native Vita/PSP development cohort from VNDB screenshots, and its labelling pipeline (docs/devnative.md)
realplus/                      large real training pool: every training-side VNDB screenshot, devnative's labelling (docs/realplus.md)
data/vocab.txt                 output character set
teacher/                       real-screenshot labels: MeikiOCR first, local owocr engines vote (docs/teacher.md)
synth/                         synthetic dialogue-box renderer and corpus builder
nvsynth/                       Japanese line crops from NVIDIA OCR-Synthetic-Multilingual-v1 for pretraining (docs/nvsynth.md)
train/                         model, CTC training (train.py; train2.py: the release recipe), export.py (weights blob for the C runtime)
runtime/                       on-console C99: recognizer (vocr.h/vocr.c), line finder (vocr_lines.c), tests
psp/                           PSP build: pspdev toolchain, test EBOOT, PPSSPP-headless parity and timing
vita/                          PS Vita: VitaSDK build, test app (VPK), Vita3K check, Cortex-A9 cycle estimate
docs/                          research notes and results
.github/, ci/, tests/          CI: lint, unit tests, training smoke, C runtime checks (ci/README.md)
```

## Data policy

Game screenshots, rendered shards, downloaded corpora, fonts and checkpoints stay local and are **never committed**. Screenshots are copyrighted game images. The repo holds code, manifests, small derived tables such as `vocab.txt`, and results. Model weights are published as release assets.

## Setup

```sh
./setup_env.sh
.venv/bin/python data/fetch_assets.py all
```

## Third-party data

Pretraining uses line crops cut from the Japanese subset of [OCR-Synthetic-Multilingual-v1](https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v1) by NVIDIA Corporation (Chesler, 2026), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), at revision `69696a1cc543ef3a0f8e9892a89c17293e915263`. The crops are modified: single horizontal text lines are cut from the pages, converted to grey and resized to height 32, and lines with characters outside `data/vocab.txt` are dropped (`nvsynth/`, `docs/nvsynth.md`). Neither the dataset nor the crops are redistributed here; `nvsynth/nv_fetch.py` rebuilds them from the Hub. Model weights trained on them carry the same attribution in their release notes.

The character-set analysis and the VN-ordered vocabularies (`docs/vocab-expand.md`, `data/vocab_vn_*.txt`) are derived from [Jiten](https://jiten.moe/frequency-dictionaries)'s word and kanji frequency lists, licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Only the pinned list manifest (`data/manifests/jiten_frequency_lists_20261014.json`: URLs + sha256), the derived vocab files and small result tables are committed; the Jiten CSVs stay local. CC BY-SA 4.0 is one-way compatible with this repository's GPL-3.0-or-later.

The vertical-text evaluation sets (`eval/manifests/vertical-*.jsonl`, `docs/vertical.md`) hold the transcribed text of columns of *ブラックジャックによろしく* (Give My Regards to Black Jack) by 佐藤秀峰 (Shuho Sato), vols. 1 and 13, whose author allows its free secondary use under the terms at [densho810.com/free](https://densho810.com/free/) (title and author credited as here); the page images are not redistributed, the manifests hold their hashes and view parameters only.

The screenshot manifests and game lists (`eval/manifests/`, `eval/linefind/manifests/`, `realdata/manifests/`, `teacher/manifests/`, `devnative/manifests/`, `realplus/manifests/`, `data/manifests/`) hold data from [VNDB](https://vndb.org): visual novel, release and screenshot ids, titles, relations, screenshot URLs and their content flags, and, in the exclusion lists (`data/manifests/synth3_exclude_*.json`), character names of the protected titles. VNDB makes its database available under the [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1.0/) and its contents under the [Database Contents License 1.0](https://opendatacommons.org/licenses/dbcl/1.0/) ([vndb.org/d17](https://vndb.org/d17)); these manifests are offered under the ODbL 1.0 as well. From synth3 on, the synthetic generators also draw speaker names from a VNDB dump, which stays local (v1 and synth2 generate theirs). The screenshots belong to their publishers and are not redistributed: the manifests hold their URLs and sha256 only, with this project's boxes and labels. The fresh native textbox set is not redistributed either: its two files (`eval/fresh_native_source_provenance_20261012.json`, `eval/fresh_native_textbox_20261012.json`) hold the URLs and sha256 of its 157 screenshots, from press and blog sites (4gamer.net, Impress Game Watch, fc2 and Hatena blogs, e+), not VNDB, and no labels.

### What the released weights were trained on

The recognizers on this repository's releases, except v0.2.2-meiki.1 (MeikiOCR's own recognizer, converted; its licences ship with that release), were trained on:

- **Synthetic renders** (`synth/`): text drawn in fonts over scenes, many of them crops of VNDB screenshots, of training-side games only from synth2 on. v0.1.0-baseline was trained on these alone; its v1 renders excluded only the `is_test_vn` hash split, so their scene pool held 9 of vita-test's 74 screenshots (v16566, v19850) and 13 other screenshots of test-reserved games as unlabelled backgrounds (`docs/results.md`, "Known leak into v1"). The provenance snapshot of v0.2.2-fl10-general.1, v0.2.3-vocr2.1 and v0.2.4-fl10-head.1 used procedural scenes only. The renders of v1, synth2 and the fl10 research use OFL fonts and Aozora Bunko's public-domain texts through the [aozorabunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean) dataset (CC BY 4.0), v1 and synth2 also Japanese Wikipedia (CC BY-SA); the fl10 research's corrective kanji pairs come from the [similar-kanji](https://github.com/siikamiika/similar-kanji) graph (MIT, `data/manifests/licenses/similar-kanji-MIT.txt`), allowed and weighted by Jiten's kanji list, which also weights the kanji of the generated names (CC BY-SA 4.0; `data/manifests/fl10_kanji_policy_20261011.json`). From synth3 on (VOCR3 and later), the renders use the generators' `personal` profile: besides fonts and texts whose licences allow commercial use, it includes two font families free for non-commercial use only (JF Dot non-commercial, Zpix), non-commercial text pools (JDD and NUCC: CC BY-NC-ND 4.0; BSD: CC BY-NC-SA 4.0; Aratako's gpt-4o-mini roleplay: CC BY-NC-SA 4.0 and OpenAI's terms), text of unclear licence (OpenSubtitles, なりきり posts), and from synth4 on (VOCR5) two fonts the owner supplied (`docs/synth3.md`, `docs/synth4.md`). `data/manifests/synth3_text_sources.json` pins every text and name source with its licence, among them SyosetuNames (ODC-BY 1.0), Tatoeba (CC BY 2.0 FR) and JESC, Wikipedia, Wiktionary and Wikiquote (CC BY-SA). `synth4.py shards --profile open` renders with the commercial-use fonts and texts alone (its scenes are still VNDB screenshot crops); no released model has been trained that way.
- **NVIDIA's crops** above, from v0.2.0 on (directly, or through the released FL10 initializer).
- **Real PS Vita, PSP and scaled PC screenshots from VNDB**, labelled by agreement of local OCR engines (PP-OCRv5/v6, MeikiOCR, Chrome Screen AI as a local voter, manga-ocr in the first rounds; `realdata/README.md`, `docs/teacher.md`) and, in later pools, reviewed by Claude (`docs/realplus.md`), with a Codex second vote in realplus v3 and the native pool (`docs/native-pool.md`). v0.2.0's fine-tune also used PP-OCRv6 small's own readings of unlabelled lines where its lowest character confidence is at least 0.7 (`ps`), and its F20 and FL10 (the later research's initializer) frame-level distillation from PP-OCRv6 small (`docs/train2.md`); the transcriptions of the fl10 research's 22 native crops were checked by an AI assistant (`docs/hiragana.md`).
- **The owner's own console captures**: VOCR4-W20 (v0.2.5-vocr4.1) and VOCR5 (`docs/vocr4.md`, `docs/owner-pool.md`).
- **The owner's hooked PS Vita and PSP corpus** (private, copyrighted game data: native frames with each game's hooked text;
  `docs/vocr6-data.md`): VOCR6-N and VOCR6-L (v0.4.0-vocr6.1) and VOCR6-S, -N and -L (v0.4.1-vocr6.2) trained on the
  train, guard and hval pages of its 48 training titles and on the line finder's crops of those pages (console crops
  v2, `docs/console-crops.md`);
  VOCR6-V (v0.4.2-vertical6.1, `docs/vocr6v.md`) on its one vertical title, Yunohana SpRING! (all of its pages), and on
  the column finder's crops of them; model B's
  console-test candidates (v0.3.7-modelb-test.1, `docs/vocr6-psp.md`) trained on its PSP titles.
- **Sources of two private research experiments**: the `fresh_anime_s1` run of v0.2.2-fl10-training.1 trained on AnimeText crops ([JustANormalTinkerer/animetext-ocr](https://huggingface.co/datasets/JustANormalTinkerer/animetext-ocr), revision `9c70c6b5609c4378d46ac0df4927c4119ebfd83d`, CC BY-NC-SA 4.0; `docs/hiragana.md`), and both runs of v0.2.2-fl10-bitmap.1 trained on renders of the owner's own CLANNAD PS Vita bitmap fonts, read from the owner's game files (`docs/native-bitmap-fonts.md`, `docs/native-bitmap-training.md`).

Release v0.5.0 republishes these weights unchanged; [`docs/models.md`](docs/models.md) names each file's attribution file and licence.

The screenshots, crops, fonts and texts are training input only; none is redistributed. Google Lens appears as a comparison and never as a training-label source; its readings were among the unattributed candidates of the second and third reads of the CLANNAD test references (`docs/clannad-v1-v2-v3-lens.md`). The character LMs of v0.2.6-ctclm.1, v0.2.6-ctclm.2 and v0.3.7-lm6.2 list their sources and licences in their `ATTRIBUTION.md` (CC BY-SA 4.0); v0.3.7-lm6.2's 1 MB and 4 MB LMs also carry character n-gram counts of the owner's hooked console captures (`docs/vocr6-lm.md`).

## License

GPL-3.0-or-later, matching Rubi.
