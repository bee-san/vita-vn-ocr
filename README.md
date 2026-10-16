# vita-vn-ocr

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/charts/vocr-vs-lens-dark.svg">
  <img src="docs/charts/vocr-vs-lens-light.svg" width="100%" alt="Bar chart of dialogue character error rate, lower is better. CLANNAD, PS Vita, 2,052 lines: VOCR5 Small 1.38%, Normal 0.96%, Large 0.73%, Google Lens 1.53%. vita-test, PS Vita, 153 lines: 1.00%, 0.96%, 0.59%, Lens 3.72%. psp-test, PSP, 183 lines: 2.22%, 1.30%, 1.00%, Lens 7.52%; Lens's vita-test and psp-test figures include the furigana it reads. The default model is 1.2 MB of int8 weights and is built to run offline in the game's memory; Google Lens is a cloud service that needs an internet connection.">
</picture>

**VOCR5 reads visual-novel dialogue as well as Google Lens on three test sets, and it needs no internet: it is built to run on the PS Vita itself.**

A tiny Japanese OCR made only for visual-novel dialogue on PS Vita and PSP. It is meant to run on the console itself, inside [vita-jp-overlay](https://github.com/bee-san/vita-jp-overlay), in as little memory as possible.

General OCR models are too big for this. The best general option, a trimmed int8 PP-OCRv5 mobile, still peaks at about 13 MiB (see `docs/research-findings.md`). VN dialogue is a much narrower problem: one or a few horizontal lines in a dialogue box, a handful of game fonts, and a fixed screen size (960×544 on Vita, 480×272 on PSP). A recognizer trained only for that case can be an order of magnitude smaller.

## Models (release [v0.3.0](https://github.com/bee-san/vita-vn-ocr/releases/tag/v0.3.0))

| model | tier | int8 weights | arena (W8I) | dialogue CER: CLANNAD / vita-test / psp-test | est. time per 3-line text box, 333 / 444 MHz | model card |
|---|---|---:|---:|---|---|---|
| [`VOCR5_S_w8.vocr`](https://github.com/bee-san/vita-vn-ocr/releases/download/v0.3.0/VOCR5_S_w8.vocr) | Small | 819,792 B | 124,832 B | 1.38% / 1.00% / 2.22% | 0.63 / 0.50 s | [Small](https://github.com/bee-san/vita-vn-ocr/releases/download/v0.3.0/MODEL_CARD_VOCR5_S.md) |
| [`VOCR5_N_w8.vocr`](https://github.com/bee-san/vita-vn-ocr/releases/download/v0.3.0/VOCR5_N_w8.vocr) | Normal (the default) | 1,185,632 B | 182,944 B | 0.96% / 0.96% / 1.30% | 0.92 / 0.72 s | [Normal](https://github.com/bee-san/vita-vn-ocr/releases/download/v0.3.0/MODEL_CARD_VOCR5_N.md) |
| [`VOCR5_L_w8.vocr`](https://github.com/bee-san/vita-vn-ocr/releases/download/v0.3.0/VOCR5_L_w8.vocr) | Large | 2,473,696 B | 261,024 B | 0.73% / 0.59% / 1.00% | 2.08 / 1.63 s | [Large](https://github.com/bee-san/vita-vn-ocr/releases/download/v0.3.0/MODEL_CARD_VOCR5_L.md) |
| Google Lens | a cloud service | | | 1.53% / 3.72%\* / 7.52%\* | needs an internet connection | |

They are made for [vita-jp-overlay](https://github.com/bee-san/vita-jp-overlay)'s three model sizes (its VOCR5 update: bee-san/vita-jp-overlay#167): choose Small, Normal (the default) or Large in its Overlay app; when a game has too little free memory for the chosen size, a job runs the next smaller one. The times are estimates for one Cortex-A9 core with the runtime the overlay uses (#85), without prefetch credit, calibrated to VOCR3-H15's measured console rate; **VOCR5 has not run on a console yet**. The CER figures are measured on the host through the plain-C W8I runtime with greedy decoding; the overlay's beam search with the 1 MB character LM, which ships in the same release, reads vita-test and psp-test dialogue at 1.15% / 0.72% / 0.53% (Small / Normal / Large). Details: `docs/vocr5.md`, the model cards and the release notes.

How to read the chart:
- **Dialogue** means lines without a speaker-name style, the metric the overlay's users care about (they crop name plates out). Speaker names go either way: Lens is far better on CLANNAD's (0.68% against 4.66-12.45%) and vita-test's (0.00% against 0.58-2.33%), far worse on psp-test's (25.84% against 3.83-6.70%).
- **CLANNAD** (2,052 dialogue lines of a PS Vita playthrough in Vita3K; references read twice by Claude and frozen before any score): every VOCR5 model is below Lens's 1.53%, but Lens's interval is wide (0.90 to 2.25): its errors cluster on a few lines, and it reads more lines exactly (96.5% against 89.9% for Large). Over all 3,048 lines, names included, Large minus Lens is −0.50 points (paired 95% interval −1.15 to +0.09), so on CLANNAD the two are level within the noise. Lens's readings were among the unattributed candidates of the references' second and third reads.
- **vita-test and psp-test** (153 and 183 dialogue lines of 16 and 14 games, from VNDB screenshots; every reader got the same line crops): every VOCR5 model is below Lens, with paired 95% intervals below zero over lines and over games. \* Part of Lens's error is the furigana above a line, which Lens transcribes and the references leave out: 116 of its 227 psp-test errors (name lines included) fall on the 20 lines with furigana, 30 of 89 on vita-test's 3.
- **Disclosure:** every VOCR5 model trained on the owner's console captures, which include planetarian's PSP port, a Key / Prototype release like CLANNAD's Vita port.
- **Lens as the reference** (its reading counted as 100% right): VOCR5 Small / Normal / Large differ from Lens's text in 2.63% / 2.24% / 2.07% of the characters of CLANNAD's dialogue lines ([#90](https://github.com/bee-san/vita-vn-ocr/issues/90)).

The chart is drawn by `docs/charts/vocr_vs_lens.py` from `docs/charts/vocr-vs-lens.json`, whose every number `tests/test_vocr_vs_lens_chart.py` recomputes from `eval/vocr5_dialogue_20261016.json` and `eval/vocr5_clannad_20261016.json`.

## Status

The table records each component and each earlier round. The VOCR5 models above replace the earlier rounds' recommended models.

| | |
|---|---|
| Model | `train/model.py`: a CTC line recognizer (stem plus depthwise-separable blocks, factorised head), grey input at height 32. VOCR5 (v0.3.0) ships three sizes: Small (width 1.0, 0.73M parameters, 4,987 classes), Normal, the default (width 1.5, 1.07M, 5,631 classes) and Large (late-wide, width 2.0 early and 3.0 late, 2.32M, 5,631 classes). v0.2.0's H15 was width 1.5, 1.04M parameters, 5,355 classes |
| **v0.3.0 release (VOCR5, the final round)** | Three int8 recognizers, one per tier of vita-jp-overlay's model sizes: **Small** `VOCR5_S_w8.vocr` (819,792 B, FL10's topology, 4,987 classes), **Normal** (the default) `VOCR5_N_w8.vocr` (1,185,632 B, H15's topology, 5,631 classes) and **Large** `VOCR5_L_w8.vocr` (2,473,696 B, late-wide, 5,631 classes). Each replaced its VOCR4 predecessor under a pre-registered rule on development data. vita-test + psp-test dialogue CER (fixed lines, console polarity): **1.60% / 1.13% / 0.79%** (0.53% for Large with the overlay's beam + 1 MB LM); whole dialogue 8.31% / 7.98% / 7.08%; CLANNAD dialogue 1.38% / 0.96% / **0.73%** (Google Lens 1.53%). Estimated 0.63 / 0.92 / 2.08 s per 3-line box at 333 MHz on one A9 core without prefetch credit (#85's runtime; estimates, nothing of VOCR5 has run on a console). `docs/vocr5.md` |
| v0.2.0 release (train2) | Pretrained on NVIDIA OCR-Synthetic-Multilingual-v1 Japanese crops (nvsynth) mixed with VN-style synthetic boxes (synth2), fine-tuned on teacher-labelled real crops. **H15** (recommended, width 1.5): real Vita screenshots (`eval/`, vita-test) **1.56% CER, 83.5% exact**; real PSP (psp-test) **2.63% CER, 82.4% exact**, int8 through the C runtime. That beats trimmed int8 PP-OCRv5 mobile (2.30% / 5.34%) on both sets, at 1.16 MB of int8 weights and a 183 KB arena. Not every style is at or below that model's yet: H15 is above it on names, coloured text, PSP furigana and text without a box (4-62 lines each, mostly within noise). **F20** (width 2.0): 1.79% / 2.79%. **FL10** (PSP-lean, width 1.0, 3,500-char vocab, 658 KB): 2.30% / 5.18%, so it ties the bar on vita-test. An estimated 166-302 ms per 640-px line for H15 (w8i, one 444 MHz A9 core; about 220-400 ms at the 333 MHz games run at by default). H15 has not run on a console; FL10 has (see Hardware). Distillation from PP-OCRv6 small was tried and did not help. Ablation, per-style tables and the selection rule are in `docs/train2.md` |
| v1 baseline | Trained on synthetic dialogue boxes only. Synthetic validation: 0.34% CER, 95.6% of lines exact (`docs/v1-metrics.jsonl`). Real Vita screenshots (`eval/`, vita-test, 206 lines from 16 games): 23.75% CER, 52.4% exact, 81.8% CER on its 43 dark-on-light lines. Real PSP screenshots (psp-test, 245 lines from 14 games): 46.45% CER, 30.2% exact, 86.8% CER on its 110 dark-on-light lines (`docs/results.md`). |
| Runtime | `runtime/`: plain-C99 recognizer for Vita (NEON) and PSP (scalar), no malloc, 116-125 KB arena at any line width; int8 weights 0.86 MB (`runtime/README.md`) |
| Architecture sweep | `docs/arch-sweep.md`: width, depth, temporal context, head bottleneck, vocab, input height and int8 on the v1 recipe, 5,000-step runs (plus a 15,000-step width pair) scored on vita-test and psp-test. Recommendation for train2: width 2.0 (1.42M parameters, 1.56 MB int8, 241 KB W8I arena; the later cycle model estimates 234-411 ms per 640-px line on one 444 MHz A9 core, against the sweep's own MAC-rate estimate of 0.59-0.74 s); width 1.5 is the lighter fallback and v1's width 1.0 the PSP-lean variant. Post-training int8 is enough; no QAT needed |
| Vita | `vita/`: VitaSDK build of the runtime plus a test app (VPK). In the Vita3K emulator it reproduces the host bit for bit on 728 test cases (vita-test, psp-test, synth-val) in w8i, and on 52 of them in all three modes. Estimated 113-212 ms per 640-px line (w8i, one Cortex-A9 core at 444 MHz) from the Vita build's instruction mix (`vita/README.md`) |
| Console budget | `docs/vita-memory-budget.md`: three tiers for a recognizer in the game's own process. Always safe: int8 weights ≤ 900 KB, W8I arena ≤ 128 KiB, an estimated ≤ 300 ms per 640-px line on one 333 MHz A9 core (FL10's class; at most about 1.5 MiB of the game's memory at a job's peak). Safe with the overlay's free-memory preflight and an always-safe fallback: weights ≤ 1.6 MB, arena ≤ 256 KiB, ≤ 550 ms (H15, F20). Never: beyond that, and nothing of the recognizer in SceShell at any size (its 50 MiB partition runs within tens of KiB of empty beside a game). Inside a game, memory is not the limit: a ceiling job needs about 2.0-2.4 MB against 56-87 MiB seen free, and the preflight refuses what does not fit. Speed and the shared 2 MiB L2 bind first; the game's share of the L2 and the cost on the system core are unmeasured. It also lists why vita-jp-overlay's OCR attempts failed: never the recognizer's size; the blocking failures were integration bugs and the Shell plugin's own footprint |
| Platform memory | `docs/vita-platform-memory.md`: which PS Vita memory pool the recognizer runs in (the game's 256 MiB budget, never SceShell's 50 MiB), what that means for model size, and the corrections to the budget doc. The full cited reference is vita-jp-overlay's `docs/vita-memory.md` |
| Hardware | FL10 has run on a PCH-1001 (firmware 3.65) inside vita-jp-overlay, in CLANNAD: model load 213-359 ms and recognition 36-145 ms per job over 14 logged jobs of varying dialogue, at an unknown clock, with no hardware CER (`notes/26`). Neither H15 nor `vita/`'s test app has run on a console, and nothing has run on a PSP; `vita/README.md` has the steps for a PCH-1000. The PSP build (`psp/`) gives the host runtime's exact output in PPSSPP, at 0.38 s emulated for the median psp-test line (`psp/README.md`) |

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

The screenshot manifests and game lists (`eval/manifests/`, `eval/linefind/manifests/`, `realdata/manifests/`, `teacher/manifests/`, `devnative/manifests/`, `realplus/manifests/`, `data/manifests/`) hold data from [VNDB](https://vndb.org): visual novel, release and screenshot ids, titles, relations, screenshot URLs and their content flags, and, in the exclusion lists (`data/manifests/synth3_exclude_*.json`), character names of the protected titles. VNDB makes its database available under the [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1.0/) and its contents under the [Database Contents License 1.0](https://opendatacommons.org/licenses/dbcl/1.0/) ([vndb.org/d17](https://vndb.org/d17)); these manifests are offered under the ODbL 1.0 as well. From synth3 on, the synthetic generators also draw speaker names from a VNDB dump, which stays local (v1 and synth2 generate theirs). The screenshots belong to their publishers and are not redistributed: the manifests hold their URLs and sha256 only, with this project's boxes and labels. The fresh native textbox set is not redistributed either: its two files (`eval/fresh_native_source_provenance_20261012.json`, `eval/fresh_native_textbox_20261012.json`) hold the URLs and sha256 of its 157 screenshots, from press and blog sites (4gamer.net, Impress Game Watch, fc2 and Hatena blogs, e+), not VNDB, and no labels.

### What the released weights were trained on

The recognizers on this repository's releases, except v0.2.2-meiki.1 (MeikiOCR's own recognizer, converted; its licences ship with that release), were trained on:

- **Synthetic renders** (`synth/`): text drawn in fonts over scenes, many of them crops of VNDB screenshots, of training-side games only from synth2 on. v0.1.0-baseline was trained on these alone; its v1 renders excluded only the `is_test_vn` hash split, so their scene pool held 9 of vita-test's 74 screenshots (v16566, v19850) and 13 other screenshots of games reserved today as unlabelled backgrounds (`docs/results.md`, "Known leak into v1"). The provenance snapshot of v0.2.2-fl10-general.1, v0.2.3-vocr2.1 and v0.2.4-fl10-head.1 used procedural scenes only. The renders of v1, synth2 and the fl10 research use OFL fonts and Aozora Bunko's public-domain texts through the [aozorabunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean) dataset (CC BY 4.0), v1 and synth2 also Japanese Wikipedia (CC BY-SA); the fl10 research's corrective kanji pairs come from the [similar-kanji](https://github.com/siikamiika/similar-kanji) graph (MIT, `data/manifests/licenses/similar-kanji-MIT.txt`), allowed and weighted by Jiten's kanji list, which also weights the kanji of the generated names (CC BY-SA 4.0; `data/manifests/fl10_kanji_policy_20261011.json`). From synth3 on (VOCR3 and later), the renders use the generators' `personal` profile: besides fonts and texts whose licences allow commercial use, it includes two font families free for non-commercial use only (JF Dot non-commercial, Zpix), non-commercial text pools (JDD and NUCC: CC BY-NC-ND 4.0; BSD: CC BY-NC-SA 4.0; Aratako's gpt-4o-mini roleplay: CC BY-NC-SA 4.0 and OpenAI's terms), text of unclear licence (OpenSubtitles, なりきり posts), and from synth4 on (VOCR5) two fonts the owner supplied (`docs/synth3.md`, `docs/synth4.md`). `data/manifests/synth3_text_sources.json` pins every text and name source with its licence, among them SyosetuNames (ODC-BY 1.0), Tatoeba (CC BY 2.0 FR) and JESC, Wikipedia, Wiktionary and Wikiquote (CC BY-SA). `synth4.py shards --profile open` renders with the commercial-use fonts and texts alone (its scenes are still VNDB screenshot crops); no released model has been trained that way.
- **NVIDIA's crops** above, from v0.2.0 on (directly, or through the released FL10 initializer).
- **Real PS Vita, PSP and scaled PC screenshots from VNDB**, labelled by agreement of local OCR engines (PP-OCRv5/v6, MeikiOCR, Chrome Screen AI as a local voter, manga-ocr in the first rounds; `realdata/README.md`, `docs/teacher.md`) and, in later pools, reviewed by Claude (`docs/realplus.md`), with a Codex second vote in realplus v3 and the native pool (`docs/native-pool.md`). v0.2.0's fine-tune also used PP-OCRv6 small's own readings of unlabelled lines where its lowest character confidence is at least 0.7 (`ps`), and its F20 and FL10 (the later research's initializer) frame-level distillation from PP-OCRv6 small (`docs/train2.md`); the transcriptions of the fl10 research's 22 native crops were checked by an AI assistant (`docs/hiragana.md`).
- **The owner's own console captures**: VOCR4-W20 (v0.2.5-vocr4.1) and VOCR5 (`docs/vocr4.md`, `docs/owner-pool.md`).
- **Sources of two private research experiments**: the `fresh_anime_s1` run of v0.2.2-fl10-training.1 trained on AnimeText crops ([JustANormalTinkerer/animetext-ocr](https://huggingface.co/datasets/JustANormalTinkerer/animetext-ocr), revision `9c70c6b5609c4378d46ac0df4927c4119ebfd83d`, CC BY-NC-SA 4.0; `docs/hiragana.md`), and both runs of v0.2.2-fl10-bitmap.1 trained on renders of the owner's own CLANNAD PS Vita bitmap fonts, read from the owner's game files (`docs/native-bitmap-fonts.md`, `docs/native-bitmap-training.md`).

The screenshots, crops, fonts and texts are training input only; none is redistributed. Google Lens appears as a comparison and never as a training-label source; its readings were among the unattributed candidates of the second and third reads of the CLANNAD test references (`docs/clannad-v1-v2-v3-lens.md`). The character LMs of v0.2.6-ctclm.1 and v0.2.6-ctclm.2 list their sources and licences in their `ATTRIBUTION.md` (CC BY-SA 4.0).

## License

GPL-3.0-or-later, matching vita-jp-overlay.
