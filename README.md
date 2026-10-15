# vita-vn-ocr

A tiny Japanese OCR made only for visual-novel dialogue on PS Vita and PSP. It is meant to run on the console itself, inside [vita-jp-overlay](https://github.com/bee-san/vita-jp-overlay), in as little memory as possible.

General OCR models are too big for this. The best general option, a trimmed int8 PP-OCRv5 mobile, still peaks at about 13 MiB (see `docs/research-findings.md`). VN dialogue is a much narrower problem: one or a few horizontal lines in a dialogue box, a handful of game fonts, and a fixed screen size (960×544 on Vita, 480×272 on PSP). A recognizer trained only for that case can be an order of magnitude smaller.

## Status

| | |
|---|---|
| Model | `train/model.py`: a CTC line recognizer (stem plus depthwise-separable blocks, factorised head), grey input at height 32. The recommended v0.2.0 model (H15) is width 1.5, 1.04M parameters, 5,355 classes |
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

The screenshot manifests and game lists (`eval/manifests/`, `eval/linefind/manifests/`, `realdata/manifests/`, `teacher/manifests/`, `devnative/manifests/`, `realplus/manifests/`, `data/manifests/`) hold data from [VNDB](https://vndb.org): visual novel, release and screenshot ids, titles, relations, screenshot URLs and their content flags, and, in the exclusion lists (`data/manifests/synth3_exclude_*.json`), character names of the protected titles. VNDB makes its database available under the [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1.0/) and its contents under the [Database Contents License 1.0](https://opendatacommons.org/licenses/dbcl/1.0/) ([vndb.org/d17](https://vndb.org/d17)); these manifests are offered under the ODbL 1.0 as well. The synthetic generators also draw speaker names from a VNDB dump, which stays local. The screenshots belong to their publishers and are not redistributed: the manifests hold their URLs and sha256 only, with this project's boxes and labels.

### What the released weights were trained on

The recognizers on this repository's releases, except v0.2.2-meiki.1 (MeikiOCR's own recognizer, converted; its licences ship with that release), were trained on:

- **Synthetic renders** (`synth/`): text drawn in fonts over scenes, many of them crops of training-side VNDB screenshots. v0.1.0-baseline was trained on these alone. The renders of v1 and synth2 use OFL fonts and Aozora/Wikipedia text. From synth3 on (VOCR3 and later), the renders use the generators' `personal` profile: besides fonts and texts whose licences allow commercial use, it includes two font families free for non-commercial use only (JF Dot non-commercial, Zpix), non-commercial text pools (JDD and NUCC: CC BY-NC-ND 4.0; BSD: CC BY-NC-SA 4.0; Aratako's gpt-4o-mini roleplay: CC BY-NC-SA 4.0 and OpenAI's terms), text of unclear licence (OpenSubtitles, なりきり posts), and from synth4 on (VOCR5) two fonts the owner supplied (`docs/synth3.md`, `docs/synth4.md`). `data/manifests/synth3_text_sources.json` pins every text and name source with its licence, among them SyosetuNames (ODC-BY 1.0), Tatoeba (CC BY 2.0 FR) and JESC, Wikipedia, Wiktionary and Wikiquote (CC BY-SA). `synth4.py shards --profile open` renders with the commercial-use fonts and texts alone (its scenes are still VNDB screenshot crops); no released model has been trained that way.
- **NVIDIA's crops** above, from v0.2.0 on (directly, or through the released FL10 initializer).
- **Real PS Vita, PSP and scaled PC screenshots from VNDB**, labelled by agreement of local OCR engines (PP-OCRv5/v6, MeikiOCR, Chrome Screen AI as a local voter, manga-ocr in the first rounds; `data/label_real.py`, `docs/teacher.md`) and, in later pools, reviewed by Claude (`docs/realplus.md`), with a Codex second vote in realplus v3 and the native pool (`docs/native-pool.md`).
- **The owner's own console captures**: VOCR4-W20 (v0.2.5-vocr4.1) and VOCR5 (`docs/vocr4.md`, `docs/owner-pool.md`).
- **Sources of two private research experiments**: the `fresh_anime_s1` run of v0.2.2-fl10-training.1 trained on AnimeText crops ([JustANormalTinkerer/animetext-ocr](https://huggingface.co/datasets/JustANormalTinkerer/animetext-ocr), revision `9c70c6b5609c4378d46ac0df4927c4119ebfd83d`, CC BY-NC-SA 4.0; `docs/hiragana.md`), and both runs of v0.2.2-fl10-bitmap.1 trained on renders of the owner's own CLANNAD PS Vita bitmap fonts, read from the owner's game files (`docs/native-bitmap-fonts.md`, `docs/native-bitmap-training.md`).

The screenshots, crops, fonts and texts are training input only; none is redistributed. Google Lens appears as a comparison and never as a training-label source; its readings were among the unattributed candidates of the second and third reads of the CLANNAD test references (`docs/clannad-v1-v2-v3-lens.md`). The character LMs of v0.2.6-ctclm.1 and v0.2.6-ctclm.2 list their sources and licences in their `ATTRIBUTION.md` (CC BY-SA 4.0).

## License

GPL-3.0-or-later, matching vita-jp-overlay.
