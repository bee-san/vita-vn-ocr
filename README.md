# vita-vn-ocr

A tiny Japanese OCR made only for visual-novel dialogue on PS Vita and PSP. It is meant to run on the console itself, inside [vita-jp-overlay](https://github.com/bee-san/vita-jp-overlay), in as little memory as possible.

General OCR models are too big for this. The best general option, a trimmed int8 PP-OCRv5 mobile, still peaks at about 13 MiB (see `docs/research-findings.md`). VN dialogue is a much narrower problem: one or a few horizontal lines in a dialogue box, a handful of game fonts, and a fixed screen size (960×544 on Vita, 480×272 on PSP). A recognizer trained only for that case can be an order of magnitude smaller.

## Status

| | |
|---|---|
| Model | `train/model.py`: a CTC line recognizer with 0.76M parameters (stem plus depthwise-separable blocks, factorised 5,355-class head), grey input at height 32 |
| v1 baseline | Trained on synthetic dialogue boxes only. Synthetic validation: 0.34% CER, 95.6% of lines exact (`docs/v1-metrics.jsonl`). Real Vita screenshots (`eval/`, vita-test, 206 lines from 16 games): 23.75% CER, 52.4% exact, 81.8% CER on its 43 dark-on-light lines. Real PSP screenshots (psp-test, 245 lines from 14 games): 46.45% CER, 30.2% exact, 86.8% CER on its 110 dark-on-light lines (`docs/results.md`). |
| Runtime | `runtime/`: plain-C99 recognizer for Vita (NEON) and PSP (scalar), no malloc, 116-125 KB arena at any line width; int8 weights 0.86 MB (`runtime/README.md`) |
| Architecture sweep | `docs/arch-sweep.md`: width, depth, temporal context, head bottleneck, vocab, input height and int8 on the v1 recipe, 5,000-step runs (plus a 15,000-step width pair) scored on vita-test and psp-test. Recommendation for train2: width 2.0 (1.42M parameters, 1.56 MB int8, 229 KB arena, an estimated 0.59-0.74 s for a long Vita line on one 444 MHz A9 core); width 1.5 is the lighter fallback and v1's width 1.0 the PSP-lean variant. Post-training int8 is enough; no QAT needed |
| Vita | `vita/`: VitaSDK build of the runtime plus a test app (VPK). In the Vita3K emulator it reproduces the host bit for bit on 728 test cases (vita-test, psp-test, synth-val) in w8i, and on 52 of them in all three modes. Estimated 113-212 ms per 640-px line (w8i, one Cortex-A9 core at 444 MHz) from the Vita build's instruction mix (`vita/README.md`) |
| Hardware | Nothing has run on a PCH-1000 or a PSP yet; `vita/README.md` has the steps for a PCH-1000. The PSP build (`psp/`) gives the host runtime's exact output in PPSSPP, at 0.38 s emulated for the median psp-test line (`psp/README.md`) |

## Layout

```
setup_env.sh, requirements.*   Python env (uv; CPU torch)
data/fetch_assets.py           OFL fonts, Aozora/Wikipedia text, VNDB screenshot list
data/label_real.py             first real-line pseudo-labeller (PP-OCRv5 server + manga-ocr agreement)
realdata/                      real training lines from Vita, PSP and scaled PC screenshots (4-reader vote)
data/vocab.txt                 output character set
teacher/                       real-screenshot labels: MeikiOCR first, local owocr engines vote (docs/teacher.md)
synth/                         synthetic dialogue-box renderer and corpus builder
nvsynth/                       Japanese line crops from NVIDIA OCR-Synthetic-Multilingual-v1 for pretraining (docs/nvsynth.md)
train/                         model, CTC training, export.py (weights blob for the C runtime)
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

## License

GPL-3.0-or-later, matching vita-jp-overlay.
