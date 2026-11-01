# Changelog

vita-vn-ocr's releases, newest first, in the style of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Every model is published as release assets; the pre-releases between the normal releases are listed under the
release that collects them. Numbers are measured on the host unless marked as estimates; nothing of VOCR6 or the
spotter has run on a console.

## [Unreleased]

### Changed

- **A faster runtime on the Vita, bit for bit** (#165, `docs/c-speed.md`; thanks to astra, whose ONNX report started
  it). NEON kernels only: the recognizer's matrix pass in `asm` with aligned loads, its 3×3 stem and its top-K filter;
  textspot8's 1×1 layers 4 output channels at a time, its depthwise layers and stem, and a cheaper exact
  requantisation. Est. per 3-line box at 333 MHz, one core, no prefetch credit: VOCR6.2 S / N / L 0.627 → 0.585,
  0.774 → 0.748, 1.880 → 1.822 s; textspot8 81.9 → 61.0 ms per 900×150 region, 264.3 → 192.2 ms per frame (estimates;
  nothing has run on a console). The same logits, texts, top-K, maps and boxes in every build and in Vita3K; arenas
  unchanged; builds without NEON compile to the same machine code.

### Added

- **ONNX files for desktop apps** (pre-release v0.5.1-onnx.2, #163, [`docs/onnx.md`](docs/onnx.md)): VOCR6.2 Small,
  Normal and Large and the textspot8 spotter converted from v0.5.0's int8 blobs by `tools/export_onnx.py`, unchanged
  weights, the blobs' non-commercial terms.
  - **exact**: the C runtime's W8I arithmetic in every float32 step and order, with the blobs' head gate and two-level
    head; on 7,745 development line reads per tier and 1,710 development frames and regions every logit, text,
    confidence, character cell and spotter box equals the C runtime's (`eval/onnx_parity_20261031.json`), also at 1,
    5 and 8 threads and every graph optimisation level (`eval/onnx_checks_20261101.json`).
  - **fast**: ONNX Runtime's own float32 kernels (W8F's classes on every development frame; 17 of 792,654 frames and
    2 line texts differ from W8I).
  - `tools/vocr_onnx.py`: the reference pipeline, NumPy and ONNX Runtime only (spotter, recognizer, the beam + LM
    decoder, character cells).
  - The method follows a report by an outside researcher, astra (int16 digits through two `MatMulInteger` calls,
    int32 reconstruction, float64 requantisation in the spotter); the C-order convolutions and the gated head are this
    export's. Host times against `runtime/` at `-O2` and `-O3 -march=native`: `docs/onnx.md`, "Speed".
- `vita/par`, `vita/vita3k_parity.py`, `vita/scripts/run_vita3k_par.sh`: the runtime's host drivers as a Vita app, run
  in Vita3K and compared with the host byte for byte (Rubi's head options and the spotter included).
  `vita/ts_speed_parity.py`: the spotter's five-build check of two runtime versions. `vita/speed_parity.py
  --no-logits` and `vita/a9_layers.py --vocr-cflags`.

## [0.5.0] - 2026-10-28

**VOCR6: recognizers, dialogue-line spotter, LMs and every model so far.** One normal release with every model from
VOCR3 on (ctclm.1's superseded LMs aside) and VOCR2's trial blob, byte for byte as its own release published it, and
one `SHA256SUMS` (86 files from 20 releases, plus `SHA256SUMS` and `v0.5.0-assets.json`).
[`docs/models.md`](docs/models.md) lists each file's size, sha256, role, tier, measured numbers, status and licence.

### Added

- **VOCR6-S, VOCR6-N and VOCR6-L** (first published as pre-release v0.4.1-vocr6.2, #125, #157): Rubi's Small, Normal
  (the default) and Large model sizes since Rubi 0.10.0. Each is its VOCR5 tier fine-tuned again on the owner's hooked
  PS Vita and PSP games and the line finder's crops of their pages (`docs/vocr6.md`).
  - Hooked games' held-out pages, Rubi's shipped path, the mean of three seeds of each tier's recipe (stage 4d's for
    Small; stage 4a's 15,000-step recipe for Normal and Large): 1.177 → 1.049%, 1.008 → 0.936%, 0.990 → 0.917%
    (VOCR5 → VOCR6, each resolved better). The shipped blobs then trained on these pages too, Normal and Large three
    times as long.
  - Other games, VOCR5's development score for the set as it ships (Rubi's cascade S → N → L): 1.674 → 1.549,
    1.482 → 1.389, 1.439 → 1.396 (not resolved). Small's drop comes from the better Normal above it: VOCR6-S against
    VOCR5-S in the same cascade is +0.041 [−0.016, +0.099], worse in point (not resolved; its recipe was the third
    tried at Small).
  - Read once: CLANNAD's dialogue lines 0.73 / 0.66 / 0.60% → 0.47 / 0.33 / 0.31% on Rubi's path (resolved better;
    two training games draw dialogue in CLANNAD's glyph raster, so not an unseen-game result). vita-test + psp-test
    level, psp-test not better.
  - Est. 0.63 / 0.77 / 1.88 s per 3-line box on one 333 MHz Cortex-A9 core (estimates).
- **The textspot8 dialogue-line spotter** (pre-release v0.3.10-textspot8.1, #149, #156): Rubi's default line finder
  since Rubi 0.10.0, 52,176 B of int8 weights, est. 82 ms per 900×150 region. Unseen development games 1.03% page-path
  dialogue CER against Rubi's previous finder's 2.38% (`docs/textspot8.md`).
- **The game-text LMs** (pre-release v0.3.7-lm6.2, #127, #137): Rubi's 1 MB (default) and 4 MB character LM tiers,
  beside ctclm.2's 0.5 MB and 2 MB (`docs/vocr6-lm.md`).
- **Every earlier model of these rounds**, marked in `docs/models.md` as used by Rubi, previous or research / not
  selected: v0.4.0-vocr6.1's VOCR6-N and -L, textspot7, textspot6, textspot3 and textspot, ctclm.2's replaced 1 MB
  and 4 MB LMs, VOCR5 with its head gate and two-level head (v0.3.5, v0.3.6) and plain (v0.3.0), VOCR5-V, the
  text-presence gate, the text detector, model B's two PSP candidates, VOCR4, VOCR3 and VOCR2's trial blob; every
  attribution file of those releases.
- **VOCR6 against other OCR engines** (report only; `eval/other_readers_protocol.json`, registered before the read,
  `eval/other_readers_20261028.json`): MeikiOCR, PP-OCRv6 small, trimmed int8 PP-OCRv5 mobile and manga-ocr read the
  same lines as VOCR on CLANNAD, vita-test, psp-test and the hooked test title; Google Lens's existing readings.
  VOCR6 Large has the lowest dialogue CER on all four sets; every VOCR6 tier reads CLANNAD, vita-test and psp-test
  better than Lens (intervals below 0). The README's chart and table show it (`docs/charts/vocr_vs_lens.py`).

### Not included

- The FL10 research snapshots before VOCR3 (v0.2.2-fl10-training.1, v0.2.2-fl10-bitmap.1, v0.2.2-fl10-general.1,
  v0.2.4-fl10-head.1: 639,150,581 B together), VOCR2's training checkpoint, MeikiOCR's converted recognizer
  (v0.2.2-meiki.1, a third-party model), v0.3.7-lm6.1 and v0.2.6-ctclm.1 (superseded) and model B's console test zip
  stay on their own pre-releases; v0.1.0-baseline, v0.2.0 and v0.2.1 are normal releases already.

## Earlier releases

- **[0.3.0] - 2026-10-16:** VOCR5 Small / Normal / Large (`docs/vocr5.md`). Pre-releases after it, now collected in
  0.5.0: v0.3.1-textdet.1, v0.3.2-textgate.1, v0.3.2-vertical.1, v0.3.3-textspot.1, v0.3.4-textspot3.1,
  v0.3.5-headgate.1, v0.3.6-head2level.1, v0.3.7-lm6.1, v0.3.7-lm6.2, v0.3.7-modelb-test.1, v0.3.8-textspot6.1,
  v0.3.9-textspot7.1, v0.3.10-textspot8.1, v0.4.0-vocr6.1 and v0.4.1-vocr6.2.
- **[0.2.1] - 2026-10-11:** v0.2.0's models with the merged Vita ports. Pre-releases after it: v0.2.2-meiki.1,
  v0.2.2-fl10-training.1, v0.2.2-fl10-bitmap.1, v0.2.2-fl10-general.1, v0.2.4-fl10-head.1 (research), v0.2.3-vocr2.1
  (VOCR2), v0.2.5-vocr3.1 (VOCR3), v0.2.5-vocr4.1 (VOCR4), v0.2.6-ctclm.1 and v0.2.6-ctclm.2 (the first LMs); 0.5.0
  collects VOCR2's blob and everything from VOCR3 on.
- **[0.2.0] - 2026-10-10:** the first real-screenshot models, FL10, H15 and F20 (`docs/train2.md`).
- **[0.1.0-baseline] - 2026-10-09:** the v1 checkpoint, trained on synthetic dialogue boxes only.

[0.5.0]: https://github.com/bee-san/vita-vn-ocr/releases/tag/v0.5.0
[0.3.0]: https://github.com/bee-san/vita-vn-ocr/releases/tag/v0.3.0
[0.2.1]: https://github.com/bee-san/vita-vn-ocr/releases/tag/v0.2.1
[0.2.0]: https://github.com/bee-san/vita-vn-ocr/releases/tag/v0.2.0
[0.1.0-baseline]: https://github.com/bee-san/vita-vn-ocr/releases/tag/v0.1.0-baseline
