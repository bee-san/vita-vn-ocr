# Private native bitmap font rendering

The OFL renderer did not correct CLANNAD's development hiragana errors in the
first six training trials. An optional bitmap adapter can now render the user's
own local CLANNAD Vita fonts alongside the existing OFL fonts. Console runtime
code and model topology are unchanged. This is a dataset experiment, with no
measured recognition improvement yet.

`data/manifests/clannad_native_fonts_20261011.json` pins four FNT/PRO pairs from
the local `PCSG00415` game. It records the game's official product URL, filenames,
sizes and SHA256 hashes. The loader never downloads or redistributes font bytes.
Fonts, generated crops, corpora and screenshots stay in ignored local folders.

The observed FNT files contain a 16-byte header, 7,120 sorted CP932 codes and two
sequential banks of 32×32, high-nibble-first 4bpp cells. PRO files contain 8,064
three-byte metric records indexed by a single-byte/JIS-grid mapping. These are
inferences from measured bytes and visual inspection, not an authoritative
format specification. The banks are called `strike0` and `strike1`; their
original purpose is unknown. The adapter interprets the PRO bytes as signed
x/y bearings and unsigned advances. Japanese kana and kanji usually have the
observed tuple `(0, 0, 32)`.

The adapter exposes only canonical, round-tripping CP932 Unicode characters
with positive advances. Each actual font covers 3,491/3,500 released vocabulary
characters; unsupported characters are `♡ ♥ ゔ ゕ ゖ ヷ ヸ ヹ ヺ`. A region containing
an unsupported glyph uses one covered OFL font for the entire region. It never
substitutes a different Unicode label or renders a missing-glyph box. Ruby
distractor readings also respect the selected font's coverage.

Use the existing hash-verified corpus, reviewed scenery and bounded Jiten
policy with an explicit source directory and probability:

```sh
.venv-ci/bin/python synth/fresh_dataset.py render \
  --root build/hiragana/native-bitmap-dataset --split train \
  --vocab build/hiragana/vocab.txt \
  --kanji-policy data/manifests/fl10_kanji_policy_20261011.json \
  --background-manifest data/manifests/fresh_jiten_backgrounds.json \
  --native-font-manifest data/manifests/clannad_native_fonts_20261011.json \
  --native-font-root /path/to/your/PCSG00415 \
  --native-font-probability 1 \
  --finder grad --invert-dark --crop-edge-jitter \
  --seed 20261021 --workers 2 \
  --out build/hiragana/native-bitmap-dataset/general --shards 8 --per-shard 500
```

The probability applies once per eligible rendered region; coverage fallbacks
and the line finder affect the final crop mixture. Metadata records actual font
use, family, bank, source hashes and requested probability. The dataset manifest
checks those bindings and reports measured native/OFL and bank counts. The
default probability is zero and preserves ordinary OFL selection and RNG use.

Bitmap regions render at 24/28/32 px on Vita (20/30/50 percent) and 12/14/16 px
on PSP (10/30/60 percent), with equal bank probabilities. Resizing uses Pillow's
pinned Lanczos implementation; native 32 px cells preserve their source pixels.
PSP examples are downsampled Vita font variants, not extracted PSP fonts. Existing
effects, console polarity, reviewed text-free scenery and crop-edge jitter apply.
Main and name baselines use the bitmap cell bottom, keeping top-row strokes
inside the declared line band; ruby uses the corresponding cell-bottom anchor.
Each native line's band also includes its measured glyph bounds, preserving
negative bearings and Latin descenders during crop matching and edge jitter.
Per-character core bounds use the same fractional Pillow origins and alpha
threshold 128 as the rendered ink mask. A native crop that cuts labelled core
strokes is rejected; proposed jitter that would do so reverts to the original
finder crop. Ruby, cursors and typewriter ghosts are excluded from this check.
The ordinary OFL baseline rule is preserved.

Validation uses generated binary fixtures, not extracted proprietary glyphs:

```sh
.venv-ci/bin/python -m pytest -q tests/test_native_bitmap_font.py \
  tests/test_native_font_manifest.py tests/test_fresh_dataset.py tests/test_render.py
.venv-ci/bin/python -m ruff check .
```

All four local fonts also passed 32 Pillow baseline/stroke checks at sizes
14, 20.5, 32 and 64 across both banks; painted bounds matched reported bounds.
The final 64-crop native-font smoke run produced 63 native and one covered OFL
fallback crop from 54 regions, with zero rendering exceptions. An independent
per-glyph replay exactly reproduced stored pixels and labels, with no clipped
native core strokes, unsupported glyphs or CTC-infeasible crops. It included
31 changed edge-jitter crops and four ruby cases. Images were visually reviewed
locally and remain ignored. This validates rendering, not OCR accuracy.

Train and development books and scenery games remain separate. Do not train on
development transcriptions or select checkpoints using the shared benchmark.
A deployment candidate must improve real development hiragana with full-CER and
synthetic guards in the C int8 runtime, then pass the unchanged held-out gate.
