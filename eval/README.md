# eval: the real-VN benchmark

One command scores a recognizer (and optionally a line finder) on hand-checked
dialogue lines from real PS Vita and PSP screenshots, plus the held-out synthetic
shard:

```sh
eval/fetch.py                                    # once: rebuild the screenshots from VNDB (sha256-checked)
eval/run.py --model models/v1/ckpt_best.pt       # every set: vita-test, psp-test, synth-val
eval/run.py --model X --set vita-test --errors 20 --out results/x.json
```

Run it through `$HERD/heavy.sh` on the shared box. Measured on all three sets in
one 5-thread slot: v1 27 s, the PP-OCRv5 ONNX presets about 80 s, the ncnn
models 3–4.5 min (one thread), manga-ocr 17 min.

On the shared box the screenshots are already in `$HERD/shared/artifacts/eval`:
point at them with `--data`, `export VOCR_EVAL_DATA=$HERD/shared/artifacts/eval`
or a `data/eval` symlink instead of fetching. synth-val reads
`data/synth/synth_00039.npz` (override with `VOCR_SYNTH_VAL`).

## What to pass as `--model`

| `--model` | what runs | preprocessing |
|---|---|---|
| `path/ckpt.pt` | `train/model.py` checkpoint (the v1 recipe), built with the checkpoint's `width` and `bottleneck` when it records them | grey (integer luma), bilinear to the checkpoint's `height` (32 if absent), width ≤ 1024, padded to a multiple of 8 by repeating the last column; one line per forward pass; greedy CTC. A class longer than one character (such as `<unk>`) decodes to U+FFFD, one wrong character |
| `path/model.onnx` | an ONNX export: 1 input channel = the tiny model (same preprocessing, at the export's input height), 3 = a PaddleOCR rec | as above, or the Paddle row below |
| `ppocr-v5-mobile`, `ppocr-v5-mobile-jis` | PP-OCRv5 mobile rec in ONNX Runtime, full head or masked to JIS X 0208 | vita-jp-overlay's: BGR, height 48, width `(w*48 + h/2)/h` clamped to [8, 1024], `(p - 127.5)/127.5` |
| `ppocr-v5-server`, `ppocr-v6-small[-jis]` | the pseudo-labelling engines, for reference | same |
| `ncnn:DIR` | an ncnn param/bin + `dict.txt` in DIR, run by `eval/ncnn/vocr_ncnn_rec` (the overlay's pinned ncnn 3798be64, one thread, fp16 off) | the overlay's `from_pixels_resize` path, exactly |
| `manga-ocr` | manga-ocr base (reference only; ~400 MB) | long lines re-flowed into a square-ish block first, as `data/label_real.py` does (`manga-ocr-raw`: no re-flow) |
| `cmd:COMMAND` | any external program, e.g. the C runtime's host CLI | crops written as binary PPM (or `--cmd-format pgm\|png`) |

`cmd:` protocol: if COMMAND contains `{}`, it runs once per crop with `{}`
replaced by the crop's path and prints that line's text; otherwise it runs once,
reads one crop path per stdin line and prints exactly one UTF-8 line per input,
in order. Example for the runtime: `--model 'cmd:runtime/build/vocr_cli weights.bin'`.

For the ncnn baselines, build the helper and copy the research's models once:

```sh
$HERD/heavy.sh eval/ncnn/build.sh                # pinned ncnn, int8 on, ~3 min
eval/ncnn/prepare_models.py                      # -> data/eval/models/ncnn/{ppocr-v5-mobile-shipped,ppocr-v5-mobile-jis-int8}
eval/run.py --model ncnn:data/eval/models/ncnn/ppocr-v5-mobile-jis-int8
```

`prepare_models.py` checks every copied param, bin and dict against the sha256
the published baselines were measured with, and exits non-zero on a mismatch.

## Sets

| set | lines | source |
|---|---|---|
| `vita-test` | 206 (2,564 chars) from 16 games, 74 screenshots | 960×544 screenshots of Japanese PS Vita releases on VNDB |
| `psp-test` | 245 (2,510 chars) from 14 games, 83 screenshots | 480×272 screenshots of Japanese PSP releases on VNDB |
| `synth-val` | 3,000 | the first 3,000 lines of `data/synth/synth_00039.npz`, the shard `train/train.py` validates on (`--limit 0` for all 10,000) |

Every real line was read by eye on a contact sheet at ≤ 1280 px (2× zoom for
small kana, dakuten and lookalikes), against PP-OCRv5 server, PP-OCRv6 small,
manga-ocr and, where they existed, `data/label_real.py`'s pseudo-labels.
Lines that could not be decided are kept as `"uncertain": true` and never
scored. UI labels, system menus, logos, vertical and rotated text are not in the
sets. In-story choice options are, since the player reads them like dialogue
(e.g. v16566_148766, v17286_92108, v19850_122862, v21789_116091).

**Test games.** A test game is never used for training, calibration or as a
synthetic background. The rule is: every VNDB VN with
`synth/common.is_test_vn(id)` true, plus the ids listed in
`manifests/test_games.json` (`all`). Check against both before adding data.

`all` holds the set games, every `is_test_vn` game with Vita or PSP screenshots,
any VN that shares a VNDB release with a set screenshot, and each set game's
direct VNDB relations that reuse its scenes, UI and fonts (`family`: alternative
versions, originals, fandiscs, sequels, prequels, side and parent stories,
series). The split is by game **and** capture session: every line records its
session (the VNDB release its screenshot was uploaded under), and
`test_games.json` lists them all in `sessions`. Never use a screenshot from a
listed session. A game can be in both sets (v10856 and v13072 are, from their
separate Vita and PSP releases), but no session is split between the sets; the
tests check this. `eval/tools/leakcheck.py POOL_DIR...` compares a training pool
with the test screenshots by thumbnail correlation and lists near-duplicates,
after skipping reserved games as training must.

### Manifest rows (`manifests/<set>.jsonl`)

| field | meaning |
|---|---|
| `id` | stable line id |
| `vn`, `title` | game: VNDB VN id and title |
| `session` | capture session: the VNDB release id the screenshot was uploaded under (one release = one session, the conservative choice) |
| `url`, `sha256` | the VNDB screenshot and the sha256 of its bytes |
| `dims` | screenshot size |
| `region` | `[x, y, w, h]`: the dialogue region a player would draw (the union of the box's lines plus a margin of 12 px on Vita, 6 px on PSP) |
| `box` | `[x, y, w, h]`: the line's ink extent, main text plus any furigana sitting on it; the crop adds `--pad` px (default 4 on Vita, like the overlay's line finder; 2 on PSP, which is half the scale with about 4 px between lines) |
| `text` | ground truth for the main text only, in the training-label convention (full-width ！？～, … for ellipses); furigana, page-cursor icons and button glyphs are never part of it |
| `styles` | tags, below |
| `uncertain` | not scored |

Style tags: `bright-on-dark` and `dark-on-light` (text fill against its
background; for outlined text, the fill against its outline), `outlined`
(outline or heavy shadow), `name` (speaker name), `coloured` (fill colour other
than white/black/grey), `coloured-name` (both), `furigana` (ruby above the line),
`mincho` (serif face), `no-box` (text drawn over the scene without a dialogue
box), `small` (main text under 16 px tall at native size). `manifest.py` guesses
the polarity from the share of ink brighter than the box border. That share
includes outline pixels, so the guess is checked by eye and overridden in the
decisions where it is wrong.

## Metrics

Both prediction and ground truth go through `metrics.norm()`: NFKC, all
whitespace removed, an ellipsis counts as one character (`…`, `‥`, `...` and
`・・・` fold), wave dashes and tildes fold to one character, and so do the
horizontal dashes (`―—─-`…). The long-vowel mark `ー`, the kanji `一`, small
kana and dakuten are **not** folded: confusing them is an error.

- **CER** = total Levenshtein distance / total ground-truth characters.
- **exact** = share of lines whose normalised prediction equals the ground truth.
- **game-macro CER** = mean of the per-game CERs (each game counts once).
- per-style rows are the same numbers over the lines carrying that tag.
- `ms/line` is host wall time, informational only; it says nothing about a console.

`--out` writes everything as JSON, including every line's prediction and the
ground-truth characters the model cannot emit (`oov_chars`).

### Polarity

By default crops reach the recognizer as they are on screen (`--polarity native`).
`--polarity console` hands them over the way the console does under the polarity
contract of `docs/linefind.md`: a crop whose text `synth/common.line_dark` calls
dark is inverted (`255 - grey`, the same grey in all three channels) before
recognition. Use it for a recognizer that will run behind the console's inversion.
On vita-test the rule inverts 37 of the 43 dark-on-light lines and 7 of the other
163. Each inverted line carries `"inverted": true` in `--out`. Both modes
(recognizer and `--finder`) honour it.

### Line finder mode

`--finder overlay|console|gradient|py:MOD:FUNC|cmd:COMMAND` runs a finder on each
dialogue region instead and recognizes what it returns:

- `overlay`: `synth/common.find_lines(region, "legacy")`, the overlay's shipped
  bright-pixel finder;
- `console`: `synth/common.find_lines(region, "grad")`, the bit-exact twin of the
  console's finder `runtime/vocr_lines.c` (`docs/linefind.md`). With
  `--polarity console` this scores the console's whole path;
- `gradient`: the research's polarity-agnostic gradient-energy projection;
- `py:MOD:FUNC`: any Python function `f(rgb_region) -> [(x, y, w, h)] | None`
  (None = region rejected);
- `cmd:COMMAND`: an external program that reads the region (PPM) and prints
  `x y w h` lines, exit status non-zero = rejected.

A detection matches a ground-truth line when their vertical overlap is at least
`--match` (0.5) of the smaller height and they overlap horizontally; matching is
one-to-one. Reported: regions rejected, **line recall** (scored lines found),
**line precision** (detections that match a line, among detections not on an
uncertain line), **pipeline CER** (matched pairs' edit distance, plus every
character of a missed line, plus every character read from a detection that
matches nothing; a detection on an uncertain line is ignored) and **concat CER**
(the research's measure: all detections read top to bottom against all lines, on
regions without uncertain lines).

## Reproducing the sets

The tools in `eval/tools/` built the sets; they are not needed to run the benchmark.

1. `vndb_list.py` lists VNDB screenshots per platform (kana API, 1 req/s).
2. `collect.py vita|psp` gathers the candidate games' native-size screenshots
   (sexual < 0.3, violence < 0.5). The candidates are every `is_test_vn` game in
   the Japanese-original listing, plus the extra Vita games listed in
   `collect.py` (`EXTRA`).
3. `candidates.py` (heavy) finds text with PP-OCRv5 server det and reads every box
   with PP-OCRv5 server, PP-OCRv6 small and manga-ocr.
4. `assemble.py` merges fragments, refines boxes to ink, attaches furigana.
5. `sheets.py` + `boxfix.py` make the contact sheets; `review.py` records each
   decision; `relations.py` fetches the VNDB relations; `manifest.py` writes the
   manifests and `test_games.json`.

Tests: `python -m pytest tests/test_eval.py` (or `python tests/test_eval.py`
without pytest): metrics, backends' preprocessing, the finders and the manifest
schema, sessions and reservations. No network or screenshots needed.

The local working files (screenshots, readings, decisions) live in
`$HERD/shared/artifacts/eval/`; only the manifests are committed.
