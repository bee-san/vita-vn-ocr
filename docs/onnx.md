# ONNX files of VOCR6 and textspot8 for desktop apps (issue #163)

Pre-release [v0.5.1-onnx.1](https://github.com/bee-san/vita-vn-ocr/releases/tag/v0.5.1-onnx.1) holds the VOCR6.2
recognizers (Small, Normal and Large) and the textspot8 line spotter as ONNX files, for apps that run OCR on a PC
rather than on the console: GameSentenceMiner (GSM) and others. They are conversions of v0.5.0's int8 blobs, the files
Rubi runs, with the same weights and the same non-commercial terms. `tools/vocr_onnx.py` is the reference pipeline:
one file that needs only NumPy and ONNX Runtime.

The method follows a report by **astra**, an outside researcher on the GSM side, who converted these models to ONNX
first and wrote up which conversions keep VOCR's output and which do not ("VOCR 6.2 and textspot8: ONNX conversion and
CPU precision findings", 2026-10-31; vita-vn-ocr #164 reproduces the rest of it). Thank you, astra. The int16 digit
split with two `MatMulInteger` calls, the int32 reconstruction, and float64 for the spotter's requantisation are
astra's. This export adds three things: the stem and depthwise convolutions in the C runtime's own order, so that every
logit is bit-identical (astra's files used ONNX Runtime's float convolutions); a head that skips the frames the head
gate skips (astra's limit: their files computed the full head and masked it); and the parity run against the C
runtime on this repository's development data.

## The files

| file | bytes | from (v0.5.0) | what |
|---|---:|---|---|
| `VOCR6.2_S_exact.onnx` | 895,567 | `VOCR6.2_S_w8.vocr` | Small, exact |
| `VOCR6.2_S_fast.onnx` | 837,393 | `VOCR6.2_S_w8.vocr` | Small, fast |
| `VOCR6.2_N_exact.onnx` | 1,363,804 | `VOCR6.2_N_w8g2.vocr` | Normal (Rubi's default), exact, with its head gate and two-level head |
| `VOCR6.2_N_fast.onnx` | 1,264,708 | `VOCR6.2_N_w8g2.vocr` | Normal, fast |
| `VOCR6.2_L_exact.onnx` | 2,659,133 | `VOCR6.2_L_w8_2l.vocr` | Large, exact, with its two-level head |
| `VOCR6.2_L_fast.onnx` | 2,525,242 | `VOCR6.2_L_w8_2l.vocr` | Large, fast |
| `textspot8_exact.onnx` | 94,490 | `textspot8_t64n8c16_w8.vtsp` | the dialogue-line spotter (float32 convolutions on its integer codes) |
| `textspot8_int8.onnx` | 95,060 | `textspot8_t64n8c16_w8.vtsp` | the same with `ConvInteger` |

The release also carries `vocr_onnx.py` (this repository's `tools/vocr_onnx.py` at the release commit), the 1 MB
character LM `vocr_lm6_o3_1000k.vlm` (Rubi's default LM tier, byte for byte from v0.5.0, with `LM-settings.json`),
the attribution files and model cards of the blobs, `onnx-manifest.json` (each file's size, sha256, source blob and
the commit it was exported from) and `SHA256SUMS`. The SHA-256 of every file is in `SHA256SUMS` and
`onnx-manifest.json`; the source blobs are the ones astra pinned (`VOCR6.2_S_w8.vocr` `864bbd19…`,
`VOCR6.2_N_w8g2.vocr` `107cf084…`, `VOCR6.2_L_w8_2l.vocr` `65310013…`, `textspot8_t64n8c16_w8.vtsp` `e129d41a…`).

All weights are stored as int8, as in the blobs; ONNX Runtime turns the fast files' weights and the exact files'
convolution kernels into float32 constants when it loads them. Loaded and run once, a model added 18 to 40 MiB to a
Python process's peak resident memory here (S / N / L exact 23 / 29 / 35 MiB, fast 18 / 31 / 40 MiB, the spotter
16 MiB). The graphs use standard operators only (opset 17, IR version 8). `tools/export_onnx.py release` makes the same
bytes from the same blobs on any machine: the files hold no date or commit.

## Which one

- **exact**: the C runtime's W8I arithmetic (`runtime/vocr.c`, what Rubi runs on the console) in every float32 step and
  order. On every development read below, every logit of every frame equals the C runtime's bit for bit, so the text,
  the confidence and every character's frames and cell are the same as Rubi's for the same crop.
- **fast**: the same layers in plain float32, with ONNX Runtime's own fused kernels: the C runtime's W8F mode (int8
  weights dequantised, float activations). It reads 3 to 4 times faster than exact. On the development reads it gave
  W8F's class on every frame, and differed from W8I (the exact files and Rubi) on 17 of 792,654 frames (12 / 2 / 3 for
  S / N / L) and in the text of 2 of 23,235 line reads. On the fixed development lines' dialogue (17,748 reference
  characters) both read with the same number of errors: 254 / 209 / 128 for S / N / L. With the 1 MB LM the fast
  files gave the C runtime's text on every read.
- **textspot8**: `exact` and `int8` give the same codes (both bit-identical to the C spotter). Take `exact`, the faster
  of the two on this host ("Speed").
- **Not shipped: astra's "native INT8" form** (the exact matrices with ONNX Runtime's own float convolutions; issue
  #164 suggested shipping it as the exact-arithmetic file). It is not bit-exact, and on the same development reads it
  was no closer to the C runtime than the fast files: it differed from W8I on 9 / 3 / 4 frames (S / N / L) and in
  2 / 2 / 0 line texts, the fast files on 12 / 2 / 3 frames and 1 / 1 / 0 texts, at twice the fast files' time. So the
  release has the bit-exact form and the fast one. `tools/export_onnx.py recognizer --mode exact --convs ort` still
  writes it.

The tiers are Rubi's: Small (`S`, 4,987 classes), Normal (`N`, 5,631, the default) and Large (`L`, 5,631). On a PC the
Large model is cheap ("Speed"), and it makes the fewest errors: 128 against Normal's 209 and Small's 254 on the fixed
development lines here, and the fewest of every OCR engine measured on each test set (README, "VOCR6 against other
OCRs").

## Using them in an app

```sh
pip install onnxruntime numpy pillow        # ONNX Runtime 1.30.0 is what the exactness was checked with
```

```python
from vocr_onnx import Spotter, Recognizer, CharLM, read_region

spot = Spotter("textspot8_exact.onnx")
rec = Recognizer("VOCR6.2_L_exact.onnx")
lm = CharLM("vocr_lm6_o3_1000k.vlm", tier="L")        # optional: beam 8 + the character LM at Rubi's settings

lines = read_region(frame, spot, rec, region=(x, y, w, h), lm=lm)   # frame: H x W grey or H x W x 3/4 RGB(A) uint8
for ln in lines:
    print(ln.text, ln.box, ln.tag)        # the text, (x, y, w, h) in frame px, 0 dialogue / 1 name / 2 choice / ...
    for c in ln.text_chars:               # every character of ln.text with its cell along the line, in frame px
        print(c.text, c.x0, c.x1, c.y0, c.y1)
```

Or from a shell: `python3 vocr_onnx.py --models DIR --tier L --lm vocr_lm6_o3_1000k.vlm screenshot.png` prints one
JSON line per image.

- **The frame.** VOCR was trained and measured on console frames at their own size: 960×544 on a PS Vita, 480×272 on a
  PSP and a PSP game shown 2× at 960×544 (Adrenaline), and PC screenshots scaled to fit 960×544. Give it frames at that
  size: scale an emulator window that shows the game larger (Vita3K or PPSSPP at 2× or 3×) down to 960×544 first.
  Other scales were not measured. The spotter halves any frame wider than 480 px before its network, as on the
  console.
- **The region.** `read_region` reads a region the way Rubi reads the dialogue box a player draws: the spotter on the
  region alone, at the scale of the frame it was cut from, then every horizontal line it finds, top to bottom, each
  read with its dark flag (dark text is inverted, as the models were trained). Without `region` it reads the whole
  frame: then keep the lines whose `tag` you want (`spot.classes`: dialogue, name, choice, UI, scene) and, for one
  dialogue block, the spotter's `pick` (`spot.spot(frame).pick`). `read_region` skips the spotter's vertical boxes:
  vertical text needs VOCR6-V and the column finder (`docs/vocr6v.md`), which are not exported here.
- **Characters.** `ln.chars` are the greedy characters, each with its run of CTC frames (`f0`, `f1`) and its cell
  `[x0, x1)` along the line (`runtime/vocr_charbox.c`'s cells: calibrated frame centres, cut halfway between
  neighbours; `docs/char-boxes.md`); `y0`, `y1` are the line's rows. The cells tile the line, so a mouse position picks
  one: that is what a hover lookup needs. `ln.text_chars` are the same for `ln.text`, which the LM may change: each
  character is aligned to the greedy ones (`vocr_charbox_align`'s rule), and a character the LM inserted shares its
  neighbour's cell. The calibration in the files is VOCR5's, fit per tier on VOCR5-S/N/L (`eval/charboxes_cal.json`);
  VOCR6 fine-tunes the same shapes and was not refit.
- **The LM.** `CharLM` is `runtime/vocr_lm.c`'s CTC prefix beam search with the character n-gram LM, in Python, with
  Rubi's settings for that file (beam 8, top-8, gate 8 nats, bracket 8 nats; `vocr_lm6_o3_1000k.vlm`: weight 0.6,
  bonus 2.5 nats, the start-of-line context). It reads the top-8 logits the recognizer returns, and takes 0.6 to
  0.9 ms a line on this host ("Speed").
- **Threads.** Each line is one `session.run` on one line; `Recognizer(path, threads=n)` sets ONNX Runtime's intra-op
  threads ("Speed" has 1 and 5).
- **What Rubi adds.** Rubi's own line path does more than this file: the counter re-decode (`runtime/vocr_post.c`,
  6 / 4 nats), the cascade (a line whose greedy confidence is under 0.7 is read again by the next tier up: S → N,
  N → L), its text normalisation and its filter that drops lines with no kana or kanji (`eval/pagepath.py`). The
  greedy text and confidence (`ln.conf_min`) are here, so an app can add the cascade; with Large on a PC it has
  nothing to cascade to.

## Inputs and outputs

**Recognizer.** Input `grey`: uint8 `[1, 1, 32, W]`, the line as `runtime/vocr.c` preprocesses it, which
`vocr_onnx.preprocess()` does bit for bit: integer luma `(77 R + 150 G + 29 B) >> 8`, `255 − grey` for dark text, OpenCV
`INTER_LINEAR`'s fixed-point resize to height 32 and width `round(w · 32 / h)` (half to even) clamped to [8, 1024],
right pad to a multiple of 8 by repeating the last column. Outputs per CTC frame (T = W / 8): `logits` `[T, classes]`
float32; `best` `[T]` int64, each frame's class (class 0 is the CTC blank); `topk_logits` / `topk_classes` `[T, 8]`, the
8 largest logits (ties to the lower class); `blank_logit` `[T]`. Normal and Large also output `escape_score` (and
Normal `gate_score`) and take a second input, `head_sections` (bool scalar):

- `true`: Rubi's reading (`vocr_opts.head_gate = ON`, `two_level = 1`). A frame whose gate score is above the
  threshold gets its blank logit alone (best 0, top-8 `[blank, −inf, …]`, classes 0); a kept frame gets the 504 most
  frequent classes (the blank included) and the rest only where its escape score is above the escape threshold.
  `logits` is −inf where a class was not computed. This is what the console runs, and the graph computes only that:
  the gate skipped 46.5% of Normal's frames in the development reads below.
- `false`: the plain blobs' reading (`VOCR6.2_N_w8.vocr`, `VOCR6.2_L_w8.vocr`): the full head on every frame.

The sections are speed-only: each moved VOCR6.2's development score by 0.000 to +0.008 points (`docs/vocr6.md`, the
follow-up's gate fits). Pass `sections=` to `Recognizer.read` / `read_region` (default `True`).

**Spotter.** Input `grey`: uint8 `[1, 1, h, w]`, the frame or region at the network's scale (halved by the 2×2 rounded
mean when the frame is wider than 480 px). Output `head`: int8 `[1, 32, ceil(h / 8), ceil(w / 8)]`, the network's codes;
`Spotter.spot` decodes them into boxes as `runtime/vocr_textspot6.c` does (`synth/textspot6_twin.py`, `docs/textspot8.md`).

**Metadata** (`session.get_modelmeta().custom_metadata_map`, values JSON): `vocr.kind`, `vocr.mode`, `vocr.vocab` (the
classes' text), `vocr.height`, `vocr.stride`, `vocr.max_width`, `vocr.topk`, the thresholds, `vocr.charbox_cal`, the
spotter's scale rule and classes, `vocr.source_file` / `vocr.source_sha256`, `vocr.licence` and `vocr.attribution`.

## How exact is exact

The C runtime's W8I mode keeps the stem and the depthwise convolutions in float32 and runs every pointwise layer, the
row collapse and the head with int8 weights on activations quantised per position to int16
(`q = trunc(x · 32767 / max|x| ± 0.5)`), as exact int32 sums over chunks of 512 terms, each chunk's sum converted to
float32 and added in order, then times `scale · max|x| / 32767` plus the bias. The exact files build that from standard
operators:

- **Convolutions.** One grouped `Conv` whose kernels each hold one tap's weight (a product plus zeros: exact on any
  kernel, with or without fused multiply-adds) and one more channel for the bias, then `CumSum` adds bias, tap 0,
  tap 1, … in the C code's order.
- **Matrices.** The int16 values split into two base-256 digits (`q = 256 h + l`), two `MatMulInteger` calls with uint8
  operands and zero points per 512-term chunk (U8U8: exact on every ONNX Runtime CPU kernel; U8S8 can saturate its
  intermediate sums on some AVX2 kernels), and `256 · hi + lo` in int32 before the cast to float32: as float32 a
  multiply and add there is open to ONNX Runtime's fusions, which astra saw move a rounding boundary. With weights in
  [−127, 127] (all `train/export.py` writes; the exporter refuses −128) no int32 sum can overflow.
- **The head gate and the escape.** Their first layer through the same integer path, their score sums in order
  (`CumSum`), the thresholds as in the C code.

Why not ordinary ONNX quantization: VOCR's activations are 16-bit, rounded half away from zero, while ONNX's
`QuantizeLinear` rounds half to even, saturates to 8 bits by default and has no integer kernel for 16-bit activations.
astra measured what 8-bit activations do on 345 line checks of their set: dynamic W8A8 changed the text of 93 and
calibrated QDQ of 50; full and mixed FP16 kept the text but moved character positions on 2. The exact files are the
"native int8 weights, 16-bit activations" form astra found exact, with the convolutions in C order on top.

The exactness was checked with ONNX Runtime 1.30.0's CPU execution provider on x86-64 (AVX-512), with its default
session options (denormals are not flushed to zero; keep `session.set_denormal_as_zero` off), 1, 5 and 16 intra-op
threads and every graph optimisation level; other versions, other execution providers (GPU, DirectML, CoreML) and
other CPUs (ARM64's integer kernels) were not checked.
`tests/test_onnx_export.py` checks the pieces against `runtime/tests/vocr_ref.py` (K = 16 to 1,792, zero, tiny and
extreme rows) and whole random-init models against the C runtime in CI.

## Parity with the C runtime (measured; development data only)

`eval/onnx_parity.py`, issue #163's rules pre-registered before the run, on development data only (no protected set
was read): the 1,724 fixed development line crops of #119's development set (teacher_val dev and devnative: 986
distinct lines of 101 VNs, 500 of them in two or three crops) in both polarities; the 1,161 development
whole-dialogue regions, each read as Rubi reads a region; the 354 rebuilt development frames and textspot-dev's 188
frames, each read whole; every horizontal line the C spotter finds in them; and random crops (1×1 to 3000×40) and
frames (1×1 to 1921×1081). Record: `eval/onnx_parity_20261031.json` (ONNX Runtime 1.30.0, the files above by sha256).

| check | reads | result |
|---|---:|---|
| R1 exact recognizers vs W8I, sections off: every logit (264,218 frames × all classes per tier), best, top-8, blank, text, confidence | 7,745 per tier | 0 differences (S, N, L) |
| R1 with the sections on (N: gate + two-level, L: two-level): best, top-8, blank, text, confidence, gate and escape scores | 7,745 per tier | 0 differences (N gated 122,932 of 264,218 frames; escapes fired on 26,130 (N) and 29,525 (L)) |
| R1 the beam + 1 MB LM (`CharLM` on the exact files vs `runtime/vocr_lm.c` on the C runtime's) | 5,727 / 5,720 / 5,730 | 0 differences |
| R2 the spotter (exact and int8): decode maps vs `vocr_textspot6_test --maps`, every box, block, pick, truncated, presence vs `--batch` | 1,710 frames and regions | 0 differences |
| R4 the pipeline (`read_region`, exact files, the 1 MB LM) vs the C path on the same region: boxes, dark flags, greedy and LM texts, every character's frames and cell | 1,710 × 3 tiers | 0 differences |
| R3 fast vs W8F (reported): best and text | 7,745 per tier | 0 differences |
| R3 fast vs W8I (reported): frames whose class differs / line texts that differ | 264,218 frames per tier | S 12 / 1, N 2 / 1, L 3 / 0; largest logit difference 0.032 (W8F: 0.00024) |
| reported, not shipped: astra's native-int8 form (`--variants ortconv`) vs W8I: logits not bit-equal / frames / texts | 7,745 per tier | S 7,744 / 9 / 2, N 7,744 / 3 / 2, L 7,744 / 4 / 0 |

## Speed on this host (measured)

`eval/onnx_timing.py` on a 64-core AMD EPYC 9R14 host (x86-64 with AVX-512; other agents' jobs ran on the box, load
average about 10 to 30), ONNX Runtime 1.30.0's CPU execution provider, gcc 11.5.0. Inputs, development data only: 200
line crops the spotter cuts from development regions (mean 394 × 39 px), 120 development regions and 60 textspot-dev
frames. Every backend reads the same inputs; 9 passes after 2 warm-ups, the backends in a shuffled order each pass;
the median of the passes' mean ms per input (every range is in `eval/onnx_timing_20261031.json`; no 1-thread row's
passes lie more than 6% from its median). The C runtime is called through ctypes, built two ways: `-O2` (`runtime/Makefile`'s
release flags) and `-O3 -march=native`, both `-std=c99 -ffp-contract=off`; the two builds gave the same logits and
boxes on every timed input. Nothing here is a console time.

**One line, one thread** (ms per line; N and L with Rubi's head sections on, in C and ONNX):

| | Small | Normal | Large |
|---|---:|---:|---:|
| C runtime, gcc `-O2` | 16.1 | 18.1 | 47.1 |
| C runtime, gcc `-O3 -march=native` | 3.9 | 4.2 | 9.3 |
| ONNX exact (`Recognizer.read`: resize, network, CTC, confidence, cells) | 4.2 | 6.0 | 10.2 |
| ONNX fast | 1.4 | 1.8 | 3.2 |
| for comparison: astra's native-int8 form (ONNX Runtime's float convolutions; not bit-exact, below) | 2.8 | 3.5 | 6.5 |
| the network alone (`session.run`): exact / fast | 3.7 / 1.0 | 5.3 / 1.2 | 9.5 / 2.7 |

- **Against the `-O2` build** the fast files are 10 to 15 times faster and the exact ones 3 to 5 times, the pattern
  astra measured against the MSVC `/O2` build (per line, C / native int8 / fast: 12.30 / 3.27 / 1.47 ms for Small,
  13.77 / 5.06 / 2.31 for Normal, 34.05 / 7.43 / 3.58 for Large, on a Ryzen 7 5800X).
- **Against `-O3 -march=native`**, the C runtime's best build here (4.1 to 5.1 times faster than `-O2`, the same
  output), the exact files are slower per line (by 7%, 41% and 10% for S, N and L) and the fast files 2.4 to 2.9 times
  faster. So most of the lead over the `-O2` build is the compiler build, not ONNX; why is issue #164's subject.
- **The head sections** save 0.6 ms (exact) and 0.3 ms (fast) per Normal line and 0.6 / 0.3 ms per Large line in ONNX
  (`head_sections` false: 6.5 / 2.0 and 10.8 / 3.5 ms).
- **Threads.** With 5 intra-op threads one line is not faster (exact 5.7 / 8.0 / 11.5 ms, fast 1.4 / 2.1 / 2.5 ms):
  a line is too small to split. Running several lines at once was not measured.
- **astra's form** (the exact matrices with ONNX Runtime's own float convolutions, `--convs ort`) is faster than the
  exact files but not bit-exact: its logits differed from the C runtime's on 7,744 of 7,745 reads per tier (in the last
  bits), and its class on a few frames ("Which one"). Issue #164 measured clang `-O3 -march=native` faster still than
  gcc's build, on its own line sample.

**The spotter** (ms; a region read as Rubi reads it, halved at the frame's scale; a whole frame):

| | region | frame |
|---|---:|---:|
| C runtime, `-O2` | 2.9 | 43.3 |
| C runtime, `-O3 -march=native` | 1.2 | 17.6 |
| ONNX exact (`Spotter.spot`: halving, network, decode in Python) | 0.9 | 8.4 |
| ONNX int8 (`ConvInteger`) | 1.0 | 9.3 |
| the network alone: exact / int8 | 0.6 / 0.8 | 7.5 / 8.4 |

The int8 form gives the same codes and is slower, as astra found (25.53 against 23.01 ms per image on their machine).

**A region or a frame, spotter and every line** (ms, one thread, greedy, no LM):

| | region S / N / L | frame S / N / L |
|---|---|---|
| C runtime, `-O2` | 27.3 / 30.3 / 74.1 | 83.0 / 90.3 / 161.5 |
| C runtime, `-O3 -march=native` | 7.2 / 7.6 / 15.2 | 27.4 / 28.6 / 41.0 |
| ONNX exact (`read_region`) | 7.4 / 10.2 / 16.7 | 20.6 / 25.0 / 36.3 |
| ONNX fast | 3.1 / 3.7 / 5.9 | 12.7 / 14.1 / 17.9 |

**The LM** (beam 8 + the 1 MB LM at Rubi's settings, per line on its top-8): `runtime/vocr_lm.c` 0.02 ms, the Python
`CharLM` 0.6 to 0.9 ms, the same text on all 199 lines with text per tier (5 passes after 1 warm-up).

## Reproduce

```sh
# the files, from v0.5.0's blobs (checked against its SHA256SUMS)
python3 tools/export_onnx.py release --models DIR_OF_V0.5.0 --out onnx/
# parity (development data, the fleet's shared box), then the timings
python3 tools/export_onnx.py recognizer --blob DIR_OF_V0.5.0/VOCR6.2_N_w8g2.vocr --mode exact --convs ort \
    --out onnx/VOCR6.2_N_ortconv.onnx              # and S, L: the reported variant
python3 eval/onnx_parity.py --onnx onnx/ --blobs DIR_OF_V0.5.0 --out parity.json --variants ortconv
python3 eval/onnx_timing.py --onnx onnx/ --blobs DIR_OF_V0.5.0 --out timing.json --lines 200 --regions 120 --frames 60
python3 eval/onnx_timing.py --onnx onnx/ --blobs DIR_OF_V0.5.0 --out timing_lm.json --parts lm --threads 1 \
    --passes 5 --warmup 1 --lines 200 --regions 1 --frames 1
```

Both evaluation scripts read development data from the shared box (`$HERD/shared/artifacts`) and write counts only.
`tests/test_onnx_export.py` runs in CI on generated models.

## Licence and attribution

The files carry the int8 blobs' terms: GPL-3.0-or-later (bee-san/vita-vn-ocr), **for non-commercial use only**, because
the training inputs carry non-commercial terms (`VOCR6-ATTRIBUTION.md` for the recognizers, `SPOTTER-ATTRIBUTION.md`
for the spotter, both in the release; the LM's terms are in `LM-ATTRIBUTION.md`). The weights were trained in part on
the owner's private captures of copyrighted games, used as training input only; nothing of them is in the files.
`ONNX-ATTRIBUTION.md` in the release says this for the ONNX files and credits astra. The code (`tools/export_onnx.py`,
`tools/vocr_onnx.py`) is the repository's, GPL-3.0-or-later.
