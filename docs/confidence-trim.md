# Trailing low-confidence trim (`vocr_opts.tail_conf`)

A recognizer-side post-process that drops the trailing run of low-confidence
characters from a line. It targets the finder's wide speaker-name crops, where
the GRAD finder's box spans the whole name bar — the name text **and** the
decorative plate art beside it — and the recognizer appends glyphs read off the
art after the name. It is off by default (`tail_conf = 0`), and when off the
output is bit-for-bit the untrimmed output.

This is the one technique from the GameSentenceMiner / owocr / meikiocr survey
(`docs/gsm-research` in `~/herd-synth3/reports/gsm-research.md`) that measured a
real, dialogue-safe gain on our development data through the C runtime. The idea
is meikiocr's recognition-confidence threshold; no third-party code was copied
(the trim is ~30 lines of original C, a watermark over greedy CTC, and its
identifiers appear in none of the surveyed clones):

- **meikiocr** `52aa607e` (Apache-2.0), `meikiocr/ocr.py:104`
  (`run_ocr(…, rec_threshold=0.1, punct_conf_factor=1.0)`), `:369-370`
  (`_postprocess_recognition_results`), `:386-387` (`if scr < rec_conf_threshold:
  continue`), `:438-443` (punctuation candidates' confidence scaled before NMS):
  drops recognized characters whose score is below a threshold.
- **owocr** `0339f341` and **GameSentenceMiner** `c3499a03` apply meikiocr's filter
  through their MeikiOCR engines — `owocr/ocr.py:2511` and
  `GameSentenceMiner/owocr/owocr/ocr.py:4342` both call
  `run_ocr(image_np, punct_conf_factor=0.2)`, so meikiocr's `rec_threshold=0.1`
  applies. (GSM's own `ocr/gsm_ocr.py` controller filters only by text similarity,
  not recognition confidence.) Both GPL-3.0.

Our recognizer is a greedy-CTC line model, not a per-character detector, so the
reimplementation is a **trailing-only** trim: a character at the end of the line
is dropped when its emitting CTC frame's softmax-max confidence is below the
threshold, and so is every character after it. Interior characters are never
dropped (dialogue ending in a confident glyph pulls the keep-watermark past any
earlier low-confidence character). Trailing-only is deliberate: the plate-art
hallucination is appended **after** the name, and ordinary substitution errors
inside a line are not confidence-separable, so trimming the interior would only
risk good text.

## What it does, exactly

`runtime/vocr.h`:

```c
unsigned tail_conf;   /* trailing low-confidence trim, in 1/1000 (0: off) */
```

In `runtime/vocr.c`:

- When `tail_conf` is non-zero, `run_head` accumulates the softmax denominator
  per CTC frame alongside the argmax it already computes (a running-max rescaled
  sum of `exp(logit - max)`, so it is numerically stable). The confidence of a
  frame is `1 / sum`, i.e. the softmax probability of the argmax class.
- `ctc_push` records, as it emits each character, a watermark (`used_hi`,
  `chars_hi`) that advances only through characters whose confidence is at least
  `tail_conf / 1000`. At the end of `vocr_run`, the text is truncated to the
  watermark.
- With `tail_conf = 0` none of this runs: the `if (conf)` branch is hoisted out of
  the per-class loop, so the disabled head loop is byte-for-byte main's (measured:
  +0.17 % instructions, within noise), the softmax array is not allocated, and the
  arena and output are identical to before. The arena grows by one `acc_frames`-float
  array (80 bytes at the default tiling, for every blob) **only** when the trim is on.

No heap, no recursion, C99, **no libm**: the softmax term uses `vexpf_nonpos`, a
runtime-local float `exp` for x ≤ 0 (a 2^x exponent-bit construction plus a degree-6
minimax polynomial on `[0,1)`). The Vita object's undefined symbols are exactly
main's (`__aeabi_idiv`, `__aeabi_uidiv`, `__aeabi_uidivmod`, `memcpy`, `memset`), so
a `-nostdlib` plugin links it unchanged, and the same code runs on host, Vita and
PSP. `tail_conf` outside `[0, 1000]` is rejected by `vocr_init` / `vocr_arena_bytes`.

## Cost on an A9 (estimate)

The confidence pass adds one `vexpf_nonpos` per class per CTC frame, on the enabled
path only. Measured ARM instruction counts (`vocr.c` built with `arm-vita-eabi-gcc
15.2` and the Vita flags, run under the repo's plugin qemu-arm with
`vita/qemu/insn_mix`, on three 670-728 px dev lines, W8I):

| model | tail 0 → 500 instructions/line | overhead | default (tail 0) vs main |
|---|---:|---:|---:|
| VOCR3-FL10 | 38.5M → 46.3M | **+20.3 %** | +0.17 % (within noise) |
| VOCR3-W20 | 89.5M → 101.5M | **+13.4 %** | — |

Through `vita/a9_cycles.py`'s A9 cost model (as the model-tiers and VOCR3 docs do),
that is roughly **+18-26 ms per 640-px line on FL10 and +27-40 ms on W20 at 444 MHz**
(≈ +35-53 ms on W20 at 333 MHz), about **+50-120 ms per typical 3-line textbox**.
Estimates; nothing has run on a console. They stay within the per-textbox speed
limits (`docs/model-tiers.md`: Small ≤ 1 s, Normal ≤ 2 s, Large ≤ 3 s at 444 MHz).
The overhead is real — one `exp` per class per frame over a 3,500-5,355-class head.
A host wall-time ratio (FL10 +2.3 %, W20 +1.2 %) badly understates it, because the
head runs on NEON and the `exp` on scalar VFP; it is kept only as a sanity check.

## Measured gain (development data only)

Pipeline: the C GRAD finder cuts each labelled dialogue region of **devnative v3**
(the native Vita/PSP development cohort, `docs/devnative.md`; v1331, the owner's
Memories Off, excluded) into line crops, the C W8I recognizer reads each, and
detections are matched one-to-one to the hand-labelled ink lines by vertical
overlap ≥ 0.5. Harness: `eval/gsm/devpipe.py`. Protocol:
`eval/gsm/confidence_trim_protocol.json`, **frozen with this change** — the
threshold `tail_conf = 500` (confidence 0.5) was chosen by a sweep on this same
543-line cohort as the lowest threshold that captures the names gain while leaving
dialogue flat, so the table below is the **in-sample** result of that selection.
Development data only, disclosed not hidden. No protected set was read.

CER %, lower is better; exact % is the share of lines read exactly; "notext" is
the characters the recognizer writes on finder false positives (scene art / plate
decoration), lower is better.

| model | names CER | names exact | dialogue CER | dialogue exact | notext chars |
|---|---:|---:|---:|---:|---:|
| VOCR3-FL10, off | 51.55 | 31.97 | 8.42 | 77.45 | 1081 |
| VOCR3-FL10, tail 0.5 | **43.04** | 36.07 | 8.33 | 77.45 | 1002 |
| VOCR3-H15, off | 43.81 | 41.80 | 8.07 | 81.25 | 1172 |
| VOCR3-H15, tail 0.5 | **40.98** | 42.62 | 8.09 | 81.25 | 1100 |
| VOCR3-W20, off | 41.49 | 38.52 | 7.72 | 80.43 | 1144 |
| VOCR3-W20, tail 0.5 | **36.34** | 45.08 | 7.70 | 80.98 | 1092 |

Name CER drops on every tier and name exact-match rises; dialogue is flat
(±0.1 points); the no-text count falls. A paired bootstrap over lines (10,000
resamples) gives name-CER changes of −9.0 points for FL10 [−14.5, −4.4] and −5.9
for W20 [−9.5, −2.9], both clear of zero, but **−2.8 for H15 [−5.9, 0.0], which
cannot be told apart from zero** on its 122 name lines. Dialogue changes are within
noise on all three. The protocol's decision rule (names down, dialogue up ≤ 0.1
points, no-text not up) holds for all three; H15's names gain is within noise.

W20 threshold sweep on the cohort (names / dialogue CER): 0.50 → 36.34 / 7.70;
0.55 → 35.31 / 7.78; 0.60 → 34.54 / 7.74; 0.65 → 34.28 / 7.87; 0.70 → 34.54 / 7.94.
Higher thresholds trade a little more names gain for a slowly rising dialogue CER;
0.50 is the knee where dialogue is still flat.

At 0.5 the trim empties a few whole lines (per tier, 7-13 of 368 dialogue and 4-7 of
122 name lines) — all already misread, but on the console such a line now shows
nothing instead of a wrong guess.

(The numbers shifted by ≤ 0.8 points from an earlier `expf` draft when `expf` was
replaced by the libm-free `vexpf_nonpos`; the gains and dialogue-safety are
unchanged.)

### Why not a geometry name-crop

The obvious fix — have the finder tight-crop the name plate to its text — was
tried and **rejected**. Three geometry variants all hurt dialogue more than they
helped names (devnative v3, W20, through the C runtime):

| variant | names CER | dialogue CER |
|---|---:|---:|
| baseline | 41.5 | 7.7 |
| densest contrast run | 49.0 | 12.4 |
| dense-glyph span | 49.5 | 55.7 |
| edge-only blank trim | 40.7 | 8.4 |

The plate art is textured and produces non-trivial contrast, so no column-density
rule separates a name plate from an ordinary (already-tight) dialogue box without
also cutting dialogue. An oracle crop to the hand-labelled ink box shows the gain
that a *perfect* name crop would give (names 10.5 %, dialogue 5.3 % on the
ok-GRAD subset), but geometry cannot deliver it safely, and the finder
(`runtime/vocr_lines.c`) is frozen. The confidence trim captures part of the same
gain from the recognizer side, without touching the finder and without risking
dialogue. The full survey and these prototypes (development scratch, not committed)
are in `~/herd-synth3/reports/gsm-research.md`.

## Reproduce

```sh
for b in VOCR3_FL10_w8 VOCR3_H15_w8 VOCR3_W20_w8; do
  .venv-ci/bin/python eval/gsm/devpipe.py --blob $b.vocr --tail 0
  .venv-ci/bin/python eval/gsm/devpipe.py --blob $b.vocr --tail 500
done
```

The crops are devnative v3 under `$HERD/shared/artifacts/devnative/` (never
committed). `runtime/tests/test_vocr.c::test_tail_conf` checks the C trim against
an independent reference built from the captured per-frame logits, that
`tail_conf = 0` is bit-identical to the untrimmed output, and that the trimmed
text is always a prefix of it.

## Not done

- No console or emulator run; the A9 cost is an estimate from the Vita build's
  instruction mix through `vita/a9_cycles.py`'s model.
- The overlay does not set `tail_conf` yet. Because the runtime stays libm-free,
  shipping it to the console is a one-line overlay change (set `opts.tail_conf = 500`
  where it fills `vocr_opts`), to be done against the overlay's next build through
  its CI and audit, not the build under test in overlay #74.
- The threshold is a single development-selected constant; it is not per-tier or
  per-style tuned.
