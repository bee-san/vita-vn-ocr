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
is theirs; no third-party code was copied (all are GPL-3.0 except meikiocr,
Apache-2.0, so copying would be licence-compatible, but a reimplementation from
the idea is cleaner here):

- **meikiocr** (`meikiocr/ocr.py`, `run_ocr(rec_threshold=0.1)`,
  `_postprocess_recognition_results`): drops recognized characters whose score is
  below a threshold, and down-weights punctuation (`punct_conf_factor`) so an
  overlapping real glyph wins. Apache-2.0.
- **owocr / GameSentenceMiner**: filter and dedupe by recognition confidence in
  the two-pass controller (`GameSentenceMiner/ocr/gsm_ocr.py`). GPL-3.0.

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
  sum of `exp(logit - max)`, so it is numerically stable and independent of class
  order). The confidence of a frame is `1 / sum`, i.e. the softmax probability of
  the argmax class.
- `ctc_push` records, as it emits each character, a watermark (`used_hi`,
  `chars_hi`) that advances only through characters whose confidence is at least
  `tail_conf / 1000`. At the end of `vocr_run`, the text is truncated to the
  watermark.
- With `tail_conf = 0` none of this runs and the softmax array is not even
  allocated, so the arena and the output are identical to before. The arena grows
  by one `acc_frames`-float array (80 bytes at the W20 head batch) **only** when
  the trim is enabled.

No heap, no recursion, C99; `expf` from `<math.h>` (already a runtime
dependency). The confidence pass adds one `expf` per class per CTC frame, but only
on the enabled path.

## Cost on an A9 (estimate)

The confidence accumulation runs inside the output head, which already computes
every class logit per frame, so the extra work is a small fraction of recognition.
Measured on the host (Xeon, through the C runtime), the per-line overhead of
turning the trim on is **+0.6 ms on FL10 (+2.3%) and +0.9 ms on W20 (+1.2%)** over
a 640-px line. Scaled to one 444 MHz Cortex-A9 core by the same ratio, the trim
adds on the order of **1-3% to recognition time** — a few milliseconds per line,
within every tier's speed budget (`docs/model-tiers.md`). This is a host-ratio
estimate; nothing has run on a console.

## Measured gain (development data only)

Pipeline: the C GRAD finder cuts each labelled dialogue region of **devnative v3**
(the native Vita/PSP development cohort, `docs/devnative.md`; v1331, the owner's
Memories Off, excluded) into line crops, the C W8I recognizer reads each, and
detections are matched one-to-one to the hand-labelled ink lines by vertical
overlap ≥ 0.5 (the `eval/linefind` rule). Harness: `eval/gsm/devpipe.py`.
Pre-registered protocol: `eval/gsm/confidence_trim_protocol.json`. The threshold
`tail_conf = 500` (confidence 0.5) was chosen on this development cohort as the
lowest threshold that captures the names gain while leaving dialogue flat; this
selection used development data only and is disclosed here, not hidden. No
protected set was read.

CER %, lower is better; exact % is the share of lines read exactly; "notext" is
the characters the recognizer writes on finder false positives (scene art / plate
decoration), lower is better.

| model | names CER | names exact | dialogue CER | dialogue exact | notext chars |
|---|---:|---:|---:|---:|---:|
| VOCR3-FL10, off | 51.55 | 31.97 | 8.42 | 77.45 | 1081 |
| VOCR3-FL10, tail 0.5 | **42.53** | 36.07 | 8.33 | 77.45 | 1002 |
| VOCR3-H15, off | 43.81 | 41.80 | 8.07 | 81.25 | 1172 |
| VOCR3-H15, tail 0.5 | **40.98** | 42.62 | 8.09 | 81.25 | 1098 |
| VOCR3-W20, off | 41.49 | 38.52 | 7.72 | 80.43 | 1144 |
| VOCR3-W20, tail 0.5 | **35.57** | 45.90 | 7.70 | 80.98 | 1090 |

On every tier, name CER drops (−9.0 FL10, −2.8 H15, −5.9 W20), name exact-match
rises, dialogue is flat (±0.1 points, well inside noise) and the hallucinated
no-text character count falls. The decision rule in the protocol (names strictly
down, dialogue up by no more than 0.1 points, no-text not up) holds for all three.

### Why not a geometry name-crop

The obvious fix — have the finder tight-crop the name plate to its text — was
tried and **rejected**. Three geometry variants (densest contrast run, dense-glyph
span, edge-only blank trim) all hurt dialogue more than they helped names, because
the plate art is textured and produces non-trivial contrast, so no column-density
rule separates a name plate from an ordinary (already-tight) dialogue box without
also cutting dialogue. An oracle crop to the hand-labelled ink box shows the gain
that a *perfect* name crop would give (names 10.5 %, dialogue 5.3 % on the
ok-GRAD subset), but geometry cannot deliver it safely, and the finder
(`runtime/vocr_lines.c`) is frozen. The confidence trim captures part of the same
gain from the recognizer side, without touching the finder and without risking
dialogue. The negative evidence is in the report.

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

- No console or emulator run; the A9 cost is a host-ratio estimate.
- The overlay does not set `tail_conf` yet. Shipping it to the console is a
  one-line overlay change (set `opts.tail_conf = 500` where it fills `vocr_opts`),
  to be done against the overlay's next build through its CI and audit, not the
  build under test in overlay #74.
- The threshold is a single development-selected constant; it is not per-tier or
  per-style tuned.
