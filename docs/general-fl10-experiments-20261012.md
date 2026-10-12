# General VN FL10 experiments, 2026-10-12

These are rejected research snapshots. The released FL10 recognizer remains
selected; none of these weights is an improved runtime. No protected test or
Google candidate scoring was performed.

The new synthetic snapshot in [the source manifest](https://github.com/bee-san/vita-vn-ocr/blob/f99050c83d0a16059e5de686cbc2c9d21536c45e/data/manifests/provenance_synthetic_20261011.json)
contains 20,000 training and 1,000 development crops from nine OFL fonts,
book-separated Aozora source pools and procedural textbox backgrounds. Game
fonts, scripts and screenshot backgrounds were absent. The vocabulary stayed at
3,500 characters; all supported kanji belong to the pinned Jiten allowlist.

The real development cohort has 237 native frames, 517 lines and 6,614 normalized
characters, including 3,554 hiragana. It contains three PSP and three Vita VNs.
Historical baseline title exposure is disclosed: these are new native frames,
not unseen-title results. Assistant source readings checked the labels; there
is no human review claim. Haitaka's 58 body lines have two independently frozen
blind readings; its 17 names have one blind reading followed by another
assistant's native source/geometry verification. Protected benchmark data
stayed separate from training and selection.

| Console | VNDB ID | Frames | Lines | Body characters | Body hiragana |
| --- | --- | ---: | ---: | ---: | ---: |
| PSP | Amagami, v1281 | 50 | 87 | 1,217 | 642 |
| PSP | Corpse Party, v2632 | 44 | 94 | 777 | 419 |
| PSP | Grisaia, v5154 | 50 | 131 | 1,608 | 904 |
| Vita | Suran, v18161 | 43 | 82 | 1,264 | 740 |
| Vita | Nil Admirari, v18163 | 21 | 48 | 588 | 330 |
| Vita | Haitaka, v18643 | 29 | 75 | 855 | 479 |

## Training and line-crop selection

One seed-1, 2,000-step run fine-tuned all model parameters with frozen BatchNorm
statistics. Batch size was 32, learning rate 0.00002, warmup 50 steps, final LR
factor 0.05, weight decay 0.01 and six CPU threads. It used no distillation or
training augmentation. Scheduled checkpoints every 200 steps were retained.

The 64,000 rows actually forwarded and consumed by the optimizer comprised
62,457 broad, 278 kana corrective and 1,265 kanji corrective rows. All ten prefix
cap audits passed: at most 0.5% per deliberately targeted kana/kanji pattern and
2% total kanji corrective exposure. Prefetch draws are excluded from these
counts.

Each exported checkpoint was measured in scalar native C W8I on frozen line
crops with console polarity and padding. Every checkpoint changed Amagami's
all/body hiragana errors from 3 to 4, so all ten failed the per-VN selection
guard. Aggregate gains alone did not select a replacement. The final scheduled
step is shown as a rejected example, not a winner.

| Line-crop measure | Released FL10 | Rejected step 2000 | Fixed quarter blend |
| --- | ---: | ---: | ---: |
| All full errors / characters | 183 / 6,614 | 142 / 6,614 | 156 / 6,614 |
| All hiragana errors / characters | 65 / 3,554 | 52 / 3,554 | 56 / 3,554 |
| Body full errors / characters | 133 / 6,309 | 86 / 6,309 | 108 / 6,309 |
| Body hiragana errors / characters | 47 / 3,514 | 30 / 3,514 | 37 / 3,514 |

The synthetic development guard also improved at every checkpoint. At step
2000 it changed full errors from 648 to 487 of 14,834 characters and hiragana
errors from 167 to 105 of 8,725. These synthetic measurements did not establish
real-screenshot accuracy or override a per-VN failure.

## Bounded blend follow-up and actual textbox rejection

After observing the first run's DEV failures, a separate prospective follow-up
froze seven whole-parameter fractions: 0, 1/16, 1/8, 1/4, 1/2, 3/4 and 1.
All 54 BatchNorm buffers were identical and copied unchanged. Alpha 0 exactly
reproduced baseline aggregate scores; alpha 1 exactly reproduced step 2000.
Both endpoints' exported weight payloads matched their pinned parent payloads.

The fixed line-selector chose alpha 1/4. Its per-VN and synthetic line guards
passed. Only that one locked candidate was measured through the already-frozen
native C GRAD finder on 711 full-textbox cases: base, +4 and +8 pixel symmetric
expansions, 1,453 detected line crops per model. Every C output was scored in
reading order; there was no reference-based matching, masking or filtering.

The quarter blend failed the textbox guard. In base Nil Admirari ROIs, all-text
errors rose from 227 to 229 of 663 characters. With +8-pixel Haitaka ROIs,
all/body hiragana errors rose from 16 to 19 of 479. Other per-VN strata passed;
all variants had balanced hiragana gains and at least two title gains. The
individual regressions still rejected the candidate. No alternative blend was
tried against this textbox guard.

Full per-VN counters, exact fractional macro rates, commands and preparation,
source-isolation, compiler/library/model/dependency hashes are retained in
[the training results](benchmarks/general-fl10-20261012.json) and
[the blend results](benchmarks/general-fl10-blends-20261012.json). Private
absolute input paths are replaced with placeholders; counters remain unchanged.

## Retained weights and reproducibility

The [research release](https://github.com/bee-san/vita-vn-ocr/releases/tag/v0.2.2-fl10-general.1)
retains all ten original checkpoints, seven derived blend checkpoints and all
17 corresponding int8 exports. The original checkpoints contain optimizer and
exposure state; blend checkpoints are derived weights without resume state.
[The asset inventory](benchmarks/general-fl10-snapshot-20261012.json) gives each
archive/member size and SHA256. Both ZIPs were read back and every member's CRC,
size and SHA256 verified before publication. No rendered shards, screenshots,
crops, corpora or fonts are in Git or this weights release.

The environment was x86_64 Linux on Core Ultra 7 165U, Python 3.11.15,
PyTorch 2.5.1+cpu, NumPy 2.4.6, Pillow 12.3.0 and GCC 16.2.1. Commands were:

```sh
.venv-ci/bin/python train/generic_fl10_train.py run --prepared private/general/training-prepared.json
.venv-ci/bin/python train/general_vn_select.py select --lock private/general/protocol.json \
  --checkpoints 'private/general/runs/general_ofl_all_s1/ckpt_step*.pt' --out private/general/selection.json
.venv-ci/bin/python train/fl10_blend.py --initial private/fl10_init.pt \
  --original-selection private/general/selection.json --baseline private/FL10_w8.vocr --out private/blend
.venv-ci/bin/python train/general_vn_textbox_guard.py compare \
  --candidate-lock private/blend/textbox-prepared-candidate.json --out private/blend/textbox-result.json
```

The recognizer uses the unchanged 657,664-byte blob and 124,832-byte W8I arena
at maximum line width 1024. These are host C measurements. Physical-console
latency, memory residency and compatibility were not measured by this run.
Original historical training shards and per-chunk Aozora book IDs remain
unavailable; the fresh source pools, book cohorts, recipe and hashes are pinned.
