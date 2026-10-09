# Results

Real-screenshot numbers come from the shared benchmark, `eval/run.py` (see
`eval/README.md`). Host speed is informational only; nothing here was measured
on a console.

## Baselines

Measured 2026-10-09 on the shared x86 host: Python 3.11.15, torch 2.5.1+cpu,
onnxruntime 1.30.0, and the ncnn helper `eval/ncnn/vocr_ncnn_rec` (ncnn
3798be64, one thread, fp16 off, int8 on, the overlay's preprocessing). Decoding
is greedy and the crops are fixed, so a rerun gives identical text; one run per
row.

```sh
M1=$HERD/shared/models/v1/ckpt_best.pt
eval/run.py --model $M1 --set vita-test,synth-val --out v1.json
eval/run.py --model ncnn:data/eval/models/ncnn/ppocr-v5-mobile-shipped --set vita-test,synth-val
eval/run.py --model ncnn:data/eval/models/ncnn/ppocr-v5-mobile-jis-int8 --set vita-test,synth-val
eval/run.py --model $M1 --set vita-test --finder overlay     # and --finder gradient
```

vita-test is 206 scored lines (2,564 characters) from 16 games. synth-val is the
first 3,000 lines (54,306 characters) of `data/synth/synth_00039.npz`.

| model | weights | peak memory (research) | synth-val CER / exact | vita-test CER / exact | vita-test game-macro CER |
|---|---|---|---|---|---|
| v1 tiny (`train/model.py`, 761,771 params, 5,355 classes) | 3,047,084 B fp32 | not measured yet (the runtime's arena report) | 0.32% / 95.80% | 23.75% / 52.43% | 23.74% |
| PP-OCRv5 mobile as shipped in vita-jp-overlay (ncnn fp16, 18,385 classes) | 8,242,276 B | 39.52 MiB tensor-heap peak over the research's 30 boxes; 42.4 MiB worst real region | 2.40% / 73.70% | 3.43% / 75.24% | 3.34% |
| PP-OCRv5 mobile, head pruned to JIS X 0208, int8 (ncnn, 6,513 classes; the research's recommendation) | 2,749,540 B | 14.29 MiB heap peak, 11.77 MiB with the raw buffer freed; 15.90 / 13.38 MiB with every line at the 1024-px clamp; RSS 20.9 MiB | 2.02% / 76.63% | 2.30% / 79.13% | 2.49% |
| manga-ocr base (reference) | about 400 MB | not measured | pending | pending | pending |

Memory figures are the research's measurements in the overlay's own `vjo-ocr`
(`docs/research-findings.md`; `scratch/int8/sa250291/results_summary.json`).
For the ONNX presets the research's liveness measure at 48×1024 is 33.6 MiB for
PP-OCRv5 mobile (4,113,194 params) and 16.6 MiB once JIS-pruned (2,676,682
params, 5,369,252-byte ncnn bin).

Host time per line, one thread for ncnn and two for torch: v1 5.8 ms, shipped
44.9 ms, JIS int8 62.1 ms (vita-test).

Things to keep in mind when reading the table:

- synth-val is v1's own training distribution, so its 0.32% says little about
  real games. It also contains characters PP-OCRv5 cannot emit (42 for the
  shipped head, 54 for the JIS head: ゐ, ヱ, ◯, ※, 呟, 欝...), which count as errors.
- v1 cannot emit one vita-test character (瘴).

### vita-test by style

CER / exact-line share per tag. A line can carry several tags.

| style | lines | chars | v1 tiny | PP-OCRv5 shipped | JIS int8 |
|---|---|---|---|---|---|
| bright-on-dark | 142 | 1,809 | 7.46% / 70.4% | 3.48% / 75.4% | 2.38% / 78.2% |
| dark-on-light | 64 | 755 | 62.78% / 12.5% | 3.31% / 75.0% | 2.12% / 81.2% |
| outlined | 72 | 939 | 28.01% / 48.6% | 5.11% / 66.7% | 3.09% / 72.2% |
| name | 53 | 172 | 21.51% / 60.4% | 3.49% / 88.7% | 4.07% / 86.8% |
| coloured | 17 | 186 | 43.01% / 41.2% | 4.84% / 70.6% | 4.30% / 58.8% |
| coloured-name | 7 | 19 | 26.32% / 57.1% | 5.26% / 85.7% | 10.53% / 71.4% |
| furigana | 3 | 64 | 7.81% / 0.0% | 1.56% / 66.7% | 3.12% / 33.3% |
| mincho | 47 | 676 | 1.33% / 85.1% | 0.89% / 87.2% | 1.33% / 85.1% |

v1 was trained only on bright text on dark boxes, and it shows: dark-on-light
lines are 62.8% CER and outlined ones 28.0%, against 7.5% for bright-on-dark.
The furigana and coloured-name rows have too few lines to rank models.

### Line finders on vita-test (v1 recognizer)

84 dialogue regions holding 206 scored lines.

| finder | regions rejected | line recall | line precision | pipeline CER | concat CER |
|---|---|---|---|---|---|
| `overlay` (the shipped bright-pixel finder) | 25 | 64.08% | 97.08% | 42.32% | 41.38% |
| `gradient` (the research's polarity-agnostic finder) | 0 | 88.84% | 100.00% | 43.21% | 40.90% |

With v1 as the recognizer the finder barely moves the pipeline CER. The 25
regions the overlay finder rejects hold 66 scored lines, 46 of them
dark-on-light (of 64 in the set), and v1 reads those 66 lines at 65.3% CER even
from hand-drawn boxes.

## Test-set hygiene

- **Split by game and capture session.** Each manifest line records its game
  (`vn`) and session (`session`, the VNDB release its screenshot belongs to). The
  16 vita-test games come from 16 releases, all listed in
  `eval/manifests/test_games.json` (`sessions`).
- **Related games are reserved too.** 7 of the 16 games have direct VNDB
  relations that reuse their scenes, UI or fonts (alternative versions,
  originals, fandiscs, sequels, prequels). Those 10 VNs are in `all` (`family`):
  v3, v10606, v18160, v18717, v19338, v20226, v24312, v25914, v26075 and v40027.
  v1's `real_train.npz` has 12 lines from two of them (v3, v18717, both relations
  of v7721), and v1 reads v7721 at 68.55% CER, so there is no sign this helped it.
- **Near-duplicate audit.** `eval/tools/leakcheck.py` compared the 74 vita-test
  screenshots with 7,675 non-reserved pool images (v1's background pool plus
  realdata's vita, psp and hires shots). No pair reached 0.95 thumbnail
  correlation; the best was between 0.90 and 0.95, for one screenshot. A test
  screenshot with its text painted out, recompressed and rescaled still scores at
  least 0.958 against itself, and random pairs from different games peak at 0.78.
- **Known leak into v1.** v16566 and v19850 are not `is_test_vn` games, so
  `synth/render.py` could use 9 of these 74 screenshots as unlabelled backgrounds
  for v1's shards. v1's CER on those 28 lines is 79.3%, against 15.7% on the other
  178, so the leak did not inflate it. Both games are reserved now; later shards
  must skip `all`.

## Not measured yet

- psp-test: hand-checking in progress.
- manga-ocr, and the PP-OCRv5 mobile ONNX presets (`ppocr-v5-mobile`,
  `ppocr-v5-mobile-jis`), on every set.
- Any console speed or memory figure.
