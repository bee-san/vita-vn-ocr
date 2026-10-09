# Results

Real-screenshot numbers come from the shared benchmark, `eval/run.py` (see
`eval/README.md`). Host speed is informational only; nothing here was measured
on a console.

## Baselines

Measured 2026-10-09 on the shared x86 host: Python 3.11.15, torch 2.5.1+cpu,
onnxruntime 1.30.0, and the ncnn helper `eval/ncnn/vocr_ncnn_rec` (ncnn
3798be64, one thread, fp16 off, int8 on, the overlay's preprocessing). Decoding
is greedy and the crops are fixed, so a rerun gives identical text; one run per
row. The PR #2 review reran every row independently and got the same prediction
on every line:
- the screenshots were re-fetched from VNDB and sha256-checked;
- the ncnn helper was rebuilt;
- the models were re-copied and hash-checked by `prepare_models.py`.

```sh
M1=$HERD/shared/models/v1/ckpt_best.pt; NC=data/eval/models/ncnn
eval/run.py --model $M1 --set vita-test,synth-val --out v1.json
eval/run.py --model ncnn:$NC/ppocr-v5-mobile-shipped --set vita-test,synth-val
eval/run.py --model ncnn:$NC/ppocr-v5-mobile-jis-int8 --set vita-test,synth-val
eval/run.py --model $M1 --set vita-test --finder overlay     # and --finder gradient; the same for both ncnn models
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

Host time per line on vita-test is informational only. Each figure is one pass
on a loaded shared host, with one thread for ncnn and one or two for torch, so
the range covers the author's and the reviewer's runs: v1 5.8–7.8 ms, shipped
45–60 ms, JIS int8 62–83 ms.

Things to keep in mind when reading the table:

- synth-val is v1's own training distribution, so its 0.32% says little about
  real games. It also contains characters PP-OCRv5 cannot emit (42 for the
  shipped head, 54 for the JIS head: ゐ, ヱ, ◯, ※, 呟, 欝...), which count as errors.
- **Why train.py reports 0.34%.** The same checkpoint scores 0.34% / 95.6% on
  the same 3,000 lines in train.py (`docs/v1-metrics.jsonl`). The difference is
  scoring, not the model: train.py compares class ids, while eval compares
  `metrics.norm()` text, which folds width and dash variants. Measured on these
  lines:
  - class ids, one line per pass: 0.35% / 95.43%;
  - the same predictions normalised: 0.32% / 95.80% (eval);
  - train.py's padded batches: 0.34% / 95.63%.
- v1 cannot emit one vita-test character (瘴).

### vita-test by style

CER / exact-line share per tag. A line can carry several tags. Polarity is the
text fill against its background, and for outlined text the fill against its
outline (`eval/README.md`); the PR #2 review corrected 26 tags that had been
decided over the outline instead (decisions `rev01_tags.txt` in the eval
artifacts).

| style | lines | chars | v1 tiny | PP-OCRv5 shipped | JIS int8 |
|---|---|---|---|---|---|
| bright-on-dark | 163 | 2,016 | 7.99% / 66.3% | 3.82% / 74.8% | 2.48% / 77.9% |
| dark-on-light | 43 | 548 | 81.75% / 0.0% | 2.01% / 76.7% | 1.64% / 83.7% |
| outlined | 74 | 989 | 26.90% / 47.3% | 4.85% / 67.6% | 3.03% / 71.6% |
| name | 53 | 172 | 21.51% / 60.4% | 3.49% / 88.7% | 4.07% / 86.8% |
| coloured | 19 | 195 | 45.64% / 36.8% | 6.15% / 68.4% | 5.13% / 57.9% |
| coloured-name | 8 | 21 | 33.33% / 50.0% | 4.76% / 87.5% | 9.52% / 75.0% |
| furigana | 3 | 64 | 7.81% / 0.0% | 1.56% / 66.7% | 3.12% / 33.3% |
| mincho | 47 | 676 | 1.33% / 85.1% | 0.89% / 87.2% | 1.33% / 85.1% |

v1 was trained only on bright text on dark boxes, and it shows: dark-on-light
lines are 81.8% CER, with not one of the 43 read exactly, and outlined ones
26.9%, against 8.0% for bright-on-dark. The furigana and coloured-name rows have
too few lines to rank models.

### Line finders on vita-test

84 dialogue regions holding 206 scored lines. A detection on an uncertain line
counts neither as a hit nor as a false alarm.

| recognizer | finder | regions rejected | line recall | line precision | pipeline CER | concat CER |
|---|---|---|---|---|---|---|
| v1 tiny | `overlay` (the shipped bright-pixel finder) | 25 | 64.08% | 97.06% | 42.32% | 41.38% |
| v1 tiny | `gradient` (the research's polarity-agnostic finder) | 0 | 88.83% | 100.00% | 43.21% | 40.90% |
| PP-OCRv5 shipped | `overlay` | 25 | 64.08% | 97.06% | 38.34% | 37.55% |
| PP-OCRv5 shipped | `gradient` | 0 | 88.83% | 100.00% | 17.00% | 14.17% |
| JIS int8 | `overlay` | 25 | 64.08% | 97.06% | 37.71% | 36.99% |
| JIS int8 | `gradient` | 0 | 88.83% | 100.00% | 18.64% | 15.12% |

What the finder results show:
- **The 25 regions `overlay` rejects** hold 66 scored lines. 36 of them are
  dark-on-light (of 43 in the set). The other 30 are 25 lines of outlined light
  text and 5 speaker names that share a region with dark-on-light (4) or
  outlined (1) lines.
- **With v1 the finder barely moves the pipeline CER.** v1 reads those 66 lines
  at 65.3% CER even from hand-drawn boxes.
- **With a PP-OCRv5 recognizer, `gradient` cuts the pipeline CER by about half**
  (shipped 38.34% to 17.00%, JIS int8 37.71% to 18.64%). The finder becomes the
  main error source: the JIS int8 model reads hand-drawn boxes at 2.30% but goes
  through `gradient` at 18.64%.
- **Tuning overlap.** The research chose `gradient`'s rules on its 16
  hand-transcribed regions, linefind's `research_a`
  (`eval/linefind/manifests/`), which sit on 4 vita-test screenshots:
  v7721_95057, v19850_122860, v19850_122861 and v21782_112473. The replication
  set `research_b` adds v21782_151627 and v21789_116094. Leaving out every region
  of those 4 games leaves 60 regions and 149 lines:

  | recognizer | hand-drawn boxes | `overlay` | `gradient` |
  |---|---|---|---|
  | v1 tiny | 22.66% | 40.26% | 44.88% |
  | PP-OCRv5 shipped | 2.67% | 36.19% | 19.54% |
  | JIS int8 | 2.28% | 36.36% | 21.83% |

  Recall on those 60 regions is 65.10% for `overlay` and 87.25% for `gradient`,
  so the in-sample screenshots flatter `gradient` by about 3 points. `overlay`
  is a port of the shipped C finder and was not tuned on any of them.

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
  The review reran it on 8,096 non-reserved images, adding synth2's PSP
  backgrounds (`artifacts/synth2/vndb_psp/shots`), and again found no pair at
  0.95 or above. With `--keep-reserved` it finds exactly the 74 vita-test
  screenshots in v1's VNDB cache (`data/vndb/shots`), each at 0.99 or above.
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
