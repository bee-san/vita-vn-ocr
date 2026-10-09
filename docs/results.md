# Results

Real-screenshot numbers come from the shared benchmark, `eval/run.py` (see
`eval/README.md`). Host speed is informational only; nothing here was measured
on a console.

## Baselines

Measured 2026-10-09 between 13:14 and 13:42Z at 049c86d, in one `heavy.sh` slot
of the shared x86 host (5 threads for torch and ONNX Runtime). 049c86d is c4a5b9d
before the rebase onto #3. Re-running v1 on all sets, and v1 and JIS int8 with
both finders, at c4a5b9d (one thread) gave the same predictions on every line.
The other rows go through the same scoring code, which #3 did not touch. Software: Python 3.11.15, torch 2.5.1+cpu, onnxruntime 1.30.0,
manga-ocr 0.1.16, and the ncnn helper `eval/ncnn/vocr_ncnn_rec` (ncnn 3798be64,
one thread, fp16 off, int8 on, the overlay's preprocessing). Decoding is greedy
and the crops are fixed, so a rerun gives the same text. The v1, shipped and JIS
int8 predictions on vita-test match PR #2's runs at c9790d7 line for line, so
each row is a single run.

```sh
M1=$HERD/shared/models/v1/ckpt_best.pt; NC=data/eval/models/ncnn
eval/run.py --model $M1 --set all --out v1.json                   # all = synth-val, vita-test, psp-test
eval/run.py --model ncnn:$NC/ppocr-v5-mobile-shipped --set all
eval/run.py --model ncnn:$NC/ppocr-v5-mobile-jis-int8 --set all
eval/run.py --model ppocr-v5-mobile --set all                     # and ppocr-v5-mobile-jis
eval/run.py --model manga-ocr --set all
eval/run.py --model $M1 --set vita-test,psp-test --finder overlay   # and gradient; the same for the JIS int8 model
```

The sets:
- vita-test: 206 scored lines (2,564 characters) from 16 games.
- psp-test: 245 lines (2,510 characters) from 14 games.
- synth-val: the first 3,000 lines (54,306 characters) of `data/synth/synth_00039.npz`.

Each cell is CER / exact-line share.

| model | weights | peak memory (research) | synth-val | vita-test | psp-test | game-macro CER, vita / psp |
|---|---|---|---|---|---|---|
| v1 tiny (`train/model.py`, 761,771 params, 5,355 classes) | 3,047,084 B fp32 | not measured yet (the runtime's arena report) | 0.32% / 95.80% | 23.75% / 52.43% | 46.65% / 28.16% | 23.74% / 48.85% |
| PP-OCRv5 mobile as shipped in vita-jp-overlay (ncnn fp16, 18,385 classes) | 8,242,276 B | 39.52 MiB tensor-heap peak over the research's 30 boxes; 42.4 MiB worst real region | 2.40% / 73.70% | 3.43% / 75.24% | 10.40% / 53.88% | 3.34% / 10.38% |
| PP-OCRv5 mobile, head pruned to JIS X 0208, int8 (ncnn, 6,513 classes; the research's recommendation) | 2,749,540 B | 14.29 MiB heap peak, 11.77 MiB with the raw buffer freed; 15.90 / 13.38 MiB with every line at the 1024-px clamp; RSS 20.9 MiB | 2.02% / 76.63% | 2.30% / 79.13% | 5.58% / 63.67% | 2.49% / 5.74% |
| PP-OCRv5 mobile, ONNX fp32 (`ppocr-v5-mobile`) | 16,631,306 B | 33.6 MiB liveness at 48×1024 (4,113,194 params) | 2.42% / 73.63% | 3.47% / 74.76% | 10.52% / 52.65% | 3.38% / 10.45% |
| the same, CTC head masked to JIS X 0208 (`ppocr-v5-mobile-jis`) | same file | 16.6 MiB once pruned (2,676,682 params) | 2.41% / 73.80% | 3.43% / 75.24% | 10.40% / 52.65% | 3.34% / 10.30% |
| manga-ocr base (reference only) | 444,023,808 B fp32 (111,005,952 params) | not measured | 3.87% / 59.83% | 8.46% / 55.83% | 14.54% / 39.59% | 8.75% / 15.26% |

The memory figures are the research's measurements in the overlay's own
`vjo-ocr` (`docs/research-findings.md`;
`scratch/int8/sa250291/results_summary.json`), and the ONNX ones are its
liveness estimates. None was measured here.

Host time per line (vita-test / psp-test / synth-val) is informational only:
- v1: 5.2 / 4.2 / 7.0 ms
- shipped ncnn: 39.6 / 30.2 / 60.0 ms
- JIS int8: 60.3 / 44.7 / 79.2 ms
- ONNX: 15.9 / 13.3 / 22.9 ms
- ONNX JIS: 18.2 / 15.8 / 23.2 ms
- manga-ocr: 265 / 260 / 298 ms

Each figure is one pass on a loaded shared host.

Reading the table:

- **int8 beats its own fp32 parent on real screenshots**, though the research
  saw them track within 0.2 points.
  - The float models (the shipped ncnn one with fp16-stored weights, ONNX, ONNX
    JIS) agree with each other within 0.5 points on every style of both sets. The JIS head and the ncnn runtime are
    therefore not the cause. Nor is the squeeze-excitation rewrite, whose float output
    the research found identical. That leaves the int8 quantisation (KL-calibrated
    on about 300 synthetic Aozora-text lines).
  - On psp-test the int8 model has fewer errors than the fp32 JIS head on 71
    lines (156 fewer errors) and more on 29 (35 more). On vita-test the split is
    28 better and 15 worse.
  - On PSP lines the fp32 models drop characters mid-line. For example,
    「戻った時には、破門になっているやも (16 px mincho) comes back as
    「戻っ時に、になっていも, and the int8 model reads it exactly.
  - Why is not known yet. Calibrating on real lines might widen the gap or close it.
- **psp-test is much harder than vita-test** for every model: 1.7 to 3.0 times
  the CER. The screen has half the resolution (106 of 245 lines are under 16 px
  tall), and 110 lines are dark-on-light.
- **For v1 the problem is polarity, not size.** It reads the 44 small
  bright-on-dark PSP lines at 2.8% CER, but the 48 dark-on-light lines that are
  not small at 78.8% and the 62 small ones at 93.4%.
  - The six psp-test games it reads above 70% CER (v12155, v3542, v4147, v3396,
    v11504, v11095) are the six whose text is mostly dark-on-light.
  - Outlined text drawn over artwork is its next weakness: v12940 44.2%, v10856
    37.6%.
- synth-val is v1's own training distribution, so its 0.32% says little about
  real games. It also contains characters PP-OCRv5 cannot emit (42 for the
  shipped head, 54 for the JIS head: ゐ, ヱ, ◯, ※, 呟, 欝...), which count as
  errors. v1 cannot emit one vita-test character (瘴). No model has an
  unemittable psp-test character.
- manga-ocr is a reference, not a candidate (444 MB). It re-flows long lines
  into a block as `data/label_real.py` does. Apart from v1 it is the weakest on
  dark-on-light text and name plates, but it is the best on psp-test's furigana
  lines (7.77%).

### By style

CER / exact-line share per tag; a line can carry several tags. The two ONNX
presets are left out: they are within 0.5 points of the shipped model on every
row.

vita-test:

| style | lines | chars | v1 tiny | PP-OCRv5 shipped | JIS int8 | manga-ocr |
|---|---|---|---|---|---|---|
| bright-on-dark | 142 | 1,809 | 7.46% / 70.4% | 3.48% / 75.4% | 2.38% / 78.2% | 6.08% / 65.5% |
| dark-on-light | 64 | 755 | 62.78% / 12.5% | 3.31% / 75.0% | 2.12% / 81.2% | 14.17% / 34.4% |
| outlined | 72 | 939 | 28.01% / 48.6% | 5.11% / 66.7% | 3.09% / 72.2% | 9.90% / 54.2% |
| name | 53 | 172 | 21.51% / 60.4% | 3.49% / 88.7% | 4.07% / 86.8% | 15.70% / 71.7% |
| coloured | 17 | 186 | 43.01% / 41.2% | 4.84% / 70.6% | 4.30% / 58.8% | 21.50% / 52.9% |
| coloured-name | 7 | 19 | 26.32% / 57.1% | 5.26% / 85.7% | 10.53% / 71.4% | 15.79% / 71.4% |
| furigana | 3 | 64 | 7.81% / 0.0% | 1.56% / 66.7% | 3.12% / 33.3% | 0.00% / 100.0% |
| mincho | 47 | 676 | 1.33% / 85.1% | 0.89% / 87.2% | 1.33% / 85.1% | 2.51% / 74.5% |

psp-test:

| style | lines | chars | v1 tiny | PP-OCRv5 shipped | JIS int8 | manga-ocr |
|---|---|---|---|---|---|---|
| bright-on-dark | 135 | 1,435 | 16.73% / 51.1% | 11.15% / 51.1% | 5.44% / 63.0% | 8.08% / 55.6% |
| dark-on-light | 110 | 1,075 | 86.61% / 0.0% | 9.40% / 57.3% | 5.77% / 64.5% | 23.16% / 20.0% |
| outlined | 69 | 740 | 19.86% / 50.7% | 8.78% / 59.4% | 5.00% / 62.3% | 8.24% / 55.1% |
| name | 62 | 209 | 63.16% / 19.4% | 15.31% / 67.7% | 6.22% / 82.3% | 36.36% / 41.9% |
| coloured | 20 | 214 | 64.95% / 15.0% | 8.88% / 55.0% | 7.94% / 50.0% | 20.09% / 25.0% |
| coloured-name | 7 | 24 | 87.50% / 0.0% | 8.33% / 71.4% | 12.50% / 71.4% | 83.33% / 0.0% |
| furigana | 20 | 206 | 45.15% / 0.0% | 22.33% / 20.0% | 9.71% / 65.0% | 7.77% / 60.0% |
| mincho | 90 | 984 | 44.51% / 30.0% | 11.79% / 47.8% | 5.49% / 60.0% | 13.41% / 37.8% |
| small | 106 | 973 | 56.32% / 31.1% | 9.15% / 55.7% | 5.55% / 67.9% | 15.21% / 38.7% |
| no-box | 4 | 59 | 59.32% / 0.0% | 15.25% / 75.0% | 5.08% / 50.0% | 6.78% / 50.0% |

The furigana, coloured-name and no-box rows have too few lines to rank models.
Per-game numbers are in each result JSON (`sets.<set>.games`).

### Line finders

vita-test has 84 dialogue regions holding 206 scored lines; psp-test has 93
holding 245. The vita-test rows are partly in-sample for `gradient`, as explained
below the table. psp-test is out of sample for both finders.

| recognizer | finder | set | regions rejected | line recall | line precision | pipeline CER | concat CER |
|---|---|---|---|---|---|---|---|
| v1 tiny | `overlay` (the shipped bright-pixel finder) | vita-test | 25 | 64.08% | 97.08% | 42.32% | 41.38% |
| v1 tiny | `overlay` | psp-test | 41 | 45.71% | 99.11% | 68.45% | 67.49% |
| v1 tiny | `gradient` (the research's polarity-agnostic finder) | vita-test | 0 | 88.83% | 100.00% | 43.21% | 40.90% |
| v1 tiny | `gradient` | psp-test | 0 | 78.78% | 98.97% | 75.06% | 73.27% |
| JIS int8 | `overlay` | vita-test | 25 | 64.08% | 97.08% | 37.71% | 36.99% |
| JIS int8 | `overlay` | psp-test | 41 | 45.71% | 99.11% | 61.12% | 58.77% |
| JIS int8 | `gradient` | vita-test | 0 | 88.83% | 100.00% | 18.64% | 15.12% |
| JIS int8 | `gradient` | psp-test | 0 | 78.78% | 98.97% | 30.60% | 27.13% |

What the finder results show:
- **With a good recognizer, the finder is now the main source of pipeline
  error.** The JIS int8 model reads hand-drawn boxes at 2.30% (Vita) and 5.58%
  (PSP), but its gradient-finder pipeline CER is 18.64% and 30.60%.
- **Switching from `overlay` to `gradient` halves the pipeline CER** for that
  recognizer. It doesn't for v1, which can't read the extra dark-on-light lines
  `gradient` finds.
- **What `overlay` rejects.** It throws away whole regions:
  - vita-test: 25 regions holding 66 lines, 46 of them dark-on-light;
  - psp-test: 41 regions holding 109 lines, 85 of them dark-on-light (of 110 in
    the set). These include every line of v11504, v12155, v12940 and v3542, and
    14 of v11095's 20.
- **`gradient` rejects nothing but still misses lines:** 11% of the Vita lines
  and 21% of the PSP ones.
- **Tuning overlap on vita-test.** The research chose `gradient`'s rules on its
  16 hand-transcribed regions, which are linefind's `research_a`. Together with
  the replication `research_b`, those sets sit on 6 vita-test screenshots of v7721,
  v19850, v21782 and v21789 (`docs/linefind.md`, `eval/linefind/manifests/`).
  Leaving out every region of those 4 games leaves 60 regions and 149 lines:
  - v1: `overlay` 40.26% and `gradient` 44.88% pipeline CER (22.66% from
    hand-drawn boxes);
  - JIS int8: `overlay` 36.36% and `gradient` 21.83% (2.28% from hand-drawn
    boxes);
  - recall: 65.10% for `overlay` and 87.25% for `gradient`.
  So the in-sample screenshots flatter `gradient` by about 3 points on vita-test.
  `overlay` is a port of the shipped C finder and was not tuned here.

## Test-set hygiene

- **Split by game and capture session.** Each manifest line records its game
  (`vn`) and session (`session`, the VNDB release its screenshot belongs to). The
  16 vita-test games come from 16 releases and the 14 psp-test games from 15; all
  31 are listed in `eval/manifests/test_games.json` (`sessions`). Two games are in
  both sets from separate releases: v10856 (Vita r22078, PSP r22079) and v13072
  (Vita r56156, PSP r28150 and r28151). No session is in both sets.
- **Related games are reserved too.** Set games' direct VNDB relations that reuse
  their scenes, UI or fonts (alternative versions, originals, fandiscs, sequels,
  prequels) add 15 VNs to `all` (`family`): v3, v10606, v18160, v18717, v19338,
  v20226, v24312, v25914, v26075 and v40027 for vita-test, and v2251, v9915,
  v11639, v11709 and v12159 for psp-test (v11707 was already reserved).
  `all` now holds 172 ids.
- **linefind's sets (#3).** Its dev and holdout manifests share no screenshot,
  session or set game with vita-test or psp-test. Its tuning sets `research_a` and
  `research_b` include 6 vita-test screenshots, which is why the finder numbers
  above carry a caveat.
- **Near-duplicate audit.** `eval/tools/leakcheck.py` compared the 157 test
  screenshots (74 vita-test, 83 psp-test) with 7,603 non-reserved pool images
  (v1's background pool `data/vndb/shots` plus realdata's vita, psp and hires
  shots; 184 images of reserved games skipped). No pair reached 0.95 thumbnail
  correlation; the best was between 0.90 and 0.95, for one screenshot. A test
  screenshot with its text painted out, recompressed and rescaled still scores at
  least 0.958 against itself, and random pairs from different games peak at 0.78.
  With `--keep-reserved` (7,787 images) there are 74 pairs, every one a vita-test
  screenshot found in v1's VNDB cache under its own name (correlation ≥ 0.99), so
  any user of that cache must skip `all`. No psp-test screenshot is in any pool.
- **What the training pools hold today** (checked 2026-10-09 13:05Z). realdata's
  labels have no line from a set game, but 170 of their 24,211 rows come from
  family-reserved VNs (86 of them from the five psp-test relations), and
  `real_train.npz` has 61 such crops (v10606, v11639, v11709, v12159, v2251, v3,
  v9915), `real_val.npz` 10 (v18717). v1's own `real_train` has 12 lines from v3
  and v18717, both relations of v7721, which v1 reads at 68.55% CER, so there is
  no sign they helped it.
- **Known leak into v1.** v16566 and v19850 are not `is_test_vn` games, so
  `synth/render.py` could use 9 of the 74 vita-test screenshots as unlabelled
  backgrounds for v1's shards. v1's CER on those 28 lines is 79.3%, against 15.7%
  on the other 178, so the leak did not inflate it. Both games are reserved now;
  later shards must skip `all`.

## Not measured yet

- Any console speed or memory figure, and v1's peak memory (the runtime's arena report).
- Repeated timing samples: each ms/line figure is one pass on a shared, loaded host.
- The reference presets `ppocr-v5-server` and `ppocr-v6-small[-jis]` (wired in, not baselined).
- Why int8 beats fp32 on real lines, and whether calibrating on real crops changes it.
- linefind's `grad` finder (#3, `synth/common.find_lines(region, "grad")`, which inverts
  dark-text crops) is not wired in as a `--finder` name. `eval/linefind/evaluate.py
  --eval-set vita-test` scores it.
