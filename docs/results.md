# Results

Real-screenshot numbers come from the shared benchmark, `eval/run.py` (see
`eval/README.md`). Host speed is informational only; nothing here was measured
on a console.

## The release models (train2, v0.2.0)

`train2` pretrains on NVIDIA OCR-Synthetic-Multilingual-v1 Japanese line crops (nvsynth) mixed
with synth2's VN boxes, then fine-tunes on teacher-labelled real crops. Method, ablation,
per-style tables and the selection rule are in `docs/train2.md`.

Three models are published as `v0.2.0`. The numbers are int8 through the C runtime, scored by
`eval/run.py`. The C runtime decodes the same text as the PyTorch checkpoints on every test line
(0 argmax-mismatch frames in f32, w8f and w8i; the logits agree within the parity test's
tolerance, not bit for bit), so it scores the same CER.

| model | width / vocab | int8 blob | arena (w8f / w8i) | vita-test CER / exact | psp-test CER / exact | est. A9 ms, 640-px line (w8i) |
|---|---|---|---|---|---|---|
| bar: JIS-trimmed int8 PP-OCRv5 mobile | — | 2,749,540 B | 13.4 MiB peak (research) | 2.30% / 79.1% | 5.34% / 66.1% | — |
| **H15 (recommended)** | 1.5 / 5,355 | 1,155,968 B | 173,472 / 182,944 B | **1.56% / 83.5%** | **2.63% / 82.4%** | 166-302 ms |
| F20 | 2.0 / 5,355 | 1,558,016 B | 228,512 / 241,056 B | 1.79% / 84.0% | 2.79% / 79.6% | 235-416 ms |
| FL10 (PSP-lean) | 1.0 / 3,500 | 657,664 B | 118,432 / 124,832 B | 2.30% / 77.7% | 5.18% / 69.8% | 94-178 ms |

Intervals are 95% paired bootstraps over lines; differences are in points of CER.

- **H15 beats the bar on both sets.**
  - vita-test −0.74 (CI −1.44 to −0.04), psp-test −2.71 (CI −4.05 to −1.36).
  - It uses 0.42× the bar's int8 weight bytes and, with the line finder, about 1.42 MB of memory
    on the console.
  - Its CER is 15× and 18× lower than v1's (int8: 23.87% / 46.57%).
- **F20 beats the bar on psp-test** (−2.55, CI −3.84 to −1.30). On vita-test it is −0.51 (CI
  −1.33 to +0.37).
- **FL10 ties the bar on vita-test** (59 errors each) and is −0.16 on psp-test (CI −1.66 to
  +1.42).
- **The per-style criterion is not met.**
  - Every model is above the bar on vita `coloured`, `coloured-name` and `name`, and on psp
    `name` and `no-box`. These are mostly short rare-kanji crops.
  - H15 and FL10 are also above it on psp `furigana`, and FL10 on four more styles.
  - Most of these differences, on 4 to 62 lines, are within noise.
- **The NVIDIA + synth2 pretrain does most of the work.** At width 1.5 alone it gets 1.72% /
  3.19%.
- **Fine-tuning on real crops lowers the selection CER (teacher_val) by 0.9 points at width
  1.5.** On the test sets the gain is within noise (pooled −0.35, CI −0.76 to +0.04).
- **Frame-level distillation from PP-OCRv6 small did not help.** F15, H15's recipe plus the KD
  term, is +0.53 worse on teacher_val (CI +0.32 to +0.74) and level on the test sets.
- **The A9 times are estimates** from the Vita compiler's machine code (`vita/a9_cycles.py`).
  Nothing has run on a console.

## Baselines

Measured 2026-10-09 on the shared x86 host with:
- Python 3.11.15, torch 2.5.1+cpu, onnxruntime 1.30.0 and manga-ocr 0.1.16;
- manga-ocr's weights `kha-white/manga-ocr-base` at commit aa6573bd, and the
  PP-OCRv5 mobile ONNX file that rapidocr 3.10.0 downloads (sha256 5825fc7e…;
  `eval/backends.py` pins both);
- the ncnn helper `eval/ncnn/vocr_ncnn_rec` (ncnn 3798be64, one thread, fp16
  off, int8 on, the overlay's preprocessing).

Decoding is greedy and the crops are fixed, so a rerun gives identical text, and
each row is a single run.

How the rows were produced:
- **Author's runs.** PR #5's author measured every row at 049c86d (c4a5b9d before
  the rebase onto #3) between 13:14 and 13:42Z, in one `heavy.sh` slot (5 threads
  for torch and ONNX Runtime).
- **Independent re-runs.** The PR #2 review re-ran the vita-test and synth-val rows
  of v1 and both ncnn models. It re-fetched the screenshots from VNDB and
  sha256-checked them, rebuilt the ncnn helper, and re-copied and hash-checked the
  models with `prepare_models.py`. The PR #5 review re-ran every other row with one
  thread, after re-fetching and sha256-checking the psp-test screenshots. Both got
  the same prediction on every line.
- **One exception.** The review checked manga-ocr's synth-val figure on the first
  600 lines only (4.02% / 60.17%). A full run takes about 50 min on one thread.
- **psp-test corrections.** The PR #5 review then fixed 10 psp-test entries
  (`psp/decisions/rev01.txt` in the eval artifacts):
  - five boxes that cut off the closing 」 or 。;
  - one 【 that had been typed as `[`;
  - four polarity tags.

  It re-ran every psp-test row at 7a88cc4, and the psp-test figures below come
  from those runs. Only the five re-boxed lines read differently.
  - 7a88cc4 is b116906 before the rebase onto #9.
  - #9 leaves the default scoring alone, and v1 and JIS int8 give the same
    psp-test predictions at 1ecd583.

```sh
M1=$HERD/shared/models/v1/ckpt_best.pt; NC=data/eval/models/ncnn
eval/run.py --model $M1 --set all --out v1.json                   # all = synth-val, vita-test, psp-test
eval/run.py --model ncnn:$NC/ppocr-v5-mobile-shipped --set all
eval/run.py --model ncnn:$NC/ppocr-v5-mobile-jis-int8 --set all
eval/run.py --model ppocr-v5-mobile --set all                     # and ppocr-v5-mobile-jis
eval/run.py --model manga-ocr --set all
eval/run.py --model $M1 --set vita-test,psp-test --finder overlay   # and gradient; the same for both ncnn models
```

The sets:
- vita-test: 206 scored lines (2,564 characters) from 16 games.
- psp-test: 245 lines (2,510 characters) from 14 games.
- synth-val: the first 3,000 lines (54,306 characters) of `data/synth/synth_00039.npz`.

Each cell is CER / exact-line share.

| model | weights | peak memory (research) | synth-val | vita-test | psp-test | game-macro CER, vita / psp |
|---|---|---|---|---|---|---|
| v1 tiny (`train/model.py`, 761,771 params, 5,355 classes) | 3,047,084 B fp32 | not measured yet (the runtime's arena report) | 0.32% / 95.80% | 23.75% / 52.43% | 46.45% / 30.20% | 23.74% / 48.69% |
| PP-OCRv5 mobile as shipped in vita-jp-overlay (ncnn fp16, 18,385 classes) | 8,242,276 B | 39.52 MiB tensor-heap peak over the research's 30 boxes; 42.4 MiB worst real region | 2.40% / 73.70% | 3.43% / 75.24% | 10.04% / 55.10% | 3.34% / 10.06% |
| PP-OCRv5 mobile, head pruned to JIS X 0208, int8 (ncnn, 6,513 classes; the research's recommendation) | 2,749,540 B | 14.29 MiB heap peak, 11.77 MiB with the raw buffer freed; 15.90 / 13.38 MiB with every line at the 1024-px clamp; RSS 20.9 MiB | 2.02% / 76.63% | 2.30% / 79.13% | 5.34% / 66.12% | 2.49% / 5.52% |
| PP-OCRv5 mobile, ONNX fp32 (`ppocr-v5-mobile`) | 16,631,306 B | 33.6 MiB liveness at 48×1024 (4,113,194 params) | 2.42% / 73.63% | 3.47% / 74.76% | 10.16% / 53.88% | 3.38% / 10.13% |
| the same, CTC head masked to JIS X 0208 (`ppocr-v5-mobile-jis`) | same file | 16.6 MiB once pruned (2,676,682 params) | 2.41% / 73.80% | 3.43% / 75.24% | 10.04% / 53.88% | 3.34% / 9.99% |
| manga-ocr base (reference only) | 444,023,808 B fp32 (111,005,952 params) | not measured | 3.87% / 59.83% | 8.46% / 55.83% | 14.38% / 40.82% | 8.75% / 15.13% |

Where the memory figures come from:
- They are the research's measurements in the overlay's own `vjo-ocr`
  (`docs/research-findings.md`; `scratch/int8/sa250291/results_summary.json`).
- The ONNX figures are the research's liveness estimates.
- None was measured here.

Host time per line (vita-test / psp-test / synth-val) is informational only:
- v1: 5.2 / 4.2 / 7.0 ms
- shipped ncnn: 39.6 / 30.2 / 60.0 ms
- JIS int8: 60.3 / 44.7 / 79.2 ms
- ONNX: 15.9 / 13.3 / 22.9 ms
- ONNX JIS: 18.2 / 15.8 / 23.2 ms
- manga-ocr: 265 / 260 / 298 ms

These are the author's 5-thread runs, one pass each on a loaded shared host.

Reading the table:

- **int8 beats its own fp32 parent on real screenshots**, though the research
  saw the two track within 0.2 points.
  - The float models agree with each other within 0.5 points on every style of
    both sets: the shipped ncnn one (fp16-stored weights), ONNX and ONNX JIS. So
    the JIS head and the ncnn runtime are not the cause.
  - Nor is the squeeze-excitation rewrite, whose float output the research found
    identical. That leaves the int8 quantisation, which was KL-calibrated on about
    300 synthetic Aozora-text lines.
  - Against the fp32 JIS head, line by line:
    - psp-test: int8 has fewer errors on 71 lines (153 fewer in total) and more on
      29 (35 more);
    - vita-test: 28 lines better and 15 worse.
  - On PSP lines the fp32 models drop characters mid-line. For example, the 16 px
    mincho line 「戻った時には、破門になっているやも comes back from ONNX as
    「戻っ時に、になっていも. The int8 model reads it exactly.
  - Why this happens is not known yet. Calibrating on real lines might widen the
    gap or close it.
- **psp-test is much harder than vita-test** for every model, at 1.7 to 2.9 times
  the CER.
  - The screen has half the resolution: 106 of the 245 lines are under 16 px tall.
  - 110 lines are dark-on-light.
- **For v1 the problem is polarity, not size.**
  - On psp-test, v1 reads the 44 small bright-on-dark lines at 2.5% CER.
  - Dark-on-light lines are far worse: 79.7% for the 48 that are not small and
    93.4% for the 62 small ones.
  - The six psp-test games it reads above 70% CER (v12155, v3542, v4147, v3396,
    v11504, v11095) are the six whose text is mostly dark-on-light.
  - Outlined text drawn over artwork is its next weakness: v12940 44.2%, v10856
    37.6%.
- **synth-val is v1's own training distribution**, so its 0.32% says little about
  real games.
  - It also contains characters PP-OCRv5 cannot emit, which count as errors: 42
    for the shipped head and 54 for the JIS head (ゐ, ヱ, ◯, ※, 呟, 欝...).
  - v1 cannot emit one vita-test character (瘴). No model has an unemittable
    psp-test character.
- **Why train.py reports 0.34%.** The same checkpoint scores 0.34% / 95.6% on the
  same 3,000 lines in train.py (`docs/v1-metrics.jsonl`).
  - The difference is in the scoring, not the model. train.py compares class ids,
    while eval compares `metrics.norm()` text, which folds width and dash
    variants.
  - Measured on these lines:
    - class ids, one line per pass: 0.35% / 95.43%;
    - the same predictions normalised: 0.32% / 95.80% (eval);
    - train.py's padded batches: 0.34% / 95.63%.
- **manga-ocr is a reference, not a candidate** (444 MB).
  - It re-flows long lines into a block, as `data/label_real.py` does.
  - Apart from v1, it is the weakest model on dark-on-light text and name plates.
  - It is the best on psp-test's furigana lines (7.77%).

### By style

CER / exact-line share per tag; a line can carry several tags.
- Polarity is the text fill against its background. For outlined text it is the
  fill against its outline (`eval/README.md`).
- The PR #2 review corrected 26 vita-test tags that had been decided over the
  outline, and the PR #5 review corrected 4 psp-test tags (`rev01` decisions in
  the eval artifacts).
- The two ONNX presets are left out, because they are within 0.5 points of the
  shipped model on every row.

vita-test:

| style | lines | chars | v1 tiny | PP-OCRv5 shipped | JIS int8 | manga-ocr |
|---|---|---|---|---|---|---|
| bright-on-dark | 163 | 2,016 | 7.99% / 66.3% | 3.82% / 74.8% | 2.48% / 77.9% | 6.35% / 65.6% |
| dark-on-light | 43 | 548 | 81.75% / 0.0% | 2.01% / 76.7% | 1.64% / 83.7% | 16.24% / 18.6% |
| outlined | 74 | 989 | 26.90% / 47.3% | 4.85% / 67.6% | 3.03% / 71.6% | 9.40% / 55.4% |
| name | 53 | 172 | 21.51% / 60.4% | 3.49% / 88.7% | 4.07% / 86.8% | 15.70% / 71.7% |
| coloured | 19 | 195 | 45.64% / 36.8% | 6.15% / 68.4% | 5.13% / 57.9% | 23.59% / 52.6% |
| coloured-name | 8 | 21 | 33.33% / 50.0% | 4.76% / 87.5% | 9.52% / 75.0% | 14.29% / 75.0% |
| furigana | 3 | 64 | 7.81% / 0.0% | 1.56% / 66.7% | 3.12% / 33.3% | 0.00% / 100.0% |
| mincho | 47 | 676 | 1.33% / 85.1% | 0.89% / 87.2% | 1.33% / 85.1% | 2.51% / 74.5% |

v1 was trained only on bright text on dark boxes, and it shows on vita-test:
- dark-on-light lines are 81.8% CER, and not one of the 43 is read exactly;
- outlined lines are 26.9%;
- bright-on-dark lines are 8.0%.

psp-test:

| style | lines | chars | v1 tiny | PP-OCRv5 shipped | JIS int8 | manga-ocr |
|---|---|---|---|---|---|---|
| bright-on-dark | 135 | 1,409 | 14.90% / 54.8% | 10.36% / 54.1% | 5.18% / 66.7% | 7.66% / 57.8% |
| dark-on-light | 110 | 1,101 | 86.83% / 0.0% | 9.63% / 56.4% | 5.54% / 65.5% | 22.98% / 20.0% |
| outlined | 69 | 740 | 19.86% / 50.7% | 8.78% / 59.4% | 5.00% / 62.3% | 8.24% / 55.1% |
| name | 62 | 209 | 63.16% / 19.4% | 14.83% / 69.4% | 5.74% / 83.9% | 36.36% / 41.9% |
| coloured | 20 | 214 | 64.95% / 15.0% | 8.41% / 60.0% | 7.48% / 55.0% | 20.09% / 25.0% |
| coloured-name | 7 | 24 | 87.50% / 0.0% | 4.17% / 85.7% | 8.33% / 85.7% | 83.33% / 0.0% |
| furigana | 20 | 206 | 45.15% / 0.0% | 22.33% / 20.0% | 9.71% / 65.0% | 7.77% / 60.0% |
| mincho | 90 | 984 | 44.00% / 35.6% | 10.98% / 50.0% | 4.98% / 65.6% | 13.01% / 41.1% |
| small | 106 | 973 | 56.22% / 32.1% | 9.04% / 56.6% | 5.45% / 68.9% | 15.21% / 38.7% |
| no-box | 4 | 59 | 59.32% / 0.0% | 15.25% / 75.0% | 5.08% / 50.0% | 6.78% / 50.0% |

The furigana, coloured-name and no-box rows have too few lines to rank models.
Per-game numbers are in each result JSON (`sets.<set>.games`).

### Line finders

The sets' dialogue regions:
- vita-test has 84 regions holding 206 scored lines;
- psp-test has 93 regions holding 245 lines.

A detection on an uncertain line counts neither as a hit nor as a false alarm.
psp-test has no uncertain lines.

The vita-test rows are partly in-sample for `gradient`, as explained below the
table. psp-test is out of sample for both finders.

| recognizer | finder | set | regions rejected | line recall | line precision | pipeline CER | concat CER |
|---|---|---|---|---|---|---|---|
| v1 tiny | `overlay` (the shipped bright-pixel finder) | vita-test | 25 | 64.08% | 97.06% | 42.32% | 41.38% |
| v1 tiny | `overlay` | psp-test | 41 | 45.71% | 99.11% | 68.45% | 67.49% |
| v1 tiny | `gradient` (the research's polarity-agnostic finder) | vita-test | 0 | 88.83% | 100.00% | 43.21% | 40.90% |
| v1 tiny | `gradient` | psp-test | 0 | 78.78% | 98.97% | 75.06% | 73.27% |
| PP-OCRv5 shipped | `overlay` | vita-test | 25 | 64.08% | 97.06% | 38.34% | 37.55% |
| PP-OCRv5 shipped | `overlay` | psp-test | 41 | 45.71% | 99.11% | 62.67% | 60.72% |
| PP-OCRv5 shipped | `gradient` | vita-test | 0 | 88.83% | 100.00% | 17.00% | 14.17% |
| PP-OCRv5 shipped | `gradient` | psp-test | 0 | 78.78% | 98.97% | 31.87% | 28.33% |
| JIS int8 | `overlay` | vita-test | 25 | 64.08% | 97.06% | 37.71% | 36.99% |
| JIS int8 | `overlay` | psp-test | 41 | 45.71% | 99.11% | 61.12% | 58.77% |
| JIS int8 | `gradient` | vita-test | 0 | 88.83% | 100.00% | 18.64% | 15.12% |
| JIS int8 | `gradient` | psp-test | 0 | 78.78% | 98.97% | 30.60% | 27.37% |

What the finder results show:
- **With a good recognizer, the finder is now the main source of pipeline
  error.** The JIS int8 model reads hand-drawn boxes at 2.30% (Vita) and 5.34%
  (PSP). Through `gradient`, its pipeline CER is 18.64% and 30.60%.
- **With a PP-OCRv5 recognizer, `gradient` roughly halves the pipeline CER of
  `overlay`** on both sets (JIS int8: 37.71% to 18.64% on Vita, 61.12% to 30.60%
  on PSP).
  - It does not help v1, which cannot read the extra dark-on-light lines
    `gradient` finds.
  - v1 reads the 66 Vita lines `overlay` rejects at 65.3% CER even from
    hand-drawn boxes.
- **What `overlay` rejects.** It throws away whole regions:
  - vita-test: 25 regions holding 66 lines. 36 of them are dark-on-light (of 43
    in the set). The other 30 are 25 lines of outlined light text and 5 speaker
    names that share a region with dark-on-light (4) or outlined (1) lines.
  - psp-test: 41 regions holding 109 lines, 85 of them dark-on-light (of 110 in
    the set). They include every line of v11504, v12155, v12940 and v3542, and 14
    of v11095's 20.
- **`gradient` rejects nothing but still misses lines:** 11% of the Vita lines
  and 21% of the PSP ones.
- **Tuning overlap on vita-test.** The research chose `gradient`'s rules on its
  16 hand-transcribed regions, linefind's `research_a` (`eval/linefind/manifests/`).
  - Those regions sit on 4 vita-test screenshots: v7721_95057, v19850_122860,
    v19850_122861 and v21782_112473.
  - The replication set `research_b` adds v21782_151627 and v21789_116094.
  - Leaving out every region of those 4 games leaves 60 regions and 149 lines,
    which score:

  | recognizer | hand-drawn boxes | `overlay` | `gradient` |
  |---|---|---|---|
  | v1 tiny | 22.66% | 40.26% | 44.88% |
  | PP-OCRv5 shipped | 2.67% | 36.19% | 19.54% |
  | JIS int8 | 2.28% | 36.36% | 21.83% |

  - Recall on those 60 regions is 65.10% for `overlay` and 87.25% for
    `gradient`. So the in-sample screenshots flatter `gradient` by about 3
    points on vita-test.
  - `overlay` is a port of the shipped C finder and was not tuned on any of
    them.

## Test-set hygiene

- **Split by game and capture session.** Each manifest line records its game
  (`vn`) and session (`session`, the VNDB release its screenshot belongs to).
  - The 16 vita-test games come from 16 releases and the 14 psp-test games from
    15. All 31 are listed in `eval/manifests/test_games.json` (`sessions`).
  - Two games are in both sets, from separate releases:
    - v10856: Vita r22078, PSP r22079;
    - v13072: Vita r56156, PSP r28150 and r28151.
  - No session is in both sets.
- **Related games are reserved too.** The set games' direct VNDB relations that
  reuse their scenes, UI or fonts (alternative versions, originals, fandiscs,
  sequels, prequels) add 15 VNs to `all` (`family`):
  - for vita-test: v3, v10606, v18160, v18717, v19338, v20226, v24312, v25914,
    v26075 and v40027;
  - for psp-test: v2251, v9915, v11639, v11709 and v12159 (v11707 was already
    reserved).

  `all` now holds 172 ids.
- **linefind's sets (#3).** Its dev and holdout manifests share no screenshot,
  session or set game with vita-test or psp-test. Its tuning sets `research_a` and
  `research_b` include 6 vita-test screenshots, which is why the finder numbers
  above carry a caveat.
- **Near-duplicate audit.** `eval/tools/leakcheck.py` compared the 157 test
  screenshots (74 vita-test, 83 psp-test) with every screenshot pool on the shared
  host:
  - v1's background pool `data/vndb/shots`;
  - realdata's vita, psp and hires shots;
  - synth2's PSP backgrounds `artifacts/synth2/vndb_psp`;
  - linefind's `psp_dev/shots`;
  - teacher's `spot`;
  - `data/real_v0_discarded`.

  The results:
  - Against the 8,472 non-reserved images (162 images of reserved games
    skipped), no pair reached 0.95 thumbnail correlation. The best was between
    0.90 and 0.95, for one screenshot.
  - For scale, a test screenshot with its text painted out, recompressed and
    rescaled still scores at least 0.958 against itself, while random pairs from
    different games peak at 0.78.
  - With `--keep-reserved` (8,634 images) there are 74 pairs. Every one is a
    vita-test screenshot found under its own name in v1's VNDB cache (correlation
    ≥ 0.99), so any user of that cache must skip `all`.
  - An exact sha256 comparison over the same pools and v1's line crops finds the
    same 74 and nothing else. No psp-test screenshot is in any pool.
  - synth2's `vndb_psp` directory does hold 32 files of four family-reserved VNs
    (v9915, v10606, v11639, v12159). Its `backgrounds.json`, the list the
    renderer uses, names none of them.
- **What the training pools hold** (checked 2026-10-09 at 17:40Z, after #7).
  - None of these has a line from any reserved VN (`all` or `is_test_vn`):
    - realdata's 24,041 label rows;
    - `real_train.npz` (9,811 lines), `real_val.npz` (1,495) and
      `real_unlabelled.npz` (11,174);
    - teacher's train, val and distillation npz files.
  - v1's own `real_train` has 12 lines from v3 and v18717, both relations of
    v7721. v1 reads v7721 at 68.55% CER, so there is no sign they helped it.
- **Known leak into v1.** v16566 and v19850 are not `is_test_vn` games, so
  `synth/render.py` could use 9 of the 74 vita-test screenshots as unlabelled
  backgrounds for v1's shards.
  - v1's CER on those 28 lines is 79.3%, against 15.7% on the other 178, so the
    leak did not inflate it.
  - Both games are reserved now, and later shards must skip `all`.

## PSP build in PPSSPP

`psp/` runs the C runtime's PSP build (pspdev, Allegrex, `-O2 -G0`) as a test EBOOT in
PPSSPPHeadless 1.20.4 (`psp/README.md`; `psp/run_all.sh`). Its logits are bit-identical to the
host runtime's on every line of vita-test, psp-test and synth-val (first 300) in f32, w8f and
w8i. Through `eval/run.py --model 'cmd:python psp/ppsspp_cli.py v1_w8.vocr --mode w8f --stdin'`
it returns the host's string on all 451 real lines: int8 v1 is 23.87% CER on vita-test and
46.57% on psp-test, as on the host (manifests byte-identical to main's: psp-test sha256
2b8db084…, vita-test 5fa68d04…).

| v1, int8 (w8f, tile 16) | blob | arena | emulated ms per frame | median / p90 psp-test line |
|---|---|---|---|---|
| scalar build (default) | 859,904 B | 353,536 B (118,432 at tile 4) | 12.07 | 0.38 / 0.73 s |
| VFPU build (`make -C psp VFPU=1`) | same | same | 7.96 | 0.25 / 0.47 s |

PPSSPP's time is emulated (about one cycle per instruction at 333 MHz, no cache model), not
console time. The console estimate in `psp/README.md` is 1.05-1.45× for the scalar build.

## Not measured yet

- Any console speed or memory figure (`psp/README.md` has the PPSSPP-emulated PSP figures and
  v1's measured arena), and v1's peak memory on a Vita.
  For the PS Vita there is so far a bit-exact check in the Vita3K emulator and a
  Cortex-A9 cycle estimate (113-212 ms per 640-px line for v1 in w8i at 444 MHz),
  both in `vita/README.md`; neither is a console measurement.
- Repeated timing samples: each ms/line figure is one pass on a shared, loaded host.
- The reference presets `ppocr-v5-server` and `ppocr-v6-small[-jis]` (wired in, not baselined).
- Why int8 beats fp32 on real lines, and whether calibrating on real crops changes it.
- linefind's `grad` finder (#3, `synth/common.find_lines(region, "grad")`, the
  console's finder). #9 wired it in as `--finder console`, with `--polarity
  console` for the console's inversion, but it has no baseline here yet.
- A few visible psp-test lines that were never detected as candidates:
  - the ？？？ name plates in v3396_55752 and v11504_52181;
  - 「！！」 in v11708_55782;
  - the names 玲 in v12155_64363 and チャーリー in v15910_89382.

  Adding them would only grow the set.
