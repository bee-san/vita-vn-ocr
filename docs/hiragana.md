# FL10 hiragana fine-tuning

This implements the host evaluation, checkpoint recovery, data preparation and
matched experiment workflow. It has **not produced an accepted replacement
model**. The original five training source snapshots are absent locally; the
9 × 5,000-step experiment has not run. Rebuild those inputs before comparing
the control, targeted and AnimeText arms. Existing released weights remain the
baseline. CLANNAD was in the original training collection, so these new sessions
measure additional known-game captures, not an unseen-game benchmark.

## Measured baseline, 2026-10-11

The released FL10 W8I model was evaluated with console polarity, using the
existing shared screenshots and two new native 960×544 Vita3K capture blocks.
The development block precedes the held-out block; their screenshots and text
were hand checked separately, and blank/fading/typewriter-transition frames
were excluded before scoring. Each block has its own session and split.
Private images and transcriptions remain under ignored `build/hiragana/`.
The new captures are a small opening-scene sample, not coverage of the whole game.

| Set | Lines | Full errors / chars | Full CER | Hiragana errors / chars | Hiragana CER | Finder pipeline CER |
|---|---:|---:|---:|---:|---:|---:|
| Shared Vita | 206 | 61 / 2,564 | 2.38% | 15 / 1,417 | 1.06% | 16.46% |
| Shared PSP | 245 | 133 / 2,510 | 5.30% | 30 / 1,382 | 2.17% | 36.18% |
| CLANNAD development | 19 | 5 / 207 | 2.42% | 2 / 127 | 1.57% | 4.83% |
| CLANNAD held out | 25 | 7 / 388 | 1.80% | 3 / 229 | 1.31% | 6.70% |

Development substitutions were `ず → す` and `っ → つ`. Target derivation uses
only that development report. Shared and held-out test reports cannot be passed
to `--target-report`. Full CER and finder results stay visible alongside the
hiragana diagnostic. Hiragana scoring aligns normalized **full text** first;
its numerator is substitutions/deletions of reference hiragana plus hiragana
insertions. Substitutions into hiragana from another script are reported
separately. Small kana, dakuten and handakuten remain distinct.

Host: Intel Core Ultra 7 165U, x86_64 Linux, Python 3.11.15, dependencies from
`requirements-ci.lock`; native scalar C library, W8I, max width 1,024. Accuracy
is deterministic; timings were single host samples and are not console latency
measurements. Blob SHA-256:
`500e0b842e5f0cb326f55e7d991a52ab66dd24977ccc89bd188228aedc825e84`.
Model and arena are 657,664 and 124,832 bytes respectively, queried through C.
Aggregate evidence is in `eval/hiragana_baseline_20261011.json`.

## Recover the trainable model

```sh
.venv-ci/bin/python train/restore_blob.py --blob /private/FL10_f32.vocr \
  --out build/hiragana/fl10_init.pt --vocab-out build/hiragana/vocab.txt
```

Restoration reverses the collapse/matrix layouts and preserves the embedded
vocabulary, including its seven kanji differences from `data/vocab_3500.txt`.
Equivalent affine BatchNorm layers represent the folded biases. Historical BN
statistics and optimizer state cannot be recovered; running statistics remain
frozen, while affine parameters can train. `train2.py` honors the recovery flag
automatically and rejects a different vocabulary order.

Measured recovery checks: C/PyTorch argmax parity on fixed random inputs of
widths 64, 256, 640 and 1,024 (maximum logit difference 0.000011921);
re-exported int8 weights retain every prediction on the 451 shared lines and
both byte limits. A three-step restored-model **fixture smoke** verified frozen
running statistics and trainable parameters. It is not an accuracy experiment.

## Import and audit AnimeText

```sh
.venv-ci/bin/python data/import_animetext.py import --vocab build/hiragana/vocab.txt \
  --out build/hiragana/animetext --max-crops 20000
.venv-ci/bin/python data/import_animetext.py audit --input build/hiragana/animetext \
  --out build/hiragana/audit --font /private/OFL-font.ttf --n 200
# Inspect all sheets and fill audit.json: correct, layout_ok, and any uncertainty/correction.
.venv-ci/bin/python data/import_animetext.py curate --input build/hiragana/animetext \
  --audit build/hiragana/audit/audit.json --vocab build/hiragana/vocab.txt \
  --out build/hiragana/animetext-curated
```

The source is [JustANormalTinkerer/animetext-ocr](https://huggingface.co/datasets/JustANormalTinkerer/animetext-ocr),
revision `9c70c6b5609c4378d46ac0df4927c4119ebfd83d`, CC-BY-NC-SA-4.0, for this
private personal experiment. YOLO detection weights are not needed for these
recognition crops and are not added to the console.

The importer verifies each upstream shard's LFS SHA-256, reads Parquet in
batches, and retains Japanese horizontal lines or single glyphs with sequence
confidence ≥0.95 and minimum token confidence ≥0.9. It rejects whitespace,
out-of-vocabulary labels, incompatible aspect ratios/multiple bands, and
unreachable CTC labels, including adjacent repeated characters. Native crop
resolution is retained, and console polarity is recorded. Pixel duplicates are
removed; image-id hashing fixes the train/holdout group independently of shard
order. Confidence and geometry filters still require visual auditing.

This run read 724,241 rows from 73 pinned shards, retaining 20,000 crops:
18,072 training and 1,928 holdout. All retained crop ids and pixel hashes are
unique, and the source-image intersection between splits is empty. The audit
sample was selected with seed 1 from the first 8,000 flushed training crops,
before the import finished. All 200 were inspected: 187 decidable labels agreed,
13 were uncertain, and 11 had incompatible/clipped layouts (some overlap).
Uncertain labels were not counted as confirmed transcription errors. Curation
excluded problematic samples and the ambiguous kana classes `ぁ`, `あ`, `お`,
`ぷ`; see the committed source hashes, audit counts and resulting crop count in
`data/manifests/animetext_import_20261011.json`. This audit does not establish
the accuracy of the complete dataset or of a trained recognizer.

## Private capture and matched training

Run Vita3K using a disposable preference directory; do not use live console
saves. `eval/capture_vita3k.py capture` records a named native-resolution window
and capture session. Manually label stable frames with the schema in
`eval/README.md`. The packer permits development data only:

```sh
.venv-ci/bin/python eval/capture_vita3k.py pack --manifest /private/dev/labels.jsonl \
  --out /private/dev.npz
.venv-ci/bin/python eval/run.py --model vocr:/private/FL10_w8.vocr \
  --set clannad-dev --manifest /private/dev/labels.jsonl --polarity console \
  --out /private/development.json
```

Populate a disposable rendering checkout with the original natural Japanese
corpus and pinned OFL fonts. Keep its training backgrounds separate from both
capture blocks and the shared test games. Render targeted data with the existing
VN styles, outlines and native Vita/PSP resolutions:

```sh
.venv-ci/bin/python synth/render.py shards --finder grad --invert-dark \
  --vocab build/hiragana/vocab.txt --target-report /private/development.json \
  --out /private/targeted --shards 2 --per-shard 10000
```

Fill `train/hiragana_config.example.json` with actual training paths. The `ps`
source must be a materialized hard-label snapshot; unlabelled distillation
crops cannot silently enter this hard-label-only experiment. Run:

```sh
.venv-ci/bin/python train/hiragana_experiment.py --config /private/experiment.json \
  --init build/hiragana/fl10_init.pt --vocab build/hiragana/vocab.txt \
  --out runs/hiragana --run
```

All arms start from the same recovered FL10, with seeds 1/2/3, batch 96,
5,000 steps, LR 1e-4, AdamW, 100-step warmup and cosine floor 5%. The original
mixture retains nv/s2/tt/rd/ps proportions 25/25/27.5/12.5/10. The arms use
100% original; 90% original + 10% targeted; and 85% original + 10% targeted +
5% curated AnimeText. There is no frame distillation. Frozen BN and exact class
order are preserved; selection uses development hiragana CER every 500 steps.
The runner checks split/session separation, curated manifest identity, hashes
all input files, and refuses to resume another experiment configuration.

Choose checkpoints and the reported seed from development results before
opening held-out results. Export with `train/export.py --int8`, then evaluate
each baseline/control/candidate with the C `vocr:` backend:

```sh
.venv-ci/bin/python eval/run.py --model vocr:/private/candidate.vocr \
  --set vita-test,psp-test,clannad-test --data /private/shared-eval \
  --local-set clannad-test=/private/test/labels.jsonl --polarity console \
  --out /private/candidate.json
# Repeat with --finder gradient and retain production-crop results too.
.venv-ci/bin/python train/hiragana_gate.py --baseline /private/baseline.json \
  --targeted /private/targeted.json --candidate /private/candidate.json \
  --blob /private/candidate.vocr --out /private/gate.json
```

The recognizer gate requires the same held-out manifests, improved CLANNAD
hiragana errors over both released and targeted controls, no full-CER
regression on any of the three sets, identical scored lines, padding and console
polarity, exact candidate blob identity and unchanged
memory limits. Inspect finder/pipeline regressions separately before promotion.
Physical Vita/PSP latency, memory headroom while a game runs, and accuracy need
separate hardware validation. Only a qualified candidate goes on a private
release and into a separate overlay hash-update PR.
