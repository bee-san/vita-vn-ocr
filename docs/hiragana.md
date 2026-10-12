# FL10 hiragana fine-tuning

The active experiment fine-tunes the released FL10 using new, reproducible
Vita/PSP dialogue crops, development-targeted hiragana, bounded similar-kanji
practice and audited AnimeText. The original training shards are unavailable;
`hiragana_experiment.py` remains an optional historical reproduction workflow.
The fresh experiment does not depend on those shards. CLANNAD was in the original
training collection, so its new sessions measure additional known-game captures.
An accuracy improvement requires the held-out native C W8I gate below.

## Measured baseline, 2026-10-11

The released FL10 W8I model was evaluated with console polarity, using the
existing shared screenshots and two new native 960×544 Vita3K capture blocks.
The development block precedes the held-out block; their screenshots and text
were hand checked separately, and blank/fading/typewriter-transition frames
were excluded before scoring. Each block has its own session and split.
Private images and transcriptions remain under ignored `build/hiragana/`.
The new captures are a small opening-scene sample, not coverage of the whole game.

| Set | Lines | Full errors / chars | Full CER | Hiragana errors / chars | Hiragana CER | Console finder pipeline CER | Research gradient finder CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| Shared Vita | 206 | 61 / 2,564 | 2.38% | 15 / 1,417 | 1.06% | 8.54% | 16.46% |
| Shared PSP | 245 | 133 / 2,510 | 5.30% | 30 / 1,382 | 2.17% | 17.93% | 36.18% |
| CLANNAD development | 19 | 5 / 207 | 2.42% | 2 / 127 | 1.57% | — | 4.83% |
| CLANNAD held out | 25 | 7 / 388 | 1.80% | 3 / 229 | 1.31% | 5.93% | 6.70% |

The console finder is the bit-exact Python twin of `runtime/vocr_lines.c`;
the research gradient finder uses different geometry. Both consume annotated
dialogue regions. Console finder recall is 95.15% on Vita, 90.20% on PSP and
96.00% on held-out CLANNAD. Their hiragana pipeline CERs are 4.66%, 11.00% and
2.18%. These include missed lines; recognizer-only CER assumes correctly
extracted lines. A model improvement alone cannot fix missing line detections.

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

## Fresh Vita/PSP training snapshots

`synth/fresh_dataset.py` rebuilds pinned corpus/font assets, natural dialogue and
narration pools, native VN rendering, role metadata and hash manifests under a
configurable ignored root. A deterministic split by Aozora book ID separates
training and synthetic development books. Background screenshots also use
separate training/development games and exclude the shared benchmark games and
CLANNAD. Native PSP/Vita VNDB screenshots contribute scenery only, from explicitly
hand-reviewed, text-free rectangles. Title, menu, load and name-entry screens are
excluded. A requested region must fit entirely within an approved patch, or the
renderer uses synthetic scenery. Downloaded images, fonts, corpora, NPZ shards and
checkpoints stay local.
Commit source URLs/hashes, configuration and aggregate results; put weights on
GitHub releases. Retain the exact scripts, logs and data manifest for each run.

The app receives a selected dialogue region, rather than an entire screen.
Rendering creates those regions at Vita 960×544, PSP 480×272 and PSP 2× scales;
the gradient finder extracts training lines. `--crop-edge-jitter` varies
spare line margins on 40% of draws, retains labelled ink, avoids adjacent text
rows, and recalculates console polarity after changing a crop. This models small
edge inaccuracies; it does not claim to recover glyphs a user cropped away.
Crops whose labels cannot fit the recognizer’s CTC sequence after height-32 resize
are rejected, including the blank steps needed between repeated characters.
`eval/crop_stress.py` checks development-only edge margins and optional dialogue
ROI offsets. The shared finder benchmark already uses annotated dialogue regions.

The pinned [official Jiten kanji list](https://jiten.moe/frequency-dictionaries)
is an allowlist, intersected with the exact released vocabulary. Its ranks are
not occurrence counts. All 3,112 released kanji are allowed, covering 1,990 of the
2,000 characters with rank ≤2,000. The model cannot emit the remaining 7,937 Jiten
characters; the training recipe leaves the vocabulary and memory size unchanged.
`data/manifests/fl10_kanji_policy_20261011.json` records the derived table,
coverage, source hashes and attribution. The older kanji_anki CSV and dictionary
KANJIDIC-only fallback characters do not override the official allowlist.

The MIT [similar-kanji graph](https://github.com/siikamiika/similar-kanji) supplies
5,212 supported visual pairs, including 土/士, 未/末 and 己/已/巳. Pair sampling
uses an explicit inverse-rank heuristic. Render one natural-context crop per
endpoint; discard wrapped crops that lose the selected character. Inside each
finite corrective shard, pair quotas with maximum endpoint ranks 2,001–3,000,
3,001–4,000 and >4,000 are capped at 10%, 4% and 1%; the rest go to common pairs.
Missing corpus contexts return their budget to common pairs. Generated names also
use rank weights, replacing uniform kanji draws. The manifest records actual
glyph exposure and paired-sample bands so rare overweighting remains measurable.

Three matched, seed-1 arms start at the same restored checkpoint, use 1,000
steps, batch 32, LR 2e-5, 50-step warmup, six CPU threads, frozen BN statistics,
no distillation and evaluation every 100 steps:

| Arm | General synthetic | Targeted hiragana | Curated AnimeText | Similar kanji |
|---|---:|---:|---:|---:|
| General | 98% | 0% | 0% | 2% |
| Targeted | 88% | 10% | 0% | 2% |
| AnimeText | 83% | 10% | 5% | 2% |

These percentages are source probabilities per sampled line, independent of
shard count. Natural dialogue keeps most training weight. A single-seed
experiment is exploratory; it cannot isolate a dataset's causal effect.

```sh
.venv-ci/bin/python synth/fresh_dataset.py assets --root build/hiragana/fresh-jiten
.venv-ci/bin/python synth/fresh_dataset.py pools --root build/hiragana/fresh-jiten \
  --vocab build/hiragana/vocab.txt
# Replay the pinned background manifest into the same root, then render each source.
.venv-ci/bin/python synth/fresh_dataset.py render --root build/hiragana/fresh-jiten \
  --split train --vocab build/hiragana/vocab.txt --finder grad --invert-dark \
  --kanji-policy data/manifests/fl10_kanji_policy_20261011.json --crop-edge-jitter \
  --background-manifest data/manifests/fresh_jiten_backgrounds.json \
  --out build/hiragana/fresh-jiten/general --shards 24 --per-shard 500 --workers 2
# Repeat into targeted with --target-report /private/development.json;
# into kanji with --kanji-pairs; and synthetic-dev with --split dev and another seed.
.venv-ci/bin/python synth/fresh_dataset.py manifest --root build/hiragana/fresh-jiten \
  --vocab build/hiragana/vocab.txt --kanji-policy data/manifests/fl10_kanji_policy_20261011.json \
  --background-manifest data/manifests/fresh_jiten_backgrounds.json \
  --source general:train --source targeted:train --source kanji:train --source synthetic-dev:dev
.venv-ci/bin/python train/hiragana_fresh.py --config /private/fresh-config.json \
  --init build/hiragana/fl10_init.pt --vocab build/hiragana/vocab.txt \
  --out runs/hiragana-fresh --run
.venv-ci/bin/python train/hiragana_select.py --baseline /private/FL10_w8.vocr \
  --dev-manifest /private/dev/labels.jsonl \
  --synthetic-dev build/hiragana/fresh-jiten/synthetic-dev/synth_00000.npz \
  --checkpoints 'runs/hiragana-fresh/*/ckpt_step*.pt' --out /private/selection.json
```

Start with `train/hiragana_fresh_config.example.json`. Preparation checks labelled
training roles, exact vocabulary and Jiten policy, curated AnimeText audit
identity, corrective shard hashes, selected pair endpoint presence, and
session/image/pixel separation from development data. The experiment snapshot
records commands, inputs, code/dependency hashes and host environment. Seeds and
pinned inputs permit a complete restart. Checkpoint resume is approximate because
train2 does not save RNG/source/mixer state.

The final selector exports **every** saved evaluation checkpoint to int8 and
scores it through native scalar C W8I. A checkpoint must avoid full-CER regression
on both CLANNAD development and independent synthetic development, then strictly
reduce development hiragana errors. Ranking uses hiragana errors, real full
errors, synthetic full errors, step and checkpoint path. An existing selection
file cannot be overwritten: its chosen blob SHA is locked before test evaluation.
If no checkpoint qualifies, retain the released baseline. Held-out test data never
selects checkpoints. The fresh gate compares the locked candidate with the release;
`--targeted` additionally checks an independent targeted control when available.

## Native dialogue recovery and experiment results

The fresh snapshot contains 12,000 general, 4,000 targeted and 1,500 paired-kanji
training crops, plus 500 independent synthetic development crops. All training
sources have zero CTC-infeasible or out-of-vocabulary labels and no development
pixel overlap. Every one of the 30 saved checkpoints from the three matched
arms was exported and scored through C W8I. Synthetic full errors improved from
510 to 441–474, but real development full errors increased from 5 to 7–11 and
hiragana errors from 2 to 3–5. None qualified. Synthetic gains did not justify
replacing the release.

For native font anchors, `data/import_native.py` packs 22 dialogue crops from
13 training-side VNDB games, evenly divided between PSP and Vita. The assistant
visually checked crop boundaries and transcriptions; ambiguous, clipped and mixed
rows were removed. These are training data, not an independently human-validated
test set. Images and transcriptions remain private. The importer binds source
URLs/hashes, annotations, vocabulary and Jiten policy, verifies complete crop
geometry and rejects CTC-infeasible labels. Shared test games, CLANNAD and
development source games are excluded.

```sh
.venv-ci/bin/python data/import_native.py --root build/hiragana/fresh-jiten \
  --sources data/manifests/fresh_jiten_backgrounds.json \
  --annotations /private/native-annotations.jsonl --vocab build/hiragana/vocab.txt \
  --policy data/manifests/fl10_kanji_policy_20261011.json \
  --out build/hiragana/native-train
.venv-ci/bin/python train/hiragana_native.py \
  --config train/hiragana_native_config.example.json \
  --init build/hiragana/fl10_init.pt --vocab build/hiragana/vocab.txt \
  --out build/hiragana/native-runs --run
```

The two frozen-encoder trials use head-only learning rates 2e-5 and 1e-4, seed 1,
1,000 steps and the same development protocol. General/targeted/native/kanji
sampling is 83/10/5/2 percent; inversion augmentation is zero because stored
crops already use console polarity. The new targeted snapshot contains 4,000
crops targeting the reference-side development confusions `ず` and `っ` only.
Encoder, bottleneck and BatchNorm state stay frozen. All 20 C exports retained
the baseline's 5 full errors and 2 hiragana errors; none gave a strict hiragana
improvement. A separately frozen full-model trial with these same inputs is
recorded in `data/manifests/fl10_native_full_experiment_20261011.json`. Its ten
C exports had 7–11 full errors and 2–5 hiragana errors on real development
data, so none qualified either. Across all six runs and 60 C exports, the
released model remains selected. No trained candidate was scored on held-out
screenshots. The single-seed, small-development experiment does not prove that
AnimeText or synthetic data cannot help; it shows these recipes failed to
improve this released model.

The real kanji diagnostic records 4 graph-listed similar substitutions on Vita
and 17 on PSP. Across the shared sets there are just five reference kanji with
Jiten ranks 3,001–4,000 and none above 4,000. This cannot establish accuracy on
rare kanji, despite explicit bounded coverage in training. Aggregate experiment
evidence is in `eval/hiragana_fresh_20261011.json`; the source manifests preserve
shard hashes, actual glyph exposure, commands and code identities. Training uses
source commits `a52749b` and `d6ef77d`; complete restart is reproducible, while
checkpoint resume still omits RNG/source/mixer state.

The research snapshot retains all 60 evaluation checkpoints, initialization,
vocabulary, settings, training logs, frozen experiment recipes and filtered
source archives. Each archived member has a SHA-256 and size; checkpoints also
receive ZIP CRC validation. Images, annotations, corpora, fonts and rendered
shards are excluded. It is explicitly a research prerelease, without qualified
replacement deployment weights. Failed experiments remain reproducible rather
than being presented as an OCR accuracy release.

## Private capture and historical matched training

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

For historical reproduction only, populate a disposable rendering checkout with the original natural Japanese
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
# Repeat with --finder console and retain production dialogue-region results too.
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
