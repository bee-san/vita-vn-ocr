# Matched native bitmap trials

The renderer snapshot adds 4,000 general and 4,000 naturally occurring `ずっ`
dialogue crops from privately owned CLANNAD Vita bitmap fonts, plus 500 separate
development crops. Actual native-font use is 3,859 general, 3,798 targeted and
455 development crops; other regions use covered OFL fallback. All native crops
preserve labelled core ink, all labels fit CTC and the released vocabulary, and
train/development pixel intersection is zero. The general shard contains 14,306
kanji occurrences, including seven ranked above 4,000; targeted contains 15,613,
including three above 4,000. All are in the pinned Jiten allowlist. Similar-kanji
correction remains a separate, capped two-percent training source.

`train/hiragana_bitmap.py` freezes two 2,000-step comparisons from the same
released FL10 initialization: head-only at learning rate 0.0001 and all parameters
at 0.00002. Both use seed 1, batch 32, six CPU threads, frozen batch-normalization
statistics, no distillation, and ten retained evaluation checkpoints. Sampling
probabilities are 66% old OFL general, 20% new bitmap general, 10% new bitmap
targeted, 2% reviewed native training crops and 2% bounded similar-kanji crops.
These are sampling probabilities, not per-character quotas. Recorded mixer counts
include prefetched draws; report them separately from the 64,000 consumed lines
per completed arm.

This is a bounded hypothesis using one game's bitmap-font family. The remaining
OFL fonts and dialogue sources keep the experiment broader than that family, but
separate text/books and crop pixels do not establish transfer to unseen fonts or
games. The 30% bitmap sampling cap is not a claim that this family represents all
PSP or Vita typography.

Inversion augmentation is zero because the saved crops already follow console
polarity. Bitmap and real native sources also disable trainer augmentation:
the renderer already provides visual/crop variation, and trainer vertical slicing
could otherwise discard labelled ink. OFL general and corrective kanji sources
retain their declared augmentation.

```sh
.venv-ci/bin/python train/hiragana_bitmap.py \
  --config train/hiragana_bitmap_config.example.json \
  --init build/hiragana/fl10_init.pt --vocab build/hiragana/vocab.txt \
  --out build/hiragana/bitmap-runs --run
```

The runner verifies shard roles, labels, Jiten policy, exact snapshot membership
and hashes, real training provenance, and train/development image/session/pixel
separation. It records commands, code/dependency/input hashes and host environment.
Complete restarts are reproducible; checkpoint resume still omits RNG/source/mixer
state. Fonts, images, corpora, labels, shards and weights remain private and ignored.

Development selection retains the original CLANNAD/full-text and original
synthetic guards, requiring strict CLANNAD hiragana improvement. Three additional
guards must each retain both full-text and hiragana error counts: 22 PSP lines at
pad 2, nine Vita lines at pad 4, and the 500 new bitmap-development crops. Their
gains cannot offset any original-set regression. The real additions were visually
reviewed by two assistants; they are development evidence, not independent human
test annotations. No development transcripts enter training.

Measured released baseline in scalar C W8I on x86_64 Linux, Core Ultra 7 165U:

| Development set | Full errors / characters | Hiragana errors / characters |
|---|---:|---:|
| Original CLANNAD | 5 / 207 | 2 / 127 |
| Original synthetic | 510 / 7,447 | 190 / 4,400 |
| Additional PSP | 2 / 265 | 0 / 132 |
| Additional Vita | 1 / 80 | 1 / 43 |
| Bitmap synthetic | 425 / 6,804 | 156 / 3,906 |

The deployment topology remains 657,664 model bytes and 124,832 arena bytes.
These counts do not measure physical-console speed or memory peaks. A candidate
must be locked by development data before the unchanged shared/private held-out
gate is run. The separate long CLANNAD/Google comparison cannot select or train
these trials. No accuracy upgrade is established by preparing this recipe.

Both arms completed 2,000 steps. All 20 retained checkpoints were exported and
measured through scalar C W8I using the frozen rules and the five development
sets above; see `eval/hiragana_bitmap_20261011.json` for commands and hashes.
The four head-only checkpoints through step 800 retain the original five full
and two hiragana errors; none strictly improves its hiragana count. All-parameter
checkpoints increase the original real-development errors. Other development
sets improve for some candidates, but those gains do not satisfy this trial's
replacement rule. The released baseline is retained and no candidate was tested
on the held-out benchmark.

The original improvement objective is narrow: its primary real-development set
is the short CLANNAD opening. Additional PSP/Vita guards currently combine games
within each console, so they also cannot prove per-game nonregression. Future
experiments should preregister a balanced objective across VN/font/console
development strata and per-stratum guards before training. The separate long
CLANNAD test remains diagnostic only and must not determine that objective.
