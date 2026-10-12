# Rejected head-only FL10 adaptation

This 2,000-step experiment produced **no eligible replacement** for released
FL10. Updating only the character classifier preserved the visual encoder but
did not preserve recognition across the six-game development cohort. The
released model and overlay defaults remain unchanged.

The first retained checkpoint passed the per-title nonregression guard but
did not improve all-text hiragana or the required number of titles. Every later
checkpoint regressed on Vita development strata. No candidate proceeded to a
textbox comparison, protected/fresh test, Google comparison or promotion.

## Controlled follow-up

The [earlier all-parameter adaptation](general-fl10-experiments-20261012.md)
improved aggregate development scores but failed individual-game guards. This
follow-up changed one training recipe field: `train_scope` from `all` to `head`.
It tested whether a fixed visual encoder could preserve those individual games.

The run used the same restored released initializer, 3,500-character ordered
vocabulary, Jiten kanji allowlist, 20,000 synthetic training crops and source
exposure limits. The data uses nine OFL fonts and book-separated Aozora pools;
no fresh test screenshot, CLANNAD reference or Iwaihime data was added. Training
used batch 32, learning rate 0.00002, final factor 0.05, warmup 50, weight decay
0.01, seed 1 and six CPU threads. BatchNorm statistics stayed frozen. There was
no distillation or additional augmentation. All ten 200-step checkpoints were
retained with optimizer and exposure state.

The optimizer consumed exactly 64,000 rows: 62,457 broad OFL, 278 kana
corrective and 1,265 kanji corrective (1.9766%). The largest correction pattern
appeared 278 times (0.4344%); there was zero game-asset exposure. All 6,000
recorded intent/forward/optimizer boundaries stayed within the existing caps.
These are consumed rows, excluding unconsumed prefetch draws.

The prospective recipe, line selector, textbox guard, source policy, code,
environment and output locations were frozen before training. Selection used
the unchanged per-title all/body full/hiragana guards, book-separated synthetic
guard, console-balanced hiragana ranking and two-title improvement requirement.
The actual driver source is commit `d80ca3fba113cdd1b9fccefecc9fb6edf8dd8a80`;
the new preparations have distinct output locations and hashes.

## Real development results

The cohort is unchanged: 237 native screenshots, 517 lines, three PSP titles
(Amagami, Corpse Party, Grisaia) and three Vita titles (Suran, Nil Admirari,
Haitaka). Assistant readings checked the references; no human-review claim is
made. These are development results with known historical baseline title
exposure, not unseen-title test results. References remain excluded from this
training run. The original historical training snapshot is still unavailable.

All candidates were exported and scored once through the pinned scalar C W8I
runtime with console-specific crop padding and polarity. The baseline record
exactly reproduces the earlier run's baseline. Each row below is an edit count;
lower is better. The complete per-title counters and failure reasons are in
[the derived JSON](benchmarks/general-fl10-head-20261012.json).

| Step | All text / 6,614 | All hiragana / 3,554 | Body text / 6,309 | Body hiragana / 3,514 | Selector outcome |
|---|---:|---:|---:|---:|---|
| Released FL10 | 183 | 65 | 133 | 47 | Kept |
| 200 | 180 | 65 | 130 | 46 | No qualifying hiragana/title gain |
| 400 | 186 | 68 | 134 | 49 | Rejected |
| 600 | 180 | 68 | 131 | 49 | Rejected |
| 800 | 180 | 69 | 132 | 50 | Rejected |
| 1000 | 175 | 68 | 127 | 49 | Rejected |
| 1200 | 175 | 67 | 127 | 48 | Rejected |
| 1400 | 175 | 68 | 127 | 49 | Rejected |
| 1600 | 175 | 68 | 127 | 49 | Rejected |
| 1800 | 175 | 68 | 127 | 49 | Rejected |
| 2000 | 175 | 68 | 127 | 49 | Rejected |

At step 2000, full-text development errors fall by eight (4.37%), while
hiragana errors rise by three (4.62%). Those mixed results do not establish an
improvement. Synthetic development errors fall from 648 to 576 of 14,834
characters and hiragana errors from 167 to 137 of 8,725; that does not override
real-game regressions. This recipe therefore stays a rejected research result.

An [independent read-only audit](../eval/fl10_head_adaptation_audit_20261012.json) verified every checkpoint's 109 non-head tensor
entries, including all 54 BatchNorm buffers, against the initializer. Both head
tensors changed; optimizer state contained only those two entries. It rebuilt
all ten int8 exports byte-identically in memory, checked the 6,000-record
exposure hash chain and checkpoint bindings, verified reference denominators
and reproduced the selector decision from saved counters. No additional neural,
finder or teacher call occurred. Per-line predictions were not retained by this
selector, so the audit **did not independently recompute edit distances**.

Each int8 export is 657,664 bytes and retains the 124,832-byte recognizer arena
contract at width 1,024. No additional console runtime memory is introduced.
Console latency and total residency were not measured by this experiment.

## Retained snapshot and reproduction

The [research weight release](https://github.com/bee-san/vita-vn-ocr/releases/tag/v0.2.4-fl10-head.1)
retains all ten original checkpoints and ten corresponding int8 exports in
`generic-fl10-head-checkpoints.zip`. It is a prerelease containing rejected
weights, not an overlay update. The derived JSON records the archive and every
member's size and SHA256. A [separate archive audit](../eval/fl10_head_release_audit_20261012.json)
verified all twenty members against the original source files and the public
counter projection against the saved selector output. No rendered shards, screenshots, crops, corpora,
fonts or dialogue references are included in Git or the weight archive.

Environment: x86-64 Linux, Python 3.11.15, PyTorch 2.5.1+cpu, NumPy 2.4.6,
Pillow 12.3.0 and GCC 16.2.1. The scalar library is pinned to SHA256
`4a5635ba9a0721373f35e8957c0788b74e1725ea8cd96b8200cc5e427536b11b`.
Training and selection commands, using the retained private preparations, were:

```sh
python train/generic_fl10_train.py run \
  --prepared build/prospective-general-vn-head-20261012/training-prepared.json
python train/general_vn_select.py select \
  --lock build/prospective-general-vn-head-20261012/protocol.json \
  --checkpoints 'build/prospective-general-vn-head-20261012/runs/general_ofl_head_s1/ckpt_step*.pt' \
  --out build/prospective-general-vn-head-20261012/selection.json
```

The [preparation/publication source snapshots](../eval/reproducibility/fl10-head-20261012/README.md)
preserve the small scripts used to create the new frozen preparations and public
counter/weight artifacts. They require the retained private inputs. The generic
trainer and selector are already versioned source; neither was changed for this run.
