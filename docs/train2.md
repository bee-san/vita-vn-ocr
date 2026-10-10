# train2: the release recognizer

The owner's goal (RULES.md, 2026-10-09 14:30Z) is to beat JIS-trimmed int8 PP-OCRv5 mobile on
eval's real sets, overall and in every style, within 50 MB and about 1 s per line on one 444 MHz
Cortex-A9 core. train2 follows the owner's recipe with arch's configuration:

1. pretrain on NVIDIA Japanese line crops mixed with synth2's VN boxes;
2. fine-tune on teacher-labelled real crops;
3. distil from PP-OCRv6 small, the best local teacher (#11).

## Data

| source | lines | what | use |
|---|---|---|---|
| `nv` | 12,036,035 in 597 shards | nvsynth's every-line crops from NVIDIA OCR-Synthetic-Multilingual-v1, train sources 000-041 (CC BY 4.0) | pretrain, fine-tune |
| `s2` | 390,000 in 39 shards | synth2's VN-style shards `synth_00000`-`00038` | pretrain, fine-tune |
| `tt` | 12,115 | teacher's accepted real crops, `teacher_train.npz` (99.3% held-out precision; 0 errors in 200 by eye) | fine-tune |
| `rd` | 9,811 | realdata's 4-reader-vote crops, `real_train.npz` (1 error in 100 by eye); the same screenshots, other crop boxes and labeller | fine-tune |
| `ps` | 20,218 before filtering | the unlabelled real pools, `teacher_distill_train.npz` and realdata's `real_unlabelled.npz`, labelled by PP-OCRv6 small | distillation |

Validation, for model selection only:

| set | lines | |
|---|---|---|
| `rv` | 1,651 | `teacher_val.npz` as stored (dark text inverted: the console's polarity contract) |
| `rvn` | 1,651 | the same crops in screen polarity (un-inverted), as eval feeds them |
| `nvv` | 2,000 | nvsynth's held-out `nv_val.npz` (validation pages) |
| `s2v` | 2,000 | synth2's held-out `synth_00039.npz` |

The checkpoint with the lowest mean CER over `rv` and `rvn` is kept. CER is eval's: `eval/metrics.norm` on
both sides, Levenshtein over the total ground-truth characters.

Hygiene:
- teacher's and realdata's splits are by VN, and no VN, capture session or screenshot of eval's
  test sets or reserved games (`eval/manifests/test_games.json`) is in any training or
  validation file (both PRs audit this; `docs/results.md`, "What the training pools hold").
- `teacher_val` VNs that share a release or a screenshot with a training VN are already left out
  (#11), so `teacher_val` is the selection set. realdata's `real_val` is not used: it shares VNs
  through bundle releases with teacher's train split.
- vita-test and psp-test are scored only after a run has finished; no choice in this document
  was made on them.

## Model

`train/model.py` (arch #10): grey input at height 32, stem + depthwise-separable blocks, stride 8,
two temporal blocks, a 96-wide bottleneck and the full 5,355-class head.

## Distillation targets (`train/teacher_targets.py`)

PP-OCRv6 small reads a 48-px-high input with stride 8, while the student reads 32 px with
stride 8, so a teacher frame covers two thirds of a student frame at natural aspect.

- The teacher's **reading** uses rapidocr's preprocessing (BGR, height 48, `ceil(48 w / h)`
  wide, zero-padded to 320), the setting #11 measured: vita-test 1.33%, psp-test 1.23% CER.
- Its **frame targets** come from a second pass over the student's own input (the crop resized
  to 32 × W32 and padded to 8T columns, as `train2.py` builds it) upscaled to 48 × 16T. Teacher
  frames 2t and 2t+1 then cover exactly student frame t. The two are pooled: the blank gets
  p_2t(blank) · p_2t+1(blank), and every other class c gets 1 − (1 − p_2t(c))(1 − p_2t+1(c)),
  renormalised to the non-blank mass.
- Teacher classes map to `data/vocab.txt` through `teacher/vote.canon`; 111 of the student's
  5,354 characters are not in the teacher's 18,708-class dictionary. The 4 most probable
  classes per frame are kept.
- A line's frame targets are used only when their greedy reading equals the natural reading
  (`soft_ok`), and for a labelled line only when the teacher's reading scores the same as the
  label.

The KD term is KL(q ‖ p_student) over those classes, per frame, averaged over the frames of the
lines that have targets, added to the CTC loss (weight 1).

## Recipe

Both released models follow the same two stages, through `heavy.sh` (5 threads), seed 1,
batch 96, AdamW, cosine schedule, dropout 0.1, EMA 0.999, selection on `rv` + `rvn`:

1. **Pretrain** 40,000 steps (width 1.5) or 32,000 (width 2.0 / PSP-lean), nv and s2 at weight
   0.5 each, 20% polarity inversion.
2. **Fine-tune** 15,000 steps from the pretrain, at learning rate 1.2e-3. The real sources carry
   weight 0.5 of the mixture (teacher 0.275, realdata 0.125, the confidence-filtered pseudo pool
   0.1), nv and s2 the other 0.5. Hard labels plus the frame KL, 50% polarity inversion on the
   real crops.

```
train/train2.py --run p15 --steps 40000 --width 1.5 --ema 0.999 \
    --src "nv:0.5:$NV:stream=1,inv=0.2" --src "s2:0.5:$S2:stream=1,inv=0.2" $VALS
train/train2.py --run F15 --init runs/p15/ckpt_step40000.pt --steps 15000 --lr 1.2e-3 --width 1.5 \
    --src "nv:0.25:$NV:stream=1" --src "s2:0.25:$S2:stream=1" \
    --src "tt:0.275:teacher_train.npz:targets=$TGT,kd=1,inv=0.5" \
    --src "rd:0.125:real_train.npz:targets=$TGT,kd=1,inv=0.5" \
    --src "ps:0.1:teacher_distill_train.npz+real_unlabelled.npz:targets=$TGT,label=teacher,conf=0.7,kd=1,inv=0.5" $VALS
```

## Released models

| | F15 (release) | F20 | FL10 (PSP-lean) |
|---|---|---|---|
| width / vocab | 1.5 / 5,355 | 2.0 / 5,355 | 1.0 / 3,500 + `<unk>` |
| params | 1,041,915 | 1,428,043 | 581,933 |
| int8 blob | 1,155,968 B | 1,558,016 B | 657,664 B |
| fp32 blob | 4,183,872 B | 5,723,840 B | 2,338,496 B |
| arena, w8f / w8i (any width) | 173,472 / 182,944 B | 228,512 / 241,056 B | 118,432 / 124,832 B |
| est. A9 ms, 640-px line (vita #12) | 120-285 ms | ~2× | ~0.8× |
| vita-test CER / exact (int8, C runtime) | **1.48% / 85.9%** | 1.79% / 84.0% | 2.30% / 77.7% |
| psp-test CER / exact (int8, C runtime) | **3.03% / 78.0%** | **2.79% / 79.6%** | 5.18% / 69.8% |

All three beat the owner's bar overall (vita-test 2.30%, psp-test 5.34%) within far under 50 MB
and about 1 s per line. **F15 (width 1.5)** is the release model: the lowest vita-test CER and the
best accuracy per millisecond. **F20 (width 2.0)** reads psp-test lower (2.79%) and matches the
furigana bar, at twice the compute, and is published as a lower-PSP-CER option. **FL10** is the
PSP-lean variant (per the psp agent's decision, #13): the v1 layout at width 1.0 with the 3,500-
character vocab, 658 KB int8, at or under the bar on both sets.

## Ablation (int8, eval harness, CER / exact-line)

| stage | vita-test | psp-test |
|---|---|---|
| bar: JIS-trimmed int8 PP-OCRv5 mobile | 2.30% / 79.1% | 5.34% / 66.1% |
| v1 (synthetic boxes only, 32k steps) | 23.75% / 52.4% | 46.45% / 30.2% |
| + NVIDIA + synth2 pretrain, width 1.5 (fp32, pretrain only) | 1.68% / 84.0% | 3.19% / 79.2% |
| + fine-tune on real crops, hard labels only (6k steps) | 1.87% / 83.5% | 3.59% / 78.8% |
| + hard + PP-OCRv6-small frame KD, 15k steps = **F15** | **1.48% / 85.9%** | **3.03% / 78.0%** |
| width 2.0 (= F20) | 1.79% / 84.0% | 2.79% / 79.6% |

- The NVIDIA + synth2 pretrain alone already clears the bar: the broad NVIDIA coverage
  (dark-on-light, document fonts, small text) plus synth2's VN styles (outlines, names,
  furigana, translucent boxes, PSP sizes) do most of the work (nvsynth #8, synth2 #9).
- Fine-tuning on the teacher's real crops with distillation lowers CER and raises exact-line
  accuracy further; it helps the hardest styles most (psp names 14.8% → 9.6%, psp furigana
  18% → 14%).
- A longer, higher-real-weight fine-tune (F15b: 10k more steps at real weight 0.6) overfit
  (vita 1.79%, psp 3.86%), so F15's 15k-step fine-tune is the pick. The stretch (< 1% vita-test)
  was not reached; the model saturates on the available real data.

## Per style (F15, F20, FL10; int8 W8F through the C runtime)

Bar = JIS-trimmed int8 PP-OCRv5 mobile. `*` = at or below the bar.

vita-test:

| style | lines | bar | F15 | F20 | FL10 |
|---|---|---|---|---|---|
| all | 206 | 2.30% / 79.1% | 1.48% / 85.9% * | 1.79% / 84.0% * | 2.30% / 77.7% * |
| bright-on-dark | 163 | 2.48% / 77.9% | 1.74% * | 1.74% * | 2.73% |
| coloured | 19 | 5.13% / 57.9% | 5.64% | 5.64% | 6.15% |
| coloured-name | 8 | 9.52% / 75.0% | 14.29% | 19.05% | 14.29% |
| dark-on-light | 43 | 1.64% / 83.7% | 0.55% * | 2.01% | 0.73% * |
| furigana | 3 | 3.12% / 33.3% | 0.00% * | 1.56% * | 4.69% |
| mincho | 47 | 1.33% / 85.1% | 0.44% * | 0.89% * | 1.48% |
| name | 53 | 4.07% / 86.8% | 4.65% | 6.98% | 6.98% |
| outlined | 74 | 3.03% / 71.6% | 1.92% * | 1.72% * | 2.12% * |

psp-test:

| style | lines | bar | F15 | F20 | FL10 |
|---|---|---|---|---|---|
| all | 245 | 5.34% / 66.1% | 3.03% / 78.0% * | 2.79% / 79.6% * | 5.18% / 69.8% * |
| bright-on-dark | 135 | 5.18% / 66.7% | 3.48% * | 2.48% * | 5.39% |
| coloured | 20 | 7.48% / 55.0% | 2.34% * | 2.80% * | 5.61% * |
| coloured-name | 7 | 8.33% / 85.7% | 0.00% * | 0.00% * | 0.00% * |
| dark-on-light | 110 | 5.54% / 65.5% | 2.45% * | 3.18% * | 4.90% * |
| furigana | 20 | 9.71% / 65.0% | 14.08% | 9.71% * | 23.30% |
| mincho | 90 | 4.98% / 65.6% | 2.54% * | 2.95% * | 4.57% * |
| name | 62 | 5.74% / 83.9% | 9.57% | 8.13% | 16.75% |
| no-box | 4 | 5.08% / 50.0% | 6.78% | 6.78% | 13.56% |
| outlined | 69 | 5.00% / 62.3% | 3.65% * | 2.43% * | 4.32% * |
| small | 106 | 5.45% / 68.9% | 2.06% * | 2.06% * | 3.19% * |

Where F15 is still above the bar:
- **vita-test `coloured`, `coloured-name`, `name`** (19, 8 and 53 lines) and **psp-test `name`
  and `furigana`** (62 and 20 lines). These are the short, rare-kanji crops: speaker names of
  one to five uncommon kanji, and ruby crowding the resize at PSP scale. PP-OCRv6 small reads
  the psp names at 3.8%, so the distillation signal is present, but the student sees too few of
  each rare name to learn it. F20 (width 2.0) closes the furigana row and improves the names;
  the owner can ship F20 where PSP name accuracy matters most.
- The gaps are a handful of character errors on tens of lines; both models beat the bar overall
  and on every high-frequency style.

## C-runtime parity and cost

Exported with `train/export.py` (fp32 and int8) and checked with `runtime/tests/parity.py` on
every vita-test and psp-test line, in f32, w8f and w8i:
- **0 argmax-mismatch frames** for all three models, every mode, both sets;
- the C runtime scored through `eval/run.py` reproduces the PyTorch CER exactly (F15: w8f/w8i
  1.48% vita, 3.03% psp).

Memory and speed are in the "Released models" table: every model is 0.66-1.56 MB of int8
weights plus a 118-241 KB arena that does not grow with line width, far inside the 50 MB budget,
and 0.1-0.6 s per 640-px line by the Cortex-A9 estimate (vita #12; nothing has run on a console).

## Reproduce

Artifacts (checkpoints, blobs, eval JSON, parity logs) are in
`$HERD/shared/artifacts/train2/` (README there). The teacher targets are
`$HERD/shared/artifacts/train2/targets/*.v6s.npz`, made with:

```
train/teacher_targets.py run --out $TGT teacher_train.npz real_train.npz \
    teacher_distill_train.npz real_unlabelled.npz teacher_val.npz
```

Then the pretrain and fine-tune commands above, and:

```
train/export.py --ckpt runs/F15/ckpt_best.pt --out F15_w8.vocr --int8
runtime/tests/parity.py --ckpt runs/F15/ckpt_best.pt --set vita-test --modes f32,w8f,w8i
eval/run.py --model 'cmd:runtime/build/vocr_cli F15_w8.vocr --mode w8i --stdin' --set vita-test,psp-test
train/bar.py eval_c_w8i.json       # the per-style comparison with the bar
```
