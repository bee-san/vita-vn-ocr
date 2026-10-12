# train2: the v0.2.0 recognizers

The owner's goal (RULES.md, 2026-10-09 14:30Z) is to beat JIS-trimmed int8 PP-OCRv5 mobile on
eval's real sets within 50 MB and well under 1 s per line on one 444 MHz Cortex-A9 core:
- vita-test CER below 2.30%;
- psp-test CER below 5.34% (the merged psp-test figure; the owner quoted 5.58% before eval #5's
  fixes);
- every style at or below that model's per-style CER.

train2 follows the owner's recipe:
1. pretrain on NVIDIA Japanese line crops mixed with synth2's VN boxes;
2. fine-tune on teacher-labelled real crops;
3. try distillation from PP-OCRv6 small, the best local teacher (#11).

## Result

Every number below is int8 through the C runtime, scored by `eval/run.py` on the hand-checked
sets. Intervals are 95% paired bootstraps over lines (10,000 resamples, seed 0); differences
are in points of CER.

| model | width / vocab | fine-tune | int8 blob | arena w8f / w8i | vita-test CER / exact | psp-test CER / exact | est. A9 ms, 640-px line, w8i / w8f |
|---|---|---|---|---|---|---|---|
| bar: JIS int8 PP-OCRv5 mobile | — | — | 2,749,540 B | 13.4 MiB peak | 2.30% / 79.1% | 5.34% / 66.1% | — |
| **H15 (recommended)** | 1.5 / 5,355 | hard labels | 1,155,968 B | 173,472 / 182,944 B | **1.56% / 83.5%** | **2.63% / 82.4%** | 166-302 / 294-365 |
| F20 | 2.0 / 5,355 | hard labels + KD | 1,558,016 B | 228,512 / 241,056 B | 1.79% / 84.0% | 2.79% / 79.6% | 235-416 / 439-579 |
| FL10 (PSP-lean) | 1.0 / 3,500 + `<unk>` | hard labels + KD | 657,664 B | 118,432 / 124,832 B | 2.30% / 77.7% | 5.18% / 69.8% | 94-178 / 157-194 |

- **H15 and F20 beat the bar overall on both sets.**
  - H15: vita-test −0.74 (CI −1.44 to −0.04), psp-test −2.71 (CI −4.05 to −1.36).
  - F20: vita-test −0.51 (CI −1.33 to +0.37), psp-test −2.55 (CI −3.84 to −1.30).
  - F20's vita-test interval includes 0, and H15's only just clears it: 206 lines cannot
    resolve differences much under a point.
- **FL10 ties the bar on vita-test** (59 errors each); it does not beat it. On psp-test it is
  −0.16 (CI −1.66 to +1.42).
- **No model meets the per-style criterion.**
  - H15 is above the bar on vita `coloured`, `coloured-name` and `name`, and on psp `furigana`,
    `name` and `no-box`.
  - These styles have 4 to 62 lines each, and most of the differences are not significant
    either way (psp `name`, H15 − bar: +3.35, CI −2.36 to +10.65).
  - They are the short rare-kanji crops (speaker names of a few uncommon kanji) and ruby
    crowding the resize at PSP scale. PP-OCRv6 small reads the psp names at 3.8% (#11), but the
    student sees too few of each rare name.
- **Distillation did not help.** F15, H15's recipe with PP-OCRv6-small frame KD, is worse on
  teacher_val (+0.53, CI +0.32 to +0.74) and level on the test sets (Ablation). F20 and FL10
  were trained with KD and have no matched control, so they may improve without it.
- **Not reached:** the < 1% vita-test stretch.
- **Nothing has run on a console.** The A9 times are estimates from `vita/a9_cycles.py` (the
  Vita compiler's machine code for the runtime under plugin qemu-arm, 40 lines, a cycle model
  as a range; `vita/README.md`).
- **On-console memory.** H15 needs its 1.16 MB of int8 weights, a 183 KB w8i arena and the line
  finder's 83 KB arena (960 × 544): about 1.42 MB plus the runtime's code, far inside 50 MB.

H15 is the run called `F15_nokd` in the artifacts: F15's recipe with the frame-level KL switched
off. It was chosen by teacher_val alone (Selection).

## Data

| source | lines | what | use |
|---|---|---|---|
| `nv` | 12,036,035 in 597 shards | nvsynth's every-line crops from NVIDIA OCR-Synthetic-Multilingual-v1, train sources 000-041 (CC BY 4.0) | pretrain, fine-tune |
| `s2` | 390,000 in 39 shards | synth2's VN-style shards `synth_00000`-`00038` | pretrain, fine-tune |
| `tt` | 12,115 | teacher's accepted real crops, `teacher_train.npz` (99.3% held-out precision; 0 wrong of 200 by eye, #11) | fine-tune |
| `rd` | 9,811 | realdata's 4-reader-vote crops, `real_train.npz` (1 wrong of 100 by eye, #7); the same screenshots, other crop boxes and labeller | fine-tune |
| `ps` | 10,706 of 20,218 | the unlabelled real pools `teacher_distill_train.npz` and realdata's `real_unlabelled.npz`, hard-labelled by PP-OCRv6 small where its lowest character confidence is at least 0.7 | fine-tune |

The `ps` labels are noisy. In 50 random lines that pass the filter, read by eye in review, 2
hold characters outside the vocab (so train2 drops them), and 6 of the other 48 are wrong
(12.5%):
- 選択 read as 退択, and 前の選択肢 as 前の択肢;
- テトメト read as テトメ;
- a dropped ―― on two lines, and an extra … on one.

On teacher_val, H15 (with `ps`) and the hard-labels-only control without it are level (−0.02,
CI −0.18 to +0.15).

## Selection

Validation, for model selection only:

| set | lines | |
|---|---|---|
| `rv` | 1,651 | teacher's `teacher_val.npz` as stored (dark text inverted: the console's polarity contract) |
| `rvn` | 1,651 | the same crops in screen polarity, as eval feeds them |
| `nvv` | 2,000 | nvsynth's held-out `nv_val.npz` (validation pages) |
| `s2v` | 2,000 | synth2's held-out `synth_00039.npz` |

- **How a run picks its checkpoint.** It keeps the checkpoint with the lowest mean CER over `rv`
  and `rvn`. CER is eval's: `eval/metrics.norm` on both sides, Levenshtein over the total
  ground-truth characters.
- **Hygiene.**
  - `teacher_val` has 181 VNs. It shares no VN, release or screenshot with any real training pool. synth2's
    scene list (a synthetic source) holds 22 of its VNs and 110 of its screenshots (docs/synth3.md, Study 1).
  - No training or validation file holds a VN of eval's test sets or reserved games
    (`eval/manifests/test_games.json`), a test capture session, or a test screenshot, by id or
    by sha256 (re-checked in review).
  - realdata's `real_val` is not used: it shares VNs through bundle releases with teacher's
    train split.
- **How the release models were chosen.**
  - The first version of this PR chose F15 for its vita-test CER, and rejected a longer
    fine-tune (F15b) for its test-set CER, although F15b's teacher_val was lower.
  - The review replaced that with a rule written down before any new run was scored on the test
    sets: the width-1.5 release is the run with the lowest teacher_val among F15, F15b and two
    matched controls, and it replaces F15 only if its interval against F15 is below 0.
  - That run is H15: 2.29% against F15's 2.82%, −0.53 (CI −0.74 to −0.32).
  - F20 and FL10 were kept as the width-2.0 and PSP-lean variants.
  - The test sets were scored only after this choice, and every candidate's scores are reported
    below.
- **Why width 1.5.** arch (#10) recommends width 2.0, with width 1.5 as the lighter fallback.
  H15 has a lower teacher_val than F20 (−0.34, CI −0.56 to −0.12) at about 0.7× its A9 time,
  so the width-1.5 model is the recommended one.

## Model

`train/model.py` (arch #10):
- grey input at height 32;
- a stem and depthwise-separable blocks, stride 8;
- two temporal blocks;
- a 96-wide bottleneck before the head.

H15 and F20 have the full 5,355-class head. FL10 has 3,501 classes: the 3,500-entry vocab and
the CTC blank.

FL10's vocab is not quite main's `data/vocab_3500.txt`.
- It was trained on the arch branch's version of that file, from before arch's review stopped
  eval's test-game character names from informing the ranking.
- The two differ in 7 of 3,500 rare kanji: FL10 has 枷柾沫盧穗錫黛, main has 柘櫂瓏籐耆蕾鴫.
- None of those 14 characters occurs in vita-test or psp-test, so no test score depends on it.
- The exact file is `$HERD/shared/artifacts/train2/release/FL10_vocab_3500.txt` (sha256
  7f1cb2eb…), and the FL10 blob embeds it.

## Distillation targets (`train/teacher_targets.py`)

PP-OCRv6 small reads a 48-px-high input with stride 8. The student reads 32 px with stride 8,
so at natural aspect a teacher frame covers two thirds of a student frame.

- **Reading.** The teacher's reading uses rapidocr's preprocessing (BGR, height 48,
  `ceil(48 w / h)` wide, zero-padded to 320). That is the setting #11 measured: vita-test 1.33%,
  psp-test 1.23% CER.
- **Frame targets.** A second pass reads the student's own input (the crop resized to 32 × W32
  and padded to 8T columns, as `train2.py` builds it), upscaled to 48 × 16T.
  - Teacher frames 2t and 2t+1 then cover exactly student frame t.
  - The two are pooled: the blank gets p_2t(blank) · p_2t+1(blank), and every other class c
    gets 1 − (1 − p_2t(c))(1 − p_2t+1(c)), renormalised to the non-blank mass.
  - The 4 most probable classes per frame are kept.
- **Class map.** Teacher classes map to `data/vocab.txt` through `teacher/vote.canon`.
  - 111 of the student's 5,354 characters are not in the teacher's 18,708-class dictionary.
  - After the canon mapping, 188 student classes get no teacher mass at all: full-width Latin
    letters, digits and symbols, rare kanji.
- **Which lines get targets.** A line's frame targets are used only when their greedy reading
  equals the natural reading (`soft_ok`). For a labelled line, the teacher's reading must also
  score the same as the label.
  - That covers 10,552 of 12,115 `tt` lines, 8,344 of 9,811 `rd` and 9,221 of 10,706 `ps`.
  - Those lines skip the horizontal-stretch augmentation, which would break the frame grid.

The KD term is KL(q ‖ p_student) over those classes per frame, averaged over the frames that have
targets, and added to the CTC loss with weight 1 (`kd=1`). With `kd=0` the lines train on their
hard labels only.

## Recipe

All runs:
- go through `heavy.sh` (5 threads), seed 1;
- use batch 96, AdamW (weight decay 0.01), cosine schedule to 5% of the peak learning rate,
  dropout 0.1 and EMA 0.999;
- select on `rv` + `rvn`.

1. **Pretrain**, from scratch: nv and s2 at weight 0.5 each, 20% polarity inversion, lr 2e-3 with
   1,000 warm-up steps. p15 (width 1.5) runs 40,000 steps; p20 (width 2.0) and pl10 (width 1.0,
   3,500 vocab) run 32,000.
2. **Fine-tune**, 15,000 steps from the pretrain: lr 1.2e-3 with 500 warm-up steps.
   - Real sources carry half the mixture: `tt` 0.275, `rd` 0.125, `ps` 0.1. nv and s2 carry the
     other half, at 0.25 each.
   - Real crops get 50% polarity inversion.
   - H15 uses hard labels (`kd=0`). F15, F20 and FL10 add the frame KD (`kd=1`).

```sh
A=$HERD/shared/artifacts
NV="$A/nvsynth/full/train/nv_train_0[0-3]?_b*.npz+$A/nvsynth/full/train/nv_train_04[01]_b*.npz"
S2="$A/synth2/shards/synth_000[0-2]?.npz+$A/synth2/shards/synth_0003[0-8].npz"
TGT=$A/train2/targets
VALS="--val rv:$A/teacher/teacher_val.npz --val rvn:$A/teacher/teacher_val.npz:native=1
      --val nvv:$A/nvsynth/shards/nv_val.npz:n=2000 --val s2v:$A/synth2/shards/synth_00039.npz:n=2000
      --select rv,rvn"
PRE=(--src "nv:0.5:$NV:stream=1,pool=3,inv=0.2" --src "s2:0.5:$S2:stream=1,pool=3,inv=0.2")
ft() {  # fine-tune mixture; $1 = kd (0: hard labels, 1: + frame KD)
  FT=(--src "nv:0.25:$NV:stream=1,pool=3,inv=0.2" --src "s2:0.25:$S2:stream=1,pool=3,inv=0.2"
      --src "tt:0.275:$A/teacher/teacher_train.npz:targets=$TGT,kd=$1,inv=0.5"
      --src "rd:0.125:$A/realdata/real_train.npz:targets=$TGT,kd=$1,inv=0.5"
      --src "ps:0.1:$A/teacher/teacher_distill_train.npz+$A/realdata/real_unlabelled.npz:targets=$TGT,label=teacher,conf=0.7,kd=$1,inv=0.5")
}
# teacher targets (readings for ps, frame targets for KD)
train/teacher_targets.py run --out $TGT --workers 5 $A/teacher/teacher_val.npz $A/teacher/teacher_train.npz \
    $A/teacher/teacher_distill_train.npz $A/realdata/real_train.npz $A/realdata/real_unlabelled.npz
# pretrains (--keep-evals writes the ckpt_step<N>.pt files the fine-tunes start from)
train/train2.py --run p15 --out runs --steps 40000 --lr 2e-3 --width 1.5 --ema 0.999 --eval-every 4000 --keep-evals "${PRE[@]}" $VALS
train/train2.py --run p20 --out runs --steps 32000 --lr 2e-3 --width 2.0 --ema 0.999 --eval-every 4000 --keep-evals "${PRE[@]}" $VALS
V3500=$A/train2/release/FL10_vocab_3500.txt     # FL10's exact vocab (see Model)
train/train2.py --run pl10 --out runs --steps 32000 --lr 2e-3 --width 1.0 --vocab $V3500 --ema 0.999 \
    --eval-every 4000 --keep-evals "${PRE[@]}" $VALS
# fine-tunes
FTARGS=(--steps 15000 --lr 1.2e-3 --warmup 500 --ema 0.999 --eval-every 2500)
ft 0; train/train2.py --run H15 --out runs --init runs/p15/ckpt_step40000.pt --width 1.5 "${FTARGS[@]}" "${FT[@]}" $VALS
ft 1; train/train2.py --run F15 --out runs --init runs/p15/ckpt_step40000.pt --width 1.5 "${FTARGS[@]}" "${FT[@]}" $VALS
ft 1; train/train2.py --run F20 --out runs --init runs/p20/ckpt_step32000.pt --width 2.0 "${FTARGS[@]}" "${FT[@]}" $VALS
ft 1; train/train2.py --run FL10 --out runs --init runs/pl10/ckpt_best.pt --width 1.0 --vocab $V3500 \
    "${FTARGS[@]}" "${FT[@]}" $VALS                   # pl10's best checkpoint is its step 28,000
```

## Ablation

All rows are int8 through the C runtime (W8F; for the four models in the release table, W8I
gives the same CER).
teacher_val is the selection metric. The hard-only control, H15 and F15 all start from p15's
step-40k checkpoint and share the step count, learning rate, schedule and seed. F15b continues
from F15.

| run | what changes | teacher_val | vita-test | psp-test |
|---|---|---|---|---|
| bar: JIS int8 PP-OCRv5 mobile | | — | 2.30% / 79.1% | 5.34% / 66.1% |
| v1 | synthetic boxes only (32k steps, width 1.0) | 23.69% | 23.87% / 52.4% | 46.57% / 30.2% |
| p15 | NVIDIA + synth2 pretrain, width 1.5, 40k steps | 3.18% | 1.72% / 84.0% | 3.19% / 79.6% |
| hard only (`F15_hard`) | + fine-tune: `tt` 0.35, `rd` 0.15, no `ps`, `kd=0` | 2.30% | 1.25% / 87.9% | 2.63% / 84.1% |
| **H15** (`F15_nokd`) | + `ps` pseudo-labels (`tt` 0.275, `rd` 0.125, `ps` 0.1), `kd=0` | **2.29%** | 1.56% / 83.5% | 2.63% / 82.4% |
| F15 | + PP-OCRv6-small frame KD (`kd=1`) | 2.82% | 1.48% / 85.9% | 3.03% / 78.0% |
| F15b | F15 + 10k steps at real weight 0.6, lr 8e-4 | 2.72% | 1.79% / 82.5% | 3.86% / 74.3% |
| p20 → F20 | width 2.0 pretrain (32k) → fine-tune with KD | 2.99% → 2.63% | 1.68% → 1.79% | 2.75% → 2.79% |
| pl10 → FL10 | width 1.0 + 3,500 vocab pretrain (32k) → fine-tune with KD | 4.67% → 4.05% | 2.85% → 2.30% | 4.98% → 5.18% |

Pooled test below means vita-test and psp-test together.

| comparison | teacher_val | pooled test |
|---|---|---|
| fine-tuning (H15 − p15) | −0.89 (CI −1.12 to −0.66) | −0.35 (CI −0.76 to +0.04) |
| fine-tuning (hard only − p15) | −0.87 (CI −1.10 to −0.65) | −0.51 (CI −0.86 to −0.19) |
| fine-tuning (F20 − p20) | −0.36 (CI −0.61 to −0.11) | +0.08 (CI −0.34 to +0.50) |
| fine-tuning (FL10 − pl10) | −0.62 (CI −0.93 to −0.32) | −0.18 (CI −0.70 to +0.33) |
| distillation (F15 − H15) | +0.53 (CI +0.32 to +0.74) | +0.16 (CI −0.22 to +0.54) |
| `ps` pseudo-labels (H15 − hard only) | −0.02 (CI −0.18 to +0.15) | +0.16 (CI −0.17 to +0.48) |
| longer fine-tune (F15b − F15) | −0.10 (CI −0.25 to +0.05) | +0.57 (CI +0.27 to +0.89) |

- **Pretraining does most of the work.** The NVIDIA + synth2 pretrain alone takes v1's 23.9% /
  46.6% to 1.72% / 3.19%, and clears the bar on both sets (pooled −1.36, CI −2.22 to −0.48).
- **Fine-tuning on the real crops helps on teacher_val in every pair.** On the 451 test lines
  only the hard-label fine-tune's gain is outside noise.
- **Distillation hurts teacher_val, and does not help on the test sets.**
  - F15 − H15 is a tight control: same init, data, mixture, schedule and seed. It differs in the
    KL term and in the horizontal stretch that KD lines skip, which also shifts the shared random
    stream, so the two runs draw different batches after the first step.
  - The interval covers the sampling of validation lines, not training noise: each run is one
    seed. The two kd=0 runs, with different mixtures, land 0.02 apart (2.29% and 2.30%), while
    both KD comparisons are 0.5 to 0.8 points worse.
  - Possible causes, none tested:
    - the 2:1 frame pooling and the 1.33× horizontal stretch of the aligned pass;
    - 188 student classes with no teacher mass;
    - the KD term weighing as much as CTC.
  - The earlier 6k-step probe agrees: ftC (KD + `ps`) against ftA (hard), both from p15 at
    step 16k, is +0.80 on teacher_val (CI +0.55 to +1.06).
- **The pseudo-labels neither help nor hurt measurably**, despite their 12.5% line error rate.
- **Longer fine-tuning is not clearly better.** F15b's teacher_val is only slightly lower than
  F15's, and its test CER is higher.

## Per style (int8 through the C runtime)

Bar = JIS-trimmed int8 PP-OCRv5 mobile. `*` = meets the bar: for the overall row below it, for a
style at or below it. `=` marks an overall tie.

vita-test:

| style | lines | bar | H15 | F20 | FL10 | F15 (KD) |
|---|---|---|---|---|---|---|
| all | 206 | 2.30% / 79.1% | 1.56% / 83.5% * | 1.79% / 84.0% * | 2.30% / 77.7% = | 1.48% / 85.9% * |
| bright-on-dark | 163 | 2.48% | 1.79% * | 1.74% * | 2.73% | 1.74% * |
| coloured | 19 | 5.13% | 6.67% | 5.64% | 6.15% | 5.64% |
| coloured-name | 8 | 9.52% | 14.29% | 19.05% | 14.29% | 14.29% |
| dark-on-light | 43 | 1.64% | 0.73% * | 2.01% | 0.73% * | 0.55% * |
| furigana | 3 | 3.12% | 0.00% * | 1.56% * | 4.69% | 0.00% * |
| mincho | 47 | 1.33% | 0.74% * | 0.89% * | 1.48% | 0.44% * |
| name | 53 | 4.07% | 5.23% | 6.98% | 6.98% | 4.65% |
| outlined | 74 | 3.03% | 2.12% * | 1.72% * | 2.12% * | 1.92% * |

psp-test:

| style | lines | bar | H15 | F20 | FL10 | F15 (KD) |
|---|---|---|---|---|---|---|
| all | 245 | 5.34% / 66.1% | 2.63% / 82.4% * | 2.79% / 79.6% * | 5.18% / 69.8% * | 3.03% / 78.0% * |
| bright-on-dark | 135 | 5.18% | 2.98% * | 2.48% * | 5.39% | 3.48% * |
| coloured | 20 | 7.48% | 3.74% * | 2.80% * | 5.61% * | 2.34% * |
| coloured-name | 7 | 8.33% | 4.17% * | 0.00% * | 0.00% * | 0.00% * |
| dark-on-light | 110 | 5.54% | 2.18% * | 3.18% * | 4.90% * | 2.45% * |
| furigana | 20 | 9.71% | 14.08% | 9.71% * | 23.30% | 14.08% |
| mincho | 90 | 4.98% | 1.93% * | 2.95% * | 4.57% * | 2.54% * |
| name | 62 | 5.74% | 9.09% | 8.13% | 16.75% | 9.57% |
| no-box | 4 | 5.08% | 6.78% | 6.78% | 13.56% | 6.78% |
| outlined | 69 | 5.00% | 2.97% * | 2.43% * | 4.32% * | 3.65% * |
| small | 106 | 5.45% | 1.54% * | 2.06% * | 3.19% * | 2.06% * |

FL10 is significantly worse than the bar on psp `name` (+11.0, CI +4.3 to +18.3) and psp
`furigana` (+13.6, CI +1.0 to +25.9). The bar reads PSP names better than any of these
students, and PSP furigana better than all but F20, which ties it.

## C-runtime parity, memory and speed

Each model was exported with `train/export.py` (fp32 and int8) and checked with
`runtime/tests/parity.py` on every vita-test and psp-test line, in f32, w8f and w8i. For all
three released models and every mode:
- the decoded text is the same as PyTorch's on every line;
- 0 frames have a different argmax;
- the logits agree within parity.py's relative tolerance. The largest max |C − PyTorch| /
  max(1, max |logit|) is 2.2e-6 in f32 and w8f (tolerance 1e-3) and 2.5e-4 in w8i (tolerance
  5e-3), both from H15. H15's logits are about 70× larger than 1, so its absolute differences
  reach 1.6e-4 and 1.7e-2.

The C runtime scored through `eval/run.py` gives the PyTorch checkpoint's CER exactly. The
outputs match, but the arithmetic is not bit for bit.

Weights and arenas are the exporter's exact figures. The arena does not grow with line width.
Each model's A9 time is the vita agent's estimator run on its own blob. A 640-px line is about
the p90 real line width (`train/cost.py`).

| model | MACs per CTC frame | w8i | w8f |
|---|---|---|---|
| v1 | 1,002,720 | 113-212 ms | 188-230 ms |
| H15 (F15: the same within 1 ms) | 1,564,992 | 166-302 ms | 294-365 ms |
| F20 | 2,339,232 | 235-416 ms | 439-579 ms |
| FL10 | 824,736 | 94-178 ms | 157-194 ms |

On the Vita, w8i is the fastest mode. On the PSP (no NEON) it is the slowest, so w8f is the mode
to use there (psp #13). psp #13 also asks for a PSP-lean variant of v1's size for anything larger than
v1; FL10 is that variant.

## Limitations

- No console run. Speed is an estimate, and so is the cache behaviour inside it.
- One seed per run. Most per-style differences, and the vita-test margins, are within noise.
- The per-style criterion is not met. FL10 only ties the bar on vita-test.
- F20 and FL10 were trained with KD, which hurt at width 1.5. A hard-label fine-tune of each
  (`ft 0`) is the obvious next run.
- teacher's acceptance rule was tuned on eval's test VNs (#11, `docs/teacher.md`). No test line
  enters training, but the training-label filter was calibrated on those games.
- The `ps` pool's hard labels are about 12.5% wrong by line (6 of 48 by eye).

## Reproduce

Artifacts are in `$HERD/shared/artifacts/train2/` (README there), with `A=$HERD/shared/artifacts`
as in the Recipe:
- `release/`: the published blobs and checkpoints, with sha256;
- `runs/`: every run's args, metrics, eval JSONs and logs, the checkpoints the fine-tunes start
  from, and the review's controls, parity logs and A9 reports;
- `targets/`: the teacher targets.

With the Recipe's commands, then for each model:

```sh
export HERD VOCR_EVAL_DATA=$A/eval VITASDK=$A/vita/vitasdk   # a9_cycles.py finds the shared qemu via $HERD
train/export.py --ckpt runs/H15/ckpt_best.pt --out H15_f32.vocr
train/export.py --ckpt runs/H15/ckpt_best.pt --out H15_w8.vocr --int8
make -C runtime
runtime/tests/parity.py --ckpt runs/H15/ckpt_best.pt --set vita-test --modes f32,w8f,w8i   # and psp-test
eval/run.py --model 'cmd:runtime/build/vocr_cli H15_w8.vocr --mode w8i --stdin' \
    --set vita-test,psp-test --out eval_c_w8i.json
train/bar.py eval_c_w8i.json                               # the per-style comparison with the bar
vita/a9_cycles.py measure --out a9/H15 --ckpt runs/H15/ckpt_best.pt --modes w8i,w8f --eval-data $A/eval
vita/a9_cycles.py report a9/H15                            # the A9 estimate
```
