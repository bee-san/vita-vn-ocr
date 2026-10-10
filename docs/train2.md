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
