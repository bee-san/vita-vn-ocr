# VOCR5: the final round, three recognizers (Small, Normal, Large)

VOCR5 trains one recognizer per tier of the owner's final ceilings on everything the 2026-10-14 13:25Z policy
allows. It uses vocab-expand's tier vocabularies and model-tiers' shapes as far as the final sizing allows, and it
trains the model to cope with the crops the console's line finder hands it. Each tier's model replaces that tier's
VOCR4 model only under a pre-registered paired rule on the fixed development sample, with dialogue CER as the
primary metric. The protected sets (vita-test, psp-test, CLANNAD) are scored once, after every selection is
committed, through the plain-C W8I runtime.

This document starts as the frozen protocol (`eval/vocr5_protocol.json`, which decides wherever the two differ).
Results are added below it once they exist. Every console time here is a host estimate, labelled as one.

## The tiers

The sizing comes from the owner's caps (18:09Z) and the conductor's final note (10:07Z / 10:21Z). Speed limits are
per typical full textbox: 3 lines of about 640 px, which is 240 CTC frames plus 0.1 s for the finder, on one
Cortex-A9 core. At freezing, #85 and overlay #93 were open and test.3 had not confirmed the 444 MHz boost, so
Normal's ceiling was 1.3 MB. Amendment 3 (2026-10-16, before any candidate was scored) raised it to the owner's
1.6 MB cap once test.3 confirmed the boost, and made the already-trained VOCR4-W20 Normal's incumbent. Normal's
VOCR5 shape is unchanged, because a 1.6 MB VOCR5 shape would have needed a new pretrain, about a day.

| tier | shape | vocab | int8 weights | W8I arena | MACs/frame | incumbent |
|---|---|---|---:|---:|---:|---|
| Small (< 1 MB, < 1 s) | `s_w100v4987`: FL10's topology | `data/vocab_4987_proof.txt` (4,987) | 819,792 B | 124,832 B | 967,392 | VOCR4-FL10 (657,664 B) |
| Normal (≤ 1.6 MB since amendment 3; about 1 s per box) | `n_w150v5631`: H15's topology | `data/vocab_vn_5631.txt` (5,631) | 1,185,632 B | 182,944 B | 1,591,488 | VOCR4-W20 (1,558,016 B; VOCR4-H15 until amendment 3) |
| Large (≤ 2.5 MB, < 3 s) | `l_lw300v5631`: late-wide 2.0 → 3.0 | `data/vocab_vn_5631.txt` (5,631) | 2,473,696 B | 261,024 B | 3,402,720 | VOCR4-W20 (1,558,016 B) |

Estimated time per 3-line box. The method is ocr-speed's: `vita/a9_layers.py` at #85's head, top-8, scaled to
VOCR3-H15's measured console rate of 4,941.7 µs per frame at an assumed 333 MHz. It reproduces
`ceilings4.json`'s own figures to the third decimal. All of these are estimates:

| shape | main's runtime, 333 / 444 MHz | #85, 333 / 444 | #85 without prefetch credit, 333 / 444 |
|---|---|---|---|
| `s_w100v4987` | 0.94 / 0.73 s | 0.57 / 0.45 s | 0.63 / 0.50 s |
| `n_w150v5631` | 1.34 / 1.03 s | 0.81 / 0.63 s | 0.92 / 0.72 s |
| `l_lw300v5631` | 2.68 / 2.08 s | 1.47 / 1.13 s | 2.08 / 1.63 s |
| VOCR4-W20 (Normal's incumbent) | 1.77 / 1.35 s | 1.08 / 0.84 s | 1.26 / 0.97 s |

VOCR4-W20 costs an estimated 0.20-0.43 s more per box than `n_w150v5631`, depending on the runtime and clock: 0.25 s
at 444 MHz and 0.34 s at 333 MHz without prefetch credit.

Why these shapes:

- **Small.** FL10's topology with vocab-expand's 4,987-class proof vocab. That vocab is a subset of
  `data/vocab.txt`, so every class already has real and synthetic data and no render is needed. The VN-ordered
  `vocab_vn_4987` would miss 0.107 VN characters per 1,000 instead of 0.187, which is 0.008% CER. To get that, it
  trades 240 kanji the real pools hold for 240 they never show.
- **Normal.** None of #74's Normal shapes fits the final 1.3 MB. The H15 topology with the 5,631-class VN head
  keeps a 1 s box at 333 MHz on #85's runtime (0.81-0.92 s). Width 1.6 (1,254,560 B) sits at the edge of that
  estimate.
- **Large.** #74's four Large shapes are all over 2.5 MB. The late-wide `n_lw300v5631` is the parameter-heavy,
  compute-light kind the owner asked for: 2.47 MB of weights for 3.40 M MACs per frame, against 3.92 M for a
  plain width 2.75 of 2.39 MB. With vocab-expand's 6,459-class Large vocab it would weigh 2,564,368 B, over the
  cap, and on VN text that vocab covers nothing 5,631 misses (0.000 per 1,000 either way).
- **Incumbents.** At freezing VOCR4-W20 was over Normal's 1.3 MB, so Normal's incumbent was VOCR4-H15, which the
  task names as the lighter Normal alternative. Since amendment 3 it is VOCR4-W20, the task's original Normal.

## Data

- **Synthetic.** Every tier keeps nvsynth, synth2 and synth3 as in VOCR4.
  - Small uses the C mix (synth3 v2), the recipe that FL10 won with in VOCR3 and VOCR4.
  - Normal and Large use the M mix (synth3 v3, VOCR4's only resolved gain) plus one tenth of `s4v5631`: a synth4
    render with `data/vocab_vn_5631.txt`, its curriculum rebuilt so the 610 new kanji get their exposures
    (1,000,000 crops, seed 20261015). The helper `vocr5-render` makes it, and it is pinned in
    `data/manifests/synth4_shards_v5631.json` before the Normal and Large pretrains start. If it is not pinned by
    20:00Z, Normal and Large fall back to `data/vocab.txt`.
  - nvsynth is not cut again. Its lines were dropped only for characters outside `data/vocab.txt`, and the new
    kanji are about 0.01% of NVIDIA's characters, so a new cut would cost a 400 GB download for a handful of
    exposures.
- **Real.** Every pool the policy allows:
  - the train parts of teacher_train, teacher_val, real_train, real_val and devnative (the re-carve of #84);
  - native pool v1;
  - owner pools v1 and v2 merged: 7,179 ink, 5,624 GRAD and 13,278 margin crops of 73 VNs (private);
  - realplus v3: the newest snapshot published with RESERVE_OK before the fine-tune pin (v3i6 at freezing:
    135,689 ink and 97,089 GRAD crops, including 23,895 lines of the reclaimed VNDB tier).

  A row is dropped when it falls on a `reserved.json` VN (test or dev_sample), on a development or test
  screenshot, or on a test session. Lines are deduplicated by screenshot, label and crop style, so a GRAD crop
  stays when its ink crop is in another pool. `eval/tools/reserve_check.py --dump --pixels` runs on every pool.
  The pools are pinned in `eval/vocr5_pools.sha256` before the first fine-tune and before any score.

## Arms and runs

The pretrains run longer than VOCR4's: Small 216,000 steps (128,000 in VOCR4), Normal 150,000 (40,000), Large
108,000 (32,000). Each is about 22.5 h at the speeds measured on the first 1,000 steps. There are 2 seeds for
Small and 3 each for Normal and Large: 8 pretrains, 8 at once. Each pretrain gets two fine-tunes (VOCR4's
recipe, 15,000 steps), which differ only by the arm:

- **P**: every pool, real half 0.5: tt + tvt 0.15, rd + rvt 0.075, rp3 0.175, own 0.075, dn + nat 0.025.
- **F**: P with finder-tighten's robustness work:
  - `fa=` augmentation on the ink crops: the finder's extra rows, with box borders, neighbour lines, furigana and
    scenery, plus line-end cursors and portrait edges; the label is unchanged;
  - refined copies of every GRAD crop, made by the merged refinement, beside the GRAD crops themselves.

  The helper `vocr5-aug` implements both.

## Selection (development sample only)

The cohorts are the fixed development sample (`reserved.json` dev_sample, 116 VNs):

- teacher_val's dev part (959 lines);
- devnative's four dev files (765 crops);
- the whole-dialogue regions of finder-tighten's tv_dlg / dn_dlg cohorts on those VNs (336 + 169 regions at
  base, +4 and +8 px).

**Dev score** = ½ × the fixed-line dialogue CER (the mean of teacher_val and devnative, both polarity views) +
½ × the whole-dialogue CER (C GRAD finder with and without the refinement). Name lines are a secondary column.

Decisions use stratified VN-cluster bootstraps (10,000 resamples), with the components' VNs resampled
independently:

1. **F over P**, per tier. F is adopted when the interval of the mean seed-pair difference lies below 0, every
   seed pair improves, and no component's interval lies above 0.
2. **The adopted arm over the incumbent.** The same rule against the incumbent's single blob. If it fails, the
   incumbent stays the tier's model.
3. **The run.** The adopted arm's lowest dev score among the blobs within the tier's limits.

## Locked report, CLANNAD, release

The locked report runs once, after the three selections are committed: `eval/vocr3_report.py` with a VOCR5 spec.
VOCR5's three models are scored against VOCR1, VOCR_H15, F20 and the VOCR3 and VOCR4 models on:

- fixed lines and whole dialogue of vita-test and psp-test, dialogue primary;
- VOCR2's gates (reported);
- the deployed beam + 1 MB LM decode (report only).

CLANNAD is read once through owner-drive-ingest's extra-arms path. Every VOCR5 model trained on the owner's frames,
which include planetarian's PSP port, a Key / Prototype release like CLANNAD's Vita port.

The three blobs then go on a full release with SHA256SUMS and the NVIDIA CC BY 4.0 attribution. A vita-jp-overlay
PR makes them the Small / Normal / Large tiers, with Normal as the default.
