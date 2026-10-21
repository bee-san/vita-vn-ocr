# textspot3: a 500 KB text spotter (issue #108)

textspot (#104, `docs/textspot.md`) selected a full-frame text spotter of 5,732 B of int8 weights, `B_h2_s4_c24n4_seed2`
with det+refine (est. 139 ms per 960×544 frame on one 333 MHz A9 core). textspot2 (#106, `docs/textspot2.md`) tried
spotters up to 18.6 KB and selected none: nothing was better than the winner on all-text CER, which UI and menu text
dominate, and the winner is the best of its recipe's three seeds. Its d2s design (a stride-8 trunk with a 2×2
sub-pixel head) cut development dialogue CER from 3.01% to 1.13% at 7,416 B and est. 80 ms, but was not better on
all-text CER (35.75%). The owner (2026-10-21 08:48Z): "Textspotter is good, but let's go even more. Can we build a
500kb model or less? to get better accuracy".

So textspot3 looks for the most accurate spotter up to 500 KB, est. 300 ms and a 256 KiB arena. Dialogue CER is the
primary metric (owner, 2026-10-14). All-text CER and F1 must not get worse, and a bigger model must earn its size on
all-text CER.

This document starts as the pre-registered protocol (`eval/textspot3_protocol.json`, with the rule as code in
`eval/textspot3_rule.py`). It is frozen before any textspot3 training run, data job and score. Results are added
below "Results" as they are measured. Nothing above "Results" changes except through a dated amendment, listed in
the protocol's `amendments`.

## Protocol

### What is reused unchanged

From textspot and textspot2, as merged (`7a6dce0`, `5af7741`):
- **The development set** `eval/manifests/textspot-dev.jsonl`: 188 frames of 81 dev_sample VNs, 885 instances,
  sha256 `0392a1d3…`. It is the only data the rule reads.
- **The metrics**, `eval/textspot_metrics.py`:
  - P / R / F1 at IoU ≥ 0.5, per-class recall;
  - the end-to-end all-text CER (every class, plus the characters read from boxes that match no reference);
  - the dialogue CER (classes dialogue and choice, without those readings);
  - the VN-cluster bootstrap: 10,000 resamples, seed 0.
- **The harness**, `eval/textspot.py`'s `det:` and `detref:` arms:
  - every box is read by the C runtime in W8I with greedy decoding, VOCR5-N for horizontal boxes and VOCR5-V for
    vertical ones (the four recognizers' sha256 as textspot pins them);
  - C parity with the twin per frame;
  - through textspot2's wrapper `eval/textspot2.py` (its reading cache, its VTSP version dispatch), extended for
    version 3 by a textspot3 wrapper.
- **The baseline** `eval/textspot_baseline_20261019.json` (eight arms). It is reported only here: the rule
  compares with `ref`.
- **The data**:
  - the synthetic half, textspot's `data/manifests/textspot_shards_v1.json` (80,000 whole frames);
  - the real half, textspot2's `data/manifests/textspot2_shards_v1.json` (52,785 canvases: its meiki labels on
    textspot's 28,265 canvases plus 24,520 more screenshots, RESERVE_OK);
  - textspot's mix per source.
- **The spotters' formats and runtimes**:
  - VTSP version 1 (`synth/textspot_twin.py`, `runtime/vocr_textspot.c`);
  - VTSP version 2, the d2s format (`synth/textspot2_twin.py`, `train/textspot2_export.py`,
    `runtime/vocr_textspot2.c`);
  - textspot's decode, blocks, order and pick.
- **The trainer**: `train/textspot2_train.py` and the files it imports, unchanged; textspot3's recipes extend it in a
  new file.
- **The estimates**: the A9 cycle model's high bound × 1.0007 at 333 MHz (`eval/textspot_a9.py`'s method), plus
  1.4 ms per refined horizontal box at the arm's median on the development Vita frames for det+refine.
- **The protected sets**: the test set `eval/manifests/textspot-test.jsonl` (textspot amendment 3) and the CLANNAD
  long run, read through `eval/textspot_protected.py`'s functions.
- **The teacher ceiling** (meiki.text.detect v0.1, `docs/textspot2.md`): report only, not re-run.

### The reference

| role | arm | VTSP B | est. ms @333 | arena B | dev F1 | CER all | dialogue |
|---|---|---:|---:|---:|---:|---:|---:|
| `ref`: the textspot winner | `B_h2_s4_c24n4_seed2` det+refine | 5,732 | 139.09 | 80,799 | 0.5416 | 32.04 | 3.01 |

- Its development units (VOCR5-N) are textspot's, read, not re-run; the protocol pins their sha256.
- `ref` is the best of its recipe's three seeds (all-text CER 32.04 / 34.59 / 36.28%, textspot2). So "not resolved
  worse than `ref`" is a strict guard.

### Budget

- VTSP blob ≤ **512,000 B**: the whole file, so int8 weights, biases, requantisation constants and headers. The
  int8 weight count is reported beside it.
- Est. ≤ **300 ms** per 960×544 frame on one A9 core at 333 MHz, on the runtime that runs the blob's version:
  - the A9 cycle model's high bound × 1.0007 for the default call (dark pass included);
  - for det+refine, plus 1.4 ms per refined horizontal box at the arm's median on the development Vita frames.
- Arena ≤ **262,144 B** (the Normal tier's ceiling) for any frame up to 960×544: the runtime's
  `*_arena_bytes(m, 960, 544)`, and for det+refine the larger of it and the refinement's 72,391 B. A shape over it
  is not eligible, and the record says what needs the room.
- The C runtime gives the twin's boxes on every development frame.
- Reported: 444 MHz (× 333/444), the PSP's 480×272, VitaSDK `-Os` code sizes, the stack peak.

### Two families

**T, the trunk** (VTSP version 2, textspot2's d2s format; runtime `runtime/vocr_textspot2.c` as merged, no runtime
change):
- `train/textdet_model.layers`' grammar: stem 3×3 stride 2, down, down8, separable blocks with dilations;
- a stride-8 trunk of up to 64 channels (the runtime's limit) with a 1×1 head to 16 channels: the 2×2 sub-pixel head,
  read as stride-4 maps;
- the scale `half_above` 480 (a frame wider than 480 px is halved).

On every development and test frame (widths 960, 728, 725 and 480), 480 halves exactly the frames that textspot's
600 halves. It also bounds the up-to arena by a 480-px-wide network input instead of a 600-px one.

**M, the multi-scale family** (a new VTSP version 3). The specification is `synth/textspot3_twin.py`, the
networks and export are `train/textspot3_model.py` and its exporter, and the runtime is `runtime/vocr_textspot3.c`.
All are to be written and pinned by an amendment before any M run.
- **Input**: textspot's grey and its exact 2×2 halving, applied when the frame is wider than 480 px **or** taller
  than 272 px. The network's input is then at most 480×272 for any frame up to 960×544. On every development and
  test frame this is the halving that textspot's `half_above` 600 applies.
- **Stream part**: textdet_model's grammar (stem, down, down8, blocks) to stride 8, computed row by row as version 1
  computes it. Its last layer's rows form a whole stride-8 map.
- **Map part**: an ordered list of layers on whole maps. There are two kinds:
  - a 3×3 depthwise layer, stride 1 or 2, dilation d, zero padding;
  - a 1×1 layer over the concatenation of up to three earlier maps, each read through a nearest upsampling by
    2^u (u ≤ 3) to the layer's resolution. The sources of one 1×1 layer share one activation scale.

  Both use version 1's integer arithmetic: int8 weights per output channel, an int32 bias, the fixed-point
  requantisation, and a ReLU or none. This builds stride-16, 32 and 64 stages of separable blocks and an FPN-style
  top-down neck back to stride 8.
- **Head**: the last layer is a 1×1 to 16 channels at stride 8, in version 2's d2s channel order, read as stride-4
  maps. textspot's decode then runs unchanged at stride 4 (min cells 2).
- **Versions**: a version-1 or version-2 blob runs bit for bit as before in the new twin and runtime, and the frozen
  files are not touched. A runtime without the extension refuses version 3.
- **Shapes**: three size points, S, M and L, at about 100 KB, 250 KB and up to 500 KB. Each must be est. ≤ 300 ms with
  a margin, and each puts most of its parameters at strides 16-64, so they are parameter-heavy and compute-light.
  They are chosen by an amendment after the runtime's A9 rows for probe shapes exist, before any M run.

### Recipes

- **`base`** is textspot2's `data` plus `ema`, the extra textspot2's stage 2 kept:
  - textspot2's real labels with textspot's synthetic half;
  - an exponential moving average of the weights (decay 0.999 from step 1,000). At the QAT start the averaged
    weights replace the live ones and the BatchNorm statistics are re-estimated on 32 batches. During QAT an
    average of the QAT weights is kept, and it is the run's model;
  - 30,000 steps of 16 windows of 256×512 at the network's scale;
  - AdamW 4e-3, 500 warm-up steps, cosine to 1e-4;
  - QAT from step 24,000;
  - the last step is the run's model.
- **`size`** is the UI/menu emphasis: `base`, with every positive line's kernel cells weighted by
  clamp(√(64 / n), 1, 4), where n is the line's kernel cell count at the map's stride. The weight multiplies the
  line's label weight. A two-character UI label weighs up to 4× a dialogue line; dice and the normalisations are
  unchanged.
- **`clean`** is cleaner meiki labels: `base`, with two rules at load time on the real half.
  - Agreement filtering: a positive of label weight < 1 becomes an ignore area. Such a positive is a meiki box of
    score < 0.8 that neither a reading-pipeline line nor a textspot winner box agrees with.
  - A vertical positive longer than 16 times its width, or narrower than 6 px at the canvas's scale, becomes an
    ignore area. These are the long thin strips textspot2's QA found.
- **`long`** is `base` at 60,000 steps: peak LR 2e-3, 1,000 warm-up steps, cosine to 1e-4, QAT from step 48,000.
- **`kd`** (stage 2) is `base` plus the kept extras plus textspot2's distillation loss from a teacher spotter, on
  the stride-4 maps:
  - the label loss;
  - BCE(K, σ(K_t)) over the valid cells;
  - 0.5 BCE(O, σ(O_t)), and smooth L1 on (A1 − A1_t) / 8 and (A2 − A2_t) / 8, on the cells where σ(K_t) > 0.5.

  The teacher's maps come from the same window at the same scale.

New trainer: `train/textspot3_train.py`. It imports textspot2's trainer and leaves textspot2's pinned files
unchanged; it is pinned by an amendment before any run.

### Seeds

- Every arm (a recipe on a shape) is trained with seeds 1, 2 and 3, and every seed is reported.
- For each decode (det, det+refine), the arm is represented by its **median seed by development dialogue CER**
  (`textspot3_rule.median_seed`). The seeds are sorted by dialogue CER, then all-text CER, then −F1, then seed, and
  the representative is index n // 2 (with two finished seeds, the worse one). It is never the best seed.
- Eligibility is checked on that seed's blob, with no substitution.
- A crashed run restarts from scratch with the same seed, at most three attempts.

### The rule (pre-registered; `eval/textspot3_rule.apply_rule3`)

- **Candidates**: the median-seed representatives, per decode, of every textspot3 arm.
- **Eligible**: within the budget above.
- **Qualifies**: against `ref`, on the paired development frames with VN-cluster 95% intervals, all three hold:
  - the dialogue CER difference is resolved below 0;
  - the all-text CER difference is not resolved above 0;
  - the F1 difference is not resolved below 0.
- **Cost order**: VTSP bytes, then est. ms at 333 MHz, then arena, then det before det+refine, then the name. So
  smaller comes first: memory first, as `AGENTS.md` orders it.
- **The pick**: the cheapest qualifier. A costlier qualifier replaces it when, compared with it, the all-text CER
  difference is resolved below 0 and the dialogue CER and F1 differences are not resolved worse. The cheapest such
  replaces it, and this repeats until none does. So a bigger model is picked only when it is resolved better on
  all-text CER than the smaller one it replaces.
- **No qualifier**: the verdict is negative and textspot's winner stands.
- **Report only**:
  - textspot's bar (`textspot_metrics.beats` against its eight baselines) for every candidate;
  - recall per class. If UI/menu recall does not rise with capacity, the results say so;
  - the Pareto front of bytes, est. ms, arena, F1, CER all, dialogue CER and per-class recall;
  - "what each arm bought";
  - VOCR5-S and VOCR5-L readings of the pick.

### Arms and stages

**Stage 1** runs after the amendments that pin the data, the trainer, the shapes and their A9 rows:
- T: `base` on each registered T shape;
- M: `base` on S, M and L; `size` and `clean` on M; `long` on M;
- seeds 1-3 each.

**Stage 2** runs after every stage-1 run is scored, by fixed rules that read stage 1's median seeds. Each choice is
recorded by an amendment, with the figures it read, before its runs. Stage 2's development intervals are therefore
optimistic, and the protected sets are the check.
- **Kept extras**:
  - `size` is kept when, on M, its det+refine all-text CER is below `base`'s and its dialogue CER is at most
    0.25 points above `base`'s;
  - `clean` is kept when its det+refine dialogue CER is below `base`'s and its all-text CER is at most 1.0 point
    above `base`'s.
- **The `kd` teacher**: the eligible stage-1 candidate of the largest M shape that has one, with the lowest
  all-text CER (any recipe or decode). Its median seed's final model is the teacher.
- **The `kd` students**: the two smallest shapes (by VTSP bytes, T or M) other than the teacher's whose stage-1
  `base` candidate qualifies against `ref`. When fewer than two qualify, the smallest eligible shapes make up two.
  Each is trained with `base` + the kept extras + `kd`, seeds 1-3.
- **The kept extras on L**: `base` + the kept extras on L, seeds 1-3, when an extra is kept.

### Data

**Excluded from all training** (as textspot2): every `reserved.json` test and dev_sample VN and test session; every
eval and eval/linefind screenshot and every linefind VN; CHAOS;CHILD (v14018); every development and test frame's
screenshot. The shards are textspot's and textspot2's, unchanged and already checked; no new pool is drawn.

### Estimates

- **Binding**: the A9 cycle model, as textspot used it:
  - the Vita compiler's code of the runtime that runs the blob's version (`vocr_textspot2.c` for T,
    `vocr_textspot3.c` for M);
  - random blobs of each shape, run under the plugin qemu-arm on textspot's pinned procedural frames at 960×544 and
    480×272;
  - priced by `vita/a9_cycles.py`.

  The tables are pinned by an amendment before any score.
- **Reported**: 444 MHz, the PSP's 480×272, VitaSDK `-Os` code sizes and the stack peak. MACs and bytes are exact.
- Every console time and memory figure is a host estimate: nothing runs on a console, since hardware work is paused.

### Protected sets, release and Rubi

- **Protected sets**: after the selection is committed and pushed, the pick is read **once** through
  `eval/textspot3_protected.py`, with attempt records under `$HERD/shared/private/textspot3`:
  - vita-test + psp-test as whole frames;
  - CLANNAD.

  Its comparisons are paired against `ref`'s stored protected units (textspot's record, read, not re-run).
  Nothing is read when nothing is selected. No model of this work runs on a test frame or a CLANNAD page before that.
- **Release**: the pick is published as a pre-release, with SHA256SUMS, ATTRIBUTION.md and a model card. The card
  says that meiki.text.detect (LGPL-3.0) gave training labels only and its weights are not shipped.
- **Rubi**: a DRAFT Rubi_for_vita PR on top of draft #228's branch pins the pick. Its memory ledger loads the bigger
  blob only while scan screen runs. It is never merged by the fleet.

### Disclosures registered in advance

- **References built from MeikiOCR's boxes.** textspot's development references started from MeikiOCR's boxes
  refined to ink, and spotters trained on meiki's boxes are favoured by that construction. The test set's dialogue
  lines (eval's) and CLANNAD's references are independent of it.
- **The arms read development errors.** They were chosen after reading textspot's and textspot2's development
  errors by class and by text height: issue #108's table, `textspot_metrics.match_frame` on the stored boxes.
  - Of the 519 UI references, 235 are under 8 px tall at the network's scale, 215 of them on the scaled PC frames.
  - The spotters find 32-42 of those; meiki at 960×544 finds 201 and at 320×192 finds 113.

  So the development set informs the design, though not the selection.
- **`ref` is a lucky seed.** It is the best of its recipe's three seeds.
- **Chance near 0.** The development set has 81 VNs. With about 30 candidates, intervals near 0 can still be chance.

### Amendments

Each amendment is dated in `eval/textspot3_protocol.json` and precedes the runs and scores it names.

1. **Family T's shapes** (2026-10-21T09:47Z, before every textspot3 run and score).
   - Three shapes of the T_ROWS grid (`train/textspot3_shapes.CANDIDATES`, measured in
     `eval/textspot3_a9_t_20261021.json` on `runtime/vocr_textspot2.c` as merged, at `half_above` 480, all with c1 16):

     | shape | what it scales | VTSP B | est. ms @333 | arena B |
     |---|---|---:|---:|---:|
     | `t64n4c16` | width: 64 channels at textspot2's depth of 4 blocks | 27,456 | 157.7 | 129,311 |
     | `t48n8c16d8` | depth and receptive field: 8 blocks of 48 channels, a dilation of 8, RF 383 px | 32,560 | 182.8 | 238,271 |
     | `t64n8c16` | width and depth: 8 blocks of 64 channels | 51,008 | 260.8 | 240,991 |

   - Not registered: the 64-channel trunks with a dilation of 8, which are over the arena budget (309,343 and
     312,911 B, for a 17-row ring of 64 planes). The other grid points lie between the registered ones.
   - Pinned as measured: `train/textspot3_shapes.py` (the registry), `eval/textspot3_a9.py` (a thin wrapper over
     `eval/textspot2_a9.py`'s measure and `eval/textspot_a9.py`'s report, with `half_above` as a parameter) and the
     table. The table also holds textspot2's `s8_c24n4x2` and `s8_c48n4x2` at `half_above` 600, all 8 rows identical
     to `eval/textspot2_a9_d2s_20261020.json`.
   - These rows are the T shapes' eligibility source, unless a later amendment pins a re-measurement on the same
     runtime. Every time is a host estimate.
2. **The scoring wrapper** (2026-10-21T10:15Z, before any score of a textspot3 arm).
   - `eval/textspot3.py` (`cbuild`, `seed-cache`, `run`, `read-pick`, `headline`, `select`, `tables`) and
     `eval/textspot3_protected.py` (`tests`, `clannad`: the pick only, once per command, with attempt records under
     `$HERD/shared/private/textspot3`).
   - They import `eval/textspot.py`, `eval/textspot_metrics.py`, `eval/textspot2.py` and `eval/textspot3_rule.py`
     unchanged.
   - Family T's version-2 blobs go through textspot2's path exactly. Family M's version-3 blobs go through
     `synth/textspot3_twin.py` and `runtime/vocr_textspot3_test` once those are merged. If the wrapper's version-3
     path has to change, a later amendment re-pins it, before any family-M score.
   - Eligibility is `eval/textspot.py`'s, with each shape's own exact MACs checked against its A9 row. A9 rows come
     only from tables this protocol pins. A seed without a row, or a finished run that is not scored, is refused.
   - The reading cache `$HERD/shared/artifacts/textspot3/eval/cache` is seeded once by a byte-for-byte copy of
     textspot2's development cache.
3. **The trainer, the sweep and stage 1's T plan** (2026-10-21T10:58Z, before every textspot3 run).
   - `train/textspot3_train.py` (recipes `base`, `size`, `clean`, `long`, `kd`; families T and M), merged from
     the train helper's `9d9c02b`.
     - `base` equals textspot2's `data` + `ema` step for step on tiny shards (logs, QAT checkpoint, int8 network).
     - Family T exports VTSP version 2 at `half_above` 480. Family M (`train/textspot3_model.py`, the version-3
       format of `2d42f9a`, pinned here with `synth/textspot3_twin.py`) exports VTSP version 3 at 480 / 272.
     - The T path equals the probed trainer: 10 of 10 identity checks are identical.
   - `clean` over the pinned pool turns 26,097 of 299,986 real positives into ignore areas, on 10,050 of 52,785
     canvases:
     - 25,210 of label weight below 1;
     - 887 by the vertical length rule;
     - none by the 6 px width rule, since textspot2's labels already keep 6-96 px.
   - The sweep is `eval/reproducibility/textspot3-20261021/` (`sweep.sh`, `textspot3_plan.py`, `check_run.py`,
     `runs_summary.py`), run from a detached worktree at this amendment's commit.
     - Stage 1's T plan `stage1_T_plan.jsonl` has 9 runs: `base` on the three T shapes, seeds 1-3. The probe puts
       it at about 14.8 slot-hours, with `base` at 137.6 / 193.6 / 242.7 ms per step.
     - It runs at most 6 at once for now.
   - Family M's stage-1 plan comes with the amendment that registers S, M and L.
