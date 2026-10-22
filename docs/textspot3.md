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
4. **Family M's size points** (2026-10-21T11:46Z, before every family-M run and every family-M score).
   - Chosen from the A9 rows of 18 probe shapes on `runtime/vocr_textspot3.c`
     (`eval/textspot3_a9_m_probe_20261021.json`: random version-3 blobs on textspot's procedural frames; no
     development frame or score was read):

     | point | shape | multi-scale part on T's `t64n4c16` trunk | VTSP B | MACs / frame | est. ms @333 | arena B |
     |---|---|---|---:|---:|---:|---:|
     | S | `qS4` | stages 64 / 128 / 192, no blocks; neck T 64, T8 48, 1 smoothing block | 117,184 | 75.1 M | 204.0 | 256,247 |
     | M | `qM4` | stages 96 (1 block) / 192 / 384; neck T 96, T8 48 | 247,392 | 96.1 M | 228.7 | 256,247 |
     | L | `qL5` | stages 96 / 192 / 384, one block each; neck T 96, T8 48 | 447,392 | 107.3 M | 246.0 | 256,247 |

   - All three keep family T's `t64n4c16` trunk (64 channels, 4 stride-8 blocks), so `qS4` minus `t64n4c16` measures
     what the multi-scale part buys at a fixed trunk. From S to L, most of the added weights sit at strides 32 and
     64 (L: 263 KB at stride 64), for little extra compute.
   - Not chosen:
     - the probes with a 2-block stride-8 trunk (113-260 ms), which would change the trunk and the deep part at once
       against family T;
     - `qL9`, which only widens L's stride-16 stage;
     - `pL4` and `qL5n6`, which are over the budget (arenas 265,575 and 327,255 B).
   - These rows are upper bounds: the runtime's kernels may still get faster, bit for bit. The rows of record for
     family M are the ones the RUNTIME_READY amendment pins, before any family-M score. If no such amendment comes,
     these rows are the eligibility source.
5. **Stage 1's family-M plan** (2026-10-21T12:28Z, before every family-M run and score).
   - `stage1_M_plan.jsonl` has 18 runs: `base` on `qS4`, `qM4` and `qL5`, and `size`, `clean` and `long` on
     `qM4`, seeds 1-3 each. It comes from the pinned planner and the probe record
     `eval/textspot3_train_probe_stage1m_20261021.json`:
     - `base` runs at 242.5 / 252.3 / 294.6 ms per step on S / M / L;
     - the plan is about 46.7 slot-hours in all.

     The trainer and sweep are amendment 3's.
   - Both sweeps share the runs directory and are capped together at 7 runs at once.
   - The version-3 runtime of `4acee6a` is pinned here for the training sweep's run checks only. The scoring, the
     code sizes and the A9 rows of record use the runtime the RUNTIME_READY amendment pins, before any family-M score.
   - The scoring wrapper's version-3 path is re-pinned (`eval/textspot3.py`, `tests/test_textspot3_eval.py`): its
     real-v3 test now reads `PROBES`, and a family-M blob that does not halve at 480 × 272 is refused. Its
     version-2 path is unchanged.
6. **The version-3 runtime of record and family M's A9 rows** (2026-10-21T13:58Z, before any family-M score).
   - `runtime/vocr_textspot3.c` at `d75b721`, after a speed pass (fewer instructions in the map part, bit for bit
     the same outputs), and its driver.
   - Checks on it:
     - `ci/runtime.sh` with gcc and clang;
     - large ASan + UBSan parity runs: 4,574 and 3,961 runs, with every version-1/2 run equal to `vocr_textspot2_test`;
     - qemu-arm Cortex-A9 parity of the Vita compiler's own objects;
     - strict VitaSDK and psp-gcc builds;
     - an independent audit (AUDIT_OK).
   - The rows of record for family M, `eval/textspot3_a9_20261021.json`:

     | shape | est. ms @333 / @444 | arena B |
     |---|---:|---:|
     | `qS4` | 195.1 / 146.3 | 256,247 |
     | `qM4` | 216.6 / 162.4 | 256,247 |
     | `qL5` | 231.8 / 173.8 | 256,247 |

     They replace amendment 4's probe rows. Family T keeps `eval/textspot3_a9_t_20261021.json` (re-measured: 20 of
     20 rows identical).
   - Code (VitaSDK `--gc-sections` `.text`): 23,880 B at -O2 NEON and 19,452 B at -Os. The stack peak is 2,712 B.
7. **Stage 2** (2026-10-21T20:58Z, after all 27 stage-1 runs were trained and scored, before every stage-2 run and the
   selection). The fixed rules, applied to stage 1's median seeds:
   - `size` is not kept: on `qM4` det+refine its all-text CER is 36.45, not below `base`'s 35.80.
   - `clean` is not kept: its dialogue CER is 1.45, not below `base`'s 1.33.
   - The kd teacher is `base_qL5_seed2`: `qL5` det+refine, all-text CER 34.58, the lowest of the largest M shape.
   - The kd students are `t64n8c16` and `qS4`, the two smallest shapes whose `base` candidate qualifies against
     `ref`. `t64n4c16` and `t48n8c16d8` do not qualify: their all-text CER is resolved worse, with lower bounds
     +0.26 and +0.61.
   - Disclosed: `qS4`'s all-text interval against `ref` has a lower bound of exactly 0.0000 after the metrics'
     four-decimal rounding. As the rule reads it, that is not resolved worse, so it qualifies: a knife edge.
   - No `base` + extras on L, since no extra is kept.
   - The runs: `stage2_plan.jsonl` has 6 runs, `kd_t64n8c16_seed1-3` and `kd_qS4_seed1-3`. Every record carries
     the teacher's `final.pt` sha256, and every command line re-checks it. The plan is about 15.3 slot-hours.
   - The sweep's two bookkeeping fixes (the 5-run notice, the end of each sweep in a shared log) are re-pinned for
     stage 2. The run command and the checks are unchanged. The stage-1 runs record is pinned with them.

## Results

Every number here is a host measurement on the development set unless it says otherwise:
- 188 frames of 81 VNs, read by the C runtimes;
- horizontal boxes read by VOCR5-N and vertical ones by VOCR5-V;
- textspot's frozen metrics, with VN-cluster 95% intervals.

Every console time is an estimate from the A9 cycle model. Nothing of this work has run on a console.

### Answer first

- **The rule has a pick: `kd_t64n8c16_seed3` with det+refine.** It is family T's widest and deepest trunk: 8
  separable blocks of 64 channels at stride 8 with the d2s head, in VTSP version 2, distilled from the 447 KB
  multi-scale `qL5`.
  - Its blob, `textspot3_t64n8c16_kd_w8.vtsp`, is **51,008 B** (39,840 int8 weights, sha256 `20c3f7cc…`).
  - Est. **267.8 ms** per 960×544 frame on one A9 core at 333 MHz, and 200.9 ms at 444 MHz. That is the spotter's
    260.8 ms plus the refinement of 5 lines.
  - Arena: **240,991 B** for any frame up to 960×544.
- **Development** (`eval/textspot3_dev_20261022.json`; the selection ran once):

  | | F1 | CER all | dialogue | UI recall |
  |---|---:|---:|---:|---:|
  | textspot's winner (`ref`, 5,732 B, est. 139.1 ms) | 0.542 | 32.04 | 3.01 | 0.324 |
  | **the pick** | **0.661** | **32.51** | **1.15** | **0.389** |
  | pick − `ref` (95% interval) | +0.119 (+0.073 to +0.168) | +0.47 (−4.39 to +5.19) | −1.86 (−3.31 to −0.77) | |

  - Dialogue CER and F1 are resolved better; all-text CER is level.
  - It also clears textspot's bar against all eight of its baselines (report only).
  - Read by VOCR5-S and VOCR5-L (report only): F1 0.661 for both, CER all 33.21% / 32.82%, dialogue 1.61% / 1.05%.
- **How the rule got there.** All 22 candidates are eligible, and 7 of them qualify (dialogue resolved better, no
  guard resolved worse). The chain runs as follows:
  1. The cheapest qualifier is `base_t64n8c16` with det+refine (51,008 B, est. 267.8 ms; all-text CER 36.10%).
  2. Its distilled twin, `kd_t64n8c16`, replaces it: all-text −3.59 (−6.12 to −1.21), with dialogue
     (−0.25, −0.94 to +0.26) and F1 (+0.036) not worse.
  3. No bigger qualifier is resolved better than the pick on all-text CER: `qS4` +4.28 / +2.61 (base / kd),
     `qM4` +3.29, `qM4` long +0.12, `qL5` +2.07.

  So the 117-447 KB multi-scale models qualify but do not earn their size. The largest one's gain reaches the pick
  through distillation.
- **The protected sets, read once** (after the selection commit `9b1b737`, 00:34:14Z):
  - **vita-test + psp-test** as whole frames (176 frames of 28 VNs):
    - F1 **0.776** against the winner's 0.694 (+0.082, +0.039 to +0.132);
    - all-text CER **20.57%** against 20.02% (+0.55, −2.59 to +4.19, level);
    - dialogue CER **2.29%** against 3.00% (−0.70, −1.44 to −0.05, resolved better);
    - VOCR5-S and VOCR5-L give 2.53% and 2.23% dialogue, both resolved better than the winner read the same way.
  - **CLANNAD whole frames** (2,052 dialogue lines on 1,867 pages): dialogue CER **0.99%** against the winner's
    1.14% (−0.16, −0.23 to −0.09); S 1.34% against 1.44%, L 0.80% against 0.94%, each resolved better.
- **What it does not do well:**
  - **UI labels.** Recall is 0.389 on development, but 0.371 on the test frames against the winner's 0.485.
  - **Merged lines.** The textspot3 spotters merge neighbouring lines or labels into one box more often than the
    winner does: 45-53 such boxes in the arms checked, against its 34. The readings of those boxes count as junk.
    That is the main reason all-text CER did not fall with the gains in dialogue and F1.
  - **Vita frames.** On development Vita frames all-text CER is 29.54% against 18.76%. On vita-test it is 23.42%
    against 20.95%, and dialogue 2.00% against 1.83%. The gains come from the PSP and Adrenaline frames.
- **The representative is the arm's luckiest seed on all-text CER.** The rule's median seed is taken by dialogue
  CER, and for `kd_t64n8c16` that seed is also its best on all-text CER (32.51% against 35.55% and 35.66%). The chain
  step above compares it with `base_t64n8c16`'s middle seed. The protected sets agree with the development verdict
  on dialogue and F1, and show all-text level.

### The selection (`eval/textspot3_dev_20261022.json`)

- **The arms:**
  - stage 1: 27 runs, 9 of family T and 18 of family M;
  - stage 2: 6 kd runs;
  - two decodes each, 66 seeds in all.

  Each arm and decode is represented by its median seed by dialogue CER: 22 candidates.
- **Eligibility:** all 22 are eligible, at est. 157.7-267.8 ms, arenas of 129,311-256,247 B and blobs of
  27,456-447,392 B. The C runtime equals the twin on all 188 frames for all 66 arms.
- **Qualifiers:** 7, all of them det+refine:
  - `base_t64n8c16`, `kd_t64n8c16`;
  - `qS4` base and kd;
  - `qM4` base and long;
  - `qL5`.

  No det decode qualifies: their dialogue CER is not resolved better.

| arm (median seed, det+refine) | VTSP B | est. ms @333 / @444 | arena B | F1 | CER all | dialogue | UI recall | qualifies |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `base_t64n4c16` (seed 1) | 27,456 | 164.7 / 123.5 | 129,311 | 0.599 | 35.83 | 1.76 | 0.329 | no: all-text +3.80 (+0.26 to +7.26) |
| `base_t48n8c16d8` (seed 2) | 32,560 | 189.8 / 142.3 | 238,271 | 0.614 | 36.06 | 1.53 | 0.314 | no: all-text +4.02 (+0.61 to +7.48) |
| `base_t64n8c16` (seed 2) | 51,008 | 267.8 / 200.9 | 240,991 | 0.625 | 36.10 | 1.40 | 0.335 | yes |
| **`kd_t64n8c16` (seed 3): the pick** | **51,008** | **267.8 / 200.9** | **240,991** | **0.661** | **32.51** | **1.15** | **0.389** | **yes** |
| `base_qS4` (S; seed 3) | 117,184 | 202.1 / 151.6 | 256,247 | 0.634 | 36.79 | 1.73 | 0.335 | yes (all-text lower bound 0.00) |
| `kd_qS4` (seed 1) | 117,184 | 202.1 / 151.6 | 256,247 | 0.606 | 35.13 | 1.20 | 0.326 | yes |
| `base_qM4` (M; seed 3) | 247,392 | 223.6 / 167.7 | 256,247 | 0.638 | 35.81 | 1.33 | 0.339 | yes |
| `size_qM4` (seed 3) | 247,392 | 223.6 / 167.7 | 256,247 | 0.619 | 36.45 | 1.20 | 0.335 | no: all-text +4.42 (+0.09 to +9.03) |
| `clean_qM4` (seed 3) | 247,392 | 223.6 / 167.7 | 256,247 | 0.614 | 36.34 | 1.45 | 0.306 | no: all-text +4.30 (+0.25 to +8.79) |
| `long_qM4` (seed 2) | 247,392 | 223.6 / 167.7 | 256,247 | 0.653 | 32.63 | 1.15 | 0.370 | yes |
| `base_qL5` (L; seed 2) | 447,392 | 237.4 / 178.0 | 256,247 | 0.636 | 34.58 | 1.50 | 0.368 | yes |

The det decodes, every seed and every comparison are in the record and in `eval/textspot3.py tables`' output.

### Recall per class: does UI recall rise with capacity?

| det+refine, median seed | dialogue (249) | name (96) | choice (15) | UI (519) | scene (6) |
|---|---:|---:|---:|---:|---:|
| `ref` | 0.944 | 0.688 | 0.867 | 0.324 | 0.500 |
| T `t64n4c16` / `t48n8c16d8` / `t64n8c16` | 0.984 / 0.996 / 0.992 | 0.708 / 0.698 / 0.750 | 0.933 | 0.329 / 0.314 / 0.335 | 0.167 / 0.000 / 0.167 |
| M `qS4` / `qM4` / `qL5` | 0.984 / 0.996 / 0.984 | 0.812 / 0.781 / 0.771 | 1.000 / 1.000 / 0.933 | 0.335 / 0.339 / 0.368 | 0.333 / 0.000 / 0.000 |
| `long_qM4` | 0.996 | 0.802 | 0.933 | 0.370 | 0.000 |
| **the pick** | **0.988** | **0.823** | **0.933** | **0.389** | **0.167** |
| meiki.text.detect 960×544, det (textspot2's teacher ceiling, report only) | 0.992 | 0.875 | 1.000 | 0.780 | 0.333 |

- **UI recall rises with capacity, but only a little.**
  - The Spearman correlation with VTSP bytes over the 11 arms is 0.31 (det+refine) and 0.39 (det).
  - Along the M ladder, S → M → L gives 0.335 → 0.339 → 0.368.
  - The pick's 0.389 comes from distillation, not size.
  - The teacher's 0.78 stays far away.
- **Names gain too**: from 0.69 to 0.70-0.82.
- **Dialogue recall is near saturation everywhere**: 0.98-1.00.

### Where the all-text errors are

These are character errors in the development references by class, plus readings of boxes that match no reference
(det+refine, median seeds unless named). The reference characters are: dialogue 3,851, name 298, choice 136, UI
2,455, scene 30.

| arm | dialogue | name | choice | UI | scene | unmatched readings | total | CER all | unmatched boxes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ref` | 114 | 74 | 6 | 1,557 | 22 | 396 | 2,169 | 32.04 | 421 |
| `base_t64n4c16` | 65 | 70 | 5 | 1,547 | 24 | 715 | 2,426 | 35.83 | 281 |
| `base_t64n8c16` | 51 | 69 | 5 | 1,550 | 26 | 743 | 2,444 | 36.10 | 233 |
| `base_qM4` | 52 | 50 | 1 | 1,550 | 28 | 743 | 2,424 | 35.81 | 213 |
| `long_qM4` | 42 | 55 | 4 | 1,505 | 30 | 573 | 2,209 | 32.63 | 210 |
| `base_qL5` | 55 | 56 | 5 | 1,513 | 28 | 684 | 2,341 | 34.58 | 238 |
| **the pick** (`kd_t64n8c16`) | **41** | **49** | **5** | **1,414** | **28** | **664** | **2,201** | **32.51** | **214** |

- **Every textspot3 arm cuts the dialogue errors by 40-63%** (114 → 42-68) and leaves 33-50% fewer boxes unmatched
  (421 → 210-281). UI errors are level (1,505-1,606 against 1,557), except the pick's 1,414.
- **They read 45-98% more characters from unmatched boxes** (573-783 against 396). Most of those are boxes that
  overlap two or more references: neighbouring lines or UI labels merged into one box.
  - `ref` has 34 such boxes reading 251 characters.
  - `base_t64n8c16` has 53 reading 576, `long_qM4` 48 reading 470, `qL5` 52 reading 550, and the pick 47 reading 530.

  A merged box matches neither reference, so both are missed and the box's reading counts as junk. The stride-8
  trunk with the d2s head merges more than textspot's stride-4 trunk. A follow-up would start there, with a
  stride-4 level in the head or a decode that splits merged components.
- `size` weights small instances up and `clean` drops low-agreement labels. Neither changed the UI errors (`size`
  1,548, `clean` 1,606).

### What each arm bought (paired development differences, median seeds, det+refine)

| what | F1 Δ | CER all Δ, points | dialogue Δ, points |
|---|---:|---:|---:|
| multi-scale part at a fixed trunk: `qS4` − `t64n4c16` | +0.034 (−0.005 to +0.075) | +0.96 (−2.13 to +4.17) | −0.03 (−0.41 to +0.40) |
| `qM4` − `qS4` (117 → 247 KB) | +0.004 (−0.024 to +0.033) | −0.99 (−2.87 to +0.81) | −0.40 (−0.97 to +0.00) |
| `qL5` − `qM4` (247 → 447 KB) | −0.001 (−0.024 to +0.021) | −1.23 (−3.13 to +0.56) | +0.18 (−0.28 to +0.74) |
| width and depth: `t64n8c16` − `t64n4c16` (conductor's report, the same compare) | +0.025 (−0.006 to +0.058) | +0.27 (−2.82 to +3.49) | **−0.35 (−0.77 to −0.02)** |
| `size` − `base` (`qM4`) | −0.019 (−0.040 to +0.002) | +0.65 (−2.08 to +3.18) | −0.13 (−0.80 to +0.38) |
| `clean` − `base` (`qM4`) | −0.023 (−0.050 to +0.002) | +0.53 (−1.85 to +2.87) | +0.13 (−0.19 to +0.57) |
| `long` − `base` (`qM4`, 60,000 steps) | +0.015 (−0.013 to +0.045) | **−3.18 (−6.28 to −0.42)** | −0.18 (−0.97 to +0.34) |
| `kd` − `base` (`t64n8c16`, teacher `qL5`) | **+0.036 (+0.014 to +0.060)** | **−3.59 (−6.12 to −1.21)** | −0.25 (−0.94 to +0.26) |
| `kd` − `base` (`qS4`) | −0.028 (−0.058 to +0.001) | −1.67 (−4.47 to +0.84) | −0.53 (−1.36 to +0.06) |

In short:
- **Distillation from the largest model and the longer schedule bought all-text CER.** These are the only two
  resolved all-text gains.
- **Parameters at low resolution bought small, unresolved steps.** From S to L, all-text CER falls 2.2 points in two
  steps, and none of them is resolved. Width and depth in the trunk bought dialogue (−0.35, resolved), not all-text CER.
- **The size-balanced UI loss and the agreement-filtered labels bought nothing.**
- Against `ref`, every arm buys dialogue (−1.25 to −1.86 points) and F1 (+0.06 to +0.12).

### Seeds

Seeds of one arm still differ by several points. These are det+refine, all-text CER and then dialogue:
- `kd_t64n8c16`: 32.51 / 35.55 / 35.66% and 1.15 / 0.95 / 1.43%;
- `long_qM4`: 34.25 / 32.63 / 34.12% and 1.03 / 1.15 / 1.30%;
- `base_qM4`: 38.40 / 34.49 / 35.81%.

The rule reads the median seed by dialogue CER and never the best. For `kd_t64n8c16` and `long_qM4`, that seed happens
to be the arm's best on all-text CER (see "Answer first").

### The speed / size front (development)

13 of the 22 eligible representatives are on the front of VTSP bytes, est. ms at 333 and 444 MHz, arena, F1, CER
all, dialogue CER and UI recall. The main points:

| arm | VTSP B | est. ms @333 / @444 | arena B | F1 | CER all | dialogue |
|---|---:|---:|---:|---:|---:|---:|
| `ref` | 5,732 | 139.1 / 104.3 | 80,799 | 0.542 | 32.04 | 3.01 |
| `base_t64n4c16` det+refine | 27,456 | 164.7 / 123.5 | 129,311 | 0.599 | 35.83 | 1.76 |
| `base_t48n8c16d8` det+refine | 32,560 | 189.8 / 142.3 | 238,271 | 0.614 | 36.06 | 1.53 |
| **`kd_t64n8c16` det+refine (pick)** | **51,008** | **267.8 / 200.9** | **240,991** | **0.661** | **32.51** | **1.15** |
| `kd_qS4` det+refine | 117,184 | 202.1 / 151.6 | 256,247 | 0.606 | 35.13 | 1.20 |
| `long_qM4` det+refine | 247,392 | 223.6 / 167.7 | 256,247 | 0.653 | 32.63 | 1.15 |

- **Time binds before weights.** The pick uses 10% of the byte budget and 89% of the time budget.
- **Parameter-heavy shapes are fast but not better.** The multi-scale models add 90-420 KB, mostly at strides 32-64,
  for 37-73 ms over their trunk. `long_qM4` comes close to the pick's accuracy in 44 ms less, but the rule prefers the
  smaller blob unless the bigger one is resolved better on all-text CER.

### Per part (development, the pick against `ref`)

| part | frames | pick F1 / CER all / dialogue | `ref` F1 / CER all / dialogue |
|---|---:|---|---|
| vita | 39 | 0.686 / 29.54 / 1.24 | 0.623 / 18.76 / 1.50 |
| psp | 64 | 0.778 / 16.04 / 0.80 | 0.601 / 20.20 / 4.06 |
| adrenaline | 20 | 0.797 / 15.77 / 2.10 | 0.652 / 17.95 / 1.87 |
| pc | 40 | 0.504 / 56.13 / 1.14 | 0.385 / 58.16 / 3.71 |
| text-free | 25 | 0 frames with a box | 5 frames with a box |

The dialogue pick lands on 81 of 126 frames with dialogue, against the winner's 86 (report only).

### The protected sets, once (`eval/textspot3_protected_20261022.json`, `eval/textspot3_clannad_20261022.json`)

- The selection was committed and pushed first (`9b1b737`, 2026-10-22T00:34:14Z; tag
  `textspot3-selection-9b1b737`).
- The pick was then read once on each set; the attempt records are dated 00:34:42Z and 00:35:37Z.
- The C runtime equals the twin on all 259 test frames and all 2,177 CLANNAD pages.
- The comparisons use the winner's stored protected units (textspot's record, sha256-checked and reproduced
  before the attempts).

**vita-test + psp-test as whole frames** (176 scored frames of 28 VNs; 714 instances, 6,229 characters, dialogue
4,707):

| | P | R | F1 | CER all (N) | dialogue (N / S / L) | text-free frames with a box |
|---|---:|---:|---:|---:|---:|---:|
| **the pick** | 0.801 | 0.754 | **0.776** | 20.57 | **2.29** / 2.53 / 2.23 | 2 / 19 |
| textspot's winner | 0.625 | 0.780 | 0.694 | 20.02 | 3.00 / 3.38 / 3.00 | 6 / 19 |

- **Against the winner** (VOCR5-N):
  - F1 +0.082 (+0.039 to +0.132);
  - all-text CER +0.55 (−2.59 to +4.19), level;
  - dialogue −0.70 (−1.44 to −0.05), resolved better.

  With VOCR5-S and VOCR5-L, dialogue is −0.85 (−1.66 to −0.13) and −0.76 (−1.53 to −0.06).
- The pick beats each of the eight baselines as textspot's bar asks.
- **Per part:**
  - vita-test: F1 0.764, CER 23.42% / 2.00%, against the winner's 0.692 / 20.95% / 1.83%;
  - psp-test: 0.795 / 17.38% / 2.60%, against 0.705 / 18.99% / 4.21%;
  - report only, psp-test at 2× (Adrenaline): 0.795 / 17.05% / 2.78%, against 0.633 / 20.85% / 6.98%.
- **Recall:** dialogue 0.984 (winner 0.955), names 0.918 (0.926), choices 0.839 (0.839), UI 0.371 (0.485), scene
  0.40 (0.45).
- **Tightness:** mean IoU 0.844, ink coverage 0.980.
- **The dialogue pick** lands on the dialogue block on 65% of test frames with dialogue (the winner: 62%).

**CLANNAD long run** (owner-drive-ingest's frozen references, `refs_sha256` `35411688…`; whole frames; a reference
line reads the detections that lie mostly inside its GRAD box):

| | dialogue (2,052 lines) N / S / L | all 3,048 lines N / S / L |
|---|---:|---:|
| **the pick** | **0.99 / 1.34 / 0.80** | 1.73 / 2.03 / 1.38 |
| textspot's winner | 1.14 / 1.44 / 0.94 | 1.89 / 2.07 / 1.51 |
| #97's detector run on whole frames (`s8c24_spot`) | 1.29 / 1.43 / 1.03 | 1.33 / 1.75 / 1.20 |

- **Dialogue:**
  - against the winner: −0.16 (−0.23 to −0.09) with N, −0.10 (−0.17 to −0.03) with S, −0.14 (−0.22 to −0.07)
    with L;
  - against `s8c24_spot`: −0.30 (−0.45 to −0.16) with N.
- **All lines with speaker names:**
  - better than the winner: −0.16 (−0.23 to −0.09);
  - worse than #97's detector: +0.29 (+0.05 to +0.52) against `s8c24_textdet` and +0.40 (+0.23 to +0.58) against
    `s8c24_spot`. Like the winner, it finds fewer name plates.
- **The 133 pages without reference text:** the pick returns 504 boxes there, the winner 790.
- **For scale, not a paired comparison:** GRAD + refine on the hand-drawn dialogue region reads the same lines at
  0.92% (VOCR5-N, #97's record).
- **Training disclosure:** the real half includes the owner's planetarian PSP frames (Key / Prototype, like
  CLANNAD's Vita port). No CLANNAD-family frame is in any training pool.

### Console time and memory (host estimates)

| the pick (`t64n8c16`, VTSP v2, det+refine) | 960×544 frame | 480×272 frame |
|---|---:|---:|
| MACs (network) | 84.7 M | 84.7 M (a PSP frame is read as it is) |
| A9 cycle model, high bound × 1.0007, @333 / @444 MHz | 260.8 / 195.6 ms | 253.0 / 189.8 ms |
| + refinement (1.4 ms × 5 lines @333) | +7.0 ms | +7.0 ms |
| arena (any frame up to 960×544; the refinement's 72,391 B fits in it) | 240,991 B | 240,031 B |

- **Weights:** 51,008 B, 8.9× the winner's 5,732 B. The arena is 3.0× the winner's 80,799 B, because of the
  rings of 8 dilated 64-channel blocks.
- **Code:** the version-2 runtime `runtime/vocr_textspot2.c`, the merged d2s runtime of #106:
  - VitaSDK gcc 15.2 `.text` 11,844 B at -O2 NEON and 9,746 B at -Os;
  - 280-408 B more than `vocr_textspot.c`;
  - stack peak 2,904 B.

  Rubi's scan screen, which pins `vocr_textspot.c` and a version-1 blob, needs that file instead.
- **Every time above is an estimate**: nothing ran on a console.

### Measured and estimated

- **Measured on the host:**
  - every P / R / F1, CER, interval, recall and count above (the C runtimes, the frozen metrics);
  - blob, arena and code sizes and stack frames, which are exact.
- **Estimated:** every console time. The method is the A9 cycle model on the Vita compiler's code under qemu,
  the binding one; 444 MHz is that time × 333/444.
- **Not measured:** nothing has run on a console.

### Limits

- **One development set** of 81 VNs, built from MeikiOCR's boxes. Its 519 UI references are 59% of all references.
- **Stage 2's development intervals are optimistic.** Its choices read stage 1's development scores (amendment 7),
  and the pick comes from stage 2. The protected sets, read once, agree on dialogue and F1, and show all-text CER
  level with the winner.
- **The design read earlier development errors** (the disclosure registered in advance). The arms were chosen
  after reading textspot's and textspot2's development errors by class and text height.
- **Seed luck:** see "Seeds". The winner is itself the best of its recipe's three seeds.
- **The knife edge:** `qS4`'s base candidate qualified with an all-text lower bound of exactly 0. That is not a
  rounding effect: the resamples around the 2.5th percentile differ by exactly 0, so the rule's `ci[0] > 0` reads it
  as registered.
  It made `qS4` a kd student, but it is not the pick.
- **UI text, merged lines and Vita frames** are the weak spots ("Answer first", "Where the all-text errors are").
- **Notes from the audit of #113**:
  - **Text above "Results" is frozen** and keeps three slips:
    - amendment 7 gives `base_qM4`'s all-text CER as 35.80; it is 35.81 (2,424 / 6,770);
    - amendment 7 gives `qS4`'s lower bound as 0.0000 "after the metrics' four-decimal rounding"; it is exactly 0
      before rounding too (see the knife edge above);
    - amendment 3's "887 by the vertical length rule" counts that rule alone: 898 positives break it, 11 of them
      also of weight below 1.
  - **Torch against blob**: on trained checkpoints each layer of the fake-quantised torch network agrees with the
    integer blob within one code. Through 22-26 layers, though, 59-61% of the head codes are equal end to end.
    Nothing reported depends on that: every score and the shipped blob are the integer network (C == twin).
  - **Small recipe details**:
    - `size` counts a line's kernel cells inside the training window;
    - kd's O and A terms can include one stride-4 row or column past an odd-sized region.

    Neither changed a choice: `size` was not kept, and the kd effect is negligible.
  - **`runtime/vocr_textspot3.c`'s NEON 1×1 loop** forms, after an odd last cell, a pointer one cell beyond
    one-past-the-end. It is never dereferenced, but forming it is undefined in C99. The audit also lists four
    comment and portability nits in the same files. The files are pinned as the runtime of record of the family-M
    scores, and no console build uses them, so they are left as they are; a follow-up fixes them under a re-pin.
- **The multi-scale runtime** `runtime/vocr_textspot3.c` (VTSP version 3) is merged as a host and research tool.
  The pick does not need it: it is a version-2 blob. Nothing on a console uses version 3.

### Reproduce

```sh
H=$HERD; T3=$H/shared/artifacts/textspot3; P3=$H/shared/private/textspot3
# A9 rows (estimates): T on vocr_textspot2.c, M on vocr_textspot3.c
.venv/bin/python eval/textspot3_a9.py measure --runtime v2 --half-above 480 --out $T3/a9/t_main --jobs 5 --vitasdk $H/shared/artifacts/vita/vitasdk
.venv/bin/python eval/textspot3_a9m.py measure --shapes qS4,qM4,qL5 --out $T3/a9/rec_m --jobs 5 --vitasdk $H/shared/artifacts/vita/vitasdk
.venv/bin/python eval/textspot3_a9m.py measure --shapes qS4,qM4,qL5 --nodark --out $T3/a9/rec_m_nodark --jobs 5 --vitasdk $H/shared/artifacts/vita/vitasdk
.venv/bin/python eval/textspot3_a9r.py report $T3/a9/rec_m --also $T3/a9/rec_m_nodark --also-label VOCR_TEXTSPOT_NO_DARK \
    --t-json <the T report> --code-size <codesize JSON> --stack <stack JSON> --json eval/textspot3_a9_20261021.json   # the M rows of record
# training, from detached worktrees at 6f2bff0 (stage 1 T), 9bcdf51 (stage 1 M) and 8a0f8d2 (stage 2)
MAX_PAR=6 eval/reproducibility/textspot3-20261021/sweep.sh $T3/runs eval/reproducibility/textspot3-20261021/stage1_T_plan.jsonl
MAX_PAR=7 eval/reproducibility/textspot3-20261021/sweep.sh $T3/runs eval/reproducibility/textspot3-20261021/stage1_M_plan.jsonl
MAX_PAR=7 eval/reproducibility/textspot3-20261021/sweep.sh $T3/runs eval/reproducibility/textspot3-20261021/stage2_plan.jsonl
# scoring (private tmpfs for every gcc step), the selection once, the tables
.venv/bin/python eval/textspot3.py cbuild --out $T3/eval/cbuild
.venv/bin/python eval/textspot3.py run --runs $T3/runs --out $T3/eval/units --cbin $T3/eval/cbuild --models N
.venv/bin/python eval/textspot3.py select --units $T3/eval/units --a9 eval/textspot3_a9_t_20261021.json \
    --a9 eval/textspot3_a9_20261021.json --cbin $T3/eval/cbuild --runs $T3/runs --out eval/textspot3_dev_20261022.json
.venv/bin/python eval/textspot3.py read-pick --names kd_t64n8c16_seed3 --dev eval/textspot3_dev_20261022.json --models S,L
.venv/bin/python eval/textspot3.py tables --dev eval/textspot3_dev_20261022.json --units $T3/eval/units
# protected reads, once each (attempt records refuse a second run)
.venv/bin/python eval/textspot3_protected.py tests --cand T3:kd_t64n8c16_seed3 --decode det+refine \
    --dev eval/textspot3_dev_20261022.json --out $P3/protected/kd_t64n8c16_seed3_detref/tests --cbin $T3/eval/cbuild
.venv/bin/python eval/textspot3_protected.py clannad --cand T3:kd_t64n8c16_seed3 --decode det+refine \
    --dev eval/textspot3_dev_20261022.json --work $H/shared/private/owner-drive-20261013/clannad-compare \
    --out $P3/protected/kd_t64n8c16_seed3_detref/clannad --cbin $T3/eval/cbuild
```
