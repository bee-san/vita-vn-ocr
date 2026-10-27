# textspot6: a text spotter trained on the hooked pages (issue #128)

Finding and cutting the text is the largest remaining error source in Rubi's page path. On the development sample,
88% of VOCR5-N's whole-dialogue errors in the console view come from how the lines are found and cut (#119). Every
learned finder so far trained on synthetic frames and on MeikiOCR's boxes: #97 (`docs/textdet.md`) and the
textspot rounds (`docs/textspot.md`, `docs/textspot2.md`, `docs/textspot3.md`). The owner's hooked corpus (#125) now
gives real supervision: 55,493 native pages (Vita 960×544, PSP 480×272) of 49 titles, with the game's own text, the
renderer's line boxes and 99,071 machine-verified lines.

textspot6 trains a spotter on those pages. For Rubi it outputs, from a whole frame:
- the line boxes in reading order, with their orientation;
- a class per box: dialogue, speaker name, choice, UI text or scene text;
- the dialogue region (the picked block of dialogue boxes);
- a presence score (the best dialogue box's confidence; 0 when the frame has no dialogue).

It is scored the way Rubi reads, through vocr6-finder's page-path benchmark (#129, `eval/pagepath.py`), against
GRAD + refine (Rubi's finder). The headline is the development games, which are not in the corpus, because users
mostly play games that are not in it.

This document starts as the pre-registered protocol (`eval/textspot6_protocol.json`). It is frozen before any
textspot6 training run and any score. Results are added below "Results" as they are measured. Nothing above
"Results" changes except through a dated amendment, listed in the protocol's `amendments`.

## Protocol

### The spotter (VTSP version 4)

`synth/textspot6_twin.py` is the specification and the bit-exact twin of the C runtime (`runtime/vocr_textspot6.c`).
- **The network** is one of textspot3's two families, with one change: the head has 16 + 4 T channels instead of
  16.
  - Family T: the d2s trunk (a VTDT network: stem, down, down8, separable blocks at stride 8, at most 64 channels).
  - Family M: the multi-scale network (a VTMS section: stride-16 to 64 stages and an FPN neck back to stride 8).
- **The head.** Channels 0..15 are version 2's d2s head (K, O, A1, A2 per sub-cell, read as stride-4 maps).
  Channels 16..16 + 4T − 1 are T tag maps per sub-cell: class t + 1's logit against class 0's.
  - T = 4, so the classes are 0 dialogue (and narration), 1 name, 2 choice, 3 UI and 4 scene: textspot's classes.
- **The decode** is textspot's, with three additions:
  - Each box gets the class with the largest summed logit over its kernel cells (class 0 scores 0; ties go to the
    lower class), and a tag confidence.
  - Blocks join only boxes of one class.
  - The pick is the largest dialogue block (textspot's area rule), or −1 when the frame has none. The presence
    score is the best dialogue box's confidence.
- **Versions 1-3** decode as before: every box has class 0, and the results are textspot3's bit for bit.
- Exporter: `train/textspot6_model.py`.

### Data

**Hooked pages** (`synth/textspot6_pages.py`; vocr6-ingest's page pool and splits, `splits_v1.json`): whole native
frames in #97's shard format. The targets per page:
- **Positives:**
  - every verified line of kind `line` (class dialogue) or `name` (class name), with its box (`box_main` when it has
    furigana) refined to ink (`teacher/geometry.refine`, on the transpose for a vertical line; the quad box when
    the refinement leaves it);
  - MeikiOCR detector boxes (score ≥ 0.5) that meet no renderer box: text the hook does not cover (UI panels,
    dates, captions, pre-rendered plates). Their class set is "not dialogue" (name, choice, UI or scene), and their
    weight is textspot2's score rule.
- **Ignore areas:** ruby readings, renderer boxes that no verified line holds (unverified but hooked text),
  dropped lines (`geometry`, `conflict`), and meiki boxes below 0.5. They are neither positives nor negatives.
- **Pages:** the ingest's 22 untrusted pages are left out.
- **Splits:** train pages only for screening. `hval` (4% of each train title, in blocks) and its guard bands are
  held out. Test titles (Dance with Devils, PCSG00807; CLANNAD when added) are never read in training or selection.
  Final models retrain on train + guard + hval.

**Synthetic frames:** textspot's 80,000 whole frames (`data/manifests/textspot_shards_v1.json`): Vita, PSP,
Adrenaline 2×, PC and text-free frames, with exact ink boxes and classes.

**Real half:** textspot2's 52,785 meiki-labelled VNDB screenshots (`data/manifests/textspot2_shards_v1.json`), with
textspot2's skip lists. These are games outside the development sample, so they add the variety of games the hooked
corpus does not have. Their class is unknown, so they have no tag target.

**The mix** (per window): hooked pages 0.5, synthetic 0.25, real half 0.25 (`base`). A hooked title is drawn with
p ∝ pages^0.5, then a page uniformly. The `nohook` arm uses synthetic 0.5 and real half 0.5: it measures what the
hooked pages buy.

**Augmentation:**
- Adrenaline: a PSP page is scaled 2× bilinear with p 0.5, as Rubi gets PSP games.
- Zoom: 0.8-1.25 at the network's scale.
- Photometric: contrast and brightness, gamma, polarity, blur, noise and JPEG.
- A blend with a text-free synthetic frame (translucent boxes over busy scenes).

### Model, recipe, seeds

- **Trainer:** `train/textspot6_train.py`.
- **Loss:** textspot2's (K BCE with the line weights, dice, O BCE, A smooth L1), plus the tag loss: −log of the
  softmax mass of each kernel cell's class set.
- **Recipe `base`** is textspot3's base schedule: 30,000 steps of 16 windows of 256×512 at the network's scale;
  AdamW 4e-3, 500 warm-up steps, cosine to 1e-4; EMA 0.999; QAT from step 24,000 with BatchNorm re-estimation and
  #97's calibration; the last step is the run's model.
- **Recipe `kd`** (stage 2) adds textspot2's distillation loss on K/O/A from a teacher run and the KL divergence of
  the tag softmax.
- **Seeds:** at least 2 per arm (1 and 2; 1-3 since amendment 4), every seed reported. An arm's representative is
  its median seed by unseen-dev page-path dialogue CER (with 2 seeds, the worse one), never the best.

**Stage 1 arms** (`base` unless named):

| arm | shape | what it tests |
|---|---|---|
| `base_t64n8c16` | family T, 8 blocks of 64 channels (textspot3's pick's shape), half scale | the reference shape with the new data |
| `base_qS4` | family M, about 118 KB | the size sweep |
| `base_qL5` | family M, about 448 KB | the size sweep's top (under the 500 KB ceiling) |
| `base_<T full-res>` | family T at full resolution (half_above 0), registered by an amendment with its A9 row | input scale |
| `nohook_t64n8c16` | as `base_t64n8c16` without hooked pages | what the hooked pages buy |

**Stage 2** (a fixed rule read from stage 1's median seeds): `kd_t64n8c16`, distilled from the stage-1 M arm with
the lowest unseen-dev page-path dialogue CER (its median seed's final model), seeds 1 and 2 (1-3 and every decode
since amendments 4 and 5).

### Metrics

- **Primary: page-path dialogue CER** through `eval/pagepath.py` (#129), pinned by an amendment before any score.
  - **The read** (Rubi's composition): the boxes → crops → VOCR5-N in W8I with the top 8 → the counter re-decode →
    beam 8 + the 1 MB LM → the cascade to VOCR5-L below a greedy confidence of 0.7 → Rubi's text rules.
  - **The units:**
    - unseen dev: #119's 1,161 whole-dialogue units (387 regions × base / +4 / +8, 93 VNs of teacher_val and
      devnative, other games than the corpus), in the console view (PSP at Adrenaline's 2×);
    - hval: the page pool's dialogue regions of the hval pages (base / +4 / +8).
  - **Spotter boxes** (`eval/textspot6.py spot` then `pagepath-boxes`): the spotter runs on the whole frame, and
    each unit reads the horizontal boxes whose centre lies in its region. The decodes are `det` (the boxes as
    decoded) and `det+refine` (the horizontal boxes through `VOCR_REFINE_DEFAULT`, as textspot's det+refine).
  - **Baseline:** GRAD + refine, Rubi's finder as `eval/pagepath.py --finder rubi` runs it on the unit's region
    (VOCR_LINES_TEXTBOX_RECOVERY, the short-line probe, refine on PSP).
  - **Intervals:** paired, with 10,000 cluster-bootstrap resamples (seed 0): VN clusters on dev, title clusters on
    hval.
- **Report only:**
  - **Whole frames** (`eval/textspot6.py tsdev`, textspot's 188 development frames of 81 VNs): F1, all-text and
    dialogue CER, recall per class, and the pick. Plus the tagged dialogue CER: what Rubi shows when it reads only
    the dialogue-tagged boxes, with every character read from another text counted as an error. Plus presence on
    the 25 text-free frames.
  - **hval boxes:** P / R / F1 against the verified lines, recall per class, and the tag accuracy of matched boxes,
    per view (Vita, PSP native, PSP 2×) and per title.
  - textspot3's pick through the same paths.

### The rule (pre-registered)

- **Eligible:**
  - VTSP blob ≤ 512,000 B;
  - est. ≤ 300 ms per 960×544 frame on one A9 core at 333 MHz (textspot3's budget and A9 cycle model on
    `runtime/vocr_textspot6.c`; plus 1.4 ms per refined horizontal box at the arm's median for det+refine);
  - arena ≤ 262,144 B;
  - the C runtime equals the twin on every development frame.
- **Beats GRAD + refine** when both hold, for a candidate (an arm and a decode, its median seed):
  - unseen dev: its page-path dialogue CER minus GRAD + refine's has a 95% interval entirely below 0;
  - hval: the same difference is not resolved above 0.
- **Pick:**
  - the cheapest qualifier by VTSP bytes, then est. ms at 333 MHz, then arena, then det before det+refine;
  - a costlier qualifier replaces it when, against it, its unseen-dev page-path CER is resolved below 0 and hval is
    not resolved above 0, and this repeats.
- **No qualifier:** the verdict is negative. The eligible candidate with the lowest unseen-dev page-path CER is
  released as a research pre-release, and no Rubi PR is opened.
- **With a pick:** the pick is retrained on train + guard + hval with the same recipe and seed. That final model is
  released (pre-release, SHA256SUMS), read once on the test sets, and wired into a Rubi DRAFT PR (never merged by
  the fleet). The screening model's numbers stay the development record.

### Test sets, read once after the selection is committed

- The test title(s) (Dance with Devils) through the page path (its page-pool units).
- CLANNAD whole frames (the frozen long-run references, as textspot3 read them).
- vita-test + psp-test as whole frames (`eval/manifests/textspot-test.jsonl`).

The textspot3 pick and GRAD + refine are read the same way for the paired comparison.

### Budget and estimates

- Weights ≤ 500 KB (owner, 2026-10-26).
- Time within textspot3's budget unless a bigger model clearly pays; the trade is reported.
- 2 heavy slots, about 30-40 slot-hours of training.
- Console times are A9 cycle-model estimates (textspot3's method), at 333 and 444 MHz. Nothing runs on a console:
  hardware work is paused.

### Disclosures registered in advance

- **Hval is seen titles:** the hval pages are blocks of training titles. The unseen-game dev units are the
  selection's headline for that reason.
- **One dev game is seen:** the real half's meiki labels and textspot's synthetic frames exclude the dev sample's
  VNs, but the hooked corpus holds one dev_sample VN as a training title (PCSG00576 = v12188). The 99 development
  VNs (#119's units and textspot's frames) were checked against the 43 hooked titles' VNDB ids and the releases of
  the 6 titles without one: only v12188 matches, and v18432 is the test title Dance with Devils, never trained on.
  The headline unseen-dev score leaves v12188's units out; the full set is reported beside it.
- **The dev regions were drawn by people,** so the region-assisted page path gives the spotter the same region as
  GRAD + refine. The whole-frame tagged path is reported beside it.

### Amendments

Each amendment is dated in `eval/textspot6_protocol.json` and precedes the runs and scores it names.

1. **Data, shapes, sweep** (2026-10-26T09:54Z, before every run and score).
   - The hooked page shards are pinned: `data/manifests/textspot6_pages_v1.json`, the post-audit page pool v1.
     - train: 50,243 pages; positives dialogue 81,542, name 7,809, "not dialogue" 34,336; 146,968 ignore areas.
     - hval 2,191 pages, guard 2,337.
     - 32 untrusted pages are left out; test pages are never read.
     - MeikiOCR ran on all 54,803 non-test pages.
   - Stage 1's shapes are registered: `t64n8c16`, `qS4`, `qL5`.
   - The sweep is `eval/reproducibility/textspot6-20261026/sweep.py` with `stage1_plan.jsonl`: 8 runs, at most 2 at
     once.
   - Probes, in ms per step before QAT: 291.9 / 212.6 / 256.8 (T / qS4 / qL5).
2. **Crop decodes** (before any score).
   - Each arm also runs as `crop`: the spotter on each unit's region alone, cut as `eval/pagepath.py` cuts it, with
     every horizontal box read. `crop+refine` adds `VOCR_REFINE_DEFAULT`.
   - This is how Rubi would use the spotter as a drop-in for GRAD + refine inside the player's region, without a
     whole-frame capture.
   - The candidates are every arm × {crop, crop+refine, det, det+refine}.
   - The cost order puts the decode after bytes, ms and arena: crop first, det+refine last.
3. **The crop decodes' scale** (before any score). A region is read at the scale the blob's rule gives the frame it
   came from (`textspot6_twin.spot(..., scale=f)`; the runtime's `VOCR_TEXTSPOT6_SCALE1` / `_SCALE2` flags).
   - Every console frame is 960×544, so a half-scale blob reads every console region halved, as it reads the whole
     frame.
   - Read at its own size, a region narrower than 480 px would show the network text twice the size it trained on.
4. **Owner directive, 3 seeds, 3 slots** (2026-10-26, before any score).
   - The owner's 10:17Z directive replaces "a Rubi DRAFT PR, never merged". With an adopted pick, the Rubi_for_vita
     PR is a normal PR, merged after GitHub CI (package e2e included), the memory ledger and an independent audit's
     MERGE_OK. It comes with a Rubi test pre-release and a test issue for the console checks.
   - The new console path keeps a setting to turn it off. Its default is the measured-best choice.
   - Every arm is trained with seeds 1-3 (vocr6-common: at least 3 seeds for an adoption claim). The representative
     is the median seed.
   - The label has 3 slots. `sweep.py --cap-file` holds the number of training runs, and the third slot scores.
   - The test reads are `eval/textspot6_test.py` (`title`, `clannad`, `tests`), once each.
5. **Stage 2's teacher** (before any arm's score). It is the M arm whose representative has the lower unseen-dev
   page-path CER in any of the four decodes.
6. **The page-path benchmark is pinned** (before any arm's score): vocr6-finder's `eval/pagepath.py` and
   `pagepath_find.c` at PR #131's head `fadc3c6`.
   - The reference scores posted on #128 used the finder's working copy, which differs only in the module
     docstring.
   - They are scored again with the pinned files before the selection.
7. **The Rubi merge gate** (owner, 11:33Z; process only). A Rubi_for_vita PR merges as soon as the local gate
   passes: host ctest, both Vita builds with their import, metadata and memory-ledger gates, and `e2e_package.py`
   when packaging changed. Its audit runs in parallel, and its findings become a follow-up PR. The label's 3 slots
   include its helper and auditors.
8. **Two slots and one scheduler** (before any arm's score). The label has 2 slots from 12:37Z.
   - `sched.py` runs every heavy job: pending scoring first, then the training plan. It leaves one slot to the
     runtime helper until the helper's runtime is ready.
   - `nohook_t64n8c16` becomes a report-only control with 2 seeds. The candidates are `base_t64n8c16`, `base_qS4`,
     `base_qL5` and stage 2's `kd_t64n8c16`, 3 seeds each.
9. **The benchmark at #135, Rubi's next finder, Rubi's default** (13:50Z).
   - **Disclosed:** this comes after a partial look at one score. The first scoring of `base_t64n8c16_seed1` was
     stopped for this change after its crop runs, and their lines were seen.
   - Nothing in the rule, the arms or the candidates changes.
   - The benchmark moves to vocr6-finder's PR #135 (`2d3971b`). It must reproduce #131's GRAD + refine and
     textspot3-pick scores unit for unit. If it does not, the rule reads #131's scores.
   - A report-only row is added: Rubi's next finder, `--finder rubi,fix=5` (#129's COLGAP + JUNK).
   - Rubi's default: the adopted pick becomes the default finder only if it also beats `rubi,fix=5` by the rule's own
     test. Otherwise the spotter ships behind its setting, off by default.
10. **The runtime and A9 rows of record** (after the seed-1 scores).
    - The runtime is `runtime/vocr_textspot6.c` (merged at `956494e`). The A9 rows are
      `eval/textspot6_a9_20261026.json`: t64n8c16 est. 264 ms at 333 MHz, qS4 199 ms, qL5 237 ms. All arms are
      eligible.
    - The C == twin check runs in scoring (`eval/textspot6_parity.py`).
11. **Scheduling** (no rule change): the remaining stage-1 runs, the M arms' seeds first (`stage1d_plan.jsonl`), so
    that stage 2's teacher is known sooner.
12. **Rubi's next finder is `rubi,fix=1`** (#129 amendment 3: COLGAP alone). JUNK deletes text lines in tight
    regions, so `rubi,fix=1` replaces `rubi,fix=5` in amendment 9's row and test, and fix=5 stays as a report row.
    The pick and the baselines are also read on vocr6-finder's tight-region units (0, 2 and 1 px around the ink),
    report only.
13. **How the test sets are read** (`eval/textspot6_test.py`, before any test read). The final weights are the
    pick's recipe and seed retrained on train + guard + hval. Every check that needs no protected pixel runs first,
    then an exclusive attempt record refuses a second read.
    - Test title: `pagepath.py units --set test`, read by GRAD + refine, `rubi,fix=1` (report), textspot3's pick
      and the final weights in the pick's decode, as hval is read; page-cluster intervals (one title).
    - CLANNAD, textspot3's way: whole frames, the final weights in det and det+refine (the primary is the pick's
      decode on a whole frame), against textspot3's pick and textspot's arms from their stored boxes, which must
      reproduce their committed records first. Report only: each page's ROI through the page path, as Rubi reads
      a region.
    - vita-test + psp-test: textspot3's way, against textspot3's stored units and textspot's stored arms; plus the
      tagged dialogue CER and presence on text-free frames.
    - The script runs first on non-protected stand-ins (`--dry-run`) with a stage-1 run as the weights.
14. **Scheduling** (no rule change): stage 2's three distilled seeds run before the nohook control's two. The
    teacher by amendment 5's rule is `base_qS4` seed 2 (the best representative decode: crop, 1.3868% unseen-dev
    CER, against `base_qL5`'s 1.4418%), so stage 2 distils from `base_qS4_seed2/final.pt`
    (`eval/textspot6_teacher_20261026.json`).
15. **The test sets with a negative verdict** (before the selection; no test set read). They are still read once,
    report only, with the research pre-release's blob (the best eligible candidate as trained, no retrain) through
    the same commands (`--research`), so the owner has its test numbers. No decision changes: no Rubi default, no
    Rubi PR. Registered after the M arms' scores, whose representatives are not resolved below GRAD + refine on
    unseen dev.

## Results

Every number is measured on the host unless it is labelled an estimate. The records are:
- `eval/textspot6_dev_20261027.json`: the selection, run once after every run was scored;
- `eval/textspot6_teacher_20261026.json`: stage 2's teacher;
- `eval/textspot6_tables_20261027.json`: the report rows, written by `eval/textspot6_tables.py`;
- `eval/textspot6_test_20261027.json`: the test reads, made once.

Page-path CER is dialogue CER through `eval/pagepath.py` (#135's snapshot, Rubi's textnorm at 213fd60). Intervals are
95% paired cluster bootstraps: VNs on dev, titles on hval, pages on the test title and CLANNAD.

### Verdict: negative

- **No candidate qualifies.** All 48 seed rows (4 arms × 3 seeds × 4 decodes) are eligible:
  - VTSP ≤ 448,304 B;
  - est. ≤ 269.8 ms at 333 MHz;
  - arena ≤ 256,247 B;
  - C == twin on every development frame and crop.

  Every one of the 16 representatives (the median seed of an arm and a decode) reads unseen dev at 1.39-2.07%,
  against GRAD + refine's 3.08%, and every one is resolved better on hval. But no unseen-dev VN-cluster interval
  lies entirely below 0: the upper ends run from +0.08 to +0.85.
- **Research pre-release** v0.3.8-textspot6.1: the rule's named candidate, `base_qS4` crop seed 2, the eligible
  representative with the lowest unseen-dev CER (118,096 B), as trained on train. There is no final retrain.
- **No Rubi PR.** Rubi's finder stays GRAD + refine, which is `rubi,fix=1` since Rubi_for_vita#330. The helper's
  integration branch (`feat/vocr6-spotter-finder` 7089934, `ocr_finder = grad | spotter`, default grad) passed
  Rubi's local fast gate and stays unmerged as a reference.
- **vocr6-finder's own rule** (#129 amendment 2: the spotter replaces `rubi,fix=1` only if hval is resolved below
  it and dev and the tight sets are not worse by more than +0.05) also keeps `rubi,fix=1`. Run on this blob as a
  report row, it fails all three conditions (#129, comment 6433642955).

### Page path, every representative

The seed column gives the representative's seed, then the unseen-dev CER of seeds 1 / 2 / 3. Dev is the unseen dev set
(92 VNs, 1,143 units; v12188, a hooked training title, is left out). Hval is 46 titles and 6,327 units. Both are in
%.

| arm | decode | seed (seeds) | dev | dev Vita / PSP | hval | hval Vita / PSP | dev − GRAD+refine | hval − GRAD+refine |
|---|---|---|---:|---|---:|---|---|---|
| GRAD + refine (`rubi`) | | | 3.08 | 2.95 / 3.21 | 3.88 | 4.53 / 2.70 | | |
| `rubi,fix=1` (Rubi's finder now) | | | 2.38 | 1.97 / 2.74 | 3.29 | 3.90 / 2.20 | −0.71 [−1.29, −0.25] | −0.58 [−0.93, −0.31] |
| `base_t64n8c16` (52 KB) | crop | 3 (2.01 / 1.80 / 1.87) | 1.87 | 1.62 / 2.08 | 1.55 | 1.59 / 1.46 | −1.22 [−3.86, +0.67] | −2.33 [−6.01, −0.29] |
| | crop+refine | 3 (2.17 / 1.94 / 2.07) | 2.07 | 1.52 / 2.57 | 1.64 | 1.72 / 1.49 | −1.01 [−3.67, +0.85] | −2.24 [−5.92, −0.20] |
| | det | 1 (1.41 / 1.47 / 1.38) | 1.41 | 1.47 / 1.35 | 1.30 | 1.37 / 1.18 | −1.68 [−4.31, +0.12] | −2.57 [−6.23, −0.54] |
| | det+refine | 3 (1.59 / 1.74 / 1.65) | 1.65 | 1.70 / 1.60 | 1.69 | 1.40 / 2.20 | −1.44 [−4.04, +0.31] | −2.19 [−5.86, −0.32] |
| `kd_t64n8c16` (52 KB) | crop | 2 (1.54 / 1.65 / 1.91) | 1.65 | 1.45 / 1.82 | 1.51 | 1.61 / 1.32 | −1.44 [−4.06, +0.36] | −2.37 [−6.10, −0.25] |
| | crop+refine | 2 (1.71 / 1.74 / 2.07) | 1.74 | 1.43 / 2.02 | 1.58 | 1.71 / 1.33 | −1.34 [−3.97, +0.47] | −2.30 [−5.95, −0.24] |
| | det | 2 (1.27 / 1.52 / 1.65) | 1.52 | 1.30 / 1.71 | 1.33 | 1.25 / 1.48 | −1.57 [−4.17, +0.28] | −2.54 [−6.25, −0.55] |
| | det+refine | 2 (1.47 / 1.83 / 1.96) | 1.83 | 1.75 / 1.91 | 1.43 | 1.39 / 1.51 | −1.25 [−3.80, +0.47] | −2.44 [−6.15, −0.45] |
| `base_qS4` (118 KB) | **crop** | **2** (1.32 / 1.39 / 1.56) | **1.39** | 0.96 / 1.77 | 1.61 | 1.84 / 1.21 | −1.70 [−4.33, +0.08] | −2.26 [−5.77, −0.21] |
| | crop+refine | 2 (1.57 / 1.58 / 1.84) | 1.58 | 1.07 / 2.05 | 1.51 | 1.71 / 1.16 | −1.50 [−4.16, +0.25] | −2.36 [−6.06, −0.26] |
| | det | 2 (1.34 / 1.47 / 1.58) | 1.47 | 1.49 / 1.46 | 1.57 | 1.27 / 2.12 | −1.61 [−4.22, +0.21] | −2.30 [−5.97, −0.41] |
| | det+refine | 1 (1.73 / 1.68 / 1.85) | 1.73 | 1.83 / 1.64 | 1.51 | 1.41 / 1.68 | −1.35 [−3.94, +0.40] | −2.37 [−6.09, −0.42] |
| `base_qL5` (448 KB) | crop | 1 (1.44 / 1.29 / 1.87) | 1.44 | 1.22 / 1.64 | 1.58 | 1.72 / 1.33 | −1.64 [−4.28, +0.12] | −2.30 [−5.84, −0.30] |
| | crop+refine | 1 (1.60 / 1.41 / 1.88) | 1.60 | 1.25 / 1.91 | 1.60 | 1.76 / 1.32 | −1.49 [−4.11, +0.28] | −2.27 [−5.98, −0.21] |
| | det | 1 (1.50 / 1.25 / 1.60) | 1.50 | 1.42 / 1.58 | 1.31 | 1.29 / 1.33 | −1.58 [−4.18, +0.23] | −2.57 [−6.27, −0.53] |
| | det+refine | 3 (1.77 / 1.47 / 1.60) | 1.60 | 1.45 / 1.74 | 1.47 | 1.43 / 1.53 | −1.48 [−4.06, +0.23] | −2.41 [−6.10, −0.49] |
| report: `rubi,fix=5` | | | 2.36 | 1.93 / 2.74 | 1.90 | 1.74 / 2.18 | −0.72 [−1.30, −0.27] | −1.98 [−5.19, −0.38] |
| report: textspot3's pick | crop | | 1.33 | 1.01 / 1.62 | 1.53 | 1.64 / 1.34 | −1.75 [−4.35, −0.01] | −2.34 [−5.77, −0.37] |
| report: textspot3's pick | det | | 1.32 | 1.26 / 1.38 | 1.27 | 1.38 / 1.07 | −1.76 [−4.40, +0.04] | −2.60 [−6.17, −0.57] |

- **Against `rubi,fix=1`** (amendment 12's test for Rubi's default, moot without a pick):
  - unseen dev: every representative is lower (−0.31 to −0.99 points), but no difference is resolved (upper ends
    +0.65 to +1.42);
  - hval: the det decodes of all four arms, and `base_qL5` det+refine, are resolved lower (upper ends −0.004 to
    −0.07).
- **textspot3's released pick** (51,008 B; trained on synthetic frames and meiki labels, no hooked page) reads as
  well as any textspot6 arm. Its crop decode is the only row of the study whose unseen-dev interval lies below 0.
  It is one released model and was not a textspot6 candidate (no median seed); its selection read development
  frames of some of these VNs.
- The per-title (hval) and per-VN (dev) rows of the research release, beside GRAD + refine's and `rubi,fix=1`'s,
  are in `eval/textspot6_tables_20261027.json` (`per_title`).

### Why the dev intervals stay open

The gain over GRAD + refine is a few VNs' worth (`spread` in the tables record):

| representative | dev − GRAD+refine | VNs better / worse / level | largest gains (errors GRAD+refine → spotter) | largest losses | without v27449 (report only) |
|---|---|---|---|---|---|
| `base_qS4` crop s2 | −1.70 [−4.33, +0.08] | 29 / 29 / 34 | v27449 311 → 20, v17972 68 → 3, v21321 42 → 3 | v1971 28 → 57, v20170 3 → 23, v9178 1 → 19 | −0.57 [−1.52, +0.27] |
| `base_t64n8c16` det s1 | −1.68 [−4.31, +0.12] | 27 / 16 / 49 | v27449 311 → 24, v17972 68 → 9, v7290 42 → 0 | v20524 0 → 36, v1971 28 → 56, v8410 1 → 24 | −0.57 [−1.53, +0.37] |
| `base_qL5` crop s1 | −1.64 [−4.28, +0.12] | 26 / 36 / 30 | v27449 311 → 24, v17972 68 → 1, v7290 42 → 0 | v1971 28 → 53, v3496 0 → 22, v454 0 → 21 | −0.53 [−1.48, +0.32] |
| `kd_t64n8c16` det s2 | −1.57 [−4.17, +0.28] | 29 / 18 / 45 | v27449 311 → 33, v17972 68 → 4, v7290 42 → 0 | v20524 0 → 45, v9178 1 → 45, v1971 28 → 54 | −0.49 [−1.51, +0.55] |

- **v27449** holds 311 of GRAD + refine's 785 unseen-dev errors, 40% of them: 311 errors on its 837 characters.
  GRAD + refine merges its 45 lines into 6 merged boxes, and `rubi,fix=1` reads it the same way. Every spotter
  boxes the 45 lines apart and reads it with 20-33 errors.
  - With v27449 the research release's interval is 4.4 points wide; without it, 1.8.
  - Without it, the representatives' differences run from +0.13 to −0.57 points, and none is resolved.
- **The losses** are concentrated:
  - side-by-side choice buttons (v1971, every arm);
  - a few VNs that GRAD + refine reads perfectly but a spotter splits, cuts or merges (v20524 and v8410 in the det
    decodes; v20170, v3496 and v454 in the crop decodes).
- The crop decodes make more VNs worse (29-45 of 92) than the det decodes (13-22).

### What each choice bought (report only)

- **Size:** the 52 KB T trunk, the 118 KB `qS4` and the 448 KB `qL5` read unseen dev about equally in det
  (1.41 / 1.47 / 1.50). Only the crop decode gains from the multi-scale networks (1.87 → 1.39 / 1.44), probably
  from their wider context, since a region alone holds less of the frame.
- **Distillation** (stage 2, teacher `base_qS4` seed 2 by amendment 5's rule): `kd_t64n8c16` against
  `base_t64n8c16` reads crop 1.65 against 1.87 (−0.22 [−0.56, +0.06]) and det 1.52 against 1.41
  (+0.11 [−0.26, +0.56]), representatives paired. Neither difference is resolved.
- **The hooked pages** (the `nohook_t64n8c16` control, seeds 1-2):
  - unseen dev det 1.48 / 1.35 and crop 2.14 / 1.66, against `base_t64n8c16`'s 1.41 / 1.47 / 1.38 and
    2.01 / 1.80 / 1.87: no gain on unseen games;
  - on the seen titles they help: hval box F1 (Vita view) 0.89-0.90 against 0.75-0.77, and dialogue tag accuracy
    0.997-0.998 against 0.96;
  - textspot's development frames: tagged dialogue CER 26-29% against 33-34%.
- **Refinement** (`VOCR_REFINE_DEFAULT` on the spotter's boxes) is worse for every representative: +0.10 to +0.32
  points on unseen dev against the same arm and decode without it.

### Regions drawn tight to the text (amendment 12, report only)

vocr6-finder's tight units: `dev_tight` has 1 px around the ink (381 units without v12188); `hval_tight` (0 px) and
`hval_tight2` (2 px) have 2,109 units each. All values are page-path CER in %. The two research-release rows are the
same weights.

| set | GRAD + refine | `rubi,fix=1` | research release, crop | research release, det (frame boxes clipped to the region) |
|---|---:|---:|---:|---:|
| dev_tight | 3.16 | 2.16 | 9.15 (vs fix=1 +6.99 [+3.54, +10.60]) | 1.78 (−0.38 [−2.93, +1.26]) |
| hval_tight | 2.57 | 1.93 | 3.82 (+1.89 [+0.65, +3.00]) | 1.92 (−0.01 [−0.17, +0.14]) |
| hval_tight2 | 2.24 | 1.62 | 2.56 (+0.94 [−0.12, +1.80]) | 1.56 (−0.06 [−0.23, +0.11]) |

- The **crop decode fails** when the region leaves no margin: on dev_tight's Vita units it finds only 58% of the
  lines whole.
- The **det decode** reads the whole frame and keeps the boxes inside the region, so it does not depend on how the
  region was drawn.
- A next round that puts a spotter inside the player's region should register det (or a cascade with
  `rubi,fix=1`) and train with tight crops. vocr6-finder records that this would be post hoc on these units.

### Whole frames and boxes (report only)

- **textspot's 188 development frames** (`tsdev`; every textspot6 run is in the tables record):
  - box F1 is 0.56-0.59 for every textspot6 run, against textspot3's 0.654;
  - UI recall is 0.21-0.25, against textspot3's 0.36;
  - the tagged dialogue CER (what Rubi shows if it reads only dialogue-tagged boxes) is 21-30%. textspot3, which
    has no tags, scores 61.8%;
  - no textspot6 arm puts a dialogue box on any of the 25 text-free frames.
- **hval boxes** (Vita view): F1 0.85-0.92 (textspot3 0.81). Dialogue tag accuracy is ≥ 0.996, name tag accuracy
  0.87-1.00.

### Test sets, read once with the research release (amendment 15, report only)

`eval/textspot6_test.py --research`, after the selection commit, against stored arms that reproduce their records
first.

| set | research release | textspot3's pick | GRAD + refine | `rubi,fix=1` |
|---|---:|---:|---:|---:|
| Dance with Devils (PCSG00807, 2,055 units on 685 pages; page path, crop) | 1.39 | 1.48 | 1.24 | **0.98** |
| CLANNAD, each page's region through the page path (2,044 pages; crop) | 0.88 | 0.85 | **0.82** | 0.95 |
| CLANNAD whole frames, dialogue (2,052 lines; det; textspot3's read) | **0.94** [0.79, 1.11] | 0.99 | | |
| CLANNAD whole frames, all lines with names (3,048) | **1.10** | 1.73 | | |
| vita-test + psp-test whole frames (176 frames; det): dialogue CER | **2.02** | 2.29 | | |
| vita-test + psp-test: box F1 / all-text CER | 0.771 / 21.82 | 0.776 / 20.57 | | |

All values are CER in % unless marked F1.

- **Dance with Devils:**
  - against GRAD + refine +0.15 [−0.17, +0.45] (not resolved);
  - against `rubi,fix=1` +0.41 [+0.15, +0.68] (resolved worse);
  - against textspot3 −0.09 [−0.28, +0.11].
- **CLANNAD whole frames:**
  - dialogue −0.04 [−0.15, +0.06] against textspot3;
  - all lines −0.62 [−0.76, −0.49]: the gain is in the name lines, since dialogue is level;
  - 587 boxes on the 133 pages without text, against textspot3's 504.
- **CLANNAD regions:** every finder is level (research release − GRAD + refine +0.06 [−0.19, +0.26]).
- **vita-test + psp-test:**
  - dialogue −0.28 [−0.94, +0.28] against textspot3, and resolved better than textspot's spotter
    (−0.98 [−1.86, −0.17]);
  - F1 −0.005 [−0.027, +0.015];
  - the tagged dialogue CER is 11.1%;
  - 1 of 19 text-free frames gets a dialogue box (textspot3: 2);
  - without v21789's 9 frames (see below): 2.09% against 2.46%.
- The harness re-spotted textspot3's pick:
  - CLANNAD: the same boxes on all 2,177 pages;
  - test sets: every score equal to its stored units' (only the C-check fields differ: this harness runs no C
    driver on test frames).

  Textspot's stored arms reproduce their committed records. The readers are textspot's library, rebuilt from
  d214a05 (hash-checked); on 600 development crops it reads the same as today's runtime.

### Disclosures found after the protocol

- **Release partners of training titles** (vocr6-ingest's overlap audit):
  - PCSG00007 (v5) and ULJM05383 (v34) share VNDB release r3748 with CLANNAD (v4);
  - PCSG00866 (v18160) shares four releases with v21789, a vita-test VN (9 of the 176 scored frames).

  All three titles were trained on, as the owner's 2026-10-25 policy allows (it keeps out only CLANNAD's own
  releases). The test numbers without v21789 are given above.
- **Registered after scores** (rule unchanged): amendment 9 (a partial look at one score), amendment 12 (`rubi,fix=1`
  as Rubi's finder) and amendment 15 (the research release's test reads).
- **Not run:** the oracle baseline (each reference line's own box), a report row, dropped for slot time.

### Cost (estimates; nothing has run on a console)

These come from the A9 cycle model on the Vita compiler's -O2 code, with the dark pass, per 960×544 frame
(`eval/textspot6_a9_20261026.json`). A crop decode costs at most the same for any region of the frame.

| shape | VTSP bytes | est. ms @333 / @444 MHz | arena |
|---|---:|---:|---:|
| `t64n8c16` (`base`, `kd`) | 52,176 | 264.2 / 198.2 | 244,607 |
| `qS4` (research release) | 118,096 | 199.1 / 149.4 | 256,247 |
| `qL5` | 448,304 | 237.0 / 177.7 | 256,247 |

### Reproduction

```sh
S=eval/reproducibility/textspot6-20261026
# training: sched.py runs sweep.py's commands from a live plan (stage1d_plan.jsonl, stage2_plan.jsonl)
HEAVY_SLOTS=12 .venv/bin/python $S/sched.py --runs $RUNS --plan $S/stage2_plan.jsonl --cap-file CAP --pagepath $PP
# scoring of one run (dev + hval in four decodes, tsdev, hval boxes, C == twin)
PAGEPATH=$PP RUBI=$TEXTNORM $S/score_run.sh $RUNS/base_qS4_seed2 base_qS4_seed2
PAGEPATH=$PP RUBI=$TEXTNORM $S/score_run.sh --baseline grad_refine_fix1 rubi,fix=1
PAGEPATH=$PP RUBI=$TEXTNORM $S/score_tight.sh $RUNS/base_qS4_seed2 base_qS4_seed2 "crop det"
# teacher, table, selection, report tables
.venv/bin/python eval/textspot6_teacher.py --eval $EVAL --runs $RUNS --out eval/textspot6_teacher_20261026.json
.venv/bin/python eval/textspot6_table.py --eval $EVAL --runs $RUNS --out TABLE.json
.venv/bin/python eval/textspot6_select.py --table TABLE.json --out eval/textspot6_dev_20261027.json
.venv/bin/python eval/textspot6_tables.py --selection eval/textspot6_dev_20261027.json \
    --per-title base_qS4_seed2:crop --json eval/textspot6_tables_20261027.json
# the test reads, once each (attempt records refuse a second run), then the public record
.venv/bin/python eval/textspot6_test.py {title,tests,clannad} --research --selection eval/textspot6_dev_20261027.json \
    --final $RUNS/base_qS4_seed2 --out $PV/vocr6-spotter/protected/research_base_qS4_seed2_crop \
    [--pagepath $PP] [--reader-src <checkout of d214a05>]
.venv/bin/python eval/textspot6_test.py record --out $PV/vocr6-spotter/protected/research_base_qS4_seed2_crop \
    --json eval/textspot6_test_20261027.json
```
