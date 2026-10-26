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

## Results

(none yet)
