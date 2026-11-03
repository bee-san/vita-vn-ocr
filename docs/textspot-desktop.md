# textspot-desktop: a fast desktop text-line detector with tight boxes (issue #173)

The owner (2026-11-01 19:38Z): "a text detection algo that has really tight bounding boxes but is ultra fast."

vocr-desktop (#169) trains a desktop line detector aimed at MeikiOCR-level accuracy (its d64 arm: 0.81 G MACs at
960×544). This work is the speed tier beside it. It reuses #169's harness, #171's data and the desktop C path, and
trains a much smaller network whose boxes are meant to be tight to the ink, so that VOCR's recognizer reads them as
well as it reads exact boxes.

**The result: a research pre-release, not adopted.** Every number is measured on the host; nothing here is a console
time.

- **It is fast:**
  - 51,226 float32 parameters and 95 M multiply-adds at 960×544.
  - In C (AVX2 + FMA) the whole detector, from an RGB frame in memory to its refined boxes, takes **2.4 ms at
    960×544 and 2.9 ms at 1920×1080 on one thread** (0.9 and 1.2 ms on four).
  - That is a third of textspot8's ONNX network time and a fortieth of MeikiOCR's detector.
  - The C detector adds 10-12 MiB to its process.
  - The C path's crops equal the ONNX path's on every dev-v2 frame.
- **Its boxes are close to the ink where it finds a line:** after the refinement, the median edge error against the
  hooked pages' exact ink boxes is 0 px on the left and right and about 1 px on the top and bottom (1.02 and 0.99 px at
  long:960). The p90 errors are 2.1-9.8 px.
- **It does not read well enough.** Its pre-registered rule asked for:
  - unseen console screenshots no worse than textspot8;
  - dense PC pages at 2% or less;
  - every edge's median error at 1 px or less and p90 at 2 px or less;
  - at most 5 junk boxes on the ornament-strip title.

  It fails all four. It passes the speed and C-parity clauses.
- **On real PC screenshots** (#171's PC test set, read once), the released model reads dialogue at **2.94%**:
  - better than textspot8 + VOCR6.2-L's 3.88% (−0.94 points [−1.93, −0.04]);
  - far from MeikiOCR's detector + VOCR6.2-L's 0.70%.

| dialogue CER, VOCR6.2-L fast, greedy | dev-v2: 84 unseen VNs, console screens | dense_dev: 100 unseen VNs, PC pages with ≥ 8 lines | PC test: 139 unseen VNs, PC screens (read once) |
|---|---:|---:|---:|
| R: MeikiOCR's detector + L | **0.77%** | **0.35%** | **0.70%** |
| textspot8 + L | 1.35% | 7.53% | 3.88% |
| **this detector + L** (the release model, long:960, refinement on) | 1.82% | 2.39% | 2.94% |
| the rule's representative (seed 2, long:960) | 3.41% | 3.34% | |
| MeikiOCR 0.3.5 itself | 2.54% (#169) | | 1.97% |
| mida + L (MeikiOCR's author's detector, report only) | 1.42% (#169) | | 1.38% |
| vocr-desktop's d64 + L (#175, its own reads, long:960 / its release at long:640) | 0.98% / 1.26% | | 1.67% / 2.75% |

**Thanks to MeikiOCR's author.** They proposed the desktop detector between textspot8 and MeikiOCR's. They pointed
out that detectors must be trained at the sizes they will read at, and they advised comparing at 960×544. They also
shared their own detector for benchmarks.

## How it works

### The network (`train/tsdesk_net.py`)

A float32 fully convolutional network that takes the grey frame (integer luma, as VOCR's recognizer reads it):

- **Stem:** a 4×4 convolution at stride 4, which cuts the frame into 4×4 patches and maps each to 16 channels.
- **Stages:** separable stages at strides 8, 16 and 32 (32, 64 and 96 channels). Each is a stride-2 separable down
  block and residual separable blocks with dilations.
- **Global context:** the mean of the stride-32 features through a 1×1 layer and ReLU, added back at every position.
  It is page-level context for 9,216 multiply-adds, the cheap stand-in for attention.
- **Pyramid:** a top-down pyramid of 1×1 laterals to 32 channels back to stride 8, and one separable smoothing block.
- **Head:** ten maps at stride 8:
  - K, a line kernel's logit;
  - O, the orientation logit;
  - A1 and A2, the distances from a cell's centre to the line's two edges across it;
  - E1 and E2, the distances from a cell's centre to the line's start and end along it;
  - four tag logits.

  The distance maps are in input px.

The family's sizes (MACs counted from the layer shapes):

| shape | parameters | MACs at 960×544 | MACs at 1920×1088 | what differs |
|---|---:|---:|---:|---|
| p8a | 41,914 | 94.9 M | 379.7 M | the base |
| **p8g (the pick's shape)** | 51,226 | 94.9 M | 379.7 M | p8a + global context |
| p8s | 22,354 | 53.5 M | 214.0 M | narrower |
| p4a | 37,434 | 88.1 M | 352.6 M | a stride-4 head with 16 pyramid channels |

For scale: textspot8 is 86.8 M MACs (it halves a Vita frame first), vocr-desktop's d64 0.81 G, MeikiOCR's author's
detector (mida) 1.04 G and MeikiOCR's detector 4.41 G.

### Tight boxes: the targets and the reference (`train/tsdesk_targets.py`)

**The renderer gives each verified hooked line exact geometry:** its glyph quads, grouped into one cell per character
of the label. A glyph cell is the glyph's whole em box, so it is not tight to the ink. The repository's usual ink
refinement (`teacher/geometry.refine`, which `synth/textspot6_pages.refined_line` applies to the line's box) takes the
smallest of three masks' boxes. **On titles whose punctuation or a coloured word is fainter than the other glyphs, it
drops end glyphs:** a leading 「 or ――, a trailing 。, 」 or ……. Its plain contrast extent also runs into dithered and
patterned boxes.

Measured on hval (geometry only, and VOCR6.2-L reading each line's box with the eval pad):

| a line's box | VOCR6.2-L CER (662 seeded hval lines) | lines that lose an end of ≥ 0.3 line heights (vs the new box) |
|---|---:|---|
| the renderer's box | 0.361% | |
| `refined_line` (the old targets and reference) | 1.073% | 6.2% of Vita lines, 12.4% of PSP lines |
| the ink inside the glyph cells (the new box) | 0.484% | |
| the glyph cells' union | 0.389% | |

**The new box** (`ink_box`) is the ink inside the union of the line's glyph cells:
- the background is the median of a one-pixel ring around the cells;
- the threshold adapts to the background's own texture;
- a pixel counts when it differs from the background and lies on a 3×3 local-contrast edge (a gradient alone never
  counts);
- specks under 3 px are dropped.

It is the exact reference of the protocol's edge error, and the training target of every verified hooked line, in
the native pages and in #171's desktop captures (the box is mapped through each capture's exact geometry, rebuilt from
its seed) (amendment 1). The along-line maps E1 and E2 learn only from sources whose boxes come from renderer or
rendering geometry: the hooked pages, their captures and the synthetic frames.

**Its limits** (measured by the independent audit of #176):
- **Neighbouring lines.** On titles whose line pitch is smaller than a glyph's em box, the cells' union reaches into
  the next line, and that line's glyph tops or bottoms count as ink.
  - On hval, 172 of 4,720 lines (3.6%, in 9 of 71 titles) have an ink box whose top or bottom lies inside a
    neighbouring line's box. Most are on four titles: PCSG00918 (68 of 107 lines), PCSG00917 (39 of 95), PCSG00916 (38
    of 69) and PCSG00510 (17 of 29). On the five most affected titles that edge lies 5 px beyond `refined_line`'s
    (median; at most 6).
  - In the training targets, 5,323 of 143,595 hooked positives (3.7%, 17 titles) overlap a neighbouring positive.
  - It touches R3's top and bottom reference, the exact-box reads (the tightness cost, the pad study's hval half) and
    those lines' A1 / A2 targets. R3 fails on the left and right p90 regardless, so the decision stands.
  - The fix (clip the cells' union across at half the gap to the neighbouring line, as `eval/tsdesk_dense.regions`
    does) changes the registered reference, so it is a follow-up: #177.
- **Lines at the frame's edge.** When the ring around the cells holds the line's own ink, as for a line touching the
  frame's edge, it falls back to the cells (1 of 1,519 sampled hval lines).

### The decode (`tools/tsdesk.py`) and the refinement (`tools/tsdesk_refine.py`)

The decode gives each line's **ink box, with no margin**:
1. It takes the 4-connected components of K > 0 at stride 8.
2. Across the line, it averages centre − A1 and centre + A2 over the component's cells.
3. Along the line, it averages centre − E1 over the component's first column and centre + E2 over its last.

Each end may lie up to 3 line heights beyond the kernel, but never inside the first or last cell's far edge.

**The refinement** then snaps each horizontal box's edges to its line's ink on the full-resolution grey frame (helper
textspot-desktop-refine). It never grows into another of the frame's boxes, and it lets the start cross gaps of at most
35% of the line height (a 「 sits next to the glyph after it; a frame ornament does not). It never shrinks across.
In C it costs about 0.06 ms per 960×544 frame and 0.17 ms per 1920×1080 frame (the C helper's profile of seed 1's
blob; the refinement helper measured 0.05 and 0.16 ms on Sd's boxes).

The C path (`desktop/vocr_tsd.c`, helper textspot-desktop-c) gives the Python decode's and refinement's boxes bit for
bit from the same maps.

### The reading pad

**The recognizer reads each tight box grown by a pad.** The pad was chosen before any candidate was read, on exact
boxes only (detector-free; amendment 1): every hval line's ink box and every dense_screen line's box, read by
VOCR6.2-L at seven pad rules (kH = max(2, round(k H / 544)) px horizontally, likewise vertically):

| pad rule (horizontal / vertical k) | hval (4,720 lines) | dense_screen (1,220 lines) | mean |
|---|---:|---:|---:|
| k3 (3 / 3) | 0.444% | 0.218% | 0.331% |
| k4 (4 / 4; eval/desktop.py's `auto`) | 0.451% | 0.236% | 0.343% |
| k5 (5 / 5) | 0.430% | 0.218% | 0.324% |
| k6 (6 / 6) | 0.445% | 0.332% | 0.389% |
| k8 (8 / 8) | 0.513% | 0.665% | 0.589% |
| k4v6 (4 / 6) | 0.473% | 0.314% | 0.394% |
| **k6v4 (6 / 4)** | **0.423%** | **0.169%** | **0.296%** |

**k6v4 is the reading pad:** 6 px left and right and 4 px top and bottom at 960×544, and 12 and 8 px at 1920×1080.

## The protocol

`eval/textspot_desktop_protocol.json` was registered before any candidate was trained or any decision set read. Each
amendment came before the reads it governs, with one exception: amendment 3 says it came before arm Sd was read, but
Sd's screening reads had finished 44 s to 2 min earlier (disclosures). The experiment log has a line for each
amendment, and the disclosures below say what had been read.

- **Screening:** seed 1 per arm, scored on hval and a held-out dense PC screening set (dense_screen). The score is the
  mean of the four dialogue CERs: hval and dense_screen, at fit:960 and long:960.
- **Decision:** 3 seeds of the chosen arm, decided on dev-v2 (84 unseen VNs) and a held-out dense PC decision set
  (dense_dev).
- **The rule** for a candidate (the representative seed at an input policy):
  - R1: dev-v2 CER ≤ textspot8 + VOCR6.2-L, with Vita and PSP each not resolved worse;
  - R2: dense_dev ≤ 2.00%;
  - R3: on hval, per edge median |error| ≤ 1 px and p90 ≤ 2 px against the exact ink boxes, and the tightness cost
    (CER from its boxes against CER from exact boxes) not resolved above 0;
  - R4: at most 5 junk boxes in PCSG00854, the ornament-strip title;
  - R5: the C AVX2 detector at 1 thread, p50 ≤ 3.0 ms on 960×544 frames and ≤ 8.0 ms on 1920×1080 frames;
  - R6: C crops equal the ONNX path's on ≥ 99% of dev-v2 frames, CER within 0.05 points.
- **The PC test set** (#171, 942 dialogue lines of 142 VNs) is read once, after the decision record. vocr-desktop
  reads it under its own protocol, and this read is independent of theirs.

**The dense sets** (`eval/tsdesk_dense.py`, `data/manifests/tsdesk_dense_v1.json`) are 1b v3 frames with at least 8
labelled lines, from 200 VNs held out of training: every frame of those VNs is left out of 1b and of the real half.
Five of them have screenshots among the synthetic frames' backgrounds (disclosures).
- dense_screen: 100 VNs, 120 frames, 1,220 lines.
- dense_dev: 100 other VNs, 120 frames, 1,203 lines.

MeikiOCR's author found textspot8 weak on screens with this much text.

## Screening (hval and dense_screen)

The baselines on the same sets:

| pipeline | hval | dense_screen | hval box edges vs exact, median / p90 |L/T/R/B| (px) |
|---|---:|---:|---|
| R: MeikiOCR's detector (+4 px) + VOCR6.2-L | 2.346% | 0.483% | 2/1/1/1 / 4/2/3/2 |
| textspot8 + VOCR6.2-L | 2.606% | 6.677% | 8/5/8/5 / 14/7/17/7 (its boxes carry a pad) |
| the exact boxes (+ k6v4) | 0.423% | 0.363% | |

The arms (seed 1, CER in %, fit:960 / long:960; the edges at fit:960, before the refinement):

| arm | recipe | steps | ONNX ms | hval | dense_screen | score | PCSG00854 junk | edges median / p90 |
|---|---|---:|---:|---|---|---:|---:|---|
| Sa p8a | base | 40k | 2.32 | 5.27 / 3.81 | 6.16 / 2.58 | 4.455 | 39 | 1.7/0.9/2.2/0.8 / 4.9/2.3/9.8/2.0 |
| Sb p8s | base | 40k | 1.81 | 6.61 / 3.95 | 4.98 / 4.13 | 4.916 | 67 | 1.8/1.0/2.3/0.9 / 5.8/2.5/11.9/2.2 |
| Sc p4a | base | 40k | 2.38 | 11.18 / 10.23 | 14.30 / 10.12 | 11.455 | 114 | 3.2/1.0/2.8/0.8 / 9.4/2.5/15.3/2.3 |
| Sd p8g | base | 40k | 2.38 | 3.54 / 2.99 | 3.70 / 2.25 | 3.122 | 53 | 1.5/1.0/1.9/0.9 / 4.8/2.3/10.0/2.2 |
| Se p8a | junk | 40k | 2.33 | 4.45 / 2.98 | 4.90 / 2.69 | 3.755 | 111 | 1.5/0.8/1.9/0.7 / 5.1/2.1/8.0/1.9 |
| Sf p8a | junk + end-ignore | 40k | 2.33 | 4.18 / 2.62 | 5.02 / 2.95 | 3.692 | 61 | 2.0/0.9/2.4/0.8 / 5.1/2.2/11.0/1.9 |
| Sh p8g | junk + end-ignore | 40k | 2.39 | 5.19 / 4.26 | 5.54 / 3.75 | 4.684 | 87 | 1.7/1.1/1.9/0.9 / 4.9/2.5/8.6/2.4 |
| Sj p8g | junk + end-ignore + ornaments | 40k | 2.39 | 3.01 / 1.69 | 4.28 / 2.77 | 2.939 | 54 | 1.6/1.0/2.1/0.8 / 4.9/2.3/8.7/2.1 |
| **Si p8g** | junk + end-ignore | **80k** | 2.45 | 2.85 / 1.48 | 3.63 / 2.08 | **2.511** | 66 | 1.5/1.1/1.9/0.9 / 4.6/2.4/8.3/2.2 |
| Sk p8g | junk + end-ignore + ornaments | 80k | 2.37 | 2.03 / 1.51 | 4.28 / 2.50 | 2.582 | 14 | 1.6/0.8/2.1/0.8 / 4.4/2.1/9.8/1.9 |

The recipe options:
- **junk:** the hooked pages' weak MeikiOCR boxes (score < 0.5) are background, not ignore areas.
- **end-ignore:** for the sources whose boxes are not exact, the cells within one line height past a line's ends are
  ignore areas.
- **ornaments:** a quarter of the windows get a row of repeated small ornaments drawn as background.

- **The pick is Si** (the rule's lowest score). The refinement is adopted by rule 3, which asks for a lower hval half
  and a lower mean median edge error:
  - hval half 2.165% → 2.122%;
  - the mean median edge error 1.31 px → 0.49 px.

  Its CER gain on Si is not resolved: hval −0.048 points [−0.118, +0.031] at fit:960, −0.037 [−0.101, +0.038] at
  long:960. Its last change was made on hval (disclosures).
- **Steps matter most:** Si and Sh share every setting but the steps.
- **One seed per arm adds noise:** Sh scores below Sd although it only adds two recipe options.
- **The ornament rows** cut PCSG00854's junk boxes to 14 (Sk). Sk's score is 0.07 above Si's at the same cost, so the
  rule keeps Si.
- **PSP frames at their own 480×272** read poorly at fit:960: hval PSP 3.2-10.5%, and 13.4% for Sc. long:960 doubles
  them to 960×544: 0.4-2.8%, and 10.0% for Sc.

## The decision (dev-v2, dense_dev; `eval/textspot_desktop_decision.json`)

The chosen arm's three seeds (80k steps each, the refinement on; CER in %):

| seed | dev-v2 fit:960 / long:960 | dense_dev fit:960 / long:960 | hval fit:960 / long:960 |
|---|---|---|---|
| 1 (the screening run) | 8.17 / 1.49 | 4.02 / 3.39 | 2.80 / 1.45 |
| **2 (the representative: the median by dev-v2 at fit:960)** | 9.81 / 3.41 | 4.01 / 3.34 | 3.81 / 3.33 |
| 3 | 10.97 / 2.44 | 4.61 / 2.82 | 3.65 / 2.85 |

**Baselines.**

| | dev-v2 | dense_dev |
|---|---:|---:|
| textspot8 + L | 1.355% | 7.534% |
| R | 0.770% | 0.353% |
| the exact boxes (+ k6v4) | 0.863% | 0.347% |

**The representative against the rule:**

| clause | fit:960 | long:960 | needed |
|---|---|---|---|
| R1 dev-v2 | 9.81% | 3.41% | ≤ 1.355% |
| R1 Vita view minus textspot8 | −0.15 [−0.86, +0.58] | −0.20 [−0.90, +0.50] | not resolved above 0 |
| R1 PSP view minus textspot8 | +15.70 [+8.49, +23.55] | +3.96 [+0.50, +8.14] | not resolved above 0 |
| R2 dense_dev | 4.01% | 3.34% | ≤ 2.00% |
| R3 hval edges, median / p90 (L/T/R/B) | 0/1.05/0/0.96 / 4/2.6/11/2.2 px | 0/1.02/0/0.99 / 3/2.5/9.8/2.1 px | ≤ 1 / ≤ 2 px each |
| R3 tightness cost | +0.97 [+0.47, +1.50] | +0.81 [+0.34, +1.32] | not resolved above 0 |
| R4 PCSG00854 junk boxes | 56 | 56 | ≤ 5 |
| R5 C detector + refinement, 1 thread | 2.20 / 2.75 ms | 2.18 / 2.71 ms | ≤ 3.0 / ≤ 8.0 ms (960×544 / 1920×1080) |
| R6 C crops = ONNX crops on dev-v2 | 313 / 313 frames, CER equal | 313 / 313, CER equal | ≥ 99%, CER within 0.05 |

**No candidate qualifies.**
- **It fails on PSP:** on dev-v2's Vita view the detector is level with textspot8, but at 480×272, and even doubled to
  960×544, it cuts and misses small console text.
- **The seeds differ by up to 2.8 points on dev-v2 at fit:960 (8.17-10.97%) and 1.9 at long:960 (1.49-3.41%),** with
  each seed's errors in a few VNs. The screening chose the arm through its best seed: one seed per arm gives the lowest
  screening score a winner's curse.

### The research candidate (amendment 6)

The rule names the research candidate: seed 2 retrained on train + guard + hval, at long:960, refinement on. Its
own reads (hval is in its training):

| set | fit:960 | long:960 |
|---|---:|---:|
| dev-v2 | 10.05% | **1.82%** (Vita 0.25, PSP 2.12, PSP from PC screenshots 2.52, Vita from PC screenshots 1.67) |
| dense_dev | 3.76% | **2.39%** |
| hval (in its training) | 3.65% | 2.33% |

dev-v2's Vita frames resampled to other heights (long:960):

| height | CER |
|---|---:|
| 720 | 1.42% |
| 1080 | 1.51% |
| 1440 | 1.58% |
| 2160 | 1.67% |

The same set read by others (#169):

| pipeline | CER |
|---|---|
| R | 0.65-0.73% |
| textspot8 | 1.31-1.47% |
| d64 | 0.96-1.04% |

**Like for like** (every frame fit into 960×544, MeikiOCR's detector size):

| pipeline | dev-v2 | hval |
|---|---:|---:|
| this, fit:960 | 1.75% | 2.34% |
| R | 0.68% | 2.31% |
| textspot8 | 1.47% | 2.46% |
| MeikiOCR itself | 2.47% | 4.44% |
| d64's release (#169) | 1.24% | |

The R, textspot8, MeikiOCR and d64 figures are #169's reads with the same harness.

## The PC test set, read once (`eval/textspot_desktop_final.json`)

#171's real PC test set: 518 clean units, 886 dialogue lines of 139 VNs, VNDB PC screenshots at their own size.
It was read once, after the decision record was committed. The attempt record refuses a second read under this
protocol. vocr-desktop read it under its own protocol, and the two reads are independent.

| pipeline | all | without the 27 flagged VNs | width ≤ 800 | 801-1279 | 1280-1919 | ≥ 1920 | frames fit into 960×544 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **this, long:960 + refinement** | **2.94%** | 3.39% | 4.98% | 0.67% | 1.75% | 2.90% | 2.95% |
| textspot8 + L | 3.88% | 4.50% | 5.43% | 1.01% | 2.53% | 4.45% | 3.51% |
| R: MeikiOCR's detector + L | 0.70% | 0.74% | 0.94% | 0.34% | 0.49% | 0.78% | 0.77% |
| the exact boxes (+ k6v4) | 1.12% | 1.33% | 2.89% | 0.34% | 0.53% | 0.50% | |
| MeikiOCR 0.3.5 itself | 1.97% | 2.07% | 1.61% | 1.28% | 2.23% | 2.22% | 2.65% |
| mida + L (report only) | 1.38% | 1.41% | 1.90% | 0.74% | 1.04% | 1.40% | |

- **Against textspot8:** −0.94 points [−1.93, −0.04]. The guard (not resolved above textspot8) holds.
- **Against R:** +2.23 [+0.79, +4.03].
- **vocr-desktop's d64** reads this set at 1.67% at long:960 and 2.75% for its release at long:640 (#175, its own
  read).
- The "flagged" VNs are the set's family and release-partner VNs (`docs/desktop-data.md`).
- The columns other than "all" are re-scored from the stored reads' unit counts by `eval/tsdesk_pctest_breakdown.py`
  (`eval/textspot_desktop_pctest_breakdown.json`); it reads no frame.
- **The exact boxes read small PC screens poorly at k6v4** (2.89% at widths ≤ 800 px, against R's 0.94%). The pad was
  chosen on frames of about 960 px; the set's reference boxes come from MeikiOCR's proposals refined to ink.

## Speed and memory (`eval/tsdesk_time.py`, the protocol's timing pages)

p50 ms per frame on 60 hval Vita frames (960×544), the same frames resampled to 1920×1080, and 30 1b training frames of
1920×1080 (pc1080), in one heavy slot at load 9-15 (the release model):

| pipeline | 1 thread: 960×544 / 1920×1080 | 4 threads: 960×544 / 1920×1080 |
|---|---|---|
| **C detector + refinement (AVX2 + FMA, `-O3 -march=x86-64-v3`)** | **2.36 / 2.88** | **0.89 / 1.15** |
| C detector, the plain-C build (`make ARCH=`: no `-march`, so the scalar kernels) | 16.1 / 21.4 | 4.7 / 6.3 |
| C network alone | 2.27 / 2.27 | 0.74 / 0.74 |
| ONNX Runtime network alone | 2.75 / 2.77 | 4.28 / 4.29 |
| ONNX detector (cv2 resize, the numpy refinement) | 5.2 / 21.0 | 6.4 / 10.9 |
| C page: detector + VOCR6.2-L (`runtime/vocr.c` W8I), k6v4 | 24.6 / 32.8 | 13.9 / 15.2 |
| ONNX page: detector + VOCR6.2_L_fast.onnx | 14.3 / 33.3 | 12.0 / 17.3 |

The other detectors' networks alone, at 1 thread and 960×544 (ONNX Runtime; mida-ocr-eval's measurements on hval
pages at load 31-36, so not the same session):

| network | ms |
|---|---:|
| textspot8 | 7.7 |
| vocr-desktop's d64 (S2) | 12.9 |
| mida | 25.9 |
| MeikiOCR's detector | 109.5 |

**Memory** of the release model (`eval/tsdesk_time.py mem`: a fresh process per pipeline, one frame in memory at a
time, 1 thread; `res/timing/mem_release_e35bf00_*.json`), 960×544 / 1920×1080:

| pipeline | peak | added by the pipeline |
|---|---:|---:|
| C detector + refinement | 46.0 / 51.3 MiB | 10.0 / 11.6 MiB |
| ONNX detector | 96.8 / 116.6 MiB | 61.7 / 76.9 MiB |
| C page with VOCR6.2-L | 47.4 / 55.9 MiB | 12.5 / 16.4 MiB |
| ONNX page | 126.3 / 158.9 MiB | 91.3 / 119.4 MiB |

- The peaks include Python, numpy and the frame, 35-40 MiB.
- At 4 threads every figure is within 1.1 MiB of these, except the ONNX page, which peaks at 145.2 / 177.3 MiB.
- **The numpy refinement** costs about 18 ms at 1080p in the ONNX path. Its C twin costs about 0.17 ms.

### The int8 detector (round 3, VTSD version 2)

The C detector also runs an int8 copy of a model (`desktop/vocr_tsd_q8.c`). `train/tsdesk_quant.py quant` writes it
from a float32 blob and a calibration pass, as a VTSD version-2 blob (`train/tsdesk_blob.py` documents the format).
`vocr_tsd_load` reads both versions, so every entry point (`ctsd:BLOB.vtsd`, `cbox:`, `cpage:`, `hpage:`, the page
pipeline) runs a version-2 blob unchanged.

- **The scheme.** The weights are int8 per output channel (max |w| → 127). The maps between layers are uint8 with one
  scale per tensor and zero point 0, since every map is a ReLU or a sum of ReLUs. Each layer sums its products exactly
  in int32, then requantizes in float32 per output channel. The head's maps stay float32, so the decode reads them as it
  reads the float network's. gc keeps its float32 weights.
- **One result everywhere.** The plain-C, AVX2 (exact int16 products) and AVX-512 VNNI kernels give the same maps bit for
  bit at any thread count, and so does the numpy twin `tsdesk_quant.q8_maps`. `tests/test_tsdesk_q8.py` checks this for
  every shape in `train/tsdesk_net.SHAPES` and the extra shapes.
- **Builds.** The VNNI kernels need `make ARCH="-march=x86-64-v4 -mavx512vnni"` (Ice Lake, Zen 4 and later CPUs);
  `-march=x86-64-v4` alone has no VNNI. On a CPU without VNNI the int8 detector runs about as fast as the float32 one,
  so the float32 blob is the better choice there.
- **Calibration** uses training-side frames only: desktop-data 1a (the hooked pools' train pages as desktop captures),
  1b (realplus v3) and 1c (synthetic). The timing pages are left out by key and by VN.
  `tsdesk_quant.py frames` takes the calibration frames and the check frames from disjoint shards. The scale methods are
  max, mse, and the 99.9 / 99.99 / 99.999th percentiles. `+hb` adds the head's bias correction, which sets the K map's
  mean error to 0 on cells near the decision and every other map's to 0 on kernel cells.

**Error against float32** (`tsdesk_quant.py err`; 104 check frames at long:1120: 1a 32, 1b 48, 1c 24). The boxes are
the decode's, before the refinement and the post-pass. A box matches at IoU ≥ 0.5.

| blob (`mse+hb`) | int8 blob | K sign flips (share of kernel cells) | mean \|ΔA\| / \|ΔE\| px on kernel cells | box recall / precision | mean IoU of matches | same tag |
|---|---:|---:|---:|---:|---:|---:|
| K1 (q8c) | 559,068 B | 1.8% | 0.14 / 0.51 | 97.8% / 97.2% | 0.980 | 99.5% |
| S2 (q8b) | 388,756 B | 2.2% | 0.27 / 0.90 | 97.3% / 96.8% | 0.973 | 99.3% |

The other methods' recall and precision are within 1.5 points of these. max is 1-2.3 points worse
(`res/q8err_*_v1.json` in the speed helper's artifacts).

**Speed** (`res/q8time_v1_5feacf1.json`): p50 ms per frame of the C detector with the refinement and C8's detector
options (amendment 9; the goal's det1 pipeline) on the timing pages. All pipelines ran interleaved per page in one heavy
slot. The 1-minute load was 488 at the start (a burst of new processes on the host) and 55 at the end.

| model, policy | 1 thread: float32 (AVX2) → int8 (VNNI), vita / pc1080 | 4 threads: float32 → int8, vita / pc1080 |
|---|---|---|
| S2 (q8b), long:960 | 11.1 → 4.1 / 11.9 → 5.0 | 3.9 → 1.7 / 4.6 → 2.2 |
| S2 (q8b), long:1120 | 18.0 → 6.2 / 18.8 → 7.1 | 5.8 → 2.3 / 6.3 → 2.9 |
| K1 (q8c), long:960 | 19.6 → 6.9 / 20.3 → 7.8 | 6.5 → 2.5 / 7.1 → 3.1 |
| K1 (q8c), long:1120 | 31.7 → 10.9 / 32.5 → 11.7 | 9.5 → 3.7 / 10.2 → 4.2 |

At 1 thread the int8 detector runs 2.4-3.0 times as fast as the float32 one. The int8 kernels of the AVX2 build run at
0.85-1.03 times the float32 speed at 1 thread and 0.96-1.10 times at 4 threads. The accuracy of an int8 blob has to be read on its own (`ctsd:` on the version-2
blob): its maps differ from the float32 network's by more than the float paths differ from each other.

## Does it help on the Vita?

**Not as it stands.**
- **Cost:** at 95 M multiply-adds per 960×544 frame it costs about as much as textspot8 (86.8 M, which halves the frame
  first). Scaled from textspot8's estimated 264 ms per frame at 333 MHz by multiply-adds, an int8 NEON port would
  take about 290 ms (95 / 86.8 × 264). That is an extrapolation, not an A9 cycle estimate.
- **Accuracy:** it reads console screens no better (dev-v2's Vita view level with textspot8, its PSP view worse).

**What could transfer to a smaller Vita variant (a follow-up, not this work):**
- the ink-in-cells training targets: `refined_line`, which textspot8's hooked targets use too, cuts end glyphs;
- the ornament-row background augmentation;
- the longer schedule.

## What would close the gap

- **Training length.** Twice the steps took the same recipe from a screening score of 4.68 to 2.51, and the loss was
  still falling at 80k.
- **Seed stability.** The three seeds read dev-v2 at 1.49-3.41% at long:960. A detector this small needs more steps,
  more PSP-scale data or more capacity before one seed predicts another.
- **Capacity within the time budget.** In C the detector uses 2.4 of R5's 3.0 ms at 960×544, about 25% headroom.
- **PSP text.** Even doubled to 960×544, PSP frames read at about 2.1% against textspot8's 1.5%.
- **Ornament strips.** The ornament rows cut PCSG00854's junk boxes from 87 to 54 at 40k steps (Sh → Sj) and from 66
  to 14 at 80k (Si → Sk). Sk scored 0.07 above the chosen arm.
- **Line ends** (R3's right-edge p90 of 8-10 px):
  - page cursors and icons next to a line's last glyph are boxed with the line;
  - some lines split at full-width spaces.
  The refinement snaps every edge to ink, so it cannot drop an icon that sits on the same ink row.


## Reproduction

The scripts as they ran are in `eval/reproducibility/textspot-desktop-20261101/` (their paths are the shared box's).

```sh
S=eval/reproducibility/textspot-desktop-20261101; H="HEAVY_SLOTS=12 $HERD/heavy.sh timeout 36000"
# the dense sets and the ink-box target cache (amendment 1)
.venv/bin/python eval/tsdesk_dense.py build --out-dir $PV/textspot-desktop/frames --manifest-out data/manifests/tsdesk_dense_v1.json
$H .venv/bin/python train/tsdesk_targets.py build --pool $A/b2-ingest2/pagepool/v2.1 --pages $A/b2-ingest2/pages \
    --out $PV/textspot-desktop/targets_v1 --splits train,guard,hval --workers 5
# the reading pad on exact boxes (detector-free)
$H .venv/bin/python eval/tsdesk_eval.py pads --frames $PV/vocr-desktop/frames/hval.pkl $PV/textspot-desktop/frames/dense_screen.pkl --out pads_exact.json
# a screening arm (e.g. Si) and its reads; the refinement's rule 3
$H $S/train.sh Si_p8g_junk_endig_80k p8g 1 train junk --end-ignore 1.0 --steps 80000
$H $S/screen_arm.sh Si_p8g_junk_endig_80k; $H $S/screen_refine.sh Si_p8g_junk_endig_80k
.venv/bin/python eval/tsdesk_rule.py screen --res $A/textspot-desktop/res --runs $A/textspot-desktop/runs --arms ... --out eval/textspot_desktop_screening.json
# seeds 2-3 and the train + guard + hval retrains; the decision reads, R6 parity, R5 timing; the rule
$S/launch_seeds.sh p8g junk Si_p8g_junk_endig_80k --end-ignore 1.0 --steps 80000
$H $S/decide_reads.sh Si 2 runs/Si_p8g_junk_endig_80k_s2/final.onnx; $H $S/baselines_decide.sh
$H $S/parity_dev.sh Si_s2 blobs/Si_s2/textspot_desktop_Si_s2.vtsd
.venv/bin/python eval/tsdesk_rule.py decide --res $A/textspot-desktop/res --arm Si --timing r5_input.json --parity r6_input.json \
    --out eval/textspot_desktop_decision.json
# after the decision record is committed: the PC test set once, the report reads, the timing; the release
$H $S/pctest_once.sh blobs/final_Si_s2/textspot_desktop_final_s2.onnx long:960 "..."
$H $S/report_res.sh blobs/final_Si_s2/textspot_desktop_final_s2.onnx; $H $S/time_final.sh
.venv/bin/python $S/package_release.py --run runs/final_Si_p8g_junk_endig_80k_s2 --repo . --out release/ --tag TAG --card CARD.md
```

## Disclosures

- **Amendments 3-5 added screening arms after some arms had been read** on hval and dense_screen. Each amendment
  records which, with one error:
  - Amendment 3 (21:53:04Z) replaced the screening rule's two steps with one rule. It says it came before arm Sd was
    read, but Sd's four screening reads had finished 44 s to 2 min before it (21:51:07-21:52:20Z, the result files'
    `created` fields). Amendment 4 lists Sd as read.
  - The pick does not depend on it: the original two-step rule, with the later arms read as shape arms, also picks Si.
    Its recipe step keeps junk (Se 3.755 < Sa 4.455), and the lowest shape score is Si's 2.511, on the junk recipe.
  - The protocol file stays as registered: its sha256 is in the PC test's attempt record.
- **Speculative runs were stopped unread.** Seeds and retrains of arm Sj were launched before Si and Sk were read,
  then stopped once Si's score excluded Sj. Two arms (Sg; and the first Sa, Se, Sb on the old targets) were stopped
  before any score. Amendment 4 says Sg stopped at about step 3,000; its log reached step 8,800 (729 s).
- **The refinement was designed, and last changed, on hval.**
  - The helper's design read hval boxes and contact sheets; its later choices moved to train pages.
  - Its last change was made on hval. Version 5 failed rule 3 on Sd, Sf, Se and Sb through PCSG00854's junk boxes
    (experiment log, 23:22Z). Version 6, the adopted one, changes only the bound on shrinking across. That change was
    first seen on hval, then checked on train pages.
  - Rule 3 adopted version 6 on Si's hval reads (00:09-00:11Z), where its CER gain is not resolved (screening, above).
- **The exact reference of the edge error is an algorithm** (`ink_box`). On textured or dithered boxes it can still
  miss a faint 。 or 」, which counts against a detector that boxes it. On tight-spaced titles it reaches into the
  neighbouring line: 3.6% of hval lines and 3.7% of the hooked training targets ("Its limits", above; #177).
- **The synthetic frames' backgrounds** include screenshots of 5 of the 200 held-out dense VNs (the audit's count):
  - textspot's scene pool holds 16 screenshots of v1001, v1120, v14922 and v2836 (dense_screen) and v6682 (dense_dev);
  - 165 of textspot's 80,000 synthetic frames and 38 of desk_synth's 10,000 are drawn over them;
  - none of them is a dense-set screenshot;
  - dense_dev without v6682 reads 2.401% against 2.394% (the release model) and 3.346% against 3.342% (the
    representative).
- **The dense sets' boxes come from 1b's labels** (MeikiOCR proposals refined to ink, Claude-reviewed). The PC test
  set's lines are MeikiOCR's proposals (`docs/desktop-data.md`).
- **mida's licence is not stated.** It was used for measurement only, through `eval/mida_det.py`: not redistributed,
  not trained on.
- **Release partners of training titles** are as in #169: PCSG00007 and ULJM05383 share a VNDB release with CLANNAD;
  PCSG00062 and PCSG00829 are vita-test / psp-test VNs. The owner's 2026-10-25 policy allows them. CLANNAD,
  vita-test and psp-test were never read.
