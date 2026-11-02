# textspot-desktop: a fast desktop text-line detector with tight boxes (issue #173)

The owner (2026-11-01 19:38Z): "a text detection algo that has really tight bounding boxes but is ultra fast."

vocr-desktop (#169) trains a desktop line detector aimed at MeikiOCR-level accuracy (its d64 arm: 0.81 G MACs at
960×544). This work is the speed tier beside it. It reuses #169's harness, #171's data and the desktop C path, and
trains a much smaller network whose boxes are meant to be tight to the ink, so that VOCR's recognizer reads them as
well as it reads exact boxes.

__RESULTS_SUMMARY__

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

### The decode (`tools/tsdesk.py`) and the refinement (`tools/tsdesk_refine.py`)

The decode gives each line's **ink box, with no margin**:
1. It takes the 4-connected components of K > 0 at stride 8.
2. Across the line, it averages centre − A1 and centre + A2 over the component's cells.
3. Along the line, it averages centre − E1 over the component's first column and centre + E2 over its last.

Each end may lie up to 3 line heights beyond the kernel, but never inside the first or last cell's far edge.

**The refinement** then snaps each horizontal box's edges to its line's ink on the full-resolution grey frame (helper
textspot-desktop-refine). It never grows into another of the frame's boxes, and it lets the start cross gaps of at most
35% of the line height (a 「 sits next to the glyph after it; a frame ornament does not). It never shrinks across.
It costs 0.05 ms per page in C.

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
amendment came before the reads it governs; the experiment log has a line for each, and the disclosures below say what
had been read.

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

- **The pick is Si** (the rule's lowest score). The refinement is adopted by rule 3:
  - hval half 2.165% → 2.122%;
  - the mean median edge error 1.31 px → 0.49 px.
- **Steps matter most:** Si and Sh share every setting but the steps.
- **One seed per arm adds noise:** Sh scores below Sd although it only adds two recipe options.
- **The ornament rows** cut PCSG00854's junk boxes to 14 (Sk). Sk's score is 0.07 above Si's at the same cost, so the
  rule keeps Si.
- **PSP frames at their own 480×272** read poorly at fit:960 (hval PSP 3.6-10.5%). long:960 doubles them to 960×544
  (0.4-2.8%).

__RESULTS__

## Reproduction

__REPRO__
