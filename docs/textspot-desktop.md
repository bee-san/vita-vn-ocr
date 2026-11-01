# textspot-desktop: a fast desktop text-line detector with tight boxes (issue #173)

The owner (2026-11-01 19:38Z): "a text detection algo that has really tight bounding boxes but is ultra fast."

vocr-desktop (#169) trains a desktop line detector aimed at MeikiOCR-level accuracy (its d64 arm: 0.81 G MACs at
960×544). This work is the speed tier beside it. It reuses #169's harness, #171's data and the desktop C path, and
trains a much smaller network whose boxes are tight to the ink. VOCR's recognizer then reads them as well as it reads
exact boxes.

__RESULTS_SUMMARY__

## How it works

### The network (`train/tsdesk_net.py`)

A float32 fully convolutional network that takes the grey frame (integer luma, as VOCR's recognizer reads it):

- **Stem:** a 4×4 convolution at stride 4, which cuts the frame into 4×4 patches and maps each to c0 channels.
- **Stages:** separable stages at strides 8, 16 and 32. Each is a stride-2 separable down block and n residual
  separable blocks with dilations.
- **Global context** (one arm, `p8g`): the mean of the stride-32 features through a 1×1 layer, added back everywhere.
- **Pyramid:** a top-down pyramid of 1×1 laterals back to the head's stride (8, or 4 for one arm).
- **Head:** ten maps:
  - K, a line kernel's logit;
  - O, the orientation logit;
  - A1 and A2, the distances from a cell's centre to the line's two edges across it;
  - E1 and E2, the distances from a cell's centre to the line's start and end along it;
  - four tag logits.

The distance maps are in input px. The family's sizes at 960×544:

__FAMILY_TABLE__

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
the native pages and in #171's desktop captures (the box is mapped through each capture's exact geometry)
(amendment 1). The along-line (E) maps learn only from sources whose boxes come from renderer or rendering geometry:
the hooked pages, their captures and the synthetic frames.

### The decode (`tools/tsdesk.py`)

The decode gives each line's **ink box, with no margin**:
1. It takes the 4-connected components of K > 0 at the head's stride.
2. Across the line, it averages centre − A1 and centre + A2 over the component's cells.
3. Along the line, it averages centre − E1 over the component's first column and centre + E2 over its last.

Each end may lie up to 3 line heights beyond the kernel, but never inside the first or last cell's far edge. The C
decode gives the Python decode's boxes bit for bit from the same maps.

### The reading pad

**The recognizer reads each tight box grown by a pad.** The pad was chosen before any candidate was read, on exact
boxes only (detector-free; amendment 1): every hval line's ink box and every dense_screen line's box, read by
VOCR6.2-L at seven pad rules:

__PAD_TABLE__

The rule with the lowest mean CER, **k6v4**, is the reading pad: max(2, round(6 H / 544)) px left and right and
max(2, round(4 H / 544)) px top and bottom. At 960×544 that is 6 and 4 px; at 1920×1080, 12 and 8 px.

## The protocol

`eval/textspot_desktop_protocol.json` was registered before any candidate was trained or any decision set read. Its
amendments 1 and 2 were registered before any candidate was read; the experiment log has a line for each.

- **Screening:** five arms, seed 1, 40,000 steps, scored on hval and a held-out dense PC screening set (dense_screen).
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
- **The PC test set** (#171, 942 dialogue lines of 142 VNs) is read once, after the decision record, as a guard
  against textspot8. vocr-desktop reads it under its own protocol, independently of this read.

**The dense sets** (`eval/tsdesk_dense.py`, `data/manifests/tsdesk_dense_v1.json`) are 1b v3 frames with at least 8
labelled lines, from 200 VNs held out of training: every frame of those VNs left out of 1b and of the real half. 100 VNs
form dense_screen (120 frames, 1,220 lines) and 100 form dense_dev (120 frames, 1,203 lines). MeikiOCR's author found
textspot8 weak on screens with this much text.

__RESULTS__

## Reproduction

__REPRO__
