# textdet: a learned text detector for the console finder (issue #96)

The recognizers now read hand-cut lines far better than the lines the GRAD finder cuts: VOCR5 reads vita-test +
psp-test dialogue at 1.60 / 1.13 / 0.79% CER on fixed lines and at 8.31 / 7.98 / 7.08% on finder-cut whole dialogue
(`docs/vocr5.md`). Issue #96 proposes a tiny learned detector for 960×544 and 480×272 VN screens that returns
ink-tight line boxes, whether the region holds text at all, and each box's orientation, run by a dependency-free C99
runtime next to `vocr_lines.c`. The owner's steering (2026-10-17 21:16Z): "we need it to run on a ps vita, so
smaller == better; play around with sizings".

This document starts as the pre-registered protocol (frozen 2026-10-17T21:44:14Z, before any detector training run,
`eval/textdet_protocol.json`). Results are added below it as they are measured; nothing above "Results" changes
except through a dated amendment.

## Protocol

### What is compared

- **Finders.** GRAD (`runtime/vocr_lines.c`, frozen), GRAD + refine (`runtime/vocr_lines_refine.c`,
  `VOCR_REFINE_DEFAULT`) and the detector candidates (`det`: an int8 blob run by `runtime/vocr_textdet.c` and its
  bit-exact twin `synth/textdet_twin.py`; a candidate whose C and twin outputs differ on any development region is
  ineligible).
- **Recognizers.** The released VOCR5 models of v0.3.0 (`VOCR5_S_w8.vocr`, `VOCR5_N_w8.vocr`, `VOCR5_L_w8.vocr`;
  sha256 in the protocol), in W8I through the C runtime, each crop with `VOCR_INVERT` from the box's dark flag.

### Development data (selection reads only these)

- **wd: whole dialogue.** VOCR5's development cohort, unchanged (`eval/vocr5_select.load_wd`): finder-tighten's
  tv_dlg and dn_dlg regions of the reserved.json `dev_sample` VNs, tv_dlg plus the dn_dlg frames tv_dlg does not hold,
  at the base region and grown by 4 and 8 px: 1,161 units, 93 VNs. Every crop a finder returns is read, joined in the
  finder's order and scored by `transcript_score` (dialogue CER; these regions hold no speaker name). Primary figure:
  the three variants pooled. Intervals: 10,000 VN-cluster bootstrap resamples (`eval/textdet.vn_diff`).
- **native_vita: the CLANNAD-dev-style subset.** The wd units whose frame is a native PS Vita screenshot (teacher_val
  frames whose teacher read is of kind `vita`, devnative's Vita frames): 150 units, 9 VNs. This is the kind of text
  on which finder-tighten's refinement was level on development and worse on CLANNAD.
- **empty: text-free regions.** eval/linefind's `empty` regions of vita_dev and psp_dev (the regions of a reserved test
  VN left out): 66 regions. The empty rate is the share for which a finder returns no line at all.
- **linefind detection** (vita_dev, psp_dev; vita_holdout and psp_holdout report only): line recall / precision / F1
  at IoU ≥ 0.5, region exact, merged / split / cut. research_a/b, which hold vita-test screenshots, are never read.
- **Report only:** the owner's CHAOS;CHILD native Vita frames (finder-tighten's own_dlg Vita units; the detector never
  trains on v14018, the VOCR5 models did), the dn pairs of the dev_sample VNs (the detector's box against the
  hand-cut crop of the same line), and gate arms (GRAD or GRAD + refine with every line dropped when the detector finds
  no text in the region).

### The frozen baseline (`eval/textdet_baseline_20261017.json`)

Re-scored here through `eval/finder_tighten.py score` on the wd cohorts; the units equal VOCR5's stored selection
units for the three selected runs, unit for unit.

| development | VOCR5-S | VOCR5-N | VOCR5-L |
|---|---:|---:|---:|
| wd dialogue CER, GRAD | 7.44% | 6.87% | 6.20% |
| wd dialogue CER, GRAD + refine | 5.21% | 4.78% | 4.46% |
| native_vita, GRAD | 0.83% | 1.07% | 0.94% |
| native_vita, GRAD + refine | 0.91% | 1.02% | 0.80% |

Errors are concentrated: on GRAD + refine with VOCR5-N, 10 of the 93 VNs hold 81% of the errors, mostly lines merged
or straddled on PSP-sized text over busy or translucent boxes, NVL pages and side-by-side choices. On the 66 empty
regions GRAD returns no line for 5 (0.076), GRAD + refine the same.

linefind detection, GRAD → GRAD + refine (line F1, region exact; empty rate):

| set | regions | F1 | region exact | empty rate |
|---|---:|---|---|---|
| vita_dev | 204 | 0.716 → 0.723 | 0.441 → 0.461 | 0.027 |
| psp_dev | 159 | 0.635 → 0.669 | 0.321 → 0.352 | 0.138 |
| vita_holdout | 48 | 0.582 → 0.590 | 0.313 → 0.313 | 0.000 |
| psp_holdout | 90 | 0.722 → 0.757 | 0.444 → 0.422 | 0.067 |

### The detector

- **Network** (`train/textdet_model.py`): a 3×3/2 stem, a downsampling stage to output stride 4 or 8, 3-5 blocks
  (separable: a dilated 3×3 depthwise then a 1×1; or a plain dilated 3×3) and a 1×1 head with four maps per cell: K
  (text kernel), O (vertical text), A1 and A2 (distances from the cell centre to the line's near and far edges across
  the line). BatchNorm is folded; weights and activations are int8, sums int32, requantisation fixed-point: integer
  arithmetic only, so the twin is exact.
- **Targets.** A line's box is its main-text ink box (fill, outline and shadow; furigana, emphasis dots, cursors and
  icons excluded). The kernel is the cells whose centre lies within max(h/4, s/2) of the line's centre line (h: the
  ink size across the line, s: the stride) and whose span meets the box along the line.
- **Decode** (integers only, fixed by the protocol): kernel cells K > 0, 4-connected components; components under 2
  cells (stride 4) or 1 (stride 8) dropped; vertical iff the component's O codes sum above 0; with c2 = twice a cell's
  centre across the line and n cells, near edge = floor(Σ(c2 − A1) / 2n) and far edge = ceil(Σ(c2 + A2) / 2n); along
  the line, the component's cells; a pad of max(2, (18h + 50) / 100) px on every side, clipped to the region; boxes
  under 4 px across dropped; horizontal lines top to bottom (left to right where two overlap by half the smaller
  height), then vertical lines right to left; at most 16 (`VocrLineSet`); the dark flag is `vocr_line_dark` on the
  padded box. A region returns text iff at least one line comes back. Vertical boxes are flagged and never handed to
  the horizontal recognizers.
- **Runtime** (`runtime/vocr_textdet.c`): C99, no malloc, a caller arena, integer only, the same row-callback form as
  `vocr_lines_rows.h` (each layer keeps only the rows its kernel spans, so the arena depends on the region's width, not
  its height, and a strip height changes no output), builds for mips32 and the VitaSDK. A new finder mode behind a
  flag; `vocr_lines.c` stays frozen and bit-identical when the flag is off.

### The size sweep (`eval/textdet_protocol.json` "sweep")

18 shapes, 24 runs (seed 1 for every shape, seed 2 for six): stride 4 separable at block widths 4, 6, 8, 12, 16 and
24 (and 5-block / 4-block variants), stride 4 plain 3×3 at 4 and 8, stride 8 separable at 8, 16, 24, 32 and 48 (and a
4-block 32), and two head ablations without the edge maps. They span 260-10,104 int8 weight bytes and 3.0-25.7 M MACs
on a 900×150 text box (11.6-97.7 M on a whole 960×544 screen). Training: `train/textdet_train.py` through
`HEAVY_SLOTS=12 $HERD/heavy.sh`, 5 threads, 30,000 steps of 16 windows of 192×512, AdamW 4e-3 cosine, int8 fake
quantisation from step 24,000, half synthetic and half real per batch; the last step is the run's model. Outside the
rule: an int4-weight study (report only) and the arena table. A trunk shared with the recognizer is not swept: the
recognizers read 32-px line crops resized after detection, so none of their layers can run on the region before the
lines are known.

### Training data (`synth/textdet_data.py`, pinned before the first run)

- Synthetic: synth3 version 3's region renderer, whose ink, fill and ruby masks give exact boxes, in canvases of
  several sizes (the region, loose crops, whole screens on text-free backgrounds), plus vertical-text renders and empty
  negatives.
- Real: MeikiOCR's line boxes refined to ink and its detector boxes (with their vertical flag) from realplus, teacher,
  devnative's training part and the owner's frames, cut into player-like and text-free regions.
- Excluded: every reserved.json `test` and `dev_sample` VN and `sessions` release; every screenshot of vita-test,
  psp-test and eval/linefind and every VN of the four linefind sets; every wd frame; the owner's CHAOS;CHILD (v14018).
  `eval/tools/reserve_check.py` runs on every real shard.

### The promotion rule (pre-registered)

- **Eligible:** int8 blob < 100,000 B; arena at 960×544 ≤ 131,072 B (the runtime's own function); ≤ 30 M MACs on a
  900×150 region; C equals the twin on every development region.
- **Size order:** blob bytes, then MACs on a 900×150 region.
- **Qualifies:** for each of VOCR5-S, -N and -L, both det − GRAD and det − refine on wd (points, variants pooled)
  have a VN-cluster 95% interval entirely below 0; on native_vita no difference (det − GRAD or det − refine, any
  model) has an interval entirely above 0; the development empty rate is at least 0.75.
- **Winner:** the smallest qualifying candidate W. A larger qualifying candidate replaces W only when the mean over
  the three models of its wd CER difference to W has a VN-cluster interval entirely below 0 (the smallest such
  candidate, repeated until none is left).
- **No winner:** the verdict is negative, GRAD (+ refine where the overlay uses it) stays, and the protected sets are
  read once for the eligible candidate with the lowest mean wd CER, labelled not promoted.
- **Fine-tune on detector crops** (issue #96's plan): only if no candidate qualifies while one passes the native_vita and
  empty conditions with wd CER below GRAD's and refine's (point estimates) for every model; then each tier's released
  checkpoint gets 3,000 steps of VOCR5's arm F recipe with that candidate's crops in place of the GRAD and refined
  crops, and the rule is re-applied with the fine-tuned models in every arm.

### Protected sets (once, after the selection is committed)

vita-test + psp-test whole dialogue (eval's regions, base / +4 / +8; GRAD, refine and det with each VOCR5 model; body
and all text; VN-cluster intervals) and detection against eval's line boxes; the CLANNAD long run against the frozen
references (each page's ROI; a reference line reads the finder lines that lie mostly inside its GRAD box; whole-page
bootstraps; plus the lines returned where the references hold no text). Attempt records refuse a second run.

### Estimates

Every console time and memory figure is a host estimate until measured on a console. Two methods: the MAC rate of
VOCR3-H15 on the console (1,564,992 MACs per CTC frame at 5.0-5.2 ms on one A9 core, 333 MHz assumed: 0.301-0.313
GMAC/s, ×444/333 at 444 MHz), and `vita/a9_cycles.py` on the Vita compiler's code under plugin qemu-arm with
ocr-speed's calibration. Reported per 900×150 and 450×80 text box and per 960×544 and 480×272 screen.

## Results

Pending.
