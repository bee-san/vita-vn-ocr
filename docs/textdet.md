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

Every number here is a host measurement through the C runtimes (the detector, the GRAD finder, the refinement and the
VOCR5 recognizers in W8I) or a host estimate of console time, labelled as such. Nothing of the detector has run on a
console.

### Answer first

- **No candidate met the pre-registered promotion rule in any of the three rounds, so the verdict is negative: the
  learned detector does not replace GRAD + refine as the overlay's default finder.** The best one is resolved better
  than GRAD and has a lower development CER than GRAD + refine for all three models, but no interval against GRAD +
  refine excludes zero, and on CLANNAD it is resolved worse for Normal and Large (as scored).
- **The best detector is tiny:** `s8_c24` (round 3's report candidate `s8_c24_seed4`): **5,700 B of int8 weights**,
  an arena of 92,639 B at 960×544 with the refinement (the always-safe tier's 131,072 B), 10.6 M MACs on a 900×150
  text box, an estimated 53 ms on one Cortex-A9 core at 333 MHz (40 ms at 444) per box and 15 / 11 ms per 450×80 PSP
  box (A9 cycle model, high bound; estimates).
- **Development whole dialogue** (det+refine, VOCR5-S / N / L): **3.67 / 3.39 / 3.12%** against GRAD 7.44 / 6.87 /
  6.20% (−3.8 / −3.5 / −3.1 points, every VN interval below 0) and GRAD + refine 5.21 / 4.78 / 4.46% (−1.5 / −1.4 /
  −1.3 points; VN intervals reach +0.79 / +0.87 / +0.83). Most of the gain over GRAD + refine is one VN: on v27449 (a
  translucent box over a busy scene) GRAD and GRAD + refine read 40-47% of the characters wrong and det+refine 6-9%;
  without it det+refine − refine is −0.33 / −0.22 / −0.26 points (VN intervals −2.1 to +1.2). Native Vita frames 0.29
  / 0.19 / 0.16% against refine's 0.91 / 1.02 / 0.80% (intervals include 0). By platform (VOCR5-N, without
  v27449): the Vita frames are resolved better than GRAD + refine (0.97% against 2.36%, −1.40 points, VN interval
  −2.85 to −0.22), the PSP frames are not (5.24% against 4.44%, +0.80, −1.78 to +3.13). Empty regions: no line in
  **92%** of them (GRAD 8%; most of them lie on screenshots the synthetic training canvases also drew backgrounds
  from, and the rate is the same on the others: "What changed after the freeze").
- **vita-test + psp-test whole dialogue, once** (body text, all variants): **6.28 / 5.99 / 5.62%** against GRAD 10.53
  / 10.23 / 9.42% (resolved, −4.3 / −4.2 / −3.8 points) and GRAD + refine 9.51 / 9.16 / 8.35% (−3.2 / −3.2 / −2.7,
  intervals reach +0.35 / +0.18 / +0.44). On these sets it is all PSP: psp-test 6.08 / 5.58 / 5.33% against
  refine's 12.33 / 11.53 / 10.92% (−6.2 / −6.0 / −5.6 points, resolved), vita-test level (6.47 / 6.40 / 5.90% against
  6.73 / 6.83 / 5.81%). The development data point the other way (above): their PSP frames gain nothing once v27449
  is set aside.
- **CLANNAD, once** (2,052 dialogue lines, as scored: the scorer's line-to-reference rule is stricter than the
  registered one, below): 1.54 / 1.15 / 1.02% against GRAD 1.38 / 0.96 / 0.73% and refine 1.43 / 0.92 / 0.63%:
  **worse for Normal and Large** (+0.23 and +0.39 points against refine, page intervals above 0), level for Small.
- **Is it worth it over GRAD + refine?** Not as the default. Its gain over GRAD + refine is large where GRAD fails
  (psp-test: about half of the whole-dialogue errors; the development VN v27449), small or absent on the development
  data as a whole without v27449 (−0.2 to −0.3 points; their PSP frames +0.8) and on vita-test, resolved on the
  development Vita frames (−1.4 points without v27449), and on CLANNAD's native Vita text, where the GRAD crops already read near their ceiling, it loses 0.1-0.4
  points as scored. It also returns no line for 92% of empty regions, where GRAD does so for 8%. The evidence does not
  single out a platform: psp-test (14 games) favours PSP text, the development PSP frames (54 VNs without v27449) do
  not. The case for Rubi is therefore an opt-in finder, chosen per title where GRAD visibly fails (merged lines,
  translucent boxes over busy scenes, junk on empty boxes) with A/B logging on the console, or the detector as a
  text-presence gate in front of GRAD + refine (below). This recommendation is made after the protected sets were
  read: a deployment caution, not a selection.

### What changed after the freeze (amendments 1-4)

| amendment | when | what |
|---|---|---|
| 1 | before any run | the twin (`synth/textdet_twin.py`) is the decode's exact specification; the export's arithmetic |
| 2 | before any run | the data pin (`data/manifests/textdet_shards_v1.json`, 117 shards); the training details; the two `ko` ablations dropped (no defined box under the frozen decode): 22 runs |
| 3 | after round 1's scores, before round 2's | round 2: det+refine (the detector's horizontal lines through `VOCR_REFINE_DEFAULT`, unchanged) for the same 22 blobs; round 1's triggered fine-tune not run (moved to the final round) |
| 4 | after round 2's scores, before round 3's runs | round 3, the last: the strict kernel rule (`--strict-kernel`), 12 runs over seven round-1 and five deeper shapes, det+refine; the fine-tune clause withdrawn |

Two deviations from the letter of the frozen protocol, recorded in the amendments: round 1's fine-tune clause, which
its scores triggered (as did rounds 2 and 3's), never ran (the losses are box geometry, not the recognizers' input
distribution: the units whose line count the detector gets right already read better than GRAD + refine with the
unchanged recognizers), and `train/textdet_export.py` gained an optional `--wbits` after amendment 2 pinned it (all
34 runs' blobs re-export byte for byte with the pinned export, with `520d759`'s and with the final one). Three rounds
on one development sample make its intervals optimistic for the last round's candidates; the protected sets were
read once, at the end, for round 3's report candidate.

Not in any amendment, and disclosed here:

- **Probes and a smoke test.** Before the sweep three 120-step speed probes (`s4_c8`, `s8_c48`, `s4_c24`, on a
  100-canvas test manifest, 22:06-22:07Z) and, after amendment 2, a harness smoke test (a probe blob on 60 wd cases,
  read by VOCR5-S, 22:20Z) ran outside the sweep. Nothing of them is used or committed.
- **A working-tree twin in one run.** `s8_c16_seed2`'s `run.json` records a `synth/textdet_twin.py` sha256
  (`7773edd1…`) that is neither the pin nor in git: the twin was briefly edited in the working tree for the decode
  check below when that run started. Training does not use the twin, and the blob re-exports byte for byte with the
  pinned twin.
- **Decode checks.** `eval/textdet_decode_check.py` tried two decode variants (centre-consistent connectivity;
  higher kernel thresholds) with round-1 blobs on held-out canvases only: synthetic ones of an unused seed and
  realplus screenshots of training VNs that the per-VN cap left out of the pool, never development data
  (22:57Z with six finished runs, 00:05-00:08Z with five round-1 blobs). Neither variant was adopted; the decode is
  the frozen one.
- **Amendment 3's empty-rate sentence** says `eval/textdet.py` checks that det+refine never adds a line to a region
  the detector returns none for; there is no such check in the code. The claim holds: the det+refine empty rates
  equal round 1's det rates for all 22 blobs.
- **Amendment 4's "seven round-1 shapes"** are seven runs of six shapes (`s8_c24` twice).
- **Two report-only analyses were not run before the protected read:** the dn pairs (the detector's box for a line
  against the hand-cut crop: the protocol gives no rule for which detector line is "the line") were not run at all,
  and the owner's CHAOS;CHILD native Vita whole dialogue was run afterwards (below, report only).
- **The synthetic scene pool holds eval/linefind screenshots and v14018** (found after the protected read). The
  protocol says every screenshot and VN of eval/linefind, and v14018, are excluded from the training data, but
  `synth/textdet_data.py` (whose pinned docstring says the same) excluded them from the real canvases only. The
  synthetic canvases draw their backgrounds and empty negatives from synth3's scene pool (test and
  development-sample VNs excluded), which holds 94 of vita_dev's 111 screenshots, 59 of psp_dev's 78, 23 of
  vita_holdout's 26, 35 of psp_holdout's 42, 8 of research_a's 16 and 11 of research_b's 19 (119 of the 142 linefind
  VNs), and three PC screenshots of CHAOS;CHILD (v14018). 54 of the 66 development empty regions lie on such a
  screenshot. The wd cohort, the protected sets and every real canvas are unaffected. On the 12 development empty
  regions whose screenshot is not in the pool, `s8_c24_seed4` returns no line for 11 (0.92, the same as on the other
  54), and on the 6 such holdout regions for all 6 (`eval/textdet_empty_pool_check_20261018.json`), so the
  empty-region rate does not look inflated by it; the linefind detection rows (whose tight and loose regions are
  mostly the dialogue boxes in the lower screen, while the scene crops come from the upper 62%) carry the same
  caveat. It changes no decision: no candidate met the rule's
  whole-dialogue conditions, which this does not touch.
- **CLANNAD was scored with a stricter line-to-reference rule than registered.** The protocol says a reference line
  reads the finder lines with most of their height inside its GRAD box; the scorer (`eval/textdet_protected.assign`)
  gives a detector line to the GRAD box it overlaps most only when that box holds at least half the line's area.
  The area rule implies the height rule for that box, so it can only leave more detector lines unread for their
  reference. Their characters then count as missed, but under the height rule such a line, which reaches well
  outside the box along the line, could also have added insertions: the effect is most likely against the detector,
  and its size and sign cannot be measured. On development data (no
  protected data read) the two rules disagree on 0 of the 138 base-region det+refine lines of the native-Vita subset
  and on 2 of the 705 others. The CLANNAD per-line outputs were not stored and a re-read would be a second protected
  attempt, so the CLANNAD figures stand as scored.

### The runtime (`runtime/vocr_textdet.c`, `runtime/README.md`)

C99, no malloc, no recursion, integer only, a caller arena that depends on the region's width alone, and the
row-callback form of `vocr_lines_rows.h` (rows asked once in increasing order, then each line's box once for its
dark flag). `vocr_find_lines_det(..., VOCR_LINES_TEXTDET, ...)` is the new finder mode behind a flag; any other finder
value calls `vocr_find_lines` unchanged, and `vocr_lines.c`, `vocr_lines_rows.c` and `vocr_lines_refine.c` are byte for
byte unchanged (their sha256 are the protocol's). NEON kernels on the Vita, a scalar path for the PSP; both give the
twin's bits. Built strictly with gcc 11.5 and clang 18.1.8 (ASan + UBSan), VitaSDK gcc 15.2 (Thumb-2: 8,912 B of
text at -O2) and psp-gcc 15.2 (mips32: 13,068 B at -O2). `tests/textdet_parity.py` compares it with the twin on
random, probe and wild blobs of every shape and on the 22 round-1 blobs, also as Cortex-A9 NEON and scalar builds
under qemu-arm, and on every development case the C runtime equals the twin for all 34 trained blobs, det and
det+refine (`eval/textdet.py detect --cbin`). Stack peak 2,848 B on the Vita (through `vocr_line_dark`). In the
finder mode a count of 0 also results when the detector finds only vertical lines (they are dropped there);
`vocr_textdet_find` tells the two cases apart (the header's comment said "no text" until the audit; that comment-only
edit came after the A9 measurement, so `eval/textdet_a9_*.json` record the earlier header's sha256).

### Round 1: the detector alone (22 runs, `eval/textdet_round1_dev_20261018.json`)


| run | int8 B | arena B | MACs box / screen (Vita) | est. ms box @333 (A9 model) | wd S / N / L | det − GRAD, N (VN CI) | det − refine, N (VN CI) | native Vita N | empty | qualifies |
|---|---:|---:|---|---:|---|---|---|---:|---:|---|
| s4_c4_seed1 | 1,004 | 38,431 | 3.1 M / 12 M | 29 | 16.50 / 15.60 / 14.38 | +8.73 (+4.31 to +13.50) | +10.81 (+6.04 to +15.57) | 3.83 | 0.48 | no |
| s4_c4_seed2 | 1,004 | 38,431 | 3.1 M / 12 M | 29 | 13.68 / 13.09 / 12.40 | +6.22 (+2.18 to +10.73) | +8.31 (+3.62 to +13.01) | 2.81 | 0.53 | no |
| s4p_c4_seed1 | 1,100 | 38,079 | 6.3 M / 24 M | 33 | 16.23 / 15.36 / 14.77 | +8.49 (+3.22 to +13.77) | +10.58 (+4.98 to +15.82) | 1.28 | 0.48 | no |
| s4_c6_seed1 | 1,280 | 47,015 | 4.2 M / 16 M | 37 | 15.07 / 14.32 / 13.56 | +7.46 (+3.37 to +11.94) | +9.54 (+5.17 to +13.85) | 3.61 | 0.58 | no |
| s4_c8_seed1 | 1,540 | 55,591 | 5.6 M / 21 M | 41 | 9.94 / 9.25 / 8.97 | +2.38 (-1.41 to +6.01) | +4.47 (+0.11 to +8.47) | 8.91 | 0.59 | no |
| s4_c8_seed2 | 1,540 | 55,591 | 5.6 M / 21 M | 41 | 14.00 / 13.27 / 12.44 | +6.41 (+1.85 to +11.12) | +8.49 (+2.94 to +14.18) | 3.96 | 0.50 | no |
| s8_c8_seed1 | 1,884 | 39,623 | 3.0 M / 12 M | 25 | 13.31 / 12.64 / 12.06 | +5.77 (+1.37 to +10.70) | +7.86 (+2.63 to +13.22) | 5.67 | 0.70 | no |
| s8_c8_seed2 | 1,884 | 39,623 | 3.0 M / 12 M | 25 | 8.10 / 7.70 / 7.23 | +0.84 (-2.80 to +4.54) | +2.92 (-1.35 to +7.21) | 3.45 | 0.73 | no |
| s4_c8n5_seed1 | 2,228 | 71,767 | 7.9 M / 30 M | 55 | 7.17 / 6.73 / 6.45 | -0.13 (-3.81 to +3.07) | +1.95 (-2.72 to +6.88) | 2.49 | 0.71 | no |
| s4_c12_seed1 | 2,276 | 75,711 | 9.8 M / 37 M | 63 | 8.68 / 8.11 / 7.75 | +1.24 (-2.80 to +4.65) | +3.33 (-1.79 to +8.65) | 4.52 | 0.64 | no |
| s4p_c8_seed1 | 2,700 | 55,047 | 18.7 M / 72 M | 70 | 7.41 / 7.00 / 6.61 | +0.13 (-3.59 to +3.28) | +2.22 (-2.53 to +7.25) | 0.86 | 0.67 | no |
| s4_c16_seed1 | 3,108 | 95,823 | 14.9 M / 57 M | 76 | 6.53 / 6.01 / 5.59 | -0.85 (-4.32 to +2.15) | +1.23 (-2.79 to +4.98) | 6.18 | 0.80 | no |
| s4_c16_seed2 | 3,108 | 95,823 | 14.9 M / 57 M | 76 | 8.57 / 8.07 / 7.55 | +1.20 (-2.81 to +4.77) | +3.29 (-1.14 to +7.28) | 5.27 | 0.79 | no |
| s8_c16_seed1 | 3,276 | 57,623 | 4.9 M / 19 M | 32 | 8.44 / 7.85 / 7.55 | +0.99 (-2.93 to +5.15) | +3.07 (-1.49 to +7.53) | 1.74 | 0.71 | no |
| s8_c16_seed2 | 3,276 | 57,623 | 4.9 M / 19 M | 32 | 10.07 / 9.51 / 9.02 | +2.64 (-1.91 to +7.00) | +4.73 (-0.92 to +10.60) | 1.66 | 0.82 | no |
| s4_c16n4_seed1 | 3,860 | 107,999 | 18.4 M / 70 M | 92 | 5.07 / 4.56 / 4.22 | -2.30 (-5.90 to +0.89) | -0.22 (-4.04 to +2.96) | 0.70 | 0.76 | no |
| s4_c24_seed1 | 4,412 | 99,791 | 23.0 M / 88 M | 99 | 7.55 / 7.09 / 6.75 | +0.22 (-3.39 to +3.28) | +2.31 (-2.17 to +6.79) | 10.17 | 0.70 | no |
| s8_c24_seed1 | 5,700 | 87,599 | 10.6 M / 40 M | 53 | 4.09 / 3.67 / 3.37 | -3.19 (-5.85 to -1.06) | -1.11 (-4.41 to +2.08) | 0.51 | 0.86 | no |
| s8_c32_seed1 | 7,924 | 105,599 | 14.3 M / 54 M | 63 | 4.87 / 4.40 / 4.17 | -2.47 (-5.75 to +0.68) | -0.38 (-3.79 to +2.55) | 1.12 | 0.94 | no |
| s8_c32_seed2 | 7,924 | 105,599 | 14.3 M / 54 M | 63 | 5.76 / 5.29 / 5.06 | -1.57 (-4.88 to +1.46) | +0.51 (-3.31 to +3.96) | 0.40 | 0.86 | no |
| s8_c32n4_seed1 | 9,876 | 109,487 | 17.1 M / 65 M | 72 | 4.64 / 4.24 / 3.93 | -2.63 (-6.27 to +1.11) | -0.55 (-4.19 to +2.81) | 0.70 | 0.85 | no |
| s8_c48_seed1 | 14,188 | 122,127 | 25.7 M / 98 M | 92 | 4.61 / 4.21 / 3.94 | -2.65 (-5.96 to +0.46) | -0.57 (-3.99 to +2.33) | 0.43 | 0.85 | no |

Baseline (the same units): GRAD 7.44 / 6.87 / 6.20, GRAD + refine 5.21 / 4.78 / 4.46 (S / N / L).

Rule: winner None; report candidate s8_c24_seed1; qualifying []; Pareto front (bytes vs mean wd CER) ['s4_c4_seed2', 's4_c8_seed1', 's8_c8_seed2', 's4_c8n5_seed1', 's4_c16_seed1', 's4_c16n4_seed1', 's8_c24_seed1'].

s8_c24_seed1, all three models:

| model | GRAD | GRAD + refine | det | − GRAD (VN CI) | − refine (VN CI) | native Vita GRAD / refine / det | native − refine (VN CI) |
|---|---:|---:|---:|---|---|---|---|
| VOCR5-S | 7.44 | 5.21 | 4.09 | -3.35 (-6.13 to -1.13) | -1.12 (-4.60 to +2.22) | 0.83 / 0.91 / 0.78 | -0.13 (-1.03 to +0.76) |
| VOCR5-N | 6.87 | 4.78 | 3.67 | -3.19 (-5.85 to -1.06) | -1.11 (-4.41 to +2.08) | 1.07 / 1.02 / 0.51 | -0.51 (-1.31 to +0.27) |
| VOCR5-L | 6.20 | 4.46 | 3.37 | -2.84 (-5.32 to -0.75) | -1.09 (-4.22 to +1.93) | 0.94 / 0.80 / 0.35 | -0.45 (-1.33 to +0.39) |

linefind development (vita_dev + psp_dev): line F1 0.839, region exact 0.680, merged / split / cut 9 / 33 / 72; empty rate 0.864.

Round 1's arena column and its C == twin check come from the runtime's early, bit-exact version (`f9d4ad3`,
amendment 3). The final runtime's detector arenas are 0.9-7.7 KB larger (its NEON kernels keep channel-planar
rings; s8_c24 92,639 B, s8_c48 129,855 B, all within 131,072 B) and it returns the same lines on every development
case (`eval/textdet_round1_final_runtime_20261018.json`); rounds 2 and 3 use the final runtime throughout.

### Round 2: det+refine, the same 22 blobs (`eval/textdet_round2_dev_20261018.json`)


| run | int8 B | arena B | MACs box / screen (Vita) | est. ms box @333 (A9 model) | wd S / N / L | det+refine − GRAD, N (VN CI) | det+refine − refine, N (VN CI) | native Vita N | empty | qualifies |
|---|---:|---:|---|---:|---|---|---|---:|---:|---|
| s4_c4_seed1 | 1,004 | 72,391 | 3.1 M / 12 M | 29 | 9.91 / 9.61 / 9.28 | +2.75 (-1.75 to +6.75) | +4.83 (+2.28 to +7.65) | 2.01 | 0.48 | no |
| s4_c4_seed2 | 1,004 | 72,391 | 3.1 M / 12 M | 29 | 9.75 / 9.34 / 8.83 | +2.48 (-1.83 to +6.38) | +4.56 (+1.76 to +7.32) | 3.21 | 0.53 | no |
| s4p_c4_seed1 | 1,100 | 72,391 | 6.3 M / 24 M | 33 | 11.91 / 11.17 / 10.51 | +4.31 (-0.87 to +9.07) | +6.39 (+2.35 to +10.10) | 1.52 | 0.48 | no |
| s4_c6_seed1 | 1,280 | 72,391 | 4.2 M / 16 M | 37 | 10.14 / 9.69 / 9.25 | +2.82 (-1.65 to +6.60) | +4.91 (+2.56 to +7.40) | 3.13 | 0.58 | no |
| s4_c8_seed1 | 1,540 | 72,391 | 5.6 M / 21 M | 41 | 8.58 / 8.22 / 7.96 | +1.36 (-2.96 to +5.47) | +3.44 (-0.22 to +6.77) | 8.61 | 0.59 | no |
| s4_c8_seed2 | 1,540 | 72,391 | 5.6 M / 21 M | 41 | 10.74 / 10.33 / 9.77 | +3.46 (-0.42 to +7.49) | +5.55 (+1.79 to +9.20) | 2.17 | 0.50 | no |
| s8_c8_seed1 | 1,884 | 72,391 | 3.0 M / 12 M | 25 | 8.98 / 8.39 / 7.60 | +1.52 (-1.97 to +4.95) | +3.61 (+0.87 to +6.23) | 4.12 | 0.70 | no |
| s8_c8_seed2 | 1,884 | 72,391 | 3.0 M / 12 M | 25 | 5.54 / 5.08 / 4.72 | -1.78 (-5.38 to +1.56) | +0.30 (-2.40 to +2.65) | 0.67 | 0.73 | no |
| s4_c8n5_seed1 | 2,228 | 73,799 | 7.9 M / 30 M | 55 | 5.14 / 4.82 / 4.46 | -2.05 (-5.66 to +1.16) | +0.04 (-3.45 to +2.81) | 1.95 | 0.71 | no |
| s4_c12_seed1 | 2,276 | 77,967 | 9.8 M / 37 M | 63 | 7.01 / 6.42 / 5.99 | -0.44 (-3.98 to +2.67) | +1.64 (-2.28 to +5.02) | 4.01 | 0.64 | no |
| s4p_c8_seed1 | 2,700 | 72,391 | 18.7 M / 72 M | 70 | 5.72 / 5.30 / 4.99 | -1.56 (-4.99 to +1.53) | +0.52 (-2.88 to +3.31) | 0.80 | 0.67 | no |
| s4_c16_seed1 | 3,108 | 98,815 | 14.9 M / 57 M | 76 | 5.34 / 4.93 / 4.50 | -1.94 (-5.56 to +1.24) | +0.15 (-3.35 to +2.86) | 5.46 | 0.80 | no |
| s4_c16_seed2 | 3,108 | 98,815 | 14.9 M / 57 M | 76 | 6.17 / 5.78 / 5.30 | -1.09 (-5.61 to +3.00) | +1.00 (-2.52 to +3.94) | 4.41 | 0.79 | no |
| s8_c16_seed1 | 3,276 | 72,391 | 4.9 M / 19 M | 32 | 6.87 / 6.30 / 5.81 | -0.56 (-3.91 to +2.64) | +1.52 (-2.05 to +4.75) | 1.66 | 0.71 | no |
| s8_c16_seed2 | 3,276 | 72,391 | 4.9 M / 19 M | 32 | 7.26 / 6.68 / 6.26 | -0.18 (-3.87 to +3.36) | +1.90 (-1.92 to +5.31) | 1.39 | 0.82 | no |
| s4_c16n4_seed1 | 3,860 | 111,375 | 18.4 M / 70 M | 92 | 3.92 / 3.62 / 3.21 | -3.24 (-6.80 to -0.14) | -1.16 (-4.59 to +1.41) | 0.43 | 0.76 | no |
| s4_c24_seed1 | 4,412 | 103,103 | 23.0 M / 88 M | 99 | 5.98 / 5.67 / 5.24 | -1.20 (-5.12 to +2.48) | +0.89 (-2.50 to +3.66) | 9.18 | 0.70 | no |
| s8_c24_seed1 | 5,700 | 92,639 | 10.6 M / 40 M | 53 | 3.40 / 3.00 / 2.61 | -3.86 (-6.90 to -1.20) | -1.78 (-4.60 to +0.35) | 0.29 | 0.86 | no |
| s8_c32_seed1 | 7,924 | 111,919 | 14.3 M / 54 M | 63 | 4.01 / 3.72 / 3.18 | -3.15 (-6.58 to -0.08) | -1.06 (-4.03 to +1.25) | 0.91 | 0.94 | no |
| s8_c32_seed2 | 7,924 | 111,919 | 14.3 M / 54 M | 63 | 4.12 / 3.72 / 3.33 | -3.14 (-6.59 to -0.11) | -1.06 (-4.05 to +1.19) | 0.24 | 0.86 | no |
| s8_c32n4_seed1 | 9,876 | 116,063 | 17.1 M / 65 M | 72 | 3.82 / 3.56 / 3.11 | -3.30 (-6.84 to -0.04) | -1.22 (-4.40 to +1.36) | 0.32 | 0.85 | no |
| s8_c48_seed1 | 14,188 | 129,855 | 25.7 M / 98 M | 92 | 4.07 / 3.72 / 3.39 | -3.14 (-6.50 to -0.16) | -1.06 (-4.14 to +1.32) | 0.27 | 0.85 | no |

Baseline (the same units): GRAD 7.44 / 6.87 / 6.20, GRAD + refine 5.21 / 4.78 / 4.46 (S / N / L).

Rule: winner None; report candidate s8_c24_seed1; qualifying []; Pareto front (bytes vs mean wd CER) ['s4_c4_seed2', 's4_c8_seed1', 's8_c8_seed2', 's4_c8n5_seed1', 's4_c16n4_seed1', 's8_c24_seed1'].

s8_c24_seed1, all three models:

| model | GRAD | GRAD + refine | det+refine | − GRAD (VN CI) | − refine (VN CI) | native Vita GRAD / refine / det+refine | native − refine (VN CI) |
|---|---:|---:|---:|---|---|---|---|
| VOCR5-S | 7.44 | 5.21 | 3.40 | -4.05 (-7.13 to -1.35) | -1.82 (-4.66 to +0.39) | 0.83 / 0.91 / 0.54 | -0.37 (-1.26 to +0.49) |
| VOCR5-N | 6.87 | 4.78 | 3.00 | -3.86 (-6.90 to -1.20) | -1.78 (-4.60 to +0.35) | 1.07 / 1.02 / 0.29 | -0.72 (-1.71 to +0.23) |
| VOCR5-L | 6.20 | 4.46 | 2.61 | -3.60 (-6.68 to -0.90) | -1.85 (-4.42 to +0.08) | 0.94 / 0.80 / 0.21 | -0.59 (-1.51 to +0.31) |

linefind development (vita_dev + psp_dev): line F1 0.839, region exact 0.680, merged / split / cut 3 / 25 / 69; empty rate 0.864.

### Round 3: the strict kernel rule, det+refine (12 runs, `eval/textdet_round3_dev_20261018.json`)


| run | int8 B | arena B | MACs box / screen (Vita) | est. ms box @333 (A9 model) | wd S / N / L | det+refine − GRAD, N (VN CI) | det+refine − refine, N (VN CI) | native Vita N | empty | qualifies |
|---|---:|---:|---|---:|---|---|---|---:|---:|---|
| s8_c8_seed3 | 1,884 | 72,391 | 3.0 M / 12 M | 25 | 9.56 / 9.07 / 8.51 | +2.21 (-1.52 to +5.99) | +4.29 (+0.80 to +7.49) | 10.38 | 0.77 | no |
| s4_c8n5_seed3 | 2,228 | 73,799 | 7.9 M / 30 M | 55 | 6.90 / 6.45 / 5.85 | -0.42 (-3.13 to +2.12) | +1.67 (-1.08 to +4.10) | 0.83 | 0.70 | no |
| s4_c12n4_seed3 | 2,808 | 87,399 | 11.9 M / 46 M | 74 | 5.61 / 5.27 / 4.65 | -1.60 (-5.16 to +1.68) | +0.49 (-2.87 to +3.28) | 1.74 | 0.76 | no |
| s8_c16_seed3 | 3,276 | 72,391 | 4.9 M / 19 M | 32 | 6.33 / 5.79 / 5.33 | -1.08 (-4.24 to +1.87) | +1.01 (-2.04 to +3.68) | 0.62 | 0.80 | no |
| s4_c16n4_seed3 | 3,860 | 111,375 | 18.4 M / 70 M | 92 | 4.74 / 4.37 / 3.97 | -2.49 (-6.58 to +1.07) | -0.41 (-3.54 to +2.07) | 3.34 | 0.80 | no |
| s8_c16n4_seed3 | 4,028 | 72,391 | 5.8 M / 22 M | 36 | 5.32 / 4.89 / 4.40 | -1.97 (-5.64 to +1.27) | +0.11 (-2.40 to +2.31) | 0.21 | 0.85 | no |
| s4_c20n4_seed3 | 4,732 | 129,175 | 23.7 M / 90 M | 112 | 4.38 / 4.14 / 3.80 | -2.73 (-6.27 to +0.39) | -0.64 (-4.03 to +1.91) | 2.03 | 0.73 | no |
| s8_c24_seed3 | 5,700 | 92,639 | 10.6 M / 40 M | 53 | 4.60 / 4.12 / 3.73 | -2.75 (-6.25 to +0.43) | -0.66 (-3.69 to +1.72) | 0.40 | 0.80 | no |
| s8_c24_seed4 | 5,700 | 92,639 | 10.6 M / 40 M | 53 | 3.67 / 3.39 / 3.12 | -3.48 (-6.97 to -0.39) | -1.39 (-4.47 to +0.87) | 0.19 | 0.92 | no |
| s4_c24n4_seed3 | 5,700 | 121,919 | 29.8 M / 114 M | 124 | 6.91 / 6.42 / 5.84 | -0.44 (-4.56 to +3.40) | +1.64 (-2.05 to +4.84) | 5.91 | 0.77 | no |
| s8_c24n4_seed3 | 6,988 | 109,295 | 12.3 M / 47 M | 59 | 3.67 / 3.49 / 3.02 | -3.37 (-6.68 to -0.45) | -1.29 (-4.19 to +0.90) | 0.32 | 0.89 | no |
| s8_c32_seed3 | 7,924 | 111,919 | 14.3 M / 54 M | 63 | 4.07 / 3.81 / 3.50 | -3.05 (-6.66 to +0.11) | -0.97 (-3.81 to +1.14) | 0.40 | 0.82 | no |

Baseline (the same units): GRAD 7.44 / 6.87 / 6.20, GRAD + refine 5.21 / 4.78 / 4.46 (S / N / L).

Rule: winner None; report candidate s8_c24_seed4; qualifying []; Pareto front (bytes vs mean wd CER) ['s8_c8_seed3', 's4_c8n5_seed3', 's4_c12n4_seed3', 's4_c16n4_seed3', 's4_c20n4_seed3', 's8_c24_seed4'].

s8_c24_seed4, all three models:

| model | GRAD | GRAD + refine | det+refine | − GRAD (VN CI) | − refine (VN CI) | native Vita GRAD / refine / det+refine | native − refine (VN CI) |
|---|---:|---:|---:|---|---|---|---|
| VOCR5-S | 7.44 | 5.21 | 3.67 | -3.77 (-7.46 to -0.55) | -1.54 (-4.72 to +0.79) | 0.83 / 0.91 / 0.29 | -0.62 (-1.42 to +0.19) |
| VOCR5-N | 6.87 | 4.78 | 3.39 | -3.48 (-6.97 to -0.39) | -1.39 (-4.47 to +0.87) | 1.07 / 1.02 / 0.19 | -0.83 (-1.75 to +0.13) |
| VOCR5-L | 6.20 | 4.46 | 3.12 | -3.08 (-6.49 to -0.06) | -1.34 (-4.20 to +0.83) | 0.94 / 0.80 / 0.16 | -0.64 (-1.44 to +0.11) |

linefind development (vita_dev + psp_dev): line F1 0.834, region exact 0.661, merged / split / cut 5 / 21 / 70; empty rate 0.924.

The arenas in rounds 2 and 3 are the det+refine arenas (the larger of the detector's and the refinement's, 72,391 B
for the refinement at 960×544). The "est. ms" column is the A9 cycle model's high bound × 1.0007 at 333 MHz for the
network and decode of a 900×150 box (`eval/textdet_a9_20261018.json`, `eval/textdet_a9_round3_20261018.json`;
random blobs of each shape, which cost what trained ones do); the refinement adds about 0.7-1.4 ms per line
(`docs/finder-tighten.md`).

What the rounds show:

- **Size.** Below about 2 KB (stride 4 with 4-8 channels, stride 8 with 8) the detector misses and merges lines:
  every such run is worse than GRAD + refine, most of them worse than GRAD. From about 4 KB (stride 4 with four
  16-channel blocks, stride 8 with 24-48 channels) some runs are resolved better than GRAD (det+refine): `s8_c24` for
  all three models in rounds 2 and 3 (one seed of two in round 3), and `s8_c32`, `s8_c32n4`, `s8_c48`, `s8_c24n4` and
  `s4_c16n4` for some or all models.
  Above 6 KB nothing more is gained: the front ends at `s8_c24` (5,700 B) in every round. Seeds of one shape differ
  by up to 5 points (`s8_c8`: 12.6 and 7.7% in round 1), more than neighbouring shapes do.
- **det+refine** lowered the mean development CER of every round-1 blob (`s8_c24_seed1` by 0.7-0.8 points): the
  refinement's SPLIT repairs boxes the detector merged on lines set close together, and TIGHTEN its loose rows.
- **The strict kernel rule** (round 3) did not help: of the six round-1 shapes it retrained, five read the same or
  worse (VOCR5-N, `s8_c24`: 4.12 and 3.39% for seeds 3 and 4, against round 2's 3.00% for seed 1); only `s8_c16`
  read better (5.79% against 6.30 and 6.68%). Round 2's `s8_c24_seed1` has the
  lowest development CER of all 34 runs (3.40 / 3.00 / 2.61%); as amendment 4 pre-registered, the protected sets were
  read for round 3's report candidate, `s8_c24_seed4`.
- **Where it gains and loses** (`s8_c24_seed4`, det+refine against GRAD + refine, development): in the 1,072 of 1,161
  units where it returns as many lines as the reference holds it makes about 590-640 fewer errors (tighter crops);
  in the 11 units where it returns fewer (lines set close together merged or missed: PSP-sized NVL pages, two-line
  boxes at about 1.1 line heights' pitch) and the 78 where it returns more (splits, stray boxes) it makes about 230-250
  more. The +4 px variants gain least. By platform, without v27449's 12 Vita and 12 PSP units: on the Vita frames
  det+refine − GRAD + refine is −1.38 / −1.40 / −1.28 points (VOCR5-S / N / L; VN intervals end at −0.28 / −0.22 /
  −0.10), on the PSP frames +0.57 / +0.80 / +0.62 (intervals include 0); round 2's `s8_c24_seed1` and `s8_c32_seed1`
  and round 3's `s8_c24n4_seed3` show the same pattern (VOCR5-N: Vita −1.1 to −1.6, PSP −0.4 to +1.2). With v27449,
  which holds 414 of the PSP frames' characters at 53% CER for GRAD + refine against 14% for det+refine (VOCR5-N),
  the PSP frames read 5.50% against 5.90%.
- **Empty regions** (66 text-free development regions): GRAD returns no line in 8% of them, every candidate from
  about 3 KB in 70-94%.

**A gate instead of a finder** (report only): GRAD + refine with every line dropped when the detector returns no
line for the region costs 0.02-0.03 points of development whole-dialogue CER with `s8_c24_seed4` (3 of the 1,515
cases, all +4 / +8 px variants of two regions, lose their lines; VOCR5-N 4.78 → 4.80%) and takes the detector's
empty-region rate (61 of the 66 text-free regions return nothing, against GRAD's 5).

**CHAOS;CHILD, native Vita console captures** (report only, run after the protected read;
`eval/textdet_own_dlg_20261018.json`): the owner's v14018 frames as whole dialogue (finder-tighten's own_dlg Vita
units, 145 regions at base / +4 / +8 px, 13,866 characters). GRAD 0.28 / 0.37 / 0.24%, GRAD + refine 0.30 / 0.41 /
0.24%, det+refine 0.48 / 0.48 / 0.38% (VOCR5-S / N / L): +0.18 / +0.07 / +0.14 points against GRAD + refine, region
intervals from about −0.1 to +0.4. The same direction as CLANNAD: on these two titles' clean native console text the
GRAD crops already read near-perfectly and the detector's are a little worse, while the development native-Vita
subset (150 units, 9 VNs) reads the other way (det+refine 0.29 / 0.19 / 0.16% against refine's 0.91 / 1.02 / 0.80%,
intervals include 0). The recognizers trained on these frames; the detector did
not (only three VNDB PC screenshots of v14018 are in its synthetic scene pool).

**int4 weights** (report only, `eval/textdet_int4_20261018.json`): limiting the trained weights to ±7 codes without
retraining costs +1.11, +0.49 and +1.90 points of development whole-dialogue CER (VOCR5-N, det+refine) for
`s8_c24_seed4`, `s8_c24_seed1` and `s4_c16n4_seed1`. Packed, it would save 1,636 of `s8_c24`'s 5,700 B; the blob is
3,272 B of int8 weights, 1,980 B of per-channel bias, multiplier and shift, and 448 B of headers.

### The protected sets, once (`eval/textdet_protected_20261018.json`, `eval/textdet_clannad_20261018.json`)

After the round-3 record was committed and pushed (`572d395`, 2026-10-18T02:12:50Z; tag
`textdet-round3-record-572d395`), round 3's report candidate `s8_c24_seed4` was read once with det+refine (attempt
records 2026-10-18T02:13:05Z), the C runtime checked against the twin on every protected region. The branch was later
rebased onto main, which rewrote its commit ids; the original commits are kept at the tags `textdet-protocol-e32ade6`,
`textdet-round3-record-572d395` and `textdet-history-20261018` (the same final tree).

**vita-test + psp-test whole dialogue** (eval's 175 regions, 2 vita-test regions with an uncertain line left out, at
base / +4 / +8 px; body = dialogue text; VN-cluster intervals):

| model | GRAD | GRAD + refine | det+refine | − GRAD (VN CI) | − refine (VN CI) | psp-test − refine (VN CI) | vita-test − refine (VN CI) |
|---|---:|---:|---:|---|---|---|---|
| VOCR5-S | 10.53 | 9.51 | 6.28 | −4.25 (−9.41 to −0.13) | −3.23 (−7.58 to +0.35) | −6.24 (−13.70 to −0.29) | −0.26 (−4.46 to +4.05) |
| VOCR5-N | 10.23 | 9.16 | 5.99 | −4.24 (−9.22 to −0.21) | −3.17 (−7.08 to +0.18) | −5.95 (−12.50 to −0.69) | −0.43 (−4.84 to +4.07) |
| VOCR5-L | 9.42 | 8.35 | 5.62 | −3.80 (−8.69 to −0.01) | −2.73 (−6.51 to +0.44) | −5.59 (−11.97 to −0.40) | +0.09 (−3.95 to +4.08) |

With speaker names (all text) det+refine − refine is −4.53 / −4.29 / −3.88 points (VN intervals below 0). Detection
against eval's line boxes, pooled: line recall 0.90 (GRAD 0.68, refine 0.73), precision 0.71 (0.69, 0.73), F1 0.80
(0.68, 0.73), no merged line (GRAD 37, refine 16).

**CLANNAD long run** (owner-drive-ingest's frozen references, `refs_sha256` `35411688…`; each reference line reads
the detector's lines that lie mostly inside its GRAD box: a line goes to the GRAD box it overlaps most when that box
holds at least half its area, stricter than the protocol's "most of its height", see "What changed after the
freeze"; whole-page intervals; as scored):

| model | dialogue GRAD | refine | det+refine | − GRAD (page CI) | − refine (page CI) | all lines − refine (page CI) |
|---|---:|---:|---:|---|---|---|
| VOCR5-S | 1.38 | 1.43 | 1.54 | +0.16 (−0.01 to +0.33) | +0.10 (−0.06 to +0.28) | −0.26 (−0.49 to −0.02) |
| VOCR5-N | 0.96 | 0.92 | 1.15 | +0.19 (+0.04 to +0.35) | +0.23 (+0.07 to +0.39) | +0.62 (+0.42 to +0.83) |
| VOCR5-L | 0.73 | 0.63 | 1.02 | +0.29 (+0.14 to +0.45) | +0.39 (+0.25 to +0.55) | +0.78 (+0.61 to +0.97) |

On the 133 CLANNAD pages none of whose lines the references score (lines excluded as no text, cut by the ROI, UI or
unreadable) the detector returned 373 lines and GRAD 161; whether those pages show text was not checked. Every
VOCR5 model trained on the owner's planetarian PSP frames (Key / Prototype, like CLANNAD's Vita port); the detector
trained on none of the owner's CLANNAD frames (CLANNAD's family is excluded from every pool) but on other owner
frames, including planetarian's.

### Console time and memory (host estimates)

| `s8_c24` | 900×150 box | 450×80 PSP box | 960×544 screen | 480×272 screen |
|---|---:|---:|---:|---:|
| MACs | 10.6 M | 2.8 M | 40.4 M | 10.1 M |
| MAC-rate estimate @333 / @444 MHz (VOCR3-H15's console rate) | 34-35 / 25-26 ms | 9 / 7 ms | 129-134 / 97-101 ms | 32-34 / 24-25 ms |
| A9 cycle model @333 / @444 MHz (high bound × 1.0007) | 53 / 40 ms | 15 / 11 ms | 191 / 143 ms | 50 / 37 ms |

Plus the refinement, about 0.7-1.4 ms per line at 333 MHz. For comparison, the GRAD finder's measured console
compute was 23-24 ms for an 883×112 region (v0.8.0-test.1, `docs/finder-tighten.md`); scaled by pixels, the cycle
model puts the detector at about 39 ms for that size (about 1.6×), inside the 0.1 s the tier estimates give the
finder. Memory: 5,700 B of weights (resident with the
recognizer's), about 9 KB of code, and an arena of 92,639 B at 960×544, which the job workspace already holds
(VOCR5's W8I arenas are 124,832 / 182,944 / 261,024 B and the finder runs before the recognizer). Both A9 methods
are estimates: nothing of the detector has run on a console.

### Measured and estimated

Measured on the host: every CER, interval, detection metric and empty rate above (the C runtimes), the blob and
arena sizes (exact), the code sizes and stack frames (compiler output). Estimated: every console time (the MAC-rate
method calibrated to VOCR3-H15's 5.0-5.2 ms per CTC frame, and the qemu-arm cycle model calibrated by ocr-speed); the
console memory is the arithmetic above.

### Limits

- The development sample is 93 VNs whose errors are concentrated (10 VNs hold 80% of GRAD's); intervals over VN
  clusters are wide, which is what kept every candidate from qualifying against GRAD + refine.
- Three rounds were run on that sample; round 3's report candidate's development intervals are therefore
  optimistic. The protected sets, read once, agree in direction for vita-test + psp-test and disagree for CLANNAD.
- The platform evidence conflicts: psp-test's gain is all PSP, the development data's (without v27449) all Vita.
- CLANNAD was scored with a line-to-reference rule stricter than the registered one (area, not height); its
  figures stand as scored and cannot be re-scored without a second protected attempt.
- The synthetic scene pool holds most eval/linefind screenshots (the empty-region and linefind rows carry that
  caveat) and three PC screenshots of v14018.
- The training labels of real screenshots are MeikiOCR's boxes refined to ink, not hand-checked; synthetic scenes
  carry some unlabelled scene text.
- Vertical text: the detector flags vertical boxes (they are never read), but no vertical text is in any scored set.
- Nothing ran on a console; the time and memory figures are estimates.

### Reproduce

```sh
H=$HERD; T=$H/shared/artifacts/textdet; P=$H/shared/private/textdet; V5=$H/shared/artifacts/vocr5-train
.venv/bin/python synth/textdet_data.py synth|real ...                    # shards, pinned by data/manifests/textdet_shards_v1.json
eval/reproducibility/textdet-20261017/sweep.sh $T/sweep                  # round 1 (22 runs); sweep_r3.sh $T/sweep_r3 (round 3)
eval/reproducibility/textdet-20261017/evaluate.sh $T/sweep $T/eval CBIN  # round 1; evaluate_r2.sh / evaluate_r3.sh for rounds 2-3
.venv/bin/python eval/textdet.py select --runs-from 4 ...                # the rule (round 3)
.venv/bin/python eval/textdet_protected.py tests|clannad --refine ...    # once (attempt records refuse a second run)
.venv-ci/bin/python eval/textdet_tables.py round eval/textdet_round3_dev_20261018.json --a9 ...   # the tables above
# the report-only records (eval/textdet_reports.py: own-dlg, int4, round1-final, empty-pool):
.venv-ci/bin/python eval/finder_tighten.py score --cohorts $P/own_dlg/cohort --cohort own_dlg --arm VOCR5-N=... --out $P/own_dlg/base/VOCR5-N
.venv/bin/python eval/textdet.py detect --refine --names own_dlg --cand s8_c24_seed4=... --cohorts $P/own_dlg/cohort --out $P/own_dlg/dets --cbin CBIN
.venv/bin/python eval/textdet.py read --names own_dlg --model VOCR5-N=... --dets $P/own_dlg/dets --cohorts $P/own_dlg/cohort --out $P/own_dlg/units
.venv/bin/python eval/textdet_reports.py own-dlg --private $P/own_dlg --out eval/textdet_own_dlg_20261018.json
.venv/bin/python eval/textdet_reports.py int4|round1-final|empty-pool --artifacts $T ... --out eval/textdet_<record>_20261018.json
```

`$P/own_dlg/cohort/own_dlg.pkl` is finder-tighten's `own_dlg.pkl` restricted to its cases of kind `vita` (all
v14018); the int4 blobs are `train/textdet_export.py --ckpt RUN/final.pt --wbits 4` of the three runs, scored with
`detect --refine` and `read` (VOCR5-N) on the wd cohorts.
