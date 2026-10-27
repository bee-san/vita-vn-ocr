# textspot7: textspot6's spotters with a GRAD fallback and decode fixes, and a fresh confirmation set (issue #128)

textspot6 (#128, `docs/textspot6.md`) trained tagged spotters on the owner's hooked pages. On average they read
unseen-game dialogue far better than Rubi's finder through Rubi's page path (unseen dev 1.39-2.07% against GRAD +
refine's 3.08%, and `rubi,fix=1`'s 2.38%), but its rule selected none: no unseen-dev VN-cluster interval lies below
0. One VN, v27449, holds 40% of GRAD + refine's errors there, and a few VNs that GRAD reads perfectly lose.

textspot7 trains nothing. It takes textspot6's trained spotters (each arm's median seed) and adds:
- **F, a per-region fallback** to `rubi,fix=1`: run both, and show the spotter's lines unless the spotter is unsure or
  its line structure shows one of its known failures;
- **D, a decode post-pass** on the spotter's boxes (region assignment, fragment joining, edge extension over ink,
  vertical tightening);
- **F+D**.

They are decided by textspot6's rule unchanged, plus one condition: not resolved worse than `rubi,fix=1` on a fresh
confirmation set of unseen games, because the losing VNs were found on the decision set itself.

This document starts as the pre-registered protocol (`eval/textspot7_protocol.json`), frozen before any read of a new
arm. The commits named in it and in the records (the protocol ae1dd78, the fits 359437d and a2fa039, the dev record
f62c2e8, the units' freeze 3781f2f, the confirmation record 91a39a0, the release target a5b7d34) are on the branch's
original history, kept as tag `textspot7-history-20261027a`; the branch was rebased onto main after #147 merged. Results are added under "Results"; nothing above it changes except by a dated amendment in the protocol.

## Diagnosis (from textspot6's stored dev rows, before the protocol)

Per VN, what each finder does on the units that decide the dev interval. Counts are page-path dialogue errors of the
VN's units (3 variants per region). `fix1` is `rubi,fix=1`; the four det representatives are `base_t64n8c16` seed 1,
`kd_t64n8c16` seed 2, `base_qS4` seed 2 and `base_qL5` seed 1.

| VN | chars | GRAD + refine | fix1 | det representatives | what happens |
|---|---:|---:|---:|---|---|
| v27449 | 837 | 311 | 311 | 24-33 | GRAD reads 2 lines as one box in both regions of one screenshot (a translucent box over a busy scene); every spotter boxes them apart |
| v1971 | 273 | 28 | 28 | 54-56 | all of it is one region of two side-by-side choices: the reference lists the right choice first (its ink box is 1 px higher); every spotter reads both choices whole and correct, left to right; GRAD reads only the right one |
| v20524 | 192 | 0 | 0 | 30-45 | one region (a dark-on-light UI-style line): the spotters give low-confidence fragments (conf 4-14 of 127) or nothing |
| v8410 | 300 | 1 | 1 | 24 | one region: every spotter's box joins this line with text to its left, so the box's centre falls outside the region and the region gets no box |
| v3496 | 87 | 0 | 0 | 6-21 | one region of 2 tightly spaced lines: three of the four put both lines in one box |
| v20170, v454 | 366, 114 | 3, 0 | 3, 0 | 0-12, 0-1 | losses of the crop decodes (a line cut into two boxes or cut short); the det decodes read them like GRAD |

- The **v1971 loss is the reference's order**, not a reading error. No arm is aimed at it; its sensitivity (the two
  choices scored in either order) is a report row.
- **hval** (the fitting set; existing arms only): `rubi,fix=1` makes 5,349 errors of 162,471 characters, 2,581 of them
  in PCSG00854 (GRAD's junk rows at the region's top edge) and 677 in ULJM05674 (merged rows). The det
  representatives make 2,116-2,552. Their own failures there are mostly lines cut at an edge (1-2 characters, a
  median 0.4-0.6 line heights beyond the box), then merged lines (`base_qS4`). The best per-unit choice between the
  two would make 1,415-1,650 errors.

## Protocol

### Arms (no training)

The spotters are textspot6's representatives in the det decode: the spotter runs on the whole frame in the console
view, and a region keeps the horizontal boxes whose centre lies in it, clipped to it.

**F: spotter + GRAD fallback.** Per region, run both `rubi,fix=1` (G) and the spotter (S), and show one of them:
- Mode S (the spotter by default): show G when S is empty and G is not, or when S's median box confidence is below
  τ, or (λ on) when G has a line that S lacks (a G box at least λ times S's median line height, whose rows the S
  boxes overlapping it cover less than half of: S missed or merged a line), or (κ on) when a G line reaches beyond
  the S boxes of its row by more than κ of its height at either end (S cut it short). Else show S.
- Mode G (GRAD by default): show S when G is empty and S is not, or when one G box holds the centres of S boxes of 2
  or more rows (GRAD merged lines) and S's median confidence is at least τ. Else show G.
- The grid is mode {S, G} × τ {0, 8, 16, 24, 32, 48, 64, 80, 96} × λ {0.6, 0.75, 0.9, off} × κ {0.5, 1.0, off}. It is
  fit per representative on hval only: the fewest hval page-path errors, ties to mode S, then the larger τ, then the
  smaller λ and κ (off last). F shows crops that G's and S's stored runs already read, so the fit needs no new
  reading; the fitted F is then run through the page path, which must reproduce it unit for unit.
- Amendment 1 (before any read of a new arm) made the coverage test a union of the S boxes (as first written, it
  would have sent v27449 to GRAD), added κ, and fixed every definition in integers for the C twin.
- **Not in amendment 1's text** (audit N2): the twin and the C count an S box in S_short's row only when it also
  overlaps the G box horizontally. With amendment 1's literal definition, 68-72 of the 117 grid points' hval errors
  change per representative, but all four fits stay the same (and F is ineligible anyway).

**D: decode post-pass on the spotter's boxes** (no GRAD), steps in this order:
- *assign*: a frame box also belongs to the region when its part inside the region is at least 0.6 of its height tall
  and one height wide, clipped to the region (for v8410's kind of box);
- *join(γ)*: boxes of one row with a gap of at most γ heights become one box;
- *extend*: each box grows left and right over inked columns of its own rows (the page path's ink threshold), at most
  1.5 heights, ending at a gap over 0.5 heights, the region's edge or another box;
- *tighten*: `VOCR_REFINE_TIGHTEN` alone (the box's rows cut to its text band plus 16%).

It is fit on hval only, pooled over the four representatives: each step alone (assign, join 0.5, join 1.0, extend,
tighten) is kept when it lowers the pooled hval errors; D is every kept step, the same for every representative.

**F+D:** F's fallback over D's boxes, F re-fit on D's hval rows, as one confirmation pair after the single arms.

The candidates are every representative × {D, F, F+D}.

### The rule: textspot6's, unchanged

- **Eligible:** VTSP ≤ 512,000 B; est. ≤ 300 ms at 333 MHz per 960×544 frame *including the fallback's cost*; arena
  ≤ 262,144 B; the C code equal to its twin on every development unit.
  - F's cost is the spotter's A9 row of record plus `rubi,fix=1` on the region: the A9 cycle model's high bound
    (textspot6's calibration) on the largest of #119's 387 base regions. The mean region and a full frame are
    reported beside it. The arena is the largest of the parts, which run one after another.
- **Beats GRAD + refine:** unseen dev (92 VNs; v12188, a hooked title, left out) page-path dialogue CER difference
  with a 95% VN-cluster interval entirely below 0, and hval not resolved above 0.
- **Confirmation:** not resolved worse than `rubi,fix=1` on the confirmation set (VN clusters), read once for every
  qualifier together after the dev decision is recorded.
- **Pick:** the cheapest qualifier that meets the confirmation condition (bytes, then est. ms, then arena, then D
  before F before F+D, then the name); a costlier one replaces it when resolved better on unseen dev and not resolved
  worse on hval.
- **VOCR6 guard:** if VOCR6-N is released before the decision, the pick read with it (as Rubi ships it) must not be
  resolved worse than `rubi,fix=1` on unseen dev or hval.
- **Rubi's default** only if, against `rubi,fix=1`, unseen dev is resolved below 0 and hval not above 0; otherwise the
  pick ships behind its setting, off by default.
- **With a pick:** retrain its spotter on train + guard + hval (same recipe and seed), release it with SHA256SUMS,
  read the test sets once (report only), and open a normal Rubi PR (the fallback in C, an off-switch, the memory
  ledger, both Vita builds, the fast gate). **No qualifier:** a negative verdict with the per-VN table.

The page path is textspot6's pinned one (`eval/pagepath.py` at #135, VOCR5-N W8I with its top 8, the counter
re-decode, beam 8 + the 1 MB LM, the cascade to VOCR5-L, Rubi's text rules), so every number pairs with textspot6's.

### The confirmation set

- **Exclusions** (VN ids, agreed with vocr6-spotter2, which trains the next spotter): every hooked title, textspot2's
  real half (52,785 canvases, 11,318 VNs), the VNs of textspot's synthetic scene backgrounds, #119's units and the dev
  sample with its release partners, textspot's development frames, the test sets with their release partners and
  capture sessions, the owner's screenshot VNs and eval/linefind's sets (where GRAD was tuned): 11,677 VNs.
- **Frames** (`data/manifests/textspot7_confirm_frames_v1.json`, frozen before any new arm is read): 454 frames of
  50 VNs (PSP 42 VNs, Vita 8): native-size VNDB screenshots of 45 VNs and the fresh native textbox frames (press and
  blog screenshots) of 5 VNs. URLs and sha256 only.
- **Units:** the whole-dialogue regions of up to 6 accepted frames per VN in the frozen review order (dialogue,
  narration and in-story choices; speaker names left out), each region the union of its lines' ink boxes plus 12 px
  (Vita) / 6 px (PSP), with the +4 / +8 variants.
- **Labels:** PP-OCRv6 small and MeikiOCR proposals, then Claude adjudication on contact sheets, one frame per sheet
  (`reviewer_kind` assistant). The reading order is rows top to bottom and left to right within a row, fixed now. The
  reviewers never see an arm's output. The reference text stays private; the units' sha256 is committed before the
  read.
- **Read:** the console view (PSP upscaled 2x), the spotter on the whole frame, every finder on the unit's region, the
  same page path, once.

## Results

Every number is measured on the host unless it is labelled an estimate. Page-path CER is dialogue CER through
textspot6's pinned page path. Intervals are 95% paired cluster bootstraps: VNs on dev and on the confirmation set,
titles on hval. Dev is the unseen dev set without v12188 (92 VNs, 1,143 units); hval is 6,327 units of 46 titles. The
records are:
- `eval/textspot7_fit_20261027.json`: the fits of F and D, both on hval only;
- `eval/textspot7_a9_20261027.json`: the A9 rows;
- `eval/textspot7_dev_20261027.json`: the selection's dev phase, committed before the confirmation set was read.

### The fits (hval only)

- **F barely fires on hval.** Its fitted parameters fall back on 0-4 of 6,327 units and change the four arms' hval
  errors from 2,116 / 2,164 / 2,552 / 2,121 to 2,106 / 2,164 / 2,552 / 2,121. On their own training titles the
  spotters rarely fail in the ways F detects; F's structure tests (λ, κ) cost errors there wherever GRAD has extra
  rows. The fit keeps the always-on rule "the spotter found nothing and GRAD found a line → GRAD", τ 8 for
  `base_t64n8c16` and `base_qL5`, and λ 0.9 for `kd_t64n8c16`, `base_qS4` and `base_qL5`.
- **D = join10, then extend.** Pooled hval errors of the four arms, each step alone:

  | plain | assign | join5 | join10 | extend | tighten |
  |---:|---:|---:|---:|---:|---:|
  | 8,953 | 8,953 | 8,713 | 8,641 | 8,361 | 9,513 |

  The plain decode through `--boxes` reproduces textspot6's stored det rows unit for unit. Combined, D reads hval at
  1.000 / 1.165 / 1.506 / 1.097% (t64 / kd / qS4 / qL5). F+D's fit on D's hval rows is D itself (it falls back on no
  hval unit beyond the empty-spotter rule).

### Cost (estimates)

A9 cycle model, high bound × 1.0007 at 333 MHz (textspot6's calibration):

| part | est. ms @333 |
|---|---:|
| `rubi,fix=1` (GRAD + refine with COLGAP, the probe) on the largest development region (834×220, PSP view) | 122.4 |
| ... on the mean / median development region | 17.2 / 14.7 |
| ... on a full 960×544 region (Vita / PSP title) | 230.3 / 248.5 |
| D's post-pass (join10 + extend) on the largest development regions (max of the five largest) | 31.5-33.5 |
| ... on the mean development region | 4.3 |
| F's choice | < 0.02 |

- **So F and F+D are over the 300 ms budget for every shape:** 199-264 ms of spotter + 122 ms of GRAD.
- **D fits:** t64 264.2 + 32.0 = 296.3 ms; qS4 231.3; qL5 270.4; kd 296.8.
- **Disclosed, and it decided the pick (amendment 5).** The post-pass's first C code computed each pixel's 3×3
  contrast with 9 reads (est. 95-99 ms on the largest regions). It was rewritten row-separable after the dev and hval
  reads, with their results in view, and before the dev record was committed. The output is the same (C == twin was
  re-checked on every dev, hval and confirmation unit), and the rule reads the code's cost. The final code's rows are
  the ones above; the first rows are in the A9 record.
  - With the first code, only `base_qS4` D was within the budget (199.1 + 95.5 = 294.6 ms). It qualifies and passes the
    confirmation condition, so it would have been the pick.
  - With the rewritten code, `base_t64n8c16` D (296.3 ms), `kd_t64n8c16` D and `base_qL5` D became eligible, and t64 D,
    the cheapest by bytes, is the pick: 3.7 ms under the budget, an estimate.
  - Rubi's default is GRAD either way.

### Unseen dev and hval (the dev phase of the rule)

GRAD + refine reads dev at 3.08% and hval at 3.88%; `rubi,fix=1` at 2.38% and 3.29%. Rows are page-path CER in %.

| candidate | est. ms @333 | dev | dev − GRAD+refine | hval | hval − GRAD+refine | dev − `rubi,fix=1` | qualifies |
|---|---:|---:|---|---:|---|---|---|
| `base_t64n8c16` det (textspot6) | 264.2 | 1.41 | −1.68 [−4.31, +0.12] | 1.30 | −2.57 [−6.23, −0.54] | −0.97 [−3.59, +0.71] | no |
| **`base_t64n8c16` D** | 296.3 | **1.18** | **−1.91 [−4.55, −0.12]** | **1.00** | −2.88 [−6.59, −0.81] | −1.20 [−3.83, +0.47] | **yes** |
| `base_t64n8c16` F | 386.7 | 1.10 | −1.98 [−4.59, −0.26] | 1.30 | −2.58 [−6.24, −0.54] | −1.28 [−3.87, +0.32] | no (time) |
| `base_t64n8c16` F+D | 418.7 | 1.01 | −2.07 [−4.70, −0.31] | 1.00 | −2.88 [−6.59, −0.81] | −1.36 [−3.99, +0.27] | no (time) |
| `kd_t64n8c16` D | 296.8 | 1.40 | −1.69 [−4.25, +0.14] | 1.17 | −2.71 [−6.29, −0.77] | −0.98 [−3.55, +0.75] | no |
| `kd_t64n8c16` F+D | 419.2 | 1.03 | −2.06 [−4.58, −0.38] | 1.17 | −2.71 [−6.29, −0.77] | −1.35 [−3.87, +0.20] | no (time) |
| **`base_qS4` D** | 231.3 | **1.14** | **−1.94 [−4.52, −0.17]** | 1.51 | −2.37 [−5.71, −0.63] | −1.23 [−3.78, +0.42] | **yes** |
| `base_qS4` F+D | 353.7 | 0.88 | −2.21 [−4.75, −0.56] | 1.51 | −2.37 [−5.71, −0.63] | −1.50 [−4.02, +0.02] | no (time) |
| **`base_qL5` D** | 270.4 | **1.16** | **−1.93 [−4.57, −0.13]** | 1.10 | −2.78 [−6.35, −0.80] | −1.22 [−3.86, +0.46] | **yes** |
| `base_qL5` F+D | 392.9 | 0.83 | −2.26 [−4.89, −0.54] | 1.10 | −2.78 [−6.35, −0.80] | −1.55 [−4.14, +0.04] | no (time) |

- **Three qualifiers:** `base_t64n8c16` D, `base_qS4` D and `base_qL5` D. Every F row is shown in
  `eval/textspot7_dev_20261027.json`.
- **None is resolved against `rubi,fix=1`** on dev. So, by the rule, even a pick does not become Rubi's default
  finder.

### The diagnosed VNs under each arm (dev errors per VN, all three variants)

| VN | GRAD + refine | `rubi,fix=1` | t64 det | t64 D | t64 F | t64 F+D | qS4 D | qL5 D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v27449 | 311 | 311 | 24 | 22 | 24 | 22 | 30 | 21 |
| v1971 | 28 | 28 | 56 | 57 | 56 | 57 | 57 | 56 |
| v20524 | 0 | 0 | 36 | 21 | 0 | 21 | 24 | 21 |
| v8410 | 1 | 1 | 24 | 24 | 0 | 0 | 24 | 24 |
| v3496 | 0 | 0 | 21 | 21 | 21 | 21 | 21 | 21 |
| v20170 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| v454 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |

- **D** removes most cut lines. Over dev (without v12188), the cut lines go 30 → 6 (t64), 21 → 6 (qS4) and 42 → 6
  (qL5), and the errors 358 → 300, 375 → 291 and 383 → 295 (v20524 36 → 21). It cannot reach the GRAD-shaped losses
  (missed and merged lines stay at 6-9 and 3).
- **F** reaches v8410 (no spotter box → GRAD) and, at λ 0.9 (qS4, qL5), v3496. But F is over the time budget.
- **v1971** is the reference-order artifact: every spotter reads both choices whole and correct.
- **Without v27449** (report only), t64 D is −0.80 [−1.79, +0.12] against GRAD + refine.

### Regions drawn tight to the text (report only)

vocr6-finder's tight units: dev_tight has 1 px around the ink (381 units without v12188); hval_tight (0 px) and
hval_tight2 (2 px) have 2,109 units each. Values are page-path CER in %, and D − `rubi,fix=1` with its 95% interval.

| set | GRAD + refine | `rubi,fix=1` | t64 det → D | qS4 det → D | qL5 det → D | t64 D − `rubi,fix=1` |
|---|---:|---:|---|---|---|---|
| dev_tight | 3.16 | 2.16 | 1.69 → 1.39 | 1.78 → 1.50 | 1.86 → 1.52 | −0.77 [−3.36, +0.84] |
| hval_tight | 2.57 | 1.93 | 1.65 → 1.58 | 1.92 → 1.74 | 1.67 → 1.46 | −0.35 [−0.65, −0.14] |
| hval_tight2 | 2.24 | 1.62 | 1.30 → 0.99 | 1.56 → 1.35 | 1.30 → 1.04 | −0.63 [−1.39, −0.21] |

- The det decode reads the whole frame, so D does not depend on how the player draws the region.
- On both hval tight sets, all three qualifiers are resolved better than `rubi,fix=1` (qS4: −0.20 [−0.34, −0.07]
  and −0.27 [−0.42, −0.14]; qL5: −0.47 [−1.14, −0.08] and −0.58 [−1.28, −0.19]).

### The confirmation set

- **What it holds:** 41 unseen VNs with accepted frames (out of the 50 in the frozen frame list; the others had no
  accepted Japanese dialogue frame within the view cap). 170 frames, 172 base regions (516 units with the +4 / +8
  variants), 320 lines, 4,737 characters; PSP 154 and Vita 18 base regions. The public manifest is
  `eval/textspot7_confirm_units_v1.json` (amendment 3): ids, VNs, URLs, sha256, regions and line boxes, no text.
- **How it was labelled:** 9 Claude helpers decided 343 of the 383 sheets offered (one frame each, ≤ 6 accepted
  frames per VN, in the frozen order; amendment 3's "323" is an erratum, corrected by amendment 5). The second look
  covered the 30 accepted lines whose text matched neither engine's reading or that were added by hand: 29 confirmed,
  1 corrected.
  - **Deviation:** amendment 2 named another helper for the second look. No agent capacity was free, so this
    label's own session did it: a different Claude session than every first reviewer, which had seen no arm output
    on any confirmation frame.
- **Ordering, disclosed:** the arms' dev and hval reads (and the dev-phase record) came before the units were frozen.
  The frame list, the review order and the review rules were frozen before any arm read (amendments 1-2), the
  reviewers never saw an arm's output, and no arm ran on a confirmation frame before the freeze.
- **The read:** once (an attempt record refuses a second), for the three qualifiers and the baselines, with F and
  F+D of the same spotters as report rows. C == twin held for D on every unit.
- **Checked by the audit** (N4, N5): the thumbnail cross-check against vocr6-spotter2's training pool that this round
  had promised was not run before the freeze (only duplicates within the list were dropped). The auditor's check by
  sha256 and VNDB screenshot number over 445 manifests and textspot2's samples found no accepted frame anywhere else.
  Two confirmation VNs (v6006, v6011) share their only release; counted as one cluster, the pick's interval against
  `rubi,fix=1` is [−5.45, −0.15] instead of [−5.49, −0.18], and no decision changes.

| finder | confirmation CER | PSP / Vita | − `rubi,fix=1` | − GRAD + refine |
|---|---:|---|---|---|
| GRAD + refine (`rubi`) | 5.01 | 5.40 / 1.43 | +0.72 [+0.06, +1.92] | |
| `rubi,fix=1` | 4.29 | 4.67 / 0.79 | | −0.72 [−1.92, −0.06] |
| `base_t64n8c16` det | 2.42 | 2.56 / 1.07 | −1.87 [−5.00, +0.41] | −2.59 [−5.73, −0.11] |
| **`base_t64n8c16` D (the pick)** | **1.92** | 2.06 / 0.64 | **−2.37 [−5.49, −0.18]** | −3.09 [−6.17, −0.70] |
| `base_qS4` D | 2.81 | 3.12 / 0.00 | −1.48 [−3.21, +0.06] | −2.20 [−4.09, −0.42] |
| `base_qL5` D | 2.09 | 2.31 / 0.07 | −2.21 [−3.95, −0.78] | −2.92 [−4.81, −1.27] |
| report: `base_qL5` F+D (over budget) | 2.14 | 2.36 / 0.07 | −2.16 [−3.90, −0.75] | −2.87 [−4.74, −1.22] |

- **All three qualifiers pass the confirmation condition** (not resolved worse than `rubi,fix=1`). On these fresh
  games they are all lower, and t64 and qL5 are resolved lower.
- **One VN carries much of the gain again.** For the pick, 20 VNs get better, 6 worse and 15 are level against
  `rubi,fix=1`, but v12571 alone goes 282 → 73 errors. Without it (report only), the pick reads 1.49% against
  `rubi,fix=1`'s 2.44%: −0.95 [−1.99, +0.09], not resolved. qL5 D stays resolved without it: −1.49 [−2.56, −0.59].
- **D adds** −0.50 points to the plain det decode here (t64: 2.42 → 1.92), as it did on dev and hval.

### The decision

`eval/textspot7_confirm_20261027.json`:

- **The pick is `base_t64n8c16` D.** It is the cheapest confirmed qualifier: 52,176 B, est. 296.3 ms per 960×544
  frame at 333 MHz, arena 244,607 B. qS4 D (231 ms but 118 KB) and qL5 D (448 KB) cost more bytes, and neither is
  resolved better on dev against it.
- **Rubi's default stays GRAD (`ocr_finder = grad`).** Against `rubi,fix=1`, the pick's unseen-dev difference
  (−1.20 [−3.83, +0.47]) is not resolved, so by the rule the spotter ships behind its setting, off by default. (It is
  resolved lower than `rubi,fix=1` on the confirmation set and on hval.)
- **The VOCR6 guard is moot:** no VOCR6 recognizer was released at the decision (vocr6-lead's release was staged,
  not out).
- **Next, by the rule:** the pick's spotter is retrained on train + guard + hval (`base` recipe, t64n8c16, seed 1:
  54,771 hooked pages), then released, read once on the test sets (report only), and wired into Rubi behind
  `ocr_finder = spotter`.
- **Not in Rubi (amendment 4, vocr6-lead's ruling):** Rubi gets one spotter. vocr6-spotter2's textspot8 student (#149,
  a spotter distilled toward MeikiOCR's detector and read in the crop decode) passed its own registered default test,
  so it ships as Rubi's default finder. textspot7's pick is released in vita-vn-ocr and is not pinned in Rubi, so no
  Rubi PR is opened for it. Its runtime file `vocr_textspot7.c` stays in vita-vn-ocr. A Rubi integration branch built
  for it (`feat/vocr6-spotter3-finder`: the spotter on the whole frame, then `vocr_textspot7_post_rows`, opt-in)
  remains an unmerged reference.

### The final weights and the test sets (read once, report only)

- **Final weights:** `base_t64n8c16`'s recipe and seed retrained on train + guard + hval (54,771 pages, 30,000 steps,
  from f62c2e8): 52,176 B, sha256 `78330f96…` (`eval/textspot7_final_20261027.json`). C == twin holds on every
  development frame and crop and for D on its dev boxes. Published as the pre-release v0.3.9-textspot7.1.
- **The read:** `eval/textspot7_test.py`, once each (attempt records), after the release. Record:
  `eval/textspot7_test_20261027.json`. These weights in D beside det; the baselines as textspot6 read them.

| set | the pick (D) | det | GRAD + refine | `rubi,fix=1` | D − `rubi,fix=1` |
|---|---:|---:|---:|---:|---|
| Dance with Devils (PCSG00807; 2,055 units, page path) | **0.67** | 1.12 | 1.24 | 0.98 | −0.31 [−0.52, −0.13] |
| CLANNAD, each page's region (2,052 dialogue lines, page path) | 0.82 | 0.92 | 0.82 | 0.95 | −0.13 [−0.36, +0.04] |
| CLANNAD whole frames, dialogue (textspot3's read) | 1.06 | 1.02 | | | (textspot3 0.99) |
| vita-test + psp-test whole frames, dialogue CER | 2.15 | 2.19 | | | (textspot3 2.29) |
| vita-test + psp-test whole frames, **all-text CER** | **30.23** | 22.52 | | | (textspot3 20.57) |
| vita-test + psp-test whole frames, box F1 | 0.667 | 0.748 | | | (textspot3 0.776) |
| vita-test + psp-test whole frames, name recall (Vita / PSP) | 0.509 / 0.477 | 0.912 / 0.908 | | | (textspot3 0.930 / 0.908) |

Values are CER in % unless marked F1. Intervals are page clusters (one title, CLANNAD) and VN clusters (vita-test +
psp-test).

- **Dance with Devils:** D is resolved better than `rubi,fix=1` (the det decode alone is worse than it).
- **CLANNAD's regions:** level with both baselines.
- **Whole frames: D loses names and UI text there.** Dialogue CER is level with det and with textspot3's pick. But
  all-text CER is resolved worse than det's (+7.71 [+5.77, +9.94]) and than textspot3's (+9.66 [+6.97, +12.79]), box F1
  drops (−0.08 [−0.12, −0.05] against det), name recall roughly halves (Vita 0.912 → 0.509, PSP 0.908 → 0.477), and
  the tagged dialogue CER rises from 11.88% to 13.36%.
  - **The cause is D's edge extension (EXTEND), not its join.** On a whole frame, EXTEND grows short boxes such as
    speaker names and UI labels over the ink beside them, past the box metric's overlap threshold. No name loses its
    box to D: on a whole frame every det box lies inside a D box.
    - The re-audit (audit-vocr6-spotter3-b) measured this on the one read's stored spotter boxes alone, counts only.
      Its rule: a box covers a name when it holds half the name's reference box, and is the name's own when it is at
      most twice the name's width.
    - Of the 187 speaker-name references (the 122 scored ones and 65 in the report-only Adrenaline copies), det gives
      173 a box of their own and the join alone 175. The join alone has the same 705 boxes as D, and by the box
      metric scores better than det on whole frames (F1 0.771 against 0.748, no interval). EXTEND then loses 52
      names and 13 UI labels between the join alone and D.
    - That is not measured through OCR, so the advice below stands.
  - **Disclosed:** a first count of D's name boxes in this round re-ran D on the test frames' pixels after the one read
    (no OCR; no decision depends on it). That went beyond the one test read, so its numbers are withdrawn; the counts
    above need only the stored boxes.
  - **D is a decode for the player's region**, which is how Rubi reads (and where it was fit and decided). It is not
    for a whole frame: a scan-screen mode should read the det decode. The whole-frame rows apply D to the frame only
    to keep textspot6's harness.
- **Disclosed** (as textspot6): three training titles share a VNDB release with CLANNAD or with a vita-test VN.
