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
arm. Results are added under "Results"; nothing above it changes except by a dated amendment in the protocol.

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
