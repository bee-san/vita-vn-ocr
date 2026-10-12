# devnative: a native Vita/PSP development cohort

teacher_val, the development set train2 and synth3 select on, has 1,651 lines from 181 VNs, but
two thirds of them (1,064) are PC screenshots scaled down to console size. devnative holds only
native console frames: VNDB screenshots of PS Vita releases at 960×544 and PSP releases at
480×272. Each line is cut the way the console cuts it, and labelled by two local OCR engines and
independent Claude reviews.

It is a development cohort, for model selection only. No model was scored on it while it was
built. Labels were reviewed by an assistant (Claude Opus 5.5), not by a human.

## Result

| file (under `$HERD/shared/artifacts/devnative/`) | lines | VNs | Vita / PSP | chars | sha256 |
|---|---|---|---|---|---|
| `devnative_tv_native_ink.npz` | 607 | 48 | 163 / 444 | 6,992 | `7fb36fc291ff7213d06bf023f0e7cb6fb391a88524d411d02b43d15c1c984d32` |
| `devnative_tv_native_grad.npz` | 503 | 48 | 142 / 361 | 5,957 | `f810690caec0164f47224b29ad537efa3500f41b1cc08653a49abdf6f6f94747` |
| `devnative_strict_ink.npz` | 38 | 14 | 0 / 38 | 411 | `ab2cc6890a0da59f49e9c1b491a0a1bc4e68cb9d7715ac485674b80cf2414f51` |
| `devnative_strict_grad.npz` | 26 | 12 | 0 / 26 | 314 | `94b3dba500ef6f3297fd3e44515ec9e5004dbde85585ddbc4272ea1e78edc61e` |
| scene_vn | 0 | 0 | | | (no line survived review) |

645 lines in all (7,403 characters) from 48 VNs, 249 screenshots and 50 capture sessions; 175 of
the lines are speaker names. Ink height is 12/14/19 px (p10/50/90) on PSP and 20/23/27 px on
Vita. The plan's target was 3,000+ lines from 60+ VNs. The conductor dropped that minimum on
2026-10-12, because this is everything the rules leave on VNDB (see Supply).

Report every file separately. The conductor's decision is never to pool them, and never to
pool tv_native with teacher_val.

- **tv_native**: teacher_val's own native screenshots, re-cut through the console's path and
  relabelled here. 421 of its lines overlap a teacher_val line (crop IoU ≥ 0.3, recorded as
  `tv_id`), and all 421 have the same agreement key as teacher_val's label. That is a
  consistency check, not an independent one: MeikiOCR fed both.
- **strict**: other screenshots of the same reserved VNs, ones no training pool, scene list,
  eval set or other set contains.
- **scene_vn**: native screenshots of reserved VNs whose other screenshots were synthetic scene
  backgrounds, the background images themselves excluded. Its 3 screenshots gave 1 candidate
  line, and review dropped it.

Every devnative VN is a teacher_val VN, because the reserved list is teacher_val's 181 VNs plus
the general-VN development VNs. All 14 strict VNs also appear in tv_native. strict differs from
teacher_val by screenshot, not by game.

The GRAD files are the subset of lines for which the console finder returns a box that holds the
whole line and exactly its label (see Lines). Lines without a clean GRAD box are in the ink files
only.

### Format

The format is teacher_val's: `data`, `offsets`, `shapes`, and `labels` and `meta` as JSON. Crops
are grey integer luma (`synth/common.to_gray`) in console polarity: a crop whose text
`common.line_dark` calls dark is stored inverted, recorded as `dark` and `inverted`.

`meta` holds:
- teacher_val's fields: `id`, `vn`, `session`, `kind`, `target`, `box`, `dark`, `inverted`,
  `how`, `pol`, and `chars` (MeikiOCR's character boxes, crop-relative);
- devnative's fields: `subset`, `crop` (`ink` or `grad`), `flags`, `readers`, `support`,
  `decision`, `looks`, `reviewer_kind` (`"assistant"`), `styles`, and `tv_id`.

`eval/synth3_ablation.py`'s `load_dev` reads all four files unchanged (checked with the scorer
as committed in `b3f2757`, unchanged up to `feat/synth3` `ebb4dfb`):

```sh
.venv-ci/bin/python eval/synth3_ablation.py score --blob NAME=BLOB.vocr \
    --dev $HERD/shared/artifacts/devnative/devnative_tv_native_grad.npz --out scores_tv_grad.json
```

## Selection (`devnative/select.py`)

1. **Listing.** A fresh VNDB kana-API listing (2026-10-12) of the 190 VNs in
   `$S3/mining/exclude_dev_vns.txt` returned 1,742 screenshots. It also fetched the VNDB
   relations of CLANNAD (19) and of the 8 fresh native textbox titles (5). Requests used a
   project User-Agent, 1.5 s apart. The query and response hashes are in
   `devnative/manifests/review.json`.
2. **VN rules.** A VN is excluded when it is:
   - a test VN: `test_games.json` "all", or `is_test_vn`;
   - a fresh native textbox title (PR #49: PSP v14909 v25459 v29454, Vita v21289 v21976 v20519,
     reserves v16509 v19140) or one of their 5 relations;
   - CLANNAD (v4) or any of its 19 relations;
   - a VN of any training label pool: teacher_train, teacher_distill_train, real_train,
     real_unlabelled, realdata's or teacher's training-split screenshots, or the FL10
     native-training sources. 10 reserved VNs fall here: v1015 v11109 v12775 v1281 v13766 v2835
     v4 v5154 v5297 v7204.
3. **Screenshot rules.** A screenshot is kept only when:
   - it is a psv release at 960×544 or a psp release at 480×272;
   - its release is Japanese-only, with VNDB flags sexual < 0.3 and violence < 0.5;
   - it is not from one of eval's capture sessions or of a training-label release.
4. **Duplicate checks.** Every candidate is fetched (local mirrors first, else VNDB) and checked
   against 7,041 reference screenshots (8,319 local images). The checks are by VNDB id, file
   sha256, decoded-pixel sha256, and same frame re-encoded (thumbnail correlation ≥ 0.95 and a
   full-size grey p99 difference < 48). The references cover eval's sets, the training label
   pools, teacher_val, the scene-background lists, linefind's sets and the general-VN development
   sources.
5. **Fresh-test sources.** Candidates are also checked against the fresh-test sources' file and
   PIL-RGB pixel hashes. None of those 8 titles or their relations was in the reserved list or
   the listing, so nothing from them was downloaded or used.
6. **Other sets in tv_native.** Screenshots of linefind's sets and of the general-VN development
   sources stay in tv_native only, flagged `linefind` (94 lines) or `general_vn_dev` (86 lines).
   They are not test sets, and the conductor's tv_native definition covers all of teacher_val's
   native screenshots.

**Supply.** The listing gives 1,742 screenshots. These were excluded:

| reason | screenshots |
|---|---|
| not native (another platform, or a size other than native) | 1,207 |
| label-pool VN | 82 |
| rating | 29 |
| other-set screenshot | 25 |
| scene-background screenshot | 25 |
| language | 18 |
| CLANNAD family | 10 |

346 screenshots remain (`devnative/manifests/shots.jsonl`: URL, VNDB id, sha256, pixel sha256,
subset, flags):

| subset | PSP | Vita | VNs |
|---|---|---|---|
| strict | 77 | 0 | 26 |
| tv_native | 203 | 63 | 51 |
| scene_vn | 3 | 0 | 3 |

## Lines (`read.py`, `lines.py`, `gradcheck.py`, `edgecheck.py`)

**Reading.** Each screenshot is read with pinned engines, offline (`unshare -rn`, one heavy.sh
slot):
- MeikiOCR 0.3.5 detects and reads at native size. teacher's geometry then merges row pieces,
  refines each box to its ink and attaches furigana.
- Each line gets four readings: MeikiOCR on its own box (`M`), MeikiOCR on a wider view (`m`),
  PP-OCRv6 small on the view (`V`), and PP-OCRv6 small on the crop (`v`).

1,477 lines were read. 1,037 are candidates: 10-72 px ink, not at the screen edge, and at least
one Japanese reading. Re-running the read with the frozen code gave identical readings for all 346
screenshots.

**Ink crop.** The ink box plus furigana, widened left and right to the labelled characters'
boxes (teacher's `crop_of` rule), plus 4 px on Vita or 2 px on PSP.

**GRAD crop.** Built as follows:
1. Dialogue regions are the lines pass 1 kept (decided, unsure or cut; not UI) grouped with
   eval's rule: vertical gap ≤ 1.6 line heights, plus a 12 px (Vita) or 6 px (PSP) margin.
2. `common.find_lines_ex(region, "grad")`, the bit-exact twin of `runtime/vocr_lines.c`, runs on
   each region. Its boxes are matched one-to-one to the lines (eval's finder match, overlap ≥ 0.5).
3. A box is used only when it:
   - holds the whole ink line;
   - holds less than a quarter of any other detected text box;
   - covers the final ink crop's labelled extent.
4. A box that runs beyond the ink crop (more than max(3 px, 0.3 ink heights) sideways, or 0.6
   ink heights vertically) is used only after a contact-sheet check that it shows exactly the
   label. The detector misses some glyphs (card digits, a bracket), and those would make the label
   wrong for that crop.

Of the 668 lines pass 1 labelled, the finder found none for 53 and merged 27 with other text. Its
box cut 15 of them and was rejected after the check for 6. Of the final lines, 22 more have a GRAD
box that does not reach the widened crop's labelled extent, so they get no GRAD crop.

**Edge check.** In the blind re-check, five disagreements came from crop edges: MeikiOCR's box
had left out an end 。, 」 or 「. So:
- every line whose final crop changed by ≥ 3 px, or whose row has ink continuing just past the
  crop, or which a later look called cut, went to an edge check: 178 lines. 169 passed;
- 4 were cut, 2 held parts of two lines, and 2 were unsure. Those 8 were dropped.

## Labels (`sheets.py`, `decide.py`)

**Sheets.** Contact sheets are at most 1280 px per side. Each shows the red-framed crop in context
at an integer zoom (PSP 3×, Vita 2×, nearest neighbour), with numbered candidate readings below;
characters that differ from candidate 1 are red. Each stage ran in its own Claude Opus 5.5
sessions, which were told not to open any other stage's files (they reported they did not; that
was not enforced). Pass 2 sees pass 1's label only as an unmarked candidate.

| stage | lines | result |
|---|---|---|
| pass 1: every candidate, engine support shown | 1,037 | 657 picked, 11 typed, 12 unsure, 304 UI, 27 cut, 26 bad |
| pass 2: every label read by one engine family (102) or none (11); candidates shuffled and unattributed, pass 1's text among them unmarked | 113 | 109 agree; 3 unsure and 1 cut dropped |
| GRAD check | 179 | 173 exact, 5 other text, 1 unsure |
| blind re-check (below) | 300 | |
| edge check | 178 | 169 pass |
| confirming reads: every kept line not yet read twice independently, blind, on the final crop | 173 + 133 | 298 agree; 5 UI and 3 unsure |

**Third look.** I resolved six lines myself:
- kept 渋澤焔 twice: 焰 is outside CP932 and the vocab;
- kept キョウ: measured as a small ョ, 9×9 px against 11 px;
- corrected one label from コリウス to ユリウス. All four engines and pass 1 read コ; the edge check
  and I read ユ at 12×;
- dropped 厳/嚴 and 橘/橋 as unsettled.

**Rule.** A label is kept when:
- both engine families read it (consensus), or pass 2 agreed;
- no later look disagreed, except where a third-look resolution kept it;
- it passed the edge check.

Every kept line has at least two independent agreeing reads: 596 have two and 49 have three.
`devnative/manifests/lines.jsonl` records per line: readers (engine to agreement, no text),
`how`, `support`, every stage's decision, `looks`, the boxes and `reviewer_kind`.
`dropped.jsonl` lists the 392 adjudicated lines not kept, with the reason (4 of them are a third
copy of the same label in one VN, teacher's cap).

## Label quality: blind re-check

Seed 20261012 drew 300 lines at random from the 664 kept after passes 1 and 2. Fresh sessions
read them with shuffled, unattributed engine candidates; they never saw a decision. Agreement is
on eval's scoring key.

| | agree | lines | rate |
|---|---|---|---|
| all | 291 | 300 | **97.0%** (Wilson 95%: 94.4-98.4%) |
| consensus labels | 242 | 251 | 96.4% |
| pass-2-confirmed (one engine or none) | 49 | 49 | 100% |
| PSP | 212 | 221 | 95.9% |
| Vita | 79 | 79 | 100% |
| strict | 11 | 12 | 91.7% |
| tv_native | 280 | 288 | 97.2% |

The 9 disagreements:
- 5 crop-edge cases: an end 。 or bracket outside the reviewed crop;
- 3 glyph variants the re-check could not settle: 焔/焰, 未/末, 厳/嚴;
- 1 scope call: an item-name plate the re-check called UI. A confirming read and pass 1 kept it.

They led to the final-crop widening, the edge check and the confirming reads above. After those,
292 of the 300 sampled lines are in the cohort, and the blind read agrees with the final label on
287 (98.3%). Of the other 5, 3 are crop-edge cases the widened crop fixed and the edge check
re-confirmed, 1 is the scope call, and 1 is 焔.

What this measures: agreement between independent assistant sessions on the same pixels. An
error every look makes the same way is not caught. The ユ/コ correction, which only the fifth look
noticed, shows such errors can exist.

## Pipeline history

`PIPELINE_FROZEN` was first written at `35a86ca`, which covers select, read, sheets and the
pass-1/2 and blind rules. Two amendments followed:
- `0cb084e`: the GRAD containment check;
- this PR: final-crop widening, the edge check, confirming reads and resolutions.

Each came from evidence found during review. The selection and reading stages are unchanged, and
their output reproduces exactly.

`build.py` without `--grad` and `--edge` (the realplus default) leaves out any GRAD box that runs
past the ink crop, and cuts final crops without the edge check.

## Reproduce

`devnative/README.md` lists every command. Screenshots, readings, sheets, decisions and crops stay
local, under `$HERD/shared/artifacts/devnative/`. Committed are: code, `devnative/manifests/`
(shots, lines, dropped, stats, review summary; no text, no pixels), this document and
`tests/test_devnative.py`.
