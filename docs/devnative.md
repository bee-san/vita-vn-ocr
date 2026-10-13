# devnative: a native Vita/PSP development cohort

teacher_val, the development set train2 and synth3 select on, has 1,651 lines from 181 VNs, but
two thirds of them (1,064) are PC screenshots scaled down to console size. devnative holds only
native console frames: VNDB screenshots of PS Vita releases at 960×544 and PSP releases at
480×272. Each line is cut the way the console cuts it, and labelled by two local OCR engines and
independent Claude reviews.

It is a development cohort, for model selection only. No model was scored on it while it was
built. Labels were reviewed by an assistant (Claude Opus 5.5), not by a human.

## Versions

| version | commit | files (under `$HERD/shared/artifacts/devnative/`) | lines | used by |
|---|---|---|---|---|
| v1 | `f5a1dbd` (tag `devnative-cohort-v1`) | `devnative_*.npz` | 645 | study 1b (its protocol pins the four sha256s) and study 2 (its FL10 scoring recorded `COHORT_FROZEN f5a1dbd` and the sha256s at 2026-10-12T22:59:35Z; its H15 scoring reads the same files) |
| v2 | `ef76d1e` (tag `devnative-cohort-v2`) | `v2/devnative_*.npz` | 553 | study 1b reports it next to v1, reported only (its amendment 3, `06a998d`, 2026-10-13T00:23:15Z, pins the four sha256s) |
| v3 | `14ae22d` (tag `devnative-cohort-v3`) | `v3/devnative_*.npz` | 551 | VOCR3's selection (its amendment 3: the latest version announced in `devnative.status` when its first selection score is computed; `COHORT_V3 14ae22d`), and any protocol written from now on |

**v2: no eval/linefind screenshot.** v1 kept 41 screenshots of eval/linefind's sets in
tv_native, flagged `linefind`, on the grounds that they were not test sets. That was wrong:
`vita_holdout` and `psp_holdout` are the line finder's held-out sets ("nothing was tuned on
them", `docs/linefind.md`), and the fleet's rule is that a new cohort holds no screenshot of any
of eval's sets. v2 drops all 41, from every eval/linefind manifest (research, dev, holdout and
candidate sets, and the PSP pool), so devnative can never feed line-finder tuning:
- tv_native loses 94 lines: 55 Vita and 39 PSP, 81 of them with GRAD crops. It also loses 4 VNs
  (v12794 v18432 v18857 v3333). Vita goes from 163 lines to 108.
- 2 PSP lines come in, v12436_63023_6 and v8063_46799_0. v1's repeat cap had left them out, and
  their first copies were on dropped screenshots. Both are consensus labels with two looks.
- Every other crop, label and meta entry is byte-identical to v1's, in the same order.

**v3: eval's crop, padded once.** v1's and v2's ink crops were padded twice. read.py stores its
padded crop (ink + furigana + 4/2 px) under the line's `crop` key, the box `lines.crop_box` takes
as the text box, so every crop came out at ink + furigana + 8 px (Vita) / 4 px (PSP). The docs said
4/2 px. realplus found this (PR #52 comment, 2026-10-12 19:30Z). The contact sheets drew their
frames the same doubled way, so labels and crops agreed with each other, but not with eval's and
teacher_val's crops, nor with the plan (ink + 4/2 px). v3 cuts eval's crop (`lines.eval_line`):
ink + furigana + 4/2 px, widened to the labelled characters' boxes plus 4/2 px.
- A tighter crop can cut a glyph the wider one held, so the edge check ran again on every v3
  crop (see Lines). A line must pass both checks, v2's and v3's.
- tv_native ink loses 2 lines (v55136_209480_1, v55136_209481_5): an end 。 sits on the tight
  crop's right edge. Their GRAD crops leave too.
- 2 GRAD crops come in (v12451_57128_0, v16770_154904_4). v2's `covers` test measured the GRAD
  box against the doubled crop, less one padding, and those boxes fell 1-2 px short of it.
- Median crop height falls from 22 to 18 px on PSP and from 40 to 32 px on Vita (v1: 22 and 39).
- 16 tv_native ink crops change polarity, because `common.line_dark` decides on the crop's own
  pixels.
- Labels, selection, GRAD boxes and every other meta field are v2's. strict_grad is v2's file.

The v1 files stay where they are, unchanged: study 1b's committed protocol pins them by path and
sha256, and study 2's scoring reads them from there. Pick one version in a protocol before
scoring. Never choose between versions by their scores. teacher_val still holds the 41
screenshots; devnative does not change it.

## Result (v3)

| file (under `$HERD/shared/artifacts/devnative/v3/`) | lines | VNs | Vita / PSP | chars | sha256 |
|---|---|---|---|---|---|
| `devnative_tv_native_ink.npz` | 513 | 44 | 108 / 405 | 5,876 | `a3d2c9d91f0460c23aeb303ecb4583de3448f634ababdc5f3163eb0f2d95aec5` |
| `devnative_tv_native_grad.npz` | 423 | 44 | 94 / 329 | 4,995 | `0ed617b97fe3050936c5ca062ec0c7352416a6dc20f92f7efbb18cc4b6ca1782` |
| `devnative_strict_ink.npz` | 38 | 14 | 0 / 38 | 411 | `a1f0f83e78a0382d8c8efa276d5500162b4a5ebeba6fb1ce958567613f8e2589` |
| `devnative_strict_grad.npz` | 26 | 12 | 0 / 26 | 314 | `94b3dba500ef6f3297fd3e44515ec9e5004dbde85585ddbc4272ea1e78edc61e` |
| scene_vn | 0 | 0 | | | (no line survived review) |

v1's files, for the protocols that use them (padded twice, with the 41 linefind screenshots):

| file (under `$HERD/shared/artifacts/devnative/`) | lines | VNs | Vita / PSP | chars | sha256 |
|---|---|---|---|---|---|
| `devnative_tv_native_ink.npz` | 607 | 48 | 163 / 444 | 6,992 | `7fb36fc291ff7213d06bf023f0e7cb6fb391a88524d411d02b43d15c1c984d32` |
| `devnative_tv_native_grad.npz` | 503 | 48 | 142 / 361 | 5,957 | `f810690caec0164f47224b29ad537efa3500f41b1cc08653a49abdf6f6f94747` |
| `devnative_strict_ink.npz` | 38 | 14 | 0 / 38 | 411 | `ab2cc6890a0da59f49e9c1b491a0a1bc4e68cb9d7715ac485674b80cf2414f51` |
| `devnative_strict_grad.npz` | 26 | 12 | 0 / 26 | 314 | `94b3dba500ef6f3297fd3e44515ec9e5004dbde85585ddbc4272ea1e78edc61e` |

v2's files and sha256s are in this document as of tag `devnative-cohort-v2`.

v3 has 551 lines (6,287 characters) from 44 VNs, 216 screenshots and 46 capture sessions:
- PSP: 443 lines from 37 VNs, ink height 12/14/20 px (p10/50/90), crop height 18 px (median);
- Vita: 108 lines from 8 VNs, ink height 20/24/28 px, crop height 32 px;
- 151 of the lines are speaker names.

The plan's target was 3,000+ lines from 60+ VNs. The conductor dropped that minimum on
2026-10-12, in a message to this agent that no file records, because this is everything the
rules leave on VNDB (see Supply).

Report every file separately unless a protocol pools them, as study 2 pools strict and tv_native
(both crops, `eval/synth3_study2.py merge`). Never pool tv_native with teacher_val: they are the
same screenshots.

- **tv_native**: teacher_val's own native screenshots, re-cut through the console's path and
  relabelled here. In v3, 353 of its lines overlap a teacher_val line (crop IoU ≥ 0.3, recorded
  as `tv_id`), and all 353 have the same agreement key as teacher_val's label. That is a
  consistency check, not an independent one: MeikiOCR fed both.
- **strict**: other screenshots of the same reserved VNs, ones no training pool, scene list,
  eval set or other set contains.
- **scene_vn**: native screenshots of reserved VNs whose other screenshots were synthetic scene
  backgrounds, the background images themselves excluded. Its 3 screenshots gave 1 candidate
  line, and review dropped it.

Every devnative VN is a teacher_val VN, because the reserved list is teacher_val's 181 VNs plus
the 9 general-VN and CLANNAD development VNs (synth3's `GENERAL_VN_DEV`). All 14 strict VNs also
appear in tv_native. strict differs from teacher_val by screenshot, not by game.

**Scene backgrounds in tv_native.** devnative's task listed synth2's `backgrounds.json` and
`data/vndb/shots` among the training pools whose VNs the cohort leaves out. The conductor's
subset decision of 2026-10-12 applies that only to strict (no scene VN) and scene_vn (no scene
screenshot). tv_native is teacher_val's native screenshots, and teacher_val already shares them:
110 of its 271 native screenshots, including 59 of its 64 Vita ones, are on synth2's scene list.
So tv_native keeps them, flagged `bg_shot` in `shots.jsonl`, `lines.jsonl` and every npz row's
`meta.flags`.

synth2's renderer (`render.py`) draws its scene backgrounds from this list, cropping the upper 62%
of a screen behind synthetic text. So models trained on synth2 data (train2's release, and every
arm that mixes synth2 in) can have seen those parts of these screens. synth3's scenes leave out
every VN on the reserved list (`exclude_dev_vns.txt`, `synth/synth3.py` `_backgrounds`), so
synth3 data has not.

| | v3 | v1 |
|---|---|---|
| screenshots on synth2's scene list (its `vita_shots` / `psp_shots`) | 73 of 216 (37 Vita, 36 PSP) | 88 of 249 (51 Vita, 37 PSP) |
| lines on them | 199 of 551: all 108 Vita lines, and 91 PSP | 238 of 645: 145 of the 163 Vita lines, and 93 PSP |
| GRAD crops on them | 180 of 449 | 212 of 529 |
| ink crops inside / straddling / below the upper 62% | 16 / 16 / 167 | 17 / 17 / 204 |
| GRAD crops inside / straddling / below the upper 62% | 10 / 18 / 152 | 11 / 19 / 182 |

`bg_shot` marks every screenshot of a scene source select.py knows. That covers synth2's list
and every file under `data/vndb/shots`. In v3 the flag and synth2's list coincide (73
screenshots). v1 has 5 more flagged screenshots (18 lines, Vita): `data/vndb/shots` files
outside synth2's list, all of them eval/linefind screenshots, which v2 dropped.

A protocol comparing models with and without synth2 data can report `bg_shot` as a stratum.
`scene_vn` marks a line whose VN has scene-background screenshots: 221 v3 lines, 278 v1 lines.

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

One line, v1877_57776_0 (a name plate), has `pol: null`, because teacher's geometry recorded no
polarity hint for it. `load_dev` files it in a `pol:None` stratum; teacher_val has none. Its
`dark`/`inverted` come from the crop like every other line's (it is one of the 16 that flip
between v2 and v3).

`eval/synth3_ablation.py`'s `load_dev` reads all four files unchanged. v1's files were checked
with the scorer as committed in `b3f2757`, unchanged up to `feat/synth3` `ebb4dfb`. v2's and
v3's were checked with `feat/synth3` `002dfc3`, and v3's again with main's copy after synth3
merged (`eab1ab8`, #50). No model was scored:

```sh
.venv-ci/bin/python eval/synth3_ablation.py score --blob NAME=BLOB.vocr \
    --dev $HERD/shared/artifacts/devnative/v3/devnative_tv_native_grad.npz --out scores_tv_grad.json
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
   - it is not from one of eval's capture sessions or of a training-label release;
   - it is in none of eval's sets: vita-test, psp-test, or any eval/linefind manifest (v2).
4. **Duplicate checks.** Every candidate is fetched (local mirrors first, else VNDB) and checked
   against 7,041 reference screenshots (8,319 local images). The checks are by VNDB id, file
   sha256, decoded-pixel sha256, and same frame re-encoded (thumbnail correlation ≥ 0.95 and a
   full-size grey p99 difference < 48). The references cover eval's sets, the training label
   pools, teacher_val, the scene-background lists, linefind's sets and the general-VN development
   sources. A near-duplicate (correlation ≥ 0.95 without being the same frame) of a vita-test,
   psp-test or linefind screenshot is dropped too.
5. **Fresh-test sources.** Candidates are also checked against the fresh-test sources' file and
   PIL-RGB pixel hashes. None of those 8 titles or their relations was in the reserved list or
   the listing, so nothing from them was downloaded or used.
6. **General-VN development sources.** Their screenshots stay in tv_native only, flagged
   `general_vn_dev` (72 lines in v3). They are a development set, like teacher_val, not an eval
   set.

**Supply.** The listing gives 1,742 screenshots. These were excluded (v2 and v3 share this
selection):

| reason | screenshots |
|---|---|
| not native (another platform, or a size other than native) | 1,207 |
| label-pool VN | 82 |
| eval/linefind screenshot (v2; 41 of them were in v1's tv_native) | 54 |
| rating | 29 |
| scene-background screenshot | 25 |
| language | 18 |
| general-VN development screenshot outside tv_native | 12 |
| CLANNAD family | 10 |

305 screenshots remain (`devnative/manifests/shots.jsonl`: URL, VNDB id, sha256, pixel sha256,
subset, flags):

| subset | PSP | Vita | VNs |
|---|---|---|---|
| strict | 77 | 0 | 26 |
| tv_native | 184 | 41 | 47 |
| scene_vn | 3 | 0 | 3 |

v1 had 346: v2's 305 plus the 41 (19 PSP, 22 Vita, all tv_native). Re-running `select.py select
--no-fetch` with the v2 rule gives v1's manifest minus those 41 rows, every other row unchanged.

## Eval overlap check (`devnative/overlap.py`)

`overlap.py` re-implements the duplicate checks without select.py's reference list. It compares
every cohort screenshot with every screenshot of vita-test, psp-test and the 11 eval/linefind
manifests: 610 distinct screenshots, all with local copies. Each pair is compared by VNDB id,
file sha256, decoded-pixel sha256, same frame and near-duplicate thumbnails. The report is
`devnative/manifests/overlap.json`: ids, hashes and set names, no text.

- v3 (and v2, the same screenshots): no match.
- v1: 41 screenshots (94 lines), all in tv_native and all flagged `linefind`. Each is the same
  file as a linefind screenshot by all four strong tests. None matches vita-test or psp-test,
  and no screenshot is only a near-duplicate.

| linefind set | its role (`docs/linefind.md`) | v1 screenshots | v1 lines |
|---|---|---|---|
| research_a | tuning | 3 | 12 |
| research_b | tuning | 3 | 8 |
| vita_dev (its candidates) | development | 10 (12) | 24 (24) |
| vita_holdout (its candidates) | held out | 2 (7) | 8 (23) |
| psp_dev (its candidates) | development | 5 (7) | 13 (13) |
| psp_holdout (its candidates) | held out | 5 (5) | 14 (14) |
| psp_pool | the PSP screenshots both PSP sets were drawn from | 19 | 39 |

A screenshot can be in several sets. CI checks the committed manifests against eval's and
linefind's manifests by VNDB id and file sha256 (`tests/test_devnative.py`).

## Lines (`read.py`, `lines.py`, `gradcheck.py`, `edgecheck.py`)

The reading and review below covered all 346 of v1's screenshots, so their counts include the
lines of the 41 that v2 dropped afterwards. Those lines were not reviewed again; they are simply
left out.

**Reading.** Each screenshot is read with pinned engines, offline (`unshare -rn`, one heavy.sh
slot):
- MeikiOCR 0.3.5 detects and reads at native size. teacher's geometry then merges row pieces,
  refines each box to its ink and attaches furigana.
- Each line gets four readings: MeikiOCR on its own box (`M`), MeikiOCR on a wider view (`m`),
  PP-OCRv6 small on the view (`V`), and PP-OCRv6 small on the crop (`v`). The crop `v` reads is
  read.py's ink + furigana + 4/2 px.

1,477 lines were read. 1,037 are candidates: 10-72 px ink, not at the screen edge, and at least
one Japanese reading. Re-running the read gives identical readings for all 346 screenshots. This
was checked once in the first session. It was checked again on 2026-10-13 at 01:53Z, with the
code at `14ae22d`, offline, into a scratch directory: every field matched except the timing, and
the engine files were the same.

**Review frame.** The contact sheets framed each line at ink + furigana + 8 px (Vita) / 4 px
(PSP): `lines.crop_box` padded read.py's already padded crop again. Every label was decided on
that frame.

**Ink crop (v3).** eval's crop (`lines.eval_line`, as in `eval/sets.py` and teacher's `crop_of`):
the ink box plus furigana, widened left and right to the labelled characters' boxes, plus 4 px on
Vita or 2 px on PSP. It lies inside the review frame, except where the widening reaches past it.
v1 and v2 widened the review frame instead.

**GRAD crop.** Built as follows:
1. Dialogue regions are the lines pass 1 kept (decided, unsure or cut; not UI), grouped with
   eval's rule (vertical gap ≤ 1.6 line heights), plus a 12 px (Vita) or 6 px (PSP) margin. The
   rule ran on read.py's padded boxes, so a region reaches 16 / 8 px beyond the text, and the
   gap is measured between padded boxes. v3 keeps these regions and boxes, so the GRAD crops and
   their review stay valid.
2. `common.find_lines_ex(region, "grad")`, the bit-exact twin of `runtime/vocr_lines.c`, runs on
   each region. Its boxes are matched one-to-one to the lines (eval's finder match, overlap ≥ 0.5).
3. A box is used only when it:
   - holds the whole ink line;
   - holds less than a quarter of any other detected text box;
   - covers the final ink crop's labelled extent (the crop less its padding), to 1 px.
4. A box that runs beyond the review frame (more than max(3 px, 0.3 ink heights) sideways, or 0.6
   ink heights vertically) is used only after a contact-sheet check that it shows exactly the
   label. The detector misses some glyphs (card digits, a bracket), and those would make the label
   wrong for that crop. Inside that margin, the box shows nothing the labellers did not see.

Of the 668 lines pass 1 labelled, the finder found none for 53 and merged 27 with other text. Its
box cut 15 of them and was rejected after the check for 6. Of v3's final lines, 17 more have a
GRAD box that does not reach the widened crop's labelled extent, so they get no GRAD crop.

**Edge check (v1, v2).** In the blind re-check, five disagreements came from crop edges:
MeikiOCR's box had left out an end 。, 」 or 「. So every line went to an edge check whose final
crop (the review frame, widened) changed by ≥ 3 px, whose row has ink continuing just past the
crop, or which a later look called cut. That was 178 lines:
- 169 passed, one of them by typing its label (the ユリウス correction);
- 8 were dropped: 4 cut, 2 holding parts of two lines, 2 unsure;
- 1, v8063_46795_0, had already been dropped by the third look before the check was applied.

**Edge check (v3).** The same rule ran on v3's tighter crops (`edgecheck.py --crop eval`) for
all 562 kept labels on v2's screenshots. It also flagged every label that no MeikiOCR reading
spells, because such a label has no character boxes to widen the crop to. 205 lines went to the
check. A line can carry several reasons:

| reason | lines |
|---|---|
| ink beyond the crop, left / right | 71 / 61 |
| label without character boxes | 66 |
| crop widened ≥ 3 px past the review frame, left / right | 60 / 17 |
| a later look typed other text / called the line cut | 3 / 2 |

Five independent sessions checked them on contact sheets framed on the v3 crop, one batch each.
They passed 201 and cut 4. In each of the 4, an end 。 or 」 sits on or just past the crop's
right edge. Two of the 4 were already v2 drops; the other two are v3's two losses. A line must
pass v2's check as well (`build.py --edge-prior`), so the 5 v2 drops that v3's check passed stay
out, among them the 未/末 line.

## Labels (`sheets.py`, `decide.py`)

**Sheets.** Contact sheets are at most 1280 px per side. Each shows the red-framed crop in context
at an integer zoom (PSP 3×, Vita 2×, nearest neighbour), with numbered candidate readings below;
characters that differ from candidate 1 are red. Each stage ran in its own Claude Opus 5.5
sessions, which were told not to open any other stage's files (they reported they did not; that
was not enforced). Pass 2 sees pass 1's label only as an unmarked candidate.

A sheet-level default (`{"sheet": ..., "default": 1}`) accepts candidate 1 of every
single-candidate entry on that sheet not decided on its own line. It decided:
- 83 of pass 1's 1,037 entries, each one a line all four readings agreed on;
- 171 of 179 GRAD checks;
- 137 of 178 v1/v2 edge checks;
- 190 of 205 v3 edge checks.

Pass 2, the blind re-check and the confirming reads used none.

| stage | lines | result |
|---|---|---|
| pass 1: every candidate, engine support shown | 1,037 | 657 picked, 11 typed, 12 unsure, 304 UI, 27 cut, 26 bad |
| pass 2: every label read by one engine family (102) or none (11); candidates shuffled and unattributed, pass 1's text among them unmarked | 113 | 109 agree; 3 unsure and 1 cut dropped |
| GRAD check | 179 | 173 exact, 5 other text, 1 unsure |
| blind re-check (below) | 300 | |
| edge check | 178 | 169 pass, 8 dropped; 1 dropped by the third look |
| confirming reads: every kept line not yet read twice independently, blind, on the final crop | 173 + 133 | 298 agree; 5 UI and 3 unsure |
| edge check on v3's crops | 205 | 201 pass, 4 cut |

**Third look.** I resolved six lines myself:
- kept 渋澤焔 twice: 焰 is outside CP932 and the vocab;
- kept キョウ: measured as a small ョ, 9×9 px against 11 px;
- corrected one label from コリウス to ユリウス. All four engines and pass 1 read コ; the edge check
  and I read ユ at 12×;
- dropped 厳/嚴 and 橘/橋 as unsettled.

All four kept resolutions are in v3.

The 厳/嚴 drop is the name plate v8063_46795_0. Its PP-OCRv6 crop reading said 嚴, the blind
re-check was unsure, and the third look could not settle it. v8063_46795_2, a dialogue line on the
same screenshot that starts with the same name, is kept as 厳: all four readings and its blind
re-check read 厳, so nothing disputes it. The rule is applied line by line. If the glyph were 嚴,
that label would be wrong too; the evidence on that line says 厳.

**Rule.** A label is kept when all of these hold:
- **Agreement.** Both engine families read it (consensus), or an independent pass 2 agreed. One
  label is a third-look text change instead: ユリウス (v1874_11826_0). No engine reads it; its
  agreeing reads are the edge-check session's typed text and my third look.
- **No disagreeing confirming read.** Two lines are kept by a third-look resolution after their
  confirming read was unsure: 渋澤焔 on v12858_60789 and キョウ.
- **Edge check.** It passed the edge check, and in v3 the edge check on its own crop too.

The blind re-check is a measurement: by itself it never drops or keeps a line. Each of its
disagreements was given a third look or a check (above). Three kept lines have a blind read that
disagrees:
- v12858_60790_0 (渋澤焔): the re-check was unsure between 焔 and 焰, and a third look kept it.
- v10091_56559_0, with no third look: the re-check called the item plate UI, while pass 1 and a
  confirming read read it as text.
- v12858_60791_2, with no third look: the re-check called it cut, because its closing 」 lay outside
  the review frame. The final crop is widened to it, and both edge checks passed it.

Every kept line has at least two independent agreeing reads: in v3, 510 have two and 41 have
three. `devnative/manifests/lines.jsonl` records per line: readers (engine to agreement, no
text), `how`, `support`, every stage's decision, `looks`, the boxes and `reviewer_kind`.
- The boxes are `ink`, `text_box` (v3: ink + furigana, unpadded; v1/v2: read.py's padded box),
  `reviewed_box` (the review frame), `ink_box` (the crop cut) and the GRAD box.
- `dropped.jsonl` lists the 486 adjudicated lines not kept, with the reason. 94 of them are on
  the 41 linefind screenshots (`linefind-shot`) and 2 fail the v3 edge check
  (`eval-edge-cut`). Another 2 are a third copy of the same label in one VN (teacher's cap).

## Label quality: blind re-check

Seed 20261012 drew 300 lines at random from the 664 kept after passes 1 and 2 (v1's screenshots).
Fresh sessions read them with shuffled, unattributed engine candidates; they never saw a decision.
They read them on the review frame. Agreement is on eval's scoring key. The v3 columns are the
same measurement restricted to the 256 sampled lines whose screenshot v3 (like v2) keeps.
Dropping a screenshot for being in a linefind set does not depend on its labels.

| | v1: agree / lines | rate | v3: agree / lines | rate |
|---|---|---|---|---|
| all | 291 / 300 | **97.0%** (Wilson 95%: 94.4-98.4%) | 247 / 256 | **96.5%** (93.5-98.1%) |
| consensus labels | 242 / 251 | 96.4% | 206 / 215 | 95.8% |
| pass-2-confirmed (one engine or none) | 49 / 49 | 100% | 41 / 41 | 100% |
| PSP | 212 / 221 | 95.9% | 195 / 204 | 95.6% |
| Vita | 79 / 79 | 100% | 52 / 52 | 100% |
| strict | 11 / 12 | 91.7% | 11 / 12 | 91.7% |
| tv_native | 280 / 288 | 97.2% | 236 / 244 | 96.7% |

All 9 disagreements are on screenshots v3 keeps; the 44 sampled lines on the dropped screenshots
all agreed. The 9 were:
- 5 crop-edge cases: an end 。 or bracket outside the reviewed crop;
- 3 glyph variants the re-check could not settle: 焔/焰, 未/末, 厳/嚴;
- 1 scope call: an item-name plate the re-check called UI. A confirming read and pass 1 kept it.

They led to the final-crop widening, the edge check and the confirming reads above. After those,
248 of the sampled lines are in v3, and the blind read agrees with the final label on 245 (98.8%;
v2: 245 of 250, 98.0%; v1: 287 of 292, 98.3%). The 2 lines v3 loses to its edge check are two of
the end-punctuation disagreements. Of the other 3:
- 1 is a crop-edge case that the widened crop fixed and both edge checks re-confirmed;
- 1 is the scope call;
- 1 is 焔.

The figures after the fixes are post hoc. The fixes were made because of this sample's
disagreements, so the figures are not an independent estimate. The measured agreement is 96.5%
(93.5-98.1%), above.

What this measures: agreement between independent assistant sessions on the same pixels. An
error every look makes the same way is not caught. The ユ/コ correction, which only the fifth look
noticed, shows such errors can exist.

## Pipeline history

The hashes below are the commits on `feat/devnative` as they were when each status was written
and each protocol cited them. They stay reachable from the tags `devnative-cohort-v1`,
`devnative-cohort-v2` and `devnative-cohort-v3`. The branch was rebased onto main after some of
them and then squash-merged, so main has different hashes for the same content.

| commit | what |
|---|---|
| `35a86ca` | `PIPELINE_FROZEN` for realplus: select, read, sheets, and the pass-1/2 and blind rules |
| `0cb084e` | the GRAD containment check |
| `f523e30` | final-crop widening, the edge check, confirming reads and resolutions |
| `c83ba3f`, `f5a1dbd` | cohort v1, `COHORT_FROZEN f5a1dbd` (tag `devnative-cohort-v1`) |
| `6d9e003`, `ef76d1e` (tag `devnative-cohort-v2`) | cohort v2: `select.py` excludes eval/linefind screenshots from every subset, `build.py --selection`, `overlap.py` |
| `cb7f9ed`, `14ae22d` (tag `devnative-cohort-v3`) | cohort v3: eval's single-padded ink crop (`lines.eval_line`, `edgecheck.py --crop eval`, `build.py --crop eval --edge-prior`) and its edge check |
| after `14ae22d` | text only, from audit-devnative's report: the scene-background disclosure, each version's use, the rule's wording and number fixes. No cohort file changed, so v3 stays `14ae22d` |

Each amendment came from evidence found during review or audit.

v2 and v3 change no reading or decision. `build.py` keeps every version reproducible: with each
version's options it rebuilds that version byte for byte. For v1 that covers the four sha256s,
`lines.jsonl`, `dropped.jsonl`, `stats.json` and `lines_full.jsonl`; for v2 (`--selection`) and v3
(`--crop eval --edge-prior`), the npz files and manifests. realplus, which calls these functions
on its own single-padded lines, is unaffected.

`build.py` without `--grad` and `--edge` (the realplus default) leaves out any GRAD box that runs
past the review frame, and cuts final crops without the edge check.

## Reproduce

`devnative/README.md` lists every command. Screenshots, readings, sheets, decisions and crops stay
local, under `$HERD/shared/artifacts/devnative/`: v1 at the top level, v2 under `v2/` and v3
under `v3/`. v1's selection report and manifests are kept under `v1/`.

Committed:
- the code;
- `devnative/manifests/`: shots, lines, dropped, stats, the review summary and the overlap
  report (no text, no pixels);
- this document;
- `tests/test_devnative.py`.
