# realplus: a large real training pool from training-side VNDB screenshots

realplus is a pool of real Japanese visual-novel text lines with clean labels, for fine-tuning the
console recognizer. It uses every usable training-side VNDB screenshot and labels the lines with
devnative's pipeline (PP-OCRv6 small and MeikiOCR read every line; Claude adjudicates on contact
sheets; devnative's training rule decides). The screenshots, crops, labels and npz files stay local
under `$HERD/shared/artifacts/realplus`. The repository holds the code (`realplus/`), a text-free
screenshot manifest (`realplus/manifests/shots_NN.jsonl`: URL, file and pixel sha256, VN, release,
capture session, line and crop counts), the counts at every stage (`realplus/manifests/stats.json`),
this report and the tests.

## Result

| | |
|---|---|
| distinct lines | **43,264**, from **9,140 VNs** and 25,325 screenshots |
| characters | 475,091 (eval's scoring key); median line 8 characters, mean 11.0 |
| ink crops | 71,219: 43,139 Vita, 28,080 PSP (`realplus_train_ink.npz`, 286 MB) |
| GRAD crops | 53,626: 33,759 Vita, 19,867 PSP (`realplus_train_grad.npz`, 257 MB) |
| labels | 41,218 lines (95.3%) agreed by all four readings (auto); 2,046 reviewed by Claude, of which 1,990 are read by both engine families and 56 by one |
| blind re-check | **495 of 500** random accepted lines agree (99.0%, Wilson 95% 97.7-99.6%) |

Compared with train2's real fine-tune data (about 32k crops: tt 12,115, rd 9,811, ps 10,706), the pool
has 2.2 times as many ink crops and 3.8 times as many crops in all. The goal was 100k+ lines, and
43,264 distinct lines falls short of it; 124,845 crops is the count in train2's units. The limit was
review capacity, not screenshots (see Limits).

The files are under `$HERD/shared/artifacts/realplus`. `realplus/manifests/stats.json` pins every
file's sha256:

- the two npz files;
- the local line manifests: `manifests/lines.jsonl` (crop boxes and provenance, no text) and
  `lines_full.jsonl` (the same rows with labels);
- `labels.jsonl`.

## Sources and selection

The listing comes from VNDB's daily database dump (`vndb-db-2026-10-12.tar.zst`, sha256
`c6ada5e33d3163d1de97fc7e6e24b91048506a2a0979172c5205f439376ef4aa`, 190,763,803 bytes, dump
timestamp 2026-10-12 08:00:26+00), not thousands of API pages. Screenshots come from VNDB's image
server over HTTPS (`https://t.vndb.org/sf/NN/ID.jpg`), at most two requests at a time with a 0.5 s
delay, under a descriptive User-Agent.

`realplus/shots.py select` keeps a screenshot when:

- its VN is a Japanese original, and its release is Japanese-only or Japanese plus English, so no
  Chinese or Korean text is on screen;
- its VNDB flag averages are sexual < 0.3 and violence < 0.5, the repository's filters (realdata,
  data/fetch_assets.py);
- it is one of realdata's three kinds:
  - vita: a psv release at 960x544;
  - psp: a psp release at 480x272;
  - hires: any other release's shot, aspect 1.25 to 1.85, at least 640x400. It is later scaled to
    fit 960x544 and 480x272.
- its VN is on the training side (next section);
- the screenshot is in no existing pool or evaluation set, by VNDB image id at selection and by
  decoded pixels after the download.

Every training-side native Vita and PSP screenshot was already used. Each one sits in realdata's or
teacher's pools, in eval's sets, in devnative's cohort, or in a release shared with a reserved VN. So
the pool is all hires: PC and home-console screenshots, scaled the way teacher scales its hires kind
(`cv2.INTER_AREA` to fit 960x544, then 480x272 for the PSP crops).

Selected: 43,198 screenshots from 10,436 VNs. The drops at selection were:

| reason | screenshots |
|---|---:|
| VN not a Japanese original | 59,199 |
| rating (sexual >= 0.3 or violence >= 0.5) | 39,558 |
| VN is a test VN (`synth/common.is_test_vn`) | 18,758 |
| VN in the validation bucket (15 <= md5(vn) % 100 < 25) | 12,587 |
| VN shares a release with a reserved VN | 11,017 |
| release language other than ja / ja+en | 4,112 |
| already in a pool or set (realdata, teacher, synth2/3, data/vndb, linefind, eval candidates, fresh-synthetic) | 4,458 |
| size or aspect outside every kind | 1,789 |
| development VN | 1,732 |
| VN of `eval/manifests/test_games.json` "all" | 1,426 |
| family of CLANNAD, a fresh native title or a development VN | 1,081 |
| fresh native textbox title | 34 |
| relation of a fresh native textbox title | 27 |
| CLANNAD | 23 |

## Exclusions

These sets are never used for anything: vita-test, psp-test, the fresh native textbox test,
CLANNAD and the CLANNAD long-run references. A VN is off limits (`shots.reserved_vns`, first match
wins) when it is:

- CLANNAD (`v4`, or "clannad" / "クラナド" in a title), and CLANNAD's direct family as the conductor
  listed it (`shots.CLANNAD_FAMILY`): v12, v4060, v28752, v27548, v10058, v29951, v23354, v16901,
  v5698, v7983, v54298. All 11 were already reserved, by title or by relation; none was selected,
  downloaded or read;
- in `eval/manifests/test_games.json` "all";
- a test VN (`synth/common.is_test_vn`);
- a fresh native textbox title:
  - PSP v14909, v25459, v29454; Vita v21289, v21976, v20519; reserves v16509, v19140;
  - the explicit list in `shots.FRESH_NATIVE`, plus every VN id in eval's fresh-native files;
- a development VN:
  - synth3's `exclude_dev_vns.txt`;
  - the general-VN and CLANNAD development manifests;
  - devnative's cohort.
- in the validation bucket;
- related to a fresh native title by any VNDB relation kind (v15053, v52836, v25385, v28146,
  v57035);
- family of CLANNAD or of a development VN (alternative version, original, fan disc, parent, side
  story, sequel, prequel, same series);
- sharing a release with any of the above. For test and validation VNs, this applies only when the
  release holds two VNs or the partner is family.

9,816 test VNs, 6,623 validation VNs, 172 test-set VNs, 189 development VNs, 8 fresh native titles,
4 more fresh-native relations, 7 CLANNAD entries, 146 family VNs and 1,840 shared-release VNs were
reserved. Separately, 8,459 screenshot ids of existing pools and sets were reserved, along with eval's
31 test capture sessions.

The fresh native textbox titles were added to the exclusions on 2026-10-12, before any labelling.
None of the 8 titles or their 5 relations was ever selected, and no screenshot of them was
downloaded. Nothing had to be removed. `dedupe` also drops any screenshot whose file sha256 or
decoded RGB pixels match an image recorded by the fresh native test. At build time, the reserved
VNs are recomputed from the current inputs, and screenshots that dedupe marked as copies are left
out.

After the download, `shots.py dedupe` compares decoded pixels (devnative's `pixel_sha256`). It
checks against every local copy of an existing pool or set (`screenshot_dirs`, hashed by
`pool_pixel_hashes`) and against earlier realplus screenshots.
No realplus screenshot matched a pool or set image (7,845 pixel hashes) or an image the fresh
native test records. Two pairs of realplus screenshots were pixel-identical copies under different
image ids; for each pair, the copy not yet read was dropped.

`eval/tools/leakcheck.py` compares 32x18 grey thumbnails and flags pairs that correlate at 0.95 or
more. It ran in two passes:

- Against vita-test and psp-test (157 screenshots): it found two cross-VN pairs at 0.955, both with
  the same vita-test screenshot (v20326_144968). Both pool screenshots were dropped
  (`manifests/leak_drop.json`), which removed one kept line.
- Against devnative's 346 cohort screenshots: it found no pair.

The fresh native test's frames are not stored locally. They are covered by VN id, by relation and by
the hashes the test records.

## Pipeline

The labelling pipeline is devnative's (`devnative/README.md`), reused unchanged:

- read and sheets as frozen at 35a86ca;
- the GRAD containment check from 0cb084e;
- final crops, the edge check and looks from f523e30 (branch head f5a1dbd).

`realplus/pool.py` drives it over the pool:

1. `prep` writes read.py's input (43,196 screenshots; the two pixel copies are not read).
   `devnative/read.py` runs MeikiOCR's detector and four readings per line: MeikiOCR on its own
   box, MeikiOCR on a wider view, PP-OCRv6 small on that view, and PP-OCRv6 small on the crop. It
   runs in a network namespace with no network, through `heavy.sh`.
2. `plan` sorts the candidate lines (devnative's `is_candidate`: Japanese, 10-72 px of ink, not at
   the screen edge) into tiers:
   - auto: all four readings agree;
   - t1: the most supported candidate is read by both engine families;
   - t2: two candidates;
   - t3: three or more candidates.
   A review line is skipped when every candidate is a label its VN already has twice among the auto
   lines; the repeat cap would drop it anyway.
3. `batches` makes disjoint pass-1 batches of 200 lines with devnative's contact sheets. The order
   is by tier, then round-robin over VNs.
4. Claude reviews each batch: one review-rpN helper agent per batch, at most 6 at a time (8 after
   the owner's 19:05Z update).
   - The helper picks a candidate, types the text, marks the line unsure, or skips it as ui, cut or
     bad.
   - It saves its decisions after every sheet.
   - Every decision has `reviewer_kind: "assistant"`.
5. `final` applies `decide.py final --mode train --auto-consensus`:
   - a label needs at least one engine family after review;
   - a changed label is dropped;
   - pass 2 is optional and must agree when given;
   - lines that all four readings agree on are kept without review.
   Pass 2 (`decide.py queue`, then `sheets.py make --mode pass2`) re-read the 66 labels that only
   one engine family supports, blind. The 5 labels it disagreed with were dropped.
6. `edge-boxes`, `edge-batches` and `edge-apply` run devnative's edge check. See "Crops".
7. `build` cuts the crops and writes the npz files. `blind` draws the blind re-check sample.
   `publish` writes the committed manifests.

## Crops

Each kept line yields up to four grey crops, in console polarity:

- a Vita crop from the screenshot scaled to fit 960x544;
- a PSP crop from it scaled to fit 480x272, when the line has at least 10 px of ink at that scale;
- each of these as an ink crop and, where the GRAD check passes, as a GRAD crop.

A dark-on-light crop is inverted, and meta `dark` / `inverted` record it.

- The ink crop is eval's and teacher's crop: the ink box with furigana plus 4 px (Vita) or 2 px
  (PSP). It is then widened left and right to the boxes MeikiOCR gives the labelled characters
  (`lines.final_crop`), so every labelled glyph is inside. The reads store the padded box under
  `crop` and `crop_box` pads it again, so devnative's own crops and contact-sheet frames are
  ink + 8 / 4 px. realplus cuts the single-padded crop (`pool.read_line`); this was reported on
  PR #52.
- An ink crop is dropped when a quarter or more of another detected text box lies inside it. This
  is common with the tight line spacing of 640x480 PC games.
- Edge check: devnative's `edgecheck.flags` runs on the frame the reviewers saw. It flags a line
  when ink continues past that frame, or when its labelled characters lie beyond it. A label that
  no MeikiOCR reading spells cannot be widened, so it is flagged too. A flagged line is kept only
  when an edge-check batch (contact sheets in mode edge, framed on the final crop) picked its
  label. Otherwise it is dropped as `edge-unchecked`.
- The GRAD crop is the box the console's GRAD finder returns for the line's dialogue region
  (`lines.grad_assign`). It is used only when:
  - it holds the whole line and no other detected text (status "ok");
  - it does not run beyond the frame the reviewers saw (devnative's `grad_default`), at either scale;
  - it covers the final crop's labelled extent (`build.covers`).
- Labels with characters outside `data/vocab.txt` are dropped. Identical labels are kept at most
  twice per VN (devnative) and eight times overall (teacher).

The npz files use teacher_train's shard format (`teacher/render.py`: data, offsets, shapes, labels,
meta), one file per crop kind, with Vita and PSP crops together (meta `target`):

- `realplus_train_ink.npz`
- `realplus_train_grad.npz`

Meta per crop:

- provenance: real, pool, id, vn, session, release, kind "hires", target, crop, box;
- polarity: dark, inverted, pol;
- review: how, support, readers, decision, looks, reviewer_kind, name;
- characters: chars, chars_match;
- geometry: src_dims, screen.

Build counts. The edge check covered 3,800 flagged lines: 3,643 passed and 157 failed (111 cut,
36 bad, 9 other text, 1 unsure). Another 2,093 flagged lines were not checked and were left out.

Other lines left out at the build:

| reason | lines |
|---|---:|
| identical label already kept twice in the VN, or 8 times overall | 4,960 |
| no crop left after the per-crop rules | 375 |
| label character outside `data/vocab.txt` | 15 |
| screenshot dropped by the leak check | 1 |

Per-crop drops:

- 18,033 lines are below 10 px of ink at PSP scale, so they have no PSP crop.
- 507 Vita and 472 PSP ink crops take in a quarter of a neighbouring line.
- GRAD crops are dropped when the box:
  - is missing because the finder returned no box for the line (5,545 Vita, 3,782 PSP);
  - holds a quarter or more of other detected text (2,371, 3,957);
  - runs beyond the reviewed frame (2,645, 1,713);
  - holds only part of the line (1,418, 197);
  - cuts the labelled extent (774, 20).

Crop polarity:

- 14.6% of the ink crops and 13.5% of the GRAD crops were dark-on-light, and are stored inverted.
- In the Vita ink crops: 36,657 bright-on-dark and 6,482 dark-on-light.

All 71,219 ink crops load through `train/train2.py`'s `load_shard` with the repository vocabulary
(none dropped).

## Precision

The blind re-check sampled kept lines at random (seed 20261013) from the 50,874 kept at the
freeze, in a pre-registered order. Each line was framed on the final Vita ink crop the npz holds.
It was shown as devnative's blind sheets show lines: candidates shuffled and unattributed, engine
support hidden. Separate review-rpN helper agents decided the lines; none of them had reviewed
these lines before.

Lines drawn and scored:

- 540 lines were drawn first, and 70 more later from the same frozen set. Some lines leave the
  pool at the build, so the first draw held fewer than 500 pool lines.
- The first 500 lines of the order that are in the built pool were scored with devnative's
  `decide.blind_agreement`. A line agrees when the blind reading's scoring key equals the label's;
  a skip or "unsure" counts as a disagreement.

| lines | agree | rate | Wilson 95% |
|---|---:|---:|---|
| all | 495 / 500 | 99.0% | 97.7-99.6% |
| auto (all four readings) | 476 / 481 | 99.0% | |
| reviewed, both engine families | 18 / 18 | 100% | |
| reviewed, one engine family | 1 / 1 | 100% | |

A third look (assistant, `review/resolutions.jsonl`) at the five disagreements found:

- two wrong labels, both on auto lines: an ellipsis run of three cells read as two, and 。 read
  where the screen has 、;
- one line whose first kana could be either of a hiragana / katakana pair, so the image does not
  settle it;
- two cut lines, where the crop holds exactly the label but the line goes on just outside it: a
  second ！ after a gap, and a thin 一 that ends at the crop's edge.

All five were dropped from the pool. The 99.0% above was measured before these drops.

devnative's `decide_line` returns before its resolution block when a line has no pass-1 decision.
So realplus applies third looks to auto lines itself (`pool.apply_auto_resolutions`), with the same
semantics.

## Limits

- All crops are hires. Every crop comes from a PC or home-console screenshot scaled down to the
  console screen; no training-side native Vita or PSP screenshot was left. Downscaled text is
  smoother than native console rendering, so the pool complements the native pools and synth3's
  console rendering rather than replacing them.
- Most lines were kept by engine agreement. 95.3% of the lines are lines all four readings
  agree on, kept without review. These are text the engines already read well.
  - A misreading both engines share survives; the blind check measures how often (see Precision).
  - Hard styles are under-represented: stylised fonts, coloured or outlined text, low contrast.
    Their lines mostly go to review, and only 17 review batches were reviewed.
- Review coverage is small. The four readings did not all agree on 85,109 candidate lines; pass 1
  reviewed 3,400 of them (17 batches). The other 81,709 are undecided and not in the pool. More
  pass-1 review would add lines at about 65% yield, plus an edge check for about one line in nine.
  In this run, the herd's 12-agent cap left 3-5 review helpers running at a time.
- The pool holds every kind of text the engines read on screen, not only dialogue:
  - speaker names;
  - some menus, buttons and status text, because auto lines are not reviewed and so UI is not
    skipped there;
  - 35% of the lines have at most four characters; most are names, some are UI strings
    or the ends of wrapped lines.
  Repeats are capped at 2 per VN and 8 overall, so no string appears more than 8 times. Meta `name`
  is set only on reviewed lines.
- The edge check was not finished. 2,093 flagged lines had no edge check at the freeze and
  were dropped; the 11 remaining edge batches were withdrawn.
- devnative's edge rule misses some cut lines:
  - ink after a gap wider than 0.6 ink heights, such as a second ！;
  - a thin stroke one or two pixels tall, such as 一.
  The blind check found two such lines. A stricter rule (any ink within one ink height) would flag
  40% of the unflagged lines, which is too many to check, so the rule was left as devnative froze it.
- All reviews are Claude's, recorded as reviewer_kind "assistant", not a human's. This covers
  pass 1, pass 2, the edge check, the blind check and the third looks.
- The blind check uses helper agents of the same model as pass 1. They run in separate sessions
  with no shared context, and see shuffled, unattributed candidates.
- Precision was measured on 500 lines; the Wilson 95% interval is about +-1 point.

## Reproduce

```sh
HERD=/local/home/skerraut/herd-vita-ocr; R=$HERD/shared/artifacts/realplus; PY=.venv/bin/python
$PY realplus/shots.py dump --date 2026-10-12
$PY realplus/shots.py select
$PY realplus/shots.py fetch --workers 2 --delay 0.5
$PY realplus/shots.py dedupe
$PY realplus/pool.py feed --loop --max 8 --kinds pass2,edge,pass1   # reads, pass-1 batches, review helpers
$PY realplus/pool.py final                                          # labels.jsonl: the freeze
$PY devnative/decide.py --reads $R/reads/reads.jsonl queue --pass1 $R/review/pass1/b* --out Q.json
#   ... the single-family ids of Q.json -> review/pass2_queue.json, then
$PY devnative/sheets.py make --mode pass2 --lines $R/review/pass2_queue.json --out $R/review/pass2 --chunk 150
$PY realplus/pool.py edge-boxes && $PY realplus/pool.py edge-batches --n 99 --size 200 --min 1
$PY realplus/pool.py blind && $PY realplus/pool.py blind --extend 70 --size 70    # 540 + 70 lines
$PY realplus/pool.py feed --loop --ready 0 --max 8 --kinds blind,pass2,edge      # their helpers
$PY realplus/pool.py edge-apply && $PY realplus/pool.py final
HEAVY_SLOTS=12 $HERD/heavy.sh timeout 7200 $PY realplus/pool.py build --procs 5
$PY realplus/pool.py blind-score            # review/blind_report.json: the measured agreement
#   third looks at the disagreements -> review/resolutions.jsonl, then
$PY realplus/pool.py final && HEAVY_SLOTS=12 $HERD/heavy.sh timeout 7200 $PY realplus/pool.py build --procs 5
HEAVY_SLOTS=12 $HERD/heavy.sh timeout 7200 $PY eval/tools/leakcheck.py $R/shots --data $R/evaldata
$PY realplus/pool.py publish                # realplus/manifests/shots_NN.jsonl, stats.json
```
