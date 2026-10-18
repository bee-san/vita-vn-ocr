# Vertical text: a column finder and a dedicated vertical recognizer (issue #99)

The owner (2026-10-18 06:50Z): "i think a vertical text is worth it because of noboru on ps vita". Then, at 07:40Z:
train "a small, fast model specifically for manga / vertical text". Until this work, every finder and recognizer
here was horizontal-only:
- GRAD and its refinement find rows;
- the learned detector of #96 flags vertical boxes and drops them;
- the teacher pipeline drops vertical lines;
- no scored set held vertical text.

This document is the protocol's record (`eval/vertical_protocol.json`, frozen before any score, amendments 1-3) and
its results. Every number is a host measurement through the C runtime (W8I) unless it is marked as an estimate.
Nothing here has run on a console.

## Answer first

- **NOBORU is a PS Vita homebrew manga and comics reader** ([Creckeryop/NOBORU](https://github.com/Creckeryop/NOBORU)), not a VNDB title. No PS Vita VN on VNDB is called that. So the NOBORU use case is manga balloons, which are vertical text read right to left. Vertical VN dialogue (the protocol's primary metric) is the other target.
- **The dedicated recognizer VOCR5-V** reads the column turned 90° anticlockwise. It has VOCR5-S's shape: 819,792 B of int8 weights, a W8I arena of 124,832 B and 967,392 MACs per frame. On the development set it reads dialogue columns at **10.64% CER on VN pages** and **2.70% on manga**. The best code-only arm is the column cut into characters and read by the released VOCR5-N (1.19 MB). It reads the same columns at 36.70% and 4.18%. VN difference: −26.07 points, VN-cluster interval −35.13 to −16.80. By platform, VOCR5-V reads PSP columns at 3.09% and PC pages scaled to the Vita at 13.24%.
- **The column finder** is GRAD on the region's transpose plus a valley split. It is `runtime/vocr_vertical.c`, with a bit-exact Python twin (`synth/vertical_twin.py`):
  - development column F1 0.837;
  - 4,264 B of Thumb-2 code at -O2;
  - an arena of 83,619 B at 960×544, about GRAD's 83,323 B;
  - no weights.

  #96's detector's vertical output is far worse (F1 0.403). End to end (finder, the columns joined right to left, VOCR5-V): development VN 19.47%, manga 5.19%.
- **Speed (estimates):**
  - A typical development VN region is 63 CTC frames at VOCR5-S's per-frame cost, about 0.14 s at 333 MHz on one Cortex-A9 core plus the finder's pass. A manga balloon is 46 frames, about 0.10 s.
  - Per frame: 2.21 ms at 333 MHz, 1.67 ms at 444; VOCR5-S's #85 estimate without prefetch credit, which is calibrated to VOCR3-H15's measured console rate.
- **The test set, read once after the selection was committed** (16 VN clusters, *Black Jack ni Yoroshiku* vol. 13):

  | test (measured) | VN | manga |
  |---|---:|---:|
  | VOCR5-V, fixed columns | **7.17%** | **0.86%** |
  | P1 + VOCR5-V, end to end | **11.29%** | **2.45%** |
  | best code-only arm R\* (a1 + VOCR5-N), fixed columns | 33.71% | 4.03% |

  VOCR5-V − R\* on VN: −26.54 points (−34.93 to −12.60). The owner's Trick x Logic PSP frames read at 1.95%.

## What NOBORU is, and the census

VNDB (dump 2026-10-12 and the live API):
- 646 VNs have a PS Vita release; none has のぼる / ノボル / 昇 / 登 / "noboru" in a title, alias or release title;
- the three "Noboru" titles are PC-only (v7659, v41401, v49972);
- the only character named Noboru in a VN with a Sony handheld release is in v7867 (PSP);
- the nearest Vita title by sound, Hana Oboro (v19288), has horizontal dialogue (checked by eye).

NOBORU's README lists "Vertical orientation in Reader" and per-manga orientation, read direction and zoom.

The census (`eval/vertical_count.py`, `eval/vertical_census_20261018.json`) merges every stored read of every real screenshot:
- MeikiOCR's detector boxes with their vertical flag;
- Chrome Screen AI's lines in the teacher's reads, whose label pipeline drops vertical lines before it votes.

A vertical line is at least twice as tall as wide and holds at least 3 kana or kanji. A vertical page has two such lines of 6 or more characters.

| pool | screenshots read | with a vertical line | vertical pages | VNs with a vertical page |
|---|---:|---:|---:|---:|
| VNDB PS Vita releases | 614 | 32 | 4 | 4 |
| VNDB PSP releases | 2,382 | 108 | 55 | 15 |
| VNDB PC releases (read at the Vita size) | 67,700 | 1,830 | 624 | 306 |
| owner's PS Vita frames | 2,049 | 2 | 1 | 1 |
| owner's PSP frames | 2,942 | 20 | 2 | 1 |

Of the 62 console vertical pages, 53 are dialogue by eye (30 vertical ADV boxes, 21 balloons, 2 NVL pages) and 9 are UI.

The console VNs with vertical dialogue are:
- PSP: Harukanaru Toki no Naka de 1-4 (six releases), Bakumatsu Rock, Genrou, Ren'ai Banchou 1-2, Gekka Ryouran Romance and Trick x Logic (the owner's frames);
- PS Vita: Yunohana SpRING!.

## Data

**Pages** (`eval/vertical_data.py`):
- VNDB pages: the census's console vertical-dialogue pages, and one PC vertical page per VNDB-relation component, read at 960×544.
- Manga pages: *Black Jack ni Yoroshiku* (Shuho Sato, [densho810.com/free](https://densho810.com/free/); the terms allow any use, commercial included, with the credit "ブラックジャックによろしく 佐藤秀峰"). They are shown as NOBORU shows a page in its fit-to-width mode: the page scaled to 960 px wide and cut into 544-row windows. Volume 1 is development, volume 13 test.
- The owner's Trick x Logic frames (PSP): test only, private.

Excluded everywhere: every reserved.json test VN and test session, and every eval screenshot.

**Split by series cluster:**
- development: Harukanaru (6 VNs), Genrou, Gekka Ryouran and 40 PC clusters;
- test: Bakumatsu Rock, Ren'ai Banchou, Yunohana SpRING!, Trick x Logic and 25 PC clusters.

**Columns.** MeikiOCR's vertical detector boxes are refined to the ink of their main text (`column_box`: the heaviest run of the box's column profile; a narrower run to its right is furigana and is left out). MeikiOCR (its page read and its vertical recognizer on the crop) and PP-OCRv6 small (on the crop turned 90°) give candidate readings.

**Labels.** Two independent, blind Claude passes per column (candidates shuffled and unattributed in the second); disagreements get a third look. A column is labelled only when two looks agree.
- Development: 474 of the 477 columns that both passes read as text agreed; 13 disagreements were settled by the third look. A blind spot check of 26 random agreed labels found no error.
- Test: 3 disagreements were settled by the third look.
- Conventions: the main text only (never furigana), and vertical forms written as their ordinary characters (︙ as …, ﹁ as 「, a vertical bar as ー). In-story choice options count as dialogue.

| set | VN dialogue columns / characters / clusters | manga columns / characters / pages |
|---|---|---|
| development (`eval/manifests/vertical-dev-*.jsonl`) | 202 / 2,275 / 25 (PSP, PC) | 262 / 1,483 / 37 windows of 18 pages |
| test (`eval/manifests/vertical-test-*.jsonl` + 22 private owner columns) | 173 / 2,453 / 16 (PSP, PS Vita, PC, owner PSP) | 126 / 694 / 25 windows |

Regions (for the end-to-end measure) are a page's labelled dialogue columns grouped by adjacency: their union plus 12 px (Vita) or 6 px (PSP). Collinear pieces of one column (MeikiOCR cuts a column at a full-width space) are merged. A name or a skipped column inside a region is an ignore area.

## Protocol (`eval/vertical_protocol.json`)

**Primary metric:** VN development dialogue CER on fixed column crops (the box plus 4 px, or 2 px on the PSP; console polarity), C runtime W8I, greedy. Intervals come from 10,000 bootstrap resamples over series clusters; manga has one series per split, so its unit is the book page (amendment 1).

**Arms:**
- a0 and a1: the column cut into character cells by its row profile and laid out left to right as squares, so characters stay upright, and read by the released VOCR5 models. a1 also turns a cell's lone tall thin stroke (a vertical long-vowel bar) back to horizontal, and moves a small mark in a cell's upper right (vertical 、。 and small kana) to its lower left.
- VOCR5-V: the column turned anticlockwise.
- (b), a vertical head on the VOCR5 trunk, is not run: the head and bottleneck are 502,171 of VOCR5-S's 726,075 parameters. With the turned input the trunk has to change anyway, and with the re-tiled input (b) keeps a0/a1's segmentation limits.

**Rules:**
- a1 replaces a0 only if a1 − a0 (VOCR5-N, VN) has an interval entirely below 0.
- VOCR5-V is adopted when both hold:
  - (i) the upper bound of the VN interval of V − R\* is below +0.5 points;
  - (ii) V's VN point estimate is lower, or V − R\* on manga is resolved below 0.
- Among adopted sizes, the smallest ships.
- Finder: P1 replaces P0, and the detector replaces the projection finder, only with an end-to-end VN interval entirely below 0.
- The test set is read once, after the selection is committed.

**The dedicated model's recipe** (pinned before training):
- start: VOCR5-S's checkpoint (`v5f_S_seed2`);
- `train/train2.py`, 30,000 steps of 96, lr 1e-3 cosine, EMA 0.999;
- data, mixed 0.75 / 0.15 / 0.10:
  - 75%: 1,000,000 synthetic columns (`synth/vertical_render.py`, `data/manifests/vertical_shards_v1.json`);
  - 15%: 1,424 real columns where MeikiOCR and PP-OCRv6 small agree, from training-side pages only (no VN in a development or test VN's relation component, no reserved or dev-sample VN), each also at half scale;
  - 10%: 3,066 columns read by PP-OCRv6 small alone with a lowest character confidence of 0.9 or more;
- the run's model is its last checkpoint.

The synthetic columns use synth3's fonts, texts, scenes and exclusions. Typesetting is the fonts' own vertical layout through HarfBuzz (the `vert` forms and the vertical positions of 、。 and small kana). There are three looks:
- Vita ADV/NVL columns over scenes or boxes, 16-34 px, with outline or shadow;
- PSP sizes, sometimes 1-bit glyphs or shown 2x;
- manga balloons;

and sometimes furigana or the edges of neighbouring columns in the crop.

## Results on the development set

Dialogue CER on fixed column crops (measured):

| arm | model | weights | VN | manga |
|---|---|---:|---:|---:|
| a0 | VOCR5-S / N / L | 0.82 / 1.19 / 2.47 MB | 39.82 / 37.89 / 36.18% | 7.22 / 6.95 / 6.88% |
| a1 | VOCR5-S / N / L | 0.82 / 1.19 / 2.47 MB | 38.20 / **36.70** / 35.08% | 4.38 / **4.18** / 4.11% |
| rot | **VOCR5-V seed 1** | 0.82 MB | **10.64%** | **2.70%** |

Rule steps, in order:
1. a1 − a0 (VOCR5-N, VN): −1.19 points (−2.47 to −0.42), so R\* = a1 + VOCR5-N.
2. VOCR5-V − R\* on VN: −26.07 points (−35.13 to −16.80); on manga −1.48 (−3.90 to +1.85). VOCR5-V is adopted.

By platform, VOCR5-V reads PSP columns at 3.09% and PC columns at 13.24%. Half of the PC errors are one cluster (v27023, a newspaper-style NVL page whose kanji-dense text is 9-15 px wide after scaling, over a dark texture). Speaker names (17 columns) read badly (54%); the owner does not use them.

Why the re-tiling fails on VN text: cutting a column into characters by its row profile breaks on small PSP kana (っ read as つ, ゃ as や) and on outlined text or busy backgrounds (cells merged or split). Manga balloons, clean black on white, survive it. A dedicated CTC model on the turned column needs no segmentation. PP-OCRv6 small, which reads vertical text the same way, pointed there.

**Column finders** (development regions; column F1 at IoU ≥ 0.5; end to end with VOCR5-V; measured):

| finder | F1 | regions exact | end to end VN | end to end manga |
|---|---:|---:|---:|---:|
| P0: GRAD transposed | 0.825 | 66% | 20.57% | 6.14% |
| **P1: P0 + valley split and density-gated growth** (`VOCR_COLUMNS_VALLEYS`) | **0.837** | **69%** | **19.47%** | **5.19%** |
| D: #96's detector, its vertical boxes | 0.403 | 20% | 52.88% | 46.19% |

P1 − P0, end to end on VN: −1.10 points (−2.11 to −0.31), so P1 is selected. The gap between end to end (19.47%) and fixed columns (10.64%) is the finder: columns it merges or misses, and clutter at their ends.

## The smaller head and the second seed

**VOCR5-V3500** was pre-registered in amendment 2, once seed 1 was adopted.
- Head: `data/vocab_vn_3500.txt`, the 3,499 most frequent VN characters of `data/vocab_vn_4987.txt` plus `<unk>`.
- Start: seed 1's last checkpoint with its head re-targeted by `train/remap_head.py` (3,495 rows copied, 5 from `<unk>`).
- Training: 10,000 steps at lr 5e-4.
- Size: 657,616 B of int8 weights (−162,176 B, −20%) and 824,736 MACs per frame (−15%).

It reads development VN at 11.21% and manga at 2.70%:

| against seed 1 | difference | interval |
|---|---:|---|
| VN | +0.57 points | −0.12 to +1.03 |
| manga | 0.00 points | −0.18 to +0.20 |

The VN upper bound is over the +0.5-point margin, so it is **not adopted** and the 4,987-class VOCR5-V ships. The test read confirms the direction (report only): VN +0.45 points (+0.05 to +0.90).

**Seed 2** (report only: the spread between seeds): SEED2_RESULT

## The test set, once

The selection was committed and pushed first (amendment 4, [49c08ab](https://github.com/bee-san/vita-vn-ocr/commit/49c08ab)). Then `eval/vertical_test_once.py` read the test set once:
- attempt record `eval/vertical_test_attempt_20261018.json`, 2026-10-18T10:50:39Z to 10:51:08Z;
- record `eval/vertical_test_20261018.json` (aggregates only; the owner's frames are scored privately);
- the selection (P1 + VOCR5-V) and, on the same read and report only, R\* and VOCR5-V3500.

The attempt record hashes the local copies of the test manifests, which are the committed rows plus each page's local path.

| test, dialogue CER (measured) | VN | manga |
|---|---:|---:|
| **VOCR5-V, fixed columns** | **7.17%** (176/2,453) | **0.86%** (6/694) |
| **P1 + VOCR5-V, end to end** | **11.29%** | **2.45%** |
| R\* (a1 + VOCR5-N), fixed columns | 33.71% | 4.03% |
| R\*, end to end with P1 | 34.04% | 11.96% |
| VOCR5-V3500, fixed columns | 7.62% | 0.86% |

VOCR5-V − R\* (VN cluster / manga page intervals):

| | VN | manga |
|---|---|---|
| fixed columns | −26.54 points (−34.93 to −12.60) | −3.17 (−6.15 to −0.99) |
| end to end | −22.75 (−31.12 to −8.54) | −9.51 (−12.03 to −7.24) |

VOCR5-V's test VN by platform (fixed columns):

| platform | CER | characters | columns |
|---|---:|---:|---:|
| PC pages scaled to the Vita | 8.26% | 1,876 | 115 |
| PSP (VNDB) | 5.52% | 290 | 34 |
| the owner's Trick x Logic PSP frames | 1.95% | 257 | 22 |
| PS Vita (Yunohana SpRING!) | 0.00% | 30 | 2 |

The test VN cohort (16 clusters) reads better than development's (10.64%). Development's errors concentrate in one hard PC cluster (v27023).

## Runtime (`runtime/vocr_vertical.h`, `.c`)

- `vocr_find_columns(im, flags, arena, bytes, out)` and `vocr_find_columns_rows(src, ...)` return the columns as `VocrLine`s in reading order: right to left, and top to bottom for columns that share a right edge. The boxes are padded 3 px and carry `vocr_line_dark`'s polarity flag; at most 16.
  - The method is GRAD's on the transpose: the contrast mask is stored column-major, and runs, bands, clusters, growth and the narrow-line drop work on columns.
  - Then boxes wider than 5/4 of their height (horizontal text) are dropped.
  - `VOCR_COLUMNS_VALLEYS` (the selected variant) splits a band at columns holding under a quarter of its median ink, and lengthens and clusters a column only over rows with at least max(2, width/8) on-pixels.
  - The row form reads the region's rows in two passes, then each column's box once.
- `vocr_column_rotate()` turns a column crop anticlockwise into grey. `vocr_run` then reads the strip unchanged (`VOCR_GREY8`, plus `VOCR_INVERT` when the column is dark).
- C99, no malloc, integer only.
  - Arena: 83,619 B at 960×544, 26,467 B at 480×272.
  - Code (gcc 15.2, -O2): 4,264 B Thumb-2 (VitaSDK) and 6,528 B mips32 (psp-gcc).
  - Stack: about 3.5 KB at peak (VitaSDK `-fstack-usage` frames: 1,264 B for the finder plus 2,120 B for `vocr_line_dark`, or 2,176 B for `vocr_line_dark_rows` and its helper).
- `vocr_lines.c`, `vocr_lines_rows.c`, `vocr_lines_refine.c`, `vocr_textdet.c` and `vocr.c` are unchanged: with vertical mode off nothing changes.
- **Checks** (`make -C runtime check-vertical`, in `check`):
  - strict compiles;
  - an ASan/UBSan self-test: in-memory, row and RGBA forms agree; the arena is exact; error paths; the rotation;
  - `tests/vertical_parity.py`: the C finder against the twin.

  Locally the C finder equals the twin on 425 regions (150 procedural and 275 real development regions), both flag sets and both forms, 1,977 columns.

## Speed and memory (estimates)

VOCR5-V has VOCR5-S's shape, so it costs what VOCR5-S costs per CTC frame. docs/vocr5.md's estimate for Small with #85's runtime, without prefetch credit, is 0.63 s at 333 MHz and 0.50 s at 444 MHz per 3-line box of 240 frames plus 0.1 s for the finder. That is about 2.21 ms per frame at 333 MHz and 1.67 ms at 444, calibrated to VOCR3-H15's measured 5.0-5.2 ms per frame.

| unit (development, median) | CTC frames | VOCR5-V at 333 / 444 MHz |
|---|---:|---:|
| VN column | 29 | 64 / 48 ms |
| VN region | 63 | 139 / 105 ms |
| manga column | 19 | 42 / 32 ms |
| manga balloon | 46 | 102 / 77 ms |

Plus the column finder: about GRAD's cost (GRAD measured 23-24 ms of console compute for an 883×112 region). The finder's two passes are GRAD's, and the column clustering touches each column's bits once.

Memory in the game: the model is one 0.82 MB file that replaces the job's tier model while vertical mode is on. Its W8I arena is VOCR5-S's (124,832 B), and the finder's arena is 83,619 B. Rubi adds a rotation buffer of at most one strip (56 KiB). The job stays below the Normal and Large tiers' jobs.

## Limits and disclosures

**Before the freeze.** The protocol says it was frozen "before any vertical CER, finder score or training run beyond a 200-step smoke test". That is true of CER and training, not of the finder:
- Between 07:57Z and 08:15Z, about ten minutes before the protocol commit (08:23Z), the column finder ran on the development regions as first built: MeikiOCR's boxes, with PP-OCRv6 small's readings as stand-in labels. It scored column F1 0.740 for P0.
- I looked at 12 of P0's failure regions there and designed P1's valley split and density gate after that look (F1 0.750 on the same provisional regions).
- No recognizer read a development column before the freeze. The finder rule was then registered with both variants.

So the development comparison P1 − P0 favours P1, whose design saw the development regions' geometry. The test read used P1 alone and is unaffected. The 200-step smoke test scored synthetic validation crops only.


- **Exposure:**
  - VOCR5-V starts from VOCR5-S, which trained on horizontal lines of real screenshots, among them other lines of some development and test VNs (their vertical lines were never in any pool).
  - The synthetic scenes are synth3's scene pool (test VNs excluded), which may hold other screenshots of some development or test VNs as backgrounds, never their text.
  - The manga development and test volumes share their letterer and fonts.
- **The development VN cohort has no native PS Vita page.** Vita-resolution text comes from PC pages scaled to 960×544; the test set holds the native Vita cluster.
- NOBORU's portrait mode turns the whole screen 90°, so its columns arrive as rows of sideways characters. It is not handled; use the default landscape orientation.
- Tate-chu-yoko numbers (two or three digits set side by side in one cell) and sideways Latin words are rare in the data and read poorly.
- The labels are assistant-reviewed (two blind passes), not human.
- Manga109-s: requested from the owner (issue #99); not used by any model here.
- Nothing ran on a console. The times above are estimates.

## Reproduce

```sh
H=$HERD; A=$H/shared/artifacts/vertical
.venv-ci/bin/python eval/vertical_count.py $A/count/shots_v2.json                       # the census
.venv-ci/bin/python eval/vertical_data.py pages --census ... --classes ... --out $A/data  # pages
.venv-ci/bin/python eval/vertical_data.py manga --zip pages_001.zip --vol 1 --split dev --out $A/manga --first 8 --every 9 --pages 20
HEAVY_SLOTS=12 $H/heavy.sh unshare -rn teacher/.venv/bin/python eval/vertical_data.py read --pages ... --out $A/data/reads
.venv-ci/bin/python eval/vertical_data.py sheets|decide ...                               # review batches, labels
.venv-ci/bin/python eval/vertical.py build --reads ... --labels ... --out $A/sets
.venv/bin/python synth/vertical_render.py shards --out $A/synth/shards_v1 --shards 100 --vocab data/vocab_4987_proof.txt
.venv-ci/bin/python eval/vertical_data.py shard --reads $A/data/reads_train/reads.jsonl --select agree|ps --half --out ...
.venv/bin/python train/train2.py --run vV_S_seed1 ... (eval/vertical_protocol.json "VOCR5-V".recipe)
.venv/bin/python train/export.py --ckpt $A/runs/vV_S_seed1/ckpt_last.pt --out VOCR5_V_w8.vocr --int8
.venv-ci/bin/python eval/vertical.py columns --manifest $A/sets/columns_dev.jsonl --arm rot --model 'cmd:runtime/build/vocr_cli VOCR5_V_w8.vocr --mode w8i --stdin' --out ...
.venv-ci/bin/python eval/vertical.py e2e --manifest $A/sets/regions_dev.jsonl --finder py1 --arm rot --model ... --out ...
.venv-ci/bin/python eval/vertical.py diff A.json B.json --cohort vn
```
