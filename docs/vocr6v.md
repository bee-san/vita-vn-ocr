# VOCR6-V: the owner's Manga109 download, and vertical text from the hooked vertical game (issue #159)

The owner (2026-10-28 07:25Z): "I have obtained manga109 by applying, we can use it to improve our vertical text ocr."
This document covers four things:
- what that download holds;
- what could be taken from it;
- VOCR6-V, VOCR5-V's successor (#99, `docs/vertical.md`);
- the record of the pre-registered protocol (`eval/vocr6v_protocol.json`, frozen before any score of this
  workstream; Rubi #335's figures for the incumbent through the same job were known) and its results.

Every number is a host measurement unless it is marked as an estimate. Nothing here has run on a console.

## Answer first

- **The download is not the annotated Manga109.** It is Manga109's super-resolution benchmark version:
  - the 109 volume **covers** (810×1080);
  - 128×128 / 32×32 tiles cut from those covers;
  - no interior pages, no text boxes, no transcriptions.

  So it holds no speech balloons. Its vertical text is 83 cover columns of titles, authors and blurbs, and 24 of them
  could be labelled.

  The official release adds the volumes' 21,142 interior pages with XML annotations: every text box with its
  transcription, and frame, face and body boxes. That is Hugging Face `hal-utokyo/Manga109-s` (87 volumes, for
  non-academic projects such as this one), and its terms allow publishing models trained on it. **To get it the owner
  applies on that page** (the form asks for an outline of the use; the Manga109 team approves by hand in "2-3 days or
  a week") and saves a read token at `~/.cache/huggingface/token` on the box. A suggested outline is in
  `herd-synth3/vertical.request-manga109s`.
- **VOCR6-V replaces VOCR5-V.** It has the same shape, size and arena: 819,792 B of int8 weights, 124,832 B W8I arena,
  4,987 classes. So the console's memory and the est. time per column are VOCR5-V's. It is VOCR5-V fine-tuned on:
  - the owner's hooked vertical game, Yunohana SpRING! (PCSG00653): its columns' box crops and the column finder's own
    crops of its pages;
  - a new synthetic render;
  - VOCR5-V's real columns.
- **Results through Rubi's own job** (`ocr_vertical = on`, tight regions; three seeds, pre-registered rule):

  | set | VOCR5-V | VOCR6-V (recipe, mean of 3 seeds) | difference [95%] |
  |---|---:|---:|---|
  | Yunohana's held-out pages (hval, 35 pages) | 7.92% | 4.64% | **−3.28 [−4.48, −2.14]** |
  | #103's development VN regions (25 clusters, other games) | 17.58% | 16.13% | **−1.45 [−2.25, −0.58]** |
  | #103's development manga (*Black Jack ni Yoroshiku* vol. 1) | 4.18% | 3.69% | **−0.49 [−1.14, −0.06]** |
  | #103's test VN, read once (16 clusters) | 7.26% | 7.05% (the released blob) | −0.20 [−0.69, +0.24] |
  | #103's test manga, read once (vol. 13) | 2.88% | 2.74% | −0.14 [−0.49, 0.00] |

  The released blob is the recipe retrained with hval folded in, as VOCR6 does. It reads development VN at 16.13% and
  manga at 3.91%.
- **The Manga109-s arm** (V6hm: plus the 19 labelled Manga109-s cover columns) **was not adopted.** Its development
  manga is +0.14 [−0.09, +0.55] against V6h (rule: resolved below 0). So no Manga109 material is in a released model.
- **What is left is mostly the column finder.**
  - On the held-out pages the recognizer reads the hooked columns' box crops at 0.35% (seed mean), and the finder's
    own crops of the same pages at 2.60%.
  - The finder finds 89% of the columns in a tight box, for every model.
  - In a box drawn once around the text box's place, the finder's false columns (precision 0.55) cost about 7 points:
    11.70% against 4.64% tight.

## The download (measured 2026-10-28)

`/home/skerraut/manga109databse.zip`: 166,820,233 B, sha256 `0cbea252…52d967f2` (checked). It holds JPEGs and a 4-line
ImageMagick tiling script, and no licence text, README or annotation file. Its manifest is
`data/manifests/manga109_download_20261028.json`: every cover's name, size, dimensions, sha256 and Manga109-s flag,
and each tile directory's file count, bytes and list hash.

| directory | files | what |
|---|---:|---|
| `Originals` / `ValidOriginals` | 89 / 20 | one colour cover per volume, 810×1080 (one 742×1072): 109 volumes |
| `HR` / `ValidHR` | 2,670 / 600 | 30 random 128×128 tiles of each cover (one `BatchTile.sh` too) |
| `LR` / `ValidLR` | 2,670 / 600 | the same tiles at 32×32 (4× smaller) |

87 covers are of Manga109-s volumes and 22 of volumes in Manga109 only (manga109.org's title list, 2026-10-28).

**Licence.** Which terms the owner's application came with is not known: none came in the zip, and none was passed
on. The published ones (manga109.org; the Hugging Face cards) are:

- **Manga109:** academic, non-commercial use only; no part may be redistributed to third parties.
- **Manga109-s:** machine-learning use is allowed, and so is publishing the results, pre-trained models included,
  when the published work says clearly that Manga109-s was used. Its images may not be redistributed, sold or
  treated as products.

The download holds all 109 volumes' covers, the 22 Manga109-only ones included, which points to the Manga109
(academic) route. So until the owner confirms Manga109-s terms (or applies for Manga109-s, below), the whole
download is treated as Manga109 under the academic terms: no training data for a released model. V6hm trained on 19
Manga109-s cover columns (below); it was not adopted, and its blobs and checkpoints stay private, never released.

How the files were handled:
- They stay on the box (`shared/private/manga109`, 700/600), and are never committed, uploaded or sent to Codex,
  Lens or any other service.
- MeikiOCR and PP-OCRv6 small read them offline (`unshare -rn`).
- Claude (the fleet's assistant reviewer, the model this workstream runs on) viewed 8 contact sheets of the covers'
  columns, as the task and the fleet's rules direct for its own sheet reviews. Claude is a hosted model, so those
  sheets' pixels reached its API as part of that review; they went nowhere else.

The private licence note is `shared/private/manga109/LICENCE_NOTE.md`.

### What the covers gave (`data/manga109_covers.py`)

**The pages.** Each cover was shown as NOBORU shows a page in its fit-to-width mode
(`eval/vertical_data.noboru_views`): 327 windows of 960×544.

**The reading.** MeikiOCR's detector and PP-OCRv6 small read them with #103's reader (`eval/vertical_data.py read`).
- 404 horizontal lines and **83 vertical columns** on 33 of the 109 covers.
- Mostly stylised title logos, author names and a few blurbs, often cut by a window edge.

**The labels.** One Claude review pass on 8 contact sheets (`reviewer_kind: assistant`). A column was labelled only
when the decided text equals one engine's reading.

| outcome | Manga109-s | Manga109 only |
|---|---:|---:|
| labelled | 19 (74 characters) | 5 |
| cut by the window or the cover's edge | 31 | 8 |
| logo or sideways Latin (`ui`) | 8 | 3 |
| no engine read the decided text | 4 | 1 |
| furigana, unsure | 2 | 2 |

**Detection, report only** (`manga109_covers.py detect`). These counts are for a check, not supervision: the covers
are no balloon data.

| finder | vertical boxes | MeikiOCR's 83 columns matched (IoU ≥ 0.5) | the 24 labelled columns matched |
|---|---:|---:|---:|
| textspot8, Rubi's default spotter (its vertical flag) | 132 | 14 | 4 |
| the column finder P1, the whole 960×544 window | 3,101 | 33 | 8 |

On a whole cover, P1 takes the art's edges for columns. Rubi's guidance stands: draw the box around the text. The
column finder has no weights, so there was nothing to train on the covers for it.

## Data of VOCR6-V

- **`vsyn`:** `synth/vertical_render.py` **generation 3**, 40 shards × 10,000 columns, seed 20261028
  (`data/manifests/vertical_shards_g3.json`, 3,034,664,986 B). Generation 2 left out the vertical sets' pages from
  the scene pools. Generation 3 leaves out every screenshot of the 78 VNs of the vertical development and test sets
  and the owner's private test VN (Trick x Logic). Generation 2 still held 20 such screenshots of 9 of them,
  docs/vertical.md's "Limits". The fonts, texts and looks are VOCR5-V's.
- **`vra`, `vps`:** #103's real training columns, unchanged: 1,424 agreed crops and 3,066 PP-OCRv6-small crops.
- **`vhk`:** the owner's hooked vertical columns, `vocr6-ingest` pools v1: Yunohana SpRING!'s 1,461 train columns of
  798 train pages, box crops turned anticlockwise, in console polarity.
- **`vcil`:** the same pages' **console columns** (`data/vertical_hooked.py cil`). This is the vertical version of
  VOCR6's crop-in-the-loop factor S2cil, the one data factor VOCR6 adopted. Each crop is made as follows:
  - the column finder P1 (the C runtime's bit-exact twin) runs on each page's tight region (its columns' union plus
    12 px, Rubi's `vertical_eval.py` unit);
  - a found box is kept when it holds the centre of every glyph cell of exactly one hooked column and nothing of
    another text box;
  - it is labelled with that column's hooked text, and cut, turned and inverted as Rubi's job does.

  Of the train pages' 1,498 columns, 1,286 came out whole; the finder's other boxes were 179 partial and 2 empty
  (`eval/vocr6v_data_20261028.json`, `console_columns`: a rebuild gives the same shards byte for byte). A
  contact sheet of 40 random crops, checked by eye (Claude), had no wrong label.
- **Held out:** Yunohana's 35 hval pages (50 columns) and the 30 guard pages around them, until the fold-in run.
  #103's development and test sets are in no source.

## Protocol (`eval/vocr6v_protocol.json`, tag `vocr6v-protocol-2bbb9e1`)

**Arms.** Each arm is a fine-tune of VOCR5-V's checkpoint: 10,000 steps of 96, lr 5e-4 cosine to 5%, warmup 300,
EMA 0.999. The last checkpoint is the model. Seeds 1, 2 and 3.

| arm | vsyn | vra | vps | vhk | vcil | Manga109-s covers |
|---|---:|---:|---:|---:|---:|---:|
| V6h | 0.60 | 0.10 | 0.05 | 0.15 | 0.10 | |
| V6hm | 0.57 | 0.10 | 0.05 | 0.15 | 0.10 | 0.03 (38 crops: 19 columns, each also at half scale\*) |

**Scoring: Rubi's own job.**
- The tool: Rubi main `eef0fc41`'s `host/page_ocr.c` (`vjo-page-ocr --vertical on --line-fix on`, Rubi's defaults
  otherwise), with vita-vn-ocr runtime `f8a1856` (Rubi's pin), through Rubi's `tools/vertical_eval.py`.
- The build is measurement-only (`eval/vocr6v_rubi_measure.patch`):
  - the job skips the SHA-256 check of every file it reads (their sizes are still checked), so the candidate blob
    can sit in `VOCR5_V_w8.vocr`'s place. Every other file of each arm's models directory was checked instead: all
    15 equal Rubi `eef0fc41`'s pins (`eval/vocr6v_data_20261028.json`, `scoring_models_dirs`);
  - for hval, the 1 MB LM tier is the screening build `h64_1000k`, whose text holds no hval page: the shipped
    `vocr_lm6_o3_1000k` holds hval's text (docs/vocr6-lm.md).
- The metric is dialogue CER per unit (`eval/vocr6v.py`, `vertical_eval.py`'s own `score_unit`).
- The units:
  - **hval:** Yunohana's 35 tight hval units;
  - **dev VN:** #103's 84 VN development regions (25 series clusters; PSP shown 2×, PC fit to 960×544), shipped LM;
  - **dev manga:** its 97 manga regions (18 book pages).
- The incumbent through this path reproduces Rubi #335's own figure on #103's development regions (12.29%).

**Rule (`decision.adopt`, VOCR6's).** Y over X iff three things hold:
- the hval interval of mean(Y) − mean(X) lies below 0;
- every seed pair's hval difference is below 0;
- neither the dev VN nor the dev manga interval lies above 0.

V6hm over V6h needs, in addition, the dev manga interval below 0. The adopted arm's seed 1, retrained with hval and
guard folded in, ships unless its dev VN or manga interval against its sibling lies above 0. Intervals: 10,000
bootstrap resamples, seed 0, by page (hval), series cluster (VN) and book page (manga).

## Results (`eval/vocr6v_decisions_20261028.json`)

The decisions record was committed before the test read (tag `vocr6v-selection-d6499da`). Since then it has
only gained each comparison's bootstrap `unit` (page, cluster, book), no value. The counts and report-only checks
below are in `eval/vocr6v_data_20261028.json`.

**V6h over VOCR5-V: adopted.**

| | VOCR5-V | V6h seeds 1 / 2 / 3 | mean difference [95%] | pairs |
|---|---:|---:|---|---|
| hval | 7.92% | 4.84 / 4.55 / 4.55% | −3.28 [−4.48, −2.14] | −3.08, −3.37, −3.37 |
| dev VN | 17.58% | 16.35 / 16.04 / 16.00% | −1.45 [−2.25, −0.58] | |
| dev manga | 4.18% | 3.71 / 3.64 / 3.71% | −0.49 [−1.14, −0.06] | |

**V6hm over V6h: not adopted.**

| | V6h | V6hm seeds 1 / 2 / 3 | mean difference [95%] |
|---|---:|---:|---|
| hval | 4.64% | 4.40 / 4.55 / 4.55% | −0.15 [−0.37, +0.05] |
| dev VN | 16.13% | 16.09 / 15.96 / 15.78% | −0.19 [−0.75, +0.47] |
| dev manga | 3.69% | 3.78 / 3.78 / 3.91% | +0.14 [−0.09, +0.55] |

**The fold-in (V6h seed 1 with hval and guard folded in): ships.** Against V6h seed 1:
- dev VN 16.35 → 16.13% (−0.22 [−0.76, +0.15]);
- dev manga 3.71 → 3.91% (+0.20 [−0.07, +0.53]).

Its hval, 3.67%, is not held out. The released blob is `VOCR6_V_w8.vocr`, sha256 `40485fce…2eb4eea81`.

**Diagnostics (report only):**

| | VOCR5-V | V6h seeds 1 / 2 / 3 | fold-in |
|---|---:|---:|---:|
| hval, the hooked columns' box crops (W8I greedy) | 1.98% | 0.61 / 0.15 / 0.30% | (trained) |
| hval, the finder's crops of the same pages | 9.09% | 2.78 / 2.41 / 2.60% | (trained) |
| hval, the box drawn once (one box per place, 34 units) | 20.93% | 11.90 / 12.35 / 10.84% | (trained) |
| dev, PSP VN regions / PC VN regions | 1.20 / 23.23% | 0.51-0.86 / 21.34-21.69% | 0.86 / 21.39% |
| dev, #103's fixed VN columns (its primary metric) | 10.64% | 9.14 / 9.76 / 9.54% | 9.54% |
| dev, #103's fixed manga columns | 2.70% | 2.76 / 2.76 / 2.70% | 2.63% |

#103's fixed-column metric agrees on VN: V6h − VOCR5-V is −1.50, −0.88 and −1.10 points, each interval below 0
(`eval/vocr6v_data_20261028.json`, `fixed_columns_dev`).
Manga columns are level. Column recall on the tight units is 0.891 on hval for every model and 0.83 on development
regions (PC 0.69, PSP 1.00): the finder did not change.

## The test set, once (`eval/vocr6v_test_20261028.json`)

After the selection was committed and pushed (tag `vocr6v-selection-d6499da`), `eval/vocr6v_test_once.py` read
#103's vertical test set once, 2026-10-28T09:59:14Z to 09:59:29Z. A dry run before it, on development data
relabelled as test, read no test data.

**The attempt record hashes 5 of the read's 7 inputs:** the private sets, both blobs and the tool. The script keyed
the four sets by file name, and the private sets share the public ones' names, so the public manifests' hashes were
lost to that collision. The script now keys every input apart, and a test checks it. As provenance, the public
manifests' sha256 today equal #103's attempt record's (`731051b9…`, `4ad6df40…`), and the files were last modified on
2026-10-18 (`eval/vocr6v_data_20261028.json`, `test_inputs_provenance`). The read used those bytes.

Report only:

| test, dialogue CER | VOCR5-V | VOCR6-V (released) | difference [95%] |
|---|---:|---:|---|
| VN through Rubi's job (16 clusters, 2,453 characters) | 7.26% | 7.05% | −0.20 [−0.69, +0.24] |
| VN, the public VNDB pages (15 clusters) | 8.01% | 7.74% | −0.27 [−0.82, +0.20] |
| VN, the owner's Trick x Logic PSP frames (257 characters) | 0.78% | 1.17% | +0.39 (1 cluster) |
| VN without Yunohana SpRING! (seen: see below) | 7.35% | 7.14% | −0.21 [−0.71, +0.23] |
| manga through Rubi's job (12 book pages) | 2.88% | 2.74% | −0.14 [−0.49, 0.00] |
| #103's fixed VN columns | 7.17% | 7.13% | |
| #103's fixed manga columns | 0.86% | 0.86% | |

On the test set VOCR6-V is level with VOCR5-V: no interval excludes 0. #103's test VN cohort reads much better than
its development cohort (7.26% against 17.58% for VOCR5-V): #103 found the development errors concentrated in its
hard PC clusters, so the test cohort has less error to remove.

## Limits and disclosures

- **One vertical hooked title.** hval is 35 pages of one game, so its gain is that game's. The unseen-game signal is
  #103's development set, where both cohorts improved (VN by 1.45 points). On the test set the change is level.
- **Yunohana SpRING! is seen in #103's test set.** It is one PS Vita page there (30 characters) and the hooked title
  VOCR6-V trained on. Rubi's 1 MB LM holds its hooked text too. The table gives the test VN without it.
- **Repeated text in hval.** 4 of hval's 55 columns (23 of 682 characters) have the same text as a train page's
  column. Without the 4 pages that hold them, V6h − VOCR5-V on hval is −3.39 [−4.61, −2.24], every pair below 0
  (`eval/vocr6v_data_20261028.json`, `hval_repeated_text`), so the result does not rest on them.
- **The cover labels** come from one Claude review pass plus an engine's agreement, not #103's two blind passes. They
  are in no released model.
- **\* A deviation from the protocol:** V6hm's cover source holds each of the 19 columns at full and at half scale
  (`manga109_covers.py shard --half`, as #103's real columns are), which the frozen protocol does not say. It was
  decided at 08:16Z, after the protocol and before any V6hm score. V6hm was not adopted.
- **The finder's bias in development.** P1's design saw #103's development regions (docs/vertical.md, "Before the
  freeze"). The finder is the same for both arms here, so the comparison between recognizers is not affected.
- **Speed and memory** are VOCR5-V's (same shape): est. 140 / 106 ms per median VN region at 333 / 444 MHz on one A9
  core. Nothing has run on a console.

## Reproduce

```sh
H=$HERD; W=$H/shared/artifacts/vocr-manga109; P=$H/shared/private/manga109/work; A=$H/shared/artifacts
# the covers
.venv/bin/python data/manga109_covers.py views --root $P/../extracted/manga109 --inventory $P/inventory_20261028.json --out $P
HEAVY_SLOTS=12 $H/heavy.sh unshare -rn teacher/.venv/bin/python eval/vertical_data.py read --pages $P/pages_m109.jsonl --out $P/reads
teacher/.venv/bin/python eval/vertical_data.py sheets --reads $P/reads/reads.jsonl --out $P/review --mode pass1
teacher/.venv/bin/python data/manga109_covers.py labels --reads $P/reads/reads.jsonl --review $P/review --out $P/labels_m109.jsonl
teacher/.venv/bin/python data/manga109_covers.py shard --reads $P/reads/reads.jsonl --labels $P/labels_m109.jsonl --out $P/shard_m109s.npz --half
.venv/bin/python data/manga109_covers.py detect --reads $P/reads/reads.jsonl --blob textspot8_t64n8c16_w8.vtsp --labels $P/labels_m109.jsonl
# the data
.venv/bin/python synth/vertical_render.py shards --out $W/synth/shards_g3 --shards 40 --seed 20261028 --gen 3 --vocab data/vocab_4987_proof.txt
.venv/bin/python data/vertical_hooked.py cil --pagepool $A/vocr6-ingest/pagepool/v1/PCSG00653.jsonl --pages $A/vocr6-ingest/pages --splits train --out $W/hooked/vcil_train.npz
# a run (eval/vocr6v_protocol.json "recipe" and "arms"), its export and its scores
.venv/bin/python train/train2.py --run v6h_s1 --init $A/vertical/runs/vV_S_seed1/ckpt_last.pt --steps 10000 ... --src ...
.venv/bin/python train/export.py --ckpt $W/runs/v6h_s1/ckpt_last.pt --out v6h_s1.vocr --int8
python3 RUBI/tools/vertical_eval.py run --units yuno_hval.jsonl --tool vjo-page-ocr --models M --tool-args "--vertical on --line-fix on" --out R.jsonl
.venv-ci/bin/python eval/vocr6v.py diff --units U.jsonl --x X.jsonl --y Y.jsonl --unit page|cluster|book [--manifest eval/manifests/vertical-dev-regions.jsonl --cohort vn|manga] --paired
.venv-ci/bin/python eval/vocr6v.py decide --protocol eval/vocr6v_protocol.json --comparison V6h_vs_VOCR5V --hval H.json --vn V.json --manga G.json
```
