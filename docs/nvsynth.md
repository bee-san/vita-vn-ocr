# nvsynth: Japanese line crops from NVIDIA OCR-Synthetic-Multilingual-v1

The owner's training plan pretrains the recognizer on the Japanese subset of
[NVIDIA OCR-Synthetic-Multilingual-v1](https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v1)
(Chesler, 2026; CC BY 4.0), then fine-tunes on real console screenshots. `nvsynth/` turns that
subset into single-line crops in the format `train/train.py` already loads.

## Source

- Revision `69696a1cc543ef3a0f8e9892a89c17293e915263` (the Hub's `main` on 2026-10-09; last
  change 2026-04-20). `nvsynth/nv_fetch.py` pins it, and every raw shard is checked against the
  Hub's LFS sha256 before it is used.
- `ja/` has 67 train shards (656 GB, about 9.8 GB and 22,400 pages each), 67 validation shards
  (82 GB) and 67 test shards (82 GB). Only Japanese is read; the other five languages (most of
  the 5.45 TB) don't help a Japanese VN reader.
- A shard is HDF5: `images` (JPEG, quality 100), `annotations` (JSON per page: word, line and
  paragraph boxes with text and quads), `dimensions`, `labels` (page text), `sample_ids`.
- Pages are SynthDoG-style documents, 513-1,525 px a side: text blocks over photos, flat or
  gradient fills and translucent panels, in many fonts, sizes and colours. About 20 lines per
  page. A line box spans the font's line height, so the ink sits about 17% of the box height
  inside its top and bottom edges. 7.7% of the annotated lines are vertical (the funnel below);
  no rotated line was found.

## What `nvsynth/` extracts

`nv_extract.py` decides, per annotated line (`check()`), whether the recognizer can learn from it:

| filter | rule |
|---|---|
| horizontal | axis-aligned quad, and not taller than wide (vertical text is out of scope) |
| one word box | a two-word line is a list or contents entry (`76 …… text`, `text ____ 3`) whose leader dots or rules between the words are drawn but not in the label, often with a bigger page number; 7% of the usable lines |
| on the page | box inside the page (1 px slack) |
| size | box at least 14 px tall (about 9 px of ink) |
| text | `synth/build_corpus.py` `norm()` (the corpus convention: NFC, half-width kana to full width, 〜 to ～, no spaces); 2+ characters; every character in `data/vocab.txt`; at least one kana or kanji; at most half ASCII |

Then `process_page()` takes the page's usable lines in a random order and `crop()` cuts them.
Two sets are built from the same code:

- **One line per page** (`--per-page 1`, `$HERD/shared/artifacts/nvsynth/shards/`): the first
  line of the random order that survives the crop filters. A page has one look, so this keeps
  the 1.5M train pages as 1.5M different backgrounds, fonts and colours in a set small enough
  for `train.py` to hold in RAM (about 11 KB per line at height 32). The A/B below uses it.
- **Every line** (`--per-page 0`, `$HERD/shared/artifacts/nvsynth/full/`): all of them, from all
  three splits. The one-line set's line of a page is the first line kept here, with the same
  margins, so don't mix the two sets.

`crop()`:

1. **Ink box.** Rows of |dx| energy above 20% of the box's peak, merged over 2 px gaps, give the
   ink's top and bottom; columns of |dy| energy give its left and right (gradients, so any text
   polarity works, as in the line finders). If the profile is unclear (busy background) the
   box's median ink band is used instead (0.17 / 0.16 of the box height off the top / bottom).
   About 80% of the crops get a clean profile.
2. **Margin.** A random 10-24% of the ink height on every side. The Vita line finder adds 4 px
   around 20-26 px glyphs (15-20%), eval crops add 4 px on Vita and 2 px on PSP.
3. **Grey and height 32.** `common.to_gray` (the runtime's integer luma), then bilinear to
   height 32 like `train.py`'s `resize32`. Crops taller than 48 px (big headings) are first
   area-downscaled to a random 28-40 px, a native Vita line height, so the model never sees
   aliased 2-4x bilinear downscales the console doesn't produce.
4. **Still readable in grey.** Rejected as `faint` when max(p99 - p50, p50 - p1) of the grey
   crop is under 16, or when the ink band's grey edge energy is under 25% of its strongest
   colour channel's (`grey_ratio`: hue-on-hue text that vanishes in luma). Both thresholds were
   set from contact sheets binned by the measure: below them most lines are unreadable.
5. **Trainable.** At most 1,024 px wide at height 32 (longer lines are dropped, not squeezed)
   and CTC-feasible at stride 8 (`train.py`'s `ctc_ok`).

Every page gets its own generator seeded from (seed, shard, sample id), so the output does not
depend on how pages are split across worker processes.

`nv_fetch.py run` streams the shards: downloads run ahead of the cutting in threads, the raw files
on disk (finished or in flight) never exceed `--max-raw-gb` (36 GB for the one-line set, 100 GB
for the every-line set), a download waits while it would leave under `--min-free-gb` (70 GB)
free on the output or the raw filesystem, and each raw file is deleted as soon as its crops are
written. Downloads take that budget strictly in file order: the cutting loop consumes the files
in order, so a later file that took the space an earlier one was waiting for would deadlock the
run (the builds that made the shared artifacts predate this; in one, train_040 waited 26 minutes
while six later files took the budget). Worker processes are spawned, not forked, and a source
file whose workers die or hang for 30 minutes is redone on fresh workers (the first one-line
build hung once after its workers vanished).

In the every-line set a source file's lines are spread over `round(lines / 20,000)` shards,
`nv_<split>_NNN_bBB.npz`, by a CRC32 of (file, page, line). Every bucket is therefore a uniform
sample of that file's lines, spread over all its pages rather than a page range. A train file has
12-16 buckets, so the same bucket of every file is a sample stratified by file, not exactly a
uniform sample of the split: `full/train/nv_train_*_b00.npz` is 1/12 to 1/16 of each file
(848,808 lines, 1/14.2 of the 42 train files built by 2026-10-09), from pages all over the split,
without opening the other shards. `nvsynth/subset.py` links a chosen set of buckets (plus a
held-out shard named `zz_val.npz`, which `train.py` then validates on, and optionally other
shards such as synth2's) into one directory for `train.py --synth 'DIR/*.npz'`.

## Outputs

| set | split | sources | lines | lines per source | shards | size |
|---|---|---|---|---|---|---|
| one line per page | train | 42 (000-041) | 845,519 | 19,520-20,776 | 42 | 5.4 GB |
| one line per page | held-out (`nv_val.npz`) | validation 000-003 | 10,065 | 2,447-2,582 | 1 | 63 MB |
| every line | train | 000-041 of 66 readable (042 is damaged) | 12,036,035 | 241,007-318,556 | 597 (12-16 per source) | 74.8 GB |
| every line | validation | 0 of 67 | | | | |
| every line | test | 0 of 67 | | | | |

The one-line set stops at 42 of the 67 train sources. Source 042 is damaged (see "Damaged source
file"; the first build deadlocked on it), and by then the conductor had asked for every line of
every page, which contains these lines too. The A/B below needs only sources 000-019.

The every-line rows are the state when this PR merged (2026-10-09): train 043-066, validation
and test were still to be built, by a resumable job of the nvsynth agent
(`$HERD/shared/artifacts/nvsynth/README` has its state, log and resume command). A source's
shards and `.json` record are written only when that source is complete, so train2 can use
whatever has landed. When the build ends it writes the every-line manifest,
`$HERD/shared/artifacts/nvsynth/full/nvsynth_ja_full.json`, to be committed as
`nvsynth/manifests/nvsynth_ja_full.json` in a follow-up. Per-source line counts, shard names, sizes
and sha256s of the one-line set are in `nvsynth/manifests/nvsynth_ja.json`. On
`validation_004`, the every-line mode kept 36,501 lines from 2,989 pages: 12.2 per page, 89% of
the usable lines (the rest: 2,415 too wide, 1,876 faint), 6.2 KB per line compressed.

Over the 987,775 pages of those 46 sources (21.7M annotated lines, 22 per page; reasons are
exclusive, in `check()`'s order):

| | lines | share of annotated lines |
|---|---|---|
| vertical (or not horizontal) | 1,679,045 | 7.7% |
| two word boxes (list entries) | 1,506,252 | 6.9% |
| off the page | 387 | 0.0% |
| box under 14 px | 1,354,356 | 6.2% |
| under 2 characters | 1,029,254 | 4.7% |
| a character outside `data/vocab.txt` | 269,411 | 1.2% |
| no kana or kanji, or over half ASCII | 2,397,684 | 11.1% |
| **usable** | **13,446,380** | **62.0%** (13.6 per page) |

855,584 pages (86.6%) yielded a line. Picking it rejected 192,056 faint, 172,912 too wide and 21
CTC-infeasible candidates along the way.

## Vocab coverage (for arch)

Counted over every horizontal line of every processed page after `norm()`, cropped or not
(`nvsynth/oov_report.py`; arch's reduced vocabs from `arch/sweep`, their `<unk>` not counted as
a character):

| vocab | size | characters missed | of which kanji | ASCII | symbols, punctuation | other (Hangul, Latin letters, emoji, combining marks, kana) |
|---|---|---|---|---|---|---|
| `data/vocab.txt` | 5,354 | 0.245% | 0.014% | 0.047% | 0.130% | 0.055% |
| arch's vocab_3500 | 3,499 | 0.485% | 0.079% | 0.193% | 0.147% | 0.067% |
| arch's vocab_2500 | 2,499 | 1.331% | 0.377% | 0.741% | 0.147% | 0.067% |

That's over 305M characters in 20.0M horizontal lines (46 sources, 9,831 distinct characters).
2.70% of horizontal lines have at least one character outside `data/vocab.txt`.
`$HERD/shared/artifacts/nvsynth/oov_report.md` and `.json` list the top missing characters.

The characters `data/vocab.txt` misses on this text are web symbols and kaomoji parts, not
dialogue: en dash, caret (`(^^)`), bullet, ¥ (the vocab has ￥), €, £, backtick, combining
accents, », backslash, the combining handakuten, ▶, ·, ®, é, Д and д (`(´Д｀)`), ♬ and ♫ (the
vocab has ♪), ❤ (it has ♥ ♡). Only 0.014% of characters are kanji outside the vocab. The
reduced vocabs cost more: vocab_3500 also misses 矯 丼 炒 餃 鍼 梱 璧 酎 姜 芯 茹 (0.083% of
characters are kana or kanji it lacks), vocab_2500 麺 匿 繍 湿 雰 菌 燥 妊 濯 缶 貼 貯 (0.38%), and
both drop some ASCII. Lines with any out-of-vocab character are not in the shards, so a train2
run on a reduced vocab sees `<unk>` only for characters the shards contain and the reduced
vocab lacks.

The text is web text, not dialogue (per 1,000 characters, all train pages so far vs eval's
vita-test ground truth):

| | NVIDIA | vita-test |
|---|---|---|
| hiragana / katakana / kanji / ASCII | 32.0% / 21.0% / 24.0% / 17.5% | 55.3% / 5.8% / 26.1% / 0.2% |
| … | 0.00 (written `...`) | 17.9 |
| ！ ？ | 0.00 (written `!` `?`: 5.0 and 2.1) | 10.5, 7.4 |
| 「 」 | 2.8, 3.2 | 17.9, 17.9 |
| 、 。 | 11.5, 25.4 | 33.9, 12.5 |

The page text was evidently NFKC-normalised before rendering (no full-width ！？, no …), and the
images match their labels, so these are not label errors; but the dialogue punctuation the VN
model needs most has to come from the synthetic VN set and the real crops.

## Does it help? A/B on the real test sets

The v1 architecture (`train/model.py`, 0.76M parameters) trained for 5,000 steps (batch 96,
lr 2e-3 with v1's warm-up and cosine schedule) from the same seed on three data sets, then
scored at step 5,000 (`ckpt_last`) by eval's harness (`eval/run.py` as merged in #5: vita-test
206 lines from 16 Vita games, psp-test 245 lines from 14 PSP games), and on both held-out
synthetic shards:

- **synth**: v1's synthetic dialogue boxes, `data/synth` (39 shards, 390k lines; light text on
  dark translucent boxes only);
- **nv**: the first 20 one-line-per-page NVIDIA train shards (about 400k lines);
- **mix**: both (about 790k lines, so about half of each arm's samples).

Seeds 1 and 2, `train.py` from arch's branch (`8af9144`: the same model and schedule as v1,
plus `--steps`, `--seed` and `--out`; this repo's `train.py` now has `--out` too), run by
`nvsynth/ab.sh` through `heavy.sh`, compared by `nvsynth/ab_compare.py` (per-seed CER, the seeds'
errors pooled, and a paired bootstrap over games for the difference to the synth arm). nv-val is
the first 3,000 lines of `nv_val.npz` (the harness's synth-val limit). Its pages are held out,
but its text only partly: 29% of `nv_val`'s lines have a label that also occurs in the one-line
train set, mostly sentence-end fragments such as ます。 (86 distinct ones of 12 or more
characters, web boilerplate). Training is deterministic: the reviewer retrained mix seed 1 and
got a bit-identical checkpoint, and the synth seed-1 checkpoint is bit-identical to arch's
independent `base_s1` run.

CER per seed (seed 1 / seed 2), with the exact-line share pooled over both seeds:

| arm (training data) | vita-test CER | vita exact | psp-test CER | psp exact | synth-val CER | nv-val CER |
|---|---|---|---|---|---|---|
| synth: v1's synthetic boxes (390k lines) | 28.51% / 30.03% | 39.1% | 51.27% / 53.67% | 25.9% | **1.25% / 1.24%** | 86.66% / 88.71% |
| nv: NVIDIA crops (400k) | 20.94% / 17.47% | 24.8% | 23.82% / 21.35% | 24.3% | 13.04% / 11.19% | **13.62% / 13.02%** |
| mix: both (790k) | **9.75% / 10.18%** | **49.0%** | **14.98% / 14.70%** | **42.4%** | 2.50% / 2.50% | 14.63% / 14.30% |
| *v1 for reference (synth, 8 epochs = 32.5k steps)* | *23.75%* | *52.4%* | *46.45%* | *30.2%* | *0.32%* | |
| *JIS-trimmed int8 PP-OCRv5 mobile* | *2.30%* | *79.1%* | *5.34%* | *66.1%* | | |

The reference rows are `docs/results.md`; the owner's bar quotes 2.30% / 5.58%, the int8
model's psp-test CER before #5's psp-test fixes.

Pooled over both seeds, CER difference to synth with a 95% bootstrap interval over games:

| | vita-test | psp-test |
|---|---|---|
| mix − synth | −19.3 points (−38.2 to −5.5); better in 10 of 16 games, worse in 5 | −37.6 points (−58.9 to −17.3); better in 11 of 14, worse in 3 |
| nv − synth | −10.1 points (−31.1 to +5.5); better in 6 of 16, worse in 10 | −29.9 points (−52.4 to −8.6); better in 8 of 14, worse in 6 |
| mix − nv | −9.2 points (−13.0 to −5.7); better in 15 of 16, worse in 0 | −7.8 points (−10.4 to −5.3); better in 13 of 14, worse in 1 |

The intervals resample games over the pooled errors, so they cover which games are in the test
sets, not training noise beyond these two seeds; the per-seed columns show that noise (up to 3.5
points on vita-test for the nv arm).

Per style (CER, both seeds pooled, eval's style tags as merged):

| style | vita lines | synth | nv | mix | psp lines | synth | nv | mix |
|---|---|---|---|---|---|---|---|---|
| bright-on-dark | 163 | 13.14% | 21.28% | **10.42%** | 135 | 22.60% | 28.35% | **18.38%** |
| dark-on-light | 43 | 88.59% | 11.59% | **8.30%** | 110 | 90.69% | 15.21% | **10.31%** |
| outlined | 74 | 34.88% | 26.29% | **13.75%** | 69 | 30.47% | 29.80% | **19.05%** |
| mincho | 47 | **2.96%** | 11.69% | 3.62% | 90 | 46.85% | 15.70% | **8.89%** |
| coloured | 19 | 55.90% | 36.15% | **32.82%** | 20 | 69.86% | 25.70% | **23.36%** |
| coloured-name | 8 | 47.62% | 42.86% | **26.19%** | 7 | 95.83% | **39.58%** | 41.67% |
| name | 53 | 28.78% | 42.15% | **28.49%** | 62 | 72.25% | 51.67% | **45.22%** |
| furigana | 3 | 18.75% | 33.59% | **16.41%** | 20 | **66.50%** | 75.73% | 67.48% |
| small (PSP) | | | | | 106 | 57.66% | 13.16% | **8.12%** |

What this says:

- **The NVIDIA crops help, a lot, but as a mix.** At the same 5,000 steps, adding them to the VN
  boxes cut vita-test CER by about two thirds and psp-test by about 70%, on both seeds, beating
  v1's full 8-epoch run on both sets. The gains are where v1's data has nothing: dark-on-light
  text (89% → 8% on Vita, 91% → 10% on PSP), small PSP glyphs (58% → 8%), and serif and outlined
  faces.
- **They don't replace the VN-style data.** Alone, they lose bright-on-dark accuracy (13% → 21%
  on Vita) and exact lines, because light text on dark translucent boxes is rare in the set
  (18% of crops) and its text has no dialogue punctuation. Mix beats nv in 15 of 16 Vita games
  and 13 of 14 PSP games.
- **Still unsolved by either:** names and name plates (28-45%) and lines with furigana above them
  (67% on PSP). They need real crops (realdata, teacher) or synth2's name-box and furigana styles.
- Each model is best on its own synthetic validation set, and synth-only is useless on the
  NVIDIA one (88% CER). Synthetic validation says nothing about the real sets here, which is why
  AGENTS.md insists on eval's real sets.
- All of this is 5,000 steps of a 0.76M-parameter model; train2's longer pretraining and real
  fine-tuning are where the bar (2.30% / 5.58%) gets attacked. For train2: pretrain on NVIDIA
  crops **mixed with** VN-style synthetic boxes (synth2's renderer now covers dark-on-light,
  outlines, names and furigana too), not on NVIDIA crops alone.

## Damaged source file

`ja/train/train_042.h5` at this revision can't be read: HDF5 reports "wrong B-tree signature" on
its root group, so not even its dataset names can be listed. Its 10,103,761,681 bytes match the
Hub's LFS sha256 (`85df20a8…`), so the file is damaged as published, not in transit. The first
two builds deadlocked on it (an exception in the cutting loop left the download threads waiting
for budget). `nv_fetch.py` now records an unreadable source in its JSON with the error, deletes
the raw file, and goes on; the manifests list it under `skipped_sources`. Pages that raise are
counted (`stats.errors`) and skipped, and any other error aborts the waiting downloads instead of
hanging. The one-line set stops just before it, at 000-041.

## Label quality (spot check)

The labels are the renderer's own text, so the question is whether the crop shows exactly that
text. 102 random kept crops of `validation_001` (the final filters, seed 23), read by eye on
contact sheets at 1x, with a second sheet for the right end of the 16 lines longer than 30
characters or 680 px:

- 0 crops whose visible text differs from the label;
- 1 crop with one character partly hidden behind a picture (ウイスキー ジン チュー**ハ**イ);
- 2 crops where the text sits small inside a crop about twice its height (the median-trim
  fallback on a busy background);
- a dozen faint lines (light grey on white, text over photos), all still legible at 1x.

A second, independent check (the reviewer, 60 random kept crops: 20 each of `nv_train_007`,
`nv_val` and `full/train/nv_train_030_b05`, read on sheets at 1.2x with 2.5x zooms of the doubtful
ones) agrees that the labels are the rendered text: no crop shows a wrong character. But some
labelled text cannot be seen in grey:

- 2 of 60 crops (3%) are mostly invisible: light text runs into a light photo, so of
  「日、TBSの29歳・テレビ局...」 only 「日、TBSの2」 shows, and of
  「昨日は最近人気の…【FF14日記】」 little more than 「【…日記】」;
- 1 crop's final 。 is invisible, and 1 crop is a lone small kana (ぁ。) that the tight crop makes
  indistinguishable from あ.

The faint filter misses these because it measures the whole crop, and part of each line has
contrast. A windowed measure catches them: the share of 32-px windows whose grey range
(p98 - p2) is under 16 flags exactly those 2 of the 60 at 25% or more, and flags 2.5% of `nv_val`
(1.2% at 50%). Such lines teach the model to emit text it cannot see. A windowed contrast filter is
left as a follow-up because it changes the built sets; until then this is a known label-noise
floor of about 2-3% of lines.

An earlier sheet, before the one-word filter, had 2 label errors in 34 lines, both list entries
whose leader dots (`76 ……… そのため`, `増えて ……… l`) were drawn but not labelled. That would
have taught the model to ignore ……, VN dialogue's most common punctuation, so two-word lines
are now dropped (7% of usable lines; with one line per page the yield barely changes).

## How the NVIDIA crops differ from real console crops

Contact sheets (`nvsynth/nv_sheet.py compare`, kept local): NVIDIA crops next to real Vita and PSP
line crops from realdata's screenshots, both grey at height 32, the real ones cut at their ink
box plus eval's pad (4 px Vita, 2 px PSP) and restricted to lines two OCR readings agree on.

| | NVIDIA crops | real Vita | real PSP |
|---|---|---|---|
| polarity | 80% dark-on-light (2,045 of 2,542 in `validation_001`; 82% of the one-line set) | 52% light-on-dark (realdata agreed lines); 79% of eval's vita-test lines are bright-on-dark | 66% light-on-dark (realdata); 55% of psp-test |
| ink height at native size (p5 / p50 / p95) | 11 / 20 / 56 px (one-line set; every-line bucket 00: 10 / 18 / 41) | 17 / 26 / 42 px | 13 / 17 / 24 px |
| characters per line (p50 / p95) | 9 / 32 | 12.5 / 24 (vita-test) | |
| image quality | JPEG q100 renders, crisp anti-aliasing | VNDB JPEG screenshots: ringing, chroma blur | half the resolution: 12-17 px glyphs, blur |
| look | web and print layouts: flat fills, gradients, photos, translucent panels; many document fonts and weights (gothic, mincho, rounded, pop, brush, italic, very heavy display faces) | game gothics and rounded faces (Rodin, Seurat-like), minchos, heavy outlines and drop shadows, translucent dark boxes, coloured names, furigana | the same at small sizes, some bitmap-like faces |
| text | web text: product names, prices, dates, Latin, digits, 【】, katakana loanwords, fragments cut mid-word by the page's wrapping | dialogue and narration: kana-heavy speech, 「」『』, ……, ！？, ～, names | same |

So the NVIDIA set brings what v1's light-on-dark synthetic boxes lack (dark text on light
backgrounds, a wide font and weight range, busy photo backgrounds, digits and Latin), and lacks
what VN boxes have most: outlined or shadowed light text on translucent dark boxes, low-resolution
glyphs, furigana, and dialogue punctuation. Vertical text (7.7% of NVIDIA's annotated lines) is
dropped by design. That is why the A/B above also tries a mix with VN-style synthetic data, and
why the owner's plan fine-tunes on real crops afterwards.

## JSSODa

[llm-jp/JSSODa](https://huggingface.co/datasets/llm-jp/JSSODa) (Sasagawa et al., LREC 2026):

- **Licence:** CC BY 4.0, so usable with attribution, like the NVIDIA set.
- **Content:** 17,991 train and 2,246 validation pages of LLM-written expository prose, rendered
  black on white in 1-4 columns, 18-30 px, 24-28 characters per line (half that in multi-column
  layouts), with an optional larger title. 8,958 of the train pages (50%) are vertical.
- **Lines:** the dataset has only the page text: no line boxes and no per-line text
  (`renderer.py` in llm-jp/eval_vertical_ja breaks lines by measured glyph advance, which the
  dataset doesn't record). Lines could be cut from the horizontal half by row projection (clean
  white pages), but each line's text would then have to be recovered by aligning an OCR reading
  of every line to the page text. About 9k pages, roughly 150k lines.
- **Expected value:** low. It is the plainest corner of what the NVIDIA set already covers
  (black text on white in document fonts), with none of the VN look and prose rather than
  dialogue. Not built and not measured here: the alignment step costs CPU the shared box is
  short of, for data unlikely to move the real-set CER. Worth revisiting only for font variety
  (its free-fonts.jp faces) if a later error analysis shows font coverage is the bottleneck.
