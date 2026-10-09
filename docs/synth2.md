# synth2: synthetic lines that look like PS Vita and PSP visual novels

v1 was trained on one look only: bright text on a dark box. The overlay's line finder only sees
bright pixels, so the old renderer could only cut bright text. On real screenshots that shows:
v1 reads light-on-dark lines at 9.9% CER, but dark-on-light lines at 58% (provisional hand set,
below). Under the owner's training plan (RULES.md, 2026-10-09 10:55Z), the Japanese subset of
NVIDIA OCR-Synthetic-Multilingual-v1 is the broad-coverage pretraining set, so synth2 concentrates
on what a generic synthetic set lacks: dialogue boxes, outlines, translucent boxes, PSP-sized
text and busy scenes.

## What render.py draws

A sample is a dialogue region, like the one a user selects, with one to eight horizontal lines.
Shares are over the 400,000 rendered lines (`$HERD/shared/artifacts/synth2/style_stats.json`).

| | |
|---|---|
| Screens | Vita 960x544 (58%), PSP 480x272 at native size (30%; vector text 12-16 px, mostly 12-14), PSP upscaled 2x as on a Vita (12%, nearest or bilinear) |
| Layouts | ADV box (50%), NVL page (14%), band across the screen (12%), bare text over the scene (9%), choice bars (9%), speech balloon (5%) |
| Polarity | light on dark (57%), dark on light (23%), light text outlined on a light box (13%), white on a saturated box (5%), dark outlined text (2%) |
| Text effects | outlines of 1-3 px, sometimes doubled (42% of lines); hard or soft drop shadows (37%); glows (5%); vertical fill gradients (10%); accent-coloured keywords or whole lines (8%) |
| Boxes | translucent (half of the dark boxes at 35-70% opacity) or opaque, vertical alpha fades, colour gradients, textures (checks, stripes, dots, noise), borders, inner frame lines, corner ornaments, glossy top bands |
| Names | a separate name tab above the box, a name line inside it, or an inline name before the bracket (regions holding 19%, 21% and 6% of the lines); often in an accent colour; "？？？" and 【】 forms; name crops are 14% of the lines |
| Distractors (never labelled) | furigana over kanji runs (gap 0-3 px) and emphasis dots, page cursors (▼, chevron, ring with triangle, diamond, sparkle, star, arrow, snowflake, feather) inline or in the box corner, the faint next glyph of a typewriter-partial line |
| Line breaking | kinsoku: closing marks and small kana hang (up to two), opening brackets never end a line, …… and ―― are not split, optional breaks after 、。！？, continuation lines after 「 indented by one character, full-width spaces after ！ and ？ (not labelled) |
| Fonts | 34 OFL families: gothic 45%, mincho 19%, rounded 18%, pixel 10%, display 4%, pen 4%. Pixel faces (DotGothic16, Fusion Pixel 10 and 12 px) are drawn without anti-aliasing at multiples of their pixel size; vector faces are also drawn without anti-aliasing on 45% of PSP regions |
| Capture | sub-native rendering then upscale (Vita, 12%), blur, noise, JPEG with a random 8x8 grid phase (60% of PSP and 50% of Vita regions) |
| Scenes | 85% crops of the upper 62% of training-side Vita and PSP screenshots (mirrored half the time), otherwise gradients with blobs, clutter and blur |

Labels contain only vocab characters and no spaces. A typewriter-partial line is labelled
with its visible prefix. Furigana, emphasis dots, cursors and the faint glyph never appear in a
label.

## Crops follow the line finder

`render.py --finder` picks how regions are cut into line crops:

| `--finder` | what it is |
|---|---|
| `mask` (default) | `mask_finder.find_lines_mask`: the overlay's projection rules applied to the renderer's exact text-ink mask, plus the furigana rule (runs under 0.55x the median height are dropped). Rules: rows with at least 2 ink pixels form runs, a gap of 3 rows ends a run, runs under 8 rows are ignored, a run over 88 rows or more than 8 lines rejects the region, boxes are padded by 4 px (5 below). |
| `grad` | `common.find_lines_ex(img, "grad")`, the bit-exact twin of the console's finder `runtime/vocr_lines.c` (#3) |
| `legacy` | `common.find_lines(img, "legacy")`, the overlay's bright-pixel finder that cut v1's data |

The mask finder is an ideal finder of either polarity. Where the furigana rule does not fire,
its boxes equal the legacy finder's on bright text: identical on 4,817 of 4,817 random
bright-only images (`runs/finder_equiv.py`, scratch). Each region uses either the text core
(fills only, as the bright finder sees outlined text) or the full ink (with outline and shadow),
50/50. A crop is kept only if it matches exactly one rendered line. Image-only finders would bias
the set toward easy styles. On 1,000 synth2 regions (2,228 mask crops,
`calibrate.py finders --n 1000 --seed 5`), usable crops per mask crop:

| scheme | grad (console) | legacy |
|---|---|---|
| light on dark | 0.60 | 0.39 |
| dark on light | 0.58 | 0.10 |
| outlined on light | 0.49 | 0.19 |
| white on colour | 0.71 | 0.22 |
| all | 0.58 | 0.28 |

**Polarity.** The console inverts a line when `common.line_dark` calls it dark (the polarity
contract of #3), so a recognizer trained on light text only can read it. synth2 crops keep their
rendered polarity by default, and every crop's meta records the rule's verdict (`dark`).
`--invert-dark` stores the crops as the console would hand them over. The rule is not a clean
split on these styles: on 800 regions it marks 85% of dark-on-light crops dark, but also 38% of
outlined light text on light boxes and 27% of dark outlined text. A recognizer therefore meets
both polarities on the console either way, which is why the default keeps them. The A/B below
uses the default.

## Calibration against real screenshots

`synth/calibrate.py` compares crops from training-side games only (not `is_test_vn`, and not in
`data/manifests/synth2_bg_exclude.txt`):

- vita-det: linefind's detections on `data/vndb/shots`, horizontal boxes in the lower half, re-cut with the console's finder (`grad`). This includes UI and copyright text.
- psp-band: the console's finder on the lower 42% of the PSP screenshots
- real-bright: data/real pseudo-label crops cut by the overlay's finder

It makes 32-px side-by-side contact sheets and computes these statistics. Text polarity uses the
same estimator for every source; on synthetic crops it agrees with the renderer's ground truth
85% of the time.

| Source | Crops | Height p10/50/90 (px) | Light-text share | Contrast p10/50/90 | Background p50 | Margin busyness p50 |
|---|---|---|---|---|---|---|
| real Vita (vita-det) | 1,500 | 23 / 31 / 40 | 0.58 | 113 / 179 / 231 | 106 | 18.4 |
| real PSP (psp-band) | 556 | 20 / 25 / 36 | 0.67 | 107 / 156 / 211 | 108 | 41.0 |
| real-bright (data/real) | 761 | 25 / 33 / 50 | 0.80 | 120 / 178 / 223 | 75 | 21.2 |
| old shards (v1) | 4,000 | 25 / 29 / 34 | 1.00 | 156 / 191 / 216 | 47 | 8.2 |
| synth2, Vita | 2,364 | 26 / 31 / 39 | 0.59 | 92 / 145 / 195 | 112 | 15.7 |
| synth2, PSP native | 1,129 | 19 / 22 / 26 | 0.70 | 92 / 145 / 195 | 99 | 20.7 |
| synth2, PSP 2x | 507 | 30 / 35 / 44 | 0.74 | 83 / 127 / 175 | 96 | 16.8 |

The real and old rows use seed 0, and the synth2 rows come from shards 00000-00001 of the
artifact set (`calibrate.py stats`). "Contrast" is |text level − background median|, where the
text level is the 98th or 2nd grey percentile. "Busyness" is the standard deviation of the top
and bottom 3 rows.

The style mix was set from a hand tally of contact sheets:

- 41 training-side Vita dialogue boxes: 51% light on dark, 37% dark on light, 12% outlined or boxless; about half have a separate name plate; a third show a page cursor; 1 has furigana
- about 25 PSP boxes: 68% / 16% / 16%; glyphs of about 12-14 px, often with a 1 px dark outline

Three iterations against the statistics:

1. PSP polarity bias, which moved the light-text share on PSP from 0.59 to 0.70-0.75 (real: 0.67-0.74, depending on the finder that cuts the real crops)
2. near-white and near-black fills and more opaque light boxes, which moved median contrast from 144 to about 150
3. the owner's emphasis on translucent boxes, frames, PSP text and busy scenes, which moved median margin busyness from 11 to 16-21 and brought contrast back to 145

Remaining gaps:

- Synthetic contrast stays below the real Vita median (179), because of translucent dark boxes over bright scenes and the deliberately hard styles.
- Real margins are busier still. This is partly an artefact: real re-cut crops include parts of neighbouring lines, ruby and box borders.
- The real sets include UI and copyright strings that synth2 does not draw.

## Exclusions

Backgrounds come only from training-side screenshots: 650 Vita and 469 PSP shots from 154 VNs,
listed in `$HERD/shared/artifacts/synth2/backgrounds.json`. `synth/bg_exclude.py` builds
`data/manifests/synth2_bg_exclude.txt` from these sources:

- every `is_test_vn` game, which is also excluded in code
- eval's `test_games.json` as of 13:30Z: 172 reserved VN ids (the set games, every `is_test_vn` game, release siblings and VNDB relations)
- eval's 31 test capture sessions, which cover 106 local screenshots
- the 28 screenshots of the research's hand-transcribed lines

`calibrate.py` uses the same exclusions. The old shards excluded only `is_test_vn` games, so they
could contain eval's extra reserved games and the hand-set screenshots as backgrounds. That
favours the old arm on those sets.

## A/B: v1 architecture, old vs synth2 shards

**Setup.**

- Recipe: `train/train.py --steps 5000 --seed S --eval-every 2500` with every other option at v1's default (0.76M parameters, batch 96, lr 2e-3 with warm-up and cosine).
- Code: frozen at 73275ea, before the rebase; `train.py` is identical on this branch.
- 5,000 steps x 96 = 480k samples, about 1.23 epochs of either set:
  - old arm: v1's 40 shards (`$HERD/shared/data/synth`), 390,018 training lines
  - synth2 arm: `$HERD/shared/artifacts/synth2/shards`, 389,989 training lines
- In both arms the last shard is the validation shard.
- Runs and threads: synth2 at seeds 1 and 2, the old arm at seed 1, two runs per heavy slot at 2 threads each. `arch_base_s1` is the arch agent's run of the same recipe (old shards, 5,000 steps, seed 1; arch/sweep's `train.py` with channels_last, 5 threads), used as a second sample of the old arm. The old arm's seed-2 run was not trained: it waited 1.5 h for a heavy slot and was cancelled.
- Scoring: eval's benchmark, `eval/run.py` at bebd4f9 (PR #5's head, not merged yet). That is vita-test (206 scored lines, 16 VNs) and psp-test (245 lines, 14 VNs) on ground-truth line boxes, padded by 4 px on Vita and 2 px on PSP, with crops not inverted. synth-val is the first 3,000 lines of v1's shard 39. Host CPU, 1 thread. CER is over the whole set, after eval's normalisation.

| model | data, steps | vita-test CER / exact | psp-test CER / exact | synth-val (old) |
|---|---|---|---|---|
| int8 JIS PP-OCRv5 mobile (the owner's bar) | | 2.30% / 79.1% | 5.58% / 63.7% | |
| v1 | old, 32,496 | 23.75% / 52.4% | 46.65% / 28.2% | 0.32% |
| old_s1 | old, 5,000 | 29.25% / 40.8% | 52.35% / 24.5% | 1.22% |
| arch_base_s1 | old, 5,000 | 28.51% / 39.3% | 51.39% / 22.9% | 1.25% |
| **new_s1** | synth2, 5,000 | **5.38%** / 59.2% | **10.76%** / 43.3% | 2.55% |
| **new_s2** | synth2, 5,000 | **5.66%** / 56.8% | **9.68%** / 51.8% | 2.46% |

A paired bootstrap over lines (4,000 resamples) of CER(new_s1) − CER(old_s1) gives:

- vita-test: −23.9 points, 95% CI [−30.1, −18.1]
- psp-test: −41.6 points, 95% CI [−47.4, −35.6]
- synth-val (old): +1.3 points, 95% CI [+1.2, +1.5]

new_s2 against arch_base_s1 gives −22.9 [−28.8, −17.2] and −41.7 [−47.4, −35.8].

Per style, eval's tags, CER for old_s1 → new_s1 (new_s2):

| style | vita-test | psp-test | bar (int8 PP-OCRv5) |
|---|---|---|---|
| bright on dark | 11.4% → 5.5% (5.5%) | 23.6% → 10.8% (9.3%) | 2.4% / 5.4% |
| dark on light | 72.1% → 5.2% (6.1%) | 90.7% → 10.7% (10.1%) | 2.1% / 5.8% |
| outlined | 34.9% → 5.2% (5.2%) | 29.2% → 8.5% (6.9%) | 3.1% / 5.0% |
| coloured | 55.4% → 11.8% (11.8%) | 71.0% → 13.1% (10.8%) | 4.3% / 7.9% |
| names | 32.6% → 16.9% (16.9%) | 76.6% → 38.8% (32.5%) | 4.1% / 6.2% |
| small (PSP) | | 57.9% → 8.9% (7.8%) | 5.6% |
| furigana (vita n=3) | 18.8% → 9.4% (12.5%) | 64.1% → 40.3% (37.4%) | 3.1% / 9.7% |
| mincho | 3.3% → 4.7% (4.0%) | 45.9% → 8.9% (8.7%) | 1.3% / 5.5% |

The provisional sets in `synth/ab_eval.py` agree. Hand-transcribed lines (58 lines, 28 screenshots):

| model | CER |
|---|---|
| v1 | 30.6% |
| old_s1 | 38.2% |
| new_s1 | 5.5% |
| new_s2 | 5.9% |

On the 13 hand lines from reserved games, CER falls from 60.2% to 7.2-7.6%. Agreed pseudo-labels of bright text (PP-OCRv5 and manga-ocr agree, cut by the bright finder):

| set | old_s1 | new_s1 / new_s2 |
|---|---|---|
| test split | 8.4% | 3.5% / 4.4% |
| train split | 13.9% | 5.8% / 5.4% |

On synth2's own validation shard, CER falls from 44.8% to 9.0% / 8.6%.

**Reading.**

- At the same budget, synth2 cuts real-screenshot CER about five-fold. It also beats v1, which trained 6.5 times longer on the old set.
- The gain is largest where the old set had nothing: dark-on-light, outlined, coloured and PSP-sized text. Light-on-dark also halves.
- The old validation shard gets worse, from 1.2% to 2.5%. It holds one easy style; the model now shares its capacity across many.
- Vita mincho is slightly worse at this budget: 3.3% becomes 4.0-4.7%, on 47 lines from 3 games. v1, with 8 epochs on the old set, has 1.3%.

**Not reached.** No 5,000-step synthetic-only run comes near the owner's bar (2.30% / 5.58%), and none was expected to. synth2 is one input to train2's recipe, with the NVIDIA pretraining set, teacher-labelled real crops and distillation. The weakest styles are follow-ups for the renderer:

- Names: 17% on Vita and 33-39% on PSP. The errors are mostly kanji substitutions in 2-4 character crops; names draw from the 1,200 most frequent kanji, and real VN names use rarer ones.
- PSP furigana: 37-40%. Eval's line boxes include the ruby above the line, which shrinks small PSP text further after the resize to 32 px.
- Coloured names.

## Cost

The 40 x 10,000-line set renders in 1,115 s with 5 workers in one heavy slot: 72 lines/s per
worker. The old renderer did 66.8 lines/s in one process (`runs/time_old.py`, scratch). The set
is 2.8 GB, the same size as v1's.

## Reproduce

```sh
$V synth/fetch_synth2.py all                     # Fusion Pixel fonts + PSP screenshots (manifests in data/manifests)
ln -s $HERD/shared/artifacts/synth2/fonts_extra data/fonts_extra
ln -s $HERD/shared/artifacts/synth2/vndb_psp data/vndb_psp
$V synth/render.py preview                       # synth/preview_*.png, all <= 1280 px
$HERD/heavy.sh $V synth/render.py shards --out OUT --shards 40 --per-shard 10000 --workers 5 --seed 20261009
$V synth/calibrate.py stats --synth "old=data/synth/synth_0000[0-1].npz" --synth "new=OUT/synth_0000[0-1].npz"
$V synth/calibrate.py sheets --synth "new=OUT/synth_0000[0-1].npz" --out runs/calib
$V synth/calibrate.py finders --n 1000
$V synth/ab_eval.py --ckpt old=... --ckpt new=... --synth-val old=data/synth/synth_00039.npz \
    --synth-val synth2=OUT/synth_00039.npz --pairs old:new
```
