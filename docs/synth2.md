# synth2: synthetic lines that look like PS Vita and PSP visual novels

v1 was trained on one look only: bright text on a dark box. The overlay's line finder only sees
bright pixels, so the old renderer could only cut bright text. On eval's vita-test
(`docs/results.md`), v1 reads bright-on-dark lines at 8.0% CER but dark-on-light lines at 81.8%
and outlined ones at 26.9%. The console's polarity rule (`docs/linefind.md`) inverts dark lines
before recognition, which takes v1's dark-on-light lines to 11.1%, but outlined (13.8%), coloured
(49.2%) and name lines (22.1%) stay weak. Under the owner's training plan (RULES.md, 2026-10-09 10:55Z), the Japanese subset of
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
its boxes equal the legacy finder's on bright text (`tests/test_synth2.py` checks this on 201
random line masks drawn white on black). Each region uses either the text core
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
split on these styles: on the 2,228 mask crops above it marks 84% of dark-on-light crops dark,
but also 40% of outlined light text on light boxes and 21% of dark outlined text (2% of
light-on-dark and white-on-colour crops). A recognizer therefore meets both polarities on the
console either way, which is why the default keeps them. The A/B below trains on the default and
scores both ways.

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
- eval's `test_games.json` as of 13:30Z: 172 reserved VN ids (the set games, every `is_test_vn` game, release siblings and VNDB relations). bg_exclude.py merges the committed `eval/manifests/test_games.json` with eval's working copy in `$HERD`, which carries psp-test's reservations ahead of #5.
- eval's 31 test capture sessions, which cover 106 local screenshots
- the 28 screenshots of the research's hand-transcribed lines

`render.py` and `calibrate.py` read the exclusions through one loader
(`bg_exclude.load_exclusions`), which also skips every VN in the committed
`eval/manifests/test_games.json`, so a stale manifest cannot let a reserved game back in.
`tests/test_synth2.py` fails when the manifest stops covering that list.

The PSP screenshots (`data/manifests/synth2_psp_shots.jsonl`, 469 rows from 68 VNs) are
training-side too. `fetch_synth2.py` fetches exactly those files and checks each sha256. Its
first pick filtered on `is_test_vn` only and also fetched 32 screenshots of four VNs eval
reserves (v9915, v10606, v11639, v12159). render.py had already skipped them, so the shards
don't contain them.

The old shards excluded only `is_test_vn` games, so their backgrounds could include eval's other
reserved games and the hand-set screenshots. That favours the old arm on those sets.

## A/B: v1 architecture, old vs synth2 shards

**Setup.**

- Recipe: `train/train.py --steps 5000 --seed S --eval-every 2500` with every other option at v1's default (0.76M parameters, batch 96, lr 2e-3 with warm-up and cosine).
- Code: frozen at 73275ea, before the rebase; `train.py` is identical on this branch.
- 5,000 steps x 96 = 480k samples, about 1.23 epochs of either set:
  - old arm: v1's 40 shards (`$HERD/shared/data/synth`), 390,018 training lines
  - synth2 arm: `$HERD/shared/artifacts/synth2/shards`, 389,989 training lines
- In both arms the last shard is the validation shard.
- Runs and threads:
  - synth2 at seeds 1 and 2, and the old arm at seed 1, two runs per heavy slot at 2 threads each.
  - `arch_base_s1` and `arch_base_s2` are the arch agent's runs of the same recipe on the old shards, seeds 1 and 2, with arch/sweep's `train.py`: the same flags and semantics, plus channels_last and 5 threads.
  - That makes three old-arm samples and two synth2 samples.
- Scoring:
  - eval's benchmark `eval/run.py` on ground-truth line boxes, padded by 4 px on Vita and 2 px on PSP. Host CPU, 1 thread. CER is over the whole set, after eval's normalisation.
  - vita-test (206 scored lines, 16 VNs) and synth-val (the first 3,000 lines of v1's shard 39) use the manifests on main.
  - psp-test (245 lines, 14 VNs) uses PR #5's manifest at bebd4f9, not merged yet.
- Two polarity conditions:
  - **native** (`--polarity native`, the default): crops as they are on screen.
  - **console** (`--polarity console`): the console's polarity contract (`docs/linefind.md`). A crop whose text `common.line_dark` calls dark is inverted before recognition. The rule inverts 37 of vita-test's 43 dark-on-light lines and 7 of its other 163. On psp-test it inverts 102 of 110 and 5 of 135, and on synth-val none.
- Pipeline: `--finder console --polarity console`. The console's finder (`common.find_lines(..., "grad")`, the twin of `runtime/vocr_lines.c`) runs on eval's dialogue regions, then the inversion, then the recognizer.

CER / exact-line share:

| model | data, steps | vita-test native | vita-test console | psp-test native | psp-test console | synth-val (old) |
|---|---|---|---|---|---|---|
| int8 JIS PP-OCRv5 mobile (the owner's bar) | | 2.30% / 79.1% | not run | 5.58% / 63.7% | not run | |
| v1 | old, 32,496 | 23.75% / 52.4% | 8.93% / 64.6% | 46.65% / 28.2% | 10.96% / 58.4% | 0.32% |
| old_s1 | old, 5,000 | 29.25% / 40.8% | 13.18% / 51.9% | 52.35% / 24.5% | 16.93% / 47.3% | 1.22% |
| arch_base_s1 | old, 5,000 | 28.51% / 39.3% | 12.91% / 51.0% | 51.39% / 22.9% | 17.09% / 46.1% | 1.25% |
| arch_base_s2 | old, 5,000 | 30.03% / 38.8% | 14.00% / 51.0% | 53.86% / 25.7% | 17.29% / 46.1% | 1.24% |
| **new_s1** | synth2, 5,000 | **5.38%** / 59.2% | **5.11%** / 62.1% | **10.76%** / 43.3% | **10.56%** / 45.7% | 2.55% |
| **new_s2** | synth2, 5,000 | **5.66%** / 56.8% | **5.50%** / 57.3% | **9.68%** / 51.8% | **9.36%** / 52.2% | 2.46% |

Whole console path (finder plus inversion), pipeline CER. The finder is the same for every
model: no region rejected, line recall 95.15% / 90.20%, precision 95.61% / 94.85%.

| model | vita-test (84 regions) | psp-test (93 regions) |
|---|---|---|
| v1 | 20.16% | 30.92% |
| old_s1 | 27.15% | 37.97% |
| arch_base_s1 | 25.86% | 37.05% |
| arch_base_s2 | 27.18% | 36.97% |
| **new_s1** | **12.52%** | **22.87%** |
| **new_s2** | **13.69%** | **23.19%** |

Paired bootstrap, CER(B) − CER(A) in points with a 95% CI. It uses 4,000 resamples with seed 0,
over lines for the crop columns and over regions for the pipeline:

| B − A | vita-test native | vita-test console | psp-test native | psp-test console | vita pipeline | psp pipeline |
|---|---|---|---|---|---|---|
| new_s1 − old_s1 | −23.9 [−29.9, −18.1] | −8.1 [−11.5, −5.0] | −41.6 [−47.4, −35.8] | −6.4 [−9.3, −3.6] | −14.6 [−20.3, −9.7] | −15.1 [−19.1, −11.5] |
| new_s2 − arch_base_s1 | −22.9 [−28.7, −17.3] | −7.4 [−10.9, −4.4] | −41.7 [−47.5, −36.0] | −7.7 [−11.0, −4.7] | −12.2 [−17.7, −7.3] | −13.9 [−17.7, −10.4] |
| new_s1 − arch_base_s2 | −24.6 [−30.7, −18.9] | −8.9 [−12.4, −5.6] | −43.1 [−49.2, −37.2] | −6.7 [−9.6, −4.0] | −14.7 [−20.4, −9.6] | −14.1 [−18.3, −10.4] |
| new_s1 − v1 | −18.4 [−24.1, −13.0] | −3.8 [−6.8, −1.2] | −35.9 [−41.8, −30.2] | −0.4 [−2.5, +1.7] | −7.6 [−12.3, −3.6] | −8.0 [−11.1, −5.1] |
| new_s2 − v1 | −18.1 [−23.8, −12.8] | −3.4 [−6.4, −0.9] | −37.0 [−42.8, −31.2] | −1.6 [−3.9, +0.6] | −6.5 [−11.2, −2.4] | −7.7 [−10.9, −4.8] |

On synth-val (old), new_s1 − old_s1 is +1.3 [+1.2, +1.5].

Per style, CER for old_s1 → new_s1 (new_s2), with v1 under the console rule and the bar.
The vita-test tags are main's, including the 26 corrections from #2's review. The psp-test tags
are PR #5's.

vita-test:

| style | lines | native | console | v1, console | bar (JIS int8, native) |
|---|---|---|---|---|---|
| bright-on-dark | 163 | 12.8% → 5.6% (5.7%) | 12.7% → 5.5% (5.8%) | 8.3% | 2.48% |
| dark-on-light | 43 | 89.8% → 4.7% (5.5%) | 15.0% → 3.7% (4.6%) | 11.1% | 1.64% |
| outlined | 74 | 34.0% → 5.3% (5.3%) | 19.7% → 5.0% (5.4%) | 13.8% | 3.03% |
| coloured | 19 | 56.4% → 12.3% (12.3%) | 54.9% → 11.8% (12.8%) | 49.2% | 5.13% |
| name | 53 | 32.6% → 16.9% (16.9%) | 32.6% → 14.0% (16.3%) | 22.1% | 4.07% |
| furigana | 3 | 18.8% → 9.4% (12.5%) | 18.8% → 9.4% (12.5%) | 7.8% | 3.12% |
| mincho | 47 | 3.3% → 4.7% (4.0%) | 3.3% → 4.7% (4.0%) | 1.3% | 1.33% |

psp-test:

| style | lines | native | console | v1, console | bar (JIS int8, native) |
|---|---|---|---|---|---|
| bright-on-dark | 135 | 23.6% → 10.8% (9.3%) | 21.7% → 11.2% (9.2%) | 14.4% | 5.44% |
| dark-on-light | 110 | 90.7% → 10.7% (10.1%) | 10.5% → 9.7% (9.6%) | 6.4% | 5.77% |
| outlined | 69 | 29.2% → 8.5% (6.9%) | 28.9% → 9.3% (6.6%) | 18.9% | 5.00% |
| coloured | 20 | 71.0% → 13.1% (10.8%) | 22.0% → 12.2% (10.8%) | 15.4% | 7.94% |
| name | 62 | 76.6% → 38.8% (32.5%) | 53.6% → 37.8% (30.1%) | 40.7% | 6.22% |
| small | 106 | 57.9% → 8.9% (7.8%) | 7.5% → 8.3% (8.1%) | 4.2% | 5.55% |
| furigana | 20 | 64.1% → 40.3% (37.4%) | 63.1% → 41.3% (36.9%) | 43.2% | 9.71% |
| mincho | 90 | 45.9% → 8.9% (8.7%) | 8.4% → 8.8% (7.9%) | 6.1% | 5.49% |

**Reading.**

- **Under the console's polarity rule**, which is how the console hands crops to a recognizer, synth2 at the same budget cuts CER by 38-61%:
  - vita-test 13.2% → 5.1% and psp-test 16.9% → 10.6% (old_s1 → new_s1);
  - the whole console path 27.2% → 12.5% and 38.0% → 22.9%;
  - all three old-arm runs and both synth2 runs agree.
- **Without the inversion the gap is much larger** (29.3% → 5.4% and 52.4% → 10.8%), because models trained on the old set can't read dark text at all (dark-on-light 89.8% and 90.7%). Most of that gap is polarity, which the console's inversion already closes for a light-only model.
- **synth2 models read both polarities**: their native and console CERs are within 0.3 points.
- **Against v1**, which trained 6.5 times longer on the old set: under the console rule synth2 is better on vita-test (−3.8 points) and on the whole console path (−7.6 and −8.0 points). On psp-test's ground-truth boxes it is level with v1 (−0.4 [−2.5, +1.7]).
- **Where synth2 helps** (console rule, old_s1 → new_s1):
  - outlined: 19.7% → 5.0% on Vita, 28.9% → 9.3% on PSP
  - bright-on-dark: 12.7% → 5.5% and 21.7% → 11.2%
  - coloured: 54.9% → 11.8% and 22.0% → 12.2%
  - names: 32.6% → 14.0% and 53.6% → 37.8%
- **Where it doesn't, at this budget:**
  - PSP small text: 7.5% → 8.3%, v1 4.2%
  - PSP dark-on-light once inverted: 10.5% → 9.7%, v1 6.4%
  - mincho: 3.3% → 4.7% on Vita (47 lines from 3 games), 8.4% → 8.8% on PSP
  - The native small-text row (57.9% → 8.9%) is a polarity effect: 62 of psp-test's 106 small lines are dark on light.
- The old validation shard gets worse, 1.2% → 2.5%. It holds one easy style, and the model now shares its capacity across many.

**Not reached.** No 5,000-step synthetic-only run comes near the owner's bar (2.30% / 5.58%), and none was expected to. synth2 is one input to train2's recipe, with the NVIDIA pretraining set, teacher-labelled real crops and distillation. The weakest styles are follow-ups for the renderer:

- Names: 14-17% on Vita and 30-39% on PSP. The errors are mostly kanji substitutions in 2-4 character crops. Names draw from the 1,200 most frequent kanji, and real VN names use rarer ones.
- PSP furigana: 37-41%. Eval's line boxes include the ruby above the line, which shrinks small PSP text further after the resize to 32 px.
- PSP small text and inverted PSP dark-on-light text, where v1's longer run is still better.
- Coloured names.

**Provisional sets** (`synth/ab_eval.py`, from before eval merged; native polarity). They point
the same way. On the research's hand-transcribed lines (58 lines, 28 screenshots):

| model | CER |
|---|---|
| v1 | 30.6% |
| old_s1 | 38.2% |
| new_s1 | 5.5% |
| new_s2 | 5.9% |

On the 13 hand lines from reserved games, CER falls from 60.2% to 7.2-7.6%. Agreed pseudo-labels
of bright text (PP-OCRv5 and manga-ocr agree, cut by the bright finder):

| set | old_s1 | new_s1 / new_s2 |
|---|---|---|
| test split | 8.4% | 3.5% / 4.4% |
| train split | 13.9% | 5.8% / 5.4% |

The train split isn't held out from synth2: 252 of its 268 lines come from 175 screenshots that
are also in synth2's background pool (the backgrounds are crops of the upper 62% of the screen).
On synth2's own validation shard, CER falls from 44.8% to 9.0% / 8.6%.

## Cost

The 40 x 10,000-line set rendered in 1,115 s with 5 workers in one heavy slot (the render log in
the artifact README), about 72 lines/s per worker. The set is 2.8 GB, the same size as v1's.

## Reproduce

```sh
$V synth/fetch_synth2.py all                     # Fusion Pixel fonts + PSP screenshots, sha256-checked against data/manifests
ln -s $HERD/shared/artifacts/synth2/fonts_extra data/fonts_extra
ln -s $HERD/shared/artifacts/synth2/vndb_psp data/vndb_psp
$V synth/bg_exclude.py                           # re-run before every render: eval's reserved list only grows
$V synth/render.py preview                       # synth/preview_*.png, all <= 1280 px
$HERD/heavy.sh $V synth/render.py shards --out OUT --shards 40 --per-shard 10000 --workers 5 --seed 20261009
$V synth/calibrate.py stats --synth "old=data/synth/synth_0000[0-1].npz" --synth "new=OUT/synth_0000[0-1].npz"
$V synth/calibrate.py sheets --synth "new=OUT/synth_0000[0-1].npz" --out runs/calib
$V synth/calibrate.py finders --n 1000 --seed 5
export VOCR_EVAL_DATA=$HERD/shared/artifacts/eval  # psp-test needs PR #5's manifest until it merges
$V eval/run.py --model CKPT --out native.json
$V eval/run.py --model CKPT --polarity console --out console.json
$V eval/run.py --model CKPT --set vita-test,psp-test --finder console --polarity console --out pipeline.json
$V synth/ab_eval.py --ckpt old=... --ckpt new=... --synth-val old=data/synth/synth_00039.npz \
    --synth-val synth2=OUT/synth_00039.npz --pairs old:new
```

The A/B checkpoints (including both arch runs) and every eval output behind these tables are in
`$HERD/shared/artifacts/synth2/ab/` (see its README).
