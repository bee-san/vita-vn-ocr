# synth4: a rare-character curriculum and a style mix measured on real crops

synth4 is synth3 version 3 (`docs/synth3-v3.md`) with the changes that `docs/synth4-gap.md`'s
measurements of the training-side real data call for: a curriculum that gives every vocab class a
minimum number of exposures in real-word contexts, a font, raster and style mix moved to the
measured real distribution, and the owner's two fonts. The generator is `synth/synth4.py`; it
patches `synth/synth3.py`'s configuration and a handful of its functions and leaves the rest
(text pools, scenes, layouts, crops, capture chains) exactly as version 3 has them.

**Result (pre-registered FL10 ablation, `eval/synth4_protocol.json`): synth4 is not measured better
than synth3 version 3 at a 16,000-step pretrain.** Two FL10 models pretrained from scratch on NVIDIA
crops plus synth4 (F_s4) or synth3 v3 (D_s3v3, study 1b's runs reused), two seeds each, read through
the C runtime (W8I) on the development cohorts:
- teacher_val (decides): F_s4 − D_s3v3 = **+0.279 points of CER** (VN-cluster 95% CI +0.08 to +0.48,
  both seeds positive: +0.287, +0.270). synth4 is slightly *worse*.
- devnative v3 pooled (guard): +0.165 (CI −0.11 to +0.47): no measured difference.

So by the frozen rule synth3 version 3 stays the recommended VN-style synthetic data. The curriculum
did what it set out to (every vocab class far better covered: median exposures 65 → 102, classes
under 10 exposures 1,167 → 29, classes drawn by under 5 fonts 575 → 29), and the style mix moved
toward the measured real distribution, but neither lowered CER at this budget. One signal points the
other way and is worth a longer study: on **native console crops** — where the 1-bit, contrast and
font changes land — synth4 is level or slightly better (teacher_val native Vita −0.36 [−1.01, +0.18],
devnative native Vita −0.40 [−0.68, −0.19], devnative native ink (tv_native) −0.16 [−0.40, +0.10]); the
regression is on teacher_val's PC-screenshot majority (hires +0.48 [+0.23, +0.74]) and its short and
name lines (len 1-4 +0.91, name-like +0.68). The limits section lists why a 16k-step pretrain may
not show a curriculum's gain, and what study would.

## What synth4 changes, and the evidence

| part | synth3 v3 | synth4 | evidence (docs/synth4-gap.md) |
|---|---|---|---|
| curriculum | focus lines 25% of dialogue (12.5% of narration), drawn with weight (exposures + 20)^-0.6 x hard x VN prior | every vocab class at least **K = 80** exposures per 1,000,000 crops if VN text uses it, **25** otherwise; the deficit against synth3 v3's measured exposures (144,288 occurrences over 2,784 classes) is spread over the render's 100 shards with seeded stochastic rounding; a focus draw takes a class with shard quota left first (75%), else synth3's weighting; focus lines 40% of dialogue (20% of narration) | median vocab class 65 exposures in synth3 v3; 2,040 VN classes under 50, 774 under 10; 1,229 classes drawn by fewer than 10 fonts |
| contexts | a pool line holding the focus character | for a quota class: a Jiten VN word holding it in a short VN carrier sentence (35%: 「{w}」, {w}って、何？, {w}だ。, ...; verbs and adjectives get verb carriers), a VNDB or web-novel name holding it (15%: {n}さん！, {n}の部屋。), else synth3's natural pool line; 30% of speaker names hold a quota class while one has quota left | real words and names, never random strings (owner); rare kanji in VN text sit in names and rare words |
| hard characters | focus weight x (1 + 4 x hard), hard from synth3's mining | hard = 0.5 x synth3 v3's + 0.5 x the VOCR3 selections' error rate on 2,890 unseen training-side lines (shrunk: errors / (occurrences + 10)) | the selections' confusions: dakuten pairs, 。/、, 』/」, ー/一, っ/つ, katakana with handakuten |
| 1-bit text | pixel fonts 10% (Vita) / 28% (PSP) of native regions; 1-bit raster 6% / 28% of the rest | pixel fonts 4% / 8%, 1-bit raster 2% / 4% (4-bit anti-aliased raster 52% of PSP regions) | synth3 49% of native PSP crops 1-bit; on the edge measure real native PSP crops match synth3's anti-aliased renders, but eye checks call 10-54% of them hard-edged: the real 1-bit share is not measured, so this is a design choice the ablation tests |
| pixel fonts | 13 families at 10-20 px strikes; 10-11 px fonts never on the Vita | the same 13 plus NDS12 (owner) at its exact strike; 10-11 px fonts also on the Vita at 2x nearest | the owner: "more pixel/bitmap fonts at their native strikes", within the measured 1-bit share |
| owner fonts | none | cinecaption 2.26 (category pen) and NDS12 (pixel, 10 ppem, 1x on PSP, 2x nearest on Vita), licence class `owner`, `--profile personal` only | owner, 14:28Z |
| outlines | synth2's own outline (34% of light-on-dark, 16% of dark-on-light schemes) plus synth3's rates (Vita .40/.12/.7, PSP .60/.20/.8, scaled .55/.2/.8) | synth2's own outline dropped on 45% of the schemes that do not need one; synth3's rates .20/.06/.5, .25/.08/.5, .35/.10/.6 | synth3 61-73% of crops outlined; real estimates 0-60% (weak detector) |
| contrast | synth2's palette | dialogue boxes darker (x U(0.3, 0.75)), light boxes lighter (distance to white x U(0.3, 0.7)), 75% of plain light fills pure white, dark fills darker (x U(0.2, 0.7)) | synth3 p50 contrast 134-153 grey levels against real 145-191: text dimmer (fill p50 226-235 vs 242-252) and boxes lighter (p10 35-39 vs 10-15) |
| colour | coloured spans 14%, lines 10%, names 65%; synth2's accents | spans 8%, lines 6%, names 30%; 40% of accent-coloured fills plain; green and cyan accents (hue 0.17-0.55) redrawn once with p 0.85; half the gradient fills | synth3 17% coloured fills, a fifth green; real 10.4%, 8% green (2.5x); gradients 10.3% vs ~7% |
| scaled text | Vita scaled em N(20.5, 3.3) | N(19.0, 3.3) | synth3's scaled Vita ink 1-2 px taller than real |
| blur | 15% | 10% | contrast |

Unchanged from version 3: the 14 text pools and their document split, the length gate, names,
layouts (ADV, band, NVL, balloon, choice, bare, UI), furigana, cursors, typewriter lines, scenes
(training-split screenshots, never a reserved VN), capture chains, gt and GRAD crops, console
polarity, and the vocab (`data/vocab.txt`, 5,354 classes: a vocab change needs a new head, section
1 of the gap analysis recommends one for the next round).

### Calibration

Two calibration renders (15,000 crops each, seed 3001, never a training seed) checked the style
changes with the gap analysis's own measures (`synth4_gap.py styles` and `colour`, thresholds
fitted on synth3 v3's metadata) against the same training-side real crops; round 2 (`c2`) is the
pinned configuration:

| | synth3 v3 | synth4 c1 | **synth4 c2** | real (teacher_train + real_train) |
|---|---|---|---|---|
| contrast p50, all / native Vita / native PSP | 142.5 / 153 / 146 | 153 / 166 / 155 | **160 / 172 / 160** | 167 / 191 / 177 |
| 1-bit detector, native PSP (obs) | 0.366 | 0.131 | **0.142** | 0.026 |
| outline, all (obs / Rogan-Gladen est.) | 0.465 / 0.68 | 0.361 / 0.36 | **0.357 / 0.35** | 0.375 / 0.40 |
| coloured fills (synth3 v3 re-rendered) | 0.168 | 0.136 | **0.096** | 0.104 |
| hue of coloured fills r / g / b | .51 / .20 / .29 | .55 / .16 / .29 | **.58 / .12 / .30** | .60 / .08 / .32 |
| ink height p10/50/90, scaled to Vita | 15/23/34 | 14/20/32 | **14/20/32** | 15/21/31 |
| background texture p50 | 15.8 | 15.9 | **16.4** | 17.6 |

The 1-bit share on native PSP is about 12% (truth: 8% pixel fonts plus 4% 1-bit raster) against
synth3's 49%. The gap analysis first read the edge measure as "real native text is almost never
1-bit"; eye checks on real crops (its section 3, corrected after the audit) do not support that:
real PSP text is mostly crisp, and its 1-bit share is unmeasured. The 12% was fixed before that
correction, so the ablation tests it as part of the package; a synth4 with more 1-bit text is an
open alternative. Native Vita contrast stays 19 levels short of real.

## Dataset

### Render and pin

- `data/manifests/synth4_shards_v1.json` pins 1,000,000 crops in 100 shards (4,678,737,317 bytes) of
  `synth/synth4.py` (sha256 `a16fa7f3…`, with synth3.py, render.py, common.py and synth3_text.py at
  main's bytes; every code hash is in the recipe), seed 20261013, `--profile personal`: 317 font
  files in 112 families (307 open, 8 non-commercial, 2 owner), synth3 v3's 14 text pools and
  scenes, 87 excluded VNs (fresh native 13, CLANNAD family 24, the owner's 50; the three synth3
  exclusion manifests, as synth3 v3). Rendered 15:58-16:29Z in 4 heavy.sh jobs of 5 workers (about
  345 s per 10,000-crop shard per worker), 0 render errors.
- **Reserved VNs the render does not exclude (disclosure; conductor accepted 2026-10-13 19:40Z).**
  The generator loads the three synth3 exclusion manifests — `synth3_exclude_fresh_native.json` (13),
  `synth3_exclude_clannad.json` (24) and `synth3_exclude_owner_games.json` (the owner's 50) = 87 VNs
  (recipe `exclusions`) — so #63's 50-VN owner set (Aiyoku no Eustia, 428, 街 and relations) **is**
  reserved. **Cause:** two VNs the fleet reserves only through a shared VNDB release are in none of
  those manifests, so they are not dropped: 弟切草 (v661, through 街's PS1 release) and Ever 17
  (v19373, through the Never 7 pack). synth3 v3's own render has the same exposure (synth3-v3
  Hygiene), which #63's audit accepted. Measured on the 100 pinned shards:
  - v661: 2 narration crops holding the title 弟切草, and about 28 name crops of names only its
    characters produce (公平, 透明人間, アインシュタイン博士, フランケンシュタインの怪物, 怪魚 — all
    generic strings); 31 of 1,000,000 crops; no scene (no v661 screenshot is local);
  - the Never 7 release partners (v13, v17, v247, v19373): about 29 name crops and two scene VNs.

  No protected test reads any of these VNs; the gap analysis (#65) excludes v661 and v19373 from its
  statistics; and both arms are reserved identically, so the ablation stands. **Before any further
  synth4 render**, keep `synth3_exclude_owner_games.json` in the generator's exclusions (it is already
  there) and add v661 and the Never 7 partners.
- The protocol was committed and pushed at 15:57:22Z (`8a1a932`), the pin at 16:30:32Z (`304bfeb`),
  before F_s4's first step (16:31Z). Both commits are kept at tag `synth4-history-20261013`; the branch
  was later rebased onto the corrected gap analysis (its commits are reachable from the branch tip;
  the generator, protocol and pin files are unchanged across the rebases), which changed no file of
  the render.
- Determinism: re-rendering the first 1,000 crops of shards 0 and 57 from the committed code gives
  byte-identical pixels, shapes, labels and metadata.
- Crop count per licence: open 990,793, non-commercial 5,616, owner 3,591 (cinecaption 3,538,
  NDS12 53). NDS12 draws only 333 vocab classes (kana, ASCII and symbols; no kanji), so it can take
  only kana-only lines, at 10 px (PSP, 1x) and 20 px (Vita, 2x nearest); a contact sheet of its crops
  showed exact 1-px bricks on the PSP and 2x2 blocks on the Vita. Pixel fonts are 8.2% of native PSP
  regions and 4.1% of native Vita regions, 1-bit raster 3.8% / 1.9%.

### What the curriculum achieved

Occurrences per vocab class in the 1,000,000 crops (`synth4_gap.py exposures`):

| | synth3 v3 | synth4 |
|---|---|---|
| H15 vocab (5,354): exposures p10 / p50 / p90 | 4 / 65 / 2,087 | **63 / 102 / 2,065** |
| classes never rendered | 66 | 28 |
| classes under 10 / under 50 exposures | 1,167 / 2,487 | **29 / 427** |
| classes under their K (80 VN, 25 other) | 2,784 | 1,336 |
| classes drawn by fewer than 10 / 5 font files | 1,229 / 575 | **30 / 29** |
| FL10 vocab (3,500): p10 / p50; under 10 / under 50 | 27 / 266; 138 / 670 | 73 / 281; 12 / 112 |
| share of all characters that are curriculum classes | 0.50% | 1.94% |

The quota is not met for every class: a focus draw takes a quota class only 75% of the time and
only 40% of dialogue lines have a focus character, so 1,336 classes end between their synth3 count
and K (most of them above 60). The 28 never rendered are the half-width ASCII ! ( ) ? ~, Cyrillic,
‐ ─ ゕ ゖ and 13 old-form kanji with no pool line, word or name (the pools use full-width forms).
By region (a region's crops share its metadata): 61,207 crops (6.1%) come from regions with a Jiten
word carrier, 6,721 from regions with a name line, and 116,301 (11.6%) from regions whose speaker
name was drawn to hold a quota class; every other quota line is a natural pool line.

### Label audit

`synth/synth3_audit.py`, PP-OCRv6 small on 1,500 random crops (seed 1) in screen polarity:

| set | teacher vs label CER | exact lines | names | pen fonts | pixel fonts | native PSP |
|---|---|---|---|---|---|---|
| synth3 v3 | 3.83% | 77.9% | | | | 5.16% |
| synth4 | 4.56% | 76.7% | 7.97% | 11.76% (88 lines) | 5.50% (53) | 5.84% |

More disagreement is expected: synth4 holds four times the share of rare characters, and the
teacher misreads rare kanji and names more often. Reading the 36 worst crops by eye found pixel,
pen and tiny fonts, rare-kanji names and two near-invisible lines: white text on a 31%-opaque
band over a bright scene (16 grey levels of range), and light text with a light outline on a light
box (synth2's palette gives an outlined-on-light scheme a light outline when an accent fill is
darker than the box). Both causes are synth3's too; the share of crops whose text contrast is under
40 grey levels is 0.14% in synth4 (calibration render), 0.20% in synth3 v3 and 0.10% in real
crops, and under 12 (unmeasurable) 0.10% / 0.13% / 0.06%. Not fixed in the pinned code.

### Calibration against training-side real crops

`synth/synth3_stats.py` on 20,000 crops of each source (teacher_train + real_train with every reserved
and development VN dropped, 454 rows) and the gap analysis's style measures on shards 0-1:

| | real | synth3 v3 | synth4 |
|---|---|---|---|
| KL(real ‖ source), characters | | 0.060 | 0.078 |
| label length p10 / p50 / p90 | 2 / 8 / 24 | 2 / 8 / 24 | 2 / 8 / 24 |
| name-like crops | 0.40 | 0.35 | 0.34 |
| Vita crop height, PC shots scaled (p10/50/90) | 23 / 28 / 35 | 23 / 29 / 36 | 22 / 27 / 34 |
| edge busyness p50 | 21.6 | 18.5 | 18.5 |
| text contrast p50, all / native Vita / native PSP | 167 / 191 / 178 | 142.5 / 153 / 146 | 158 / 173 / 161 |
| 1-bit detector, native PSP (obs) | 0.026 | 0.366 | 0.146 |
| outlined (obs / est.) | 0.376 / 0.40 | 0.465 / 0.68 | 0.357 / 0.35 |
| coloured fills; hue r / g / b | 0.104; .60 / .08 / .32 | 0.168; .51 / .20 / .29 | 0.109; .58 / .13 / .28 |

The character KL rises because the curriculum oversamples the tail on purpose (the curriculum
classes are 0.5% of real-like text and 1.9% of synth4's characters).

### The public recipe (`--profile open`)

`synth4.py shards --profile open` renders with free fonts and open text only. A check render (one
shard of 2,000 crops, seed 20261013, 52 s, 0 errors) loaded 307 font files in 108 families, all
licence class open (no non-commercial family, neither owner font), and 8 of the 14 text pools, all
class open (Aozora dialogue and narration, Wikipedia, Tatoeba, JESC, Aratako DeepSeek roleplay,
Wiktionary, Wikiquote); its recipe records `fonts_by_licence: {"open": 307}`. It was not trained or
scored: the ablation's both arms use the personal profile, as synth3 v3's pinned render did.

## The ablation (pre-registered)

The protocol (`eval/synth4_protocol.json`, frozen and pushed at 15:57:22Z, before the shard pin at
16:30:32Z and F_s4's first step at 16:31Z; both commits at tag `synth4-history-20261013`) is study
1b's exactly, with the arms D_s3v3 (synth3 v3, study 1b's runs, reused with the conductor's approval)
and F_s4 (synth4). Both train the FL10 topology from scratch for 16,000 steps on NVIDIA one-line
crops (0.5) plus the VN-style source (0.5), seeds 1 and 2, at trainer commit `2b132aa` (the commit
D_s3v3 trained at); the step-16000 EMA checkpoints are exported to the C W8I runtime and scored on
the VOCR3 selection's amendment-8 copies of teacher_val and devnative v3 (no owner-reserved title).
D_s3v3's reused checkpoints are re-exported and checked byte-for-byte against study 1b's blobs
(`eval/synth4_score.sh`). teacher_val decides; devnative v3 pooled is a guard (must not be entirely
worse). The decision adds nothing else. Result: `eval/synth4_ablation_20261013.json`.

### teacher_val (decides)

| run | rv CER | rvn CER |
|---|---|---|
| D_s3v3 seed 1 | 5.16% | 5.21% |
| D_s3v3 seed 2 | 5.38% | 5.37% |
| F_s4 seed 1 | 5.55% | 5.40% |
| F_s4 seed 2 | 5.68% | 5.60% |

(These are the final C W8I scores of `eval/synth4_score.sh`; the in-training evaluation logs, which the
protocol uses for monitoring only, read a little lower.) Arm CER (both views, both seeds): D_s3v3 5.28%,
F_s4 5.56%. **F_s4 − D_s3v3 = +0.279 points, VN-cluster 95% CI [+0.08, +0.48]**, both seed pairs positive
(+0.287, +0.270). The interval is above 0, so synth4 is not better; it is slightly worse. Of teacher_val's
180 VNs, 56 improve under synth4, 83 worsen and 41 are equal.

| stratum | lines | D_s3v3 | F_s4 | F − D, points (VN-cluster 95% CI) |
|---|---|---|---|---|
| native Vita | 161 | 2.95% | 2.59% | **−0.36 (−1.01 to +0.18)** |
| native PSP | 424 | 3.16% | 3.09% | −0.06 (−0.46 to +0.33) |
| Latin / digits | 62 | 10.95% | 10.36% | −0.59 (−2.53 to +0.67) |
| ink 28 px and taller | 44 | 4.03% | 3.55% | −0.47 (−1.62 to +0.58) |
| PC renders scaled (hires) | 1,058 | 6.31% | 6.79% | **+0.48 (+0.23 to +0.74)** |
| dark-on-light | 601 | 7.06% | 7.66% | +0.60 (+0.26 to +0.96) |
| name-like | 706 | 13.43% | 14.11% | +0.68 (−0.15 to +1.50) |
| 1-4 characters | 604 | 13.98% | 14.89% | +0.91 (−0.10 to +1.93) |
| ink under 14 px | 481 | 11.80% | 12.48% | +0.69 (+0.18 to +1.22) |
| tail-any (curriculum class in the line) | 268 | 4.37% | 4.81% | +0.45 (+0.03 to +0.86) |
| tail-kanji | 40 | 7.90% | 8.43% | +0.53 (−0.89 to +1.62) |
| rare kanji (Jiten rank > 2,000) | 116 | 6.48% | 6.73% | +0.25 (−0.68 to +1.07) |

The regression is concentrated on teacher_val's PC-screenshot majority (hires, dark-on-light, short
and name lines); native console strata go the other way. The curriculum's own strata (tail-any,
tail-kanji) do not improve: more exposures of a rare character did not lower its error on these 268
lines at this budget, and the oversampling of the tail cost a little on the common text.

### devnative v3 (guard)

Pooled (strict + tv_native, ink + grad): **F_s4 − D_s3v3 = +0.165 points, VN-cluster 95% CI
[−0.11, +0.47]** — no measured difference, so the guard passes (not entirely worse). By file:

| file | lines | F − D, points (VN-cluster 95% CI) |
|---|---|---|
| strict_ink | 34 | −0.85 (−2.61 to +1.04) |
| strict_grad | 23 | +0.70 (−2.22 to +3.95) |
| tv_native_ink | 509 | −0.16 (−0.40 to +0.10) |
| tv_native_grad | 420 | +0.60 (+0.06 to +1.19) |

The native-Vita strata favour synth4 (native Vita −0.40 [−0.68, −0.19], ink 20-27 px −0.45); native
PSP ink is flat and PSP GRAD slightly worse. Short and name lines are noisier here (len 1-4 +1.74,
CI crosses 0).

The in-training evaluations used study 1b's `--val rv/rvn` (the full teacher_val npz, which still
holds 8 lines of the owner's Memories Off VN v1331); nothing was selected on them (the step-16000
checkpoint is fixed, no checkpoint selection), but a future run should pass the amendment-8 copy as
`--val`. The scoring that the decision rests on read only the amendment-8 copies (no v1331); no
vita-test, psp-test, fresh native, CLANNAD or other protected test was read or scored.

## Development error profile of the VOCR3 selections (report only)

The task's item 1c asked which characters and styles the current VOCR3 selections get wrong on
teacher_val and devnative. That profile was computed only after synth4's design and protocol were
committed (15:57Z), from the VOCR3 selection's own score files (C runtime W8I, console polarity;
`eval/synth4_devprofile.py`, `eval/synth4_devprofile_20261013.json`); it informed nothing above.

| stratum | FL10 selection, teacher_val | FL10, devnative v3 | H15 selection, teacher_val | H15, devnative v3 |
|---|---|---|---|---|
| all | 2.27% (17,050 chars) | 2.32% (11,494) | 2.09% | 2.16% |
| 1-4 characters | 6.73% | 19.12% | 5.53% | 16.01% |
| name-like | 6.58% | 17.89% | 5.60% | 15.48% |
| rare kanji (outside Jiten's top 2,000) | 2.72% | 6.11% | 2.65% | 3.82% |
| curriculum classes, kanji (tail-kanji) | 3.75% (614 chars) | 3.12% (160) | 2.77% | 0.62% |
| characters outside FL10's vocab | 6.76% (74) | 25.0% (8) | 0.00% | 0.00% |
| ink under 14 px | 5.74% | 4.36% | 5.55% | 3.96% |
| kanji / katakana / hiragana (per character) | 4.67 / 2.94 / 0.82% | 3.74 / 3.75 / 0.58% | 4.14 / 3.36 / 0.67% | 3.23 / 4.25 / 0.69% |

On devnative the short and name lines are the selections' worst stratum by far (15-19% CER against
2.2-2.3% overall), as on the unseen training-side lines; rare kanji follow.

## Limits

- One pilot render, one recipe (study 1's 16,000-step pretraining without a real fine-tune), two
  seeds per arm, FL10 only. Whether a difference survives the release recipe (pretrain plus real
  fine-tune) or holds at H15 width is not measured. FL10's 3,500-class vocab reads every curriculum
  class outside it as `<unk>`, so the ablation measures the curriculum only on FL10's classes.
- synth4 changes several things at once (curriculum, raster and font mix, outlines, contrast,
  colour); the ablation measures the package, not each part.
- The style targets come from detectors calibrated on synth3's own metadata; outline, shadow and
  furigana are weak detectors, and the 1-bit measure disagrees with eye checks on real crops
  (docs/synth4-gap.md): synth4's 1-bit share (about 12% of native PSP crops, synth3 49%) is a design
  choice, not a measured match. Native Vita text contrast stays below real (173 against 191 grey
  levels).
- The curriculum leaves 1,336 classes below their K (most above 60 exposures); 28 vocab classes
  have no context at all. The tail-kanji stratum of the development cohorts is small (614 characters
  on teacher_val, 160 on devnative), so its intervals are wide.
- NDS12 covers only kana, ASCII and symbols and drew 53 crops; cinecaption drew 3,538. Neither the
  owner's fonts nor any synth4 crop is redistributed; the open-profile recipe was rendered, not
  trained.
- The 13:10Z owner titles were missed by the first gap-analysis run (98 teacher_train/real_train
  rows); everything here is from the corrected run, and the curriculum the render used is identical
  to the corrected one (docs/synth4-gap.md, Correction).
- The D_s3v3 runs' development scores were published before this protocol (study 1b, the
  native-frames study); they are a fixed baseline, and nothing was selected or tuned on them.
- Labels were checked by a teacher-agreement audit and 36 crops read by eye, not by hand at scale.
- The protocol fixes three curriculum strata (tail-kanji, tail-any, fl10-unk); the result tables show
  tail-kanji and tail-any but not fl10-unk, because only 4 teacher_val lines carry a character outside
  FL10's vocab and `compare` drops strata under 5 lines. FL10 reads those 4 lines' out-of-vocab
  characters as `<unk>` under both arms, so the stratum cannot separate them anyway.
- synth4's Jiten-word carriers are filtered only by the exclusion manifests' text markers (as synth3's
  pool lines are), not by the fuller name-pool reservation: 326 of the 54,460 curriculum words are VNDB
  names of reserved-VN characters (mostly ordinary words; about 7 distinctive names, ~22 crops), in 912
  Jiten-region crops. A later render should filter the carrier words too. The B2 disclosure above counts
  the v661 and Never 7 exposure.
- Owner update 2026-10-13 20:47Z (planetarian v34 is now training data, to be removed from the eval
  and development sets in a new manifest version): it does not affect this work. synth4 reserved v34
  through the CLANNAD-family exclusion, so no synth4 crop renders it, and v34 has 0 lines in the
  amendment-8 teacher_val and devnative copies the ablation scored. Re-versioning the shared eval and
  development sets, and rescoring the released models, is a separate conductor/eval task.
- Nothing here ran on a console.

## Reproduce

```sh
A=$HERD/shared/artifacts; S4=$A/synth4-design; G=$S4/gap
# owner fonts (private, never committed) into $S4/fonts/owner/, then the manifest
synth/synth4.py fonts --out data/manifests/synth4_fonts.json && synth/synth4.py fonts-verify
# the curriculum and focus targets (gap analysis outputs: docs/synth4-gap.md, Reproduce). The pinned
# curriculum_v1.json (sha256 2b366252...) was built at 15:43Z from the first gap run's text.json and label files;
# this command, on the corrected inputs, gives the same quota, words and targets for every class (only the
# file's inputs record differs), which is all the renderer reads
HEAVY_SLOTS=12 $HERD/heavy.sh synth/synth4.py curriculum --exposures $G/exposures_s3v3.json --text $G/r2/text.json \
    --real realplus_v3=$G/r2/inputs/realplus_v3_labels.jsonl --real native_frames=$G/r2/inputs/native_frames_labels.jsonl \
    --real teacher_train=$A/teacher/teacher_train.npz --real real_train=$A/realdata/real_train.npz --mine $G/mine.json \
    --base-targets $A/synth3-v3/text/targets_v3/targets.json --exclude $A/synth3/mining/exclude_dev_vns.txt \
    --out $S4/curriculum/curriculum_v1.json --targets-out $S4/curriculum/targets_s4.json
# the pilot render (4 heavy.sh jobs: --first 0, 25, 50, 75), then the pin
HEAVY_SLOTS=12 $HERD/heavy.sh timeout 7200 synth/synth4.py shards --out $S4/shards/train_v1 --shards 25 --first F \
    --per-shard 10000 --workers 5 --seed 20261013 --curriculum $S4/curriculum/curriculum_v1.json \
    --targets $S4/curriculum/targets_s4.json --scene-hashes $A/synth3-v3/exclude/scene_hashes.json
synth/synth4.py manifest --dir $S4/shards/train_v1 --out data/manifests/synth4_shards_v1.json
# checks: exposures, label audit, calibration, styles
synth/synth4_gap.py exposures --shards "$S4/shards/train_v1/synth4_000??.npz" --out $G/exposures_s4v1.json
.venv/bin/python synth/synth3_audit.py --shards "$S4/shards/train_v1/synth4_000[0-9]?.npz" --n 1500 --seed 1 \
    --out audit.json --sheet worst.png
synth/synth3_stats.py measure --name real=$A/teacher/teacher_train.npz+$A/realdata/real_train.npz \
    --name s3v3="$A/synth3-v3/shards/train_v3/synth3_0000[0-1].npz" --name s4="$S4/shards/train_v1/synth4_0000[0-1].npz" \
    --exclude $A/synth3-v3/mining/forbid_vns_v3.txt --out stats.json
# the ablation (eval/synth4_protocol.json): F_s4 seeds 1 and 2 from a worktree at 2b132aa, then the scoring
git worktree add --detach $S4/train_wt 2b132aaba600ab7dbceeeb1f3a71877d3202e853
TRAIN_REPO=$S4/train_wt HEAVY_SLOTS=10 $HERD/heavy.sh timeout 36000 bash eval/synth4_run.sh 1    # and 2
HEAVY_SLOTS=12 $HERD/heavy.sh timeout 7200 bash eval/synth4_score.sh
# the report-only development profile of the VOCR3 selections
eval/synth4_devprofile.py --select-dir $A/synth3/vocr3/select --model FL10=rp_FL10_C_mix_pre128k_seed1 \
    --model H15=rp_H15_C_mix_seed2 --curriculum $S4/curriculum/curriculum_v1.json \
    --fl10-vocab $A/train2/release/FL10_vocab_3500.txt --out eval/synth4_devprofile_20261013.json
```
