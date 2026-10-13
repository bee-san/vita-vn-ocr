# synth4 gap analysis: what the real VN data says synthetic data must supply

The owner (2026-10-13 14:27Z): real data will not hold enough rare kanji, so synthetic data must;
the model is small, so the synthetic data has to be excellent; decide from the non-synthetic data
what synth4 needs. This document measures the training-side real data against synth3 version 3
(`docs/synth3-v3.md`, the pinned 1,000,000-crop render that studies 1b and native-frames trained
on) and turns the gaps into synth4's design inputs. Every number comes from
`synth/synth4_gap.py`; the result files are in `eval/synth4_gap/`.

## Answers first

- **Vocabulary.** On VN running text (Jiten's visual-novel word list, 2,104 scripts) FL10's
  3,500 classes miss **2.90 of every 1,000 characters** (8.0 per 1,000 kanji), H15's 5,354 miss
  **0.22** (0.67 per 1,000 kanji). On the real training-side labels, which their pipelines already
  filtered to H15's vocab, FL10 cannot emit 0.50-1.19 characters per 1,000. Both vocabs are poorly
  ordered for VN text: H15 holds 335 kanji that the 2,104 scripts never use (one occurrence in
  757,000 real label characters) while lacking 611 that they do; FL10 lacks VN kanji such as 璧 篭 攣
  丼 芯 and the full-width capitals Ｐ Ｖ Ｅ Ｌ Ｕ. Recommendation (section 1): a VN-ordered vocab of
  about **4,986 classes for the always-safe tier** (est. 819,652 B blob, 271 ms, 22 times fewer
  missing characters than FL10's) and **5,630 classes for the preflight tier** (est. 1,185,976 B,
  405 ms, every VN-list kanji). A vocab change needs a new head, so it is a recommendation for the
  next model round; synth4 itself renders H15's vocab.
- **The exposure tail.** In synth3 v3's 1,000,000 crops the median vocab character appears 65
  times. 2,040 characters of H15's vocab that VN text uses appear fewer than 50 times (774 fewer
  than 10, 17 never), 1,229 vocab characters are drawn by fewer than 10 font files, and 66 vocab
  characters are never rendered. Those characters are 0.44% of VN running text. synth4's
  curriculum has to lift them (section 2).
- **Styles.** Measured with the same detectors on real and synthetic crops (section 3), the
  largest gap is **1-bit text**: synth3 draws 49% of its native PSP crops and 15% of its native
  Vita crops without anti-aliasing (pixel fonts and 1-bit raster), while the real native PSP and
  Vita crops are almost all anti-aliased (2.6% and 15% under the threshold, which is the detector's
  false-positive rate on anti-aliased synthetic text). I checked this by eye on a contact sheet.
  synth3 also has more outlined text (truth 68%; real estimates 0-60% by kind), lower-contrast
  text (p50 134-153 grey levels against 145-191), and more coloured fills (17% against 10.5%, with
  three times the share of green). Ink heights, stroke weights, furigana and background texture
  are close. 
- **Errors.** On 2,890 realplus v3 lines that no model has trained on (8,238 crops), the VOCR3
  selections read 2.56% (FL10 class) and 2.23% (H15 class) CER through the C runtime. The errors
  concentrate where the real data is thin: 1-4 character lines and names (8-10%), lines with
  furigana (6-8%), rare kanji (4-5%), PSP-scale crops (4.0-4.4% against 1.2-1.5% at Vita scale),
  katakana (3.5-4.0% of characters) and kanji (4.5-5.1%), and the dakuten pairs (プ/ブ, バ/パ,
  ペ/ベ, て/で, か/が), 。/、, 』/」, ー/一 and っ/つ (section 4).

## Data and hygiene

Training-side only, as the task and the conductor (14:45Z) require:

| source | what is used | lines |
|---|---|---|
| realplus v2 (`realplus_train_{ink,grad}_v2.npz`, #57) | crops (styles) | 20,000 + 20,000 sampled |
| realplus v3 (`$HERD/shared/artifacts/realplus-v3/labels.jsonl`, in progress) | kept labels (text); the 4,308 lines kept since v2 (mining) | 50,526 |
| native-frames pool (`native-frames/pool/labels.jsonl`, in progress) | kept labels (text) | 1,839 |
| teacher_train + real_train | labels (text), crops (styles), RGB cuts of their screenshots (colour) | 16,014 / 20,000 / 11,432 |
| Jiten VN word list (`frequency_list_visual_novel.csv`, CC BY-SA 4.0, as synth3) | 249,118 words with ranks | |
| VNDB character names (`vndb_chars.tsv`, synth3's dump) | 103,104 Japanese names of usable VNs | |
| synth3 v3 (`data/manifests/synth3_shards_v3.json`) | all 100 shards' labels and metadata (exposures), shards 0-1 (styles), 3,000 regions rendered again in RGB with seed 4242 (colour) | 1,000,000 |

Every row of a reserved VN is dropped and counted (`reserved_dropped` in each result file): eval's
test games (`test_games.json` "all"), `is_test_vn`, the fresh native textbox titles, CLANNAD and
its family, the owner's Memories Off series (the three synth3 exclusion manifests), the owner's
13:10Z set (`native-frames/owner_games_vns.json`, 50 VNs) and the development VNs
(`exclude_dev_vns.txt`: teacher_val's 181 and the 9 general-VN ones).

The development cohorts (teacher_val and devnative v3, the amendment-8 copies the VOCR3 selection
uses) appear only in section 3's style table, as a check that the training-side measurements
describe them too. They are not a design input, and no model was scored on them here. The
conductor's rule for this work: mining for design runs on training-side pools only, because the
synth4 ablation is scored on the development cohorts. The task also asked for the selections'
errors on teacher_val and devnative; those are reported in `docs/synth4.md` after the synth4
design and its ablation protocol are committed, so they cannot steer either.

## 1. Vocabulary: coverage of VN running text

Jiten publishes ranks, not counts. A word of rank r is weighted r^-s and each character of its
dictionary form counts with that weight. s is fitted so that the kanji distribution it implies is
closest (KL) to the kanji of the 68,379 real training-side labels below: s = 0.95 (KL 0.166; s = 0.7:
0.205, s = 1.4: 0.372). The real labels themselves cannot show what lies beyond H15's vocab,
because every labelling pipeline dropped such lines, so they only measure FL10.

| vocab | classes | VN characters outside, per 1,000 | VN kanji outside, per 1,000 |
|---|---|---|---|
| FL10 (`FL10_vocab_3500.txt`) | 3,500 | 2.903 | 8.035 |
| H15 (`data/vocab.txt`) | 5,354 | 0.224 | 0.672 |

| real training-side labels | lines | VNs | characters | FL10 cannot emit, per 1,000 | lines holding one |
|---|---|---|---|---|---|
| realplus v3 (kept) | 50,526 | 9,643 | 565,544 | 0.626 | 332 |
| native-frames pool (kept) | 1,839 | 226 | 18,571 | 1.185 | 17 |
| teacher_train | 9,178 | 1,240 | 97,196 | 0.504 | 45 |
| real_train | 6,836 | 1,198 | 75,583 | 0.662 | 42 |

(Labels are counted once per VN. The native-frames pool, the only native-console source, has the
highest FL10 gap.) VNDB character names: 1.72% of the 103,104 names hold a character outside
FL10's vocab, 0.46% one outside H15's (鴇 椛 杠 洸 椚 檸 檬 ... lead).

FL10's most frequent misses by VN weight are 膣 璧 篭 攣 咥 猥 芯 丼 痙 腺 痒 軋 茹 炒 and the full-width
capitals Ｐ Ｖ Ｅ Ｌ Ｕ: 100 of H15's non-kanji classes are not in FL10 (full-width Latin, Greek,
Cyrillic, box drawing, ①②③). H15's most frequent misses are 侭 蠕 腟 剝 搔 藹 諤 鬩 嚙 吽 鴇 忸 怩 (several
are JIS X 0213 forms such as 剝 搔 嚙 繫 吞). Every one of the 611 VN kanji H15 lacks is drawn by at
least 20 of the 244 smooth font files of synth3's pool, so synthetic data can supply them.

### Vocab size per tier

The head costs about 109 bytes and 0.024 ms (high bound, 640-px line, 333 MHz) per class at either
width, and the arena does not depend on it (`docs/vita-memory-budget.md`'s sweep: FL10 → `w100`
and `w150v3500` → `w150v7000` → `w150v11000`). A VN-ordered vocab is H15's 486 non-kanji classes
plus the N kanji with the most VN weight:

| VN kanji | classes | VN chars outside, per 1,000 | FL10 topology: est. blob, ms | H15 topology: est. blob, ms |
|---|---|---|---|---|
| 2,500 | 2,986 | 3.507 | 601,486 B, 222.5 ms | 897,463 B, 340.6 ms |
| 3,000 | 3,486 | 1.597 | 656,028 B, 234.6 ms | 952,023 B, 352.7 ms |
| 3,500 | 3,986 | 0.716 | 710,569 B, 246.8 ms | 1,006,583 B, 364.8 ms |
| 4,000 | 4,486 | 0.318 | 765,111 B, 258.9 ms | 1,061,143 B, 376.9 ms |
| **4,500** | **4,986** | **0.133** | **819,652 B, 271.0 ms** | 1,115,703 B, 389.1 ms |
| 5,000 | 5,486 | 0.027 | 874,194 B, 283.2 ms | 1,170,263 B, 401.2 ms |
| all 5,144 | **5,630** | 0 | 889,902 B, 286.7 ms | **1,185,976 B, 404.7 ms** |

Tiers (#60): always safe ≤ 900,000 B, ≤ 131,072 B arena, ≤ 300 ms; preflight ≤ 1,600,000 B,
≤ 262,144 B, ≤ 550 ms. Blob and time are estimates from the measured per-class costs, not exports.

- **Always-safe tier (FL10 class): about 4,986 VN-ordered classes.** It cuts the missing share
  from 2.90 to 0.13 per 1,000 VN characters for an estimated +162 KB and +36 ms, and leaves 80 KB
  and 29 ms of the tier's margin. Even at FL10's own size a VN-ordered vocab misses half as much
  (3,486 classes: 1.60 per 1,000). All 5,630 classes also fit (889,902 B, 287 ms) but leave 10 KB
  and 13 ms. With real data the width-1.0 model was level with either vocab in the architecture
  sweep (`v3500_r` +0.06 points, CI -0.61 to +0.73), so the larger head has no measured cost.
- **Preflight tier (H15 class): 5,630 classes**, H15's vocab without the 335 kanji the VN scripts
  never use and with the 611 they use: est. +30 KB and +7 ms. Adding only the top 300 of the 611
  leaves 0.06 per 1,000.
- Neither change is in this round: a new head means a new model. synth4 renders H15's vocab and
  its curriculum gives every class enough exposures, so a later vocab change only needs the
  missing characters added to the renderer's vocab (`--vocab`).

## 2. The exposure tail of synth3

Occurrences of each vocab character in the labels of the 1,000,000 pinned v3 crops (a 16,000-step
FL10 pretrain at 96 lines and a 0.5 synthetic share streams 768,000 crops, 0.77 of the render):

| exposures in synth3 v3 | H15 vocab chars that VN text uses | of them kanji | their share of VN text, per 1,000 | FL10 vocab chars | share, per 1,000 |
|---|---|---|---|---|---|
| < 1 | 17 | 17 | 0.006 | 0 | 0 |
| < 10 | 774 | 762 | 0.558 | 48 | 0.015 |
| < 50 | 2,040 | 2,001 | 4.430 | 551 | 2.086 |
| < 100 | 2,501 | 2,441 | 9.164 | 987 | 6.686 |
| < 300 | 3,218 | 3,116 | 27.85 | 1,697 | 25.20 |

- H15's vocab: exposures p10 / p50 / p90 = 4 / 65 / 2,087; 66 classes never rendered (ASCII ! ( ) ?
  ~, Greek, Cyrillic, ‐ ⇔ ∀ and old-form kanji); 4,874 of the 5,354 classes occur in VN text.
- Font variety: 143 classes are drawn by one font file only, 575 by fewer than 5, 1,229 by fewer
  than 10 (FL10's vocab: 27, 75, 151).
- synth3's focus sampling (25% of dialogue lines built around a tail-weighted character) can only
  pick a character that a text pool line holds after vocab filtering; the characters below 10
  exposures mostly have no such line. synth4 needs contexts for them that are real words (the
  Jiten VN words holding the character), not random strings.

## 3. Styles: real crops against synth3

`synth4_gap.py styles` measures every crop the same way (grey crop as stored, console polarity,
text oriented bright):

| feature | definition |
|---|---|
| ink height | `px_ink()` (teacher's refine() rule, as synth3_stats.py) |
| contrast | p99 of the ink rows minus the border-ring median |
| aa | share of text-edge pixels at 20-80% of the way from background to text level |
| stroke | p25 of the horizontal and vertical runs of text pixels in the ink box |
| halo (outline) | outer background level minus the darkest quarter of the 1-2 px ring around the fill, over contrast |
| shadow | ring above-left minus ring below-right, over contrast |
| nearest2x | even/odd asymmetry of equal neighbour pairs at text edges (a PSP frame upscaled 2x nearest) |
| ruby | an ink band right above the main band, 0.2-0.7 of its height, under half its ink |
| texture | std of the background pixels outside the dilated text |

Each binary detector's threshold maximises TPR - FPR against synth3's own metadata (what the
generator drew) on 19,975 crops of shards 0-1; real prevalence is then estimated with the
Rogan-Gladen correction (observed - FPR) / (TPR - FPR), per capture kind where the kind has enough
synthetic crops of both classes ("-" where the correction is undefined):

| detector | threshold | TPR | FPR |
|---|---|---|---|
| binary (1-bit text: pixel fonts at their strike, 1-bit raster) | aa < 0.389 | 0.53 | 0.10 (0.07 native PSP, 0.22 native Vita) |
| outline | halo > 0.096 | 0.57 | 0.24 |
| shadow (no outline) | > 0.021 | 0.60 | 0.20 |
| nearest 2x | > 0.109 | 0.93 | 0.04 |
| furigana | band rule | 0.22 | 0.005 |

The outline, shadow and furigana detectors are weak (they confuse anti-aliasing, JPEG and busy
scenes with halos), so their estimates are coarse; binary and nearest-2x separate well.

| source, target:kind | crops | ink h p10/50/90 | contrast p50 | aa p10/50/90 | binary obs (est) | outline obs (est) | shadow obs (est) | nearest 2x obs | furigana obs (est) |
|---|---|---|---|---|---|---|---|---|---|
| synth3 v3, native Vita | 4,817 | 19/25/36 | 153 | .22/.52/.79 | .291 (truth .147) | .457 (truth .609) | .282 (truth .455) | .117 | .016 (truth .049) |
| real, native Vita | 1,646 | 18/24/36 | 191 | .36/.51/.72 | .153 (0) | .375 (.38) | .207 (.20) | .036 | .006 (.01) |
| synth3 v3, native PSP | 4,692 | 11/16/22 | 146 | .00/.52/.81 | .366 (truth .493) | .557 (truth .730) | .371 (truth .470) | .036 | .017 (truth .054) |
| real, native PSP | 3,766 | 12/15/23 | 177 | .48/.64/.79 | .026 (0) | .314 (0) | .279 (.19) | .033 | .012 (.03) |
| synth3 v3, PC renders scaled to Vita | 5,748 | 15/23/34 | 138 | .42/.65/.84 | .067 (truth .066) | .465 (truth .676) | .346 (truth .494) | .027 | .015 (truth .064) |
| real, PC shots scaled to Vita | 10,094 | 15/21/31 | 168 | .51/.69/.84 | .020 | .411 (.60) | .373 (.52) | .035 | .009 (.02) |
| realplus v2 ink, scaled to Vita | 12,240 | 15/20/29 | 181 | .50/.66/.81 | .018 | .324 (.33) | .385 (.56) | .028 | .004 (0) |
| synth3 v3, PC renders scaled to PSP | 3,776 | 10/14/21 | 134 | .51/.72/.86 | .022 | .389 (truth .677) | .397 (truth .510) | .073 | .019 (truth .056) |
| real, PC shots scaled to PSP | 4,483 | 10/14/20 | 145 | .62/.77/.88 | .007 | .344 (.39) | .357 (.47) | .057 | .023 (.09) |
| realplus v2 ink, scaled to PSP | 7,755 | 9/11/16 | 166 | .62/.76/.86 | .008 | .245 (.08) | .320 (.35) | .126 | .005 (0) |
| synth3 v3, PSP upscaled on a Vita | 942 | 22/31/43 | 134 | .32/.74/.87 | .118 (truth .468) | .357 | .305 | .455 (truth .331) | .013 |
| dev: teacher_val, native PSP / Vita | 424 / 161 | 12/14/22, 19/24/31 | 181 / 191 | .44/.63/.80, .40/.54/.74 | .050 / .074 | .290 / .280 | .307 / .329 | .033 / .056 | .007 / .012 |
| dev: devnative v3, native PSP / Vita | 784 / 202 | 12/14/27, 20/25/34 | 177 / 204 | .48/.65/.80, .42/.53/.74 | .041 / .040 | .305 / .233 | .367 / .272 | .028 / .035 | .006 / .000 |

"real" is teacher_train + real_train (20,000 sampled crops). The full table, with background
texture, trends and realplus GRAD crops, is `tables eval/synth4_gap/stylesum_20261013.json`.

What it shows:
- **1-bit text is over-represented.** synth3 draws 49% of native PSP crops (28% pixel fonts plus
  1-bit raster on 28% of the rest) and 15% of native Vita crops without anti-aliasing. The real
  native PSP crops sit almost entirely above the threshold (aa p10 0.48 against synth3's 0.00): 2.6%
  fall below it, about the detector's false-positive rate on anti-aliased synthetic PSP text (7%),
  so the estimate is 0. Native Vita: 15% below against a 22% false-positive rate. The development
  cohorts agree (4-7%). On a contact sheet of the 10 lowest-aa and 10 median-aa crops of 400 random
  real native PSP crops (3x nearest; not kept), the lowest were crisp bitmap-like text (system
  menus, a few dialogue fonts) and the median ones anti-aliased glyphs, so the measure reads what
  it should. Pixel fonts at their native
  strike do occur in real PSP titles, but rarely.
- **Outlines are over-represented** (synth3 61-73% of crops by kind; real estimates 0-60%,
  realplus 8-33%), with the caveat of a weak detector. Shadows are close.
- **Text contrast is too low** on every kind: synth3 p50 134-153, real 145-191 grey levels. The
  translucent boxes, gamma and JPEG chains of v3 together overshoot.
- **Ink heights match** per kind (synth3's scaled Vita text is 1-2 px taller at the median);
  realplus's PSP-scaled crops are smaller (ink p50 11 px) than teacher's (14 px).
- **Furigana**: the detector sees it on 1.6% of synth3 and 0.6-2.3% of real crops; with its 22%
  recall that is about 5% (synth3's truth) against 1-9%: no measured gap.
- **Nearest-2x frames** (a PSP game on a Vita) exist only in synth3 (33% of its 5% psp2x share);
  the training-side real crops have none, and the owner's Adrenaline captures, the case they stand
  for, are test-only. Keep them small.
- **Background texture** (std outside the text) is close: synth3 p50 15.8, real 17.6 (native
  13-16 both); PC shots scaled to PSP are busier in real crops (20.1 against 16.4).
- **Names**: 35.4% of synth3 crops are name-like (at most 6 characters, no dialogue punctuation),
  real 36.7-39.9% (teacher_val 42.9%).

### Colour and gradient fills

`synth4_gap.py colour` cuts each crop again from its screenshot in RGB (teacher_train; native
shots as they are, PC shots scaled as teacher scales them; a cut must equal the stored grey crop
to the grey level) and renders 3,000 synth3 regions in RGB (seed 4242, never a training seed). A
fill (text pixels at 70% of the way from background to text level) is coloured when its median
HSV saturation exceeds 0.35 at a value above 0.25:

| source | crops | coloured fill | partly coloured | gradient (detector) | hue of coloured fills r / g / b |
|---|---|---|---|---|---|
| real (teacher_train) | 11,432 | 0.105 | 0.102 | 0.009 | .59 / .09 / .32 |
| real, native PSP | 2,377 | 0.079 | 0.082 | 0.004 | .65 / .13 / .22 |
| real, native Vita | 978 | 0.110 | 0.114 | 0.002 | .77 / .03 / .20 |
| synth3 v3 | 6,926 | 0.172 | 0.147 | 0.011 (truth 0.094) | .46 / .23 / .31 |
| synth3 v3, native PSP | 1,724 | 0.121 | 0.141 | 0.009 | .50 / .24 / .26 |
| synth3 v3, native Vita | 1,641 | 0.169 | 0.126 | 0.015 | .45 / .22 / .34 |

synth3 has 1.6 times the real share of coloured fills, and a quarter of them are green against a
tenth of the real ones (real coloured text is mostly red/orange/pink and blue). The gradient
detector finds only 6% of synth3's gradient fills (FPR 0.6%), so its real estimate is about 5%
(Rogan-Gladen) against synth3's 9.4%: weak evidence of too many gradients.

## 4. Errors of the current selections on unseen training-side lines

The mining set is every realplus v3 line kept since v2 (Claude and Luna agree, or both engine
families do; 4,308 lines) that is not on a reserved VN and does not wait for an edge check, cut by
main's `realplus/pool.py` exactly as realplus builds its pool: 2,890 lines from 2,217 VNs, as
2,862 Vita ink, 2,343 Vita GRAD, 1,811 PSP ink and 1,222 PSP GRAD crops (8,238 crops, 126,604
characters; `minepool_20261013.json`). No VOCR3 candidate trained on these lines: every fine-tune
read realplus v2, which they are not in. They are PC and home-console shots scaled to the console;
the native-frames pool could not serve, because 1,910 of its 1,924 kept lines sit on screenshots
that teacher_train, real_train, realplus v2 or the pseudo-label pools already hold
(`nativepool_20261013.json`). Labels are engine-plus-assistant agreed, not hand-checked, so a
share of the "errors" are label errors.

Each crop is read through the C runtime (W8I, console polarity, as the console reads it) by the
VOCR3 selections (`rp_FL10_C_mix_pre128k_seed1`, the FL10-class pick of 13:32Z, and
`rp_H15_C_mix_seed2`, the H15 pick of 12:15Z) and, for reference, the released v0.2.0 FL10 and
H15. Strata come from the crop's metadata, label and section 3's crop measures, never from a
model's output.

| stratum | chars | FL10 selection | H15 selection | FL10 v0.2.0 | H15 v0.2.0 |
|---|---|---|---|---|---|
| all | 126,604 | 2.56% | 2.23% | 4.57% | 2.69% |
| Vita-scale crops | 80,350 | 1.50% | 1.23% | 2.95% | 1.56% |
| PSP-scale crops | 46,254 | 4.41% | 3.98% | 7.39% | 4.66% |
| ink crops / GRAD crops | 71,510 / 55,094 | 1.93% / 3.40% | 1.65% / 2.98% | 3.60% / 5.83% | 2.10% / 3.45% |
| 1-4 characters | 4,121 | 9.97% | 8.47% | 15.43% | 10.87% |
| name-like | 5,149 | 9.69% | 8.27% | 14.70% | 10.33% |
| furigana (detected) | 1,004 | 8.47% | 6.28% | 12.15% | 8.07% |
| rare kanji (outside Jiten's top 2,000) | 10,373 | 4.89% | 4.02% | 7.90% | 4.98% |
| ink 28 px and taller | 14,864 | 4.02% | 3.46% | 7.50% | 4.08% |
| busy background (texture > 20) | 56,214 | 3.39% | 2.94% | 5.79% | 3.42% |
| dark text (inverted) | 16,955 | 3.23% | 2.73% | 5.77% | 3.43% |
| outline (detected halo) | 38,267 | 2.70% | 2.34% | 4.87% | 2.77% |

| per reference character | FL10 selection | H15 selection |
|---|---|---|
| hiragana (70,228) | 1.16% | 0.98% |
| katakana (8,397) | 4.03% | 3.45% |
| kanji (31,025) | 5.09% | 4.49% |
| Latin letters and digits (449) | 12.0% | 10.0% |
| punctuation and symbols (16,505) | 2.24% | 1.87% |

Top confusions of the FL10 selection (reference → read, count): 。→、 30, き→さ 24, 』→」 21,
、→。 21, プ→ブ 20, バ→パ 17, ー→一 16, て→で 16, ブ→プ 14, で→て 14, ペ→ベ 14, パ→バ 13, 「→『 12, っ→つ 11.
H15's are the same set (』→」 24 first). The worst characters with at least 20 occurrences are
katakana with handakuten (ペ 16 of 33 wrong for FL10, プ 23 of 82, パ, ポ) and kanji of names and
rare words (薬 透 霊 麗 観 智 菜). The released FL10 also confuses ぁ→あ (33): the VOCR3 FL10
selection fixed that. The most frequent insertion is ・ (17).

## What synth4 takes from this

| gap | measured | synth4 |
|---|---|---|
| rare vocab characters | 2,040 VN characters under 50 exposures; 1,229 under 10 fonts | a curriculum: every vocab class at least K times across at least F fonts, in real-word contexts (Jiten VN words, pool lines) |
| 1-bit text | PSP 49% synthetic against ~0-3% real; Vita 15% against ~0 | 1-bit share cut to a floor (pixel fonts kept for variety, the owner's NDS12 among them, at a far smaller total share) |
| outlines | 61-73% synthetic against 0-60% real (weak detector) | outline rates lowered toward the real estimates |
| contrast | p50 134-153 against 145-191 | box opacity and gamma moved toward higher contrast |
| coloured fills | 17% against 10.5%; green 23% of them against 9% | coloured spans and lines rarer; palette away from green |
| short lines, names, furigana, dakuten, 。/、, 』/」, katakana | the selections' worst strata on unseen lines | focus weights from this mining (training-side), name and short-line shares kept or raised |

These are inputs; synth4's design and its ablation are in `docs/synth4.md`.

## Limits

- The Jiten list gives ranks, so its weights are a Zipf model fitted to real labels (s = 0.95),
  not counts; lemmas, not surface forms; and whatever Jiten's parser calls a word (names may be
  missing). The real labels are VN screenshot text but were filtered to H15's vocab.
- Tier costs are extrapolated from measured per-class costs of exported sweep models, not from
  exports of these vocabs.
- The style detectors were calibrated on synth3's own crops; outline, shadow and furigana are weak
  (TPR 0.22-0.60), so their real estimates are coarse. The binary measure was checked by eye on one
  contact sheet.
- Colour was measured on teacher_train only (its boxes index its screenshots); synth3's side is a
  fresh RGB render of the same generator, not the pinned shards.
- The mining set is PC shots scaled to the console (no unseen native console lines exist yet), and
  its labels are not hand-checked.
- Nothing here ran on a console.

## Reproduce

```sh
A=$HERD/shared/artifacts; G=$A/synth4-design/gap; X="$A/synth3/mining/exclude_dev_vns.txt $A/native-frames/owner_games_vns.json"
HEAVY_SLOTS=12 $HERD/heavy.sh synth/synth4_gap.py exposures --shards "$A/synth3-v3/shards/train_v3/synth3_000??.npz" --hash --out $G/exposures_s3v3.json
HEAVY_SLOTS=12 $HERD/heavy.sh synth/synth4_gap.py text --exposures $G/exposures_s3v3.json \
    --jiten-vn $A/synth3/text/jiten/frequency_list_visual_novel.csv --real realplus_v3=$A/realplus-v3/labels.jsonl \
    --real native_frames=$A/native-frames/pool/labels.jsonl --real teacher_train=$A/teacher/teacher_train.npz \
    --real real_train=$A/realdata/real_train.npz --vndb-chars $A/synth3/text/vndb_chars.tsv \
    --vocab-fl10 $A/train2/release/FL10_vocab_3500.txt --exclude $X --out $G/text.json
HEAVY_SLOTS=12 $HERD/heavy.sh synth/synth4_gap.py styles --limit 20000 --source "s3v3=$A/synth3-v3/shards/train_v3/synth3_0000[0-1].npz" \
    --source "real_tt_rt=$A/teacher/teacher_train.npz+$A/realdata/real_train.npz" \
    --source realplus_ink=$A/realplus/realplus_train_ink_v2.npz --source realplus_grad=$A/realplus/realplus_train_grad_v2.npz \
    --exclude $X --out-dir $G/styles --out $G/styles/styles.json
S=$A/synth3/vocr3/select_pools      # development cohorts: reported only
HEAVY_SLOTS=12 $HERD/heavy.sh synth/synth4_gap.py styles --limit 20000 --source dev_teacher_val=$S/teacher_val.npz \
    --source "dev_devnative=$S/devnative_v3/devnative_*.npz" --exclude $A/native-frames/owner_games_vns.json \
    --out-dir $G/styles --out $G/styles/styles_dev.json
synth/synth4_gap.py stylesum --synth $G/styles/s3v3.features.jsonl --source real_tt_rt=$G/styles/real_tt_rt.features.jsonl \
    --source realplus_ink=... --source realplus_grad=... --source dev_teacher_val=... --source dev_devnative=... --out $G/styles/stylesum.json
HEAVY_SLOTS=12 $HERD/heavy.sh synth/synth4_gap.py colour --real $A/teacher/teacher_train.npz --synth-regions 3000 \
    --scene-hashes $A/synth3-v3/exclude/scene_hashes.json --targets $A/synth3-v3/text/targets_v3/targets.json --exclude $X --out $G/colour.json
HEAVY_SLOTS=12 $HERD/heavy.sh synth/synth4_gap.py minepool --realplus $A/realplus-v3 --v2-labels $A/realplus/labels.jsonl \
    --exclude $X --out-dir $G/minepool --out $G/minepool/minepool.json
HEAVY_SLOTS=12 $HERD/heavy.sh synth/synth4_gap.py mine --pool vita_ink=$G/minepool/minepool_vita_ink.npz ... \
    --blob FL10_sel=$A/synth3/vocr3/blobs/rp_FL10_C_mix_pre128k_seed1_w8.vocr --blob H15_sel=$A/synth3/vocr3/blobs/rp_H15_C_mix_seed2_w8.vocr \
    --blob FL10_v020=$A/train2/release/FL10_w8.vocr --blob H15_v020=$A/train2/release/H15_w8.vocr --out $G/mine.json
synth/synth4_gap.py tables eval/synth4_gap/text_20261013.json      # and stylesum, colour, mine
```

The realplus v3 and native-frames labels were read as they stood at 2026-10-13 15:00Z (both pools
are still growing); `minepool_20261013.json` records the counts.
