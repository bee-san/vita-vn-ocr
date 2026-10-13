# synth3: synthetic VN dialogue lines calibrated to real console crops

synth3 is a new synthetic training set for the recognizer: grey line crops of Japanese
visual-novel dialogue, names, choices and UI as the PS Vita and PSP show them, in shards that
`train/train2.py` reads unchanged, plus a development split whose text, scenes and names share
no source document with training. It is meant to replace synth2 as the VN-style source in
train2's mixture, or to be mixed with it. The generator is `synth/synth3.py`.

**Status (2026-10-13 00:30Z).**
- **Study 1 (frozen ablation, short recipe): under the protocol's rule the mix of synth2 and
  synth3 is the recommended VN-style synthetic data.** FL10 models pretrained for 16,000 steps on
  NVIDIA crops plus synth2 (A), synth3 (B) or half of each (C), two seeds per arm, read through the
  C runtime (W8I) on the development cohort teacher_val (1,651 real lines, 181 VNs):
  - C − A = −0.51 points of CER (VN-cluster 95% CI −0.76 to −0.26) and C − B = −0.49 (CI −0.74
    to −0.25), with both seeds below zero in both comparisons.
  - The gain is measured on teacher_val's PC/console screenshots scaled to console size (64% of
    its lines: C − A −0.70, CI −1.02 to −0.39). On its 587 native Vita and PSP lines no arm
    separates from another (C − A −0.09, CI −0.53 to +0.31).
  - synth3 alone shows no measured difference from synth2 (B − A −0.01, CI −0.34 to +0.30).
  - On the protected test sets (scored once, after the decision was committed) no comparison is
    resolved: pooled C − A −0.16 (CI −0.62 to +0.26), B − A −0.12, C − B −0.04 (console
    polarity).
- **Secondaries (exploratory, one run per arm).** After a 5,000-step real fine-tune the mix still
  leads (ft C − ft A −0.32, CI −0.57 to −0.08), with less than half the pre-fine-tune gap. At
  H15 width synth3 alone is lower than synth2 in one run per arm (−0.52, CI −0.80 to −0.25; the
  interval covers line sampling only; not replicated), also on the native lines (−0.38, CI −0.70
  to −0.10); the test sets do not resolve it.
- **Study 2 (full release recipe), FL10: no measured difference** between the mix and synth2
  (C − A −0.07, CI −0.24 to +0.09 on teacher_val; −0.09, CI −0.29 to +0.12 on the native
  cohort devnative), so synth2 stays at FL10 size. The real fine-tune absorbs most of the
  pretrain gap (pretrain-only C − A −0.46). Both arms reach 3.3-3.4% on teacher_val, against the
  released FL10's 4.05%. H15 (2 seeds per arm): the same, no measured difference (C − A −0.14, CI
  −0.30 to +0.02), both arms level with the released H15's 2.29%.
- **Pretrain length matters more than the synthetic arm:** FL10 C_mix pretrained 128,000 steps instead
  of 32,000 reaches 2.60% after the same fine-tune (−0.70, CI −1.03 to −0.40; one seed). **More synth3
  crops do not:** the learning curve is flat from 0.25M to 2M crops in study 1's recipe (−0.10 points).

Commit hashes cited in this document, the protocols and the result files refer to the branch's
history before it was rebased onto main; that history, with its original commit times, is kept at
tag `synth3-history-20261012`.

## What changed from synth2, and why

Every choice below was set from training-side evidence only: real crops of teacher_train and
real_train (statistics), the released FL10 and H15 against PP-OCRv6 small on training-side
unlabelled pools (errors), and the research notes listed under Sources. No development or test
line informed it.

### Measured gaps of synth2

`synth/synth3_stats.py` on 20,000 random real training crops (teacher_train + real_train) and
20,000 synth2 crops (shards 00000-00001):

| | real | synth2 |
|---|---|---|
| label length p10 / p50 / p90 | 2 / 8 / 24 | 4 / 15 / 25 |
| crops of 2-3 characters | 24.0% | 7.5% |
| name-like crops (at most 6 characters, no dialogue punctuation) | 40% | 16% |
| ！ / ？ / ～ / ー per 1,000 characters | 6.8 / 6.8 / 1.3 / 8.0 | 1.7 / 3.2 / 0.1 / 2.7 |
| 、 per 1,000 characters | 26.7 | 39.1 |
| crop height p10 / p50 / p90, PSP target | 16 / 19 / 25 | 19 / 22 / 26 |
| crop height p10 / p50 / p90, Vita target | 23 / 28 / 36 | 26 / 32 / 40 |
| contrast p50 / edge busyness p50 | 206 / 21.6 | 191 / 16.5 |
| KL(real ‖ source) of the character distribution (20k lines each) | | 0.143 |

synth2 lines are about twice as long as the real crops, it has a third as many short crops
(speaker names, 「えっ？」), too few exclamations and long vowels, too many commas (Aozora prose),
and its crops sit looser around the text.

Teacher's character boxes give the real ink geometry per capture kind (teacher_train):

| kind, target | crops | crop height p50 | ink height p10 / p50 / p90 | margin above / below ink, x ink height |
|---|---|---|---|---|
| native Vita | 998 | 32 | 17 / 24 / 30 | 0.12 / 0.19 |
| native PSP | 2,474 | 19 | 12 / 15 / 17 | 0.07 / 0.15 |
| PC/console shot scaled to Vita | 6,355 | 27 | 13.5 / 18.7 / 24 | 0.20 / 0.28 |
| PC/console shot scaled to PSP | 2,288 | 16 | 9.5 / 10.9 / 13.7 | 0.20 / 0.27 |

Two thirds of the real training and development crops are PC or home-console screenshots
area-scaled to a console resolution (teacher's `hires` kind). synth2 has no such bucket.

### Where the released models fail (hard-example mining)

`synth/synth3_mine.py` read 3,339 training-side real lines with the released int8 blobs through
the C runtime (W8I). The pools are teacher_distill_train and realdata's real_unlabelled
(training-split games only; no teacher_val or test VN), and a line counts only
when PP-OCRv6 small reads it with lowest character confidence at least 0.9 and another reader
agrees. These are consensus references, not hand labels, and part of this pool was a
pseudo-label source in train2's fine-tune. The miner's forbid list held teacher_val's 181 VNs but
not yet the nine general-VN and CLANNAD development VNs, so 26 of the 3,339 lines come from six of
those (v1015, v1281, v12775, v13766, v5154, v5297); they are 0.8% of the lines behind the focus
weights.

| stratum | lines | FL10 CER | H15 CER |
|---|---|---|---|
| all | 3,339 | 2.56% | 1.24% |
| 1-4 characters | 1,182 | 5.03% | 2.78% |
| name-like | 1,368 | 5.61% | 3.16% |
| 25+ characters | 320 | 1.98% | 0.72% |
| console polarity dark (inverted) | 566 | 4.24% | 2.42% |
| ink 32 px and taller | 572 | 3.82% | 1.74% |
| native Vita | 259 | 3.41% | 1.79% |
| Latin letters and digits (by character) | 295 chars | 18.0% | 9.8% |
| kanji / katakana / hiragana (by character) | | 3.6% / 3.3% / 1.1% | 1.7% / 1.6% / 0.3% |

The most frequent single error of both models is an inserted ― (16 for FL10, 22 for H15). The
most frequent substitutions are dakuten and handakuten (バ→パ, プ→ブ, ペ→ベ, ば→は, が→か, じ→し,
ず→す), small kana (っ→つ, ゃ→や), 『』→「」, ー/一, and dropped edge punctuation (」、。…！). Some of
H15's top kanji "substitutions" are Chinese variants outside `data/vocab.txt` in the consensus
references (值→値, 歷→歴, 擊→撃, 錄→録): reference artifacts counted as model errors. Crops of 1-4
characters have 2.5x (FL10) and 3.9x (H15) the CER of crops of 25 or more.

### Design

| part | synth3 | evidence |
|---|---|---|
| text | 14 pools, split by source document (1 in 20 documents is dev): Aozora speech in 「」 (modern-kana books only, by 作品ID), Aozora narration, Wikipedia, Tatoeba, JESC and OpenSubtitles lines, NUCC, JDD and BSD conversations, synthetic Japanese roleplay (Aratako), なりきり roleplay posts, Wiktionary and Wikiquote examples, plus templates (interjections, stammers, onomatopoeia, name calls). Lines with any character outside `data/vocab.txt` are dropped whole | real VN text is short spoken lines with ！？…ー; Aozora prose alone over-represents 、 and long sentences (table above) |
| dressing | 「」 on 60% of named and 30% of unnamed speech (『』 3%), terminal punctuation on subtitle lines that lack it, interjections, stammers, ……, elongations, VN-style Latin and digits (CG, Lv, 3年B組, 18:30) | measured punctuation rates; Latin/digits are FL10's worst script (18% CER) |
| focus characters | 25% of dialogue (12.5% of narration) is a natural line chosen to contain one vocab character drawn with weight (exposure so far + 20)^-0.6 x (1 + 4 x mined error rate, with a bonus for dakuten, small-kana and similar-kanji sets) x VN frequency^0.35 (real labels + the Jiten visual-novel word list) | SynthTIGER's character-distribution control (+1.0 to +2.5 points on Japanese at p 0.25-0.5, -5.6 at p 1.0); a tail weight alone pulled classical-Chinese contexts in from Wiktionary, so dictionary pools are a last resort |
| lengths | rejection sampling of crops toward the real label-length histogram (floor 15%) | SynthTIGER's length control (+2.1); 24% of real crops are 2-3 characters |
| names | VNDB character names (102,536 kept after removing every character of a test VN, an is_test_vn VN, a teacher_val VN or a development VN), 300,000 web-novel names (SyosetuNames), generated katakana and Jiten-weighted kanji names; family, given or full name, honorifics, 【】, ？？？, role names (少女, 店員) | names are the weakest real style; synth2 drew names from the 1,200 most frequent kanji |
| fonts | 120 families, 317 files: synth2's 35 plus 85 researched free families (GenEi, Mgen+, Gen Jyuu, IPA, Ume, Koruri, LINE Seed, Kosugi, M+ weights and others), 13 pixel families at 10-20 px (JF Dot / Shinonome / K12 / k8x12 / PixelMplus / Ark / Fusion / Unifont). Families are drawn per category, then a file (medium to bold weights favoured); a line is drawn only by a font that has an outline for every character | Vita and PSP VNs use Rodin/Seurat/Matisse-like faces and bitmap fonts; 25 of the new files map hundreds of code points to empty glyphs, which a cmap check would have rendered as blanks under labels |
| raster | 8-bit FreeType AA, 4-bit (16-level) coverage, glyph atlases (16-48 px cells, 4-bit, bilinear or area downscale, integer origins), 1-bit hinted glyphs, pixel fonts at their exact strike (nearest 2x on the Vita), 2x supersampling, coverage gamma 0.7-1.45 | PSP GE and many Vita engines draw 4-bit atlas glyphs (MAGES: 32/48 px cells shown at 0.75x/0.5x); PSP text is often 1-bit |
| screens | Vita 46%, PSP 480x272 30% (12-20 px), PSP upscaled 2x as on a Vita 13% (bilinear 60%, nearest 30%, 1.995x 5%, scanlines 5%), large renders area-scaled to Vita (ink about 19 px) or PSP (about 11 px) 11% | the ink-height table; Adrenaline and the PSP emulator's scale modes |
| capture | Vita: clean framebuffer 50%, VNDB-like JPEG chain 50% (console Q82-94 4:2:0, then VNDB's Q90, 4:2:0 85% / 4:4:4 15%); PSP: clean 35%, Q90 35%, double 30%, RGB565 ordered dither 45%; scaled: Q90 85% | VNDB re-encodes uploads with libvips at Q90 (its source); 85% of the local Vita screenshots carry standard Q90 4:2:0 tables |
| crops | half the regions: the labelled ink (fill, outline, shadow, attached furigana) plus 4 px (Vita) / 2 px (PSP) like eval's and teacher's boxes, 55% with an extra margin up to 0.12-0.24 x ink height per side; the other half: the console's GRAD finder (`common.find_lines_ex`, the twin of `runtime/vocr_lines.c`) matched to one rendered line | eval crops are ink + 4/2 px, teacher crops are looser, the console reads GRAD boxes |
| polarity | stored as the console hands crops over: dark text inverted by `common.line_dark` | teacher and realdata store real crops the same way |
| layouts | synth2's ADV / band / NVL / balloon / choice / bare, plus VN UI menus and buttons (セーブ, オート, Q.LOAD, タイトルに戻る, 第三章, 4月1日（月）), tighter line pitch (from 1.12 em) so crop edges catch neighbouring lines, more translucent boxes and name plates, ruby 15% (Vita) / 20% (PSP), cursors 33%, typewriter lines 8% (fading glyphs labelled only at alpha 0.5 or more) | real crops include UI and cut neighbours; real edge busyness 21.6 against synth2's 16.5 |
| scenes | crops of the upper 62% of training-split screenshots (synth2's list plus realdata's Vita, PSP and PC shots; version 2: 5,238 screenshots of 1,182 VNs, version 1: 6,260 of 1,185), else synth2's procedural scenes | synth2's scene list holds 22 of teacher_val's 181 VNs and 110 of its screenshots; synth3 drops every development VN |

## Dataset

### Sources and exclusions

- Fonts: `data/manifests/synth3_fonts.json` pins 317 files of 120 families (1.46 GB, never
  committed): synth2's 35 families and 85 found and licence-checked for synth3. 118 families are
  free including commercial use (OFL-1.1, Apache-2.0, IPA, M+, public-domain and author licences);
  2 are free for non-commercial use only (`jfdot_noncommercial`, `zpix`). `--profile open` drops
  those two and every non-commercial or unlicensed text pool.
- Text: `data/manifests/synth3_text_sources.json` pins every source (URL, revision, size, sha256,
  licence and class) and every built pool. 14 pools hold 3.65M training and 0.18M development
  lines. Classes: open (Aozora, Wikipedia, Tatoeba, JESC, Aratako DeepSeek roleplay, Wiktionary,
  Wikiquote), nc (JDD, BSD, NUCC, Aratako gpt-4o-mini roleplay) and grey (OpenSubtitles,
  なりきり posts). The owner uses the recognizer privately and accepts non-commercial sources;
  nothing is redistributed.
- Names: 102,536 VNDB character names (dump of 2026-10-12, ODbL / DbCL) after dropping 24,331
  characters that appear in a reserved VN: the 362 listed ones (eval's 172 test and family VNs,
  teacher_val's 181 VNs and the 9 general-VN / CLANNAD development VNs) and every is_test_vn VN;
  plus 300,000 SyosetuNames (ODC-BY).
- Scenes (version 2): upper 62% of training-split screenshots whose text lines are known
  (realdata's line boxes, linefind's detections), and only patches that touch none of them.
- Vocabulary: `data/vocab.txt` (5,354 classes); lines with any other character are dropped
  whole. FL10's 3,500-class vocabulary reads the rest as `<unk>`, as with synth2.

### Renders

| render | generator | crops | shards | bytes | pin |
|---|---|---|---|---|---|
| version 1 (superseded before any scoring) | version 1 | 1,000,000 | 100 | 5.95 GB | `data/manifests/synth3_shards_20261012.json` |
| **version 2 (the ablations' s3)** | version 2 | 1,000,000 | 100 | 5,517,419,265 B | `data/manifests/synth3_shards_v2.json` |
| version 2 extension (learning curve's 2M point) | version 2, shards 100-199 | 1,000,000 | 100 | 5,520,646,261 B | `data/manifests/synth3_shards_v2_ext.json` |
| version 2 development split (`dev_v2`) | version 2, `--split dev` | 20,000 | 2 | | not pinned; never trained on |

- Version 2: seed 20261012, `--profile personal`, 315 font files under 110 family names (the loader
  skips files covering fewer than 1,500 vocab characters, 300 for pixel fonts), 14 text pools, scenes from 1,182 training-split VNs (947 Vita, 1,317 PSP and
  2,974 PC screenshots). Rendered 14:21-14:44Z in 5 heavy.sh jobs of 5 workers each (about 310 s
  per 10,000-crop shard per worker); 0 render errors.
- Determinism: re-rendering the first 1,000 crops of shards 0 and 57 from the committed code gave
  byte-identical pixels, shapes, labels and metadata. Each shard is seeded by (seed, shard index),
  so the extension's shards 100-199 are new crops from the same generator; its `recipe.json`
  (configuration, code sha256s, fonts, pools, scenes) equals version 2's.
- Version 1 is kept only for the audit below.

### Label audit (teacher agreement)

`synth/synth3_audit.py` reads 1,500 random crops of a set with PP-OCRv6 small in screen polarity
(seed 1) and compares the reading with the label. A disagreement is either a wrong label or a
teacher misread; the worst crops are then read by eye.

| set | teacher vs label CER | exact lines | GRAD crops | gt crops | mincho | pixel fonts | names |
|---|---|---|---|---|---|---|---|
| synth2 (shards 00000-00039) | 3.00% | 73.3% | | | 2.94% | 4.06% | 13.39% |
| synth3 version 1 | 4.52% | 76.7% | 4.46% | 4.55% | 3.60% | 6.67% | 11.09% |
| **synth3 version 2** | **3.56%** | **80.1%** | 2.92% | 3.89% | 1.96% | 4.30% | 8.24% |

- Reading the 36 worst version-1 crops showed two generator faults: a GRAD box that merged real
  text from a scene screenshot with a labelled line (the label then covers part of the crop), and
  gt boxes widened across a neighbouring line whose ink reached into the line's band.
- Version 2 removes both causes: scenes are cut only from screenshots with known text boxes and
  avoid them, ink columns come from each line's own glyph extent, and a GRAD box must hold and hug
  exactly one labelled line (`tests/test_synth3.py` checks the matcher).
- Of the 36 worst version-2 crops, 32 are teacher misreads of legible hard renders (pixel, pen
  and tiny fonts, rare-kanji names). Four cannot be read from their pixels (a washed-out
  pixel-font name, a blurred two-kanji name, a smudged 【雫】 and a name whose last two kanji are
  scattered dots; found by the independent audit at 4x zoom): for those the image does not
  support the label. The rate of such crops across the set was not measured.
- Metadata over 100,000 crops of each version (`meta_check.py`, the same definitions for both):

| fault | version 1 | version 2 |
|---|---|---|
| gt crops whose ink box is much wider than the label (width / (characters x ink height) > 1.6) | 1.86% (1,188 of 63,815) | 0.17% (116 of 67,431) |
| GRAD crops more than 2.2x their ink height | 4.86% (1,760 of 36,185) | 0.61% (199 of 32,569) |

The protocol amendment quotes 2.8% for the first fault in version 1; that figure used an earlier,
looser definition. Both definitions show the same drop.

### Calibration against real training crops

`synth/synth3_stats.py` on 20,000 random crops of each source against 20,000 real training crops
(teacher_train + real_train; never development or test data):

| | real | synth2 | synth3 v1 | synth3 v2 |
|---|---|---|---|---|
| KL(real ‖ source), characters | | 0.143 | 0.078 | 0.076 |
| label length p10 / p50 / p90 | 2 / 8 / 24 | 4 / 15 / 25 | 2 / 8 / 24 | 2 / 8 / 24 |
| name-like crops | 39.9% | 15.9% | 37.0% | 36.0% |
| dark text (stored inverted) | 21.0% | 0% | 24.2% | 23.8% |
| Vita crop height p10 / p50 / p90 | 23 / 28 / 36 | 26 / 32 / 40 | 27 / 34 / 46 | 27 / 34 / 44 |
| PSP crop height p10 / p50 / p90 | 16 / 19 / 25 | 19 / 22 / 26 | 16 / 20 / 27 | 16 / 20 / 25 |
| Vita margin p50, top / bottom / left / right (x ink height) | .19 / .27 / .24 / .30 | | .18 / .18 / .18 / .18 | .17 / .17 / .17 / .17 |
| edge busyness p50 | 21.6 | 16.5 | 14.4 | 13.9 |
| Latin share of characters | 0.57% | 0.60% | 1.94% | 1.92% |
| ！ / ？ per 1,000 characters | 6.8 / 6.8 | 1.7 / 3.2 | 6.1 / 8.9 | 6.2 / 9.1 |
| 」 per 1,000 characters | 16.1 | 12.1 | 9.6 | 9.3 |

synth3 is close to the real text distribution in label lengths, name share and characters (KL
halved), and matches the PSP crop heights and the share of dark text. It does not match all
punctuation (？ 9.1 against 6.8 per 1,000 characters, 」 9.3 against 16.1, ～ 0.58 against 1.32)
or the PSP margins (p50 top / bottom / left / right .19 / .20 / .20 / .20 against real
.14 / .21 / .17 / .25). Gaps that a version 3 targets (separate work, not in these ablations):
- Vita crops are too tall and too tight; real crops have more margin, mostly below and to the
  right.
- Backgrounds are too clean (edge busyness 13.9 against 21.6).
- Latin is 3x too frequent, and there are too few closing 」.

## Study 1: the frozen ablation

The protocol (`eval/synth3_ablation_protocol.json`) was committed before any run, and amended
once before any scoring to point at the version-2 shards. It trains FL10-sized models (width 1.0,
FL10's 3,500-entry vocabulary) from scratch for 16,000 steps with train2's pretraining recipe, on
the NVIDIA one-line-per-page set (0.5) plus synth2 (A_s2), synth3 (B_s3) or both at 0.25 each
(C_mix), seeds 1 and 2. The step-16000 EMA checkpoints are exported (per-channel int8) and read
through the C runtime (W8I) on teacher_val in both polarities (`rv` as stored, `rvn` in screen
polarity), errors summed over seeds and views; VN-cluster paired bootstraps with 10,000
resamples (`eval/synth3_ablation.py`). The decision rule was fixed in advance. All six runs trained
together, 14:54-16:21Z, on a 32-core EPYC 9R14 box (5 threads each); result
`eval/synth3_ablation_20261012.json`. The tables below are rounded once from the stored per-line
errors (`eval/reproducibility/synth3-20261012/doc_tables.py`); the result JSON rounds deltas to
3 decimals first, so a last digit can differ.

| run | rv | rvn |
|---|---|---|
| A_s2_seed1 | 5.88% | 5.81% |
| A_s2_seed2 | 5.56% | 5.50% |
| B_s3_seed1 | 5.67% | 5.62% |
| B_s3_seed2 | 5.72% | 5.67% |
| C_mix_seed1 | 5.09% | 5.12% |
| C_mix_seed2 | 5.26% | 5.25% |

| comparison | CER X → Y | Y − X, points (VN-cluster 95% CI) | line CI | seed 1 | seed 2 | VNs better / worse / equal | decision |
|---|---|---|---|---|---|---|---|
| B_s3 vs A_s2 | 5.69% → 5.67% | −0.01 (−0.34 to +0.30) | −0.25 to +0.22 | −0.20 | +0.17 | 69 / 70 / 42 | no decision |
| C_mix vs A_s2 | 5.69% → 5.18% | **−0.51 (−0.76 to −0.26)** | −0.72 to −0.30 | −0.74 | −0.27 | 96 / 40 / 45 | **C better** |
| C_mix vs B_s3 | 5.67% → 5.18% | **−0.49 (−0.74 to −0.25)** | −0.70 to −0.29 | −0.54 | −0.44 | 88 / 47 / 46 | **C better** |

C beats both A and B, so under the protocol the mix is the recommended VN-style synthetic data.

Strata (reported, not decided; teacher_val's own metadata; VN-cluster intervals; CER is errors
over both views and both seeds; 54 intervals without multiplicity control):

| stratum | lines | A_s2 | B_s3 | C_mix | B − A | C − A | C − B |
|---|---|---|---|---|---|---|---|
| target Vita | 972 | 4.70% | 4.69% | 4.12% | −0.00 (−0.37 to +0.35) | −0.57 (−0.89 to −0.28) | −0.57 (−0.88 to −0.27) |
| target PSP | 679 | 7.11% | 7.09% | 6.71% | −0.03 (−0.53 to +0.43) | −0.41 (−0.85 to +0.00) | −0.38 (−0.76 to −0.01) |
| native Vita | 161 | 2.95% | 2.86% | 2.61% | −0.09 (−1.48 to +0.77) | −0.34 (−1.46 to +0.39) | −0.25 (−0.57 to +0.17) |
| native PSP | 426 | 3.26% | 3.46% | 3.27% | +0.20 (−0.30 to +0.72) | +0.01 (−0.42 to +0.46) | −0.19 (−0.54 to +0.17) |
| native Vita + PSP (not a protocol stratum) | 587 | | | | +0.11 (−0.39 to +0.58) | −0.09 (−0.53 to +0.31) | −0.20 (−0.47 to +0.08) |
| PC/console shot scaled | 1,064 | 6.88% | 6.80% | 6.17% | −0.08 (−0.50 to +0.33) | −0.70 (−1.02 to −0.39) | −0.63 (−0.95 to −0.30) |
| bright on dark | 1,050 | 4.49% | 4.53% | 4.03% | +0.03 (−0.29 to +0.33) | −0.47 (−0.78 to −0.18) | −0.50 (−0.76 to −0.26) |
| dark on light | 601 | 7.63% | 7.54% | 7.07% | −0.09 (−0.70 to +0.51) | −0.57 (−0.99 to −0.16) | −0.48 (−0.92 to −0.01) |
| ink under 14 px | 484 | 12.62% | 12.46% | 11.67% | −0.17 (−0.98 to +0.57) | −0.95 (−1.63 to −0.34) | −0.78 (−1.47 to −0.12) |
| ink 14-19 px | 760 | 3.61% | 3.61% | 3.30% | −0.00 (−0.38 to +0.36) | −0.31 (−0.63 to −0.00) | −0.31 (−0.59 to −0.03) |
| ink 20-27 px | 363 | 3.08% | 3.22% | 2.70% | +0.13 (−0.32 to +0.56) | −0.38 (−0.76 to −0.05) | −0.52 (−0.88 to −0.16) |
| ink 28 px and taller | 44 | 3.85% | 3.50% | 2.78% | −0.36 (−2.37 to +1.41) | −1.07 (−3.82 to +1.13) | −0.71 (−1.92 to +0.85) |
| 1-4 characters | 608 | 16.61% | 15.39% | 15.13% | −1.22 (−2.79 to +0.19) | −1.47 (−2.73 to −0.28) | −0.26 (−1.29 to +0.77) |
| 5-12 characters | 453 | 6.22% | 6.05% | 5.89% | −0.17 (−0.67 to +0.33) | −0.33 (−0.83 to +0.16) | −0.16 (−0.71 to +0.37) |
| 13+ characters | 590 | 3.89% | 4.10% | 3.47% | +0.21 (−0.13 to +0.53) | −0.41 (−0.66 to −0.18) | −0.63 (−0.89 to −0.37) |
| name-like | 710 | 15.74% | 14.65% | 14.08% | −1.09 (−2.32 to −0.01) | −1.66 (−2.76 to −0.61) | −0.57 (−1.46 to +0.35) |
| dakuten / small kana | 831 | 4.41% | 4.43% | 3.93% | +0.02 (−0.33 to +0.34) | −0.48 (−0.74 to −0.24) | −0.50 (−0.75 to −0.25) |
| rare kanji | 116 | 7.28% | 6.95% | 6.87% | −0.33 (−1.60 to +0.78) | −0.41 (−1.64 to +0.70) | −0.08 (−0.87 to +0.71) |
| Latin / digits | 62 | 10.09% | 11.06% | 10.12% | +0.97 (−0.90 to +2.08) | +0.03 (−2.14 to +1.41) | −0.94 (−2.70 to +0.50) |

What this says:
- The mix's gain is measured on the 64% of teacher_val that is PC/console screenshots scaled to
  console size (C − A −0.70, CI −1.02 to −0.39). On the native Vita and PSP captures no arm
  separates from another: C − A −0.09 (CI −0.53 to +0.31) on their 587 lines together. The
  native development cohort (devnative) is what can test the gain on console text.
- Across teacher_val's strata (64% of its lines are scaled shots) the mix is lower in both
  polarities, at every ink height up to 27 px, on 1-4 and 13+ character lines, names and
  dakuten/small kana; 5-12 characters is not resolved (−0.33, CI −0.83 to +0.16). One
  explanation, not tested: the diversity of two generators with different fonts, layouts and
  text.
- synth3 alone shows no measured difference from synth2 overall. Of its strata, only name-like
  excludes 0, and only just (−1.09, CI −2.32 to −0.01); 1-4 characters is −1.22 (CI −2.79 to
  +0.19); long lines (+0.21) and Latin (+0.97) are not resolved either way.
- 40 of 181 VNs are worse under C than A; the largest per-VN regressions are on VNs with 8-189
  characters (no per-VN veto; listed in the result JSON).

**Overlap with synth2's scenes.** synth2's background list (which arms A and C use, and B does
not) holds 110 of teacher_val's own screenshots (253 of its lines) and 22 of its VNs (311 lines).
The protocol's note that teacher_val shares no screenshot with any training pool is therefore
wrong for synth2's scenes, which crop the upper 62% of those screenshots (the protocol file is left
unchanged because the result pins its sha256). The overlap favours the arms with more synth2, so
it works against the reported gain. Without the 22 VNs (the audit's sensitivity check, protocol
statistics): C − A −0.56 (CI −0.85 to −0.29), C − B −0.53 (CI −0.82 to −0.24), B − A −0.03 (CI
−0.41 to +0.33); the decision stands. Study 2 inherits the overlap: both its arms use synth2.

### Study 1 on the protected test sets (reported only)

After the decision and result were committed (6803fd9), every primary export was scored once on
vita-test (206 lines, 16 games) and psp-test (245 lines, 14 games; 28 distinct VNs in the pooled
cluster bootstrap) with `eval/run.py --model vocr:BLOB` (W8I), in console and native polarity,
and summarised by `eval/synth3_test_report.py` (`eval/synth3_ablation_test_20261012.json`). The
released FL10 (`FL10_w8.vocr`, v0.2.0) was scored the same way as a reference: it reproduces its
published 2.30% / 5.18% in native polarity. These numbers cannot change the decision.

| arm (2 seeds) | console: vita-test | console: psp-test | console: pooled | native: vita-test | native: psp-test | native: pooled |
|---|---|---|---|---|---|---|
| A_s2 | 3.32% | 6.18% | 4.73% | 3.35% | 6.22% | 4.77% |
| B_s3 | 2.94% | 6.31% | 4.61% | 2.89% | 6.10% | 4.47% |
| C_mix | 2.96% | 6.22% | 4.57% | 3.02% | 6.31% | 4.65% |
| reference: released FL10 (32k pretrain + fine-tune) | 2.38% | 5.30% | 3.82% | 2.30% | 5.18% | 3.72% |

| difference, points (VN-cluster CI) | vita-test | psp-test | pooled |
|---|---|---|---|
| console, B − A | −0.37 (−1.19 to +0.29) | +0.14 (−0.27 to +0.57) | −0.12 (−0.57 to +0.28) |
| console, C − A | −0.35 (−1.22 to +0.33) | +0.04 (−0.39 to +0.51) | −0.16 (−0.62 to +0.26) |
| console, C − B | +0.02 (−0.36 to +0.35) | −0.10 (−0.68 to +0.53) | −0.04 (−0.37 to +0.33) |
| native, B − A | −0.47 (−1.28 to +0.21) | −0.12 (−0.79 to +0.51) | −0.30 (−0.82 to +0.17) |
| native, C − A | −0.33 (−1.24 to +0.38) | +0.10 (−0.42 to +0.62) | −0.12 (−0.62 to +0.33) |
| native, C − B | +0.14 (−0.21 to +0.43) | +0.22 (−0.39 to +0.86) | +0.18 (−0.15 to +0.52) |

No difference is resolved: every interval includes 0 (451 lines from 28 VNs give pooled intervals
of about ±0.45 points). In native polarity synth3 alone (B) has the lowest pooled CER and C − B
has the opposite sign to teacher_val. The 16k-step pretrain-only models are 0.75-0.91 points
behind the released FL10 in console polarity and 0.75-1.04 in native polarity; the released FL10
had twice the pretraining and a real fine-tune, which study 2 adds.

### Study 1's secondaries (exploratory)

The protocol's secondary runs, scored like the primaries; one seed each, so an interval covers the
sampling of development lines, not training noise. The comparisons were fixed in
`eval/synth3_ablation_secondary.json` before any secondary was exported or scored (the
fine-tunes' own in-training evaluations of their final checkpoints were already logged); result
`eval/synth3_ablation_secondary_20261012.json`.
- **Fine-tunes:** from each arm's seed-1 pretrain, 5,000 steps on nv 0.25 + the arm's synthetic
  data 0.25 + teacher_train 0.35 + real_train 0.15 (unfiltered pools: they hold CLANNAD and its
  family, see study 2's amendment 2).
- **H15:** arms A and B at width 1.5 with `data/vocab.txt`, study 1's recipe, seed 1.

| teacher_val | CER X → Y | Y − X (VN-cluster CI) | native Vita + PSP lines (587) |
|---|---|---|---|
| ft B − ft A | 4.66% → 4.52% | −0.14 (−0.44 to +0.16) | +0.26 (−0.15 to +0.72) |
| ft C − ft A | 4.66% → 4.34% | −0.32 (−0.57 to −0.08) | −0.08 (−0.44 to +0.26) |
| ft C − ft B | 4.52% → 4.34% | −0.18 (−0.50 to +0.15) | −0.35 (−0.86 to +0.10) |
| fine-tuning gain, A | 5.84% → 4.66% | −1.19 (−1.65 to −0.78) | −0.67 (−1.20 to −0.22) |
| fine-tuning gain, B | 5.64% → 4.52% | −1.13 (−1.60 to −0.72) | −0.35 (−0.76 to +0.09) |
| fine-tuning gain, C | 5.11% → 4.34% | −0.77 (−1.21 to −0.39) | −0.63 (−1.07 to −0.20) |
| C − A, seed 1, before fine-tuning | 5.84% → 5.11% | −0.74 (−1.05 to −0.44) | −0.13 (−0.66 to +0.38) |
| **H15: B − A** | 4.52% → 4.00% | **−0.52 (−0.80 to −0.25)** | **−0.38 (−0.70 to −0.10)** |

- The fine-tune keeps the order (ft C < ft B < ft A) but the seed-1 C − A gap falls from −0.74
  to −0.32 (43% of it): real data gains more for A and B than for C. This is study 2's question,
  in one seed.
- At H15 width, in one run per arm, synth3 alone is lower than synth2, on the native lines too,
  with names −2.20 (CI −3.38 to −1.08), Latin/digits −2.76 (CI −4.78 to −1.35) and dark-on-light
  −0.88 (CI −1.40 to −0.41). The intervals cover line sampling only: at FL10 width the same
  comparison was −0.20 in seed 1 and +0.17 in seed 2. Not replicated; the mix was not run at H15
  width in study 1.

On the protected test sets (each once, after the secondary result was committed;
`eval/synth3_ablation_secondary_test_20261012.json`; console polarity, pooled vita-test +
psp-test): ft A 4.57%, ft B 4.73%, ft C 4.14%; H15 A 3.29%, H15 B
3.05%. Differences: ft C − ft A −0.43 (CI −1.01 to +0.09), ft C − ft B −0.59 (CI −1.20 to −0.03),
ft B − ft A +0.16 (CI −0.35 to +0.66), H15 B − H15 A −0.24 (CI −0.98 to +0.39). Native polarity:
ft A 4.73%, ft B 4.63%, ft C 4.30%; H15 A 3.43%, H15 B 3.07%.

## Study 2: the full release recipe

The protocol (`eval/synth3_study2_protocol.json`) was committed before any study-2 run trained.
- Recipe: train2's release recipe with hard labels: pretrain from scratch on nvsynth's
  every-line set (rebuilt on this machine, 0.5) plus synth2 (A_s2, 0.5) or the mix (C_mix,
  0.25 + 0.25), FL10 for 32,000 steps and H15 for 40,000; then 15,000 fine-tune steps on nv 0.25
  + the arm's synthetic data 0.25 + teacher_train 0.275 + real_train 0.125 + PP-OCRv6
  pseudo-labels 0.1 (kd=0). The fine-tune's final EMA checkpoint is scored; no selection.
- Arms and seeds: A against C, FL10 seeds 1-3 and H15 seeds 1-2, scored on teacher_val with
  study 1's statistics and decision rule per architecture.
- devnative is co-primary: its cohort was committed (COHORT_FROZEN f5a1dbd, 19:14:50Z) before
  the first study-2 score (23:12Z). Its decision set is the strict and tv_native subsets, ink and
  GRAD crops pooled (1,174 crops, 48 VNs, both views). Both architectures use that version. Later
  versions (v2 drops 41 screenshots that are also in eval/linefind's finder sets; v3 fixes ink
  crops that v1 padded twice) are reported, not decided on.
- Amendment 1 (before any score): how devnative's subset files are pooled.
- Amendment 2 (before any fine-tune step): the four fine-tune pools are copies without CLANNAD
  (v4) and its VNDB family (45 of 42,144 rows: v4, v4060, v12), made by
  `eval/synth3_filter_pools.py` and pinned in `eval/synth3_study2_pools.sha256`.
- Reported only: pretrain-only scores, the fine-tuning gain, a pretrain-length curve (FL10 C_mix
  at 64k and 128k pretrain steps) and a synth3 learning curve (study 1's B_s3 at 0.25M, 0.5M,
  1M and 2M crops).

### FL10: no measured difference

`eval/synth3_study2_FL10_20261012.json` (`eval/synth3_study2.py result`, C W8I, both views):

| | teacher_val CER | devnative CER |
|---|---|---|
| A_s2 (3 seeds) | 3.41% | 3.99% |
| C_mix (3 seeds) | 3.33% | 3.90% |
| C − A, points (VN-cluster 95% CI) | −0.07 (−0.24 to +0.09) | −0.09 (−0.29 to +0.12) |
| per seed (1, 2, 3) | −0.20, +0.09, −0.11 | +0.07, −0.16, −0.17 |

Neither interval excludes 0, so under the frozen rule neither arm is better and synth2 (the
released recipe) stays at FL10 size. Reported only:
- **The real fine-tune absorbs most of the pretrain gap.** Pretrain-only (32k steps), the mix is
  still ahead as in study 1: C − A −0.46 (CI −0.64 to −0.29) on teacher_val. The fine-tune gains
  −1.06 (CI −1.45 to −0.72) for A and −0.67 (CI −1.06 to −0.35) for C. On devnative it gains
  1.30 points for A and 0.80 for C.
- **Strata (teacher_val):** only 1-4 characters is resolved, −0.91 (CI −1.66 to −0.19); name-like
  −0.56 (CI −1.27 to +0.13); native Vita −0.20 and native PSP −0.11, neither resolved.
- **Both arms are far below the released FL10** on teacher_val (3.4% against 4.05%), with the same
  topology and vocabulary. Possible causes, not separated: hard labels instead of frame KD (train2
  found KD hurt H15 by 0.53), the fine-tune starting from the final pretrain rather than the best
  teacher_val checkpoint, and nvsynth rebuilt here.

Protected test sets (`eval/synth3_study2_FL10_test_20261012.json`; each export scored once, after
the result was committed; reported only):

| pooled vita-test + psp-test | console polarity | native polarity |
|---|---|---|
| A_s2 fine-tuned (3 seeds) | 3.40% | 3.38% |
| C_mix fine-tuned (3 seeds) | 3.13% | 3.18% |
| C − A, points (VN-cluster CI) | −0.27 (−0.60 to +0.01) | −0.20 (−0.54 to +0.07) |
| A_s2 pretrain-only | 3.74% | 3.77% |
| C_mix pretrain-only | 3.03% | 3.06% |
| pretrain-only C − A | −0.72 (−1.10 to −0.42) | −0.72 (−1.10 to −0.41) |
| released FL10 (v0.2.0) | 3.82% | 3.72% |

On the test sets the fine-tune changes A by −0.34 (console, CI −0.72 to +0.01) and C by +0.10
(CI −0.12 to +0.34), neither resolved, and the pretrain-only gap of the mix stays resolved. These
numbers cannot change the decision or the recipe; the development cohorts, where the fine-tune
helps both arms, remain the selection evidence.

### H15: no measured difference either

`eval/synth3_study2_H15_20261013.json` (`eval/synth3_study2.py result`, computed 2026-10-13T07:57Z
when the last H15 fine-tune had finished; C W8I, both views; devnative v1 as for FL10):

| | teacher_val CER | devnative CER |
|---|---|---|
| A_s2 (2 seeds) | 2.37% | 2.50% |
| C_mix (2 seeds) | 2.23% | 2.39% |
| C − A, points (VN-cluster 95% CI) | −0.14 (−0.30 to +0.02) | −0.11 (−0.30 to +0.08) |
| per seed (1, 2) | −0.14, −0.14 | −0.14, −0.07 |

Both seed pairs and both cohorts favour the mix, but neither interval excludes 0 (teacher_val's upper
bound is +0.02), so under the frozen rule neither arm is better and synth2 stays at H15 size as well.
Reported only:
- **Pretrain-only, the mix is ahead by less than at FL10 size:** C − A −0.27 (CI −0.44 to −0.11) on
  teacher_val (FL10: −0.46; study 1, with the one-line nvsynth set: −0.51). The fine-tune gains −0.79
  (CI −1.17 to −0.48) for A and −0.66 (CI −1.05 to −0.34) for C on teacher_val, and 0.90 and 0.77
  points on devnative.
- **Strata (teacher_val):** 1-4 characters −0.91 (CI −1.57 to −0.24) and name-like lines −0.73
  (CI −1.34 to −0.13) are resolved in the mix's favour, as the short-line stratum was at FL10 size;
  native Vita −0.27 (CI −0.80 to +0.12) and native PSP −0.10 (CI −0.41 to +0.20) are not; rare kanji
  +0.31 (CI −0.47 to +1.06).
- **Strata (devnative):** the native Vita crops (305 of 1,174) favour the mix, −0.36 (CI −0.57 to
  −0.19), the PSP crops do not, −0.01 (CI −0.25 to +0.23). A stratum of a reported-only cohort cannot
  decide anything, but it is the first sign in these studies that the synth3 mix helps on native Vita
  frames.
- **Level with the released H15.** teacher_val 2.37% (A) and 2.23% (C) against the released H15's
  2.29% (`docs/train2.md`, train2's own measurement), which was trained with the same hard-label recipe
  and synth2. At FL10 size the study-2 runs were far below the released FL10 (3.4% against 4.05%),
  which was trained with frame KD; H15 was not, so the two sizes agree: the gap at FL10 was the
  recipe, not the synthetic data.

Protected test sets (`eval/synth3_study2_H15_test_20261013.json`; each export scored once, after
the result was committed; reported only):

| pooled vita-test + psp-test | console polarity | native polarity |
|---|---|---|
| A_s2 fine-tuned (2 seeds) | 2.10% | 2.11% |
| C_mix fine-tuned (2 seeds) | 2.08% | 2.09% |
| C − A, points (VN-cluster CI) | −0.02 (−0.29 to +0.30) | −0.02 (−0.27 to +0.29) |
| A_s2 pretrain-only | 2.40% | 2.42% |
| C_mix pretrain-only | 2.00% | 1.98% |
| pretrain-only C − A | −0.40 (−0.72 to −0.10) | −0.44 (−0.74 to −0.16) |
| released H15 (v0.2.0) | 2.21% | 2.09% |

Per set (console): vita-test A 1.27%, C 1.11% (C − A −0.16, CI −0.44 to +0.13), released H15 1.56%;
psp-test A 2.95%, C 3.07% (+0.12, CI −0.35 to +0.70), released H15 2.87%. The fine-tune changes A by
−0.30 (console, CI −0.52 to −0.07) and C by +0.08 (CI −0.17 to +0.38) on the test sets, so the
pretrain-only mix is the lowest pooled number here (2.00%), as at FL10 size (3.03%); on the
development cohorts the fine-tune helps both arms. These numbers cannot change the decision or the
recipe.

### Pretrain length: longer pretraining pays at FL10 size (reported only, one seed)

`eval/synth3_study2_curves_20261013.json` (the reported-only comparisons of
`eval/synth3_study2_reported.json`, computed by `eval/synth3_study2.py result`; teacher_val, C W8I, both
views). FL10 C_mix seed 1 pretrained for 32,000 (study 2's run), 64,000 and 128,000 steps, each followed
by the same 15,000-step fine-tune:

| pretrain steps | pretrain-only CER | fine-tuned CER | fine-tuned − 32k, points (VN-cluster 95% CI) |
|---|---|---|---|
| 32,000 | 4.04% | 3.31% | |
| 64,000 | 3.42% | 2.81% | −0.50 (−0.75 to −0.26) |
| 128,000 | 3.18% | 2.60% | −0.70 (−1.03 to −0.40); vs 64k −0.21 (−0.45 to +0.03) |

The gain is broad: every stratum but the two tallest ink bands improves, most on ink under 14 px
(−1.78, CI −2.82 to −0.89), name-like lines (−1.37, CI −2.47 to −0.32), PSP targets (−1.06, CI −1.67 to
−0.56) and dark-on-light text (−1.05, CI −1.64 to −0.53). One seed, so the size of the gain is
uncertain, but its sign is not: at FL10 size, four times the pretraining (12.3M lines seen instead of
3.1M, from the same 13M available) moves teacher_val by more than any synthetic-data arm did. The
128k fine-tune, 2.60%, is close to the H15 arms (2.23-2.37%) at 57% of their weight bytes.

### Learning curve: more synth3 crops do not help this recipe (reported only)

Study 1's B_s3 recipe (16,000 steps, nvsynth one-line set 0.5 + synth3 v2 0.5, FL10, seeds 1 and 2),
with the synth3 source restricted or extended. The 1M point is study 1's runs on the 32-core box; the
other points trained here.

| synth3 crops | seed 1 / seed 2 CER | arm CER | vs the next smaller point (VN-cluster 95% CI) |
|---|---|---|---|
| 0.25M (shards 0-24) | 5.55% / 5.47% | 5.51% | |
| 0.5M (shards 0-49) | 5.28% / 5.65% | 5.46% | −0.05 (−0.24 to +0.14) |
| 1M (study 1's B_s3) | 5.64% / 5.70% | 5.67% | +0.21 (+0.01 to +0.41) |
| 2M (shards 0-99 + the pinned extension 100-199) | 5.48% / 5.33% | 5.41% | −0.26 (−0.52 to −0.03) |

From 0.25M to 2M crops the arm CER moves by −0.10 points, with the two resolved steps pointing in
opposite directions, so this recipe is saturated: 16,000 steps × 96 × 0.5 see 768,000 synth3 samples,
about three passes over 0.25M crops and well under one pass over 2M. Rendering more synth3 is not worth
it for this recipe; the pretrain-length curve above says where the remaining headroom is (seeing more
samples, not having more distinct ones). The 1M point is the only one from the other machine, which
confounds its +0.21 step.

### Provenance of the runs

- The ten FL10/H15 runs were launched from the conductor's clone at commits 0617f72-ebb4dfb
  (pre-rebase history, tag `synth3-history-20261012`); their logs record that commit at pretrain
  start. Amendment 2's files were then in that clone's working tree, byte-identical to the
  committed ones, and every fine-tune's `args.json` records the CLANNAD-free pools.
- The fine-tune command of the runs already pretraining was changed by rewriting only the part
  of `eval/synth3_study2_run.sh` after the pretrain block, in place: bash reads a running script
  from its saved offset, so each run read the new fine-tune block (checked on a test script and
  on all six FL10 fine-tunes' `args.json`; the H15 fine-tunes are checked when they start).
- The learning-curve runs (`eval/synth3_curve_run.sh`) and the realplus fine-tunes ran from the
  detached worktree `runner-578821d` (tag `vocr3-runner-578821d`, feat/vocr3's amendment-5 commit)
  through `study2_queue6.sh`; the step-curve runs are study-2 runs (`eval/synth3_study2_run.sh FL10
  C_mix 1 64000|128000`).
- Runs launched after 21:22Z (H15 C_mix seed 2 excepted) run from a detached worktree at
  c464aa8 (tag `vocr3-runner-c464aa8`), whose study-2 scripts are byte-identical to the
  committed ones; the scoring driver is `eval/reproducibility/synth3-20261012/study2_score_when_ready.sh`.

## Limits

- teacher_val is the only development cohort on this machine so far (the six-title general-VN
  cohort and the CLANNAD captures are private and not here). Two thirds of its lines are
  PC/console screenshots scaled to console size, and study 1's gain is measured there: on native
  Vita and PSP captures no arm separates (see Study 1).
- teacher_val shares 110 screenshots (253 lines) with synth2's scene list (see Study 1); the
  overlap favours synth2 arms.
- Study 1 is a short recipe (16,000 steps, no real fine-tune). Whether the gain survives the
  release recipe is study 2's question; in one seed the secondary fine-tunes leave less than
  half of it.
- Two seeds per arm in study 1; seed-to-seed spread within an arm was up to 0.32 points.
- Hard-example mining used machine-consensus references on training-side pools, part of which
  was a pseudo-label source in train2's fine-tune; 26 of its 3,339 lines are from six general-VN
  development VNs (see the mining section).
- The learning curve's 1M point reuses study 1's runs from the 32-core box; cross-machine
  equivalence rests on identical logged losses and draws for the first 700 steps of a replay.
- Native game fonts were not used: the private CLANNAD font files are not on this machine. The
  dataset uses free and non-commercial fonts only (`--profile open` drops the non-commercial
  ones; not separately evaluated).
- The rendering parameters for engines and captures come partly from documentation and source
  code (VNDB, PPSSPP, impacto) and partly from estimates; the measured ones are the VNDB JPEG
  tables (768 of 789 local Vita screenshots carry the standard Q90 tables, 667 are 4:2:0 and 122
  4:4:4) and the real crop geometry above.
- Nothing here has run on a console.
- **CLANNAD in the inherited real pools.** teacher_train and real_train hold 8 crops of CLANNAD
  (v4) and 23 of its family (v4060, v12); the pseudo-label pools hold 14 family crops. train2's
  fine-tunes trained on them, so the released v0.2.0 models and study 1's secondary fine-tunes
  did too; study 2 fine-tunes on copies without them (amendment 2). synth3 read them only as
  calibration and mining data, at a negligible scale (second audit, recounted from the pinned
  inputs): its focus-character targets count every label character of teacher_train + real_train,
  363 of whose 235,997 characters come from those rows (v4 116, v4060 213, v12 34); its length
  target and calibration tables sample 20,000 crops of the same pools (how many CLANNAD rows the
  sample drew is not recorded); and 4 of the 3,339 mined lines are family lines (v12 2, v4060 2;
  none from v4). Its scenes include 5 family screenshots (four of v4060, one of v12; neither v4
  screenshot). Study 1's and study 2's hygiene notes ("synth3 ... never read ... CLANNAD") hold
  for v4's names and scenes but not for this calibration, the family's scenes or the family's
  names below; the protocol files stay unchanged because results pin their sha256. synth3's own exclusion list
  holds v4 but not its family, and VNDB lists CLANNAD's main cast again as characters of Tomoyo
  After (v12) and Hikari Mimamoru Sakamichi de (v4060): 19 name parts of those 22 entries (古河,
  渚, 岡崎, 坂上, 智代, 春原, 陽平, 伊吹, 風子, 藤林, 杏, 椋, 秋生, 早苗, ...) are VNDB rows of
  `names.train.tsv`, and 7 version-2 crops hold a full cast name (春原陽平 2, 坂上智代 2,
  伊吹風子, 岡崎汐, 藤林椋); single parts occur in more crops but are mostly common words or
  names (`clannad_names.py.snapshot` in `eval/reproducibility/synth3-20261012/`). realdata's
  shots, a scene source, hold 6 screenshots of v12 and v4060. So any synth3-trained model is
  exposed to CLANNAD's character names, which matters for the CLANNAD long run; the released
  models are exposed through the real crops above. Later renders should exclude CLANNAD's
  family from names, scenes, calibration and mining.
- **Fresh native textbox test exposure (names only).** synth3's name pool was built before that
  test's six titles were published (PR #49) and did not exclude them (PSP v14909, v25459,
  v29454; Vita v21289, v21976, v20519; reserves v16509, v19140). About 70 VNDB names occur only in
  those titles' entries and are in `names.train.tsv` (39 single names and 31 two-part names). 340
  of the 1,000,000 version-2 crops (0.034%) have a label containing one of the 39 single names,
  and 13 a full two-part name; most single-name hits are common words or substrings (ラゴン inside
  ドラゴン, 常盤/常磐, 駐在), the distinctive ones (トリリオン, フェゴール, ペルペル, ...) are a few
  dozen crops. No screenshot, scene or real crop of those titles is in any synth3 input. Any model
  trained on synth3 v2 carries this exposure into that test. Later renders should add the eight
  VNs and their relations to the name exclusion (`synth3_text.py names --dev-vns`).
- **Test speaker names in the name pool.** 33 of the 94 distinct speaker names of vita-test and
  psp-test are in `names.train.tsv`, through characters of unreserved VNs or SyosetuNames
  (for example 宮ノ杜 勇 of test VN v7634, through a character of v21784, a 'set' relation that
  eval's family rule does not reserve); 856 version-2 crops carry one as their whole label, 545
  of them 少女. This is within eval's rule.

## Reproduce

```sh
A=$HERD/shared/artifacts; S3=$A/synth3; P=$S3/text/pools3
# fonts: pinned by data/manifests/synth3_fonts.json (synth/synth3_fonts.py verify)
# text: raw sources as pinned in data/manifests/synth3_text_sources.json, then
synth/synth3_text.py pools --raw $S3/text/raw --aozora $HERD/shared/data/corpus/aozora/aozorabunko-dedupe-clean.jsonl.gz \
    --wiki $HERD/shared/data/corpus/wiki.txt --out $P --workers 5
synth/synth3_text.py dev-vns --dev $A/teacher/teacher_val.npz --out $S3/mining/exclude_dev_vns.txt
synth/synth3_text.py names --vndb-chars $S3/text/vndb_chars.tsv --syosetu $S3/text/raw/syosetu_names/syosetu-names.parquet \
    --dev-vns $S3/mining/exclude_dev_vns.txt --out $P
synth/synth3_text.py index --pools $P --split train && synth/synth3_text.py index --pools $P --split dev
synth/synth3_mine.py run --pool $A/teacher/teacher_distill_train.npz --pool $A/realdata/real_unlabelled.npz \
    --targets $A/train2/targets --blob FL10=$A/train2/release/FL10_w8.vocr --blob H15=$A/train2/release/H15_w8.vocr \
    --conf 0.9 --forbid-vns $(cat $S3/mining/dev_vns.txt) --out $S3/mining/mining_v6s_consensus.json
#   (as run: dev_vns.txt is teacher_val's 181 VNs; exclude_dev_vns.txt, written later, adds the 9 development VNs)
synth/synth3_text.py targets --pools $P --real $A/teacher/teacher_train.npz+$A/realdata/real_train.npz \
    --jiten-vn $S3/text/jiten/frequency_list_visual_novel.csv --mining $S3/mining/mining_v6s_consensus.json
# render (two heavy.sh slots of 5 workers each), then pin
$HERD/heavy.sh synth/synth3.py shards --out $S3/shards/train_v2 --split train --shards 50 --first 0 --per-shard 10000 --workers 5
$HERD/heavy.sh synth/synth3.py shards --out $S3/shards/train_v2 --split train --shards 50 --first 50 --per-shard 10000 --workers 5
synth/synth3.py manifest --dir $S3/shards/train_v2 --out data/manifests/synth3_shards_v2.json
synth/synth3_stats.py measure --name real=$A/teacher/teacher_train.npz+$A/realdata/real_train.npz \
    --name s2=$A/synth2/shards/synth_0000[0-1].npz --name s3v1="$S3/shards/train/synth3_0000[0-1].npz" \
    --name s3="$S3/shards/train_v2/synth3_0000[0-1].npz" --out stats.json
# the ablation: eval/synth3_run_arm.sh ARM SEED (through heavy.sh), then
eval/synth3_ablation.py export --ckpt $S3/ablation/runs/RUN/ckpt_step16000.pt --out RUN_w8.vocr
eval/synth3_ablation.py score --blob RUN=RUN_w8.vocr --dev $A/teacher/teacher_val.npz --out S/RUN.json   # one file per run
eval/synth3_ablation.py compare --scores S/{A_s2,B_s3,C_mix}_seed{1,2}.json --protocol eval/synth3_ablation_protocol.json \
    --out result.json
# the exact scripts that ran, and the doc's tables: eval/reproducibility/synth3-20261012/
# the protected-test report, once the decision is committed (each export once per polarity)
VOCR_EVAL_DATA=$A/eval eval/run.py --model vocr:RUN_w8.vocr --set vita-test,psp-test --polarity console --out T/RUN_console.json
VOCR_EVAL_DATA=$A/eval eval/run.py --model vocr:RUN_w8.vocr --set vita-test,psp-test --polarity native --out T/RUN_native.json
eval/synth3_test_report.py --protocol eval/synth3_ablation_protocol.json --dir T --ref REF --out test_report.json
# study 2: nvsynth's every-line set (docs/nvsynth.md), the v2 extension, then the runs (HEAVY_SLOTS=10)
$HERD/heavy.sh nvsynth/nv_fetch.py run --split train --per-page 0 --files 0-20 --out $A/nvsynth/full/train --max-raw-gb 50
$HERD/heavy.sh nvsynth/nv_fetch.py run --split train --per-page 0 --files 21-41 --out $A/nvsynth/full/train --max-raw-gb 50
$HERD/heavy.sh synth/synth3.py shards --out $S3/shards/train_v2_ext --split train --shards 25 --first 100 --per-shard 10000 --workers 5
#   (and --first 125, 150, 175), then synth/synth3.py manifest --dir $S3/shards/train_v2_ext --out data/manifests/synth3_shards_v2_ext.json
HEAVY_SLOTS=10 $HERD/heavy.sh eval/synth3_study2_run.sh FL10 C_mix 1        # ARCH ARM SEED [PRETRAIN_STEPS]
HEAVY_SLOTS=10 $HERD/heavy.sh eval/synth3_curve_run.sh 025M 1               # SIZE SEED
```

## Sources

- SynthTIGER (Yim et al., ICDAR 2021, https://arxiv.org/abs/2107.09313): rendering ablations,
  length and character-distribution control on Japanese.
- Baek et al., "What is wrong with scene text recognition model comparisons?" (ICCV 2019) and
  "What if we only use real datasets?" (CVPR 2021): synthetic diversity, real fine-tuning.
- VNDB image processing (https://code.blicky.net/yorhel/vndb, `util/imgproc.c`: Q90 libvips) and
  database dumps (https://vndb.org/d14, ODbL 1.0 / DbCL 1.0).
- PPSSPP (`Core/Screenshot.cpp`, `Core/Font/PGF.cpp`), pspsdk `pspfont.h`, vitasdk `psp2/pvf.h`,
  `psp2/jpegenc.h`; CommitteeOfZero/impacto (MAGES glyph atlases, outline sheets, typewriter
  fade); Adrenaline and Sharpscale scale modes.
- Jiten frequency lists (https://jiten.moe/frequency-dictionaries, CC BY-SA 4.0): the visual-novel
  word list (2,104 VN scripts) and the kanji list.
