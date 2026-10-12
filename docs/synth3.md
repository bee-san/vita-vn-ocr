# synth3: synthetic VN dialogue lines calibrated to real console crops

synth3 is a new synthetic training set for the recognizer: grey line crops of Japanese
visual-novel dialogue, names, choices and UI as the PS Vita and PSP show them, in shards that
`train/train2.py` reads unchanged, plus a development split whose text, scenes and names share
no source document with training. It is meant to replace synth2 as the VN-style source in
train2's mixture. The generator is `synth/synth3.py`.

**Status (2026-10-12 13:35Z): no accuracy result yet.** The frozen ablation
(`eval/synth3_ablation_protocol.json`) had started when the owner stopped all runs to move to a
bigger machine; nothing was exported or scored. Generator version 1 was rendered (1,000,000
crops, `data/manifests/synth3_shards_20261012.json`) and then superseded by version 2 before any
scoring, after the label audit below found crops with unlabelled text. The 1M version-2 render and
the six primary runs are the next steps (`synth-dataset.handoff` outside the repo has the
commands).

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
(training-split games only, never a teacher_val, test or development VN), and a line counts only
when PP-OCRv6 small reads it with lowest character confidence at least 0.9 and another reader
agrees. These are consensus references, not hand labels, and part of this pool was a
pseudo-label source in train2's fine-tune.

| stratum | lines | FL10 CER | H15 CER |
|---|---|---|---|
| all | 3,339 | 2.56% | 1.25% |
| 1-4 characters | 1,182 | 5.03% | 2.78% |
| name-like | 1,368 | 5.61% | 3.16% |
| 25+ characters | 320 | 1.98% | 0.72% |
| console polarity dark (inverted) | 566 | 4.24% | 2.42% |
| ink 32 px and taller | 572 | 3.82% | 1.74% |
| native Vita | 259 | 3.41% | 1.79% |
| Latin letters and digits (by character) | 295 chars | 18.0% | 9.8% |
| kanji / katakana / hiragana (by character) | | 3.6% / 3.3% / 1.1% | 1.7% / 1.6% / 0.3% |

The most frequent substitutions are dakuten and handakuten (バ→パ, プ→ブ, ペ→ベ, ば→は, が→か,
じ→し, ず→す), small kana (っ→つ, ゃ→や), 『』→「」, ー/一, and dropped edge punctuation (」、。…！).
Short crops have about twice the CER of long ones, for both models.

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
| scenes | crops of the upper 62% of 6,260 training-split screenshots (synth2's list plus realdata's Vita, PSP and PC shots) from 1,185 VNs, else synth2's procedural scenes | synth2's scene list held 22 of teacher_val's 181 VNs; synth3 drops every development VN |

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
  characters that appear in any of 362 reserved VNs (eval's 172 test and family VNs, every
  is_test_vn VN, teacher_val's 181 VNs and the general-VN / CLANNAD development VNs), plus 300,000
  SyosetuNames (ODC-BY).
- Scenes (version 2): upper 62% of training-split screenshots whose text lines are known
  (realdata's line boxes, linefind's detections), and only patches that touch none of them.
- Vocabulary: `data/vocab.txt` (5,354 classes); lines with any other character are dropped
  whole. FL10's 3,500-class vocabulary reads the rest as `<unk>`, as with synth2.

### Version 1 render and label audit

Version 1 (1,000,000 crops, 100 shards, 5.95 GB, seed 20261012, rendered in 80 min with 10
cores, 0 render errors) pins in `data/manifests/synth3_shards_20261012.json`. The committed code
of that commit re-renders it byte for byte (first 1,000 crops of shards 0 and 73 checked).

`synth/synth3_audit.py` read 1,500 random crops of each set with PP-OCRv6 small in screen
polarity:

| set | teacher vs label CER | exact lines |
|---|---|---|
| synth2 (shards 00000-00039) | 3.00% | 73.3% |
| synth3 version 1 (shards 00000-00099) | 4.52% | 76.7% |

Most disagreements are the teacher misreading hard renders (pixel fonts, pen fonts, tiny PSP
names, faint text). Reading the 36 worst version-1 crops by eye also showed two generator
faults: a GRAD box that merged real text from a scene screenshot with a labelled line (the
label then covers part of the crop), and gt boxes widened across a neighbouring line whose ink
reached into the line's band. Over 100,000 version-1 crops, 2.8% of gt crops have an ink box
much wider than their label and 4.8% of GRAD crops are more than 2.2x their ink height. Version
2 removes both causes (scenes avoid known screen text; ink columns come from each line's own
glyph extent; a GRAD box must hold and hug exactly one labelled line; `tests/test_synth3.py`
checks the matcher). Version 2 has not been rendered or audited yet.

### Calibration (version 1 test renders; version 2 to be re-measured)

On 4,000 crops of a version-1 test render against 4,000 real training crops:

| | real | synth2 | synth3 v1 |
|---|---|---|---|
| label length p10 / p50 / p90 | 2 / 8 / 24 | 4 / 15 / 25 | 2 / 9 / 23 |
| crops of 2-3 characters | 24.5% | 7.9% | 20.5% |
| name-like crops | 40% | 16% | 34% |
| KL(real ‖ source), characters | | 0.199 | 0.137 |
| ！ / ？ / … per 1,000 characters | 5.8 / 6.4 / 4.8 | 1.9 / 3.8 / 3.5 | 7.2 / 8.4 / 4.9 |

## Ablation

The protocol (`eval/synth3_ablation_protocol.json`, committed before any run, amended once
before any scoring to point at the version-2 shards) trains FL10-sized models from scratch for
16,000 steps on the NVIDIA one-line-per-page set (0.5) plus synth2, synth3 or both (0.5), seeds 1
and 2, and compares the step-16000 EMA checkpoints through the C runtime (W8I) on teacher_val
in both polarities, with VN-cluster paired bootstraps, per-stratum and per-VN tables
(`eval/synth3_ablation.py`). It fixes the decision rule in advance. A release-comparable study
(the full train2 recipe with real fine-tuning, more seeds, an H15 round and a synth3 learning
curve) follows on the new machine, with its own protocol written before scoring. Results will be
added here; none exist yet.

## Limits

- No accuracy result yet (see Status).
- teacher_val is the only development cohort on this machine (the six-title general-VN cohort and
  the CLANNAD captures are private and not here). Two thirds of its lines are PC/console
  screenshots scaled to console size, not native console captures.
- Hard-example mining used machine-consensus references on training-side pools, part of which
  was a pseudo-label source in train2's fine-tune.
- Native game fonts were not used: the private CLANNAD font files are not on this machine. The
  dataset uses free and non-commercial fonts only.
- The rendering parameters for engines and captures come partly from documentation and source
  code (VNDB, PPSSPP, impacto) and partly from estimates; the measured ones are the VNDB JPEG
  tables (768 of 789 local Vita screenshots carry the standard Q90 tables, 667 are 4:2:0 and 122
  4:4:4) and the real crop geometry above.

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
    --conf 0.9 --forbid-vns $(cat $S3/mining/exclude_dev_vns.txt) --out $S3/mining/mining_v6s_consensus.json
synth/synth3_text.py targets --pools $P --real $A/teacher/teacher_train.npz+$A/realdata/real_train.npz \
    --jiten-vn $S3/text/jiten/frequency_list_visual_novel.csv --mining $S3/mining/mining_v6s_consensus.json
# render (two heavy.sh slots of 5 workers each), then pin
$HERD/heavy.sh synth/synth3.py shards --out $S3/shards/train_v2 --split train --shards 50 --first 0 --per-shard 10000 --workers 5
$HERD/heavy.sh synth/synth3.py shards --out $S3/shards/train_v2 --split train --shards 50 --first 50 --per-shard 10000 --workers 5
synth/synth3.py manifest --dir $S3/shards/train_v2 --out data/manifests/synth3_shards_v2.json
synth/synth3_stats.py measure --name real=$A/teacher/teacher_train.npz+$A/realdata/real_train.npz \
    --name s2=$A/synth2/shards/synth_0000[0-1].npz --name s3="$S3/shards/train_v2/synth3_0000[0-1].npz" --out stats.json
# the ablation: eval/synth3_run_arm.sh ARM SEED (through heavy.sh), then
eval/synth3_ablation.py export --ckpt $S3/ablation/runs/RUN/ckpt_step16000.pt --out RUN_w8.vocr
eval/synth3_ablation.py score --blob RUN=RUN_w8.vocr ... --dev $A/teacher/teacher_val.npz --out scores.json
eval/synth3_ablation.py compare --scores scores.json --protocol eval/synth3_ablation_protocol.json --out result.json
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
