# synth6: synthetic lines in the real game fonts, in each training title's own style

synth6 (VOCR6 track T2, [#125](https://github.com/bee-san/vita-vn-ocr/issues/125)) renders VN dialogue lines in the
**real fonts the owner's farm harvested from the games** (`fonts/` of the hooked corpus). Each line is drawn **the
way one training title draws its text**: that title's measured rows, ink height, pen advance, fill, outline or
halo, shadow and box. It goes **on that title's own pages**, on a text row the page leaves empty, with every text
box of the page masked out. synth3 and synth4 drew free fonts in a sampled palette over VNDB scenes; synth6 draws
the typefaces and screens the recognizer will read, with any text, including rare kanji and #119's confusions.

**Result.** In a pre-registered proof (4 runs from VOCR5-N, equal steps), synth6 is **better** than the same weight
of synth4. On the hval clean dialogue lines it reads −0.030 points [−0.061, −0.002] (0.444% → 0.413%; both seed pairs
lower). On the unseen-game development sets it scores −0.116 [−0.230, −0.014], where the extra synth4 weight had cost
VOCR5-N's development score. The page path is level: −0.010 [−0.051, +0.040]. The proof trained on v1 (see its
caveats below).

**The shard set to use is v3** (`data/manifests/synth6_shards_v3_{vita,psp2x,psp}.json`). v1
(`synth6_shards_v1_*`) is kept only because the proof's runs trained on it.
- v1: the independent audit found three renderer defects in about 1.5-2% of its crops.
- v2: fixed those; its own label audit then showed two more label-noise sources, a shifted font table and GRAD
  boxes cut through glyphs, about 0.5% of its PSP crops. v2 was never used and is not pinned.
- v3: fixes both (see "The shard sets"). The generator is `synth/synth6.py`
(with `synth6_fonts.py`, `synth6_assets.py` and `synth6_text.py`). The font inventory with every exclusion and check
is `data/manifests/synth6_fonts.json`, and the measured styles are `data/manifests/synth6_styles.json`. Nothing of
the corpus (pages, fonts, text) is committed, and no shard is either.

**Use.** The fonts are the games' own and the Vita/PSP system fonts, extracted by the owner's farm for private
training use only (`fonts/README.md` of the corpus). The backgrounds are the owner's captures. Neither they nor any
render is redistributed. A released model trained on synth6 must say so in its release notes, as for the real
hooked lines.

## What a synth6 line is

| part | how | measured on / checked by |
|---|---|---|
| title | one of the corpus' 46 drawable `train` titles (31 Vita, 15 PSP), drawn uniformly (title-balanced) | NPJH50127 has no verified lines and PCSG00653 only vertical ones: neither is drawn |
| background | a random **train** page of that title (vocr6-ingest's `splits_v1.json`: never an hval, guard or test page) | |
| placement (rows) | the title's dialogue rows that no box of this page touches (box ± 4 px Vita, 2 px PSP); 1-4 lines, up to 4 pages tried. The lines of one region form a chain: each next row is the nearest one below that does not overlap the previous row and starts within 2.5 row heights of it | rows clustered from the verified lines' boxes, at least 8 lines and 2% of the title's lines per row |
| placement (scene) | a free rectangle in the page's scene (top 6-72% of the frame, a line height from every box), bare only where the title's look keeps it legible (a hard halo, or a calm background 100+ luma levels from the fill), else on a box of the colour real text sits on in that title (`scene_box`, opacity 0.8-1.0) | |
| font, matched (55% where the title has one) | the title's own **verified** dialogue font at its native size or scale and letter spacing (fidelity records of the font pack), spacing ± 0.6 px | 21 of the 46 titles have one |
| font, cross | any of 293 drawable harvested fonts (128 faces from 58 games' files and the Vita/PSP system fonts), sized so its full-width ink is 0.82-0.93 of the title's pen advance, its spacing setting the title's advance (−10% to +12%); a face whose strikes would all need a scale outside 0.5-1.7 is redrawn | the cell-heavy atlases (cell up to 1.9× the ink) get the negative spacing their engines use |
| style | the title's fill (an accent colour for its share of non-plain lines), halo (radius, width, opacity, colour, hard/soft), drop shadow (offset, opacity, colour), jittered (± 10 RGB, ± 0.12 opacity); cross mode borrows another same-polarity title's halo and shadow 12% of the time | measured on 160 real lines per title (below) |
| name | a speaker name on a free name row that no drawn line overlaps (35%), in the title's name style | the title's name rows |
| text | below | |
| view | **vita** 960×544; **psp** 480×272 (model B); **psp2x** the PSP frame upscaled 2× after drawing, as Rubi captures a PSP game through Adrenaline (bilinear 80%, nearest 15%, bicubic 5%) | |
| crop | the renderer cell box + pad (the hooked pool's convention, 35%), the ink box + pad + extra margin (eval's, 35%), or the console's GRAD finder box on the dialogue region (30%; a line without a one-line GRAD box, or whose GRAD box holds less than 90% of its ink height or not all of its ink width, gets its ink crop). Pad 4 px Vita and 2×, 2 px PSP. Console polarity: a dark-text crop is inverted (`common.line_dark`) | real boxes on the page block a GRAD box that merges them |
| never cropped | a line whose ink or cell box leaves the frame; a line whose ink overlaps another drawn line's by a quarter of the smaller height; a line with a fill contrast under 60 luma levels, or any character cell under 40, or a busy surround (luma sd over 45) that no halo or shadow of half opacity frames | the real titles' median dialogue contrast is 80-250 levels (lowest NPJH50621, 80) |

### Text

| source | share | what |
|---|---:|---|
| pool | 40% | synth3's VN-register pools through synth3's own `dialogue_text` / `narration_text` / `short_text` (its focus control and CLANNAD exclusion markers) |
| hook | 25% | another train title's hook text (train pages only), re-wrapped to this title's rows: a title's own text is never drawn in its own style, so no synthetic line copies a real crop |
| target | 25% | a line holding a character drawn with weight (count + 5)^-0.7 over its count in the train titles' verified lines (×4 for a known confusable), restricted to what the drawing font can draw: a synth3 pool line holding it, a Jiten VN word holding it in a short carrier (synth4's curriculum words), or a name |
| confusion | 10% | #119's measured confusions: ＝ between katakana words, Latin letters and digits inside Japanese lines (I, l, D, BGM, VTuber, mSv/h, ...), 『』 against 「」, small kana, (han)dakuten katakana, ー against 一 and the lookalike kana and kanji groups |

The train titles' verified lines hold 1,291,940 characters. **2,783 of the 5,629 vocab classes never occur in them**
(2,575 of the 5,142 kanji), and 3,998 occur fewer than 20 times. These are the target source's main draws. A
full-width space follows a mid-line ！ or ？ (60% of them) in 20% of utterances, the gap that cuts lines in #119.
The wrap lets one closing mark (、。」ー… or a small kana) hang past the row's limit, as engines do, but not a
second. Labels are `teacher/vote.canon` of the drawn text, the hooked pool's label form.

**Never drawn:**

- a line sharing an 8-character run with a test title's line (Dance with Devils, PCSG00807), or equal to one of four
  or more characters;
- any line from the corpus' own text that equals one of the 6,876 hval or guard lines, or a pool, target or
  confusion line of six or more characters that equals one;
- CLANNAD's character names and title: synth3's exclusion markers filter its pools, names and (from v2 on) the
  Jiten-word carriers. CLANNAD has no farm text.

## Fonts: what is drawn and what is not

394 fonts in the pack (`data/manifests/synth6_fonts.json`):

| | fonts |
|---|---:|
| a test title's (CLANNAD PCSG00415 ×4, Dance with Devils PCSG00807 ×1) | 5 |
| byte-identical (content or file sha256) to a test title's font (PCSG00638's `gothp32` and `minp32`, which ship CLANNAD's files) | 2 |
| **near-identical** to a test title's font: 95% or more of their common glyph bitmaps identical (99.9-99.99% measured: the Key-port fonts of PCSG00414 Rewrite ×4, PCSG00638 Amatsumisora ×2, PCSG00768 Walpurgis ×4, PCSG00926 Hoshi Ori ×4, against CLANNAD's four) | 14 |
| decoder mismatch on its own title's pages and verified nowhere | 9 |
| fewer than 300 glyphs | 21 |
| vertical-writing punctuation at the horizontal code points (、。 top-right: tategaki fonts) | 5 |
| fewer than 2,000 glyphs (not drawn, kept in the inventory) | 43 |
| **failed the OCR check** (the 10 px and 15 px strikes of PCSG00216's `textfont_2`, whose strokes break at that size) | 2 |
| **drawn** | **293** |

**CLANNAD's glyph raster is still drawn, through 24 LUCA fonts** (measured by the second audit round). The
`gothic32`, `mincho32`, `modern32` and `marugothic32` of six Prototype titles (PCSG00183, PCSG00206, PCSG00620
Grisaia Spinout, PCSG00785, PCSG00988, PCSG01000) hold CLANNAD's 32 px raster, re-quantised through another
palette. None of their glyphs is byte-identical, so the rule above does not catch them, but for 92-98% of the
characters they share with CLANNAD's matching font the glyph has the same ink box and differs by at most one of 16
grey levels per pixel. They are drawn in 19,106 v3 Vita crops (3.8%), 0 PSP; 10,837 of those are PCSG00620's own
fonts, whose `gothic32` is its verified dialogue font. v1 drew them in 17,705 Vita crops, besides the 14 Key-port
copies (77k crops).

**They are kept, and disclosed.** Grisaia Spinout (PCSG00620) is a hooked train title. Its 1,696 real train lines
are drawn in that same raster, so every VOCR6 model trained on the hooked lines sees it whatever synth6 does, and
removing the fonts from synth6 would not make CLANNAD's raster unseen. It would also take away a train title's
matched font. So: a CLANNAD score of a model trained on synth6 v3 (or on the hooked lines) has seen CLANNAD's glyph
raster, though not its font files, text or pages, and VOCR6's CLANNAD reports must say so. Little Busters' LUCA
fonts are close but not the same (within two grey levels for 64-83% of the glyphs of three of them, by the audit's
measure).

Dance with Devils' FOT-ModeMinA Std B has no near copy in the pack, even by ink-box alignment (the audit's measure).
The same typeface is drawn from the Otomate titles' FFU atlases.

Four checks guard what a glyph shows, since a wrong glyph teaches a wrong label. The renderer never draws a
character that any of them denies in that font.

- **OCR check** (`synth6.py font-check`): 19 lines of common text with punctuation, digits and Latin, drawn in each
  font and read by VOCR5-L; a font passes at a CER of at most 8%.
- **Glyph check** (`synth6.py glyph-check`): every covered kana and kanji (lines of 8 in random order) is read by
  VOCR5-L and compared with three reference fonts (the Vita system gothic and mincho, Noto Sans JP) on its ink box.
  A glyph is denied in that font when:
  - the recognizer reads another (non-lookalike) character, and the glyph matches that character's references
    better than its own by 0.05 or more;
  - or the same image is drawn for three or more characters (a placeholder);
  - or 40% of a 40-character Shift-JIS window is denied (a block).

  It found one real decoder error. **Custom Drive's `lt18` (NPJH50649, verified on its pages) draws other kanji
  for its JIS level-2 codes** (丕 as 傲, 乖 as 嗅, and so on). The font's fidelity check (20 real lines) probably
  could not see this, because real lines rarely hold level-2 kanji. The block rule also denies 27 correct level-1
  kanji next to them (和, 話, 論 and others; measured by the audit), a coverage loss in the title's own font, not a
  label error.
  Placeholders came from the Otomate sysfonts (①②③ⅠⅡⅢ℃♡) and AnitoStd (a box for ゔヷヸヹヺ and JIS2004 forms
  such as 剝 and 噓). The rule also denies o in the Otomate sysfonts, where Latin o, Greek ο and Cyrillic о share one
  glyph, another coverage loss. Stylised faces lose some complex kanji to false positives.
- **Glyph geometry** (`synth6.py glyph-geometry`, `synth6_fonts.misplaced`): a glyph whose ink lies mostly (over
  40% of its coverage) outside its own pen cell, or that has no visible ink (maximum coverage under 0.3), is denied.
  The 164 LUCA fonts store ・：； in a full cell with a half-width advance, which drops the mark into the next
  character's cell (the audit found this in v1). Also denied: н in 40 fonts, ~ in 10, PCSG01083's , and .,
  PCSG00380 font_Vita's faint box-drawing glyphs, NPJH50132 jp8's │ and PCSG00007 mincho20's _. 167 fonts lose 1-6
  characters.
- **Shifted tables** (`synth6.py glyph-geometry`, `synth6_fonts.swapped`). The two checks above read kana and kanji
  and a few lines of text, so a symbol table can be wrong unseen. Every symbol, punctuation mark, Latin letter and
  digit is compared in Shift-JIS order with the reference glyphs of itself and of its neighbours 1-2 codes away
  (shape cosine). A run of 8 or more characters that match a neighbour better than themselves, at one shift, is a
  table off by that many codes, and the whole run is denied (one odd glyph is not enough: tiny marks like 。．
  match their neighbours by chance).
  - **NPJH50649's `lt18`**: correct up to （）, then shifted by one from 0x8176 on: 」 is drawn as 「 (the font has
    no 「), 『 as 」, 【 as 』, ＝ as ÷, ○ as ★, ° as ♀, ■ as □ ... 63 characters.
  - **PCSG00629's `advfont`**: 16 half-width marks !"#$%&'()*+,-./0.
  - No other drawn font. In v2, 0.38-0.39% of the PSP crops drew a symbol of lt18's run (＝ in 40% of them; 『 and
    』 in most of the rest), and 0.035% of the Vita crops.
  - The shift counts assigned codes only, as a decoder's table does: ◯ (0x81FC) follows four unassigned codes, and
    the first version of the rule, which counted raw codes, missed it. v3 rendered with that version, so 6 of its
    PSP crops show ◯ as ¶; main's rule denies ◯ too.
- **Single wrong glyphs** at rarely used codes are not checked automatically outside a shifted run. The second audit
  round's per-symbol screen found 5, confirmed by eye: ┐ drawn as a wave in PCSG00617's and PCSG00838's `font24`, "
  as a box icon in PCSG00838's, ≪ as a long dash in NPJH50621's `font16`, and 〒 as 」 in PCSG01083's. They are listed
  by hand (`synth6.MANUAL_DENY`) and denied on main. v3, rendered before, holds 21 crops of them.

Per drawn font, the median is 5 denied characters and the p90 is 23 (before the shifted-table check).

## Styles, measured

`synth6_assets.py` measures 160 random verified lines of each title (and 60 names) on its train pages. Each crop is
the box + 5 px. Polarity comes from `common.line_dark`. The fill is the Otsu-bright core (the dark text is
inverted first). Rings 1-4 px outside the glyph mask (anti-aliased edge included) are compared with the far
background, which gives the outline or halo; the asymmetry of the mask shifted down-right against up-left gives the
drop shadow.

The box opacity comes from the empty text rows: a box of opacity a over changing scenes varies across pages
(1 − a)² times as much as the scene does. It is descriptive only. A row placement keeps the page's own box. A
`scene_box` placement draws a flat box (no texture: NPJH50621's textured box is not reproduced) in the colour real
text sits on in that title (`bg_rgb_p50`), at opacity U(0.8, 1.0). A soft halo renders a little fainter than
measured: its measured colour is the blended ring colour, and compositing applies its opacity again. Hard outlines
are not affected. Examples (`synth6_styles.json`):

| title | fill | halo | shadow | box (rgb, opacity) |
|---|---|---|---|---|
| PCSG00620 Grisaia Spinout | white | soft dark, r2, 0.23 | (1, 2), 0.38 | blue, 0.53 |
| PCSG00469 Clock Zero | black (dark text) | hard white, r1 w2, 0.88 | - | light, 0.78 |
| PCSG00866 Collar x Malice | light grey | hard black, r2, 0.84 | - | none (about 0: text on the scene) |
| NPJH50132 Memories Off Sorekara | dark | - | white (1, 1), 0.56 | white, 0.66 |
| ULJM05383 planetarian | white | - | - | black, 0.69 |

## Checks

- **By eye**, on contact sheets at 32 px line height (local only, as the data policy requires): real crops of a
  title next to synth6 crops of its style (`synth6.py compare`), for all 46 titles (Vita native, PSP native and PSP
  2×), and previews of every view.
  - Matched lines look like the real ones on the sheets: the same glyphs, size, spacing, halo and shadow.
  - Cross lines carry the title's look in another game's typeface.
  - Found and fixed before v1:
    - a uint8 overflow in the luma measure;
    - cross fonts drawn with their atlas cell as the advance (the wide gaps of the cell-heavy atlases);
    - the tategaki fonts' punctuation;
    - illegible bare text on busy scenes, the cause of the legibility rule and the scene rule;
    - stray one-page layouts, the cause of the 8-line row minimum.
- **The independent audit** of v1 (`audit-vocr6-synth6`, round 1) re-derived the pins, the determinism, the
  font and label checks and every table, and found what the sheets had missed (B1-B4). Every finding is fixed in
  v2 (and v3) and pinned by a test:
  - B1: lines of two overlapping layouts stacked on each other;
  - B2: glyphs cut at the frame edge, and runs of closing marks hanging past it;
  - B3: ・：； drawn outside their cell;
  - B4: CLANNAD's glyph bitmaps in 14 fonts of 4 other titles;
  - F2: lines partly invisible over a bright patch.
- **v2's own label audit** (the same measure as v1's; its worst-crop sheets read by eye) found what B1-B4 had
  hidden: lt18's shifted symbol table (□□□ for ■■■, ♀ for °) and GRAD boxes cut through the glyphs (a 15 px band
  across a 30 px line). Both are fixed in v3, with tests.
- **The second audit round** (on v3) reproduced the pins, the re-render, the label audit and the proof. It found:
  - B5: CLANNAD's raster in 24 LUCA fonts, kept and disclosed (above);
  - 38 residual crops: F7, a GRAD box holding only part of a line at the region edge; F8, ◯ in lt18 and 5 single
    wrong symbols. These are fixed on main with tests, and v3 keeps them.
- **Unit tests** (`tests/test_synth6.py`, CI fixtures only):
  - drawing: atlas drawing at scale 1 is exact; scale and spacing move the pen; the wrap and its one-mark overhang;
  - fonts: every inventory exclusion, including near-identical copies; the misplaced-glyph rule;
  - styles and layout: the outline, shadow and polarity measures on drawn crops; rows and free rows; row chains
    never stack;
  - text: the test-text filter; the marker filter on word carriers;
  - compositing: legibility, per character too;
  - crops: lines off the frame or stacked are not cropped; a GRAD box must cover its line;
  - shifted tables: a run is found, a single odd glyph is not;
  - output: a region, a shard's format and its byte-for-byte determinism; the 2× view.

## The shard sets

### v3: the set to use

`data/manifests/synth6_shards_v3_{vita,psp2x,psp}.json` pin 1,000,000 crops of `synth/synth6.py` at commit
`17e3365` (every code file's sha256 and every asset's are in the recipes), seed 20261028, under
`$HERD/shared/artifacts/vocr6-synth6/shards/v3/`:

- **vita**: 500,000 crops in 50 shards (4.63 GB);
- **psp2x**: 250,000 in 25 (2.89 GB), model A's view of PSP games;
- **psp**: 250,000 in 25 (0.93 GB), native, for model B.

It was rendered and checked in one heavy slot (5 workers) in 66 minutes, with 0 render errors. **Determinism:** the first 1,000 crops
of 6 pinned shards (vita 0 and 31, psp2x 0 and 12, psp 0 and 20), re-rendered in a fresh process, are byte-identical
(pixels, shapes, labels and meta).

**The audit's defects, measured on every v3 crop** (from the metas; `work/defect_rates.py`):

| | v1 (the audit's measure) | v3 |
|---|---:|---:|
| a line drawn over another line (B1) | 0.35-0.43% | 0 |
| ink beyond the frame (B2) | 1,183 / 5,459 / 5,432 crops (vita / psp2x / psp) | 0 |
| ・：； in a font that draws them outside their cell (B3) | 0.19% | 0 |
| a GRAD box holding less than 80% of its line's ink height | (v2: 0.12 / 0.31 / 0.01%) | 0 |
| a GRAD box holding only part of its line's width (the line at the region edge; F7) | | 11 psp crops (NPJH50279) |
| a symbol of lt18's shifted table | (v2: 0.035 / 0.39 / 0.38%) | 6 crops (◯; F8) |
| a single wrong symbol glyph (F8) | | 21 crops |

The 0.006-0.010% of v3 crops that hold ・：； in PCSG01083's font are drawn inside that font's narrow advance; in
cross mode a negative spacing can still press the dot into the next glyph (1 of 4 such crops the audit viewed, 80
crops in all). The 38 residual crops (F7, F8) are fixed in main's renderer, not in v3: 38 in 1,000,000 did not
justify another render. Also in v3, measured by the audit:
- a page-UI triangle (▶) over the first glyph in 2 of 126 random crops, with the label right;
- in tall-cell cross fonts, a cell crop holding part of the next drawn line (625 / 1,045 / 676 crops at 25% or more of
  its height);
- drawn ink over 25% or more of a real text box's height in 34 PSP crops, mostly NPJH50297.

| | vita | psp2x | psp |
|---|---:|---:|---:|
| titles / fonts drawn | 31 / 262 | 15 / 158 | 15 / 156 |
| matched / cross font | 32.1 / 67.9% | 14.8 / 85.2% | 14.6 / 85.4% |
| rows / scene_box / bare scene | 72.1 / 22.8 / 5.2% | 64.8 / 25.6 / 9.6% | 64.9 / 25.4 / 9.7% |
| crop: ink / cell / GRAD | 41.0 / 34.9 / 24.0% | 44.4 / 34.9 / 20.7% | 38.8 / 35.1 / 26.2% |
| inverted (dark text) | 29.1% | 17.9% | 17.8% |
| names | 1.6% | 1.4% | 1.4% |
| text: pool / hook / target / confusion | 40.2 / 24.2 / 24.3 / 9.7% | 40.3 / 24.4 / 24.2 / 9.8% | 40.4 / 24.3 / 24.4 / 9.6% |
| label length p10 / p50 / p90 | 5 / 14 / 25 | 5 / 13 / 24 | 5 / 13 / 24 |
| fill contrast p10 / p50 (luma levels) | 116 / 172 | 101 / 146 | 101 / 146 |
| Adrenaline filter: bilinear / nearest / bicubic | | 80.0 / 15.0 / 5.0% | |

The real train lines are 7 / 16 / 25 characters long (p10 / p50 / p90) on the Vita and 5 / 15 / 24 on the PSP.

**Coverage.** The vita set draws 5,543 of the 5,629 vocab classes. The 3,998 classes that occur fewer than 20 times
in the real train lines get a median of 34 exposures in it (p10 16, p90 122), and 16-17 in each PSP set. synth4's
s4v5631 gives them a median of 76 in 1,000,000 crops, with a curriculum aimed at exactly that. synth6 spends its
crops on the real fonts and styles instead; its targets are a quarter of the lines.

**Balance.** Titles are drawn uniformly (Vita 3.23% each, PSP 6.67%). The crops follow within ±20% for 27 of 31
Vita titles but only 5 (psp2x) and 6 (psp) of 15 PSP titles. Under-produced titles:
- PCSG00751 0.98% and PCSG00468 1.73% (scattered rows);
- NPJH50621 2.1% and NPJH50356 2.3%, whose text has the lowest real contrast (80 and 87 levels) on textured boxes,
  so many of their lines fail the legibility rule;
- NPJH50551 4.6% and NPJH50297 5.1%.

ULJM05674 (9.6%) and NPJH50641 (8.6%) take their share. A title-weighted acceptance would even this out; v3 does
not have it.

**Label audit** (`synth/synth3_audit.py`, as synth4's: PP-OCRv6 small reads 1,500 random crops per view, seed 1,
in screen polarity). The teacher reads the labels as follows:

| teacher vs label CER (exact lines) | vita | psp2x | psp |
|---|---:|---:|---:|
| **v3** | **1.52%** (84.3%) | **1.68%** (83.7%) | **1.57%** (84.7%) |
| v3, matched / cross font | 1.47 / 1.54% | 1.07 / 1.78% | 0.95 / 1.68% |
| v3, rows / scene_box / bare scene | 1.61 / 0.95 / 2.30% | 1.73 / 1.37 / 2.31% | 1.52 / 1.14 / 3.35% |
| v2 | 1.19% | 1.80% | 1.75% |
| v1 | 1.51% (84.9%) | 2.28% (79.0%) | 2.27% (79.4%) |
| synth4 v1 (`docs/synth4.md`, same measure) | 4.56% (76.7% exact), all views | | |
| synth3 v3 | 3.83% (77.9%), all views | | |

The worst 36 crops of v1's vita and psp2x audits, v2's psp2x audit and v3's vita audit were read by eye. Most
disagreements are teacher errors: rare kanji in stylised faces, single characters, brackets read as [ ], ー against
―, small kana. The sheets found:
- **v1**: real text the page's boxes miss lying under the drawn line, 3-4 of the 1,500 crops in each of the two
  audits, 0.2-0.3% (Clopper-Pearson 95%: 0.04-0.58% for 3, 0.07-0.68% for 4). It shows only when it misleads the
  teacher, so this is a lower bound. These are old speech bubbles, earlier NVL lines or unboxed screen text, on
  titles where text sits on the scene (NPJG00016, PCSG00648, NPJH50356). In 37 random crops of the 8 titles with
  scattered rows or no box, 1 is a clear case and 1 unclear. The audit's own random sample (126 crops) had none,
  and 1 drawn-line overlap (B1).
- **v2**: lt18's shifted table and the cut GRAD bands, both fixed in v3.
- **v3**: nothing systematic.

Two detectors for real text under a row were measured and rejected:
- a GRAD line in a box-free row is found on 0-100% of rows per title, but on the sheets it is mostly box decoration
  (Starry Sky's constellations, Clock Zero's gears, Tenshin Ranman's logo) or scenery, which real text also sits on
  (`work/free_row_ink.json`);
- masking the previous 8 pages' boxes as well leaves the inked share unchanged: NPJG00016's free rows are inked 0.74
  when no previous page boxed them and 0.83 when one did (`work/prev_boxes.json`).

### v1: the proof's set

`data/manifests/synth6_shards_v1_{vita,psp2x,psp}.json` pin the 1,000,000 crops the proof trained on. They were
rendered at `d912850` (tag `synth6-v1-render-d912850`), seed 20261026, with 307 drawn fonts; each composition
share is within 2.5 points of v3's. The first 1,000 crops of 6 shards re-render byte-identical from the tag.

**Do not train on v1.** Its measured defects are in the table above. It also has:
- the 14 near-identical CLANNAD fonts, in 7.7% of its crops;
- 2 crops showing 有紀寧, a CLANNAD character's given name, from a Jiten word carrier for 寧. The carrier filter
  came later.

v2 (code `5483ee8`, tag `synth6-v2-render-5483ee8`, seed 20261027) was rendered, measured and replaced by v3
without being used; it is not pinned.
v3 was rendered at `17e3365` (tag `synth6-v3-render-17e3365`) with the asset files whose sha256 its recipes record.
Main's renderer adds F7 and F8 (the GRAD width rule, the assigned-code shift, the manual denials), so a shard
re-rendered with main can differ from v3 where those apply. v3 has 3 crops whose label holds ことみ inside
ということみたい (hook text), no CLANNAD name.

## The proof (pre-registered)

`eval/synth6_proof_protocol.json`, committed at `00132a1` (tag `synth6-proof-protocol-00132a1`) and logged in the
VOCR6 experiment log at 10:07:34Z, before any proof run (10:52Z). The render code `d912850` and the v1 pins came
after the protocol commit but before the first run: no outcome could inform them.

- **Arms.** Both start from the released VOCR5-N (`v5f_N_seed3/ckpt_step15000.pt`) and fine-tune it 8,000 steps
  (lr 0.0012, warmup 500, batch 96, EMA 0.999). The mix is vocr6-lead's stage-1 mix m1, with the hooked train lines
  at 0.25, at tag `vocr6-runner-88a0c24`. The arms differ only in one 0.15 synthetic slot:
  - P4: s4v5631, raised from 0.05 to 0.15;
  - P6: synth6 v1, Vita 0.10 and PSP 2× 0.05.
  - Seeds 1 and 2; the model is the final step.
- **Scoring.** The shipped form: W8I C runtime + Rubi's per-line decode (counter re-decode, beam 8 + the 1 MB LM,
  textnorm, the non-Japanese filter; no cascade). It uses the lead's `eval/vocr6_score.py` plans (hval: 46 titles'
  screening pages; dev: #119's tv / dn / wd without the hooked titles' VNs) and `eval/vocr6_select.py compare`.
- **Rule.** hval clean dialogue CER is primary: synthetic lines teach the recognizer, which clean crops isolate from
  the finder.
  - better: its interval below 0, both seed pairs below 0, and dev not resolved worse;
  - not worse: no component resolved worse;
  - worse: otherwise.
  - The proof adopts nothing: synth6 enters VOCR6 through vocr6-lead's stage 2 (S2s6).

| | VOCR5-N | P4 seed 1 / 2 | P6 seed 1 / 2 | P6 − P4 (95% interval) |
|---|---:|---:|---:|---:|
| hval clean dialogue | 0.56% | 0.44 / 0.45% | 0.42 / 0.41% | **−0.030 [−0.061, −0.002]** (pairs −0.020, −0.041) |
| ... Vita / PSP 2× | 0.50 / 0.68% | | | −0.027 [−0.063, +0.009] / −0.037 [−0.091, +0.006] |
| hval page path (junk-free units) | 2.35% | 2.23 / 2.26% | 2.22 / 2.26% | −0.010 [−0.051, +0.040] |
| hval page path, junk boxes included | 3.24% | 3.51 / 2.73% | 3.33 / 3.48% | +0.291 [−0.045, +0.997] |
| dev score (0.25 tv + 0.25 dn + 0.5 wd) | 1.990 | 2.177 / 2.056 | 1.959 / 2.043 | **−0.116 [−0.230, −0.014]** |
| ... tv / dn / wd | 0.66 / 0.66 / 3.32 | | | −0.108 / −0.087 / −0.134 |

**Verdict: better** (the registered rule; `$A/proof/summary.json`). It passes at the rule's edge: the hval clean
interval's upper bound is −0.002. The lead's stage rule, page-path primary, would
not adopt on this proof alone: the page path is level.

Reported, not decided:
- **Greedy decode:** hval clean −0.038 [−0.069, −0.007], hval page −0.074 [−0.124, −0.029], dev −0.039
  [−0.104, +0.030].
- **Rare characters:** the 288 hval clean lines (5,888 characters) that hold a character with fewer than 20
  occurrences in the real train lines read 0.433% → 0.365% with Rubi's decode: −0.068 [−0.146, +0.011], not
  resolved. Greedy: +0.008 [−0.109, +0.173].
- **The extra synth4 weight** costs VOCR5-N's development score (point values, no interval: +0.19 and +0.07 for the
  two P4 seeds); synth6 does not (−0.03 and +0.05).

**Caveats.**
- The proof trained on **v1**: about 1.5-2% of its crops carry the defects v2 and v3 fix. It also drew CLANNAD's
  glyph raster: the 14 Key-port copies (77k crops) and the 24 LUCA re-quantised copies (17,705 Vita crops). No
  CLANNAD score was taken.
- The measured gain is for v1 against synth4 at 8,000 steps, at one tier (N). v3's cleaner labels are not measured.
  Stage 2's S2s6 measures synth6 v3 on/off in the stage's own recipe.

## Limits

- **46 titles.** synth6 teaches these titles' looks (and 128 faces' glyphs) with any text. It does not show a box
  style no train title has. The cross mode and the unseen-game dev score are the guard against overfitting to them.
- **Styles are measured, not hand-checked per title.** The halo, shadow and box detectors are simple. Eye checks on
  the compare sheets agree with them, but a measured opacity can be off where a box is textured, or where the text
  sits on the scene.
- **The glyph checks are filters, not a proof.** For kana and kanji they catch a glyph read as another character by
  two independent measures. They also catch placeholders, glyphs outside their cell and shifted symbol tables. A
  single wrong symbol outside a shifted run is not checked automatically: the 5 the audit found are denied by hand,
  and others may remain.
- **Real text under a drawn line**, where a page shows text its boxes miss (old speech bubbles, earlier NVL lines):
  about 0.2-0.3% of crops in v1's label audits, more on bubble and NVL titles; v3 has not changed it. A later
  version could draw such titles only in `scene_box` (its own box hides what lies below).
- **Two low-contrast PSP styles are under-represented** (NPJH50621, NPJH50356): the legibility rule rejects many of
  their lines.
- **No furigana, ruby, page cursors, typewriter-partial lines or vertical text** (train2's finder augmentation adds
  cursors and neighbour rows). Names are 2% of the crops.
- **The PSP-native view is rendered but not proven.** It is for model B (vocr6-psp).
- Nothing here ran on a console.

## Reproduce

```sh
A=$HERD/shared/artifacts/vocr6-synth6
# the font pack, extracted once (private; never committed)
mkdir -p $A/fontpack && zstd -dc $HERD/shared/private/hooked-b2/fonts/font-pack.tar.zst | tar -x -C $A/fontpack
# per-title styles, layout, train hook text, counts, the font inventory, the test and hval text guards
HEAVY_SLOTS=15 $HERD/heavy.sh synth/synth6.py assets --out $A/assets --workers 5
synth/synth6.py inventory --assets $A/assets            # the inventory alone (exclusions by hash and glyph identity)
# the OCR check, the per-glyph check (VOCR5-L) and the glyph geometry
CK=$HERD/shared/artifacts/vocr5-train/runs/v5f_L_seed2/ckpt_step15000.pt
HEAVY_SLOTS=15 $HERD/heavy.sh synth/synth6.py font-check --ckpt $CK --out $A/assets/font_check.json
HEAVY_SLOTS=15 $HERD/heavy.sh synth/synth6.py glyph-check --ckpt $CK --out $A/assets/glyph_check.json
synth/synth6.py glyph-geometry --out $A/assets/glyph_geometry.json
synth/synth6.py manifests --assets $A/assets          # data/manifests/synth6_fonts.json, synth6_styles.json
# eye checks
synth/synth6.py compare --title PCSG00620,PCSG00146 --n 3 --matched --out $A/work/cmp/cmp.png
synth/synth6.py preview --view psp2x --out $A/work/prev
# the v3 render (one heavy slot), the pins, the label audit
for v in vita:50 psp2x:25 psp:25; do
  HEAVY_SLOTS=15 $HERD/heavy.sh synth/synth6.py shards --view ${v%%:*} --out $A/shards/v3/${v%%:*} --shards ${v##*:} \
      --per-shard 10000 --workers 5 --seed 20261028
done
for v in vita psp2x psp; do synth/synth6.py manifest --dir $A/shards/v3/$v --out data/manifests/synth6_shards_v3_$v.json; done
for v in vita psp2x psp; do
  HEAVY_SLOTS=15 $HERD/heavy.sh .venv/bin/python synth/synth3_audit.py --shards "$A/shards/v3/$v/synth6_${v}_000??.npz" \
      --n 1500 --seed 1 --keys platform,view,crop,kind,mode,placement,font_kind --out audit_$v.json --sheet worst_$v.png
done
```

Training with it: `--src 's6v:W:$A/shards/v3/vita/synth6_vita_000??.npz:stream=1,pool=3,inv=0.2'` (Vita),
`s6x` with `psp2x/synth6_psp2x_000??.npz` (model A's PSP view) and `s6p` with `psp/synth6_psp_000??.npz` (model B).
The crops are native size in console polarity, and train2 resizes them to its input height.
