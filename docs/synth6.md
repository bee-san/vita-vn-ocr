# synth6: synthetic lines in the real game fonts, in each training title's own style

synth6 (VOCR6 track T2, [#125](https://github.com/bee-san/vita-vn-ocr/issues/125)) renders VN dialogue lines in the
**real fonts the owner's farm harvested from the games** (`fonts/` of the hooked corpus). Each line is drawn **the
way one training title draws its text**: that title's measured rows, ink height, pen advance, fill, outline or
halo, shadow and box. It goes **on that title's own pages**, on a text row the page leaves empty, with every text
box of the page masked out. synth3 and synth4 drew free fonts in a sampled palette over VNDB scenes; synth6 draws
the typefaces and screens the recognizer will read, with any text, including rare kanji and #119's confusions.

**Result:** PROOF_RESULT_PLACEHOLDER

The generator is `synth/synth6.py` (with `synth6_fonts.py`, `synth6_assets.py` and `synth6_text.py`). The shard set
is pinned by `data/manifests/synth6_shards_v1_{vita,psp2x,psp}.json`. The font inventory with every exclusion and
check is `data/manifests/synth6_fonts.json`, and the measured styles are `data/manifests/synth6_styles.json`.
Nothing of the corpus (pages, fonts, text) is committed, and no shard is either.

**Use.** The fonts are the games' own and the Vita/PSP system fonts, extracted by the owner's farm for private
training use only (`fonts/README.md` of the corpus). The backgrounds are the owner's captures. Neither they nor any
render is redistributed. A released model trained on synth6 must say so in its release notes, as for the real
hooked lines.

## What a synth6 line is

| part | how | measured on / checked by |
|---|---|---|
| title | one of the corpus' 46 drawable `train` titles (31 Vita, 15 PSP), drawn uniformly (title-balanced) | NPJH50127 has no verified lines and PCSG00653 only vertical ones: neither is drawn |
| background | a random **train** page of that title (vocr6-ingest's `splits_v1.json`: never an hval, guard or test page) | |
| placement (rows) | the title's dialogue rows that no box of this page touches (box ± 4 px Vita, 2 px PSP); 1-4 lines, up to 4 pages tried | rows clustered from the verified lines' boxes, at least 8 lines and 2% of the title's lines per row |
| placement (scene) | a free rectangle in the page's scene (top 6-72% of the frame, a line height from every box), bare only where the title's look keeps it legible (a hard halo, or a calm background 100+ luma levels from the fill), else on a box of the colour real text sits on in that title (`scene_box`, opacity 0.8-1.0) | |
| font, matched (55% where the title has one) | the title's own **verified** dialogue font at its native size or scale and letter spacing (fidelity records of the font pack), spacing ± 0.6 px | 21 of the 46 titles have one |
| font, cross | any of 307 drawable harvested fonts (142 faces from 62 games' files and the Vita/PSP system fonts), sized so its full-width ink is 0.82-0.93 of the title's pen advance, its spacing setting the title's advance (−10% to +12%) | the cell-heavy atlases (cell up to 1.9× the ink) get the negative spacing their engines use |
| style | the title's fill (an accent colour for its share of non-plain lines), halo (radius, width, opacity, colour, hard/soft), drop shadow (offset, opacity, colour), jittered (± 10 RGB, ± 0.12 opacity); cross mode borrows another same-polarity title's halo and shadow 12% of the time | measured on 160 real lines per title (below) |
| name | a speaker name on a free name row (35%), in the title's name style | the title's name rows |
| text | below | |
| view | **vita** 960×544; **psp** 480×272 (model B); **psp2x** the PSP frame upscaled 2× after drawing, as Rubi captures a PSP game through Adrenaline (bilinear 80%, nearest 15%, bicubic 5%) | |
| crop | the renderer cell box + pad (the hooked pool's convention, 35%), the ink box + pad + extra margin (eval's, 35%), or the console's GRAD finder box on the dialogue region (30%; a line without a one-line GRAD box gets its ink crop). Pad 4 px Vita and 2×, 2 px PSP. Console polarity: a dark-text crop is inverted (`common.line_dark`) | real boxes on the page block a GRAD box that merges them |
| legibility | a line is cropped only with a fill contrast of 60+ luma levels and a calm surround (luma sd ≤ 45) unless a halo or shadow of half opacity frames it | the real titles' median dialogue contrast is 80-250 levels (lowest NPJH50621, 80) |

### Text

| source | share | what |
|---|---:|---|
| pool | 40% | synth3's VN-register pools through synth3's own `dialogue_text` / `narration_text` / `short_text` (its focus control and CLANNAD exclusion markers) |
| hook | 25% | another train title's hook text (train pages only), re-wrapped to this title's rows: a title's own text is never drawn in its own style, so no synthetic line copies a real crop |
| target | 25% | a line holding a character drawn with weight (count + 5)^-0.7 over its count in the train titles' verified lines (×4 for a known confusable), restricted to what the drawing font can draw: a synth3 pool line holding it, a Jiten VN word holding it in a short carrier (synth4's curriculum words), or a name |
| confusion | 10% | #119's measured confusions: ＝ between katakana words, Latin letters and digits inside Japanese lines (I, l, D, BGM, VTuber, mSv/h, ...), 『』 against 「」, small kana, (han)dakuten katakana, ー against 一 and the lookalike kana and kanji groups |

The train titles' verified lines hold 1,291,940 characters. **2,783 of the 5,629 vocab classes never occur in them**
(2,575 of the 5,142 kanji), and 3,998 occur fewer than 20 times. These are the target source's main draws. A
full-width space follows a mid-line ！ or ？ (60% of them) in 20% of utterances, the gap that cuts lines in #119. Labels are
`teacher/vote.canon` of the drawn text, the hooked pool's label form.

**Never drawn:**

- a line sharing an 8-character run with a test title's line (Dance with Devils, PCSG00807), or equal to one of four
  or more characters;
- any line from the corpus' own text that equals one of the 6,876 hval or guard lines, or a pool, target or
  confusion line of six or more characters that equals one;
- CLANNAD's character names and title: synth3's exclusion markers filter its pools and names, and (from the commit
  after the v1 render) the Jiten-word carriers too. CLANNAD has no farm text. 2 v1 crops hold one name; see the v1
  disclosures below.

## Fonts: what is drawn and what is not

394 fonts in the pack (`data/manifests/synth6_fonts.json`):

| | fonts |
|---|---:|
| a test title's (CLANNAD PCSG00415 ×4, Dance with Devils PCSG00807 ×1) | 5 |
| byte-identical (content or file sha256) to a test title's font (PCSG00638's `gothp32` and `minp32`, which ship CLANNAD's files) | 2 |
| decoder mismatch on its own title's pages and verified nowhere | 9 |
| fewer than 300 glyphs | 21 |
| vertical-writing punctuation at the horizontal code points (、。 top-right: tategaki fonts) | 5 |
| fewer than 2,000 glyphs (not drawn, kept in the inventory) | 43 |
| **failed the OCR check** (two brush faces of PCSG00216) | 2 |
| **drawn** | **307** |

Two checks guard the glyph-to-character map, since a wrong map teaches a wrong label:

- **OCR check** (`synth6.py font-check`): 19 lines of common text with punctuation, digits and Latin, drawn in each
  font and read by VOCR5-L; a font passes at a CER of at most 8%. 307 of 309 passed.
- **Glyph check** (`synth6.py glyph-check`): every covered kana and kanji (lines of 8 in random order) is read by
  VOCR5-L and compared with three reference fonts (the Vita system gothic and mincho, Noto Sans JP) on its ink box.
  A glyph is denied in that font when:
  - the recognizer reads another (non-lookalike) character, and the glyph matches that character's references
    better than its own by 0.05 or more;
  - or the same image is drawn for three or more characters (a placeholder);
  - or 40% of a 40-character Shift-JIS window is denied (a block).

  It found one real decoder error. **Custom Drive's `lt18` (NPJH50649, verified on its pages) draws other kanji
  for its JIS level-2 codes** (丕 as 傲, 乖 as 嗅, and so on). The block rule also denies the last level-1
  characters before them. The font's fidelity check (20 real lines) probably could not see this, because real
  lines rarely hold level-2 kanji. Placeholders came from the Otomate sysfonts (①②③ⅠⅡⅢ℃♡) and AnitoStd
  (a box for ゔヷヸヹヺ and JIS2004 forms such as 剝 and 噓). Stylised faces lose some complex kanji to false
  positives.

  Per drawn font, the median is 2 denied characters and the p90 is 22.

The CLANNAD typeface (Prototype's ゴシック) is also drawn by train titles in other files: Little Busters' LUCA
FONT.PAK, which is a train title with real pages, and other Key/Prototype ports. Only CLANNAD's own files and
byte-identical copies are excluded, so a CLANNAD score of a model trained on synth6 (or on Little Busters' real
lines) has seen its typeface but not its font files, text or pages. Dance with Devils' FOT-ModeMinA Std B is
likewise drawn from the Otomate titles that ship that typeface as FFU atlases.

## Styles, measured

`synth6_assets.py` measures 160 random verified lines of each title (and 60 names) on its train pages. Each crop is
the box + 5 px. Polarity comes from `common.line_dark`. The fill is the Otsu-bright core (the dark text is
inverted first). Rings 1-4 px outside the glyph mask (anti-aliased edge included) are compared with the far
background, which gives the outline or halo; the asymmetry of the mask shifted down-right against up-left gives the
drop shadow.

The box opacity comes from the empty text rows: a box of opacity a over changing scenes varies across pages
(1 − a)² times as much as the scene does. Examples (`synth6_styles.json`):

| title | fill | halo | shadow | box (rgb, opacity) |
|---|---|---|---|---|
| PCSG00620 Grisaia Spinout | white | soft dark, r2, 0.23 | (1, 2), 0.38 | blue, 0.53 |
| PCSG00469 Clock Zero | black (dark text) | hard white, r1 w2, 0.88 | - | light, 0.78 |
| PCSG00866 Collar x Malice | light grey | hard black, r2, 0.84 | - | black, 0.23 |
| NPJH50132 Memories Off Sorekara | dark | - | white (1, 1), 0.56 | white, 0.66 |
| ULJM05383 planetarian | white | - | - | black, 0.69 |

## Checks before the render

- **By eye**, on contact sheets at 32 px line height (local only, as the data policy requires): real crops of a
  title next to synth6 crops of its style (`synth6.py compare`), for all 46 titles (Vita native, PSP native and PSP
  2×), and previews of every view.
  - Matched lines look like the real ones on the sheets: the same glyphs, size, spacing, halo and shadow.
  - Cross lines carry the title's look in another game's typeface.
  - What the sheets found and what was fixed:
    - a uint8 overflow in the luma measure;
    - cross fonts drawn with their atlas cell as the advance (the wide gaps of the cell-heavy atlases);
    - the tategaki fonts' punctuation;
    - illegible bare text on busy scenes, the cause of the legibility rule and the scene rule;
    - stray one-page layouts, the cause of the 8-line row minimum.
- **Unit tests** (`tests/test_synth6.py`, CI fixtures only): atlas drawing at scale 1 is exact; scale and spacing
  move the pen; every inventory exclusion; the outline, shadow and polarity measures on drawn crops; rows and free
  rows; the test-text filter; the wrap; compositing and legibility; a region, a shard's format and its byte-for-byte
  determinism; the 2× view.

## The shard set (v1)

`data/manifests/synth6_shards_v1_{vita,psp2x,psp}.json` pin 1,000,000 crops of `synth/synth6.py` at commit
`d912850` (every code file's sha256 and every asset's are in the recipes), seed 20261026, under
`$HERD/shared/artifacts/vocr6-synth6/shards/v1/`:

- **vita**: 500,000 crops in 50 shards (4.65 GB);
- **psp2x**: 250,000 in 25 (2.88 GB), model A's view of PSP games;
- **psp**: 250,000 in 25 (0.92 GB), native, for model B.

It rendered on 5 workers per view, with 0 render errors: vita in 2,066 s (about 48 crops/s per worker), psp2x in
559 s, psp in 387 s. Two earlier starts were stopped after their first shards: checks of those shards found the hval-text
overlap (the first start) and the row and legibility issues (the second). Both were deleted; nothing of them is
pinned. **Determinism:**
the first 1,000 crops of 6 pinned shards (vita 0 and 7, psp2x 0 and 17, psp 0 and 24), re-rendered in a fresh
process, are byte-identical (pixels, shapes, labels and meta).

| | vita | psp2x | psp |
|---|---:|---:|---:|
| titles / fonts drawn | 31 / 275 | 15 / 170 | 15 / 170 |
| matched / cross font | 31.7 / 68.3% | 13.6 / 86.4% | 13.7 / 86.3% |
| rows / scene_box / bare scene | 72.6 / 22.2 / 5.2% | 67.1 / 23.9 / 9.0% | 67.3 / 23.8 / 8.9% |
| crop: ink / cell / GRAD | 41.3 / 35.0 / 23.8% | 44.0 / 35.0 / 21.0% | 38.9 / 34.9 / 26.1% |
| inverted (dark text) | 29.1% | 17.0% | 16.9% |
| names | 1.6% | 1.3% | 1.3% |
| text: pool / hook / target / confusion | 40.1 / 24.0 / 24.5 / 9.8% | 40.6 / 24.4 / 23.8 / 10.0% | 40.5 / 24.3 / 24.0 / 10.0% |
| label length p10 / p50 / p90 | 5 / 14 / 25 | 5 / 13 / 24 | 5 / 13 / 24 |
| fill contrast p10 / p50 (luma levels) | 114 / 171 | 98 / 147 | 98 / 147 |
| Adrenaline filter: bilinear / nearest / bicubic | | 80.1 / 15.0 / 4.9% | |

The real train lines are 7 / 16 / 25 characters long (p10 / p50 / p90) on the Vita and 5 / 15 / 24 on the PSP.

**Coverage.** The vita set draws 5,542 of the 5,629 vocab classes. The 3,998 classes that occur fewer than 20 times
in the real train lines get a median of 33.5 exposures in it (p10 16, p90 123), 16 in each PSP set. synth4's
s4v5631 gives them a median of 76 in 1,000,000 crops, with a curriculum aimed at exactly that. synth6 spends its
crops on the real fonts and styles instead; its targets are a quarter of the lines.

**Balance.** Titles are drawn uniformly (Vita 3.23% each, PSP 6.67%), and the crops follow within ±20% for 29 of
31 Vita titles and 10-11 of 15 PSP titles. Four titles render far fewer crops: PCSG00751 0.98%, PCSG00468 1.92%,
NPJH50621 2.1% and NPJH50356 2.6%. Their rows are scattered (the first two), or their text has the lowest real
contrast (80 and 87 levels) on textured boxes, so many of their lines fail the legibility rule. The other PSP titles
take their share (ULJM05674 up to 9.1%). synth6 thus under-represents the two lowest-contrast PSP styles.

**Label audit** (`synth/synth3_audit.py`, as synth4's: PP-OCRv6 small reads 1,500 random crops per view, seed 1,
in screen polarity). The teacher reads the labels as follows:

| | teacher vs label CER | exact lines | matched font | cross font | rows / scene_box / bare scene |
|---|---:|---:|---:|---:|---|
| vita | **1.51%** | 84.9% | 1.20% | 1.64% | 1.54 / 1.18 / 2.40% |
| psp2x | **2.28%** | 79.0% | 1.26% | 2.44% | 2.52 / 1.56 / 2.12% |
| psp | **2.27%** | 79.4% | 1.41% | 2.41% | 2.50 / 1.30 / 2.91% |
| synth4 v1 (`docs/synth4.md`, same measure) | 4.56% | 76.7% | | | |
| synth3 v3 | 3.83% | 77.9% | | | |

On the 36 worst crops of the vita and psp2x audits (two sheets read by eye), most disagreements are teacher
errors: rare kanji in stylised faces, single characters, brackets read as [ ], ー against ―, small kana. One
systematic fault shows: **real text the page's boxes miss, under the drawn line.** These are old speech bubbles,
earlier NVL lines or unboxed screen text, on titles where text sits on the scene (NPJG00016, PCSG00648,
NPJH50356). They are 3-4 of the 1,500 crops in each of the two audits, about 0.2-0.3%. That is a lower bound: an
overlap shows on these sheets only when it misleads the teacher. In 37 random crops of the 8 titles with scattered
rows or no box, 1 is a clear case (NPJG00016, a speech-bubble game) and 1 is unclear.

Two candidate detectors were measured and rejected. A GRAD line in a box-free row is found on 0-100% of rows per
title, but on the sheets it is mostly box decoration (Starry Sky's constellations, Clock Zero's gears, Tenshin
Ranman's logo) or scenery, which real text also sits on. Masking the previous 8 pages' boxes as well leaves the
inked share unchanged (NPJG00016 0.74 against 0.83). v1 keeps these crops.

**Disclosures (v1).**

- 2 psp2x crops (`synth6_psp2x_00000.npz` row 5028, `_00003.npz` row 9636) show 有紀寧, the given name of a CLANNAD
  character. A rare-character target for 寧 drew it from the Jiten VN word list, which covers CLANNAD's script, and
  synth6's word carriers did not apply the exclusion markers when v1 rendered. The code now does, as synth4's do,
  and a test pins it. The two crops stay in v1 (2 in 1,000,000); a v2 render would not hold them.
- 2 crops hold the marker string ことみ inside ということみたい: other titles' ordinary text, no name.
- v1 was rendered at commit `d912850`, kept at tag `synth6-v1-render-d912850` (the squash merge drops branch
  commits), and v1 reproduces exactly from that tag (the determinism check above). The renderer on main differs from
  it only in the marker filter above. A shard rendered with main's code can therefore diverge from v1 once a
  carrier word for 有, 紀, 寧, 朋, 也, 芳 or 野 is drawn.

## The proof (pre-registered)

PROOF_PLACEHOLDER

## Limits

- **46 titles.** synth6 teaches these titles' looks (and 142 faces' glyphs) with any text. It does not show a box
  style no train title has. The cross mode and the unseen-game dev score are the guard against overfitting to them.
- **Styles are measured, not hand-checked per title.** The halo, shadow and box detectors are simple. Eye checks on
  the compare sheets agree with them, but a measured opacity can be off where a box is textured, or where the text
  sits on the scene.
- **The glyph check is a filter, not a proof.** It catches characters read as another character by two independent
  measures. A wrong glyph that neither the recognizer nor the template match can tell from the right one would
  pass. Stylised faces lose some complex kanji to false positives.
- **No furigana, ruby, page cursors, typewriter-partial lines or vertical text** (train2's finder augmentation adds
  cursors and neighbour rows). Names are 2% of the crops.
- **Real text under a drawn line**, where a page shows text its boxes miss (about 0.2-0.3% of PSP crops, more on
  bubble and NVL titles). A v2 could draw such titles only in `scene_box` (its own box hides what lies below), or
  mask text found by difference with the neighbouring pages.
- **Two low-contrast PSP styles are under-represented** (NPJH50621, NPJH50356: 2.1-2.6% of PSP crops against 6.7%):
  the legibility rule rejects many of their lines.
- **The PSP-native view is rendered but not proven.** It is for model B (vocr6-psp).
- Nothing here ran on a console.

## Reproduce

```sh
A=$HERD/shared/artifacts/vocr6-synth6
# the font pack, extracted once (private; never committed)
mkdir -p $A/fontpack && zstd -dc $HERD/shared/private/hooked-b2/fonts/font-pack.tar.zst | tar -x -C $A/fontpack
# per-title styles, layout, train hook text, counts, the font inventory, the test and hval text guards
HEAVY_SLOTS=15 $HERD/heavy.sh synth/synth6.py assets --out $A/assets --workers 5
# the OCR check and the per-glyph check (VOCR5-L)
CK=$HERD/shared/artifacts/vocr5-train/runs/v5f_L_seed2/ckpt_step15000.pt
HEAVY_SLOTS=15 $HERD/heavy.sh synth/synth6.py font-check --ckpt $CK --out $A/assets/font_check.json
HEAVY_SLOTS=15 $HERD/heavy.sh synth/synth6.py glyph-check --ckpt $CK --out $A/assets/glyph_check.json
synth/synth6.py manifests --assets $A/assets          # data/manifests/synth6_fonts.json, synth6_styles.json
# eye checks
synth/synth6.py compare --title PCSG00620,PCSG00146 --n 3 --matched --out $A/work/cmp/cmp.png
synth/synth6.py preview --view psp2x --out $A/work/prev
# the render (two heavy slots), then the pins
HEAVY_SLOTS=15 $HERD/heavy.sh synth/synth6.py shards --view vita --out $A/shards/v1/vita --shards 50 --workers 5
HEAVY_SLOTS=15 $HERD/heavy.sh synth/synth6.py shards --view psp2x --out $A/shards/v1/psp2x --shards 25 --workers 5
HEAVY_SLOTS=15 $HERD/heavy.sh synth/synth6.py shards --view psp --out $A/shards/v1/psp --shards 25 --workers 5
for v in vita psp2x psp; do synth/synth6.py manifest --dir $A/shards/v1/$v --out data/manifests/synth6_shards_v1_$v.json; done
```

Training with it: `--src 's6v:W:$A/shards/v1/vita/synth6_vita_000??.npz:stream=1,pool=3,inv=0.2'` (Vita),
`s6x` with `psp2x/synth6_psp2x_000??.npz` (model A's PSP view) and `s6p` with `psp/synth6_psp_000??.npz` (model B).
The crops are native size in console polarity, and train2 resizes them to its input height.
