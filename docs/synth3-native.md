# synth3 version 4: native console frames

synth3 version 3 renders a native Vita or PSP dialogue region as a region: the capture chain (VNDB's
JPEG, the PSP's 16-bit dither, a PSP-on-Vita upscale) acts on the region alone, and its glyph and effect
statistics were calibrated against all training-side real crops, three quarters of which are PC
screenshots scaled down. Version 4 (`CFG["frames"]`, `data/manifests/synth3_v4_native_cfg.json`) renders
native regions inside a whole console frame:

1. The region is placed in a 960x544 (Vita) or 480x272 (PSP) frame. A training-side native screenshot of
   the console covers the rest of the frame. ADV boxes and bands go near the bottom; choices, menus and
   NVL pages near the centre.
2. The console's own scaling acts on the frame: a Vita game rendered below 960x544 and upscaled by the
   GPU, or a 4:3 picture stretched across the 16:9 screen.
3. A PSP frame is dithered to 16 bits. A PSP-on-Vita frame (`psp2x`) is the whole PSP frame upscaled to
   the Vita's screen: bilinear, nearest, 1.995x or scanlines, as in version 3.
4. The frame is captured: clean, or VNDB's chain as measured on training-side native frames below. The
   JPEG grid is aligned to the frame's origin.
5. The region's window is cut back out. The ink, fill and furigana masks and the line layout follow every
   geometric step, and the crop path is version 3's: the gt crop, or the console's GRAD finder box.

Native lines also get version 4's style settings for native platforms: fewer outlines and shadows behind
opaque boxes, a little extra letter spacing on some lines, fewer furigana. With `CFG["frames"] = None` the code renders
version 3 byte for byte. Re-rendering the first 400 crops of pinned v3 shards 0 and 57 with this code
gave identical pixels, shapes, labels and metadata (`$A/native-frames/checks/v3_identity.py`).

## Measurements (training-side data only)

`synth/native_calib.py` measured these on data no development or test set shares. The full record is
`eval/synth3_native_calibration_20261013.json`.

- **Frames**: 1,678 native VNDB screenshots of training-side VNs (1,355 PSP, 323 Vita). These are the
  frames of realdata's and teacher's pools that `realplus/native.py` selects for the native pool. None is
  an eval, linefind, devnative, teacher_val, CLANNAD or owner-game frame.
- **Crops**: teacher_train and real_train native crops, with reserved and development VNs dropped.

### VNDB's JPEG

Every native frame is a JPEG whose luma table is exactly IJG quality 90: 1,354 of 1,355 PSP frames and
317 of 323 Vita frames. The others fit Q95 / Q94. Chroma subsampling splits by console:

| frames | Q90 table | 4:2:0 | 4:4:4 |
|---|---:|---:|---:|
| PSP | 1,354 / 1,355 | 220 (16%) | 1,135 (84%) |
| Vita | 317 / 323 | 304 (94%) | 19 (6%) |

Version 3 used 4:2:0 for 85-100% of its PSP JPEGs. Version 4's chains follow the table:

| platform | clean | single Q90 | console JPEG, then Q90 |
|---|---|---|---|
| PSP | 25% | 63%, 4:4:4 | 12%, 4:2:0 |
| Vita, PSP on Vita | 25% | | 75%: the console's JPEG at a quality drawn from 82-94, then Q90, 4:2:0 in 94% |

The clean share is kept for the console's own framebuffer, which is what the deployed recognizer reads.

### Console scaling on the frame

The frame-level test looks for the periodic second-difference energy that an upscale by 4/3, 3/2 or 2
leaves, and for the column-pair pattern of a nearest-neighbour 2x upscale.

- No native frame shows a nearest 2x upscale.
- 11 of 323 Vita frames (3.4%) and 3 of 1,355 PSP frames show a strong 4/3 peak: a Vita game rendered
  at 720 px width and upscaled.
- No frame shows a 3/2 peak.

Version 4 renders 3% of Vita frames below native resolution (0.75, 2/3 or 5/6, area down then bilinear
up) and stretches 3% of frames horizontally by 1.12-1.33. The stretch share is set by the glyph-shape
measurement below: about 3% of real kanji boxes are wider than 1.15.

### Glyph shape and spacing

The measure is ink width per full-width character over ink height, on lines of five or more full-width
characters, measured the same way for every source (`synth3_stats.px_ink`).

| crops | PSP p10 / p50 / p90 | PSP < 0.80 / > 1.15 | Vita p10 / p50 / p90 | Vita < 0.80 / > 1.15 |
|---|---|---|---|---|
| real native (training side) | 0.69 / 1.02 / 1.19 | 21% / 17% | 0.78 / 1.03 / 1.20 | 12% / 18% |
| synth3 v3 native renders | 0.60 / 0.84 / 1.10 | 43% / 5% | 0.63 / 0.92 / 1.12 | 35% / 7% |
| v3, plain text only (no outline, shadow, furigana) | 0.74 / 0.97 / 1.13 | | 0.75 / 1.02 / 1.14 | |
| version 4, c3 (effects dropped everywhere; superseded) | 0.68 / 0.94 / 1.18 | 25% / 14% | 0.71 / 0.99 / 1.20 | 23% / 18% |
| version 4, c4 (the render's settings) | 0.65 / 0.90 / 1.17 | 32% / 12% | 0.67 / 0.98 / 1.20 | 27% / 16% |

MeikiOCR's character boxes on the real crops show what causes the gap. Real kanji are square: w/h
p10 / p50 / p90 0.88 / 1.00 / 1.09 on PSP and 0.92 / 1.00 / 1.10 on Vita. Their pitch is one kanji
height: p50 1.00 on PSP and 1.03 on Vita, p90 1.13. So real native text is neither stretched (3% of kanji
boxes are over 1.15 wide) nor widely spaced.

Version 3's plain text matches the real crops. Its outlined and shadowed lines do not: outline alone
gives a median of 0.84, outline plus shadow 0.80, and furigana 0.58-0.70. About 78% of its native PSP
lines carry an outline, a shadow or both, against 14% plain.

Version 4 therefore drops most sampled outlines and shadows of native lines, but only where plain text
stays legible:

- An outline is kept on 8% of PSP and 10% of Vita lines, a shadow on 15%.
- The drop happens only when the box behind the text has alpha 0.6 or more. Text without a box keeps
  its effects.
- Furigana goes from 15-20% of lines to 5%, and some lines get 1-3 px of extra pitch.

The first full render (12:08Z, c3's settings) dropped effects behind any box. Its label audit failed:

| crops | teacher-vs-label CER (PP-OCRv6 small, 1,500 crops) |
|---|---|
| version 3, native strata | Vita 2.5%, PSP 5.2%, PSP-on-Vita 6.0%; gt crops 3.9% |
| version 4 render of 12:08Z (c3) | 6.0% overall, gt crops 7.5%, text without a box 14% |
| version 4, c4 (the render's settings) | 4.2% overall, gt crops 4.6%, Vita 2.7%, PSP 5.8%, PSP-on-Vita 5.3% |

The 12:08Z render's worst crops were plain text on translucent boxes over busy scenes, unreadable even by
eye. It was discarded before any training (protocol amendment 1), and the render was redone with the
legibility gate.

Four calibration rounds (c1-c4, seed 3001, 5,000 crops each) moved the median from 0.84 to 0.90 on PSP
and from 0.92 to 0.98 on Vita. The rest of the gap, mostly a narrow tail, comes from three groups:
outlined text on translucent boxes, text without a box, and NVL narration with close neighbour lines.

### Sharpness

p90 gradient over the crop's contrast, inside the ink box:

| crops | PSP p10 / p50 / p90 | Vita p10 / p50 / p90 |
|---|---|---|
| real native | 0.43 / 0.50 / 0.55 | 0.43 / 0.50 / 0.54 |
| v3 | 0.42 / 0.51 / 0.68 | 0.39 / 0.49 / 0.53 |
| version 4 (c4) | 0.41 / 0.50 / 0.67 | 0.38 / 0.48 / 0.53 |

The medians match. The crisp PSP tail (p90 0.66) comes from clean captures of 1-bit and pixel fonts. The
soft Vita tail comes from the double JPEG with 4:2:0 and from blur. Version 4 does not tune either.

## Data: s3e

Version 4 renders native platforms only, in version 3's proportions (Vita 24, PSP 24, PSP-on-Vita 5) and
with version 3's native-crop length histogram as the length target. Its output is 100 shards of 5,500
crops (seed 20261012, `$A/native-frames/shards/native_v4`). `synth/synth3_native_merge.py` builds s3e:

- Shard i is version 3's pinned shard i, crop for crop.
- Each vita, psp or psp2x crop (528,418 of 1,000,000) is replaced, in place and in order, by the next
  crop of version 4's native shard i.
- Version 3's area-scaled PC crops (471,582) are kept byte for byte.

So the two arms of the study stream the same structure and differ only in how their native crops were
rendered. `data/manifests/synth3_shards_e.json` pins both sets: every shard's sha256, the recipes, and
the per-shard replacement counts.

## Study (`eval/synth3_native_protocol.json`)

The protocol was frozen and pushed before E_s3nat trained or anything was scored. It keeps study 1's
recipe, architecture, steps, seeds, export, views, metric, bootstrap and B-better-than-A rule:

- **Arms.** D_s3v3 is study 1b's runs, reused; it trained on this machine at 2b132aa. E_s3nat trains
  from a worktree of that commit with s3e in place of s3v3.
- **Decision.** devnative v3 decides: strict and tv_native, ink and GRAD crops pooled, with VN-cluster
  intervals. The rows of Memories Off After Rain (v1331), a series title, are removed first, which leaves
  986 crops of 543 lines from 43 VNs (`eval/synth3_native_dev.py`).
- **Reported only.** teacher_val and each devnative file.

RESULTS_PLACEHOLDER

## Hygiene

- Version 4 uses every exclusion version 3 does. It also uses
  `data/manifests/synth3_exclude_owner_games.json`, the owner's test-only Memories Off series (45 VNs,
  `docs/owner-games.md`, owner 2026-10-13 10:51Z). Against version 3's render this drops 34 PSP and 15 PC
  scene screenshots and 954 more name rows (1,406 against 452).
- A first version-4 render (10:17Z) used the 09:25Z owner rule (STEINS;GATE and planetarian) instead. It
  was stopped at 57% when that rule was superseded and is not used anywhere.
- s3e keeps version 3's scaled crops. Those were rendered before the rule, and their scenes can include
  20 PC screenshots of the series (`docs/owner-games.md`).
- The calibration used training-side frames and crops only. It was repeated without the series once
  the series was reserved: the 4:2:0 share of Vita frames is 92% (the patch uses the first
  measurement, 94%), PSP 4:4:4 84%, glyph aspect within 0.003. No setting was tuned on devnative,
  teacher_val or a protected set.

## Reproduce

```sh
A=$HERD/shared/artifacts; V3=$A/synth3-v3; NF=$A/native-frames
.venv-ci/bin/python synth/native_calib.py jpeg   --shots $NF/pool/reads/shots.jsonl --out jpeg.json
.venv-ci/bin/python synth/native_calib.py frames --shots $NF/pool/reads/shots.jsonl --out frames.json
.venv-ci/bin/python synth/native_calib.py crops --real $A/teacher/teacher_train.npz+$A/realdata/real_train.npz \
    --synth "$V3/shards/train_v3/synth3_0000[0-1].npz" --exclude $A/synth3/mining/exclude_dev_vns.txt --limit 12000 --out crops.json
HEAVY_SLOTS=12 $HERD/heavy.sh timeout 14400 .venv-ci/bin/python synth/synth3.py shards --out $NF/shards/native_v4 \
    --split train --shards 100 --per-shard 5500 --workers 5 --seed 20261012 --cfg @data/manifests/synth3_v4_native_cfg.json \
    --scene-hashes $V3/exclude/scene_hashes.json --targets $V3/text/targets_v3/targets.json
HEAVY_SLOTS=12 $HERD/heavy.sh timeout 7200 .venv-ci/bin/python synth/synth3_native_merge.py merge \
    --v3 $V3/shards/train_v3 --native $NF/shards/native_v4 --out $NF/shards/e_v4
.venv-ci/bin/python synth/synth3.py manifest --dir $NF/shards/e_v4 --dir $NF/shards/native_v4 --out data/manifests/synth3_shards_e.json
TRAIN_REPO=<worktree at 2b132aa> HEAVY_SLOTS=10 $HERD/heavy.sh timeout 36000 bash eval/synth3_native_run.sh 1   # and 2
HEAVY_SLOTS=12 $HERD/heavy.sh timeout 7200 bash eval/synth3_native_score.sh
```
