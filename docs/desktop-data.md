# Desktop data: resolution-diverse training pages and a real PC test set

Issue #169's data side. vocr-desktop (#169) builds a desktop text detector for apps that read a game window on a PC
(GameSentenceMiner and others). Its training data were almost only native Vita and PSP pages (the hooked pools v2.1:
960 x 544 and 480 x 272), and its only PC set held 101 frames of 51 VNs, read for report only. Desktop users run
games from 1280 x 720 to 4K, fullscreen or in a window, with the game's or the emulator's own scaling, and some share
JPEG or downscaled captures. The owner (2026-11-01 06:49Z): "cause desktops can have different resolutions, you do
whatever you want and think is best for this."

This adds three frozen training sets and a frozen real PC test set. Every number below was measured on the host;
nothing here trains or scores a model.

**Summary**

- **1a, the hooked pages as desktop captures:** 20,000 of the hooked pools v2.1's 74,091 train pages (all 136
  titles that have train pages; 15,497 Vita and 4,503 PSP pages), each put through one draw of a desktop capture
  model: 2,764 distinct capture sizes from 404 x 303 (an 800 x 600 window, downscaled by half) to 3840 x 2160,
  letterboxed, pillarboxed, stretched or windowed, six resamplers, DPI stretches, downscaled and JPEG captures. Their labels are vocr-desktop's own (its
  page shards' lines and ignore areas), mapped exactly. Only split train is read: hval, guard and the test titles
  stay out exactly as in pools v2.1.
- **1b, realplus at its own size:** 49,267 of realplus v3's PC screenshots (14,457 VNs) at the size VNDB has them
  (640 x 400 to 3840 x 2160), with realplus's 101,317 labelled line boxes and 482,260 ignore areas for every region
  its labels do not cover (lines realplus did not keep, MeikiOCR's boxes at thresholds 0.5 and 0.2).
- **1c, synthetic desktop frames:** 10,000 frames drawn at 13 desktop resolutions (1280 x 720 to 3840 x 2160, and
  800 x 600 to 1280 x 960 for 4:3), text rasterised at its own size with synth3's non-pixel fonts (all
  open-licensed), main-text em 24-64 px, thin weights, outlines, shadows, translucent and opaque boxes, box-less
  text, choices and UI labels; 30% of them drawn at a game resolution and put through the capture model.
- **The PC test set:** 942 dialogue lines (plus 217 speaker names) from 142 VNs that are in no pool, on 504 VNDB PC
  screenshots at their own size (640 x 390 to 2560 x 1440, 21 named engines), labelled by MeikiOCR and PP-OCRv6
  small's readings and two Claude passes, 214 of them checked line by line by me (213 right; the wrong one
  corrected). Frozen with its manifest (sha256 `20576c40…`) at 09:23Z, before vocr-desktop's decision; its VNs are
  refused by `realplus/shots.reserved_vns`. The target was 1,000 lines: VNDB holds no more such screenshots (section 2).
- **The owner's desktop captures:** `desktopdata/owner.py` imports them when they come (the hooked bucket's format,
  platform `pc`), into the same pools and into native-size detector pages.

Everything bulky stays local: the training pages under `$HERD/shared/artifacts/desktop-data/pages/`, the test set's
screenshots and review files under `$HERD/shared/artifacts/desktop-data/pctest/`. The repository holds the code
(`desktopdata/`), the manifests (`data/manifests/desktop_data/`, `eval/manifests/desktop-pc-test-v1.jsonl`), the
counts and this document.

## 1. Training pages

### The capture model (`desktopdata/capture.py`)

A capture is a game frame shown on a desktop and captured there. One draw picks:

| step | choices (weights in `capture.MIX`, recorded in every manifest) |
|---|---|
| setup | fullscreen 0.80 (the frame fitted with its aspect: bars where the aspect differs), an emulator window at a multiple of the frame 0.10 (Vita x1.25-x3, PSP x2-x5, as Vita3K and PPSSPP size their windows), a user-sized window 1024-2560 px wide 0.06, the frame stretched to the screen 0.04 |
| fullscreen screens | 16:9: 1280 x 720, 1366 x 768, 1600 x 900, 1920 x 1080 (0.22), 2560 x 1440 (0.12), 3840 x 2160 (0.07); 16:10: 1280 x 800 to 2560 x 1600 (letterbox); 21:9: 2560 x 1080, 3440 x 1440 (pillarbox); 4:3: 800 x 600 to 1600 x 1200 (letterbox) |
| resampler | bilinear 0.33, bicubic 0.15, Lanczos 0.14, nearest 0.12, sharp bilinear 0.14 (nearest to the largest integer multiple, then bilinear), bilinear with a sharpening pass 0.12 (an FSR / CAS-like unsharp mask); on a downscale, INTER_AREA half the time |
| DPI | a window at 125-175% display scaling, 30% of the windows: drawn at the logical size, then stretched bilinear to the physical size, as Windows does for a program that is not DPI-aware |
| bars | black 0.85, dark grey (6-40) 0.15 |
| capture | downscaled 0.10 (INTER_AREA, x0.5-0.8: a shared or thumbnail-sized capture), JPEG 0.25 (quality 70-95, 4:2:0, 4:4:4 one time in five) |

Every geometric step is an affine map per axis, so a box maps exactly: its edges go through the composed map (start
edges rounded down, end edges up, eval's convention). `tests/test_desktop_data.py` checks it against the ink of
rendered boxes for every resampler and setup.

The realized mix:

<!-- generated:mix -->
| capture (frames) | 1a hooked desktop | 1c synthetic desktop |
|---|---:|---:|
| setup: fullscreen | 15,966 | 2,727 |
| setup: emulator window (multiple of the frame) | 1,980 | 0 |
| setup: user-sized window | 1,222 | 178 |
| setup: stretch | 832 | 118 |
| fullscreen on 16:9 | 10,979 | 1,841 |
| fullscreen on 16:10 | 2,325 | 384 |
| fullscreen on 21:9 | 1,146 | 207 |
| fullscreen on 4:3 | 1,516 | 295 |
| resampler: bilinear | 6,677 | 943 |
| resampler: bicubic | 3,036 | 442 |
| resampler: lanczos | 2,733 | 463 |
| resampler: nearest | 2,499 | 398 |
| resampler: sharp_bilinear | 2,685 | 437 |
| resampler: bilinear_sharpen | 2,370 | 340 |
| DPI stretch (125-175%) | 992 | 55 |
| capture downscaled | 1,945 | 303 |
| capture saved as JPEG | 5,012 | 768 |
| captured (of all frames) | 20,000 of 20,000 | 3,023 of 10,000 |
<!-- /generated:mix -->

### 1a: the hooked pages as desktop captures (`desktopdata/hooked.py`)

- **Pages.** 20,000 of the 74,091 train pages of vocr-desktop's hooked page shards (`data/manifests/
  desktop_pages_v21.json` on feat/vocr-desktop, 1b4e074: synth/textspot6_pages.py on page pool v2.1 `bbadfcbf`),
  spread over the titles as vocr-desktop's HookPool samples them (a title's weight its pages^0.5, never more than its
  pages: 8 to 426 pages a title), a title's pages drawn uniformly (seed 20261101).
- **Labels.** vocr-desktop's, as they are: positives (verified dialogue and name lines refined to ink; MeikiOCR's
  other text as class `notdlg`), ignore areas (ruby, dropped lines, renderer boxes no verified line holds, weak
  MeikiOCR boxes). Each shard's json sha256 is checked against that manifest before use.
- **Splits.** Only split `train` is read and every page is checked to be train: hval (3,010 pages), guard (3,780)
  and the test titles (PCSG00807, NPJH50882) stay out exactly as in pools v2.1.
- **Pixels.** Each page's colour PNG (sha256 checked) through one capture draw (seeded by title and page), then the
  integer luma the runtime uses. 584 pages have no positive line (all their text is an ignore area).
- **Format.** textspot6-pages shards (HookPool reads them unchanged): `pages/hooked_v1/train/<TID>_NNN`, at most
  200 pages a shard, kind `desktop` (so the trainers' PSP 2x step never applies), the page's platform in
  `meta.src_kind`, the capture's settings in `meta.capture`. 183 shards, 47 GB.

### 1b: realplus at its own size (`desktopdata/realplus.py`)

**What exists on disk.** realplus labelled its lines on each screenshot scaled to fit 960 x 544; the originals are
kept at VNDB's size: `$ART/realplus/shots/hires` (43,198 screenshots of v1-v2's selection) and
`$ART/reserve-policy/realplus/shots/hires` (20,572 of v3's reclaim tier, plus 295 PSP-size shots under `psp/`).
`$ART/realplus-v3/shots` is empty (v3 reused them). Every one of v3's 49,285 screenshots with a kept line is on disk.

- **Frames.** 49,267 of them at their own size: 18 left out (13 of VNs in textspot7's confirmation set, which
  vocr-desktop reads once at the end, and 5 of textspot2's eval-screenshot skip list). Also left out, finding none:
  the reservation's test and dev_sample VNs, #119's development units, the PC test set's VNs, eval/linefind's VNs.
- **Positives.** Each kept v3 line: its reads line's ink box (main text, no furigana) widened left and right to v3's
  final ink crop, scaled to native size and refined to ink there (kept only when it stays inside the box grown by
  max(3, 3 / s) px; all 101,317 refined). Class `name` for a speaker name, else `text` (no tag target: realplus does
  not tell dialogue from choices).
- **Ignore areas.** Every other line realplus's reads found (210,298 areas), every MeikiOCR box of the reads at
  threshold 0.5 (103,679) and of a fresh native pass at threshold 0.2 (167,972) that meets no positive, and the kept
  lines' furigana (311); an area already 90% inside another is not added twice. realplus kept about a third of the
  lines its reads found, so most text on these frames is an ignore area, never background.
- **Format.** #97's frames (FramePool reads them as source `realplus`): `pages/realplus_v1/desk_realplus_v1/
  rp_PP_NNN`, kind `pc`, at most 400 frames a shard. 124 shards, 27 GB.

### 1c: synthetic desktop frames (`desktopdata/synth.py`)

Each frame is a whole PC VN screen drawn at the screen's own resolution:

- **Scene:** synth/textspot_data.load_scene (textspot's clean scene pool: training-side VNDB screenshots,
  cover-cropped, their known text inpainted), synth2's synthetic background 8% of the time.
- **Dialogue:** an ADV box at the bottom 52% (at the top 3%), a full-screen NVL page 13%, a band 8%, text with no box
  18%, none 6%; boxes translucent (alpha 0.15-0.8, 65%) or opaque (0.8-0.97), rounded or square, some with a border;
  a name plate, a name line, an inline name or none.
- **Fonts:** synth3's pool without its pixel families (gothic 0.42, mincho 0.28, rounded 0.14, display 0.08,
  pen 0.08): 244 files of 97 families as synth3 loads them, every one licence class `open`
  (`data/manifests/synth3_fonts.json`; the two non-commercial families are pixel faces and never drawn here). A thin
  weight (100-300) where the family has one, 22% of the draws (7.7% of the dialogue elements got one).
- **Texts:** synth3's dialogue, narration, choice and name sources in its `personal` profile, as textspot's synthetic
  frames use them (`data/manifests/synth3_text_sources.json`, some of them non-commercial: README "What the released
  weights were trained on").
- **Sizes:** the main text's em is 2.8-5.0% of the screen height clipped to 24-64 px, or (15%) uniform in 24-64 px.
- **Effects:** outlines (a quarter of the boxed lines, 85% of the box-less ones), hard and soft shadows, coloured
  names, furigana (ignore areas), page cursors (background).
- **Other text:** choice bars (class `choice`), UI labels at the screen's edges, 14-28 px (class `ui`).
- **Text-free:** 7% of the frames (the scene, or the scene with an empty box).
- **A guard:** MeikiOCR's detector (threshold 0.3) on every finished frame; a box that meets no label and no ignore
  area becomes an ignore area (5,294 of them: scene text the inpainting missed).
- **Capture:** 70% native (the game runs at the screen's resolution; a fifth of those saved as JPEG q 75-95), 30%
  drawn at a game resolution (800 x 600 to 1920 x 1080) and put through the capture model.

Boxes are each instance's ink box from its own coverage (synth/textspot_data's rule). #97's frames (FramePool reads
them as source synth): `pages/synth_v1/desk_synth_v1/desk_synth_NNNN`, src `synth:desktop`, kind `pc`, 250 frames a
shard. 40 shards, 21 GB. Rendering the 10,000 took about 5 minutes in two heavy slots.

**Reproduction.** One shard of each set rebuilt from the committed code matched its manifest's sha256 byte for byte
(1a NPJG00016_000, 1b rp_00_000 and rp_00_030, 1c desk_synth_0007).

### Counts and text heights (`desktopdata/stats.py`, `data/manifests/desktop_data/stats_v1.json`)

The Vita / PSP pools are vocr-desktop's hooked page shards' train pages; "real half" is textspot2's MeikiOCR-labelled
real canvases (realplus and teacher screenshots fitted to 960 x 544), which vocr-desktop trains on today. Frames
per resolution class (by width):

<!-- generated:resolution -->
| resolution class (width) | hooked Vita (native) | hooked PSP (native) | real half (fit 960) | 1a hooked desktop | 1b realplus native | 1c synthetic desktop | pc-test |
|---|---:|---:|---:|---:|---:|---:|---:|
| <=640 | 0 | 21,142 | 2,690 | 60 | 13,672 | 18 | 92 |
| 641-1023 | 52,949 | 0 | 46,024 | 935 | 22,003 | 351 | 62 |
| 1024-1279 | 0 | 0 | 0 | 1,320 | 4,348 | 518 | 22 |
| 1280-1599 | 0 | 0 | 0 | 4,090 | 7,828 | 2,572 | 157 |
| 1600-1919 | 0 | 0 | 0 | 1,742 | 264 | 619 | 2 |
| 1920-2559 | 0 | 0 | 0 | 6,343 | 1,122 | 3,724 | 157 |
| 2560-3839 | 0 | 0 | 0 | 4,264 | 21 | 1,703 | 12 |
| >=3840 | 0 | 0 | 0 | 1,246 | 9 | 495 | 0 |
| frames | 52,949 | 21,142 | 48,714 | 20,000 | 49,267 | 10,000 | 504 |
| distinct sizes | 1 | 1 | 136 | 2764 | 451 | 470 | 25 |
<!-- /generated:resolution -->

Dialogue line heights (the ink box of each horizontal positive of class dialogue or text) at the frame's own size
and at vocr-desktop's default input policy fit:960 (the longer side down to 960 px, never up):

<!-- generated:heights -->
| set | lines | native p5 / p25 / p50 / p75 / p95 (px) | at fit:960 p5 / p25 / p50 / p75 / p95 (px) |
|---|---:|---|---|
| hooked Vita (native) | 84,359 | 20 / 22 / 24 / 26 / 32 | 20 / 22 / 24 / 26 / 32 |
| hooked PSP (native) | 32,463 | 13 / 14 / 15 / 17 / 21 | 13 / 14 / 15 / 17 / 21 |
| real half (fit 960) | 284,180 | 8 / 12 / 16 / 21 / 29 | 8 / 12 / 16 / 21 / 29 |
| 1a hooked desktop | 31,340 | 26 / 38 / 50 / 64 / 96 | 18.7 / 22.5 / 25.5 / 29.2 / 36 |
| 1b realplus native | 93,976 | 15 / 18 / 22 / 25 / 35 | 14 / 17.5 / 20.2 / 23 / 30 |
| 1c synthetic desktop | 15,826 | 23 / 31 / 40 / 51 / 70 | 15 / 18.8 / 22 / 25.3 / 34.7 |
| pc-test | 942 | 16 / 22 / 25 / 32 / 46 | 11.5 / 15 / 18.8 / 22.5 / 27.3 |
<!-- /generated:heights -->

<!-- generated:hist -->
| dialogue line height at native size (px) | hooked Vita (native) | hooked PSP (native) | real half (fit 960) | 1a hooked desktop | 1b realplus native | 1c synthetic desktop | pc-test |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0-8 | 0.2% | 0.1% | 3.4% | 0.1% | 0.0% | 0.0% | 0.0% |
| 8-12 | 0.1% | 0.5% | 19.5% | 0.1% | 0.8% | 0.1% | 0.0% |
| 12-16 | 0.2% | 54.0% | 21.9% | 0.3% | 8.1% | 0.3% | 4.4% |
| 16-20 | 3.8% | 36.2% | 22.5% | 0.8% | 21.9% | 0.8% | 8.6% |
| 20-24 | 35.8% | 7.2% | 18.1% | 2.1% | 36.2% | 4.8% | 24.9% |
| 24-32 | 54.3% | 2.0% | 11.0% | 8.8% | 25.6% | 19.5% | 35.2% |
| 32-40 | 5.1% | 0.0% | 2.1% | 15.2% | 5.0% | 22.7% | 16.8% |
| 40-48 | 0.5% | 0.0% | 0.7% | 17.3% | 1.8% | 19.9% | 5.7% |
| 48-64 | 0.0% | 0.0% | 0.5% | 28.8% | 0.5% | 23.9% | 3.9% |
| 64-96 | 0.0% | 0.0% | 0.3% | 20.9% | 0.1% | 6.7% | 0.3% |
| 96-128 | 0.0% | 0.0% | 0.0% | 4.6% | 0.0% | 1.1% | 0.1% |
| 128-inf | 0.0% | 0.0% | 0.0% | 0.9% | 0.0% | 0.2% | 0.0% |
<!-- /generated:hist -->

- At their own size, 1a's and 1c's lines are 2-3x the Vita pool's (median 50 and 40 px against 24); at fit:960 they
  come back to the Vita pool's range (median 25.5 and 22 px), with the resampling, JPEG and DPI traces of the
  capture. A detector that reads at `native` or `long:N` sizes sees the full range.
- 1b is mostly 640 x 480 to 1280 x 720 (VNDB's PC screenshots are mostly older games); its median line is 22 px, as
  tall as a Vita line, and about 9% of its lines are under 16 px.
- The PC test set's lines (median 25 px at their own size) are smaller at fit:960 than the Vita pool's lines (median
  18.8 px against 24), between the PSP pool (15) and the Vita pool, as 1b's are (20.2); 1a's and 1c's stay near the
  Vita pool's (25.5, 22). The detector's input-size policy and its training scale range decide how it sees them.

### Reading them in vocr-desktop's trainer (`train/desktop_det.py`)

- 1a: `HookPool(hooked_desktop_v1.json, $A/desktop-data/pages/hooked_v1, ("train",))` reads it unchanged (manifest
  kind textspot6-pages); a second hook source in `Sources` (or a share of the `hook` source) mixes it with the native
  pages.
- 1b and 1c: pass their manifests to `--manifest` and their directories to `--shards`
  (`.../realplus_v1/desk_realplus_v1`, `.../synth_v1/desk_synth_v1`); FramePool files 1b under its `realplus`
  source and 1c under `synth`. Neither manifest is a textspot data pin, so the real half's skip lists are not
  required for them (1b applied them itself).
- The windows of `make_window_ms` cut a region and scale it by a log-uniform factor in `--scale`; for 4K frames a
  wider low end (or a first downscale of the canvas) keeps the windows from always seeing a small part of the screen.
- Checked: feat/vocr-desktop's `train/desktop_det.py` (1b4e074, read-only worktree) loads the three frozen manifests
  as above (HookPool: 136 titles, 20,000 pages; FramePool: synth 10,000, realplus 49,267) and draws batches from
  them with `make_batch_ms` (mix hook 0.5 / synth 0.25 / real 0.25, scale 0.42-1.9: 0.08 s per batch of 16).

## 2. The real PC test set (`desktopdata/pctest.py`, `eval/manifests/desktop-pc-test-v1.jsonl`)

**What was frozen** (2026-11-01 09:23Z, before vocr-desktop's decision; `$A/desktop-data/frozen_v1/FROZEN.sha256`):

- **942 dialogue lines** (16,838 characters) plus 217 speaker names, from **142 VNs** (143 with the names) and 504
  screenshots: 518 clean dialogue units holding 886 of the lines for the page path.
- Resolutions: 25 screenshot sizes from 640 x 390 to 2560 x 1440: 1920-2559 px wide 317 lines (44 VNs), 1280-1919
  px 241 (49), 801-1279 px 83 (9), 800 px or less 266 (37), 2560 px 35 (3). Engines: 21 named in VNDB (Unity 195 lines,
  KiriKiri 163, NScripter 75, TyranoScript 48, Artemis 32, Adobe AIR 22, LiveMaker 20, Godot 17, ...); 296 lines come
  from releases without an engine on VNDB. Releases from 1998 to 2026, three fifths of the lines from 2020 on. 689
  lines bright on dark, 253 dark on light.
- `eval/manifests/desktop-pc-test-v1.jsonl` (sha256 `20576c40…`): one row per line: VNDB ids (VN, release, screenshot),
  URL, file and pixel sha256, size, the line's ink box (`box`, x y w h, no furigana), its box with furigana
  (`text_box`), the label (`text`, eval's conventions), `name`, how the engines read it (`how`: consensus / single /
  changed, `support`), the review record (`pass1`, `pass2`, `hand_checked`, `looks`), `reviewer_kind` assistant,
  and the flags below. `eval/desktop_pc_test_v1.json` is its summary.
- `frames_pctest_v1.pkl` (sha256 `aa3785d2…`, local): the same set in vocr-desktop's `eval/desktop.py` frames format
  (src kind `pctest`, the screenshot's path and sha256, units with region, line boxes and even cells; groups `pc`
  and `console-hd`); its `load_frames`, `check_sha` and `load_rgb` read it unchanged (checked on its branch's code).

**Candidates.** The rules were fixed before any screenshot was downloaded (`pctest.py` docstring, `candidates.json`),
on VNDB's dump of 2026-10-31:

- a Japanese-original VN in **no pool**: every VN named by a training pool (realplus v1-v3 and its reclaim tier, by
  every screenshot selected or on disk; teacher; realdata; the native pool; the owner's pools and games; the hooked
  pools v1 and v2.1), a development or test set (devnative, eval's manifests and records, eval/linefind, #119's
  units, textspot7's confirmation set, the reservation's test and dev_sample), a scene list (realdata's and
  data/vndb's shots, synth2's backgrounds) or a data manifest: 16,416 VNs;
- no VNDB release shared with a pool or reserved VN (amendment 2: unless the screenshot's own release holds none;
  flagged `release_partner_of`, 9 VNs, 55 lines);
- a family relation to a pool VN is allowed and flagged (`family_of`, amendment 1: 21 VNs, 152 lines);
- the screenshot's VNDB flags sexual < 0.3, violence < 0.5; its release a PC one (win, mac, lin) listing Japanese,
  or (amendment 2) an HD console release (Switch, PS4, PS5, Xbox, at least 1280 x 720: group `console-hd`, 4 VNs, 31
  lines; desktop users capture these from emulators); at least 640 x 360, aspect 1.25-2.4.

That left 1,618 screenshots of 317 VNs (all downloaded, sha256 and pixel sha256 recorded, no pixel duplicate among
them or with vocr-desktop's confirmation frames). The rest of VNDB's screenshots: 127,511 of pool VNs, 59,721 of
non-Japanese VNs, 5,807 over the content flags, 2,254 of other platforms, 2,206 of releases without Japanese, 726 of
other sizes and 183 on a release shared with a pool VN.

- **Amendment 1** (07:21Z, after the reads, before any review): family-related VNs kept and flagged, the per-VN
  caps raised (the first rules gave 1,157 candidate lines of 136 VNs).
- **Amendment 2** (07:48-07:51Z, while pass 1 ran, before any label of a line it adds): the release-partner and
  console-HD groups above, caps 12 screenshots and 40 lines per VN, and a line of 12 px or more at its own size kept
  although under 10 px at the 960 x 544 scale (pass 1's first batches skipped a third of the lines as UI).

**Reads and the review set.** Each screenshot at its own size: MeikiOCR's detector (threshold 0.5) and recognizer,
teacher/geometry's rows, ink boxes and furigana, and devnative's four readings per line (MeikiOCR on its box and on
a wider view, PP-OCRv6 small on the view and on the crop; pinned engine files, no network). A screenshot counts
when one of its lines reads at least two kana (a Chinese, English or Korean screen of a multi-language release does
not: 976 screenshots); a line when it is not at the screen's edge, at most 72 px at the fit scale and at least 10 px
there or 12 px at its own size, and Japanese in some reading. 1,885 lines of 626 screenshots and 171 VNs.

**Labels** (devnative's development rule, `devnative/decide.decide_line` mode dev; `reviewer_kind` assistant):

- Pass 1: 20 batches of contact sheets (<= 1280 px, devnative's format with large text scaled to 30 px and small text
  zoomed), read by ten Claude helper sessions (`desktop-data-review-1`..`-10`): pick a candidate, type the text,
  unsure, or skip as ui / cut / bad. 515 lines were UI, 151 not one readable Japanese line, 51 cut.
- Pass 2: the 216 labels that both engine families do not read (single 175, changed 41), shuffled and unattributed
  with pass 1's text as one more candidate, read by two fresh sessions (`-11`, `-12`); kept when they read the same
  key: 212 of 216 (1 disagreed, 3 unsure).
- Kept: 1,159 lines: 946 read by both engine families (consensus), 173 by one, 40 by neither (a reviewer's text, both
  passes agreeing; one of them the corrected label below).
- **The hand check** (me, line by line): 214 kept lines (one or two per VN, then at random; 200 drawn when 1,082
  lines were kept, 14 more from the 77 kept after), shown with no candidate, typed from the image (zoomed where a
  character was small; dot runs and repeats counted by column projection). 208 matched the label. Of the other six:
  four were my misreadings (two closing 」 that are not on screen, a page cursor read as く, two single-dot cells
  read as one …: the zoom confirmed the label each time), one stayed unsure (a 一 too faint to see; the label holds
  what is visible) and **one label was wrong** (薫 under the ruby まゆずみ is 黛; both engines and both passes read
  薫): corrected. So 213 of the 214 checked labels were right (99.5%, Wilson 95% 97.4-99.9%). The third looks are in
  `review/resolutions.jsonl`.

**Units.** Each screenshot's kept dialogue lines (names left out) are grouped as a player draws a region (eval's
vertical-gap rule), the union plus eval's margin scaled to the frame (12 px at 960 x 544, 24 px at 1080p). A unit is
clean when no other text the reads found (a line not kept, a name, a raw detector box) has its centre in the region:
518 of 552 units, holding 886 dialogue lines. Only clean units go into the frames file.

**Keeping it out of training.** The frozen manifest is the set's reservation: `realplus/shots.desktop_pc_test_vns()`
reads its 143 VNs, and `realplus/shots.reserved_vns()` (the reservation every VNDB pool builder applies: realplus,
its reclaim tier, the native pool) refuses them with the reason `desktop-pc-test`. `eval/manifests/reserved.json` and
`synth/common.py` are pinned by sha256 in frozen protocols (textspot, textspot2, textspot3, textspot6 among them), so
they do not list these VNs; `synth/common.is_reserved_vn` does not know them, and a builder that uses it alone (synth
scene pools) must also check `desktop_pc_test_vns()` until the next reservation carve adds them with re-pinning
amendments. The set's VNs are a 0.4% share of the real lines (942 of about 219,000 with realplus v3's 101,350 and
the hooked pools' 116,987), inside AGENTS.md's 10% budget; none of them was in any pool before.

**Not met: the 1,000-line target.** 942 dialogue lines, 58 short. VNDB's eligible space is exhausted: every Japanese
PC screenshot of a VN in no pool was downloaded and read, and the selection took every Japanese screen. The owner's
desktop captures (below) or later VNDB uploads can extend it as a v2.

**Erratum** (`eval/desktop_pc_test_v1_errata.json`). At 09:53Z, after the freeze, a pass-2 reviewer called four
v21170 lines cut: their frame misses the glyph tops. On the image, the ink refinement took only the lower half of
that game's two-tone glyph fill. All lines whose ink box is under 0.6 of MeikiOCR's detector box height (names left
out) are these 7 lines of v21170. Their labels are right and their units' regions hold the whole line, so dialogue
CER is unchanged; their reference boxes hold the lower half of the glyphs, so for line recall a detector box that
cuts their tops still counts as whole. The erratum gives corrected boxes (their rows from the detector box and the ink box). The reviewer's
four post-freeze decisions are kept apart (`review/pass2/b03/decisions_postfreeze.jsonl`), so `final` still
rebuilds v1 byte for byte.

**How to read it.** As vocr-desktop reads its other sets (`eval/desktop.py detect` / `read` on the frames file), once,
after its decision is recorded, under a committed protocol: never for screening, tuning or a threshold. Report `pc`
and `console-hd` apart, and the set with and without the flagged VNs.

## The owner's desktop captures (`desktopdata/owner.py`)

When the owner sends desktop captures in the hooked bucket's format (`<platform>/<TITLE_ID> <title>/` with
pages.tar.zst, pages.jsonl.zst, lines.jsonl.zst and metadata.json; platform `pc`, `resolution` the capture's size):

```sh
PY=.venv/bin/python; W=$HERD/shared/artifacts/desktop-data/owner; B=<the bucket's local copy>
$PY desktopdata/owner.py import --root $B --work $W --vndb-map <map>      # import_hooked: extract, split, lines, pages
HEAVY_SLOTS=12 $HERD/heavy.sh timeout 7200 teacher/.venv/bin/python desktopdata/owner.py meiki --work $W
HEAVY_SLOTS=12 $HERD/heavy.sh timeout 7200 $PY desktopdata/owner.py shards --work $W --root $B --out $W/shards
$PY desktopdata/owner.py manifest --work $W --shards $W/shards --out data/manifests/desktop_data/owner_desktop_v1.json
```

- The import is data/import_hooked.py's, unchanged: the recognizer pools (line crops at native size), the splits
  (hval / guard blocks, a `test` title kept out) and the page pool. One addition: the page pool's dialogue regions
  take eval's 12 px margin scaled to the frame, max(12, round(12 H / 544)) (24 px at 1080p); Vita and PSP keep 12 /
  6 px.
- The shards are synth/textspot6_pages.py's, unchanged, at the frame's own size: kind `pc` (the page's platform),
  positives the verified lines, MeikiOCR's other text as `notdlg`.
- `tests/test_desktop_data.py` runs the whole path on a two-page 1920 x 1080 bucket.

## Reproduce

```sh
H=$HERD/shared; A=$H/artifacts; D=$A/desktop-data; PV=$H/private; PY=.venv/bin/python; TPY=teacher/.venv/bin/python
M=data/manifests/desktop_data
# 1a: the plan, the build (one heavy slot, 7 minutes), the manifest
git show 1b4e074:data/manifests/desktop_pages_v21.json > $D/inputs/desktop_pages_v21.json
$PY desktopdata/hooked.py plan --labels-manifest $D/inputs/desktop_pages_v21.json \
    --labels-shards $PV/vocr-desktop/shards_v21 --n 20000 --out $D/hooked/plan_v1.json
HEAVY_SLOTS=12 $HERD/heavy.sh timeout 10800 $PY desktopdata/hooked.py build --plan $D/hooked/plan_v1.json \
    --pool $A/b2-ingest2/pagepool/v2.1 --pages $A/b2-ingest2/pages --labels-shards $PV/vocr-desktop/shards_v21 \
    --out $D/pages/hooked_v1 --workers 5
$PY desktopdata/hooked.py manifest --plan $D/hooked/plan_v1.json --shards $D/pages/hooked_v1 --out $M/hooked_desktop_v1.json
# 1b: the plan, MeikiOCR at 0.2 (two slots, 11 minutes), the build (two slots, 70 s), the manifest
$PY desktopdata/realplus.py plan --out $D/realplus/plan_v1.json
for i in 0 1; do HEAVY_SLOTS=12 $HERD/heavy.sh timeout 10800 unshare -rn $TPY desktopdata/realplus.py meiki \
    --plan $D/realplus/plan_v1.json --out $D/realplus/meiki02 --part $i/2 & done; wait
for i in 0 1; do HEAVY_SLOTS=12 $HERD/heavy.sh timeout 14400 $PY desktopdata/realplus.py build \
    --plan $D/realplus/plan_v1.json --meiki $D/realplus/meiki02 --out $D/pages/realplus_v1 --part $i/2 & done; wait
$PY desktopdata/realplus.py manifest --plan $D/realplus/plan_v1.json --shards $D/pages/realplus_v1 --out $M/realplus_native_v1.json
# 1c: 40 shards of 250 frames (two slots, 5 minutes each), the manifest
for p in 0 1; do HEAVY_SLOTS=12 $HERD/heavy.sh timeout 14400 $TPY desktopdata/synth.py render \
    --out $D/pages/synth_v1/desk_synth_v1 --shards 20 --first $((p * 20)) --per-shard 250 --workers 5 & done; wait
$PY desktopdata/synth.py manifest --dir $D/pages/synth_v1/desk_synth_v1 --out $M/synth_desktop_v1.json
# the counts and this document's tables (--check: what CI runs)
$PY desktopdata/stats.py run --set "hooked Vita (native)=$D/inputs/desktop_pages_v21.json:$PV/vocr-desktop/shards_v21:train:vita" \
    --set "hooked PSP (native)=$D/inputs/desktop_pages_v21.json:$PV/vocr-desktop/shards_v21:train:psp" \
    --set "real half (fit 960)=data/manifests/textspot2_shards_v1.json:<textspot2's lab_* and more_* shard dirs>" \
    --set "1a hooked desktop=$M/hooked_desktop_v1.json:$D/pages/hooked_v1:train" \
    --set "1b realplus native=$M/realplus_native_v1.json:$D/pages/realplus_v1" \
    --set "1c synthetic desktop=$M/synth_desktop_v1.json:$D/pages/synth_v1" --out $M/stats_v1.json
$PY desktopdata/stats.py tables --stats $M/stats_v1.json --doc docs/desktop-data.md
# QA sheets (local, never committed)
$PY desktopdata/qa.py sheet --shard $D/pages/hooked_v1/train/PCSG00845_000 --out $D/qa/hooked.png
# 2: the PC test set (pctest.py; its working files under $D/pctest)
curl -sSfL -A "<UA>" -o $D/vndb/vndb-db-2026-10-31.tar.zst https://dl.vndb.org/dump/vndb-db-2026-10-31.tar.zst
tar -I zstd -xf $D/vndb/vndb-db-2026-10-31.tar.zst -C $D/vndb/dump          # sha256 2048bc35…
$PY desktopdata/pctest.py pools && $PY desktopdata/pctest.py candidates && $PY desktopdata/pctest.py fetch
$PY desktopdata/pctest.py dedupe --against $PV/vocr6-spotter3/confirm/shots
HEAVY_SLOTS=12 $HERD/heavy.sh timeout 10800 unshare -rn $TPY desktopdata/pctest.py read --procs 5
$PY desktopdata/pctest.py select
$PY desktopdata/pctest.py sheets --mode pass1 --out $D/pctest/review/pass1 --chunk 100   # Claude helpers decide
$PY desktopdata/pctest.py queue2 --pass1 $D/pctest/review/pass1 --out $D/pctest/review/pass2_queue.json
$PY desktopdata/pctest.py sheets --mode pass2 --lines $D/pctest/review/pass2_queue.json --out $D/pctest/review/pass2 --chunk 70
$PY desktopdata/pctest.py checksample --pass1 $D/pctest/review/pass1 --pass2 $D/pctest/review/pass2 --n 200 \
    --out $D/pctest/review/check_ids_1.json
$PY desktopdata/pctest.py sheets --mode check --lines $D/pctest/review/check_ids_1.json --out $D/pctest/review/check --chunk 50
$PY desktopdata/pctest.py final --pass1 $D/pctest/review/pass1 --pass2 $D/pctest/review/pass2 \
    --resolve $D/pctest/review/resolutions.jsonl --check $D/pctest/review/check     # the manifest, summary, frames
```

The pass-2 queue was built in three parts as pass 1 finished (`pass2_queue_{a,b,c}.json`, 138 + 60 + 19 lines; one
line of part b was later turned to unsure by its pass-1 reviewer, so 216 of the 217 count) and the hand check in two
(200 lines drawn when 1,082 lines were kept, then 14 of the 77 kept after, at the same rate); `final` reads every
batch, so the parts give the same labels as one queue.

## Limits

- **1a is an emulation of a desktop capture.** The emulators' own internal-resolution rendering (Vita3K and PPSSPP
  draw at 2-10x internally, sharper than any resample of the native frame) is approximated by the resamplers; window
  chrome (menu bars, title bars) is not drawn.
- **1b's positives are realplus's kept lines only.** Most text on its frames is an ignore area, so it teaches the
  detector what real PC text looks like where it is labelled and nothing where it is not; text MeikiOCR misses at 0.2
  and realplus did not label is background.
- **1c's scenes are VNDB screenshots of training-side games** (textspot's pool), some upscaled to 4K, so their
  backgrounds are softer than a native 4K render.
- **Names.** synth3's VNDB name pool (93,169 names, the source of 1c's and every VOCR5/6 render's speaker names) was
  built before the PC test set existed: 1,103 of its entries (1.2%) equal the full name of one of the test VNs' 922
  characters, many of them common names other VNs share. The set's dialogue metric leaves speaker names out, but a
  dialogue line can name a character. Every recognizer trained on synth3 renders carries this, as it carries the
  CLANNAD names the README discloses.
