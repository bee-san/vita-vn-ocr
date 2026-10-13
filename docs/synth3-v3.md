# synth3 version 3: closing the calibration gaps of version 2

synth3 version 2 (docs/synth3.md) matched the training-side real crops in label length,
name share, polarity and PSP geometry, but five gaps remained (new-desktop plan, section 1.2):
Vita crops too tall, crop margins too tight, edge busyness too low, Latin letters three times too
frequent and too few closing 」. Version 3 (`synth/synth3.py`, `CFG["version"] == 3`) keeps every
version-2 fix and changes what the measurements below found behind each gap. It also applies the
fleet's newer hygiene rules: the fresh native textbox titles and CLANNAD are excluded with their
VNDB families.

Study 1b compared version 3 with version 2 under study 1's frozen recipe and decision rule
(`eval/synth3_study1b_protocol.json`).

**Result:** version 3 **lowers teacher_val CER against version 2** under study 1's frozen rule: 5.67% -> 5.26% (both views, both seeds), D - B -0.41 points, VN-cluster 95% interval [-0.65, -0.18], and below 0 for both seed pairs (-0.48, -0.34). devnative (native console frames, reported only) shows no measured difference. So version 3 is the recommended VN-style synthetic data for the next studies.

## What was behind each gap

Every number in this section comes from training-side real crops only: teacher_train plus
real_train, 20,000 random lines, after dropping every row of a reserved or development VN (220
rows: eval's test games, `is_test_vn`, teacher_val's and the general-VN development VNs, the
fresh native textbox titles and CLANNAD with their families). They were measured with
`synth/synth3_stats.py` against 20,000 crops of the version-2 shards 0-1.

### Crop height: the composition, not the renderer

73% of the real crops (86% on the Vita target) are PC or home-console screenshots area-scaled
to a console size (teacher's and realdata's `hires` kind). Version 2 drew only 12%
of its Vita-target crops that way, 68% as native Vita renders and 20% as PSP renders upscaled
2x (a case the real crops do not contain at all). Per capture kind, version 2 was close (native
Vita and PSP within 1-2 px; its few PC renders scaled to PSP had smaller text):

| target:kind | real crops | real height p10/50/90 | real pixel ink p10/50/90 | v2 crops | v2 height | v2 pixel ink |
|---|---|---|---|---|---|---|
| vita:hires | 10,061 | 23/28/35 | 15/21/32 | 1,636 | 25/30/38 | 16/21/32 |
| vita:vita | 1,639 | 25/32/41 | 18/24/36 | 9,226 | 27/34/43 | 18/25/37 |
| vita:psp2x | 0 | | | 2,696 | 30/37/47 | 22/28/41 |
| psp:hires | 4,501 | 15/18/22 | 10/14/20 | 672 | 14/17/21 | 9/11/18 |
| psp:psp | 3,799 | 17/21/27 | 12/15/23 | 5,770 | 16/20/25 | 11/14/22 |

"Pixel ink" is one measure applied to every source (`px_ink()`: teacher/geometry.py refine()'s
rule run on the stored crop).

### Margins: a measurement artifact

The margins in section 1.2 (real top/bottom/left/right .19/.27/.24/.30 of ink height, v2 .17)
used MeikiOCR's character boxes for real crops and the renderer's ink box for synthetic ones.
The two are different measures. On the same teacher_train crops (7,295 Vita, 4,695 PSP, reserved
VNs dropped), MeikiOCR's union box sits inside the pixel ink at the right (median 1.8 px on
Vita, 1.0 px on PSP) and at the bottom (1.0 px), and outside it on the left (1.2 / 1.0 px);
`diag/inset.py`. With one pixel-ink measure for both, version 2's margins were close to real:

| target | real t/b/l/r (px ink) | v2 t/b/l/r (px ink) |
|---|---|---|
| Vita | .18/.19/.23/.18 | .18/.18/.18/.18 |
| PSP | .14/.14/.23/.15 | .20/.22/.21/.23 |

So the Vita crops were not too tight (only the left margin was short) and the PSP crops were too
loose: version 2's extra margins (up to .12-.24 of ink height, from the character-box numbers)
over-padded small text. `synth3_stats.py` now reports the pixel-ink margins for every source,
by target and capture kind; the character-box columns stay for comparison with docs/synth3.md.

### Edge busyness: clean boxes and a selection bias

Edge busyness is the std of the crop's top and bottom two rows. `synth3_stats.py` now splits it
into the share of those pixels that are ink (a neighbouring line cut by the crop), the std of the
others (box and scene texture) and the difference between the top and bottom rows:

| source | busy p50 | edge ink (mean) | texture p50 | top-bottom p50 |
|---|---|---|---|---|
| real | 21.6 | .093 | 15.9 | 8.1 |
| v2 | 13.9 | .061 | 11.9 | 5.7 |

Per kind, real native Vita crops are the calmest (14.8) and PC shots scaled to PSP the busiest
(26.1); version 2 was 12.7-14.8 everywhere. Two causes stood out in version 2's metadata:
- GRAD-finder crops were the cleanest (12.4 against 14.9 for gt crops). The finder's box is
  kept only when it holds and hugs one labelled line; lines beside clutter or a close neighbour
  failed that check and produced no crop at all, so busy regions were under-sampled.
- Busyness falls steeply with box opacity. On a version-3 calibration render (scaled PC text,
  busyness p50 by box alpha): 22.9 at 0.2-0.4, 21.0 at 0.4-0.6, 14.7 at 0.6-0.8 and 9.1 at
  0.8-1.0; text without a box: 47. Version 2's boxes were mostly at the opaque end.

### Latin letters: UI menus

Version 2 had 19.3 Latin letters or digits per 1,000 characters against 5.7 in the real crops.
UI menus were 12.4% of its crops and brought 9.8 of those 19.3 (a quarter of the menu words are
Latin: SAVE, Q.LOAD, CONFIG, CGモード). Wikipedia narration, subtitle and roleplay lines and the
alnum dressing (4% of dialogue lines) brought most of the rest. Real Latin is mostly digits.

### Closing brackets: only the first page

| source | 「 / 」 per 1,000 | lines opening / closing / both |
|---|---|---|
| real | 14.8 / 16.1 | .156 / .174 / .087 |
| v2 | 14.1 / 9.3 | .124 / .101 / .047 |

Version 2 wrapped each utterance into the box and dropped what did not fit, so a long utterance
never reached its closing bracket. Names written inline before 「 (5.9% of version 2's crops) also
put an opening bracket in mid-line on 3.6% of lines, three times the real 1.2%.

## What version 3 changes

| change | CFG | why |
|---|---|---|
| large renders area-scaled to the console 47% of regions (v2 11%), native Vita 24% (46%), PSP 24% (30%), PSP 2x 5% (13%); 40% of the scaled ones go to the PSP target (30%) | `platform`, `scaled_to_psp` | composition (above); native console renders stay about half of the crops because the console's own screens are the deployment target |
| scaled em N(20.5, 3.3) in [13, 30] px on Vita (v2 N(21, 2.5) in [16, 27]), N(13.5, 2.0) in [9, 19] on PSP (N(11.5, 1.5) in [9, 15]); native PSP sizes one step larger | `scaled_em`, `psp_sizes` | pixel ink of the real hires and native PSP crops |
| VNDB's Q90 JPEG at the source resolution, before the area downscale | `capture.scaled` | how a PC screenshot reaches the real crops (VNDB, then teacher's or realdata's scaling) |
| per-target extra margins, small except on the left (Vita left up to .12, PSP left up to .2 of ink height) | `pad_extra` | the pixel-ink margins above |
| gt boxes of outlined or shadowed white or black text take their rows from the glyph fill (+1 px), so the outline and shadow above and below sit in the pad; the columns stay the full ink's; on scaled renders the fill is found before the downscale | (code: `fill_ink`) | eval's and teacher's boxes come from refine(), whose white-text and black-text masks find the fill, on the source screenshot |
| scenes on 92% of regions (82%), scene zoom 1.0-1.2 (1.0-1.4) | `background`, `scene_zoom` | more scene detail behind boxes |
| box opacity per platform: scaled PC renders mostly 0.15-0.6 (on dark), PSP 0.2-0.65, native Vita unchanged | `box_alpha` | the busyness-opacity relation above; native Vita was already close to real (13.4 against 14.8) |
| outlines and shadows more frequent on scaled renders | `outline.scaled`, `shadow.scaled` | PC VN text is usually outlined or shadowed |
| a line whose GRAD box fails the one-line check gets its gt crop; GRAD regions 75% (50%) so that genuine GRAD crops stay about a third of all crops | `grad_fallback`, `crop` | the selection bias above |
| UI menus 5% of regions (5%) with 2-5 lines (2-6); Latin menu words 8% of menu-word draws (26%); place and status labels (教室, 体力, 所持金...) 45% of UI lines | `layout.ui`, `ui` | Latin share; short unpunctuated labels are common in the real crops |
| a pool line with Latin letters or digits is kept with probability 0.22, else redrawn (up to 8 times); focus contexts for a non-Latin character prefer lines without Latin; alnum dressing 1.2% (4%) | `latin_keep`, `latin_redraws`, `dressing.alnum` | Latin share |
| text that overflows the box continues on later pages and a render shows any page | `pages` | closing brackets |
| inline names 6% / 8% (18% / 22%), brackets on 70% of named and 35% of unnamed lines (60% / 30%) | `name_mode`, `brackets` | mid-line 「; bracket rates after paging |

Calibration renders used seed 3001 (never the training seed) and 20,000 crops each: five tuning
rounds and a sixth that checked the fill-box fix (`$HERD/shared/artifacts/synth3-v3/calib/`,
c1-c6).

## Hygiene

- **Fresh native textbox test** (main, PR #49): `data/manifests/synth3_exclude_fresh_native.json`
  lists the six selected titles, the two reserves, their direct VNDB relations of any kind
  (v15053 v52836 v25385 v28146 v57035, queried on the kana API) and the VNs of their source
  releases: 13 VNs. It also holds the 157 source screenshots' file and decoded-RGB sha256 and the
  name hashes of the 142 VNDB characters of those VNs.
  - None of the 13 VNs is in teacher_train, teacher_val, teacher_distill_*, real_train, real_val
    or real_unlabelled, in synth2's background list, in realdata's shots or in data/vndb/shots.
  - 0 of 7,582 scene candidates match a source screenshot by file or pixel sha256
    (`synth3.py scene-hashes`).
  - 0 text-pool lines mention one of their Japanese titles.
  - The names of 124 of those 142 characters appear as rows of version 2's VNDB name pool (13 in
    its dev pool). Version 3 drops every name row whose name hashes to one of them.
- **CLANNAD as a title**: `data/manifests/synth3_exclude_clannad.json` lists CLANNAD (v4), its 19
  direct VNDB relations (Tomoyo After v12, Hikari Mimamoru Sakamichi de v4060 and the rest) and
  the four other Key titles that share a VNDB release with it in the Key 10th Memorial BOX and
  the Key Trilogy Premium Box (v5 Little Busters!, v33, v34 planetarian, v36): 24 VNs.
  - Version 2 already excluded v4's own characters and scenes. Version 3 also drops the
    family's: the VNDB entries of v12 and v4060 repeat CLANNAD's cast (古河 渚, 岡崎 朋也, ...;
    15 of their 26 characters were rows of version 2's name pool), and version 2's scenes held
    14 screenshots of v12, v4060, v5 and v36 (1,182 scene VNs in version 2, 1,178 in version 3).
  - The subtitle pools (JESC, OpenSubtitles) hold CLANNAD anime dialogue, e.g.
    「朋也くんも、食べたいですか」. Version 3 redraws every pool line, focus context or name row
    holding one of 41 text markers: the full names of the characters of v4, v12 and v4060, their
    distinctive given and family names, and the title. These hit 159 of about 3.8M pool lines.
    Anime lines that name no character cannot be identified and remain, in version 2 as in
    version 3.
- **Focus targets and mining**: version 2's focus targets counted characters over all of
  teacher_train and real_train, which hold 8 CLANNAD crops (4 + 4; audit-synth3 B1), and its
  mining report includes 26 lines of general-VN development VNs (audit F6). Version 3 renders
  with targets rebuilt from those labels without any reserved, development, fresh-native or
  CLANNAD-family VN (21,706 lines, 220 dropped), and from a re-mined report (`synth3_mine.py`,
  same blobs and threshold, `--forbid-vns` with all of those VNs: 3,293 lines). The result
  correlates 1.000 (real frequencies) and 0.999 (error rates) with version 2's targets.
- **Calibration rows**: `synth3_stats.py` drops every row of eval's test games, `is_test_vn` and
  both exclusion manifests by default; the calibration runs also passed the development VNs.
- The Jiten visual-novel word list (2,104 VN scripts, used for focus weights since version 1)
  is an aggregate frequency list that may count CLANNAD or other protected titles; it carries no
  per-title text.

## Dataset

### Render and pin

- `data/manifests/synth3_shards_v3.json` pins 1,000,000 crops in 100 shards (4,780,813,115
  bytes) of generator version 3 at `5ffdabf`, seed 20261012, rendered on the 64-core box in 12
  heavy.sh jobs of 5 workers (2026-10-12 22:56Z to 2026-10-13 00:00Z), 0 render errors. The pin
  was committed and pushed at 00:01:49Z (`aa02dcb`), before any D_s3v3 step. The recipe records
  the code, font, vocabulary, text, exclusion-manifest, scene-hash and focus-target hashes.
- Two earlier version-3 renders never fed a run:
  - the render of `6344ad4` was discarded unpinned when the CLANNAD text markers were added;
  - the render of `e633396` was pinned (`5b67247`) and then superseded. Its gt crops of outlined or
    shadowed white or black text lost thin glyphs: after an area downscale the fill of 。 or 」
    can vanish, and the fill box was narrowed to the fill's columns. 2.9% of its gt crops had ink
    touching a side edge (version 2: 0.23%); the audit's worst crops showed it (label す。」 on a
    crop showing す). D_s3v3 had been queued and was stopped before it got a slot.
- Determinism: re-rendering the first 1,000 crops of shards 0 and 57 gives byte-identical pixels,
  shapes, labels and metadata.
- On shards 0-4 (50,000 crops): genuine GRAD-finder crops are 32.9% of crops (version 2:
  32.6%); gt crops whose ink touches a side edge 0.27% (0.23%); gt crops whose ink touches the
  top or bottom edge 2.6% (0.36%), which is the fill-box rule putting outlines and shadows into
  the pad.

### Label audit

`synth/synth3_audit.py`: PP-OCRv6 small reads 1,500 random crops (seed 1) in screen polarity.

| set | teacher vs label CER | exact lines |
|---|---|---|
| synth3 version 1 | 4.52% | 76.7% |
| synth3 version 2 | 3.56% | 80.1% |
| synth3 version 3 | 3.83% | 77.9% |

| stratum | v2 CER (lines) | v3 CER (lines) |
|---|---|---|
| native Vita | 2.42% (697) | 2.50% (362) |
| native PSP | 4.27% (428) | 5.16% (344) |
| PSP 2x | 4.71% (201) | 6.01% (78) |
| PC renders scaled down | 4.95% (174) | 3.76% (716) |
| gt crops | 3.89% (1,018) | 3.93% (1,009) |
| GRAD crops | 2.92% (482) | 3.63% (491) |

Disagreement is not label error: the teacher misreads hard renders too. Reading the 36 worst
version-3 crops by eye found pixel and pen fonts, tiny PSP-scaled names, faint text over bright
scenes and rare kanji, and one generator fault that version 2 shares: a speaker name drawn as
its own line, or in a name tab, is not counted in the region's width, so a name longer than the
region is clipped while its label keeps the whole name (エロデカ・ボイン on a crop that ends at ボ).
Ink touching a side edge marks it: 2.0% of version 2's name-line crops and 1.8% of version 3's,
0.29% of all crops in both; name tabs 0.09% (v2) and 0.16% (v3). It is not fixed in the pinned
code; the fix is to count the name's width in the region's.

### Calibration (training-side real crops)

`synth/synth3_stats.py` on 20,000 real crops (teacher_train + real_train, 220 reserved or
development rows dropped) and 20,000 crops of shards 0-1 of each version:

| | real | synth3 v2 | synth3 v3 |
|---|---|---|---|
| KL(real ‖ source), characters | | 0.075 | 0.060 |
| label length p10 / p50 / p90 | 2 / 8 / 24 | 2 / 8 / 24 | 2 / 8 / 24 |
| name-like crops | 40.0% | 36.0% | 35.4% |
| dark text (stored inverted) | 21.0% | 23.8% | 24.7% |
| Vita crop height p10 / p50 / p90 | 23 / 28 / 36 | 27 / 34 / 44 | 25 / 31 / 40 |
| Vita hires-kind height | 23 / 28 / 35 | 25 / 30 / 38 | 23 / 29 / 36 |
| Vita native height | 25 / 32 / 41 | 27 / 34 / 43 | 27 / 33 / 41 |
| PSP crop height | 16 / 19 / 25 | 16 / 20 / 25 | 15 / 19 / 25 |
| Vita pixel-ink margin t/b/l/r (p50) | .18/.19/.23/.18 | .18/.18/.18/.18 | .17/.17/.18/.17 |
| PSP pixel-ink margin t/b/l/r (p50) | .14/.14/.23/.15 | .20/.22/.21/.23 | .14/.14/.21/.15 |
| edge busyness p50 | 21.6 | 13.9 | 18.5 |
| edge ink (mean) / texture p50 / top-bottom p50 | .093 / 15.9 / 8.1 | .061 / 11.9 / 5.7 | .091 / 14.3 / 6.8 |
| Latin letters and digits per 1,000 characters | 5.7 | 19.3 | 5.9 |
| 「 / 」 per 1,000 characters | 14.8 / 16.1 | 14.1 / 9.3 | 15.6 / 13.7 |
| lines opening / closing / both | .156 / .174 / .087 | .124 / .101 / .047 | .156 / .148 / .073 |

By capture kind (busyness p50, real / v2 / v3): native Vita 14.8 / 13.4 / 15.0, Vita hires
20.8 / 12.7 / 19.3, native PSP 21.3 / 14.7 / 19.6, PSP hires 26.1 / 13.1 / 20.3.
Version 3's Vita hires crops read 15/23/34 px of pixel ink against real 15/21/32. On those crops
the renderer's labelled ink box (with outline and furigana) is 15/21/28 px: in 44% of them the
pixel-ink rule's run is at least 2 px taller, because scene detail or a neighbour's ink at the
crop edge joins it. The real crops are measured by the same rule, so the two columns compare
like with like, but how much of the difference is text size and how much is busier edges cannot
be separated.

## Study 1b

The protocol (`eval/synth3_study1b_protocol.json`, frozen 2026-10-12T20:10:20Z before any run,
with three amendments each before any score) is study 1's exactly, with the arms B_s3 (generator
version 2, as in study 1, rerun here on the 64-core box) and D_s3v3 (version 3). Both train the
FL10 topology from scratch for 16,000 steps on NVIDIA one-line crops (0.5) plus the VN-style
source (0.5), seeds 1 and 2 (`eval/synth3_study1b_run.sh`); the step-16000 EMA checkpoints are
exported to the C W8I runtime and scored on teacher_val, both views, by
`eval/synth3_study1b_score.sh`. teacher_val decides; devnative (both cohort versions) is reported
only. Result: `eval/synth3_study1b_20261012.json`.

### teacher_val (decides)

| run | rv CER | rvn CER |
|---|---|---|
| B_s3 seed 1 | 5.67% | 5.62% |
| B_s3 seed 2 | 5.73% | 5.67% |
| D_s3v3 seed 1 | 5.14% | 5.19% |
| D_s3v3 seed 2 | 5.36% | 5.35% |

Arm CER (both views, both seeds): B_s3 5.671%, D_s3v3 5.261%. D - B -0.41 points, VN-cluster 95%
interval [-0.65, -0.18], line-level [-0.61, -0.22]; per seed -0.48 [-0.80, -0.16] and
-0.34 [-0.65, -0.03]; per view rv -0.44 [-0.69, -0.21], rvn -0.38 [-0.62, -0.14]. 83 of 181 VNs
improve, 53 worsen, 45 are equal. **D_s3v3 is better than B_s3** (interval below 0 and every seed
pair below 0); the mirrored rule does not hold. B_s3 here matches study 1's B_s3 to the digit
(it was already on this box), so the machine change adds nothing.

| stratum | lines | CER B | CER D | D - B, points (VN-cluster 95% CI) |
|---|---|---|---|---|
| PC renders scaled (hires) | 1,064 | 6.80% | 6.28% | -0.52 (-0.83 to -0.20) |
| native PSP | 426 | 3.46% | 3.16% | -0.29 (-0.64 to +0.02) |
| native Vita | 161 | 2.86% | 2.95% | +0.09 (-0.25 to +0.50) |
| target Vita | 972 | 4.69% | 4.28% | -0.41 (-0.70 to -0.13) |
| target PSP | 679 | 7.09% | 6.68% | -0.41 (-0.76 to -0.08) |
| 1-4 characters | 608 | 15.39% | 13.94% | -1.45 (-2.69 to -0.17) |
| name-like | 710 | 14.65% | 13.40% | -1.25 (-2.21 to -0.22) |
| ink < 14 px | 484 | 12.46% | 11.74% | -0.71 (-1.30 to -0.14) |
| dakuten / small kana | 831 | 4.43% | 4.14% | -0.29 (-0.53 to -0.05) |
| bright-on-dark | 1,050 | 4.52% | 4.16% | -0.37 (-0.63 to -0.11) |
| dark-on-light | 601 | 7.54% | 7.06% | -0.48 (-0.88 to -0.08) |
| rare kanji | 116 | 6.95% | 6.48% | -0.47 (-1.43 to +0.36) |
| Latin / digits | 62 | 11.06% | 10.95% | -0.12 (-1.53 to +1.52) |

The gain is on teacher_val's PC-screenshot majority, where version 3 made most of its changes; it
is largest on short and name-like lines. Native Vita and PSP strata (161 + 426 lines) do not
separate on their own, as in study 1.

### devnative (native console frames, reported only, cannot change the decision)

Pooled (strict + tv_native, ink + grad), D - B VN-cluster: cohort v1 +0.14 points [-0.13, +0.41],
cohort v2 +0.26 [-0.05, +0.58]. Neither interval excludes 0; the point estimate slightly favours
version 2. By file, the strict subsets (native screenshots of the reserved VNs, no overlap with
teacher_val, 38 + 26 lines) move the other way (strict_ink -0.61, strict_grad -0.95) but their
intervals are wide; tv_native (teacher_val's own native screenshots, re-cut) is near 0 or slightly
positive. These are 553-645 lines from 44-48 VNs, almost all PSP; the gap between this and the
teacher_val gain is why devnative is reported, not decisive. Full tables for every subset and both
cohorts are in the result JSON.

## Limits

- The pooled Vita crop height stays taller than real (25/31/40 against 23/28/36 on calibration
  renders): about half of version 3's Vita-target crops are still native console renders, while
  86% of the real ones are PC shots scaled down. Per capture kind the heights match.
- Edge busyness closed about 60% of its gap (18.5 against real 21.6, version 2 13.9). PC shots
  scaled to PSP stay calmer than real (20.3 against 26.1).
- Closing brackets: 13.7 per 1,000 characters against 16.1 (version 2 9.3); opening brackets
  rose to 15.6 (real 14.8). Paging makes them roughly symmetric, while the real crops have more
  closing than opening lines.
- The calibration matches distributions of training-side crops, 73% of which are PC shots;
  matching them is not evidence of better accuracy on native console text. Study 1b and
  devnative are.
- Hygiene limits are listed under Hygiene (unnamed anime lines, the Jiten list).
- Every probability not listed above is version 2's. Labels were checked by a teacher-agreement
  audit, not by hand.

## Commit history

This branch was rebased twice with commit times kept: onto the rewritten `feat/synth3` (the
earlier history is at tag `synth3-v3-history-20261012a`) and onto main after #50 was
squash-merged (tag `synth3-v3-history-20261013`). This document cites the commits of the final
branch, kept at tag `synth3-v3-20261013` after the squash merge. The protocol's amendments cite
the earlier hashes; they map as follows:

| before (tag synth3-v3-history-20261013) | on main's history | what |
|---|---|---|
| ebe146a | ac4c4d3 | generator version 3 |
| ab5db0e | 6344ad4 | CLANNAD exclusion, clean focus targets |
| 0ba3bc0 | 7bb72b2 | study-1b protocol frozen (2026-10-12T20:10:20Z) |
| 1ba727c | e633396 | CLANNAD text markers |
| 099bda9 | d734639 | protocol amendment 1 |
| f62ae57 | 5b67247 | first v3 pin (superseded) |
| edb8443 | 5ffdabf | fill-box fix; the pinned render's code |
| a32b79b | 9e948b8 | protocol amendment 2 |
| 6e194d1 | aa02dcb | v3 pin (2026-10-13T00:01:49Z) |
| 567fc7d | a48246a | scoring script |

The render's code is identified by the file hashes in the shard manifest's recipe, which are the
same before and after the rebases.

## Reproduce

```sh
A=$HERD/shared/artifacts; S3=$A/synth3; V3=$A/synth3-v3
# exclusions: data/manifests/synth3_exclude_fresh_native.json and synth3_exclude_clannad.json (committed);
# VNDB relations from the kana API (queried 2026-10-12), names from $S3/text/vndb_chars.tsv
synth/synth3.py scene-hashes --out $V3/exclude/scene_hashes.json --workers 5
# focus targets without reserved, development, fresh-native or CLANNAD-family VNs
cat $S3/mining/exclude_dev_vns.txt <(jq -r '.vns[]' data/manifests/synth3_exclude_{fresh_native,clannad}.json) \
    | sort -u > $V3/mining/forbid_vns_v3.txt
synth/synth3_mine.py run --pool $A/teacher/teacher_distill_train.npz --pool $A/realdata/real_unlabelled.npz \
    --targets $A/train2/targets --blob FL10=$A/train2/release/FL10_w8.vocr --blob H15=$A/train2/release/H15_w8.vocr \
    --conf 0.9 --forbid-vns $(cat $V3/mining/forbid_vns_v3.txt) --out $V3/mining/mining_v6s_consensus_v3.json
synth/synth3_real_labels.py --real $A/teacher/teacher_train.npz+$A/realdata/real_train.npz \
    --forbid $S3/mining/exclude_dev_vns.txt data/manifests/synth3_exclude_{fresh_native,clannad}.json \
    --out $V3/text/real_labels_v3.npz
synth/synth3_text.py targets --pools $V3/text/targets_v3 --real $V3/text/real_labels_v3.npz \
    --jiten-vn $S3/text/jiten/frequency_list_visual_novel.csv --mining $V3/mining/mining_v6s_consensus_v3.json
# render: 12 heavy.sh jobs of 5 workers (first 0, 9, 18, 27 with 9 shards; 36, 44, ..., 92 with 8)
HEAVY_SLOTS=12 $HERD/heavy.sh timeout 14400 synth/synth3.py shards --out $V3/shards/train_v3 --split train \
    --shards N --first F --per-shard 10000 --workers 5 --seed 20261012 --scene-hashes $V3/exclude/scene_hashes.json \
    --targets $V3/text/targets_v3/targets.json
synth/synth3.py manifest --dir $V3/shards/train_v3 --out data/manifests/synth3_shards_v3.json
# calibration (training-side real crops only) and label audit
synth/synth3_stats.py measure --name real=$A/teacher/teacher_train.npz+$A/realdata/real_train.npz \
    --name s2="$A/synth2/shards/synth_0000[0-1].npz" --name s3v2="$S3/shards/train_v2/synth3_0000[0-1].npz" \
    --name s3v3="$V3/shards/train_v3/synth3_0000[0-1].npz" --exclude $V3/mining/forbid_vns_v3.txt --out stats.json
synth/synth3_audit.py --shards "$V3/shards/train_v3/synth3_000[0-9]?.npz" --n 1500 --seed 1 --out audit.json --sheet worst.png
# study 1b
HEAVY_SLOTS=10 $HERD/heavy.sh timeout 36000 bash eval/synth3_study1b_run.sh B_s3 1   # and seed 2, and D_s3v3 1, 2
HEAVY_SLOTS=12 $HERD/heavy.sh timeout 7200 bash eval/synth3_study1b_score.sh
```
