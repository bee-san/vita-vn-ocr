# Counters: ッ月 → ヶ月, numerals before ヶ, and lines cut by the region (issue #109)

On the owner's PS Vita, Rubi read CLANNAD's dialogue line 「九ヶ月です」 as `ルッ月です」`. Three characters are wrong: the 「 is dropped, 九 becomes ル and the small ke ヶ becomes the small tsu ッ. The dictionary lookup then picked る. This change adds three pieces, each off unless its caller turns it on:

| piece | file | what it does | development verdict |
|---|---|---|---|
| **R**, the counter rule | `runtime/vocr_post.c` `vocr_post_counters()` | turns ッ into ヶ before 月 年 (any left), 国 条 (after a numeral or 何数幾) and 所 (after one of those other than 一) | changes **0** development lines; it fixes the owner's string; it never breaks text the corpora hold |
| **A**, the counter re-decode | `runtime/vocr_post.c` `vocr_post_redecode()` | at X + ッ/ヶ/ヵ + 月/所/年/国/条, where X is a letter but not a numeral, re-scores X's frames restricted to numerals, and a missing 「 at the blank frames before it; works from the per-frame top-K | thresholds chosen by the pre-registered rule: **6 nats** for the numeral, **4 nats** for the bracket; **0** real development lines changed; the synthetic counter set improves, **0** lines worse |
| **W**, the edge continuation | `runtime/vocr_lines_widen.c` `vocr_line_widen()` | continues a found line past the region's left or right edge through the ink that the edge cuts | large gain on regions drawn into the text; a small, unresolved loss on regions with a margin; **not adopted** as a default, so it stays behind its flag, off |

Everything is plain C99 with no malloc and no libm, has a bit-exact Python twin (`eval/vocr_post.py`, `synth/lines_widen.py`) and runs in `make -C runtime check` (`check-post`, `check-widen`: strict compiles, C self-tests, C-versus-twin parity under ASan and UBSan). Code with the VitaSDK compiler (arm-vita-eabi-gcc 15.2.0, Cortex-A9, Thumb-2): `vocr_post.o` is 3,964 B at -O2 and 2,598 B at -Os, with an 896 / 832 B stack frame for the re-decode; `vocr_lines_widen.o` is 756 / 568 B with a 72 / 64 B frame. No arena or weights are needed. The time is a few thousand integer operations per line (est., not measured on a console), next to a recognizer that takes tens of milliseconds a line.

## R: the text rule, from corpus counts

`eval/kanum_counts.py` counts every ッ, ヶ, ヵ, ケ and 箇 (and か, カ, っ, ツ for reference) before 月 所 年 国 条 村 町 日 in **700,680,047 characters** of training-side text, split by the character on the left. The text is Japanese Wikipedia (the shard `data/fetch_assets.py` fetched, 412 M), Aozora Bunko (230 M), synth3's train-split text pools other than its Aozora and Wikipedia ones (56.5 M), and the real label pools: teacher, realdata, realplus v3, native-frames and owner pool v2, 1.7 M, with every reserved test VN and every `dev_sample` VN removed. No eval set, no Jiten list and no photo went into it. Every ッ hit's context was read:

| ッ before | ッ (any left) | ッ (left a numeral / 何数幾) | ヶ ヵ ケ 箇 (any left) | ヶ ヵ ケ 箇 (numeral left) | the ッ hits |
|---|---:|---:|---:|---:|---|
| 月 | **0** | 0 | 14,883 | 14,790 | none |
| 年 | **0** | 0 | 1,522 | 1,488 | none |
| 国 | 2 | 0 | 1,610 | 1,576 | えッ国上山, 為ッ国の為ッて (valid) |
| 条 | 1 | 0 | 1,180 | 845 | 命に別ッ条ない (valid) |
| 所 | 1 | 1 | 13,306 | 8,890 | 一ッ所に立つて (ひとつところ, Aozora: valid) |
| 日 | 5 | 1 | 414 | 399 | 三ッ日月, イマドキッ日曜日, アッ日の丸, じゃッ日曜日 (valid) |
| 村 | 3 | 0 | 760 | 707 | Ainu and Austerlitz place names |
| 町 | 15 | 0 | 109 | 101 | Malay place names, えーッ町長 |

ッ月 and ッ年 occur nowhere in it. The issue's pre-registered criterion covers a pattern only when its ッ hits are at most one per 1,000 counter forms and none of them is valid text that the rewrite would break. That gives the rule:

- ッ → ヶ before **月** or **年**, whatever stands on its left;
- before **国** or **条** when the left character is a numeral (〇一二三四五六七八九十百千万, 0-9, ０-９) or 何 数 幾;
- before **所** when it is one of those other than 一 (the valid 一ッ所 stays).

日 is out: 三ッ日月 is valid with a numeral on its left. 村 and 町 have no numeral-left hit, but only ~100-700 counter forms and archaic counters, so the narrowest rule leaves them out. ヵ, ケ, 箇, か and カ are valid counters and stay. The replacement is always ヶ, the commonest counter form (ヶ月 11,367 against ヵ月 2,500 after numerals). A line break is never a numeral, so the rule never reaches across one. The rule never makes CER worse unless the original was a valid ッ before a counter, and none occurs in the corpora.

**Development.** R changes 0 lines on teacher_val dev, devnative dev, whole dialogue and the synthetic counter set, for every tier, greedy and with the LM. The models never read ッ before a counter on these data. The console's ッ comes from CLANNAD's own font at the framebuffer's resolution, which no development crop shows. On the owner's string it gives `ルッ月です」` → `ルヶ月です」`.

## A: the counter-context re-decode

`vocr_post_redecode(model, extra, {tau, open}, stats)` reads a `vocr_run()`'s per-frame top-K (`vocr_extra.topk` ≥ 1; Rubi computes the top-8 for its LM already) and its greedy path, each frame's top-1 CTC-collapsed. At each X + ッ/ヶ/ヵ + 月/所/年/国/条, where X is a letter (kana, kanji, Latin) and not a numeral or 何数幾, it scores every numeral in X's frames' top-K. The cost of a numeral d is the greedy path's score over a window (X's frames and up to 4 blank frames on each side) minus that of the best path holding one run of d and the blank elsewhere. Logit differences are rounded to Q10 nats as `vocr_lm.c` rounds them. The cheapest d is accepted when its cost is at most `tau`. A run of d right after a run of d would merge (CTC), so it may not start there. When X is the line's first character and the line's first bracket is a closing 」 or 』, the matching opener is scored at the blank frames before X and accepted at most `open`.

An accepted change is written into the extras as a swap of the chosen class with the frame's top-1 class; the logits stay where they are. So `vocr_post_greedy()` gives the edited greedy text and `vocr_lm_decode()` reads it as the top-1. `tau = open = 0` changes nothing and reads nothing.

**The threshold, chosen on development data by the issue's pre-registered rule:** the largest value of the grid {0.5, 1, 2, 3, 4, 6, 8} nats that makes no line worse on the real development cohorts or the synthetic counter set, for all three tiers, greedy and with the LM. The numeral and the bracket are chosen separately; an amendment made before Large's scores. The real references hold no counter context (0 of 3,270 lines), so the synthetic set carries the choice. It is every line of the development splits of synth3's text pools (`pools3/*.dev.txt`, which no model trained on) with a numeral or 何数幾 before ヶヵケ箇 and 月所年国条: 189 lines. Each gives three texts: 「…」, bare, and a continuation line `…」` whose opener is on an earlier line, where an inserted 「 is wrong (the amendment's third form). Each text is drawn in one of nine open training fonts and cropped seven times, with the left edge 3 px before the first ink column, at it, and 2 to 10 px into it: 3,969 crops (`eval/kanum_dev.py render`).

Lines better / worse on the synthetic set (real development: 0 / 0 at every value; G greedy, L beam + LM):

| tau (nats) | S G | S L | N G | N L | L G | L L |
|---|---|---|---|---|---|---|
| numeral 0.5 | 92 / 0 | 2 / 0 | 67 / 0 | 1 / 0 | 65 / 0 | 1 / 0 |
| 2 | 432 / 0 | 60 / 0 | 322 / 0 | 52 / 0 | 345 / 0 | 53 / 0 |
| 4 | 616 / 0 | 146 / 0 | 624 / 0 | 199 / 0 | 585 / 0 | 175 / 0 |
| **6** | 627 / 0 | 157 / 0 | 646 / 0 | 221 / 0 | 602 / 0 | 194 / 0 |
| 8 | 633 / 0 | 163 / 0 | 647 / 0 | 222 / 0 | 607 / 0 | 199 / **1** |
| bracket 2 | 5 / 0 | 4 / 0 | 9 / 0 | 4 / 0 | 11 / 0 | 8 / 0 |
| **4** | 14 / 0 | 11 / 0 | 26 / 0 | 17 / 0 | 24 / 0 | 20 / 0 |
| 6 | 23 / 0 | 19 / 0 | 49 / 0 | 39 / 0 | 49 / **3** | 41 / **3** |
| 8 | 41 / 3 | 37 / 2 | 80 / 2 | 69 / 2 | 78 / 22 | 66 / 22 |

So `tau = 6000`, `open = 4000` (1/1000 nat). Both together, VOCR5 W8I (dialogue CER %; dev score as VOCR5's protocol computes it):

| tier | decode | teacher_val dev | devnative dev | whole dialogue | dev score | synthetic: off → on | better / worse |
|---|---|---:|---:|---:|---:|---|---|
| S | greedy | 1.363 | 1.110 | 6.328 | 3.782 (unchanged) | 4.807 → **3.550** | 627 / 0 |
| S | beam + LM | 0.819 | 0.978 | 5.931 | 3.415 (unchanged) | 3.780 → **3.446** | 166 / 0 |
| N | greedy | 0.738 | 0.984 | 5.823 | 3.342 (unchanged) | 5.184 → **3.866** | 646 / 0 |
| N | beam + LM | 0.675 | 0.672 | 5.494 | 3.084 (unchanged) | 4.081 → **3.605** | 235 / 0 |
| L | greedy | 0.494 | 0.678 | 5.332 | 2.959 (unchanged) | 4.292 → **3.054** | 602 / 0 |
| L | beam + LM | 0.463 | 0.576 | 5.148 | 2.834 (unchanged) | 3.396 → **2.976** | 204 / 0 |

The real development columns are the same with and without it: **0 lines changed**. On the synthetic set it fixed the classic look-alikes: ー→一 (the most), ニ→二, 可→何, 枚/敦/改/女→数, フ→7, ミ→三, ヒ→七; and 「 where it was cut. The LM already fixes many of these from context, so it gains less there. Numbers: `eval/counters_dev_20261021.json`.

## W: lines the region's edge cuts

`vocr_line_widen(strip, rx, rw, &x, &w)` takes a found line's rows over the region plus a margin on each side (`rx`: the region's first column in the strip). The finder only sees the region, so it cannot give back what a tight region cut off; the caller usually has the pixels around it. A box that starts at the region's first column (or ends at its last) with ink in the edge's two columns is extended through that ink across gaps shorter than max(2, h/8) columns, plus the finder's 3 px padding. A column holds ink when its grey range over the line's rows is at least max(16, C/3), C being the largest range inside the box. The extension is kept only if the ink ends in such a gap inside the margin: ink that runs on to the margin's end (a box frame, a textured scene) is no cut character. That condition came from a first development run, in which 225 of 357 widened standard lines ran to the margin's end and the arm lost 0.04 to 0.13 points. Every other line is left as found.

Development (`eval/kanum_widen.py`; whole-dialogue cohorts tv_dlg + dn_dlg as VOCR5 pools them; GRAD and GRAD + refine; greedy W8I). The pixels around each region come from its frame: the cohorts were rebuilt from their frames by finder_tighten's own builders, all 1,515 pixel for pixel, with 16 px on each side. Intervals are the paired VN-cluster bootstrap's 95% (10,000 resamples, seed 0):

| tier | standard regions (CER %) | change | tight regions (left edge at the text + 0-6 px) | change |
|---|---|---|---|---|
| S | 6.328 → 6.339 (26 better, 37 worse) | +0.012 [−0.028, +0.049] | 7.943 → 6.250 | **−1.69** [−1.94, −1.46] |
| N | 5.823 → 5.846 (23, 33) | +0.023 [−0.018, +0.071] | 7.684 → 5.908 | **−1.78** [−2.06, −1.48] |
| L | 5.332 → 5.372 (16, 36) | +0.041 [−0.010, +0.104] | 6.897 → 5.404 | **−1.49** [−1.72, −1.27] |

By cut (N): 0 px 5.709 → 5.761 (unresolved), 2 px 6.927 → 5.535, 4 px 8.436 → 5.831, 6 px 9.666 → 6.504. On the standard regions 243 of 3,424 lines are widened. No tier's interval lies entirely above 0, but the mean change over the tiers is +0.025 points, so the pre-registered rule (mean ≤ 0) does not adopt it as a default. It stays behind its flag, off: a caller that knows its regions are drawn tight can turn it on.

## The owner's photos (local check only)

The photos are of CLANNAD, a protected test title, so they are never committed, uploaded or used to choose anything. Every threshold above was fixed before the photo was read with these pieces on. A phone photo is not the console's framebuffer: even rectified to 960×544 (perspective from the screen's edges, INTER_AREA) it adds blur, moiré and a glare band, and it does not reproduce the console's ッ (every model reads ヶ there).

Through Rubi's own pipeline (`vjo-ocr`: grey, TEXTBOX_RECOVERY finder, the refinement, W8I, LM o3 1000k), a region that starts at or before the 「 reads 「九ヶ月です」 on every model and arm. Moving the region's left edge into the line in 2 px steps reproduces the console's error. From 2 px into the 「 it is lost (S, N; Large from 4 px), and from 20 px in (inside 九) Normal reads `ルヶ月です」` and Large `しヶ月です」` at 22 px. The finder and the refinement never cut the line on this frame; the region's own edge does. With the beam + LM:

| region | arms | S | N | L |
|---|---|---|---|---|
| margin before 「 | all | 「九ヶ月です」 | 「九ヶ月です」 | 「九ヶ月です」 |
| edge 8 px into the line | rule only / + re-decode | 九ヶ月です」 | 九ヶ月です」 | 九ヶ月です」 |
| | + edge continuation / all three | **「九ヶ月です」** | **「九ヶ月です」** | **「九ヶ月です」** |
| edge 20 px in (into 九) | rule only / + continuation | 九ヶ月です」 | ルヶ月です」 | 九ヶ月です」 |
| | + re-decode / all three | 九ヶ月です」 | **九ヶ月です」** | 九ヶ月です」 |
| edge 22 px in | rule only / + continuation | 九ヶ月です」 | ルヶ月です」 | しヶ月です」 |
| | + re-decode / all three | 九ヶ月です」 | 1ヶ月です」 | **九ヶ月です」** |

Over the 15 left-edge positions (−6 to +22 px), the exact readings with the LM are S 4 / 4 / 9 / 9, N 4 / 4 / 9 / 9 and L 5 / 5 / 10 / 10 for rule only / + re-decode / + continuation / all three. The continuation does not fire when the edge lies inside 九: the edge columns' range is under its threshold, so the 「 there stays lost. The re-decode turns the cut 九's ル or し back into 九, except Normal at 22 px, which picks 1. On the console string itself, the rule turns `ルッ月です」` into `ルヶ月です」`.

So the likely cause of the console's reading is a region drawn tight against or into the line. The follow-up for Rubi is #110: capture a margin around the region, then continue the cut lines or pad them. **To test on the console** (hardware work is paused): the same page with the region's left edge 10 px before the 「, with the counter rule and the re-decode on; and with the edge continuation on and a region drawn against the text.

## Using it

```c
#include "vocr_post.h"
vocr_post_opts po = {6000, 4000};                    /* tau, open: 1/1000 nat (the development choice) */
rc = vocr_run(&ctx, &img, text, cap, &ex);           /* ex.topk >= 1 */
if (vocr_post_redecode(&model, &ex, &po, NULL) > 0)  /* edits ex's top-K classes */
    vocr_post_greedy(&model, &ex, text, cap, NULL);  /* or vocr_lm_decode(&dec, &ex, text, cap, NULL) */
vocr_post_counters(text);                            /* ッ月 -> ヶ月 ... in place */
```

`vocr_cli` takes `--post` (the rule on every printed line) and `--redecode TAU[,OPEN]`. `pyvocr` has `post_counters(text)` and `Vocr.redecode(run_topk_output, tau, open)`. Reproduce: `eval/kanum_counts.py count`, then `eval/kanum_dev.py render`, `cache` (with and without `--synth --synth-only`), `score` and `summary`, then `eval/kanum_widen.py outer` and `score --outer`, all through `heavy.sh`.
