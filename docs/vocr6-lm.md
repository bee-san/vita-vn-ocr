# vocr6-lm: Rubi's character LM rebuilt with the hooked games' text (T4, #127)

Rubi decodes every OCR line with a CTC prefix beam search and a character 3-gram LM (`runtime/vocr_lm.c`; release
v0.2.6-ctclm.2, `docs/ctc-lm-decode.md`). This round rebuilt that LM with the text of the owner's hooked PS Vita and
PSP captures (#125), retuned the decoder on the hooked hval pages and VOCR5's development cohorts through Rubi's own
line path, and tried a dictionary-aware variant.

**Everything here is development data, and every console time is an estimate.** No protected set was read: not
CLANNAD, vita-test, psp-test, the fresh native test, or the hooked test title (Dance with Devils). Selection follows
`eval/vocr6_lm_protocol.json`, frozen before any score was read, with two dated amendments (below). Machine-readable:
`eval/vocr6_lm_dev_20261026.json`. The tables are generated from it by `eval/vocr6_lm.py tables`.

## Answers first

- **Two of Rubi's four LM tiers change: 1 MB (the default) and 4 MB.** Both are rebuilt with the hooked train titles'
  lines and get new decoder settings. 0.5 MB and 2 MB keep ctclm.2 and its settings: the pre-registered rule
  adopted nothing there. The new blobs are no larger than the ones they replace (982,320 B against 987,600 B;
  3,603,552 B against 3,781,072 B).
- **The owner's condition, and the ruling on it.** The VOCR6 rules (under the owner's directive of 2026-10-26
  10:17Z) let a game-text LM ship "if it's measurably better on the page path (unseen-game dev sets AND hval)". The
  new LMs are resolved better on hval and level on the unseen-game dev sets, at both tiers. So the condition holds if
  it means plan section 7's adoption rule, and not if it means a resolved gain on each set. The conductor, who wrote
  the rule, ruled at 16:38Z (audit-vocr6-lm B1) that it means plan section 7's pre-registered rule: "better on hval
  and not worse on the unseen-game dev sets (game-bootstrap 95% interval)". Both tiers meet it ("The owner's
  condition" below).
- **New settings for the adopted tiers.** LM weight 0.6 (was 0.5) and a per-character bonus of 2.5 nats (was 3.5).
  The start-of-line context is on and the end-of-line term is off (`VOCR_LM_BOS`, was `VOCR_LM_EOS`). The cascade
  threshold stays 700.
- **Measured gain (VOCR5-N with its cascade to L, cross-validated, Rubi's display path):** hooked hval page path
  dialogue CER 4.12% → 3.39% (1 MB) and 4.03% → 3.57% (4 MB). hval clean lines 0.49% → 0.40% and 0.46% → 0.41%.
  The unseen-game dev score is level: 2.93 → 2.90 and 2.89 → 2.90, neither resolved. VOCR5-S and VOCR5-L
  move the same way (models table).
- **Most of the page-path gain is one title.** PCSG00854's dialogue box has a decoration that the finder reads
  as a line of junk. The old LM starts that line with a kana, so Rubi's non-Japanese filter keeps it on screen;
  the new LM does not, and the filter drops it. Without that title the gains are smaller but still resolved:
  page path −0.10 points (−0.17 to −0.03) at 1 MB and −0.05 (−0.11 to −0.002) at 4 MB, clean lines −0.09
  (−0.14 to −0.03) at 1 MB.
- **The game text itself is what helps.** The same recipe without game text (family A, with the retuned
  settings) is not resolved better than the incumbent at any tier. With game text (family B) minus without (A),
  cross-validated: at 1 MB, page path −0.54 points (−1.53 to −0.07), clean lines −0.10 (−0.14 to −0.05) and dev
  score −0.01 (−0.09 to +0.06); at 4 MB, −0.45 (−1.32 to −0.04), −0.06 (−0.11 to −0.02) and −0.05 (−0.09 to −0.01).
- **What the game-text LM keeps.** It is order 3, so no stored fragment is longer than 3 characters. 1.5% of the
  1 MB blob's bigrams and trigrams (7.7% at 4 MB) occur only in the game text. Of all its bigrams and trigrams,
  26% (19% at 4 MB) are ones the same recipe without game text does not keep: the game counts also lift public
  n-grams over the pruning threshold. Apart from runs of one repeated small kana (up to 22 characters), greedy
  continuation recovers at most 12 characters of a training line (a game's title in its own text), against 10 for
  the LM without game text.
- **The dictionary-aware variant does not help.** A word bonus from JMdict's headwords ranked by Jiten (the files
  Rubi ships) is worse than the plain LM at every setting tried, so it is not built for the console. Its cost
  would be 0.23-2.5 MB more memory and about 200 trie lookups per line (estimates).
- **On the shared page-path benchmark** (#129, Rubi's actual finder, report only): hval 3.88% → 2.23% (without
  PCSG00854: 2.33% → 2.21%, resolved); unseen-game dev 3.04% → 3.14%, not resolved. One VN's box-border junk line
  accounts for that rise (without it: 3.07% → 3.04%).
- **On the original line path nothing passes.** Without Rubi's display rules (amendment 1), the junk lines drive
  the selection toward reading fewer characters, and no tier's choice meets the rule.

## What ships

<!-- table:shipped -->
| LM tier | ships | recipe (game-text weight, re-cut) | alpha | beta | flags | cascade |
|---|---|---|---:|---:|---|---:|
| 500k | ctclm.2 (unchanged) | – | 500 | 3000 | EOS | 700 |
| 1000k | `vocr_lm6_o3_1000k.vlm` | game text x64, order 3 | 600 | 2500 | BOS | 700 |
| 2000k | ctclm.2 (unchanged) | – | 500 | 3000 | EOS | 700 |
| 4000k | `vocr_lm6_o3_4000k.vlm` | game text x16, re-cut, order 3 | 600 | 2500 | BOS | 700 |
<!-- /table:shipped -->

The flags are `vocr_lm_opts.flags`; beam 8, top-8, gate 8 nats and bracket 8 nats are unchanged everywhere. One blob
serves every model, as before. The decoder workspace is 41,632 B for VOCR5-N/L and 39,712 B for VOCR5-S, under Rubi's
`VJO_VOCR_LM_WS_BYTES` (49,152 B). The search is the same, so the per-line time is unchanged: est. 0.4-0.7 ms per
640-px line (`docs/ctc-lm-decode.md`). The blobs are smaller, so their one-time load is not longer. Rubi needs a code
change, because its `lm_opts()` uses one LM weight and one flag set for every tier; the Rubi PR makes both per tier.

## The owner's condition

The owner's directive of 2026-10-26 10:17Z was "ship it if accuracy is good. i dont want to decide anything". The
VOCR6 rules the conductor wrote under it say, of this round, in full: "LM from game hook text: allowed. Ship it if
it's measurably better on the page path (unseen-game dev sets AND hval), and still build the no-game-text variant to
measure the gain. Prune to the smallest order/size that keeps the gain (less verbatim script, less memory);
attribution as for the models trained on game data."

What was measured per set, new − incumbent in points (95% cluster-bootstrap interval), VOCR5-N with its cascade,
cross-validated, Rubi's display path. Bold: the interval is entirely below 0.

| | 1 MB `vocr_lm6_o3_1000k` | 4 MB `vocr_lm6_o3_4000k` |
|---|---|---|
| hval page path | **−0.72 (−2.15 to −0.05)** | **−0.46 (−1.38 to −0.01)** |
| hval page path without PCSG00854 | **−0.10 (−0.17 to −0.03)** | **−0.05 (−0.11 to −0.002)** |
| hval clean lines | **−0.08 (−0.14 to −0.03)** | −0.05 (−0.10 to +0.004) |
| unseen-game dev, whole dialogue | −0.04 (−0.14 to +0.05) | +0.01 (−0.08 to +0.10) |
| unseen-game dev score | −0.03 (−0.10 to +0.04) | +0.01 (−0.05 to +0.07) |
| unseen-game dev, #129 benchmark (Rubi's finder) | +0.10 (−0.18 to +0.46) | not run |
| objective (0.5 hp + 0.5 dev score) | **−0.37 (−1.09 to −0.02)** | −0.22 (−0.69 to +0.01) |
| objective without PCSG00854 | **−0.06 (−0.11 to −0.01)** | −0.02 (−0.06 to +0.02) |
| game text − no game text (B − A), dev score | −0.01 (−0.09 to +0.06) | **−0.05 (−0.09 to −0.01)** |

- **Read as "measurably better on each set":** not met at either tier. hval is resolved better; the unseen-game dev
  sets are level, with no resolved gain or loss.
- **Read as plan section 7's adoption rule** ("better on hval page-path dialogue CER AND not worse on the
  unseen-game dev sets"), which the same rules name for every VOCR6 decision and which this round's (a) and (b)
  encode: met at both tiers.
- **The ruling.** The audit (audit-vocr6-lm, B1) left the reading to the rule's author. The conductor ruled at
  2026-10-26 16:38Z that the bullet means plan section 7's rule, frozen before any score: "better on hval and not
  worse on the unseen-game dev sets (game-bootstrap 95% interval)". Both tiers ship, 4 MB included ("users only get
  it by choosing 4000k"); 0.5 MB and 2 MB stay ctclm.2. The ruling is in `vocr6/experiments.jsonl`, on #125 and in
  the VOCR6 rules.
- **In one line:** better on held-out pages of the hooked games, level on unseen games (no measurable gain or
  loss); measured on development data on the host.

## How it was measured

- **Models.** VOCR5-S/N/L (v0.3.0), W8I through the C runtime, head gate off (what Rubi main runs), top-8 extras.
  Primary: VOCR5-N with its cascade to VOCR5-L; S (cascading to N) and L (no cascade) must not be resolved worse.
- **Line path** (Rubi `core/vocr_ocr.c` `finish_line`, modelled by `eval/headgate_eval.Decoder.rubi`). An empty
  line stays empty. Then the counter re-decode, the beam + LM, and the counter text rule. A line whose greedy
  confidence is under the cascade threshold shows the next size's reading of the same crop. After amendment 1
  comes what Rubi shows: the job's lines through Rubi's own `core/textnorm.c` (built from Rubi main `bd130a8`;
  its border rule drops lines like `ーニーー…`), then the Shell's non-Japanese line filter. Without amendment 1,
  the harness reproduces `eval/headgate_dev_20261023.json`'s scores exactly.
- **Sets.**
  - dev: VOCR5's development cohorts, from the head gate's cache: tv, dn (both views) and wd (GRAD and GRAD +
    refine, three region margins). Two VNs are left out: v12188 is a hooked train title, v18432 the hooked test
    title.
  - hval: the ingest's screening pages (`splits_v1.json`, 4% of each train title's pages in blocks with 15-page
    guard bands). hc is its verified dialogue lines (3,170); hp is its whole-dialogue regions through GRAD and
    GRAD + refine (12,654 region reads of 46 titles). PSP pages are in the Adrenaline view, upscaled 2× bilinear.
  - The LMs screened on hval never saw hval or guard pages (`lm/hook_text.py --keep train`). Text repeats across
    pages, though. 837 of the 6,327 hval regions (13%; 1,674 of the 12,654 region reads) and 165 of the 3,170 clean
    lines (5.2%) hold a line whose text (`eval/metrics.norm`) is also a train line. Those lines are 4.2% of the page
    path's reference characters; PCSG01110 has 90% of its page-path characters in such regions. Without those units
    the 1 MB result stands: hval page path −0.79 (−2.38 to −0.05), and −0.10 (−0.18 to −0.03) without PCSG00854 as
    well. At 4 MB, without both, it is no longer resolved: −0.05 (−0.12 to +0.001).
- **Score.** Dialogue CER. The dev score is 0.25 tv + 0.25 dn + 0.5 wd (VOCR5's). The objective is 0.5 hp + 0.5 dev
  score. Intervals: 10,000-resample cluster bootstraps (VNs for dev, titles for hval).
- **Selection.** Per LM tier, over each family's recipes, the grid (alpha 0.3-0.8, beta 1.0-5.0 nats, flags
  0-3) and cascade thresholds 500/600/700, 2-fold cross-validated by cluster. The choice made on one fold scores
  the other fold. The adoption rule:
  - (a) hp resolved better;
  - (b) dev score not resolved worse;
  - (c) S and L not resolved worse;
  - (d) bytes and workspace within Rubi's limits.
- **Amendment 1** (10:05Z, before any cross-validated number). Add Rubi's display rules to the line path. On the
  original path, PCSG00854's junk lines made up about half of all hval page-path errors and pulled the in-sample
  choice to the grid's lowest per-character bonus, with clean lines and dev worse. Seen before it, all in-sample:
  its record lists tune3's grid; tune1 (09:22-09:28Z, dev only, original path) had also given in-sample dev scores
  over the whole grid for greedy, the incumbent, ctclm.2's 1 MB LM retuned and nine new 1 MB recipes (eight with
  game text).
- **Amendment 2** (10:22Z, conductor and owner directive). Decide two families separately: A without game text, B
  with it. B ships instead of A only if, cross-validated, B − A is resolved better on hp and on the objective, the
  dev score is not resolved worse, and S and L are not resolved worse. B stays at order 3 unless a higher order is
  resolved better. Its record lists nothing seen before it. On the amended path, tune4 had given the in-sample
  objectives of greedy, the four incumbents and the no-game-text rebuild of ctclm.2's 1 MB recipe (10:09-10:13Z),
  and the first no-game-text recipe's grid (`v6b_1000k`) had just finished (10:17:46Z). No game-text recipe had
  been decoded on that path (the first, 10:31:00Z), and no cross-validated number existed for either amendment (the
  first selection, 1000k, by 11:31:54Z; the 13:15Z run of every tier rewrote its files).
- **Two protocol dates were corrected without a note** in amendment 2's commit: `created` 09:55:00Z → 09:14:00Z,
  and amendment 1's `time` 10:20:00Z → 10:05:00Z. Both originals were later than the commits that carried them
  (09:16:54Z, 10:06:50Z). Nothing the rule reads changed.

## Results

Dialogue CER, %, VOCR5-N with its cascade (threshold 700 for the incumbent, the families' own choices otherwise);
greedy is VOCR5-N alone, without the LM and the cascade:

<!-- table:results -->
| LM tier | part | greedy | incumbent | A in-sample | A cross-validated | B in-sample | B cross-validated |
|---|---|---:|---:|---:|---:|---:|---:|
| 500k | hval page path | 4.220 | 3.983 | 3.456 | 4.164 | 3.280 | 3.345 |
| 500k | hval clean lines | 0.760 | 0.521 | 0.525 | 0.531 | 0.421 | 0.449 |
| 500k | dev score | 3.404 | 2.929 | 2.894 | 2.966 | 2.963 | 3.029 |
| 500k | dev whole dialogue | 5.926 | 5.273 | 5.235 | 5.334 | 5.298 | 5.383 |
| 500k | objective | 3.812 | 3.456 | 3.175 | 3.565 | 3.121 | 3.187 |
| 1000k | hval page path | 4.220 | 4.116 | 3.426 | 3.937 | 3.248 | 3.393 |
| 1000k | hval clean lines | 0.760 | 0.488 | 0.515 | 0.501 | 0.404 | 0.404 |
| 1000k | dev score | 3.404 | 2.929 | 2.831 | 2.915 | 2.869 | 2.904 |
| 1000k | dev whole dialogue | 5.926 | 5.294 | 5.190 | 5.288 | 5.212 | 5.253 |
| 1000k | objective | 3.812 | 3.522 | 3.129 | 3.426 | 3.058 | 3.149 |
| 2000k | hval page path | 4.220 | 3.962 | 3.363 | 3.784 | 3.257 | 3.711 |
| 2000k | hval clean lines | 0.760 | 0.480 | 0.474 | 0.488 | 0.398 | 0.409 |
| 2000k | dev score | 3.404 | 2.898 | 2.838 | 2.996 | 2.828 | 2.937 |
| 2000k | dev whole dialogue | 5.926 | 5.265 | 5.226 | 5.397 | 5.212 | 5.298 |
| 2000k | objective | 3.812 | 3.430 | 3.100 | 3.390 | 3.042 | 3.324 |
| 4000k | hval page path | 4.220 | 4.026 | 3.330 | 4.021 | 3.255 | 3.570 |
| 4000k | hval clean lines | 0.760 | 0.462 | 0.447 | 0.478 | 0.394 | 0.413 |
| 4000k | dev score | 3.404 | 2.889 | 2.832 | 2.947 | 2.826 | 2.901 |
| 4000k | dev whole dialogue | 5.926 | 5.269 | 5.172 | 5.354 | 5.208 | 5.277 |
| 4000k | objective | 3.812 | 3.457 | 3.081 | 3.484 | 3.041 | 3.235 |
<!-- /table:results -->

The rule, per tier and family (cross-validated differences from the incumbent, points, 95% interval):

<!-- table:rule -->
| LM tier | family | chosen (in-sample) | hp, CV − incumbent | dev score, CV − incumbent | (a) | (b) | (c) | (d) | adopted |
|---|---|---|---|---|---|---|---|---|---|
| 500k | A (no game text) | `repro_500k` a500 b2000 f1 c700 | +0.181 (+0.009 to +0.529) | +0.037 (-0.016 to +0.096) | no | yes | yes | yes | no |
| 500k | B (game text) | `h256rl_500k` a700 b3000 f1 c700 | -0.638 (-1.990 to +0.001) | +0.099 (+0.020 to +0.181) | no | no | no | yes | no |
| 1000k | A (no game text) | `v6b_1000k` a500 b2000 f1 c700 | -0.179 (-0.634 to +0.047) | -0.014 (-0.088 to +0.068) | no | yes | yes | yes | no |
| 1000k | B (game text) | `h64_1000k` a600 b2500 f1 c700 | -0.722 (-2.151 to -0.046) | -0.025 (-0.099 to +0.041) | yes | yes | yes | yes | yes |
| 2000k | A (no game text) | `v6b_2000k` a500 b2000 f1 c700 | -0.178 (-0.659 to +0.064) | +0.097 (+0.028 to +0.166) | no | no | no | yes | no |
| 2000k | B (game text) | `h16rl_2000k` a600 b2500 f1 c700 | -0.251 (-0.737 to +0.009) | +0.039 (-0.022 to +0.096) | no | yes | no | yes | no |
| 4000k | A (no game text) | `v6b_4000k` a600 b3000 f1 c700 | -0.005 (-0.084 to +0.056) | +0.058 (-0.006 to +0.124) | no | yes | no | yes | no |
| 4000k | B (game text) | `h64rlo4_4000k` a600 b2500 f1 c700 | -0.543 (-1.676 to -0.002) | +0.057 (-0.013 to +0.123) | yes | yes | yes | yes | yes |
| 4000k | B, order 3 | `h16rl_4000k` a600 b2500 f1 c700 | -0.456 (-1.382 to -0.014) | +0.012 (-0.048 to +0.071) | yes | yes | yes | yes | yes |
<!-- /table:rule -->

Family comparisons (cross-validated, points):

<!-- table:families -->
| LM tier | comparison | objective | hval page path | hval clean lines | dev score | ships |
|---|---|---|---|---|---|---|
| 500k | B − A | -0.378 (-1.224 to +0.025) | -0.819 (-2.520 to -0.038) | -0.082 (-0.134 to -0.033) | +0.063 (-0.008 to +0.135) | incumbent |
| 1000k | B − A | -0.277 (-0.774 to -0.028) | -0.543 (-1.526 to -0.068) | -0.098 (-0.144 to -0.052) | -0.011 (-0.094 to +0.065) | B |
| 2000k | B − A | -0.066 (-0.100 to -0.034) | -0.073 (-0.124 to -0.029) | -0.078 (-0.119 to -0.040) | -0.058 (-0.104 to -0.012) | incumbent |
| 4000k | B − A | -0.270 (-0.803 to -0.007) | -0.538 (-1.604 to -0.027) | -0.053 (-0.101 to -0.008) | -0.001 (-0.054 to +0.051) | B |
| 4000k | B (order 3) − A | -0.249 (-0.677 to -0.037) | -0.452 (-1.316 to -0.039) | -0.065 (-0.115 to -0.018) | -0.046 (-0.088 to -0.006) | B, order 3 |
| 4000k | B (order 4) − B (order 3) | -0.021 (-0.130 to +0.043) | -0.086 (-0.301 to +0.030) | +0.012 (-0.014 to +0.037) | +0.045 (+0.011 to +0.080) | B, order 3 |
<!-- /table:families -->

At 4 MB the B selection's own choice was order 4 (`h64rlo4_4000k`). Amendment 2's order clause compares it with
the best order-3 B: order 4 is not resolved better (objective −0.02, −0.13 to +0.04), and its dev score is resolved
worse (+0.045, +0.011 to +0.080). So the order-3 B ships.

Each model with the shipped LM and setting, against the incumbent (cross-validated, points):

<!-- table:models -->
| LM tier | model | hval page path | hval clean lines | dev score | objective |
|---|---|---|---|---|---|
| 1000k | VOCR5-S + cascade to N | -0.268 (-0.723 to -0.023) | -0.059 (-0.115 to -0.009) | +0.022 (-0.083 to +0.136) | -0.123 (-0.363 to +0.024) |
| 1000k | VOCR5-N + cascade to L | -0.722 (-2.151 to -0.046) | -0.084 (-0.137 to -0.030) | -0.025 (-0.099 to +0.041) | -0.374 (-1.090 to -0.025) |
| 1000k | VOCR5-L | -0.732 (-2.152 to -0.072) | -0.065 (-0.113 to -0.015) | +0.018 (-0.052 to +0.085) | -0.357 (-1.070 to -0.014) |
| 4000k | VOCR5-S + cascade to N | -0.351 (-1.045 to -0.007) | -0.047 (-0.100 to +0.004) | +0.025 (-0.060 to +0.111) | -0.163 (-0.520 to +0.027) |
| 4000k | VOCR5-N + cascade to L | -0.456 (-1.382 to -0.014) | -0.049 (-0.102 to +0.004) | +0.012 (-0.048 to +0.071) | -0.222 (-0.689 to +0.010) |
| 4000k | VOCR5-L | -0.471 (-1.394 to -0.028) | -0.047 (-0.091 to -0.004) | -0.009 (-0.065 to +0.044) | -0.240 (-0.706 to -0.010) |
<!-- /table:models -->

Vita and PSP (VOCR5-N + cascade, CER %):

<!-- table:platform -->
| LM tier | platform | hc incumbent | hc B (CV) | hp incumbent | hp B (CV) |
|---|---|---:|---:|---:|---:|
| 1000k | vita | 0.430 | 0.370 | 4.593 | 3.485 |
| 1000k | psp | 0.598 | 0.467 | 3.260 | 3.229 |
| 4000k | vita | 0.415 | 0.373 | 4.473 | 3.773 |
| 4000k | psp | 0.552 | 0.490 | 3.223 | 3.204 |
<!-- /table:platform -->

- **Names (secondary).** devnative's speaker names read much better: 8.66% → 2.89% (VOCR5-N, 1 MB, in-sample).
  The gain comes from the retune with the end-of-line term off, not from the game text or the start-of-line
  context. At the shipped weight and bonus, the flags none / BOS / EOS / both give 2.80 / 2.89 / 4.58 / 4.67% with
  the game-text LM, and 2.80 / 2.89 / 4.07 / 4.16% with ctclm.2's. teacher_val's name-like lines are slightly worse
  (2.91% → 3.19%).
- **Cascade threshold (report only).** At the shipped 1 MB setting, the objective is 3.13 / 3.08 / 3.06 / 3.03 /
  3.02 at thresholds 500 / 600 / 700 / 800 / 900. A higher threshold re-reads more lines, and Rubi's 2 s ceiling
  bounds that time, so 700 stays.
- **The original line path** (no display rules), for comparison. Nothing ships at any tier:

<!-- table:original -->
| LM tier | path | A chosen | B chosen | ships |
|---|---|---|---|---|
| 500k | original | `v6brl_500k` a600 b1000 f3 | `h256rl_500k` a800 b1000 f3 | incumbent |
| 1000k | original | `v6brl_1000k` a700 b1000 f3 | `h1024rl_1000k` a600 b1000 f3 | incumbent |
| 2000k | original | `v6b_2000k` a700 b1000 f3 | `h256rl_2000k` a600 b1000 f3 | incumbent |
| 4000k | original | `v6brl_4000k` a600 b1000 f3 | `h64rlo4_4000k` a800 b1000 f3 | incumbent |
<!-- /table:original -->

- **Final blobs.** The shipped blobs were rebuilt with hval folded back in (`hook_final`). On dev they are not
  resolved worse than the screening builds at the shipped setting: 1 MB N −0.006 points (−0.024 to +0.010), 4 MB N
  +0.001 (−0.010 to +0.011). hval cannot score them, since its text is now in them.

### On the shared page-path benchmark (#129, report only)

`eval/pagepath.py` reads whole dialogue with Rubi `bd130a8`'s finder (its `rubi` preset, frozen): TEXTBOX_RECOVERY
with its short-line probe, and the refinement on PSP titles. The rule above used VOCR5's GRAD and GRAD + refine
regions. This run is VOCR5-N with its cascade: the incumbent against the shipped 1 MB setting (`--lm-alpha 600
--lm-beta 2500 --lm-flags 1`). #135 has since made COLGAP the finder default that Rubi gets; its hval gain (3.875%
→ 3.292%, `docs/finder-fix.md`) and this LM's were measured separately, on the same baseline.
hval uses the screening build `h64_1000k`, since the shipped blob holds hval's text; the dev units (#119's) use the
shipped blob. The setting was chosen in-sample on this round's own hval, so the hval number here is optimistic.

| units | incumbent | new | difference, points (95% interval) |
|---|---:|---:|---|
| hval, 46 titles | 3.875% | 2.227% | −1.65 (−5.14 to −0.07) |
| hval without PCSG00854 | 2.330% | 2.205% | −0.12 (−0.20 to −0.06) |
| dev (#119's units, 93 VNs) | 3.036% | 3.137% | +0.10 (−0.18 to +0.46) |
| dev without v21004 | 3.072% | 3.037% | −0.04 (−0.23 to +0.17) |

The intervals are a 10,000-resample bootstrap over titles and VNs of the two runs' rows
(`select_extra/pagepath_check.json`), not `pagepath.py compare`'s, which differ in the third digit (hval −5.09 to
−0.08). #119's dev units include 21 of the two VNs this round's rule leaves out (v12188: 18, v18432: 3). Both arms
read them with 0 errors, so only the denominators move (450 of 25,854 reference characters).

v21004 is the mirror image of PCSG00854. Its box border is read as a junk line, and the old LM spelt it with ー and
ニ, which Rubi's border rule drops. The new one spells it as 35 ー, which the rule keeps, because it reads as a
pause: +35 errors in one unit. Both junk lines come from the finder; a finder that leaves box borders out
(vocr6-finder) removes both effects.

### Held-out cross-entropy (the screen)

Nats per character with `</s>`, on hval's lines and on the dev references (lower is better). The screen ordered
the recipes; decisions come only from decoding.

<!-- table:screen -->
| LM | bytes | hval lines, BOS | hval, no BOS | dev references, BOS | dev, no BOS |
|---|---:|---:|---:|---:|---:|
| `old_1000k` | 987,600 | 3.493 | 3.658 | 3.904 | 4.045 |
| `v6b_1000k` | 986,416 | 3.495 | 3.659 | 3.912 | 4.054 |
| `h1_1000k` | 975,872 | 3.427 | 3.600 | 3.894 | 4.036 |
| `h4_1000k` | 985,168 | 3.346 | 3.533 | 3.862 | 4.013 |
| `h16_1000k` | 979,872 | 3.234 | 3.440 | 3.817 | 3.980 |
| `h64_1000k` | 987,216 | 3.153 | 3.368 | 3.810 | 3.976 |
| `h16rb_1000k` | 984,112 | 3.274 | 3.479 | 3.827 | 3.994 |
| `h16rl_1000k` | 986,208 | 3.235 | 3.438 | 3.807 | 3.972 |
| `h16o4_1000k` | 984,624 | 3.311 | 3.542 | 3.896 | 4.076 |
| `h64rl_1000k` | 981,872 | 3.152 | 3.369 | 3.798 | 3.969 |
| `h256_1000k` | 986,128 | 3.139 | 3.356 | 3.833 | 3.996 |
| `h256rl_1000k` | 987,488 | 3.138 | 3.356 | 3.824 | 3.990 |
| `h1024rl_1000k` | 987,584 | 3.145 | 3.362 | 3.848 | 4.010 |
<!-- /table:screen -->

## The game text: what it adds and what it keeps

Measured with `lm/memorization.py`. Stored n-grams are checked against the public text the builder reads (the five
corpora, their visual-line re-cut and the Jiten words). Greedy runs are measured on the hooked train lines the blobs
were built from, and on hval's lines. Those are lines of held-out pages, not unseen text: 251 of hval's 3,302 lines
(7.6%) are also train lines, verbatim. A greedy run starts at a position of a line and follows the LM's own most
probable next character for as long as it matches the line.

| LM | game text | stored bigrams / trigrams | of them only in the game text | longest greedy run on its own training lines | lines with a run of ≥ 8 / ≥ 12 / ≥ 20 | longest run on held-out lines |
|---|---|---:|---:|---:|---|---:|
| ctclm.2, 1 MB | none | 70,817 / 121,536 | 0 / 0 | 10 | 45 / 0 / 0 | 9 |
| family A, 1 MB (`v6b_1000k`) | none | 70,436 / 120,116 | 0 / 0 | 10 | 45 / 0 / 0 | 9 |
| family A, 4 MB (`v6b_4000k`) | none | 236,941 / 541,242 | 0 / 0 | 10 | 46 / 0 / 0 | 9 |
| screening `h64_1000k` | ×64 | 66,089 / 131,980 | 384 / 2,286 | 22 | 97 / 2 / 1 | 10 |
| screening `h16rl_4000k` | ×16 | 207,304 / 670,744 | 3,857 / 56,366 | 22 | 109 / 2 / 1 | 10 |
| shipped `vocr_lm6_o3_1000k` | ×64 | 66,123 / 130,260 | 416 / 2,551 | 22 | 112 / 4 / 1 | (hval is in its text) |
| shipped `vocr_lm6_o3_4000k` | ×16 | 199,552 / 648,198 | 4,156 / 61,011 | 22 | 116 / 4 / 1 | (hval is in its text) |

- **What is kept.** Order 3 stores no fragment longer than 3 characters. The fragments found only in the game text
  are 1.5% of the shipped 1 MB blob's bigrams and trigrams, and 7.7% of the 4 MB blob's, which prunes less. That
  is not how much of the blob the game text changed. 26% of the 1 MB blob's bigrams and trigrams (51,578 of 196,383;
  trigrams alone 32%) are not kept by the same recipe without game text (`v6b_1000k`), and 19% of the 4 MB blob's
  (164,203 of 847,750) are not kept by `v6brl_4000k`. The game counts lift public n-grams over the pruning
  threshold, and they raise the threshold itself (136 against 50 at 1 MB, 11 against 8 at 4 MB).
- **What greedy continuation recovers.** The longest runs of the shipped blobs are a repeated small kana (ぁ × 22,
  ぇ × 14-15) and a game's title inside its own text (12 characters). No other training line gives a run of 12 or
  more; the LMs without game text reach 10, on common phrases. On held-out lines of the same games, the screening
  builds reach at most 10, as the LMs without game text reach 9.
- **Pruning: order.** Order 3 is the smallest order built in this round and in ctclm.2's. The decoder accepts
  orders 2-6 (`runtime/vocr_lm.c`, `lm/vlm.py`); order 2 was not tried. Order 4 was not resolved better (4 MB,
  above).
- **Pruning: size.** The byte budgets are Rubi's tiers, no larger than before.
  - The 1 MB blob's count threshold is 136 (ctclm.2: 49). A trigram the public text does not have stays only if it
    occurs at least three times in the game's distinct lines (64 each).
  - The 4 MB blob's threshold is 11 at weight 16, so it keeps every trigram of the game text.
  - At the 4 MB tier, the 3.6 MB game-text blob is not better than the 1 MB one. VOCR5-N, the selections'
    cross-validated errors on the same units, 4 MB − 1 MB: hval page path +0.18 (−0.02 to +0.59), dev score −0.00
    (−0.05 to +0.05), objective +0.09 (−0.02 to +0.29). L is level too; only S is better with the 4 MB blob
    (objective −0.06, −0.13 to −0.01).
  - The 1 MB blob against the 4 MB incumbent: hval page path −0.63 (−1.97 to −0.005), dev score +0.01 (−0.05 to
    +0.08).
  - The 4 MB blob holds 10× the 1 MB one's game-only bigrams and 24× its trigrams (4,156 / 61,011 against 416 /
    2,551). The conductor's ruling ships it under the same rule as the 1 MB one, since users get it only by
    choosing the 4000k tier.

**Licence.** The blobs stay CC BY-SA 4.0, as their open sources (Jiten, JESC) require. The game text enters only as
character n-gram counts, as the recognizers' weights carry what they learnt from the owner's screenshots, and the
release's `ATTRIBUTION.md` discloses it in the same way. The VOCR6 rule of 2026-10-26: "LM from game hook text:
allowed. Ship it if it's measurably better on the page path (unseen-game dev sets AND hval) [...]; attribution as
for the models trained on game data." The LMs are better on hval and level on the unseen-game dev sets. They ship
under the conductor's ruling of 16:38Z that this means plan section 7's rule ("The owner's condition", above). The
games' text belongs to its publishers, and no running text is in the blobs.

## The dictionary-aware variant

The prototype adds a bonus when a prefix of the beam ends in a word of a lexicon. The lexicon is the JMdict forms
in Rubi's pinned `JMdict-plain.zip`, ranked by the Jiten global list Rubi ships (`jiten_freq_global.zip`), 2-8
characters, in vocabulary (`lm/lexicon.py`). The bonus is `bonus × (L − 1)` for the longest word length `L`. It ran
in `lm/vlm.py`'s Python twin of the decoder (`decode(lexicon=…)`), on the shipped 1 MB LM and setting, over a
4 × 3 grid: bonus 0.25-2 nats, words of Jiten rank up to 20k / 100k / 300k.

- **CER.** Every setting is worse than the plain LM (objective 3.067-3.439 against 3.058, VOCR5-N + cascade,
  in-sample). The cross-validated choice (bonus 0.25 nat, rank ≤ 20k) is not resolved better on hp, and S or L is
  resolved worse. Not adopted, so no C code was written.
- **Cost if it were built (estimates).** Memory: a reversed trie of 33k / 155k / 355k nodes for the 20k / 100k /
  300k lists, about 0.23 / 1.1 / 2.5 MB at 7 B a node. Time: about 200 trie steps a line (VOCR5-N), next to the
  beam's 158 LM calls a line. Scaled from the decoder's est. 0.4-0.7 ms for those calls, that is est. +0.5-0.9 ms
  per 640-px line, within the protocol's +5 ms limit. Its CER is what rules it out.
- **Why it does not help.** The 3-gram LM and the Jiten word counts already in it cover most short words, and a
  word bonus also rewards wrong splits that happen to form words.

## LM settings on VOCR6's logits (report)

vocr6-lead saw ctclm.2's LM at Rubi's settings raise VOCR6 stage-1 clean-line CER (A15: 0.298% → 0.391%) and asked
whether VOCR6's sharper logits need other settings. This run decodes two VOCR6 N recognizers, stage 2's adopted
S2cil and stage 1's A15 (two seeds each), with VOCR5-N as the control. Tier N alone, without the cascade, on Rubi's
line path with its display rules, on hval (the screening LM builds) and VOCR5's dev units. The grid is alpha
0.1-0.8, beta 0-5 nats, flags 0-3 and the confidence gate 2/4/8 nats, about 1,056 settings, 2-fold
cross-validated on the protocol's folds. It is a report, outside the protocol's decision. Machine-readable:
`eval/vocr6_lm_vocr6logits_20261026.json`.

| recognizer, seed | LM | greedy hc / hp / dev | LM at its VOCR5 setting: hc / hp / dev | hc, LM − greedy | objective, CV setting − VOCR5 setting |
|---|---|---|---|---|---|
| S2cil, 1 | ctclm.2 1 MB | 0.476 / 4.631 / 3.294 | 0.423 / 5.060 / 3.074 | −0.053 (−0.105 to +0.002) | +0.001 (−0.036 to +0.045) |
| S2cil, 3 | ctclm.2 1 MB | 0.437 / 4.088 / 3.355 | 0.390 / 3.945 / 3.182 | −0.047 (−0.109 to +0.009) | −0.012 (−0.046 to +0.021) |
| S2cil, 1 | game-text 1 MB | 0.476 / 4.631 / 3.294 | 0.358 / 3.392 / 3.047 | −0.117 (−0.171 to −0.067) | +0.394 (−0.040 to +1.332) |
| S2cil, 3 | game-text 1 MB | 0.437 / 4.088 / 3.355 | 0.345 / 3.496 / 3.185 | −0.092 (−0.152 to −0.035) | +0.110 (−0.012 to +0.351) |
| S2cil, 1 | game-text 4 MB | 0.476 / 4.631 / 3.294 | 0.333 / 3.485 / 2.995 | −0.143 (−0.203 to −0.085) | +0.307 (−0.006 to +0.961) |
| S2cil, 3 | game-text 4 MB | 0.437 / 4.088 / 3.355 | 0.317 / 3.499 / 3.097 | −0.119 (−0.186 to −0.059) | +0.047 (−0.005 to +0.119) |
| A15, 1 | ctclm.2 1 MB | 0.437 / 4.311 / 3.332 | 0.386 / 4.556 / 3.157 | −0.051 (−0.106 to +0.004) | −0.506 (−1.585 to −0.009) |
| A15, 3 | ctclm.2 1 MB | 0.443 / 3.913 / 3.368 | 0.396 / 3.808 / 3.121 | −0.047 (−0.097 to −0.002) | −0.285 (−0.863 to −0.010) |
| A15, 1 | game-text 1 MB | 0.437 / 4.311 / 3.332 | 0.339 / 3.334 / 3.110 | −0.098 (−0.158 to −0.043) | +0.064 (−0.074 to +0.317) |
| A15, 3 | game-text 1 MB | 0.443 / 3.913 / 3.368 | 0.349 / 3.229 / 3.127 | −0.094 (−0.156 to −0.037) | −0.016 (−0.052 to +0.027) |
| VOCR5-N | ctclm.2 1 MB | 0.760 / 4.220 / 3.404 | 0.564 / 3.814 / 3.136 | −0.196 (−0.341 to −0.084) | +0.045 (−0.029 to +0.159) |
| VOCR5-N | game-text 1 MB | 0.760 / 4.220 / 3.404 | 0.435 / 3.332 / 3.144 | −0.325 (−0.511 to −0.173) | +0.186 (−0.006 to +0.569) |

CER %, differences in points with 95% cluster intervals; "VOCR5 setting" is ctclm.2's (alpha 0.5, beta 3.5, EOS)
for ctclm.2's LM and the shipped one (alpha 0.6, beta 2.5, BOS) for the game-text LMs.

- **On equal terms the LM does not hurt VOCR6's clean lines.** The greedy arm above runs through Rubi's display
  rules too. Without them, A15 seed 1 reads hval's clean lines at 0.294% (150 errors), vocr6-lead's figure. The same
  greedy text through `core/textnorm.c` and the Shell's non-Japanese filter is at 0.437%. The rules drop 14
  dialogue lines made only of punctuation (the `……` kind), 75 reference characters. They do the same for every
  recognizer (`show_cost` in the JSON: +0.13-0.14 points for each). A greedy arm without the rules, compared with
  an LM arm with them, puts this cost on the LM. With the rules on both sides, the shipped game-text LMs lower
  VOCR6's clean-line CER by 0.09-0.14 points, resolved for each seed. ctclm.2's lowers it by about 0.05.
- **The gain shrinks as the recognizer sharpens.** The game-text 1 MB LM takes 0.33 points off VOCR5-N's clean
  lines and 0.09-0.12 off VOCR6's.
- **Retuning does not beat the shipped settings for the shipped LMs.** Cross-validated against the shipped setting:
  objective −0.02 to +0.39 on the game-text LMs, none resolved, most worse in point. A lower confidence gate (4
  nats) wins some in-sample bests but does not survive the cross-validation. ctclm.2's 1 MB LM would gain from a
  retune on A15 (−0.51 and −0.29), mostly on the hval page path (−0.95 and −0.53, with wide intervals), but it no
  longer ships at 1 MB.
- **So VOCR6 N keeps Rubi's per-tier settings** (1 and 4 MB alpha 0.6, beta 2.5, BOS; 0.5 and 2 MB ctclm.2's). The
  run repeats per recognizer in 10-15 minutes in one heavy slot (Reproduce). It is to be run on the final
  S/N/L blobs before they are pinned, and with the 0.5 and 2 MB LMs.
- **For the 1% target.** Rubi's display rules alone cost 0.14 points of hval clean-line CER, on lines that carry no
  word to look up. Keeping punctuation-only lines in the Shell's filter is a display choice for Rubi; box borders
  are handled by the border rule, not the filter.

## For Rubi

- New files: `vocr_lm6_o3_1000k.vlm` and `vocr_lm6_o3_4000k.vlm` (pre-release v0.3.7-lm6.2, CC BY-SA 4.0, with
  `ATTRIBUTION.md`, `SHA256SUMS` and the settings), cut from this PR's merge. v0.3.7-lm6.1 holds the same blobs. It
  went out from this PR's branch before the audit, with an `ATTRIBUTION.md` and notes that missed Little Busters!
  and called Collar x Malice a vita-test game, and is superseded.
- The 0.5 and 2 MB tiers keep ctclm.2's files and settings.
- `core/vocr_ocr.c`: the LM table carries each tier's alpha, beta and flags.

## Reproduce

```sh
P=$HERD/shared/private/vocr6-lm; A=$HERD/shared/artifacts/vocr6-lm; I=$HERD/shared/artifacts/vocr6-ingest
# hook text (train pages only; --keep final for the shipped builds)
.venv-ci/bin/python lm/hook_text.py export --root $HERD/shared/private/hooked-b2 --splits $I/splits_v1.json --keep train \
  --vocab data/vocab.txt --vocab data/vocab_vn_5631.txt --out $P/text/hook_train.txt --heldout $P/text/hval_lines.txt \
  --manifest $P/text/hook_train.json --test-vns $P/text/test_vns.txt
# an LM (the shipped 1 MB recipe; ctclm.2's own command still gives its bytes bit for bit)
.venv-ci/bin/python lm/build.py --out vocr_lm6_o3_1000k.vlm --manifest vocr_lm6_o3_1000k.json --bytes 987600 --order 3 \
  --jiten vn=0.4,novel=0.2 --corpora aozora_dialogue,aozora_narration,aratako_dsv3,jesc,tatoeba,hook_final:64 \
  --protect vocr6 --vocab-extra vn5631 --exclude-vns $P/text/test_vns.txt
#   4 MB: --bytes 3781072, hook_final:16, --recut lines
# hval reads (the ingest's pools and page pool), then the grid on both line paths, then the selection
.venv-ci/bin/python eval/vocr6_lm.py hval-units --pools $I/pools/v1 --pagepool $I/pagepool/v1 --pages $I/pages --out $P/hval_rubi
for t in S N L; do .venv-ci/bin/python eval/vocr6_lm.py hval-cache --tier $t --dir $P/hval_rubi; done
.venv-ci/bin/python eval/vocr6_lm.py tune --lm NAME=LM.vlm --grid GRID --tiers S,N,L --cascade 500,600,700,800,900 \
  --hval-cache $P/hval_rubi --rubi RUBI_CHECKOUT --both --out $P/tune4/NAME.npz
bash $A/select_all.sh        # select (families A, B; both paths) and families, per tier
.venv-ci/bin/python lm/memorization.py report --lm NAME=LM.vlm ... --train TEXT --heldout HELD --public PUBLIC --out MEMO.json
.venv-ci/bin/python eval/vocr6_lm.py lex-tune --lm ... --config SETTING --lexicon LEX.tsv --grid LEXGRID ...
.venv-ci/bin/python eval/vocr6_lm.py checks --select-dir $P/select --priv $P --hval-cache $P/hval_rubi \
  --pagepool $I/pagepool/v1 --tune4 $P/tune4 --lm-dir $P/lm --final-dir $P/final --pagepath $A/pagepath \
  --out $P/select_extra/checks.json     # the audit's re-derived numbers (the JSON's checks block)
.venv-ci/bin/python eval/vocr6_lm.py summary ... --checks $P/select_extra/checks.json \
  --out eval/vocr6_lm_dev_20261026.json
.venv-ci/bin/python eval/vocr6_lm.py tables --json eval/vocr6_lm_dev_20261026.json --doc docs/vocr6-lm.md
```

For another recognizer (VOCR6's), with its blob as tier N:

```sh
B=BLOB.vocr; T=$P/v6/tune_NAME
.venv-ci/bin/python eval/vocr6_lm.py dev-cache --tier N --blob N=$B --out $P/v6/dev_N_NAME.pkl   # VOCR5's dev reads
.venv-ci/bin/python eval/vocr6_lm.py hval-cache --tier N --blob N=$B --dir $P/hval_rubi          # reads_N.<stem>.pkl
.venv-ci/bin/python eval/vocr6_lm.py tune --tiers N --blob N=$B --dev-cache N=$P/v6/dev_N_NAME.pkl --cascade 700 \
  --hval-cache $P/hval_rubi --rubi RUBI_CHECKOUT --lm NAME=LM.vlm --grid GRID --out $T/NAME.npz  # and --lm greedy=none
.venv-ci/bin/python eval/vocr6_lm.py blob-report --name NAME --run 1=$T ... --lm NAME:ALPHA,BETA,FLAGS --out REPORT.json
.venv-ci/bin/python eval/vocr6_lm.py show-cost --blob N=$B --hval-cache $P/hval_rubi --rubi RUBI_CHECKOUT --out COST.json
```

`dev-cache` reproduces the head gate cache's reads of VOCR5-N bit for bit (8,233 reads checked). The LM and the
harness do not depend on the recognizer.

## Limits

- **The page-path gain rests mostly on one title.** Without PCSG00854's junk lines it is −0.10 points (resolved). The
  finder that reads the decoration as a line is T3's problem (vocr6-finder).
- **The hval pages are the train titles' own games.** The LM saw other passages of the same games, so hval
  measures "a game the LM knows". The unseen-game dev cohorts show no resolved gain or loss.
- **The dev references are labelled by engine agreement plus an assistant reviewer**, not hand-checked.
- **Two-fold cross-validation with a 216-setting grid is noisy.** The two folds chose different recipes or settings
  at every tier, in both families. What failed at 0.5 and 2 MB is the rule (rule table). At 0.5 MB, B missed (a),
  (b) and (c), and A missed (a). At 2 MB, B missed (a) and (c), and A missed (a), (b) and (c). An in-sample choice
  would look better than the cross-validated numbers show.
- **Seen titles.** Three hooked train titles are related to a protected set, and their text is in the new LMs:
  - Little Busters! Converted Edition (PCSG00007; VNDB v5 by title, unmapped in its metadata) shares a release
    with CLANNAD. 594 of its lines (9,362 characters) are in the screening text, 648 (10,292) in the shipped blobs.
  - planetarian (ULJM05383, v34) shares a release with CLANNAD: 2,467 lines (2,672 in the shipped blobs).
  - Collar x Malice (PCSG00866, v18160) shares a release with vita-test's Collar x Malice -Unlimited- (v21789):
    825 lines (896).

  Any vita-test, psp-test or CLANNAD score decoded with these LMs must disclose them. The Jiten caveat of
  `docs/ctc-lm-decode.md` still holds. The protocol's `seen_flags` (frozen) list only the last two, and call
  Collar x Malice a vita-test game.
- **A seen series in dev.** Haitaka no Psychedelica (v18643, 74 dev units) is of the same series as a hooked train
  title, Kokuchou no Psychedelica (PCSG00468, v15771 by title, unmapped in its metadata). So the unseen-game dev set
  holds one VN of a series whose text is in the LMs. Without it the dev score difference is unchanged: −0.03 (−0.10
  to +0.04) at 1 MB, +0.01 (−0.05 to +0.07) at 4 MB.
- **No console run.** Every time and memory figure for the console is an estimate.
