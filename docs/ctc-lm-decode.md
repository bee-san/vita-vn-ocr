# A character LM and CTC prefix beam search for the console runtime

The recognizers decode greedily: each CTC frame's top class, repeats collapsed. Most dialogue errors that remain
are a class the network nearly got right: a similar kanji (蓮/達, 題/選), a dakuten (パ/バ, て/で), a small kana
(ゃ/や), a punctuation mark (。/、, 』/」, ー/一) (`docs/model-tiers.md` §2.1). This change reads the same network
output with a beam search over each frame's best few classes and scores the candidates with a small character
n-gram language model, in plain C on the console: `runtime/vocr_lm.c`, no malloc, integer scores, an LM blob of
0.5, 1, 2 or 4 MB used in place.

**Everything here is development data or an estimate.** The tuning and every score below read only teacher_val,
devnative v3 and the owner pool (the owner's 2026-10-14 13:25Z rule: only CLANNAD, vita-test and psp-test are test
sets; none of them was read). Dialogue CER is the primary metric and names a secondary column (13:57Z). Console
times are `vita/a9_cycles.py` estimates; nothing has run on a console.

Machine-readable: `docs/ctc-lm-decode.json`, generated whole by `eval/ctclm.py summary`; every table below is
`eval/ctclm.py tables --doc docs/ctc-lm-decode.md` of it. API: `runtime/README.md` ("Top-K, confidence and the
beam + LM decoder").

## Answers first

- **Dialogue CER falls by 30–42 % at every LM size, for every model.** Pooled over the six recognizers (VOCR3 and
  VOCR4 FL10/H15/W20), development dialogue errors go from 1,656 to 1,156 / 1,044 / 996 / 964 with the 0.5 / 1 /
  2 / 4 MB LM (CER 0.93 % → 0.65 / 0.59 / 0.56 / 0.54 %). Every model's interval lies below zero at every size, and
  the 2-fold VN cross-validated CER (settings chosen on one half of the VNs, scored on the other) is within 0.05
  points of the in-sample one. With the 1 MB LM, VOCR3-W20 reads dialogue at 0.42 % (0.72 % greedy; 0.43 %
  cross-validated) and VOCR4-FL10, the Small model, at 0.71 % (1.13 %; 0.73 %).
- **The families the task named shrink most** (4 MB, six models summed): kanji↔kanji substitutions 527 → 298,
  dakuten 197 → 58, small kana 54 → 26, ー/一 and other dash pairs 16 → 5, deletions 238 → 140. Punctuation pairs
  stay level (79 → 75) once the decoder pairs brackets; insertions rise (44 → 81), the price of the per-character
  bonus that cuts the deletions.
- **Names improve too (secondary):** name errors fall at every size and for every model (pooled 960 → 841 at
  1 MB). A tighter gate protects nothing: at 0.5 nat both dialogue (1,459 errors) and names (912) are worse than at
  8 nats (1,044 / 841). What the LM does break is listed below: per character it fixes about 4.6 for each one it
  breaks in dialogue and 3.5 in names; rare kanji (Jiten rank > 2,000) come out even (dialogue: 5 broken, 4 fixed
  of 762; names: 9 broken, 5 fixed of 564), and ＝ between katakana (a name separator) turns into ー (12 times in
  dialogue at 1 MB).
- **One setting serves all six models** at each LM size: beam 8, top-8 candidates per frame, LM weight 0.5, an
  end-of-line term, a gate and a bracket-pairing cost of 8 nats each, and a per-character bonus of 3–3.5 nats.
- **Cost (estimates):** the search itself adds 0.1–0.5 ms to a 640-px line and the top-8 tracking in the output
  head 2.4–7.3 ms, next to the network's 93–409 ms (444 MHz). A 3-line textbox stays at 0.29–0.54 s for FL10
  (Small ≤ 1 s) and 0.71–1.25 s for W20 (Normal ≤ 2 s, Large ≤ 3 s). Memory is the LM blob (resident, read-only:
  498,496 / 996,960 / 1,991,184 / 3,817,696 B), a 35–41 KB decoder workspace and 6.6 KB of top-K buffers. The
  first use of an LM in a session costs 28–459 ms; the optional CRC check is 89–95 % of its instructions.
- **Greedy is unchanged and recoverable bit for bit:** the top-K extras change neither the logits nor the greedy
  text nor the arena, and the decoder with `alpha = beta = bracket = 0` returns the greedy text exactly (tested on
  random lines, generated images and all 3,223 development lines of all six models).

## Results (development data only)

<!-- table:settings -->
The recommended settings (`vocr_lm_opts`; alpha in 1/1000, beta, gate and bracket in 1/1000 nat), one per LM tier, shared by all six models:

| tier | LM file | bytes | beam | top-K | alpha | beta | gate | bracket | flags |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0.5MB | `vocr_lm_o3_500k.vlm` | 498,496 | 8 | 8 | 500 | 3000 | 8000 | 8000 | VOCR_LM_EOS |
| 1MB | `vocr_lm_o3_1000k.vlm` | 996,960 | 8 | 8 | 500 | 3500 | 8000 | 8000 | VOCR_LM_EOS |
| 2MB | `vocr_lm_o3_2000k.vlm` | 1,991,184 | 8 | 8 | 500 | 3000 | 8000 | 8000 | VOCR_LM_EOS |
| 4MB | `vocr_lm_o3_4000k.vlm` | 3,817,696 | 8 | 8 | 500 | 3500 | 8000 | 8000 | VOCR_LM_EOS |
<!-- /table:settings -->

### Dialogue (primary)

<!-- table:dialogue -->
Dialogue CER, greedy → beam + LM (2-fold VN cross-validated in brackets):

| model | dialogue chars | greedy | 0.5MB LM | 1MB LM | 2MB LM | 4MB LM |
|---|---:|---:|---:|---:|---:|---:|
| VOCR3-FL10 | 31,356 | 1.03% | 0.77% (0.76%) | 0.65% (0.67%) | 0.63% (0.66%) | 0.61% (0.63%) |
| VOCR3-H15 | 31,356 | 0.93% | 0.62% (0.64%) | 0.58% (0.59%) | 0.55% (0.56%) | 0.55% (0.56%) |
| VOCR3-W20 | 31,356 | 0.72% | 0.46% (0.48%) | 0.42% (0.43%) | 0.38% (0.39%) | 0.37% (0.38%) |
| VOCR4-FL10 | 31,356 | 1.13% | 0.81% (0.81%) | 0.71% (0.73%) | 0.69% (0.70%) | 0.63% (0.63%) |
| VOCR4-H15 | 31,356 | 0.87% | 0.62% (0.64%) | 0.57% (0.58%) | 0.55% (0.56%) | 0.55% (0.58%) |
| VOCR4-W20 | 20,467 | 0.90% | 0.61% (0.65%) | 0.61% (0.61%) | 0.58% (0.59%) | 0.56% (0.56%) |
| pooled (six models) | 177,247 | 0.93% | 0.65% | 0.59% | 0.56% | 0.54% |
<!-- /table:dialogue -->

<!-- table:delta -->
Change in dialogue CER, points, with the 95% VN-cluster bootstrap interval (in-sample selection; cross-validated):

| model | 0.5MB | 1MB | 2MB | 4MB | 4MB CV |
|---|---|---|---|---|---|
| VOCR3-FL10 | -0.27 (-0.48 to -0.14) | -0.38 (-0.62 to -0.23) | -0.41 (-0.67 to -0.25) | -0.42 (-0.70 to -0.26) | -0.41 (-0.66 to -0.26) |
| VOCR3-H15 | -0.31 (-0.51 to -0.19) | -0.35 (-0.55 to -0.24) | -0.38 (-0.59 to -0.26) | -0.39 (-0.60 to -0.26) | -0.37 (-0.57 to -0.25) |
| VOCR3-W20 | -0.26 (-0.46 to -0.14) | -0.30 (-0.53 to -0.17) | -0.34 (-0.59 to -0.20) | -0.35 (-0.60 to -0.20) | -0.34 (-0.58 to -0.20) |
| VOCR4-FL10 | -0.32 (-0.55 to -0.18) | -0.42 (-0.70 to -0.25) | -0.44 (-0.73 to -0.27) | -0.50 (-0.82 to -0.31) | -0.50 (-0.82 to -0.31) |
| VOCR4-H15 | -0.25 (-0.42 to -0.15) | -0.30 (-0.49 to -0.19) | -0.33 (-0.54 to -0.20) | -0.32 (-0.53 to -0.20) | -0.29 (-0.48 to -0.17) |
| VOCR4-W20 | -0.29 (-0.49 to -0.13) | -0.29 (-0.49 to -0.14) | -0.32 (-0.52 to -0.17) | -0.34 (-0.53 to -0.19) | -0.34 (-0.52 to -0.20) |
| pooled | -0.28 (-0.45 to -0.17) | -0.34 (-0.54 to -0.23) | -0.37 (-0.58 to -0.24) | -0.39 (-0.60 to -0.26) | |
<!-- /table:delta -->

### Names and the console finder's crops (secondary)

<!-- table:secondary -->
Secondary views: names (CER) and the console finder's GRAD crops of devnative's dialogue lines:

| model | names greedy | names 0.5MB | names 1MB | names 2MB | names 4MB | GRAD greedy | GRAD 4MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| VOCR3-FL10 | 6.38% | 5.53% | 5.46% | 5.21% | 5.21% | 1.71% | 1.30% |
| VOCR3-H15 | 5.39% | 5.03% | 4.79% | 4.43% | 4.40% | 1.93% | 1.47% |
| VOCR3-W20 | 5.07% | 4.79% | 4.61% | 4.29% | 4.50% | 1.53% | 1.12% |
| VOCR4-FL10 | 6.35% | 5.74% | 5.56% | 5.60% | 5.50% | 2.02% | 1.43% |
| VOCR4-H15 | 5.64% | 5.07% | 4.71% | 4.68% | 4.75% | 1.89% | 1.38% |
| VOCR4-W20 | 5.21% | 4.86% | 4.68% | 4.22% | 4.22% | 1.22% | 1.12% |
<!-- /table:secondary -->

### Confusion families

<!-- table:families -->
Confusion families on dialogue lines, summed over the six models (eval/model_gaps.py's pair_kind; substitutions, then insertions and deletions):

| family | greedy | 0.5MB | 1MB | 2MB | 4MB |
|---|---:|---:|---:|---:|---:|
| kanji_kanji | 527 | 369 | 339 | 307 | 298 |
| dakuten | 197 | 84 | 67 | 61 | 58 |
| small_kana | 54 | 36 | 31 | 30 | 26 |
| hira_kata | 15 | 10 | 9 | 6 | 4 |
| dash_like | 16 | 10 | 3 | 5 | 5 |
| punct_punct | 79 | 76 | 75 | 73 | 75 |
| punct_other | 36 | 35 | 32 | 29 | 21 |
| other_sub | 450 | 293 | 260 | 257 | 256 |
| insertion | 44 | 55 | 81 | 53 | 81 |
| deletion | 238 | 188 | 147 | 175 | 140 |
| **all** | 1656 | 1156 | 1044 | 996 | 964 |
<!-- /table:families -->

<!-- table:families_model -->
Per model at the 1MB tier (greedy → beam):

| model | kanji_kanji | dakuten | small_kana | punct_punct | dash_like | other_sub | deletion | insertion |
|---|---|---|---|---|---|---|---|---|
| VOCR3-FL10 | 94 → 65 | 40 → 13 | 9 → 5 | 17 → 22 | 5 → 0 | 86 → 50 | 54 → 31 | 8 → 11 |
| VOCR3-H15 | 96 → 52 | 37 → 11 | 12 → 6 | 15 → 11 | 1 → 0 | 79 → 56 | 40 → 27 | 8 → 13 |
| VOCR3-W20 | 77 → 44 | 26 → 10 | 12 → 5 | 7 → 7 | 1 → 0 | 58 → 32 | 31 → 15 | 6 → 11 |
| VOCR4-FL10 | 108 → 72 | 43 → 19 | 9 → 7 | 17 → 19 | 6 → 2 | 98 → 46 | 49 → 34 | 9 → 14 |
| VOCR4-H15 | 82 → 59 | 29 → 7 | 9 → 7 | 18 → 10 | 1 → 0 | 83 → 45 | 37 → 23 | 8 → 23 |
| VOCR4-W20 | 70 → 47 | 22 → 7 | 3 → 1 | 5 → 6 | 2 → 1 | 46 → 31 | 27 → 17 | 5 → 9 |
<!-- /table:families_model -->

### What the LM breaks: the guard

The gate is the guard the task asked for: a frame whose top-1 leads its top-2 by at least `gate` keeps its top-1,
so the LM can only re-rank characters the network was unsure of. On development data the best gate is wide (8
nats, a probability ratio of about 3,000). Most of the gain comes from frames with margins under 2.5 nats (the
gate sweep: 560 of the 612 errors removed at 1 MB), it is flat from 8 nats on, and tighter gates cost dialogue and
names alike: the LM weight of 0.5 already keeps the LM from overruling a clear reading, so the gate never had to
protect names.

<!-- table:changes -->
What the LM changes, dialogue lines summed over the six models (characters fixed and introduced, lines better and worse; the last three columns: introduced-error lines with a rare kanji (Jiten rank > 2,000), mostly katakana, or Latin/digits in the reference):

| tier | lines changed | chars fixed | chars introduced | lines better | lines worse | rare kanji | katakana | Latin/digits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5MB | 850 | 699 | 199 | 575 | 191 | 21 | 31 | 40 |
| 1MB | 894 | 785 | 173 | 636 | 165 | 22 | 28 | 31 |
| 2MB | 900 | 818 | 158 | 662 | 156 | 19 | 32 | 32 |
| 4MB | 928 | 852 | 160 | 683 | 157 | 21 | 32 | 27 |
<!-- /table:changes -->

<!-- table:names_changes -->
The same for name lines:

| tier | lines changed | chars fixed | chars introduced | lines better | lines worse |
|---|---:|---:|---:|---:|---:|
| 0.5MB | 353 | 167 | 82 | 155 | 79 |
| 1MB | 380 | 201 | 82 | 186 | 80 |
| 2MB | 413 | 234 | 76 | 211 | 74 |
| 4MB | 421 | 241 | 87 | 218 | 83 |
<!-- /table:names_changes -->

<!-- table:by_class -->
Reference characters by class, summed over the six models at the 1MB tier: read correctly by the greedy text and by the beam, and the characters the beam broke (greedy right, beam wrong) and fixed:

| view | class | chars | greedy correct | beam correct | broke | fixed |
|---|---|---:|---:|---:|---:|---:|
| dialogue | hiragana | 100,874 | 100,488 | 100,717 | 36 | 265 |
| dialogue | kanji_common | 42,482 | 41,865 | 42,098 | 42 | 275 |
| dialogue | kanji_rare | 762 | 695 | 694 | 5 | 4 |
| dialogue | katakana | 11,301 | 11,095 | 11,208 | 16 | 129 |
| dialogue | latin_digit | 1,350 | 1,231 | 1,248 | 19 | 36 |
| dialogue | other | 64 | 51 | 31 | 20 | 0 |
| dialogue | punct | 20,414 | 20,210 | 20,288 | 42 | 120 |
| names | hiragana | 2,352 | 2,298 | 2,310 | 4 | 16 |
| names | kanji_common | 8,772 | 8,105 | 8,219 | 31 | 145 |
| names | kanji_rare | 564 | 516 | 512 | 9 | 5 |
| names | katakana | 4,566 | 4,421 | 4,449 | 10 | 38 |
| names | latin_digit | 198 | 188 | 186 | 2 | 0 |
| names | punct | 474 | 474 | 472 | 2 | 0 |
<!-- /table:by_class -->

<!-- table:gate_sweep -->
The gate from 0.5 to 64 nats at the o3j2_1000k settings otherwise (pooled errors over the six models; greedy 1656 dialogue / 960 names):

| gate (nat) | 0.5 | 1 | 1.5 | 2 | 2.5 | 3.5 | 5 | 8 | 12 | 20 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dialogue errors | 1459 | 1288 | 1197 | 1120 | 1096 | 1058 | 1051 | 1044 | 1044 | 1044 | 1044 |
| name errors | 912 | 889 | 872 | 858 | 847 | 843 | 841 | 841 | 841 | 841 | 841 |
<!-- /table:gate_sweep -->

<!-- table:gate_tiers -->
The gate at each tier's other settings, pooled errors over the six models (dialogue / names; greedy 1656 / 960):

| tier | gate 3.5 nat | gate 5 nat | gate 8 nat | gate 12 nat |
|---|---|---|---|---|
| 0.5MB | 1162 / 878 | 1161 / 875 | 1156 / 875 | 1156 / 875 |
| 1MB | 1058 / 843 | 1051 / 841 | 1044 / 841 | 1044 / 841 |
| 2MB | 1010 / 807 | 1000 / 804 | 996 / 802 | 996 / 802 |
| 4MB | 979 / 806 | 967 / 806 | 964 / 806 | 964 / 806 |
<!-- /table:gate_tiers -->

### Beam width and candidates per frame

A wider beam and more candidates help a little past 8 × 8 (1 MB: 1,044 → 1,027 errors at 16 × 16), at a search
cost that was not measured (it grows with beam × candidates). Beam 1 rescores each frame's candidates against one
prefix only and gets about a quarter of the gain (1,656 → 1,485).

<!-- table:ablation -->
Beam width × top-K at the 1MB tier's other settings (pooled dialogue errors, six models):

| beam | top-2 | top-4 | top-8 | top-16 |
|---|---:|---:|---:|---:|
| 1 | 1493 | 1494 | 1485 | 1485 |
| 2 | 1199 | 1179 | 1168 | 1167 |
| 4 | 1174 | 1101 | 1073 | 1069 |
| 8 | 1170 | 1079 | 1044 | 1035 |
| 16 | 1171 | 1076 | 1036 | 1027 |

Beam width × top-K at the 4MB tier's other settings (pooled dialogue errors, six models):

| beam | top-2 | top-4 | top-8 | top-16 |
|---|---:|---:|---:|---:|
| 1 | 1454 | 1443 | 1429 | 1426 |
| 2 | 1144 | 1111 | 1088 | 1086 |
| 4 | 1110 | 1026 | 999 | 995 |
| 8 | 1105 | 1002 | 964 | 957 |
| 16 | 1104 | 994 | 960 | 948 |
<!-- /table:ablation -->

<!-- table:oracle -->
Greedy dialogue substitutions whose reference character is among the emitting frame's top 8 (the ones a top-8 search can fix at all):

| model | substitutions | reference in the top 8 |
|---|---:|---:|
| VOCR3-FL10 | 262 | 210 (80%) |
| VOCR3-H15 | 245 | 207 (84%) |
| VOCR3-W20 | 189 | 164 (87%) |
| VOCR4-FL10 | 296 | 237 (80%) |
| VOCR4-H15 | 229 | 187 (82%) |
| VOCR4-W20 | 153 | 129 (84%) |
<!-- /table:oracle -->

### Cost and memory (estimates)

<!-- table:a9 -->
Estimated Cortex-A9 cost (vita/a9_cycles.py on the Vita build, development lines), ms per 640-px line (80 CTC frames) at 444 MHz, low–high (333 MHz in brackets):

| model | network (greedy) | + top-8 | + beam + 0.5MB LM | + beam + 1MB LM | + beam + 2MB LM | + beam + 4MB LM | LM load, once |
|---|---|---|---|---|---|---|---|
| VOCR3-FL10 | 92.5–175.9 (123.3–234.4) | 2.4–4.9 (3.2–6.5) | 0.2–0.3 (0.2–0.4) | 0.2–0.3 (0.2–0.4) | 0.2–0.3 (0.2–0.4) | 0.2–0.3 (0.2–0.5) | 0.5MB 28–59; 1MB 54–114; 2MB 107–225; 4MB 203–444 (CRC ≤ 95%) ms |
| VOCR3-W20 | 231.9–409.2 (309.2–545.5) | 3.6–7.3 (4.8–9.7) | 0.1–0.4 (0.2–0.5) | 0.1–0.4 (0.2–0.5) | 0.1–0.5 (0.2–0.6) | 0.1–0.5 (0.2–0.6) | 0.5MB 29–63; 1MB 55–120; 2MB 107–234; 4MB 204–459 (CRC ≤ 95%) ms |
| VOCR4-W20 | 231.9–409.2 (309.2–545.5) | 3.6–7.3 (4.8–9.7) | 0.1–0.4 (0.2–0.5) | 0.1–0.4 (0.2–0.5) | 0.1–0.5 (0.2–0.6) | 0.1–0.5 (0.2–0.6) | 0.5MB 29–63; 1MB 55–120; 2MB 107–234; 4MB 204–459 (CRC ≤ 95%) ms |

Per typical full textbox (3 lines of about 640 px) at 444 MHz, the network plus top-K plus beam + LM, estimated, against the tier limits (Small ≤ 1 s, Normal ≤ 2 s, Large ≤ 3 s):

| model | greedy box | with the 0.5MB LM | with the 1MB LM | with the 2MB LM | with the 4MB LM |
|---|---|---|---|---|---|
| VOCR3-FL10 | 0.28–0.53 s | 0.29–0.54 s | 0.29–0.54 s | 0.29–0.54 s | 0.29–0.54 s |
| VOCR3-W20 | 0.70–1.23 s | 0.71–1.25 s | 0.71–1.25 s | 0.71–1.25 s | 0.71–1.25 s |
| VOCR4-W20 | 0.70–1.23 s | 0.71–1.25 s | 0.71–1.25 s | 0.71–1.25 s | 0.71–1.25 s |
<!-- /table:a9 -->

How the estimate was made: `vita/a9_cycles.py measure` (the Vita compiler's machine code for `vocr.c` and
`vocr_lm.c` under qemu-arm with instruction counting and a simulated 32 KiB L1 / 2 MiB L2) on 40 development
dialogue lines (20 teacher_val, 20 devnative; 347 px and 61 frames per line on average) and on the same 40 twice,
with the network alone, with top-8, and with each tier's beam + LM. A component's per-line cost is the difference
of two runs on the same pack (instruction counts are deterministic) and the slope from 40 to 80 lines, so the
LM's one-time load drops out and is shown on its own; the cost model (`a9_cycles.model_mode`) turns the
differential instruction mix and cache misses into a low–high cycle range. The search is branchy integer code
with a few hundred kilobytes of LM touched per line; in the simulation its trie lookups stay in the caches (under
25 L1 and 4 L2 misses per frame). In a game the L2 is shared, so expect more misses than this.

<!-- table:memory -->
Memory per tier: the LM blob (resident while the game runs, read-only), the decoder workspace (vocr_lm_workspace_bytes, a 1,024-px line) and the caller's top-K buffers:

| tier | LM file | LM bytes | sha256 | workspace (FL10 / H15, W20) | top-8 buffers |
|---|---|---:|---|---|---:|
| 0.5MB | `vocr_lm_o3_500k.vlm` | 498,496 | `7014027b340454ce…` | 35,248 / 40,816 B | 6,656 B |
| 1MB | `vocr_lm_o3_1000k.vlm` | 996,960 | `ea439f8b18f5d593…` | 35,248 / 40,816 B | 6,656 B |
| 2MB | `vocr_lm_o3_2000k.vlm` | 1,991,184 | `c785c8c39ec2f1a2…` | 35,248 / 40,816 B | 6,656 B |
| 4MB | `vocr_lm_o3_4000k.vlm` | 3,817,696 | `3d2b801832f9f8e6…` | 35,248 / 40,816 B | 6,656 B |
<!-- /table:memory -->

### The greedy path's confidence

<!-- table:confidence -->
The greedy path's confidence (vocr_extra.conf_min over top-8) as a flag for a dialogue line with an error (greedy; AUROC, and the share of lines flagged / errors caught at three thresholds):

| model | lines (with an error) | AUROC | < 0.5 | < 0.7 | < 0.9 |
|---|---|---:|---|---|---|
| VOCR3-FL10 | 1,924 (206) | 0.882 | 4% / 30% | 23% / 71% | 42% / 94% |
| VOCR3-H15 | 1,924 (176) | 0.870 | 4% / 30% | 21% / 71% | 39% / 89% |
| VOCR3-W20 | 1,924 (141) | 0.859 | 3% / 27% | 22% / 70% | 39% / 91% |
| VOCR4-FL10 | 1,924 (221) | 0.886 | 5% / 33% | 21% / 71% | 41% / 94% |
| VOCR4-H15 | 1,924 (171) | 0.882 | 4% / 31% | 19% / 74% | 37% / 88% |
| VOCR4-W20 | 1,333 (107) | 0.904 | 5% / 41% | 21% / 79% | 40% / 94% |
<!-- /table:confidence -->

`conf_min` is what the overlay's confidence cascade (re-read a doubtful line with a larger model) needs: at a
threshold of 0.7 it flags about a fifth of the dialogue lines and catches 69–79 % of the lines with an error.

## How it works

### Top-K and confidence on the greedy path (`vocr.h`)

`vocr_extra.topk = K` makes `vocr_run()` keep each frame's K largest logits (`topk_logit`, `topk_class`; ties
keep the lower class first, so `[0]` is the argmax) and the blank's logit, in caller buffers: the arena and the
default path are untouched (the `topk = 0` head loop is the old one). It also writes `conf_min` and `conf_mean`,
the smallest and mean softmax-max probability over the frames that emit the greedy characters, computed over the
top K with a local float `exp` (no libm).

### The search (`vocr_lm.c`)

A Viterbi CTC prefix beam search: per frame, every kept prefix has a score ending in a blank and one ending in a
character, and it moves to

- the same prefix through a blank (`s + q`) or a repeat of its last character (`s_char + q`), and
- a longer prefix through a new character `c` (from either score; for its own last character only after a blank),
  scoring `s + q + alpha · log P_LM(c | the last two characters) + beta`,

where `q` is the candidate's logit minus the frame's top-1 logit (a log-probability difference; no softmax is
needed), rounded to 1/1024 nat and capped at 64 nats. A new prefix equal to a kept one is merged into it (max,
not sum: the score is the best alignment's). The beam keeps the `beam` best prefixes; ties keep the earlier one.
At the end of the line, `VOCR_LM_EOS` adds `alpha · log P(end | last two characters)`.

- **Candidates and the gate.** A frame's candidates are its top-1 and the other top-K classes (and the blank, from
  `blank_logit`) whose margin to the top-1 is below `gate`.
- **Exact greedy at weight zero.** Every non-top-1 candidate costs at least 1/1024 nat, so with `alpha = beta =
  bracket = 0` the greedy path is the unique best path, for any beam, top-K and gate.
- **Integers.** Scores are int32 in Q10 nats; the LM term per character is clamped to [−64, 0] nats and a line to
  1,024 frames, so no sum can overflow. The only float operations are the margin's subtraction and rounding (IEEE
  single precision, no contraction), the same on x86-64, ARM VFP and the PSP's FPU.
- **Brackets.** A character LM sees two characters back, so it pulls 』 and 】 towards the common 」. Each prefix
  carries its last two unclosed opening brackets (「『（【〈《〔［ and ASCII "("), and a closer that does not match the
  innermost open one costs `bracket`; a closer with nothing open on the line is free (its opener may be on the
  line before). The LM itself reads every pair as 「」, so the picture alone decides a pair's style. Both were added
  after reading the first rounds' errors: without them the LM introduced 45 』→」 substitutions at 1 MB, all but
  one with the 『 on the same line, and with the pairing cost alone it turned the opener 『 into 「 instead.
- **Memory.** The workspace holds the class → LM-token map (2 B per class), the bracket roles (1 B per class), a
  prefix tree of at most 1 + frames × beam nodes (parent, class, length, hash, bracket state, LM state) and the
  per-frame candidates: 35,248 B for FL10 (3,501 classes) and 40,816 B for H15/W20 (5,355) at beam 8, top-8, order
  3 and 1,024 px. `vocr_lm_workspace_bytes()` is exact; nothing else is allocated. `vocr_lm_decode()` polls
  `vocr_extra.cancel` every 8 frames.

`lm/vlm.py` is the specification: the same blob reader, integer scorer and search in Python, step for step.
`tests/test_ctclm.py` requires the C output to equal it on random lines (exact ties, with and without the blank's
logit, every option) and checks the scorer state by state; while tuning, C and Python also agreed on 300 + 491
development lines under seven settings.

### The LM (`lm/build.py`, `VOLM` blobs)

- **Text.** synth3's pinned text pools (`synth/synth3_text.py`'s display form, pieces of at most 60 characters,
  every character in `data/vocab.txt`): Aozora Bunko dialogue and narration, Aratako's synthetic role-play
  dialogue (MIT), JESC subtitles and Tatoeba, 1,829,895 pieces and 47.2 M characters. Each piece is `<s> … </s>`.
- **Jiten.** The Jiten frequency lists of visual novels (2,162 VN scripts, 241,004 usable words) and novels (3,161
  novels, 317,051 words) (CC BY-SA 4.0): each word of at least two in-vocabulary characters is counted as a
  boundary-delimited pseudo-sentence `max(1, round(S / rank))` times, with S set so the VN list carries 40 % and
  the novel list 20 % of the corpus' character count. The boundary token is for counting only; its n-grams are
  dropped from the model.
- **Model.** Character 3-grams, interpolated modified Kneser-Ney (Chen and Goodman 1998), pruned by count to fit
  the byte budget (bigram and trigram thresholds searched; every kept trigram's context is kept: 1/132/132 at
  0.5 MB, 1/49/49 at 1 MB, 1/18/18 at 2 MB, 1/7/7 at 4 MB), backoff weights recomputed over what is kept,
  log-probabilities and backoffs quantised to 256 levels per order (Q10 nats). Order 3 beat orders 4 and 5 at
  every size (recipe table): at a fixed budget the 4-gram entries cost more bigrams than they bring.
- **Blob.** `VOLM` version 1: a 64-byte header (magic, version, size, order, tokens, CRC-32, the vocabulary's
  offset, a sha256 prefix of the build manifest), an order table, and 16-byte-aligned sections: per order the
  entries' last token (u16), probability and backoff codes (u8), child ranges (u32) and two 256-entry codebooks;
  the vocabulary as code points. Order 1 is indexed by token; the children of an entry are contiguous and sorted,
  so a lookup is a binary search over a short range. The blob is read in place: load it in 32 KiB chunks or any
  other way into one 16-byte-aligned block and call `vocr_lm_load()`.
- **Reproducible.** The same pinned inputs (`lm/sources.json`, each file checked by sha256) give the same bytes
  (the tests build twice). The four release blobs were rebuilt from the committed builder and are byte-identical
  to the ones tuned except the 16-byte manifest prefix in the header, which the scorer never reads (the summary
  checks this).

<!-- table:recipes -->
LM recipes (round 2's grid; pooled dialogue errors at each LM's best shared configuration, cross-validated, names; held-out cross-entropy on realplus v2 labels):

| LM | bytes | order | corpora | Jiten share | held-out nats/char | dialogue | CV | names |
|---|---:|---:|---|---|---:|---:|---:|---:|
| mixb_500k | 498,752 | 4 | Aozora + Aratako + JESC + Tatoeba | vn 0.2, novel 0.1 | 4.220 | 1317 | 1331 | 892 |
| mixb_o3_500k | 498,048 | 3 | Aozora + Aratako + JESC + Tatoeba | vn 0.2, novel 0.1 | 4.075 | 1239 | 1255 | 888 |
| o3j2_500k | 498,496 | 3 | Aozora + Aratako + JESC + Tatoeba | vn 0.4, novel 0.2 | 4.085 | 1220 | 1238 | 878 |
| o3j4_500k | 498,944 | 3 | Aozora + Aratako + JESC + Tatoeba | vn 0.8, novel 0.4 | 4.114 | 1202 | 1231 | 884 |
| mixb_1000k | 996,720 | 4 | Aozora + Aratako + JESC + Tatoeba | vn 0.2, novel 0.1 | 4.013 | 1233 | 1254 | 878 |
| mixb_jiten2x_1000k | 998,944 | 4 | Aozora + Aratako + JESC + Tatoeba | vn 0.4, novel 0.2 | 4.003 | 1192 | 1204 | 867 |
| mixb_o3_1000k | 998,784 | 3 | Aozora + Aratako + JESC + Tatoeba | vn 0.2, novel 0.1 | 3.915 | 1123 | 1147 | 842 |
| o3j2_1000k | 996,960 | 3 | Aozora + Aratako + JESC + Tatoeba | vn 0.4, novel 0.2 | 3.934 | 1110 | 1127 | 838 |
| o3j4_1000k | 995,232 | 3 | Aozora + Aratako + JESC + Tatoeba | vn 0.8, novel 0.4 | 3.983 | 1108 | 1118 | 827 |
| o3j6_1000k | 995,488 | 3 | Aozora + Aratako + JESC + Tatoeba | vn 1.2, novel 0.6 | 4.028 | 1111 | 1122 | 818 |
| o3j4vn_1000k | 989,920 | 3 | Aozora + Aratako + JESC + Tatoeba | vn 1.2 | 3.981 | 1098 | 1112 | 826 |
| o4j4_1000k | 994,464 | 4 | Aozora + Aratako + JESC + Tatoeba | vn 0.8, novel 0.4 | 4.016 | 1188 | 1214 | 857 |
| mixb_2000k | 1,992,128 | 4 | Aozora + Aratako + JESC + Tatoeba | vn 0.2, novel 0.1 | 3.833 | 1122 | 1138 | 843 |
| o3_2000k | 1,985,456 | 3 | Aozora + Aratako + JESC + Tatoeba | vn 0.2, novel 0.1 | 3.818 | 1065 | 1070 | 817 |
| o3j2_2000k | 1,991,184 | 3 | Aozora + Aratako + JESC + Tatoeba | vn 0.4, novel 0.2 | 3.845 | 1053 | 1073 | 799 |
| o3j4_2000k | 1,992,912 | 3 | Aozora + Aratako + JESC + Tatoeba | vn 0.8, novel 0.4 | 3.900 | 1061 | 1065 | 784 |
| mixb_4000k | 3,862,368 | 4 | Aozora + Aratako + JESC + Tatoeba | vn 0.2, novel 0.1 | 3.711 | 1036 | 1056 | 787 |
| o3_4000k | 3,977,840 | 3 | Aozora + Aratako + JESC + Tatoeba | vn 0.2, novel 0.1 | 3.758 | 1020 | 1041 | 800 |
| o3j2_4000k | 3,817,696 | 3 | Aozora + Aratako + JESC + Tatoeba | vn 0.4, novel 0.2 | 3.800 | 1019 | 1040 | 794 |
| o3j4_4000k | 3,838,048 | 3 | Aozora + Aratako + JESC + Tatoeba | vn 0.8, novel 0.4 | 3.861 | 1035 | 1056 | 784 |
| o3j6_4000k | 3,962,432 | 3 | Aozora + Aratako + JESC + Tatoeba | vn 1.2, novel 0.6 | 3.911 | 1037 | 1037 | 790 |
| o4j2_4000k | 3,857,200 | 4 | Aozora + Aratako + JESC + Tatoeba | vn 0.4, novel 0.2 | 3.722 | 1041 | 1074 | 781 |
| o5j2_4000k | 3,969,296 | 5 | Aozora + Aratako + JESC + Tatoeba | vn 0.4, novel 0.2 | 3.767 | 1067 | 1098 | 800 |
<!-- /table:recipes -->

## Tuning protocol (development data only)

- **Data.** VOCR3's pinned copies of teacher_val (1,643 lines, 180 VNs) and devnative v3 (543 hand-cut ink crops
  and 443 console-finder GRAD crops of 43 VNs, all of them also teacher_val VNs), and the owner pool v1 (594 ink
  crops: planetarian, CHAOS;CHILD, Sharin no Kuni), each read once through the C runtime in W8I mode in the console
  polarity (crops as stored, dark text inverted) by each of the six selections: VOCR3 `rp_FL10_C_mix_pre128k_seed1`,
  `rp_H15_C_mix_seed2`, `rp_W20_C_mix_seed1` and VOCR4 `v4c_FL10_seed1`, `v4c_H15_seed1`, `v4m_W20_seed2` (blob
  sha256 in the JSON). Per frame the top-16 logits and the blank's were stored (`eval/ctclm.py extract`); every
  setting below re-decodes those through the C library.
- **Views.** Dialogue (primary): the lines of teacher_val, devnative ink and the owner pool that are not names, a
  name being devnative's `name` style, the owner pool's name flag, or in teacher_val (no style tags) a name-like
  line by the repo's rule (≤ 6 characters, no dialogue punctuation; this also drops UI labels). VOCR4-W20 trained
  on the owner pool (arm M), so its dialogue view leaves the owner pool out. Names: the name lines. GRAD: devnative's
  finder crops of the dialogue lines (a second, noisier crop of lines already in the view).
- **Search.** Beam 8 and top-8 throughout; six rounds over the LM weight, the per-character bonus, the gate, the
  start/end-of-line terms and the bracket cost (`docs/ctc-lm-decode.json` `selection.grids`). Rounds 2–3 compared 24
  LMs (recipe table); the recipe was then fixed to order 3 with the 2× Jiten share at every size, the best or within
  noise of it at each. Rounds 4–6 added the bracket state and folding and extended the grid past its edges; round 6
  is final.
- **Selection.** At each LM size, the one setting with the fewest dialogue errors summed over the six models
  (first minimum in grid order). It is in-sample; the cross-validated column re-selects on the VNs of one fold
  (sha256 of the VN id, two folds) and scores the other, so it carries the selection's own optimism.
- **Intervals.** 10,000 bootstrap resamples of VNs (a VN's teacher_val, devnative and owner lines together), 95 %
  percentile interval of the CER difference.

## For the overlay (vita-jp-overlay)

Per line: `vocr_run()` with `vocr_extra.topk = 8` (the greedy text, the top-K and `conf_min`), then
`vocr_lm_decode()` (the beam + LM text). Both texts are there for A/B logging. One LM blob serves every model: it
has its own character vocabulary, and `vocr_lm_init()` maps the model's classes to it once. Pick the tier by free
memory; the blob stays resident like the weights (`USER_RW`, read-only after load), and the workspace (≤ 41 KB)
and the top-K buffers (6.6 KB) belong to the job. Pin each tier by its size and sha256 (memory table); verify the
file's sha256 at load instead of `vocr_lm_check_crc()` if the job's first-use time matters (the CRC is 89–95 % of
it). The settings per tier are the first table above (`docs/ctc-lm-decode.json` `tiers.*.config`).

## Reproduce

```sh
A=$HERD/shared/artifacts; ART=$A/ctc-lm-decode
# 1. development extracts: six blobs, 3,223 lines each (tv, dn_ink, dn_grad, own)
.venv-ci/bin/python eval/ctclm.py extract --blob v3FL10=$A/synth3/vocr3/blobs/rp_FL10_C_mix_pre128k_seed1_w8.vocr \
    --blob v3H15=$A/synth3/vocr3/blobs/rp_H15_C_mix_seed2_w8.vocr --blob v3W20=$A/synth3/vocr3/blobs/rp_W20_C_mix_seed1_w8.vocr \
    --blob v4FL10=$A/vocr4-train/blobs/v4c_FL10_seed1_w8.vocr --blob v4H15=$A/vocr4-train/blobs/v4c_H15_seed1_w8.vocr \
    --blob v4W20=$A/vocr4-train/blobs/v4m_W20_seed2_w8.vocr --out $ART/extract
# 2. the LM tiers, about 10 minutes each
for b in 500000 1000000 2000000 4000000; do .venv-ci/bin/python lm/build.py --out $ART/release/vocr_lm_o3_${b%000}k.vlm \
    --manifest $ART/release/vocr_lm_o3_${b%000}k.json --bytes $b --order 3 --jiten vn=0.4,novel=0.2 \
    --corpora aozora_dialogue,aozora_narration,aratako_dsv3,jesc,tatoeba; done
# 3. round 6 for each model and tier (--in-sample-own for v4W20), then selection, report, ablation, gate sweep
GRID='{"alpha":[400,500,600],"beta":[2500,3000,3500,4000],"gate":[3500,5000,8000,12000],"flags":[2],"bracket":[0,1000,2000,3000,5000,8000,12000,20000,64000]}'
.venv-ci/bin/python eval/ctclm.py tune --extract $ART/extract/v3W20.npz --lm $ART/lm/o3j2_1000k.vlm --grid "$GRID" \
    --out $ART/tune/r6_v3W20_o3j2_1000k.npz
.venv-ci/bin/python eval/ctclm.py select --extract $ART/extract --tune $ART/tune --prefix r6_ --lm-dir $ART/lm \
    --lms o3j2_500k,o3j2_1000k,o3j2_2000k,o3j2_4000k --out $ART/select/final_r6.json
.venv-ci/bin/python eval/ctclm.py report --extract $ART/extract --select $ART/select/final_r6.json --lm-dir $ART/lm \
    --tune $ART/tune --out $ART/select/report_r6.json        # re-decodes; refuses if errors differ from the tuning
.venv-ci/bin/python eval/ctclm.py ablate --extract $ART/extract --select $ART/select/final_r6.json --tier 1MB \
    --lm-dir $ART/lm --out $ART/select/ablate_r6_1MB.json     # and --tier 4MB
# 4. A9: base, --topk 8 and each tier's settings, on the 40-line pack and the same twice ($ART/a9/run_final.sh)
.venv-ci/bin/python vita/a9_cycles.py measure --out $ART/a9/f_v3W20_lm1MB_dev40 --modes w8i --blob BLOB \
    --pack $ART/a9/dev40.bin --cli-args "--lm LM.vlm --beam 8 --topk 8 --alpha 500 --beta 3500 --gate 8000 --lm-flags 2 --bracket 8000"
.venv-ci/bin/python eval/ctclm.py a9 --dir $ART/a9 --out $ART/select/a9.json
# 5. the JSON and this doc's tables
$ART/select/summary_args.sh docs/ctc-lm-decode.json $ART/select/a9.json
.venv-ci/bin/python eval/ctclm.py tables --doc docs/ctc-lm-decode.md
```

The extracts, tuning files, LM builds and A9 runs are under `$HERD/shared/artifacts/ctc-lm-decode/` (private:
they hold development text).

## Limits

- Development data only, labelled by engine agreement plus an assistant reviewer, not hand-checked: some
  "errors" are label errors, and the LM may fix or make those too. The settings were chosen on the same data; the
  cross-validated column bounds that optimism for the settings but not for the design decisions taken after
  reading development errors (the bracket state and folding, order 3, the Jiten share).
- teacher_val and devnative share all 43 devnative VNs and 348 line ids (the same text from a scaled PC
  screenshot and a native console capture); the VN-cluster bootstrap keeps them together, but the pooled CER
  weights those texts twice.
- The Jiten lists are word frequencies over 2,162 VN scripts, which may include development or test titles; they
  hold no running text, but the LM knows VN vocabulary from them.
- Every console time is an estimate: the Vita build's instruction mix under qemu-arm with a simulated 2 MiB L2
  to ourselves. A game's share of the L2 and its DRAM traffic will raise the LM's misses. The decoder has not run
  on a Vita or a PSP (it compiles warning-free for both).
- The search sees a frame's top 8: 13–20 % of the greedy substitutions have the right character outside them and
  cannot be fixed (oracle table), and it cannot add a character the network never proposed.
- Lines are decoded one at a time: a closing bracket whose opener is on the previous line is never penalised,
  and the LM starts each line without context.
- A lexicon bonus from JMdict (the task's optional idea) is not implemented.

## Attribution of the LM asset

The `VOLM` blobs are derived data, distributed under **CC BY-SA 4.0** (Jiten's and JESC's ShareAlike terms) as
release assets only, never in the repository:

- Jiten frequency lists (visual novels, novels), https://jiten.moe/frequency-dictionaries, CC BY-SA 4.0.
- Aozora Bunko texts (public domain), via globis-university/aozorabunko-clean, CC BY 4.0.
- Aratako/Synthetic-Japanese-Roleplay-SFW-DeepSeek-V3-0324-20k, MIT.
- JESC: Japanese-English Subtitle Corpus (Pryzant et al. 2018), CC BY-SA 4.0.
- Tatoeba, https://tatoeba.org/, CC BY 2.0 FR.
