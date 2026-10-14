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

- **Dialogue errors fall by 30–41 % pooled over the six recognizers** (VOCR3 and VOCR4 FL10/H15/W20): from 1,656
  to 1,153 / 1,048 / 1,000 / 969 with the 0.5 / 1 / 2 / 4 MB LM (CER 0.93 % → 0.65 / 0.59 / 0.56 / 0.55 %). **Per model
  the drop is 26–49 %**, and from the 1 MB LM on at least 32 % for every model. Every model's interval lies below
  zero at every size, and the 2-fold VN cross-validated CER (settings chosen on one half of the VNs, scored on the
  other) is within 0.05 points of the in-sample one. With the 1 MB LM, VOCR3-W20 reads dialogue at 0.42 % (0.72 %
  greedy; 0.43 % cross-validated) and VOCR4-FL10, the Small model, at 0.71 % (1.13 %; 0.74 %).
- **Counting each crop once changes little:** the dialogue view scores 241 crops twice (devnative's copies of
  teacher_val's own native crops). Without the second copies the pooled CER is 1.01 % greedy → 0.71 / 0.64 / 0.61 /
  0.60 % (30–41 % fewer errors).
- **The families the task named shrink most** (4 MB, six models summed): kanji↔kanji substitutions 527 → 307,
  dakuten 197 → 58, small kana 54 → 26, ー/一 and other dash pairs 16 → 5, deletions 238 → 139. Punctuation pairs
  stay level (79 → 75) once the decoder pairs brackets; insertions rise (44 → 80), the price of the per-character
  bonus that cuts the deletions.
- **Names improve too (secondary):** name errors fall at every size and for every model (pooled 960 → 849 at
  1 MB). A tighter gate protects nothing: at 0.5 nat both dialogue and names are worse than at 8 nats (gate sweep).
  What the LM does break is listed below: per character it fixes about 4.6 for each one it breaks in dialogue and
  3.3 in names. Rare kanji (Jiten rank > 2,000) come out slightly worse (dialogue: 695 → 694 of 762 read right;
  names: 516 → 511 of 564), and ＝ between katakana (a name separator) turns into ー (12 of the 20 symbols the beam
  breaks in dialogue at 1 MB; the other 8 are /).
- **The protected titles' names are filtered out, and the development titles' names do not carry the gain.** The
  builder drops every corpus piece and Jiten word that names a character or title of a protected VN (vita-test,
  psp-test, their families and CLANNAD's: 206 VNs, 2,316 characters) unless the name is also a common word or
  surname; the Jiten VN list does count CLANNAD's script among its 2,162, so its ordinary word frequencies still
  carry that script's share (Limits). Dropping the 183 development VNs' names too leaves the result where it was
  (1,054 against 1,048 errors at 1 MB), and an LM without the Jiten lists is 4–5 % worse up to 2 MB and level at
  4 MB (ablation table).
- **One setting serves all six models** at each LM size: beam 8, top-8 candidates per frame, LM weight 0.5, an
  end-of-line term, a gate and a bracket-pairing cost of 8 nats each, and a per-character bonus of 3–3.5 nats.
- **Cost (estimates calibrated to the console):** every A9 estimate is scaled to the owner's measured
  VOCR3-H15 time on the console (5.0–5.2 ms per CTC frame, v0.8.0-test.1 logs). On that scale the beam + LM search
  costs 0.4–0.7 ms per 640-px line and the top-8 tracking in the output head 6–10 ms, against the network's
  227–246 ms (FL10), 400–416 ms (H15) and 568–573 ms (W20): a 3-line textbox goes from 0.68–0.74 to 0.70–0.76 s
  (FL10, Small < 1 s) and from 1.71–1.72 to 1.73–1.75 s (W20, Normal < 2 s). What the LM's size costs is its
  one-time load in a session (68–87 / 131–166 / 259–325 / 493–636 ms for the 0.5 / 1 / 2 / 4 MB tier, 89–95 % of it
  the optional CRC check) and its resident memory (499,824 / 987,600 / 1,971,568 / 3,781,072 B), plus a
  35–41 KB workspace and 6.6 KB of top-K buffers per job. With the load in a session's first box, the 0.5 and 1 MB
  tiers keep FL10 under 1 s and W20 under 2 s; the 2 MB tier is at the limit and the 4 MB tier over it for that
  first box only (tier advice below).
- **Greedy is unchanged and recoverable bit for bit:** the top-K extras change neither the logits nor the greedy
  text nor the arena, and the decoder with `alpha = beta = bracket = 0` returns the greedy text exactly (tested on
  random lines, generated images and all 3,223 development lines of all six models).

## Results (development data only)

<!-- table:settings -->
The recommended settings (`vocr_lm_opts`; alpha in 1/1000, beta, gate and bracket in 1/1000 nat), one per LM tier, shared by all six models:

| tier | LM file | bytes | beam | top-K | alpha | beta | gate | bracket | flags |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0.5MB | `vocr_lm_o3_500k.vlm` | 499,824 | 8 | 8 | 500 | 3000 | 8000 | 8000 | VOCR_LM_EOS |
| 1MB | `vocr_lm_o3_1000k.vlm` | 987,600 | 8 | 8 | 500 | 3500 | 8000 | 8000 | VOCR_LM_EOS |
| 2MB | `vocr_lm_o3_2000k.vlm` | 1,971,568 | 8 | 8 | 500 | 3000 | 8000 | 8000 | VOCR_LM_EOS |
| 4MB | `vocr_lm_o3_4000k.vlm` | 3,781,072 | 8 | 8 | 500 | 3500 | 8000 | 8000 | VOCR_LM_EOS |
<!-- /table:settings -->

### Dialogue (primary)

<!-- table:dialogue -->
Dialogue CER, greedy → beam + LM (2-fold VN cross-validated in brackets):

| model | dialogue chars | greedy | 0.5MB LM | 1MB LM | 2MB LM | 4MB LM |
|---|---:|---:|---:|---:|---:|---:|
| VOCR3-FL10 | 31,356 | 1.03% | 0.77% (0.76%) | 0.65% (0.67%) | 0.63% (0.63%) | 0.62% (0.64%) |
| VOCR3-H15 | 31,356 | 0.93% | 0.62% (0.64%) | 0.58% (0.60%) | 0.56% (0.56%) | 0.55% (0.56%) |
| VOCR3-W20 | 31,356 | 0.72% | 0.46% (0.50%) | 0.42% (0.43%) | 0.38% (0.39%) | 0.37% (0.38%) |
| VOCR4-FL10 | 31,356 | 1.13% | 0.82% (0.82%) | 0.71% (0.74%) | 0.69% (0.69%) | 0.64% (0.64%) |
| VOCR4-H15 | 31,356 | 0.87% | 0.62% (0.64%) | 0.57% (0.58%) | 0.55% (0.56%) | 0.56% (0.59%) |
| VOCR4-W20 | 20,467 | 0.90% | 0.61% (0.66%) | 0.61% (0.61%) | 0.58% (0.58%) | 0.56% (0.56%) |
| pooled (six models) | 177,247 | 0.93% | 0.65% | 0.59% | 0.56% | 0.55% |
<!-- /table:dialogue -->

<!-- table:delta -->
Change in dialogue CER, points, with the 95% VN-cluster bootstrap interval (in-sample selection; cross-validated):

| model | 0.5MB | 1MB | 2MB | 4MB | 4MB CV |
|---|---|---|---|---|---|
| VOCR3-FL10 | -0.27 (-0.48 to -0.14) | -0.38 (-0.62 to -0.23) | -0.41 (-0.67 to -0.25) | -0.41 (-0.69 to -0.26) | -0.40 (-0.65 to -0.25) |
| VOCR3-H15 | -0.32 (-0.51 to -0.19) | -0.35 (-0.55 to -0.23) | -0.38 (-0.58 to -0.25) | -0.39 (-0.61 to -0.26) | -0.38 (-0.59 to -0.25) |
| VOCR3-W20 | -0.26 (-0.46 to -0.14) | -0.30 (-0.52 to -0.17) | -0.34 (-0.59 to -0.19) | -0.35 (-0.60 to -0.20) | -0.34 (-0.58 to -0.20) |
| VOCR4-FL10 | -0.31 (-0.55 to -0.17) | -0.41 (-0.70 to -0.25) | -0.44 (-0.73 to -0.27) | -0.49 (-0.81 to -0.31) | -0.49 (-0.81 to -0.30) |
| VOCR4-H15 | -0.25 (-0.42 to -0.15) | -0.30 (-0.49 to -0.19) | -0.32 (-0.54 to -0.19) | -0.32 (-0.53 to -0.19) | -0.29 (-0.48 to -0.17) |
| VOCR4-W20 | -0.30 (-0.49 to -0.14) | -0.29 (-0.49 to -0.14) | -0.33 (-0.52 to -0.18) | -0.35 (-0.53 to -0.20) | -0.34 (-0.52 to -0.20) |
| pooled | -0.28 (-0.46 to -0.18) | -0.34 (-0.53 to -0.23) | -0.37 (-0.57 to -0.24) | -0.39 (-0.60 to -0.26) | |
<!-- /table:delta -->

<!-- table:distinct -->
Each crop once (the dialogue view without the second copy of each devnative crop that is teacher_val's own crop of the line, i.e. the same reference and the same runtime output; dialogue CER, greedy → beam + LM):

| model | lines | chars | greedy | 0.5MB | 1MB | 2MB | 4MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| VOCR3-FL10 | 1683 | 27,831 | 1.11% | 0.81% | 0.70% | 0.67% | 0.66% |
| VOCR3-H15 | 1683 | 27,831 | 0.98% | 0.67% | 0.63% | 0.60% | 0.59% |
| VOCR3-W20 | 1683 | 27,831 | 0.78% | 0.50% | 0.46% | 0.42% | 0.41% |
| VOCR4-FL10 | 1683 | 27,831 | 1.20% | 0.89% | 0.77% | 0.75% | 0.69% |
| VOCR4-H15 | 1683 | 27,831 | 0.95% | 0.67% | 0.61% | 0.60% | 0.60% |
| VOCR4-W20 | 1096 | 16,984 | 1.05% | 0.70% | 0.71% | 0.67% | 0.65% |
| pooled | | 156,139 | 1.01% | 0.71% (29.9% fewer errors) | 0.64% (36.3% fewer errors) | 0.61% (39.1% fewer errors) | 0.60% (40.7% fewer errors) |
<!-- /table:distinct -->

### Names and the console finder's crops (secondary)

<!-- table:secondary -->
Secondary views: names (CER) and the console finder's GRAD crops of devnative's dialogue lines:

| model | names greedy | names 0.5MB | names 1MB | names 2MB | names 4MB | GRAD greedy | GRAD 4MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| VOCR3-FL10 | 6.38% | 5.46% | 5.53% | 5.21% | 5.28% | 1.71% | 1.30% |
| VOCR3-H15 | 5.39% | 5.00% | 4.82% | 4.50% | 4.50% | 1.93% | 1.47% |
| VOCR3-W20 | 5.07% | 4.75% | 4.68% | 4.32% | 4.54% | 1.53% | 1.12% |
| VOCR4-FL10 | 6.35% | 5.71% | 5.60% | 5.64% | 5.53% | 2.02% | 1.47% |
| VOCR4-H15 | 5.64% | 5.03% | 4.75% | 4.71% | 4.75% | 1.89% | 1.38% |
| VOCR4-W20 | 5.23% | 4.95% | 4.73% | 4.23% | 4.34% | 1.22% | 1.12% |
<!-- /table:secondary -->

### Confusion families

<!-- table:families -->
Confusion families on dialogue lines, summed over the six models (eval/model_gaps.py's pair_kind; substitutions, then insertions and deletions):

| family | greedy | 0.5MB | 1MB | 2MB | 4MB |
|---|---:|---:|---:|---:|---:|
| kanji_kanji | 527 | 370 | 342 | 313 | 307 |
| dakuten | 197 | 82 | 67 | 60 | 58 |
| small_kana | 54 | 36 | 31 | 31 | 26 |
| hira_kata | 15 | 10 | 9 | 6 | 4 |
| dash_like | 16 | 10 | 4 | 5 | 5 |
| punct_punct | 79 | 75 | 74 | 73 | 75 |
| punct_other | 36 | 35 | 32 | 27 | 20 |
| other_sub | 450 | 293 | 260 | 257 | 255 |
| insertion | 44 | 54 | 82 | 52 | 80 |
| deletion | 238 | 188 | 147 | 176 | 139 |
| **all** | 1656 | 1153 | 1048 | 1000 | 969 |
<!-- /table:families -->

<!-- table:families_model -->
Per model at the 1MB tier (greedy → beam):

| model | kanji_kanji | dakuten | small_kana | punct_punct | dash_like | other_sub | deletion | insertion |
|---|---|---|---|---|---|---|---|---|
| VOCR3-FL10 | 94 → 66 | 40 → 13 | 9 → 5 | 17 → 21 | 5 → 0 | 86 → 50 | 54 → 31 | 8 → 11 |
| VOCR3-H15 | 96 → 52 | 37 → 11 | 12 → 6 | 15 → 11 | 1 → 1 | 79 → 56 | 40 → 27 | 8 → 13 |
| VOCR3-W20 | 77 → 44 | 26 → 10 | 12 → 5 | 7 → 7 | 1 → 0 | 58 → 32 | 31 → 15 | 6 → 12 |
| VOCR4-FL10 | 108 → 74 | 43 → 19 | 9 → 7 | 17 → 19 | 6 → 2 | 98 → 46 | 49 → 34 | 9 → 14 |
| VOCR4-H15 | 82 → 59 | 29 → 7 | 9 → 7 | 18 → 10 | 1 → 0 | 83 → 45 | 37 → 23 | 8 → 23 |
| VOCR4-W20 | 70 → 47 | 22 → 7 | 3 → 1 | 5 → 6 | 2 → 1 | 46 → 31 | 27 → 17 | 5 → 9 |
<!-- /table:families_model -->

### What the LM breaks: the guard

The gate is the guard the task asked for: a frame whose top-1 leads its top-2 by at least `gate` keeps its top-1,
so the LM can only re-rank characters the network was unsure of. On development data the best gate is wide (8
nats, a probability ratio of about 3,000). Most of the gain comes from frames with margins under 2.5 nats (gate
sweep), it is flat from 8 nats on, and tighter gates cost dialogue and names alike: the LM weight of 0.5 already
keeps the LM from overruling a clear reading, so the gate never had to protect names.

<!-- table:changes -->
What the LM changes, dialogue lines summed over the six models (characters fixed and introduced, lines better and worse; the last three columns: introduced-error lines with a rare kanji (Jiten rank > 2,000), mostly katakana, or Latin/digits in the reference):

| tier | lines changed | chars fixed | chars introduced | lines better | lines worse | rare kanji | katakana | Latin/digits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5MB | 852 | 702 | 199 | 578 | 191 | 21 | 31 | 40 |
| 1MB | 894 | 783 | 175 | 634 | 167 | 22 | 28 | 31 |
| 2MB | 895 | 813 | 157 | 659 | 155 | 19 | 31 | 30 |
| 4MB | 931 | 850 | 163 | 683 | 160 | 21 | 32 | 27 |
<!-- /table:changes -->

<!-- table:names_changes -->
The same for name lines:

| tier | lines changed | chars fixed | chars introduced | lines better | lines worse |
|---|---:|---:|---:|---:|---:|
| 0.5MB | 363 | 172 | 83 | 160 | 80 |
| 1MB | 377 | 195 | 84 | 181 | 82 |
| 2MB | 408 | 231 | 78 | 207 | 76 |
| 4MB | 423 | 235 | 91 | 213 | 87 |
<!-- /table:names_changes -->

<!-- table:by_class -->
Reference characters by class, summed over the six models at the 1MB tier: read correctly by the greedy text and by the beam, and the characters the beam broke (greedy right, beam wrong) and fixed:

| view | class | chars | greedy correct | beam correct | broke | fixed |
|---|---|---:|---:|---:|---:|---:|
| dialogue | hiragana | 100,874 | 100,488 | 100,717 | 36 | 265 |
| dialogue | kanji_common | 42,482 | 41,865 | 42,095 | 42 | 272 |
| dialogue | kanji_rare | 762 | 695 | 694 | 5 | 4 |
| dialogue | katakana | 11,301 | 11,095 | 11,207 | 17 | 129 |
| dialogue | latin_digit | 1,350 | 1,231 | 1,248 | 19 | 36 |
| dialogue | other | 64 | 51 | 31 | 20 | 0 |
| dialogue | punct | 20,414 | 20,210 | 20,289 | 42 | 121 |
| names | hiragana | 2,352 | 2,298 | 2,310 | 4 | 16 |
| names | kanji_common | 8,762 | 8,095 | 8,202 | 32 | 139 |
| names | kanji_rare | 564 | 516 | 511 | 10 | 5 |
| names | katakana | 4,566 | 4,421 | 4,449 | 10 | 38 |
| names | latin_digit | 198 | 188 | 186 | 2 | 0 |
| names | punct | 474 | 474 | 472 | 2 | 0 |
<!-- /table:by_class -->

<!-- table:gate_sweep -->
The gate from 0.5 to 64 nats at the p_o3j2_1000k settings otherwise (pooled errors over the six models; greedy 1656 dialogue / 960 names):

| gate (nat) | 0.5 | 1 | 1.5 | 2 | 2.5 | 3.5 | 5 | 8 | 12 | 20 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dialogue errors | 1456 | 1288 | 1201 | 1125 | 1101 | 1063 | 1055 | 1048 | 1048 | 1048 | 1048 |
| name errors | 915 | 893 | 880 | 866 | 855 | 851 | 849 | 849 | 849 | 849 | 849 |
<!-- /table:gate_sweep -->

<!-- table:gate_tiers -->
The gate at each tier's other settings, pooled errors over the six models (dialogue / names; greedy 1656 / 960):

| tier | gate 3.5 nat | gate 5 nat | gate 8 nat | gate 12 nat |
|---|---|---|---|---|
| 0.5MB | 1159 / 874 | 1158 / 871 | 1153 / 871 | 1153 / 871 |
| 1MB | 1063 / 851 | 1055 / 849 | 1048 / 849 | 1048 / 849 |
| 2MB | 1015 / 813 | 1004 / 809 | 1000 / 807 | 1000 / 807 |
| 4MB | 986 / 816 | 972 / 816 | 969 / 816 | 969 / 816 |
<!-- /table:gate_tiers -->

### Beam width and candidates per frame

A wider beam and more candidates help a little past 8 × 8 (see the 16 × 16 cells), at a search cost that was not
measured (it grows with beam × candidates). Beam 1 rescores each frame's candidates against one prefix only and
gets about a quarter of the gain.

<!-- table:ablation -->
Beam width × top-K at the 1MB tier's other settings (pooled dialogue errors, six models):

| beam | top-2 | top-4 | top-8 | top-16 |
|---|---:|---:|---:|---:|
| 1 | 1491 | 1492 | 1483 | 1483 |
| 2 | 1202 | 1181 | 1166 | 1165 |
| 4 | 1176 | 1103 | 1079 | 1074 |
| 8 | 1173 | 1080 | 1048 | 1039 |
| 16 | 1173 | 1079 | 1040 | 1031 |

Beam width × top-K at the 4MB tier's other settings (pooled dialogue errors, six models):

| beam | top-2 | top-4 | top-8 | top-16 |
|---|---:|---:|---:|---:|
| 1 | 1458 | 1446 | 1432 | 1428 |
| 2 | 1148 | 1116 | 1093 | 1091 |
| 4 | 1117 | 1032 | 1006 | 1002 |
| 8 | 1111 | 1008 | 969 | 963 |
| 16 | 1111 | 1000 | 964 | 955 |
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

### Where the gain comes from: the Jiten lists and the development titles

Three LM recipes at each size, all with the protected-name filter and round 7's grid: the released recipe, the same
without the Jiten lists, and (1 and 4 MB) the same with the 183 development VNs' character names and titles removed
too (teacher_val's, devnative's and the owner pool's titles, by the same rule as the protected ones: 388 VNs, 5,021
characters, 26,898 corpus pieces and 2,745 + 2,634 Jiten words dropped). Removing the development titles' names
changes nothing beyond noise, so the LM does not win on development data by knowing its characters' names; the
Jiten lists help most at the small sizes, where the corpus' n-grams are pruned hardest.

<!-- table:ablations -->
Where the gain comes from (round 7's grid, every LM with the protected-name filter; pooled errors over the six models at each LM's best shared setting, 2-fold cross-validated, names):

| LM | bytes | Jiten | development VNs' names removed too | held-out nats/char | dialogue | CV | names |
|---|---:|---|---|---:|---:|---:|---:|
| p_o3j2_500k | 499,824 | vn 0.4, novel 0.2 | no | 4.085 | 1153 | 1187 | 871 |
| p_o3nj_500k | 498,528 | none | no | 4.100 | 1202 | 1203 | 901 |
| p_o3j2_1000k | 987,600 | vn 0.4, novel 0.2 | no | 3.938 | 1048 | 1071 | 849 |
| p_o3nj_1000k | 993,568 | none | no | 3.934 | 1094 | 1095 | 874 |
| p_o3j2dev_1000k | 993,024 | vn 0.4, novel 0.2 | yes | 3.939 | 1054 | 1073 | 852 |
| p_o3j2_2000k | 1,971,568 | vn 0.4, novel 0.2 | no | 3.853 | 1000 | 1006 | 807 |
| p_o3nj_2000k | 1,916,944 | none | no | 3.832 | 1040 | 1054 | 855 |
| p_o3j2_4000k | 3,781,072 | vn 0.4, novel 0.2 | no | 3.808 | 969 | 991 | 816 |
| p_o3nj_4000k | 3,891,184 | none | no | 3.774 | 964 | 980 | 828 |
| p_o3j2dev_4000k | 3,742,880 | vn 0.4, novel 0.2 | yes | 3.812 | 973 | 996 | 831 |
<!-- /table:ablations -->

### Cost and memory (estimates)

<!-- table:a9 -->
Calibrated to the console (owner 2026-10-14 18:09Z): every estimate scaled so that VOCR3-H15's network costs the measured 5.0–5.2 ms per CTC frame (equivalent to an A9 at 181–317 MHz under the cost model's low and high bounds). ms per 640-px line (80 CTC frames), low–high:

| model | network (greedy) | + top-8 | + beam + 0.5MB LM | + beam + 1MB LM | + beam + 2MB LM | + beam + 4MB LM | LM load, once |
|---|---|---|---|---|---|---|---|
| VOCR3-FL10 | 226.8–246.3 | 5.9–6.8 | 0.4–0.5 | 0.4–0.5 | 0.4–0.5 | 0.4–0.5 | 0.5MB 68.3–82.5; 1MB 131.4–158.4; 2MB 258.7–312.2; 4MB 492.9–615.5 ms |
| VOCR3-H15 (calibration reference) | 400.0–416.0 | – | – | – | – | – | – |
| VOCR3-W20 | 568.3–572.9 | 8.9–10.2 | 0.4–0.7 | 0.4–0.6 | 0.4–0.6 | 0.4–0.6 | 0.5MB 70.3–87.4; 1MB 133.4–166.4; 2MB 260.7–324.5; 4MB 494.9–636.2 ms |
| VOCR4-W20 | 568.3–572.9 | 8.9–10.2 | 0.4–0.7 | 0.4–0.5 | 0.4–0.6 | 0.4–0.6 | 0.5MB 70.3–87.4; 1MB 133.4–166.5; 2MB 260.7–324.5; 4MB 494.9–636.2 ms |

Per typical full textbox (3 lines of about 640 px), calibrated, against the tier limits (Small < 1 s, Normal < 2 s, Large < 3 s); the first-box columns add the LM's one-time load (CRC check, parse, class map; file reading not counted) to a session's first box:

| model | greedy box | beam + LM box (any tier) | first box, 0.5MB LM | first box, 1MB LM | first box, 2MB LM | first box, 4MB LM |
|---|---|---|---|---|---|---|
| VOCR3-FL10 | 0.68–0.74 s | 0.70–0.76 s | 0.77–0.84 s | 0.83–0.92 s | 0.96–1.07 s | 1.19–1.38 s |
| VOCR3-H15 | 1.20–1.25 s | – | – | – | – | – |
| VOCR3-W20 | 1.71–1.72 s | 1.73–1.75 s | 1.80–1.84 s | 1.87–1.92 s | 1.99–2.08 s | 2.23–2.39 s |
| VOCR4-W20 | 1.71–1.72 s | 1.73–1.75 s | 1.80–1.84 s | 1.87–1.92 s | 1.99–2.08 s | 2.23–2.39 s |

The uncalibrated estimate (vita/a9_cycles.py's cost model on the Vita build, development lines), ms per 640-px line at 444 MHz, low–high (333 MHz in brackets):

| model | network (greedy) | + top-8 | + beam + 0.5MB LM | + beam + 1MB LM | + beam + 2MB LM | + beam + 4MB LM | LM load, once |
|---|---|---|---|---|---|---|---|
| VOCR3-FL10 | 92.5–175.9 (123.4–234.5) | 2.4–4.9 (3.2–6.5) | 0.2–0.4 (0.2–0.5) | 0.2–0.4 (0.2–0.5) | 0.2–0.4 (0.2–0.5) | 0.2–0.4 (0.2–0.5) | 0.5MB 28–59; 1MB 54–113; 2MB 106–223; 4MB 201–440 (CRC ≤ 95%) ms |
| VOCR3-W20 | 231.9–409.2 (309.2–545.5) | 3.6–7.3 (4.8–9.7) | 0.1–0.5 (0.2–0.6) | 0.1–0.4 (0.2–0.5) | 0.2–0.4 (0.2–0.5) | 0.2–0.4 (0.2–0.6) | 0.5MB 29–62; 1MB 54–119; 2MB 106–232; 4MB 202–454 (CRC ≤ 95%) ms |
| VOCR4-W20 | 231.9–409.2 (309.2–545.5) | 3.6–7.3 (4.8–9.7) | 0.1–0.5 (0.2–0.6) | 0.1–0.4 (0.2–0.5) | 0.1–0.4 (0.2–0.5) | 0.2–0.4 (0.2–0.5) | 0.5MB 29–62; 1MB 54–119; 2MB 106–232; 4MB 202–454 (CRC ≤ 95%) ms |

The same per typical full textbox (3 lines of about 640 px), uncalibrated; the last column adds the LM's one-time load to a session's first box (4 MB tier, file reading not counted):

| model | greedy, 444 MHz | greedy, 333 MHz | beam + LM (any tier), 444 MHz | beam + LM, 333 MHz | first box with the 4MB LM, 444 MHz |
|---|---|---|---|---|---|
| VOCR3-FL10 | 0.28–0.53 s | 0.37–0.70 s | 0.29–0.54 s | 0.38–0.72 s | 0.49–0.98 s |
| VOCR3-W20 | 0.70–1.23 s | 0.93–1.64 s | 0.71–1.25 s | 0.94–1.67 s | 0.91–1.71 s |
| VOCR4-W20 | 0.70–1.23 s | 0.93–1.64 s | 0.71–1.25 s | 0.94–1.67 s | 0.91–1.71 s |
<!-- /table:a9 -->

**Calibration (owner 2026-10-14 18:09Z).** The console runs VOCR3-H15 (`cb3c61a8…`, the blob of this document's
`v3H15` rows) at about 5.0–5.2 ms per CTC frame (vita-jp-overlay v0.8.0-test.1 logs), well above the cost model's
2.0–3.7 ms at 444 MHz. Every estimate above is therefore scaled per bound: a cycle count of the model's low bound is
multiplied by 5.0 ms ÷ VOCR3-H15's low-bound cycles per frame, and the high bound by 5.2 ms ÷ its high-bound
cycles, so H15's network comes out at exactly the measured 5.0–5.2 ms per frame and every other figure keeps its
ratio to H15's estimate. The search, the top-K tracking and the CRC are integer or scalar code with another
instruction mix than the network's NEON matrix products, so their calibrated figures are less certain than the
network's; the uncalibrated estimates follow for reference.

How the estimate was made: `vita/a9_cycles.py measure` (the Vita compiler's machine code for `vocr.c` and
`vocr_lm.c` under qemu-arm with instruction counting and a simulated 32 KiB L1 / 2 MiB L2) on 40 development
dialogue lines and on the same 40 twice, with the network alone, with top-8, and with each tier's beam + LM. The
pack is `eval/ctclm.py pack`: 20 teacher_val and 20 devnative dialogue lines of at least 10 characters, distinct by
line id and by crop, 490 px wide on average at the model's height of 32. A component's per-line cost is the
difference of two runs on the same pack and the slope from 40 to 80 lines, so the LM's one-time load drops out and
is shown on its own; the cost model (`a9_cycles.model_mode`) turns the differential instruction mix and cache
misses into a low–high cycle range. The runtime's instruction counts repeat to about seven significant digits
(the CLI's timing printf shares libgcc's division helpers with the runtime). Cache misses also move with the memory
layout of each run, so a component's differential miss count is noisy and can come out negative; the cost model
takes it clamped at zero (`docs/ctc-lm-decode.json` keeps the signed values). The search is branchy integer code
with a few hundred kilobytes of LM touched per line; in the simulation its trie lookups stay in the caches. In a game
the L2 is shared, so expect more misses than this. The one-time load (CRC, `vocr_lm_load`, `vocr_lm_init`) does not
include reading the blob from storage.

<!-- table:memory -->
Memory per tier: the LM blob (resident while the game runs, read-only), the decoder workspace (vocr_lm_workspace_bytes, a 1,024-px line) and the caller's top-K buffers:

| tier | LM file | LM bytes | sha256 | workspace (FL10 / H15, W20) | top-8 buffers |
|---|---|---:|---|---|---:|
| 0.5MB | `vocr_lm_o3_500k.vlm` | 499,824 | `38ea94e0c4ec9386…` | 35,248 / 40,816 B | 6,656 B |
| 1MB | `vocr_lm_o3_1000k.vlm` | 987,600 | `993000611555171d…` | 35,248 / 40,816 B | 6,656 B |
| 2MB | `vocr_lm_o3_2000k.vlm` | 1,971,568 | `ac33e461bd346617…` | 35,248 / 40,816 B | 6,656 B |
| 4MB | `vocr_lm_o3_4000k.vlm` | 3,781,072 | `01f49be1691a1aab…` | 35,248 / 40,816 B | 6,656 B |
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
- **Protected titles.** Before counting, the builder drops every corpus piece and every Jiten word that names a
  protected VN's character or title: the VNs of `eval/manifests/test_games.json` (its "all", the vita-test and
  psp-test games with their release partners under today's reservation (`docs/reservations.md`), together with
  "all_20261012", the earlier rule's wider list with their families) and of `data/manifests/synth3_exclude_clannad.json`
  (the CLANNAD family): 206 VNs with 2,316 VNDB characters and 234 titles (synth3's pinned VNDB extracts). The markers start from synth's rule
  (`synth/owner_games.marker_candidates`) and go further: always every title of ≥ 3 Japanese or ≥ 5 Latin
  characters, every full name of two or more parts and the CLANNAD manifest's 41 curated markers; and every other
  full name or name part of ≥ 2 characters that occurs in at most 50 corpus pieces. A piece or Jiten word holding a
  marker is dropped: 14,126 of 1,829,895 pieces and 1,350 + 1,331 Jiten words (4,003 markers). 524 name parts occur
  in more pieces and stay as ordinary words or surnames (岡崎, 伊吹, 坂上, but also 世界 or 太陽). Of CLANNAD's main
  names, 春原, 朋也, 古河, 藤林, 風子, 智代, 一ノ瀬, 陽平 and 芽衣 leave the corpus and the Jiten lists; the 4 MB
  tier still holds the bigrams 智代 and 芽衣 from other words that contain them (美智代, 芽衣子). On development data
  the filter costs nothing measurable (1,044 → 1,048 dialogue errors at 1 MB).
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
  (the tests build twice). The four release blobs were rebuilt from the committed builder with the commands under
  "Reproduce" and are byte-identical to the ones tuned (the summary checks this).

<!-- table:recipes -->
LM recipes (rounds 2–3, LMs built before the protected-name filter; pooled dialogue errors at each LM's best shared configuration, cross-validated, names; held-out cross-entropy on realplus v2 labels):

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
- **Search.** Beam 8 and top-8 throughout; seven rounds over the LM weight, the per-character bonus, the gate, the
  start/end-of-line terms and the bracket cost (`docs/ctc-lm-decode.json` `selection.grids`). Round 1 tuned 6 LMs
  and rounds 2–3 23 (24 recipes in all: the recipe table plus round 1's Aozora-only `mixa_1000k`); the recipe was
  then fixed to order 3 with the 2× Jiten share at every size, the best or within noise of it at each. Rounds 4–6
  added the bracket state and folding and extended the grid past its edges. Round 7, after the first audit, runs
  round 6's grid on the LMs rebuilt with the protected-name filter, and on the ablations; it is final. It ran first
  with a looser filter (synth's rule as is, which keeps two-character given names such as 朋也) and was repeated
  after checking CLANNAD's names in the blobs; the looser runs are archived and changed nothing measurable either.
  Five more LMs were built during the exploration and never tuned.
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
and the top-K buffers (6.6 KB) belong to the job. This document assumes the LM does not count against a
recognizer tier's weight ceiling (`docs/model-tiers.md`: Small < 1 MB of int8 weights): it is a separate block
the overlay's free-memory preflight has to fit (Small with the 0.5 MB LM holds 657,664 + 499,824 B resident); the
conductor decides whether the tiers should budget it. The owner's ceilings of 2026-10-14 18:09Z (int8 weights:
Small < 1 MB, Normal ≤ 1.6 MB, Large ≤ 2.5 MB; time per 3-line box: < 1 / < 2 / < 3 s) come from console time, and
the LM barely changes a box's time; its size shows in the first box of a session (calibrated table above). Advice,
from that table:

- **Small** (VOCR4-FL10): the 1 MB LM (first box 0.83–0.92 s, later boxes 0.70–0.76 s); the 0.5 MB LM if memory is
  short. The 2 MB LM's first box (0.96–1.07 s) reaches the 1 s limit.
- **Normal** (VOCR4-W20 or H15, ≤ 1.6 MB): the 1 MB LM (W20's first box 1.87–1.92 s); the 2 MB LM's first box is at
  1.99–2.08 s.
- **Large** (≤ 2.5 MB, < 3 s): any tier; W20 with the 4 MB LM is 2.23–2.39 s in the first box.
- Loading the LM together with the recognizer's weights (once per session, not in a job), or skipping
  `vocr_lm_check_crc()` when the file is verified another way, takes the load out of the first box and lets every
  LM tier fit every recognizer tier on time. Pin each tier by its size and sha256 (memory table); verify the
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
for b in 500000 1000000 2000000 4000000; do .venv-ci/bin/python lm/build.py --out $ART/release2/vocr_lm_o3_${b%000}k.vlm \
    --manifest $ART/release2/vocr_lm_o3_${b%000}k.json --bytes $b --order 3 --jiten vn=0.4,novel=0.2 \
    --corpora aozora_dialogue,aozora_narration,aratako_dsv3,jesc,tatoeba; done
# (the ablations: --jiten "" for no Jiten; --exclude-vns $ART/lm/dev_vns.txt for the development titles too)
# 3. round 7 for each model and tier (--in-sample-own for v4W20), then selection, report, ablation, gate sweep
GRID='{"alpha":[400,500,600],"beta":[2500,3000,3500,4000],"gate":[3500,5000,8000,12000],"flags":[2],"bracket":[0,1000,2000,3000,5000,8000,12000,20000,64000]}'
.venv-ci/bin/python eval/ctclm.py tune --extract $ART/extract/v3W20.npz --lm $ART/lm/o3j2_1000k.vlm --grid "$GRID" \
    --out $ART/tune/r7_v3W20_p_o3j2_1000k.npz        # p_o3j2_* = the release tiers (step 2), plus the ablations
.venv-ci/bin/python eval/ctclm.py select --extract $ART/extract --tune $ART/tune --prefix r7_ --lm-dir $ART/lm \
    --lms p_o3j2_500k,p_o3j2_1000k,p_o3j2_2000k,p_o3j2_4000k --out $ART/select/final_r7.json
.venv-ci/bin/python eval/ctclm.py report --extract $ART/extract --select $ART/select/final_r7.json --lm-dir $ART/lm \
    --tune $ART/tune --out $ART/select/report_r7.json        # re-decodes; refuses if errors differ from the tuning
.venv-ci/bin/python eval/ctclm.py ablate --extract $ART/extract --select $ART/select/final_r7.json --tier 1MB \
    --lm-dir $ART/lm --out $ART/select/ablate_r7_1MB.json     # and --tier 4MB
# 4. A9: the development pack, then base, --topk 8 and each tier's settings on it and on it twice
.venv-ci/bin/python eval/ctclm.py pack --out $ART/a9/devp40   # devp40.bin, devp40x2.bin, devp40.json
.venv-ci/bin/python vita/a9_cycles.py measure --out $ART/a9/f_v3W20_lm1MB_devp40 --modes w8i --blob BLOB \
    --pack $ART/a9/devp40.bin --cli-args "--lm LM.vlm --beam 8 --topk 8 --alpha 500 --beta 3500 --gate 8000 --lm-flags 2 --bracket 8000"
.venv-ci/bin/python eval/ctclm.py a9 --dir $ART/a9 --packs devp40,devp40x2 --out $ART/select/a9.json
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
- teacher_val and devnative share all 43 devnative VNs and 348 line ids, and 349 of devnative's 543 ink crops are
  pixel-identical to one of teacher_val's own native Vita/PSP crops (344 under the same line id; both sets were cut
  from the same native screenshots): the dialogue view scores 241 crops twice (the same reference and runtime
  output) and the names view about 100. teacher_val itself cuts 251 lines twice, at Vita and at PSP scale
  (different crops of one line). The VN-cluster bootstrap keeps them together; the
  pooled CER counts them twice. The "each crop once" table drops the second copies: 1.01 % greedy → 0.71 / 0.64 /
  0.61 / 0.60 %.
- **Exposure to protected titles (disclose with any protected-set score).** The Jiten VN list is built from 2,162
  VN scripts and includes CLANNAD (one of Jiten's VN decks), and may include vita-test or psp-test titles. The
  builder removes their character names and titles from the Jiten lists and the corpora (above) unless a name is
  also a common word or surname (> 50 corpus pieces), and longer words that contain a name keep its n-grams (the
  4 MB tier holds 智代 and 芽衣 that way); the frequencies of ordinary words still carry those scripts' share.
  JESC's subtitles include CLANNAD's anime, of which only the lines naming a character or title are removed. Any score of a model
  decoded with one of these LMs on a protected set (CLANNAD, vita-test, psp-test) must disclose this; the conductor
  decides whether such a score is admissible. An LM built without the Jiten lists (`--jiten ""`, ablation table)
  removes the Jiten part of the exposure and is level with the released recipe at 4 MB.
- Every console time is an estimate: the Vita build's instruction mix under qemu-arm with a simulated 2 MiB L2
  to ourselves, calibrated to one console measurement (VOCR3-H15's network time per frame). The decoder's own code
  has another instruction mix and gets the network's calibration factors; a game's share of the L2 and its DRAM
  traffic will raise the LM's misses. The decoder has not run on a Vita or a PSP (it compiles warning-free for
  both).
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
