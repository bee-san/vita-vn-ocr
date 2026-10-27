# vocr6-final-lm: Rubi's decoder settings on VOCR6's final blobs (stage 4, #125, #127)

Rubi decodes every OCR line with a beam search and a character LM. Each LM tier has its own decoder setting:

- the 1 MB (default) and 4 MB game-text LMs of #137: alpha 0.6, beta 2.5 nats, BOS;
- the 0.5 MB and 2 MB ctclm.2 LMs: alpha 0.5, beta 3.0, EOS.

Before the LM, the counter re-decode (#109) runs at tau 6 / open 4 nats. All of these values were chosen on VOCR5's
logits. This pass checks, on VOCR6's final blobs, whether they still hold. #109's synthetic counter set is the guard.

**Development data only, read on the host; every console figure is an estimate; nothing ran on a console.** No
protected set was read: not CLANNAD, vita-test, psp-test or the hooked test title PCSG00807.

- Protocol: `eval/vocr6_final_lm_protocol.json`. It was committed before any final blob existed, with two dated
  amendments, also made before any final read.
- Result: `eval/vocr6_final_lm_result_20261027.json`, all of it counts, rates, settings and hashes.

## Answers first

- **The default 1 MB tier and the 4 MB tier keep their settings on VOCR6-N and VOCR6-L** (alpha 0.6, beta 2.5, BOS).
  - The best setting of the grid, cross-validated, is worse in point: objective +0.21 / +0.33 points at 1 MB (N /
    L), +0.24 / +0.25 at 4 MB.
  - Its in-sample choice would also keep fewer counters on #109's set, by 0.7-1.4 points at N and L (resolved).
- **The 0.5 MB tier keeps ctclm.2's setting.** Retuning gains in point at N and L, but neither gain is resolved:
  objective −0.364 (−1.156 to +0.004) and −0.346 (−1.142 to +0.024).
- **The 2 MB tier changes for Normal jobs only:** alpha 0.4, beta 1.5, BOS and EOS (`flags` 3), cascade 700,
  counter 6 / 4.
  - Cross-validated against the shipped setting, the objective is −0.260 points (−0.800 to −0.006).
  - No guard is resolved worse: hp −0.485 (−1.562 to +0.001), hc −0.008, dev score −0.036 (−0.093 to +0.011).
  - On the counter set, the CER is resolved better, 3.035% → 2.959%. The kept count is level, 3,296 → 3,288.
  - **Most of the page-path gain is one title.** PCSG00854's box decoration, read by the finder as a line of junk,
    accounts for −1,519 of the −1,575 hp errors. Without that title, the objective is −0.027 (−0.058 to −0.000),
    still resolved.
  - By platform (in-sample, shipped → new): hp Vita 4.474 → 3.230 and PSP 3.047 → 3.026; hc Vita 0.326 → 0.320 and
    PSP 0.376 → 0.353. PCSG00854 is a Vita title.
  - Large jobs at 2 MB keep the shipped setting: −0.244 (−0.781 to +0.010), not resolved.
- **The counter thresholds stay 6 / 4 nats at every tier.** No cell of the 5 × 3 grid changes a single real
  development or hval unit, so none can be resolved better on the objective. On the synthetic counter set, larger
  thresholds keep a few more counters: +12 to +14 crops at N with 8 / 5.
- **Small stays VOCR5-S** (stage 4a did not adopt VOCR6-S). Its job path, VOCR5-S with the cascade to VOCR6-N, is
  reported only. No setting would pass there either.
- **What VOCR6 reads with Rubi's shipped 1 MB setting** (dialogue CER %; hval read by the 4a sibling):

  | job | hval page path | hval clean | dev score | objective | counter set: crops keeping the counter (of 3,969), CER |
  |---|---:|---:|---:|---:|---|
  | VOCR6-N + cascade to L | 3.180 | 0.321 | 2.865 | 3.022 | 3,322, 3.019% |
  | VOCR6-L | 3.148 | 0.288 | 2.889 | 3.018 | 3,326, 2.853% |
  | VOCR5-S + cascade to VOCR6-N (report) | 3.487 | 0.470 | 3.069 | 3.278 | 3,257, 3.339% |
  | VOCR5-N + cascade to VOCR5-L, for comparison | 3.248 | 0.404 | 2.863 | 3.055 | 3,064, 3.612% |

  Rubi's decode at the shipped 1 MB setting (LM and cascade) against VOCR6-N's greedy text alone: clean lines
  −0.155 points (−0.226 to −0.091), dev score −0.469 (−0.663 to −0.304).

## For Rubi

| model (job tier) | 0.5 MB | 1 MB (default) | 2 MB | 4 MB |
|---|---|---|---|---|
| Small (VOCR5-S) | 500 / 3000 / EOS | 600 / 2500 / BOS | 500 / 3000 / EOS | 600 / 2500 / BOS |
| Normal (VOCR6-N) | 500 / 3000 / EOS | 600 / 2500 / BOS | **400 / 1500 / BOS + EOS** | 600 / 2500 / BOS |
| Large (VOCR6-L) | 500 / 3000 / EOS | 600 / 2500 / BOS | 500 / 3000 / EOS | 600 / 2500 / BOS |

The cells give `vocr_lm_opts` alpha / beta (1/1000 nat) / flags. Everything else is unchanged everywhere: gate 8000,
bracket 8000, beam 8, top-8, cascade under 700, counter tau / open 6000 / 4000.

- The one change applies to a Normal job's own decode and to its cascade's (VOCR6-L) decode. Rubi's `lms[]` is per LM
  tier today, so this needs a per-model override.
- The decoder workspace is 41,632 B for N and L at the new setting, under `VJO_VOCR_LM_WS_BYTES` (49,152 B).
- `result.json` carries the table as data (`rubi`).
- The crafted 「九ヶ月です」 host test at the 1 MB tier keeps today's expectation (「九月です」, that setting's known
  loss), because the 1 MB setting does not change. On the counter set, arm b's 1 MB choices keep fewer counters than
  the shipped setting (N 3,322 → 3,268). Larger counter thresholds keep up to 12 more, but they change no real line.

## How it was measured

- **Blobs.** vocr6-lead's FINAL BLOBS (2026-10-27 08:52Z):
  - N `v6f4h_N_seed1_w8` (`11a306c3…`) and L `v6f4h_L_seed1_w8` (`d32744ca…`), stage 4b, hval folded in;
  - S is VOCR5-S, read from v0.3.0's `v5f_S_seed2_w8` (`f175bfcb…`). Rubi's `VOCR5_S_w8g` is the same weights with
    a head gate.

  All are W8I through the C runtime, head gate off, top-8.
  - The final blob reads the development cohorts and the counter set.
  - hval is read by a recognizer that never trained on it: N's and L's stage-4a seed-1 siblings, `v6s2cil_N_seed1`
    (`5bd63326…`) and `v6f4_L_seed1` (`7113f109…`). VOCR5-S reads hval itself.
- **Line path.** Rubi's `finish_line`, as `eval/vocr6_lm.py` models it:
  1. the counter re-decode of the top-8;
  2. the beam + LM decode;
  3. the counter rule;
  4. the cascade: a line under the confidence threshold shows the next tier's reading of the same crop, decoded at
     the same setting;
  5. what Rubi shows: Rubi's `core/textnorm.c` and the Shell's non-Japanese line filter.

  The dev units and the counter set are decoded with the shipped LM blobs. hval is decoded with the screening builds
  (h64_1000k, h16rl_4000k; ctclm.2's own bytes at 0.5 and 2 MB), because the shipped game-text blobs hold hval's text.
- **Sets.**
  - dev: VOCR5's cohorts (tv, dn, wd), without v12188 and v18432: 3,344 units, 92 / 33 / 91 VNs.
  - hval: vocr6-lm's hval_rubi set: 3,170 clean lines and 12,654 region reads (GRAD and GRAD + refine) of 46
    titles.
  - counter set: #109's 3,969 synthetic crops = 189 public development-pool lines × 3 forms × 7 left-edge cuts.
- **Score.** Dialogue CER. The objective is #127's: 0.5 hp + 0.5 dev score, with the dev score 0.25 tv + 0.25 dn
  + 0.5 wd. It is computed on the job path: N with its cascade to L, L alone, S with its cascade to N. A counter
  crop keeps its counter when the shown text holds a numeral + ヶヵケ箇 + 月所年国条, vocr6-rubi-ship's test.
- **Arms**, per job tier and LM tier:
  - (a) the shipped setting;
  - (b) the best of vocr6-lm's grid: alpha 0.1-0.8 × beta 0-5 nats × flags 0-3, gate 8 nats, with cascade 500 /
    600 / 700. That is 1,056 options for S and N, and 352 for L;
  - (c) tau {4, 5, 6, 7, 8} × open {3, 4, 5} nats at the shipped setting.

  (b) and (c) are 2-fold cross-validated on #127's folds: each fold's units are scored by the other fold's choice.
  Ties go to the shipped option. The gate stays at 8 nats because docs/counters.md needs the re-decode's thresholds
  below the LM's gate.
- **Rule.** Keep the shipped setting unless (b) or (c) meets all four conditions:
  - its cross-validated objective is resolved better (95% cluster-bootstrap interval below 0);
  - hp, hc and the dev score are not resolved worse;
  - the counter set is not resolved worse in CER, and its kept share is not resolved lower, at the setting that would
    ship (189 source-line clusters);
  - the setting is admissible.

  A change applies only where it passes. Amendment 2 (before FINAL BLOBS): Small stays VOCR5-S, so the rule decides
  the N and L tiers.
- **Checks.**
  - The harness reproduces #127's numbers on VOCR5: greedy VOCR5-N hp 4.220 / hc 0.760 / dev 3.404, and #137's
    shipped 1 MB numbers.
  - It reproduces vocr6-rubi-ship's counter numbers exactly: 3,104 crops keep the counter and 1,847 errors with
    ctclm.2 1 MB at 500 / 3500 / EOS; 3,015 and 1,936 with the game-text 1 MB at 600 / 2500 / BOS.
  - The N sibling's hval reads equal vocr6-lm's reads of the same blob bit for bit (23,453 of 23,453).
  - VOCR5-S's dev reads equal the head gate cache's bit for bit (8,233 of 8,233).
  - In every tune, arm c's shipped cell equals arm b's shipped setting.

## Results

Cross-validated (b) − shipped, points, 95% interval. The objective's guards are hp, hc and the dev score:

| tier | LM | decision | shipped: obj / hp / hc / dev | b (CV) - shipped, objective | b chosen (in-sample) | b guards: hp / hc / dev | c (CV) - shipped, objective | c chosen |
|---|---|---|---|---|---|---|---|---|
| S | 1000k | report only | 3.278 / 3.487 / 0.470 / 3.069 | +0.220 (-0.056 to +0.790) | a500 b1500 BOS+EOS c700 | +0.474 (-0.051 to +1.608) / +0.008 (-0.024 to +0.039) / -0.035 (-0.095 to +0.025) | +0.000 (+0.000 to +0.000) | 6000/4000 |
| N | 1000k | keep | 3.022 / 3.180 / 0.321 / 2.865 | +0.212 (-0.014 to +0.680) | a400 b1500 BOS+EOS c700 | +0.401 (-0.029 to +1.337) / -0.004 (-0.019 to +0.011) / +0.022 (-0.023 to +0.072) | +0.000 (+0.000 to +0.000) | 6000/4000 |
| L | 1000k | keep | 3.018 / 3.148 / 0.288 / 2.889 | +0.333 (-0.001 to +1.044) | a400 b1500 BOS+EOS | +0.641 (-0.001 to +2.065) / +0.004 (-0.005 to +0.014) / +0.025 (-0.030 to +0.076) | +0.000 (+0.000 to +0.000) | 6000/4000 |
| S | 4000k | report only | 3.301 / 3.582 / 0.443 / 3.021 | +0.195 (-0.025 to +0.636) | a600 b2000 BOS+EOS c700 | +0.390 (-0.011 to +1.261) / +0.020 (-0.012 to +0.052) / -0.000 (-0.076 to +0.085) | +0.000 (+0.000 to +0.000) | 6000/4000 |
| N | 4000k | keep | 2.992 / 3.160 / 0.304 / 2.824 | +0.240 (+0.010 to +0.733) | a400 b1500 BOS+EOS c700 | +0.428 (-0.014 to +1.405) / +0.000 (-0.013 to +0.012) / +0.052 (+0.016 to +0.095) | +0.000 (+0.000 to +0.000) | 6000/4000 |
| L | 4000k | keep | 2.990 / 3.149 / 0.286 / 2.832 | +0.250 (+0.015 to +0.742) | a400 b1500 BOS+EOS | +0.443 (+0.001 to +1.417) / +0.016 (+0.000 to +0.037) / +0.057 (+0.005 to +0.119) | +0.000 (+0.000 to +0.000) | 6000/4000 |
| S | 500k | report only | 4.029 / 4.953 / 0.537 / 3.105 | -0.409 (-1.330 to +0.014) | a500 b2000 BOS+EOS c700 | -0.787 (-2.632 to +0.037) / -0.002 (-0.035 to +0.028) / -0.032 (-0.081 to +0.017) | +0.000 (+0.000 to +0.000) | 6000/4000 |
| N | 500k | keep | 3.439 / 3.941 / 0.374 / 2.937 | -0.364 (-1.156 to +0.004) | a400 b1500 BOS+EOS c700 | -0.693 (-2.277 to +0.015) / -0.024 (-0.052 to +0.000) / -0.035 (-0.101 to +0.019) | +0.000 (+0.000 to +0.000) | 6000/4000 |
| L | 500k | keep | 3.418 / 3.924 / 0.341 / 2.911 | -0.346 (-1.142 to +0.024) | a400 b1500 BOS+EOS | -0.690 (-2.268 to +0.015) / -0.016 (-0.042 to +0.009) / -0.002 (-0.068 to +0.058) | +0.000 (+0.000 to +0.000) | 6000/4000 |
| S | 2000k | report only | 3.948 / 4.889 / 0.494 / 3.006 | -0.626 (-2.074 to +0.031) | a400 b1500 BOS+EOS c700 | -1.246 (-4.144 to +0.043) / -0.012 (-0.041 to +0.015) / -0.005 (-0.063 to +0.053) | +0.000 (+0.000 to +0.000) | 6000/4000 |
| N | 2000k | change (b) | 3.410 / 3.963 / 0.343 / 2.858 | -0.260 (-0.800 to -0.006) | a400 b1500 BOS+EOS c700 | -0.485 (-1.562 to +0.001) / -0.008 (-0.035 to +0.018) / -0.036 (-0.093 to +0.011) | +0.000 (+0.000 to +0.000) | 6000/4000 |
| L | 2000k | keep | 3.397 / 3.945 / 0.317 / 2.848 | -0.244 (-0.781 to +0.010) | a400 b1500 BOS+EOS | -0.466 (-1.538 to +0.019) / +0.012 (-0.006 to +0.031) / -0.023 (-0.079 to +0.026) | +0.000 (+0.000 to +0.000) | 6000/4000 |

#109's counter set (3,969 crops). The shipped setting and arm b's and arm c's in-sample choices, with the
differences from the shipped setting:

| tier | LM | arm | counters kept (of 3,969) | counter-set CER | kept share - shipped | CER - shipped |
|---|---|---|---:|---:|---|---|
| S | 1000k | shipped | 3,257 | 3.339% | – | – |
| S | 1000k | b (in-sample) | 3,193 | 3.433% | -1.613 (-2.142 to -1.083) | +0.094 (+0.035 to +0.149) |
| S | 1000k | c (in-sample 6000/4000) | 3,257 | 3.339% | +0.000 (+0.000 to +0.000) | +0.000 (+0.000 to +0.000) |
| N | 1000k | shipped | 3,322 | 3.019% | – | – |
| N | 1000k | b (in-sample) | 3,268 | 3.044% | -1.361 (-1.864 to -0.882) | +0.025 (-0.026 to +0.076) |
| N | 1000k | c (in-sample 6000/4000) | 3,322 | 3.019% | +0.000 (+0.000 to +0.000) | +0.000 (+0.000 to +0.000) |
| L | 1000k | shipped | 3,326 | 2.853% | – | – |
| L | 1000k | b (in-sample) | 3,300 | 2.832% | -0.655 (-1.109 to -0.202) | -0.021 (-0.083 to +0.033) |
| L | 1000k | c (in-sample 6000/4000) | 3,326 | 2.853% | +0.000 (+0.000 to +0.000) | +0.000 (+0.000 to +0.000) |
| S | 4000k | shipped | 3,306 | 3.208% | – | – |
| S | 4000k | b (in-sample) | 3,273 | 3.249% | -0.831 (-1.159 to -0.529) | +0.041 (+0.004 to +0.077) |
| S | 4000k | c (in-sample 6000/4000) | 3,306 | 3.208% | +0.000 (+0.000 to +0.000) | +0.000 (+0.000 to +0.000) |
| N | 4000k | shipped | 3,354 | 2.912% | – | – |
| N | 4000k | b (in-sample) | 3,299 | 2.964% | -1.386 (-1.864 to -0.932) | +0.053 (+0.008 to +0.098) |
| N | 4000k | c (in-sample 6000/4000) | 3,354 | 2.912% | +0.000 (+0.000 to +0.000) | +0.000 (+0.000 to +0.000) |
| L | 4000k | shipped | 3,380 | 2.713% | – | – |
| L | 4000k | b (in-sample) | 3,344 | 2.742% | -0.907 (-1.285 to -0.554) | +0.029 (-0.014 to +0.070) |
| L | 4000k | c (in-sample 6000/4000) | 3,380 | 2.713% | +0.000 (+0.000 to +0.000) | +0.000 (+0.000 to +0.000) |
| S | 500k | shipped | 3,187 | 3.474% | – | – |
| S | 500k | b (in-sample) | 3,190 | 3.423% | +0.076 (-0.151 to +0.302) | -0.051 (-0.090 to -0.010) |
| S | 500k | c (in-sample 6000/4000) | 3,187 | 3.474% | +0.000 (+0.000 to +0.000) | +0.000 (+0.000 to +0.000) |
| N | 500k | shipped | 3,258 | 3.138% | – | – |
| N | 500k | b (in-sample) | 3,262 | 3.054% | +0.101 (-0.202 to +0.403) | -0.084 (-0.123 to -0.043) |
| N | 500k | c (in-sample 6000/4000) | 3,258 | 3.138% | +0.000 (+0.000 to +0.000) | +0.000 (+0.000 to +0.000) |
| L | 500k | shipped | 3,273 | 2.964% | – | – |
| L | 500k | b (in-sample) | 3,287 | 2.840% | +0.353 (-0.050 to +0.756) | -0.125 (-0.175 to -0.075) |
| L | 500k | c (in-sample 6000/4000) | 3,273 | 2.964% | +0.000 (+0.000 to +0.000) | +0.000 (+0.000 to +0.000) |
| S | 2000k | shipped | 3,260 | 3.277% | – | – |
| S | 2000k | b (in-sample) | 3,232 | 3.267% | -0.706 (-1.083 to -0.353) | -0.010 (-0.058 to +0.041) |
| S | 2000k | c (in-sample 6000/4000) | 3,260 | 3.277% | +0.000 (+0.000 to +0.000) | +0.000 (+0.000 to +0.000) |
| N | 2000k | shipped | 3,296 | 3.035% | – | – |
| N | 2000k | b (in-sample) | 3,288 | 2.959% | -0.202 (-0.504 to +0.076) | -0.076 (-0.119 to -0.028) |
| N | 2000k | c (in-sample 6000/4000) | 3,296 | 3.035% | +0.000 (+0.000 to +0.000) | +0.000 (+0.000 to +0.000) |
| L | 2000k | shipped | 3,322 | 2.838% | – | – |
| L | 2000k | b (in-sample) | 3,329 | 2.738% | +0.176 (-0.202 to +0.580) | -0.100 (-0.141 to -0.059) |
| L | 2000k | c (in-sample 6000/4000) | 3,322 | 2.838% | +0.000 (+0.000 to +0.000) | +0.000 (+0.000 to +0.000) |

- **The page path is dominated by one title at every LM tier.** PCSG00854's junk line moves hp by 1,236-4,104 errors
  between settings, in both directions, out of about 325,000 reference characters.
  - At 0.5 and 2 MB, the shipped EOS setting spells that line with kana, and the filter keeps it. Arm b's BOS + EOS
    choices do not, which is most of their page-path gain.
  - Without PCSG00854:
    - N 2 MB: −0.027 (−0.058 to −0.000), resolved;
    - N 0.5 MB: −0.019 (−0.055 to +0.011);
    - L 2 MB: −0.011 (−0.041 to +0.016);
    - N 1 MB: +0.007 (−0.019 to +0.034).
- **What the 4b blobs read on their own hval pages (report only: memorised pages).** At N 2 MB, b − shipped: hc
  −0.020 (−0.041 to −0.002), hp −0.279 (−0.961 to +0.040). The siblings' figures above are the ones the rule reads.
- **The counter thresholds (arm c).** Every one of the 15 cells gives the same objective at every tier and LM tier.
  On the counter set the kept count rises with tau, for example:
  - N 1 MB: 3,210 (4 / 3), 3,322 (6 / 4), 3,334 (8 / 5);
  - L 1 MB: 3,300, 3,326, 3,328.

  None can pass the rule's objective clause.

## Report only: what the head gate does to these numbers

Rubi runs the gated blobs, with `ocr_head_gate = on` (amendment 1). For Rubi 0.9.8 as it ships today, the VOCR5
gated blobs with the counter-context re-read (`eval/vocr6_final_lm.py gated-cache`), against v0.3.0's blobs with the
gate off, at the 1 MB setting:

- **Real units.** The objective moves +0.008 (S + cascade), +0.004 (N + cascade) and +0.012 (L, +0.001 to +0.024),
  with 47-77 units changed.
- **Counter set.** The crops keeping the counter go 3,127 → 3,093, 3,064 → 2,987 and 3,200 → 3,179 (kept share
  resolved lower at every tier). The CER rises +0.06 / +0.15 / +0.06 points (resolved).
- **Where the losses are** (N alone at the 1 MB setting, no cascade, no display rules; `eval/vocr6_final_lm.py
  counter-loss`, in the result's `reports.counter_loss_vocr5_N_1000k`):
  - 75 crops lose the counter, and 0 gain one.
  - All 75 are the bare and `…」` forms (33 and 42), whose first character is the numeral. No `「…」` crop is among
    them.
  - They are cut 2-10 px into the ink: 8 / 19 / 19 / 24 / 5 crops at 2 / 4 / 6 / 8 / 10 px.
  - With the gate on, none of the 75 greedy texts starts with a numeral (53 of 75 with the gate off), and 42 of the
    75 have identical greedy text both ways.
  - With the gate off, the LM restores the numeral from the edge frames' top-K. With the gate on, all 75 runs skip
    frames there, none finds a counter context, so none is read again.

**The blobs as they ship.** vocr6-final-gate's result of 09:45:42Z gives N and L the two-level head and no gate;
Small keeps `VOCR5_S_w8g`. Read this way (amendment 1, report only):

- **The 2 MB change on these reads** (Normal job, new − shipped):
  - dev score −0.032 (−0.096 to +0.023);
  - counter set: kept 3,295 → 3,287, CER −0.076 (−0.119 to −0.028);
  - the 4b blob's own (memorised) hval pages: hc −0.020 (−0.041 to −0.002), hp −0.279 (−0.961 to +0.040).

  These are close to the same comparison with the gate off: new − shipped at fixed settings, dev −0.034 (−0.098 to
  +0.022), counter-set CER −0.076.
- **The two-level head against no two-level head**, same weights and setting, at 1 MB, at the shipped 2 MB setting
  and at the new one:
  - Normal job: objective +0.003 / +0.004 / +0.005, with 22-26 units changed; the counter set moves by at most one
    crop.
  - Large: the objective is within ±0.001. The kept count is unchanged in all three comparisons, and the errors move
    by at most one crop.
  - Small, with its head gate in front of the cascade (report only): 13-15 fewer crops keep the counter (kept share
    −0.33 to −0.38 points, resolved). The objective moves +0.009 to +0.014.

Files: `artifacts/vocr6-final-lm/gateon_final_*.json`, embedded in the result as `reports`.

## Limits

- **hval is vocr6-lm's page path,** GRAD and GRAD + refine on the ingest's regions. It is not Rubi's shipped finder
  (COLGAP), and PCSG00854's junk line weighs on every page-path comparison.
- **The 4b blobs trained on hval.** The rule reads hval through their 4a siblings, which share the recipe and seed
  but not the hval pages, so the hval half measures the setting on a recognizer as sharp as the final one, not the
  final one itself. The 4b blobs' own hval is reported.
- **The development references** are engine-agreement labels with an assistant reviewer, not hand-checked.
- **N's sibling `v6s2cil_N_seed1` is vocr6-lm's #141 S2cil seed 1.** Its N-alone retune numbers were known before
  this protocol.
- **Two-fold cross-validation over 1,056 options is noisy.** At all 12 job tiers × LM tiers, the two folds chose
  different options (in the result's `chosen`).
- **No console run;** reads with the head gate off except the report-only section.

## Reproduce

```sh
ART=$HERD/shared/artifacts; A=$ART/vocr6-final-lm
# 1. the reads (vocr6_lm.py dev-cache / hval-cache, vocr6_final_lm.py counter-cache), the four LM tiers' tunes, the rule
HEAVY_SLOTS=15 $HERD/heavy.sh bash $A/final.sh S=$ART/vocr5-train/blobs/v5f_S_seed2_w8.vocr \
  N=$ART/vocr6-final-train/blobs/v6f4h_N_seed1_w8.vocr RN=$ART/vocr6-s2-a/blobs/v6s2cil_N_seed1_w8.vocr \
  L=$ART/vocr6-final-train/blobs/v6f4h_L_seed1_w8.vocr RL=$ART/vocr6-final-train/blobs/v6f4_L_seed1_w8.vocr
# 2. report only: the VOCR5 dry run (gate-off tune + continuity), Rubi 0.9.8 gate on, the shipped VOCR6 blobs gate on,
#    and the counter-loss breakdown
HEAVY_SLOTS=15 $HERD/heavy.sh bash $A/dry.sh
HEAVY_SLOTS=15 $HERD/heavy.sh bash $A/gateon.sh
HEAVY_SLOTS=15 $HERD/heavy.sh bash $A/gateon_final.sh
HEAVY_SLOTS=15 $HERD/heavy.sh bash $A/counter_loss.sh
# 3. the record (every report embedded)
.venv-ci/bin/python eval/vocr6_final_lm.py summary --select $A/select.json --protocol eval/vocr6_final_lm_protocol.json \
  --extra gateon_vocr5_1000k=$A/gateon_vocr5_1000k.json --extra continuity_vocr5_N=$A/continuity_N.json \
  --extra gateon_final_2000k_new_minus_shipped=$A/gateon_final_2000k_new_minus_shipped.json \
  --extra gateon_final_2000k_new_on_minus_off=$A/gateon_final_2000k_new_on_minus_off.json \
  --extra gateon_final_2000k_ship_on_minus_off=$A/gateon_final_2000k_ship_on_minus_off.json \
  --extra gateon_final_1000k_ship_on_minus_off=$A/gateon_final_1000k_ship_on_minus_off.json \
  --extra counter_loss_vocr5_N_1000k=$A/counter_loss_vocr5_N_1000k.json \
  --out $A/result.json --md $A/result.md     # = eval/vocr6_final_lm_result_20261027.json except created
```

`continuity_N.json` is the dry run's `continuity` output (`$HERD/shared/private/vocr6-final-lm/dry/continuity_N.json`,
copied to `$A`).
