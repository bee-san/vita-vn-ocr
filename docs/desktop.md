# A desktop line detector for VOCR: ONNX Runtime and C (issue #169)

MeikiOCR's author suggested this study: a line detector for desktop apps, between textspot8 (VOCR's 52 KB console
spotter) and MeikiOCR's detector with VOCR's recognizer, close to MeikiOCR's accuracy at a fraction of its time. It
runs with VOCR6.2's recognizer on ONNX Runtime's CPU provider, or in plain C. The protocol is
`eval/desktop_protocol.json`, with six amendments, each registered before the reads it governs. The records are
`eval/desktop_dev_20261101.json` (the dev-v2 decision reads), `eval/desktop_final_20261101.json` (the decision and
every final read), `eval/desktop_timing_20261101.json` (every timing, memory figure, profile and MAC count) and
`eval/desktop_report_20261102.json` (the figures derived from them: width classes, error breakdowns, the C path's
checks, mida's reads, the budget timings).

**The result: a research pre-release, not adopted.** "R" below is MeikiOCR's detector with VOCR6.2-L, the reference
the rule compares against. With VOCR6.2-L, the new detector (210 k parameters) reads the hooked test titles far
better than R. Its page takes 0.25-0.31 of R's time and under a fifth of MeikiOCR's. But it reads unseen VNDB console
screenshots (dev-v2) 0.2-0.5 points worse than R, and real PC screenshots 1-2 points worse. So it fails the
registered accuracy clause, which asks for a CER within 20% of R's on dev-v2 and on the PC test set. Every number is
measured on the host.

| dialogue CER | dev-v2: 84 unseen VNs, console screens | PC test: unseen VNs, PC screens | test titles: 2 hooked games |
|---|---:|---:|---:|
| MeikiOCR 0.3.5 itself | 2.54% | 1.97% | 5.91% |
| R: MeikiOCR's detector + VOCR6.2-L | **0.77%** | **0.70%** | 3.50% |
| textspot8 + VOCR6.2-L | 1.35% | 3.88% | 2.05% |
| **this detector + VOCR6.2-L** (the release model, long:640) | 1.26% | 2.75% | **1.19%** |

| page at a 4-thread budget (p50 ms: Vita / PSP / PC timing pages) | ms | peak memory |
|---|---|---:|
| MeikiOCR 0.3.5 itself | 120 / 106 / 122 | 670 MiB |
| R | 74 / 71 / 77 | 245 MiB |
| **this detector + VOCR6.2-L, ONNX** | **21 / 18 / 24** | 160 MiB |
| **the same in C** (`desktop/`: both networks in C, called from Python) | 24 / 20 / 26 | 83 MiB (22-30 MiB as a C program) |

The decision's own timing put the representative seed at 0.247 / 0.222 of R's time on the Vita / PSP pages, just
under the rule's 1/4; the release model reads at 0.28 / 0.25 / 0.31 of R's in the report's timing above.

## Thanks

- **MeikiOCR's author** proposed the detector and asked how much of ONNX Runtime's speed comes from threads and how
  much from fusion and vectorisation. Their dynamic-size variant of MeikiOCR lost accuracy at sizes it was never
  trained on, so this detector trains at many sizes from the start. They advised comparing every pipeline on frames
  resized to 960×544, MeikiOCR's own detector size. They also shared a new detector of their own, measured here as a
  report-only arm ("mida").
- **astra's report** (`docs/onnx.md`) and **an outside researcher's SIMD C build of VOCR** (numbers the owner
  forwarded) set the comparisons with SIMD C builds and the peak memory beside every time.

## The files

Pre-release **v0.5.2-desktop.1** (research, not adopted):

| file | bytes | what |
|---|---:|---|
| `vocr_desktop_det.onnx` | 844,538 | the detector, float32 ONNX (opset 17; any input size, multiples of 32 after padding) |
| `vocr_desktop_det.vdsk` | 826,868 | the same weights for the C path (`desktop/vocr_desk.c`) |
| `vocr_desktop.py`, `vocr_onnx.py` | | the Python pipeline from `tools/`: detector, reading order, recognizer, character cells |
| `VOCR6.2_L_fast.onnx`, `VOCR6.2_L_exact.onnx` | | VOCR6.2-L, byte for byte from v0.5.1-onnx.2 |
| `VOCR6.2_L_w8_2l.vocr` | | VOCR6.2-L's int8 blob, byte for byte from v0.5.0 (what the C path reads) |
| `DESKTOP-ATTRIBUTION.md`, `MODEL_CARD_DESKTOP.md`, `desktop-manifest.json`, `SHA256SUMS` | | terms, card, sizes and sha256 |

The C source is `desktop/` with `runtime/vocr.c`. Build it from the repository after #175's merge: the tag's commit
(`1192fa8`) lacks the input bounds the audit asked for (a size policy or a model file could make the arena too small;
"The C path" below).

There is no int8 detector. A static QDQ export, per channel and calibrated on 64 hval pages, keeps the FP32 file's
crops on only 63 of 313 dev-v2 frames. As a network it is only 1.25 times faster: 7.4 against 9.3 ms
(`eval/desktop_int8.py`).

## Results

### The sets

- **dev-v2, the decision set.** #119's 339 whole-dialogue units of 84 unseen VNs, cut from VNDB screenshots: PS Vita
  at 960×544, PSP at 480×272, and PC screenshots scaled to fit 960×544 as #119 cut them. The 9 of #119's 93 VNs that
  pools v2.1 trains on are left out.
- **The PC test set, a decision set since amendment 2.** desktop-data's set: 942 dialogue lines of 142 VNs that no
  training pool holds, on 504 VNDB PC screenshots at their own size. The page path reads its 518 clean units: 886
  lines of 139 VNs, with no other text inside their regions, on 480 of the screenshots, 640-2560 px wide and
  370-1440 px tall (`docs/desktop-data.md`).
- **hval, screening only.** The held-out pages of the hooked pools v2.1 (#166): 2,836 units of 71 titles.
- **confirm-v2, a guard.** textspot7's confirmation units without the 4 of its 41 VNs that pools v2.1 now trains on:
  155 units of 37 VNs.
- **The test titles.** The two titles the hooked corpus marks test: Dance with Devils (PCSG00807, Vita) and
  うたわれるもの PORTABLE (NPJH50882, PSP). 998 units, read once.
- **The PC sample, report only.** dev-v2's units cut from PC screenshots, read at the screenshot's own size: 115
  units of 51 VNs.
- **Not read:** vita-test, psp-test and CLANNAD.

**How a page is read.** Every pipeline reads the whole frame the same way.
1. The detector runs on the frame.
2. Every horizontal box whose centre lies in a dialogue unit's region is cut from the frame. VOCR6.2-L's fast ONNX
   file reads it: greedy CTC, no LM, no line filter.
3. The unit's texts, in reading order, are scored against its lines, as `eval/pagepath.py` scores a unit.

Line recall is the share of lines found whole; box precision is the share of boxes that are not junk. The intervals
are paired 95% cluster bootstraps: by VN on dev-v2, the PC sets and confirm-v2, by title on hval, by page on the test
titles.

### Accuracy

Dialogue CER, in %. Δ is the arm minus R, in points, with its interval. "Representative" is the seed the rule judged
(F3, trained without hval); "release model" is its retrain with hval folded in, which is the published file.

| arm | dev-v2 (Vita / PSP) | PC test | confirm-v2 | hval | test titles (Vita / PSP) |
|---|---|---|---|---|---|
| MeikiOCR 0.3.5 itself | 2.54 (0.61 / 4.16) | 1.97 | 2.81 | 4.65 | 5.91 (2.45 / 16.16) |
| **R** | **0.77** (0.35 / 1.12) | **0.70** | **1.15** | 2.35 | 3.50 (1.35 / 9.87) |
| textspot8 + L | 1.35 (1.16 / 1.52) | 3.88 | 2.35 | 2.61 | 2.05 (0.90 / 5.48) |
| mida + L, report only (amendment 5) | 1.42† (0.87 / 1.88) | 1.38 | 1.98 | 4.25† | 2.31 (0.67 / 7.13) |
| representative, long:640 | 1.10, Δ +0.33 [−0.04, +0.83] | 2.78, Δ +2.07 [+0.82, +3.74] | 1.40 | 2.65 | |
| representative, long:960 | 0.98, Δ +0.21 [−0.10, +0.61] | 1.67, Δ +0.97 [+0.34, +1.77] | 1.13 | 1.30, Δ −1.05 [−2.10, −0.06] | |
| **release model, long:640** | 1.26 (0.81 / 1.64), Δ +0.49 [+0.08, +1.01] | 2.75, Δ +2.04 [+0.82, +3.61] | 1.47 | (trained on it) | **1.19** (0.45 / 3.38), Δ **−2.31 [−2.96, −1.71]** |

† mida-ocr-eval's reads, with this harness, on the same frames, in the registered configuration (input fit into
960×544). `eval/desktop_report_20261102.json` (`mida.reads`) holds them, paired with R: dev-v2 +0.65 [+0.13, +1.33],
hval +1.90 [−2.00, +9.28].

- **The hooked games' console screens** are the detector's own domain. On the test titles it makes a third of R's
  errors. In point it reads hval better than R at every policy but long:640 and fit:640. At those two, PCSG00854 costs
  it about 2 points: VOCR6.2-L's reading of that title changes by hundreds of errors with a 1 px change of a crop's
  height (`docs/textspot8.md`).
- **VNDB console screenshots (dev-v2).** At long:960 and long:640 it is level with R within the interval, but
  0.2-0.3 points behind in point. At long:960 its 16 extra errors come from one missed PSP line (8 errors), 7 cut
  lines (11) and junk boxes (2). On the 324 units both read whole, it makes 3 fewer errors than R.
- **PC screenshots.** It is resolved worse than R at every policy.
  - Its line recall is 0.938-0.963 across the policies, against R's 1.000. On the PC test set the release model cuts
    37 lines and misses 7; R cuts none.
  - Old 4:3 games are its weakest frames: up to 800 px wide, 130 frames of 37 VNs, mostly 640×480 and 800×600 (these
    width classes and the error breakdowns are in `eval/desktop_report_20261102.json`). There the representative reads
    4.91% at long:640 and 3.15% at long:960, against R's 0.94%. On frames 1920 px wide or more it reads 2.60% at
    long:640 and 1.52% at long:960, against R's 0.78%.
- **Against MeikiOCR itself**, this pipeline makes fewer errors on every console set. On the PC test set it makes more
  at long:640 (2.75% against 1.97%) and fewer at long:960 (the representative: 1.67%). MeikiOCR's own recognizer reads
  console-size text poorly: 4.16% on dev-v2's PSP view and 16.16% on the PSP test title.

### Like-for-like: every frame fit into 960×544 (amendment 4)

MeikiOCR's detector scales every image to fit 960×544, so its author advises giving every pipeline frames already
resized to that size. Then every detector reads the same pixels: a Vita frame stays as it is, a PSP frame is doubled,
and a 1920×1080 frame becomes 960×540. On these frames this detector reads at fit:960, as MeikiOCR's does.

| arm (frames fit into 960×544) | dev-v2 | PC test | confirm-v2 | hval | test titles |
|---|---:|---:|---:|---:|---:|
| MeikiOCR itself | 2.47 | 2.65 | 2.76 | 4.44 | 5.80 |
| R | **0.68** | **0.77** | **1.13** | 2.31 | 3.43 |
| textspot8 + L | 1.47 | 3.51 | 2.21 | 2.46 | 2.02 |
| mida + L | 1.33† | 1.59 | 2.03 | 4.25† | 2.28 |
| representative, fit:960 | 1.00 | 1.71 | 1.27 | **1.29** | |
| release model, fit:960 | 1.24 | 1.65 | 1.43 | | **0.60** |

Against R on these frames:
- dev-v2: +0.32 [+0.03, +0.67] (the representative);
- the PC test set: +0.88 [+0.26, +1.71] (the release model);
- confirm-v2: level;
- hval: −1.02 [−2.07, −0.03] (the representative);
- the test titles: −2.84 [−3.67, −2.11] (the release model).

Page time on these frames at 4 threads (p50 ms, Vita / PSP / PC timing pages). Each detector ran at 4 intra-op
threads only, not at the budget's fastest variant:

| arm | ms |
|---|---|
| MeikiOCR itself | 114 / 102 / 114 |
| R | 71 / 70 / 71 |
| textspot8 + L | 23 / 22 / 22 |
| mida + L | 39 / 40 / 39 |
| this detector at fit:960 + L | 27 / 28 / 28 |
| the C page at fit:960 | 26 / 25 / 28 |

### Input sizes: training at many sizes works

MeikiOCR's author found that a detector trained at one size loses accuracy at others. This one was trained on windows
of pages scaled by a log-uniform factor in 0.42-1.9, so it saw text about 6 to 60 px tall. The screening's control
(S4) is the same network trained near one scale (0.8-1.25). hval CER, all titles / without PCSG00854:

| hval CER | fit:960 | fit:640 | fit:480 | long:960 | long:640 |
|---|---|---|---|---|---|
| multi-scale d48 (S1) | 2.66 / 1.22 | 1.31 / 0.90 | **0.97 / 0.94** | 2.47 / 1.03 | 1.15 / 0.73 |
| single-scale d48 (S4) | 1.59 / 1.29 | 3.10 / 1.33 | 3.48 / 3.53 | 0.98 / 0.68 | 2.44 / 0.66 |

- The single-scale model is good near its own scale and falls off when the text gets smaller. At fit:480 it reads
  3.53% without PCSG00854, against the multi-scale model's 0.94%.
- PSP frames read better scaled up. S1 reads hval's PSP pages at 1.12% at their own size and at 0.42% at long:1280.
  So the candidate policies include long:N, which scales the longer side to N px either way.
- The representative's dev-v2 CER is 0.98-1.77% across the 7 policies. It is worst at fit:800 and long:800, where one
  dev VN (v21004) makes 53 more errors than R.

The policies, as `tools/vocr_desktop.py` and `desktop/` read them:
- **fit:N** scales the longer side down to N px when it is longer, never up.
- **long:N** scales the longer side to N px, up or down.
- **native** keeps the frame as it is.

**Per resolution.** dev-v2's frames resampled to each height, aspect kept. CER on all of dev-v2; page p50 on its Vita
timing pages at 4 threads (each detector at 4 intra-op threads only):

| height | CER: R | CER: release, long:640 | CER: release, fit:960 | ms: R | ms: ONNX, long:640 | ms: C, long:640 |
|---|---:|---:|---:|---:|---:|---:|
| 720 | 0.73 | 1.22 | 1.00 | 71 | 21 | 23 |
| 1080 | 0.65 | 1.09 | 0.96 | 79 | 25 | 27 |
| 1440 | 0.68 | 1.21 | 1.04 | 84 | 33 | 27 |
| 2160 | 0.66 | 1.16 | 1.00 | 147 | 131 | 37 |

At 2160 px most of the Python page is NumPy's integer luma of the 3840×2160 frame: 73 ms per conversion in a later
check (the median of 50 calls at load 30; `luma_ms` in the report record). The timing harness converts twice for this
detector, once in the detector and once for the crops; `read_frame` converts once, and R's pipeline converts once.
The C page converts in C and stays at 37 ms.

### Speed and memory

**What is timed.** The timing pages are 60 Vita and 60 PSP frames of dev-v2 and 30 frames of the PC sample. A page
is the whole pipeline:
- the detector on the frame (preprocessing, network, decode);
- every horizontal box cut, its dark flag set and the line read by VOCR6.2-L.

Every arm is interleaved per page within a run, so a load spike on the shared host meets every arm alike. The loads
were 35-52 on 64 vCPUs: the 1-minute load as each run ended (the records' `load_1min_at_start` was read after the
passes; `eval/desktop_time.py` now reads it before them). p50 ms per page (Vita / PSP / PC):

| arm | 1 thread | 4-thread budget | 8 threads |
|---|---|---|---|
| MeikiOCR 0.3.5 itself | 224 / 193 / 222 | 120 / 106 / 122 | 105 / 90 / 105 |
| R | 136 / 131 / 138 | 74 / 71 / 77 | 63 / 57 / 66 |
| textspot8 + L | 27 / 22 / 27 | 19 / 17 / 20 | 29 / 25 / 30 |
| mida + L | 50 / 46 / 54 | 40 / 38 / 42 | 39 / 36 / 41 |
| **this detector, long:640 + L** | 30 / 24 / 31 | **21 / 18 / 24** | 22 / 19 / 25 |
| this detector, fit:960 + L | 39 / 20 / 43 | 29 / 14 / 32 (4 intra-op threads) | 30 / 15 / 35 |
| **C page, long:640** (C detector + `runtime/vocr.c` W8I) | 48 / 37 / 47 | 24 / 20 / 26 (4 C threads) | 24 / 20 / 28 |
| C textspot8 + C W8I | 49 / 43 / 48 | 32 / 31 / 34 (4 C threads) | 33 / 31 / 34 |

- **The budget (amendment 1).** Each detector runs at whichever of 1, 2 or 4 intra-op threads is fastest for it, per
  view (`budget_t4` in the report record names the variant). The lines are read on 4 Python threads sharing one
  recognizer session, or on 4 C threads. The fit:960 and C rows ran one variant only.
- **MeikiOCR's sessions.** MeikiOCR and R run on the harness's ONNX Runtime sessions, with ONNX Runtime's default
  intra-op spinning; meikiocr 0.3.5's own constructor turns spinning off. On the baselines' Vita pages that made
  MeikiOCR 25% slower at 2 threads (158.7 against 126.8 ms with spinning off) and 8% faster at 4 (111.9 against
  121.0 ms; `baselines.page_t2` and `page_t4` in the timing record). The 4-thread figures here are the faster
  setting; at 8 threads only the default was measured. R's fastest budget variant is its detector at 4 threads, also
  the faster setting there. R's detector at 2 threads without spinning was not measured: scaled as MeikiOCR's pipeline
  scales (101.8 ms × 126.8 / 158.7 ≈ 81 ms), it would still be slower than 74.2 ms at 4.
- **The decision's own timing** (`eval/desktop_timing_decide_20261101.json`, load 38) puts the representative at
  long:640 at 0.247 / 0.222 of R's time on the Vita / PSP pages, under the rule's 1/4.
- **A cost against this detector in these timings.** The harness converts each frame to luma twice for the grey
  detectors (this one and textspot8): 0.6 ms per 960×544 frame at load 30 (`luma_ms`; the auditor measured 2.9 ms at
  load 44).
- **Peak memory** per process (`desktop_time.py mem`: 60 dev-v2 frames, 4 threads, a fresh process per arm),
  including about 53 MiB of interpreter and libraries:

  | arm | peak MiB |
  |---|---:|
  | MeikiOCR | 670 |
  | R | 245 |
  | mida + L | 206 |
  | this detector + L at fit:960 | 196 |
  | this detector + L at long:640 | 160 |
  | textspot8 + L | 146 |
  | the C page | 83 |
  | C textspot8 + C L | 78 |

- **As a C program.** `desktop/vocr_desk_cli` reads the same pages at 1 / 4 threads, at 22-30 MiB peak. These are
  means per page (the CLI's summary), not p50s:

  | pages | mean ms |
  |---|---|
  | Vita | 48 / 25 |
  | PSP | 41 / 20 |
  | PC | 54 / 26 |

  The detector takes a mean 7-13 ms of that, resize and decode included.
- **MACs:**

  | network | MACs |
  |---|---:|
  | this detector, a 16:9 frame at long:640 (a 640×384 input) | 380 M |
  | this detector, 960×544 (fit:960) | 807 M |
  | this detector, 480×288 (a PSP frame at fit:960, or any frame at fit:480) | 214 M |
  | MeikiOCR's detector, per frame | 4.41 G |
  | MeikiOCR's recognizer, per line | 0.69 G |
  | textspot8 | 87 M |
  | mida, 960×544 | 1.04 G |
  | VOCR6.2-L, per 8 px of line | 3.40 M |
  | VOCR6.2-L, per 640 px line | 272 M |

- **Weights:**

  | network | parameters | file |
  |---|---:|---:|
  | this detector, in training | 209,848 | |
  | this detector, BatchNorm folded | 206,680 | 844,538 B ONNX |
  | MeikiOCR's detector | 3.55 M | 14.5 MB |
  | MeikiOCR's recognizer | 4.57 M | 18.6 MB |
  | mida | 430 k | 1.75 MB |
  | VOCR6.2-L | | 2.5 MB fast ONNX |

### Threads, fusion and vectorisation (MeikiOCR's author's question)

The networks alone, ms per input:

| network | graph optimisations off, 1 thread | on, 1 thread | on, 2 / 4 / 8 threads |
|---|---:|---:|---|
| MeikiOCR's detector, 960×544 | 131.3 | 99.4 | 88.0 / 59.1 / 41.9 |
| textspot8 (exact) | 8.2 | 6.9 | 12.0 / 11.2 / 11.9 |
| VOCR6.2-L fast, per line | 3.40 | 1.79 | 1.87 / 1.52 / 1.31 |
| VOCR6.2-L exact, per line | 6.7 | 6.5 | 8.5 / 7.8 / 6.8 |
| this detector, long:640 | 17.8 | 9.9 | 13.9 / 11.3 / not run |

The first four rows were measured on a quiet box (load 5-7), the last at load 36-40.

- **MeikiOCR's 4.4 G MAC detector** is big enough to split: 8 threads make it 2.4 times faster, graph fusion and
  layout 1.3 times.
- **VOCR's small graphs** gain little from ONNX Runtime's threads, or lose time.
- **Fusion and layout** make textspot8, the fast recognizer and this detector 1.2-1.9 times faster. The exact
  recognizer gains only 1.03 times. On this detector ONNX Runtime fuses most ReLUs into the convolutions, runs them in
  its blocked (NCHWc) layout, and spends 85% of the kernel time in convolutions.
- **The rest of the old gap was the compiler build.** VOCR's C runtime built with gcc `-O2` against
  `-O3 -march=native` takes 31.8 against 6.3 ms per VOCR6.2-L line, and 41.8 against 17.0 ms per textspot8 frame
  (#164's subject).
- **The fast ONNX recognizer** is faster still (1.8 ms) because its arithmetic is the C runtime's W8F mode: int8
  weights dequantised, float activations, no int16 activation steps (`docs/onnx.md`).
- **The C detector** (AVX2 + FMA kernels, `-O3 -march=x86-64-v3`) takes about as long as ONNX Runtime's network
  alone: on the Vita pages at long:640, 11.6 ms at 1 thread and 8.9 ms at 4, resize and decode included.

## The rule and the decision

The rule was registered in #169's body at 03:20Z on 2026-11-01, before any baseline was scored: the first baseline
results were written at 03:33Z. It has three clauses and a pick:
- **Accuracy on dev-v2:** the representative seed's CER ≤ 1.2 × R's, its paired interval not entirely above 0, and the
  Vita and PSP views each ≤ 1.5 × R's.
- **Guards:** hval and confirm-v2 not resolved above 0.
- **Speed:** at the 4-thread budget, p50 ≤ 1/4 of R's on the Vita and the PSP pages separately.
- **Pick:** the cheapest qualifying input policy.

**Where and when it was recorded.**
- **The experiment log's protocol line** is stamped 03:40Z. That is after the dev-v2 and PC-sample baselines were
  scored (03:33-03:34Z) and while the hval baselines ran. The issue body, which holds the whole rule, came first.
- **An erratum.** `eval/desktop_protocol.json` (written at 04:04Z, before any candidate existed) says the rule was in
  the log and the issue body at 03:40Z, before any baseline. That is wrong for the log line; the file is left as
  registered.
- **`rule.pick`.** Amendment 2 appended its policy list to the registered `rule.pick` text, which survives as an
  unchanged prefix. The other changes are amendment entries.
- **The per-run lines.** The experiment log's line for each training run (S1-S5, F1-F3 and the retrains) was
  appended after the decision, at 22:10Z, not as each run ran. Each run's own `run.json`, `done.json` and the
  heavy.log record its times.

**Amendments** (all in `eval/desktop_protocol.json`):
1. the thread budget: each detector at its fastest intra-op count;
2. the owner's updates:
   - the C path, peak memory and the SIMD-C comparisons;
   - the PC test set as a decision set, with the same 1.2 × R clause;
   - per-resolution reports;
   - the long:N policies, the pad rule and an extra screening arm;
3. the final runs' data: desktop-data's pages;
4. the like-for-like rows;
5. mida as a report-only arm;
6. a research release if nothing qualifies.

**Screening** (hval only, one seed each, 40,000 steps, all multi-scale):

| arm | parameters | MACs at 960×544 | hval CER at fit:960 |
|---|---:|---:|---:|
| S3 d32 | 50 k | 247 M | 4.93% |
| S1 d48 | 127 k | 506 M | 2.66% |
| S2 d64 | 210 k | 807 M | **1.30%** |

S4 (single-scale) and S5 (a ruby recipe) were run beside them.
- **Shape: d64**, which has the lowest hval CER at fit:960.
- **Recipe: unchanged.** S5 made the hooked pages' ruby background, with ruby-like MeikiOCR boxes as ignore areas. Its
  mean over the 7 policies was 2.63%, against S1's 2.37%, so it was not adopted.
- **Pad: the frame pad**, MeikiOCR's reference pad: the ink box + max(2, round(4 H / 544)) px. Its mean was 2.00%,
  against 2.37% for 18% of the size across.

**The final runs.** d64 with desktop-data's pages added at 25% of the mix (amendment 3), seeds 1-3, 40,000 steps,
about 9 h each on 4 threads.
- The representative is seed 3. By dev-v2 CER at fit:960, seeds 2 and 3 tie at 1.009% and seed 1 reads 1.341%. The
  registered rule takes the median of the three sorted by CER, and the tie was broken by seed number, so seed 3 is the
  median.
- Each seed's retrain on train + guard + hval ran beside it. The release model is seed 3's retrain.

**The verdict** (`eval/desktop_rule.py`): **no qualifier.**

| policy | dev-v2 (limit 0.924%) | PC test (limit 0.845%) | speed, Vita / PSP (limit 0.25) |
|---|---:|---:|---|
| fit:960 | 1.009 | 2.194 | 0.331 / 0.172 |
| fit:800 | 1.766 | 2.283 | 0.301 / 0.170 |
| fit:640 | 1.182 | 2.777 | **0.247 / 0.175** |
| fit:480 | 1.208 | 3.209 | **0.187 / 0.174** |
| long:960 | 0.983 | 1.674 | 0.377 / 0.347 |
| long:800 | 1.700 | 1.852 | 0.312 / 0.284 |
| long:640 | 1.102 | 2.777 | **0.247 / 0.222** |

- **No policy meets the accuracy limit** on dev-v2 or on the PC test set. The dev-v2 intervals reach below 0 at
  fit:960, long:960 and long:640, but the 1.2 × R limit is not met.
- **The guards hold.** No policy is resolved worse than R on hval or confirm-v2.
- **Three policies meet the speed clause:** fit:640, fit:480 and long:640.
- **The release** follows amendment 6: a research pre-release of the retrain at long:640, the policy with the lowest
  dev-v2 CER of those that meet the speed clause.

## What would close the gap

Most of the gap is detection: cut line ends, missed short lines, junk boxes and the odd merged pair of lines, mostly on
PC screenshots.
- At long:960, the representative's dev-v2 units with a cut or missed line or a junk box hold 21 more errors than R's;
  the rest hold 5 fewer.
- At long:640, they hold 20 more, and the rest hold 5 more.

1. **Line ends.** The decode grows a box only by the frame pad. mida's decode extends each end along the line by the
   box's size across, and it cuts fewer lines on the PC test set (14, against the release model's 37). An along-the-line
   extension, chosen on hval, is the cheapest next step.
2. **Old 4:3 PC games.** Frames up to 800 px wide (130 frames of 37 VNs, mostly 640×480 and 800×600) read at
   2.7-5.1% on the representative's policies, against R's 0.9%. Their own size is the worst scale: 5.10% at fit:960,
   which leaves them as they are. Reading them larger (long:960: 1.2-1.5 times) gives 3.15%, and smaller (fit:480:
   0.6-0.75 times) 2.69%.
   So the scale matters in both directions. A policy keyed to the text's size, or two scales with the more confident
   boxes kept, is worth trying, chosen on hval and dev-v2 first.
3. **Large frames read small.** Frames 1920 px wide or more lose small text at long:640: 2.60%, against 1.52% at
   long:960.
4. **Cleaner PC labels.** The final runs trained on desktop-data's first freeze, whose realplus pages are v1. v1 has
   two-row positives in 223 frames and unlabelled text in 564; desktop-data fixed both in v3 (`docs/desktop-data.md`,
   1b). Two-row positives can teach a detector to merge lines: the release model merges 7 line pairs on the PC
   test set, R none. Whether the v3 labels help is untested.
5. **More training and capacity.** These are 40,000 steps of a 210 k parameter network. MeikiOCR's detector is 17
   times larger.

## How the detector works

**The network** (`train/desktop_det.py`, shape d64; 0.38-0.81 G MACs per frame):
- a 3×3 stem at stride 2;
- four stages at strides 4, 8, 16 and 32, with 32, 80, 128 and 160 channels (the stem has 16). Each stage has a
  separable down-sampling block, then 1, 4, 3 and 2 separable residual blocks: 3×3 depthwise with dilations 1-4,
  then 1×1, BatchNorm and ReLU;
- a top-down feature pyramid back to stride 4 (64 channels, nearest 2× up-sampling), one separable smoothing block,
  and a 1×1 head.

**The head** gives, at stride 4:
- a line's kernel logit K and orientation logit O;
- A1 and A2, the distances to the line's two edges across it, in input px;
- 4 tag logits, for name, choice, UI and scene (textspot6's classes; dialogue scores 0).

**The input** is the grey frame as integer luma, (77 R + 150 G + 29 B) >> 8, as VOCR's recognizer takes it. It is
resized by the policy and padded right and bottom to a multiple of 32 by edge replication; the graph normalises it as
(g − 128) / 128.

**The decode** (`tools/vocr_desktop.decode`; `desktop/vocr_desk.c` gives the same boxes bit for bit on the same maps):
1. the 4-connected components of K > 0 with at least 2 cells;
2. per component, its confidence (the mean K logit) and its orientation (vertical when the mean O logit > 0);
3. across the line, the mean of the cells' centre − A1 and centre + A2; along it, the component's cells; lines under
   4 px across are dropped;
4. the box grown by the frame pad;
5. its tag, the first class with the largest summed tag logit;
6. at most 256 boxes, by confidence.

**Training** (40,000 steps, batch 16 windows of 256×512, 4 torch threads):
- AdamW at 3e-3, linear warm-up, cosine decay to 1e-4, gradient clip 5. The run's model is an EMA (0.999) of the
  weights from step 1,000, with its BatchNorm statistics re-estimated.
- textspot6's targets and losses, with textspot8's line-end weights, icons and tight windows.
- Every window's page is scaled by a log-uniform factor in 0.42-1.9 before the cut. A desktop-size page is first
  scaled to a 960 px long side.
- **The mix:**

  | source | share |
  |---|---:|
  | the hooked pools v2.1's train pages (`data/manifests/desktop_pages_v21.json`) | 34% |
  | textspot's synthetic frames | 15% |
  | textspot2's MeikiOCR-labelled real screenshots | 26% |
  | desktop-data's pages (`data/manifests/desktop_data/*_v1.json`, the first freeze), of which: | 25% |
  | the hooked train pages re-rendered as desktop captures (20,000) | 10% |
  | realplus PC screenshots at their own size (49,267) | 10% |
  | synthetic desktop frames (10,000) | 5% |

## Using it in an app

```python
from vocr_desktop import Detector, read_frame
from vocr_onnx import Recognizer

det = Detector("vocr_desktop_det.onnx", threads=1)    # long:640 by default; size="long:960" for PC screenshots
rec = Recognizer("VOCR6.2_L_fast.onnx")
lines = read_frame(frame, det, rec, workers=4)       # frame: H x W x 3 RGB (or grey) uint8, any size
for ln in lines:                                     # reading order: rows top to bottom, left to right
    ln.text, ln.box, ln.tag                          # box (x, y, w, h) in frame px; tag 0 dialogue, 1 name, ...
    for c in ln.text_chars:                          # each character's cell, for a hover or tap lookup
        c.text, c.x0, c.x1, c.y0, c.y1
```

- **Policies.**
  - long:640, the file's default, has the lowest dev-v2 CER of the policies that meet the speed clause.
  - long:960 reads PC screenshots best overall (the representative: 1.67% on the PC test set), at about 1.5 times
    the page time.
  - fit:N never scales a frame up, so it reads PSP's 480×272 frames at their own size; long:N scales them up, which
    reads them better.
- **Regions.** `read_frame(..., region=(x, y, w, h))` reads only a region, such as the dialogue box a user draws. The
  region is read at the scale and with the box pad that the policy gives the whole frame, as a whole-frame read would
  read it. No set here was read through a region.
- **Threads.** The detector is fastest at 1-4 intra-op threads here; `workers` reads the lines in parallel. Measure on
  the target machine.
- **The LM.** `read_frame(..., lm=vocr_onnx.CharLM(...))` adds Rubi's beam and character LM to each line
  (`docs/onnx.md`).
- **Vertical boxes** come back without text. Vertical text needs VOCR6-V and the column finder (`docs/vocr6v.md`).
- **C** (`desktop/vocr_desk.h`):

```c
int err;
vocr_desk_page *p = vocr_desk_page_new(det, det_bytes, rec, rec_bytes, 4 /* threads */, -1 /* the blob's policy */,
                                       0, &err);
vocr_desk_page_run_rgb(p, rgb, w, h, w * 3, 3);        /* or vocr_desk_page_run on a grey frame */
for (unsigned i = 0; i < vocr_desk_page_count(p); i++) {
    const vocr_desk_line *ln = vocr_desk_page_line(p, i);   /* ln->text (UTF-8), x, y, w, h, tag, dark, vertical */
}
vocr_desk_page_free(p);
```

Build with `make -C desktop` (AVX2 + FMA, x86-64-v3), `make -C desktop ARCH=` (plain C) or
`make -C desktop ARCH=-march=native`.

## The C path

- **What it is.** `desktop/` is C99 with libm and POSIX threads. It runs the detector from the VDSK blob, with AVX2 +
  FMA kernels when the compiler targets them and plain C otherwise.
- **The page.** The page runs the detector, then reads every horizontal box with `runtime/vocr.c` (unchanged, W8I, the
  blob's head sections as Rubi runs them). Its threads split the detector's steps into row bands, then share the
  lines. The output is the same bits at any thread count.
- **Not console code.** It allocates its own memory, and none of it is in Rubi.
- **Parity on dev-v2** (the release model at long:640; `c_crop_differences` and `c_checks` in the report record):
  - **Texts.** The C detector's boxes read by `runtime/vocr.c` (W8I, through the harness's crops) give the ONNX path's
    texts on all 339 units: 1.262% CER both ways. The ONNX path is the ONNX detector with VOCR6.2-L fast.
  - **The whole C page.** `vocr_desk_page_run_rgb` (its reading order, crop rounding, dark flags and `runtime/vocr.c`)
    gives the same 1,893 lines as that harness composition, on all 313 frames.
  - **Crops.** The C detector's crops equal the ONNX detector's on 276 of 313 frames. On each of the other 37, the
    largest edge shift of a matched box is:

    | largest edge shift | frames |
    |---|---:|
    | none (only the box count differs) | 2 |
    | 1 px | 9 |
    | 2 px | 1 |
    | 3 px | 22 |
    | 6 px | 1 |
    | 18 px | 2 |

    On 6 of them one path finds a box the other does not: 5 boxes only in ONNX, 1 only in C. 3 px is one map cell on
    a PSP frame at long:640.
  - **The cause is the resampler.** The ONNX network and the Python decode, run on the C path's own resize of each
    frame instead of OpenCV's, give the C detector's crops on all 313 frames. The C resampler lands within ±1 grey level
    of OpenCV's, and that is enough to move a decode near its threshold.
  - **The ship bar.** Amendment 2 required identical crops on 99% of dev-v2's frames before a pick could ship. This
    release is not a pick, and the C path does not meet that bar (88%).
- **Bounds** (audit-vocr-desktop F5, fixed after the release's tag).
  - A size policy is native, or fit:N or long:N with N of 1-8192; the library refuses any other with
    `VOCR_DESK_E_ARG`, and refuses a model file whose default policy is outside that with `VOCR_DESK_E_BLOB`.
  - Before the fix, a huge long:N, through the API or a crafted model file with a valid CRC, wrapped the arena's size
    and overflowed the heap.
  - The arena sizes saturate rather than wrap, and the resize and the decode check their own arguments.
  - The release model's lines are unchanged on all 150 timing frames (plain and AVX2 builds, 1 and 4 threads).
- **In CI** (`tests/test_desktop.py`, plain and AVX2 builds, random networks):
  - the C maps are within 1e-4 relative of ONNX Runtime's;
  - the C decode equals the Python decode bit for bit on the same maps;
  - frames read without a resize give identical crops.

## Limits and disclosures

- **The host.** One shared machine with 64 vCPUs (AMD EPYC 9R14, AVX-512), at loads of 30-50 during most timings.
  Other CPUs, Windows, macOS and GPU providers were not measured.
- **Labels.**
  - dev-v2's PC frames were scaled by #119 to fit 960×544, so dev-v2 is a console-scale set.
  - The PC test set's labels were drawn from MeikiOCR's and PP-OCRv6 small's readings and two Claude passes, with 214
    lines checked by desktop-data. MeikiOCR's readings among the candidates favour MeikiOCR, not this detector.
  - The set's erratum (`eval/desktop_pc_test_v1_errata.json`): 7 lines of v21170 have ink boxes that hold only the
    lower half of their glyphs. It was not applied. CER does not change; it can only flatter line recall, because a
    box that cuts those glyphs' tops still counts as whole.
- **Exposure.**
  - The detector trained on pools v2.1. PCSG00062 there is a vita-test and psp-test VN, and PCSG00829 a vita-test VN.
    PCSG00007 and ULJM05383 share a VNDB release with CLANNAD, and PCSG00866 releases with vita-test's Collar x Malice
    -Unlimited-. None of vita-test, psp-test or CLANNAD was read.
  - The test titles come from the same hooked corpus and capture setup as the training titles, so their result is
    in-domain, not unseen-game. Dance with Devils' fandisc PCSG01152 is a training title and shares a VNDB release
    with it. NPJH50882 shares 7 releases with vita-test's Utawarerumono: Itsuwari no Kamen (`docs/vocr6-data-v2.md`).
- **mida's numbers** use this repository's decode of its maps (`eval/mida_det.py`), because its author's decode is not
  published. Its own decode may read it better.
- **A screening checkpoint ran on dev-v2 frames once before the decision.** mida-ocr-eval timed vocr-desktop's S2
  checkpoint on dev-v2 frames, recording page times and line counts but no CER. Nothing here used those runs.
- **The final reads ran once.** The read script was edited while it ran, and bash stopped it between two arms on the
  PC test set and on confirm-v2. A copy was relaunched; it skipped every arm already read, so each arm read each set
  once.
- **The test titles were read without the representative.** Amendment 6's read list names the representative's 7
  policies on the PC test set, confirm-v2 and the test titles. The test titles' read was launched with `SKIP_F3=1`,
  so only R, MeikiOCR, textspot8, mida and the release model read them. The deviation was not recorded at the time.
  It means fewer protected reads than registered, so it cannot flatter a result, and the decision did not use the
  test titles. The representative will not be read on them now.
- **The release model trained before the decision**, beside its seed. The rule only chose which retrain to publish.

## Reproduce

The scripts as they ran are in `eval/reproducibility/desktop-20261101/`. Run them through `heavy.sh` (one slot, 5
threads, unless noted). They read the frames from the shared box and write counts only. The records' `repo_head`
commits (a19bede for the dev-v2 record, a614fd8 for the final reads) are the branch before its last rebase, kept by the
tag `desktop-reads-20261101`; the protocol as registered is the tag `desktop-registered-20261101`.

```sh
S=eval/reproducibility/desktop-20261101
bash $S/baselines.sh dev; bash $S/baselines.sh hval; bash $S/baselines.sh pc     # MeikiOCR, R, textspot8 + S/N/L
bash $S/baselines_res.sh; bash $S/baselines_l544.sh dev_l544 hval_l544          # per resolution, like-for-like
bash $S/train.sh S2_d64 d64 1 train                                              # a screening arm (S4: --scale 0.8,1.25)
HOOK_MANIFEST=desktop_pages_v21_rubybg.json HOOK_SHARDS=shards_v21_rubybg \
    bash $S/train2.sh S5_d48_ruby d48 1 train --ruby-filter                      # S5, the ruby recipe
bash $S/score_run.sh RUNS/S2_d64 hval "fit:960 fit:640@frame ..."                # its hval reads
python3 $S/screen_table.py S1_d48 S2_d64 S3_d32 S4_d48_single S5_d48_ruby        # the screening table
bash $S/train3.sh F3_d64_seed3 d64 3 train                                       # a final seed (amendment 3's mix)
bash $S/train3.sh final_d64_seed3 d64 3 train,guard,hval                         # its retrain
bash $S/post_seeds.sh                                                            # exports, dev-v2 reads, the representative
bash $S/timing_decide.sh MODELS/F3 F3                                            # the decision's timing
bash $S/final_reads.sh pctest; bash $S/final_reads.sh confirm                    # once each, after the dev-v2 record
SKIP_F3=1 bash $S/final_reads.sh test                                            # once, behind the protocol gate
bash $S/final3_reads.sh; bash $S/c_parity.sh                                     # the release model; the C path, INT8
python3 eval/desktop_rule.py --res RES --timing T4.json --candidates C.json --out DECISION.json
bash $S/timing_report.sh grid 4                                                  # also grid 1/2/8 (8: two slots), l544, res, mem
python3 $S/checks_20261102.py --out CHECKS.json                                  # luma; the C resampler and page checks
python3 eval/mida_det.py info MIDA.onnx --macs 960x544 > MIDA_INFO.json         # mida's size and MACs (measurement only)
python3 $S/report_20261102.py --out eval/desktop_report_20261102.json --checks CHECKS.json --mida-info MIDA_INFO.json
```

The tools: `eval/desktop.py` (sets, detectors, reads, scores), `eval/desktop_time.py` (timings, memory, profiles,
MACs), `eval/desktop_record.py` (the public records), `eval/desktop_rule.py` (the rule) and `eval/desktop_int8.py`.

## Licence

- **Code:** GPL-3.0-or-later.
- **Weights:** for **non-commercial use only**. The training inputs include non-commercial fonts and texts and the
  owner's private game captures (`DESKTOP-ATTRIBUTION.md` in the release).
- **MeikiOCR's role:** it labelled training screenshots, as it did for textspot8 (textspot2's real half).
- **mida's role:** measurement only. No mida output was trained on.
