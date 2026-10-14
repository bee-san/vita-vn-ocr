# The three recognizer tiers: Small, Normal, Large

The final deliverable is three models. The owner (2026-10-14) set the weight ceilings and the
speed targets; this document turns them into exact, selectable tier ceilings, prices one or two
architectures per tier against them with `train/export.py` random-init blobs and `vita/a9_cycles.py`
estimates, states the free-game-memory a job of each tier needs, gives the runtime selection rule
the overlay should apply, and then analyses every remaining gap of the current recognizers on
development data to say where the extra capacity should go.

**Everything about console speed and the memory a job holds is an estimate.** Nothing has run on a
console at these sizes; the only on-console numbers so far are FL10's load and recognition times in
CLANNAD, which were an API 13 build (`docs/vita-memory-budget.md` §2, `notes/26`). The gap analysis
is on development data only (teacher_val and devnative); no protected set (vita-test, psp-test, the
fresh native textbox test, CLANNAD) was read for it.

Machine-readable: `docs/model-tiers.json`, generated whole by `vita/model_tiers.py report
--taxonomy` (every candidate's sizes, MACs, estimated ms per 640-px line and per 3-line box at 333
and 444 MHz with the whole L2 and a 1 MiB share, the tier each fits, the free memory it needs, the
matched-recipe capacity curve, the recommendations). Tools: `vita/model_tiers.py` (sizing, the
memory rule) and `eval/model_gaps.py` (the taxonomy); tests `tests/test_model_tiers.py`.

## Answers first

The owner's final time target (07:33Z/08:01Z, superseding the earlier per-line limits) is **per
typical full textbox — 3 lines of about 640 px — on one Cortex-A9 core at 444 MHz, estimated**:

| | Small | Normal | Large |
|---|---|---|---|
| int8 weight ceiling | **< 1,000,000 B** | **≤ 4,000,000 B** | **≤ 9,000,000 B** |
| W8I arena ceiling (this doc) | ≤ 131,072 B | ≤ 262,144 B | ≤ 1,310,720 B |
| box time target (444 MHz, est.) | **≤ 1 s** | **≤ 2 s** | **≤ 3 s** |
| **binding constraint** | weights + arena (time has margin) | time + arena (not weights) | **time and the 2 MiB L2, not weights** |
| widest shape that fits | any ≤ 1 MB / 128 KiB here | up to ~2.5 MB, arena ≤ 256 KiB | ~3.0–3.25 width, big head, big tiles, ~3.1–3.7 MB |
| train now (vocr4-train's size) | FL10 (width 1.0, 3,500 vocab) | W20 (width 2.0) | W20 (fits with wide margin) |
| free game memory a first full job needs | **≈ 5.3 MiB** (FL10) / 5.6 MiB (ceiling) | **≈ 10.3 MiB** (W20) / 12.6 MiB (ceiling) | **≈ 18.3 MiB** (W20) / 21.0 MiB (next-round) |
| always loads? | yes (every game seen leaves ≥ 56 MiB) | preflight + Small fallback | preflight + Normal/Small fallback |

- **The 9 MB Large ceiling is far larger than the console can run in 3 s.** On the shared 2 MiB L2 a
  recognizer whose weights plus arena outgrow about 2.3 MB starts missing the L2 on every weight
  pass, and the estimated time turns superlinear (`docs/vita-memory-budget.md` §3). At the runtime's
  default 4-frame tiles the widest plain width-3.0 model (2.71 MB) is already ~2.9 s/box @444, and
  larger shapes are over 3 s. **Weights are not what caps Large; time and the shared cache are.** The
  envelope is reachable (up to about 3.7 MB) only with bigger tiles (16-frame tiles, 64-frame head batch, which trade
  arena for fewer L2 passes), and even then several shapes over 3 MB fit at about the same time (§1.4).
  The recommended Large is a parameter-heavy, compute-light wide-head shape, **`l_w300v6459b192`**
  (width 3.0, bottleneck 192, a 6,459-class VN-ordered vocab — vocab-expand's final, which covers the
  whole VN list without the Chinese-only classes a 7,501 head adds): 3.48 MB, est. 2.79 s/box @444.
  A plain width-3.25 model (`l_w325v5631`, 3.06 MB, 2.92 s/box) and a width-3.0 + 4-ctx model
  (`l_w300c4v5631`, 3.41 MB, 2.88 s/box) also fit. Which wins on accuracy per ms is unmeasured.
- **Small now has real speed headroom to trade for accuracy** (owner 07:33Z). Every Small candidate
  here is ≤ ~0.65 s/box @444, well under 1 s, so Small can spend its weight budget on a wider head
  (a VN-ordered vocab) rather than staying at FL10's 3,500 classes.
- **The gaps are as much data/finder as capacity.** On 28,544 development reference characters
  (console-polarity view) the three current VOCR3 selections read 2.29 % / 2.12 % / 1.81 % CER. The
  errors cluster where the data is thin or the crop is bad: **the line finder's crop, short 1–4-char
  dialogue lines, thinly-exposed/rare kanji, and katakana with dakuten/handakuten. Speaker-name plates
  (≈ 8–18 % CER, 44–48 %
  of all errors), the console line finder's own crops (≈ 6× the errors of a hand-cut crop on the
  same lines), kanji that the real pools expose thinly (highest error *rate*, though common kanji are still most of the kanji errors by count), katakana with dakuten/handakuten, and 1–4-character lines.** On
  matched-recipe runs the capacity curve pays most on the first step (FL10→H15 2.13 points/MB,
  H15→W20 0.80), and the full FL10→W20 gain is 1.38 points — about the size of the data/finder
  headroom, so capacity and data both matter and width is not the only lever (part 2).

## Part 1 — the budgets

### 1.1 Where a job's memory goes (facts from vita-jp-overlay #75, main `64b5726`)

OCR runs in the **game's own process** since API 19 (overlay #75): the kernel's display hook copies
the dialogue region, as 8-bit grey, a strip of rows at a time, straight into the worker's own
`USER_RW` block; nothing of the recognizer is in SceShell, and the kernel holds no capture buffer
(`docs/vita-platform-memory.md`, overlay `docs/ingame-capture.md`). What one job allocates in the
game's 256 MiB partition (`game_ocr/main.c`, `core/vocr_ocr.c`, `include/vjo_ledger_budget.h`):

| block | bytes | when | memblock type |
|---|---|---|---|
| verified weights | page-rounded int8 blob | resident between jobs while the title is active | `USER_RW` only (never CDRAM/phycont) |
| job workspace | page(max(finder arena, recognizer arena) + 3,760 B runtime structs) | per job, freed at the end | `USER_RW`, else `USER_CDRAM_RW`, else `USER_MAIN_PHYCONT_NC_RW` |
| capture block | ≤ 61,440 B (one strip + row checksums), any region | per job | same ladder |
| line block | only a band taller than a 64-row strip; ≤ 516,096 B (`VJO_VOCR_LINE_MAX_BYTES`) | per job | same ladder |
| worker module + stack + TLS | 53,248 + 32,768 + 4,096 B | permanent while the game runs | — |

Since API 19 the workspace holds **no copy of the region** (API 18 held a full grey plane: 712,704 B
for H15 at 960×544; now 188,416 B at any region). The capture block is under 64 KiB at any region,
and the line finder's arena (83,323 B at 960×544, the largest) is smaller than every recognizer
arena here, so **a job's memory no longer depends on the region** — only on the model. The dictionary
is disk-backed in SceShell and never resident; no dictionary pages are in the game.

The worker's preflight (`check_memory`): `free USER (size_user) − 4 MiB reserve ≥ page(workspace) +
page(capture) + page(weights unless resident)`. It refuses before allocating, logs the free bytes,
and falls back to a smaller pinned model that needs less. `sceKernelGetFreeMemorySize` returns the
free size minus a system margin; a negative figure means 0 free (overlay reads #74's `-2097152` that
way). The worker runs at the lowest user priority on any of the game's three cores, so a game thread
always preempts it (`game_ocr/main.c`, `WORKER_PRIORITY`). The weights are `USER_RW` only
(CPU-cached); the workspace and capture take the `USER_RW → USER_CDRAM_RW → USER_MAIN_PHYCONT_NC_RW`
ladder, and CDRAM/phycont are **uncached** and slower for the CPU (SDK Memory-Management Tutorial),
so a fallback rung costs speed as well as being a different pool.

### 1.2 Console measurements we have, and what is still unmeasured

Measured on a PCH-1001, firmware 3.65, inside CLANNAD (`notes/26`–`27`, overlay #74):

- **Free game memory:** 91,226,112 B (87 MiB) free `USER` in CLANNAD before a job (one sample; a
  second session ~87 MiB). Stardew Valley in +109 MiB mode: ≈ 56 MiB (third-party screenshot). No
  other VN sampled. CLANNAD's `param.sfo` is the plain 256 MiB mode (`ATTRIBUTE2=0`).
- **FL10 timing, API 13 (overlay v0.7.3–v0.7.5), 917×125 region:** model load 213–359 ms (once per
  session since overlay #54), recognition 36–145 ms per job, over 14 text-bearing jobs; line finding
  34–38 ms in the first session's 7 jobs. Unknown clock, varying dialogue, not a controlled
  benchmark. The API 18 build (overlay #74) failed at capture before inference, so H15's and the
  larger tiers' console times are unmeasured.
- **Still unmeasured** (the self-test's job answers them; overlay #74/#75, #76's meminfo fix merged):
  the game's free USER/CDRAM/phycont per title and per block; whether the display hook can write each
  memory type on 3.65; the capture time per strip; H15's (never run on a console) and the larger
  tiers' load and recognition times; the game's share of the shared L2; CPU3 cost; DRAM bandwidth.
  **Every ms and every per-job memory figure below is a `vita/a9_cycles.py` estimate or arithmetic on
  the overlay's layout, not a console measurement.**

The console self-test (overlay `docs/self-test.md`) already reports, per job, the fields a tier
decision needs: `free_user`, `need_workspace`, `need_weights`, `reserve_bytes`, `resident_bytes`,
each line's `frames`/`us` (so ms per frame), and the capture source/time. The field a per-title tier
table still needs is the `memory ... main=` figure per game, which one session per title fills.

### 1.3 The tier ceilings

Sizes are exact (`train/export.py`); times are `vita/a9_cycles.py` estimates on 40 synth-val lines
(no protected line read), high bound with the whole 2 MiB L2. The box time is 3 × a 640-px line (80
CTC frames at height 32, stride 8); a taller input or coarser stride is scaled to that.

| tier | int8 weights | W8I arena | box time target (444 MHz, est.) | free game memory a first full job needs | always loads? |
|---|---|---|---|---|---|
| **Small** | < 1,000,000 B | ≤ 131,072 B | < 1 s / 3-line box | **≈ 5.3 MiB** (FL10) / 5.6 MiB (ceiling) | yes |
| **Normal** | ≤ 4,000,000 B | ≤ 262,144 B | < 2 s / 3-line box | **≈ 10.3 MiB** (W20) / 12.6 MiB (ceiling) | preflight; falls back to Small |
| **Large** | ≤ 9,000,000 B | ≤ 1,310,720 B (16-frame tiles) | < 3 s / 3-line box | **≈ 18.3 MiB** (W20) / 21.0 MiB (next-round) | preflight; falls back to Normal, then Small |

For reference at 333 MHz (the shell's default clock), multiply the 444 MHz box times by about 1.33.
Free-memory needs are `job_need` (page-rounded weights + workspace + 61,440 B capture + the 516,096 B
line-block ceiling) plus the per-tier reserve, in MiB from bytes. Notes on the ceilings this document
adds on top of the owner's weight and time ceilings:

- **Arena.** Small keeps the always-tier 128 KiB; Normal the preflight-tier 256 KiB. Large needs
  16-frame tiles to stay inside 3 s/box (§1.4), which cost arena, so its arena ceiling is 1,310,720 B (1.25 MiB).
  The workspace (arena) is `USER_RW` first, then CDRAM, then phycont (§1.1), and counts in the preflight; a model's tier is judged at the
  default 4-frame tiling first (least arena/memory), and 16-frame tiles are tried only for Large and only
  when the default tiling is over 3 s/box — so most Large models (W20 included) are still judged at the
  default tiling, and only the shapes over ~2.7 MB use 16-frame tiles.
- **Reserve.** The overlay keeps a flat 4 MiB reserve on every job (untuned, one title). This document
  **proposes** a larger reserve for the larger tiers — 4 MiB (Small), 8 MiB (Normal), 16 MiB (Large)
  — because a bigger resident block, held while the game is foreground, is a bigger worst-case bite
  out of the game's own later allocations. The reserve is the overlay's to set; §1.5 states it as a
  selection parameter. Even the 21.0 MiB Large next-round need is 38 % of the 56 MiB seen free in the
  tightest game measured (24 % of CLANNAD's 87 MiB), so RAM is not expected to bind in VNs; the
  reserve is insurance, not a fit problem.
- **"Always loads."** Small's 5.3 MiB need is 9 % of the smallest free figure seen (56 MiB; 6 % of
  CLANNAD's 87 MiB), so no measured or inferred game refuses it; it needs no fallback. Normal and
  Large pass through the preflight and fall back when a game is tight.

### 1.4 Architectures priced against the ceilings

Exact `train/export.py` random-init blobs, `vita/a9_cycles.py` estimates. Full grid in
`docs/model-tiers.json`; the key rows (3-line box, high bound, whole L2; "t4" = default 4-frame tiles,
"t16" = 16-frame tiles / 64-frame head batch):

| candidate | shape | int8 B | W8I arena B (t4 / t16) | MMAC/640px | box @333 (t4 / t16) | box @444 (t4 / t16) | tier |
|---|---|---|---|---|---|---|---|
| `s_w100v3500` | FL10: w1.0, 3,500 vocab | 657,616 | 124,832 | 66 | 373–706 | 280–530 | **Small** |
| `s_w100v4987` | w1.0, 4,987 vocab | 819,792 | 124,832 | 77 | 432–814 | 324–611 | **Small** |
| `s_w100v5631` | w1.0, 5,631 vocab | 889,568 | 124,832 | 82 | 458–861 | 343–646 | **Small** |
| `n_w200` | W20: w2.0, 5,355 vocab | 1,558,016 | 241,056 | 187 | 934–1,645 | 701–1,234 | **Normal** |
| `n_w200v5631` | w2.0, 5,631 vocab | 1,587,680 | 241,056 | 189 | 945–1,665 | 709–1,249 | **Normal** |
| `n_lw300v5631` | late-wide 2.0→3.0, 5,631 vocab | 2,473,696 | 261,024 | 272 | 1,290–2,544 | 968–1,950 | **Normal** |
| `n_w250v5631` | w2.5, 5,631 vocab | 2,095,712 | 299,168 (> 262 K) | 268 | 1,285–2,392 | 964–1,820 | Large (arena over Normal's) |
| `n_w275v5631` | w2.75, 5,631 vocab | 2,389,472 | 328,224 / 1,053,696 | 314 | 1,490–2,917 / 1,480–2,861 | 1,117–2,240 / 1,110–2,190 | Large |
| `n_w300v5631` | w3.0, 5,631 vocab | 2,709,728 | 357,280 / 1,146,496 | 364 | 1,692–3,719 / 1,681–3,278 | 1,269–**2,907** / 1,261–2,510 | Large |
| `l_w300v6459b128` | w3.0, b128, 6,459 vocab | 3,025,808 | 359,712 / 1,156,608 | 388 | 1,796–3,913 / 1,785–3,437 | 1,347–3,059 / 1,339–**2,632** | **Large (t16)** |
| `l_w300v6459b192` | w3.0, b192, 6,459 vocab (recommended) | 3,476,624 | 364,576 / 1,176,832 | 424 | 1,938–4,179 / 1,927–3,642 | 1,453–3,271 / 1,445–**2,789** | **Large (t16)** |
| `l_w300v7501b128` | w3.0, b128, 7,501 vocab | 3,172,464 | 359,712 / 1,156,608 | 399 | 1,848–4,033 / 1,837–3,526 | 1,386–3,153 / 1,378–**2,699** | **Large (t16)** |
| `l_w325v5631` | **plain w3.25**, 5,631 vocab | 3,056,480 | 386,336 / 1,239,296 | 418 | 1,930–4,356 / 1,918–3,812 | 1,447–3,415 / 1,439–**2,919** | **Large (t16)** |
| `l_w300c4v5631` | w3.0, 4 ctx, 5,631 vocab | 3,410,400 | 394,144 / 1,238,656 | 418 | 1,935–4,493 / 1,920–3,745 | 1,451–3,533 / 1,440–**2,876** | **Large (t16)** |
| `l_w300v7501b192` | w3.0, b192, 7,501 vocab | 3,689,840 | 364,576 / 1,176,832 | 440 | 2,010–4,325 / 2,000–3,760 | 1,508–3,385 / 1,500–**2,878** | **Large (t16)** |
| `l_w300v11001b192` | w3.0, b192, 11,001 vocab | 4,408,144 | 364,576 / 1,176,832 | 494 | 2,255–4,778 / 2,244–4,155 | 1,691–3,734 / 1,683–**3,178** | over 3 s even with t16 |

The L2 turns the picture over past ~2.3 MB. With **16-frame tiles and a 64-frame head batch** the
spill shrinks (fewer passes over the weights, `docs/vita-memory-budget.md` §3) and the box time of the
width-3.0/3.25 shapes drops under 3 s @444. **Several shapes over 3 MB fit Large with 16-frame tiles**, at
about the same estimated box time. The recommended one is wide-head and compute-light:

- `l_w300v6459b192` **(recommended)** — width 3.0 + bottleneck 192 + a 6,459-class VN-ordered vocab
  (vocab-expand's final: the full VN list, 0.0 OOV per 1,000 chars, without the Chinese-only classes a
  7,501 head adds), 3.48 MB, 5.30 M MACs/frame, est. 1.45–**2.79** s/box @444 (t16), arena 1,176,832 B
  (1.12 MiB). Its 6,459-class b128 sibling is 3.03 MB and 2.63 s/box.
- `l_w300v7501b128` — width 3.0 + bottleneck 128 + 7,501-class vocab, 3.17 MB, 4.99 M MACs/frame, est.
  1.38–**2.70** s/box @444 (t16), arena 1,156,608 B (1.10 MiB). Faster, but the 7,501 head carries
  Chinese-only classes VN text never uses.
- `l_w325v5631` — **plain width 3.25**, 3.06 MB, 5.23 M MACs/frame, est. 1.44–**2.92** s/box @444 (t16),
  arena 1,239,296 B (1.18 MiB).
- `l_w300c4v5631` — width 3.0 + 4 temporal blocks, 3.41 MB, 5.22 M MACs/frame, est. 1.44–**2.88** s/box
  @444 (t16), arena 1,238,656 B (1.18 MiB).
- `l_w300v7501b192` — width 3.0 + bottleneck 192 + a 7,501-class vocab (nearly half the blob), 3.69 MB,
  5.51 M MACs/frame, est. 1.50–**2.88** s/box @444 (t16), arena 1,176,832 B (1.12 MiB).

The wide-head shapes (`b128`/`b192` + 7,501 vocab) are the compute-light ones the owner asked to prefer;
which wins on accuracy per ms is unmeasured. `l_w300v11001b192` (JIS X 0213 vocab, 4.41 MB) is 3.18
s/box even with t16, over the budget. Widths 3.5 and 4.0, run with 16-frame tiles too, are still over 3 s/box (3.30 and 4.25 s) and widths 3.5 and 4.0 also have t16
arenas over the Large arena ceiling (1,332,096 and 1,517,696 B); `l_w300v7501b128`, `l_w300c4v5631` and `l_lw400v7501b192`
were also measured with t16 (2.70, 2.88 and 3.68 s/box). A head/vocab wider than H15's needs a vocab
change (a new head), so the big-head Large shapes are next-round candidates, not VOCR4 changes
(vocab-expand prices the exact vocab). Earlier arch work found depth, ctx and extra bottleneck add
nothing over plain width per MAC up to F20, so the cheap place for extra parameters is width or the
head.

### 1.5 The runtime selection rule (a spec for an overlay follow-up)

The overlay already has the mechanism (preflight, resident weights, a configured fallback). The tier
scheme generalises its two-model fallback to three:

```
Given the pinned tiers, largest first: [Large, Normal, Small], each with its blob bytes,
its W8I arena at its shipping tiling, and its reserve R_tier (16/8/4 MiB).
On a job:
  free = sceKernelGetFreeMemorySize().size_user       # < 0 (under the margin) means 0 free
  for tier in [Large, Normal, Small]:
    need    = page(weights unless this tier's weights are already resident)
            + page(max(finder_arena, tier.arena) + 3760)      # workspace
            + 61440                                           # capture block
            + VJO_VOCR_LINE_MAX_BYTES                         # the line-block ceiling
    release = page(resident other tier's weights)             # freed first, so counts as free
    if free + release - R_tier >= need:  run tier;  break
  else: refuse, log free and the needs (as today)
```

So the plugin picks the **largest tier whose need ≤ free − reserve**, releasing a resident block of
another tier (counted as free), falling back down the list.

**What this needs from the overlay, which it does not do today (not changed in this task):**

- (a) `weights_alloc` refuses any blob larger than `VJO_VOCR_F20_BYTES` = 1,558,016 B
  (`game_ocr/main.c:415`): today **no Normal or Large larger than F20 can load** (W20 is exactly F20-sized and passes the size cap; only the pin table (b) stops it). The Normal and
  Large tiers above are a sizing study until this is raised.
- (b) Only the four pinned blobs load (FL10, H15, F20 and the opt-in VOCR2, each by exact size, SHA-256 and
  arena, `core/vocr_ocr.c`): new tier models must be added to that table.
- (c) The build-time memory ledger would fail the build: `VJO_LIMIT_GAME_WEIGHTS` 1,601,536, `VJO_LIMIT_GAME_WORKSPACE` 790,528 (a t16h64 Large workspace is 1.16–1.25 MB) and `VJO_LIMIT_GAME_JOB`
  2,392,064 (`include/vjo_ledger_budget.h`), enforced by `tools/check_memory_ledger.py`.
- (d) A job always uses the runtime's default 4-frame tiles and 16-frame head (`core/vocr_ocr.c`
  zeroes `vocr_opts`): the t16h64 Large shapes need a per-model tiling setting and a t16h64 arena pin.
- (e) There is one configured fallback (`r->fallback`), not a list: a three-tier fallback needs a
  pinned chain.
- (f) The line-block term above is a conservative ceiling; today's `check_memory` does not count it.

Keep what already works: resident weights are dropped when the game is not foreground, the title is
disabled, the file changes, or a job is refused; the preflight is one syscall; a failed allocation
after a passed check is normal; the reserve is per tier so a larger resident block leaves more
headroom. The console self-test's per-title `free_user` figures should set each game's achievable
tier before any of this ships.

## Part 2 — the gaps

Dev data only: teacher_val (1,643 lines, 180 VNs) and devnative v3 (986 crops, 43 VNs), read through
the plain-C W8I runtime in the console-polarity view (rv; dark text inverted). The three VOCR3
selections' aggregate tables are committed in `docs/model-tiers.json`'s `gaps` section (counts only); the full per-line taxonomy is
`eval/model_gaps.py`. VOCR4's A/C/M arms are mid-training (the vocr4-train agent); the real
decisions so far adopt arm C (A not better) for FL10 and H15. Coordinated with vocab-expand and
vocr4-train via panes; this part does not re-derive their numbers.

**Owner (2026-10-14 13:57Z): speaker-name boxes do not matter — users crop them out, so dialogue CER is the
primary metric and names are a secondary column only. The name-plate strata below are reported but are not a
target; the top-10 plan is re-weighted accordingly (name plates dropped as a priority).**

Pooled development CER (rv, 28,544 reference characters): **FL10 2.29 %, H15 2.12 %, W20 1.81 %** (the
128k-pretrain FL10 selection; the matched-recipe 32k FL10 is 3.20 %, §2.3). 380 / 346 / 315 of the
2,629 lines carry an error.

### 2.1 Error taxonomy

**By character class** (errors per 1,000 reference characters of that class; share of all errors):

| class | ref chars | FL10 /1,000 (share) | H15 | W20 | cause |
|---|---|---|---|---|---|
| kanji (all) | 7,675 | 44.7 (52 %) | 40.5 (51 %) | 35.9 (53 %) | capacity + data (below) |
| — rank ≤ 500 | 5,423 | 27.7 | 26.2 | 21.6 | mostly data/finder |
| — rank 501–2,000 | 2,043 | 73.4 | 65.6 | 62.2 | capacity + exposure |
| — rank > 2,000 (rare) | 209 | 114.8 | 69.4 | 78.7 | vocab + exposure |
| — real exposure < 100 | 1,551 | 73–235 | 67–118 | 61–235 | **data: thin real exposure** |
| katakana with dakuten/handakuten (`kata_voiced`) | 356 | 92.7 | 112.4 | 67.4 | **data: dakuten pairs (プ/ブ)** |
| katakana (plain) | 1,762 | 25.0 | 26.1 | 21.6 | data |
| hiragana (plain) | 12,491 | 7.1 | 7.2 | 6.2 | low; floor |
| long-vowel ー | 210 | 66.7 | 19.0 | 33.3 | ー/一/— confusions |
| punctuation | 3,253 | 15.7 | 17.2 | 14.4 | 。/、, 』/」 |
| Latin letters | 99 | 212.1 | 80.8 | 90.9 | **data: full-width Latin thin** |
| digits | 75 | 133.3 | 93.3 | 80.0 | data |
| OOV (vocab can't emit) | 7 (FL10) | 1000 | — (H15/W20 vocab covers them) | | **vocab (FL10 only)** |

Kanji is half of every size's errors but mostly because kanji is half the hard characters, not
because the model lacks capacity: within kanji the error rate tracks **Jiten rank and real-pool
exposure**, not model size — a kanji the real pools hold < 10 times is read at 60–235 /1,000 by every
size, a rank-≤-500 kanji at 22–28. `kata_voiced` (katakana with dakuten/handakuten) is the worst kana
class at every size.

**By style / shape** (CER; selected rows, full table in `docs/model-tiers.json`'s `gaps`):

| stratum | chars | FL10 | H15 | W20 | note |
|---|---|---|---|---|---|
| name-like | 3,228 | 9.6 % | 8.3 % | 7.5 % | **short, no context; 44–48 % of all errors** |
| length 1–4 | 2,460 | 10.3 % | 8.5 % | 7.9 % | same lines, from the length side |
| style:name (devnative) | 809 | 17.6 % | 14.5 % | 13.6 % | native name plates, worst stratum with ≥ 100 chars |
| style:furigana | 26 | FL10 38.5 %, H15 38.5 %, W20 46.2 % | | | 6 lines, noise |
| ink < 14 px | 5,190 | 5.5 % | 5.2 % | 4.5 % | small PSP text |
| frames 1–15 (short crops) | 2,392 | 7.6 % | 6.4 % | 5.9 % | short lines again |
| target PSP | 15,894 | 2.7 % | 2.7 % | 2.2 % | PSP 1.5–1.9× Vita |
| target Vita | 12,650 | 1.8 % | 1.4 % | 1.3 % | |

**Confusion families** (substitution kinds, W20): kanji↔kanji 208 (visually close pairs: 蓮/達,
題/選, 中/申), then dakuten pairs 32 (パ/バ, ベ/ペ, て/で, か/が), small-kana 17 (ゃ/や, ょ/よ),
punctuation 。/、 and 』/」, and ー/一. 404 of FL10's 654 errors fall on the first or last two
characters of the line, but 253 of those 404 are on 1–4-character lines where every position is an
edge; on lines of 9+ characters the edge rate (27.9 % of errors on 23.3 % of positions) is near the
base rate, so edge position is a weak signal except on short lines — the finder evidence (§2.2) is the
stronger pointer.

### 2.2 Line finder vs recognizer (the whole-dialogue guard)

devnative cuts each line two ways: by hand around its ink (`ink`), and by the console's own finder
(`GRAD`), the latter only where the finder returns one clean box. On the 443 lines with both crops,
read by the **same** W20 model:

| crop | lines | errors | CER |
|---|---|---|---|
| hand-cut ink | 443 | 27 | 0.51 % |
| console GRAD | 443 | 163 | 3.10 % |

**The console finder's crop makes ~6× the recognizer's own errors on the identical lines.** Reading
the GRAD errors by eye (`eval/model_gaps.py sheets`, sheets S011–S015 reviewed): the GRAD box is
taller (median 26 px vs 18) and wider (243 vs 190, over all 443) than the ink box, and it pulls in a
character portrait at the edge (`v1877_57778` "桐生"→"抱生", box ~3× wider), a decorative flower or
cursor glyph appended, or the dialogue-box border. The extra errors over the 443 pairs are
substitutions and end-of-line insertions (GRAD − ink: `s_mid` +46, `ins_end` +37, `s_start`+`s_end`
+21). This is the same thing the VOCR3 whole-dialogue guard caught (every candidate passes the
fixed-line gate and fails the finder-cut guard). **A bigger recognizer does not fix it** — the ink CER
is already ~0.5 %. It needs the finder (a textbox-aware crop, VOCR2's direction) and a whole-dialogue
development cohort, not more model.

### 2.3 Capacity curve: gain per extra MB (matched recipe)

Earlier drafts compared an FL10 trained with a 4× longer pretrain against H15/W20, which flattered that FL10 point and so understated
the first step; the curve below uses **matched-recipe runs** (the realplus fine-tunes of each size,
same arm, seeds averaged, pooled dev CER, rv). Both the C_mix and A_s2 synthetic arms are shown. The
128k-pretrain FL10 selection (2.29 %, §2.1) is a separate, better point that this per-MB comparison
excludes; H15's 40k pretrain vs W20's 32k is a smaller residual confound.

| step | ΔMB | C_mix points | C_mix points/MB | A_s2 points | A_s2 points/MB |
|---|---|---|---|---|---|
| FL10 (3.20 % / 3.34 %) → H15 (2.14 % / 2.20 %) | +0.498 | 1.06 | **2.13** | 1.14 | **2.29** |
| H15 → W20 (1.82 % / 1.91 %) | +0.402 | 0.32 | **0.80** | 0.29 | **0.72** |
| FL10 → W20 (total) | +0.900 | 1.38 | 1.53 | 1.43 | 1.59 |

**The first step pays about 2.7–3.2× more per MB than the second**, and the full FL10→W20 gain is 1.38
points. So width helps and the step to Normal is clearly worth it, while the extra MB past W20 into
Large is worth less per MB — and beyond about 3.7 MB with 16-frame tiles (about 2.7 MB at the default tiling) the console cannot run it in 3 s/box anyway (§1.4), so
Large's capacity gain is bounded by speed, not by the weight ceiling. 1.38 points is about the size of
the data/finder headroom in §2.4, so capacity and data both matter.

### 2.4 Prioritised plan: the top 10 gaps × fix × expected gain × cost

Gain is the development-CER headroom the gap holds (errors in the stratum ÷ 28,544 total chars), with
the lever. "Cost" is training/data effort; a new head or input height needs a fresh pretrain (~12 h).

| # | gap (dev evidence) | fix | lever | est. gain (dev CER) | cost |
|---|---|---|---|---|---|
| 1 | finder crop adds ~6× errors on GRAD vs ink (2.2) | textbox-aware finder + whole-dialogue cohort | **finder** | finder-cut CER 3.1 % vs 0.5 % ink; the single largest lever on the real reading path | overlay + a dev cohort; no retrain |
| 2 | short 1–4-char **dialogue** lines 8–10 % CER (2.1) — name plates excluded per owner 13:57Z | more real + synthetic short dialogue lines; names are secondary (users crop them) | **data** | the dialogue short-line share, not the ~45 % that is mostly names | synth4 round; vocr4 data |
| 3 | kanji at real-exposure < 100: 60–235 /1,000 (2.1) | synth4's rare-char curriculum (≥ K renders × F fonts in real-word contexts) | **data** | 1,551 chars; ≈ 0.3–0.4 points | synth4 (built; measure in the ablation) |
| 4 | PSP 1.5–1.9× Vita CER (2.1) | more native PSP / 1-bit-font synthetic; the owner's PSP frames | **data** | PSP is 15,894 chars at 2.2–2.7 %; −0.5 point there ≈ 0.3 pooled | vocr4 data (owner PSP frames) |
| 5 | katakana with dakuten/handakuten worst kana (67–112 /1,000) | dakuten/handakuten-focused synthetic; the contrast fix | **data** | 356 chars; ≈ 0.1 point, a visible class | synth4 |
| 6 | FL10 cannot emit 7 dev chars (OOV); 2.9/1,000 on VN text | VN-ordered 4,987-class vocab for Small | **vocab** | Small only; 22× fewer missing chars (`docs/synth4-gap.md`) | new head, next round |
| 7 | capacity: W20 − H15 = −0.32 points, 0.80 pts/MB (2.3) | ship Normal = width 2.0 (W20) | **capacity** | ≈ 0.3 points for +0.4 MB | VOCR4 (W20 trained) |
| 8 | Large headroom unused by the console | several Large shapes fit (1.4); recommended `l_w300v6459b192` (w3.0 + b192 + a 6,459-class VN vocab) | **capacity + vocab** | FL10→W20 is 1.38 points; a further ~1 MB ≈ 0.2–0.4 points if not compute-bound | new pretrain, next round |
| 9 | punctuation 。/、, 』/」, ー/一 (2.1) | punctuation-balanced synthetic; the normaliser folds dashes | **data** | 3,253 chars at 14–17 /1,000; ≈ 0.1 point | synth4 |
| 10 | small / low-contrast PSP ink (< 14 px, 4.5–5.5 %) | lower-contrast synthetic (synth4 gap §3); input height 40/48 for Large | **data + capacity** | 5,190 chars; height 40/48 priced (`n_w200h40/48`), but 1.5–1.95× slower | synth4; new pretrain for height |

The headline: **items 1–5 and 9–10 are data/finder, not capacity**, and together hold development-CER
headroom (roughly 1.2–1.8 points if half-closed) comparable to the full FL10→W20 capacity gain (1.38
points). Ship the three tiers on the current recipe now; the next round's effort goes to the finder
and synth4's data, with the vocab/height/head changes (items 6, 8, 10) as pre-registered follow-ups
that each need a fresh pretrain.

## Hand-off to vocr4-train

The three ceilings, and the architecture to train for each tier if a ceiling is above vocr4-train's
current size (VOCR4 maps Small = FL10, Normal = H15, Large = W20; this document instead recommends
W20 as the trained-now Normal and the trained-now Large, since W20 fits both and H15 sits between — so this
round trains two distinct models, not three; with W20 pinned to both tiers the selection rule's Large-vs-Normal
choice would differ only in the reserve (16 vs 8 MiB) until the next-round Large shape is trained):

- **Small < 1 MB / 131,072 B arena / box ≤ 1 s @444** (free need ≈ 5.3 MiB). Train now: FL10 (657,664 B,
  est. 0.28–0.53 s/box @444, 0.37–0.71 @333). Headroom for the next round: the FL10 shape with a
  VN-ordered 4,987-class head (`s_w100v4987`, 819,792 B, est. 0.32–0.61 s/box @444), a new head → a
  pre-registered follow-up. Small's speed headroom means the vocab can grow freely within 1 MB.
- **Normal ≤ 4 MB / 262,144 B arena / box ≤ 2 s @444** (free need ≈ 10.3 MiB). Train now: W20
  (1,558,016 B, est. 0.70–1.23 s/box @444). The biggest-capacity shape that still fits Normal is a
  late-wide model (`n_lw300v5631`, width 2.0 early → 3.0 late, 2.47 MB, arena 261,024 B, est.
  0.97–1.95 s/box @444) — a pre-registered follow-up (new pretrain).
- **Large ≤ 9 MB weights but box ≤ 3 s @444, which caps it near 3.1–3.7 MB** (free need ≈ 18.3 MiB for
  W20, 21.0 MiB for the next-round shape; arena ≤ 1,310,720 B (1.25 MiB) with 16-frame tiles). W20 is the trained-now
  Large and fits with wide margin. To use the envelope, the recommended next-round Large is a
  **parameter-heavy, compute-light** wide-head shape read with tile 16 / head 64: **`l_w300v6459b192`**
  (width 3.0, bottleneck 192, a 6,459-class VN-ordered vocab — vocab-expand's final PR #79, which covers
  the whole VN list at 0.0 OOV per 1,000 chars without the Chinese-only classes a 7,501 head adds),
  3.48 MB, est. 1.45–2.79 s/box @444 (free need ≈ 21.0 MiB). Other shapes also fit (plain width 3.25
  `l_w325v5631`, 3.06 MB; width-3.0 + 4-ctx `l_w300c4v5631`, 3.41 MB; the 7,501-head pair), all 2.63–2.92
  s/box. It needs a fresh pretrain and a new head rendered with the 6,459 vocab (the real pools are
  H15-vocab filtered, so synthetic data carries the new classes), so it is a pre-registered next round,
  not a change to VOCR4 after scores. `l_w300v11001b192` (JIS X 0213 vocab) is over 3 s even with big
  tiles.

## Reproduce

```sh
A=$HERD/shared/artifacts; export VITASDK=$A/vita/vitasdk VOCR_SYNTH_VAL=$HERD/shared/data/synth/synth_00039.npz
export ARM_CC=$A/runtime/tools/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-linux-gnueabihf/bin/arm-none-linux-gnueabihf-gcc
.venv-ci/bin/python vita/model_tiers.py blobs --out $A/model-tiers/sweep            # random-init blobs, sizes.json
HEAVY_SLOTS=12 $HERD/heavy.sh .venv-ci/bin/python vita/budget_sweep.py a9 --dir $A/model-tiers/sweep \
    --qemu $A/vita/qemu --l2 2048,1024 --jobs 5                                      # cycle estimates
HEAVY_SLOTS=12 $HERD/heavy.sh .venv-ci/bin/python vita/budget_sweep.py a9 --dir $A/model-tiers/sweep \
    --qemu $A/vita/qemu --l2 2048,1024 --tile 16 --head 64 \
    --only n_w200,n_w200v5631,n_lw300v5631,n_w250v5631,n_w250v5631b128,n_w250d2v5631,n_w250h40v5631,n_w200h40v5631,n_w200h48v5631,n_w275v5631,n_w300v5631   # the Normal-sized candidates
HEAVY_SLOTS=12 $HERD/heavy.sh .venv-ci/bin/python vita/budget_sweep.py a9 --dir $A/model-tiers/sweep \
    --qemu $A/vita/qemu --l2 2048,1024 --tile 16 --head 64 \
    --only l_w300v6459b128,l_w300v6459b192,l_w325v5631,l_w300v7501b128,l_w300v7501b192,l_w300c4v5631,l_w300v11001b192,l_w350v5631,l_w400v5631,l_lw400v7501b192,l_w250h48s16v7501b192,n_w250h48s16v5631   # the Large-sized candidates (every tier verdict is then measured)
.venv-ci/bin/python eval/model_gaps.py taxonomy --vocr3 $A/synth3/vocr3/select --vocr4 $A/vocr4-train/select \
    --pools $A/synth3/study2/pools/teacher_train.npz $A/synth3/study2/pools/real_train.npz \
    $A/synth3/vocr3/pools/realplus_train_ink.npz --fl10-vocab $A/train2/release/FL10_vocab_3500.txt \
    --out $A/model-tiers/gaps/taxonomy.json
.venv-ci/bin/python vita/model_tiers.py report --dir $A/model-tiers/sweep \
    --taxonomy $A/model-tiers/gaps/taxonomy.json --out docs/model-tiers.json       # table + capacity + recommendations
.venv-ci/bin/python eval/model_gaps.py sheets --vocr3 $A/synth3/vocr3/select --out $A/model-tiers/review/errors
```

## Limits

- Every console speed and every per-job memory figure is a `vita/a9_cycles.py` estimate or arithmetic
  on the overlay's layout; nothing has run on a console at these sizes. The L2 share a game leaves is
  unknown (the table assumes the whole 2 MiB and, in the 1 MiB column, half); the 16-frame-tile Large
  result depends on that share.
- The gap analysis is on development cohorts (teacher_val, devnative v3), one machine, the C W8I
  runtime, the rv (console-polarity) view only; labels are engine-plus-assistant agreed, not
  hand-checked, so a share of the "errors" are label errors (e.g. `早寝早起き` cut to `早寝早起` by the
  GRAD box). No protected set was read.
- The capacity curve is three points (FL10/H15/W20) from matched-recipe realplus fine-tunes, pooled rv;
  it is a development estimate, not the locked-report test CER, and H15's longer pretrain is a residual
  confound.
- The free-memory needs use the overlay's exact layout at main `64b5726`; the per-tier reserves
  (8/16 MiB for Normal/Large) are this document's proposal, not the overlay's current flat 4 MiB.
- The overlay today loads at most F20-sized weights and only the four pinned blobs (§1.5), so
  the Normal and Large tiers are a sizing study until the overlay follow-up lands.
- Tier costs for the next-round heads (4,987/5,631/7,501/11,001 classes) are exact exports here; their
  accuracy is vocab-expand's and vocr4-train's to measure.
