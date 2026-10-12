# VOCR3 locked-report harness

`eval/vocr3_report.py` produces the locked report the VOCR3 plan asks for (plan section 5,
phase D): any set of W8I blobs on vita-test and psp-test through the C runtime, per style, with
paired bootstrap intervals, the VOCR2 gates, size, arena and an estimated Cortex-A9 time, plus a
cross-check through vita-jp-overlay's own host pipeline. Two adapters run the private tests:
`eval/vocr3_fresh.py` (the fresh native textbox test) and `eval/vocr3_clannad.py` (the CLANNAD long
run).

The harness was verified only by reproducing the published numbers of the three released blobs:
FL10_w8 (VOCR1, v0.2.0), H15_w8 (VOCR_H15, v0.2.0, the overlay default) and VOCR2_FL10_w8 (VOCR2,
v0.2.3-vocr2.1). It has scored no other model on a protected set. A new model goes through it only
after its protocol is committed (`eval/vocr3_protocol.json`, phase D), and only the conductor runs
it on the protected sets (AGENTS.md, plan section 6).

## What it measures

An *arm* is a blob plus the finder it ships with (`grad` for VOCR1 and VOCR_H15, `textbox` for
VOCR2, `textbox_recovery` for C4). Arms are listed in a JSON spec; `eval/vocr3_harness_released.json`
is the one for the released blobs. Every arm's sha256 can be pinned there.

| part | how | output |
|---|---|---|
| fixed lines | `eval/run.py`'s `recognizer_mode`, unchanged: every hand-checked line, box + 4 px (Vita) / 2 px (PSP), W8I, max width 1,024; console polarity (primary: dark-text crops inverted, as on the console) and native polarity | integer errors / characters, exact lines, hiragana errors / characters, per set, per style tag, per VN and pooled |
| whole dialogue | each manifest region at its base box and expanded by 4 and 8 native pixels; the arm's own C finder; every returned crop, in finder order, with the finder's polarity; scored by the textbox guard's `transcript_score` (all text and body, full and hiragana; unmatched insertions charged to body). Regions holding an uncertain line are left out whole (2 Vita regions, 4 certain lines, 58 characters, as in VOCR2's protected audit) | per title (platform:VN) and variant, platform and pooled totals, detections |
| pairs `CAND:BASE` | CAND − BASE in points; paired bootstrap 95% interval over lines (regions for whole dialogue) and over VN clusters (strata with at least 5 VNs); 10,000 resamples, seed 0, the train2 review's `boot.py` method | per set, per style, pooled; full and hiragana |
| gates `CAND:BASE` | the VOCR2 fixed-line gate (no set's full-text or hiragana errors increase; pooled hiragana errors decrease; primary polarity) and the held-out whole-dialogue guard (no title/variant all/body full/hiragana error increase; every variant strictly improves the console-balanced hiragana rate in both scopes; at least two titles gain hiragana in both scopes) | pass/fail, every failing cell with its counts |
| resources | blob bytes; W8I and W8F arenas from `runtime/pyvocr.py` (`vocr_arena_bytes`), checked against `train/export.py`'s `plan_arena`; MACs per frame | exact bytes |
| A9 estimate | `vita/a9_cycles.py measure --blob` on each blob: the Vita compiler's machine code for `runtime/vocr.c` under plugin qemu-arm, 40 lines (20 synth-val, 20 vita-test), a cycle model as a range | cycles per CTC frame, ms per 640-px line at 444 and 333 MHz; **estimates, not console measurements** |
| overlay cross-check | vita-jp-overlay's host `vjo-ocr` (its own finder call, workspace layout and model table) built from a scratch export of a read-only clone, with the runtime at the overlay's pinned vita-vn-ocr commit; its `tools/benchmark_local_ocr.py` on both sets | region concat errors, lines read, workspace peaks; host RSS/ms are informational |

The C recognizer and the finder driver are compiled fresh into the run directory with the flags of
the VOCR2 comparisons (`-std=c99 -O2 -ffp-contract=off -DVOCR_NO_NEON`, warnings as errors) and
hashed; pyvocr's cached `runtime/build/libvocr.so` is never used.

Public JSON holds counts and hashes only: `assert_public` refuses any `text`, `pred` or `predictions`
field. Per-line predictions go to `OUT/private/` and stay out of Git. A report is never overwritten:
every part is created exclusively, so a rerun needs a new `--out`.

## Commands

On the shared box, through `heavy.sh`:

```sh
HERD=/local/home/skerraut/herd-vita-ocr; A=$HERD/shared/artifacts
export VOCR_SYNTH_VAL=$HERD/shared/data/synth/synth_00039.npz   # eval/run.py's synth-val shard
SPEC=eval/vocr3_harness_released.json; BLOBS=/path/to/blobs; OUT=/path/to/new-run
eval/vocr3_report.py shared --spec $SPEC --blobs $BLOBS --data $A/eval --out $OUT \
  --stored bar:native:vita-test=$A/eval/results/ncnn-jis-int8.json \
  --stored bar:native:psp-test=$A/eval/results/rev01-psp/jis-int8.json --pair VOCR_H15:bar --pair VOCR1:bar
eval/vocr3_report.py a9 --spec $SPEC --blobs $BLOBS --out $OUT --data $A/eval --vitasdk $VITASDK \
  --qemu $A/vita/qemu --arm-cc $A/runtime/tools/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-linux-gnueabihf/bin/arm-none-linux-gnueabihf-gcc
eval/vocr3_report.py overlay --spec $SPEC --blobs $BLOBS --out $OUT --data $A/eval --overlay /path/to/vita-jp-overlay \
  --cmake /path/to/cmake --ninja /path/to/ninja        # add --extend-table for blobs the overlay does not pin
eval/vocr3_report.py render --out $OUT [--fresh FRESH/result.json] [--clannad CLANNAD/result.json]
eval/vocr3_verify_released.py $OUT --out $OUT/verification.json   # released blobs only
```

`render` writes `report.json`, `report_summary.json` (without per-line records) and `report.md`.
`--stored NAME:POLARITY:SET=RESULT.json` adds an earlier `eval/run.py --out` result, re-scored on
today's manifest, as a reference arm for pairs (never for gates). `--local-set NAME=MANIFEST` adds a
private hand-checked capture set (`eval/sets.py register_local`).

Tools used here: VitaSDK from `vita/scripts/fetch_vitasdk.sh` (pin `sdk-snapshot-20260926.790.1`,
tarball sha256 `87b111ff…`), the shared plugin qemu-arm 10.0.13 (`vita/scripts/build_qemu_plugins.sh`),
the Arm GNU Toolchain 14.2.rel1 (`runtime/scripts/fetch_arm_tools.sh`), and for the overlay build
CMake 3.31.10 and Ninja 1.13.2 from PyPI in a separate user-space venv.

### A spec

```json
{"schema": 1, "kind": "vocr3-report-spec",
 "arms": [{"name": "VOCR1", "file": "FL10_w8.vocr", "finder": "grad", "sha256": "500e0b84…"},
          {"name": "VOCR3_FL10", "file": "VOCR3_FL10_w8.vocr", "finder": "grad", "sha256": "…"}],
 "pairs": [{"candidate": "VOCR3_FL10", "baseline": "VOCR1"}],
 "gates": [{"candidate": "VOCR3_FL10", "baseline": "VOCR1"}],
 "sets": ["vita-test", "psp-test"], "polarities": ["console", "native"], "primary_polarity": "console",
 "bootstrap": {"reps": 10000, "seed": 0, "min_clusters": 5}, "dialogue": true}
```

The overlay cross-check of a blob the overlay does not pin needs `--extend-table`: the scratch
export's `core/vocr_ocr.c` gets one model-table row (name, bytes, sha256, W8I arena, finder) per such
blob. The clone is never modified; the patched source's hash and the rows are in `overlay.json`.

## Private tests (adapters)

Both adapters follow the frozen harnesses' one-shot rule. `prepare` validates roles and bindings
before any pixel is read, hashes every input, blob, source file and the freshly built C binaries,
and writes `protocol.json` and `prepared.json` into a new directory, with zero finder calls and zero
model forwards. `compare` runs only if the authorization file equals
`vocr3_report.expected_authorization(prepared, KIND)`, which the coordinating reviewer writes after
checking the preparation (the CLIs never write it). It creates `attempt.json` exclusively; a second
attempt is refused, a failure leaves `failure.json`, and nothing is retried. Every arm must be pinned
by sha256 in the spec.

```sh
eval/vocr3_fresh.py prepare --manifest M.json --annotations A.json --spec SPEC --blobs DIR --directory NEW
eval/vocr3_fresh.py compare --prepared NEW/prepared.json --authorization AUTH.json
eval/vocr3_clannad.py prepare --session SESSION --spec SPEC --blobs DIR --directory NEW --pair VOCR3_FL10:cached
eval/vocr3_clannad.py compare --prepared NEW/prepared.json --authorization AUTH.json
```

- **Fresh native textbox** (`docs/fresh-native-textbox.md`): the cohort contract is
  `eval/fresh_native_textbox.py`'s, reused unchanged (roles, exposure-evidence binding, deterministic
  deduplication, the fixed minima of three titles per console, five frames and 500 body / 200 hiragana
  characters per title, native and ROI pixel hashes, base/+4/+8 variants, `transcript_score`). Each arm
  reads every variant of every frame with its own finder. The result has per-title counters per arm
  and variant, pairs over frames and title clusters, and the whole-dialogue guard for each gate.
- **CLANNAD long run** (`docs/clannad-longrun-results-20261012.md`): by default the session must carry
  the published pins (lock `14ba9297…`, capture journal `b6cd582c…`, comparison journal `f2754897…`) and
  counts (2,204 captures, 2,177 pages, 27 excluded, 4,271 lines, 36,915 / 40,114 reference characters,
  cached FL10 3,176 / 5,618 edits). Each arm reads the original processed line PNGs (`invert=False`)
  and each original RGB ROI through its own finder (all rows up to 16; truncation is fatal), exactly
  as the released-VOCR2 comparison did. `cached` names the cached FL10 predictions. **Google is an
  automated proxy, so these are edit disagreements, not CER, and they cannot qualify a model.**

The private inputs are not on this box. Both adapters are tested only with generated fixtures:
`tests/test_fresh_native_textbox.py`'s generated cohort with drawn frames, and a generated session
written through `eval/clannad_longrun.py`'s own capture and journal functions
(`tests/test_vocr3_adapters.py`).

## Verification on the released blobs

Run at `b1037b9` (clean tree) on the 64-core shared box (Linux x86_64, glibc 2.34, GCC 11.5.0, CPython
3.11.15, NumPy 2.4.6, OpenCV 5.0.0, Pillow 12.3.0, torch 2.5.1+cpu), with the three release assets
downloaded by `gh release download` and checked against their sha256, and vita-jp-overlay at
`1a118ef`. The check-by-check comparison is in
[`eval/vocr3_harness_verification_20261012.json`](../eval/vocr3_harness_verification_20261012.json):
72 of its 79 checks reproduce exactly. The other 7 are A9 high bounds, explained below.

| published number (source) | reproduced |
|---|---|
| blob bytes, W8F / W8I arenas: FL10 657,664, 118,432 / 124,832; H15 1,155,968, 173,472 / 182,944; VOCR2 657,664, — / 124,832 (train2.md, vocr2-qualification.md) | identical |
| native fixed lines, errors / characters and exact lines: FL10 vita 59/2,564, 77.7%; psp 130/2,510, 69.8%; H15 40/2,564, 83.5%; 66/2,510, 82.4%; bar 59, 79.1%; 134, 66.1% (train2.md) | identical; every FL10 and H15 prediction equals the train2 C-runtime run's (451 lines each) |
| native per-style CERs of the bar, H15 and FL10: 54 cells (train2.md) | identical |
| paired bootstraps (train2.md): H15 − bar vita −0.74 [−1.44, −0.04], psp −2.71 [−4.05, −1.36]; FL10 − bar psp −0.16 [−1.66, +1.42], psp names +11.0 [+4.3, +18.3], psp furigana +13.6 [+1.0, +25.9]; H15 − bar psp names +3.35 [−2.36, +10.65] | identical |
| console fixed lines (vocr2-qualification.md): FL10 vita 61, hiragana 15/1,417; psp 133, 30/1,382; VOCR2 59, 13; 119, 25; fixed-line gate passes | identical; gate passes (pooled hiragana 45 → 38 without the private CLANNAD captures) |
| FL10 console per style and per VN (hiragana_baseline_20261011.json) | identical |
| whole dialogue, PSP, every variant and side: all/body × full/hiragana errors / characters and detections (vocr2_candidate_20261012.json), e.g. base all text 443 → 424 / 2,510 | identical |
| whole-dialogue guard: fails on 19 title cells, gain titles 8 / 6 / 5 per variant (vocr2_candidate_20261012.json) | the same 19 cells with the same counts, the same titles |
| overlay `benchmark_local_ocr.py`: FL10 vita 217/2,506, psp 443/2,510, lines read 207/233, workspace peak max 914,000 / 827,216; H15 170/2,506, 369/2,510, 1,470,416 / 1,383,632 (vocr-release-20261011.json) | identical: regions, refusals, failures, errors, characters, lines read and both workspace peaks, per set |
| A9 estimate, ms per 640-px line at 444 MHz (train2.md): FL10 W8I 94-178, W8F 157-194; H15 W8I 166-302, W8F 294-365 | low bounds identical in cycles per frame (FL10 521,887 / 871,347; H15 920,390 / 1,633,471) and ms; the same 1,828 frames and decoded texts; high bounds FL10 179 / 196 ms, H15 302 / 366 ms (+0.85%, +1.08%, −0.05%, +0.11% in cycles per frame), the layout-dependent term below |

The overlay's region errors equal the harness's own whole-dialogue base all-text errors for all three
arms on both sets (VOCR2: 205/2,506 and 424/2,510), so the two pipelines agree on every region's
totals. The overlay pins the released FL10 and H15 files and the VOCR2 trial file, so no model-table
row was added (`table_patch` is null).

### Released baselines as the harness reports them

Console polarity (primary), errors / characters; whole dialogue is the base crop, all text:

| arm | vita-test lines | psp-test lines | hiragana, pooled | whole dialogue, Vita | whole dialogue, PSP | W8I arena |
|---|---|---|---|---|---|---:|
| VOCR1 (FL10_w8) | 61 / 2,564 | 133 / 2,510 | 45 / 2,799 | 217 / 2,506 | 443 / 2,510 | 124,832 |
| VOCR_H15 (H15_w8) | 40 / 2,564 | 72 / 2,510 | 21 / 2,799 | 170 / 2,506 | 369 / 2,510 | 182,944 |
| VOCR2 (VOCR2_FL10_w8, C3 finder) | 59 / 2,564 | 119 / 2,510 | 38 / 2,799 | 205 / 2,506 | 424 / 2,510 | 124,832 |

VOCR_H15 − VOCR1, pooled lines, console polarity: −1.62 points (lines CI −2.09 to −1.16; VN clusters
−2.41 to −0.91). VOCR2 − VOCR1: −0.32 (lines −0.55 to −0.10; VN clusters −0.53 to −0.12).

VOCR2's published whole-dialogue Vita totals include the private CLANNAD captures; the harness's Vita
totals do not, so only the PSP side and the per-title cells can be compared there. The VOCR2
qualification's pooled hiragana counts (48 → 40) include those captures too.

### The A9 estimate's layout sensitivity

The high bound charges every simulated L1D miss at the L2 latency, and qemu's cache simulator counts
misses for the guest's actual addresses. Those move with the guest's memory layout: the environment
and argument strings on the stack, and the length of the executable's real path. In 29 FL10 W8I runs
(environments of 0 to 8 KiB, executable paths of 36 to 107 characters, absolute and relative file
names) the misses ranged from 5,622 to 5,955 per frame; the train2 review's run had 5,640. That moves
the W8I high bound between 177.6 and 179.4 ms at 444 MHz, about 1%. The low bound (NEON and VFP issue
cycles), the frame count and the decoded text were identical in every full measurement, the review's
and this harness's.

`vita/a9_cycles.py` now fixes the layout: the guest runs from a staging copy at a fixed-length
`/tmp/vocr-a9-XXXXXXXX` path, with relative file names, an empty environment and `qemu -seed 0`.
Two runs from different output directories and caller environments then gave the same simulated L1D
misses (10,819,836) and the same report to the printed precision; their high bounds differed by 0.2
cycles per frame (a few dozen guest instructions differ, most likely in the CLI's own wall-clock timing
output). With this fixed layout FL10's high
bounds land 0.85% (W8I) and 1.08% (W8F) above the review's, H15's −0.05% and +0.11%. Every arm of a
report is measured with the same layout, so comparisons between arms are like for like.

## Limits

- Every number is a host measurement through the C runtime; nothing here ran on a Vita or a PSP.
  The A9 times are estimates from a cycle model, not console measurements.
- 451 hand-checked lines and 30 title/console groups cannot resolve differences much under a point;
  read the intervals, and the VN-cluster intervals where a stratum has several VNs.
- The CLANNAD long run is a Google proxy, and the fresh cohort's references are assistant-reviewed;
  neither is independent human ground truth.
- The overlay cross-check uses the overlay's pinned runtime (`49ff6f4`), not this checkout's: a blob
  that needs newer runtime code (C4) cannot be cross-checked there.
- `heavy.sh` controls the threads; results do not depend on them (the C runtime is single-threaded).
