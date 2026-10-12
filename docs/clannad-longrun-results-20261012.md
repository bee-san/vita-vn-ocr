# CLANNAD sustained Google OCR comparison

The sustained test completed with **40,114 normalized whole-ROI Google reference
characters**, 2,177 successful unique pages and zero pending comparisons. All
2,204 accepted captures have exactly one retained comparison attempt. The final
read-only audit verified both journals and 8,719 stored image files and decoded
pixel hashes without issuing OCR requests or running model inference.

This diagnostic exercises the released FL10 baseline while playing CLANNAD in
Vita3K. It compares the dependency-free C recognizer with Google Lens through
`owocr==1.26.7`. Both engines receive the same stored line pixels. A separate
Google request reads the original dialogue-box crop, allowing a comparison that
also includes the general line finder's coverage and reading order.

Google is an automated reference proxy. These are edit disagreements, **not
hand-checked CER**, and cannot establish which engine is correct. This run does
not evaluate VOCR2, select a checkpoint, calibrate preprocessing or supply training
data. CLANNAD is one real VN used for diagnostics; the OCR implementation remains
general across games. Accuracy qualification uses the separate shared,
hand-checked Vita/PSP benchmark.

## Results

The target is at least 40,000 normalized whole-ROI Google characters from unique,
fully successful captures. The final dataset contains **2,177** such pages,
**4,271** paired lines and **40,114** independent whole-ROI reference characters.
All captures have exactly one comparison attempt; no failed request is retried.

| Comparison | Reference characters | Edit disagreements | Disagreement rate | Hiragana disagreements / reference characters |
| --- | ---: | ---: | ---: | ---: |
| VOCR vs Google on identical line pixels | 36,915 | 3,176 | 8.60% | 993 / 20,323 (4.89%) |
| Joined line VOCR vs whole-ROI Google | 40,114 | 5,618 | 14.01% | 2,181 / 21,729 (10.04%) |
| Joined line Google vs whole-ROI Google | 40,114 | 4,156 | 10.36% | 1,467 / 21,729 (6.75%) |

The first row measures recognition disagreement after the unchanged C GRAD finder
has selected a line. The second includes line-finder omissions, ordering, names
and incidental crop-edge content. The third shows how Google's own outputs differ
between finder-selected lines and the independent full crop. It is a coverage
diagnostic, not a measured finder error rate. The three rows are not additive.

The denominator is the same set of fully successful unique pages throughout.
A page qualifies only if every paired Google/VOCR line call succeeds and the
whole-ROI Google output succeeds with nonempty normalized text. Successful empty
Google line text retains its zero reference length and any VOCR insertion
disagreement. Normalization uses the frozen `eval.metrics.norm`; small kana,
dakuten and similar kanji remain distinct. Hiragana disagreements use full-text
alignment, including hiragana substitutions, deletions and insertions.

## Captures and exclusions

| Accounting | Count |
| --- | ---: |
| Accepted completed captures | 2,204 |
| Successful unique pages | 2,177 |
| Failed comparison pages, retained without retry | 27 |
| Pending comparisons at closure | 0 |
| Duplicate accepted foreground signatures | 0 |
| Verified final-profile pilot successful pages / reference characters excluded | 9 / 127 |
| Captured pilot pages excluded from the target | 72 |
| Finder geometry pauses excluded from the target | 3 |

The 27 excluded comparison pages comprise three failed Google line requests and
24 successful whole-ROI requests with empty normalized reference text. There are
zero VOCR call failures and zero whole-ROI Google request failures. All attempts
remain in the private comparison journal and contribute no target characters.
Menus, choices, incomplete typewriter text, blank transitions and
finder-rejected frames do not enter the OCR denominator. Duplicate-screen
observations are counted separately. The collector pauses on suspicious geometry
or after 120 seconds without a new completed page. Reviewed controller recoveries
leave the locked ROI, finder, thresholds and model unchanged. Choices use the
previously declared highlighted topmost option.

The explicit final-profile pilot supplies nine successful pages and 127
whole-ROI Google reference characters, all excluded from the target. Across seven
capture/control pilot sessions, all 72 captured native frames are excluded.
Recomputing their exact thresholded foreground signatures with the final test
profile finds zero mask overlap with accepted test captures. This check does not
prove semantic or text independence across other datasets. The controller starts
the test after the excluded pilot pages. Dialogue identity uses
the fixed foreground mask with the continuation marker excluded; it never uses
Google text or moving scenery.

The native image is 960×544. The raw RGB ROI is `(32, 360, 896, 184)`, including the
name area and the completion marker. Completion uses `(874, 508, 30, 28)`, brightness
threshold 160 and at least 15 marked pixels. A page must stay stable for at least
two seconds and three observations, with a 0.25-second sampling interval. The
general scalar C GRAD finder accepts up to five lines. Each line is converted to
grayscale and inverted once when its original C polarity flag requests it;
both engines receive those exact saved lossless pixels. Capture invocations use
finite batches for controller review; batch size is outside the frozen OCR/capture
profile and does not alter image acceptance or scoring.

At the target-triggered comparator exit, 40,023 reference characters were counted
and seven previously accepted captures were still queued. Capture stopped, and
one invocation of the unchanged comparator
processed only those seven previously uncompared captures. They each succeeded;
the 27 unsuccessful pages were not retried. The final counters and journal hashes
include this queue drain and all terminal controller receipts. The emulator is
preserved; the collector, comparator and Google worker are closed.
The retained closure receipt SHA256 is
`a5692830aa3be65b71bec56c4e9f5c2b62e0ad56c5f48e5648a0b2d201179128`.

## Runtime and host measurements

The baseline is released FL10 W8I, SHA256
`500e0b842e5f0cb326f55e7d991a52ab66dd24977ccc89bd188228aedc825e84`.
Its weight blob is **657,664 bytes**, and its recognizer arena at maximum width
1024 is **124,832 bytes**. Those are model/runtime buffer sizes, not measured total
console resident memory. The scalar C library is pinned to SHA256
`4f9cef809b88025ccaf3b5ee10f0f22eabc07505c6cd118f8a6af814656bfb59`.

| Live host call | Samples | Median | p95 |
| --- | ---: | ---: | ---: |
| VOCR paired line | 4,271 | 7.38 ms | 20.55 ms |
| Google paired line | 4,271 | 483.35 ms | 1,599.00 ms |
| Google whole ROI | 2,177 | 733.80 ms | 3,532.44 ms |

These are single live calls over different inputs on an x86-64 Linux host,
using Python 3.11.15, NumPy 2.4.6, Pillow 12.3.0 and GCC 16.2.1 with
`-O2 -ffp-contract=off -DVOCR_NO_NEON`. Google timings include network latency.
There are no controlled repeated console samples, physical Vita measurements or
PSP measurements in this run. Neither timing nor emulator results establish
console speed or total memory usage.

## Integrity and reproduction

The capture/comparison harness is frozen at
[`e80690eb6c2f29ee84f5dddf55a1d21d8007c90b`](https://github.com/bee-san/vita-vn-ocr/commit/e80690eb6c2f29ee84f5dddf55a1d21d8007c90b).
The [small aggregate result](../eval/clannad_longrun_results_20261012.json) records
the final lock, source, model, library, teacher/dependency and journal hashes,
the request period, exact counters, scores, host timings and exclusions.

The final read-only audit checks both journal hash chains, one comparison per
capture with no retries, every result-to-capture binding, all native/ROI/line file
and decoded-pixel hashes, native-to-ROI and grayscale/polarity pixel relations,
the completion/stability receipts and foreground signatures, and frozen source,
library, model and installed dependency hashes. It recomputes aggregate counters
from the journals and requires a measured target, zero pending comparisons and
unchanged files throughout the audit. It makes no OCR calls or model forwards.

Accepted frames span 2026-10-11 18:32:05 to 2026-10-12 05:20:48 UTC; retained Google
requests span 2026-10-11 18:32:09 to 2026-10-12 05:33:18 UTC. The final capture
journal SHA256 is
`b6cd582c582d22af883710f202f60f9c44848be23c4a8fe2f726799c7f8f81ed`, and the
comparison journal SHA256 is
`f2754897c8b69f555706b93a3cd4f709b8066c4a7578c03baf828458db584cd9`.
The retained read-only audit exporter SHA256 is
`0b7342d5a013ea56da68acdc791d16c8137b5a068db0f18fc43fa89e44e43f75`.

The [capture guide](clannad-longrun.md) contains the actual collect/compare command
shape and pinned dependency setup. To recompute source-text-free metrics from a
retained private session using the exact frozen harness:

```sh
# Run from the frozen e80690eb6c2f29ee84f5dddf55a1d21d8007c90b checkout.
"$HOST_PYTHON" - "$SESSION_DIR" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'eval')
from clannad_longrun import aggregate, read_journal, validate_lock
session = Path(sys.argv[1])
validate_lock(session)
result = aggregate(read_journal(session / 'captures.jsonl'),
                   read_journal(session / 'comparisons.jsonl'))
assert result['target_met'] and result['counts']['pending_captures'] == 0
print(json.dumps(result, indent=2, sort_keys=True))
PY
```

`HOST_PYTHON` uses the pinned Python 3.11.15, NumPy 2.4.6 and Pillow 12.3.0
environment; `SESSION_DIR` refers to locally retained, ignored data. Screenshots,
crops, either engine's dialogue output and detailed journals are not committed.
Google's remote service has no pinned version; the local implementation,
dependencies, request timestamps and original responses document this run.

This sustained baseline diagnostic complements the shared benchmark. It supplies
no evidence of VOCR2 improvement or promotion and changes no app default. See
[VOCR2 qualification](vocr2-qualification.md) for the separate hand-checked shared
benchmark measurements and the experimental candidate's failed promotion guard.
