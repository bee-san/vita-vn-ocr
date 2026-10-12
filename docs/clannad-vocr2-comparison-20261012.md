# Released VOCR2 on the fixed CLANNAD Google-proxy session

The single approved comparison completed and an independent read-only audit
reproduced its inputs, counts and scores. Released VOCR2 has fewer edit
disagreements than cached original FL10 on this fixed session. Google is an
automated reference proxy, not hand-checked ground truth/CER; this diagnostic
does not qualify the model for promotion or establish general VN accuracy.

| Fixed measurement | Reference characters | Cached FL10 disagreements | Released VOCR2 disagreements | Fewer edits | Rate decrease, percentage points | Relative edit reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Identical original line PNGs: full text | 36,915 | 3,176 (8.60%) | 3,069 (8.31%) | 107 | 0.29 | 3.37% |
| Identical original line PNGs: hiragana | 20,323 | 993 (4.89%) | 928 (4.57%) | 65 | 0.32 | 6.55% |
| Original whole RGB ROI: full text | 40,114 | 5,618 (14.01%) | 5,138 (12.81%) | 480 | 1.20 | 8.54% |
| Original whole RGB ROI: hiragana | 21,729 | 2,181 (10.04%) | 1,889 (8.69%) | 292 | 1.34 | 13.39% |

Decreases are calculated directly from the integer edit counts before rounding:
rate decrease is `100 * (FL10 edits - VOCR2 edits) / reference characters`, and
relative edit reduction is `100 * (FL10 edits - VOCR2 edits) / FL10 edits`.
They describe fewer disagreements with the same automated reference, rather than
a percentage increase in human-verified OCR accuracy. Hiragana rates use full-text
alignment from the frozen scorer; small kana, dakuten and similar kanji stay distinct.

The line-only comparison has 107 fewer full-text and 65 fewer hiragana
disagreements. The whole-ROI comparison has 480 fewer full-text and 292 fewer
hiragana disagreements. These rows use different reference requests and cannot
be added together or subtracted to isolate a finder-only effect.

Both measurements retain the same 2,177 successful unique pages from the original
2,204-capture session. All 27 originally unsuccessful comparison pages remain
excluded and retained, without retry: three Google line request failures and
24 successful whole-ROI requests with empty normalized text. No new Google
request, capture, controller input, baseline model forward, training, tuning or
model selection occurred. Cached line references and whole-ROI references are
unchanged, including successful empty line references.

The recognizer-only comparison uses all 4,271 original processed line PNGs with
`invert=False`; their stored polarity was already applied. The whole-ROI comparison
uses the same 2,177 original RGB ROI PNGs `(32,360,896,184)` from native 960×544
frames. Released C3 mode 2 returned 4,345 lines: 483 pages with one line, 1,289 with
two, 344 with three, 54 with four, six with five and one with six. All returned
lines were processed, including the six-line page. No finder result was empty.
The 566 normalized blank line-only predictions and 607 blank whole-ROI line
predictions remain actual predictions in the score; they are not filtered.

Released VOCR2 is model SHA256
`d3adc52e5cf62b548af04c5ab07a2f57cf0419386bbdf49ba19c9703be6e1766`, with C3 finder
source revision `49ff6f4d1c55804c600d098810574a49d382a009`. C4 is not involved.
Recognition uses the exact original scalar library SHA256 `4f9cef…`, because
recognizer C/H source bytes are identical between the old frozen harness and
this release. The C3-only finder library is SHA256 `e6495076…`, compiled from the
released C3 source. Model CRC, 3,501 classes, 657,664-byte blob and 124,832-byte
W8I recognizer arena at maximum width 1,024 were verified. These are buffer sizes,
not total physical-console resident memory measurements.

There were exactly **8,616** candidate recognizer calls: 4,271 original line inputs
plus 4,345 new C3 crop inputs. The independent audit verified all 4,271 original
line input/file/pixel bindings, reconstructed all 4,345 new crop pixel hashes
from unchanged ROI pixels and recorded boxes/polarity, checked all three journal
hash chains and exact root/attempt/completion bindings, and reproduced every
published score, count and host quantile. It performed zero additional model,
native finder or Google calls. All original and candidate artifacts remained
unchanged during the audit.

Single live host-call median/p95 on the same 4,271 old line inputs was 7.38/20.55 ms
in the historical baseline run and 5.97/14.17 ms in this run. Executions took place
at different times under uncontrolled host load; these values do not establish
a speed improvement or console performance. Environment: x86-64 Linux, pinned
Python 3.11.15, NumPy 2.4.6, Pillow 12.3.0, scalar C W8I, GCC 16.2.1 with
`-O2 -ffp-contract=off -DVOCR_NO_NEON`.

The frozen command was:

```sh
"$HOST_PYTHON" fixed_compare.py compare \
  --directory "$PREPARED_DIR" --authorization "$ROOT_AUTHORIZATION"
```

`HOST_PYTHON` is the pinned Python 3.11 environment. `PREPARED_DIR` is the private
nonoverwriting preparation; approval binds its exact protocol, prepared receipt
and driver hashes. No screenshot, crop, dialogue output or detailed prediction
journal is included. The [aggregate](../eval/clannad_vocr2_comparison_20261012.json)
and [independent audit](../eval/clannad_vocr2_comparison_audit_20261012.json) preserve
their original bytes and hashes. The [source snapshot and input requirements](../eval/reproducibility/clannad-vocr2-20261012/README.md)
archive the exact comparison, generated-test and audit helpers used for this run.

| Artifact | SHA256 |
| --- | --- |
| Frozen driver | `70ba2b7956dcef89d4f22c87d7f280d0189577264a20a16b21a8c672e995de8b` |
| Protocol | `105660119aa7d0151500d396b7071715e4e784f3ae904ef7419a13fd656847e8` |
| Prepared receipt | `5170914a9c693c8cb5549b48d7e05e58e932b8166b323180e024c66306b329d1` |
| Candidate journal | `664688518e01e065e6f8943d031bae8ced7588bc66b241d4b8e0d30cf25ebc6c` |
| Public aggregate | `b5dda9225fc2187e10cd9964d4c5a3da4d3cfbc4a7f7d0b634b9582d85d2ca60` |
| Independent read-only audit | `88cb8332cdfd7f7972679fd2d19b6fad2dc6d96ec12459e1b87cf2c02fffd6ee` |

The released VOCR2 trial remains experimental under its separate shared
[hand-checked benchmark qualification](vocr2-qualification.md). This post-release CLANNAD diagnostic
does not change the model, app default or earlier promotion decision.
