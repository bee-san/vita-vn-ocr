# Experimental VOCR2 qualification

The source-frozen C3 textbox finder and quarter-blended FL10 model improve the
shared line benchmark, but **fail the strict held-out whole-dialogue guard**. This
pair is an opt-in experimental VOCR2 trial. It has not qualified for stable
promotion; the released FL10/GRAD default remains unchanged. The trial uses the
same evaluated candidate, with no replacement or tuning after this result.

[Machine-readable results](../eval/vocr2_candidate_20261012.json) contain integer
errors, denominators, platform aggregates, every title guard, failed cells and
the hashes of the frozen evidence. The measurements below are host evaluations
of real native screenshots, not synthetic validation or physical-console tests.

## Model and finder

The candidate is a fixed 25% parameter blend of a generic synthetic adaptation
and the released initializer, selected using development data. Both models retain
FL10's architecture, 3,500-character vocabulary and frozen BN buffers. The
candidate asset is `VOCR2_FL10_w8.vocr`, SHA256
`d3adc52e5cf62b548af04c5ab07a2f57cf0419386bbdf49ba19c9703be6e1766`.
The comparison baseline is `FL10_w8.vocr`, SHA256
`500e0b842e5f0cb326f55e7d991a52ab66dd24977ccc89bd188228aedc825e84`.

Training uses broad Japanese context with bounded kana and similar-kanji
corrections. The existing vocabulary's 3,112 kanji all belong to the pinned Jiten
allowlist; no larger recognition head was introduced. Forwarded training-prefix
caps limit all corrective kanji exposure to 2% and each kana intent to 0.5%.
Protected screenshots, references and the separate Google/CLANNAD long run are
excluded from training and selection. There is no CLANNAD-specific adaptation.

`VOCR_LINES_TEXTBOX` first runs the original GRAD finder. It retains ordinary old
crops, padding and polarity, and uses connected components to repair supported
merged or missing rows. Components with ambiguous topology or insufficient
capacity leave GRAD unchanged. Near-duplicate proposed rectangles are coalesced
before routing. Overlap suppression uses half of the **shorter** vertical span.
This is one generic mode for selected dialogue boxes; its constants do not vary
by game. GRAD and LEGACY behavior remain unchanged.

## Shared fixed-line test

The original `eval/run.py` recognizer evaluation uses console polarity, padding
four native pixels on Vita and two on PSP, and the same source-consistent scalar
C W8I runtime for both models. The three sets contain 476 lines and 5,462
reference characters. Every set's full-text and hiragana errors must not increase;
pooled hiragana errors must decrease. **This fixed-line gate passes.**

| Test set | Full errors / characters, baseline → candidate | Full CER | Hiragana errors / characters, baseline → candidate | Hiragana CER |
| --- | --- | --- | --- | --- |
| Shared Vita | 61 → 59 / 2,564 | 2.38% → 2.30% | 15 → 13 / 1,417 | 1.06% → 0.92% |
| Shared PSP | 133 → 119 / 2,510 | 5.30% → 4.74% | 30 → 25 / 1,382 | 2.17% → 1.81% |
| Separate CLANNAD captures | 7 → 7 / 388 | 1.80% → 1.80% | 3 → 2 / 229 | 1.31% → 0.87% |
| Pooled | 201 → 185 / 5,462 | 3.68% → 3.39% | 48 → 40 / 3,028 | 1.59% → 1.32% |

Both sides completed 476 recognizer forwards on identical fixed-line inputs.
The new, prospectively frozen cross-title gate is separate from the earlier
CLANNAD-only promotion protocol, which remains unexecuted and is not claimed
passed. The CLANNAD capture annotations were assistant reviewed; they are not an
independent human audit.

## Whole-dialogue development and held-out test

Development uses 237 native frames from six reserved titles, with the same
dialogue ROI expanded by zero, four and eight pixels. Selection occurs before
protected access. The chosen pair passes every title/variant all-text and body
full/hiragana development check; five titles gain hiragana accuracy in every
variant. These are **development** results:

| DEV crop | All-text full errors / 6,614 | All-text hiragana / 3,554 | Body full / 6,309 | Body hiragana / 3,514 |
| --- | --- | --- | --- | --- |
| Base | 473 → 339 | 185 → 95 | 397 → 260 | 185 → 95 |
| +4 pixels | 592 → 443 | 242 → 147 | 480 → 334 | 229 → 136 |
| +8 pixels | 567 → 525 | 195 → 172 | 421 → 377 | 171 → 148 |

The single held-out audit covers 195 native ROIs and 585 variant cases across 31
title/platform groups. Its shared Vita scope has 82 ROIs, 202 lines and 2,506
characters; PSP has 93 ROIs, 245 lines and 2,510 characters; the separate CLANNAD
scope has 20 ROIs, 25 lines and 388 characters. Two ambiguous shared Vita regions
were excluded whole before candidate scoring, also excluding four certain lines
and 58 characters. The fixed-line shared benchmark above includes those lines
and remains unchanged.

Each side recognizes every crop returned by its own C finder, in finder order.
There is no GT matching, pixel masking, output filtering or selection. Missing
text incurs deletions; incidental recognized UI incurs insertions. The body
score charges every unmatched insertion as well as edits to reference body
characters. All-text and body full/hiragana errors must not increase for **any**
title/variant; every variant must also improve exact console-balanced hiragana
rates in both scopes and gain on at least two titles.

The candidate improves each platform aggregate below and meets the balanced
hiragana/breadth requirements. **It fails 19 of the 372 per-title metric checks**,
so the complete promotion gate fails. Regressions are one or two full-text errors
in affected title/variant cells; base Vita `v17829` also gains one hiragana error
in both scopes. The public JSON records every failed cell, including its counts.
Aggregate gains do not erase these failures.

| TEST platform / crop | All-text full errors | All-text hiragana errors | Body full errors | Body hiragana errors |
| --- | --- | --- | --- | --- |
| PSP / base | 443 → 424 / 2,510 | 152 → 143 / 1,382 | 328 → 314 / 2,301 | 150 → 142 / 1,369 |
| PSP / +4 | 558 → 552 / 2,510 | 198 → 193 / 1,382 | 430 → 426 / 2,301 | 196 → 191 / 1,369 |
| PSP / +8 | 595 → 586 / 2,510 | 213 → 208 / 1,382 | 448 → 436 / 2,301 | 208 → 203 / 1,369 |
| Vita including CLANNAD / base | 239 → 227 / 2,894 | 70 → 68 / 1,617 | 193 → 183 / 2,722 | 67 → 65 / 1,602 |
| Vita including CLANNAD / +4 | 278 → 261 / 2,894 | 86 → 81 / 1,617 | 226 → 212 / 2,722 | 83 → 78 / 1,602 |
| Vita including CLANNAD / +8 | 291 → 279 / 2,894 | 87 → 84 / 1,617 | 233 → 223 / 2,722 | 83 → 80 / 1,602 |

Baseline and candidate completed 1,388 and 1,394 whole-dialogue recognizer
forwards. Different detection counts are allowed and reported. There was one
frozen protected comparison, with no candidate fallback or protected retuning.

## Memory and host timing

| Resource | Baseline | Candidate |
| --- | ---: | ---: |
| W8I model bytes | 657,664 | 657,664 |
| Recognition arena at maximum width 1,024 | 124,832 | 124,832 |
| Additional finder arena / heap allocation | — | 0 / 0 |
| Host GCC `-O2` finder stack frame | 1,296 | 1,408 |
| Estimated host dark-path call-chain peak | 3,568 | 3,680 |
| Host object text bytes | 7,554 | 14,746 |

The finder reuses dead GRAD scratch and the existing output array. Its unchanged
arena and lack of heap allocation do not imply zero total memory cost: host stack
grows by 112 bytes and host code size grows. Target stack, complete overlay memory
and physical Vita/PSP speed require separate measurements.

The finder alone was timed on the 711 native DEV variant records, with 30
repetitions per record, using C `clock()` process CPU time. Median per-record mean
time was **638.8 → 824.0 μs**; p95 was **1,889.4 → 2,016.7 μs**. This Linux x86_64
GCC 16.2.1 measurement performs no recognizer forwards and uses no protected
images. It is not an estimate of console elapsed time. Raw timing hashes and
frozen finder source hashes are in the JSON.

## Commands and evidence

The ordinary shared benchmark can be run with the pinned project environment:

```sh
python eval/run.py --model /path/to/FL10_w8.vocr \
  --set vita-test,psp-test --polarity console --data /path/to/eval \
  --out /path/to/private-baseline.json
python eval/run.py --model /path/to/VOCR2_FL10_w8.vocr \
  --set vita-test,psp-test --polarity console --data /path/to/eval \
  --out /path/to/private-candidate.json
```

The frozen audit used `eval/run.py`'s original `recognizer_mode` with an explicit
adapter to its pinned scalar library, avoiding a rebuilt or stale default cache.
The private whole-dialogue command was `python protected_pipeline.py compare`
from the frozen audit directory. Finder timing used
`<frozen finder binary> --bench <private native DEV batch> 30` for each side.
The captured environment was CPython 3.11.15, torch 2.5.1+cpu, NumPy 2.4.6,
Pillow 12.3.0 and OpenCV 5.0.0 on Linux x86_64/glibc 2.44, with GCC 16.2.1 and
`-std=c99 -O2 -ffp-contract=off -DVOCR_NO_NEON`.

New captures, crops, private annotations, predictions, trained checkpoints and
private audit drivers remain outside Git. Their frozen hashes permit evidence identity checks;
the private ROI audit is not publicly replayable from this aggregate alone.
Weights belong on GitHub releases. Key evidence hashes are:

| Artifact | SHA256 |
| --- | --- |
| Protected protocol | `6e8274abe09e120511d09a171e8a40ef87a03c7c001b598a63256df83760ff38` |
| Protected preparation | `3890e728459fbf45995a6f8cb78d13b8df3eaa5c42e9afe6cf1aa591d56b0009` |
| Protected paired result | `3c2c873239e1bcbdf3858650d7b628853b0f4b1f4c324ecc9519a7348c8d09f7` |
| Fixed-line result | `e30808015e50199665e6463929e60405fe4c2c383fe2eb72bfe6284aaad196ec` |

Exact source and shared-manifest hashes are recorded under `source_sha256` in
the public JSON. No Google comparison was used to select or qualify this pair.
