# Optional textbox row recovery

`VOCR_LINES_TEXTBOX_RECOVERY` adds mode 3 to the C line finder. It recovers
coherent missing rows when scene pixels inside a selected dialogue box distort
the projection height or the old width-based permission check. The recognizer
uses the unchanged released FL10 weights. This mode remains optional; the
default finder is GRAD, and this change does not promote a model or overlay build.

The measurements below are **DEVELOPMENT ONLY**, using native screenshots.
Development evidence informed the algorithm family. Independent shared/fresh
test accuracy and physical-console latency/residency remain unmeasured for this
mode. [The aggregate JSON](benchmarks/native-textbox-recovery-development-20261012.json)
contains every title/variant counter, reference hashes, fixed gate and source
hashes, without images or dialogue transcripts.

## Algorithm

Recovery runs after the existing TEXTBOX operation. Every existing crop keeps
its coordinates, padding and polarity. The shared local-contrast mask supplies
connected components with core heights 8–96 pixels. Average horizontal support
of at least twice the core height makes a component strong. This is geometric
support; the runtime has no recognized strings, character classes, title IDs or
annotations.

A 97-byte histogram in dead run scratch selects a supported core-height class
using geometry-matched old rows. Repeated supports must have compatible 5:4
heights; incompatible supported scales leave the old result. With insufficient
old support, independently coherent components can provide a scale. A lower,
separated component may override one scene-inflated anchor only with at least
5:4 greater aspect support. Competing incompatible independent scales are
ambiguous and retain the existing output.

New cores must be 55–125% of the chosen height. A crop already covering half the
candidate height and half the smaller horizontal width suppresses recovery.
Otherwise, broad or short rows require a strong compatible neighbor: center
separation between one and four main heights, and either nearby starts or
overlap across half the smaller width. Weak candidates cannot support one
another. The selected independently supported main row can support itself.

Only new crops can expand for detached marks. A bounded probe confirms a
complete small vertical span, including its closing empty row, before attaching
it. The span limit is `max(5, main_height / 4)`; horizontal joined runs must be
at most half the main height. Continuing ink, a source edge, a broad row or a
neighbor-core midpoint fence rejects the attachment. This preserves tiny paired
and ring marks while rejecting partial ruby or neighboring strokes.

Final proposed rectangles are deduplicated before and after mark attachment at
90% overlap in both dimensions. Existing TEXTBOX crops win duplicates. If all
eligible additions cannot fit the 16 output slots, the original result remains.
Unused output slots stage final geometry; candidate cores remain immutable
during the mark probes. Polarity and reading order use the existing functions.

## Frozen development comparison

The cohort contains 237 native frames from six titles: three PSP and three Vita.
Base, four-pixel and eight-pixel ROI expansions produce 711 cases. Each variant
has 6,614 all-text characters and 6,309 body characters; the hiragana denominators
are 3,554 and 3,514 respectively. These references are excluded from training.

The private, pinned audit command was `python paired_pipeline.py compare`.
One fresh-process comparison used GCC 16.2.1, CPython 3.11.15, torch 2.5.1 CPU,
NumPy 2.4.6 and Pillow 12.3.0 on x86-64 Linux. Both sides used the same explicitly
pinned scalar C W8I library and
[released FL10 asset](https://github.com/bee-san/vita-vn-ocr/releases/download/v0.2.1/FL10_w8.vocr),
SHA256 `500e0b842e5f0cb326f55e7d991a52ab66dd24977ccc89bd188228aedc825e84`.

Before candidate scoring, all 711 original GRAD detection/input records and
complete raw baseline rows reproduced the archived baseline exactly. Both sides
recognized every returned crop in finder order, including blank results.
Per-line normalization precedes concatenation. Full-transcript Levenshtein ties
prefer diagonal, deletion, then insertion. Body counters charge every unmatched
insertion plus substitutions/deletions of body reference characters. No
annotation-based matching or output filtering selects successful crops.

| ROI | Original GRAD full / 6,614 | Pure C3 full / 6,614 | C4 full / 6,614 | C3 → C4 full-error reduction |
| --- | ---: | ---: | ---: | ---: |
| Base | 473 | 341 | 264 | 22.58% |
| +4 px | 592 | 457 | 314 | 31.29% |
| +8 px | 567 | 532 | 415 | 21.99% |

| ROI | All hiragana / 3,554, GRAD → C3 → C4 | Body full / 6,309, GRAD → C3 → C4 | Body hiragana / 3,514, GRAD → C3 → C4 |
| --- | ---: | ---: | ---: |
| Base | 185 → 99 → 57 | 397 → 265 → 191 | 185 → 99 → 57 |
| +4 px | 242 → 155 → 75 | 480 → 345 → 206 | 229 → 142 → 62 |
| +8 px | 195 → 173 → 114 | 421 → 386 → 272 | 171 → 149 → 90 |

The gate was fixed before this one comparison and passed with no failure
reasons. All 72 title/variant all/body full/hiragana checks were nonregressing
against original GRAD, and all 12 pooled metrics were nonregressing against
pure C3. The required 10% pooled full-error reductions used immutable maximum
counts 306/411/478. Every variant strictly improved console-balanced body
hiragana, with at least two improving titles per console: `v2632`/`v5154` on PSP
and `v18163`/`v18643` on Vita. Consoles receive equal weight, and titles receive
equal weight within each console.

The original GRAD and C4 sides completed 1,453 and 1,563 recognizer forwards.
Detection counts were 470/470/513 versus 499/510/554. Against saved C3 outputs,
no existing crop or its raw recognition changed. Recovery added 23/34/36 crops
in 17/29/30 frames; 13/15/14 additions were blank and remain in the accounting.
Scene texture still produces false positives, and this family preserves old C3
false positives. Ambiguous scale, clipping and isolated short rows can still
leave missing text. These development results do not establish promotion.

## Memory and static target build

The FL10 blob stays 657,664 bytes and its W8I arena stays 124,832 bytes at maximum
width 1024. Finder arenas remain 83,323 bytes for 960×544, 25,355 for 480×272 and
235 for 8×8. Recovery adds no heap allocation, arena, recognition head, output
array or mark array.

An actual VitaSDK GCC 15.2.0 compile of the frozen source used effective `-O2`,
`-mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard`, and `-fstack-usage`. These are
**compile-only per-function frame sizes and object sections**, not physical
memory residency or measured runtime speed.

| Static property | Original GRAD | Packaged C3 | C4 |
| --- | ---: | ---: | ---: |
| `find_grad` frame | 1,224 B | 1,400 B | 1,440 B |
| `vocr_find_lines` frame | 80 B | 80 B | 80 B |
| `vocr_line_dark` frame | 2,120 B | 2,120 B | 2,120 B |
| Object text | 4,636 B | 9,716 B | 12,456 B |
| Object data / bss | 0 / 0 B | 0 / 0 B | 0 / 0 B |

The new `recovery_mark_span` frame is 48 bytes. C4's target object and stack-usage
SHA256 values are recorded in the aggregate JSON. Per-function frames are not
a total stack measurement.

## Generated validation

Before the real development comparison, 213 semantic/existing-finder tests
passed, with two holdout metadata tests excluded. Modes 0/1/2 matched published
commit `2c028bf655af08b6a864630671c78c917fbadb62` byte for byte on 473 generated
images, explicitly checking 1,419 calls. Both GCC 16.2.1 and Clang 22.1.8
ASan/UBSan checks passed 2,400 procedural images per compiler with exact arenas,
alignment offsets 0–3 and guard bytes. The private audit driver passed 47
generated contract tests, including equal-score/different-string baseline
rejection. Generated checks cover punctuation geometry, detached marks, ruby,
nearby short rows, boundary crops, polarity, scales and duplicate suppression.

The pytest CI checkout retrieves history so that the frozen parity test can
read that exact published source commit. Other jobs, gates and dependency pins
are unchanged.

```sh
.venv-ci/bin/python -m pytest -q tests/test_linefind_recovery.py tests/test_linefind.py \
  -k 'not manifests and not holdouts'
.venv-ci/bin/ruff check .
PYTHON=.venv-ci/bin/python ci/runtime.sh gcc
PYTHON=.venv-ci/bin/python ci/runtime.sh clang
```
