# teacher: MeikiOCR-first multi-engine labels for real VN screenshots

The owner's recipe fine-tunes the recognizer on real Vita/PSP screenshots labelled on the
PC by a strong teacher, MeikiOCR first, with other local engines voting through owocr,
uncertain lines rejected, and the teacher measured on the hand-checked test sets before
it is trusted. `teacher/` does that. It relabels realdata's screenshot collection and
supersedes realdata's PP-OCRv5 + manga-ocr labels. Pipeline and files: `teacher/README.md`.

## Engines

All of them run locally on the CPU, offline: every job runs under `unshare -rn`, a
network namespace with no interfaces. None of owocr's cloud engines is ever loaded.

| name | engine (pinned) |
|---|---|
| `meiki` | MeikiOCR 0.3.5 detector + recognizer on the whole screenshot (line boxes, detector score, per-character boxes and confidences) |
| `meikiv` | MeikiOCR's recognizer on the line's voter view (the line plus a margin) |
| `v6s`, `v5s` | PP-OCRv6 small rec and PP-OCRv5 server rec (rapidocr 3.10.0 ONNX) on the voter view |
| `manga` | manga-ocr base on the voter view, re-flowed into a block |
| `screenai` | Chrome Screen AI through `owocr.ocr.ChromeScreenAI` (owocr 1.26.8), whole screenshot |

## Each engine on eval's crops (recognizer only)

`eval/run.py --model 'cmd:… teacher/engine_cli.py ENGINE'` (`teacher/measure_engines.sh`)
reads eval's own crops: box plus 4 px on Vita, 2 px on PSP. The sets are vita-test as
merged in #2 (206 scored lines, 16 VNs) and psp-test from PR #5's head bebd4f9 (245 lines,
14 VNs). Each engine ran on one thread, so host ms/line is informational only.

| engine | vita-test CER / exact | psp-test CER / exact | host ms/line |
|---|---|---|---|
| **PP-OCRv6 small rec** | **1.33% / 89.3%** | **1.47% / 87.8%** | 499 / 399 |
| MeikiOCR rec (BGR, as its docs) | 1.60% / 86.4% | 4.10% / 76.7% | 495 / 414 |
| MeikiOCR rec (RGB, as owocr feeds it) | 1.64% / 85.4% | 4.14% / 76.3% | 499 / 412 |
| Chrome Screen AI (owocr) | 3.20% / 87.9% | 4.86% / 80.4% | 1365 / 803 |
| PP-OCRv5 server rec | 3.43% / 74.3% | not run (queued) | 4460 |
| manga-ocr (re-flowed) | 5.46% / 65.0% | not run (queued) | 879 |
| *bar: JIS-trimmed int8 PP-OCRv5 mobile (eval)* | *2.30%* | *5.58%* | |

Of the engines measured, PP-OCRv6 small reads both sets best. It is also a CTC model, so
its logits suit the soft (CTC) distillation the owner suggests. MeikiOCR drops characters
on small PSP names: 20.1% CER on psp-test's 62 name lines, against 4.3% for PP-OCRv6 small.
That is one reason the teacher needs voters.

## The teacher on eval's screenshots: choosing the rule

`label.py read --set vita-test|psp-test --no-cascade` runs the whole pipeline on the test
screenshots with every engine on every line. `measure.py rules` then matches each teacher
line to the hand-checked lines its crop overlaps; truth lines on one row are joined. The
lines it matches nothing to are UI, menus and other text eval leaves out. It scores a grid
of 17,106 rules (primary reading, voter set, votes, confidence and detector thresholds, the
confusable veto and the truncation check). The rule chosen is the one with the highest
recall whose precision is at least 99.5% on vita-test, on psp-test and pooled.

**Chosen rule** (`teacher/rule.json`): MeikiOCR's own reading of the line it detected is
the label when at least 2 of PP-OCRv6 small, manga-ocr, Chrome Screen AI and MeikiOCR's
re-read of the view read the same text under eval's scoring normalisation. A dissent that
differs only in small kana or ー/一 vetoes it, and so does a Screen AI reading that has
the label plus more at one end (the line runs on past MeikiOCR's box).

| | accepted (scored) | precision | recall of truth lines | label CER |
|---|---|---|---|---|
| vita-test | 181 (147) | 100.00% | 71.4% of 206 | 0.000% |
| psp-test | 162 (148) | 100.00% | 60.4% of 245 | 0.000% |
| pooled | 343 (295) | 100.00% (95% CI 98.7–100%) | 65.4% of 451 | 0.000% |
| **held out, 2-fold by game** (rule chosen on one half of the VNs, scored on the other) | 352 (304) | **99.01%** (91/93 and 210/211) | 62.3% / 68.9% | 0.29% / 0.04% |

The in-sample 100% is tuned on these lines, so the held-out 99.0% is the honest estimate.
Reference rules on the same lines (pooled):

| rule | precision | recall |
|---|---|---|
| MeikiOCR alone (no vote) | 77.01% | 72.1% |
| MeikiOCR + at least 1 of v6s / Screen AI | 98.16% | 71.0% |
| MeikiOCR (view) + v5s + manga-ocr all agree (label_real.py's engines) | 99.23% | 28.6% |
| MeikiOCR (view) + v6s + manga-ocr (no Screen AI) | 99.39% | 35.9% |

The vote is what makes MeikiOCR usable as a labeller. It costs 7 points of recall (72.1% →
65.4%) and lifts precision from 77% to about 99%. The engines `label_real.py` used reach
the same precision at less than half the recall. Recall by style, for the chosen rule:
outlined 75.5%, furigana 78.3%, bright-on-dark 68.8%, name 65.2%, dark-on-light 58.8%,
mincho 56.9%, small 56.6%, coloured 56.4%. Precision is 100% in every style.

## Relabel of the training screenshots

The input is realdata's collection (`shots_{vita,psp,hires}.jsonl` at 1c03d92, 6,998
screenshots). Dropped: every VN with `is_test_vn()` or in eval's `test_games.json` `all`,
every screenshot of a reserved release (`sessions`), and every screenshot of eval's
manifests (PR #5's superset of the merged list). That leaves 6,932 screenshots, read at
native resolution in one heavy.sh slot (5 single-thread workers, 1 h 28 min wall).

- **38,240 lines** read. **11,783 accepted** from 5,033 screenshots, 1,454 VNs and 1,785
  capture sessions: 120,659 characters, 10,291 distinct labels.
  - By source: PC/console 7,519, PSP 3,050, Vita 1,214.
  - By polarity: 7,553 bright-on-dark, 4,227 dark-on-light.
- Rejected:
  - not Japanese: 9,879
  - too small or too big for the target screen: 7,412
  - engines disagree: 6,999
  - cut by the screen edge: 662
  - repeated label cap (2 per VN, 8 overall): 492
  - shorter than 2 characters: 380
  - confusable dissent: 327
  - truncated: 170
  - characters outside `data/vocab.txt`: 2
- **Split by game** (as realdata: val = `15 <= md5(vn) % 100 < 25`). Train is 1,268 VNs,
  12,651 crops (7,678 at Vita and 4,973 at PSP resolution). Val is 186 VNs, 1,786 crops
  (1,052 / 734). Audited: no VN or capture session is in both splits, and no screenshot is
  from a test game, test release or test screenshot.
- `eval/tools/leakcheck.py` on the whole collection against the 157 test screenshots found
  no pair at 0.95 thumbnail correlation or above. The closest pair is 0.92: a vita-test map
  scene against another VN's building, which shares only the dialogue-box layout (checked
  by eye).

**Spot check by eye: 0 wrong of 100** random accepted lines. That is
`teacher/manifests/spot.jsonl`, each read glyph by glyph against its training crop on
48 px-high contact sheets. By the rule of three, the label error rate is at most 3% at
95% confidence.

**realdata's hand checks** (321 training-side lines, `measure.py hand`): 177 of them have
an accepted teacher label. 173 match exactly. I checked the other 4 by eye, and in each
the teacher's label is right for its own crop:
- `『お姫様』`: the screen shows 『』, and the hand label had 「」.
- In three lines the hand-check box was shorter than the teacher's crop, which also
  holds a trailing 、, a leading 「, and a closing ……」 respectively.

## Outputs (`$HERD/shared/artifacts/teacher/`, README there)

| file | what |
|---|---|
| `teacher_train.npz`, `teacher_val.npz` | the accepted lines in render.py's shard format, so `train/train.py --real` loads them unchanged (train.py drops 0 of them). |
| `teacher_distill_train.npz`, `teacher_distill_val.npz` | 8,355 + 1,229 real text lines the vote rejected, same crops, empty labels: the unlabelled pool for logit distillation. |
| `lines_full.jsonl`, `reads/` | every accepted line with its character boxes; every engine's raw reading of every line. |

Shard details:
- **Labelled shards:** meta per crop has `id, vn, session, kind, target, how, dark, inverted, box`, plus `chars` (MeikiOCR's character boxes and confidences, relative to the crop).
- **Distillation pool:** meta has every engine's reading. train.py drops empty labels, so the pool can't leak into supervised training.
- **Crops** follow eval's: ink box plus furigana, widened to the labelled characters, padded 4 px (Vita) or 2 px (PSP), at the target resolution. Dark-text crops are stored inverted, the console's polarity contract (`common.line_dark`), as realdata's npz.

## Limitations

- psp-test comes from eval's PR #5 head, which wasn't merged when I measured. If it
  changes, rerun `measure.py rules` and `build.py decide`; the reads don't change.
- Recall is about 65%, and lower for PSP (60%), mincho, small and dark-on-light text
  (57–59%). The accepted set therefore under-represents the hardest styles. The
  distillation pool holds those lines unlabelled.
- Crop extents: on the accepted test lines, 14% of Vita and 15% of PSP crops cut eval's
  hand-fixed ink box by more than the padding on some side. It is mostly the outline
  pixels of outlined names and dialogue, while the labels are right. Growing the boxes
  over edge ink cut fewer glyphs but over-extended more boxes (PSP: 7 → 17 boxes more
  than 4 px too big), so it isn't used.
- Chrome Screen AI is Google's closed-source library from Chromium's CIPD (instance
  `ac82d1f5…`). It only reads screenshots locally to vote, and nothing of it is
  redistributed. MeikiOCR (Apache-2.0, LGPL-3.0 weights), owocr (GPL-3.0), manga-ocr and
  PP-OCR (Apache-2.0) are open.
- AnimeText was not used.

## Reproduce

```sh
teacher/setup_venv.sh && teacher/.venv/bin/python teacher/label.py fetch
$HERD/heavy.sh teacher/read_all.sh                      # eval sets (all engines), then the collection
teacher/.venv/bin/python teacher/measure.py rules --out $TEACHER_OUT/results/rules.json --rule-out teacher/rule.json
teacher/.venv/bin/python teacher/measure.py hand --out $TEACHER_OUT/results/hand.json
teacher/.venv/bin/python teacher/build.py decide        # manifests + npz (render alone is byte-for-byte repeatable)
teacher/.venv/bin/python teacher/spot.py sheet          # 100 random accepted lines -> contact sheets
$HERD/heavy.sh teacher/measure_engines.sh               # each engine on eval's crops
teacher/.venv/bin/python teacher/report.py              # every table above, from the result files
```
