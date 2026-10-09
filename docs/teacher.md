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
merged in #2 (206 scored lines, 16 VNs) and psp-test as merged in #5 (245 lines, 14 VNs).

The table is the PR #11 review's rerun: one engine at a time, one thread on one core at
nice 19 of the loaded shared host, outside heavy.sh, whose slots stayed full. It ran on
PR #5's head, and gave the same prediction as the first run on every line both read (that
run had the six engines side by side in one heavy.sh slot). The merged psp-test changes
the boxes of 5 lines, one text and 4 tags. Each engine re-read those 5 crops, which now
take in a closing 」 or 。 the old boxes cut off, and all 245 lines were scored again by
`eval/run.py`. Host ms/line is informational only.

| engine | vita-test CER / exact | psp-test CER / exact | host ms/line (vita / psp) |
|---|---|---|---|
| **PP-OCRv6 small rec** | **1.33% / 89.3%** | **1.23% / 90.2%** | 64 / 53 |
| MeikiOCR rec (BGR, as its docs) | 1.60% / 86.4% | 3.86% / 79.2% | 55 / 53 |
| MeikiOCR rec (RGB, as owocr feeds it) | 1.64% / 85.4% | 3.90% / 78.8% | 56 / 54 |
| Chrome Screen AI (owocr) | 3.20% / 87.9% | 4.62% / 82.9% | 249 / 264 |
| PP-OCRv5 server rec | 3.43% / 74.3% | 11.04% / 48.6% | 5630 / 5679 |
| manga-ocr (re-flowed) | 5.46% / 65.0% | 7.65% / 62.4% | 1104 / 998 |
| *bar: JIS-trimmed int8 PP-OCRv5 mobile (eval, `docs/results.md`)* | *2.30%* | *5.34%* | |

Of the engines measured, PP-OCRv6 small reads both sets best. It is also a CTC model, so
its logits suit the soft (CTC) distillation the owner suggests. MeikiOCR drops characters
on small PSP names: 19.6% CER on psp-test's 62 name lines, against 3.8% for PP-OCRv6 small.
That is one reason the teacher needs voters.

## The teacher on eval's screenshots: choosing the rule

`label.py read --set vita-test|psp-test --no-cascade` runs the whole pipeline on the test
screenshots with every engine on every line. `measure.py rules` then matches each teacher
line to the hand-checked lines its crop overlaps; truth lines on one row are joined. The
lines it matches nothing to are UI, menus and other text eval leaves out. It scores a grid
of 3,672 rules (primary reading, voter set, votes, confidence and detector thresholds, the
confusable veto and the truncation check). Votes come only from engines other than
MeikiOCR, and a rule needs at least two of them (see "Independent votes" below). The rule
chosen is the one with the highest recall whose precision is at least 99.5% on vita-test,
on psp-test and pooled. The rule was chosen on PR #5's psp-test, then measured again on the
merged one (no OCR needed): it is the same rule, with the same numbers on every row of the
two tables below.

**Chosen rule** (`teacher/rule.json`): MeikiOCR's own reading of the line it detected is
the label when at least 2 of PP-OCRv6 small, manga-ocr and Chrome Screen AI read the same
text under eval's scoring normalisation. A dissent that differs only in small kana or ー/一
vetoes it, and so does a Screen AI reading that has the label plus more at one end (the
line runs on past MeikiOCR's box).

| | accepted (scored) | precision | recall of truth lines | label CER |
|---|---|---|---|---|
| vita-test | 175 (143) | 100.00% | 69.4% of 206 | 0.000% |
| psp-test | 154 (141) | 100.00% | 57.6% of 245 | 0.000% |
| pooled | 329 (284) | 100.00% (95% CI 98.7–100%) | 63.0% of 451 | 0.000% |
| **held out, 2-fold by game** (rule chosen on one half of the VNs, scored on the other) | 336 (291) | **99.31%** (86/87 and 203/204) | 58.9% / 66.6% | 0.11% / 0.04% |

The in-sample 100% is tuned on these lines, so the held-out 99.3% is the honest estimate.
Both held-out errors are the teacher's, with eval's truth right: a 「 that MeikiOCR's box
left out (that half's rule had no truncation check), and へえ for the screen's へぇ.
Reference rules on the same lines (pooled):

| rule | precision | recall |
|---|---|---|
| MeikiOCR alone (no vote) | 77.01% | 72.1% |
| MeikiOCR + at least 1 of v6s / Screen AI | 98.16% | 71.0% |
| MeikiOCR (view) + v5s + manga-ocr all agree (label_real.py's engines) | 99.23% | 28.6% |
| MeikiOCR (view) + v6s + manga-ocr (no Screen AI) | 99.39% | 35.9% |

The vote is what makes MeikiOCR usable as a labeller. It costs 9 points of recall (72.1% →
63.0%) and lifts precision from 77% to about 99%. The engines `label_real.py` used reach
the same precision at less than half the recall. Recall by style, for the chosen rule:
furigana 78.3%, outlined 73.4%, bright-on-dark 67.1%, name 63.5%, dark-on-light 54.9%,
mincho 54.7%, small 53.8%, coloured 53.8%. Precision is 100% in every style.

### Independent votes

The first version of this rule (PR #11 before review) also let MeikiOCR's re-read of the
voter view (`meikiv`) vote. That is one recognizer reading two crops of the same line, so
its agreement is weak evidence, and 539 accepted training lines rested on MeikiOCR plus a
single other engine plus `meikiv`. Read by eye, 2 of 30 random ones were wrong: a page
cursor ▾ read as a trailing ・, and にゃるべそ。 where the screen has ぺ. Of the 37 where a
dissenting engine read the same text with a dakuten or handakuten changed, 4 were wrong
(スケシュール, 負けたらとうする。, てきない and the にゃるべそ。 line) and 2 unclear (バ/パ at PSP
size). The
random spot check, 0 wrong of 159 lines then, could not show it, and eval's sets hold only
11 scored lines on that path.

So the grid counts only PP-OCRv6 small, manga-ocr, Screen AI and PP-OCRv5 server as votes,
and needs at least two of them (`measure.MIN_VOTES`). With one, half 0 of the 2-fold
check picks MeikiOCR plus 1 of PP-OCRv6 small / manga-ocr at a minimum character
confidence of 0.3, which scores 96.77% (90/93) on the other half. The chosen rule is the
old one without `meikiv`. It costs 2.4 points of recall on eval's sets (65.4% → 63.0%). On
the relabel it rejects 583 lines the old rule accepted (counted before the repeat cap) and
accepts 7 that only `meikiv`'s confusable dissent had vetoed (all 7 checked by eye and
right). The 2-fold held-out precision went from 99.01% (301/304) to 99.31% (289/291).

## Relabel of the training screenshots

The input is realdata's collection (`shots_{vita,psp,hires}.jsonl` at 1c03d92, 6,998
screenshots). Dropped: every VN with `is_test_vn()` or in eval's `test_games.json` `all`,
every screenshot of a reserved release (`sessions`), and every screenshot of eval's
manifests (PR #5's, now merged). That leaves 6,932 screenshots, read at
native resolution in one heavy.sh slot (5 single-thread workers, 1 h 28 min wall).

- **38,240 lines** read. **11,228 accepted** from 4,923 screenshots, 1,446 VNs and 1,768
  capture sessions: 114,552 characters, 9,840 distinct labels.
  - By source: PC/console 7,169, PSP 2,900, Vita 1,159.
  - By polarity: 7,182 bright-on-dark, 4,043 dark-on-light.
- Rejected:
  - not Japanese: 9,833
  - engines disagree: 7,708
  - too small or too big for the target screen: 7,371
  - cut by the screen edge: 662
  - repeated label cap (2 per VN, 8 overall): 418
  - shorter than 2 characters: 379
  - confusable dissent: 221
  - truncated: 91
  - characters outside `data/vocab.txt`: 2
  - on a val VN linked to a training VN (below): 37 screenshots, 53 lines the rule
    accepted and 53 for the distillation pool
- **Split by game** (as realdata: val = `15 <= md5(vn) % 100 < 25`). Train is 1,265 VNs,
  12,115 crops (7,353 at Vita and 4,762 at PSP resolution). Val is 181 VNs, 1,651 crops
  (972 / 679).
  - One capture can be filed under two VNDB ids, through a bundle release or an alternative
    version. For example, v1762 and v1763 share release r12379 and six byte-identical
    screenshots. md5(vn) alone had put 6 such releases and 14 identical screenshots in both
    splits. A val VN that shares a release or a screenshot (sha256) with a training VN
    anywhere in the collection is therefore left out (`build.linked_val_vns`, 7 VNs).
    realdata's merged split still has these pairs (v1762 in `real_val`, v1763 in
    `real_train`), so select on `teacher_val`, not `real_val`, when training on both.
  - Audited: no VN, capture session, VNDB release or screenshot is in both splits. With
    the 772 val screenshots as the test side, `leakcheck.py` finds no training screenshot at
    0.95 or above. The closest, 0.94, is another VN that reuses the same room background
    and UI, with different text. No screenshot is from a test game, test release or test
    screenshot.
- `eval/tools/leakcheck.py` on the whole collection against the 157 test screenshots found
  no pair at 0.95 thumbnail correlation or above. The closest pair is 0.92: a vita-test map
  scene against another VN's building, which shares only the dialogue-box layout (checked
  by eye).

**Spot check by eye: 0 wrong of 200** random accepted lines, each read glyph by glyph
against its training crop on 48 px-high contact sheets (`teacher/manifests/spot.jsonl`,
whose `by` field says which sample a line came from):
- the 94 of the PR's first 100 (`spot.py sheet --seed 1`) that the final rule still accepts;
- 56 more of the 60 drawn in review (`--seed 7`);
- 50 drawn in review from the rebuilt set, among lines not checked before.

By the rule of three, the label error rate is at most 1.5% at 95% confidence. The review
also checked 20 random lines of the 188 in the final set where another reading (an engine's,
or MeikiOCR's re-read) has the same text with a dakuten or handakuten changed: all 20
right, though one crop (v8576_55673_3) cuts off the tops of the glyphs, where its dakuten
sit.

**realdata's hand checks** (321 training-side lines, `measure.py hand`): 174 of them have
an accepted teacher label. 170 match exactly. I checked the other 4 by eye, and in each
the teacher's label is right for its own crop:
- `『お姫様』`: the screen shows 『』, and the hand label had 「」.
- In three lines the hand-check box was shorter than the teacher's crop, which also
  holds a trailing 、, a leading 「, and a closing ……」 respectively.

**Reproduced in review** (all at one thread, outside heavy.sh, whose slots stayed full):
- `label.py read` on a sample: 6 vita-test, 6 psp-test and 12 training screenshots (4 per
  kind), every engine, offline. 22 of the 24 rows matched the PR's reads exactly. The other
  two hold a ruby that today's reading stage attaches to its host line, while the PR's
  reads, written before the furigana rule changed (8ac0ab9), kept it as a line of its own.
  After `geometry.refurigana`, which re-applies the rule in build and measure, every line,
  crop, reading and decision was the same.
- `measure.py rules` and `build.py decide` on the PR's reads gave its rule, manifests and
  npz byte for byte. With this rule, a second `decide` reproduces the published files byte
  for byte too.

## Outputs (`$HERD/shared/artifacts/teacher/`, README there)

| file | what |
|---|---|
| `teacher_train.npz`, `teacher_val.npz` | the accepted lines in render.py's shard format, so `train/train.py --real` loads them unchanged (train.py drops 0 of them). |
| `teacher_distill_train.npz`, `teacher_distill_val.npz` | 9,044 + 1,261 real text lines the vote rejected, same crops, empty labels: the unlabelled pool for logit distillation. |
| `lines_full.jsonl`, `reads/` | every accepted line with its character boxes; every engine's raw reading of every line. |

Shard details:
- **Labelled shards:** meta per crop has `id, vn, session, kind, target, how, dark, inverted, box`, plus `chars` (MeikiOCR's character boxes and confidences, relative to the crop).
- **Distillation pool:** meta has every engine's reading. train.py drops empty labels, so the pool can't leak into supervised training.
- **Crops** follow eval's: ink box plus furigana, widened to the labelled characters, padded 4 px (Vita) or 2 px (PSP), at the target resolution. Dark-text crops are stored inverted, the console's polarity contract (`common.line_dark`), as realdata's npz.

## Limitations

- If eval's test sets change again, rerun `measure.py rules` and `build.py decide`; the
  reads don't change. `rule.json` records the sha256 of the manifests it was measured on.
- Recall is about 63%, and lower for PSP (58%) and for mincho, small, coloured and
  dark-on-light text (54–55%). The accepted set therefore under-represents the hardest
  styles. The distillation pool holds those lines unlabelled.
- Crop extents (measured with the first rule): on the accepted test lines, 14% of Vita
  and 15% of PSP crops cut eval's hand-fixed ink box by more than the padding on some
  side. It is mostly the outline pixels of outlined names and dialogue, while the labels
  are right. Growing the boxes over edge ink cut fewer glyphs but over-extended more
  boxes (PSP: 7 → 17 boxes more than 4 px too big), so it isn't used.
- Chrome Screen AI is Google's closed-source library from Chromium's CIPD (instance
  `ac82d1f5…`, fetched by that instance id). It only reads screenshots locally to vote,
  and nothing of it is redistributed. MeikiOCR (Apache-2.0, LGPL-3.0 weights), owocr
  (GPL-3.0), manga-ocr and PP-OCR (Apache-2.0) are open.
- AnimeText was not used.

## Reproduce

```sh
teacher/setup_venv.sh --frozen && teacher/.venv/bin/python teacher/label.py fetch
$HERD/heavy.sh teacher/read_all.sh                      # eval sets (all engines), then the collection
teacher/.venv/bin/python teacher/measure.py rules --out $TEACHER_OUT/results/rules.json --rule-out teacher/rule.json
teacher/.venv/bin/python teacher/measure.py hand --out $TEACHER_OUT/results/hand.json
teacher/.venv/bin/python teacher/build.py decide        # manifests + npz (render alone is byte-for-byte repeatable)
teacher/.venv/bin/python teacher/spot.py sheet          # 100 random accepted lines -> contact sheets
$HERD/heavy.sh teacher/measure_engines.sh               # each engine on eval's crops
teacher/.venv/bin/python teacher/report.py              # every table above, from the result files
```

`measure.py rules --rule-out` keeps the `source` note of each eval manifest while its sha256
is unchanged.
