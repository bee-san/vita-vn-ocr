# teacher: MeikiOCR-first, multi-engine labeller for real VN screenshots

Labels real PS Vita, PSP and PC VN screenshots for fine-tuning the tiny recognizer
(train2), following the owner's recipe: **MeikiOCR first** (it targets Japanese game
text and gives line boxes, confidences and character boxes), with other strong
**local** engines voting through [owocr](https://github.com/AuroraWright/owocr).
Uncertain or incomplete readings are rejected, the rule is chosen by measurement on
eval's hand-checked sets, and a random sample of accepted lines is checked by eye.
It supersedes `realdata/`'s PP-OCRv5 + manga-ocr labeller and reuses its screenshot
collection (realdata's `shots_<kind>.jsonl`).

## Engines (all local, CPU, run with no network)

| name | engine | reads | pinned by |
|---|---|---|---|
| `meiki` | MeikiOCR 0.3.5 (rtr46): meiki.text.detect.v0.1 960×544 + meiki.text.recognition.v0 960×32 | the whole screenshot: line boxes (+ detector score), text, per-character boxes and confidences | HF revisions, sha256 (`engines.PINS`) |
| `meikiv` | MeikiOCR's recognizer | the line's voter view (the line plus a margin) | same |
| `v6s` | PP-OCRv6 small rec (rapidocr 3.10.0 ONNX) | the voter view | sha256 |
| `v5s` | PP-OCRv5 server rec (rapidocr 3.10.0 ONNX), `label_real.py`'s engine | the voter view | sha256 |
| `manga` | manga-ocr base (kha-white), `label_real.py`'s engine; long lines re-flowed into a block | the voter view | HF revision |
| `screenai` | Chrome Screen AI through `owocr.ocr.ChromeScreenAI` (owocr 1.26.8) | the whole screenshot; its lines are matched to the teacher's | CIPD instance sha256 |

owocr's cloud engines (Google Lens, Bing, Google Vision, Azure, OCRSpace) are never
loaded. Screen AI's library describes itself as on-device ("do not send any data to
network"); every teacher job also runs under `unshare -rn`, a network namespace with no
interface, so no engine could reach the network if it tried. OneOCR, Apple Vision/Live
Text and WinRT are Windows/macOS only.

## Pipeline

```sh
teacher/setup_venv.sh                                  # teacher/.venv: owocr + MeikiOCR + voters (pinned)
teacher/.venv/bin/python teacher/label.py fetch        # model files -> $TEACHER_OUT/models (sha256-checked)
PY="unshare -rn teacher/.venv/bin/python"
# 1. read eval's test sets with every engine, and score each engine on eval's own crops
$HERD/heavy.sh $PY teacher/label.py read --set vita-test --no-cascade --procs 5
$HERD/heavy.sh teacher/measure_engines.sh vita-test,psp-test
# 2. choose the rule: recall-max at >= 99% precision, 2-fold by game
teacher/.venv/bin/python teacher/measure.py rules --out $TEACHER_OUT/results/rules.json
# 3. relabel the training screenshots (realdata's collection, minus eval's test games/sessions)
$HERD/heavy.sh $PY teacher/label.py read --set train --procs 5 --voters screenai,v6s,manga
teacher/.venv/bin/python teacher/build.py decide       # rule.json -> manifests/, npz, README
# 4. spot check
teacher/.venv/bin/python teacher/spot.py sheet         # 100 random accepted lines -> contact sheets
```

`$TEACHER_OUT` defaults to `$HERD/shared/artifacts/teacher`.

1. **Lines** (`geometry.py`): MeikiOCR detects every text line at native resolution.
   Pieces of one row are merged, each box is refined to its ink (either polarity) and
   small lines just above a line become its furigana (inside the crop, never in the
   label), all as eval builds its test lines. Vertical text is dropped.
2. **Votes** (`label.py`): every voter reads the line's *view*, the ink box widened by
   0.6 line heights left and right (stopping halfway to a neighbour). A line MeikiOCR cut
   short therefore reads longer for the voters and is rejected instead of being labelled
   with half its text.
3. **Rule** (`vote.accept`, chosen in `rule.json`): MeikiOCR's reading is the label when
   its confidences pass and enough voters read the same agreement key, which is eval's
   scoring normalisation: two readings agree exactly when they would score the same on
   the benchmark. Small kana, dakuten, ー/一 and lookalike kanji must match.
4. **Build** (`build.py`): labels outside `data/vocab.txt` are dropped, a label is kept
   at most twice per VN and 8 times overall, eval's test games and sessions are excluded
   again, the split is by game (as realdata: val = `15 <= md5(vn) % 100 < 25`), and crops
   are cut like eval's (ink box + furigana, widened to the labelled characters, plus 4 px
   on Vita or 2 px on PSP) at the target resolution (PC shots scaled to fit 960×544 and
   480×272).

## Files

| file | what |
|---|---|
| `rule.json` | the acceptance rule and the measurements that chose it |
| `manifests/shots.jsonl` | every screenshot with an accepted line: url, sha256, vn, kind, release, session, split |
| `manifests/lines_<kind>.jsonl` | every accepted line: id, crop box (source px), label, how, targets, split |
| `manifests/games.json`, `stats.json` | games (split, sessions, counts) and build statistics |
| `manifests/spot.jsonl` | the by-eye spot check of 100 random accepted lines |
| `$TEACHER_OUT/teacher_{train,val}.npz` | grey crops in render.py's shard format; meta has the character boxes |
| `$TEACHER_OUT/lines_full.jsonl`, `reads/` | accepted lines with character boxes; every engine's raw reading |

Screenshots and crops are copyrighted: they stay under `$HERD/shared/artifacts`, never
in git.
