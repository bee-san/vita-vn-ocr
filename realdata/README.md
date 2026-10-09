# realdata: real VN dialogue lines

Real training lines for the recognizer, pseudo-labelled from VNDB screenshots of
Japanese VNs: native PS Vita (960×544), native PSP (480×272), and PC / home-console
screenshots scaled down to both. Replaces `data/label_real.py` (kept unchanged, since
`eval/build/candidates.py` imports it).

## Run it

```sh
PY=$HERD/shared/venv/bin/python                 # torch, rapidocr, manga-ocr, onnxruntime
$PY realdata/vndb.py list --platform psv          # VNDB kana API, one request per 1.2 s
$PY realdata/vndb.py list --platform psp
$PY realdata/vndb.py list --platform pc --top 1500
$PY realdata/vndb.py select                       # -> manifests/shots_{vita,psp,hires}.jsonl
$PY realdata/vndb.py fetch                        # screenshots, sha256 recorded / checked
$HERD/heavy.sh $PY realdata/label.py --shard 0 --nshards 2 --procs 5   # and --shard 1
$PY realdata/build.py decide                      # rule + hand checks -> manifests, npz, README
```

Outputs go to `$REALDATA` (default `$HERD/shared/artifacts/realdata`). Screenshots and
crops are copyrighted: they stay there and are never committed.

## How a line is labelled

1. `lines.py`: PP-OCRv5 mobile det finds text boxes (PSP frames are detected at 2×);
   boxes on one row are merged; each row's ink box is refined with a gradient-energy
   profile, so any text polarity works, and split at the valley before a neighbouring
   line. Rows cut by the screen edge or too small for the target are dropped.
2. `voters.py`: four readers read the line at source resolution: PP-OCRv5 server and
   PP-OCRv6 small (rapidocr ONNX), manga-ocr, and meiki (a game-text character
   detector). `label.py` skips PP-OCRv5 server (the slowest) where `vote.v5s_can_matter()`
   proves it can't change the decision.
3. `vote.py`: a label needs three of the four readers behind one PP-OCR reading;
   manga-ocr and meiki may only differ by invented edge brackets or punctuation, and
   from PP-OCRv6 by ellipsis cell counts. A fourth reader that differs only in small
   kana or ー/一 vetoes it, and a label may start with ・ only when every reader sees it
   (otherwise it is mostly a list bullet). Labels follow the corpus conventions (`vote.canon`).
4. `build.py`: hand checks override the vote; labels outside `data/vocab.txt` are
   dropped; one label is kept at most twice per VN and 8 times in all. Crops are cut at
   the target resolution with the console's conventions: padded so that the grad line
   finder (`runtime/vocr_lines.c`) gives the same boxes, and inverted when `common.line_dark()`
   calls the text dark (docs/linefind.md's polarity contract; `--keep-polarity` skips it).
   `check_finder.py` (random accepted lines, each in a region 2 line heights wider on
   either side and 0.6 taller, as a loose selection would be; 2 × 2,000 lines, seeds 1 and
   2): the finder finds 74.9% at IoU ≥ 0.5. On those, the median edge difference is 0 px on
   every side, except +1 px at x0 with seed 2, but the spread is wide. Only 42–44% /
   70–72% / 39–40% / 62–63% of the x0 / y0 / x1 / y1 edges are within 1 px, and the
   10th–90th percentile range is about −25…+18 px at x0, −4…+2 at y0, −18…+5 at x1 and
   −3…+5 at y1. `line_dark()` on the training box agrees with the finder's dark flag on
   98.2% of the found lines.

Test games are excluded at every step: `synth/common.is_test_vn()` and the eval set's
reserved VNs (`eval/manifests/test_games.json` once merged, `eval_exclude.json` until then),
plus every screenshot of eval's test capture sessions (the `sessions` release ids in the
same files, `vndb.excluded_sessions()`). If eval reserves more VNs or sessions later:
`vndb.py prune`, then `build.py decide`.

Models (labelling only, none ships), pinned in `voters.MODELS` / `voters.MANGA`:

| reader | file | sha256 / revision | source |
|---|---|---|---|
| detector (`lines.py`) | `ch_PP-OCRv5_det_mobile.onnx` | `4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae` | rapidocr 3.10.0 ([ModelScope RapidAI/RapidOCR v3.10.0](https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx)); rapidocr checks this hash itself |
| v5s | `ch_PP-OCRv5_rec_server.onnx` | `e09385400eaaaef34ceff54aeb7c4f0f1fe014c27fa8b9905d4709b65746562a` | [rapidocr 3.10.0, ModelScope](https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_server.onnx) |
| v6s | `PP-OCRv6_rec_small.onnx` | `6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884` | [rapidocr 3.10.0, ModelScope](https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx) |
| meiki | `meiki.text.rec.v0.960x32.onnx` | `3e96bc772fbee9717e536a6353032bb944c3382dd2f6960ef4890decda43b000` | [Hugging Face rtr46/meiki.txt.recognition.v0 at a28cf587](https://huggingface.co/rtr46/meiki.txt.recognition.v0/tree/a28cf5874dc2438ebb1c86336be26bcec51e3375), LGPL-3.0 |
| manga | `kha-white/manga-ocr-base` | revision `aa6573bd10b0d446cbf622e29c3e084914df9741` | Hugging Face, through manga-ocr 0.1.16 |

`voters.model_path()` looks for each ONNX reader in `$REALDATA_MODELS` if set, then in the
installed rapidocr package's `models/` (where rapidocr keeps its downloads) or, for meiki,
the research campaign's copy (`$HERD/shared/research/scratch/eval/models/meiki/`). It checks
the sha256 before the reader loads and names the download URL when a file is missing.

## Results (2026-10-09)

| | |
|---|---|
| screenshots | 6,932 of 1,488 VNs: 574 Vita, 2,354 PSP, 4,004 PC/console |
| lines found / read | 23,752; 17,151 with Japanese text (PP-OCRv5 server skipped on 6,849 by the cascade) |
| accepted | 8,311 lines (89,629 characters, 7,602 distinct labels) from 1,405 VNs; 2,745 dark-on-light |
| crops | real_train 9,811 (Vita 5,452, PSP 4,359), real_val 1,495 (Vita 804, PSP 691) |
| split | train 1,221 VNs / 1,479 sessions, val 184 VNs / 208 sessions |
| unlabelled pool | 7,937 lines (11,174 crops) of 1,223 training-split VNs, for distillation |

Label quality, all by eye on the crop the label goes with:

- Dev sample (221 lines of 201 screenshots, sampled before the full run; 24 unreadable):
  exact-line accuracy of each reader on the 197 readable ones: PP-OCRv6 small 85.8%,
  PP-OCRv5 server 69.0%, manga-ocr 39.6%, meiki 38.1%. The old rule (`data/label_real.py`,
  PP-OCRv5 server agrees with manga-ocr) accepts 37.1% with 72/73 correct; the rule here
  accepts 53.3% with 105/105 correct. Without PP-OCRv5 server it would accept 35.0%
  (68/69), without manga-ocr 42.1%, without meiki 36.5% (`evaluate_rules.py`). The rule
  was chosen on this sample, so these precisions are optimistic.
- The dev sample was read before the valley split changed the readers' view, so the same
  hand truths were also scored against the full run's own readings, taken with the final
  geometry (`evaluate_rules.py --lines "$REALDATA/labels/lines.*.jsonl"`, 195 matched
  readable lines). The rule accepts 59.0% with 115/115 correct. The readers score
  PP-OCRv6 small 86.7%, manga-ocr 47.2% and meiki 40.5%, and PP-OCRv5 server 72.7% of the
  150 lines it read.
- Spot check (100 random accepted, vote-labelled lines of the final set): 1 wrong label
  (a decorative bullet read as ・ by three readers), 1 character error in 987.
- Leading ・ (review of PR #7): of the 8 full-run lines whose vote label started with a ・
  that PP-OCRv5 server didn't read, 7 were a list bullet or a mark in front of a name
  (the spot check's wrong line among them) and 1 the last dot of a ・・・ cut by the ink box.
  The veto rejects them all: 7 lines leave the final set (the eighth is the hand-labelled
  spot line), and no dev decision changes. The cap shuffle then sees a shorter list, so
  61 other lines were swapped for another copy of the same label.

These are training data, not a test set: hand-checked lines are labelled by hand and train
like the rest. The eval test sets come from other games.

## Files

| file | what |
|---|---|
| `manifests/shots_<kind>.jsonl` | every screenshot: `vn, kind, url, dims, release, sexual, violence, sha256, session` |
| `manifests/lines_<kind>.jsonl` | every accepted line: `id` (`<vn>_<screenshot id>_<k>`), `ink` box in source pixels, `targets`, `label`, `how`, `pol`, `split` |
| `manifests/unlabelled_<kind>.jsonl` | the unlabelled pool: every other Japanese line of a training-split game (the readers disagreed): `id`, `ink`, `targets`, `pol` and the four `reads` |
| `manifests/games.json` | every VN in the collection: title, split, kinds, sessions, screenshots, accepted lines |
| `manifests/handcheck.jsonl` | lines checked by eye: `truth` (null = unreadable), the `ink` box checked, `set` (dev / spot) |
| `manifests/stats.json` | counts from the last `build.py decide` |
| `$REALDATA/real_train.npz`, `real_val.npz` | grey crops in render.py's shard format, dark text inverted; meta per crop: `id, vn, session, kind, target, pol, dark, inverted, how`. Train with `train.py --real $REALDATA/real_train.npz`, naming that file: the three npz share one directory, so a glob such as `--real "$REALDATA/*.npz"` would also train on `real_val.npz` |
| `$REALDATA/real_unlabelled.npz` | the crops of `manifests/unlabelled_<kind>.jsonl`, cut the same way: empty labels, the four readings in meta. For soft / logit distillation from a teacher; `train.py --real` drops them all |

The split is by game: a VN is validation when `15 <= md5(vn) % 100 < 25` (`build.split_of`),
otherwise training. A capture session (`<vn>/<release>/<n>`, screenshots of one release
uploaded together) belongs to one VN, so it never straddles the split.

## Reusing the collection (e.g. for another teacher)

```python
import sys; sys.path.insert(0, "realdata")
import vndb, build
for s in vndb.load_shots():          # 6,932 screenshots; s["kind"] in vita | psp | hires
    path = vndb.shot_path(s)         # $REALDATA/shots/<kind>/<vn>_<file>; vndb.py fetch rebuilds it
    split = build.split_of(s["vn"])  # train | val, by game; s["session"] for session-level grouping
```

`vndb.excluded_vns()` plus `is_test_vn()` are the test games to keep out, and
`vndb.excluded_sessions()` the test capture sessions (release ids); `vndb.exclusion(row, ...)`
applies both to a screenshot row.
