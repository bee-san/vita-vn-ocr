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
   kana or ー/一 vetoes it. Labels follow the corpus conventions (`vote.canon`).
4. `build.py`: hand checks override the vote; labels outside `data/vocab.txt` are
   dropped; one label is kept at most twice per VN and 8 times in all; crops are
   padded like the overlay's line finder (4 px, 5 px below) at the target resolution.

Test games are excluded at every step: `synth/common.is_test_vn()` and the eval set's
reserved VNs (`eval/manifests/test_games.json` once merged, `eval_exclude.json` until then).

## Files

| file | what |
|---|---|
| `manifests/shots_<kind>.jsonl` | every screenshot: `vn, kind, url, dims, release, sexual, violence, sha256, session` |
| `manifests/lines_<kind>.jsonl` | every accepted line: `id` (`<vn>_<screenshot id>_<k>`), `ink` box in source pixels, `targets`, `label`, `how`, `pol`, `split` |
| `manifests/games.json` | every VN in the collection: title, split, kinds, sessions, screenshots, accepted lines |
| `manifests/handcheck.jsonl` | lines checked by eye: `truth` (null = unreadable), the `ink` box checked, `set` (dev / spot) |
| `manifests/stats.json` | counts from the last `build.py decide` |
| `$REALDATA/real_train.npz`, `real_val.npz` | grey crops in render.py's shard format; meta per crop: `id, vn, session, kind, target, pol, how` |

The split is by game: a VN is validation when `15 <= md5(vn) % 100 < 25` (`build.split_of`),
otherwise training. A capture session (`<vn>/<release>/<n>`, screenshots of one release
uploaded together) belongs to one VN, so it never straddles the split.

## Reusing the collection (e.g. for another teacher)

```python
import sys; sys.path.insert(0, "realdata")
import vndb, build
for s in vndb.load_shots():          # 6,998 screenshots; s["kind"] in vita | psp | hires
    path = vndb.shot_path(s)         # $REALDATA/shots/<kind>/<vn>_<file>; vndb.py fetch rebuilds it
    split = build.split_of(s["vn"])  # train | val, by game; s["session"] for session-level grouping
```

`vndb.excluded_vns()` plus `is_test_vn()` are the test games to keep out.
