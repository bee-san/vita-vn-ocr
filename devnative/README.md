# devnative: native Vita/PSP development cohort and its labelling pipeline

Builds the native development cohort described in `docs/devnative.md`, and is the labelling
pipeline other real pools reuse (realplus). Screenshots, readings, crops, sheets and labels live
under `$DEVNATIVE_OUT` (default `$HERD/shared/artifacts/devnative`); git holds the code and
`manifests/` (URLs, hashes, ids, boxes, provenance; never text or pixels).

## Pipeline

```sh
HERD=/local/home/skerraut/herd-vita-ocr; OUT=$HERD/shared/artifacts/devnative
PY=.venv/bin/python; TPY=teacher/.venv/bin/python
$PY devnative/select.py list                                   # fresh VNDB listing (kana API, polite)
HEAVY_SLOTS=12 $HERD/heavy.sh timeout 3600 $PY devnative/select.py refs     # hash every screenshot to avoid
HEAVY_SLOTS=12 $HERD/heavy.sh timeout 3600 $PY devnative/select.py select   # rules, fetch, dedupe
HEAVY_SLOTS=12 $HERD/heavy.sh timeout 7200 unshare -rn $TPY devnative/read.py run --procs 5
$PY devnative/sheets.py make --mode pass1 --out $OUT/review/pass1 --chunk 200
#   ... decisions.jsonl written in each batch (README.md there has the format)
$PY devnative/decide.py queue --pass1 $OUT/review/pass1 --out $OUT/review/pass2_queue.json
$PY devnative/sheets.py make --mode pass2 --lines $OUT/review/pass2_queue.json --out $OUT/review/pass2 --chunk 150
$PY devnative/decide.py final --pass1 $OUT/review/pass1 --pass2 $OUT/review/pass2
$PY devnative/decide.py sample --n 300 --out $OUT/review/blind_ids.json
$PY devnative/sheets.py make --mode blind --lines $OUT/review/blind_ids.json --out $OUT/review/blind --chunk 150
$PY devnative/decide.py blind --batch $OUT/review/blind --out $OUT/review/blind_report.json
$PY devnative/decide.py final --pass1 ... --pass2 ... --resolve $OUT/review/resolutions.jsonl   # third look
$PY devnative/build.py npz
```

| stage | file | what it does |
|---|---|---|
| select | `select.py` | eligibility (reserved VNs only; no test VN, CLANNAD family or training-label VN), subsets, fetch, duplicate checks against every eval, training and scene screenshot (VNDB id, file sha256, decoded-pixel sha256, same frame re-encoded) |
| read | `read.py` | MeikiOCR detection + reading on the whole screen, teacher's geometry (rows merged, ink boxes, furigana), and four readings per line: `meiki` (its own box), `meikiv` (MeikiOCR on the wider view), `v6s` (PP-OCRv6 small on the view), `v6sc` (PP-OCRv6 small on the crop). Pinned engine files, no network |
| sheets | `sheets.py` | contact sheets of at most 1280 x 1280 px: the framed crop in context, integer zoom, numbered candidates with differences in red; pass 1 shows engine support, pass 2 and the blind re-check shuffle candidates and hide it |
| decide | `decide.py` | validates `decisions.jsonl`, classifies each pass-1 label (consensus: both engine families read it; single: one; changed: none), queues single and changed labels for the independent second pass, applies the rule, samples and scores the blind re-check |
| build | `build.py` | ink crops (ink + furigana + 4/2 px) and GRAD crops (the console finder on each dialogue region, kept only when it holds the whole line and nothing else), console polarity, npz files in teacher_val's format, manifests |
| geometry | `lines.py` | regions, finder matching, crop boxes, polarity (pure numpy; tested in CI) |

## Rules

- Development labels (`decide.py final --mode dev`): a consensus label is kept after pass 1; a
  single or changed label only when the independent second pass reads the same key; unsure,
  skipped or disputed lines are dropped. The agreement key is eval's scoring normalisation
  (`teacher/vote.key`): small kana, dakuten, ー/一 and lookalike kanji must match.
- Training labels (`--mode train`): at least one engine family must read the label after review
  (changed labels are dropped); `--auto-consensus` keeps lines all four readings agree on
  without review; pass 2 is optional and, when given, must agree.
- Every decision is recorded with `reviewer_kind: "assistant"` (Claude reading the sheets).

## Reusing it on another screenshot pool

`read.py` reads any `shots.jsonl` with one row per screenshot: `shot` (unique key), `vn`, `kind`
(`vita` or `psp`: the console size it is read at; other sizes are scaled to fit, as teacher's
hires shots), `url`, `path` (the image), and optionally `release`, `subset`, `flags`. Use
`read.py run --shots X --out DIR`, `sheets.py make --reads DIR/reads.jsonl --chunk N` for
review batches (one helper per batch directory writes its `decisions.jsonl`), `decide.py
--reads ... final --mode train`, and `build.py npz --prefix NAME --manifests DIR`.
