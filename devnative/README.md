# devnative: native Vita/PSP development cohort and its labelling pipeline

Builds the native development cohort described in `docs/devnative.md`, and is the labelling
pipeline other real pools reuse (realplus). Screenshots, readings, crops, sheets and labels live
under `$DEVNATIVE_OUT` (default `$HERD/shared/artifacts/devnative`); git holds the code and
`manifests/` (URLs, hashes, ids, boxes, provenance; never text or pixels).

## Pipeline

```sh
HERD=/local/home/skerraut/herd-vita-ocr; OUT=$HERD/shared/artifacts/devnative; R=$OUT/review
PY=.venv/bin/python; TPY=teacher/.venv/bin/python
$PY devnative/select.py list                                   # fresh VNDB listing (kana API, polite)
HEAVY_SLOTS=12 $HERD/heavy.sh timeout 3600 $PY devnative/select.py refs     # hash every screenshot to avoid
HEAVY_SLOTS=12 $HERD/heavy.sh timeout 3600 $PY devnative/select.py select   # rules, fetch, dedupe
HEAVY_SLOTS=12 $HERD/heavy.sh timeout 7200 unshare -rn $TPY devnative/read.py run --procs 5
$PY devnative/sheets.py make --mode pass1 --out $R/pass1 --chunk 120          # -> decisions.jsonl per batch
$PY devnative/decide.py queue --pass1 $R/pass1 --out $R/pass2_queue.json
$PY devnative/sheets.py make --mode pass2 --lines $R/pass2_queue.json --out $R/pass2 --chunk 57
$PY devnative/gradcheck.py boxes --pass1 $R/pass1 && $PY devnative/gradcheck.py sheets --out $R/gradcheck --chunk 120
$PY devnative/gradcheck.py apply --batch $R/gradcheck
$PY devnative/decide.py final --pass1 $R/pass1 --pass2 $R/pass2
$PY devnative/decide.py sample --n 300 --seed 20261012 --out $R/blind_ids.json
$PY devnative/sheets.py make --mode blind --lines $R/blind_ids.json --out $R/blind --chunk 100
$PY devnative/decide.py blind --batch $R/blind --out $R/blind_report.json            # the measured agreement
$PY devnative/edgecheck.py boxes --evidence $R/gradcheck $R/blind && $PY devnative/edgecheck.py sheets --out $R/edgecheck --chunk 60
#   confirming reads: every kept line with one look, blind, framed on its final crop
#   ({"ids": [...], "frames": {id: final crop}} from $OUT/edge/boxes.jsonl), then
$PY devnative/sheets.py make --mode blind --lines $R/confirm_ids.json --out $R/confirm --chunk 58
$PY devnative/decide.py final --pass1 $R/pass1 --pass2 $R/pass2 --blind $R/blind --confirm $R/confirm $R/confirm2 \
    --resolve $R/resolutions.jsonl                                              # third looks (assistant)
$PY devnative/edgecheck.py boxes --evidence $R/gradcheck $R/blind $R/confirm $R/confirm2
$PY devnative/edgecheck.py apply --batch $R/edgecheck
#   cohort v1 (f5a1dbd) was built with the next line into $OUT and devnative/manifests; to check it,
#   build into a scratch directory: its four sha256s and manifests match v1's byte for byte
$PY devnative/build.py npz --grad $OUT/grad/grad.jsonl --edge $OUT/edge/edge.jsonl --out /tmp/v1 --manifests /tmp/v1/manifests
#   cohort v2: select.py's rules exclude eval/linefind screenshots everywhere; rebuild the
#   selection from the local files and build only its screenshots' lines, into v2/
$PY devnative/select.py select --no-fetch
$PY devnative/build.py npz --grad $OUT/grad/grad.jsonl --edge $OUT/edge/edge.jsonl \
    --selection devnative/manifests/shots.jsonl --out $OUT/v2
$PY devnative/overlap.py --cohort v1 $OUT/v1/manifests/shots.jsonl $OUT/v1/manifests/lines.jsonl \
    --cohort v2 devnative/manifests/shots.jsonl devnative/manifests/lines.jsonl --require-clean v2
$PY devnative/report.py                                                         # manifests/review.json
```

Never write over `$OUT/devnative_*.npz`: they are cohort v1, which study 1b's protocol pins by
sha256. v1's selection report and manifests are under `$OUT/v1/` (and tag `devnative-cohort-v1`).

| stage | file | what it does |
|---|---|---|
| select | `select.py` | eligibility (reserved VNs only; no test VN, CLANNAD family or training-label VN), subsets, fetch, duplicate checks against every eval, linefind, training and scene screenshot (VNDB id, file sha256, decoded-pixel sha256, same frame re-encoded); no screenshot of eval's or linefind's sets in any subset |
| read | `read.py` | MeikiOCR detection + reading on the whole screen, teacher's geometry (rows merged, ink boxes, furigana), and four readings per line: `meiki` (its own box), `meikiv` (MeikiOCR on the wider view), `v6s` (PP-OCRv6 small on the view), `v6sc` (PP-OCRv6 small on the crop). Pinned engine files, no network |
| sheets | `sheets.py` | contact sheets of at most 1280 x 1280 px: the framed crop in context, integer zoom, numbered candidates with differences in red; pass 1 shows engine support, pass 2 and the blind re-check shuffle candidates and hide it |
| decide | `decide.py` | validates `decisions.jsonl`, classifies each pass-1 label (consensus: both engine families read it; single: one; changed: none), queues single and changed labels for the independent second pass, applies the rule, samples and scores the blind re-check |
| GRAD check | `gradcheck.py` | the console finder on each dialogue region (the lines pass 1 called dialogue), matched to the lines; a box that holds the whole line and no other detected text is used, and one that runs beyond the reviewed ink crop only after a contact-sheet check that it shows exactly the label (the detector misses some glyphs: card digits, icons, run-on text) |
| edge check | `edgecheck.py` | final ink crops widened to the labelled characters' boxes (teacher's rule); every crop that changed, has ink continuing past it, or that a later look called cut is checked on a contact sheet (mode edge); a failed check drops the line |
| report | `report.py` | text-free summary of every stage's counts, the blind agreement and the third looks (`manifests/review.json`) |
| build | `build.py` | final ink crops and the checked GRAD crops, console polarity, npz files in teacher_val's format, manifests. Without `--grad` it uses regions of the kept lines and leaves out every GRAD box that runs beyond the ink crop. `--selection` builds only the lines of the screenshots a selection manifest holds |
| overlap | `overlap.py` | an independent re-check of a cohort's screenshots against vita-test, psp-test and every eval/linefind manifest (id, file sha256, pixel sha256, same frame, near-duplicate); `manifests/overlap.json` |
| geometry | `lines.py` | regions, finder matching, crop boxes, polarity (pure numpy; tested in CI) |

## Rules

- Development labels (`decide.py final --mode dev`): a consensus label is kept after pass 1; a
  single or changed label only when the independent second pass reads the same key; unsure,
  skipped or disputed lines are dropped. The agreement key is eval's scoring normalisation
  (`teacher/vote.key`): small kana, dakuten, ー/一 and lookalike kanji must match.
- Training labels (`--mode train`): at least one engine family must read the label after review
  (changed labels are dropped); `--auto-consensus` keeps lines all four readings agree on
  without review; pass 2 is optional and, when given, must agree.
- After the blind re-check, every kept line without two independent agreeing reads gets a blind
  confirming read on its final crop (`--confirm`); a disagreeing, unsure or skipping read drops the
  line unless a third-look resolution (`--resolve`) keeps it.
- Every decision is recorded with `reviewer_kind: "assistant"` (Claude reading the sheets).

## Reusing it on another screenshot pool

`read.py` reads any `shots.jsonl` with one row per screenshot: `shot` (unique key), `vn`, `kind`
(`vita` or `psp`: the console size it is read at; other sizes are scaled to fit, as teacher's
hires shots), `url`, `path` (the image), and optionally `release`, `subset`, `flags`. Use
`read.py run --shots X --out DIR`, `sheets.py make --reads DIR/reads.jsonl --chunk N` for
review batches (one helper per batch directory writes its `decisions.jsonl`), `decide.py
--reads ... final --mode train`, and `build.py npz --prefix NAME --manifests DIR`.
