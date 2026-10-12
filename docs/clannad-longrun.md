# CLANNAD long-run Google OCR comparison

The capture/reference workflow is general across visual novels: it uses the unchanged console finder, released model and identical-input comparison, with no game-specific OCR rules. CLANNAD supplies one external capture profile (ROI, completion marker and controller key). This run collects a new, isolated Vita3K CLANNAD **test** session and compares the released FL10 recognizer against Google Lens through `owocr==1.26.7`. It aims for 40,000 normalized whole-dialogue Google reference characters. Google output is an automated proxy, not hand-checked ground truth: the reported edit disagreements are **not human CER**. This test must never select training checkpoints, margins, thresholds, models or preprocessing rules. The existing hiragana model gate is unchanged.

The baseline model is locked to SHA256 `500e0b842e5f0cb326f55e7d991a52ab66dd24977ccc89bd188228aedc825e84`, 657,664 blob bytes and 124,832 recognizer arena bytes at width 1024. The host compiles the unchanged dependency-free C GRAD finder and C W8I recognizer without NEON. Timings are host measurements; Google times include network latency. Neither timings nor Vita3K results demonstrate physical-console performance.

## Frozen inputs and accounting

Before capture or any reference request, `lock.json` freezes the baseline model, runtime/compiler/library hashes, Python source hashes, installed engine/dependency content hashes, exact native 960×544 geometry, selected ROI, completion/stability options and actual advance key. Every accepted capture stores the full native screenshot, raw RGB dialogue ROI and each exact polarity-adjusted grayscale line as a lossless PNG, with file and decoded-pixel SHA256. The C finder runs on the selected ROI; its relative line boxes and vertical geometry remain unchanged. Its polarity flag is applied once to each grayscale line, and **both engines receive those same stored pixels**. Google Lens may encode/preprocess that image internally, as pinned in its source.

The independent whole-ROI Google request uses the original RGB ROI and does not inherit finder omissions. Its text is compared separately with both concatenated line-Google and line-VOCR outputs. This measures coverage/order disagreement as well as recognition disagreement; it cannot establish which engine is correct. Google line boxes and line counts are retained locally for inspection.

Captures and comparison attempts are append-only, hash-chained journals. Resume verifies role, lock, source bytes, decoded pixels and result-to-input binding; changed or partially written inputs fail explicitly. No OCR exception becomes an empty success. A capture contributes to both reported OCR denominators only when every paired line request and VOCR call succeeds and the whole-ROI reference succeeds with nonempty normalized text. Failures and pending captures are counted separately. Blank successful line-Google outputs retain their actual zero reference length and any VOCR insertion disagreement; they are not request failures.

Stable foreground-mask identity excludes repeated screens across resume, while ignoring the continuation marker. It uses the fixed dialogue region, not full-screen background/cursor animation. Google text is never used to deduplicate distinct inputs. Identical foreground text is conservatively treated as a duplicate even if scenery changes. Stability alone does not certify a completed typewriter page: an explicit continuation-marker box or operator acknowledgement is required. An operator must pause on choices/menus; suspicious empty, truncated or excessive-line detections stop collection for review. Completion detection is capture bookkeeping, never an OCR preprocessing change. No choice is selected automatically; blank transitions remain under controller control. A 120-second idle timeout pauses for controller review instead of hanging on a duplicate page or mistaking a menu for dialogue. The operator must verify the actual advance key: menu animation alone is not evidence of accepted input.

## Setup and sustained run

Keep screenshots, crops, transcripts, lock and journals under ignored `build/`. The optional standalone engine environment needs Python 3.11 and only the explicitly pinned packages below, not GameSentenceMiner or owocr desktop integrations:

```sh
uv venv --python 3.11 .venv-owocr
uv pip install --python .venv-owocr/bin/python --no-deps -r eval/requirements-owocr-longrun.txt
```

Before starting this new test session, use a separate ignored pilot to verify navigation and skip the previously captured development opening. Do not count known development/pilot frames toward the new 40,000-character test. The collector and comparison worker stay alive; they do not restart Python per page. An external game controller may advance the window, or the collector may hold an explicitly configured key for the configured key-hold interval (0.3 seconds in the verified Qt profile) after an accepted completed page. It focuses the selected window before each key input. Use the actual working frontend/window/key; the following geometry and marker came from the original development screenshot, not a held-out tuning run:

The excluded capture pilot expands this game's ROI to include the complete name
area and lowers the completion threshold to recognize its later triangle icon.
These are controller/capture-profile settings, not recognizer preprocessing or
training rules. The same raw ROI is passed to the general finder and whole-ROI
reference. The verified game controls use `c` to advance and `x` to reveal the
current message without advancing. An optional reveal key is disabled by default;
only configure it after checking the game's actual controls. Automatic advancement
requires an operator screen-kind file, which must be paused before opening any
menu. Bright menu controls can resemble a continuation marker.

```sh
mkdir -p build/clannad-longrun
printf '%s\n' '{"kind":"dialogue","complete":false}' > build/clannad-longrun/control.json

/home/bee/Documents/src/vita-vn-ocr/.venv-ci/bin/python eval/clannad_longrun.py collect \
  --display :97 --window 6291485 --session clannad-longrun-20261011 --split test \
  --session-dir build/clannad-longrun --model /private/FL10_w8.vocr \
  --owocr-python /home/bee/Documents/src/vita-vn-ocr/.venv-owocr/bin/python \
  --roi 32,360,896,184 --complete-box 874,508,30,28 \
  --native-size 960,544 --complete-threshold 160 --complete-pixels 15 --count 2000 \
  --control-file build/clannad-longrun/control.json \
  --interval .25 --stable-seconds 2 --stable-samples 3 --max-lines 5 \
  --advance-key c --key-hold-seconds .3 --reveal-key x --reveal-delay .5
```

The completion marker is automatically excluded from stability, but retained in the app ROI and OCR inputs. A `--control-file` may override the screen kind with JSON such as `{"kind":"pause","complete":false}` or `{"kind":"choice","complete":false}`. Without a marker, require an operator to issue a unique per-page acknowledgement `{"kind":"dialogue","complete":true,"page":"000001"}` after verifying the fully displayed text. Menus/choices/transitions and pauses never trigger automatic key input. Configured control-file path and capture options are frozen; acknowledgement content can change intentionally as controller state.

In another process:

```sh
/home/bee/Documents/src/vita-vn-ocr/.venv-ci/bin/python eval/clannad_longrun.py compare \
  --session-dir build/clannad-longrun --model /private/FL10_w8.vocr \
  --owocr-python /home/bee/Documents/src/vita-vn-ocr/.venv-owocr/bin/python \
  --follow --stop-at-target
```

`--follow` consumes newly appended frozen captures. The worker initializes Google Lens once. `summary.json` gives successful paired line characters/errors, independent ROI reference characters/errors, finder coverage disagreement, exclusions, hashes and host timing quantiles. The target is the independent ROI reference count on the **same fully successful unique captures**, not a mixture of successes from different engines. When it reaches 40,000 characters, stop capture via an ignored `STOP` file in the session directory. The same sentinel stops a following comparator. Google’s remote service has no pinned version; the lock freezes local engine/dependencies, and request times and immutable responses document the actual comparison. On resume, successful captures are not re-requested; failures remain excluded unless `--retry-errors` explicitly appends a fresh attempt. Never rewrite old attempts.

Only code, URL/hash manifests and small aggregate results may be committed. Native screenshots, cropped pixels and either engine's dialogue output remain ignored and local. Do not claim a completed 40,000-character result until `target_met` is measured true. This diagnostic run neither changes nor promotes the runtime model.

## Validation

```sh
/home/bee/Documents/src/vita-vn-ocr/.venv-ci/bin/python -m pytest -q tests/test_clannad_longrun.py
```

Tests use generated pixels and fake engines; they do not request Google or load protected benchmark images.
