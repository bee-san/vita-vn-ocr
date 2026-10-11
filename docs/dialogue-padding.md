# Horizontal dialogue crop padding

Some small kana errors change when a line crop has more horizontal background. The fixed candidate widens each existing GRAD crop by 4 px on each horizontal side, clamps to the user-selected dialogue ROI, preserves its vertical geometry, and recomputes polarity from the widened pixels. It uses horizontal padding 7 and vertical padding 3 around the contrast mask; the mask's one-pixel expansion gives approximately 8/4 px around ink. Detection bands, filtering, order, maximum line count, legacy behavior and model weights remain unchanged.

The initial development study used the released FL10 blob (`500e0b842e5f0cb326f55e7d991a52ab66dd24977ccc89bd188228aedc825e84`) and 19 hand-checked CLANNAD dialogue ROIs from session `clannad-opening-A`. The development manifest SHA256 was `a32ba010c2b8dc7ed79fe81965c53e87d0bd8ab02a7757c215614ccaba1c387b`. With a C GRAD finder and scalar C W8I recognizer, full errors changed from 9/207 to 7/207 (4.348% to 3.382% CER), and hiragana errors from 1/127 to 0/127. Both paths found 18 of 19 lines with no false positives. All 18 crops widened without ROI or resized-width clamps; polarity did not change on these captures. These are development results from one game, not held-out accuracy or console speed measurements.

The study ran on x86_64 Linux with Python 3.11.15 and GCC 16.2.1, using `-O2 -ffp-contract=off`. The FL10 blob remained 657,664 bytes and its recognition arena 124,832 bytes at maximum width 1024. Horizontal padding adds no finder arena allocation; generated tests check the existing 83,323-byte Vita and 25,355-byte PSP finder arenas.

## Prospective evaluation

`eval/padding_protocol.py` compares the fixed candidate against the original finder from commit `d6ef77d`, using the same released blob and identical dialogue ROIs. The primary score matches both prediction streams using the original detection boxes. A second score matches the candidate's emitted expanded boxes and reports every changed association. Missing reference lines and unmatched detections count as errors; uncertain lines retain the shared evaluator's treatment. This prevents improvements in matching from silently being attributed to recognition.

Before opening any held-out pixels, commit the code and create an immutable lock. It binds the clean candidate commit, every evaluation source hash, baseline source, released blob and all three manifest identities. `lock` reads manifest metadata only. `run` requires the exact preregistered lock SHA256, validates every manifest before scoring any set, and creates an exclusive receipt keyed by that lock. Reusing the lock with another output filename is refused.

From the candidate worktree, with private captures and shared screenshots kept outside Git:

```bash
PY=/home/bee/Documents/src/vita-vn-ocr/.venv-ci/bin/python
$PY -B eval/padding_protocol.py dev \
  --manifest /home/bee/Documents/src/vita-vn-ocr/build/hiragana/clannad/dev/labels.jsonl \
  --blob /home/bee/Downloads/vita-release-2026-10-11/ocr/FL10_w8.vocr \
  --out build/dialogue-padding/dev.json

$PY -B eval/padding_protocol.py lock \
  --clannad-manifest /home/bee/Documents/src/vita-vn-ocr/build/hiragana/clannad/test/labels.jsonl \
  --data /absolute/path/to/shared/eval/screenshots \
  --blob /home/bee/Downloads/vita-release-2026-10-11/ocr/FL10_w8.vocr \
  --out build/dialogue-padding/lock.json

# Substitute the exact digest printed by lock; record it before scoring.
$PY -B eval/padding_protocol.py run \
  --lock build/dialogue-padding/lock.json \
  --lock-sha256 PREREGISTERED_SHA256 \
  --out build/dialogue-padding/heldout.json
```

Acceptance requires full error nonregression on each of `vita-test`, `psp-test` and `clannad-test`, plus a strict CLANNAD hiragana error reduction, under both the fixed and emitted-box matching methods. Reference character counts must remain identical. Candidate detections must equal the original boxes with exactly the prescribed ROI-clamped expansion. The C finder must equal its Python twin, and C polarity must match the final recognition crop. Reports include ROI clips, padding clamps, width clamps, matching changes, code/data/model hashes, compiler commands and exact model/finder memory.

If the locked candidate fails, retain the original preprocessing. Do not choose a different margin from these held-out results. This protocol leaves `train/hiragana_gate.py` unchanged: trained-model acceptance continues to compare recognizer-only reports with the original equal padding. Any future combined model/preprocessing change needs its own prospective evaluation. Physical-console accuracy and timing remain separate validation work; this change authorizes no deployment.
