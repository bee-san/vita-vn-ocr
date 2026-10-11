# Horizontal dialogue crop padding

**Rejected research experiment.** The locked candidate failed held-out acceptance. Retain the original preprocessing for deployment. This branch keeps the rejected padding change as its default solely to reproduce the experiment; it is not approved for deployment, merge into the runtime, or release.

Some small kana errors change when a line crop has more horizontal background. The fixed candidate widens each existing GRAD crop by 4 px on each horizontal side, clamps to the user-selected dialogue ROI, preserves its vertical geometry, and recomputes polarity from the widened pixels. It uses horizontal padding 7 and vertical padding 3 around the contrast mask; the mask's one-pixel expansion gives approximately 8/4 px around ink. Detection bands, filtering, order, maximum line count, legacy behavior and model weights remain unchanged.

The initial development study used the released FL10 blob (`500e0b842e5f0cb326f55e7d991a52ab66dd24977ccc89bd188228aedc825e84`) and 19 hand-checked CLANNAD dialogue ROIs from session `clannad-opening-A`. The development manifest SHA256 was `a32ba010c2b8dc7ed79fe81965c53e87d0bd8ab02a7757c215614ccaba1c387b`. With a C GRAD finder and scalar C W8I recognizer, full errors changed from 9/207 to 7/207 (4.348% to 3.382% CER), and hiragana errors from 1/127 to 0/127. Both paths found 18 of 19 lines with no false positives. All 18 crops widened without ROI or resized-width clamps; polarity did not change on these captures. These are development results from one game, not held-out accuracy or console speed measurements.

The study ran on x86_64 Linux with Python 3.11.15 and GCC 16.2.1, using `-O2 -ffp-contract=off`. The FL10 blob remained 657,664 bytes and its recognition arena 124,832 bytes at maximum width 1024. Horizontal padding adds no finder arena allocation; generated tests check the existing 83,323-byte Vita and 25,355-byte PSP finder arenas.

## Completed held-out evaluation

The candidate was frozen at `91e4cd17f9b85d410560bb0e9cff796ada630616` before a single held-out run. The [committed aggregate report](../eval/dialogue_padding_20261011.json) preserves the lock, source, manifest, model and raw-report hashes. All numbers below are full console pipeline scores on real screenshots with the same released FL10 weights.

| Set | Full errors, original → candidate | Reference characters | Hiragana errors, original → candidate | Hiragana characters |
|---|---:|---:|---:|---:|
| Vita | 219 → 223 | 2,564 | 66 → 67 | 1,417 |
| PSP | 450 → 440 | 2,510 | 152 → 145 | 1,382 |
| CLANNAD | 23 → 23 | 388 | 5 → 6 | 229 |

Vita full CER regressed from 8.541% to 8.697%, and CLANNAD hiragana CER regressed from 2.183% to 2.620%. These fail the preregistered requirement of full-error nonregression on every set and strict CLANNAD hiragana improvement. PSP's improvement does not satisfy that rule. Original-box and emitted-box matching gave identical scores on all three sets, with zero changed associations. Padding was clamped in 49 Vita and 179 PSP crops; one Vita crop changed polarity. No annotated GT line was clipped by the selected ROI, and neither path clamped a resized crop at width 1024.

No further margin or device-specific rule was chosen using these results. No candidate was deployed, promoted, merged into the runtime, or released.

## Frozen evaluation protocol

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
  --data /home/bee/Documents/src/ps-vita-ocr/data/eval \
  --blob /home/bee/Downloads/vita-release-2026-10-11/ocr/FL10_w8.vocr \
  --out build/dialogue-padding/lock.json

# This completed lock was consumed once. The command is an audit record;
# its immutable receipt refuses a repeat run against the same lock.
$PY -B eval/padding_protocol.py run \
  --lock build/dialogue-padding/lock.json \
  --lock-sha256 33c73eb3c75f86b1a6529f959566d403eebca2d0e1f9cce6f12a3807cd7a21b8 \
  --out build/dialogue-padding/heldout.json
```

Acceptance requires full error nonregression on each of `vita-test`, `psp-test` and `clannad-test`, plus a strict CLANNAD hiragana error reduction, under both the fixed and emitted-box matching methods. Reference character counts must remain identical. Candidate detections must equal the original boxes with exactly the prescribed ROI-clamped expansion. The C finder must equal its Python twin, and C polarity must match the final recognition crop. Reports include ROI clips, padding clamps, width clamps, matching changes, code/data/model hashes, compiler commands and exact model/finder memory.

The locked candidate failed, so retain the original preprocessing. Do not choose a different margin from these held-out results. This protocol leaves `train/hiragana_gate.py` unchanged: trained-model acceptance continues to compare recognizer-only reports with the original equal padding. Any future combined model/preprocessing change needs its own prospective evaluation. Physical-console accuracy and timing remain unmeasured for this rejected candidate.
