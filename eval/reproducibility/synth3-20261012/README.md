# synth3, 2026-10-12: the scripts behind the reported numbers

The shell scripts and snapshots here are the exact files that produced the study-1, secondary and
protected-test results in `docs/synth3.md` (paths are the shared box's; `$S3` =
`$HERD/shared/artifacts/synth3`). They are kept verbatim, not maintained.

| file | what it produced |
|---|---|
| `score_study1.sh` | study 1's exports, teacher_val scores and `eval/synth3_ablation_20261012.json` |
| `run_test.sh` | every protected-test output in `$S3/ablation/test` (each export once per polarity; existing outputs are never rescored) |
| `score_secondary_ft.sh`, `score_secondary_h15.sh` | the secondaries' exports and teacher_val scores (`eval/synth3_ablation_secondary_20261012.json`) |
| `study2_queue.sh`, `study2_queue2.sh`, `study2_score_arch.sh` | study 2's run order (`study2_queue.sh` launched the ten FL10/H15 runs; from 21:22Z `study2_queue2.sh` runs the rest after VOCR3's width-2.0 runs) and its export and scoring |
| `study2_score_when_ready.sh` | study 2's scoring trigger: waits for an architecture's fine-tunes, records devnative's status, scores and runs `eval/synth3_study2.py result` |
| `meta_check.py.snapshot` | the metadata-fault table (100,000 crops of version 1 and version 2) |
| `fresh_names.py.snapshot` | the first count of fresh-native names in the name pool (single names only; see docs/synth3.md, Limits) |
| `doc_tables.py` | the doc's study-1, secondary and test tables, rounded once from the stored per-line errors |
| `ink_geometry.py` | the teacher_train ink-geometry table (recomputed; first computed inline) |
| `clannad_names.py.snapshot` | the CLANNAD-family name counts in the name pool and the v2 labels |

```sh
eval/reproducibility/synth3-20261012/doc_tables.py --scores $S3/ablation/scores --test $S3/ablation/test
```
