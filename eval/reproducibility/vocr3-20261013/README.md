# VOCR3, 2026-10-13: the scripts behind the selections and the locked report

The conductor's drivers, as run (paths are the shared box's; `$S3` = `$HERD/shared/artifacts/synth3`). Kept verbatim,
not maintained; the committed result files record their inputs, so the selections can be rerun without them.

| file | what it produced |
|---|---|
| `select_size.sh SIZE [export\|score\|budget\|select\|all]` | one size class's selection: exports (`eval/synth3_ablation.py export`), scores on the amendment-8 cohorts (`$S3/vocr3/select_pools`), the budget check, `eval/vocr3_select.py realplus` then `base` (`eval/vocr3_realplus_<SIZE>_20261013.json`, `eval/vocr3_selection_<SIZE>_20261013.json`). The H15 size ran once before the driver read the amendment-8 copies; that pass was superseded (commit "recomputed on amendment 8's cohorts") |
| `budget_check.sh OUT BLOB...` | `vita/budget_sweep.py check` from a main worktree (fab5d3f+: prints the tier), synth-val lines only |
| `report.sh spec\|shared\|a9\|overlay\|render` | the locked report: `eval/vocr3_make_spec.py`, then `eval/vocr3_report.py` once (`eval/vocr3_report_20261013.json`, `docs/vocr3-report-20261013.md`); `shared` ran from tag `vocr3-report-runner-9eadb90`, the other steps from `vocr3-report-runner-f978800` |
| `dev_scores.py` | `eval/vocr3_dev_scores_20261013.json`: every candidate's development scores and budget check (reported only; the rows behind docs/vocr3.md's selection tables) |
