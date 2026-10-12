# Fresh native cohort reproduction snapshots

These are the exact helper bytes used for the 2026-10-12 source join and public
provenance projection, plus the independent stored-output audit. They are stored
as `.snapshot` evidence, not importable
production entry points. Use Python 3.11.15 without optimization (`-O` disables
the helpers' assertions), with the repository's pinned dependencies.

`build_fresh_inputs.py.snapshot` verifies a private JSON configuration's input
hashes, recorded exclusions, per-source Japanese console release metadata,
actual recorded redirect URLs, native file/RGB hashes, references and 4/8-pixel
ROI expansions. It delegates admission, deduplication and the unchanged quotas
to `eval/fresh_native_textbox.py` before creating new private input files.
It performs no inference. Before final admission it correctly rejected the
under-quota Oumagatoki cohort; no prepared run or evaluation was created for it.

The configuration contains `driver` (the frozen worktree), `driver_sha256`
(the harness file hash), hash-pinned `exposure`, `source_pools`, `acquisitions`,
`references` and `reviews`, plus `titles` mapping VNDB IDs to `platform`.
Each pin is `{ "path": "<private file>", "sha256": "<hash>" }`; source pools also
provide the document's list `key`. Private references use the source-only
annotation schema shown by the helper. Do not fabricate missing labels.

`publish_source_provenance.py.snapshot` checks the input-join receipt against the
completed result and publishes a whitelist of URLs, hashes, native crop geometry,
review identity and aggregate line counts. It excludes pixels, transcripts,
predictions and absolute private paths.

`audit_stored_predictions.py.snapshot` independently recomputes edit distances
with a two-row dynamic program and ranked operation trace, then verifies all
case/aggregate counters, the fixed gate, coverage, quotas, deduplication and
frozen input/code/model/environment pins. It imports no harness scorer or gate
and makes no model, finder or teacher calls. Its saved result is
`eval/fresh_native_textbox_independent_audit_20261012.json`.

Commands, with private paths substituted:

```sh
python build_fresh_inputs.py.snapshot --config private/final-config.json \
  --output private/final-inputs
python eval/fresh_native_textbox.py prepare \
  --manifest private/final-inputs/manifest.json \
  --annotations private/final-inputs/annotations.json \
  --model private/FL10_w8.vocr --library private/libvocr_scalar.so \
  --directory private/fresh-test --compiler gcc
python eval/fresh_native_textbox.py verify \
  --prepared private/fresh-test/prepared.json
# The coordinating source/protocol reviewer supplies the exact authorization.
python eval/fresh_native_textbox.py compare \
  --prepared private/fresh-test/prepared.json \
  --authorization private/fresh-test/execution-authorization.json
python publish_source_provenance.py.snapshot \
  --inputs private/final-inputs --selection private/final-selection.json \
  --result private/fresh-test/result.json --output public-source-provenance.json
python audit_stored_predictions.py.snapshot \
  --prepared private/fresh-test/prepared.json --root private/frozen-harness \
  --out independent-audit.json
```

The original one-shot test is complete and must not be rerun or used for tuning.
Changing code, paths, compiler or environment changes the preparation identity;
an independent reproduction must preserve the method and report its own identity.
Private game assets and exact references are not distributed with these snapshots.
The source provenance JSON contains the frozen input filenames and hashes.
