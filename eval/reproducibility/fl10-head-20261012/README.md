# Head-only experiment source snapshots

The `.py.snapshot` files preserve the exact preparation and publication helpers
used for the [rejected head-only FL10 experiment](../../../docs/fl10-head-adaptation-20261012.md).
Restore their original `.py` filenames outside Git to inspect or run them. Their
historical paths require the retained private development/training snapshot;
these are not portable CI tests. They contain no game text or images.

`prepare_head_followup.py.snapshot` makes fresh line and textbox preparations
from the original development inputs using the unchanged fixed protocols. It
copies the prior training recipe and changes only `train_scope` to `head`, with
a distinct run/output identity. It performs no model forward or training.

`publish_head_snapshot.py.snapshot` requires all ten completed checkpoints and
their selector exports. It verifies their bound hashes, emits a whitelist of
stored per-title counters and provenance hashes, and packages only the ten
checkpoint and ten int8 weight files. It reads every ZIP member back and checks
CRC, exact bytes, size and SHA256. It does not perform inference or independently
recompute edit distances from predictions.

The preparation and publication helpers refuse to overwrite their output directory. Do not run them over
the historical completed run. Reproduction needs the pinned original source,
model, vocabulary, source policy, synthetic shards and real development
references; source hashes alone do not reconstruct missing inputs. The
publication helper's output JSON is preserved byte for byte in
[`docs/benchmarks/general-fl10-head-20261012.json`](../../../docs/benchmarks/general-fl10-head-20261012.json).

The driver/source revision is `d80ca3fba113cdd1b9fccefecc9fb6edf8dd8a80`.
Training uses the repository's pinned Python dependencies and frozen private
preparation. Neither preparation nor publication accesses CLANNAD, the fresh multi-title test, Google
OCR, Iwaihime or a physical console.

`audit_head_run.py.snapshot` preserves the independent audit source. It disables
model forwards, scoring, finder/build entry points and network access, verifies
head-only tensors and optimizer membership, reconstructs exports in memory,
checks the exposure journal, and reproduces selection from stored counters. It
does not independently recalculate edit distances. Its original private receipt
is projected into [`eval/fl10_head_adaptation_audit_20261012.json`](../../fl10_head_adaptation_audit_20261012.json):
the run path is removed and historical path prefixes become `head-run/` or
`previous-run/`; the original receipt hash is retained.

The additional [release audit receipt](../../fl10_head_release_audit_20261012.json)
records the exact private-to-public counter projection and all twenty archive
members. The publication helper creates the weight ZIP, results JSON and their
checksum file; publication also copies the two audit receipts and extends the
release checksum file to include them.

| Snapshot | SHA256 |
|---|---|
| `audit_head_run.py.snapshot` | `509fa0e8b4626a874cff849bc21733d439f85a51c2ec2731af7cae44ff816622` |
| `prepare_head_followup.py.snapshot` | `ec5ab7fb39fc179461094d83aa0cea5a6314b29bdfd909d1981a3763d38c5aa4` |
| `publish_head_snapshot.py.snapshot` | `d8a2e00f6379e08114a8aeb8bc7afbd7076fb8cb07637f442611fa3b7f9971a0` |
