# Machine names in committed results

The repository is public. No tracked file names the machine a result was produced on: result files hold a generic
label from `train/hostinfo.py`, `host_label()`: `"<cores>-core <CPU model> host"` (`"64-core AMD EPYC 9R14 host"` on
the build host; physical cores from `/proc/cpuinfo`) or `"build-host"` when the CPU model is unknown. Never the node
name, the socket host name or the FQDN.

| writer | field |
|---|---|
| `eval/run.py` | `env.host` |
| `eval/vocr3_report.py` (the locked reports) | `environment.host` |
| `train/hiragana_fresh.py` (and `hiragana_bitmap.py`, `hiragana_native.py`, which use it) | `environment.host` |
| `train/train2.py` (source-policy runs) | `host.node` of the execution identity (`source_execution.json`, checkpoints) |

`tests/test_repo_policy.py` runs in CI's `ruff + policy` job on every event and fails on an internal host name in
any tracked file, and on code that reads the machine's network name.

## The 2026-10-15 scrub

Until 2026-10-15 seven eval JSONs recorded the internal build machine's hostname, a corporate cloud desktop's name
(found by the publication audit of 2026-10-15). On the owner's request ("remove the hostname", 17:46Z) it was
replaced by `build-host`, and nothing else in those files changed:

- `machine` in five frozen protocols: `eval/synth3_study1b_protocol.json`, `eval/synth3_study2_protocol.json`,
  `eval/synth3_native_protocol.json`, `eval/synth4_protocol.json`, `eval/vocab_proof_protocol.json`;
- `host` in three environment records each of the VOCR3 and VOCR4 locked reports,
  `eval/vocr3_report_20261013.json` and `eval/vocr4_report_20261014.json`.

The pins of those bytes were rewritten to match, in 11 more result files, to a fixpoint (a protocol pins its base
protocol's sha256, and a result pins its protocol's). No measured number changed: old and new have the same
structure, keys and order and the same 107,159 numbers, booleans and nulls, and every string that differs is the
host string or one of the digests below. Both locked reports re-render byte for byte (`eval/vocr3_report.py`'s
`render` and `compact`) from the private full reports with the label. The protocols' own input checks
(`eval/synth3_native_score.sh`, `eval/synth4_score.sh`, `eval/vocab_proof_score.sh`, `eval/vocab_proof_run.sh`)
pass on the new files.

A frozen protocol's file on `main` is therefore no longer the frozen bytes. Those are in git history: the file at
`d214a05` (the last commit before the scrub; `git show d214a05:<file>`) has the "before" sha256 below, and differs
from today's file only in the host string. The 11 digests that changed (37 places):

| what the digest pins | before (up to `d214a05`) | after |
|---|---|---|
| sha256 of `eval/synth3_study1b_protocol.json` (14 places) | `5219b318be54bfae378268c16ca47d4de25f2798587ee15d1f72eee20b6b5205` | `e8f1bfefd3c9f4aa64321d3933eb268b3b5cfa2efc86ddb9131cba4858fea8b3` |
| sha256 of `eval/synth3_study2_protocol.json` (6) | `51bba585b98c58fa773efb32a5789651e4f6ec000b010b566dd399d30dc6c03d` | `7e51810aff2c11356629e18b78bdb4c65264588cebbee93273238b57862de91e` |
| sha256 of `eval/synth3_native_protocol.json` (7) | `87be292dd794ccda64a48ec2a0ab2b49cbc77562f67f3031d8d606a08656e2bb` | `6398fbc03db162bde946f2aee158bde933f56343216992be2f25b98cb885175f` |
| sha256 of `eval/synth4_protocol.json` (2) | `5a14d5c3753d616117767225007056999bf562cf9d3e7f037cd511549ceff3e3` | `1ea0bbc723299d563e400e4a7898ac22766b31953c9a7b4ee4cdd1adb1ffc28e` |
| sha256 of `eval/vocab_proof_protocol.json` (2) | `ca8de1e5b6fe9ebedfc2a7a55fc8dadc18205bf11ff4223209fa3177206d9c09` | `11863c1446b4c245983d88985c46ddc8d9e4f3aa34ab09fafbf4ba19f2c08107` |
| VOCR3 report `part_sha256.a9` | `e5d632491a05a0853290748af0e0fbf0a877636416ffbf4ae95ae79468704802` | `3cc8329c6d8de64762f6c26bbb99cbf01e1c675c7660a1dbcddf9bab74fae74b` |
| VOCR3 report `part_sha256.overlay` | `7ad0ffb87827292bb45e96419f1fb0d4ecaab659d128551decf2dc9423a33e74` | `990fe1c62e516d84e5e4a2a69ae66325e08605776ef00633cb6cf2232e51e3cf` |
| VOCR3 report `part_sha256.shared` | `57b842f91a9b9a35af56721590d7575c248738b1e03150ecfc7ae38085e99ec3` | `f04f96d7a1199ef6febf86ebdc564f1f30f33a2830ccee76123639eed9ba294d` |
| VOCR4 report `part_sha256.a9` | `f9264bc50788e6e6dde7521494585edcde428510f7cba8f6ad94855b81e955d5` | `60a05eb4a42d9d1a44abd3cbd2a47c934b8c625c157c7167ffe5cf0884ecdebb` |
| VOCR4 report `part_sha256.overlay` | `caa9ac37d611d6e99cdc79e56fb935bda101dbaecdbc2e0f7177b5299e96cc88` | `a6d1584027bece5458cf96942389a60751698de004b74bd2795266329024651f` |
| VOCR4 report `part_sha256.shared` | `8d31f70833f342c5e3f4a258939024155ac5ebb6af0c2bfed1dc6e369c6780fe` | `f72bea7b5d765de4606d330e596f2f88594256cebdc67d29743f3e48071bc729` |

A report's `a9` and `overlay` digests are those of its committed parts; `shared` is the digest of the private full
part (the committed file is its `compact` form without per-line records), recomputed with the same label.

The files that changed are those 7 and `eval/synth3_study1b_20261012.json`, `eval/synth3_native_20261013.json`,
`eval/synth3_study2_{FL10_20261012,FL10_test_20261012,FL10_step_test_20261013,H15_20261013,H15_test_20261013,curves_20261013}.json`,
`eval/synth4_ablation_20261013.json` and `eval/vocab_proof_{probe,strata}_20261014.json`. To check by eye:
`git diff d214a05 <scrub commit> --word-diff -- 'eval/*.json'` shows only the host fields and the digests above.

Code-identity records in older results (the sha256 of `eval/run.py`, `eval/vocr3_report.py` or
`train/hiragana_fresh.py` at the time) describe the code their result ran; they do not match the files after the
writers changed, as with any later code change.

**What the scrub does not do.** The old name is still in the repository's history (every commit before the scrub,
the pull request refs) and in any clone, fork or archive made while it was public. Only rewriting the history in a
new repository would remove it from GitHub; that is plan B of the publication audit and needs the owner's explicit
decision.
