# CI

`.github/workflows/ci.yml` runs on every non-draft pull request (and when a draft is marked ready), every push to `main` and by hand, on `ubuntu-24.04`, with each action pinned to a commit SHA. Python is 3.11.15 with the hash-checked `requirements-ci.lock`: CPU torch 2.5.1 and the same numpy, OpenCV, Pillow, fontTools and PyArrow versions as `requirements.lock`, plus pytest and ruff. Nothing copyrighted is downloaded. The one data download is the OFL font M+ 1p from google/fonts at a pinned commit, checked against its sha256.

| Job | Checks | Local command |
|---|---|---|
| ruff + policy | lint (`ruff.toml`: E, F, W, I), the repo policy tests (`tests/test_repo_policy.py`: data policy on every tracked file, action pinning, `ubuntu-24.04`), and `ci/changed.py`, which decides which jobs below run | `.venv-ci/bin/ruff check . && .venv-ci/bin/python -m pytest tests/test_repo_policy.py` |
| pytest | `tests/`: vocab, model shape and op set, `train.py` helpers, renderer smoke, CI fixtures, repo data policy, action pinning | `.venv-ci/bin/python -m pytest` |
| training smoke (50 steps) | renders 2 × 160 lines from the pinned font and `ci/fixtures/corpus`, trains exactly 50 CTC steps with `train/train.py`, then checks the checkpoint, the metrics and that the loss fell | `.venv-ci/bin/python ci/train_smoke.py --threads 2` |
| runtime (gcc), runtime (clang) | `runtime/` as C99 with `-Wall -Wextra -Werror` plus ASan and UBSan, then its parity test against PyTorch. Skipped until `runtime/` exists | `PYTHON=.venv-ci/bin/python ci/runtime.sh gcc` |
| vita (VitaSDK build) | `runtime/` and the `vita/` test app built for the PS Vita with the pinned VitaSDK and `-Werror`; uploads `vocr_test.vpk` and the code-size report as the `vocr_test-vpk` artifact, kept 7 days for a pull request and 30 days for `main` and manual runs | `vita/scripts/fetch_vitasdk.sh && VITASDK=$PWD/vita/tools/vitasdk vita/scripts/build.sh` |

## When each job runs

This is a private repository, so every job minute counts against the GitHub Actions quota, and each job is billed rounded up to the minute. `ruff + policy` runs on every event. `ci/changed.py` (tested by `tests/test_ci_changed.py`) diffs the change against the base branch (pull request) or the previous tip (push to `main`) and gates the rest on the paths it touched; a manual run, or any diff that cannot be computed, runs everything:

| Changed paths | pytest | training smoke | runtime | vita |
|---|---|---|---|---|
| only `*.md`, `docs/`, `notes/`, `LICENSE`, `.gitignore` | – | – | – | – |
| anything else (`tests/`, `eval/`, `realplus/`, ...) | ✓ | – | – | – |
| `train/`, `synth/`, `data/`, `ci/`, `requirements-ci.*`, `.github/` | ✓ | ✓ | ✓ gcc on a pull request, gcc + clang on `main` | – |
| `runtime/` | ✓ | – | ✓ gcc + clang | ✓ |
| `vita/` | ✓ | – | – | ✓ |

Skipped jobs show as skipped, not failed, so `gh pr checks` still answers "is CI green". When a job starts reading a new directory, add it to the matching set in `ci/changed.py` and to the table above.

Make the env once (a separate `.venv-ci`, because `uv pip sync` removes packages that aren't in the lock):

```sh
uv venv --python 3.11.15 .venv-ci
UV_INDEX_STRATEGY=unsafe-best-match uv pip sync --python .venv-ci/bin/python --require-hashes requirements-ci.lock
```

`ci/sandbox.py` runs `render.py` and `train.py` in a throwaway copy of the repo whose `data/` holds only the fixtures. The checkout's own `data/`, which may link to shared data, is never written.

## The runtime contract

Once `runtime/` exists, `ci/runtime.sh <cc>` checks that ASan and UBSan trap, then compiles each `runtime/*.c` on its own. If `runtime/Makefile` exists, it writes a fixed-seed random-init checkpoint (`ci/fixture_ckpt.py`, with random BatchNorm so that folding is exercised) and a 48-line shard, and runs:

```sh
make -C runtime check CC=<cc> CFLAGS=<strict + sanitizers> LDFLAGS=<sanitizers> PYTHON=<python> \
     VOCR_CKPT=<fixture .pt> VOCR_SHARD=<shard .npz> VOCR_REPORT=<report .json>
```

Once the recognizer is there (`runtime/vocr.h`), `check` must run the parity test, and the parity programs must be built with `$(CC) $(CFLAGS) … $(LDFLAGS)`. They may add flags, for example `override CFLAGS += -ffp-contract=off`, but must not drop any; put `-I` in `CPPFLAGS` and `-lm` in `LDLIBS`. The test exports `VOCR_CKPT` with `train/export.py` (fp32 and int8) and fails if the logits leave tolerance or a decoded string differs from a PyTorch reference with the same weights. `runtime/tests/parity.py` lets a string differ only on a near tie: every differing frame's top-2 margin must be within twice the measured logit difference, and the logits must still be bit-identical to its numpy model of the C arithmetic. It writes `{"ckpt": VOCR_CKPT, "shard": VOCR_SHARD, "lines": <crops compared>, "pass": true, …}` to `VOCR_REPORT`, where `lines` must be every crop in the shard, and `ci/parity_report.py` checks that report and adds it to the job summary.

Other C code that arrives in `runtime/` before the recognizer, such as the line finder, is only compiled strictly. Nothing of it runs unless it brings a Makefile whose `check` target runs its tests, and ASan and UBSan only check code that runs. The full text is at the top of `ci/runtime.sh`. The workflow sets `vm.mmap_rnd_bits=28` (ASan needs it on ubuntu-24.04 kernels) and installs clang's sanitizer runtime if it's missing. Do the same locally if the sanitizer self-test fails.

## Updating pins

- Python: edit `requirements-ci.in`, then regenerate the lock with the command in its header.
- Font: `COMMIT` and `FILES` (sha256 and size) in `ci/fonts.py`.
- Actions: a full SHA plus `# vX.Y.Z`. `tests/test_repo_policy.py` rejects anything else, and also any tracked font, image, shard, checkpoint, console binary or file over 2 MiB. A justified exception goes in its `ALLOWED` set, with the reason given in the PR.
