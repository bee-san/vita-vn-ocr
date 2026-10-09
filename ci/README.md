# CI

`.github/workflows/ci.yml` runs on every pull request and every push to `main`, on `ubuntu-24.04`, with each action pinned to a commit SHA. Python is 3.11.15 with the hash-checked `requirements-ci.lock`: CPU torch 2.5.1 and the same numpy, OpenCV, Pillow and fontTools versions as `requirements.lock`, plus pytest and ruff. Nothing copyrighted is downloaded. The one download is the OFL font M+ 1p from google/fonts at a pinned commit, checked against its sha256.

| Job | Checks | Local command |
|---|---|---|
| ruff | lint (`ruff.toml`: E, F, W, I) | `.venv-ci/bin/ruff check .` |
| pytest | `tests/`: vocab, model shape and op set, `train.py` helpers, renderer smoke, CI fixtures, repo data policy, action pinning | `.venv-ci/bin/python -m pytest` |
| training smoke (50 steps) | renders 2 × 160 lines from the pinned font and `ci/fixtures/corpus`, trains exactly 50 CTC steps with `train/train.py`, then checks the checkpoint, the metrics and that the loss fell | `.venv-ci/bin/python ci/train_smoke.py --threads 2` |
| runtime (gcc), runtime (clang) | `runtime/` as C99 with `-Wall -Wextra -Werror` plus ASan and UBSan, then its parity test against PyTorch. Skipped until `runtime/` exists | `PYTHON=.venv-ci/bin/python ci/runtime.sh gcc` |

Make the env once (a separate `.venv-ci`, because `uv pip sync` removes packages that aren't in the lock):

```sh
uv venv --python 3.11.15 .venv-ci
UV_INDEX_STRATEGY=unsafe-best-match uv pip sync --python .venv-ci/bin/python --require-hashes requirements-ci.lock
```

`ci/sandbox.py` runs `render.py` and `train.py` in a throwaway copy of the repo whose `data/` holds only the fixtures. The checkout's own `data/`, which may link to shared data, is never written.

## The runtime contract

Once `runtime/` exists, `ci/runtime.sh <cc>` checks that ASan and UBSan trap, compiles each `runtime/*.c` on its own, writes a fixed-seed random-init checkpoint (`ci/fixture_ckpt.py`, with random BatchNorm so that folding is exercised) and a 48-line shard, and then runs:

```sh
make -C runtime check CC=<cc> CFLAGS=<strict + sanitizers> LDFLAGS=<sanitizers> PYTHON=<python> \
     VOCR_CKPT=<fixture .pt> VOCR_SHARD=<shard .npz>
```

The `check` target has to build with those `CC`/`CFLAGS`/`LDFLAGS`. It may append flags (for example `override CFLAGS += -ffp-contract=off`) but must not drop any; put `-I` in `CPPFLAGS` and `-lm` in `LDLIBS`. It also has to export `VOCR_CKPT` with `train/export.py` (fp32 and int8), and fail if the logits leave tolerance or a decoded string differs from a PyTorch reference with the same weights. The full text is at the top of `ci/runtime.sh`. The workflow sets `vm.mmap_rnd_bits=28` (ASan needs it on ubuntu-24.04 kernels) and installs clang's sanitizer runtime if it's missing. Do the same locally if the sanitizer self-test fails.

## Updating pins

- Python: edit `requirements-ci.in`, then regenerate the lock with the command in its header.
- Font: `COMMIT` and `FILES` (sha256 and size) in `ci/fonts.py`.
- Actions: a full SHA plus `# vX.Y.Z`. `tests/test_repo_policy.py` rejects anything else, and also any tracked font, image, shard, checkpoint, console binary or file over 2 MiB. A justified exception goes in its `ALLOWED` set, with the reason given in the PR.
