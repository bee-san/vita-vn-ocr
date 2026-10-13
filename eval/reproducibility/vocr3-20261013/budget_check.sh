#!/usr/bin/env bash
# check.sh OUTDIR BLOB...: vita/budget_sweep.py check (docs/vita-memory-budget.md) from a main worktree; synth-val lines only.
set -uo pipefail
HERD=/local/home/skerraut/herd-vita-ocr; A=$HERD/shared/artifacts; export HERD
export VITASDK=$A/vita/vitasdk VOCR_SYNTH_VAL=$HERD/shared/data/synth/synth_00039.npz
CC=$A/runtime/tools/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-linux-gnueabihf/bin/arm-none-linux-gnueabihf-gcc
OUT=$1; shift
cd /local/home/skerraut/herd-vita-oct12/clones/main-wt
echo "git $(git rev-parse HEAD)"
.venv-ci/bin/python vita/budget_sweep.py check "$@" --out "$OUT" --qemu $A/vita/qemu --arm-cc "$CC" --vitasdk "$VITASDK"
echo "rc=$?"
