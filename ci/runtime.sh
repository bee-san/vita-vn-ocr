#!/usr/bin/env bash
# CI job "runtime (<cc>)": the C runtime built strictly with sanitizers, then its PyTorch parity test.
#
#   ci/runtime.sh gcc|clang          (from anywhere; uses $PYTHON, default: python on PATH)
#
# Until runtime/ exists this prints a notice and exits 0 (the workflow skips the job too).
#
# 1. Checks that ASan and UBSan really trap with this compiler, so green means something.
# 2. Compiles every runtime/*.c on its own with CFLAGS below: C99, -Wall -Wextra -Werror,
#    ASan + UBSan. Each source has to be warning-free under both gcc and clang.
# 3. Makes the parity inputs: a fixed-seed random-init checkpoint (ci/fixture_ckpt.py) and a
#    48-line shard rendered from the pinned OFL font (ci/sandbox.py). Only the font is downloaded.
# 4. Runs the runtime's own check target. This is the contract with runtime/:
#
#      make -C runtime check CC=<cc> CFLAGS=<CFLAGS> LDFLAGS=<LDFLAGS> PYTHON=<python> \
#           VOCR_CKPT=<fixture .pt> VOCR_SHARD=<shard .npz>
#
#    The target builds the runtime and its host test programs with these CC, CFLAGS and
#    LDFLAGS: it may append flags (override CFLAGS += -ffp-contract=off) but must not drop
#    any. Keep -I paths in CPPFLAGS and -lm in LDLIBS so they survive the command line. The
#    six variables are also set in make's environment, so a Python driver run from the
#    recipe can read them. The target exports VOCR_CKPT with train/export.py (fp32 and
#    int8), runs PyTorch and the C runtime on the crops in VOCR_SHARD (render.py's shard
#    format), and exits non-zero when logits differ beyond tolerance or any decoded string
#    differs. Compare the strings against a reference with the same weights (dequantised
#    for int8): this random-init fixture has small top-1/top-2 logit gaps. Run sanitized
#    code as executables, because an ASan build can't be loaded into Python through ctypes
#    without LD_PRELOAD.
set -euo pipefail

cc=${1:?usage: ci/runtime.sh gcc|clang}
cd "$(dirname "$0")/.."
if [ ! -d runtime ]; then
  echo "::notice title=runtime ($cc) skipped::runtime/ does not exist yet, so there is nothing to build or check."
  exit 0
fi

PYTHON=$(command -v "${PYTHON:-python}")
cflags=(-std=c99 -O2 -g -Wall -Wextra -Werror "-fsanitize=address,undefined" -fno-sanitize-recover=all
        -fno-omit-frame-pointer)
CFLAGS="${cflags[*]}"
LDFLAGS="-fsanitize=address,undefined"
export ASAN_OPTIONS=${ASAN_OPTIONS:-detect_leaks=1:abort_on_error=1:halt_on_error=1}
export UBSAN_OPTIONS=${UBSAN_OPTIONS:-print_stacktrace=1:halt_on_error=1}
work=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/vocr-runtime-$cc.XXXXXX")
"$cc" --version | head -n 1

# 1. Each sanitizer must catch a deliberate bug.
cat > "$work/asan.c" <<'EOF'
#include <stdlib.h>
int main(int argc, char **argv) {
    volatile int *p = calloc(4, sizeof *p);
    int v = p[argc + 3];                     /* one past the end */
    (void)argv;
    free((void *)p);
    return v;
}
EOF
cat > "$work/ubsan.c" <<'EOF'
#include <limits.h>
int main(int argc, char **argv) {
    int x = INT_MAX - 1 + argc;
    (void)argv;
    return x + argc;                         /* signed overflow */
}
EOF
for t in address:heap-buffer-overflow undefined:"signed integer overflow"; do
  san=${t%%:*}
  name=$([ "$san" = address ] && echo asan || echo ubsan)
  # Only this sanitizer, so the other one can't report the bug first (UBSan's object-size
  # check catches the out-of-bounds read too).
  "$cc" -std=c99 -O1 -g -fsanitize="$san" -fno-sanitize-recover=all "$work/$name.c" -o "$work/$name"
  if timeout 60 "$work/$name" > "$work/$name.log" 2>&1 || ! grep -q "${t#*:}" "$work/$name.log"; then
    cat "$work/$name.log"
    echo "error: $cc's -fsanitize=$san build did not report a deliberate bug. Missing sanitizer runtime" \
         "(clang: libclang-rt-<N>-dev)? ASan crashing with DEADLYSIGNAL: sudo sysctl vm.mmap_rnd_bits=28" >&2
    exit 1
  fi
  echo "$name traps: $(grep -m 1 "${t#*:}" "$work/$name.log")"
done

# 2. Every runtime source on its own, warning-free C99.
mkdir -p "$work/obj"
for src in runtime/*.c; do
  [ -e "$src" ] || continue
  echo "+ $cc -Iruntime $CFLAGS -c $src"
  "$cc" -Iruntime "${cflags[@]}" -c "$src" -o "$work/obj/$(basename "$src" .c).o"
done

# 3. Parity inputs.
"$PYTHON" ci/fixture_ckpt.py --out "$work/fixture.pt"
"$PYTHON" ci/sandbox.py --out "$work/sandbox" --shards 1 --per-shard 48 --workers 2
shard="$work/sandbox/repo/data/synth/synth_00000.npz"

# 4. The runtime's own build and parity test.
if [ ! -f runtime/Makefile ]; then
  echo "error: runtime/ has no Makefile with a 'check' target (see the contract at the top of ci/runtime.sh)" >&2
  exit 1
fi
vars=(CC="$cc" CFLAGS="$CFLAGS" LDFLAGS="$LDFLAGS" PYTHON="$PYTHON" VOCR_CKPT="$work/fixture.pt" VOCR_SHARD="$shard")
env "${vars[@]}" make -C runtime check "${vars[@]}" || {
  echo "error: 'make -C runtime check' failed with $cc (contract at the top of ci/runtime.sh; inputs kept in $work)" >&2
  exit 1
}
rm -rf "$work"
echo "RUNTIME_CHECK_OK $cc"
