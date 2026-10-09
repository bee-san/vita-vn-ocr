#!/usr/bin/env bash
# CI job "runtime (<cc>)": runtime/ built strictly with sanitizers, then its PyTorch parity test.
#
#   ci/runtime.sh gcc|clang          (from anywhere; uses $PYTHON, default: python on PATH;
#                                     a relative $PYTHON is taken from the current directory)
#
# Until runtime/ exists this prints a notice and exits 0 (the workflow skips the job too).
#
# 1. Checks that ASan and UBSan really trap with this compiler, so green means something.
# 2. Compiles every runtime/*.c on its own as C99 with -Wall -Wextra -Werror, ASan and UBSan.
#    Each source has to be warning-free under both gcc and clang. Nothing is run in this step.
# 3. If runtime/Makefile exists: writes the parity inputs and runs its check target.
#    The inputs are a fixed-seed random-init checkpoint (ci/fixture_ckpt.py) and a 48-line
#    shard rendered from the pinned OFL font (ci/sandbox.py); only the font is downloaded.
#
#      make -C runtime check CC=<cc> CFLAGS=<strict + sanitizers> LDFLAGS=<sanitizers> \
#           PYTHON=<python> VOCR_CKPT=<fixture .pt> VOCR_SHARD=<shard .npz> VOCR_REPORT=<.json>
#
#    The same variables are also set in make's environment, for a Python driver.
# 4. Once the recognizer exists (runtime/vocr.h), checks the parity report (ci/parity_report.py).
#
# The contract with runtime/, once runtime/vocr.h is there:
#  - runtime/Makefile has a check target that runs the C tests and the parity test.
#  - The parity programs are built with $(CC) $(CFLAGS) ... $(LDFLAGS). Flags may be added
#    (override CFLAGS += -ffp-contract=off) but not dropped, so the parity run is sanitized
#    under each compiler. Keep -I paths in CPPFLAGS and -lm in LDLIBS. Run sanitized code as
#    executables: an ASan build can't be loaded into Python through ctypes without LD_PRELOAD.
#  - The parity test exports VOCR_CKPT with train/export.py (fp32 and int8), runs PyTorch and
#    the C runtime on every crop in VOCR_SHARD (render.py's shard format), fails when the
#    logits differ beyond tolerance or any decoded string differs (runtime/tests/parity.py:
#    except on a near tie, where every differing frame's top-2 margin is within twice the
#    measured logit difference; its logits must still equal a bit-exact model of the C
#    arithmetic), and writes one JSON object
#    to VOCR_REPORT: {"ckpt": VOCR_CKPT, "shard": VOCR_SHARD, "lines": <crops compared, which
#    must be every crop in the shard>, "pass": true, plus anything else, e.g. max logit
#    difference per mode}. Compare strings against a reference with the same weights
#    (dequantised for int8): this random-init fixture has small top-1/top-2 logit gaps.
# Other C code in runtime/ before the recognizer lands (the line finder, say) gets steps 1-2,
# and step 3 if it brings a Makefile; it needs no parity report. Without a Makefile nothing of
# it is run, so the sanitizers check nothing: a check target is how its tests run under them.
set -euo pipefail

cc=${1:?usage: ci/runtime.sh gcc|clang}
# make -C runtime runs $PYTHON from runtime/, so a relative one (PYTHON=.venv-ci/bin/python)
# is made absolute here, against the caller's directory.
py=${PYTHON:-python}
PYTHON=$(command -v "$py" || true)
case $PYTHON in "" | /*) ;; *) PYTHON=$PWD/$PYTHON ;; esac
cd "$(dirname "$0")/.."
if [ ! -d runtime ]; then
  echo "::notice title=runtime ($cc) skipped::runtime/ does not exist yet, so there is nothing to build or check."
  exit 0
fi

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
  # abort_on_error=0: the expected report ends in exit(1), not SIGABRT, so bash logs no "Aborted".
  if ASAN_OPTIONS=detect_leaks=0:abort_on_error=0 timeout 60 "$work/$name" > "$work/$name.log" 2>&1 ||
     ! grep -q "${t#*:}" "$work/$name.log"; then
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

# 3. The runtime's own build and tests.
if [ ! -f runtime/Makefile ]; then
  if [ -f runtime/vocr.h ]; then
    echo "error: runtime/vocr.h is here but runtime/Makefile is not; the recognizer needs a 'check'" \
         "target that runs its parity test (contract at the top of ci/runtime.sh)" >&2
    exit 1
  fi
  echo "::notice title=runtime ($cc): compiled only, nothing run::runtime/ has no recognizer (runtime/vocr.h) and no Makefile, so its sources were compiled warning-free with the sanitizer flags but nothing was run, and ASan and UBSan only check code that runs. A 'check' target in runtime/Makefile runs tests under them (ci/runtime.sh)."
  rm -rf "$work"
  exit 0
fi
[ -n "$PYTHON" ] || { echo "error: no Python at PYTHON=$py; set PYTHON to the CI env's interpreter (ci/README.md)" >&2; exit 1; }
"$PYTHON" ci/fixture_ckpt.py --out "$work/fixture.pt"
"$PYTHON" ci/sandbox.py --out "$work/sandbox" --shards 1 --per-shard 48 --workers 2
shard="$work/sandbox/repo/data/synth/synth_00000.npz"
report="$work/parity.json"
vars=(CC="$cc" CFLAGS="$CFLAGS" LDFLAGS="$LDFLAGS" PYTHON="$PYTHON"
      VOCR_CKPT="$work/fixture.pt" VOCR_SHARD="$shard" VOCR_REPORT="$report")
env "${vars[@]}" make -C runtime check "${vars[@]}" || {
  echo "error: 'make -C runtime check' failed with $cc (contract at the top of ci/runtime.sh; inputs kept in $work)" >&2
  exit 1
}

# 4. Evidence that the recognizer's parity test ran on CI's fixture.
if [ -f runtime/vocr.h ]; then
  "$PYTHON" ci/parity_report.py "$report" --ckpt "$work/fixture.pt" --shard "$shard"
fi
rm -rf "$work"
echo "RUNTIME_CHECK_OK $cc"
