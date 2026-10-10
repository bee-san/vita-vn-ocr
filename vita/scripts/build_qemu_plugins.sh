#!/usr/bin/env bash
# Build qemu-arm (linux-user) with TCG plugins in user space, for the Cortex-A9 cycle
# estimate (vita/a9_cycles.py): the instruction-mix plugin vita/qemu/insn_mix.c and qemu's
# own cache simulator (contrib/plugins/cache.c) need a plugin-enabled, dynamically linked
# qemu, which the static Debian qemu-arm of runtime/scripts/fetch_arm_tools.sh is not.
#
#   vita/scripts/build_qemu_plugins.sh [DEST]   default DEST: vita/tools (gitignored)
#
# Source: qemu 10.0.13 from download.qemu.org (the version of runtime's qemu-arm), sha256-
# checked. Needs a C compiler, glib-2.0 headers, ninja, and a Python with tomllib (3.11+;
# set QEMU_PYTHON, default python3) and venv.
# Run it through $HERD/heavy.sh on the shared box: it compiles with $OMP_NUM_THREADS jobs.
# Result: DEST/qemu/bin/qemu-arm, DEST/qemu/plugins/{libcache.so,libinsn_mix.so}.
set -euo pipefail
cd "$(dirname "$0")/.."
VITA=$PWD
DEST=${1:-tools}
mkdir -p "$DEST/dl"
DEST=$(cd "$DEST" && pwd)
VER=10.0.13
SHA=4a59b74dac7ead8b162a7c4595dcbbeebe989b90c89cfe6bb4dec5447342f6fa
URL=https://download.qemu.org/qemu-$VER.tar.xz
UA="vita-vn-ocr-vita/1 (+https://github.com/bee-san/vita-vn-ocr)"
JOBS=${OMP_NUM_THREADS:-4}

f="$DEST/dl/qemu-$VER.tar.xz"
if [ ! -f "$f" ] || ! echo "$SHA  $f" | sha256sum -c --status; then
  curl -sSfL -A "$UA" --retry 3 -o "$f.part" "$URL"
  mv "$f.part" "$f"
fi
echo "$SHA  $f" | sha256sum -c
src="$DEST/qemu-src/qemu-$VER"
if [ ! -f "$src/configure" ]; then
  rm -rf "$DEST/qemu-src"
  mkdir -p "$DEST/qemu-src"
  tar -C "$DEST/qemu-src" -xJf "$f"
fi
mkdir -p "$src/build"
cd "$src/build"
if [ ! -f build.ninja ]; then
  ../configure --python="${QEMU_PYTHON:-python3}" --prefix="$DEST/qemu" --target-list=arm-linux-user --enable-plugins \
    --disable-docs --disable-tools --disable-werror --disable-debug-info --without-default-features \
    --enable-tcg
fi
ninja -j "$JOBS" qemu-arm contrib/plugins/libcache.so
mkdir -p "$DEST/qemu/bin" "$DEST/qemu/plugins"
cp qemu-arm "$DEST/qemu/bin/qemu-arm"
cp contrib/plugins/libcache.so "$DEST/qemu/plugins/"
# the instruction-mix plugin: one shared object against qemu's plugin header
cc -O2 -shared -fPIC -Wall -Wextra -Werror -I"$src/include/qemu" $(pkg-config --cflags glib-2.0) \
   -o "$DEST/qemu/plugins/libinsn_mix.so" "$VITA/qemu/insn_mix.c" $(pkg-config --libs glib-2.0)
"$DEST/qemu/bin/qemu-arm" --version | head -n 1
echo "QEMU=$DEST/qemu/bin/qemu-arm PLUGINS=$DEST/qemu/plugins"
