#!/usr/bin/env bash
# Build runtime/ and the vocr_test app for the PS Vita with VitaSDK.
#
#   VITASDK=... vita/scripts/build.sh [BUILD_DIR]     default BUILD_DIR: build/vita
#
# Writes BUILD_DIR/vocr_test.vpk (install it with VitaShell), BUILD_DIR/eboot.bin and
# BUILD_DIR/libvocr.a, then BUILD_DIR/size.txt: code size of vocr.o (NEON and scalar) and its
# largest stack frames, from runtime/Makefile's `size` target with the Vita compiler.
set -euo pipefail
vita=$(cd "$(dirname "$0")/.." && pwd)
root=$(dirname "$vita")
: "${VITASDK:?set VITASDK; vita/scripts/fetch_vitasdk.sh installs the pinned SDK and prints the export line}"
export PATH="$VITASDK/bin:$PATH"
build=${1:-$root/build/vita}
mkdir -p "$build"
build=$(cd "$build" && pwd)
"$VITASDK/bin/arm-vita-eabi-gcc" --version | head -n 1
cmake -S "$vita" -B "$build" -DCMAKE_TOOLCHAIN_FILE="$VITASDK/share/vita.toolchain.cmake" > "$build/configure.log"
cmake --build "$build" -j "${OMP_NUM_THREADS:-4}"
make -s -C "$root/runtime" size ARM_CC="$VITASDK/bin/arm-vita-eabi-gcc" OUT="$build/runtime-size" > "$build/size.txt"
cat "$build/size.txt"
ls -l "$build/vocr_test.vpk" "$build/eboot.bin" "$build/libvocr.a"
sha256sum "$build/vocr_test.vpk"
