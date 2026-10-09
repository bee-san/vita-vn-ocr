#!/usr/bin/env bash
# Build eval/ncnn/build/vocr_ncnn_rec against the ncnn commit vita-jp-overlay pins
# (cmake/LocalOCR.cmake: 3798be64), CPU only, int8 on, no OpenMP/Vulkan.
#   eval/ncnn/build.sh            (CPU-heavy: run it through $HERD/heavy.sh on a shared box)
# NCNN_SRC=/path/to/ncnn-src skips the download; BUILD_DIR overrides eval/ncnn/build.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${BUILD_DIR:-$HERE/build}"
COMMIT=3798be64cfeeebc0afcfcebfb43ed5c9e1b72e82
SHA256=276ed0523e7a80048957c2bff89f37242cadd8858ed026a5538222af338e9fde
JOBS="${OMP_NUM_THREADS:-4}"
mkdir -p "$BUILD"
if [ -z "${NCNN_SRC:-}" ]; then
  NCNN_SRC="$BUILD/ncnn-$COMMIT"
  if [ ! -f "$NCNN_SRC/CMakeLists.txt" ]; then
    tgz="$BUILD/ncnn-$COMMIT.tar.gz"
    [ -f "$tgz" ] || curl -fsSL -A "vita-vn-ocr-eval/0.1" -o "$tgz" \
      "https://github.com/Tencent/ncnn/archive/$COMMIT.tar.gz"
    echo "$SHA256  $tgz" | sha256sum -c -
    tar -xzf "$tgz" -C "$BUILD"
  fi
fi
cmake -S "$NCNN_SRC" -B "$BUILD/ncnn-build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_VULKAN=OFF -DNCNN_OPENMP=OFF -DNCNN_THREADS=OFF -DNCNN_INT8=ON -DNCNN_BF16=OFF \
  -DNCNN_SHARED_LIB=OFF -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF -DNCNN_PYTHON=OFF -DNCNN_SIMPLEOCV=OFF \
  -DNCNN_PIXEL=ON > "$BUILD/cmake.log"
cmake --build "$BUILD/ncnn-build" -j "$JOBS" > "$BUILD/build.log"
g++ -O2 -std=c++11 -Wall -Wextra -o "$BUILD/vocr_ncnn_rec" "$HERE/vocr_ncnn_rec.cpp" \
  -I"$NCNN_SRC/src" -I"$BUILD/ncnn-build/src" "$BUILD/ncnn-build/src/libncnn.a"
echo "built $BUILD/vocr_ncnn_rec (ncnn $COMMIT)"
