#!/usr/bin/env bash
# Build PPSSPPHeadless (the PSP emulator's test runner) in user space, at pinned releases:
#   PPSSPP 1.20.4 (release source tarball, submodules included), sha256-checked
#   SDL2 2.32.10 (only because PPSSPP's Linux CMake links its `native` library against it),
#     built shared, without X11/Wayland/audio back ends, sha256-checked
#
#   psp/scripts/build_ppsspp.sh [DEST]     # default DEST: psp/tools; run it through heavy.sh
#   -> DEST/ppsspp/PPSSPPHeadless (+ DEST/ppsspp/assets, which it looks for next to itself)
#
# FFmpeg is off (the source tarball has no prebuilt FFmpeg, and the test app plays no media),
# so are Discord and miniupnpc; GLEW_NO_GLU drops glew's GLU include, which PPSSPP never uses,
# and the X11/Wayland Vulkan surfaces are off (this SDL2 has no window back ends).
# Needs cmake >= 3.16, ninja, a C++17 compiler, and the GL/EGL, zlib and libpng headers
# (present on Amazon Linux 2023 and ubuntu-24.04).
# Threads: $OMP_NUM_THREADS (heavy.sh sets 5), else 4.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p "${1:-tools}/dl" "${1:-tools}/src" "${1:-tools}/build"
DEST=$(cd "${1:-tools}" && pwd)
JOBS=${OMP_NUM_THREADS:-4}

PPSSPP_VER=1.20.4
PPSSPP_FILE=ppsspp-$PPSSPP_VER.tar.xz
PPSSPP_URL=https://github.com/hrydgard/ppsspp/releases/download/v$PPSSPP_VER/$PPSSPP_FILE
PPSSPP_SHA=4de21db3105d9d81a465a8a7e78c68ee3c0e0bf6597d1c1d530f7555f3ad8b31
SDL_VER=2.32.10
SDL_FILE=SDL2-$SDL_VER.tar.gz
SDL_URL=https://github.com/libsdl-org/SDL/releases/download/release-$SDL_VER/$SDL_FILE
SDL_SHA=5f5993c530f084535c65a6879e9b26ad441169b3e25d789d83287040a9ca5165
UA="vita-vn-ocr-psp-tools/1 (+https://github.com/bee-san/vita-vn-ocr)"

fetch() { # url file sha256
  if [ ! -f "$2" ] || ! echo "$3  $2" | sha256sum -c --status; then
    curl -sSfL -A "$UA" --retry 3 -o "$2.part" "$1"
    mv "$2.part" "$2"
  fi
  echo "$3  $2" | sha256sum -c
}

# ---- SDL2 (shared, no video/audio back ends; PPSSPPHeadless never opens a window)
if [ ! -f "$DEST/sdl2/lib/libSDL2.so" ]; then
  fetch "$SDL_URL" "$DEST/dl/$SDL_FILE" "$SDL_SHA"
  rm -rf "$DEST/src/SDL2-$SDL_VER" "$DEST/build/sdl2"
  tar -C "$DEST/src" -xzf "$DEST/dl/$SDL_FILE"
  cmake -S "$DEST/src/SDL2-$SDL_VER" -B "$DEST/build/sdl2" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$DEST/sdl2" -DSDL_SHARED=ON -DSDL_STATIC=OFF -DSDL_TEST=OFF \
    -DSDL_X11=OFF -DSDL_WAYLAND=OFF -DSDL_KMSDRM=OFF -DSDL_VULKAN=OFF -DSDL_OPENGL=ON -DSDL_OPENGLES=OFF \
    -DSDL_PULSEAUDIO=OFF -DSDL_PIPEWIRE=OFF -DSDL_ALSA=OFF -DSDL_JACK=OFF -DSDL_SNDIO=OFF \
    -DSDL_IBUS=OFF -DSDL_DBUS=OFF -DSDL_HIDAPI=OFF -DSDL_LIBUDEV=OFF > "$DEST/build/sdl2-configure.log"
  cmake --build "$DEST/build/sdl2" -j "$JOBS" > "$DEST/build/sdl2-build.log"
  cmake --install "$DEST/build/sdl2" > /dev/null
  rm -rf "$DEST/src/SDL2-$SDL_VER" "$DEST/build/sdl2" "$DEST/dl/$SDL_FILE"
fi

# ---- PPSSPPHeadless
if [ ! -x "$DEST/ppsspp/PPSSPPHeadless" ]; then
  fetch "$PPSSPP_URL" "$DEST/dl/$PPSSPP_FILE" "$PPSSPP_SHA"
  S=$DEST/src/ppsspp-$PPSSPP_VER
  B=$DEST/build/ppsspp-$PPSSPP_VER
  [ -d "$S" ] || tar -C "$DEST/src" -xJf "$DEST/dl/$PPSSPP_FILE"
  cmake -S "$S" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DHEADLESS=ON -DUSE_FFMPEG=OFF \
    -DUSE_DISCORD=OFF -DUSE_MINIUPNPC=OFF -DUSE_CCACHE=OFF -DCMAKE_PREFIX_PATH="$DEST/sdl2" \
    -DUSING_X11_VULKAN=OFF -DUSE_WAYLAND_WSI=OFF \
    -DCMAKE_C_FLAGS=-DGLEW_NO_GLU -DCMAKE_CXX_FLAGS=-DGLEW_NO_GLU \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -Wno-dev > "$DEST/build/ppsspp-configure.log"
  cmake --build "$B" -j "$JOBS" --target PPSSPPHeadless > "$DEST/build/ppsspp-build.log"
  rm -rf "$DEST/ppsspp" && mkdir -p "$DEST/ppsspp"
  cp "$B/PPSSPPHeadless" "$DEST/ppsspp/"
  cp -r "$S/assets" "$DEST/ppsspp/assets"
  echo "PPSSPP $PPSSPP_VER $PPSSPP_SHA; SDL2 $SDL_VER $SDL_SHA" > "$DEST/ppsspp/.release"
  rm -rf "$B" "$S" "$DEST/dl/$PPSSPP_FILE"
fi
"$DEST/ppsspp/PPSSPPHeadless" --help 2>&1 | head -n 3 || true
echo "PPSSPP=$DEST/ppsspp/PPSSPPHeadless"
