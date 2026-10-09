#!/usr/bin/env bash
# Install the Vita3K emulator in user space (no root, no FUSE), sha256-pinned, for
# vita/scripts/run_vita3k.sh.
#
#   vita/scripts/fetch_vita3k.sh [DEST]     default DEST: vita/tools (gitignored)
#
# The pin is Vita3K build 4154 (v0.2.1, commit b75d5e6292d5caa6a8411bacc9db1eca1531beab,
# 2026-10-09), checked by the AppImage's sha256. Vita3K/Vita3K-builds keeps a release per
# build, so the pinned file stays downloadable; the rolling "continuous" release of
# Vita3K/Vita3K serves the same file only until upstream moves on, and is tried second.
# VITA3K_APPIMAGE=<file> uses a kept copy instead (one is in $HERD/shared/artifacts/vita/dl/).
#
# The binary needs glibc >= 2.43 and libwayland-client >= 1.24, newer than many hosts
# (Amazon Linux 2023 has 2.34), so Debian's libc6 2.43-7 and libwayland-client0 1.26.0-1 are
# unpacked beside it and run_vita3k.sh starts Vita3K through that glibc's loader. Both
# .debs are sha256-checked, from deb.debian.org with snapshot.debian.org as the fallback.
# Result: DEST/vita3k/4154/{app (the extracted AppImage), compat (the libraries)}.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${1:-tools}
mkdir -p "$DEST/dl"
DEST=$(cd "$DEST" && pwd)
BUILD=4154
APPIMAGE_SHA=b42ba0296458831018837e7fcc1a12d216d7a1190fbc79be3c3f97baf884e57a
APPIMAGE_URL=https://github.com/Vita3K/Vita3K-builds/releases/download/$BUILD/Vita3K-x86_64.AppImage
APPIMAGE_URL2=https://github.com/Vita3K/Vita3K/releases/download/continuous/Vita3K-x86_64.AppImage
LIBC=libc6_2.43-7_amd64.deb
LIBC_SHA=5c71715c51103beb4fbba9fe9f8dff604af6610dfa71c70bff342f54a7860803
LIBC_SHA1=d3b47ff25250d1e0065ca0b39417fc55b4e8e341
WAYLAND=libwayland-client0_1.26.0-1_amd64.deb
WAYLAND_SHA=b432b1d30319f371efe776209e36143a23289285dd26d41751e89ad4a88b2c33
WAYLAND_SHA1=aabcdf08f5a7ae452accc846eb443649f99ea1f7
UA="vita-vn-ocr-vita/1 (+https://github.com/bee-san/vita-vn-ocr)"
V="$DEST/vita3k/$BUILD"

fetch() { # file sha256 url...
  local f=$1 sha=$2 url
  shift 2
  if [ -f "$f" ] && echo "$sha  $f" | sha256sum -c --status; then return 0; fi
  for url in "$@"; do
    if curl -sSfL -A "$UA" --retry 3 -o "$f.part" "$url" && echo "$sha  $f.part" | sha256sum -c --status; then
      mv "$f.part" "$f"
      return 0
    fi
    rm -f "$f.part"
  done
  echo "error: no source gave $(basename "$f") with sha256 $sha" >&2
  return 1
}

if [ ! -x "$V/app/usr/bin/Vita3K" ] || [ ! -f "$V/compat/libc.so.6" ]; then
  img="$DEST/dl/vita3k-$BUILD-x86_64.AppImage"
  if [ -n "${VITA3K_APPIMAGE:-}" ]; then
    cp "$VITA3K_APPIMAGE" "$img"
  fi
  if ! fetch "$img" "$APPIMAGE_SHA" "$APPIMAGE_URL" "$APPIMAGE_URL2"; then
    echo "no download gave Vita3K build $BUILD; set VITA3K_APPIMAGE to a copy of it" >&2
    exit 1
  fi
  fetch "$DEST/dl/$LIBC" "$LIBC_SHA" "https://deb.debian.org/debian/pool/main/g/glibc/$LIBC" \
        "https://snapshot.debian.org/file/$LIBC_SHA1"
  fetch "$DEST/dl/$WAYLAND" "$WAYLAND_SHA" "https://deb.debian.org/debian/pool/main/w/wayland/$WAYLAND" \
        "https://snapshot.debian.org/file/$WAYLAND_SHA1"
  rm -rf "$V"
  mkdir -p "$V/compat"
  x=$(mktemp -d)
  chmod +x "$img"
  (cd "$x" && "$img" --appimage-extract > /dev/null)   # the AppImage runtime unpacks itself, no FUSE
  mv "$x/squashfs-root" "$V/app"
  for deb in "$DEST/dl/$LIBC" "$DEST/dl/$WAYLAND"; do
    rm -rf "$x/deb" && mkdir -p "$x/deb"
    (cd "$x/deb" && ar x "$deb" && tar -xf data.tar.*)
    cp -a "$x/deb/usr/lib/x86_64-linux-gnu/." "$V/compat/"
  done
  rm -rf "$x"
  # Vita3K finds its assets from its own path (APPDIR/usr/share/Vita3K), so the loader that
  # starts it must sit beside it as a real file, not a symlink
  cp "$V/compat/ld-linux-x86-64.so.2" "$V/app/usr/bin/ld-linux-x86-64.so.2"
fi
h=$(mktemp -d)   # Vita3K writes a config and a log even for --version: keep them out of $HOME
env HOME="$h" XDG_CONFIG_HOME="$h/config" XDG_DATA_HOME="$h/data" XDG_CACHE_HOME="$h/cache" APPDIR="$V/app" \
  "$V/app/usr/bin/ld-linux-x86-64.so.2" --library-path "$V/compat:$V/app/usr/lib:/usr/lib64:/usr/lib/x86_64-linux-gnu" \
  "$V/app/usr/bin/Vita3K" --version < /dev/null 2>/dev/null | tail -n 1
rm -rf "$h"
echo "VITA3K=$V"
