#!/usr/bin/env bash
# Install the pinned VitaSDK in user space (no root), sha256-checked.
#
#   vita/scripts/fetch_vitasdk.sh [DEST]     default DEST: vita/tools (gitignored)
#
# Installs DEST/vitasdk (arm-vita-eabi gcc, binutils, newlib, vita-headers stubs,
# vita-elf-create, vita-make-fself, vita-mksfoex, vita-pack-vpk, share/vita.cmake) from
# vitasdk/autobuilds release sdk-snapshot-20260926.790.1 (SDK version 0.20260926.321,
# buildscripts 3c0b9c7b, built on ubuntu:20.04, so glibc >= 2.31). Then:
#   export VITASDK=DEST/vitasdk PATH=DEST/vitasdk/bin:$PATH
# Nothing here is committed or shipped: the SDK only builds vita/ (vita/README.md).
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${1:-tools}
mkdir -p "$DEST/dl"
DEST=$(cd "$DEST" && pwd)

TAG=sdk-snapshot-20260926.790.1
TARBALL=vitasdk-x86_64-linux-gnu-2026-09-26_08-31-53.tar.bz2
SHA=87b111ff350c1f36bde1c2923e57467c1f86b044d54bc616631260486f2a512c
URL=https://github.com/vitasdk/autobuilds/releases/download/$TAG/$TARBALL
UA="vita-vn-ocr-vita/1 (+https://github.com/bee-san/vita-vn-ocr)"
PIN="$TAG $TARBALL $SHA"

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) ;;
  *) echo "error: this script pins the x86_64-linux-gnu SDK; other hosts: $URL's release page" >&2; exit 1 ;;
esac

if [ -x "$DEST/vitasdk/bin/arm-vita-eabi-gcc" ] && [ "$(cat "$DEST/vitasdk/.vocr-pin" 2>/dev/null)" = "$PIN" ]; then
  echo "VitaSDK $TAG already installed in $DEST/vitasdk"
else
  f="$DEST/dl/$TARBALL"
  if [ ! -f "$f" ] || ! echo "$SHA  $f" | sha256sum -c --status; then
    curl -sSfL -A "$UA" --retry 3 -o "$f.part" "$URL"
    mv "$f.part" "$f"
  fi
  echo "$SHA  $f" | sha256sum -c
  rm -rf "$DEST/vitasdk.tmp" "$DEST/vitasdk"
  mkdir -p "$DEST/vitasdk.tmp"
  tar -C "$DEST/vitasdk.tmp" -xjf "$f"
  # the tarball holds one top-level directory, vitasdk/
  mv "$DEST/vitasdk.tmp/vitasdk" "$DEST/vitasdk"
  rmdir "$DEST/vitasdk.tmp"
  echo "$PIN" > "$DEST/vitasdk/.vocr-pin"
  rm -f "$f"
fi
"$DEST/vitasdk/bin/arm-vita-eabi-gcc" --version | head -n 1
echo "export VITASDK=$DEST/vitasdk PATH=$DEST/vitasdk/bin:\$PATH"
