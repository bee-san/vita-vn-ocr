#!/usr/bin/env bash
# Fetch the pinned ARM test tools into runtime/scripts/tools (user space, no root):
#   - Arm GNU Toolchain 14.2.rel1, x86_64 host -> arm-none-linux-gnueabihf (gcc 14.2.1)
#   - qemu-arm 10.0.13 (Debian trixie's qemu-user 1:10.0.13+ds-0+deb13u1, a static binary)
# Both are checked against sha256s. Then:
#   make arm parity-arm ARM_CC=runtime/scripts/tools/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-linux-gnueabihf/bin/arm-none-linux-gnueabihf-gcc
# The tools are only for testing the NEON path under emulation; nothing here is shipped.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p "${1:-tools}/dl" "${1:-tools}/bin"
DEST=$(cd "${1:-tools}" && pwd)

TC=arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-linux-gnueabihf
TC_URL=https://developer.arm.com/-/media/Files/downloads/gnu/14.2.rel1/binrel/$TC.tar.xz
TC_SHA=32301a5a33aab47810837cdab848a5a513ca22804d3168d3ada5833828b07912
QEMU_DEB=qemu-user_10.0.13+ds-0+deb13u1_amd64.deb
QEMU_URL=https://deb.debian.org/debian/pool/main/q/qemu/$QEMU_DEB
QEMU_SHA=ca6ede739327a20ae5a3498230c8a3a9a072c586e131bc6bcc1201eb1725e336
UA="vita-vn-ocr-runtime-tools/1 (+https://github.com/bee-san/vita-vn-ocr)"

fetch() { # url file sha256
  if [ ! -f "$2" ] || ! echo "$3  $2" | sha256sum -c --status; then
    curl -sSfL -A "$UA" --retry 3 -o "$2.part" "$1"
    mv "$2.part" "$2"
  fi
  echo "$3  $2" | sha256sum -c
}

if [ ! -x "$DEST/$TC/bin/arm-none-linux-gnueabihf-gcc" ]; then
  fetch "$TC_URL" "$DEST/dl/$TC.tar.xz" "$TC_SHA"
  tar -C "$DEST" -xf "$DEST/dl/$TC.tar.xz"
  rm -f "$DEST/dl/$TC.tar.xz"
fi
if [ ! -x "$DEST/bin/qemu-arm" ]; then
  fetch "$QEMU_URL" "$DEST/dl/$QEMU_DEB" "$QEMU_SHA"
  x=$(mktemp -d)
  (cd "$x" && ar x "$DEST/dl/$QEMU_DEB" && tar -xf data.tar.xz ./usr/bin/qemu-arm)
  cp "$x/usr/bin/qemu-arm" "$DEST/bin/qemu-arm"
  rm -rf "$x" "$DEST/dl/$QEMU_DEB"
fi
"$DEST/$TC/bin/arm-none-linux-gnueabihf-gcc" --version | head -n 1
"$DEST/bin/qemu-arm" --version | head -n 1
echo "ARM_CC=$DEST/$TC/bin/arm-none-linux-gnueabihf-gcc"
echo "QEMU=$DEST/bin/qemu-arm"
