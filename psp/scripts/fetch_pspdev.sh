#!/usr/bin/env bash
# Fetch the pinned PSP toolchain (pspdev) into psp/tools (user space, no root):
#   pspdev v20261001, ubuntu-latest x86_64 build: psp-gcc 15.2.0, binutils 2.44 (Allegrex),
#   newlib, pspsdk 6f15c15 (2026-09-28). sha256-checked.
#
#   psp/scripts/fetch_pspdev.sh [DEST]        # default DEST: psp/tools
#   export PSPDEV=DEST/pspdev PATH=DEST/pspdev/bin:$PATH
#
# The release's host binaries are built on Ubuntu 24.04 and need glibc >= 2.38. On an older
# host (Amazon Linux 2023 has 2.34) the script also unpacks Ubuntu 24.04's glibc 2.39
# (libc6 2.39-0ubuntu8.9) next to the toolchain and points every host executable of the
# toolchain at it with patchelf 0.18.0: interpreter + RUNPATH, with --no-sort (sorting the
# program headers breaks cc1, and patchelf 0.17 breaks psp-gcc). Both downloads are
# sha256-checked. Only the compiler's own host processes use that glibc; nothing of it ends up
# in a PSP binary. On a host with glibc >= 2.38 (GitHub's ubuntu-24.04) nothing is patched.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p "${1:-tools}/dl"
DEST=$(cd "${1:-tools}" && pwd)

REL=v20261001
TC_FILE=pspdev-ubuntu-latest-x86_64.tar.gz
TC_URL=https://github.com/pspdev/pspdev/releases/download/$REL/$TC_FILE
TC_SHA=a86efe624e770005290919617859e9df24bd8fcd3e364f4587f23ec7364b0acd
LIBC_DEB=libc6_2.39-0ubuntu8.9_amd64.deb
LIBC_URL=https://launchpad.net/ubuntu/+archive/primary/+files/$LIBC_DEB
LIBC_SHA=ff5557d99b51f761c4b7c92368b9cc45565eda17df9bf9eb4b134d09825008be
PE_FILE=patchelf-0.18.0-x86_64.tar.gz
PE_URL=https://github.com/NixOS/patchelf/releases/download/0.18.0/$PE_FILE
PE_SHA=ce84f2447fb7a8679e58bc54a20dc2b01b37b5802e12c57eece772a6f14bf3f0
NEED_GLIBC=2.38
UA="vita-vn-ocr-psp-tools/1 (+https://github.com/bee-san/vita-vn-ocr)"

fetch() { # url file sha256
  if [ ! -f "$2" ] || ! echo "$3  $2" | sha256sum -c --status; then
    curl -sSfL -A "$UA" --retry 3 -o "$2.part" "$1"
    mv "$2.part" "$2"
  fi
  echo "$3  $2" | sha256sum -c
}

host_glibc=$(getconf GNU_LIBC_VERSION | awk '{print $2}')
shim=0
[ "$(printf '%s\n%s\n' "$NEED_GLIBC" "$host_glibc" | sort -V | head -n 1)" = "$NEED_GLIBC" ] || shim=1

if [ ! -x "$DEST/pspdev/bin/psp-gcc" ] || { [ $shim = 1 ] && [ ! -f "$DEST/pspdev/.glibc-shim" ]; }; then
  fetch "$TC_URL" "$DEST/dl/$TC_FILE" "$TC_SHA"
  rm -rf "$DEST/pspdev" "$DEST/glibc-2.39"
  tar -C "$DEST" -xzf "$DEST/dl/$TC_FILE"
  rm -f "$DEST/dl/$TC_FILE"
  echo "$REL $TC_SHA" > "$DEST/pspdev/.release"
  if [ $shim = 1 ]; then
    echo "host glibc $host_glibc < $NEED_GLIBC: the toolchain's host programs get Ubuntu's glibc 2.39"
    fetch "$PE_URL" "$DEST/dl/$PE_FILE" "$PE_SHA"
    fetch "$LIBC_URL" "$DEST/dl/$LIBC_DEB" "$LIBC_SHA"
    mkdir -p "$DEST/patchelf"
    tar -C "$DEST/patchelf" -xzf "$DEST/dl/$PE_FILE" ./bin/patchelf
    PE=$DEST/patchelf/bin/patchelf
    G=$DEST/glibc-2.39
    x=$(mktemp -d)
    (cd "$x" && ar x "$DEST/dl/$LIBC_DEB" && tar --zstd -xf data.tar.zst ./usr/lib/x86_64-linux-gnu)
    mkdir -p "$G"
    # the loader and the libraries a compiler can need (gconv, nss and the debug libs are left out)
    for f in ld-linux-x86-64.so.2 libc.so.6 libm.so.6 libmvec.so.1 libdl.so.2 libpthread.so.0 librt.so.1 libutil.so.1; do
      cp "$x/usr/lib/x86_64-linux-gnu/$f" "$G/"
    done
    rm -rf "$x" "$DEST/dl/$LIBC_DEB" "$DEST/dl/$PE_FILE"
    declare -A seen=()
    n=0
    while IFS= read -r -d '' f; do
      ino=$(stat -c %d:%i "$f")
      [ -z "${seen[$ino]:-}" ] || continue        # hard links: patch each file once
      seen[$ino]=1
      [ "$(head -c 4 "$f" | od -An -tx1 | tr -d ' ')" = 7f454c46 ] || continue
      "$PE" --print-interpreter "$f" > /dev/null 2>&1 || continue   # host executables only
      old=$("$PE" --print-rpath "$f")
      "$PE" --no-sort --set-interpreter "$G/ld-linux-x86-64.so.2" \
            --set-rpath "$G:/lib64:/usr/lib64${old:+:$old}" "$f"
      n=$((n + 1))
    done < <(find "$DEST/pspdev" -type f -perm -u+x -print0)
    echo "patched $n host executables for glibc 2.39 ($G)" > "$DEST/pspdev/.glibc-shim"
    cat "$DEST/pspdev/.glibc-shim"
  fi
fi

# smoke test: compile and link a PSP object, so a broken host binary shows up here
t=$(mktemp -d)
printf 'int vocr_smoke(int a) { return a * 3; }\n' > "$t/t.c"
"$DEST/pspdev/bin/psp-gcc" -O2 -G0 -c "$t/t.c" -o "$t/t.o"
"$DEST/pspdev/bin/psp-size" "$t/t.o" > /dev/null
rm -rf "$t"
"$DEST/pspdev/bin/psp-gcc" --version | head -n 1
echo "PSPDEV=$DEST/pspdev"
