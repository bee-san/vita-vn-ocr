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
# (libc6 2.39-0ubuntu8.9, sha256-checked) next to the toolchain, and every host program of
# the toolchain becomes a two-line sh wrapper that starts the untouched program (renamed
# .NAME.real) with that glibc's loader: ld-linux-x86-64.so.2 --library-path GLIBC --argv0 "$0".
# argv[0] stays the wrapper's path, so gcc still finds the rest of its tree, and the ELF files
# are not edited (rewriting their interpreter in place with patchelf gave programs that crash
# at start for some DEST paths). Only the compiler's own host processes use that glibc;
# nothing of it ends up in a PSP binary. On a host with glibc >= 2.38 (GitHub's ubuntu-24.04)
# nothing is wrapped.
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
NEED_GLIBC=2.38
SHIM=.glibc-wrappers   # marker of a wrapped install (older installs, patched in place, are redone)
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

sq() { # $1 as a single-quoted sh word
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

if [ ! -x "$DEST/pspdev/bin/psp-gcc" ] || { [ $shim = 1 ] && [ ! -f "$DEST/pspdev/$SHIM" ]; }; then
  fetch "$TC_URL" "$DEST/dl/$TC_FILE" "$TC_SHA"
  rm -rf "$DEST/pspdev" "$DEST/glibc-2.39" "$DEST/patchelf"
  tar -C "$DEST" -xzf "$DEST/dl/$TC_FILE"
  rm -f "$DEST/dl/$TC_FILE"
  echo "$REL $TC_SHA" > "$DEST/pspdev/.release"
  if [ $shim = 1 ]; then
    echo "host glibc $host_glibc < $NEED_GLIBC: the toolchain's host programs get Ubuntu's glibc 2.39"
    fetch "$LIBC_URL" "$DEST/dl/$LIBC_DEB" "$LIBC_SHA"
    G=$DEST/glibc-2.39
    LD=$G/ld-linux-x86-64.so.2
    x=$(mktemp -d)
    (cd "$x" && ar x "$DEST/dl/$LIBC_DEB" && tar --zstd -xf data.tar.zst ./usr/lib/x86_64-linux-gnu)
    mkdir -p "$G"
    # the loader and the libraries a compiler can need (gconv, nss and the debug libs are left out)
    for f in ld-linux-x86-64.so.2 libc.so.6 libm.so.6 libmvec.so.1 libdl.so.2 libpthread.so.0 librt.so.1 libutil.so.1; do
      cp "$x/usr/lib/x86_64-linux-gnu/$f" "$G/"
    done
    rm -rf "$x" "$DEST/dl/$LIBC_DEB"
    # The host programs: executable ELF files with a program interpreter (shared libraries have
    # none), listed with the toolchain's own readelf, run by the new loader, before any is moved.
    progs=()
    while IFS= read -r -d '' f; do
      [ "$(head -c 4 "$f" | od -An -tx1 | tr -d ' ')" = 7f454c46 ] || continue
      "$LD" --library-path "$G" "$DEST/pspdev/bin/psp-readelf" -lW "$f" 2> /dev/null |
        grep -q 'Requesting program interpreter' || continue
      progs+=("$f")
    done < <(find "$DEST/pspdev" -type f -perm -u+x -print0)
    [ ${#progs[@]} -gt 0 ] || { echo "no host programs found in $DEST/pspdev" >&2; exit 1; }
    # Each becomes a wrapper; hard links (psp-gcc, psp-gcc-15.2.0) each get their own, and their
    # .real names stay hard links of one file.
    for f in "${progs[@]}"; do
      real=$(dirname "$f")/.$(basename "$f").real
      mv "$f" "$real"
      printf '#!/bin/sh\n# glibc 2.39 for this host (psp/scripts/fetch_pspdev.sh)\nexec %s --library-path %s --argv0 "$0" %s "$@"\n' \
        "$(sq "$LD")" "$(sq "$G")" "$(sq "$real")" > "$f"
      chmod 755 "$f"
    done
    echo "wrapped ${#progs[@]} host programs for glibc 2.39 ($G)" > "$DEST/pspdev/$SHIM"
    cat "$DEST/pspdev/$SHIM"
  fi
fi

# smoke test: compile a PSP object and read it back, so a broken host program shows up here
t=$(mktemp -d)
printf 'int vocr_smoke(int a) { return a * 3; }\n' > "$t/t.c"
"$DEST/pspdev/bin/psp-gcc" -O2 -G0 -c "$t/t.c" -o "$t/t.o"
"$DEST/pspdev/bin/psp-size" "$t/t.o" > /dev/null
rm -rf "$t"
"$DEST/pspdev/bin/psp-gcc" --version | head -n 1
echo "PSPDEV=$DEST/pspdev"
