#!/usr/bin/env bash
# Run vocr_test.vpk in Vita3K, headless under Xvfb, on a bundle from vita/bundle.py, then
# compare the results with the host runtime (vita/compare.py).
#
#   vita/scripts/run_vita3k.sh BUNDLE [VPK] [OUT] [key=value ...]
#
#   BUNDLE   a vita/bundle.py output directory
#   VPK      default build/vita/vocr_test.vpk (vita/scripts/build.sh)
#   OUT      where results.tsv, vita3k.log and compare.txt go (default BUNDLE/vita3k-<UTC time>)
#   key=value  overrides for the app's config.txt for this run, e.g. models=w8i repeat=1
#   env VITA3K   the fetch_vita3k.sh install (default vita/tools/vita3k/4154)
#   env TIMEOUT  seconds before the emulator is stopped (default 3600)
#   env PYTHON   python with numpy for compare.py (default python3)
#
# Vita3K runs in a throwaway sandbox (its config, ux0:, logs and caches under $TMPDIR);
# $HOME is never written. The app runs with autoexit=1, so the SDL frontend quits with it.
# Vita3K uses the OpenGL renderer on Mesa's llvmpipe, which Xvfb provides. Its timings are
# x86 JIT timings of the emulator: they say nothing about a PS Vita.
set -euo pipefail
vita=$(cd "$(dirname "$0")/.." && pwd)
root=$(dirname "$vita")
[ $# -ge 1 ] || { sed -n '2,17p' "$0"; exit 2; }
bundle=$(cd "$1" && pwd)
vpk=$(realpath "${2:-$root/build/vita/vocr_test.vpk}")
out=${3:-$bundle/vita3k-$(date -u +%Y%m%dT%H%M%SZ)}
V=${VITA3K:-$vita/tools/vita3k/4154}
[ -x "$V/app/usr/bin/Vita3K" ] || { echo "error: no Vita3K at $V; run vita/scripts/fetch_vita3k.sh" >&2; exit 1; }
command -v xvfb-run > /dev/null || { echo "error: xvfb-run is needed (package xvfb / xorg-x11-server-Xvfb)" >&2; exit 1; }
mkdir -p "$out"
out=$(cd "$out" && pwd)
sbx=$(mktemp -d "${TMPDIR:-/tmp}/vocr-vita3k.XXXXXX")
trap 'rm -rf "$sbx"' EXIT
mkdir -p "$sbx/config/Vita3K" "$sbx/data/Vita3K/Vita3K/ux0/data/vocr" "$sbx/cache"
cat > "$sbx/config/Vita3K/config.yml" <<'EOF'
---
initial-setup: true
backend-renderer: OpenGL
validation-layer: false
discord-rich-presence: false
show-live-area-screen: false
delay-start: 0
delay-background: 0
log-level: 2
...
EOF
ux0=$sbx/data/Vita3K/Vita3K/ux0
for f in "$bundle"/cases.bin "$bundle"/*.vocr "$bundle"/font.bin "$bundle"/config.txt; do
  [ -e "$f" ] && cp "$f" "$ux0/data/vocr/"
done
touch "$ux0/data/vocr/config.txt"
sed -i '/^autoexit=/d' "$ux0/data/vocr/config.txt"
echo "autoexit=1" >> "$ux0/data/vocr/config.txt"
shift $(( $# < 3 ? $# : 3 ))
for kv in "$@"; do
  case $kv in *=*) ;; *) echo "error: override '$kv' is not key=value" >&2; exit 2 ;; esac
  sed -i "/^${kv%%=*}=/d" "$ux0/data/vocr/config.txt"
  echo "$kv" >> "$ux0/data/vocr/config.txt"
done
t0=$(date +%s)
set +e
env APPDIR="$V/app" HOME="$sbx" XDG_CONFIG_HOME="$sbx/config" XDG_DATA_HOME="$sbx/data" XDG_CACHE_HOME="$sbx/cache" \
  timeout "${TIMEOUT:-3600}" xvfb-run -a -s "-screen 0 1280x720x24" \
  "$V/app/usr/bin/ld-linux-x86-64.so.2" --library-path "$V/compat:$V/app/usr/lib:/usr/lib64:/usr/lib/x86_64-linux-gnu" \
  "$V/app/usr/bin/Vita3K" --frontend sdl "$vpk" < /dev/null > "$out/vita3k.stdout" 2>&1
rc=$?
set -e
echo "Vita3K exit $rc after $(( $(date +%s) - t0 )) s"
cp "$sbx/cache/Vita3K/vita3k.log" "$out/vita3k.log" 2>/dev/null || true
if [ ! -f "$ux0/data/vocr/results.tsv" ]; then
  echo "error: the app wrote no results.tsv (see $out/vita3k.stdout)" >&2
  exit 1
fi
cp "$ux0/data/vocr/results.tsv" "$out/results.tsv"
"${PYTHON:-python3}" "$vita/compare.py" --bundle "$bundle" "$out/results.tsv" --emulator | tee "$out/compare.txt"
