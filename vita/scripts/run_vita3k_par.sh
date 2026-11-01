#!/usr/bin/env bash
# Run vocr_par.vpk (vita/par) in Vita3K, headless under Xvfb, on a job directory of vita/vita3k_parity.py.
#
#   vita/scripts/run_vita3k_par.sh JOBDIR VPK OUT
#
#   JOBDIR   jobs.txt and every file its jobs read; copied to ux0:data/vocrpar/
#   VPK      a vocr_par.vpk (vita/scripts/build.sh builds it beside vocr_test.vpk)
#   OUT      gets ux0:data/vocrpar/ after the run (each job's out_k.txt and err_k.txt, status.txt, done.txt and every
#            file a job wrote), vita3k.log and vita3k.stdout
#   env VITA3K   the fetch_vita3k.sh install (default vita/tools/vita3k/4154)
#   env TIMEOUT  seconds before the emulator is stopped (default 7200)
#   env SANDBOX  where the throwaway sandbox goes (default $TMPDIR or /tmp); the job files are copied into it
#
# As vita/scripts/run_vita3k.sh: Vita3K runs in a sandbox (its config, ux0:, logs and caches), $HOME is never written,
# the SDL frontend on Mesa's llvmpipe under Xvfb. The app exits when its last job is done, and Vita3K with it.
set -euo pipefail
vita=$(cd "$(dirname "$0")/.." && pwd)
[ $# -ge 3 ] || { sed -n '2,16p' "$0"; exit 2; }
jobs=$(cd "$1" && pwd)
vpk=$(realpath "$2")
mkdir -p "$3"
out=$(cd "$3" && pwd)
V=${VITA3K:-$vita/tools/vita3k/4154}
[ -x "$V/app/usr/bin/Vita3K" ] || { echo "error: no Vita3K at $V; run vita/scripts/fetch_vita3k.sh" >&2; exit 1; }
command -v xvfb-run > /dev/null || { echo "error: xvfb-run is needed (package xvfb / xorg-x11-server-Xvfb)" >&2; exit 1; }
[ -f "$jobs/jobs.txt" ] || { echo "error: no $jobs/jobs.txt" >&2; exit 1; }
sbx=$(mktemp -d "${SANDBOX:-${TMPDIR:-/tmp}}/vocr-vita3k-par.XXXXXX")
trap 'rm -rf "$sbx"' EXIT
mkdir -p "$sbx/config/Vita3K" "$sbx/data/Vita3K/Vita3K/ux0/data" "$sbx/cache"
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
cp -r "$jobs" "$ux0/data/vocrpar"
rm -f "$ux0/data/vocrpar/status.txt" "$ux0/data/vocrpar/done.txt"
t0=$(date +%s)
set +e
env LP_NUM_THREADS="${LP_NUM_THREADS:-2}" APPDIR="$V/app" HOME="$sbx" XDG_CONFIG_HOME="$sbx/config" XDG_DATA_HOME="$sbx/data" XDG_CACHE_HOME="$sbx/cache" \
  timeout "${TIMEOUT:-7200}" xvfb-run -a -s "-screen 0 1280x720x24" \
  "$V/app/usr/bin/ld-linux-x86-64.so.2" --library-path "$V/compat:$V/app/usr/lib:/usr/lib64:/usr/lib/x86_64-linux-gnu" \
  "$V/app/usr/bin/Vita3K" --frontend sdl "$vpk" < /dev/null > "$out/vita3k.stdout" 2>&1
rc=$?
set -e
echo "Vita3K exit $rc after $(( $(date +%s) - t0 )) s"
cp "$sbx/cache/Vita3K/vita3k.log" "$out/vita3k.log" 2>/dev/null || true
# the outputs only: every file the jobs wrote (not the inputs, which JOBDIR already holds)
(cd "$ux0/data/vocrpar" && for f in *; do [ -e "$jobs/$f" ] || cp "$f" "$out/"; done)
if [ ! -f "$out/done.txt" ]; then
  echo "error: the app did not finish its jobs (see $out/vita3k.stdout and $out/status.txt)" >&2
  exit 1
fi
