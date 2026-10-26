#!/usr/bin/env bash
# eval/vocr6_gate.sh TIER BLOB NEXT_TIER_BLOB OUT      one tier (S, N or L) of VOCR6's head gate and two-level head
# eval/vocr6_gate.sh all OUT S_BLOB N_BLOB L_BLOB      every tier, in the protocol's order (L, N, S), then the result
#
# eval/vocr6_gate_protocol.json: per tier the head gate (#120) fit on the tier's int8 blob, for N and L the two-level
# head (#122's adopted 2Lg_Kalt_r96) re-derived on it, their development rules as frozen (VOCR5's cohorts and decodes,
# the cascade with NEXT_TIER_BLOB, the same seed's next tier: S uses N, N uses L; "-" for L), the VOCR6 path (vocr6-lead's
# kit on hval plan v2 and #119's dev plan) and the console-time / memory estimates (vita/a9_layers.py under qemu;
# estimates only). A tier's VOCR6-path arm cascades into the next tier's decided configuration, so N needs L decided
# (it runs L itself from NEXT_TIER_BLOB) and S needs N decided (run N first).
#
# Run inside ONE heavy.sh slot (5 threads): HEAVY_SLOTS=15 $HERD/heavy.sh eval/vocr6_gate.sh ... < /dev/null
# Every step is idempotent (OUT/<T>/steps/*.json, with its wall clock and slot-hours): a rerun resumes.
# Environment: VOCR6_GATE_ARGS (extra vocr6_gate.py options, e.g. "--prefix VOCR5 --all-vns --frozen-only" for the
# VOCR5 reproduction, or "--path-lm F --path-lm-opts alpha=600,beta=2500,flags=1"), VOCR6_GATE_KIT (the kit worktree).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$REPO/.venv-ci/bin/python}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-5}
read -r -a EXTRA <<< "${VOCR6_GATE_ARGS:-}"
cd "$REPO"
make -s -C runtime build/libvocr.so >/dev/null
tier() {  # TIER BLOB NEXT OUT
  local next=()
  [ "$3" != "-" ] && next=(--next-blob "$3")
  "$PY" eval/vocr6_gate.py run --tier "$1" --blob "$2" "${next[@]}" --out "$4" "${EXTRA[@]}"
}
if [ "${1:-}" = all ]; then
  [ $# -eq 5 ] || { echo "usage: $0 all OUT S_BLOB N_BLOB L_BLOB" >&2; exit 2; }
  out=$2
  tier L "$5" - "$out"
  tier N "$4" "$5" "$out"
  tier S "$3" "$4" "$out"
  "$PY" eval/vocr6_gate.py result --out "$out" --result "$out/result.json"
  "$PY" eval/vocr6_gate.py timing --out "$out"
  exit 0
fi
[ $# -eq 4 ] || { echo "usage: $0 TIER BLOB NEXT_TIER_BLOB OUT  |  $0 all OUT S_BLOB N_BLOB L_BLOB" >&2; exit 2; }
case $1 in S|N|L) ;; *) echo "TIER: S, N or L" >&2; exit 2 ;; esac
tier "$1" "$2" "$3" "$4"
"$PY" eval/vocr6_gate.py result --out "$4" --result "$4/result.json"
