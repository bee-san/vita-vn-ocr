#!/usr/bin/env bash
# The teacher's reading stage as one heavy.sh job (one slot, 5 single-thread workers), offline:
#   $HERD/heavy.sh teacher/read_all.sh [SETS]
# SETS default "vita-test train": eval's test sets are read with every engine on every line
# (--no-cascade, to measure), then realdata's collection with the default voters. Resumable:
# finished screenshots are skipped on restart. Logs in $TEACHER_OUT/reads/.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$HERE")"
SETS="${1:-vita-test train}"
READS="${TEACHER_READS:-${TEACHER_OUT:-${HERD:-/local/home/skerraut/herd-vita-ocr}/shared/artifacts/teacher}/reads}"
rc=0
for s in $SETS; do
  if [ "$s" = train ]; then
    args=(--voters screenai,v6s,manga)
  else
    args=(--voters screenai,v6s,manga,v5s --no-cascade)
  fi
  echo "$(date -u +%FT%TZ) read $s ${args[*]}"
  timeout 6h unshare -rn "$HERE/.venv/bin/python" teacher/label.py read --set "$s" --procs 5 --reads "$READS" \
    "${args[@]}" < /dev/null || rc=1
done
echo "READ_ALL_DONE rc=$rc"
exit $rc
