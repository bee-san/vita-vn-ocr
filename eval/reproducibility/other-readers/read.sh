#!/usr/bin/env bash
# The other readers' reads (eval/other_readers_protocol.json), one engine per call, offline (unshare -rn):
#   eval/reproducibility/other-readers/read.sh ENGINE P        P = the private output directory
# ENGINE: MeikiOCR | PPOCRv6_small | PPOCRv5_mobile_int8 | manga_ocr. Each reads vita-test and psp-test through
# eval/run.py (screen polarity) and the private sets through eval/other_readers.py read (CLANNAD: not MeikiOCR,
# whose arm (a) readings exist). Run it through $HERD/heavy.sh, from the repository root, after .venv, .venv-meiki
# (eval/clannad_meiki_requirements.lock) and eval/ncnn/build.sh.
set -euo pipefail
ENGINE=$1; P=$2
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
A=$HERD/shared/artifacts
export VOCR_MODELS=$A/teacher/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
case $ENGINE in
  MeikiOCR) SPEC="cmd:$PWD/.venv-meiki/bin/python $PWD/eval/meiki_cmd.py" ;;
  PPOCRv6_small) SPEC=ppocr-v6-small ;;
  PPOCRv5_mobile_int8) SPEC=ncnn:$A/eval/models/ncnn/ppocr-v5-mobile-jis-int8 ;;
  manga_ocr) SPEC=manga-ocr ;;
  *) echo "unknown engine $ENGINE" >&2; exit 2 ;;
esac
mkdir -p "$P"
for S in vita-test psp-test; do
  unshare -rn .venv/bin/python eval/run.py --model "$SPEC" --set $S --polarity native --data $A/eval \
    --out "$P/run_${ENGINE}_$S.json" > "$P/run_${ENGINE}_$S.log" 2>&1
done
for S in pcsg00807 clannad; do
  [ "$ENGINE" = MeikiOCR ] && [ $S = clannad ] && continue
  unshare -rn .venv/bin/python eval/other_readers.py read --set $S --reader "$ENGINE" --spec "$SPEC" --into "$P" \
    > "$P/read_${ENGINE}_$S.log" 2>&1
done
echo "read.sh $ENGINE finished $(date -u +%FT%TZ)"
