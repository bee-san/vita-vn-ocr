#!/usr/bin/env bash
# Every PSP measurement of psp/README.md, end to end:
#
#   psp/run_all.sh OUT [CKPT]          (from the repo root or anywhere)
#
# Needs psp/tools (scripts/fetch_pspdev.sh, scripts/build_ppsspp.sh, or PSPDEV / PPSSPP set),
# eval's screenshots ($VOCR_EVAL_DATA or data/eval), data/synth/synth_00039.npz, and a
# checkpoint (default $VOCR_CKPT, else models/v1/ckpt_best.pt). Single-threaded (PPSSPP's
# emulation is one host thread; the PyTorch references run with 1 thread).
#
#  1. EBOOTs: the scalar build (the default) and the VFPU=1 build; code size and stack chains
#  2. crop packs: psp-test, vita-test (eval's GT-box crops, RGBA) and synth-val (300, grey)
#  3. parity: runtime/tests/parity.py with the PSP build in PPSSPP as its C side
#     (--cli psp/ppsspp_cli.py): bit-exact logits against tests/vocr_ref.py, tolerance against
#     PyTorch, strings, tile invariance, VOCR_INVERT, arena plan; f32, w8f, w8i
#  4. bundles: the console self-check (host CRC-32 of every line's logits) and PPSSPP's
#     emulated time per line, per build and mode, tile 4 / head 16 and tile 16 / head 16
#  5. eval: eval/run.py through the PSP build and through the host vocr_cli (same strings)
#  then psp/summarize.py OUT prints the tables.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(mkdir -p "${1:?usage: psp/run_all.sh OUT [CKPT]}" && cd "$1" && pwd)
CKPT=${2:-${VOCR_CKPT:-$ROOT/models/v1/ckpt_best.pt}}
PY=${PYTHON:-python3}
export PSPDEV=${PSPDEV:-$ROOT/psp/tools/pspdev}
export PPSSPP=${PPSSPP:-$ROOT/psp/tools/ppsspp/PPSSPPHeadless}
export OMP_NUM_THREADS=1 TORCH_NUM_THREADS=1
SYNTH=${VOCR_SYNTH_VAL:-$ROOT/data/synth/synth_00039.npz}
cd "$ROOT"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/run.log"; }

log "1. EBOOTs (PSPDEV=$PSPDEV)"
for v in 1 0; do   # the scalar (default) build last, so psp/EBOOT.PBP is the default one
  make -s -C psp clean > /dev/null
  make -s -C psp VFPU=$v > "$OUT/make_vfpu$v.log" 2>&1
  cp psp/EBOOT.PBP "$OUT/EBOOT_$([ $v = 1 ] && echo vfpu || echo scalar).PBP"
done
make -s -C psp size > "$OUT/size.txt" 2>&1
"$PSPDEV/bin/psp-size" psp/vocr_test.elf >> "$OUT/size.txt"
mkdir -p "$OUT/blobs"
$PY train/export.py --ckpt "$CKPT" --out "$OUT/blobs/f32.vocr" > /dev/null
$PY train/export.py --ckpt "$CKPT" --out "$OUT/blobs/w8.vocr" --int8 > /dev/null
make -s -C runtime > /dev/null

log "2. packs"
for s in psp-test vita-test; do $PY psp/pack.py build --set "$s" --out "$OUT/packs/$s" | tee -a "$OUT/run.log"; done
$PY psp/pack.py build --set synth-val --n 300 --out "$OUT/packs/synth-val" | tee -a "$OUT/run.log"

log "3. parity through PPSSPP"
for b in scalar vfpu; do
  modes=f32,w8f,w8i
  for s in psp-test vita-test synth-val; do
    src=(--set "$s"); [ "$s" = synth-val ] && src=(--shard "$SYNTH" --n 300)
    PSP_EBOOT="$OUT/EBOOT_$b.PBP" $PY runtime/tests/parity.py --ckpt "$CKPT" "${src[@]}" --modes "$modes" \
      --cli "$PY psp/ppsspp_cli.py" --out "$OUT/parity_${b}_$s.json" > "$OUT/parity_${b}_$s.log" 2>&1 \
      && log "   $b $s PARITY_OK" || log "   $b $s PARITY_FAIL (see parity_${b}_$s.log)"
  done
done

log "4. bundles: console self-check + PPSSPP emulated time"
for b in scalar vfpu; do
  for m in f32 w8f w8i; do
    blob="$OUT/blobs/w8.vocr"; [ $m = f32 ] && blob="$OUT/blobs/f32.vocr"
    for s in psp-test vita-test; do
      for tl in 4 16; do
        d="$OUT/bundles/${b}_${m}_${s}_t$tl"
        $PY psp/pack.py bundle --eboot "$OUT/EBOOT_$b.PBP" --blob "$blob" --mode $m --tile "$tl" --head 16 \
          --pack "$OUT/packs/$s" --out "$d" > /dev/null
        if $PY psp/ppsspp_cli.py --bundle "$d" --stats "$d/stats.json" --keep "$d/run" > "$d/log.txt" 2>&1; then
          log "   $b $m $s tile $tl: same as host"; else log "   $b $m $s tile $tl: DIFFERS or failed"; fi
        cp "$d/run/.ppsspp/PSP/GAME/VOCR/results.txt" "$d/results.txt" 2>/dev/null || true
        rm -rf "$d/run"
      done
    done
  done
done

log "5. eval/run.py through the PSP build and the host CLI"
for m in w8f w8i; do
  $PY eval/run.py --model "cmd:$PY psp/ppsspp_cli.py $OUT/blobs/w8.vocr --mode $m --stdin" \
    --set psp-test,vita-test --out "$OUT/eval_psp_$m.json" > "$OUT/eval_psp_$m.log" 2>&1
  $PY eval/run.py --model "cmd:runtime/build/vocr_cli $OUT/blobs/w8.vocr --mode $m --stdin" \
    --set psp-test,vita-test --out "$OUT/eval_host_$m.json" > "$OUT/eval_host_$m.log" 2>&1
done
log "done"
$PY psp/summarize.py "$OUT" | tee "$OUT/summary.md"
