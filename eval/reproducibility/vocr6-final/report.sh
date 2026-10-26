#!/usr/bin/env bash
# report.sh MODELS_DIR OUT: VOCR6 stage 4's final test (eval/vocr6_final_test_protocol.json; issue #125), read once.
# Modelled on eval/reproducibility/stride16-20261023/report.sh, as one command: the S/N/L blobs in, every table out.
#   MODELS_DIR  S.vocr, N.vocr, L.vocr and models.json, written by `eval/vocr6_final_test.py stage` from vocr6-lead's
#               message "FINAL READ S=<blob> N=<blob> L=<blob>"; or '-' with PATHS=a: the validation (VOCR5 alone,
#               path (a) alone, compared with VOCR5's published records by `vocr6_final_test.py validate`)
#   OUT         a new public directory (counts, rates, intervals, hashes: report.json, report.md, validation.json,
#               log.txt); its private twin $HERD/shared/private/vocr6-final-test/<basename OUT> holds every reading,
#               plan, dump and per-unit row (protected data: never committed, never sent anywhere)
#   PATHS       a,b (default, the final read) | a (the validation)
#   RUBI        a Bee-san/Rubi_for_vita checkout of main for path (b) (default: a fresh clone of main in the private
#               directory, taken now)
# Run it through heavy.sh (one slot, 5 threads). Each step writes its own log under the private directory; nothing of
# a step's output reaches the terminal but this script's own one-line notes (no protected text, no *_DONE token).
set -euo pipefail
MODELS=$1; OUT=$(realpath -m "$2"); NAME=$(basename "$OUT")
HERD=/local/home/skerraut/herd-vita-ocr; A=$HERD/shared/artifacts; export HERD
REPO=$(cd "$(dirname "$(realpath "$0")")/../../.." && pwd); PY=$REPO/.venv-ci/bin/python
PRIV=$HERD/shared/private/vocr6-final-test/$NAME
PATHS=${PATHS:-a,b}
P=eval/vocr6_final_test_protocol.json; H=eval/vocr6_final_test.py
cd "$REPO"
for f in "$P" "$H" eval/reproducibility/vocr6-final/report.sh; do
  git ls-files --error-unmatch "$f" > /dev/null 2>&1 || { echo "$f is not committed"; exit 2; }
done
git diff --quiet HEAD -- eval runtime train synth data || { echo "tracked files differ from HEAD: commit first"; exit 2; }
if [ "$MODELS" = - ]; then
  MODE=validate; FAMS=(VOCR5); SUF=v6ftcheck
  [ "$PATHS" = a ] || { echo "the validation reads path (a) only (PATHS=a)"; exit 2; }
else
  MODE=final; FAMS=(VOCR5 VOCR6); SUF=v6ftfinal; MODELS=$(realpath "$MODELS")
  [ "$PATHS" = a,b ] || { echo "the final read reads both paths (PATHS=a,b)"; exit 2; }
  [ -s "$MODELS/models.json" ] || { echo "$MODELS/models.json is missing: run vocr6_final_test.py stage"; exit 2; }
fi
mkdir -p "$OUT" "$PRIV"/{logs,a/lm,a/pcsg,clannad,b,blobs}
LOG=$OUT/log.txt
note() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
run() {   # run LOGNAME CMD...: a step's output goes to its private log only
  local name=$1; shift
  if ! "$@" > "$PRIV/logs/$name.log" 2>&1; then note "step $name FAILED (log: $PRIV/logs/$name.log)"; exit 3; fi
}
# the attempt record: one validation and one final read, each refused a second time
$PY - "$PRIV/attempt.json" "$MODE" "$P" "${MODELS}" <<'PY'
import datetime, hashlib, json, subprocess, sys
path, mode, proto, models = sys.argv[1:5]
rec = {"mode": mode, "started": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
       "repo_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
       "protocol_sha256": hashlib.sha256(open(proto, "rb").read()).hexdigest()}
if models != "-":
    rec["models"] = {t: e["sha256"] for t, e in json.load(open(f"{models}/models.json"))["tiers"].items()}
with open(path, "x") as f:
    json.dump(rec, f, indent=1)
PY
cp "$PRIV/attempt.json" "$OUT/attempt.json"
note "start $MODE (families ${FAMS[*]}, paths $PATHS), repo $(git rev-parse --short HEAD), private $PRIV"

# ---- blobs: VOCR5's pinned files (path a: v0.3.0; path b: the gated blobs Rubi runs), VOCR6's staged ones
B=$PRIV/blobs
$PY - "$P" "$B" "$MODELS" "$MODE" <<'PY'
import hashlib, json, shutil, sys
proto, out, models, mode = sys.argv[1:5]
p = json.load(open(proto))
def put(src, dst, want):
    shutil.copyfile(src, f"{out}/{dst}")
    got = hashlib.sha256(open(f"{out}/{dst}", "rb").read()).hexdigest()
    if got != want:
        raise SystemExit(f"{src}: sha256 {got[:16]}... is not the pinned {want[:16]}...")
for t in "SNL":
    e = p["models"]["VOCR5"]["path_a"][t]
    put(e["file"], f"VOCR5_{t}_w8.vocr", e["sha256"])
    g = p["models"]["VOCR5"]["gated"][t]
    put(g["file"], f"VOCR5g_{t}.vocr", g["sha256"])
    if mode == "final":
        m = json.load(open(f"{models}/models.json"))["tiers"][t]
        put(f"{models}/{t}.vocr", f"VOCR6_{t}_w8.vocr", m["sha256"])
PY
[ "$MODE" = final ] && cp "$MODELS/models.json" "$PRIV/models.json"
(cd "$B" && sha256sum ./*.vocr) > "$OUT/blobs.sha256"
ARMS=(); for f in "${FAMS[@]}"; do for t in S N L; do ARMS+=("${f}_$t"); done; done

# ---- path (a) 1: vita-test + psp-test fixed lines and whole dialogue (eval/vocr3_report.py shared, scalar C W8I)
$PY - "$PRIV/a/spec.json" "$B" "${ARMS[@]}" <<'PY'
import hashlib, json, sys
out, blobs, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
spec = {"schema": 1, "kind": "vocr3-report-spec", "arms": [], "pairs": [], "gates": [], "sets": ["vita-test", "psp-test"],
        "polarities": ["console", "native"], "primary_polarity": "console",
        "bootstrap": {"reps": 10000, "seed": 0, "min_clusters": 5}, "dialogue": True}
for a in arms:
    f = f"{a}_w8.vocr"
    spec["arms"].append({"name": a, "file": f, "finder": "grad", "role": "vocr6 final test",
                         "sha256": hashlib.sha256(open(f"{blobs}/{f}", "rb").read()).hexdigest()})
for t in "SNL":
    if f"VOCR6_{t}" in arms:
        spec["pairs"].append({"candidate": f"VOCR6_{t}", "baseline": f"VOCR5_{t}"})
json.dump(spec, open(out, "w"), indent=1)
PY
note "path (a): vocr3_report shared (${ARMS[*]})"
run a_shared timeout 7200 "$PY" eval/vocr3_report.py shared --spec "$PRIV/a/spec.json" --blobs "$B" --data "$A/eval" \
  --out "$PRIV/a/shared"
run a_render "$PY" eval/vocr3_report.py render --out "$PRIV/a/shared"

# ---- path (a) 2: the fixed lines with beam 8 + ctclm.2's 1 MB LM at its 1MB settings (vocr_cli), console polarity
LM=$A/ctc-lm-decode/release_pub2/vocr_lm_o3_1000k.vlm
[ "$(sha256sum "$LM" | cut -c1-64)" = 993000611555171daf6361e29a503b0871a5f3dd240a19555d55f16aaa3ab3d9 ] || {
  note "$LM is not the pinned 1 MB LM"; exit 4; }
OPTS="--stdin --topk 8 --beam 8 --alpha 500 --beta 3500 --gate 8000 --lm-flags 2 --bracket 8000 --lm $LM"
run a_make make -C runtime build/vocr_cli build/libvocr.so
note "path (a): beam + 1 MB LM on the fixed lines"
for arm in "${ARMS[@]}"; do
  for s in vita-test psp-test; do
    run "a_lm_${arm}_$s" timeout 3600 "$PY" eval/run.py --model "cmd:runtime/build/vocr_cli $B/${arm}_w8.vocr $OPTS" \
      --set "$s" --data "$A/eval" --polarity console --out "$PRIV/a/lm/${arm}_$s.json"
  done
done
# the dialogue-primary tables (eval/vocr5_dialogue.py), the LM arms beside the greedy ones
$PY - "$PRIV/a/dialogue_pairs.json" "${ARMS[@]}" <<'PY'
import json, sys
arms = sys.argv[2:]
pairs = [[f"VOCR6_{t}", f"VOCR5_{t}"] for t in "SNL" if f"VOCR6_{t}" in arms]
json.dump({"locked_report": {"arms": arms, "pairs": pairs}}, open(sys.argv[1], "w"), indent=1)
PY
LA=(); for arm in "${ARMS[@]}"; do for s in vita-test psp-test; do
  LA+=(--extra "console:${arm}+LM:$s=$PRIV/a/lm/${arm}_$s.json"); done; done
run a_dialogue "$PY" eval/vocr5_dialogue.py tables --protocol "$PRIV/a/dialogue_pairs.json" \
  --report "$PRIV/a/shared/report.json" "${LA[@]}" --out "$PRIV/a/shared/dialogue.json" --md "$PRIV/a/shared/dialogue.md"

# ---- path (a) 3: CLANNAD, the frozen references through eval/clannad_refs.py's extra-arms path (one attempt)
WORK=$($PY -c "import json; print(json.load(open('$P'))['sets']['clannad']['work_dir'])")
C=$PRIV/clannad
EX=(); PAIRS=(); for a in "${ARMS[@]}"; do EX+=(--extra "$a=${a}_w8.vocr#$(sha256sum "$B/${a}_w8.vocr" | cut -c1-64)"); done
for t in S N L; do
  if [ "$MODE" = final ]; then PAIRS+=(--pair "VOCR6_$t:VOCR5_$t" --pair "VOCR6_$t:Lens"); else PAIRS+=(--pair "VOCR5_$t:Lens"); fi
done
note "path (a): CLANNAD extra arms (suffix $SUF)"
run a_clannad_arms timeout 7200 "$PY" eval/clannad_refs.py arms --work "$WORK" --blobs "$B" "${EX[@]}" --suffix "$SUF" \
  --into "$C"
run a_clannad_score "$PY" eval/clannad_refs.py score --work "$WORK" --suffix "$SUF" --into "$C" "${PAIRS[@]}" \
  --out "$C/clannad_$SUF.json"
# the dialogue pairs (not-name lines) from the attempt's per-line errors, the same page bootstrap
run a_clannad_dialogue "$PY" eval/stride16_clannad_dialogue.py --work "$WORK" --private "$C/score.$SUF.private.jsonl" \
  "${PAIRS[@]}" --out "$C/clannad_${SUF}_dialogue.json"

# ---- path (a) 4: the test title PCSG00807 as eval/vocr6_baseline.py read it (docs/vocr6-data.md)
for f in "${FAMS[@]}"; do
  note "path (a): PCSG00807 ($f)"
  MARG=(); [ "$f" = VOCR6 ] && MARG=(--models "$MODELS")
  run "a_pcsg_$f" timeout 7200 "$PY" "$H" pcsg-a --family "$f" "${MARG[@]}" --build-cache "$PRIV/a/pcsg/build" \
    --cache "$PRIV/a/pcsg/cache_$f" --out "$PRIV/a/pcsg/$f.json"
done

# ---- VOCR5's path (a) against the published records (the validation; in the final read, the same check again)
if "$PY" "$H" validate --priv "$PRIV" --suffix "$SUF" --out "$OUT/validation.json" > "$PRIV/logs/validate.log" 2>&1; then
  note "validation: every compared VOCR5 entry equals the published records"
else
  note "validation: a compared entry DIFFERS (see $OUT/validation.json)"
  [ "$MODE" = validate ] && exit 5
fi
if [ "$MODE" = validate ]; then note "validation run finished"; exit 0; fi

# ---- path (b): Rubi's shipped path, resolved from Rubi main now (before any path (b) read)
if [ -z "${RUBI:-}" ]; then
  RUBI=$PRIV/rubi; run b_rubi_clone timeout 600 git clone -q https://github.com/bee-san/Rubi_for_vita.git "$RUBI"
fi
run b_rubi_path "$PY" "$H" rubi-path --rubi "$RUBI" --out "$PRIV/rubi_path.json"
cp "$PRIV/rubi_path.json" "$OUT/rubi_path.json"
note "path (b): $(cat "$PRIV/logs/b_rubi_path.log" | tail -n 1)"
for s in vita-test psp-test clannad pcsg00807; do
  note "path (b): plan $s"
  run "b_plan_$s" timeout 7200 "$PY" "$H" plan --set "$s" --rubi-path "$PRIV/rubi_path.json" --data "$A/eval" \
    --out "$PRIV/b/plan_$s.pkl"
  DUMPS=()
  for f in "${FAMS[@]}"; do
    for t in S N L; do
      if [ "$f" = VOCR5 ]; then BL=$B/VOCR5g_$t.vocr; else BL=$B/VOCR6_${t}_w8.vocr; fi
      run "b_read_${s}_${f}_$t" timeout 7200 "$PY" "$H" read --plan "$PRIV/b/plan_$s.pkl" --blob "$BL" \
        --rubi-path "$PRIV/rubi_path.json" --out "$PRIV/b/dump_${s}_${f}_$t.pkl"
      DUMPS+=(--dump "$f:$t=$PRIV/b/dump_${s}_${f}_$t.pkl")
    done
  done
  run "b_score_$s" "$PY" "$H" score-b --plan "$PRIV/b/plan_$s.pkl" "${DUMPS[@]}" --rubi-path "$PRIV/rubi_path.json" \
    --out "$PRIV/b/rows_$s.pkl"
done

# ---- every table
run report "$PY" "$H" report --priv "$PRIV" --suffix "$SUF" --out "$OUT"
cp "$OUT/report.json" "$OUT/report.md" "$A/vocr6-final-test/"
note "final read finished: $OUT/report.md"
