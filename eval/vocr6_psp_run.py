#!/usr/bin/env python3
"""Model B (vita-vn-ocr #126): one protocol run, end to end, in one heavy slot.

  vocr6_psp_run.py ARM SEED [--final] [--winner ARM]      train, export int8, score dev + hval, log
  vocr6_psp_run.py ARM SEED --probe STEPS --out DIR       the training-speed probe (not a run: nothing kept)
  vocr6_psp_run.py ARM SEED --dry                         print the train2 command only

ARM and its seeds, init, shape and sources come from eval/vocr6_psp_protocol.json (ARM Q is the stage-1 winner's
recipe, --winner, with --qat 0.7). Outputs: $HERD/shared/artifacts/vocr6-psp/runs/<run>/ (train2's directory, the
int8 blob <run>_w8.vocr, the score rows dev.json / hval.json), and one line in vocr6/experiments.jsonl.
"""
import argparse
import datetime
import fcntl
import glob
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
A = HERD / "shared/artifacts"
ME = A / "vocr6-psp"
PROTO = ROOT / "eval" / "vocr6_psp_protocol.json"
EXPLOG = Path("/local/home/skerraut/herd-synth3/vocr6/experiments.jsonl")
ING = A / "vocr6-ingest/pools/v1"
VOCAB = ROOT / "data" / "vocab_4987_proof.txt"
VOCAB_SHA = "4a1e7cb77c627a217ae57c7ed427f0695c732dfb85158c674ec4d14994a4e04c"
INIT = {"S_d0c2": ME / "init/S_d0c2.pt", "S_d0c1": ME / "init/S_d0c1.pt", "S_d1c2": ME / "init/S_d1c2.pt",
        "S_d1c1": ME / "init/S_d1c1.pt"}
PSP_TITLE = ("NPJ", "ULJ", "ULS", "NPH", "UCJ")
NATIVE_INK = ("tt", "tvt", "rd", "rvt", "nat_ink", "dntv_ink", "dnst_ink", "own_ink", "own_m4", "own_m8")
NATIVE_GRAD = ("nat_grad", "nat_ref", "dntv_grad", "dntv_ref", "dnst_grad", "dnst_ref", "own_grad", "own_ref")
NATIVE_PRIVATE = HERD / "shared/private/vocr6-psp/pools/native"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def psp_files(split, pools=("line", "name", "furimain")):
    out = []
    for pool in pools:
        out += [f for f in sorted(glob.glob(str(ING / split / pool / "*.npz"))) if Path(f).stem.startswith(PSP_TITLE)]
    return out


def native(names):
    out = []
    for n in names:
        p = ME / "pools/native" / f"{n}_psp.npz"
        q = NATIVE_PRIVATE / f"{n}_psp.npz"
        out.append(str(q if q.exists() else p))
    for f in out:
        if not Path(f).exists():
            raise SystemExit(f"missing native pool {f}")
    return out


def sources(proto, final):
    s = proto["data"]["sources"]
    hk = psp_files("train") + (psp_files("guard") + psp_files("hval_fold") if final else [])
    hc = sorted(glob.glob(str(ME / "pools/console/train/*.npz")))
    if final:
        hc += sorted(glob.glob(str(ME / "pools/console/guard/*.npz"))) + \
            sorted(glob.glob(str(ME / "pools/console/hval/*.npz")))
    nvg = f"{A}/nvsynth/full/train/nv_train_0[0-3]?_b*.npz+{A}/nvsynth/full/train/nv_train_04[01]_b*.npz"
    s2g = f"{A}/synth2/shards/synth_000[0-2]?.npz+{A}/synth2/shards/synth_0003[0-8].npz"
    s3g = f"{A}/synth3/shards/train_v2/synth3_000??.npz"
    files = {"hk": "+".join(hk), "hc": "+".join(hc), "npi": "+".join(native(NATIVE_INK)),
             "npg": "+".join(native(NATIVE_GRAD)), "nv": nvg, "s2": s2g, "s3": s3g}
    args, record = [], {}
    for name in ("hk", "hc", "npi", "npg", "nv", "s2", "s3"):
        args += ["--src", f"{name}:{s[name]['weight']}:{files[name]}:{s[name]['opts']}"]
        if name in ("hk", "hc", "npi", "npg"):
            fl = files[name].split("+")
            record[name] = {"files": len(fl), "sha256": hashlib.sha256(
                "".join(sha256(f) for f in fl).encode()).hexdigest()}
    return args, record


def check_synth3():
    m = json.loads((ROOT / "data/manifests/synth3_shards_v2.json").read_text())
    d = A / "synth3/shards/train_v2"
    bad = [s["file"] for s in m["shards"] if sha256(d / s["file"]) != s["sha256"]]
    if bad:
        raise SystemExit(f"synth3 v2 shards differ from their manifest: {bad[:3]}")


def arm_of(proto, arm, winner):
    if arm == "Q":
        if not winner:
            raise SystemExit("arm Q needs --winner (the stage-1 winner)")
        w = dict(proto["arms"][winner])
        w["train2_args"] += " --qat 0.7"
        w["seeds"] = proto["arms"]["Q"]["seeds"]
        return w
    return proto["arms"][arm]


def log(rec):
    EXPLOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPLOG, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fcntl.flock(fh, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("arm")
    ap.add_argument("seed", type=int)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--winner", default="")
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    proto = json.loads(PROTO.read_text())
    arm = arm_of(proto, a.arm, a.winner)
    seeds = [1, 2, 3] if a.final else arm["seeds"]
    if a.seed not in seeds:
        raise SystemExit(f"seed {a.seed} is not one of {seeds}")
    if sha256(VOCAB) != VOCAB_SHA:
        raise SystemExit("the vocab differs from VOCR5-S's")
    init = INIT[arm["init"]]
    steps = 20000 if a.final else 10000
    run = f"mb{'F' if a.final else ''}_{a.arm}{('-' + a.winner) if a.arm == 'Q' else ''}_seed{a.seed}"
    out = Path(a.out) if a.probe else ME / "runs"
    srcs, src_rec = sources(proto, a.final)
    cmd = [str(ROOT / ".venv-ci/bin/python"), str(ROOT / "train/train2.py"), "--run", run, "--out", str(out),
           "--init", str(init), "--steps", str(a.probe or steps), "--batch", "96", "--lr", "1.2e-3", "--warmup", "500",
           "--final-lr", "0.05", "--wd", "0.01", "--ema", "0.999", *arm["train2_args"].split(), "--vocab", str(VOCAB),
           "--seed", str(a.seed), "--threads", "5", *srcs]
    if a.probe:
        cmd += ["--eval-every", str(10 ** 9), "--log-every", "50"]
    else:
        cmd += ["--eval-every", "2500", "--val", f"hv:{ME / 'pools/hval_lines_psp.npz'}:n=1200"]
    if a.dry:
        print(" ".join(cmd))
        return
    rdir = out / run
    if not a.probe and (rdir / f"ckpt_step{steps}.pt").exists() is False and rdir.exists():
        raise SystemExit(f"{rdir} is partial: delete it and restart (runs never resume)")
    check_synth3()
    t0 = time.time()
    start = now()
    print(f"git {subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()}"
          f" {start} {run}: {' '.join(cmd[:20])} ...", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode:
        raise SystemExit(f"train2 failed rc {r.returncode}")
    wall = time.time() - t0
    if a.probe:
        shutil.rmtree(rdir, ignore_errors=True)
        print(f"probe complete: {run} {wall:.0f} s")
        return
    ck = rdir / "ckpt_last.pt"
    blob = rdir / f"{run}_w8.vocr"
    py = str(ROOT / ".venv/bin/python")
    subprocess.run([py, str(ROOT / "train/export.py"), "--ckpt", str(ck), "--out", str(blob), "--int8"], check=True,
                   cwd=ROOT)
    for plan in ("dev", "hval"):
        subprocess.run([py, str(ROOT / "eval/modelb_eval.py"), "read", "--plan", str(ME / f"eval/{plan}_plan.pkl"),
                        "--blob", str(blob), "--mode", "w8f", "--out", str(rdir / f"{plan}.json")], check=True,
                       cwd=ROOT)
    dev = json.loads((rdir / "dev.json").read_text())["totals"]
    hv = json.loads((rdir / "hval.json").read_text())["totals"]
    sc = 0.5 * hv["hp_grad"]["cer"] + 0.25 * dev["wd_grad"]["cer"] + 0.25 * dev["fixed"]["cer"]
    rec = {"label": "vocr6-psp", "arm": a.arm + (f"({a.winner})" if a.arm == "Q" else ""), "tier": "modelB",
           "seed": a.seed, "final": a.final, "config_hash": hashlib.sha256(" ".join(cmd).encode()).hexdigest()[:16],
           "data_manifest_hash": hashlib.sha256(json.dumps(src_rec, sort_keys=True).encode()).hexdigest()[:16],
           "steps": steps, "wall_clock_s": round(wall), "slot_hours": round(wall / 3600, 2), "checkpoint": str(ck),
           "blob": str(blob), "blob_sha256": sha256(blob), "blob_bytes": blob.stat().st_size,
           "hval_cer": {"clean_lines": hv["hl"]["cer"], "page_grad_refine": hv["hp_grad"]["cer"],
                        "page_recovery_refine": hv["hp_recovery"]["cer"]},
           "dev_cer": {"native_psp_wd_grad_refine": dev["wd_grad"]["cer"],
                       "native_psp_fixed_dialogue": dev["fixed"]["cer"],
                       "native_psp_wd_recovery_refine": dev["wd_recovery"]["cer"],
                       "native_psp_names": dev["names"]["cer"]},
           "screening_score": round(sc, 4), "blank_share_dev_wd": dev["wd_grad"].get("blank_share"),
           "started": start, "time": now(),
           "notes": f"model B (#126) {arm['train2_args']} init {arm['init']}; W8F greedy, no gate (screening); "
                    "eval/vocr6_psp_protocol.json"}
    log(rec)
    (rdir / "run_record.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"run complete: {run} SC {sc:.4f} hp {hv['hp_grad']['cer']} wd {dev['wd_grad']['cer']} "
          f"fixed {dev['fixed']['cer']}")


if __name__ == "__main__":
    main()
