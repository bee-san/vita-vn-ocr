#!/usr/bin/env python3
"""One VOCR6 T1 run (issue #125, eval/vocr6_protocol.json): train it, export its final step to int8, score it in the
shipped form (eval/vocr6_score.py all) and log it in the fleet's experiment log.

  vocr6_run.py run --stage S --arm ARM --seed N --label LABEL [--protocol P] [--dry-run]
  vocr6_run.py finish --stage S --arm ARM --seed N --label LABEL      (export + score + log an already trained run)
  vocr6_run.py command --stage S --arm ARM --seed N --label LABEL     (print the train2 command only)

Run it in one heavy slot: HEAVY_SLOTS=12 $HERD/heavy.sh timeout SECS .venv-ci/bin/python eval/vocr6_run.py run ...
The run is <arm.run>_seed<N> (e.g. v6s1A15_N_seed1) under $HERD/shared/artifacts/<LABEL>/runs/; its score files go
to .../<LABEL>/scores/. The model is the run's final step (ckpt_step<steps>.pt): nothing is selected inside a run (its
--val sets are logged only). A run that stopped after a checkpoint resumes from ckpt_last.pt (train2's resume, which
reseeds the batch stream; the log line records it).

Inputs are checked before training: the vocab's sha256, the init checkpoint's sha256 (protocol), every hooked pool
file against vocr6-ingest's manifest, VOCR5's pinned pools (eval/vocr5_pools.sha256), and the synthetic shard
manifests' file sizes (their hashes were checked once when the protocol was registered).
"""
import argparse
import datetime
import fcntl
import glob
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
EXPLOG = Path(os.environ.get("VOCR6_EXPLOG", "/local/home/skerraut/herd-synth3/vocr6/experiments.jsonl"))
PY = sys.executable


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def jdigest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def load_protocol(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arm_spec(proto, stage, arm):
    st = proto["stages"][str(stage)]
    if "arms" not in st:
        raise SystemExit(f"stage {stage} has no arms yet (they are filled when the previous stage decides)")
    if arm not in st["arms"]:
        raise SystemExit(f"stage {stage} has no arm {arm} (has {sorted(st['arms'])})")
    if st["arms"][arm].get("stopped"):
        raise SystemExit(f"stage {stage} arm {arm} is stopped: {st['arms'][arm]['stopped']}")
    spec = dict(st.get("defaults", {}))
    spec.update(st["arms"][arm])
    spec["tier"] = spec.get("tier", st.get("tier", "N"))
    base = st.get("base") or {}
    uses_base = [k for k in ("init", "mix") if spec.get(k) and spec[k] == base.get("arm")]
    if uses_base:
        # the stage's base arm (e.g. W15 or R2, chosen by the previous stage's rule): its init per seed and/or its
        # mix, once the protocol records them
        filled = base.get("filled")
        if not filled:
            raise SystemExit(f"stage {stage}'s base {base['arm']} is not filled yet (protocol stages.{stage}.base)")
        if "init" in uses_base:
            spec["init"], spec["init_sha256"] = filled["init"], filled["init_sha256"]
        if "mix" in uses_base:
            spec["mix"] = filled["mix"]
        spec["base"] = filled.get("arm")
    return st, spec


def expand(globs):
    out = []
    for g in globs:
        hit = sorted(glob.glob(g))
        if not hit:
            raise SystemExit(f"no files match {g}")
        out += hit
    return out


def mix_sources(proto, mix_name):
    """[(name, weight, [files], opts)] of a mix: protocol "mixes"[name], each entry {name, weight, globs, opts}.
    An entry with "per_title": T (a temperature) becomes one source per title (its files grouped by file name, the
    title id), weighted by rows^T over the titles and summing to the entry's weight: T = 1 is the pooled source's
    expected mix, T = 0 gives every title the same share. Rows come from vocr6-ingest's manifest."""
    out = []
    for s in proto["mixes"][mix_name]:
        if float(s["weight"]) <= 0:
            continue
        files = expand(s["globs"])
        if "per_title" not in s:
            out.append((s["name"], float(s["weight"]), files, s.get("opts", "")))
            continue
        pins = proto["data_pins"]["hooked"]
        man = json.loads(Path(pins["manifest"]).read_text(encoding="utf-8"))
        rows = {str(p.resolve()): int(r) for p, r in man_rows(man, Path(pins["manifest"]).parent)}
        by_title = {}
        for f in files:
            by_title.setdefault(Path(f).stem, []).append(f)
        n = {t: sum(rows[str(Path(f).resolve())] for f in fs) for t, fs in by_title.items()}
        temp = float(s["per_title"])
        w = {t: n[t] ** temp for t in by_title if n[t] > 0}
        z = sum(w.values())
        for t in sorted(w):
            out.append((f"{s['name']}_{t}", float(s["weight"]) * w[t] / z, by_title[t], s.get("opts", "")))
    return out


def man_rows(man, base):
    files = man.get("files") or {}
    for p, h in files.items():
        q = Path(p)
        yield (q if q.is_absolute() else base / q), (h.get("rows", 0) if isinstance(h, dict) else 0)


def check_inputs(proto, spec, sources, seed):
    """Every pinned input as the protocol registered it. Returns {what: sha256 or count} for the log."""
    rec = {}
    tier = proto["tiers"][spec["tier"]]
    vocab = ROOT / tier["vocab"]["path"]
    if sha256(vocab) != tier["vocab"]["sha256"]:
        raise SystemExit(f"{vocab} differs from the protocol's sha256")
    rec["vocab_sha256"] = tier["vocab"]["sha256"]
    if spec.get("init"):
        init = spec["init"].format(seed=seed)
        want = (spec.get("init_sha256") or {}).get(str(seed))
        got = sha256(init)
        if want and got != want:
            raise SystemExit(f"{init} differs from the protocol's sha256")
        rec["init_sha256"] = got
    pins = proto["data_pins"]
    man = json.loads(Path(pins["hooked"]["manifest"]).read_text(encoding="utf-8"))
    if sha256(pins["hooked"]["manifest"]) != pins["hooked"]["sha256"]:
        raise SystemExit("vocr6-ingest's pool manifest differs from the protocol's sha256")
    known = {str(Path(p).resolve()): h for p, h in man_files(man, Path(pins["hooked"]["manifest"]).parent)}
    hooked = pins["hooked"]["sources"]
    hooked_files = [f for name, _, files, _ in sources
                    if any(name == h or name.startswith(h + "_") for h in hooked) for f in files]
    for f in hooked_files:
        h = known.get(str(Path(f).resolve()))
        if h is None or sha256(f) != h:
            raise SystemExit(f"{f} is not in vocr6-ingest's manifest with this sha256")
    rec["hooked_files"] = len(hooked_files)
    rec["hooked_manifest_sha256"] = pins["hooked"]["sha256"]
    if any(name in pins["vocr5_pools"]["sources"] for name, _, _, _ in sources):
        r = subprocess.run(["sha256sum", "--quiet", "-c", str(ROOT / pins["vocr5_pools"]["pin"])], cwd="/",
                           capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"VOCR5's pinned pools differ: {r.stdout[-500:]} {r.stderr[-500:]}")
        rec["vocr5_pools_pin_sha256"] = sha256(ROOT / pins["vocr5_pools"]["pin"])
    for m in pins.get("synthetic_manifests", []):
        if m.get("sha256") and sha256(ROOT / m["manifest"]) != m["sha256"]:
            raise SystemExit(f"{m['manifest']} differs from the protocol's sha256")
        doc = json.loads((ROOT / m["manifest"]).read_text(encoding="utf-8"))
        bad = [s["file"] for s in doc["shards"] if "bytes" in s and
               os.path.getsize(Path(m["dir"]) / s["file"]) != int(s["bytes"])]
        if bad:
            raise SystemExit(f"{m['manifest']}: {len(bad)} shards have another size")
    used = {f for _, _, files, _ in sources for f in files}
    for p, h in (pins.get("guarded") or {}).get("files", {}).items():
        if p in used and sha256(p) != h:
            raise SystemExit(f"{p} differs from the protocol's sha256 (data_pins.guarded)")
    check_extra_pools(pins, sources, rec)
    return rec


def check_extra_pools(pins, sources, rec):
    """Another track's pools (data_pins.extra_pools, e.g. vocr6-finder's console crops v1 and v2): each manifest by
    sha256, then every file of a source the pins name against the manifest whose directory holds it (one version may
    be pinned beside another: a file is checked against its own version's manifest, and a file under none fails)."""
    pinned = []
    for pin in pins.get("extra_pools", []):
        mpath = Path(pin["manifest"])
        if sha256(mpath) != pin["sha256"]:
            raise SystemExit(f"{mpath} differs from the protocol's sha256 (data_pins.extra_pools)")
        listed = {str((mpath.parent / q).resolve()): (h["sha256"] if isinstance(h, dict) else h)
                  for q, h in json.loads(mpath.read_text(encoding="utf-8"))["files"].items()}
        pinned.append((pin, mpath.parent.resolve(), listed))
    names = {s for pin, _, _ in pinned for s in pin["sources"]}
    for name, _, files, _ in sources:
        if name not in names:
            continue
        for f in files:
            fp = Path(f).resolve()
            own = [(pin, listed) for pin, root, listed in pinned if name in pin["sources"] and fp.is_relative_to(root)]
            if not own:
                raise SystemExit(f"{f} (source {name}) lies under no pinned manifest's directory")
            pin, listed = own[0]
            want = listed.get(str(fp))
            if want is None or sha256(f) != want:
                raise SystemExit(f"{f} is not in {pin['manifest']} with this sha256")
            rec[f"{pin['name']}_files"] = rec.get(f"{pin['name']}_files", 0) + 1
            rec[f"{pin['name']}_manifest_sha256"] = pin["sha256"]
    return rec


def man_files(man, base):
    """(path, sha256) of every file vocr6-ingest's manifest lists (absolute, or relative to the manifest)."""
    files = man.get("files") or {}
    items = files.items() if isinstance(files, dict) else ((f["path"], f["sha256"]) for f in files)
    for p, h in items:
        q = Path(p)
        yield (q if q.is_absolute() else base / q), (h["sha256"] if isinstance(h, dict) else h)


def run_name(spec, seed):
    return f"{spec['run']}_seed{seed}"


def dirs(label):
    base = HERD / "shared" / "artifacts" / label
    return base / "runs", base / "scores", base / "blobs"


def train_command(proto, st, spec, seed, label):
    runs, _, _ = dirs(label)
    tier = proto["tiers"][spec["tier"]]
    sources = mix_sources(proto, spec["mix"])
    cmd = [PY, str(ROOT / "train/train2.py"), "--run", run_name(spec, seed), "--out", str(runs),
           "--steps", str(spec["steps"]), "--batch", str(spec.get("batch", 96)), "--lr", str(spec["lr"]),
           "--warmup", str(spec["warmup"]), "--final-lr", str(spec.get("final_lr", 0.05)),
           "--wd", str(spec.get("wd", 0.01)), "--ema", str(spec.get("ema", 0.999)), *tier["arch"],
           *spec.get("arch_extra", []), "--vocab", str(ROOT / tier["vocab"]["path"]), "--seed", str(seed),
           "--threads", "5", "--eval-every", str(spec.get("eval_every", 5000)), "--keep-evals"]
    if spec.get("init"):
        cmd += ["--init", spec["init"].format(seed=seed)]
    for name, w, files, opts in sources:
        cmd += ["--src", f"{name}:{w}:{'+'.join(files)}" + (f":{opts}" if opts else "")]
    for v in proto.get("log_vals", []):
        cmd += ["--val", v]
    if proto.get("log_vals"):
        cmd += ["--select", proto["log_select"]]
    cmd += spec.get("train_extra", [])
    return cmd, sources


def explog(rec):
    EXPLOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPLOG, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def train(args, proto, st, spec):
    cmd, sources = train_command(proto, st, spec, args.seed, args.label)
    runs, scores, blobs = dirs(args.label)
    name = run_name(spec, args.seed)
    rd = runs / name
    final = rd / f"ckpt_step{spec['steps']}.pt"
    if args.dry_run:
        print(" ".join(shlex.quote(c) for c in cmd))
        return
    if final.exists():
        print(f"{name} is trained ({final})", flush=True)
        return
    inputs = check_inputs(proto, spec, sources, args.seed)
    resumed = (rd / "ckpt_last.pt").exists()
    rd.mkdir(parents=True, exist_ok=True)
    meta = rd / "vocr6_run.json"
    info = json.loads(meta.read_text()) if meta.exists() else {"segments": []}
    info.update({"run": name, "stage": args.stage, "arm": args.arm, "seed": args.seed, "label": args.label,
                 "tier": spec["tier"], "spec": spec, "config_sha256": jdigest(spec), "inputs": inputs,
                 "protocol_sha256": sha256(args.protocol), "command": cmd,
                 "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                       text=True).stdout.strip()})
    seg = {"start": now(), "resumed": resumed, "slot": os.environ.get("HEAVY_SLOT")}
    info["segments"].append(seg)
    meta.write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n")
    explog({"label": args.label, "status": "started", "stage": args.stage, "arm": args.arm, "tier": spec["tier"],
            "seed": args.seed, "run": name, "config_sha256": info["config_sha256"],
            "data_manifest_sha256": inputs.get("hooked_manifest_sha256"), "steps": spec["steps"], "resumed": resumed,
            "at": seg["start"], "checkpoint_dir": str(rd)})
    t0 = time.time()
    with open(rd / "train.log", "a") as log:
        log.write(f"# {now()} vocr6_run {' '.join(shlex.quote(c) for c in cmd)}\n")
        log.flush()
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    seg.update({"end": now(), "seconds": round(time.time() - t0), "rc": rc})
    meta.write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n")
    if rc != 0 or not final.exists():
        raise SystemExit(f"train2 ended rc={rc} without {final}: see {rd / 'train.log'}")


def finish(args, proto, st, spec):
    runs, scores, blobs = dirs(args.label)
    name = run_name(spec, args.seed)
    rd = runs / name
    final = rd / f"ckpt_step{spec['steps']}.pt"
    if not final.exists():
        raise SystemExit(f"{name} has not finished ({final})")
    blobs.mkdir(parents=True, exist_ok=True)
    blob = blobs / f"{name}_w8.vocr"
    if not blob.exists():
        subprocess.run([PY, str(ROOT / "train/export.py"), "--ckpt", str(final), "--out", str(blob), "--int8"],
                       check=True, cwd=ROOT)
    t0 = time.time()
    subprocess.run([PY, str(ROOT / "eval/vocr6_score.py"), "all", "--name", name, "--blob", str(blob),
                    "--protocol", str(args.protocol), "--out", str(scores), "--workers", "5"], check=True, cwd=ROOT)
    sc = json.loads((scores / f"score_{name}.json").read_text(encoding="utf-8"))
    info = json.loads((rd / "vocr6_run.json").read_text())
    secs = sum(s.get("seconds", 0) for s in info["segments"])
    rec = {"label": args.label, "status": "done", "stage": args.stage, "arm": args.arm, "tier": spec["tier"],
           "seed": args.seed, "run": name, "config_sha256": info["config_sha256"],
           "data_manifest_sha256": info["inputs"].get("hooked_manifest_sha256"), "steps": spec["steps"],
           "wall_clock_s": secs, "slot_hours": round(secs / 3600, 2), "score_s": round(time.time() - t0),
           "checkpoint": str(final), "blob": str(blob), "blob_sha256": sc["blob_sha256"],
           "blob_bytes": sc["blob_bytes"], "arena_bytes": sc["arena_bytes"],
           "hval_clean_cer": sc["summary"]["rubi"].get("hval_clean_dlg"),
           "hval_page_cer": sc["summary"]["rubi"].get("hval_page"),
           "hval_page_all_cer": sc["summary"]["rubi"].get("hval_page_all"),
           "dev_score": sc["summary"]["rubi"].get("dev_score"), "summary": sc["summary"],
           "score_file": str(scores / f"score_{name}.json"), "at": now(),
           "notes": "W8I + Rubi per-line decode (no cascade); final step; dev without the hooked titles' VNs"}
    explog(rec)
    print("run finished:", name, json.dumps(rec["summary"]["rubi"]), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "finish", "command"))
    ap.add_argument("--stage", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--protocol", default=str(ROOT / "eval/vocr6_protocol.json"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    proto = load_protocol(a.protocol)
    st, spec = arm_spec(proto, a.stage, a.arm)
    seeds = st.get("seeds")
    if seeds and a.seed not in seeds:
        raise SystemExit(f"seed {a.seed} is not one of stage {a.stage}'s ({seeds})")
    if spec.get("seed_only") and a.seed != int(spec["seed_only"]):
        raise SystemExit(f"arm {a.arm} runs seed {spec['seed_only']} only")
    if a.cmd == "command":
        a.dry_run = True
        train(a, proto, st, spec)
    elif a.cmd == "run":
        train(a, proto, st, spec)
        if not a.dry_run:
            finish(a, proto, st, spec)
    else:
        finish(a, proto, st, spec)


if __name__ == "__main__":
    main()
