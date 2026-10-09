#!/usr/bin/env python3
"""Read real screenshots with MeikiOCR and the voter engines (the slow, model-heavy stage).

  label.py fetch                                   pinned model files -> $TEACHER_MODELS
  label.py read --set train|vita-test|psp-test [--shard K --nshards N] [--procs P]
                [--voters v6s,manga,screenai,v5s] [--limit-shots M] [--no-cascade]

For each screenshot, at its native resolution: MeikiOCR detects and reads every text line
(geometry.lines_of merges row fragments, refines ink boxes, attaches furigana); Chrome Screen
AI reads the whole screenshot once and its lines are matched to MeikiOCR's by overlap;
PP-OCRv6 small, manga-ocr and PP-OCRv5 server read each line's voter view (the line plus a
margin, so a line MeikiOCR cut short reads longer for them); MeikiOCR re-reads the view too
("meikiv"). Nothing is decided here: build.py applies the vote (vote.accept) with the
thresholds measure.py chose, so they can change without re-reading.

Inputs:
  train      realdata's screenshot collection (its shots_<kind>.jsonl manifests, images in
             $REALDATA/shots/<kind>/), minus eval's test games (re-checked in build.py).
  vita-test, psp-test   eval's hand-checked sets (images from $VOCR_EVAL_DATA), to measure.
Output: $TEACHER_OUT/reads/<set>.<K>of<N>.jsonl, one row per screenshot (all its lines with
every reading and box), and <set>.<K>of<N>.done (screenshot URLs finished). Every .done of
the set counts on restart, so a killed run resumes even with a different sharding.

Run it offline and inside one heavy.sh slot:
  $HERD/heavy.sh unshare -rn teacher/.venv/bin/python teacher/label.py read --set train --procs 5
`unshare -rn` gives the job a network namespace with no interfaces: no engine can reach the
network. --procs P runs P single-thread workers (the readers scale poorly with threads on
short lines), pinned to the slot's cores.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from teacher import geometry as G  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
OUT = Path(os.environ.get("TEACHER_OUT") or HERD / "shared" / "artifacts" / "teacher")
REALDATA = Path(os.environ.get("REALDATA") or HERD / "shared" / "artifacts" / "realdata")
EVAL_DATA = Path(os.environ.get("VOCR_EVAL_DATA") or HERD / "shared" / "artifacts" / "eval")
KINDS = ("vita", "psp", "hires")
EVAL_SETS = {"vita-test": "vita", "psp-test": "psp"}
SLOW = ("manga", "v5s")


def shots_manifests():
    """realdata's shots_<kind>.jsonl: the merged copy in this checkout if present, else the
    copy taken from realdata's branch into $TEACHER_OUT/inputs/ (see README)."""
    for d in (ROOT / "realdata" / "manifests", OUT / "inputs"):
        paths = [d / f"shots_{k}.jsonl" for k in KINDS]
        if all(p.exists() for p in paths):
            return paths
    sys.exit(f"realdata's shots_<kind>.jsonl not found in {ROOT / 'realdata' / 'manifests'} or {OUT / 'inputs'}")


def eval_manifest(name):
    for d in (ROOT / "eval" / "manifests", OUT / "inputs" / "eval"):
        if (d / f"{name}.jsonl").exists():
            return d / f"{name}.jsonl"
    sys.exit(f"eval manifest {name}.jsonl not found")


def load_shots(set_name):
    """[{url, sha256, vn, kind, dims, release, path}] for a set."""
    rows = []
    if set_name == "train":
        for p in shots_manifests():
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                r["path"] = str(REALDATA / "shots" / r["kind"] / f"{r['vn']}_{r['url'].rsplit('/', 1)[-1]}")
                rows.append(r)
        return rows
    platform = EVAL_SETS[set_name]
    seen = {}
    for line in eval_manifest(set_name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            seen.setdefault(r["url"], r)
    for url, r in seen.items():
        rows.append({"url": url, "sha256": r["sha256"], "vn": r["vn"], "kind": platform, "dims": r["dims"],
                     "release": None,
                     "path": str(EVAL_DATA / platform / "shots" / f"{r['vn']}_{url.rsplit('/', 1)[-1]}")})
    return rows


def shot_key(r):
    return f"{r['vn']}_{r['url'].rsplit('/', 1)[-1].rsplit('.', 1)[0]}"


def shard_of(url, n):
    return int(hashlib.md5(url.encode()).hexdigest(), 16) % n


def japanese_ish(text):
    from teacher import vote
    return vote.japanese(text)


def match_sai(ink, sai):
    """Chrome Screen AI's reading of a line: its horizontal lines that overlap the ink box
    (vertically by at least half the smaller height, horizontally at all, and at least 0.65x
    the ink height, so furigana stays out), joined left to right. A Screen AI line wider
    than MeikiOCR's line brings its extra text, so the vote fails on a truncated line."""
    ih = ink[3] - ink[1]
    hits = [s for s in sai if not s["vertical"] and s["text"] and G.vov(s["box"], ink) >= 0.5
            and min(s["box"][2], ink[2]) - max(s["box"][0], ink[0]) > 0
            and s["box"][3] - s["box"][1] >= 0.65 * ih]
    if not hits:
        return ["", 0.0, 0.0]
    hits.sort(key=lambda s: s["box"][0])
    cs = [s["conf"] for s in hits]
    return ["".join(s["text"] for s in hits), round(sum(cs) / len(cs), 4), round(min(cs), 4)]


def read_shot(r, eng, voters, cascade):
    import cv2
    bgr = cv2.imread(r["path"], cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = bgr[:, :, ::-1]
    H, W = bgr.shape[:2]
    secs = {}

    def timed(name, f, *a):
        t0 = time.perf_counter()
        res = f(*a)
        secs[name] = secs.get(name, 0.0) + time.perf_counter() - t0
        return res

    lines = G.lines_of(timed("meiki", eng["meiki"].ocr, bgr), rgb)
    sai = timed("screenai", eng["screenai"].ocr, bgr) if "screenai" in eng else None
    out = []
    for k, ln in enumerate(lines):
        rec = {"k": k, "det": ln["det"], "det_conf": ln["det_conf"], "parts": ln["parts"], "ink": ln["ink"],
               "crop": ln["crop"], "view": ln["view"], "edge": ln["edge"], "hint": ln["hint"],
               "meiki": {"text": ln["text"], "chars": ln["chars"]}, "reads": {}}
        if ln.get("furi"):
            rec["furi"], rec["furi_text"] = ln["furi"], ln["furi_text"]
        tg = G.targets_of(r["kind"], [W, H], ln["ink"])
        rec["targets"] = tg
        x0, y0, x1, y1 = ln["view"]
        view = bgr[y0:y1, x0:x1]
        if sai is not None:
            rec["reads"]["screenai"] = match_sai(ln["ink"], sai)
        if view.size and (tg and not ln["edge"] or not cascade):
            t, mean, low, chars = timed("meikiv", eng["meiki"], view)
            rec["reads"]["meikiv"] = [t, mean, low, [[c[0], c[1] + x0, c[2] + y0, c[3] + x0, c[4] + y0, c[5]]
                                                     for c in chars]]
        # no rule accepts a line without a target, at the screen edge, or with no Japanese reading
        skip = cascade and (not tg or ln["edge"] or not (
            japanese_ish(ln["text"]) or japanese_ish(rec["reads"].get("meikiv", [""])[0])))
        if not skip and view.size:
            for name in voters:
                if name != "screenai":
                    rec["reads"][name] = timed(name, eng[name], view)
        elif skip:
            rec["skip"] = True
        out.append(rec)
    row = {"shot": shot_key(r), "url": r["url"], "sha256": r.get("sha256"), "vn": r["vn"], "kind": r["kind"],
           "release": r.get("release"), "session": r.get("session"), "dims": [W, H], "lines": out,
           "secs": {k: round(v, 3) for k, v in secs.items()}}
    if sai is not None:
        row["sai"] = sai
    return row


def pin_to_slot():
    """Keep this job (and its children) on the 5 cores of its heavy.sh slot. Workers that
    spawn() started inherit the set and don't re-pin."""
    slot = os.environ.get("HEAVY_SLOT")
    if not slot or os.environ.get("TEACHER_PINNED") or not hasattr(os, "sched_setaffinity"):
        return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    n = os.cpu_count() or 16
    per = max(1, int(os.environ.get("OMP_NUM_THREADS", "5")))
    cores = {((int(slot) - 1) * per + i) % n for i in range(per)}
    os.sched_setaffinity(0, cores)
    os.environ["TEACHER_PINNED"] = "1"
    return sorted(cores)


def spawn(args):
    n = args.procs
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCH_NUM_THREADS"):
        env[var] = "1"
    skip = {"--procs", "--shard", "--nshards"}
    rest, it = [], iter(sys.argv[2:])
    for a in it:
        if a.split("=")[0] in skip:
            if "=" not in a:
                next(it, None)
            continue
        rest.append(a)
    reads = Path(args.reads)
    reads.mkdir(parents=True, exist_ok=True)
    procs = []
    for i in range(n):
        k, nn = args.shard * n + i, args.nshards * n
        cmd = [sys.executable, str(Path(__file__).resolve()), "read", "--shard", str(k), "--nshards", str(nn), *rest]
        log = open(reads / f"worker.{args.set}.{k}of{nn}.log", "a")
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL))
    print(f"{n} workers, 1 thread each: pids {[p.pid for p in procs]}", flush=True)
    rcs = [p.wait() for p in procs]
    print(f"workers done, exit codes {rcs}", flush=True)
    if any(rcs):
        sys.exit(1)
    print("READ_DONE", flush=True)


def cmd_read(args):
    cores = pin_to_slot()
    if args.procs > 1:
        print(f"cores {cores}", flush=True)
        return spawn(args)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    out = Path(args.reads) / f"{args.set}.{args.shard}of{args.nshards}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done_path = out.with_suffix(".done")
    done = set()
    for p in sorted(out.parent.glob(f"{args.set}.*.done")):
        done |= set(p.read_text().split())
    rows = [r for r in load_shots(args.set) if shard_of(r["url"], args.nshards) == args.shard
            and r["url"] not in done and Path(r["path"]).exists()]
    if args.set == "train":                 # never even read a test game (build.py re-checks)
        from teacher import build
        ex = build.eval_exclusions()
        rows = [r for r in rows if not build.excluded(r, ex)]
    if args.kinds:
        rows = [r for r in rows if r["kind"] in args.kinds.split(",")]
    if args.limit_shots:
        rows = rows[:args.limit_shots]
    from teacher import engines
    voters = [v for v in args.voters.split(",") if v]
    eng = engines.load(["meiki", *voters])
    print(f"{args.set} shard {args.shard}/{args.nshards}: {len(rows)} screenshots, voters {voters}, "
          f"threads {engines.threads()}, cores {cores}", flush=True)
    t0, n_lines = time.time(), 0
    with open(out, "a", encoding="utf-8") as fo, open(done_path, "a") as fd:
        for i, r in enumerate(rows):
            row = read_shot(r, eng, voters, not args.no_cascade)
            if row is None:
                print(f"unreadable: {r['path']}", flush=True)
            else:
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                fo.flush()
                n_lines += len(row["lines"])
            fd.write(r["url"] + "\n")
            fd.flush()
            if i % 20 == 0:
                dt = time.time() - t0
                print(f"{i + 1}/{len(rows)} shots, {n_lines} lines, {dt:.0f}s, {dt / (i + 1):.2f}s/shot", flush=True)
    print(f"done: {len(rows)} shots, {n_lines} lines in {time.time() - t0:.0f}s", flush=True)
    print("READ_DONE", flush=True)


def cmd_augment(args):
    """Read extra voters on lines already read, but only where they can still change the
    decision of the rule in --rule (vote.undecided); rewrites the reads files in place
    (atomically, per file). Used to add a slow voter after the rule has been measured."""
    import cv2

    from teacher import engines, vote
    pin_to_slot()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    rule = json.loads(Path(args.rule).read_text(encoding="utf-8"))["rule"]
    names = [v for v in args.voters.split(",") if v]
    eng = engines.load(names)
    paths = sorted(Path(args.reads).glob(f"{args.set}.*.jsonl"))
    paths = [p for i, p in enumerate(paths) if i % args.nshards == args.shard]
    t0, n = time.time(), 0
    for p in paths:
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        for r in rows:
            todo = [ln for ln in r["lines"] if not ln.get("skip") and any(v not in ln["reads"] for v in names)
                    and vote.undecided(ln, rule)]
            if not todo:
                continue
            src = next(s for s in load_shots(args.set) if s["url"] == r["url"]) if args.set != "train" else \
                {"path": str(REALDATA / "shots" / r["kind"] / f"{r['vn']}_{r['url'].rsplit('/', 1)[-1]}")}
            bgr = cv2.imread(src["path"], cv2.IMREAD_COLOR)
            for ln in todo:
                x0, y0, x1, y1 = ln["view"]
                for v in names:
                    if v not in ln["reads"] and vote.undecided(ln, rule):
                        ln["reads"][v] = eng[v](bgr[y0:y1, x0:x1])
                        n += 1
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        tmp.replace(p)
        print(f"{p.name}: {n} reads so far, {time.time() - t0:.0f}s", flush=True)
    print("AUGMENT_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--mirror", action="append", default=[], help="directory with already-downloaded model files")
    p = sub.add_parser("augment")
    p.add_argument("--set", default="train", choices=["train", *EVAL_SETS])
    p.add_argument("--voters", default="v5s")
    p.add_argument("--rule", default=str(HERE / "rule.json"))
    p.add_argument("--reads", default=str(OUT / "reads"))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p = sub.add_parser("read")
    p.add_argument("--set", default="train", choices=["train", *EVAL_SETS])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--procs", type=int, default=1)
    p.add_argument("--voters", default="screenai,v6s,manga,v5s")
    p.add_argument("--kinds", default="", help="train: only these kinds (vita,psp,hires)")
    p.add_argument("--limit-shots", type=int, default=0)
    p.add_argument("--reads", default=str(OUT / "reads"), help="output directory")
    p.add_argument("--no-cascade", action="store_true",
                   help="read every line with every voter (default: skip lines no rule can accept)")
    args = ap.parse_args()
    if args.cmd == "fetch":
        from teacher import engines
        engines.fetch(mirrors=args.mirror)
        print("FETCH_DONE", flush=True)
    elif args.cmd == "augment":
        cmd_augment(args)
    else:
        cmd_read(args)


if __name__ == "__main__":
    main()
