#!/usr/bin/env python3
"""Train one family B run of the full-frame text spotter (issue #104, docs/textspot.md, eval/textspot_protocol.json
"candidates.family_b"; runs are named <scale>_<shape>_seed<n>).

  textspot_train.py --shape S --scale h2|f1 --seed N --manifest M ... --shards DIR ... --out RUN \
                    [--steps 30000] [--threads 5]
  textspot_train.py ... --probe 200      speed probe: one warm-up step, then 200 steps of the run, timed
                                         (RUN/probe.json; no model is written)

This is #97's trainer (train/textdet_train.py) on whole frames. Its targets (the round-1 kernel rule, not
--strict-kernel), losses, optimiser and schedule (AdamW 4e-3, 500 warm-up steps, cosine to 1e-4, weight decay on
conv weights, gradient clip 5), QAT (activation scales from 32 batches, BatchNorm folded, int8 fake quantisation
from 80 % of the run: step 24,000 of 30,000), checkpoint, export (train/textdet_export.py) and log records are
imported from it and used unchanged. What the protocol's recipe changes is here:

- Every canvas is a whole frame. With --scale h2 a canvas wider than 600 px (the VTSP blob's half_above) is first
  halved by synth/textspot_twin.halve, the runtime's exact 2x2 mean, and its boxes and ignore areas are mapped to
  the halved image: start floor(v / 2), end ceil(v / 2), which is exactly where the halved ink of [v0, v1) lies.
  So the network trains at the scale it runs at. f1 never halves.
- Windows of 256 x 512 at that scale (#97: 192 x 512). A canvas that fits is used whole, top left, and every
  layer's output is masked to it as #97 does; a bigger one gives a window at a uniformly random offset (each side
  whole where it fits). No player-like region crops.
- 16 windows per batch, each drawn from #97's mix: half synthetic (samples whose src starts with "synth"), half
  real (realplus 0.35, teacher 0.35, owner 0.25, devnative 0.05 of the real half).
- Every line of every class is a positive (the network has no class head: the class is not read); ignore areas
  carry no loss; a line under 4 px across at the network's scale (the decode drops such boxes) becomes an ignore
  area. A line cut across by the window becomes an ignore area and one cut along it keeps its inside part, as #97.

Data: the shards (#97's format, synth/textdet_data.py) of every --manifest that lie in one of the --shards
directories, matched by the directory's name as #97's trainer matches them; a manifest's shards in a directory
that is not given are not read (run.json lists what was read and what was not). Every shard read is checked
against its manifest's npy and json sha256 before training. All five sources of the mix must be present (tests
pass --allow-missing-sources, and #97's Pool then renormalises the mix over the sources present, as #97's does).

Skip lists: the textspot data pin (data/manifests/textspot_shards_v1.json, synth/textspot_real_half.py) lists the
real canvases the protocol's exclusions still remove from #97's pinned real half: real.measured.skip's
eval_screenshot.skip_shots (screenshots that a later eval or eval/linefind list names) and linefind_list_vn.skip_vns
(VNs of an eval/linefind list). Every real canvas (a sample whose src does not start with "synth") whose shot is in
skip_shots or whose vn is in skip_vns is dropped while the pool is built, so it is never drawn. Every --manifest
that carries real.measured.skip contributes its lists; a textspot data pin (kind textspot-shards, or a file named
textspot_shards_*.json) without both lists is refused. run.json records each list's length and sha256 (of its
items, each followed by a newline, in the manifest's order) and the canvases dropped per source: by shot, by VN
alone (shot not listed), the overlap (both, counted once) and the total, beside the counts the pin itself records.

Outputs (the last step is the run's model; nothing is selected inside a run; RUN must not exist or be empty, so a
crashed run restarts from scratch in a fresh directory):
  RUN/run.json    started (UTC), the arguments, shape, scale, seed, the recipe, the manifests' sha256, the skip
                  lists and the canvases they dropped, the shards read, the pool per source, the sha256 of every
                  train/ and synth/ file the process imported, the protocol's sha256 and the checkout's commit (with
                  any modified tracked file)
  RUN/log.jsonl   #97's records: every 200 steps (step, lr, sec, k_bce, dice, o, a) and the QAT record
  RUN/final.pt    #97's quantisation-aware checkpoint (kind textdet-qat)
  RUN/final.vtd   the int8 VTDT blob (train/textdet_export.export_checkpoint)
  RUN/final.vtsp  textspot_twin.pack(final.vtd, half_above 600 for h2 or 0 for f1, conf_min 1)
  RUN/done.json   finished (UTC), steps, wall time, the blobs' bytes and sha256
"""
import argparse
import contextlib
import datetime
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_export as EX  # noqa: E402
import textdet_model as TM  # noqa: E402
import textdet_train as TT  # noqa: E402
import textspot_twin as TS  # noqa: E402

WIN_H, WIN_W = 256, 512
BATCH = TT.BATCH
MIX = TT.MIX
HALF_ABOVE = {"h2": 600, "f1": 0}      # eval/textspot_protocol.json candidates.family_b.scales
MIN_ACROSS = TS.MIN_ACROSS             # 4: the decode drops boxes under 4 px across at the network's scale
CLIP = 5.0                             # #97's gradient-norm clip (textdet_train.main)
PROTOCOL = ROOT / "eval" / "textspot_protocol.json"


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextlib.contextmanager
def window_size():
    """#97's targets() and calibrate() size their maps by textdet_train's module constants WIN_H x WIN_W
    (192 x 512). The recipe's windows are 256 x 512, so those two run with the constants set to it and restored
    after (#97's trainer, textgate's and their tests keep 192 x 512)."""
    old = TT.WIN_H, TT.WIN_W
    TT.WIN_H, TT.WIN_W = WIN_H, WIN_W
    try:
        yield
    finally:
        TT.WIN_H, TT.WIN_W = old


def factor(scale, width):
    """2 when the scale halves a canvas of this width (textspot_twin.scale_of's rule), else 1."""
    return 2 if HALF_ABOVE[scale] and width > HALF_ABOVE[scale] else 1


def half_box(b):
    """[x0, y0, x1, y1] on a canvas -> the same box on its halved image (textspot_twin.halve): every start
    floor(v / 2), every end ceil(v / 2). The halved pixel i reads canvas pixels 2i and 2i + 1, so it meets
    [v0, v1) exactly when floor(v0 / 2) <= i < ceil(v1 / 2)."""
    return [int(b[0]) // 2, int(b[1]) // 2, (int(b[2]) + 1) // 2, (int(b[3]) + 1) // 2]


def at_scale(lines, ignore, f):
    """A canvas's lines [[x0, y0, x1, y1, vertical, class...]] and ignore areas [[x0, y0, x1, y1]] at the network's
    scale (f = 2: halved): (positives [[x0, y0, x1, y1, vertical]], ignore areas). Every line is a positive whatever
    its class, unless it is under MIN_ACROSS px across there (its height, or its width when vertical): then it
    becomes an ignore area."""
    def to(b):
        return half_box(b) if f == 2 else [int(v) for v in b[:4]]

    ign = [to(b) for b in ignore]
    pos = []
    for ln in lines:
        b = to(ln)
        vertical = int(ln[4])
        if (b[2] - b[0] if vertical else b[3] - b[1]) < MIN_ACROSS:
            ign.append(b)
        else:
            pos.append([*b, vertical])
    return pos, ign


def window(rng, canvas, scale):
    """(image, ox, oy, w, h, f): the window of a canvas at the network's scale. f = 2 when the scale halves the
    canvas; the canvas at that scale is H x W (ceil of half each side when halved) and the window is
    min(WIN_H, H) x min(WIN_W, W) at a uniformly random offset (x drawn first, as #97's window_of). A halved window
    is the halving of the canvas's own 2h x 2w block at (2 ox, 2 oy), which equals halving the whole canvas and
    cutting the window (the last row or column repeats only at the canvas's end)."""
    H, W = canvas.shape
    f = factor(scale, W)
    Hs, Ws = -(-H // f), -(-W // f)
    h, w = min(WIN_H, Hs), min(WIN_W, Ws)
    ox, oy = int(rng.integers(0, Ws - w + 1)), int(rng.integers(0, Hs - h + 1))
    if f == 2:
        img = TS.halve(canvas[2 * oy:2 * (oy + h), 2 * ox:2 * (ox + w)])
    else:
        img = np.asarray(canvas[oy:oy + h, ox:ox + w])
    return img, ox, oy, w, h, f


def list_sha256(items):
    """sha256 of a list's items, each followed by a newline, in the given order (the skip lists' fingerprint)."""
    return hashlib.sha256("".join(f"{x}\n" for x in items).encode("utf-8")).hexdigest()


def is_data_pin(path, doc):
    return doc.get("kind") == "textspot-shards" or Path(path).name.startswith("textspot_shards")


def skip_lists(manifests):
    """The real canvases the data pins exclude: (skip_shots, skip_vns, record). Each --manifest with
    real.measured.skip gives its eval_screenshot.skip_shots and linefind_list_vn.skip_vns; a textspot data pin
    without both lists is refused. The record per list source: lengths, sha256 (list_sha256) and the canvas
    counts the pin records (synth/textspot_real_half.py)."""
    shots, vns, rec = set(), set(), []
    for man in manifests:
        doc = json.loads(Path(man).read_text(encoding="utf-8"))
        skip = ((doc.get("real") or {}).get("measured") or {}).get("skip")
        if skip is None:
            if is_data_pin(man, doc):
                raise SystemExit(f"{man}: a textspot data pin without real.measured.skip")
            continue
        try:
            ev, lf = skip["eval_screenshot"], skip["linefind_list_vn"]
            s, v = list(ev["skip_shots"]), list(lf["skip_vns"])
        except (KeyError, TypeError) as e:
            raise SystemExit(f"{man}: real.measured.skip lacks eval_screenshot.skip_shots or "
                             f"linefind_list_vn.skip_vns ({e!r})") from None
        if not all(isinstance(x, str) and x for x in s + v):
            raise SystemExit(f"{man}: a skip list holds a non-string or empty item")
        shots.update(s)
        vns.update(v)
        rec.append({"manifest": str(man), "skip_shots": len(s), "skip_shots_sha256": list_sha256(s),
                    "skip_vns": len(v), "skip_vns_sha256": list_sha256(v),
                    "pin_counts": {"eval_screenshot": {k: ev.get(k) for k in ("shots", "canvases", "by_source")},
                                   "linefind_list_vn": {k: lf.get(k) for k in ("shots", "canvases", "by_source")}}})
    return frozenset(shots), frozenset(vns), rec


class FramePool(TT.Pool):
    """#97's Pool over several manifests: the shards in the given directories, memory-mapped, grouped by #97's
    sources (synth: src starting with "synth"; realplus, teacher, owner, devnative). verify: check each shard's npy
    and json against its manifest's sha256 first. skip: (shots, vns): a real canvas whose shot or vn is listed is
    dropped and counted in self.dropped[source] (by_shot, by_vn_only, both, total)."""

    def __init__(self, manifests, dirs, verify=False, skip=(frozenset(), frozenset())):
        roots = {}
        for d in dirs:
            if Path(d).name in roots:
                raise SystemExit(f"two shard directories named {Path(d).name}")
            roots[Path(d).name] = Path(d)
        skip_shots, skip_vns = (frozenset(x) for x in skip)
        self.items = {k: [] for k in MIX}
        self.dropped = {k: {"by_shot": 0, "by_vn_only": 0, "both": 0, "total": 0} for k in MIX if k != "synth"}
        self.data, self.read, self.skipped = [], [], {}
        used = set()
        for man in manifests:
            doc = json.loads(Path(man).read_text(encoding="utf-8"))
            for s in doc["shards"]:
                if s["dir"] not in roots:
                    key = f"{man}:{s['dir']}"
                    self.skipped[key] = self.skipped.get(key, 0) + 1
                    continue
                used.add(s["dir"])
                base = roots[s["dir"]] / s["name"]
                npy, js = base.with_suffix(".npy"), base.with_suffix(".json")
                if verify:
                    for path, want in ((npy, s["npy_sha256"]), (js, s["json_sha256"])):
                        if TT.sha256(path) != want:
                            raise SystemExit(f"{path}: sha256 is not {man}'s")
                d = len(self.data)
                self.data.append(np.load(npy, mmap_mode="r"))
                samples = json.loads(js.read_text(encoding="utf-8"))["samples"]
                kept = 0
                for smp in samples:
                    src = "synth" if smp["src"].startswith("synth") else smp["src"]
                    if src not in self.items:
                        raise SystemExit(f"{js}: source {smp['src']!r} is not in #97's mix")
                    if src != "synth":
                        by_shot, by_vn = smp.get("shot") in skip_shots, smp.get("vn") in skip_vns
                        if by_shot or by_vn:
                            c = self.dropped[src]
                            c["by_shot"] += by_shot
                            c["by_vn_only"] += by_vn and not by_shot
                            c["both"] += by_shot and by_vn
                            c["total"] += 1
                            continue
                    self.items[src].append((d, smp))
                    kept += 1
                self.read.append({"manifest": str(man), "dir": s["dir"], "name": s["name"], "samples": len(samples),
                                  "kept": kept})
        unused = sorted(set(roots) - used)
        if unused:
            raise SystemExit(f"no manifest shard lies in {unused}")
        self.keys = [k for k in MIX if self.items[k]]
        if not self.keys:
            raise SystemExit("no samples")
        p = np.array([MIX[k] for k in self.keys])
        self.p = p / p.sum()

    def summary(self):
        by = {k: {"canvases": len(v), "lines": sum(len(s["lines"]) for _, s in v),
                  "vertical": sum(int(ln[4]) for _, s in v for ln in s["lines"]),
                  "ignore": sum(len(s["ignore"]) for _, s in v)} for k, v in self.items.items()}
        return {"by_source": by, "mix_effective": dict(zip(self.keys, (round(float(q), 6) for q in self.p))),
                "dropped": self.dropped, "dropped_total": sum(c["total"] for c in self.dropped.values()),
                "shards_read": self.read, "shards_not_read": self.skipped}


def skip_record(lists, dropped):
    """run.json's "skip": the lists, the canvases dropped per source, the pin's own counts and where they differ
    (by_shot per source against the pin's eval_screenshot canvases, by_vn_only against its linefind_list_vn
    canvases; pin_differs: {source: {column: [dropped, pin]}})."""
    want = {}
    for r in lists:
        for key, col in (("eval_screenshot", "by_shot"), ("linefind_list_vn", "by_vn_only")):
            for src, n in ((r["pin_counts"][key] or {}).get("by_source") or {}).items():
                want.setdefault(src, {"by_shot": 0, "by_vn_only": 0})[col] += int(n)
    srcs = sorted(set(want) | set(dropped))
    differs = {src: {col: [dropped.get(src, {}).get(col, 0), want.get(src, {}).get(col, 0)]
                     for col in ("by_shot", "by_vn_only")
                     if dropped.get(src, {}).get(col, 0) != want.get(src, {}).get(col, 0)} for src in srcs}
    differs = {k: v for k, v in differs.items() if v}
    return {"lists": lists, "how": "a real canvas (src not 'synth...') whose shot is in skip_shots or whose vn is in "
            "skip_vns is dropped before the pool is built; list sha256 = sha256 of its items each followed by a "
            "newline, in the manifest's order",
            "dropped": dropped, "dropped_total": sum(c["total"] for c in dropped.values()),
            "pin_expects": want, "pin_expects_total": sum(sum(d.values()) for d in want.values()),
            "matches_pin": bool(lists) and not differs, "pin_differs": differs}


def make_batch(pool, rng, s, scale):
    """One batch: #97's tensors (x uint8 N x 1 x WIN_H x WIN_W, valid (h, w), K, O, A1, A2, valid and kernel
    masks at stride s) from 16 windows, each from a source drawn by the mix and a canvas drawn uniformly in it."""
    x = np.full((BATCH, 1, WIN_H, WIN_W), 128, np.uint8)
    valid_hw = []
    tg = [np.zeros((BATCH, WIN_H // s, WIN_W // s), np.float32) for _ in range(4)]
    vm = np.zeros((BATCH, WIN_H // s, WIN_W // s), bool)
    km = np.zeros_like(vm)
    for b in range(BATCH):
        src = pool.keys[int(rng.choice(len(pool.keys), p=pool.p))]
        lst = pool.items[src]
        d, smp = lst[int(rng.integers(len(lst)))]
        img, ox, oy, w, h, f = window(rng, pool.canvas(d, smp), scale)
        x[b, 0, :h, :w] = img
        valid_hw.append((h, w))
        lines, ignore = at_scale(smp["lines"], smp["ignore"], f)
        with window_size():
            K, O, A1, A2, v, k = TT.targets(lines, ignore, ox, oy, w, h, s)  # noqa: E741
        for t, a in zip(tg, (K, O, A1, A2)):
            t[b] = a
        vm[b], km[b] = v, k
    return (torch.from_numpy(x), torch.tensor(valid_hw), *(torch.from_numpy(t) for t in tg),
            torch.from_numpy(vm), torch.from_numpy(km))


class FrameBatches(torch.utils.data.IterableDataset):
    def __init__(self, manifests, dirs, stride, seed, scale, skip=(frozenset(), frozenset())):
        super().__init__()
        self.manifests, self.dirs = list(manifests), list(dirs)
        self.stride, self.seed, self.scale = stride, seed, scale
        self.skip = tuple(frozenset(x) for x in skip)

    def __iter__(self):
        pool = FramePool(self.manifests, self.dirs, skip=self.skip)
        rng = np.random.default_rng(self.seed)
        while True:
            yield make_batch(pool, rng, self.stride, self.scale)


def code_sha256():
    """sha256 of every train/ and synth/ Python file this process has imported (the script itself included)."""
    out = {}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f or not f.endswith(".py"):
            continue
        p = Path(f).resolve()
        try:
            rel = p.relative_to(ROOT)
        except ValueError:
            continue
        if rel.parts[0] in ("train", "synth"):
            out[rel.as_posix()] = TT.sha256(p)
    return dict(sorted(out.items()))


def git_state():
    """The checkout's commit and its modified tracked files (None outside a git checkout)."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=60)
        st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                            text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if head.returncode or st.returncode:
        return None
    return {"head": head.stdout.strip(), "modified": [x[3:] for x in st.stdout.splitlines()]}


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", required=True, choices=sorted(TM.SHAPES))
    ap.add_argument("--scale", required=True, choices=sorted(HALF_ABOVE))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--probe", type=int, default=0, metavar="N",
                    help="speed probe: one warm-up step, then N steps of the run's schedule, timed; RUN/probe.json")
    ap.add_argument("--cal-batches", type=int, default=TT.CAL_BATCHES, help="QAT calibration batches (#97: 32)")
    ap.add_argument("--workers", type=int, default=1, help="loader processes (#97: 1)")
    ap.add_argument("--allow-missing-sources", action="store_true",
                    help="tests: train on whichever of the mix's sources the shards hold")
    args = ap.parse_args(argv)
    if args.steps < 1 or args.probe < 0 or args.cal_batches < 1 or args.workers < 0:
        ap.error("--steps >= 1, --probe >= 0, --cal-batches >= 1, --workers >= 0")
    return args


def main(argv=None):
    args = parse(argv)
    t_proc = time.time()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: a run starts from scratch in a fresh directory")
    started = utc()
    skip_shots, skip_vns, lists = skip_lists(args.manifest)
    pool = FramePool(args.manifest, args.shards, verify=True, skip=(skip_shots, skip_vns))
    missing = [k for k in MIX if not pool.items[k]]
    if missing and not args.allow_missing_sources:
        raise SystemExit(f"no samples of {missing} (the mix needs all of {list(MIX)})")
    pool_summary = pool.summary()
    skip = skip_record(lists, pool.dropped)
    del pool
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    shape = TM.SHAPES[args.shape]
    s = shape["stride"]
    half_above = HALF_ABOVE[args.scale]
    model = TM.TextDet(shape)
    opt = torch.optim.AdamW(TT.param_groups(list(model.parameters())), lr=TT.LR)
    loader = torch.utils.data.DataLoader(FrameBatches(args.manifest, args.shards, s, args.seed, args.scale,
                                                      (skip_shots, skip_vns)),
                                         batch_size=None, num_workers=args.workers,
                                         persistent_workers=args.workers > 0,
                                         prefetch_factor=4 if args.workers > 0 else None)
    it = iter(loader)
    qat_start = int(TT.QAT_FRAC * args.steps)
    run = {"kind": "textspot-run", "name": f"{args.scale}_{args.shape}_seed{args.seed}", "started": started,
           "args": vars(args), "argv": sys.argv[1:] if argv is None else list(argv),
           "shape": shape, "scale": args.scale, "half_above": half_above, "conf_min": 1, "seed": args.seed,
           "steps": args.steps, "qat_start": qat_start, "probe": args.probe,
           "recipe": {"batch": BATCH, "window": [WIN_H, WIN_W], "mix": MIX, "lr": [TT.LR, TT.LR_MIN, TT.WARMUP],
                      "wd": TT.WD, "clip": CLIP, "qat_frac": TT.QAT_FRAC, "cal_batches": args.cal_batches,
                      "kernel_rule": "round 1 (textdet_train.targets strict=False)", "min_across": MIN_ACROSS,
                      "halve": "textspot_twin.halve when the canvas is wider than half_above; boxes start "
                               "floor(v/2), end ceil(v/2)",
                      "windows": "uniformly random offset; a side that fits is whole; no region crops",
                      "positives": "every line of every class; under min_across px across: ignore",
                      "loader_workers": args.workers},
           "manifests_sha256": {m: TT.sha256(m) for m in args.manifest}, "skip": skip,
           "shards_dirs": [str(Path(d)) for d in args.shards], "shards_verified": True, "pool": pool_summary,
           "code_sha256": code_sha256(),
           "protocol_sha256": TT.sha256(PROTOCOL) if PROTOCOL.exists() else None, "git": git_state(),
           "torch": torch.__version__, "numpy": np.__version__, "threads": args.threads}
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8")
    log = open(out / "log.jsonl", "a", encoding="utf-8")
    t0 = time.time()
    agg, n_agg = {}, 0
    act_scale = None
    n_steps = min(args.steps, args.probe + 1) if args.probe else args.steps
    times = []
    for step in range(n_steps):
        if step == qat_start:
            with window_size():
                act_scale = TT.calibrate(model, it, shape, args.cal_batches)
            model.eval()
            folded = TM.fold(model)
            model = TM.QTextDet(shape, folded, act_scale)
            opt = torch.optim.AdamW(TT.param_groups(list(model.parameters())), lr=TT.lr_at(step, args.steps))
            log.write(json.dumps({"step": step, "qat": True, "act_scale": act_scale}) + "\n")
            log.flush()
        ts = time.perf_counter()
        for g in opt.param_groups:
            g["lr"] = TT.lr_at(step, args.steps)
        x, vhw, K, O, A1, A2, vm, km = next(it)  # noqa: E741
        td = time.perf_counter()
        masks = TM.layer_masks(shape, vhw.tolist(), WIN_H, WIN_W)
        y = model(TT.to_input(x), masks=masks)
        loss, parts = TT.loss_of(y, K, O, A1, A2, vm, km)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        if args.probe:
            times.append((td - ts, time.perf_counter() - ts))
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v
        n_agg += 1
        if (step + 1) % args.log_every == 0 or step + 1 == n_steps:
            rec = {"step": step + 1, "lr": round(TT.lr_at(step, args.steps), 6), "sec": round(time.time() - t0, 1),
                   **{k: round(v / n_agg, 5) for k, v in agg.items()}}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            agg, n_agg = {}, 0
    if args.probe:
        timed = times[1:]
        ms = np.array([t for _, t in timed]) * 1000
        wait = np.array([t for t, _ in timed]) * 1000
        probe = {"finished": utc(), "steps_timed": len(timed), "warmup_ms": round(times[0][1] * 1000, 1),
                 "ms_per_step": round(float(ms.mean()), 2), "ms_per_step_median": round(float(np.median(ms)), 2),
                 "ms_per_step_p90": round(float(np.percentile(ms, 90)), 2),
                 "ms_waiting_for_data": round(float(wait.mean()), 2), "threads": args.threads}
        (out / "probe.json").write_text(json.dumps(probe, indent=1) + "\n", encoding="utf-8")
        print("TEXTSPOT_PROBE_DONE", json.dumps(probe), flush=True)
        return probe
    ck = {"kind": "textdet-qat", "shape": shape, "seed": args.seed, "steps": args.steps,
          "w": [w.detach().double() for w in model.w], "b": [b.detach().double() for b in model.b],
          "act_scale": act_scale, "run": run}
    torch.save(ck, out / "final.pt")
    vtd = EX.export_checkpoint(out / "final.pt")
    (out / "final.vtd").write_bytes(vtd)
    vtsp = TS.pack(vtd, half_above, 1)
    m = TS.Model(vtsp)
    if (m.half_above, m.conf_min, m.name) != (half_above, 1, shape["name"]):
        raise SystemExit("the VTSP blob does not read back")
    (out / "final.vtsp").write_bytes(vtsp)
    done = {"finished": utc(), "steps": args.steps, "seconds": round(time.time() - t0, 1),
            "wall_seconds": round(time.time() - t_proc, 1), "pt_sha256": TT.sha256(out / "final.pt"),
            "vtd_bytes": len(vtd), "vtd_sha256": hashlib.sha256(vtd).hexdigest(),
            "vtsp_bytes": len(vtsp), "vtsp_sha256": hashlib.sha256(vtsp).hexdigest()}
    (out / "done.json").write_text(json.dumps(done, indent=1) + "\n", encoding="utf-8")
    print("TEXTSPOT_TRAIN_DONE", json.dumps(done), flush=True)
    return done


if __name__ == "__main__":
    main()
