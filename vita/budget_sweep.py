#!/usr/bin/env python3
"""Memory and estimated Cortex-A9 cost of the recognizer as it grows (docs/vita-memory-budget.md).

  budget_sweep.py blobs --out DIR [--only NAME,...]
  budget_sweep.py a9 --dir DIR [--l2 2048,1024,512,256] [--only NAME,...] [--tile N] [--head N]
                     [--l2-ways 8] [--evict lru|rand] [--rep K]
                     [--lines 40] [--jobs 5] [--vitasdk D] [--arm-cc CC] [--qemu D]
  budget_sweep.py table --dir DIR [--json OUT]
  budget_sweep.py ckpt --name NAME --out CKPT     a random-init checkpoint of one GRID config
                                                  (for vita/bundle.py --ckpt and the Vita3K check)
  budget_sweep.py check BLOB [BLOB...] [--out DIR] [--lines 40] [--vitasdk D] [--arm-cc CC] [--qemu D]
                                                  an int8 blob against CEILING; exit 1 if any fails

blobs: every config of GRID as a train/model.py TinyVNOCR with random weights and random
  BatchNorm statistics, exported by train/export.py as an int8 blob (DIR/blobs/NAME_w8.vocr).
  DIR/sizes.json gets, per config: parameters; blob bytes split into int8 matrices, float32
  stem/depthwise weights, float32 biases and scales, and vocabulary/headers; the W8I and W8F
  arenas (export.plan_arena, which the parity tests hold equal to vocr_arena_bytes()) at the
  default tile (4 frames) and head batch (16) and at --tile/--head 16/64; MACs per CTC frame;
  and the weight bytes the runtime reads per frame (every backbone op once per tile, the head
  once per head batch), the traffic that has to come from DRAM when the blob does not stay in
  the L2. Classes above data/vocab.txt's 5,355 are filled with CJK ideographs the vocab lacks.
  Random weights cost the same cycles as trained ones (the kernels have no data-dependent
  branches); the table checks it with the released FL10, H15 and F20 blobs, which `a9` also
  measures when $HERD/shared/artifacts/train2/release holds them.

a9: vita/a9_cycles.py's estimate for every blob, at each L2 size in --l2 (KiB). The Vita's L2 is
  one PL310 shared by all four cores, so a game that keeps its own data there leaves the
  recognizer a smaller share; the smaller sizes model that. The machine code (arm-vita-eabi-gcc,
  Thumb-2, -O2), the qemu plugins, the guest staging layout (a fixed-length /tmp/vocr-a9-XXXXXXXX
  copy, relative names, empty environment, -seed 0) and the cost model (a9_cycles.model_mode) are
  a9_cycles.py's. Only synth-val lines are read (the first --lines lines of $VOCR_SYNTH_VAL); no
  protected test line is. --tile/--head pass the runtime's tiling to vocr_cli (0: its defaults).
  --l2-ways sets the simulated L2's associativity (a9_cycles.py: 8; the Vita's PL310: 16). --evict rand
  replaces cache lines at random in every simulated cache, as the PL310 does (AUX_CTRL bit 25 clear) instead
  of qemu's default LRU; qemu seeds it from the system, so --rep K tags repeated runs.
  One result per run: DIR/a9/NAME_l2KIB[_tTILEhHEAD][_wWAYS][_rand][_rK].json.

table: the Markdown tables of docs/vita-memory-budget.md from sizes.json and the a9 results.

check: the ceiling of docs/vita-memory-budget.md for a trained candidate. The blob bytes, the W8I arena at
  the runtime's default tiling (export.plan_arena) and the high end of the estimated W8I time of an
  80-frame (640-px) line on one 333 MHz Cortex-A9 core, from the same a9 run as above (2 MiB L2, the
  first --lines synth-val lines, so no protected test line is read). Run it through heavy.sh.

Everything about console speed here is an estimate from a cycle model, not a measurement.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "train"), str(ROOT / "eval")]
import a9_cycles  # noqa: E402

HERD = Path(os.environ["HERD"]) if os.environ.get("HERD") else None
RELEASE = HERD / "shared/artifacts/train2/release" if HERD else None
RELEASED = ("FL10", "H15", "F20")       # train2's released int8 blobs, measured as they are
FRAMES_640 = a9_cycles.FRAMES_640
CLOCKS = a9_cycles.CLOCKS
L2_BLOCK = 32
L2_WAYS = 8                             # a9_cycles.CACHE_ARGS; the Vita's PL310 has 16 (--l2-ways)
DEFAULT_TILE, DEFAULT_HEAD = 4, 16
TILINGS = ((DEFAULT_TILE, DEFAULT_HEAD), (8, 32), (4, 64), (16, 64))   # (tile, head batch) in frames
BASE = (2048, DEFAULT_TILE, DEFAULT_HEAD, L2_WAYS, "lru", 0)   # a9_cycles.py's setup: L2 KiB, tile, head, ...

# docs/vita-memory-budget.md's ceiling: int8 blob bytes, W8I arena bytes (tile 4, head 16), and the high
# end of the estimated ms per 640-px line on one Cortex-A9 core at 333 MHz (W8I, 2 MiB L2).
CEILING = {"weights": 1_600_000, "arena": 262_144, "ms": 550, "mhz": 333}


def status_line(c=CEILING):
    return f"BUDGET_READY weights<={c['weights']} arena<={c['arena']} ms<={c['ms']}"


# name: (width, bottleneck, depth, ctx, ctx_k, classes, height). Classes include the CTC blank.
GRID = {
    "w075": (0.75, 96, 1, 2, 5, 5355, 32),
    "w100": (1.0, 96, 1, 2, 5, 5355, 32),          # v1
    "w100v3500": (1.0, 96, 1, 2, 5, 3501, 32),     # FL10's shape
    "w150": (1.5, 96, 1, 2, 5, 5355, 32),          # H15's shape
    "w200": (2.0, 96, 1, 2, 5, 5355, 32),          # F20's shape
    "w250": (2.5, 96, 1, 2, 5, 5355, 32),
    "w300": (3.0, 96, 1, 2, 5, 5355, 32),
    "w400": (4.0, 96, 1, 2, 5, 5355, 32),
    "w150b48": (1.5, 48, 1, 2, 5, 5355, 32),
    "w150b64": (1.5, 64, 1, 2, 5, 5355, 32),
    "w150b128": (1.5, 128, 1, 2, 5, 5355, 32),
    "w150b192": (1.5, 192, 1, 2, 5, 5355, 32),
    "w150v2500": (1.5, 96, 1, 2, 5, 2501, 32),
    "w150v3500": (1.5, 96, 1, 2, 5, 3501, 32),
    "w150v7000": (1.5, 96, 1, 2, 5, 7001, 32),     # about JIS X 0208's 6,879 characters
    "w150v11000": (1.5, 96, 1, 2, 5, 11001, 32),   # about JIS X 0213's 11,233
    "w150d2": (1.5, 96, 2, 2, 5, 5355, 32),
    "w150d3": (1.5, 96, 3, 2, 5, 5355, 32),
    "w150c4": (1.5, 96, 1, 4, 5, 5355, 32),
    "w200b192": (2.0, 192, 1, 2, 5, 5355, 32),
    "w200h24": (2.0, 96, 1, 2, 5, 5355, 24),
    "w300b192": (3.0, 192, 1, 2, 5, 5355, 32),
    "w300v11000": (3.0, 96, 1, 2, 5, 11001, 32),
}


# ---------------------------------------------------------------- blobs

def vocab_of(n_classes):
    """n_classes - 1 output characters: data/vocab.txt's first ones, then CJK ideographs it lacks."""
    base = (ROOT / "data/vocab.txt").read_text(encoding="utf-8").splitlines()
    n = n_classes - 1
    if n <= len(base):
        return base[:n]
    have = set(base)
    extra = [chr(c) for c in range(0x4E00, 0xA000) if chr(c) not in have]
    if n - len(base) > len(extra):
        raise ValueError(f"no {n_classes} classes: only {len(base) + len(extra)} characters available")
    return base + extra[:n - len(base)]


def random_model(width, bottleneck, depth, ctx, ctx_k, n_classes, seed=0):
    """A TinyVNOCR with random weights and random BatchNorm statistics, in eval mode."""
    import torch
    import torch.nn as nn

    from model import TinyVNOCR
    torch.manual_seed(seed)
    model = TinyVNOCR(n_classes, width=width, bottleneck=bottleneck, depth=depth, ctx=ctx, ctx_k=ctx_k)
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_mean.normal_(0, 0.1)
                mod.running_var.uniform_(0.5, 2.0)
                mod.weight.uniform_(0.5, 1.5)
                mod.bias.normal_(0, 0.1)
    return model.eval()


def blob_info(data):
    """Sizes, arenas, MACs and per-frame weight traffic of an int8 blob."""
    import export
    b = export.read_blob(data)
    ops = b["ops"]
    split = {"int8_matrices": 0, "f32_conv_weights": 0, "f32_bias_scale": 0}
    per_op = []
    for op in ops:
        wb = op["w_bytes"]                     # as stored: output channels padded to cout_pad
        bs = 4 * op["cout_pad"] * ((op["bias"] is not None) + op["int8"])
        split["int8_matrices" if op["int8"] else "f32_conv_weights"] += wb
        split["f32_bias_scale"] += bs
        per_op.append((op["type"] == export.T_HEAD, wb + bs))
    split["vocab_headers_padding"] = len(data) - sum(split.values())
    head_bytes = sum(n for is_head, n in per_op if is_head)
    body_bytes = sum(n for is_head, n in per_op if not is_head)
    macs, head_macs = export.macs_per_frame(b)

    def traffic(tile, head):
        return body_bytes / tile + head_bytes / head

    arenas = {}
    for tile, head in TILINGS:
        arenas[f"t{tile}h{head}"] = {m: export.plan_arena(b, tile=tile, head=head, mode=code)
                                     for m, code in (("w8i", export.MODE_W8I), ("w8f", export.MODE_W8F))}
    return {"blob_bytes": len(data), "params": int(b.get("params", 0)), "classes": b["n_classes"],
            "height": b["height"], "split": split, "head_weight_bytes": head_bytes,
            "body_weight_bytes": body_bytes, "macs_per_frame": macs, "head_macs_per_frame": head_macs,
            "arena": arenas, "weight_bytes_per_frame": {f"t{t}h{h}": traffic(t, h) for t, h in TILINGS}}


def cmd_blobs(args):
    import export
    out = Path(args.out)
    (out / "blobs").mkdir(parents=True, exist_ok=True)
    sizes_path = out / "sizes.json"
    sizes = json.loads(sizes_path.read_text()) if sizes_path.exists() else {}
    names = args.only.split(",") if args.only else list(GRID)
    for name in names:
        width, bn, depth, ctx, ctx_k, n_classes, height = GRID[name]
        model = random_model(width, bn, depth, ctx, ctx_k, n_classes)
        data, meta = export.build_blob(export.model_ops(model, height), vocab_of(n_classes), height, int8=True)
        (out / "blobs" / f"{name}_w8.vocr").write_bytes(data)
        info = blob_info(data)
        info.update(config=dict(width=width, bottleneck=bn, depth=depth, ctx=ctx, ctx_k=ctx_k,
                                classes=n_classes, height=height), random_init=True,
                    params=meta["params"], sha256=hashlib.sha256(data).hexdigest())
        sizes[name] = info
        print(f"{name}: {len(data):,} B blob, W8I arena {info['arena']['t4h16']['w8i']:,} B, "
              f"{info['macs_per_frame']:,} MACs/frame", file=sys.stderr)
    for name in RELEASED if (RELEASE and not args.only) else ():
        path = RELEASE / f"{name}_w8.vocr"
        if not path.exists():
            continue
        data = path.read_bytes()
        shutil.copy2(path, out / "blobs" / f"{name}_w8.vocr")
        info = blob_info(data)
        info.update(released=True, sha256=hashlib.sha256(data).hexdigest())
        sizes[name] = info
    sizes_path.write_text(json.dumps(sizes, indent=1, ensure_ascii=False))


def cmd_ckpt(args):
    """The same random model as `blobs` (same seed) as a train.py-style checkpoint."""
    import torch
    width, bn, depth, ctx, ctx_k, n_classes, height = GRID[args.name]
    model = random_model(width, bn, depth, ctx, ctx_k, n_classes)
    torch.save({"model": model.state_dict(), "vocab": vocab_of(n_classes), "height": height, "width": width,
                "bottleneck": bn, "depth": depth, "ctx": ctx, "ctx_k": ctx_k, "random_init": True}, args.out)


# ---------------------------------------------------------------- a9

def cache_args(l2_bytes, ways=L2_WAYS, evict="lru"):
    """a9_cycles.CACHE_ARGS with another L2 size, associativity or replacement (same 32-byte lines)."""
    old = f"l2cachesize=2097152,l2assoc={L2_WAYS},"
    assert old in a9_cycles.CACHE_ARGS
    if l2_bytes % (ways * L2_BLOCK):
        raise ValueError(f"L2 size {l2_bytes} is not a multiple of {ways} ways x {L2_BLOCK} B")
    if evict not in ("lru", "rand", "fifo"):
        raise ValueError(f"unknown eviction policy {evict}")
    args = a9_cycles.CACHE_ARGS.replace(old, f"l2cachesize={int(l2_bytes)},l2assoc={int(ways)},")
    return args if evict == "lru" else f"{args},evict={evict}"


def write_synth_pack(path, n):
    """vocr_cli's pack format (as a9_cycles.write_pack) with the first n synth-val lines only."""
    import sets
    lines = sets.load_synth_val(os.environ.get("VOCR_SYNTH_VAL", sets.SYNTH_VAL), limit=n)
    crops = [np.ascontiguousarray(ln.crop_rgb[..., 0]) for ln in lines]
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(crops)))
        for c in crops:
            f.write(struct.pack("<III", c.shape[0], c.shape[1], 1))
            f.write(c.tobytes())
    return len(crops)


def run_name(name, l2_kib, tile, head, ways=L2_WAYS, evict="lru", rep=0):
    return (f"{name}_l2{l2_kib}" + (f"_t{tile}h{head}" if tile or head else "")
            + (f"_w{ways}" if ways != L2_WAYS else "") + (f"_{evict}" if evict != "lru" else "")
            + (f"_r{rep}" if rep else ""))


def run_one(name, blob, l2_kib, tile, head, work, qemu, out, ways=L2_WAYS, evict="lru", rep=0):
    """One qemu run of the Vita machine code on one blob; returns (run name, result dict)."""
    rn = run_name(name, l2_kib, tile, head, ways, evict, rep)
    stage = Path(tempfile.mkdtemp(prefix="vocr-a9-", dir="/tmp"))   # a9_cycles' fixed-length path
    try:
        cli = work / "vocr_cli_vitacc"
        for f in (cli, work / "lines.bin"):
            shutil.copy2(f, stage / f.name)
        shutil.copy2(blob, stage / "model_w8.vocr")                 # a9_cycles.given_blob's name
        counts, cache = out / f"counts_{rn}.txt", out / f"cache_{rn}.txt"
        cmd = [str(qemu / "bin/qemu-arm"), "-cpu", "cortex-a9", "-seed", "0",
               "-plugin", f"{qemu}/plugins/libinsn_mix.so,out={counts},classes={work}/classes.txt",
               "-plugin", f"{qemu}/plugins/libcache.so,{cache_args(l2_kib * 1024, ways, evict)}", "-d", "plugin", "-D",
               str(cache), f"./{cli.name}", "model_w8.vocr", "--mode", "w8i", "--pack", "lines.bin"]
        cmd += ["--tile", str(tile)] if tile else []
        cmd += ["--head", str(head)] if head else []
        p = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=14400,
                           env=dict(a9_cycles.QEMU_GUEST_ENV), cwd=stage)
        if p.returncode:
            raise RuntimeError(f"{rn}: qemu failed: {p.stderr[-2000:]}")
        import re
        m = re.search(r"(\d+) lines, (\d+) frames", p.stderr)
        frames = int(m.group(2))
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    rows = a9_cycles.parse_dis((work / "cli.dis").read_text())
    funcs = set((work / "vocr_functions.txt").read_text().split())
    c, extra = a9_cycles.read_counts(counts)
    res = a9_cycles.model_mode(rows, c, extra, funcs, frames, a9_cycles.read_cache(cache))
    res["ms_640px"] = {mhz: [x * FRAMES_640 / (mhz * 1000) for x in res["cycles_per_frame"][mhz]] for mhz in CLOCKS}
    res.update(name=name, l2_kib=l2_kib, l2_ways=ways, evict=evict, rep=rep, tile=tile or DEFAULT_TILE,
               head=head or DEFAULT_HEAD,
               lines=int(m.group(1)), blob_sha256=hashlib.sha256(Path(blob).read_bytes()).hexdigest(),
               texts_sha256=hashlib.sha256(p.stdout.encode()).hexdigest())
    counts.unlink()                    # 100 KB each; the result keeps what the tables use
    return rn, res


def cmd_a9(args):
    d = Path(args.dir).resolve()
    sizes = json.loads((d / "sizes.json").read_text())
    names = args.only.split(",") if args.only else list(sizes)
    vitasdk = args.vitasdk or os.environ.get("VITASDK")
    if not vitasdk:
        raise SystemExit("set VITASDK (vita/scripts/fetch_vitasdk.sh)")
    work = d / "work"
    work.mkdir(exist_ok=True)
    if not (work / "cli.dis").exists():
        arm_cc = args.arm_cc or os.environ.get("ARM_CC") or a9_cycles.default_arm_cc()
        _, dis = a9_cycles.build_cli(work, vitasdk, arm_cc)
        with open(work / "classes.txt", "w") as f:
            for a, (_, mn, ops) in sorted(a9_cycles.parse_dis(dis).items()):
                unit = a9_cycles.classify(mn, ops)[0]
                if unit != "I":
                    f.write(f"{a:x} {unit}\n")
    if not (work / "lines.bin").exists():
        n = write_synth_pack(work / "lines.bin", args.lines)
        (work / "pack.json").write_text(json.dumps({"lines": n, "source": "synth-val", "synth_val": str(
            os.environ.get("VOCR_SYNTH_VAL", "")), "sha256": hashlib.sha256((work / "lines.bin").read_bytes())
            .hexdigest()}))
    qemu = Path(args.qemu) if args.qemu else a9_cycles.default_qemu()
    out = d / "a9"
    out.mkdir(exist_ok=True)
    jobs = []
    for l2 in (int(x) for x in args.l2.split(",")):
        for name in names:
            rn = run_name(name, l2, args.tile, args.head, args.l2_ways, args.evict, args.rep)
            if not (out / f"{rn}.json").exists():
                jobs.append((name, d / "blobs" / f"{name}_w8.vocr", l2))
    # biggest first, so the long runs do not end up alone at the end
    jobs.sort(key=lambda j: -sizes[j[0]]["macs_per_frame"])
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_one, name, blob, l2, args.tile, args.head, work, qemu, out, args.l2_ways,
                          args.evict, args.rep) for name, blob, l2 in jobs]
        for fut in cf.as_completed(futs):
            rn, res = fut.result()
            (out / f"{rn}.json").write_text(json.dumps(res, indent=1))
            lo, hi = res["ms_640px"][444]
            print(f"{rn}: {res['cycles_per_frame'][444][0]:,.0f}-{res['cycles_per_frame'][444][1]:,.0f} cycles/frame, "
                  f"{lo:.0f}-{hi:.0f} ms/640 px at 444 MHz, L2 misses/frame {res['l2_misses_per_frame']:,.0f}",
                  file=sys.stderr, flush=True)


def verdict(info, run, c=CEILING):
    """[(what, value, limit, ok)] of one blob against the ceiling."""
    ms_hi = _ms(run, c["mhz"])[1]
    rows = [("int8 blob bytes", info["blob_bytes"], c["weights"]),
            ("W8I arena bytes", info["arena"]["t4h16"]["w8i"], c["arena"]),
            (f"est. ms per 640-px line, {c['mhz']} MHz, high", round(ms_hi, 1), c["ms"])]
    return [(what, v, lim, v <= lim) for what, v, lim in rows]


def cmd_check(args):
    out = Path(args.out or tempfile.mkdtemp(prefix="vocr-budget-check-")).resolve()
    (out / "blobs").mkdir(parents=True, exist_ok=True)
    sizes = {}
    for b in args.blobs:
        data = Path(b).read_bytes()
        name = Path(b).name.removesuffix(".vocr").removesuffix("_w8")
        sizes[name] = blob_info(data)
        shutil.copy2(b, out / "blobs" / f"{name}_w8.vocr")
    (out / "sizes.json").write_text(json.dumps(sizes, indent=1))
    args.dir, args.l2, args.only, args.tile, args.head = str(out), "2048", ",".join(sizes), 0, 0
    args.l2_ways, args.evict, args.rep = L2_WAYS, "lru", 0
    cmd_a9(args)
    runs = load_runs(out)
    ok = True
    for name, info in sizes.items():
        rows = verdict(info, runs[(name, *BASE)])
        ok &= all(r[3] for r in rows)
        print(f"{name}: " + "; ".join(f"{w} {v:,} <= {lim:,} {'ok' if good else 'OVER'}" for w, v, lim, good in rows))
    print(status_line())
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------- tables

def load_runs(d):
    """{(name, L2 KiB, tile, head, L2 ways, evict, rep): result} of every a9 run in DIR/a9."""
    runs = {}
    for p in sorted((Path(d) / "a9").glob("*.json")):
        r = json.loads(p.read_text())
        runs[(r["name"], r["l2_kib"], r["tile"], r["head"], r.get("l2_ways", L2_WAYS), r.get("evict", "lru"),
              r.get("rep", 0))] = r
    return runs


def _ms(r, mhz):
    """[low, high] estimated ms per 640-px line at mhz (JSON keys are strings)."""
    return r["ms_640px"].get(str(mhz), r["ms_640px"].get(mhz))


def _cyc(r, mhz):
    return r["cycles_per_frame"].get(str(mhz), r["cycles_per_frame"].get(mhz))


def _span(x, fmt="{:.0f}"):
    return f"{fmt.format(x[0])}-{fmt.format(x[1])}"


def tables(sizes, runs):
    """(markdown text, summary dict) for docs/vita-memory-budget.md."""
    order = [n for n in (*GRID, *RELEASED) if n in sizes]
    rank = {n: i for i, n in enumerate(order)}
    out, summary = [], {}
    out.append("| config | params | int8 blob B | of which head B | W8I arena B | MACs/frame | est. cycles/frame "
               "| cycles/MAC | est. ms per 640-px line, 444 MHz | 333 MHz |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for n in order:
        s = sizes[n]
        row = summary[n] = {"params": s["params"], "blob_bytes": s["blob_bytes"],
                            "head_weight_bytes": s["head_weight_bytes"], "split": s["split"],
                            "arena_w8i": s["arena"]["t4h16"]["w8i"], "arena_w8f": s["arena"]["t4h16"]["w8f"],
                            "macs_per_frame": s["macs_per_frame"], "released": bool(s.get("released"))}
        r = runs.get((n, *BASE))
        cells = ["-"] * 4
        if r:
            c4 = _cyc(r, 444)
            row.update(cycles_per_frame=c4, ms_444=_ms(r, 444), ms_333=_ms(r, 333),
                       l2_misses_per_frame=r["l2_misses_per_frame"])
            cells = [_span(c4, "{:,.0f}"), _span([c / s["macs_per_frame"] for c in c4], "{:.3f}"),
                     _span(_ms(r, 444)), _span(_ms(r, 333))]
        out.append(f"| {n} | {s['params']:,} | {s['blob_bytes']:,} | {s['head_weight_bytes']:,} | "
                   f"{s['arena']['t4h16']['w8i']:,} | {s['macs_per_frame']:,} | " + " | ".join(cells) + " |")
    # the L2 share: default tiling, 8-way LRU, each simulated L2 size
    l2s = sorted({k[1] for k in runs if k[2:] == BASE[1:]}, reverse=True)
    out.append("")
    out.append("| config | blob B | weight B read per frame | " + " | ".join(
        f"L2 {k} KiB: L2 misses/frame; est. ms 333 MHz" for k in l2s) + " |")
    out.append("|---|---|---|" + "---|" * len(l2s))
    for n in order:
        cells = []
        for k in l2s:
            r = runs.get((n, k, *BASE[1:]))
            if r:
                cells.append(f"{r['l2_misses_per_frame']:,.0f}; {_span(_ms(r, 333))}")
                summary[n].setdefault("l2", {})[k] = {"l2_misses_per_frame": r["l2_misses_per_frame"],
                                                     "l1d_misses_per_frame": r["l1d_misses_per_frame"],
                                                     "ms_333": _ms(r, 333), "ms_444": _ms(r, 444)}
            else:
                cells.append("-")
        if sum(c != "-" for c in cells) > 1:
            out.append(f"| {n} | {sizes[n]['blob_bytes']:,} | {sizes[n]['weight_bytes_per_frame']['t4h16']:,.0f} | "
                       + " | ".join(cells) + " |")
    # other tilings (8-way LRU), next to the default tiling at the same L2 size
    other = [k for k in runs if (k[2], k[3]) != BASE[1:3] and k[4:] == BASE[3:]]
    if other:
        out.append("")
        out.append("| config | L2 KiB | tile / head batch | W8I arena B | weight B read per frame | L2 misses/frame "
                   "| est. ms 333 MHz | same at tile 4 / head 16: L2 misses/frame; ms |")
        out.append("|---|---|---|---|---|---|---|---|")
        for k in sorted(other, key=lambda k: (rank[k[0]], -k[1], k[2], k[3])):
            r, s = runs[k], sizes[k[0]]
            key = f"t{k[2]}h{k[3]}"
            arena, wb = s["arena"][key]["w8i"], s["weight_bytes_per_frame"][key]
            d = runs.get((k[0], k[1], *BASE[1:]))
            ref = f"{d['l2_misses_per_frame']:,.0f}; {_span(_ms(d, 333))}" if d else "-"
            out.append(f"| {k[0]} | {k[1]} | {k[2]} / {k[3]} | {arena:,} | {wb:,.0f} | "
                       f"{r['l2_misses_per_frame']:,.0f} | {_span(_ms(r, 333))} | {ref} |")
            summary[k[0]].setdefault("tiling", {})[f"{key}_l2{k[1]}"] = {
                "arena_w8i": arena, "weight_bytes_per_frame": wb, "l2_misses_per_frame": r["l2_misses_per_frame"],
                "ms_333": _ms(r, 333), "ms_444": _ms(r, 444)}
    # the cache model itself: associativity and replacement, default tiling
    model = [k for k in runs if k[2:4] == BASE[1:3] and k[4:6] != BASE[3:5]]
    if model:
        out.append("")
        out.append("| config | L2 KiB | L2 ways | replacement | run | L1D misses/frame | L2 misses/frame "
                   "| est. ms 333 MHz | 8-way LRU: L1D misses; L2 misses; ms |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for k in sorted(model, key=lambda k: (rank[k[0]], -k[1], k[4], k[5], k[6])):
            r = runs[k]
            d = runs.get((k[0], k[1], *BASE[1:]))
            ref = (f"{d['l1d_misses_per_frame']:,.0f}; {d['l2_misses_per_frame']:,.0f}; {_span(_ms(d, 333))}"
                   if d else "-")
            out.append(f"| {k[0]} | {k[1]} | {k[4]} | {k[5]} | {k[6] or 1} | {r['l1d_misses_per_frame']:,.0f} | "
                       f"{r['l2_misses_per_frame']:,.0f} | {_span(_ms(r, 333))} | {ref} |")
            summary[k[0]].setdefault("cache_model", {})[f"l2{k[1]}_w{k[4]}_{k[5]}_r{k[6]}"] = {
                "l1d_misses_per_frame": r["l1d_misses_per_frame"], "l2_misses_per_frame": r["l2_misses_per_frame"],
                "ms_333": _ms(r, 333), "ms_444": _ms(r, 444)}
    return "\n".join(out) + "\n", summary


def cmd_table(args):
    d = Path(args.dir)
    text, summary = tables(json.loads((d / "sizes.json").read_text()), load_runs(d))
    sys.stdout.write(text)
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("blobs")
    b.add_argument("--out", required=True)
    b.add_argument("--only")
    a = sub.add_parser("a9")
    a.add_argument("--dir", required=True)
    a.add_argument("--l2", default="2048")
    a.add_argument("--only")
    a.add_argument("--tile", type=int, default=0)
    a.add_argument("--head", type=int, default=0)
    a.add_argument("--l2-ways", type=int, default=L2_WAYS)
    a.add_argument("--evict", default="lru", choices=("lru", "rand", "fifo"))
    a.add_argument("--rep", type=int, default=0)
    a.add_argument("--lines", type=int, default=40)
    a.add_argument("--jobs", type=int, default=5)
    a.add_argument("--vitasdk")
    a.add_argument("--arm-cc")
    a.add_argument("--qemu")
    t = sub.add_parser("table")
    t.add_argument("--dir", required=True)
    t.add_argument("--json")
    c = sub.add_parser("ckpt")
    c.add_argument("--name", required=True, choices=sorted(GRID))
    c.add_argument("--out", required=True)
    k = sub.add_parser("check")
    k.add_argument("blobs", nargs="+")
    k.add_argument("--out")
    k.add_argument("--lines", type=int, default=40)
    k.add_argument("--jobs", type=int, default=5)
    k.add_argument("--vitasdk")
    k.add_argument("--arm-cc")
    k.add_argument("--qemu")
    args = ap.parse_args()
    {"blobs": cmd_blobs, "a9": cmd_a9, "table": cmd_table, "ckpt": cmd_ckpt, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
