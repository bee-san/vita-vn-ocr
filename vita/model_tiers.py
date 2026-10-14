"""Small / Normal / Large: the recognizer tiers for OCR in the game's process (docs/model-tiers.md).

  model_tiers.py blobs --out DIR [--only NAME,...]   random-init int8 blobs of CANDIDATES, DIR/sizes.json
                                                     (budget_sweep.py's format: budget_sweep.py a9 --dir DIR runs them)
  model_tiers.py report --dir DIR [--dir DIR2 ...] --out JSON [--md MD] [--taxonomy GAPS.json]
                                                     the tier table: blob, arena, MACs, est. ms at 333 and 444 MHz
                                                     (640-px line, 917-px line, three-line box; 2 MiB L2 and a 1 MiB
                                                     share), the free game memory a job needs, the tier it fits;
                                                     with --taxonomy, the capacity curve and the recommendations
  model_tiers.py need --blob B --arena A             one model's free-memory need (API 19: no region dependence)

Tiers (owner, 2026-10-14, final 07:33Z/08:01Z): int8 weights Small < 1 MB, Normal <= 4 MB, Large <= 9 MB
(decimal MB; a blob under them is under the MiB reading too), the arena and line buffers on top; the time
target is per typical full textbox (3 lines of about 640 px) on one Cortex-A9 core at 444 MHz, estimated:
Small ≤ 1 s, Normal ≤ 2 s, Large ≤ 3 s. 333 MHz is reported for reference. A 640-px line is 80 CTC frames
at height 32, stride 8; a taller input or a coarser stride reads the same physical line in
FRAMES_640 * height / 32 * 8 / stride frames, and its time is scaled to that. A model fits a tier only if
its weights, its W8I arena and its estimated box time all meet the tier; the arena is checked at the tiling
the model would ship with (default 4-frame tiles, or 16-frame tiles where the doc pins them for a Large shape
that needs them to stay inside 3 s).

Times are vita/a9_cycles.py estimates (the low bound perfect overlap, the high bound no dual issue and every miss
at full latency), read from budget_sweep.py's a9 runs on synth-val lines only. They are not console measurements.
The tier test uses the high bound with the whole 2 MiB L2, as docs/vita-memory-budget.md's tiers do; the 1 MiB
column is the same estimate with the game holding half of the shared L2.

The free-memory need follows vita-jp-overlay main 64b5726 (API 19): game_ocr/main.c check_memory and
core/vocr_ocr.c layout / vjo_vocr_capture_bytes. A job holds the weights (USER_RW, page-rounded, resident between
jobs), a workspace (the line finder's arena, then the recognizer's, plus the runtime's two structs), a capture
block (one strip of grey rows, under 64 KiB at any region) and, only for a band taller than a strip, a line block
(at most VJO_VOCR_LINE_MAX_BYTES). The overlay's check counts weights + workspace + capture against free USER
minus a 4 MiB reserve; this tool also counts the line block's ceiling, and adds the tier's reserve.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "train"), str(ROOT / "eval")]
import budget_sweep as BS  # noqa: E402

PAGE = 4096
MB = 1_000_000
FRAMES_640 = 80                 # a 640-px line at height 32, stride 8 (a9_cycles.FRAMES_640)
FRAMES_917 = 115                # a full-width 917-px CLANNAD line (docs/vita-memory-budget.md: about 115 frames)
BOX_LINES = 3                   # the owner's "typical full textbox": three lines of about 640 px
CLOCKS = (333, 444)

# vita-jp-overlay main 64b5726: game_ocr/main.c, core/vocr_ocr.c, include/vjo_ledger_budget.h
OVERLAY = {
    "commit": "64b57266e1e663a6ea8f77d5739a57afc0735490",
    "runtime_structs": 3760,            # ALIGN16(sizeof(vocr_model)) + ALIGN16(sizeof(vocr_ctx)), vita/README.md
    "finder_arena_960x544": 83323,      # vocr_lines_arena_bytes(960, 544), vita/README.md
    "capture_block_max": 61440,         # vjo_vocr_capture_bytes, page-rounded, any region (docs/ingame-capture.md)
    "line_block_max": 516096,           # VJO_VOCR_LINE_MAX_BYTES = VJO_BUDGET_GAME_LINE
    "line_block_seen": 73728,           # largest band on the 372 linefind dev regions, page-rounded (ingame-capture.md)
    "reserve": 4 * 1024 * 1024,         # VJO_GAME_RESERVE
    "module_pages": 53248,              # VJO_BUDGET_WORKER_MODULE_PAGES (permanent while the game runs)
    "stack_tls": 32768 + 4096,          # VJO_BUDGET_GAME_STACK
    "weights_cap_today": 1558016,       # game_ocr/main.c weights_alloc refuses more than VJO_VOCR_F20_BYTES
}

# The owner's final ceilings (2026-10-14 07:33Z/08:01Z). The time target is per 3-line box (BOX_LINES x a
# 640-px line) on one A9 core at 444 MHz, high bound, whole 2 MiB L2. The arena ceiling and the W8I tiling a
# model ships with (t4h16, or t16h64 for a Large shape pinned to it): a model fits iff weights, arena and box
# time all meet the tier. The reserve per tier is this document's proposal (docs/model-tiers.md, "Selection
# rule"), not the overlay's current flat 4 MiB.
TIERS = (
    {"name": "Small", "weights_lt": 1 * MB, "arena_le": 131_072, "mhz": 444, "box_ms": 1000,
     "tiling": "t4h16", "reserve": 4 * 1024 * 1024},
    {"name": "Normal", "weights_le": 4 * MB, "arena_le": 262_144, "mhz": 444, "box_ms": 2000,
     "tiling": "t4h16", "reserve": 8 * 1024 * 1024},
    {"name": "Large", "weights_le": 9 * MB, "arena_le": 1_310_720, "mhz": 444, "box_ms": 3000,
     "tiling": "t16h64", "reserve": 16 * 1024 * 1024},
)

# name: (width, bottleneck, depth, ctx, ctx_k, classes incl. the CTC blank, height, frame stride, chans or None)
# Classes: 3,501 = FL10's 3,500; 4,987 and 5,631 = synth4-gap's VN-ordered proposals (4,986 / 5,630 characters);
# 5,355 = data/vocab.txt; 6,501 / 7,501 / 11,001 the larger heads vocab-expand prices. Heights 40 and 48 collapse
# 3 rows instead of 2 (the runtime's collapse takes any h_in with h_in x channels <= 4096). Stride 16 makes the
# (2,1) stage (2,2), so a line has half the frames.
CANDIDATES = {
    # Small: < 1 MB, box <= 1 s at 444 MHz (owner's final per-textbox target)
    "s_w100v3500": (1.0, 96, 1, 2, 5, 3501, 32, 8, None),        # FL10's shape (FL10's own vocab: 657,664 B)
    "s_w100v4987": (1.0, 96, 1, 2, 5, 4987, 32, 8, None),
    "s_w100v5631": (1.0, 96, 1, 2, 5, 5631, 32, 8, None),
    "s_w110v4987": (1.1, 96, 1, 2, 5, 4987, 32, 8, None),
    "s_w110v5631": (1.1, 96, 1, 2, 5, 5631, 32, 8, None),
    "s_w120v3500": (1.2, 96, 1, 2, 5, 3501, 32, 8, None),
    "s_w120v4987b64": (1.2, 64, 1, 2, 5, 4987, 32, 8, None),
    "s_w125v4987b64": (1.25, 64, 1, 2, 5, 4987, 32, 8, None),
    "s_w100v4987c3": (1.0, 96, 1, 3, 5, 4987, 32, 8, None),
    # Normal: <= 4 MB, box <= 2 s at 444 MHz
    "n_w200": (2.0, 96, 1, 2, 5, 5355, 32, 8, None),             # W20 / F20's shape
    "n_w200v5631": (2.0, 96, 1, 2, 5, 5631, 32, 8, None),
    "n_w250v5631": (2.5, 96, 1, 2, 5, 5631, 32, 8, None),
    "n_w250v5631b128": (2.5, 128, 1, 2, 5, 5631, 32, 8, None),
    "n_w275v5631": (2.75, 96, 1, 2, 5, 5631, 32, 8, None),
    "n_w300v5631": (3.0, 96, 1, 2, 5, 5631, 32, 8, None),
    "n_w250d2v5631": (2.5, 96, 2, 2, 5, 5631, 32, 8, None),
    "n_w200h40v5631": (2.0, 96, 1, 2, 5, 5631, 40, 8, None),
    "n_w200h48v5631": (2.0, 96, 1, 2, 5, 5631, 48, 8, None),
    "n_w250h40v5631": (2.5, 96, 1, 2, 5, 5631, 40, 8, None),
    "n_w250h48s16v5631": (2.5, 96, 1, 2, 5, 5631, 48, 16, None),
    "n_lw300v5631": (2.0, 96, 1, 2, 5, 5631, 32, 8, (64, 128, 256, 480, 576)),   # late-wide: width 2.0 early, 3.0 late
    # Large: <= 9 MB, box <= 3 s at 444 MHz (3-line box)
    # vocab-expand FINAL: 6,459 = the full VN list, no Chinese-only classes (fewer than 7,501)
    "l_w300v6459b192": (3.0, 192, 1, 2, 5, 6459, 32, 8, None),
    "l_w300v6459b128": (3.0, 128, 1, 2, 5, 6459, 32, 8, None),
    "l_w300v7501b128": (3.0, 128, 1, 2, 5, 7501, 32, 8, None),
    "l_w300v7501b192": (3.0, 192, 1, 2, 5, 7501, 32, 8, None),
    "l_w300v11001b192": (3.0, 192, 1, 2, 5, 11001, 32, 8, None),
    "l_w300c4v5631": (3.0, 96, 1, 4, 5, 5631, 32, 8, None),
    "l_w325v5631": (3.25, 96, 1, 2, 5, 5631, 32, 8, None),
    "l_w350v5631": (3.5, 96, 1, 2, 5, 5631, 32, 8, None),
    "l_w400v5631": (4.0, 96, 1, 2, 5, 5631, 32, 8, None),
    "l_lw400v7501b192": (3.0, 192, 1, 2, 5, 7501, 32, 8, (96, 192, 384, 640, 768)),   # late-wide: 3.0 early, 4.0 late
    "l_w250h48s16v7501b192": (2.5, 192, 1, 2, 5, 7501, 48, 16, None),
}


# ---------------------------------------------------------------- models

def collapse_rows(height):
    """Rows left after the stem and the three downsampling stages (each a 3 x 3 conv, padding 1, stride 2 in
    height): the height the collapse conv has to cover."""
    h = height
    for _ in range(4):
        h = (h + 2 - 3) // 2 + 1
    return h


def build_model(cfg, seed=0):
    """A random-init TinyVNOCR (random BatchNorm statistics, as budget_sweep.random_model) for one CANDIDATES
    entry. Heights other than 24-32 replace the 2 x 1 collapse by one over the rows that are left; stride 16
    makes the (2,1) stage (2,2)."""
    import torch
    import torch.nn as nn

    from model import DS, TinyVNOCR, cbr
    width, bn, depth, ctx, ctx_k, n_classes, height, stride, chans = cfg
    if stride not in (8, 16):
        raise ValueError(f"frame stride {stride}: only 8 and 16 are built")
    torch.manual_seed(seed)
    model = TinyVNOCR(n_classes, width=width, bottleneck=bn, depth=depth, ctx=ctx, ctx_k=ctx_k, chans=chans)
    i_stage4, i_collapse = 3 + 2 * depth, 4 + 3 * depth
    stage4, collapse = model.features[i_stage4], model.features[i_collapse]
    c128, c160 = stage4.dw[0].in_channels, stage4.pw[0].out_channels
    c192 = collapse[0].out_channels
    if stride == 16:
        model.features[i_stage4] = DS(c128, c160, (2, 2))
    rows = collapse_rows(height)
    if rows != 2:
        model.features[i_collapse] = cbr(c160, c192, (rows, 1), pad=(0, 0))
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_mean.normal_(0, 0.1)
                mod.running_var.uniform_(0.5, 2.0)
                mod.weight.uniform_(0.5, 1.5)
                mod.bias.normal_(0, 0.1)
    return model.eval()


def export_blob(cfg):
    import export
    model = build_model(cfg)
    data, meta = export.build_blob(export.model_ops(model, cfg[6]), BS.vocab_of(cfg[5]), cfg[6], int8=True)
    return data, meta


def cmd_blobs(args):
    out = Path(args.out)
    (out / "blobs").mkdir(parents=True, exist_ok=True)
    sizes_path = out / "sizes.json"
    sizes = json.loads(sizes_path.read_text()) if sizes_path.exists() else {}
    for name in (args.only.split(",") if args.only else CANDIDATES):
        cfg = CANDIDATES[name]
        data, meta = export_blob(cfg)
        (out / "blobs" / f"{name}_w8.vocr").write_bytes(data)
        info = BS.blob_info(data)
        width, bn, depth, ctx, ctx_k, n_classes, height, stride, chans = cfg
        info.update(config=dict(width=width, bottleneck=bn, depth=depth, ctx=ctx, ctx_k=ctx_k, classes=n_classes,
                                height=height, stride=stride, chans=list(chans) if chans else None),
                    random_init=True, params=meta["params"], sha256=hashlib.sha256(data).hexdigest())
        sizes[name] = info
        print(f"{name}: {len(data):,} B, W8I arena {info['arena']['t4h16']['w8i']:,} B, "
              f"{info['macs_per_frame']:,} MACs/frame, stride {meta['stride']}", file=sys.stderr)
    sizes_path.write_text(json.dumps(sizes, indent=1, ensure_ascii=False))


# ---------------------------------------------------------------- memory

def page(n):
    return -(-int(n) // PAGE) * PAGE


def job_need(blob_bytes, arena, finder_arena=None, line_block=None, resident=False):
    """Bytes of the game's USER memory one job takes, per the overlay's API 19 layout: weights (0 when
    already resident), workspace, capture block and the line block's ceiling. Since API 19 none of it depends
    on the region: the workspace holds no copy of it, the capture block is under 64 KiB at any region, and the
    finder's arena (83,323 B at 960 x 544, the largest) is smaller than every recognizer arena here."""
    finder = OVERLAY["finder_arena_960x544"] if finder_arena is None else finder_arena
    scratch = -(-max(finder, arena) // 16) * 16
    workspace = page(scratch + OVERLAY["runtime_structs"])
    weights = 0 if resident else page(blob_bytes)
    line = OVERLAY["line_block_max"] if line_block is None else line_block
    return {"weights": weights, "workspace": workspace, "capture": OVERLAY["capture_block_max"],
            "line_block": line, "total": weights + workspace + OVERLAY["capture_block_max"] + line}


def free_needed(blob_bytes, arena, reserve):
    """Free USER (sceKernelGetFreeMemorySize().size_user) the first job of a session needs: the job, the line
    block's ceiling and the reserve."""
    return job_need(blob_bytes, arena)["total"] + reserve


def select_tier(free_user, models, resident=None):
    """The runtime rule (docs/model-tiers.md): models is a list of dicts {name, tier, blob, arena, reserve},
    largest first. Pick the first whose need (weights unless resident, workspace, capture, line block) plus its
    tier's reserve fits free_user, counting another model's resident weights as free (they are released
    first). Returns the chosen name, or None when even the last does not fit. free_user < 0 (the query failed)
    chooses nothing."""
    if free_user is None or free_user < 0:
        return None
    for m in models:
        is_res = resident is not None and resident["name"] == m["name"]
        release = page(resident["blob"]) if resident is not None and not is_res else 0
        need = job_need(m["blob"], m["arena"], resident=is_res)["total"]
        if free_user + release - m["reserve"] >= need:
            return m["name"]
    return None


# ---------------------------------------------------------------- report

def frames_per_line(height, stride, frames_640=FRAMES_640):
    """CTC frames of one physical line that is 640 px wide at height 32 (frames_640 at stride 8, height 32)."""
    return frames_640 * height / 32 * 8 / stride


def run_ms(run, mhz, frames):
    """[low, high] est. ms for `frames` frames of the run's model at mhz."""
    cyc = run["cycles_per_frame"].get(str(mhz), run["cycles_per_frame"].get(mhz))
    return [c * frames / (mhz * 1000) for c in cyc]


def tier_fit(tier, info, r):
    """A model fits a tier when its weights are under the ceiling and some admissible tiling keeps the W8I
    arena under the tier's arena ceiling and the estimated high-bound 3-line-box time under its budget. The
    default 4-frame tiling is preferred (least arena/memory, and the overlay's default); 16-frame tiles are
    tried only where the tier allows them and the default tiling is over the box budget. Returns (fits,
    tiling, arena, box_ms_high), or None for fits when no needed a9 run exists for any admissible tiling."""
    w = info["blob_bytes"] < tier["weights_lt"] if "weights_lt" in tier else info["blob_bytes"] <= tier["weights_le"]
    tilings = ["t4h16", "t16h64"] if tier["tiling"] == "t16h64" else ["t4h16"]
    first = None       # the first admissible tiling with a run, for the "fits=False" fallback report
    saw_arena = False  # some tiling's arena is under the ceiling
    for t in tilings:
        arena = info["arena"][t]["w8i"]
        if arena > tier["arena_le"]:
            continue
        saw_arena = True
        box = _box_ms(r, t, tier["mhz"])
        if box is None:
            continue
        if first is None:
            first = (w and box[1] <= tier["box_ms"], t, arena, box[1])
        if w and box[1] <= tier["box_ms"]:
            return (True, t, arena, box[1])      # the first (lowest-arena) tiling that fits
    if first is not None:
        return first                             # none fit on time: report the lowest-arena tiling's figures
    if not saw_arena:
        return (False, tier["tiling"], None, None)   # no tiling's arena is under the ceiling: a definite miss
    return (None, tier["tiling"], None, None)        # arena fits but no a9 run yet: unmeasured


def _arena_for_tiling(info, tiling):
    """The W8I arena at a tiling key ('t4h16' or 't16h64')."""
    return info["arena"][tiling]["w8i"]


def _box_ms(r, tiling, mhz):
    """[low, high] est. ms per 3-line box at a tiling and clock, or None when that tiling has no a9 run
    (so a tier verdict is null, never the default tiling's time stood in for a bigger tile)."""
    key = "l22048" if tiling == "t4h16" else "l22048_t16h64"
    b = r["ms"].get(key)
    return b[str(mhz)]["box3x640"] if b else None


def row(name, info, runs):
    cfg = info.get("config", {})
    height, stride = info.get("height", cfg.get("height", 32)), cfg.get("stride", 8)
    f640, f917 = frames_per_line(height, stride), frames_per_line(height, stride, FRAMES_917)
    arena = info["arena"]["t4h16"]["w8i"]
    r = {"name": name, "config": cfg, "params": info.get("params"), "blob_bytes": info["blob_bytes"],
         "head_weight_bytes": info["head_weight_bytes"], "arena_w8i": arena,
         "arena_w8i_t16h64": info["arena"]["t16h64"]["w8i"], "macs_per_frame": info["macs_per_frame"],
         "head_macs_per_frame": info["head_macs_per_frame"], "frames_640": f640,
         "macs_per_640px_line": round(info["macs_per_frame"] * f640),
         "params_per_mac": round(info["params"] / info["macs_per_frame"], 3) if info.get("params") else None,
         "ms": {}, "l2_misses_per_frame": {}}
    for (rn, l2, tile, head, ways, evict, rep), run in runs.items():
        if rn != name or ways != BS.L2_WAYS or evict != "lru" or rep:
            continue
        key = f"l2{l2}" + ("" if (tile, head) == (BS.DEFAULT_TILE, BS.DEFAULT_HEAD) else f"_t{tile}h{head}")
        r["l2_misses_per_frame"][key] = round(run["l2_misses_per_frame"], 1)
        r["ms"][key] = {str(mhz): {"line640": [round(x, 1) for x in run_ms(run, mhz, f640)],
                                   "line917": [round(x, 1) for x in run_ms(run, mhz, f917)],
                                   "box3x640": [round(BOX_LINES * x, 1) for x in run_ms(run, mhz, f640)]}
                        for mhz in CLOCKS}
    # Each tier's verdict uses the best admissible tiling (t4, or t16 where the tier allows it): weights,
    # the arena at that tiling, and the box time at that tiling must all meet the tier.
    r["tiers"] = {}
    r["tier_detail"] = {}
    for t in TIERS:
        fits, tiling, tarena, box_hi = tier_fit(t, info, r)
        r["tiers"][t["name"]] = fits
        chosen = tiling if tarena is not None else t["tiling"]
        used_arena = tarena if tarena is not None else _arena_for_tiling(info, t["tiling"])
        r["tier_detail"][t["name"]] = {"tiling": chosen, "arena_w8i": used_arena, "box_ms_high_444": box_hi}
        r.setdefault("free_needed", {})[t["name"]] = free_needed(info["blob_bytes"], used_arena, t["reserve"])
        r.setdefault("job_need", {})[t["name"]] = job_need(info["blob_bytes"], used_arena)
    return r


def fmt_ms(r, key, mhz, what="line640"):
    ms = r["ms"].get(key, {}).get(str(mhz), {}).get(what)
    return f"{ms[0]:.0f}-{ms[1]:.0f}" if ms else "-"


def markdown(rows):
    head = ("| model | params | int8 blob B | head B | W8I arena B (t4h16/t16h64) | MACs/frame | "
            "MMAC per 640-px line | 3-line box @333 (t4/t16) | @444 (t4/t16) | @444 1 MiB L2 (t4) | fits |")
    out = [head, "|" + "|".join(["---"] * (head.count("|") - 1)) + "|"]
    for r in rows:
        fits = ", ".join(n for n, ok in r["tiers"].items() if ok) or "none"
        box333 = f"{fmt_ms(r, 'l22048', 333, 'box3x640')} / {fmt_ms(r, 'l22048_t16h64', 333, 'box3x640')}"
        box444 = f"{fmt_ms(r, 'l22048', 444, 'box3x640')} / {fmt_ms(r, 'l22048_t16h64', 444, 'box3x640')}"
        out.append(f"| {r['name']} | {r['params']:,} | {r['blob_bytes']:,} | {r['head_weight_bytes']:,} | "
                   f"{r['arena_w8i']:,} / {r['arena_w8i_t16h64']:,} | {r['macs_per_frame']:,} | "
                   f"{r['macs_per_640px_line'] / 1e6:.0f} | {box333} | {box444} | "
                   f"{fmt_ms(r, 'l21024', 444, 'box3x640')} | {fits} |")
    return "\n".join(out) + "\n"


# Matched-recipe capacity points: the realplus fine-tunes of each size, same arm and recipe, seeds averaged,
# on the pooled development cohort (rv). The arms (A_s2, C_mix) differ only in synthetic data; both are
# reported so the curve is not read off one. The FL10 128k-pretrain selection is excluded here because its 4x
# longer pretrain confounds the per-MB comparison (docs/vocr3.md: it moved FL10 by -0.85 points on its own).
CAPACITY_ARMS = {
    "C_mix": {"FL10": ["rp_FL10_C_mix_seed1", "rp_FL10_C_mix_seed2", "rp_FL10_C_mix_seed3"],
              "H15": ["rp_H15_C_mix_seed1", "rp_H15_C_mix_seed2"], "W20": ["rp_W20_C_mix_seed1"]},
    "A_s2": {"FL10": ["rp_FL10_A_s2_seed1", "rp_FL10_A_s2_seed2", "rp_FL10_A_s2_seed3"],
             "H15": ["rp_H15_A_s2_seed1", "rp_H15_A_s2_seed2"], "W20": ["rp_W20_A_s2_seed1"]},
}
SIZE_BLOB = {"FL10": 657_664, "H15": 1_155_968, "W20": 1_558_016}


def capacity(taxonomy_path):
    """The matched-recipe capacity curve from eval/model_gaps.py's taxonomy JSON (dev CER, rv)."""
    tax = json.loads(Path(taxonomy_path).read_text())
    out = {"note": "matched-recipe realplus fine-tunes, seeds averaged, pooled development CER (rv); the FL10 "
           "128k-pretrain selection is excluded (its longer pretrain confounds the per-MB comparison, "
           "docs/vocr3.md); H15's 40k pretrain vs W20's 32k is a smaller residual confound", "arms": {}}
    for arm, sizes in CAPACITY_ARMS.items():
        pts = []
        for size, runs in sizes.items():
            have = [r for r in runs if r in tax["runs"]]
            if not have:
                continue
            e = sum(tax["runs"][r]["total"]["errors"] for r in have)
            c = tax["runs"][have[0]]["total"]["chars"]
            pts.append({"size": size, "blob_bytes": SIZE_BLOB[size], "runs": have,
                        "dev_cer_rv": round(e / (c * len(have)), 5)})
        steps = []
        for i in range(1, len(pts)):
            dmb = (pts[i]["blob_bytes"] - pts[i - 1]["blob_bytes"]) / 1e6
            dp = (pts[i - 1]["dev_cer_rv"] - pts[i]["dev_cer_rv"]) * 100
            steps.append({"from": pts[i - 1]["size"], "to": pts[i]["size"], "delta_MB": round(dmb, 3),
                          "points_gained": round(dp, 3), "points_per_MB": round(dp / dmb, 2)})
        out["arms"][arm] = {"points": pts, "steps": steps}
    return out


def recommendations(by_name):
    """One trained-now shape and one next-round candidate per tier, generated from the sized rows so the
    numbers always match the table. Trained-now = vocr4-train's size for the tier; next-round names a
    bigger shape that still fits the tier (its JSON row carries the exact sizes and ms)."""
    plan = {"Small": {"train_now": "s_w100v3500", "next_round": "s_w100v4987"},
            "Normal": {"train_now": "n_w200", "next_round": "n_lw300v5631"},
            "Large": {"train_now": "n_w200", "next_round": "l_w300v6459b192"}}
    rec = {}
    for tname, names in plan.items():
        t = next(x for x in TIERS if x["name"] == tname)
        tn, nr = by_name.get(names["train_now"]), by_name.get(names["next_round"])
        rec[tname] = {
            "weights_ceiling": t["weights_lt"] if "weights_lt" in t else t["weights_le"],
            "arena_ceiling": t["arena_le"], "box_ms_ceiling_444": t["box_ms"], "reserve": t["reserve"],
            "shipping_tiling": t["tiling"],
            "train_now": _rec_model(names["train_now"], tn, t, tname),
            "next_round": _rec_model(names["next_round"], nr, t, tname),
        }
    return rec


def _rec_model(name, r, tier, tname):
    """One recommendation entry at the tiling the tier verdict chose for this model, so arena, box time and
    free_needed are self-consistent (audit R2): free_needed == free_needed(blob, arena, reserve)."""
    det = r["tier_detail"][tname]
    tiling = det["tiling"]
    return {"name": name, "config": r["config"], "blob_bytes": r["blob_bytes"],
            "head_weight_bytes": r["head_weight_bytes"], "tiling": tiling, "arena_w8i": det["arena_w8i"],
            "box3x640_444": _box_ms(r, tiling, 444), "box3x640_333": _box_ms(r, tiling, 333),
            "free_needed": r["free_needed"][tname], "fits": r["tiers"][tname]}


# The VOCR3 selections, whose aggregate taxonomy tables the report copies into the JSON (audit R4) so the
# doc's Part-2 numbers are committed, not only in the uncommitted taxonomy file.
GAP_SELECTIONS = {"FL10": "rp_FL10_C_mix_pre128k_seed1", "H15": "rp_H15_C_mix_seed2", "W20": "rp_W20_C_mix_seed1"}


def gap_tables(taxonomy_path):
    """The selections' aggregate counts (total, by_class, by_stratum, finder, confusions) from the taxonomy
    JSON, counts only, so docs/model-tiers.json carries the Part-2 tables the doc cites."""
    tax = json.loads(Path(taxonomy_path).read_text())
    out = {"cohorts": tax["cohorts"], "selections": {}}
    for size, run in GAP_SELECTIONS.items():
        r = tax["runs"].get(run)
        if not r:
            continue
        out["selections"][size] = {"run": run, "total": r["total"], "by_class": r["by_class"],
                                   "by_stratum": r["by_stratum"], "finder": r["finder"],
                                   "confusion_kinds": r["confusion_kinds"], "confusions": r["confusions"][:30],
                                   "edge": r["edge"]}
    return out


def cmd_report(args):
    rows = []
    for d in args.dir:
        sizes = json.loads((Path(d) / "sizes.json").read_text())
        runs = BS.load_runs(d)
        names = [n for n in sizes if not args.only or n in args.only.split(",")]
        rows += [row(n, sizes[n], runs) for n in names]
    rows.sort(key=lambda r: r["macs_per_640px_line"])
    by_name = {r["name"]: r for r in rows}
    out = {"kind": "model-tiers", "tiers": TIERS, "overlay": OVERLAY, "frames_640": FRAMES_640,
           "frames_917": FRAMES_917, "box_lines": BOX_LINES,
           "estimate": "vita/a9_cycles.py via budget_sweep.py a9 (synth-val lines only; host estimate, not a "
           "console measurement); box time is BOX_LINES x a 640-px line, high bound, whole 2 MiB L2",
           "models": rows}
    if args.taxonomy:
        out["capacity"] = capacity(args.taxonomy)
        out["recommendations"] = recommendations(by_name)
        out["gaps"] = gap_tables(args.taxonomy)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    if args.md:
        Path(args.md).write_text(markdown(rows))
    print(markdown(rows))


def cmd_need(args):
    j = job_need(args.blob, args.arena)
    print(json.dumps({"job": j, "free_needed": {t["name"]: j["total"] + t["reserve"] for t in TIERS}}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("blobs")
    b.add_argument("--out", required=True)
    b.add_argument("--only")
    r = sub.add_parser("report")
    r.add_argument("--dir", action="append", required=True)
    r.add_argument("--only")
    r.add_argument("--out", required=True)
    r.add_argument("--md")
    r.add_argument("--taxonomy", help="eval/model_gaps.py taxonomy JSON: adds the capacity curve and the "
                   "per-tier recommendations to the output")
    n = sub.add_parser("need")
    n.add_argument("--blob", type=int, required=True)
    n.add_argument("--arena", type=int, required=True)
    args = ap.parse_args()
    {"blobs": cmd_blobs, "report": cmd_report, "need": cmd_need}[args.cmd](args)


if __name__ == "__main__":
    main()
