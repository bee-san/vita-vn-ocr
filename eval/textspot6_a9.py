#!/usr/bin/env python3
"""Cortex-A9 estimates, exact arenas, code sizes and stack peaks of textspot6's tagged spotters (VTSP version 4) on
runtime/vocr_textspot6.c (issue #128): textspot3's method (eval/textspot3_a9.py, textspot3_a9m.py, textspot3_a9r.py)
on the new runtime. Every time here is a host estimate: nothing ran on a console.

  textspot6_a9.py measure --out DIR [--rows NAME,...] [--set SET] [--nodark] [--cflags "-O2"] [--define D ...]
                          [--sizes 960x544,480x272] [--seed 1] [--jobs 5] [--vitasdk DIR] [--arm-cc GCC] [--qemu DIR]
                          [--shapes-json FILE]
  textspot6_a9.py report DIR [--also DIR --also-label TEXT ...] --json OUT [--md OUT] [--code-size JSON]
                         [--stack JSON] [--title TEXT]
  textspot6_a9.py codesize --out JSON [--vitasdk DIR] [--pspdev DIR] [--work DIR]
  textspot6_a9.py stack --out JSON [--vitasdk DIR] [--pspdev DIR] [--work DIR]

measure: eval/textspot_a9.py's measure, unchanged in its method: the Vita compiler's objects of the runtime
(arm-vita-eabi-gcc, Thumb-2, the flags of --cflags / --define for vocr_textspot6.c, -O2 for vocr_lines*.c) linked into
the driver runtime/vocr_textspot6_test.c by an ARM Linux toolchain, run under the plugin qemu-arm with the Vita's caches
on textspot's pinned procedural frame of each size (textspot_a9.frame, grey), --empty and --batch, priced by
vita/a9_cycles.py: est. ms at 333 MHz = high bound x 1.0007 / 333 MHz, at 444 MHz that x 333/444; the arenas the
runtime's own functions give (vocr_textspot6_frame_arena_bytes, vocr_textspot6_arena_bytes) under qemu-arm. A row is a
shape with tags ("NAME_t4": a random VTSP version-4 blob, train/textspot6_model.random_blob(shape, seed 1, 4 tags)) or
without ("NAME": the same shape's version-2 (family T) or version-3 (family M) blob, exactly as textspot3's A9 tables
made it: train/textspot2_export.random_blob(NAME, 1, half_above) for T, train/textspot3_model.random_blob(shape, 1) for
M), each at its shape's scale: "h480" (half_above 480: a 960x544 frame is halved, 480x272 is read as it is), "f1"
(half_above 0: a frame is read whole) or "m" (480 / 272). The MACs per frame are train/textspot6_model.cost (tagged
rows), train/textspot2_export.cost (T) or train/textspot3_model.cost (M) at the network's input.

The shapes: family T at half_above 480 t64n8c16, t48n8c16d8, t64n4c16 (train/textspot3_shapes.SHAPES); family T at full
resolution (half_above 0) T_FULL below; family M qS4, qM4, qL5 (registered by eval/textspot3_protocol.json); more with
--shapes-json ({name: shape}).

report: per row (shape, tags, scale), the default call and VOCR_TEXTSPOT_NO_DARK at 960x544 and 480x272 (@333 and
@444), -O2 NEON and the build of every --also directory (e.g. -Os); VTSP bytes, int8 weights, exact MACs, the arena for
any frame up to 960x544 (vocr_textspot6_arena_bytes(m, 960, 544)) and 480x272; the cost split by part (textspot3_a9m's
method: the high bound of each part's own instructions, the cache term of the whole run); for a tagged row, the
extra time of the tag maps against the same shape's textspot3 row of record (eval/textspot3_a9_t_20261021.json:
runtime/vocr_textspot2.c; eval/textspot3_a9_20261021.json: runtime/vocr_textspot3.c) where one exists, split into the
runtime's part (the same version-2/3 blob on vocr_textspot6.c minus that row) and the tags' part (the tagged row minus
the version-2/3 blob on vocr_textspot6.c). With --code-size and --stack, those JSONs.

codesize: the .text of the spotter's own sections that a --gc-sections link of a caller of load, both arena functions,
find and find_rows keeps (eval/textspot2_a9.py codesize's method and vita/textgate_size.py's flags), VitaSDK gcc -O2
and -Os, NEON and -DVOCR_NO_NEON, each without and with -DVOCR_TEXTSPOT6_NO_MS; psp-gcc's object .text (-O2 -G0,
-Os -G0, without and with -DVOCR_TEXTSPOT6_NO_MS).

stack: the peak stack of each public function (gcc -fcallgraph-info=su over vocr_textspot6.c and vocr_lines.c /
vocr_lines_rows.c; eval/textspot3_a9r.peaks: every frame static, the heaviest path), VitaSDK -O2 / -Os and psp-gcc
-O2 -G0 / -Os -G0, without and with -DVOCR_TEXTSPOT6_NO_MS; find also without the dark pass.
"""
import argparse
import concurrent.futures as cf
import contextlib
import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "train", ROOT / "synth", ROOT / "vita", ROOT / "psp"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot2_a9 as W2  # noqa: E402
import textspot3_a9m as W3  # noqa: E402
import textspot3_a9r as A9R  # noqa: E402
import textspot_a9 as A9  # noqa: E402

import textspot3_model as M3  # noqa: E402
import textspot3_shapes as S3  # noqa: E402
import textspot6_model as M6  # noqa: E402

TAGS = 4
SOURCES = ("vocr_textspot6.c", "vocr_lines.c", "vocr_lines_rows.c")
DRIVER, HEADER = "vocr_textspot6_test.c", "vocr_textspot6.h"
NO_MS = "-DVOCR_TEXTSPOT6_NO_MS"
T3_ROWS = {"T": ROOT / "eval" / "textspot3_a9_t_20261021.json", "M": ROOT / "eval" / "textspot3_a9_20261021.json"}
CODE_FILES = ("eval/textspot6_a9.py", "eval/textspot_a9.py", "eval/textspot2_a9.py", "eval/textspot3_a9m.py",
              "eval/textspot3_a9r.py", "train/textspot6_model.py", "train/textspot3_model.py",
              "train/textspot3_shapes.py", "train/textspot2_export.py", "synth/textspot6_twin.py",
              "synth/textspot3_twin.py", "synth/textspot2_twin.py", "train/textdet_model.py",
              "train/textdet_export.py", "vita/a9_cycles.py")


def t_full(name, c2, n, dil):
    """A family-T shape read at full resolution (half_above 0: a 960x544 frame is read whole)."""
    return {"name": name, "family": "T", "stride": 8, "c0": 8, "c1": 16, "c2": c2, "chans": [c2] * n,
            "dil": list(dil), "block": "sep", "head": "kroa_x2", "half_above": 0, "half_above_h": 0}


# family T at full resolution (the issue's grid; added shapes say so in the report)
T_FULL = {"t32n4c16": t_full("t32n4c16", 32, 4, (1, 2, 4, 1)),
          "t32n6c16": t_full("t32n6c16", 32, 6, (1, 2, 4, 1, 2, 4)),
          "t24n6c16": t_full("t24n6c16", 24, 6, (1, 2, 4, 1, 2, 4)),
          "t48n4c16": t_full("t48n4c16", 48, 4, (1, 2, 4, 1))}
T_HALF = ("t64n8c16", "t48n8c16d8", "t64n4c16")
M_SHAPES = ("qS4", "qM4", "qL5")
EXTRA = {}          # --shapes-json, or what a measured directory's meta.json records
SETS = {"t480": [f"{n}_t{TAGS}" for n in T_HALF], "tfull": [f"{n}_t{TAGS}" for n in T_FULL],
        "m": [f"{n}_t{TAGS}" for n in M_SHAPES], "base-t480": list(T_HALF), "base-tfull": list(T_FULL),
        "base-m": list(M_SHAPES)}
SETS["v4"] = SETS["t480"] + SETS["tfull"] + SETS["m"]
SETS["base"] = SETS["base-t480"] + SETS["base-tfull"] + SETS["base-m"]
SETS["all"] = SETS["v4"] + SETS["base"]
SCALES = {"h480": 480, "f1": 0, "m": 480}
PARTS = {
    "stream": ("feed", "chain", "delivered", "dest_row", "dest_stride", "ring_row", "to_f8", "pair_sums",
               "input_codes", "setup", "setup3", "produce", "producible", "drain", "k3_block_s1", "k3_block_s2",
               "k1_block", "grey8", "grey_px", "src_row", "memset"),
    "map": ("run_pw", "run_dw", "dw_row", "dw_border", "pw_cells", "pw_sums", "pw_store", "pw_block_run",
            "pw_gather", "pw_gathers", "pw_offsets", "pw_segs", "op_params", "rq4v", "copy_cells", "map_ref", "hsum4",
            "memcpy"),
    "decode": ("run_head", "sub_row", "sub_row3", "sub_plane", "sub_plane3", "decode_row", "sweep", "finish",
               "tag_of", "keep", "group", "isort", "rows_of", "joins", "find_root", "root_of", "unite", "cap_less",
               "key_less", "field", "same_row"),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base_shapes():
    """{name: shape} of every shape this tool knows: the T shapes at 480, T_FULL, the M shapes, EXTRA."""
    out = {n: dict(S3.SHAPES[n]) for n in T_HALF}
    out.update({n: dict(s) for n, s in T_FULL.items()})
    out.update({n: M3.check_shape(dict(S3.SHAPES[n])) for n in M_SHAPES})
    for n, s in EXTRA.items():
        if n in out and out[n] != s:
            raise SystemExit(f"{n}: another shape under this name")
        out[n] = s
    return out


def row_shape(row):
    """(the base shape, its name, tags) of a row name: NAME_tT (tagged, version 4) or NAME (version 2 or 3)."""
    shapes = base_shapes()
    name, tags = row, 0
    if "_t" in row and row.rsplit("_t", 1)[1].isdigit() and row.rsplit("_t", 1)[0] in shapes:
        name, tags = row.rsplit("_t", 1)[0], int(row.rsplit("_t", 1)[1])
    if name not in shapes:
        raise SystemExit(f"{row}: unknown shape {name}")
    return shapes[name], name, tags


def scale_of_row(row):
    shape, _, _ = row_shape(row)
    if M6.family_of(shape) == "M":
        return "m"
    ha = int(shape.get("half_above", S3.T_HALF_ABOVE))
    return {480: "h480", 0: "f1"}.get(ha) or f"h{ha}"


def cost_of(shape, width, height):
    """Exact MACs etc. of a registered row's network on a width x height network input."""
    tags = shape.get("tags", 0)
    base = {k: v for k, v in shape.items() if k not in ("tags", "row")}
    base["name"] = shape.get("base", base["name"])
    if tags:
        return M6.cost(base, width, height, tags)
    if M6.family_of(base) == "M":
        return M3.cost(base, width, height)
    import textspot2_export as X
    return X.cost(S3.t_plain(base), width, height)


def figures(row):
    """VTSP bytes, int8 weights, parameters, MACs at 960x544 and 480x272 (the network's input by the row's scale)."""
    shape, name, tags = row_shape(row)
    fam = M6.family_of(shape)
    ha = SCALES[scale_of_row(row)]
    hh = 272 if fam == "M" else int(shape.get("half_above_h", 0))
    macs = {}
    for w, h in ((960, 544), (480, 272)):
        f = 2 if (ha and w > ha) or (hh and h > hh) else 1
        macs[f"{w}x{h}"] = cost_of(dict(shape, tags=tags, base=name), -(-w // f), -(-h // f))["macs"]
    c = cost_of(dict(shape, tags=tags, base=name), 480, 272)
    if tags:
        vtsp = len(M6.random_blob(shape, 1, tags, n_cal=1)[0])
    elif fam == "M":
        vtsp = M3.vtsp_bytes(shape)
    else:
        import textspot2_export as X
        vtsp = 32 + S3.vtdt_bytes(X.layers(S3.t_plain(shape)))
    return {"row": row, "shape": name, "family": fam, "tags": tags, "half_above": ha, "half_above_h": hh,
            "vtsp_bytes": vtsp, "weights_int8": c["weights_int8"], "params": c["params"], "macs": macs,
            "version": 4 if tags else (3 if fam == "M" else 2)}


def blob_for_v6(row, scale, seed):
    """The row's random blob: version 4 (tags), else the version-2 / version-3 blob textspot3's tables measured."""
    shape, name, tags = row_shape(row)
    if tags:
        return M6.random_blob(shape, seed, tags)
    if M6.family_of(shape) == "M":
        return M3.random_blob(shape, seed)
    import textspot2_export as X
    with S3.inject({name: shape}):
        return X.random_blob(name, seed, half_above=SCALES[scale])


def build_v6(out, vitasdk, arm_cc, cflags=("-O2",)):
    """textspot3_a9m.build_v3's commands with runtime/vocr_textspot6.c and runtime/vocr_textspot6_test.c."""
    saved = (W3.SOURCES, W3.DRIVER)
    try:
        W3.SOURCES, W3.DRIVER = SOURCES, DRIVER
        return W3.build_v3(out, vitasdk, arm_cc, cflags)
    finally:
        W3.SOURCES, W3.DRIVER = saved


@contextlib.contextmanager
def runtime(rows):
    """eval/textspot_a9's module state for runtime/vocr_textspot6.c: its sources, build, blobs, the rows' shapes in
    train/textdet_model.SHAPES (one entry per row) and their costs; the scales."""
    import textdet_model as TM
    saved = (A9.SOURCES, A9.build, A9.blob_for, TM.cost)
    added = [r for r in rows if r not in TM.SHAPES]
    had = {k: A9.SCALES.get(k) for k in SCALES}
    orig_cost = TM.cost

    def cost(shape, width, height):
        if shape.get("row"):
            return cost_of(shape, width, height)
        return orig_cost(shape, width, height)

    try:
        A9.SOURCES, A9.build, A9.blob_for, TM.cost = SOURCES, build_v6, blob_for_v6, cost
        A9.SCALES.update(SCALES)
        for r in added:
            shape, name, tags = row_shape(r)
            TM.SHAPES[r] = dict(shape, name=r, row=r, base=name, tags=tags)
        yield
    finally:
        A9.SOURCES, A9.build, A9.blob_for, TM.cost = saved
        for k, v in had.items():
            if v is None:
                A9.SCALES.pop(k, None)
            else:
                A9.SCALES[k] = v
        for r in added:
            TM.SHAPES.pop(r, None)


def code_hashes():
    out = {f"runtime/{s}": sha256(ROOT / "runtime" / s) for s in (*SOURCES, DRIVER, HEADER, "vocr_textspot.h")}
    for f in CODE_FILES:
        out[f] = sha256(ROOT / f)
    return dict(sorted(out.items()))


def load_extra(path):
    for name, s in json.loads(Path(path).read_text(encoding="utf-8")).items():
        if M6.family_of(s) == "M":
            EXTRA[name] = M3.check_shape(dict(s, name=name))
        else:
            EXTRA[name] = dict(s, name=name)


# ----------------------------------------------------------------------------- measure

def cmd_measure(argv):
    ap = argparse.ArgumentParser(prog="textspot6_a9.py measure")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="")
    ap.add_argument("--set", default="", help=f"a row set: {', '.join(SETS)}")
    ap.add_argument("--shapes-json", default="")
    ap.add_argument("--nodark", action="store_true")
    ap.add_argument("--cflags", default="-O2")
    ap.add_argument("--define", action="append", default=[], help="e.g. VOCR_TEXTSPOT6_NO_MS (the spotter's object)")
    ap.add_argument("--sizes", default="960x544,480x272")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--vitasdk", default=os.environ.get("VITASDK", ""))
    ap.add_argument("--arm-cc", default="")
    ap.add_argument("--qemu", default="")
    args = ap.parse_args(argv)
    if args.shapes_json:
        load_extra(args.shapes_json)
    rows = [r for r in args.rows.split(",") if r] + (SETS[args.set] if args.set else [])
    if not rows or not args.vitasdk:
        raise SystemExit("--rows or --set, and --vitasdk (or VITASDK)")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    arm_cc = args.arm_cc or os.environ.get("ARM_CC") or A9.A.default_arm_cc()
    q = Path(args.qemu) if args.qemu else A9.A.default_qemu()
    sizes = [tuple(int(v) for v in s.split("x")) for s in args.sizes.split(",")]
    cflags = args.cflags.split() + [f"-D{d}" for d in args.define]
    with runtime(rows):
        drv, dis, funcs, objs, text = A9.build(out, args.vitasdk, arm_cc, cflags)
        jobs = []
        for r in rows:
            scale = scale_of_row(r)
            blob, src = A9.blob_for(r, scale, args.seed)
            for wh in sizes:
                jobs.append((out, q, drv, r, scale, wh, "grey", args.seed, blob, src, args.nodark))
        with cf.ThreadPoolExecutor(args.jobs) as ex:
            runs = list(ex.map(A9.measure_one, jobs))
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {"kind": "textspot6-a9", "computed": now,
            "method": A9.__doc__.split("\n\n")[1].strip().replace("\n", " "),
            "sources_sha256": {f"runtime/{s}": sha256(ROOT / "runtime" / s) for s in (*SOURCES, DRIVER, HEADER)},
            "tools": {"vita_cc": A9.tool_version(str(Path(args.vitasdk) / "bin/arm-vita-eabi-gcc")),
                      "arm_cc": A9.tool_version(arm_cc), "qemu": A9.tool_version(str(q / "bin/qemu-arm")),
                      "qemu_plugins_sha256": {p.name: sha256(p) for p in sorted((q / "plugins").glob("*.so"))},
                      "a9_cycles_sha256": sha256(ROOT / "vita/a9_cycles.py")},
            "flags": ["-std=c99", *A9.A.ARCH, *cflags, "(Thumb-2: the VitaSDK default; vocr_lines*.c at -O2)"],
            "cache": A9.A.CACHE_ARGS, "vocr_textspot_o_text_bytes": text, "calibration": A9.CAL, "seed": args.seed,
            "runs": runs, "textspot6": {"argv": W2.portable(["measure", *argv]), "rows": rows,
                                        "shapes": {r: row_shape(r)[0] for r in rows}, "cflags": cflags,
                                        "nodark": args.nodark}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{len(runs)} batches measured into {out}")


# ----------------------------------------------------------------------------- report

def part_of(fn, lines_fns):
    if fn in lines_fns:
        return "dark"
    base = fn.split(".")[0]
    for part, names in PARTS.items():
        if base in names:
            return part
    return "other"


def split(d, run):
    """textspot3_a9m.split with this runtime's parts."""
    saved = (W3.PARTS, W3.part_of)
    try:
        W3.PARTS, W3.part_of = PARTS, part_of
        return W3.split(d, run)
    finally:
        W3.PARTS, W3.part_of = saved


def table(d):
    """(meta, the rows of a measured directory with their cost split)."""
    meta = json.loads((Path(d) / "meta.json").read_text())
    for name, s in meta.get("textspot6", {}).get("shapes", {}).items():
        base = name.rsplit("_t", 1)[0] if "_t" in name else name
        if base not in base_shapes():
            EXTRA[base] = s
    rows = meta["textspot6"]["rows"]
    with runtime(rows):
        _, tab = A9.table_of(d)
    runs = {(r["shape"], tuple(r["size"])): r for r in meta["runs"]}
    for t in tab:
        t["split"] = split(Path(d), runs[(t["shape"], tuple(t["size"]))])
    return meta, tab


def t3_rows():
    """{(shape, (w, h), flags): row} of textspot3's A9 tables of record (main table and every also)."""
    out = {}
    for fam, path in T3_ROWS.items():
        doc = json.loads(path.read_text(encoding="utf-8"))
        for tab in [doc["table"]] + [a["table"] for a in doc.get("also", [])]:
            for t in tab:
                if t["format"] == "grey":
                    out[(t["shape"], tuple(t["size"]), t["flags"])] = {"ms_333": t["ms_per_frame"]["333"],
                                                                       "ms_444": t["ms_per_frame"]["444"],
                                                                       "arena_up_to": t["arena_bytes"],
                                                                       "half_above": t["half_above"], "table": fam}
    return out


def cmd_report(argv):
    ap = argparse.ArgumentParser(prog="textspot6_a9.py report")
    ap.add_argument("dir")
    ap.add_argument("--also", action="append", default=[])
    ap.add_argument("--also-label", action="append", default=[])
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", default="")
    ap.add_argument("--code-size", default="")
    ap.add_argument("--stack", default="")
    ap.add_argument("--title", default="")
    args = ap.parse_args(argv)
    labels = ["main"] + [args.also_label[i] if i < len(args.also_label) else f"also{i}" for i in range(len(args.also))]
    tables = []
    sources = None
    for d, label in zip([args.dir, *args.also], labels):
        meta, tab = table(d)
        if sources is None:
            sources = meta["sources_sha256"]
        elif meta["sources_sha256"] != sources:
            raise SystemExit(f"{d} was measured on other sources")
        flags = " ".join(meta["textspot6"]["cflags"])
        mode = "nodark" if meta["textspot6"]["nodark"] else "dark"
        tables.append({"label": label, "dir": W2.portable([str(d)])[0], "cflags": flags, "mode": mode,
                       "computed": meta["computed"], "argv": meta["textspot6"]["argv"], "table": tab,
                       "vocr_textspot6_o_text_bytes": meta["vocr_textspot_o_text_bytes"]})
    first = json.loads((Path(args.dir) / "meta.json").read_text())
    ref = t3_rows()
    allrows = sorted({t["shape"] for tb in tables for t in tb["table"]})
    figs = {r: figures(r) for r in allrows}
    cells = {}
    for tb in tables:
        for t in tb["table"]:
            cells.setdefault(t["shape"], {})[(tb["cflags"], tb["mode"], tuple(t["size"]))] = t
    summary = []
    for r in sorted(allrows, key=lambda r: (figs[r]["tags"] == 0, figs[r]["family"], figs[r]["half_above"] == 0,
                                            figs[r]["vtsp_bytes"])):
        fg = figs[r]
        row = dict(fg)
        for (cfl, mode, size), t in sorted(cells[r].items()):
            key = f"{'O2' if '-O2' in cfl.split() else 'Os' if '-Os' in cfl.split() else cfl}" + \
                (" no_ms" if NO_MS in cfl.split() else "") + (" scalar" if "-DVOCR_NO_NEON" in cfl.split() else "")
            tag = f"{size[0]}x{size[1]}"
            cell = row.setdefault("ms", {}).setdefault(key, {}).setdefault(mode, {})
            cell[tag] = {"333": t["ms_per_frame"]["333"], "444": t["ms_per_frame"]["444"]}
            if mode == "dark":
                row.setdefault("arena", {})[tag] = {"up_to": t["arena_bytes"], "frame": t["frame_arena_bytes"]}
                row.setdefault("boxes", {})[tag] = t["boxes"]
                if key == "O2":
                    row.setdefault("split_333", {})[tag] = {k: round(v["ms_333"], 2) for k, v in t["split"].items()}
        o2 = row.get("ms", {}).get("O2", {})
        if fg["tags"]:
            base = fg["shape"]
            extra = {}
            for mode in ("dark", "nodark"):
                for size in ((960, 544), (480, 272)):
                    tag = f"{size[0]}x{size[1]}"
                    mine = o2.get(mode, {}).get(tag)
                    old = ref.get((base, size, "0" if mode == "dark" else "VOCR_TEXTSPOT_NO_DARK"))
                    same = figs.get(base) and cells.get(base, {}).get(("-O2", mode, size))
                    e = {}
                    if mine and old:
                        e["vs_textspot3_row_ms_333"] = mine["333"] - old["ms_333"]
                        e["textspot3_row_ms_333"] = old["ms_333"]
                        e["textspot3_table"] = str(T3_ROWS[old["table"]].relative_to(ROOT))
                    if mine and same:
                        e["tags_ms_333"] = mine["333"] - same["ms_per_frame"]["333"]
                        e["v23_on_textspot6_ms_333"] = same["ms_per_frame"]["333"]
                        if old:
                            e["runtime_ms_333"] = same["ms_per_frame"]["333"] - old["ms_333"]
                    if e:
                        extra.setdefault(mode, {})[tag] = e
            if extra:
                row["extra_of_the_tags"] = extra
            b960 = cells.get(base, {}).get(("-O2", "dark", (960, 544)))
            if b960 and "960x544" in row.get("arena", {}):
                row["arena_of_the_tags_up_to_960"] = row["arena"]["960x544"]["up_to"] - b960["arena_bytes"]
        summary.append(row)
    res = {"kind": "textspot6-a9-report", "title": args.title,
           "note": ("estimates: qemu-arm instruction counts of the Vita compiler's code priced by vita/a9_cycles.py's "
                    "Cortex-A9 model, not a console measurement; est. ms @333 = high bound x 1.0007 / 333 MHz, est. ms "
                    "@444 = that x 333/444; nothing ran on a console. Arena, code and stack bytes are exact."),
           "method": first["method"], "calibration": A9.CAL, "cache": first["cache"], "tools": first["tools"],
           "sources_sha256": sources, "code_sha256": code_hashes(), "rows_how": __doc__.split("\n\n")[1].strip(),
           "shapes": {row_shape(r)[1]: row_shape(r)[0] for r in allrows},
           "textspot3_rows_of_record": {k: str(v.relative_to(ROOT)) for k, v in T3_ROWS.items()},
           "summary": summary, "tables": [{k: v for k, v in tb.items() if k != "table"} for tb in tables]}
    if args.code_size:
        res["code_size"] = json.loads(Path(args.code_size).read_text(encoding="utf-8"))
    if args.stack:
        res["stack"] = json.loads(Path(args.stack).read_text(encoding="utf-8"))
    res["table"] = [dict(t, table=tb["label"]) for tb in tables for t in tb["table"]]
    md = markdown(res)
    print(md)
    Path(args.json).write_text(dumps(res) + "\n")
    print("wrote", args.json)
    if args.md:
        Path(args.md).write_text(md + "\n")


TABLE_KEYS = ("shape", "scale", "half_above", "size", "format", "flags", "f", "net_input", "macs", "blob_sha256",
              "blob_bytes", "frame_sha256", "rc", "boxes", "blocks", "pick", "truncated", "insns_per_frame",
              "cycles_per_frame", "ms_per_frame", "l1d_misses_per_frame", "l2_misses_per_frame", "low_bound_set_by",
              "frame_arena_bytes", "arena_bytes", "negative_count_addresses", "split", "table")


def dumps(res):
    """The JSON with every table and summary row on one line."""
    def rows(t, keys=None):
        return "[\n" + ",\n".join("  " + json.dumps({k: r[k] for k in keys if k in r} if keys else r,
                                                     separators=(",", ":")) for r in t) + "\n ]"
    head = {k: v for k, v in res.items() if k not in ("table", "summary")}
    out = json.dumps(head, indent=1)[:-2]
    return out + ',\n "summary": ' + rows(res["summary"]) + ',\n "table": ' + rows(res["table"], TABLE_KEYS) + "\n}"


def _f(v, fmt="{:.1f}"):
    return "-" if v is None else fmt.format(v)


def markdown(res):
    out = [res["title"] or "", "", res["note"], "",
           "| row | VTSP v | VTSP B | int8 weights | MACs / 960x544 | MACs / 480x272 | 960x544 @333 / @444 | no dark "
           "@333 | 480x272 @333 / @444 | -Os 960x544 @333 | -Os no dark @333 | arena up to 960x544 | arena 480x272 | "
           "tags vs textspot3 row @333 (runtime + tags) |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in res["summary"]:
        o2, os_ = r.get("ms", {}).get("O2", {}), r.get("ms", {}).get("Os", {})
        g = lambda d, m, s, k="333": d.get(m, {}).get(s, {}).get(k)  # noqa: E731
        ex = r.get("extra_of_the_tags", {}).get("dark", {}).get("960x544", {})
        exs = "-"
        if "vs_textspot3_row_ms_333" in ex:
            exs = f"{ex['vs_textspot3_row_ms_333']:+.1f} ({_f(ex.get('runtime_ms_333'), '{:+.1f}')} + " \
                  f"{_f(ex.get('tags_ms_333'), '{:+.1f}')})"
        elif "tags_ms_333" in ex:
            exs = f"tags {ex['tags_ms_333']:+.1f}"
        out.append(f"| {r['row']} | {r['version']} | {r['vtsp_bytes']:,} | {r['weights_int8']:,} | "
                   f"{r['macs']['960x544']:,} | {r['macs']['480x272']:,} | {_f(g(o2, 'dark', '960x544'))} / "
                   f"{_f(g(o2, 'dark', '960x544', '444'))} | {_f(g(o2, 'nodark', '960x544'))} | "
                   f"{_f(g(o2, 'dark', '480x272'))} / {_f(g(o2, 'dark', '480x272', '444'))} | "
                   f"{_f(g(os_, 'dark', '960x544'))} | {_f(g(os_, 'nodark', '960x544'))} | "
                   f"{r.get('arena', {}).get('960x544', {}).get('up_to', 0):,} | "
                   f"{r.get('arena', {}).get('480x272', {}).get('up_to', 0):,} | {exs} |")
    cs = res.get("code_size")
    if cs:
        out += ["", "| vocr_textspot6.c code (.text, --gc-sections) | bytes |", "|---|---:|"]
        for k, v in cs["vita"]["builds"].items():
            out.append(f"| VitaSDK {k} | {v['text']:,} (object {v['object_text']:,}) |")
        for k, v in cs.get("psp", {}).get("object_text", {}).items():
            out.append(f"| psp-gcc {k} (object) | {v:,} |")
    st = res.get("stack")
    if st:
        builds = list(st["builds"])
        out += ["", "| stack peak (B) | " + " | ".join(builds) + " |", "|---|" + "---:|" * len(builds)]
        for root in STACK_ROOTS:
            out.append(f"| {root} | " + " | ".join(str(st["builds"][b]["peaks"].get(root, {}).get("bytes", "-"))
                                                for b in builds) + " |")
        out.append("| vocr_textspot6_find without the dark pass | " + " | ".join(
            str(st["builds"][b]["no_dark"].get("vocr_textspot6_find without the dark pass", {}).get("bytes", "-"))
            for b in builds) + " |")
    return "\n".join(out)


# ----------------------------------------------------------------------------- code size and stack

CALLER = """
static vocr_textspot6 m;
static VocrTextspot6Set set;
int main(void)
{
    VocrLinesImage im;
    VocrLinesRows src = {960, 544, 1, row_of, NULL};
    size_t need;
    int rc;
    if (vocr_textspot6_load(&m, (const void *)blobp, blobn)) return 1;
    need = vocr_textspot6_arena_bytes(&m, 960, 544) + vocr_textspot6_frame_arena_bytes(&m, 960, 544);
    im.pixels = row; im.width = 960; im.height = 1; im.stride = 960; im.bpp = 1;
    rc = vocr_textspot6_find(&m, &im, 0, arena, need, &set);
    return rc + vocr_textspot6_find_rows(&m, &src, 0, arena, need, &set);
}
"""


def cmd_codesize(argv):
    import textgate_size as G
    ap = argparse.ArgumentParser(prog="textspot6_a9.py codesize")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vitasdk", default=str(W2.HERD / "shared/artifacts/vita/vitasdk"))
    ap.add_argument("--pspdev", default=str(W2.HERD / "shared/artifacts/psp/tools/pspdev"))
    ap.add_argument("--work", default="")
    args = ap.parse_args(argv)
    vcc = Path(args.vitasdk) / "bin/arm-vita-eabi-gcc"
    pcc = Path(args.pspdev) / "bin/psp-gcc"
    work = args.work or tempfile.mkdtemp(prefix="textspot6-size-")
    saved_caller, saved_src = dict(W2.CALLER), W2.V2_SOURCES
    doc = {"kind": "textspot6-code-size", "runtime": "v6", "source": "runtime/vocr_textspot6.c",
           "computed": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "how": ("eval/textspot2_a9.py codesize's: the .text of the spotter's own sections kept by a --gc-sections "
                   "link (-Wl,-Map) of a caller of load, both arena functions, find and find_rows; vita/textgate_size"
                   ".py's flags and map parser; vocr_lines*.c and libgcc listed apart, not counted; each build "
                   f"without and with {NO_MS}"),
           "sources_sha256": code_hashes(), "vita": {"compiler": G.tool_version(vcc), "builds": {}}}
    try:
        W2.CALLER["v6"] = (HEADER, CALLER)
        W2.V2_SOURCES = SOURCES
        for noms in (False, True):
            for neon in (True, False):
                for level in ("-O2", "-Os"):
                    extra = ([] if neon else ["-DVOCR_NO_NEON"]) + ([NO_MS] if noms else [])
                    flags = G.RUBI_FLAGS + G.VITA_ARCH + [level] + extra
                    tag = f"vita{level}{'' if neon else '_scalar'}{'_noms' if noms else ''}"
                    r = W2.link_size(vcc, flags, "v6", work, tag)
                    r["object_text"] = W2.object_text(vcc, Path(args.vitasdk) / "bin/arm-vita-eabi-size",
                                                      ["-std=c99", *A9.A.ARCH, level, *extra], SOURCES[0], work,
                                                      tag + "_obj")
                    r["flags"] = flags
                    doc["vita"]["builds"][" ".join([level, *extra])] = r
        if pcc.exists():
            doc["psp"] = {"compiler": G.tool_version(pcc), "object_text": {}}
            for noms in (False, True):
                for level in ("-O2", "-Os"):
                    extra = [NO_MS] if noms else []
                    doc["psp"]["object_text"][" ".join([level, "-G0", *extra])] = W2.object_text(
                        pcc, Path(args.pspdev) / "bin/psp-size", ["-std=c99", level, "-G0", *extra], SOURCES[0], work,
                        f"psp{level}{'_noms' if noms else ''}")
        else:
            doc["psp"] = {"error": f"no psp-gcc at {pcc}"}
    finally:
        W2.CALLER.clear()
        W2.CALLER.update(saved_caller)
        W2.V2_SOURCES = saved_src
    print("runtime/vocr_textspot6.c: VitaSDK --gc-sections .text (object .text)")
    for k, r in doc["vita"]["builds"].items():
        print(f"  {k:46s} {r['text']:7,d} B (+ rodata {r['rodata']} B; object {r['object_text']:,} B)")
    for k, v in doc.get("psp", {}).get("object_text", {}).items():
        print(f"  psp-gcc {k:38s} {v:7,d} B (object .text)")
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print("wrote", args.out)


STACK_ROOTS = ("vocr_textspot6_find", "vocr_textspot6_find_rows", "vocr_textspot6_load", "vocr_textspot6_maps",
               "vocr_textspot6_arena_bytes", "vocr_textspot6_frame_arena_bytes")


def cmd_stack(argv):
    ap = argparse.ArgumentParser(prog="textspot6_a9.py stack")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vitasdk", default=str(W2.HERD / "shared/artifacts/vita/vitasdk"))
    ap.add_argument("--pspdev", default=str(W2.HERD / "shared/artifacts/psp/tools/pspdev"))
    ap.add_argument("--work", default="")
    args = ap.parse_args(argv)
    work = Path(args.work or tempfile.mkdtemp(prefix="textspot6-stack-"))
    vcc = Path(args.vitasdk) / "bin/arm-vita-eabi-gcc"
    pcc = Path(args.pspdev) / "bin/psp-gcc"
    builds = {}
    for noms in (False, True):
        extra = [NO_MS] if noms else []
        sfx = f" {NO_MS}" if noms else ""
        vita = ["-mcpu=cortex-a9", "-mthumb", "-mfpu=neon", "-mfloat-abi=hard"]
        builds[f"Vita -O2{sfx}"] = (vcc, [*vita, "-O2", *extra])
        builds[f"Vita -Os{sfx}"] = (vcc, [*vita, "-Os", *extra])
        builds[f"PSP -O2 -G0{sfx}"] = (pcc, ["-O2", "-G0", *extra])
        builds[f"PSP -Os -G0{sfx}"] = (pcc, ["-Os", "-G0", *extra])
    doc = {"kind": "textspot6-stack", "computed": datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"), "how": ("eval/textspot3_a9r.py stack's: gcc -fcallgraph-info=su over vocr_textspot6.c, "
                                      "vocr_lines.c and vocr_lines_rows.c; every frame static; the heaviest path "
                                      "(eval/textspot3_a9r.peaks, psp/stack.py's parser); calls outside them listed, "
                                      "their frames not counted"),
        "sources_sha256": {f"runtime/{s}": sha256(ROOT / "runtime" / s) for s in SOURCES},
        "compilers": {"vita": A9.tool_version(str(vcc)), "psp": A9.tool_version(str(pcc))}, "builds": {}}
    for label, (cc, flags) in builds.items():
        if not Path(cc).exists():
            continue
        cis = []
        for src in SOURCES:
            d = work / label.replace(" ", "_")
            d.mkdir(parents=True, exist_ok=True)
            o = d / (Path(src).stem + ".o")
            subprocess.run([str(cc), "-std=c99", *flags, "-fcallgraph-info=su", f"-I{ROOT / 'runtime'}", "-c",
                            str(ROOT / "runtime" / src), "-o", str(o)], check=True, capture_output=True, cwd=d)
            ci = d / (Path(src).stem + ".ci")
            if not ci.exists():
                ci = next(d.glob(Path(src).stem + "*.ci"))
            cis.append(ci.read_text(encoding="utf-8"))
        p, ext = A9R.peaks(cis, STACK_ROOTS)
        nd, _ = A9R.peaks(cis, STACK_ROOTS[:1], exclude=A9R.DARK)
        doc["builds"][label] = {"flags": flags, "peaks": p,
                                "no_dark": {f"{k} without the dark pass": v for k, v in nd.items()}, "externs": ext}
        print(label, {k: v["bytes"] for k, v in p.items()}, {k: v["bytes"] for k, v in nd.items()})
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print("wrote", args.out)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    fn = {"measure": cmd_measure, "report": cmd_report, "codesize": cmd_codesize, "stack": cmd_stack}.get(cmd)
    if fn is None:
        raise SystemExit(f"unknown command {cmd}: measure, report, codesize or stack")
    fn(rest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
