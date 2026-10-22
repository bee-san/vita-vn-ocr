#!/usr/bin/env python3
"""Cortex-A9 estimates and exact arenas of textspot3's family M (VTSP version 3) on runtime/vocr_textspot3.c (issue
#108, docs/textspot3.md "Estimates"): eval/textspot_a9.py's measure and report on the new runtime, with the cost
split by part. Estimates only. (eval/textspot3_a9.py, pinned by amendment 1, measures family T on
runtime/vocr_textspot2.c; this file is its version-3 counterpart.)

  textspot3_a9m.py measure --out DIR [--shapes S,...] [--shapes-json FILE] [--nodark] [--sizes 960x544,480x272]
                           [--cflags "-O2"] [--seed 1] [--jobs 5] [--vitasdk DIR] [--arm-cc GCC] [--qemu DIR]
  textspot3_a9m.py report DIR [--also DIR2 --also-label TEXT ...] [--json OUT] [--md OUT] [--title TEXT]
  textspot3_a9m.py codesize --out JSON [--vitasdk DIR] [--pspdev DIR] [--work DIR]

measure: eval/textspot_a9.py's measure, unchanged in its method (the Vita compiler's objects of the runtime linked
into the driver by an ARM Linux toolchain, run under the plugin qemu-arm with the Vita's caches on textspot's pinned
procedural frame of each size, priced by vita/a9_cycles.py: est. ms at 333 MHz = high bound x 1.0007 / 333 MHz, at
444 MHz that x 333/444; the arenas the runtime's own functions give, under qemu-arm), with
runtime/vocr_textspot3.c and its driver runtime/vocr_textspot3_test.c in place of textspot's, and random version-3
blobs: train/textspot3_model.random_blob(shape, seed 1) (half_above 480, half_above_h 272), recorded under the scale
"m" (rows' half_above 480: a 960x544 frame is halved, 480x272 is read as it is, as version 3's rule does). The
shapes are train/textspot3_model.PROBES (not registered), the M shapes eval/textspot3_protocol.json registers
(train/textspot3_shapes.M) and, with --shapes-json, the M shape dicts of a JSON file ({name: shape}; each recorded in
meta.json, so report knows it); the MACs per frame are train/textspot3_model.cost at the network's input.

report: textspot_a9.py's report (summary, table, also) plus a "textspot3" block: the protocol's sha256 and budget,
the code's hashes, each shape's exact figures (train/textspot3_model.costs: VTSP bytes, int8 weights, parameters,
MACs in total and per part, receptive field), textspot3's budget check (VTSP <= 512,000 B, est. <= 300 ms per
960x544 frame at 333 MHz, arena_up_to_960 <= 262,144 B; for det), and the cost split of every row: the high bound's
instruction part (vita/a9_cycles.model_mode on the counts of one part's code, no cache term) for the stream (the
halving, the stride-8 layers, F8), the map ops (every 1x1 and depthwise op, the head's 1x1 included), the decode
(the head's rows into the decode, components, the cap, blocks, order, pick), the dark flags and the rest, and the
cache term of the whole run (L1 misses x 30 + L2 misses x (30 + 110 ns)); est. ms per part at 333 MHz on the same
scale (x 1.0007 / 333 MHz). --md writes the markdown table.

codesize: the .text of the spotter's own sections that a --gc-sections link of a caller of load, both arena
functions, find and find_rows keeps (eval/textspot2_a9.py codesize's method and flags) for vocr_textspot3.c:
VitaSDK gcc -O2 and -Os, NEON and -DVOCR_NO_NEON; and psp-gcc's object .text (mips32, -O2 -G0 and -Os -G0).
"""
import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "train", ROOT / "synth", ROOT / "vita"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot2_a9 as W2  # noqa: E402
import textspot_a9 as A9  # noqa: E402

import textspot3_model as M3  # noqa: E402

PROTOCOL = ROOT / "eval" / "textspot3_protocol.json"
BUDGET = {"vtsp_max": 512000, "ms_max_333_960": 300.0, "arena_max": 262144}
SCALE, HALF_ABOVE = "m", M3.HALF_ABOVE
SOURCES = ("vocr_textspot3.c", "vocr_lines.c", "vocr_lines_rows.c")
DRIVER, HEADER = "vocr_textspot3_test.c", "vocr_textspot3.h"
CODE_FILES = ("eval/textspot3_a9m.py", "eval/textspot_a9.py", "eval/textspot2_a9.py", "train/textspot3_model.py",
              "train/textspot3_shapes.py", "synth/textspot3_twin.py", "synth/textspot2_twin.py",
              "synth/textspot_twin.py", "synth/textdet_twin.py", "train/textdet_model.py", "train/textdet_export.py",
              "vita/a9_cycles.py")
# the functions of runtime/vocr_textspot3.c by part (the noinline stages; anything else is "other")
PARTS = {
    "stream": ("feed", "chain", "delivered", "dest_row", "to_f8", "pair_sums", "input_codes", "setup3", "produce",
               "drain", "k3_block_s1", "k3_block_s2", "k1_block", "grey8", "memset"),
    "map": ("run_pw", "run_dw", "dw_row", "pw_cells", "pw_tile", "pw_offsets", "pw_segs", "op_params", "rq4v",
            "rq4v.part.0", "copy_cells", "map_ref", "hsum4", "memcpy"),
    "decode": ("run_head", "sub_row3", "decode_row", "sweep", "finish", "keep", "group", "isort", "rows_of", "joins",
               "find_root", "root_of", "unite", "cap_less", "key_less", "field", "same_row"),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


EXTRA = {}   # more M shapes: --shapes-json, or the shapes a measured directory's meta.json records


def load_extra(path):
    """{name: M shape} of a JSON file (checked by train/textspot3_model.check_shape) into EXTRA."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    for name, s in doc.items():
        if isinstance(s, dict) and s.get("family") == M3.FAMILY:
            EXTRA[name] = M3.check_shape(dict(s, name=name))
    return EXTRA


def m_shapes():
    """Every M shape this tool knows: the probe grid, the protocol's registered M shapes and EXTRA."""
    import textspot3_shapes as S3
    out = dict(M3.PROBES)
    for name, s in S3.M.items():
        out[name] = M3.check_shape(s)
    for name, s in EXTRA.items():
        if name in out and out[name] != s:
            raise SystemExit(f"{name}: another shape under this name")
        out[name] = s
    return out


# ----------------------------------------------------------------------------- the v3 runtime

def build_v3(out, vitasdk, arm_cc, cflags=("-O2",)):
    """textspot_a9.build's commands with runtime/vocr_textspot3.c and runtime/vocr_textspot3_test.c; also records the
    functions of each object (functions_by_object.json) for the cost split."""
    vcc = Path(vitasdk) / "bin/arm-vita-eabi-gcc"
    objs = []
    for src in SOURCES:
        o = out / src.replace(".c", "_vita.o")
        flags = list(cflags) if src == SOURCES[0] else ["-O2"]
        subprocess.run([str(vcc), "-std=c99", "-Wall", "-Wextra", "-Werror", *A9.A.ARCH, *flags, "-c", "-o", str(o),
                        str(ROOT / "runtime" / src)], check=True)
        objs.append(o)
    drv = out / "textspot_driver_vitacc"
    subprocess.run([arm_cc, "-std=c99", "-Wall", "-Wextra", "-Werror", *A9.A.ARCH, "-O2", "-static",
                    "-Wl,--no-enum-size-warning", "-o", str(drv), *map(str, objs),
                    str(ROOT / "runtime" / DRIVER)], check=True, capture_output=True)
    objdump = arm_cc[:-3] + "objdump" if arm_cc.endswith("gcc") else "objdump"
    dis = subprocess.run([objdump, "-d", "--no-show-raw-insn", str(drv)], check=True, capture_output=True,
                         text=True).stdout
    (out / "driver.dis").write_text(dis)
    rows = A9.A.parse_dis(dis)
    with open(out / "classes.txt", "w") as f:
        for a, (_, mn, ops) in sorted(rows.items()):
            unit = A9.A.classify(mn, ops)[0]
            if unit != "I":
                f.write(f"{a:x} {unit}\n")
    funcs, by_obj = set(), {}
    nm = Path(vitasdk) / "bin/arm-vita-eabi-nm"
    for o in objs:
        txt = subprocess.run([str(nm), "--defined-only", str(o)], check=True, capture_output=True, text=True).stdout
        names = sorted({ln.split()[-1] for ln in txt.splitlines() if ln.split()[1:2] and ln.split()[1] in "tT"})
        by_obj[o.name] = names
        funcs |= set(names)
    (out / "functions.txt").write_text("\n".join(sorted(funcs)) + "\n")
    (out / "functions_by_object.json").write_text(json.dumps(by_obj, indent=1) + "\n")
    size = subprocess.run([str(Path(vitasdk) / "bin/arm-vita-eabi-size"), str(objs[0])], check=True,
                          capture_output=True, text=True).stdout.split("\n")[1].split()[0]
    return drv, rows, funcs, objs, int(size)


def blob_for_v3(shape, scale, seed):
    if scale != SCALE:
        raise SystemExit(f"{shape}: a version-3 shape is measured at the scale {SCALE!r}")
    return M3.random_blob(m_shapes()[shape], seed)


@contextlib.contextmanager
def runtime():
    """eval/textspot_a9's module state for runtime/vocr_textspot3.c: its sources, build, version-3 blobs and the M
    shapes' MACs (train/textdet_model.SHAPES / cost see them only inside this block); scale "m" = half_above 480."""
    import textdet_model as TM
    shapes = m_shapes()
    saved = (A9.SOURCES, A9.build, A9.blob_for, TM.cost)
    added = [k for k in shapes if k not in TM.SHAPES]
    had_scale = SCALE in A9.SCALES
    orig_cost = TM.cost

    def cost(shape, width, height):
        if shape.get("family") == M3.FAMILY:
            return M3.cost(shape, width, height)
        return orig_cost(shape, width, height)

    try:
        A9.SOURCES, A9.build, A9.blob_for, TM.cost = SOURCES, build_v3, blob_for_v3, cost
        A9.SCALES[SCALE] = HALF_ABOVE
        for k in added:
            TM.SHAPES[k] = shapes[k]
        yield shapes
    finally:
        A9.SOURCES, A9.build, A9.blob_for, TM.cost = saved
        if not had_scale:
            A9.SCALES.pop(SCALE, None)
        for k in added:
            TM.SHAPES.pop(k, None)


def code_hashes():
    out = {f"runtime/{s}": sha256(ROOT / "runtime" / s) for s in (*SOURCES, DRIVER, HEADER)}
    for f in CODE_FILES:
        out[f] = sha256(ROOT / f)
    return dict(sorted(out.items()))


def cmd_measure(argv):
    ap = argparse.ArgumentParser(prog="textspot3_a9m.py measure", add_help=False)
    ap.add_argument("--shapes", default="")
    ap.add_argument("--shapes-json", default="")
    ap.add_argument("--out", required=True)
    args, rest = ap.parse_known_args(argv)
    if args.shapes_json:
        load_extra(args.shapes_json)
    if "--scales" in rest:
        raise SystemExit("version-3 shapes are measured at the scale 'm' (half_above 480, half_above_h 272)")
    with runtime() as shapes:
        names = [s for s in args.shapes.split(",") if s] or list(shapes)
        unknown = [s for s in names if s not in shapes]
        if unknown:
            raise SystemExit(f"unknown M shapes {unknown}")
        A9.main(["measure", "--out", args.out, "--shapes", ",".join(names), "--scales", SCALE, *rest])
    meta_path = Path(args.out) / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["sources_sha256"] = {k: v for k, v in code_hashes().items() if k.startswith("runtime/")}
    meta["textspot3"] = {"runtime": "v3", "argv": W2.portable(["measure", *argv]),
                         "shapes": {n: m_shapes()[n] for n in names}}
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")


# ----------------------------------------------------------------------------- report

def part_of(fn, lines_fns):
    if fn in lines_fns:
        return "dark"
    for part, names in PARTS.items():
        if fn in names:
            return part
    return "other"


def split(d, run):
    """The cost split of one measured row (directory d): per part, instructions, NEON issue and the high bound of
    its instructions alone (no cache term) per frame; the whole run's cache term; est. ms at 333 MHz of each."""
    rows = A9.A.parse_dis((d / "driver.dis").read_text())
    funcs = set((d / "functions.txt").read_text().split())
    by_obj = json.loads((d / "functions_by_object.json").read_text())
    lines_fns = set(by_obj.get("vocr_lines_vita.o", [])) | set(by_obj.get("vocr_lines_rows_vita.o", []))
    w, h = run["size"]
    tag = f"{run['shape']}_{run['scale']}_{w}x{h}_{run['format']}"
    c_on, _ = A9.A.read_counts(d / f"counts_{tag}_batch.txt")
    c_off, _ = A9.A.read_counts(d / f"counts_{tag}_dry.txt")
    delta = {a: c_on.get(a, 0) - c_off.get(a, 0) for a in set(c_on) | set(c_off)}
    parts = {}
    for a, v in delta.items():
        if v <= 0:
            continue
        fn = rows.get(a, ("?", "?", ""))[0]
        if fn not in funcs and not A9.A.HELPERS.match(fn):
            continue
        parts.setdefault(part_of(fn, lines_fns), {})[a] = v
    out = {}
    for part in ("stream", "map", "decode", "dark", "other"):
        r = A9.A.model_mode(rows, parts.get(part, {}), {}, funcs, run["frames"], None)
        hi = r["cycles_per_frame"][333][1]
        out[part] = {"insns_per_frame": round(r["insns_per_frame"]), "neon_issue_per_frame": round(
            r["neon_issue_per_frame"]), "cycles_high_per_frame": round(hi),
            "ms_333": A9.CAL * hi / (333 * 1000)}
    con, coff = A9.A.read_cache(d / f"cache_{tag}_batch.txt"), A9.A.read_cache(d / f"cache_{tag}_dry.txt")
    if con and coff:
        l1 = max(0, con["l1d_misses"] - coff["l1d_misses"]) / run["frames"]
        l2 = max(0, con["l2_misses"] - coff["l2_misses"]) / run["frames"]
        mem = l1 * A9.A.L2_LAT[1] + l2 * (A9.A.L2_LAT[1] + A9.A.DRAM_NS * 333 / 1000)
        out["cache"] = {"l1d_misses_per_frame": round(l1), "l2_misses_per_frame": round(l2),
                        "cycles_high_per_frame": round(mem), "ms_333": A9.CAL * mem / (333 * 1000)}
    return out


def within(r):
    return r["vtsp_bytes"] <= BUDGET["vtsp_max"] and r["ms_333_960"] <= BUDGET["ms_max_333_960"] and \
        r["arena_up_to_960"] <= BUDGET["arena_max"]


def tables(res):
    return [("main", res["table"])] + [(a.get("label") or "also", a["table"]) for a in res.get("also", [])]


def summary_rows(res, shapes):
    cells = {}
    for _, tab in tables(res):
        for t in tab:
            if t["format"] == "grey":
                mode = "dark" if t["flags"] == "0" else "nodark"
                cells.setdefault(t["shape"], {})[(mode, tuple(t["size"]))] = t
    out = []
    for shape, c in sorted(cells.items(), key=lambda kv: (shapes[kv[0]]["vtsp_bytes"], kv[0])):
        s = shapes[shape]
        r = {"shape": shape, "point": s.get("point"), "vtsp_bytes": s["vtsp_bytes"],
             "weights_int8": s["weights_int8"], "params": s["params"], "receptive_field": s["receptive_field"],
             "macs_960": s["macs"]["960x544"], "macs_480": s["macs"]["480x272"],
             "macs_by_part_480x272": s["macs_by_part_480x272"], "weights_by_part": s["weights_by_part"]}
        for mode in ("dark", "nodark"):
            for size, tag in (((960, 544), "960"), ((480, 272), "480")):
                t = c.get((mode, size))
                suffix = "" if mode == "dark" else "_nodark"
                r[f"ms_333_{tag}{suffix}"] = t["ms_per_frame"]["333"] if t else None
                r[f"ms_444_{tag}{suffix}"] = t["ms_per_frame"]["444"] if t else None
                if t and mode == "dark":
                    r[f"arena_up_to_{tag}"] = t["arena_bytes"]
                    r[f"arena_frame_{tag}"] = t["frame_arena_bytes"]
                    r[f"boxes_{tag}"] = t["boxes"]
                    r[f"truncated_{tag}"] = t["truncated"]
                    if "split" in t:
                        r[f"split_{tag}"] = t["split"]
        if r["ms_333_960"] is not None and "arena_up_to_960" in r:
            r["within_budget"] = within(r)
        out.append(r)
    return out


def _f(v, fmt="{:.1f}"):
    return "-" if v is None else fmt.format(v)


def markdown(res):
    b = res["textspot3"]
    lines = [b["title"] or "", "",
             "Estimates (the A9 cycle model on the Vita compiler's code of runtime/vocr_textspot3.c under qemu-arm; "
             "nothing ran on a console). Arenas exact (the runtime's functions under qemu-arm). Random version-3 "
             "blobs (train/textspot3_model.random_blob seed 1, half_above 480 / 272); shapes not registered unless "
             "the protocol says so.", "",
             "| shape | point | VTSP B | int8 weights | MACs / 960x544 | RF px | est. ms 960x544 @333 / @444 (rule) | "
             "no dark @333 / @444 | 480x272 @333 / @444 | 480x272 no dark @333 | arena up to 960x544 (rule) | "
             "arena 480x272 | split @333 960x544: stream / map / decode / dark / other / cache | within budget |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in b["rows"]:
        sp = r.get("split_960")
        sps = " / ".join(f"{sp[k]['ms_333']:.1f}" for k in ("stream", "map", "decode", "dark", "other", "cache")
                         if k in sp) if sp else "-"
        lines.append(
            f"| {r['shape']} | {r.get('point') or ''} | {r['vtsp_bytes']:,} | {r['weights_int8']:,} | "
            f"{r['macs_960']:,} | {r['receptive_field']} | {_f(r['ms_333_960'])} / {_f(r['ms_444_960'])} | "
            f"{_f(r['ms_333_960_nodark'])} / {_f(r['ms_444_960_nodark'])} | {_f(r['ms_333_480'])} / "
            f"{_f(r['ms_444_480'])} | {_f(r['ms_333_480_nodark'])} | {r.get('arena_up_to_960', 0):,} | "
            f"{r.get('arena_up_to_480', 0):,} | {sps} | {'yes' if r.get('within_budget') else 'no'} |")
    return "\n".join(lines)


def cmd_report(argv):
    ap = argparse.ArgumentParser(prog="textspot3_a9m.py report")
    ap.add_argument("dir")
    ap.add_argument("--json")
    ap.add_argument("--md")
    ap.add_argument("--also", action="append", default=[])
    ap.add_argument("--also-label", action="append", default=[])
    ap.add_argument("--title", default="")
    ap.add_argument("--code-size", default="")
    args = ap.parse_args(argv)
    for d in [args.dir, *args.also]:          # the shapes each directory measured
        for name, sh in json.loads((Path(d) / "meta.json").read_text()).get("textspot3", {}).get("shapes",
                                                                                                    {}).items():
            if name not in M3.PROBES:
                EXTRA[name] = M3.check_shape(sh)
    with runtime() as shapes:
        with tempfile.TemporaryDirectory(prefix="textspot3-a9m-") as td:
            tmp = Path(td) / "a9.json"
            rest = [x for d, lab in zip(args.also, args.also_label + [""] * len(args.also))
                    for x in ("--also", d, "--also-label", lab)]
            A9.main(["report", args.dir, "--json", str(tmp), *rest])
            res = json.loads(tmp.read_text())
    dirs = [Path(args.dir)] + [Path(d) for d in args.also]
    for d, (_, tab) in zip(dirs, tables(res)):
        meta = json.loads((d / "meta.json").read_text())
        runs = {(r["shape"], tuple(r["size"]), r["format"]): r for r in meta["runs"]}
        for t in tab:
            t["split"] = split(d, runs[(t["shape"], tuple(t["size"]), t["format"])])
    figures = {n: M3.costs(shapes[n]) for n in sorted({t["shape"] for _, tab in tables(res) for t in tab})}
    proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    res["rule_reads"]["limits"] = dict(BUDGET)
    res["rule_reads"]["within_limits"] = (
        "textspot3's budget, for det: vtsp_bytes <= vtsp_max, ms_333_960 <= ms_max_333_960 and arena_up_to_960 <= "
        "arena_max; det+refine adds 1.4 ms per refined horizontal box and the refinement's 72,391-B arena")
    for r in res["summary"]:
        c = figures[r["shape"]]
        r.update(weights_int8=c["weights_int8"], vtsp_bytes=c["vtsp_bytes"], receptive_field=c["receptive_field"])
        r["within_limits"] = within(r)
    meta = json.loads((Path(args.dir) / "meta.json").read_text())
    block = {"protocol": {"path": "eval/textspot3_protocol.json", "sha256": sha256(PROTOCOL),
                          "budget": proto["budget"], "estimates": proto["estimates"]["binding"]},
             "title": args.title, "runtime": "v3",
             "runtime_files": "runtime/vocr_textspot3.c (VTSP versions 1, 2 and 3; driver "
                              "runtime/vocr_textspot3_test.c)",
             "code_sha256": code_hashes(), "shapes": {n: {"shape": shapes[n], **figures[n]} for n in figures},
             "split_parts": {k: list(v) for k, v in PARTS.items()},
             "measured": {"main": meta.get("textspot3", {}).get("argv"),
                          "also": [json.loads((Path(d) / "meta.json").read_text()).get("textspot3", {}).get("argv")
                                   for d in args.also]}}
    if args.code_size:
        block["code_size"] = json.loads(Path(args.code_size).read_text(encoding="utf-8"))
    block["rows"] = summary_rows(res, {n: dict(f, point=shapes[n].get("point")) for n, f in figures.items()})
    res["textspot3"] = block
    print(markdown(res))
    if args.json:
        Path(args.json).write_text(A9.dumps(res) + "\n")
        print("wrote", args.json)
    if args.md:
        Path(args.md).write_text(markdown(res) + "\n")
        print("wrote", args.md)


# ----------------------------------------------------------------------------- code size

CALLER = """
static vocr_textspot3 m;
static VocrTextspotSet set;
int main(void)
{
    VocrLinesImage im;
    VocrLinesRows src = {960, 544, 1, row_of, NULL};
    size_t need;
    int rc;
    if (vocr_textspot3_load(&m, (const void *)blobp, blobn)) return 1;
    need = vocr_textspot3_arena_bytes(&m, 960, 544) + vocr_textspot3_frame_arena_bytes(&m, 960, 544);
    im.pixels = row; im.width = 960; im.height = 1; im.stride = 960; im.bpp = 1;
    rc = vocr_textspot3_find(&m, &im, 0, arena, need, &set);
    return rc + vocr_textspot3_find_rows(&m, &src, 0, arena, need, &set);
}
"""


def cmd_codesize(argv):
    import textgate_size as G
    ap = argparse.ArgumentParser(prog="textspot3_a9m.py codesize")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vitasdk", default=str(W2.HERD / "shared/artifacts/vita/vitasdk"))
    ap.add_argument("--pspdev", default=str(W2.HERD / "shared/artifacts/psp/tools/pspdev"))
    ap.add_argument("--work", default="")
    args = ap.parse_args(argv)
    vcc = Path(args.vitasdk) / "bin/arm-vita-eabi-gcc"
    pcc = Path(args.pspdev) / "bin/psp-gcc"
    work = args.work or tempfile.mkdtemp(prefix="textspot3-size-")
    saved = dict(W2.CALLER)
    W2.CALLER["v3"] = (HEADER, CALLER)
    saved_src = W2.V2_SOURCES
    try:
        W2.V2_SOURCES = SOURCES     # link_size takes the spotter's source from V2_SOURCES for a non-v1 runtime
        import datetime
        doc = {"kind": "textspot3-code-size", "runtime": "v3", "source": "runtime/vocr_textspot3.c",
               "computed": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "how": ("eval/textspot2_a9.py codesize's: the .text of the spotter's own sections kept by a "
                       "--gc-sections link (-Wl,-Map) of a caller of load, both arena functions, find and find_rows; "
                       "vita/textgate_size.py's flags and map parser; vocr_lines*.c and libgcc listed apart"),
               "sources_sha256": code_hashes(), "vita": {"compiler": G.tool_version(vcc), "builds": {}}}
        for neon in (True, False):
            for level in ("-O2", "-Os"):
                flags = G.RUBI_FLAGS + G.VITA_ARCH + [level] + ([] if neon else ["-DVOCR_NO_NEON"])
                tag = f"vita{level}{'' if neon else '_scalar'}"
                r = W2.link_size(vcc, flags, "v3", work, tag)
                obj_flags = ["-std=c99", *A9.A.ARCH, level] + ([] if neon else ["-DVOCR_NO_NEON"])
                r["object_text"] = W2.object_text(vcc, Path(args.vitasdk) / "bin/arm-vita-eabi-size", obj_flags,
                                                  SOURCES[0], work, tag + "_obj")
                r["flags"] = flags
                doc["vita"]["builds"][f"{level}{'' if neon else ' -DVOCR_NO_NEON'}"] = r
        if pcc.exists():
            doc["psp"] = {"compiler": G.tool_version(pcc), "object_text": {}}
            for level in ("-O2", "-Os"):
                doc["psp"]["object_text"][f"{level} -G0"] = W2.object_text(
                    pcc, Path(args.pspdev) / "bin/psp-size", ["-std=c99", level, "-G0"], SOURCES[0], work,
                    f"psp{level}")
        else:
            doc["psp"] = {"error": f"no psp-gcc at {pcc}"}
    finally:
        W2.CALLER.clear()
        W2.CALLER.update(saved)
        W2.V2_SOURCES = saved_src
    print("runtime/vocr_textspot3.c: VitaSDK --gc-sections .text (object .text)")
    for k, r in doc["vita"]["builds"].items():
        print(f"  {k:22s} {r['text']:7,d} B (+ rodata {r['rodata']} B; object {r['object_text']:,} B)")
    for k, v in doc.get("psp", {}).get("object_text", {}).items():
        print(f"  psp-gcc {k:14s} {v:7,d} B (object .text)")
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print("wrote", args.out)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "measure":
        cmd_measure(rest)
    elif cmd == "report":
        cmd_report(rest)
    elif cmd == "codesize":
        cmd_codesize(rest)
    else:
        raise SystemExit(f"unknown command {cmd}: measure, report or codesize")
    return 0


if __name__ == "__main__":
    sys.exit(main())
