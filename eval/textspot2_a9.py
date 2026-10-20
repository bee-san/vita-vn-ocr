#!/usr/bin/env python3
"""Cortex-A9 estimates and exact arenas of textspot2's spotter shapes (issue #106, docs/textspot2.md "Estimates"):
eval/textspot_a9.py's measure and report, unchanged, on the shapes of eval/textspot2_protocol.json. Estimates only.

  textspot2_a9.py measure --out DIR [--runtime v1|v2] [--shapes S,...] [textspot_a9.py measure's other options]
  textspot2_a9.py report DIR [--json OUT] [--also DIR2 --also-label TEXT ...] [--textspot JSON] [--code-size JSON]
  textspot2_a9.py codesize --out JSON [--runtime v1|v2] [--vitasdk DIR] [--pspdev DIR] [--work DIR]

measure and report: train/textspot2_shapes.register() adds the protocol's "kroa" shapes to
train/textdet_model.SHAPES, then eval/textspot_a9.py measures and reports exactly as it did for textspot (#104): the
Vita compiler's code of the spotter under the plugin qemu-arm with the Vita's caches, on its pinned procedural frame
of each size, priced by vita/a9_cycles.py, est. ms at 333 MHz = high bound x 1.0007 / 333 MHz and at 444 MHz that x
333/444, random blobs (train/textdet_export.random_int_layers seed 1, train/textspot_twin.pack), the arenas the
runtime's own functions give under qemu-arm. The default is every "kroa" shape of the protocol at h2 (half_above
600), 960x544 and 480x272, grey; --nodark gives the report-only VOCR_TEXTSPOT_NO_DARK batch.

--runtime v2 measures runtime/vocr_textspot2.c, the spotter of VTSP version 1 and 2 blobs, instead of the frozen
runtime/vocr_textspot.c: textspot_a9.build's commands with runtime/vocr_textspot2.c in place of vocr_textspot.c and
the driver runtime/vocr_textspot2_test.c in place of vocr_textspot_test.c (the same record and output format), a d2s
shape ("kroa_x2", docs/textspot2.md "d2s") as a random version-2 blob (train/textspot2_export.random_blob, seed 1),
every other shape as textspot_a9.blob_for's version-1 blob (the same bytes as for vocr_textspot.c); everything else
is textspot_a9.py's. Its MACs per frame come from train/textspot2_export.cost for the d2s shapes.

report adds a "textspot2" block to textspot_a9.py's JSON: the protocol's sha256, the code's hashes, the shapes, and
with --textspot the comparison of every row of a shape textspot measured with its row there (identical, or each
field that differs). codesize: the .text of the spotter's own sections that a --gc-sections link of a caller of
load, both arena functions, find and find_rows keeps (VitaSDK gcc, Thumb-2, -O2 and -Os, NEON and -DVOCR_NO_NEON;
vita/textgate_size.py's flags and map parser), and psp-gcc's object .text (mips32, -O2 -G0 and -Os -G0).
"""
import argparse
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
for _p in (ROOT / "eval", ROOT / "train", ROOT / "synth", ROOT / "vita"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot_a9 as A9  # noqa: E402

import textspot2_shapes as S2  # noqa: E402

PROTOCOL = ROOT / "eval" / "textspot2_protocol.json"
TEXTSPOT_A9 = ROOT / "eval" / "textspot_a9_20261019.json"
HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
V1_SOURCES = A9.SOURCES
V2_SOURCES = ("vocr_textspot2.c", "vocr_lines.c", "vocr_lines_rows.c")
DRIVER = {"v1": "vocr_textspot_test.c", "v2": "vocr_textspot2_test.c"}
HEADER = {"v1": "vocr_textspot.h", "v2": "vocr_textspot2.h"}
# the fields of a table row that must agree for a row to reproduce another measurement of the same code
SAME = ("blob_sha256", "blob_bytes", "frame_sha256", "batch_sha256", "rc", "boxes", "blocks", "pick", "truncated",
        "frame_arena_bytes", "arena_bytes", "macs", "net_input", "insns_per_frame", "cycles_per_frame",
        "ms_per_frame", "l1d_misses_per_frame", "l2_misses_per_frame", "negative_count_addresses")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_shapes(kind="kroa"):
    """The protocol's shapes of one head kind ("kroa": the VTDT ones, "kroa_x2": d2s), in the protocol's order."""
    return [k for k, v in S2.SHAPES.items() if v["head"] == kind]


# ----------------------------------------------------------------------------- the v2 runtime

def build_v2(out, vitasdk, arm_cc, cflags=("-O2",)):
    """textspot_a9.build's commands with runtime/vocr_textspot2.c and its driver runtime/vocr_textspot2_test.c."""
    vcc = Path(vitasdk) / "bin/arm-vita-eabi-gcc"
    objs = []
    for src in V2_SOURCES:
        o = out / src.replace(".c", "_vita.o")
        flags = list(cflags) if src == V2_SOURCES[0] else ["-O2"]
        subprocess.run([str(vcc), "-std=c99", "-Wall", "-Wextra", "-Werror", *A9.A.ARCH, *flags, "-c", "-o", str(o),
                        str(ROOT / "runtime" / src)], check=True)
        objs.append(o)
    drv = out / "textspot_driver_vitacc"
    subprocess.run([arm_cc, "-std=c99", "-Wall", "-Wextra", "-Werror", *A9.A.ARCH, "-O2", "-static",
                    "-Wl,--no-enum-size-warning", "-o", str(drv), *map(str, objs),
                    str(ROOT / "runtime" / DRIVER["v2"])], check=True, capture_output=True)
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
    funcs = set()
    nm = Path(vitasdk) / "bin/arm-vita-eabi-nm"
    for o in objs:
        txt = subprocess.run([str(nm), "--defined-only", str(o)], check=True, capture_output=True, text=True).stdout
        funcs |= {ln.split()[-1] for ln in txt.splitlines() if ln.split()[1:2] and ln.split()[1] in "tT"}
    (out / "functions.txt").write_text("\n".join(sorted(funcs)) + "\n")
    size = subprocess.run([str(Path(vitasdk) / "bin/arm-vita-eabi-size"), str(objs[0])], check=True,
                          capture_output=True, text=True).stdout.split("\n")[1].split()[0]
    return drv, rows, funcs, objs, int(size)


_BLOB_FOR_V1 = A9.blob_for


def blob_for_v2(shape, scale, seed):
    """A d2s shape: a random VTSP version-2 blob; any other: textspot_a9.blob_for's version-1 blob."""
    if shape in S2.D2S:
        import textspot2_export as X
        return X.random_blob(shape, seed, half_above=A9.SCALES[scale])
    return _BLOB_FOR_V1(shape, scale, seed)


@contextlib.contextmanager
def runtime(kind):
    """eval/textspot_a9.py's module state for one runtime: v1 as it is; v2 with the new sources, driver, blobs and
    the d2s shapes' MACs (train/textdet_model.SHAPES / cost see them only inside this block)."""
    if kind == "v1":
        yield
        return
    import textspot2_export as X

    import textdet_model as TM
    saved = (A9.SOURCES, A9.build, A9.blob_for, TM.cost)
    added = [k for k in S2.D2S if k not in TM.SHAPES]
    orig_cost = TM.cost

    def cost(shape, width, height):
        return X.cost(shape, width, height) if shape.get("head") == "kroa_x2" else orig_cost(shape, width, height)

    try:
        A9.SOURCES, A9.build, A9.blob_for, TM.cost = V2_SOURCES, build_v2, blob_for_v2, cost
        for k in added:
            TM.SHAPES[k] = X.shape(k)
        yield
    finally:
        A9.SOURCES, A9.build, A9.blob_for, TM.cost = saved
        for k in added:
            TM.SHAPES.pop(k, None)


def code_hashes(kind):
    srcs = V1_SOURCES if kind == "v1" else V2_SOURCES
    out = {f"runtime/{s}": sha256(ROOT / "runtime" / s) for s in (*srcs, DRIVER[kind], HEADER[kind])}
    for f in ("eval/textspot2_a9.py", "eval/textspot_a9.py", "train/textspot2_shapes.py", "train/textdet_model.py",
              "train/textdet_export.py", "synth/textspot_twin.py", "synth/textdet_twin.py", "vita/a9_cycles.py"):
        out[f] = sha256(ROOT / f)
    if kind == "v2":
        for f in ("synth/textspot2_twin.py", "train/textspot2_export.py"):
            out[f] = sha256(ROOT / f)
    return out


def portable(argv):
    """argv with the fleet's shared directory written as $HERD and this checkout's paths relative (no local paths in
    a committed record)."""
    if argv is None:
        return None
    out = []
    for a in argv:
        a = str(a).replace(str(HERD), "$HERD").replace(str(ROOT) + "/", "")
        out.append(a)
    return out


def cmd_measure(argv):
    ap = argparse.ArgumentParser(prog="textspot2_a9.py measure", add_help=False)
    ap.add_argument("--runtime", choices=("v1", "v2"), default="v1")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--scales", default="h2")
    ap.add_argument("--out", required=True)
    args, rest = ap.parse_known_args(argv)
    S2.register()
    shapes = args.shapes or ",".join(protocol_shapes())
    unknown = [s for s in shapes.split(",") if s and s not in S2.SHAPES and s not in A9.RUN_SHAPES]
    if unknown:
        raise SystemExit(f"unknown shapes {unknown}")
    d2s = [s for s in shapes.split(",") if s in S2.D2S]
    if d2s and args.runtime != "v2":
        raise SystemExit(f"{d2s}: a d2s shape needs --runtime v2")
    with runtime(args.runtime):
        A9.main(["measure", "--out", args.out, "--shapes", shapes, "--scales", args.scales, *rest])
    meta_path = Path(args.out) / "meta.json"
    meta = json.loads(meta_path.read_text())
    # textspot_a9.cmd_measure records its own driver and header; record the files this runtime was built from
    meta["sources_sha256"] = {k: v for k, v in code_hashes(args.runtime).items() if k.startswith("runtime/")}
    meta["textspot2"] = {"runtime": args.runtime, "argv": portable(["measure", *argv])}
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")


def compare_rows(ours, theirs, label):
    """[{shape, scale, size, format, flags, identical, diffs}] for every row of ours that theirs has."""
    key = lambda t: (t["shape"], t["scale"], tuple(t["size"]), t["format"], t["flags"])  # noqa: E731
    by = {key(t): t for t in theirs}
    out = []
    for t in ours:
        o = by.get(key(t))
        if o is None:
            continue
        diffs = {f: [t.get(f), o.get(f)] for f in SAME if f in t and f in o and t.get(f) != o.get(f)}
        out.append({"table": label, "shape": t["shape"], "scale": t["scale"], "size": t["size"],
                    "format": t["format"], "flags": t["flags"], "identical": not diffs, "diffs": diffs})
    return out


def cmd_report(argv):
    ap = argparse.ArgumentParser(prog="textspot2_a9.py report")
    ap.add_argument("dir")
    ap.add_argument("--json")
    ap.add_argument("--also", action="append", default=[])
    ap.add_argument("--also-label", action="append", default=[])
    ap.add_argument("--textspot", default="", help="textspot's A9 table to compare the rows of its shapes with")
    ap.add_argument("--code-size", default="", help="a codesize JSON to include")
    ap.add_argument("--title", default="")
    args = ap.parse_args(argv)
    S2.register()
    meta = json.loads((Path(args.dir) / "meta.json").read_text())
    kind = meta.get("textspot2", {}).get("runtime", "v1")
    for d in args.also:
        m = json.loads((Path(d) / "meta.json").read_text())
        if m.get("textspot2", {}).get("runtime", "v1") != kind:
            raise SystemExit(f"--also {d} measured another runtime")
    with tempfile.TemporaryDirectory(prefix="textspot2-a9-") as td:
        tmp = Path(td) / "a9.json"
        rest = [x for d, lab in zip(args.also, args.also_label + [""] * len(args.also))
                for x in ("--also", d, "--also-label", lab)]
        with runtime(kind):
            A9.main(["report", args.dir, "--json", str(tmp), *rest])
        res = json.loads(tmp.read_text())
    proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    measured = sorted({t["shape"] for t in res["table"]})
    shapes = {}
    for name in measured:
        s = S2.SHAPES.get(name)
        if s is None:
            continue
        rec = {k: s[k] for k in S2.FIELDS}
        if s["head"] == "kroa":
            rec.update(S2.costs(name))
        else:
            import textspot2_export as X
            rec.update(X.costs(name))
        shapes[name] = rec
    block = {"protocol": {"path": "eval/textspot2_protocol.json", "sha256": sha256(PROTOCOL),
                          "estimates": proto["estimates"]["binding"]},
             "title": args.title, "runtime": kind,
             "runtime_files": {"v1": "runtime/vocr_textspot.c (frozen; driver runtime/vocr_textspot_test.c)",
                               "v2": "runtime/vocr_textspot2.c (VTSP versions 1 and 2; driver "
                                     "runtime/vocr_textspot2_test.c)"}[kind],
             "code_sha256": code_hashes(kind), "shapes": shapes,
             "measured": {"main": portable(meta.get("textspot2", {}).get("argv")),
                          "also": [portable(json.loads((Path(d) / "meta.json").read_text()).get("textspot2", {})
                                            .get("argv")) for d in args.also]}}
    if args.textspot:
        ref = json.loads(Path(args.textspot).read_text(encoding="utf-8"))
        rows = compare_rows(res["table"], ref["table"], "main")
        for a in res["also"]:
            for b in ref.get("also", []):
                if b["flags"] == a["flags"] and {t["flags"] for t in b["table"]} == {t["flags"] for t in a["table"]}:
                    rows += compare_rows(a["table"], b["table"], a["label"] or "also")
                    break
        block["reproduces"] = {"against": portable([args.textspot])[0], "sha256": sha256(args.textspot),
                               "rows": rows, "all_identical": bool(rows) and all(r["identical"] for r in rows)}
    if args.code_size:
        block["code_size"] = json.loads(Path(args.code_size).read_text(encoding="utf-8"))
    res["textspot2"] = block
    print_summary(res)
    if args.json:
        Path(args.json).write_text(A9.dumps(res) + "\n")
        print("wrote", args.json)


def print_summary(res):
    print("\n| shape | VTSP B | MACs / 960x544 | est. ms 960x544 @333 / @444 (rule) | no dark @333 | "
          "480x272 @333 / @444 | arena up to 960 (rule) | arena 960 frame | arena 480 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in res["summary"]:
        nd = f"{r['ms_333_960_nodark']:.1f}" if r.get("ms_333_960_nodark") is not None else "-"
        print(f"| {r['shape']} | {r['vtsp_bytes']:,} | {r['macs_960']:,} | {r['ms_333_960']:.1f} / "
              f"{r['ms_444_960']:.1f} | {nd} | {r.get('ms_333_480', float('nan')):.1f} / "
              f"{r.get('ms_444_480', float('nan')):.1f} | {r['arena_up_to_960']:,} | {r['arena_frame_960']:,} | "
              f"{r.get('arena_up_to_480', 0):,} |")
    rep = res.get("textspot2", {}).get("reproduces")
    if rep:
        same = sum(r["identical"] for r in rep["rows"])
        print(f"\nrows of textspot's shapes: {same} of {len(rep['rows'])} identical to {rep['against']}")
        for r in rep["rows"]:
            if not r["identical"]:
                print(f"  {r['table']} {r['shape']} {r['size']} {r['flags']}: {r['diffs']}")


# ----------------------------------------------------------------------------- code size

CALLER = {"v1": ("vocr_textspot.h", """
static vocr_textspot m;
static VocrTextspotSet set;
int main(void)
{
    VocrLinesImage im;
    VocrLinesRows src = {960, 544, 1, row_of, NULL};
    size_t need;
    int rc;
    if (vocr_textspot_load(&m, (const void *)blobp, blobn)) return 1;
    need = vocr_textspot_arena_bytes(&m, 960, 544) + vocr_textspot_frame_arena_bytes(&m, 960, 544);
    im.pixels = row; im.width = 960; im.height = 1; im.stride = 960; im.bpp = 1;
    rc = vocr_textspot_find(&m, &im, 0, arena, need, &set);
    return rc + vocr_textspot_find_rows(&m, &src, 0, arena, need, &set);
}
"""), "v2": ("vocr_textspot2.h", """
static vocr_textspot2 m;
static VocrTextspotSet set;
int main(void)
{
    VocrLinesImage im;
    VocrLinesRows src = {960, 544, 1, row_of, NULL};
    size_t need;
    int rc;
    if (vocr_textspot2_load(&m, (const void *)blobp, blobn)) return 1;
    need = vocr_textspot2_arena_bytes(&m, 960, 544) + vocr_textspot2_frame_arena_bytes(&m, 960, 544);
    im.pixels = row; im.width = 960; im.height = 1; im.stride = 960; im.bpp = 1;
    rc = vocr_textspot2_find(&m, &im, 0, arena, need, &set);
    return rc + vocr_textspot2_find_rows(&m, &src, 0, arena, need, &set);
}
""")}
PRELUDE = """#include <stddef.h>
volatile const unsigned char *blobp;
volatile size_t blobn;
static unsigned char arena[1u << 17], row[960];
static const unsigned char *row_of(void *user, unsigned y) { (void)user; (void)y; return row; }
"""


def link_size(cc, flags, kind, work, tag):
    """(.text, .rodata, {function: bytes}, {other object: bytes}, [libgcc members]) kept from the spotter's object
    by a --gc-sections link of CALLER[kind]."""
    import textgate_size as G
    header, body = CALLER[kind]
    spot = V1_SOURCES[0] if kind == "v1" else V2_SOURCES[0]
    d = Path(work) / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "main.c").write_text(f'#include "{header}"\n' + PRELUDE + body)
    objs = []
    for src in ("main.c", spot, "vocr_lines.c", "vocr_lines_rows.c"):
        o = d / (Path(src).stem + ".o")
        G.run([cc, *flags, f"-I{ROOT / 'runtime'}", "-c", d / "main.c" if src == "main.c" else ROOT / "runtime" / src,
               "-o", o])
        objs.append(o)
    G.run([cc, *flags, *objs, "-Wl,--gc-sections", f"-Wl,-Map={d / 'link.map'}", "-o", d / "prog.elf"])
    kept = G.parse_map((d / "link.map").read_text(errors="replace"))
    text = rodata = 0
    funcs, other, libgcc = {}, {}, set()
    stem = Path(spot).stem + ".o"
    for (obj, sec), size in sorted(kept.items()):
        base = Path(obj.split("(")[0]).name
        if base == stem:
            if sec.startswith(".text"):
                text += size
                funcs[sec[len(".text."):] or ".text"] = size
            elif sec.startswith(".rodata"):
                rodata += size
        elif base in ("vocr_lines.o", "vocr_lines_rows.o") and sec.startswith((".text", ".rodata")):
            other[base] = other.get(base, 0) + size
        elif "libgcc" in obj and sec.startswith(".text"):
            libgcc.add(obj[obj.find("(") + 1:-1] if "(" in obj else obj)
    return {"text": text, "rodata": rodata, "functions": funcs, "other_objects": other, "libgcc": sorted(libgcc)}


def object_text(cc, size_tool, flags, src, work, tag):
    d = Path(work) / tag
    d.mkdir(parents=True, exist_ok=True)
    o = d / (Path(src).stem + ".o")
    subprocess.run([str(cc), *flags, f"-I{ROOT / 'runtime'}", "-c", str(ROOT / "runtime" / src), "-o", str(o)],
                   check=True, capture_output=True)
    out = subprocess.run([str(size_tool), str(o)], check=True, capture_output=True, text=True).stdout
    return int(out.split("\n")[1].split()[0])


def cmd_codesize(argv):
    import textgate_size as G
    ap = argparse.ArgumentParser(prog="textspot2_a9.py codesize")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runtime", choices=("v1", "v2"), default="v1")
    ap.add_argument("--vitasdk", default=str(HERD / "shared/artifacts/vita/vitasdk"))
    ap.add_argument("--pspdev", default=str(HERD / "shared/artifacts/psp/tools/pspdev"))
    ap.add_argument("--work", default="")
    args = ap.parse_args(argv)
    vcc = Path(args.vitasdk) / "bin/arm-vita-eabi-gcc"
    pcc = Path(args.pspdev) / "bin/psp-gcc"
    spot = V1_SOURCES[0] if args.runtime == "v1" else V2_SOURCES[0]
    work = args.work or tempfile.mkdtemp(prefix="textspot2-size-")
    doc = {"kind": "textspot2-code-size", "runtime": args.runtime, "source": f"runtime/{spot}",
           "computed": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "how": ("the .text of the spotter's own sections kept by a --gc-sections link (-Wl,-Map) of a caller of "
                   "load, both arena functions, find and find_rows; vita/textgate_size.py's flags (Rubi's runtime "
                   "flags) and map parser; vocr_lines*.c and libgcc listed apart, not counted"),
           "sources_sha256": code_hashes(args.runtime), "vita": {"compiler": G.tool_version(vcc), "builds": {}}}
    for neon in (True, False):
        for level in ("-O2", "-Os"):
            flags = G.RUBI_FLAGS + G.VITA_ARCH + [level] + ([] if neon else ["-DVOCR_NO_NEON"])
            tag = f"vita{level}{'' if neon else '_scalar'}"
            r = link_size(vcc, flags, args.runtime, work, tag)
            r["object_text"] = object_text(vcc, Path(args.vitasdk) / "bin/arm-vita-eabi-size",
                                           ["-std=c99", *A9.A.ARCH, level] + ([] if neon else ["-DVOCR_NO_NEON"]),
                                           spot, work, tag + "_obj")
            r["flags"] = flags
            doc["vita"]["builds"][f"{level}{'' if neon else ' -DVOCR_NO_NEON'}"] = r
    if pcc.exists():
        doc["psp"] = {"compiler": G.tool_version(pcc), "object_text": {}}
        for level in ("-O2", "-Os"):
            doc["psp"]["object_text"][f"{level} -G0"] = object_text(
                pcc, Path(args.pspdev) / "bin/psp-size", ["-std=c99", level, "-G0"], spot, work, f"psp{level}")
    else:
        doc["psp"] = {"error": f"no psp-gcc at {pcc}"}
    print(f"{spot}: VitaSDK --gc-sections .text (object .text)")
    for k, r in doc["vita"]["builds"].items():
        print(f"  {k:22s} {r['text']:7,d} B (+ rodata {r['rodata']} B; object {r['object_text']:,} B; "
              f"vocr_lines*: {sum(r['other_objects'].values()):,} B)")
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
