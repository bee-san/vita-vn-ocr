#!/usr/bin/env python3
"""Code size of the text-presence gate (runtime/vocr_textgate.c) on the PS Vita, and on the PSP: the bytes the
protocol's size order reads (eval/textgate_protocol.json "size_order", eval/textgate.py select).

  vita/textgate_size.py [--vitasdk DIR] [--pspdev DIR] [--out eval/textgate_sizes_YYYYMMDD.json] [--work DIR]

Each option is a minimal program that calls only the entry points of one way of using the gate. It is compiled
with the VitaSDK's arm-vita-eabi-gcc (15.2) and the runtime flags Rubi builds vita-vn-ocr's runtime with
(Rubi_for_vita cmake/VocrOCR.cmake: -std=c99 -ffp-contract=off -Wall -Wextra -ffunction-sections -fdata-sections
-mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard; Thumb-2, the VitaSDK default) at -Os and at -O2, linked with
-Wl,--gc-sections, and measured from the link map: the .text and .rodata bytes kept from the gate's object.
vocr_lines.c and vocr_lines_rows.c (the finder, which Rubi already has) and libgcc's helpers are listed apart and
not counted.

  region          vocr_textgate_load, _arena_bytes, _begin, _push, _score_region: what Rubi would call
  boxes           vocr_textgate_load, _arena_bytes, _begin, _push, _score in boxes mode (the finder's boxes)
  finder_region   vocr_textgate_load, vocr_lines_gated_arena_bytes, vocr_find_lines_gated and _rows, region mode
  finder_boxes    the same in boxes mode
  textdet         #97's path for comparison: vocr_textdet_load, vocr_textdet_arena_bytes, vocr_textdet_find_rows
                  (vocr_textdet.c; Rubi #209 measured it at 7,400 B of -Os text)

code_bytes.region / .boxes (Vita, -Os) are what the selection reads; "blank" is 0 (Rubi's vjo_vocr_ink exists).
With --pspdev (or $HERD/shared/artifacts/psp/tools/pspdev present) the same at psp-gcc -Os -G0 (mips32, Allegrex).
Prints a table and writes the JSON.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RT = ROOT / "runtime"
HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
VITASDK = HERD / "shared/artifacts/vita/vitasdk"
PSPDEV = HERD / "shared/artifacts/psp/tools/pspdev"
RUBI_FLAGS = ["-std=c99", "-ffp-contract=off", "-Wall", "-Wextra", "-ffunction-sections", "-fdata-sections"]
VITA_ARCH = ["-mcpu=cortex-a9", "-mfpu=neon", "-mfloat-abi=hard"]
PSP_ARCH = ["-G0"]
LEVELS = ("-Os", "-O2")

PRELUDE = """#include <stddef.h>
volatile const unsigned char *blobp;
volatile size_t blobn;
static unsigned char arena[1u << 17], row[960];
static const unsigned char *row_of(void *user, unsigned y) { (void)user; (void)y; return row; }
"""

MAINS = {
    "region": ("vocr_textgate.h", ["vocr_textgate.c"], """
int main(void)
{
    static vocr_textgate_model m;
    static vocr_textgate_state st;
    int score = 0;
    unsigned y;
    size_t need;
    if (vocr_textgate_load(&m, (const void *)blobp, blobn)) return 1;
    need = vocr_textgate_arena_bytes(&m, VOCR_TEXTGATE_REGION, 960, 544);
    if (vocr_textgate_begin(&st, &m, VOCR_TEXTGATE_REGION, 2, 960, 544, arena, need)) return 2;
    for (y = 0; y < 544; y++) vocr_textgate_push(&st, row);
    vocr_textgate_score_region(&st, &score);
    (void)row_of;
    return score;
}
"""),
    "boxes": ("vocr_textgate.h", ["vocr_textgate.c"], """
int main(void)
{
    static vocr_textgate_model m;
    static vocr_textgate_state st;
    static VocrLineSet boxes;
    int score = 0;
    unsigned y;
    size_t need;
    if (vocr_textgate_load(&m, (const void *)blobp, blobn)) return 1;
    need = vocr_textgate_arena_bytes(&m, VOCR_TEXTGATE_BOXES, 960, 544);
    if (vocr_textgate_begin(&st, &m, VOCR_TEXTGATE_BOXES, 2, 960, 544, arena, need)) return 2;
    for (y = 0; y < 544; y++) vocr_textgate_push(&st, row);
    vocr_textgate_score(&st, &boxes, &score);
    (void)row_of;
    return score;
}
"""),
    "finder_region": ("vocr_textgate.h", ["vocr_textgate.c", "vocr_lines.c", "vocr_lines_rows.c"], """
int main(void)
{
    static vocr_textgate_model m;
    static VocrLineSet out;
    static VocrLinesImage im;
    VocrLinesRows src = {960, 544, 1, row_of, NULL};
    size_t need;
    if (vocr_textgate_load(&m, (const void *)blobp, blobn)) return 1;
    need = vocr_lines_gated_arena_bytes(&m, VOCR_TEXTGATE_REGION, 960, 544);
    im.pixels = arena; im.width = 960; im.height = 544; im.stride = 960; im.bpp = 1;
    if (vocr_find_lines_gated(&im, VOCR_LINES_GRAD, &m, VOCR_TEXTGATE_REGION, 2, 0, arena, need, &out)) return 2;
    return vocr_find_lines_gated_rows(&src, VOCR_LINES_GRAD, &m, VOCR_TEXTGATE_REGION, 2, 0, arena, need, &out);
}
"""),
    "finder_boxes": ("vocr_textgate.h", ["vocr_textgate.c", "vocr_lines.c", "vocr_lines_rows.c"], """
int main(void)
{
    static vocr_textgate_model m;
    static VocrLineSet out;
    static VocrLinesImage im;
    VocrLinesRows src = {960, 544, 1, row_of, NULL};
    size_t need;
    if (vocr_textgate_load(&m, (const void *)blobp, blobn)) return 1;
    need = vocr_lines_gated_arena_bytes(&m, VOCR_TEXTGATE_BOXES, 960, 544);
    im.pixels = arena; im.width = 960; im.height = 544; im.stride = 960; im.bpp = 1;
    if (vocr_find_lines_gated(&im, VOCR_LINES_GRAD, &m, VOCR_TEXTGATE_BOXES, 2, 0, arena, need, &out)) return 2;
    return vocr_find_lines_gated_rows(&src, VOCR_LINES_GRAD, &m, VOCR_TEXTGATE_BOXES, 2, 0, arena, need, &out);
}
"""),
    "textdet": ("vocr_textdet.h", ["vocr_textdet.c", "vocr_lines.c", "vocr_lines_rows.c"], """
int main(void)
{
    static vocr_textdet m;
    static VocrTextdetSet out;
    VocrLinesRows src = {960, 544, 1, row_of, NULL};
    size_t need;
    if (vocr_textdet_load(&m, (const void *)blobp, blobn)) return 1;
    need = vocr_textdet_arena_bytes(&m, 960, 544);
    return vocr_textdet_find_rows(&m, &src, arena, need, &out);
}
"""),
}
COUNTED = {"region": "vocr_textgate", "boxes": "vocr_textgate", "finder_region": "vocr_textgate",
           "finder_boxes": "vocr_textgate", "textdet": "vocr_textdet"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(cmd, **kw):
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, stdin=subprocess.DEVNULL, **kw)
    if p.returncode:
        raise RuntimeError(f"{' '.join(map(str, cmd))}: rc {p.returncode}\n{p.stderr[-2000:]}")
    return p.stdout


def parse_map(text):
    """{(object file, input section): bytes} of the sections the link kept (after 'Linker script and memory map')."""
    kept = {}
    lines = text.split("\n")
    try:
        start = next(i for i, ln in enumerate(lines) if ln.startswith("Linker script and memory map"))
    except StopIteration:
        raise RuntimeError("not a GNU ld map")
    pending = None
    for ln in lines[start:]:
        m = re.match(r"^ (\.[^\s]+)(?:\s+(0x[0-9a-f]+)\s+(0x[0-9a-f]+)\s+(\S.*))?$", ln)
        if m and m.group(2) is None:
            pending = m.group(1)
            continue
        if m:
            name, size, obj = m.group(1), int(m.group(3), 16), m.group(4).strip()
        elif pending:
            m2 = re.match(r"^\s+(0x[0-9a-f]+)\s+(0x[0-9a-f]+)\s+(\S.*)$", ln)
            pending, name = None, pending
            if not m2:
                continue
            size, obj = int(m2.group(2), 16), m2.group(3).strip()
        else:
            continue
        if size:
            kept[(obj, name)] = kept.get((obj, name), 0) + size
    return kept


def measure(cc, arch, level, option, work):
    header, sources, body = MAINS[option]
    d = Path(work) / f"{Path(cc).name}{level}_{option}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "main.c").write_text(f'#include "{header}"\n' + PRELUDE + body)
    flags = RUBI_FLAGS + arch + [level]
    objs = []
    for src in ["main.c"] + sources:
        o = d / (Path(src).stem + ".o")
        run([cc, *flags, f"-I{RT}", "-c", d / "main.c" if src == "main.c" else RT / src, "-o", o])
        objs.append(o)
    run([cc, *flags, *objs, "-Wl,--gc-sections", f"-Wl,-Map={d / 'link.map'}", "-o", d / "prog.elf"])
    kept = parse_map((d / "link.map").read_text(errors="replace"))
    counted = COUNTED[option]
    out = {"text": 0, "rodata": 0, "functions": {}, "other_objects": {}, "libgcc": {}}
    for (obj, sec), size in sorted(kept.items()):
        base = Path(obj.split("(")[0]).name
        if base == f"{counted}.o":
            if sec.startswith(".text"):
                out["text"] += size
                out["functions"][sec[len(".text."):] or ".text"] = size
            elif sec.startswith(".rodata"):
                out["rodata"] += size
        elif "libgcc" in obj and sec.startswith(".text"):
            member = obj[obj.find("(") + 1:-1] if "(" in obj else obj
            out["libgcc"][member] = out["libgcc"].get(member, 0) + size
        elif base in ("vocr_lines.o", "vocr_lines_rows.o", "vocr_textdet.o", "vocr_textgate.o") and \
                sec.startswith((".text", ".rodata")):
            out["other_objects"][base] = out["other_objects"].get(base, 0) + size
    return out


def tool_version(cc):
    try:
        return run([cc, "--version"]).splitlines()[0]
    except (OSError, RuntimeError, IndexError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vitasdk", default=str(VITASDK))
    ap.add_argument("--pspdev", default=str(PSPDEV))
    ap.add_argument("--options", default=",".join(MAINS))
    ap.add_argument("--out", default="")
    ap.add_argument("--work", default="")
    args = ap.parse_args(argv)
    options = [o for o in args.options.split(",") if o]
    vcc = Path(args.vitasdk) / "bin/arm-vita-eabi-gcc"
    pcc = Path(args.pspdev) / "bin/psp-gcc"
    work = args.work or tempfile.mkdtemp(prefix="textgate-size-")
    doc = {"kind": "textgate-sizes", "computed": datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"), "code_bytes": {}, "note": (
        "code_bytes: VitaSDK gcc -Os with Rubi's runtime flags, .text kept by --gc-sections from the gate's object "
        "for the option's entry points (vita/textgate_size.py); blank: 0, Rubi's vjo_vocr_ink exists already"),
        "vita": {"compiler": tool_version(vcc), "flags": RUBI_FLAGS + VITA_ARCH, "link": ["-Wl,--gc-sections"],
                 "levels": {}},
        "sources_sha256": {f: sha256(RT / f) for f in ("vocr_textgate.c", "vocr_textgate.h", "vocr_textdet.c",
                                                         "vocr_lines.c", "vocr_lines_rows.c")}}
    for level in LEVELS:
        doc["vita"]["levels"][level] = {o: measure(vcc, VITA_ARCH, level, o, work) for o in options}
    os_ = doc["vita"]["levels"]["-Os"]
    doc["code_bytes"] = {o: os_[o]["text"] for o in options if o != "textdet"}
    doc["code_bytes"]["blank"] = 0
    if pcc.exists():
        try:
            doc["psp"] = {"compiler": tool_version(pcc), "flags": RUBI_FLAGS + PSP_ARCH, "link": ["-Wl,--gc-sections"],
                          "levels": {"-Os": {o: measure(pcc, PSP_ARCH, "-Os", o, work) for o in options}}}
        except RuntimeError as e:
            doc["psp"] = {"error": str(e)[:500]}
    else:
        doc["psp"] = {"error": f"no psp-gcc at {pcc}"}
    print(f"{'option':14s} " + " ".join(f"{'vita ' + lv:>10s}" for lv in LEVELS) + f" {'rodata':>7s} {'psp -Os':>8s}"
          "  libgcc (vita -Os)")
    for o in options:
        row = [doc["vita"]["levels"][lv][o]["text"] for lv in LEVELS]
        psp = doc["psp"].get("levels", {}).get("-Os", {}).get(o, {}).get("text", "-")
        lib = ", ".join(sorted(os_[o]["libgcc"]))
        print(f"{o:14s} " + " ".join(f"{v:10,d}" for v in row) + f" {os_[o]['rodata']:7d} {psp!s:>8s}  {lib}")
    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
