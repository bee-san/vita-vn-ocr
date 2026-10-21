#!/usr/bin/env python3
"""textspot3's A9 table of record (issue #108, docs/textspot3.md "Estimates"): the registered shapes of both families
measured on the final runtimes, with the code sizes and stack peaks, in one JSON. Estimates only.

  textspot3_a9r.py report MDIR [--also DIR --also-label TEXT ...] --t-json TJSON [--code-size CS] [--stack ST]
                          --json OUT [--md OUT] [--title TEXT]
  textspot3_a9r.py stack --out JSON [--vitasdk DIR] [--pspdev DIR] [--work DIR]

report: family M's rows by eval/textspot3_a9m.py's report (pinned by amendment 4, unchanged) on its measured
directories (the registered M shapes on runtime/vocr_textspot3.c, the default call and VOCR_TEXTSPOT_NO_DARK, 960x544
and 480x272), with its cost split's part table extended by the runtime's later function names (EXTRA_PARTS; GCC's
clones name.isra.0 / name.part.0 count as their function); then a "t" block: family T's registered shapes and
textspot2's two d2s shapes measured again by eval/textspot3_a9.py (pinned by amendment 1, unchanged) on
runtime/vocr_textspot2.c, its report TJSON with --reproduce of the pinned tables (eval/textspot3_a9_t_20261021.json,
eval/textspot2_a9_d2s_20261020.json) and every compared row; and the code sizes (eval/textspot3_a9m.py codesize's JSON)
and stack peaks (stack below). The JSON keeps eval/textspot_a9.py's form (summary, table, also, one row per line).

stack: the peak stack of each public function of runtime/vocr_textspot3.c, and for reference of the frozen
runtime/vocr_textspot2.c, from gcc -fcallgraph-info=su over the spotter and vocr_lines.c / vocr_lines_rows.c (whose
vocr_line_dark gives the dark flags): VitaSDK gcc -O2 and -Os (Thumb-2, NEON), psp-gcc -O2 -G0 and -Os -G0. Every
frame must be static; the peak is the heaviest path through the call graph across the three files (psp/stack.py's
parser); calls outside them (memcpy, memset, libgcc) are listed, their frames not counted.
"""
import argparse
import datetime
import json
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "train", ROOT / "synth", ROOT / "vita", ROOT / "psp"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot3_a9 as T9  # noqa: E402
import textspot3_a9m as W  # noqa: E402
import textspot_a9 as A9  # noqa: E402

# the runtime's functions by part, beyond eval/textspot3_a9m.PARTS (its speed pass split the 1x1 and depthwise
# kernels into more functions)
EXTRA_PARTS = {"map": ("dw_border", "pw_sums", "pw_store", "pw_block_run", "pw_gather", "pw_gathers")}
STACK_SOURCES = {"v3": ("vocr_textspot3.c", "vocr_lines.c", "vocr_lines_rows.c"),
                 "v2": ("vocr_textspot2.c", "vocr_lines.c", "vocr_lines_rows.c")}
STACK_ROOTS = {"v3": ("vocr_textspot3_find", "vocr_textspot3_find_rows", "vocr_textspot3_load", "vocr_textspot3_maps",
                      "vocr_textspot3_arena_bytes", "vocr_textspot3_frame_arena_bytes"),
               "v2": ("vocr_textspot2_find", "vocr_textspot2_find_rows", "vocr_textspot2_load", "vocr_textspot2_maps")}
DARK = ("vocr_line_dark", "vocr_line_dark_rows")


def parts():
    out = {k: tuple(v) for k, v in W.PARTS.items()}
    for k, v in EXTRA_PARTS.items():
        out[k] = out.get(k, ()) + tuple(v)
    return out


def part_of(fn, lines_fns):
    """eval/textspot3_a9m.part_of with EXTRA_PARTS, GCC's clones (name.isra.0, name.part.0) counted as name."""
    if fn in lines_fns:
        return "dark"
    base = fn.split(".")[0]
    for part, names in W.PARTS.items():
        if base in names:
            return part
    return "other"


def t_block(path):
    """The "t" block from eval/textspot3_a9.py's report JSON of family T's re-measurement."""
    tc = json.loads(Path(path).read_text(encoding="utf-8"))
    tb = tc["textspot3"]
    reps = []
    for r in tb.get("reproduces", []):
        reps.append({"against": r["against"], "sha256": r["sha256"], "all_identical": r["all_identical"],
                     "rows": len(r["rows"]), "identical": sum(x["identical"] for x in r["rows"]),
                     "differ": [x for x in r["rows"] if not x["identical"]]})
    return {"how": "family T's registered shapes (VTSP version 2, half_above 480) and textspot2's s8_c24n4x2 and "
                   "s8_c48n4x2 (half_above 600) measured again by eval/textspot3_a9.py on runtime/vocr_textspot2.c as "
                   "merged, every row compared field by field with the pinned tables (eval/textspot2_a9.compare_rows)",
            "runtime": tb["runtime"], "runtime_files": tb["runtime_files"], "code_sha256": tb["code_sha256"],
            "sources_sha256": tc["sources_sha256"], "rows": tb["rows"], "reproduces": reps,
            "measured": tb.get("measured"), "markdown": T9.markdown(tc)}


def cmd_report(argv):
    ap = argparse.ArgumentParser(prog="textspot3_a9r.py report")
    ap.add_argument("dir")
    ap.add_argument("--also", action="append", default=[])
    ap.add_argument("--also-label", action="append", default=[])
    ap.add_argument("--t-json", required=True)
    ap.add_argument("--code-size", default="")
    ap.add_argument("--stack", default="")
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", default="")
    ap.add_argument("--title", default="")
    args = ap.parse_args(argv)
    saved = (W.PARTS, W.part_of)
    try:
        W.PARTS, W.part_of = parts(), part_of
        with tempfile.TemporaryDirectory(prefix="textspot3-a9r-") as td:
            tmp = Path(td) / "m.json"
            rest = [x for d, lab in zip(args.also, args.also_label + [""] * len(args.also))
                    for x in ("--also", d, "--also-label", lab)]
            extra = ["--code-size", args.code_size] if args.code_size else []
            W.cmd_report([args.dir, *rest, "--json", str(tmp), "--title", args.title, *extra])
            res = json.loads(tmp.read_text())
    finally:
        W.PARTS, W.part_of = saved
    block = res["textspot3"]
    block["split_parts"] = {k: list(v) for k, v in parts().items()}
    block["tools"] = {"m": "eval/textspot3_a9m.py (amendment 4)", "t": "eval/textspot3_a9.py (amendment 1)",
                      "record": "eval/textspot3_a9r.py", "record_sha256": W.sha256(Path(__file__))}
    block["t"] = t_block(args.t_json)
    if args.stack:
        block["stack"] = json.loads(Path(args.stack).read_text(encoding="utf-8"))
    md = markdown(res)
    print(md)
    Path(args.json).write_text(A9.dumps(res) + "\n")
    print("wrote", args.json)
    if args.md:
        Path(args.md).write_text(md + "\n")
        print("wrote", args.md)


def markdown(res):
    b = res["textspot3"]
    out = [W.markdown(res), "", "Family T and textspot2's d2s shapes (runtime/vocr_textspot2.c as merged), measured "
                                "again:", b["t"]["markdown"]]
    for r in b["t"]["reproduces"]:
        out.append(f"\nAgainst {r['against']}: {r['identical']} of {r['rows']} rows identical.")
    st = b.get("stack")
    if st:
        out += ["", "| stack peak (B) | " + " | ".join(st["builds"]) + " |",
                "|---|" + "---:|" * len(st["builds"])]
        for root in STACK_ROOTS["v3"] + STACK_ROOTS["v2"]:
            cells = [str(st["builds"][k]["peaks"].get(root, {}).get("bytes", "-")) for k in st["builds"]]
            out.append(f"| {root} | " + " | ".join(cells) + " |")
    cs = b.get("code_size")
    if cs:
        out += ["", "| code (.text, --gc-sections) | bytes |", "|---|---:|"]
        for k, r in cs["vita"]["builds"].items():
            out.append(f"| VitaSDK {k} | {r['text']:,} (object {r['object_text']:,}) |")
        for k, v in cs.get("psp", {}).get("object_text", {}).items():
            out.append(f"| psp-gcc {k} (object) | {v:,} |")
    return "\n".join(out)


# ----------------------------------------------------------------------------- stack peaks

def peaks(cis, roots, exclude=()):
    """{root: {"bytes", "chain"}} over several .ci files (a call into another file follows it there), the externs."""
    import stack as ST
    nodes, edges = {}, {}
    for text in cis:
        n, e = ST.parse(text)
        nodes.update(n)
        for k, v in e.items():
            edges.setdefault(k, []).extend(v)
    by_name = {t.split(":")[-1]: t for t, v in nodes.items() if v is not None}

    def resolve(t):
        return t if nodes.get(t) is not None else by_name.get(t.split(":")[-1], t)

    @lru_cache(maxsize=None)
    def deepest(t, depth=0):
        if depth > 64:
            raise SystemExit(f"recursion through {t}")
        t = resolve(t)
        own = nodes.get(t)
        if own is None:
            return 0, (t.split(":")[-1] + " (extern)",)
        if own[1] != "static":
            raise SystemExit(f"{t}: a {own[1]} frame")
        best, path = 0, ()
        for c in edges.get(t, []):
            if c.split(":")[-1] in exclude:
                continue
            sz, p = deepest(c, depth + 1)
            if sz > best or not path:
                best, path = sz, p
        return own[0] + best, (f"{t.split(':')[-1]} {own[0]}",) + path

    out = {}
    for r in roots:
        if r in by_name:
            total, path = deepest(by_name[r])
            out[r] = {"bytes": total, "chain": list(path)}
    externs = sorted({c.split(":")[-1] for t in nodes for c in edges.get(t, []) if nodes.get(resolve(c)) is None})
    return out, externs


def cmd_stack(argv):
    ap = argparse.ArgumentParser(prog="textspot3_a9r.py stack")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vitasdk", default=str(W.W2.HERD / "shared/artifacts/vita/vitasdk"))
    ap.add_argument("--pspdev", default=str(W.W2.HERD / "shared/artifacts/psp/tools/pspdev"))
    ap.add_argument("--work", default="")
    args = ap.parse_args(argv)
    work = Path(args.work or tempfile.mkdtemp(prefix="textspot3-stack-"))
    vcc = Path(args.vitasdk) / "bin/arm-vita-eabi-gcc"
    pcc = Path(args.pspdev) / "bin/psp-gcc"
    builds = {"Vita -O2": (vcc, ["-mcpu=cortex-a9", "-mthumb", "-mfpu=neon", "-mfloat-abi=hard", "-O2"]),
              "Vita -Os": (vcc, ["-mcpu=cortex-a9", "-mthumb", "-mfpu=neon", "-mfloat-abi=hard", "-Os"]),
              "PSP -O2 -G0": (pcc, ["-O2", "-G0"]), "PSP -Os -G0": (pcc, ["-Os", "-G0"])}
    doc = {"kind": "textspot3-stack", "computed": datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"), "how": __doc__.split("\n\n")[3].strip().replace("\n", " "),
        "sources_sha256": {f"runtime/{s}": W.sha256(ROOT / "runtime" / s) for s in
                           sorted(set(STACK_SOURCES["v3"]) | set(STACK_SOURCES["v2"]))},
        "compilers": {"vita": W.W2.A9.tool_version(str(vcc)), "psp": W.W2.A9.tool_version(str(pcc))}, "builds": {}}
    for label, (cc, flags) in builds.items():
        rec = {"flags": flags, "peaks": {}, "no_dark": {}, "externs": {}}
        for kind, srcs in STACK_SOURCES.items():
            cis = []
            for src in srcs:
                d = work / label.replace(" ", "_") / kind
                d.mkdir(parents=True, exist_ok=True)
                o = d / (Path(src).stem + ".o")
                subprocess.run([str(cc), "-std=c99", *flags, "-fcallgraph-info=su", f"-I{ROOT / 'runtime'}", "-c",
                                str(ROOT / "runtime" / src), "-o", str(o)], check=True, capture_output=True, cwd=d)
                ci = d / (Path(src).stem + ".ci")
                if not ci.exists():
                    ci = next(d.glob(Path(src).stem + "*.ci"))
                cis.append(ci.read_text(encoding="utf-8"))
            p, ext = peaks(cis, STACK_ROOTS[kind])
            rec["peaks"].update(p)
            nd, _ = peaks(cis, STACK_ROOTS[kind][:1], exclude=DARK)
            rec["no_dark"].update({f"{k} without the dark pass": v for k, v in nd.items()})
            rec["externs"][kind] = ext
        doc["builds"][label] = rec
        print(label, {k: v["bytes"] for k, v in rec["peaks"].items()},
              {k: v["bytes"] for k, v in rec["no_dark"].items()})
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print("wrote", args.out)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "report":
        cmd_report(rest)
    elif cmd == "stack":
        cmd_stack(rest)
    else:
        raise SystemExit(f"unknown command {cmd}: report or stack")
    return 0


if __name__ == "__main__":
    sys.exit(main())
