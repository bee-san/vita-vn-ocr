#!/usr/bin/env python3
"""Cortex-A9 estimates and exact arenas of textspot3's spotter shapes (issue #108, docs/textspot3.md "Estimates"):
eval/textspot_a9.py's measure and report, through eval/textspot2_a9.py's runtimes, with the blob's half_above a
parameter. Estimates only.

  textspot3_a9.py measure --out DIR [--runtime v2] [--shapes S,...] [--half-above N] [textspot_a9.py measure's
                          other options: --nodark --sizes --formats --cflags --seed --jobs --vitasdk --arm-cc --qemu]
  textspot3_a9.py report DIR [--also DIR2 --also-label TEXT ...] [--json OUT] [--md OUT] [--title TEXT]
                         [--reproduce JSON ...]

measure --runtime v2 (family T, VTSP version 2): eval/textspot2_a9.py's measure --runtime v2, unchanged:
runtime/vocr_textspot2.c (as merged) and its driver runtime/vocr_textspot2_test.c compiled by the Vita compiler
(arm-vita-eabi-gcc, Thumb-2, -O2) and linked by an ARM Linux toolchain, run under the plugin qemu-arm with the Vita's
caches on textspot's pinned procedural frame of each size, priced by vita/a9_cycles.py; est. ms at 333 MHz = the
high bound x 1.0007 / 333 MHz and at 444 MHz that x 333/444; the arenas the runtime's own functions give
(vocr_textspot2_frame_arena_bytes, vocr_textspot2_arena_bytes) under qemu-arm. The blobs are
train/textspot2_export.random_blob (seed 1) at the given half_above: the shapes of train/textspot3_shapes.py are
made visible to train/textspot2_shapes for the call (textspot3_shapes.inject), so textspot2's frozen layers, costs
and blob code build them. half_above names the scale of the rows: 600 is textspot's "h2" (so textspot2's shapes
measured at 600 compare key for key with eval/textspot2_a9_d2s_20261020.json: --reproduce), 0 "f1", any other N
"hN" (family T: 480, "h480").

report: textspot_a9.py's report (its JSON: summary, table, also), plus a "textspot3" block: the protocol's sha256
and budget, the code's hashes, each measured shape's exact figures (train/textspot3_shapes.costs: VTSP bytes, int8
weights, parameters, MACs, receptive field), and with --reproduce the comparison of every row of a shape another
table measured with its row there (eval/textspot2_a9.compare_rows: identical, or each field that differs). The
summary's within_limits is textspot3's budget (VTSP <= 512,000 B, est. <= 300 ms per 960x544 frame at 333 MHz,
arena_up_to_960 <= 262,144 B; det: det+refine adds 1.4 ms per refined box and the refinement's arena). --md writes
the markdown table of the summary.
"""
import argparse
import contextlib
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "train", ROOT / "synth", ROOT / "vita"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot2_a9 as W2  # noqa: E402
import textspot_a9 as A9  # noqa: E402

import textspot2_shapes as S2  # noqa: E402
import textspot3_shapes as S3  # noqa: E402

PROTOCOL = ROOT / "eval" / "textspot3_protocol.json"
BUDGET = {"vtsp_max": 512000, "ms_max_333_960": 300.0, "arena_max": 262144}
RUNTIMES = ("v2",)
WRAPPER_FILES = ("eval/textspot3_a9.py", "eval/textspot2_a9.py", "eval/textspot_a9.py", "train/textspot3_shapes.py",
                 "train/textspot2_shapes.py", "train/textspot2_export.py", "train/textdet_model.py",
                 "train/textdet_export.py", "synth/textspot2_twin.py", "synth/textspot_twin.py",
                 "synth/textdet_twin.py", "vita/a9_cycles.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scale_name(half_above):
    """The rows' scale name for a blob's half_above: textspot's "h2" (600) and "f1" (0), else "hN"."""
    return {600: "h2", 0: "f1"}.get(int(half_above), f"h{int(half_above)}")


@contextlib.contextmanager
def scales(half_above):
    """textspot_a9.SCALES with the scale of this half_above (removed again unless textspot_a9 defines it)."""
    name = scale_name(half_above)
    had = name in A9.SCALES
    if had and A9.SCALES[name] != int(half_above):
        raise ValueError(f"scale {name} is {A9.SCALES[name]}, not {half_above}")
    A9.SCALES[name] = int(half_above)
    try:
        yield name
    finally:
        if not had:
            A9.SCALES.pop(name, None)


@contextlib.contextmanager
def runtime(kind, half_above_values=()):
    """eval/textspot_a9's module state for one runtime: v2 = eval/textspot2_a9.runtime("v2") with
    train/textspot3_shapes' T shapes visible to textspot2's frozen code; the scales of the given half_above values
    defined."""
    if kind not in RUNTIMES:
        raise SystemExit(f"runtime {kind}: one of {RUNTIMES}")
    with contextlib.ExitStack() as st:
        st.enter_context(S3.inject())
        st.enter_context(W2.runtime(kind))
        for ha in sorted(set(int(v) for v in half_above_values)):
            st.enter_context(scales(ha))
        yield


def code_hashes(kind):
    out = W2.code_hashes(kind)
    for f in WRAPPER_FILES:
        out[f] = sha256(ROOT / f)
    return dict(sorted(out.items()))


def shape_costs(name):
    """Exact figures of a measured shape: textspot3's (train/textspot3_shapes.costs), or for textspot2's shapes
    textspot2's own (train/textspot2_export.costs / train/textspot2_shapes.costs) with their VTSP bytes."""
    if name in S3.SHAPES:
        return S3.costs(name)
    import textspot2_export as X
    if name in S2.D2S:
        c = X.costs(name)
        c["vtsp_bytes"] = 32 + S3.vtdt_bytes(X.layers(X.shape(name)))
        return {"family": "textspot2 d2s", **c}
    if name in S2.SHAPES:
        import textdet_model as TM
        S2.register()
        c = S2.costs(name)
        c["vtsp_bytes"] = 32 + S3.vtdt_bytes(TM.layers(TM.SHAPES[name]))
        return {"family": "textspot2 kroa", **c}
    raise KeyError(name)


# ----------------------------------------------------------------------------- measure

def cmd_measure(argv):
    ap = argparse.ArgumentParser(prog="textspot3_a9.py measure", add_help=False)
    ap.add_argument("--runtime", choices=RUNTIMES, default="v2")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--half-above", type=int, default=S3.T_HALF_ABOVE)
    ap.add_argument("--out", required=True)
    args, rest = ap.parse_known_args(argv)
    if "--scales" in rest:
        raise SystemExit("--half-above sets the scale")
    shapes = args.shapes or ",".join(S3.T)
    with S3.inject(), scales(args.half_above):     # eval/textspot2_a9.cmd_measure enters its runtime itself
        W2.cmd_measure(["--runtime", args.runtime, "--shapes", shapes, "--scales", scale_name(args.half_above),
                        "--out", args.out, *rest])
    meta_path = Path(args.out) / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["textspot3"] = {"runtime": args.runtime, "half_above": args.half_above,
                         "argv": W2.portable(["measure", *argv]), "registered": S3.REGISTERED}
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")


# ----------------------------------------------------------------------------- report

def half_aboves(*dirs):
    out = set()
    for d in dirs:
        meta = json.loads((Path(d) / "meta.json").read_text())
        out |= {int(r["half_above"]) for r in meta["runs"]}
    return out


def within(r):
    return r["vtsp_bytes"] <= BUDGET["vtsp_max"] and r["ms_333_960"] <= BUDGET["ms_max_333_960"] and \
        r["arena_up_to_960"] <= BUDGET["arena_max"]


def cmd_report(argv):
    ap = argparse.ArgumentParser(prog="textspot3_a9.py report")
    ap.add_argument("dir")
    ap.add_argument("--json")
    ap.add_argument("--md")
    ap.add_argument("--also", action="append", default=[])
    ap.add_argument("--also-label", action="append", default=[])
    ap.add_argument("--reproduce", action="append", default=[], help="tables whose rows of our shapes must agree")
    ap.add_argument("--title", default="")
    args = ap.parse_args(argv)
    meta = json.loads((Path(args.dir) / "meta.json").read_text())
    kind = meta.get("textspot3", meta.get("textspot2", {})).get("runtime", "v2")
    for d in args.also:
        m = json.loads((Path(d) / "meta.json").read_text())
        if m.get("textspot3", m.get("textspot2", {})).get("runtime", "v2") != kind:
            raise SystemExit(f"--also {d} measured another runtime")
    with runtime(kind, half_aboves(args.dir, *args.also)):
        with tempfile.TemporaryDirectory(prefix="textspot3-a9-") as td:
            tmp = Path(td) / "a9.json"
            rest = [x for d, lab in zip(args.also, args.also_label + [""] * len(args.also))
                    for x in ("--also", d, "--also-label", lab)]
            A9.main(["report", args.dir, "--json", str(tmp), *rest])
            res = json.loads(tmp.read_text())
        shapes = {name: shape_costs(name) for name in sorted({t["shape"] for _, tab in tables(res) for t in tab})}
    proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    res["rule_reads"]["limits"] = {"vtsp_max": BUDGET["vtsp_max"], "arena_max": BUDGET["arena_max"],
                                   "ms_max_333_960": BUDGET["ms_max_333_960"]}
    res["rule_reads"]["within_limits"] = (
        "textspot3's budget (eval/textspot3_protocol.json \"budget\"), for det: vtsp_bytes <= vtsp_max, "
        "ms_333_960 <= ms_max_333_960 and arena_up_to_960 <= arena_max; det+refine adds 1.4 ms per refined "
        "horizontal box at the arm's median and the refinement's 72,391-B arena")
    for r in res["summary"]:
        c = shapes[r["shape"]]
        r.update(weights_int8=c["weights_int8"], params=c["params"], receptive_field=c["receptive_field"])
        r["within_limits"] = within(r)
    block = {"protocol": {"path": "eval/textspot3_protocol.json", "sha256": sha256(PROTOCOL),
                          "budget": proto["budget"], "estimates": proto["estimates"]["binding"]},
             "title": args.title, "runtime": kind, "registered": S3.REGISTERED,
             "runtime_files": {"v2": "runtime/vocr_textspot2.c as merged (VTSP versions 1 and 2; driver "
                                     "runtime/vocr_textspot2_test.c)"}[kind],
             "code_sha256": code_hashes(kind), "shapes": shapes,
             "measured": {"main": meta.get("textspot3", {}).get("argv") or W2.portable(
                 meta.get("textspot2", {}).get("argv")),
                 "also": [json.loads((Path(d) / "meta.json").read_text()).get("textspot3", {}).get("argv")
                          for d in args.also]}}
    if args.reproduce:
        block["reproduces"] = []
        for path in args.reproduce:
            ref = json.loads(Path(path).read_text(encoding="utf-8"))
            rows = []
            for label, ours in tables(res):
                for _, theirs in tables(ref):
                    if {t["flags"] for t in ours} == {t["flags"] for t in theirs}:
                        rows += W2.compare_rows(ours, theirs, label)
            block["reproduces"].append({"against": W2.portable([path])[0], "sha256": sha256(path), "rows": rows,
                                        "all_identical": bool(rows) and all(r["identical"] for r in rows)})
    block["rows"] = summary_rows(res, shapes)
    res["textspot3"] = block
    print_summary(res)
    if args.json:
        Path(args.json).write_text(A9.dumps(res) + "\n")
        print("wrote", args.json)
    if args.md:
        Path(args.md).write_text(markdown(res) + "\n")
        print("wrote", args.md)


def _f(v, fmt="{:.1f}"):
    return "-" if v is None else fmt.format(v)


def tables(res):
    """[(label, rows)]: the main table and every --also table of a report."""
    return [("main", res["table"])] + [(a.get("label") or "also", a["table"]) for a in res.get("also", [])]


def summary_rows(res, shapes):
    """One row per (shape, half_above) over every table of the report (grey frames): the default call's and
    VOCR_TEXTSPOT_NO_DARK's est. ms at 960x544 and 480x272 (@333, @444), the exact arenas, the shape's figures and
    textspot3's budget check (det)."""
    cells = {}
    for _, tab in tables(res):
        for t in tab:
            if t["format"] != "grey":
                continue
            key = (t["shape"], t["half_above"])
            mode = "dark" if t["flags"] == "0" else "nodark"
            cells.setdefault(key, {})[(mode, tuple(t["size"]))] = t
    out = []
    for (shape, ha), c in sorted(cells.items(), key=lambda kv: (kv[0][1], shapes[kv[0][0]]["vtsp_bytes"], kv[0][0])):
        s = shapes[shape]
        r = {"shape": shape, "half_above": ha, "family": s.get("family"), "vtsp_bytes": s["vtsp_bytes"],
             "weights_int8": s["weights_int8"], "params": s["params"], "receptive_field": s["receptive_field"],
             "macs_960": s["macs"]["960x544"], "macs_480": s["macs"]["480x272"]}
        for mode in ("dark", "nodark"):
            for size, tag in (((960, 544), "960"), ((480, 272), "480")):
                t = c.get((mode, size))
                suffix = "" if mode == "dark" else "_nodark"
                r[f"ms_333_{tag}{suffix}"] = t["ms_per_frame"]["333"] if t else None
                r[f"ms_444_{tag}{suffix}"] = t["ms_per_frame"]["444"] if t else None
                if mode == "dark" and t:
                    r[f"arena_up_to_{tag}"] = t["arena_bytes"]
                    r[f"arena_frame_{tag}"] = t["frame_arena_bytes"]
                    r[f"boxes_{tag}"] = t["boxes"]
                    r[f"truncated_{tag}"] = t["truncated"]
        if r["ms_333_960"] is not None and "arena_up_to_960" in r:
            r["within_budget"] = within(r)
        out.append(r)
    return out


def markdown(res):
    b = res["textspot3"]
    lines = [f"{b['title']}" if b["title"] else "", "",
             "Estimates (the A9 cycle model on the Vita compiler's code under qemu-arm; nothing ran on a console). "
             "Arenas exact (the runtime's functions under qemu-arm). "
             f"Runtime: {b['runtime_files']}. Shapes registered: {'yes' if b['registered'] else 'no (candidates)'}.",
             "",
             "| shape | half_above | VTSP B | int8 weights | MACs / 960x544 | RF px | est. ms 960x544 @333 / @444 "
             "(rule) | no dark @333 / @444 | 480x272 @333 / @444 | 480x272 no dark @333 / @444 | arena up to "
             "960x544 (rule) | arena 960x544 frame | arena 480x272 | within budget |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in b["rows"]:
        lines.append(
            f"| {r['shape']} | {r['half_above']} | {r['vtsp_bytes']:,} | {r['weights_int8']:,} | {r['macs_960']:,} | "
            f"{r['receptive_field']} | {_f(r['ms_333_960'])} / {_f(r['ms_444_960'])} | "
            f"{_f(r['ms_333_960_nodark'])} / {_f(r['ms_444_960_nodark'])} | "
            f"{_f(r['ms_333_480'])} / {_f(r['ms_444_480'])} | {_f(r['ms_333_480_nodark'])} / "
            f"{_f(r['ms_444_480_nodark'])} | {r.get('arena_up_to_960', 0):,} | {r.get('arena_frame_960', 0):,} | "
            f"{r.get('arena_up_to_480', 0):,} | {'yes' if r.get('within_budget') else 'no'} |")
    for rep in b.get("reproduces", []):
        same = sum(r["identical"] for r in rep["rows"])
        lines += ["", f"Rows of shapes {rep['against']} measured: {same} of {len(rep['rows'])} identical "
                      f"(every field: counts, cycles, ms, misses, boxes, arenas, blob and frame sha256)."]
    return "\n".join(lines)


def print_summary(res):
    print(markdown(res))
    for rep in res["textspot3"].get("reproduces", []):
        for r in rep["rows"]:
            if not r["identical"]:
                print(f"  DIFFERS {r['table']} {r['shape']} {r['size']} {r['flags']}: {r['diffs']}")


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
    else:
        raise SystemExit(f"unknown command {cmd}: measure or report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
