#!/usr/bin/env python3
"""Compare the Vita app's results.tsv with the host runtime, and summarise speed and memory.

  compare.py --bundle DIR results.tsv [--emulator] [--json OUT]

DIR is the vita/bundle.py output the run used (its cases.json). Reports, per model:
  * agreement with the host build of the same C code: MATCH (same text and logits CRC-32,
    and for regions the same line boxes), TEXT (same text, other logits), DIFF, ERR, and
    cases that did not run;
  * CER and exact lines against the hand-checked ground truth, with eval/metrics.py's
    normalisation: GT-box line crops (with the console's polarity rule, so not eval's
    number), and for dialogue regions the console pipeline (line finder + recogniser),
    scored like eval/run.py's finder mode (eval/finders.match);
  * time per line from sceKernelGetProcessTimeWide (median of the per-case medians, and
    per CTC frame, and scaled to a 640-px line = 80 frames), the regions' whole-pipeline
    time, and the ARM performance counters when the firmware gave them;
  * the memory the app reported: blob, arena, line-finder arena and the runtime's stack.
--emulator marks the timings as an emulator's (Vita3K runs ARM code through an x86 JIT, so
its times say nothing about a PS Vita).
"""
import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent / "eval"


def eval_module(name):
    """eval/<name>.py, loaded by path under a private name: eval/linefind has a metrics.py of
    its own, so a bare 'import metrics' depends on sys.path order."""
    key = f"_vita_eval_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, EVAL / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


def unescape(s):
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            out.append({"t": "\t", "n": "\n", "\\": "\\"}.get(s[i + 1], s[i + 1]))
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def parse_results(path):
    """(meta lines, rows as dicts) of a results.tsv."""
    meta, rows, cols = [], [], None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("#cols "):
            cols = line[len("#cols "):].split("\t")
        elif line.startswith("#"):
            meta.append(line[1:].strip())
        elif line.strip() and cols:
            vals = line.split("\t")
            if len(vals) != len(cols):
                continue                      # a row cut short by a crash
            r = dict(zip(cols, vals))
            r["text_host"], r["text_vita"] = unescape(r["text_host"]), unescape(r["text_vita"])
            rows.append(r)
    return meta, rows


def meta_fields(meta, tag):
    """key=value pairs of the first '# tag ...' line."""
    for m in meta:
        if m.split(" ", 1)[0] == tag:
            return dict(kv.split("=", 1) for kv in m.split()[1:] if "=" in kv)
    return {}


def quantiles(v):
    if not v:
        return None
    v = sorted(v)
    q = lambda p: v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]  # noqa: E731
    return dict(n=len(v), median=statistics.median(v), p10=q(0.1), p90=q(0.9), min=v[0], max=v[-1])


def score(cases, rows_by_id):
    """CER of the console output against the ground truth: line cases, and regions the way
    eval/run.py scores a finder (pipeline and concatenated CER)."""
    finders, metrics = eval_module("finders"), eval_module("metrics")
    line = {"errs": 0, "chars": 0, "exact": 0, "lines": 0}
    by_set = {}
    reg_by_set = {}
    for c in cases:
        r = rows_by_id.get(c["id"])
        if not r or r["status"] == "ERR":
            continue
        if c["kind"] == 0:
            p, g = metrics.norm(r["text_vita"]), metrics.norm(c["gt"])
            for t in (line, by_set.setdefault(c.get("set", "?"), {"errs": 0, "chars": 0, "exact": 0, "lines": 0})):
                t["errs"] += metrics.lev(p, g)
                t["chars"] += len(g)
                t["exact"] += p == g
                t["lines"] += 1
            continue
        preds = r["text_vita"].split("\n") if r["text_vita"] else []
        boxes = c["expected"][r["model"]]["boxes"]
        if r["status"] != "MATCH" or len(boxes) != len(preds):
            continue                          # boxes are only known when they equal the host's
        reg = reg_by_set.setdefault(c.get("set", "?"), {"errors": 0, "chars": 0, "concat_errors": 0,
                                                      "concat_chars": 0, "gt_lines": 0, "found": 0})
        dets = [tuple(b[:4]) for b in boxes]
        gts = [dict(g, box_r=tuple(g["box"])) for g in c["rows"]]
        pairs, unmatched = finders.match(dets, gts)
        scored = [g for g in gts if not g["uncertain"]]
        reg["gt_lines"] += len(scored)
        reg["chars"] += sum(len(metrics.norm(g["text"])) for g in scored)
        matched = {j for _, j in pairs}
        for i, j in pairs:
            if not gts[j]["uncertain"]:
                reg["found"] += 1
                reg["errors"] += metrics.lev(metrics.norm(preds[i]), metrics.norm(gts[j]["text"]))
        reg["errors"] += sum(len(metrics.norm(g["text"])) for j, g in enumerate(gts)
                             if j not in matched and not g["uncertain"])
        reg["errors"] += sum(len(metrics.norm(preds[i])) for i in unmatched)
        if not any(g["uncertain"] for g in gts):
            order = sorted(range(len(dets)), key=lambda i: (dets[i][1], dets[i][0]))
            cat_p = "".join(metrics.norm(preds[i]) for i in order)
            cat_g = "".join(metrics.norm(g["text"]) for g in sorted(gts, key=lambda g: (g["box"][1], g["box"][0])))
            reg["concat_errors"] += metrics.lev(cat_p, cat_g)
            reg["concat_chars"] += len(cat_g)
    out = {}
    if line["lines"]:
        out["lines"] = dict(n=line["lines"], cer=line["errs"] / max(1, line["chars"]),
                            exact=line["exact"] / line["lines"])
        out["lines_by_set"] = {k: dict(n=t["lines"], cer=t["errs"] / max(1, t["chars"]), exact=t["exact"] / t["lines"])
                               for k, t in sorted(by_set.items())}
    out["regions_by_set"] = {
        k: dict(gt_lines=r["gt_lines"], line_recall=r["found"] / r["gt_lines"],
                pipeline_cer=r["errors"] / max(1, r["chars"]),
                concat_cer=r["concat_errors"] / max(1, r["concat_chars"]))
        for k, r in sorted(reg_by_set.items()) if r["gt_lines"]}
    return out


def summarise(bundle, results, emulator=False):
    manifest = json.loads((Path(bundle) / "cases.json").read_text(encoding="utf-8"))
    meta, rows = parse_results(results)
    cases = manifest["cases"]
    by_case = {c["id"]: c for c in cases}
    report = {"bundle": str(bundle), "results": str(results), "emulator": emulator, "meta": meta, "models": {}}
    for m in manifest["models"]:
        name = m["name"]
        mr = [r for r in rows if r["model"] == name]
        if not mr:
            continue
        status = {s: sum(r["status"] == s for r in mr) for s in ("MATCH", "TEXT", "DIFF", "ERR")}
        crc_bad = [r["id"] for r in mr if r["id"] in by_case and
                   r["crc_host"].lower() != f"{by_case[r['id']]['expected'][name]['crc']:08x}"]
        text_bad = [r["id"] for r in mr if r["id"] in by_case and
                    r["text_host"] != by_case[r["id"]]["expected"][name]["text"]]
        line_rows = [r for r in mr if r["kind"] == "line" and r["status"] != "ERR"]
        reg_rows = [r for r in mr if r["kind"] == "region" and r["status"] != "ERR"]
        usf = [float(r["us_per_frame"]) for r in line_rows if float(r["us_per_frame"]) > 0]
        pmu = [r for r in line_rows if int(r["cycles"]) > 0]
        rep = {
            "cases": len(mr), "missing": sorted(set(by_case) - {r["id"] for r in mr}), **status,
            "bundle_mismatch": crc_bad + text_bad,   # results.tsv from another bundle?
            "not_matching": [(r["id"], r["status"], r["text_host"], r["text_vita"]) for r in mr
                             if r["status"] != "MATCH"],
            "line_ms": quantiles([float(r["ms_median"]) for r in line_rows]),
            "us_per_frame": quantiles(usf),
            "ms_per_640px": statistics.median(usf) * 80 / 1000 if usf else None,
            "region_ms": quantiles([float(r["ms_median"]) for r in reg_rows]),
            "accuracy": score(cases, {r["id"]: r for r in mr}),
        }
        if pmu:
            frames = sum(int(r["frames"]) for r in pmu)
            tot = {k: sum(int(r[k]) for r in pmu) for k in ("cycles", "insns", "neon", "dmiss", "dstall", "nodispatch")}
            rep["pmu"] = dict(lines=len(pmu), cycles_per_frame=tot["cycles"] / frames,
                              ipc=tot["insns"] / max(1, tot["cycles"]), neon_share=tot["neon"] / max(1, tot["insns"]),
                              dmiss_per_frame=tot["dmiss"] / frames, dstall_share=tot["dstall"] / max(1, tot["cycles"]))
        report["models"][name] = rep
    report["app"] = {k: meta_fields(meta, k) for k in ("vocr_test", "config", "clock", "sizes", "stack", "pmu", "done",
                                                       "lines")}
    report["app"]["raw"] = {k: next((m for m in meta if m.split(" ", 1)[0] == k), "")
                            for k in ("vocr_test", "done", "stack", "clock", "sizes", "pmu")}
    report["app"]["model_lines"] = [m for m in meta if m.startswith("model ")]
    return report


def fmt_q(q, unit, digits=1):
    if not q:
        return "-"
    return f"{q['median']:.{digits}f} {unit} (p10 {q['p10']:.{digits}f}, p90 {q['p90']:.{digits}f}, n={q['n']})"


def print_report(rep):
    app = rep["app"]
    clock = app["clock"].get("arm_mhz", "?")
    where = "Vita3K (emulator)" if rep["emulator"] else "the console"
    print(f"results: {rep['results']}  ({where}, ARM clock {clock} MHz as reported by scePowerGetArmClockFrequency)")
    print(f"app: {app['raw']['vocr_test']}; {app['raw']['done']}")
    print(f"  {app['raw']['sizes']}; {app['raw']['pmu']}")
    for line in app["model_lines"]:
        print(f"  {line}")
    st = app["stack"]
    if st:
        print(f"  runtime stack (probe thread): {st.get('probe_used', '?')} B of {st.get('probe_stack', '?')} B; "
              f"line finder arena {app['lines'].get('arena', '?')} B")
    for name, m in rep["models"].items():
        print(f"\n[{name}] {m['cases']} cases: {m['MATCH']} MATCH, {m['TEXT']} TEXT, {m['DIFF']} DIFF, {m['ERR']} ERR"
              f", {len(m['missing'])} not run")
        if m["bundle_mismatch"]:
            print(f"  warning: {len(m['bundle_mismatch'])} rows disagree with cases.json: results from another bundle?")
        for ident, status, host, vita in m["not_matching"][:10]:
            print(f"  {status} {ident}: host {host!r} / vita {vita!r}")
        acc = m["accuracy"]
        for name_set, a in acc.get("lines_by_set", {}).items():
            print(f"  {name_set} GT-box lines (console polarity rule): CER {100 * a['cer']:.2f}%, "
                  f"exact {100 * a['exact']:.1f}% ({a['n']} lines)")
        for name_set, a in acc.get("regions_by_set", {}).items():
            print(f"  {name_set}, console pipeline: line recall {100 * a['line_recall']:.1f}%, pipeline CER "
                  f"{100 * a['pipeline_cer']:.2f}%, concat CER {100 * a['concat_cer']:.2f}% ({a['gt_lines']} GT lines)")
        tag = " [emulator time, not console time]" if rep["emulator"] else ""
        print(f"  time per line: {fmt_q(m['line_ms'], 'ms')}{tag}")
        print(f"  time per CTC frame: {fmt_q(m['us_per_frame'], 'us', 0)}; "
              f"640-px line: {m['ms_per_640px']:.1f} ms{tag}" if m["ms_per_640px"] else "  time per frame: -")
        print(f"  region (finder + lines): {fmt_q(m['region_ms'], 'ms')}{tag}")
        if "pmu" in m:
            p = m["pmu"]
            print(f"  PMU ({p['lines']} lines): {p['cycles_per_frame']:.0f} cycles/frame, IPC {p['ipc']:.2f}, "
                  f"NEON {100 * p['neon_share']:.0f}% of instructions, {p['dmiss_per_frame']:.0f} L1D misses/frame, "
                  f"{100 * p['dstall_share']:.0f}% of cycles in D-cache stalls")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", type=Path)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--emulator", action="store_true", help="the results come from an emulator")
    ap.add_argument("--json", type=Path, help="write the summary as JSON")
    args = ap.parse_args()
    rep = summarise(args.bundle, args.results, args.emulator)
    print_report(rep)
    if args.json:
        args.json.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    bad = any(m["DIFF"] or m["ERR"] or m["missing"] for m in rep["models"].values()) or not rep["models"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
