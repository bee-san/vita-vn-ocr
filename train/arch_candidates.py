#!/usr/bin/env python3
"""Recognizer design candidates inside the Vita ceiling (docs/vocr3-arch-candidates.md).

  arch_candidates.py blobs --out DIR [--only NAME,...]   random-init int8 blobs + DIR/sizes.json
  arch_candidates.py table --dir DIR [--json OUT]       the doc's cost table, joining DIR/sizes.json
                                                         with `vita/budget_sweep.py check ... --out
                                                         DIR/check` (run that through heavy.sh)
  arch_candidates.py rawckpt CKPT OUT                    CKPT with "model" = its raw (non-EMA) weights,
                                                         fused for a --rep run (for proxy scoring)
  arch_candidates.py proxies --dir PROXY_DIR [--json OUT]  the doc's proxy table from
                                                         synth3_ablation.py compare outputs

Every candidate is a train/model.py TinyVNOCR built from the runtime's existing op set (3x3 stem,
k x k depthwise, pointwise, the 2x1 row collapse, 1 x k temporal depthwise, ReLU, residual adds,
the factorised head), so `vocr_load()` runs it unchanged. Blobs use budget_sweep.py's random
weights and random BatchNorm statistics: sizes are exact, and the A9 estimate does not depend on
the weights (the kernels have no data-dependent branches; budget_sweep checks this on FL10, H15
and F20). Training-only variants (--rep, an auxiliary decoder) export to the same blob as their
plain shape. Nothing here reads a protected set.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "vita"), str(ROOT / "eval")]

# name: TinyVNOCR keyword arguments (5,355 classes, height 32 unless given). train2.py flags in
# brackets. "chans" is (stem, 8-row stage, 4-row stage, 2-row stage, frame row).
CANDIDATES = {
    "H15": dict(width=1.5),                                        # VOCR_H15's shape [--width 1.5]
    "F20": dict(width=2.0),                                        # [--width 2.0]
    "LW": dict(chans=(48, 96, 192, 320, 384)),                     # H15's early stages, F20's late ones
    "LWb64": dict(chans=(48, 96, 192, 320, 384), bottleneck=64),
    "LW224": dict(chans=(48, 96, 224, 320, 384)),
    "LW256": dict(chans=(48, 96, 256, 320, 384)),
    "LW384b80": dict(chans=(48, 96, 192, 384, 384), bottleneck=80),
    "EW": dict(chans=(64, 128, 256, 240, 288)),                    # F20's early stages, H15's late ones
    "W175": dict(width=1.75),                                      # uniform, LW's MACs
    "W20b64": dict(width=2.0, bottleneck=64),
    "W2125b64": dict(chans=(64, 136, 272, 344, 408), bottleneck=64),
}


def flags(kw):
    """The train2.py arguments that build a candidate."""
    out = []
    if kw.get("chans"):
        out.append("--chans " + ",".join(map(str, kw["chans"])))
    else:
        out.append(f"--width {kw.get('width', 1.0)}")
    for k in ("bottleneck", "depth", "ctx", "ctx_k"):
        if k in kw:
            out.append(f"--{k.replace('_', '-')} {kw[k]}")
    return " ".join(out)


def random_model(kw, n_classes=5355, seed=0):
    """budget_sweep.random_model's weights and BatchNorm statistics, for any TinyVNOCR shape."""
    import torch
    import torch.nn as nn

    from model import TinyVNOCR
    torch.manual_seed(seed)
    model = TinyVNOCR(n_classes, **kw)
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_mean.normal_(0, 0.1)
                mod.running_var.uniform_(0.5, 2.0)
                mod.weight.uniform_(0.5, 1.5)
                mod.bias.normal_(0, 0.1)
    return model.eval()


def cmd_blobs(args):
    import budget_sweep

    import export
    out = Path(args.out)
    (out / "blobs").mkdir(parents=True, exist_ok=True)
    sizes_path = out / "sizes.json"
    sizes = json.loads(sizes_path.read_text()) if sizes_path.exists() else {}
    names = args.only.split(",") if args.only else list(CANDIDATES)
    for name in names:
        kw = CANDIDATES[name]
        n_classes, height = kw.get("classes", 5355), kw.get("height", 32)
        model = random_model({k: v for k, v in kw.items() if k not in ("classes", "height")}, n_classes)
        data, meta = export.build_blob(export.model_ops(model, height), budget_sweep.vocab_of(n_classes), height,
                                       int8=True)
        (out / "blobs" / f"{name}_w8.vocr").write_bytes(data)
        info = budget_sweep.blob_info(data)
        info.update(config={k: list(v) if isinstance(v, tuple) else v for k, v in kw.items()},
                    train2_flags=flags(kw), random_init=True, params=meta["params"],
                    sha256=hashlib.sha256(data).hexdigest())
        sizes[name] = info
        print(f"{name}: {len(data):,} B, W8I arena {info['arena']['t4h16']['w8i']:,} B, "
              f"{info['macs_per_frame']:,} MACs/frame", file=sys.stderr)
    sizes_path.write_text(json.dumps(sizes, indent=1))


def cmd_table(args):
    import budget_sweep as B
    d = Path(args.dir)
    sizes = json.loads((d / "sizes.json").read_text())
    runs = B.load_runs(d / "check") if (d / "check").exists() else {}
    rows, summary = [], {}
    rows.append("| candidate | train2.py flags | params | int8 blob B | of which head B | W8I arena B | MACs/frame "
                "| est. ms per 640-px line, 333 MHz | 444 MHz | within the ceiling |")
    rows.append("|---|---|---|---|---|---|---|---|---|---|")
    for name in [n for n in CANDIDATES if n in sizes]:
        s = sizes[name]
        r = runs.get((name, *B.BASE))
        ms3 = B._ms(r, 333) if r else None
        ms4 = B._ms(r, 444) if r else None
        ok = None
        if r:
            ok = all(v[3] for v in B.verdict(s, r))
        summary[name] = {"params": s["params"], "blob_bytes": s["blob_bytes"],
                         "head_weight_bytes": s["head_weight_bytes"],
                         "arena_w8i": s["arena"]["t4h16"]["w8i"], "macs_per_frame": s["macs_per_frame"],
                         "ms_333": ms3, "ms_444": ms4, "within_ceiling": ok, "train2_flags": s.get("train2_flags"),
                         "sha256": s.get("sha256")}
        rows.append(f"| {name} | `{s.get('train2_flags', '')}` | {s['params']:,} | {s['blob_bytes']:,} | "
                    f"{s['head_weight_bytes']:,} | {s['arena']['t4h16']['w8i']:,} | {s['macs_per_frame']:,} | "
                    + (f"{B._span(ms3)} | {B._span(ms4)} | {'yes' if ok else 'no'} |" if r else "- | - | - |"))
    sys.stdout.write("\n".join(rows) + "\n")
    if args.json:
        Path(args.json).write_text(json.dumps({"ceiling": B.CEILING, "candidates": summary}, indent=1))


def cmd_proxies(args):
    """The proxy table: C W8I teacher_val CER (rv + rvn) of each run's raw and EMA step-5000 weights
    (synth3_ablation.py compare outputs), the synthetic held-out CERs train2 logged (EMA, fp32) and
    the training rate, plus every comparison's delta with its VN-cluster interval."""
    d = Path(args.dir)
    raw = json.loads((d / "compare_raw.json").read_text())
    ema = json.loads((d / "compare.json").read_text()) if (d / "compare.json").exists() else {"models": {}}
    rows = ["| run | shape (train2.py flags) | seed | threads | lines/s | teacher_val CER, raw | EMA "
            "| s2v / nvv CER (EMA, fp32) |", "|---|---|---|---|---|---|---|---|"]
    out = {"runs": {}, "comparisons": {}}
    for name in sorted(raw["models"]):
        run = "P_" + name[2:]
        rd = d / "runs" / run
        a = json.loads((rd / "args.json").read_text()) if (rd / "args.json").exists() else {}
        last = [json.loads(x) for x in (rd / "metrics.jsonl").read_text().splitlines() if x.strip()][-1] \
            if (rd / "metrics.jsonl").exists() else {}
        rates = []
        log = d / "logs" / f"{run}.log"
        for line in log.read_text().splitlines() if log.exists() else []:
            if line.startswith("step ") and " lines/s" in line:
                rates.append(float(line.split(" lines/s")[0].rsplit(" ", 1)[1]))
        rate = sorted(rates)[len(rates) // 2] if rates else None
        c_raw = raw["models"][name]["cer"]
        c_ema = ema["models"].get(run, {}).get("cer")
        flags = f"--chans {a['chans']}" if a.get("chans") else f"--width {a.get('width', '?')}"
        if a.get("rep", "0") != "0":
            flags += f" --rep {a['rep']}"
        if a.get("guide", "none") != "none":
            flags += f" --guide {a['guide']}"
        mean = lambda c: (c["rv"] + c["rvn"]) / 2 if c else None  # noqa: E731
        out["runs"][run] = {"raw": mean(c_raw), "ema": mean(c_ema), "s2v": last.get("s2v_cer"),
                            "nvv": last.get("nvv_cer"), "threads": a.get("threads"), "lines_per_s": rate,
                            "seed": a.get("seed"), "flags": flags}
        rows.append(f"| {run} | `{flags}` | {a.get('seed', '-')} | {a.get('threads', '-')} | "
                    f"{rate if rate is not None else '-'} | {100 * mean(c_raw):.2f}% | "
                    + (f"{100 * mean(c_ema):.2f}%" if c_ema else "-") + " | "
                    + (f"{100 * last['s2v_cer']:.2f}% / {100 * last['nvv_cer']:.2f}%" if last else "-") + " |")
    rows += ["", "| comparison (B - A) | raw: delta, points [VN-cluster 95% CI] | line CI | VNs better / worse "
             "| EMA: delta [VN CI] |", "|---|---|---|---|---|"]
    for name, c in raw["comparisons"].items():
        v, ln = c["all_vn_cluster"], c["all_line"]
        e = ema.get("comparisons", {}).get(name, {}).get("all_vn_cluster")
        out["comparisons"][name] = {"a": c["a"], "b": c["b"], "raw": v, "raw_line_ci95": ln["ci95"],
                                    "vns_better": c["per_vn"]["better"], "vns_worse": c["per_vn"]["worse"], "ema": e,
                                    "per_seed": {k: x["delta_points"] for k, x in c["per_seed"].items()},
                                    "strata": {k: {kk: x[kk] for kk in ("delta_points", "ci95", "lines")}
                                               for k, x in c["strata"].items()}}
        rows.append(f"| {c['b']} - {c['a']} | {v['delta_points']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] | "
                    f"[{ln['ci95'][0]:+.2f}, {ln['ci95'][1]:+.2f}] | {c['per_vn']['better']} / {c['per_vn']['worse']}"
                    " | "
                    + (f"{e['delta_points']:+.2f} [{e['ci95'][0]:+.2f}, {e['ci95'][1]:+.2f}]" if e else "-") + " |")
    sys.stdout.write("\n".join(rows) + "\n")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")


def raw_checkpoint(ck):
    """A train2.py checkpoint dict with "model" replaced by the raw (non-EMA) weights, fused into the plain
    TinyVNOCR when the run used --rep, so export.load_checkpoint / synth3_ablation.py export read it."""
    import torch  # noqa: F401  (the state tensors are torch tensors)

    from model import TinyVNOCR, fuse_rep
    raw = ck.get("model_raw", ck["model"])
    if int(ck.get("rep") or 0):
        arch = {k: ck[k] for k in ("width", "bottleneck", "depth", "ctx", "ctx_k")}
        m = TinyVNOCR(len(ck["vocab"]) + 1, **arch, chans=ck.get("chans"), rep=int(ck["rep"]))
        m.load_state_dict(raw)
        raw = fuse_rep(m.eval()).state_dict()
    return {**ck, "model": raw}


def cmd_rawckpt(args):
    import torch
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    torch.save(raw_checkpoint(ck), args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("blobs")
    b.add_argument("--out", required=True)
    b.add_argument("--only")
    t = sub.add_parser("table")
    t.add_argument("--dir", required=True)
    t.add_argument("--json")
    p = sub.add_parser("proxies")
    p.add_argument("--dir", required=True, help="the proxy directory: runs/, logs/, compare_raw.json, compare.json")
    p.add_argument("--json")
    r = sub.add_parser("rawckpt")
    r.add_argument("ckpt")
    r.add_argument("out")
    args = ap.parse_args()
    {"blobs": cmd_blobs, "table": cmd_table, "proxies": cmd_proxies, "rawckpt": cmd_rawckpt}[args.cmd](args)


if __name__ == "__main__":
    main()
