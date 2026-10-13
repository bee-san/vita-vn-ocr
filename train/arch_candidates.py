#!/usr/bin/env python3
"""Recognizer design candidates inside the Vita ceiling (docs/vocr3-arch-candidates.md).

  arch_candidates.py blobs --out DIR [--only NAME,...]   random-init int8 blobs + DIR/sizes.json
  arch_candidates.py table --dir DIR [--json OUT]       the doc's cost table, joining DIR/sizes.json
                                                         with `vita/budget_sweep.py check ... --out
                                                         DIR/check` (run that through heavy.sh)

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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("blobs")
    b.add_argument("--out", required=True)
    b.add_argument("--only")
    t = sub.add_parser("table")
    t.add_argument("--dir", required=True)
    t.add_argument("--json")
    args = ap.parse_args()
    {"blobs": cmd_blobs, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    main()
