#!/usr/bin/env python3
"""Model B (vita-vn-ocr #126): PSP cost of candidate recognizer shapes in PPSSPPHeadless.

  modelb_sweep.py --shapes SHAPES.json --pack BOX.bin --out DIR --eboot scalar=E.PBP [--eboot vfpu=V.PBP]
                  [--configs w8f:4:16,w8i:4:16,...] [--variants plain,gate0,gate1,l1:63:0,l1:63:1,...]

Each shape in SHAPES.json ({"name": ..., and any of width, chans, bottleneck, depth, ctx, ctx_k, stride,
height, classes}; defaults VOCR5-S: width 1.0, bottleneck 96, depth 1, ctx 2, ctx_k 5, stride 8, height 32,
4,987 classes) is built random-initialised (seed 0) by train/model.py, exported int8 by train/export.py, and
run by the PSP test app (psp/main.c) in PPSSPPHeadless on the crops of BOX.bin, once per build x config x
variant. The work the kernels do does not depend on the weights (the few data-dependent branches - ReLU,
argmax, CTC - are noise), so the emulated time is the shape's. Variants force the head gate and the
two-level head to their extremes, so that a measured skip or escape rate interpolates between them:

  plain        the blob as exported (no gate, full head)
  gate0        a 16-hidden gate (docs/headgate.md) that never skips: the gate's own cost on every frame
  gate1        the same gate skipping every frame: no head at all (the head's blank logit only)
  l1:B:E       the two-level head (docs/head2level.md) with level 1 = the head's first B 8-row blocks and a
               16-hidden escape that never (E = 0) or always (E = 1) fires
  g1l1:B:E     both: a gate that never skips (gate0's cost) and the two-level head

The gate and the two-level head run in W8I only (vocr_opts), so their variants are skipped in other modes.
PPSSPP's time is emulated (about one cycle per instruction at 333 MHz, no cache or pipeline model), not
console time; psp/README.md converts it. Writes DIR/sweep.json and prints a markdown table.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "train"), str(ROOT / "runtime"), str(ROOT / "synth")]

DEFAULT = {"width": 1.0, "bottleneck": 96, "depth": 1, "ctx": 2, "ctx_k": 5, "stride": 8, "height": 32,
           "classes": 4987, "chans": None}
BIG = np.float32(3.0e38)


def build_shape(shape, seed=0):
    """(blob bytes, info) of a random-initialised shape."""
    import torch

    import export as E
    from common import load_vocab
    from model import TinyVNOCR
    k = {**DEFAULT, **{key: v for key, v in shape.items() if key != "name"}}
    torch.manual_seed(seed)
    model = TinyVNOCR(k["classes"], width=k["width"], bottleneck=k["bottleneck"], depth=k["depth"], ctx=k["ctx"],
                      ctx_k=k["ctx_k"], chans=k["chans"], stride=k["stride"]).eval()
    vocab = load_vocab(ROOT / "data" / "vocab_vn_6459.txt")
    if k["classes"] - 1 > len(vocab):
        raise ValueError(f"at most {len(vocab) + 1} classes")
    vocab = vocab[:k["classes"] - 1]
    ops = E.model_ops(model, k["height"])
    blob, meta = E.build_blob(ops, vocab, k["height"], True, None, b"modelb-sweep")
    info = E.read_blob(blob)
    macs, head = E.macs_per_frame(info)
    return blob, {"knobs": k, "params": meta["params"], "blob_bytes": len(blob), "macs_per_frame": macs,
                  "head_macs_per_frame": head, "stride": meta["stride"]}


def random_gate(blob, skip_all, hidden=16, seed=1):
    """blob + a gate section with random weights whose threshold skips every frame or none."""
    import headgate as G
    rng = np.random.default_rng(seed)
    K = G.head_op(blob)["k"]
    params = G.quantise(rng.normal(0, 0.1, (hidden, K)), rng.normal(0, 0.1, hidden), rng.normal(0, 0.1, hidden),
                        0.0)
    return G.append_gate(G.base_of(blob), params, -BIG if skip_all else BIG)


def random_two_level(blob, l1_blocks, fire_all, hidden=16, seed=2):
    """blob (with or without a gate) + a VOC2 trailer: level 1 = the first l1_blocks head blocks, an escape
    with random weights whose threshold fires on every frame or none."""
    import head2level as T
    import headgate as G
    rng = np.random.default_rng(seed)
    K = G.head_op(blob)["k"]
    q = rng.integers(-127, 128, (hidden, K)).astype(np.int8)
    esc = dict(q=q, scale=np.full(hidden, 1e-4, np.float32), bias=rng.normal(0, 0.1, hidden).astype(np.float32),
               wl1=rng.normal(0, 0.1, hidden).astype(np.float32), wb0=rng.normal(0, 0.1, hidden).astype(np.float32),
               w2=rng.normal(0, 0.1, hidden).astype(np.float32), b2=np.float32(0), t=-BIG if fire_all else BIG)
    base = G.base_of(blob)
    trailer = T.write_trailer(base, l1_blocks, esc)
    return T.attach(blob, trailer)


def variant_blob(blob, variant):
    if variant == "plain":
        return blob, {}
    if variant in ("gate0", "gate1"):
        return random_gate(blob, variant == "gate1"), {"head_gate": "on"}
    m = re.fullmatch(r"(g1)?l1:(\d+):([01])", variant)
    if m:
        b = random_gate(blob, False) if m.group(1) else blob
        opts = {"two_level": True}
        if m.group(1):
            opts["head_gate"] = "on"
        return random_two_level(b, int(m.group(2)), m.group(3) == "1"), opts
    raise ValueError(f"unknown variant {variant}")


def run_psp(eboot, blob_path, mode, pack, tile, head, opts, timeout=1800):
    cmd = [sys.executable, str(HERE / "ppsspp_cli.py"), str(blob_path), "--mode", mode, "--pack", str(pack),
           "--repeat", "1", "--tile", str(tile), "--head", str(head), "--eboot", str(eboot)]
    if opts.get("head_gate"):
        cmd += ["--head-gate", opts["head_gate"]]
    if opts.get("two_level"):
        cmd += ["--two-level"]
    stats = Path(str(blob_path) + f".{mode}.{tile}.{head}.stats.json")
    cmd += ["--stats", str(stats)]
    p = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=timeout)
    if p.returncode:
        return {"error": (p.stderr or p.stdout)[-600:]}
    st = json.loads(stats.read_text())
    return {"lines": st["lines"], "frames": st["frames"], "us_total": st["us_total"],
            "us_frame": st["us_total"] / max(st["frames"], 1), "arena": st["arena_bytes"],
            "gated": st.get("gated", 0), "escaped": st.get("escaped", 0), "stack_used": st["stack_used"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eboot", action="append", required=True, help="NAME=PATH")
    ap.add_argument("--configs", default="w8f:4:16,w8i:4:16", help="mode:tile:head,...")
    ap.add_argument("--variants", default="plain")
    ap.add_argument("--only", help="comma list of shape names")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shapes = json.loads(Path(a.shapes).read_text())
    eboots = dict(e.split("=", 1) for e in a.eboot)
    res_path = out / "sweep.json"
    rows = json.loads(res_path.read_text()) if res_path.exists() else []
    done = {(r["name"], r["eboot"], r["config"], r["variant"]) for r in rows if "error" not in r["psp"]}
    for shape in shapes:
        name = shape["name"]
        if a.only and name not in a.only.split(","):
            continue
        blob, info = build_shape(shape)
        slug = re.sub(r"[^a-z0-9.]+", "_", name.lower())
        for variant in a.variants.split(","):
            vb, opts = variant_blob(blob, variant)
            vpath = out / f"{slug}__{re.sub(r'[^a-z0-9]+', '_', variant)}.vocr"
            vpath.write_bytes(vb)
            for cfg in a.configs.split(","):
                mode, tile, head = cfg.split(":")
                if opts and mode != "w8i":
                    continue
                for ename, eboot in eboots.items():
                    key = (name, ename, cfg, variant)
                    if key in done:
                        continue
                    r = run_psp(eboot, vpath, mode, a.pack, int(tile), int(head), opts)
                    row = {"name": name, "eboot": ename, "config": cfg, "variant": variant, **info, "psp": r}
                    rows.append(row)
                    done.add(key)
                    print(json.dumps({k: row[k] for k in ("name", "eboot", "config", "variant", "macs_per_frame")}
                                     | {"us_frame": r.get("us_frame"), "us_total": r.get("us_total"),
                                        "arena": r.get("arena"), "err": r.get("error", "")[:200]}), flush=True)
                    res_path.write_text(json.dumps(rows, indent=1))
    print("\n| shape | params | MACs/frame | head MACs | int8 blob B | build | config | variant | us/frame | "
          "us total | arena B |")
    print("|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|")
    for r in rows:
        p = r["psp"]
        print(f"| {r['name']} | {r['params']:,} | {r['macs_per_frame']:,} | {r['head_macs_per_frame']:,} | "
              f"{r['blob_bytes']:,} | {r['eboot']} | {r['config']} | {r['variant']} | "
              f"{p.get('us_frame', float('nan')):.1f} | {p.get('us_total', 'err')} | {p.get('arena', 'err')} |")


if __name__ == "__main__":
    main()
