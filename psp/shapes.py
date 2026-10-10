#!/usr/bin/env python3
"""PSP cost of candidate model shapes: blob, arena and PPSSPP-emulated time per CTC frame.

  shapes.py --out DIR --pack LINES.bin [--model-py PATH] [--eboot E.PBP ...] [--modes f32,w8f]

Each shape (the SHAPES list: arch's knobs, docs/arch-sweep.md on the arch branch) is built
random-initialised, exported with train/export.py (fp32 and int8), and run by the PSP test app
in PPSSPPHeadless on the crops of LINES.bin. Weights do not change the work the kernels do,
so the time is the shape's (the few data-dependent branches - ReLU, argmax, CTC - are noise).
PPSSPP's emulated time is deterministic (it counts emulated cycles, not host time), and it
is not console time: see psp/README.md for the conversion to a console estimate.

--model-py: a train/model.py whose TinyVNOCR takes width, bottleneck, depth, ctx, ctx_k (the
arch branch's); with main's model.py only width / bottleneck / vocab / height shapes run.
Writes DIR/shapes.json and prints a markdown table.
"""
import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "train"), str(ROOT / "runtime"), str(ROOT / "synth")]

# name, knobs (defaults: v1). vocab = output classes including the CTC blank.
SHAPES = [
    ("v1", {}),
    ("w0.5", {"width": 0.5}),
    ("w0.75", {"width": 0.75}),
    ("w1.5", {"width": 1.5}),
    ("w2.0", {"width": 2.0}),
    ("b64", {"bottleneck": 64}),
    ("b48", {"bottleneck": 48}),
    ("v3500", {"vocab": 3501}),
    ("v2500", {"vocab": 2501}),
    ("h24", {"height": 24}),
    ("d2", {"depth": 2}),
    ("ctx4", {"ctx": 4}),
    ("lean: v3500 b64", {"vocab": 3501, "bottleneck": 64}),
    ("lean: v3500 b64 h24", {"vocab": 3501, "bottleneck": 64, "height": 24}),
]
DEFAULT = {"width": 1.0, "bottleneck": 96, "depth": 1, "ctx": 2, "ctx_k": 5, "vocab": 5355, "height": 32}


def load_model_module(path):
    spec = importlib.util.spec_from_file_location("model", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model"] = mod          # export.model_ops imports DS, Ctx from "model"
    spec.loader.exec_module(mod)
    return mod


def build(mod, knobs, seed=0):
    import torch
    k = {**DEFAULT, **knobs}
    torch.manual_seed(seed)
    kw = {"width": k["width"], "bottleneck": k["bottleneck"]}
    for extra in ("depth", "ctx", "ctx_k"):
        if extra in knobs:
            kw[extra] = k[extra]
    try:
        m = mod.TinyVNOCR(k["vocab"], **kw)
    except TypeError:
        return None, k              # this model.py has no such knob (pass --model-py)
    return m.eval(), k


def macs_per_frame(ops, stride=8):
    import export as E
    tot, s = 0.0, 1
    for op in ops:
        s *= op["sw"]
        cols = stride / s                  # output columns per CTC frame
        if op["type"] in (E.T_CONV, E.T_DW):
            tot += op["h_out"] * op["cout"] * op["kh"] * op["kw"] * cols
        else:
            rows = 1 if op["collapse"] else op["h_in"]
            k = op["w"].shape[1]
            tot += rows * k * op["cout"] * cols
    return int(tot)


def run_psp(eboot, blob, mode, pack, tile, head):
    cmd = [sys.executable, str(HERE / "ppsspp_cli.py"), str(blob), "--mode", mode, "--pack", str(pack),
           "--repeat", "1", "--tile", str(tile), "--head", str(head), "--eboot", str(eboot)]
    p = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=7200)
    if p.returncode:
        return {"error": p.stderr[-500:]}
    m = re.search(r"(\d+) lines, (\d+) frames, ([\d.]+) ms/line, ([\d.]+) us/frame", p.stderr)
    a = re.search(r"arena (\d+) bytes", p.stderr)
    return {"lines": int(m.group(1)), "frames": int(m.group(2)), "ms_line": float(m.group(3)),
            "us_frame": float(m.group(4)), "arena": int(a.group(1))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--model-py", default=str(ROOT / "train" / "model.py"))
    ap.add_argument("--eboot", action="append", help="NAME=PATH (default scalar=psp/EBOOT.PBP)")
    ap.add_argument("--modes", default="f32,w8f")
    ap.add_argument("--tiles", default="4/16,16/16")
    ap.add_argument("--only", help="comma list of shape names")
    a = ap.parse_args()
    from pyvocr import Vocr

    import export as E
    from common import load_vocab
    mod = load_model_module(a.model_py)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eboots = dict(e.split("=", 1) for e in (a.eboot or [f"scalar={HERE / 'EBOOT.PBP'}"]))
    vocab_all = load_vocab(ROOT / "data" / "vocab.txt")
    rows = []
    for name, knobs in SHAPES:
        if a.only and name not in a.only.split(","):
            continue
        model, k = build(mod, knobs)
        if model is None:
            print(f"skipped {name}: {a.model_py} has no knob for {knobs}", flush=True)
            continue
        vocab = vocab_all[:k["vocab"] - 1]
        ops = E.model_ops(model, k["height"])
        r = {"name": name, "knobs": knobs, "params": sum(p.numel() for p in model.parameters()),
             "macs_per_frame": macs_per_frame(ops)}
        slug = re.sub(r"[^a-z0-9.]+", "_", name.lower())
        for key, int8 in (("f32", False), ("w8", True)):
            data, _ = E.build_blob(ops, vocab, k["height"], int8, 1024, b"shape:" + slug.encode())
            path = out / f"{slug}_{key}.vocr"
            path.write_bytes(data)
            r[f"blob_{key}"] = len(data)
        r["arena"] = {}
        for mode in ("f32", "w8f", "w8i"):
            blob = out / f"{slug}_{'f32' if mode == 'f32' else 'w8'}.vocr"
            for t in a.tiles.split(","):
                tile, head = (int(v) for v in t.split("/"))
                try:
                    r["arena"][f"{mode} {t}"] = Vocr(blob, mode=mode, tile=tile, head=head).arena_bytes
                except Exception as e:  # noqa: BLE001 - w8i has a reduction-length limit
                    r["arena"][f"{mode} {t}"] = f"n/a ({e})"
        r["psp"] = {}
        for ename, eboot in eboots.items():
            for mode in a.modes.split(","):
                blob = out / f"{slug}_{'f32' if mode == 'f32' else 'w8'}.vocr"
                for t in a.tiles.split(","):
                    tile, head = (int(v) for v in t.split("/"))
                    r["psp"][f"{ename} {mode} {t}"] = run_psp(eboot, blob, mode, a.pack, tile, head)
        rows.append(r)
        print(json.dumps(r), flush=True)
        (out / "shapes.json").write_text(json.dumps(rows, indent=1))
    keys = sorted({k for r in rows for k in r["psp"]})
    print("\n| shape | params | MACs/frame | int8 blob B | fp32 blob B | " +
          " | ".join(f"{k} us/frame" for k in keys) + " |")
    print("|" + "---|" * (5 + len(keys)))
    for r in rows:
        cells = [f"{r['psp'][k].get('us_frame', 'err'):.0f}" if "us_frame" in r["psp"][k] else "err" for k in keys]
        print(f"| {r['name']} | {r['params']:,} | {r['macs_per_frame']:,} | {r['blob_w8']:,} | {r['blob_f32']:,} | "
              + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
