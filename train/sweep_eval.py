#!/usr/bin/env python3
"""Score sweep checkpoints: fp32 and int8 weights, on synthetic and real VN lines.

  sweep_eval.py --run runs/NAME [--ckpt ckpt_last.pt] [--sets synth-val,real-hand,real-pseudo]

Variants: fp32, and int8 = symmetric per-output-channel int8 weights (exactly the
BN-folded export, see model.fake_quant_weight) with float activations. For a QAT
run the int8 row is the QAT result; for every other run it is post-training
quantisation of the same checkpoint.

Sets (PROVISIONAL until eval/ merges; then use eval/run.py):
  synth-val    the held-out synthetic shard (data/synth, last shard), all lines
  real-hand    hand-transcribed real Vita lines from the research campaign
               (scratch/real/gt.json confident lines, plus gt_real.json lines whose
               text is not already in gt.json), crops as the research cut them
  real-pseudo  data/real lines of held-out test VNs (common.is_test_vn) on which
               PP-OCRv5 server and manga-ocr agree exactly (pseudo-labels, the
               overlay's bright-pixel finder crops)
Dark-text crops are inverted before recognition (--polarity none disables it),
since the v1 recipe only renders light text. Scoring: CER = Levenshtein / GT
length after label_real.canon, full-width ASCII folded to ASCII, dash-like glyphs
folded to ―, whitespace removed. A reduced vocab's <unk> (〓) is always an error.
Writes <run>/eval_<ckpt stem>.json.
"""
import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "synth"))
sys.path.insert(0, str(ROOT / "data"))
from common import DATA, is_test_vn, to_gray  # noqa: E402
from model import from_checkpoint, set_weight_quant  # noqa: E402
from train import MAX_W, STRIDE, edit_distance, greedy  # noqa: E402
from vocab import RESEARCH  # noqa: E402

DASHES = set("—―─‒–－━-‐")


def canon(s):
    from label_real import canon as c
    return c(s)


def score_norm(s):
    s = canon(s)
    out = []
    for ch in s:
        o = ord(ch)
        if 0xFF01 <= o <= 0xFF5E:
            ch = chr(o - 0xFEE0)
        if ch in DASHES:
            ch = "―"
        if not ch.isspace():
            out.append(ch)
    return "".join(out)


def polarity(g):
    """Invert a crop whose text is darker than its background (median grey)."""
    bg = float(np.median(g))
    ink = g[np.abs(g.astype(np.int16) - bg) > 40]
    if ink.size and ink.mean() < bg:
        return 255 - g
    return g


def synth_val(limit=None):
    path = sorted(glob.glob(str(DATA / "synth" / "synth_*.npz")))[-1]
    z = np.load(path)
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    labels = json.loads(str(z["labels"]))
    meta = json.loads(str(z["meta"]))
    n = len(labels) if limit is None else min(limit, len(labels))
    out = []
    for i in range(n):
        h, w = (int(v) for v in shapes[i])
        out.append((data[offs[i]:offs[i + 1]].reshape(h, w), labels[i],
                    {"cat": meta[i]["cat"], "id": i}))
    return out


def real_hand(pol=True):
    items, seen = [], set()
    gt_b = RESEARCH / "gt.json"
    if gt_b.exists():
        for e in json.loads(gt_b.read_text(encoding="utf-8")):
            for k, ln in enumerate(e["lines"]):
                if not ln.get("confident", True):
                    continue
                path = RESEARCH / "lines" / Path(ln["crop"]).name
                rgb = cv2.imread(str(path))[:, :, ::-1]
                g = to_gray(rgb)
                items.append((polarity(g) if pol else g, ln["text"],
                              {"vn": e["vn"], "kind": e["kind"], "id": path.name}))
                seen.add(score_norm(ln["text"]))
    gt_a = RESEARCH / "gt_real.json"
    if gt_a.exists():
        for e in json.loads(gt_a.read_text(encoding="utf-8")):
            for k, text in enumerate(e["lines"]):
                path = RESEARCH / "lines" / f"{e['shot']}_{k}.png"
                if score_norm(text) in seen or not path.exists():
                    continue
                g = to_gray(cv2.imread(str(path))[:, :, ::-1])
                items.append((polarity(g) if pol else g, text,
                              {"vn": e["shot"].split("_")[0], "kind": e.get("notes", ""),
                               "id": path.name}))
                seen.add(score_norm(text))
    return items


def real_pseudo(pol=True):
    items = []
    lines = DATA / "real" / "lines.jsonl"
    for x in lines.read_text(encoding="utf-8").splitlines():
        r = json.loads(x)
        if r["agree"] and r["label"] and is_test_vn(r["vn"]):
            bgr = cv2.imread(str(DATA / "real" / "crops" / f"{r['id']:05d}.png"))
            g = to_gray(bgr[:, :, ::-1])
            items.append((polarity(g) if pol else g, r["label"], {"vn": r["vn"], "id": r["id"]}))
    return items


def prep(g, height):
    h, w = g.shape
    nw = max(8, min(MAX_W, int(round(w * height / h))))
    return cv2.resize(g, (nw, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)


@torch.no_grad()
def predict(model, crops, height, vocab, batch=64):
    gs = [prep(g, height) for g in crops]
    order = np.argsort([g.shape[1] for g in gs])
    out = [None] * len(gs)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        W = (max(gs[i].shape[1] for i in idx) + STRIDE - 1) // STRIDE * STRIDE
        x = np.empty((len(idx), 1, height, W), np.float32)
        for j, i in enumerate(idx):
            w = gs[i].shape[1]
            x[j, 0, :, :w] = gs[i]
            x[j, 0, :, w:] = gs[i][:, -1:]
        in_len = torch.tensor([math.ceil(gs[i].shape[1] / STRIDE) for i in idx])
        for i, seq in zip(idx, greedy(model(torch.from_numpy(x / 127.5 - 1.0)), in_len)):
            out[i] = "".join(vocab[t - 1] for t in seq)
    return out


def score(preds, refs):
    errs = chars = exact = 0
    for p, r in zip(preds, refs):
        p, r = score_norm(p), score_norm(r)
        e = edit_distance(list(p), list(r))
        errs, chars, exact = errs + e, chars + len(r), exact + (e == 0)
    return {"cer": errs / max(chars, 1), "line_acc": exact / max(len(refs), 1),
            "lines": len(refs), "chars": chars, "errors": errs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", default="ckpt_last.pt")
    ap.add_argument("--sets", default="synth-val,real-hand,real-pseudo")
    ap.add_argument("--polarity", choices=("auto", "none"), default="auto")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="first N synth-val lines (tests)")
    args = ap.parse_args()
    from train import default_threads
    torch.set_num_threads(args.threads or default_threads())
    torch.set_flush_denormal(True)
    ck = torch.load(args.run / args.ckpt, map_location="cpu", weights_only=False)
    model, meta = from_checkpoint(ck)
    pol = args.polarity == "auto"
    loaders = {"synth-val": lambda: synth_val(args.limit), "real-hand": lambda: real_hand(pol),
               "real-pseudo": lambda: real_pseudo(pol)}
    sets = {name: loaders[name]() for name in args.sets.split(",")}
    res = {"run": args.run.name, "ckpt": args.ckpt, "step": ck["step"], "width": meta["width"],
           "bottleneck": meta["bottleneck"], "height": meta["height"],
           "classes": len(meta["vocab"]) + 1, "unk": meta["unk"], "qat": ck.get("qat"),
           "polarity": args.polarity, "results": {}}
    for variant, bits in (("fp32", 0), ("int8", 8)):
        set_weight_quant(model, bits)
        for name, items in sets.items():
            t0 = time.time()
            preds = predict(model, [g for g, _, _ in items], meta["height"], meta["vocab"])
            refs = [t for _, t, _ in items]
            r = score(preds, refs)
            r["seconds"] = round(time.time() - t0, 1)
            if name != "synth-val":
                r["lines_detail"] = [{"id": str(m["id"]), "ref": t, "pred": p}
                                     for (_, t, m), p in zip(items, preds)]
            else:
                by = {}
                for (_, t, m), p in zip(items, preds):
                    by.setdefault(m["cat"], ([], []))
                    by[m["cat"]][0].append(p)
                    by[m["cat"]][1].append(t)
                r["by_cat"] = {c: score(*v)["cer"] for c, v in sorted(by.items())}
            res["results"][f"{variant}/{name}"] = r
            print(f"{args.run.name} {variant:4s} {name:11s} CER {r['cer'] * 100:6.2f}% "
                  f"line acc {r['line_acc'] * 100:5.1f}% ({r['lines']} lines, {r['chars']} chars, "
                  f"{r['seconds']}s)", flush=True)
    out = args.run / f"eval_{Path(args.ckpt).stem}{'_limit' if args.limit else ''}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("EVAL_DONE", out, flush=True)


if __name__ == "__main__":
    main()
