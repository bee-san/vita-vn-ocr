#!/usr/bin/env python3
"""Re-target a TinyVNOCR checkpoint's CTC head to another vocabulary (issue #99: VOCR5-V's smaller head).

  remap_head.py --ckpt IN.pt --vocab NEW_VOCAB.txt --out OUT.pt

Every class of the new vocabulary takes the head row (weight and bias) of the same character in the checkpoint's
vocabulary; a character the checkpoint lacks takes its <unk> row (〓, a reduced vocab's last entry), or the mean
row when it has none. The blank (class 0) keeps its row. Everything else (trunk, bottleneck) is copied unchanged,
"model" and, when present, "model_raw" (EMA and raw weights). The result is an --init for train/train2.py with
--vocab NEW_VOCAB.txt. Prints how many classes were copied and how many started from <unk>.
"""
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "synth"))
from common import load_vocab  # noqa: E402

UNK = "〓"


def remap(sd, old, new):
    """The state dict with head.weight / head.bias re-indexed from vocab `old` to vocab `new` (blank = class 0)."""
    w, b = sd["head.weight"], sd["head.bias"]
    if w.shape[0] != len(old) + 1:
        raise SystemExit(f"head has {w.shape[0]} classes, vocabulary {len(old)} + blank")
    idx = {c: i + 1 for i, c in enumerate(old)}
    unk = idx.get(UNK)
    mean_w, mean_b = w[1:].mean(0, keepdim=True), b[1:].mean()
    rows_w, rows_b, copied, fell = [w[0:1]], [b[0:1]], 0, 0
    for c in new:
        j = idx.get(c)
        if j is not None:
            rows_w.append(w[j:j + 1])
            rows_b.append(b[j:j + 1])
            copied += 1
        elif unk is not None:
            rows_w.append(w[unk:unk + 1])
            rows_b.append(b[unk:unk + 1])
            fell += 1
        else:
            rows_w.append(mean_w)
            rows_b.append(mean_b.view(1))
            fell += 1
    out = dict(sd)
    out["head.weight"] = torch.cat(rows_w).contiguous()
    out["head.bias"] = torch.cat(rows_b).contiguous()
    return out, copied, fell


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    old, new = list(ck["vocab"]), load_vocab(args.vocab)
    for key in ("model", "model_raw"):
        if key in ck:
            ck[key], copied, fell = remap(ck[key], old, new)
    ck["vocab"] = new
    ck["remapped_from"] = {"ckpt": str(args.ckpt), "classes": len(old) + 1}
    for k in ("opt", "optim", "optimizer", "sched", "scheduler"):
        ck.pop(k, None)
    torch.save(ck, args.out)
    print(f"{len(new)} classes: {copied} copied, {fell} from <unk>; -> {args.out}")


if __name__ == "__main__":
    main()
