#!/usr/bin/env python3
"""Restore a trainable checkpoint from an FL10 fp32 release blob.

The original BN statistics cannot be recovered. Equivalent affine BNs retain
the folded biases; their running statistics must stay frozen during fine-tuning.
The embedded class order is authoritative, including the unknown class.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import export
from model import DS, Ctx, TinyVNOCR


def freeze_bn_stats(model):
    """Freeze running statistics, leaving affine parameters trainable."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def restore(path, *, width=1.0, bottleneck=96, depth=1, ctx=2, ctx_k=5):
    blob = export.read_blob(path)
    if blob["int8"]:
        raise ValueError("restore requires fp32 weights; int8 cannot recover the released fp32 model")
    arch = dict(width=width, bottleneck=bottleneck, depth=depth, ctx=ctx, ctx_k=ctx_k)
    model = TinyVNOCR(blob["n_classes"], **arch).eval()
    expected = export.model_ops(model, blob["height"])
    pairs = []
    for m in model.features:
        if isinstance(m, nn.Sequential):
            pairs.append((m[0], m[1]))
        elif isinstance(m, (DS, Ctx)):
            pairs.extend(((m.dw[0], m.dw[1]), (m.pw[0], m.pw[1])))
    pairs.extend(((model.bottleneck, None), (model.head, None)))
    if len(expected) != len(blob["ops"]):
        raise ValueError("blob architecture does not match the requested model")
    keys = "type kh kw sh sw ph pw cin cout h_in h_out relu res collapse".split()
    with torch.no_grad():
        for i, ((conv, bn), want, op) in enumerate(zip(pairs, expected, blob["ops"])):
            if any(want[k] != op[k] for k in keys):
                raise ValueError(f"op {i} architecture mismatch")
            w = op["w"]
            if op["collapse"]:
                w = w.reshape(op["cout"], op["kh"], op["cin"]).transpose(0, 2, 1)[..., None]
            else:
                w = w.reshape(tuple(conv.weight.shape))
            conv.weight.copy_(torch.from_numpy(np.array(w, copy=True)))
            b = op["bias"]
            if bn is not None:
                bn.running_mean.zero_()
                bn.running_var.fill_(1)
                bn.weight.fill_(np.sqrt(1 + bn.eps))
                bn.bias.copy_(torch.from_numpy(np.array(b, copy=True)))
            elif conv.bias is not None:
                conv.bias.copy_(torch.from_numpy(np.array(b, copy=True)))
            elif b is not None and np.any(b):
                raise ValueError(f"op {i} has a bias the model cannot represent")
    # Verify the inverse layout and folded biases before producing a checkpoint.
    reconstructed = export.model_ops(model, blob["height"])
    max_weight_error = max(float(np.max(np.abs(a["w"] - b["w"])))
                           for a, b in zip(reconstructed, blob["ops"]))
    for a, b in zip(reconstructed, blob["ops"]):
        np.testing.assert_allclose(a["w"], b["w"], rtol=2e-7, atol=1e-7)
        if b["bias"] is not None:
            np.testing.assert_array_equal(a["bias"], b["bias"])
    vocab = blob["vocab"][1:]
    ck = {"model": model.state_dict(), "vocab": vocab, "height": blob["height"], **arch,
          "unk": "〓" if vocab[-1] == "〓" else None, "step": 0, "freeze_bn_stats": True,
          "restored_from": {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                            "max_folded_weight_error": max_weight_error}}
    return model, ck


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blob", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vocab-out", type=Path, required=True)
    args = ap.parse_args()
    _, ck = restore(args.blob)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.vocab_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, args.out)
    args.vocab_out.write_text("\n".join(ck["vocab"]) + "\n", encoding="utf-8")
    print(json.dumps(ck["restored_from"], indent=2))


if __name__ == "__main__":
    main()
