#!/usr/bin/env python3
"""vocr6_attn_cost.py: sizes, arena and A9 time estimates of the attention op (docs/vocr6-attn.md, "Cost").

  vocr6_attn_cost.py sizes --out SIZES.json                 parameters, int8 bytes, MACs per frame and the exact
                                                            W8I / W8F / F32 arena at 256 / 640 / 1,024 px of VOCR5-S/N/L
                                                            with and without the op (dim 96, 2 heads), also with the
                                                            adopted head gate (S) and gate + two-level head (N, L)
  vocr6_attn_cost.py blobs --out DIR                        the a9 measurement blobs: per tier the adopted speed blob
                                                            rebuilt from its checkpoint (checked byte for byte against
                                                            the released one) and the same with the op at gate 0
  vocr6_attn_cost.py time --a9 DIR --pack PACK --out T.json the A9 estimates from vita/a9_layers.py reports of those
                                                            blobs (DIR/{nopld,pld}_{S,N,L}_{ref,attn}/report.json)

The op at gate 0 (train/model.py GlobalAttn as a fine-tune from VOCR5 starts) makes z' = z exactly, so the head gate
and the two-level escape of the released blobs, which read the head's input, skip and fire on exactly the frames they
do today: the measured difference is the op's own cost (its projections, scores, softmax and weighted sums, the copy
of z' into the head's batches), nothing else. The op's arithmetic does not depend on its weights' values.

Time is ocr-speed's method (docs/vocr5.md, docs/headgate.md): vita/a9_layers.py instruction counts of the VitaSDK
-O2 NEON build under qemu-arm with its cache model, on ocr-speed's 24 development lines of 60-100 frames, top-8, the
high bound times 1.000676 (VOCR3-H15's measured console rate at an assumed 333 MHz); 444 MHz by the clock ratio. The
attention's scores and weighted sums cost T per frame on a line of T frames, so its core (phase 103) is projected
from the pack's lines to the nominal box's 3 lines of 80 frames by c = core cycles / sum(T^2) over the pack's lines.
The nominal box is 240 frames plus 0.1 s for the finder (docs/vocr5.md). Every millisecond here is an estimate.
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT / "train", ROOT / "vita", ROOT / "runtime", ROOT / "runtime" / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import export as E  # noqa: E402

HERD = Path("/local/home/skerraut/herd-vita-ocr")
A = HERD / "shared/artifacts"
V5 = A / "vocr5-train"
# tier: (the released run, the adopted speed blob, its runtime options; docs/headgate.md, docs/head2level.md)
TIERS = {
    "S": ("v5f_S_seed2", A / "head-gate/blobs/VOCR5_S_w8g.vocr", "--head-gate on"),
    "N": ("v5f_N_seed3", A / "head-2level/blobs/VOCR5_N_2Lg_K500_r96.vocr", "--two-level --head-gate on"),
    "L": ("v5f_L_seed2", A / "head-2level/blobs/VOCR5_L_2Lg_K500_r96.vocr", "--two-level --head-gate on"),
}
RELEASE = V5 / "release/v0.3.0"
DIM, HEADS = 96, 2
WIDTHS = (256, 640, 1024)
SCALE, MHZ = 1.000676, 333
NOMINAL = {"lines": 3, "frames": 80, "finder_ms": 100.0}


def sha256(b):
    return hashlib.sha256(b if isinstance(b, (bytes, bytearray)) else Path(b).read_bytes()).hexdigest()


def models(tier, attn_seed=0):
    """(VOCR5 tier model, the same with the attention op at gate 0, vocab, height, checkpoint path)."""
    import torch

    from model import TinyVNOCR, load_state
    ck_path = V5 / "runs" / TIERS[tier][0] / "ckpt_step15000.pt"
    base, vocab, height = E.load_checkpoint(ck_path)
    kw = {k: v for k, v in base.arch.items() if k not in ("rep",)}
    torch.manual_seed(attn_seed)
    attn = TinyVNOCR(base.head.out_channels, **kw, attn=1, attn_dim=DIM, attn_heads=HEADS).eval()
    assert load_state(attn, base.state_dict()) == "fresh-attn"
    return base, attn, vocab, height, ck_path


def blob(model, vocab, height, int8, ck_path):
    return E.build_blob(E.model_ops(model, height), vocab, height, int8, None,
                        hashlib.sha256(Path(ck_path).read_bytes()).digest())[0]


def adopted_like(base_v1, adopted):
    """The adopted speed blob's construction (head-gate's VOCG section; with a VOC2 trailer, head2level's row
    permutation and trailer) applied to another version-1 blob `base_v1` of the same model's head."""
    import head2level as H2
    import headgate as HG
    data = Path(adopted).read_bytes()
    two = H2.read_trailer(data)
    out = bytes(base_v1)
    if two is not None:                        # the head rows and vocab in the adopted blob's order
        new_vocab = E.read_blob(data)["vocab"]
        old_vocab = E.read_blob(out)["vocab"]
        pos = {c: i for i, c in enumerate(old_vocab)}
        out = H2.permute_blob(out, np.array([pos[c] for c in new_vocab], np.int64))
    g = HG.read_gate(data)
    params = {k: g[k] for k in ("q", "scale", "bias", "w2", "b2")}
    out = HG.append_gate(out, params, g["t"])
    if two is not None:
        out = H2.attach(out, H2.write_trailer(HG.base_of(out), two["l1_blocks"], two["esc"]))
    return out


def arena_rows(data, options):
    """Exact arena bytes (train/export.py plan_arena, = vocr_arena_bytes) per mode and width."""
    import head2level as H2
    b = E.read_blob(data)
    two = H2.read_trailer(data)
    rows = {}
    for name, mode, hg, tl in options:
        if (tl and two is None) or (hg and b["gate"] is None):
            continue
        rows[name] = {w: E.plan_arena(b, w, 0, 0, mode, hg, two if tl else None) for w in WIDTHS}
    return rows


def cmd_sizes(args):
    out = {"kind": "vocr6-attn-sizes", "attn": {"dim": DIM, "heads": HEADS}, "widths": WIDTHS, "tiers": {}}
    for tier in TIERS:
        base, attn, vocab, height, ck = models(tier)
        rec = {"run": TIERS[tier][0]}
        for name, m in (("today", base), ("attn", attn)):
            f32, w8 = blob(m, vocab, height, False, ck), blob(m, vocab, height, True, ck)
            b8 = E.read_blob(w8)
            sp = adopted_like(w8, TIERS[tier][1])
            opts = [("f32", E.MODE_F32, 0, 0)]
            r = {"train_params": sum(p.numel() for p in m.parameters()), "blob_params": b8["params"],
                 "blob_f32_bytes": len(f32), "blob_w8_bytes": len(w8), "speed_blob_bytes": len(sp),
                 "macs_per_frame": E.macs_per_frame(b8)[0],
                 "macs_per_frame_T80": E.macs_per_frame(b8, NOMINAL["frames"])[0],
                 "arena": {**arena_rows(f32, opts), **arena_rows(w8, [("w8f", E.MODE_W8F, 0, 0),
                                                                     ("w8i", E.MODE_W8I, 0, 0)]),
                           **arena_rows(sp, [("w8i+gate", E.MODE_W8I, 1, 0), ("w8i+gate+2l", E.MODE_W8I, 1, 1)])}}
            if name == "today":
                r["release_identical"] = w8 == (RELEASE / f"VOCR5_{tier}_w8.vocr").read_bytes()
                r["speed_identical"] = sp == Path(TIERS[tier][1]).read_bytes()
            rec[name] = r
        t, a = rec["today"], rec["attn"]
        rec["delta"] = {"train_params": a["train_params"] - t["train_params"],
                        "blob_params": a["blob_params"] - t["blob_params"],
                        "blob_w8_bytes": a["blob_w8_bytes"] - t["blob_w8_bytes"],
                        "blob_f32_bytes": a["blob_f32_bytes"] - t["blob_f32_bytes"],
                        "macs_per_frame": a["macs_per_frame"] - t["macs_per_frame"],
                        "macs_per_frame_T80": a["macs_per_frame_T80"] - t["macs_per_frame_T80"],
                        "arena": {k: {w: a["arena"][k][w] - t["arena"][k][w] for w in WIDTHS} for k in t["arena"]}}
        out["tiers"][tier] = rec
        print(tier, json.dumps(rec["delta"]), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")


def cmd_blobs(args):
    d = Path(args.out)
    d.mkdir(parents=True, exist_ok=True)
    rec = {}
    for tier in TIERS:
        base, attn, vocab, height, ck = models(tier)
        ref = adopted_like(blob(base, vocab, height, True, ck), TIERS[tier][1])
        if ref != Path(TIERS[tier][1]).read_bytes():
            raise SystemExit(f"{tier}: the rebuilt adopted blob differs from {TIERS[tier][1]}")
        att = adopted_like(blob(attn, vocab, height, True, ck), TIERS[tier][1])
        for name, data in (("ref", ref), ("attn", att)):
            (d / f"{tier}_{name}.vocr").write_bytes(data)
        rec[tier] = {"ref_sha256": sha256(ref), "attn_sha256": sha256(att), "cli": f"--topk 8 {TIERS[tier][2]}",
                     "adopted": str(TIERS[tier][1])}
        print(tier, rec[tier], flush=True)
    (d / "blobs.json").write_text(json.dumps(rec, indent=1) + "\n")


def pack_frames(pack, lo=60, hi=100, limit=24):
    """The CTC frame counts of the pack's lines a9_layers.py measures (its subset_pack's selection)."""
    import dev_pack
    data = Path(pack).read_bytes()
    n, off, out = struct.unpack_from("<I", data, 0)[0], 4, []
    for _ in range(n):
        h, w, ch = struct.unpack_from("<III", data, off)
        fr = dev_pack.frames(w, h)
        if lo <= fr <= hi and len(out) < limit:
            out.append(fr)
        off += 12 + h * w * ch
    return out


def cmd_time(args):
    frames_l = pack_frames(args.pack)
    sT, sT2 = sum(frames_l), sum(t * t for t in frames_l)
    nominal_frames = NOMINAL["lines"] * NOMINAL["frames"]
    nominal_T2 = NOMINAL["lines"] * NOMINAL["frames"] ** 2
    ms = lambda cyc: cyc * SCALE / (MHZ * 1000)  # noqa: E731
    out = {"kind": "vocr6-attn-time", "estimates_are_not_measurements": True, "scale": SCALE, "mhz": MHZ,
           "pack": {"sha256": sha256(args.pack), "lines": len(frames_l), "frames": sT, "sum_T2": sT2},
           "nominal_box": NOMINAL, "tiers": {}}
    for basis in ("nopld", "pld"):
        for tier in TIERS:
            ph = {}
            for name in ("ref", "attn"):
                p = Path(args.a9) / f"{basis}_{tier}_{name}" / "report.json"
                if not p.exists():
                    continue
                r = json.loads(p.read_text())
                if r["frames"] != sT:
                    raise SystemExit(f"{p}: {r['frames']} frames, the pack's selection has {sT}")
                ph[name] = {int(k): v["high"] for k, v in r["phases"].items()}
                ph[name + "_meta"] = r["meta"]
            if "ref" not in ph or "attn" not in ph:
                continue
            ref, att = ph["ref"], ph["attn"]
            n_ops = len(E.read_blob(Path(ph["attn_meta"]["blob_path"]).read_bytes())["ops"])
            proj, core = att.get(n_ops - 2, 0.0), att.get(103, 0.0)
            tot_r, tot_a = sum(ref.values()), sum(att.values())
            other = (tot_a - tot_r) - proj - core       # everything else that moved: the head, the copy, the cache
            c = core / sT2                               # core cycles per T^2
            per_frame = {"ref": ms(tot_r) / sT, "attn": ms(tot_a) / sT, "attn_proj": ms(proj) / sT,
                         "attn_core_pack": ms(core) / sT, "attn_core_T80": ms(c) * NOMINAL["frames"],
                         "other": ms(other) / sT}
            box_ref = nominal_frames * per_frame["ref"] + NOMINAL["finder_ms"]
            delta = nominal_frames * (per_frame["attn_proj"] + per_frame["other"]) + ms(c) * nominal_T2
            texts = [ph[k]["runs"]["prof"]["texts_sha256"] for k in ("ref_meta", "attn_meta")]
            same_text = texts[0] == texts[1]
            rec = {"per_frame_ms_333": {k: round(v, 4) for k, v in per_frame.items()},
                   "nominal_box_ms_333": {"ref": round(box_ref, 1), "attn": round(box_ref + delta, 1),
                                          "delta": round(delta, 1)},
                   "nominal_box_ms_444": {"ref": round(box_ref * MHZ / 444, 1),
                                          "attn": round((box_ref + delta) * MHZ / 444, 1),
                                          "delta": round(delta * MHZ / 444, 1)},
                   "pack_texts_identical": same_text,
                   "blobs": {k: ph[k + "_meta"]["blob_sha256"] for k in ("ref", "attn")},
                   "cli_opts": ph["attn_meta"]["cli_opts"], "vocr_c_sha256": ph["attn_meta"]["vocr_c_sha256"]}
            out["tiers"].setdefault(tier, {})[basis] = rec
            print(basis, tier, json.dumps(rec["per_frame_ms_333"]), json.dumps(rec["nominal_box_ms_333"]),
                  "texts identical" if same_text else "TEXTS DIFFER", flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sizes")
    p.add_argument("--out", required=True)
    p = sub.add_parser("blobs")
    p.add_argument("--out", required=True)
    p = sub.add_parser("time")
    p.add_argument("--a9", required=True)
    p.add_argument("--pack", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"sizes": cmd_sizes, "blobs": cmd_blobs, "time": cmd_time}[a.cmd](a)


if __name__ == "__main__":
    main()
