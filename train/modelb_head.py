#!/usr/bin/env python3
"""Model B (vita-vn-ocr #126): the head gate and the two-level head of a model B blob, in W8F (the PSP's mode).

  modelb_head.py permute --blob B.vocr --rank RANK.json --out P.vocr
      the blob with its head rows and vocab in the rank's frequency order (train/head2level.py permute_blob: every
      class's logit is the same, bit for bit); RANK.json from data/modelb_pools.py rank
  modelb_head.py features --blob P.vocr --sources GLOB [GLOB ...] --lines N --out F.npz
      per frame of N training-side console crops (a fixed sample: numpy default_rng(20261026) over the sources'
      rows, sorted) from the C runtime in W8F: the head's input (vocr_extra.head_in), the full W8F head's argmax and,
      for level 1 (the blank and the 503 most frequent classes: L1 = 504 rows), its largest non-blank logit l1 and
      its blank logit b0 from the same W8F logits; the row's cluster (hooked: the title id; else the VNDB id) and the
      calibration split (sha1(cluster) % 5 == 0, train/headgate.is_cal)
  modelb_head.py fit --blob P.vocr --features F.npz --out DIR [--eps 1e-3] [--recall 0.999]
      the gate (train/headgate.py fit_float, 16 hidden units; label: the W8F argmax is the blank; threshold: at most
      floor(eps n) of the n calibration frames with a non-blank argmax score above it) and the escape (head2level's
      fit_escape on [head input, l1, b0]; label: the W8F argmax lies outside level 1; threshold: recall >= `recall`
      on the calibration positives, head2level.threshold_for), both on the fit split; writes DIR/gated.vocr (the gate
      alone), DIR/gated2.vocr (the gate and the VOC2 trailer) and DIR/fit.json; checks on the calibration lines that
      the C runtime equals the twins bit for bit (gate scores in W8F; the W8F two-level head; both together)

Training-side data only. No development, test or protected row is read (asserted against eval/manifests/reserved.json
before any feature is extracted).
"""
import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (HERE, ROOT / "runtime", ROOT / "runtime" / "tests", ROOT / "data", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import head2level as H2  # noqa: E402
import headgate as HG  # noqa: E402

F32 = np.float32
L1_ROWS = 504                     # the protocol: the blank and the 503 most frequent classes (63 head blocks)
SAMPLE_SEED = 20261026


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def perm_from_rank(vocab, order):
    """perm (new class -> old class) for a blob vocab (class 0 = the blank) from a rank's character order."""
    pos = {c: i for i, c in enumerate(vocab)}
    if len(pos) != len(vocab) or set(order) != set(vocab[1:]):
        raise SystemExit("the rank's order is not this blob's vocab")
    return np.array([0] + [pos[c] for c in order], np.int64)


def cmd_permute(a):
    import export
    data = Path(a.blob).read_bytes()
    vocab = export.read_blob(data)["vocab"]
    rank = json.loads(Path(a.rank).read_text(encoding="utf-8"))
    perm = perm_from_rank(vocab, rank["order"])
    out = H2.permute_blob(HG.base_of(data), perm)
    Path(a.out).write_bytes(out)
    print(json.dumps({"out": a.out, "sha256": sha256(a.out), "bytes": len(out), "rank_sha256": sha256(a.rank)}))


def hygiene(clusters_vn):
    r = json.loads((ROOT / "eval" / "manifests" / "reserved.json").read_text(encoding="utf-8"))
    bad = {v for v in clusters_vn if v in set(r["test"]) or v in set(r["dev_sample"])}
    if bad:
        raise SystemExit(f"training rows of reserved / dev_sample VNs: {sorted(bad)[:10]}")


def cmd_features(a):
    from pyvocr import Vocr

    import modelb_pools as MP
    t0 = time.time()
    files = sorted({f for g in a.sources for f in glob.glob(g)})
    index = []
    for fi, f in enumerate(files):
        _, labels, meta = MP.load_shard(f)
        for i, m in enumerate(meta):
            index.append((fi, i, m.get("title_id") or m["vn"], m.get("vn")))
    hygiene({v for *_, v in index if v})
    rng = np.random.default_rng(SAMPLE_SEED)
    pick = sorted(rng.choice(len(index), min(a.lines, len(index)), replace=False).tolist())
    rt = Vocr(a.blob, mode="w8f", max_width=1024)
    cache, xs, am, l1, b0, line_of, cl = {}, [], [], [], [], [], []
    for n, j in enumerate(pick):
        fi, i, cluster, _ = index[j]
        if fi not in cache:
            cache.clear()
            cache[fi] = MP.load_shard(files[fi])[0]
        g = cache[fi][i]
        _, lg = rt.run(g, logits=True, head_in=True)
        xs.append(rt.last_head_in.astype(F32))
        am.append(lg.argmax(1).astype(np.int32))
        l1.append(lg[:, 1:L1_ROWS].max(1).astype(F32))
        b0.append(lg[:, 0].astype(F32))
        line_of.append(np.full(len(lg), n, np.int32))
        cl.append(cluster)
    off = np.zeros(len(pick) + 1, np.int64)
    off[1:] = np.cumsum([len(v) for v in am])
    cal_line = np.array([HG.is_cal(c) for c in cl])
    np.savez(a.out, x=np.concatenate(xs), am=np.concatenate(am), l1=np.concatenate(l1), b0=np.concatenate(b0),
             off=off, cal_line=cal_line, cluster=np.array(cl), file_index=np.array([index[j][0] for j in pick]),
             row=np.array([index[j][1] for j in pick]), files=np.array(files))
    info = {"blob": a.blob, "blob_sha256": sha256(a.blob), "files": {f: sha256(f) for f in files},
            "rows": len(index), "lines": len(pick), "frames": int(off[-1]), "clusters": len(set(cl)),
            "cal_lines": int(cal_line.sum()), "seconds": round(time.time() - t0, 1)}
    Path(str(a.out) + ".json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "files"}), flush=True)


def cmd_fit(a):
    from pyvocr import Vocr

    import modelb_pools as MP
    t0 = time.time()
    z = np.load(a.features)
    info = json.loads(Path(str(a.features) + ".json").read_text())
    base = Path(a.blob).read_bytes()
    if sha256(a.blob) != info["blob_sha256"]:
        raise SystemExit("the features come from another blob")
    x, am, l1, b0, off = z["x"], z["am"], z["l1"], z["b0"], z["off"]
    line = np.repeat(np.arange(len(off) - 1), np.diff(off))
    cal = z["cal_line"][line]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # the gate
    yb = am == 0
    g = HG.fit_float(x[~cal], yb[~cal])
    gp = HG.quantise(g["w1"], g["b1"], g["w2"], g["b2"])
    gs = HG.gate_scores(gp, x)
    gt, gm = HG.threshold(gs[cal & ~yb], a.eps)
    skip = gs > gt
    # the escape
    ye = am >= L1_ROWS
    X = np.concatenate([x, l1[:, None], b0[:, None]], 1).astype(F32)
    f = H2.fit_escape(X[~cal], ye[~cal], seed=0, threads=5)
    esc = H2.export_escape(f, K=x.shape[1])
    q, sx = H2.quant16_exact(x[cal])
    es = H2.escape_score(q, sx, l1[cal], b0[cal], esc)
    et, em = H2.threshold_for(es[ye[cal]], a.recall)
    esc["t"] = et
    fire = es > et
    base0 = HG.base_of(base)
    gated = HG.append_gate(base0, gp, gt)
    gated2 = H2.attach(gated, H2.write_trailer(base0, L1_ROWS // 8, esc))
    (out / "gated.vocr").write_bytes(gated)
    (out / "gated2.vocr").write_bytes(gated2)
    kept_cal = ~skip[cal]
    rec = {"blob": a.blob, "blob_sha256": info["blob_sha256"], "features": a.features,
           "features_sha256": sha256(a.features), "L1": L1_ROWS, "eps": a.eps, "recall": a.recall,
           "gate": {"t": float(gt), "allowed": int(gm), "fit_frames": int((~cal).sum()), "cal_frames": int(cal.sum()),
                    "blank_share_cal": round(float(yb[cal].mean()), 5),
                    "skip_share_cal": round(float(skip[cal].mean()), 5),
                    "nonblank_skipped_cal": int((skip & cal & ~yb).sum()), "loss": round(g["loss"], 6)},
           "escape": {"t": float(et), "max_missed": int(em), "fit_positives": int(ye[~cal].sum()),
                      "cal_positives": int(ye[cal].sum()), "fire_share_cal": round(float(fire.mean()), 5),
                      "fire_share_of_kept_cal": round(float(fire[kept_cal].mean()), 5) if kept_cal.any() else None,
                      "missed_cal": int((~fire & ye[cal]).sum()), "pos_weight": round(f["pos_weight"], 4)},
           "blobs": {"gated": {"sha256": sha256(out / "gated.vocr"), "bytes": len(gated)},
                     "gated2": {"sha256": sha256(out / "gated2.vocr"), "bytes": len(gated2)}}}
    # the C runtime against the twins on the calibration lines (W8F)
    files = [str(s) for s in z["files"]]
    cal_lines = np.nonzero(z["cal_line"])[0]
    rt_s = Vocr(gated, mode="w8f", max_width=1024, head_gate="score")
    rt_2 = Vocr(gated2, mode="w8f", max_width=1024, two_level=1)
    rt_b = Vocr(gated2, mode="w8f", max_width=1024, two_level=1, head_gate="on")
    vocab = [rt_b.class_text(c) for c in range(rt_b.n_classes)]
    head = H2.head_arrays(base0)
    two = H2.read_trailer(gated2)
    cache, ok = {}, {"gate_scores": True, "two_level": True, "both": True}
    for n in cal_lines[:a.check_lines]:
        fi, ri = int(z["file_index"][n]), int(z["row"][n])
        if fi not in cache:
            cache.clear()
            cache[fi] = MP.load_shard(files[fi])[0]
        g_img = cache[fi][ri]
        o = rt_s.run_topk(g_img, 8, head_in=True, gate_score=True)
        ok["gate_scores"] &= bool(np.array_equal(o["gate_score"].view(np.uint32),
                                                 HG.gate_scores(gp, o["head_in"]).view(np.uint32)))
        o2 = rt_2.run_topk(g_img, 8, head_in=True, esc=True)
        tw = H2.twolevel_exact(o2["head_in"], head, two, topk=8, mode="w8f")
        ok["two_level"] &= bool(np.array_equal(o2["esc"].view(np.uint32), tw["score"].view(np.uint32)) and
                                np.array_equal(o2["best"], tw["best"]) and np.array_equal(o2["topc"], tw["topc"]))
        sk = HG.gate_scores(gp, o2["head_in"]) > gt
        want = HG.simulate(o2, sk, vocab)
        ob = rt_b.run_topk(g_img, 8)
        ok["both"] &= bool(ob["text"] == want["text"] and np.array_equal(ob["best"], want["best"]) and
                           ob["gated"] == int(sk.sum()))
    rec["c_equals_twin"] = dict(ok, lines=int(min(len(cal_lines), a.check_lines)))
    rec["seconds"] = round(time.time() - t0, 1)
    (out / "fit.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec), flush=True)
    if not all(ok.values()):
        raise SystemExit(f"C runtime != twin: {ok}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("permute")
    p.add_argument("--blob", required=True)
    p.add_argument("--rank", required=True)
    p.add_argument("--out", required=True)
    f = sub.add_parser("features")
    f.add_argument("--blob", required=True)
    f.add_argument("--sources", nargs="+", required=True)
    f.add_argument("--lines", type=int, default=12000)
    f.add_argument("--out", required=True)
    t = sub.add_parser("fit")
    t.add_argument("--blob", required=True)
    t.add_argument("--features", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--eps", type=float, default=1e-3)
    t.add_argument("--recall", type=float, default=0.999)
    t.add_argument("--check-lines", type=int, default=400)
    a = ap.parse_args()
    {"permute": cmd_permute, "features": cmd_features, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
