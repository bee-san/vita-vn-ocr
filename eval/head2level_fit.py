#!/usr/bin/env python3
"""head2level: the escapes, K_alt and the two-level blobs (eval/head2level_protocol.json "frequent_set",
"escape_training", "r48_fine_tune").

  head2level_fit.py escapes --tier N --train PRIV/train_N.npz --blob PERM.vocr --out OUT/fit_N.json
      one r96 escape per K of the grid (K = 1,500 and K_alt's candidates), fit on the training split, thresholded on
      the calibration split with the exact twin (recall >= 0.999; the 0.9999 fallback recorded), W(K), K_alt
  head2level_fit.py build --tier N --blob PERM.vocr --fit OUT/fit_N.json --K 1500 [--rare R48.npz] --out B.vocr
      the permuted blob + its VOC2 trailer

Training-side data only (head2level_extract.py train). Nothing here reads development or test data.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT / "runtime", ROOT / "train", ROOT / "runtime" / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import head2level as H2  # noqa: E402

F32 = np.float32
KALT_GRID = [500, 750, 1000, 1250, 1750, 2000, 2500]
ESC_MACS_ROWS = 16.5          # (96 x 16 + 2 x 16 + 16) MACs / 96 per head row


def l1_rows(K):
    return 8 * ((K + 1 + 7) // 8)


def load_train(path):
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    off = z["off"]
    line_cal = np.array(meta["cal"], bool)
    cal = np.repeat(line_cal, np.diff(off))
    return dict(x=z["x"], am=z["am"].astype(np.int64), b0=z["b0"], l1=z["l1"], kgrid=[int(k) for k in z["kgrid"]],
                cal=cal, meta=meta, off=off)


def esc_to_json(e):
    return {k: (np.asarray(v).tolist() if np.ndim(v) else float(v)) for k, v in e.items()}


def esc_from_json(d):
    e = {k: np.asarray(v, F32) for k, v in d.items()}
    e["q"] = np.asarray(d["q"], np.int8)
    e["b2"], e["t"] = F32(d["b2"]), F32(d["t"])
    return e


def fit_one(tr, K, labels=None, threads=5):
    """Fit, export and threshold one escape at K (labels: y per frame, default the r96 one: argmax outside L1)."""
    L1 = l1_rows(K)
    ki = tr["kgrid"].index(K)
    y = tr["am"] >= L1 if labels is None else np.asarray(labels, bool)
    fit, cal = ~tr["cal"], tr["cal"]
    npos_fit, npos_cal = int(y[fit].sum()), int(y[cal].sum())
    X = np.concatenate([tr["x"], tr["l1"][:, ki:ki + 1], tr["b0"][:, None]], 1).astype(F32)
    t0 = time.time()
    f = H2.fit_escape(X[fit], y[fit], seed=0, threads=threads)
    esc = H2.export_escape(f, K=tr["x"].shape[1])
    xc = tr["x"][cal]
    q, sx = H2.quant16_exact(xc)
    s = H2.escape_score(q, sx, tr["l1"][cal, ki], tr["b0"][cal], esc)
    out = {"K": K, "L1": L1, "fit_frames": int(fit.sum()), "cal_frames": int(cal.sum()), "fit_positives": npos_fit,
           "cal_positives": npos_cal, "pos_weight": round(f["pos_weight"], 4), "fit_loss": round(f["loss"], 6),
           "seconds": round(time.time() - t0, 1), "thresholds": {}}
    for name, rc in (("r0.999", 0.999), ("r0.9999", 0.9999)):
        t, m = H2.threshold_for(s[y[cal]], rc)
        fire = s > t
        out["thresholds"][name] = {"t": float(t), "max_missed": int(m), "missed": int((~fire & y[cal]).sum()),
                                   "cal_recall": round(float(fire[y[cal]].mean()), 6),
                                   "cal_fire_share": round(float(fire.mean()), 6)}
    esc["t"] = F32(out["thresholds"]["r0.999"]["t"])
    out["esc"] = esc_to_json(esc)
    return out


def cmd_escapes(a):
    tr = load_train(a.train)
    head = H2.head_arrays(Path(a.blob).read_bytes())
    V, Vpad = head["V"], head["P"]
    res = {"kind": "head2level-escapes", "tier": a.tier, "train": a.train, "train_sha256": H2.sha256(a.train),
           "blob": a.blob, "blob_sha256": H2.sha256(a.blob), "classes": V, "cout_pad": Vpad,
           "frames": int(len(tr["am"])), "cal_frames": int(tr["cal"].sum()), "fits": {}}
    for K in [1500] + KALT_GRID:
        o = fit_one(tr, K, threads=a.threads)
        L1, fc = o["L1"], o["thresholds"]["r0.999"]["cal_fire_share"]
        o["eligible"] = o["fit_positives"] >= 200 and o["cal_positives"] >= 50
        o["W"] = round(L1 + ESC_MACS_ROWS + fc * (Vpad - L1), 3)
        res["fits"][str(K)] = o
        print(f"{a.tier} K{K}: pos fit {o['fit_positives']} cal {o['cal_positives']} cal fire {fc:.4f} "
              f"W {o['W']} eligible {o['eligible']} ({o['seconds']}s)", flush=True)
    cand = [(res["fits"][str(K)]["W"], -K) for K in KALT_GRID if res["fits"][str(K)]["eligible"]]
    res["K_alt"] = -min(cand)[1] if cand else None
    res["W_K1500"] = res["fits"]["1500"]["W"]
    Path(a.out).write_text(json.dumps(res) + "\n")
    print(json.dumps({"tier": a.tier, "K_alt": res["K_alt"], "W": {K: res["fits"][str(K)]["W"] for K in
                                                                   [1500] + KALT_GRID}}), flush=True)


def cmd_build(a):
    base = Path(a.blob).read_bytes()
    fit = json.loads(Path(a.fit).read_text())
    f = fit["fits"][str(a.K)] if "fits" in fit else fit["escape"] if "escape" in fit else fit
    if f["K"] != a.K:
        raise SystemExit(f"the fit is for K = {f['K']}, not {a.K}")
    esc = esc_from_json(f["esc"])
    if a.recall != "r0.999":
        esc["t"] = F32(f["thresholds"][a.recall]["t"])
    rare = None
    if a.rare:
        z = np.load(a.rare)
        rare = {k: z[k] for k in ("p_q", "p_scale", "p_bias", "r_q", "r_scale", "r_bias")}
    tr = H2.write_trailer(base, l1_rows(a.K) // 8, esc, rare)
    out = H2.attach(base, tr)
    Path(a.out).write_bytes(out)
    print(json.dumps({"out": a.out, "sha256": H2.sha256(a.out), "bytes": len(out), "base_bytes": len(base),
                      "trailer_bytes": len(tr), "K": a.K, "l1_blocks": l1_rows(a.K) // 8, "t": float(esc["t"]),
                      "rare_k": 0 if rare is None else int(rare["p_q"].shape[0])}))


def quant_all(x, chunk=200000):
    """quant16 of every row: (q int16 [F][K], sx float32 [F])."""
    q = np.empty(x.shape, np.int16)
    sx = np.empty(len(x), F32)
    for c0 in range(0, len(x), chunk):
        qq, ss = H2.quant16_exact(x[c0:c0 + chunk])
        q[c0:c0 + chunk] = qq.astype(np.int16)
        sx[c0:c0 + chunk] = ss
    return q, sx


def exact_logits(q, sx, w, scale, bias, chunk=65536):
    """W8I logits (mm_i16, K <= 512) of rows w for every frame, in chunks: float32 [F][N]."""
    out = np.empty((len(q), w.shape[0]), F32)
    wd = np.asarray(w, np.float64).T
    for c0 in range(0, len(q), chunk):
        s = (q[c0:c0 + chunk].astype(np.float64) @ wd).astype(F32)
        out[c0:c0 + chunk] = (s * (scale[None, :] * sx[c0:c0 + chunk, None]).astype(F32) + bias[None, :]).astype(F32)
    return out


def lse_rows(q, sx, w, scale, bias, chunk=65536):
    """log-sum-exp over the rows' exact W8I logits per frame (float64)."""
    out = np.empty(len(q), np.float64)
    for c0 in range(0, len(q), chunk):
        z = exact_logits(q[c0:c0 + chunk], sx[c0:c0 + chunk], w, scale, bias).astype(np.float64)
        m = z.max(1)
        out[c0:c0 + chunk] = m + np.log(np.exp(z - m[:, None]).sum(1))
    return out


def r48_argmax(q, sx, head, L1, rare, chunk=65536):
    """Per frame the first maximum over level 1's exact logits and r48's exact rare logits (the two-level head's
    argmax on a frame that escapes), and over the full released head's (today's)."""
    V = head["V"]
    a48 = np.empty(len(q), np.int64)
    afull = np.empty(len(q), np.int64)
    for c0 in range(0, len(q), chunk):
        qq = q[c0:c0 + chunk].astype(np.float64)
        ss = sx[c0:c0 + chunk]
        z1 = H2.mm_exact(qq, ss, head["q"][:L1], head["scale"][:L1], head["bias"][:L1])
        zr = H2.rare_logits(qq, ss, rare)[:, :V - L1]
        a48[c0:c0 + chunk] = np.concatenate([z1, zr], 1).argmax(1)
        zf = H2.mm_exact(qq, ss, head["q"][L1:], head["scale"][L1:], head["bias"][L1:])
        afull[c0:c0 + chunk] = np.concatenate([z1, zf], 1).argmax(1)
    return a48, afull


def cmd_r48(a):
    """protocol "r48_fine_tune": P (rk x 96) and the rare rows (R x rk) by KD to the released full head on the
    escape's fit-split head inputs and the synth4 frames; then the r48 labels, the escape and its threshold."""
    import torch
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    t0 = time.time()
    tr = load_train(a.train)
    head = H2.head_arrays(Path(a.blob).read_bytes())
    K, RK = a.K, a.rk
    L1, V, Pd = l1_rows(K), head["V"], head["P"]
    R, Rpad = V - L1, Pd - L1
    fit = ~tr["cal"]
    syn = np.load(a.synth)["x"]
    q_fit, sx_fit = quant_all(tr["x"][fit])
    q_syn, sx_syn = quant_all(syn)
    q = np.concatenate([q_fit, q_syn])
    sx = np.concatenate([sx_fit, sx_syn])
    del q_syn, sx_syn, syn
    n = len(q)
    # init: the rank-RK factorisation of the dequantised rare rows in the metric of the fit split's head inputs
    u_fit = (q_fit.astype(np.float64) * sx_fit[:, None].astype(np.float64))
    S = u_fit.T @ u_fit / len(u_fit)
    lam, E = np.linalg.eigh(S)
    lam = np.maximum(lam, lam.max() * 1e-10)
    s_half = (E * np.sqrt(lam)) @ E.T
    s_mhalf = (E / np.sqrt(lam)) @ E.T
    A_R = head["q"][L1:V].astype(np.float64) * head["scale"][L1:V, None].astype(np.float64)
    U, D, Vt = np.linalg.svd(A_R @ s_half, full_matrices=False)
    rows0 = U[:, :RK] * np.sqrt(D[:RK])[None, :]
    P0 = (np.sqrt(D[:RK])[:, None] * Vt[:RK]) @ s_mhalf
    del u_fit
    # the teacher: level 1's log-sum-exp (fixed) and the rare rows' exact W8I logits (per batch)
    lse1 = lse_rows(q, sx, head["q"][:L1], head["scale"][:L1], head["bias"][:L1])
    wR = torch.from_numpy(head["q"][L1:V].astype(np.float64).T.copy())
    sR = torch.from_numpy(head["scale"][L1:V].astype(np.float32))
    bR = torch.from_numpy(head["bias"][L1:V].astype(np.float32))
    Pm = torch.nn.Parameter(torch.from_numpy(P0.astype(np.float32)))
    pb = torch.nn.Parameter(torch.zeros(RK))
    Wr = torch.nn.Parameter(torch.from_numpy(rows0.astype(np.float32)))
    br = torch.nn.Parameter(bR.clone())
    params = [Pm, pb, Wr, br]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.0)
    bs, epochs = 4096, 4
    steps = epochs * ((n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(1, steps)))))
    g = torch.Generator().manual_seed(0)
    qt, sxt, lset = torch.from_numpy(q), torch.from_numpy(sx), torch.from_numpy(lse1)

    def kl_batch(idx):
        qb = qt[idx].to(torch.float64)
        sb = sxt[idx]
        zR = ((qb @ wR).to(torch.float32) * (sR[None, :] * sb[:, None]) + bR[None, :]).to(torch.float64)
        l1 = lset[idx]
        lse_t = torch.logaddexp(l1, torch.logsumexp(zR, 1))
        pR = torch.exp(zR - lse_t[:, None])
        u = (qb.to(torch.float32) * sb[:, None])
        zs = ((u @ Pm.T + pb) @ Wr.T + br).to(torch.float64)
        lse_s = torch.logaddexp(l1, torch.logsumexp(zs, 1))
        return ((pR * (zR - zs)).sum(1) + lse_s - lse_t).mean()

    log = []
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot, nb = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = kl_batch(idx)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            tot += float(loss)
            nb += 1
        log.append({"epoch": ep + 1, "mean_kl": round(tot / nb, 6), "seconds": round(time.time() - t0, 1)})
        print(f"{a.tier} K{K} r{RK} epoch {ep + 1}: kl {tot / nb:.6f} ({time.time() - t0:.0f}s)", flush=True)
    import export
    P_f = Pm.detach().numpy().astype(np.float64)
    Wr_f = Wr.detach().numpy().astype(np.float64)
    p_q, p_scale = export.quantize_rows(P_f)
    r_q0, r_scale0 = export.quantize_rows(Wr_f)
    r_q = np.zeros((Rpad, RK), np.int8)
    r_q[:R] = r_q0
    r_scale = np.zeros(Rpad, F32)
    r_scale[:R] = r_scale0
    r_bias = np.zeros(Rpad, F32)
    r_bias[:R] = br.detach().numpy()
    rare = dict(p_q=p_q, p_scale=p_scale.astype(F32), p_bias=pb.detach().numpy().astype(F32), r_q=r_q,
                r_scale=r_scale, r_bias=r_bias)
    # the calibration split: KL (float) and argmax agreement with today's head (float, exact W8I)
    cal = tr["cal"]
    qc, sc = quant_all(tr["x"][cal])
    with torch.no_grad():
        qt, sxt = torch.from_numpy(qc), torch.from_numpy(sc)
        lset = torch.from_numpy(lse_rows(qc, sc, head["q"][:L1], head["scale"][:L1], head["bias"][:L1]))
        tot = 0.0
        for i in range(0, len(qc), 65536):
            idx = torch.arange(i, min(len(qc), i + 65536))
            tot += float(kl_batch(idx)) * len(idx)
        kl_cal = tot / len(qc)
    a48, afull = r48_argmax(qc, sc, head, L1, rare)
    rep = {"kind": "head2level-r48", "tier": a.tier, "K": K, "L1": L1, "rare_k": RK, "rare_rows": R,
           "train_frames": int(fit.sum()), "synth_frames": int(n - fit.sum()), "steps": step, "log": log,
           "cal_frames": int(cal.sum()), "cal_mean_kl_float": round(kl_cal, 6),
           "cal_argmax_agree_exact": round(float((a48 == afull).mean()), 6),
           "cal_argmax_agree_exact_nonblank": round(float((a48 == afull)[afull != 0].mean()), 6),
           "bytes_rare96": int(Rpad * (96 + 8)), "bytes_r48": int(RK * 96 + 8 * RK + Rpad * (RK + 8))}
    # r48 labels on every training frame: the two-level head's argmax with r48 lies outside level 1
    qa, sa = quant_all(tr["x"])
    y48 = r48_argmax(qa, sa, head, L1, rare)[0] >= L1
    rep["label_share"] = round(float(y48.mean()), 6)
    rep["label_share_r96"] = round(float((tr["am"] >= L1).mean()), 6)
    o = fit_one(tr, K, labels=y48, threads=a.threads)
    o["W"] = round(L1 + ESC_MACS_ROWS + o["thresholds"]["r0.999"]["cal_fire_share"] * (Pd - L1), 3)
    rep["escape"] = o
    rep["seconds"] = round(time.time() - t0, 1)
    np.savez(a.out, **rare)
    Path(a.out + ".json").write_text(json.dumps(rep) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("escape", "log")}), flush=True)
    print(f"{a.tier} K{K} r48 escape: cal fire {o['thresholds']['r0.999']['cal_fire_share']} W {o['W']}", flush=True)


def cmd_retarget(a):
    """A copy of a two-level blob whose escape threshold is t (timing bounds: +inf never fires, -inf always)."""
    data = Path(a.blob).read_bytes()
    tr = H2.read_trailer(data)
    base = data[:tr["offset"]]
    esc = dict(tr["esc"])
    esc["t"] = F32(float(a.t))
    out = H2.attach(base, H2.write_trailer(base, tr["l1_blocks"], esc, tr["rare"]))
    Path(a.out).write_bytes(out)
    print(json.dumps({"out": a.out, "sha256": H2.sha256(a.out), "t": float(esc["t"])}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("retarget")
    p.add_argument("--blob", required=True)
    p.add_argument("--t", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("escapes")
    p.add_argument("--tier", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--blob", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=5)
    p = sub.add_parser("r48")
    p.add_argument("--tier", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--synth", required=True)
    p.add_argument("--blob", required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--rk", type=int, default=48)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=5)
    p = sub.add_parser("build")
    p.add_argument("--tier", required=True)
    p.add_argument("--blob", required=True)
    p.add_argument("--fit", required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--rare", default="")
    p.add_argument("--recall", default="r0.999", choices=("r0.999", "r0.9999"))
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"escapes": cmd_escapes, "build": cmd_build, "r48": cmd_r48, "retarget": cmd_retarget}[a.cmd](a)


if __name__ == "__main__":
    main()
