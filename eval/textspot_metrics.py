"""Metrics of the full-frame text spotter (issue #104, docs/textspot.md, eval/textspot_protocol.json), frozen with the
protocol before any training run or score.

A frame (one row of eval/manifests/textspot-*.jsonl, with its pixels) has references and an arm's detections:

    refs  [{"box": [x, y, w, h], "cls": CLASS, "vertical": bool, "text": str | None, "ignore": bool}]
          box = the instance's main-text ink box in frame pixels (fill, outline and shadow; furigana excluded);
          ignore = an area that is neither found nor missed (furigana, logos, unreadable or tiny text, ...)
    dets  [{"box": [x, y, w, h], "vertical": bool, "text": str}]   text = the recognizer's reading of the box

Every function here is pure (no I/O); eval/textspot.py runs the arms and feeds them.

Detection (match_frame): a reference's scoring box is its ink box padded by the spotter's own pad rule (pad_box:
max(2, (across 18 + 50) / 100) px on every side, across = the height of a horizontal box, the width of a vertical
one), clipped to the frame. A detection with at least half of its area inside one ignore reference's padded box is
left out. The others are matched one to one to the non-ignore references at IoU >= 0.5, greedily by IoU (ties: the
lower detection index, then the lower reference index). tp = matched pairs, fp = unmatched detections, fn =
unmatched references; recall per class. Ink tightness of a matched pair: its IoU and the ink coverage
|det & ink| / |ink| (1.0: the detection cuts none of the reference's ink).

End to end (e2e_frame): every detection not left out goes to the reference (ignore or not) whose padded box holds the
largest part of it, if that part is at least half of the detection's area (ties: the lower reference index); a
detection that goes to an ignore reference is left out, one that goes to none is junk. A reference's hypothesis is
its detections' readings joined left to right (top to bottom for a vertical reference) by (x, y) (or (y, x));
errors = Levenshtein(norm(text), norm(hypothesis)) (eval/metrics.norm, lev); junk adds len(norm(reading)). The
frame's unit: errors = the references' errors + junk, chars = the references' norm lengths. The dialogue unit counts
the references of class dialogue and choice only, and no junk.

Intervals: 10,000 VN-cluster bootstrap resamples (seed 0; sorted VN ids, numpy default_rng(seed).integers, linear
2.5 / 97.5 percentiles: eval/textdet.vn_diff's scheme, copied here so the protocol pins it); CER differences in
points (vn_diff), F1 differences on the pooled tp / fp / fn of each resample (f1_diff).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from metrics import lev, norm  # noqa: E402

CLASSES = ("dialogue", "name", "choice", "ui", "scene")
DIALOGUE = ("dialogue", "choice")
IOU_MIN = 0.5
INSIDE_MIN = 0.5
PAD_PCT, PAD_MIN = 18, 2
BOOT = {"reps": 10000, "seed": 0, "min_clusters": 5}


# ----------------------------------------------------------------------------- boxes

def pad_box(box, vertical, width, height):
    x, y, w, h = box
    across = w if vertical else h
    p = max(PAD_MIN, (across * PAD_PCT + 50) // 100)
    x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(width, x + w + p), min(height, y + h + p)
    return [x0, y0, max(0, x1 - x0), max(0, y1 - y0)]


def area(b):
    return max(0, b[2]) * max(0, b[3])


def inter(a, b):
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0


def iou(a, b):
    i = inter(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def _padded(refs, width, height):
    return [pad_box(r["box"], bool(r.get("vertical")), width, height) for r in refs]


def _left_out(dets, refs, padded):
    """Indices of detections with at least half their area inside one ignore reference's padded box."""
    out = set()
    for i, d in enumerate(dets):
        a = area(d["box"])
        for r, p in zip(refs, padded):
            if r.get("ignore") and a and 2 * inter(d["box"], p) >= a:
                out.add(i)
                break
    return out


# ----------------------------------------------------------------------------- detection

def match_frame(dets, refs, width, height):
    """{"tp", "fp", "fn", "tp_cls", "fn_cls", "pairs": [(det index, ref index, iou, ink coverage)], "left_out"}."""
    padded = _padded(refs, width, height)
    out_idx = _left_out(dets, refs, padded)
    cand = []
    for i, d in enumerate(dets):
        if i in out_idx:
            continue
        for j, (r, p) in enumerate(zip(refs, padded)):
            if r.get("ignore"):
                continue
            v = iou(d["box"], p)
            if v >= IOU_MIN:
                cand.append((-v, i, j))
    cand.sort()
    used_d, used_r, pairs = set(), set(), []
    for nv, i, j in cand:
        if i in used_d or j in used_r:
            continue
        used_d.add(i)
        used_r.add(j)
        ink = refs[j]["box"]
        cov = inter(dets[i]["box"], ink) / area(ink) if area(ink) else 0.0
        pairs.append((i, j, -nv, cov))
    scored = [j for j, r in enumerate(refs) if not r.get("ignore")]
    tp_cls = {c: 0 for c in CLASSES}
    fn_cls = {c: 0 for c in CLASSES}
    for j in scored:
        (tp_cls if j in used_r else fn_cls)[refs[j]["cls"]] += 1
    n_det = len(dets) - len(out_idx)
    return {"tp": len(pairs), "fp": n_det - len(pairs), "fn": len(scored) - len(pairs), "tp_cls": tp_cls,
            "fn_cls": fn_cls, "pairs": pairs, "left_out": sorted(out_idx)}


def f1_of(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else None


def pooled(counts):
    """counts: [{"tp", "fp", "fn", ...}] -> precision, recall, F1 over the pooled sums."""
    tp = sum(c["tp"] for c in counts)
    fp = sum(c["fp"] for c in counts)
    fn = sum(c["fn"] for c in counts)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None, "f1": f1_of(tp, fp, fn)}


def tightness(frames):
    """frames: [(match_frame result)] -> mean IoU and mean ink coverage over every matched pair."""
    ious = [p[2] for m in frames for p in m["pairs"]]
    covs = [p[3] for m in frames for p in m["pairs"]]
    return {"pairs": len(ious), "mean_iou": float(np.mean(ious)) if ious else None,
            "mean_ink_coverage": float(np.mean(covs)) if covs else None}


# ----------------------------------------------------------------------------- end to end

def assign(dets, refs, width, height):
    """[reference index | None | "out"] per detection (see the module docstring)."""
    padded = _padded(refs, width, height)
    out_idx = _left_out(dets, refs, padded)
    res = []
    for i, d in enumerate(dets):
        if i in out_idx:
            res.append("out")
            continue
        a = area(d["box"])
        best, bj = 0, None
        for j, p in enumerate(padded):
            v = inter(d["box"], p)
            if v > best:
                best, bj = v, j
        if bj is None or not a or 2 * best < a:
            res.append(None)
        elif refs[bj].get("ignore"):
            res.append("out")
        else:
            res.append(bj)
    return res


def e2e_frame(dets, refs, width, height):
    """{"all": (errors, chars), "dialogue": (errors, chars), "junk": chars of junk, "per_ref": [errors or None]}."""
    a = assign(dets, refs, width, height)
    hyp = {j: [] for j in range(len(refs))}
    junk = 0
    for d, t in zip(dets, a):
        if t == "out":
            continue
        if t is None:
            junk += len(norm(d.get("text") or ""))
        else:
            hyp[t].append(d)
    e_all = c_all = e_dlg = c_dlg = 0
    per_ref = []
    for j, r in enumerate(refs):
        if r.get("ignore"):
            per_ref.append(None)
            continue
        vert = bool(r.get("vertical"))
        parts = sorted(hyp[j], key=(lambda d: (d["box"][1], d["box"][0])) if vert else
                       (lambda d: (d["box"][0], d["box"][1])))
        ref_t = norm(r["text"])
        e = lev(ref_t, norm("".join(d.get("text") or "" for d in parts)))
        per_ref.append(e)
        e_all += e
        c_all += len(ref_t)
        if r["cls"] in DIALOGUE:
            e_dlg += e
            c_dlg += len(ref_t)
    return {"all": (e_all + junk, c_all), "dialogue": (e_dlg, c_dlg), "junk": junk, "per_ref": per_ref}


def cer_of(units):
    e = sum(u["errors"] for u in units)
    n = sum(u["chars"] for u in units)
    return {"errors": e, "chars": n, "cer": round(100 * e / n, 4) if n else None}


# ----------------------------------------------------------------------------- intervals

def _clusters(units):
    vns = sorted({u["vn"] for u in units})
    return vns, {v: i for i, v in enumerate(vns)}


def _paired(cand, base):
    a = {u["unit"]: u for u in cand}
    b = {u["unit"]: u for u in base}
    if set(a) != set(b) or any(a[k]["vn"] != b[k]["vn"] for k in a):
        raise ValueError("paired comparison needs the same frames")
    return a, b


def vn_diff(cand, base, boot=BOOT):
    """CER(cand) - CER(base) in points over frame units {"unit", "vn", "errors", "chars"}, with a VN-cluster
    bootstrap (eval/textdet.vn_diff's scheme for one pair)."""
    a, b = _paired(cand, base)
    if any(a[k]["chars"] != b[k]["chars"] for k in a):
        raise ValueError("paired comparison needs identical denominators")
    vns, ci = _clusters(a.values())
    K = len(vns)
    n = np.zeros(K)
    d = np.zeros(K)
    for k in a:
        c = ci[a[k]["vn"]]
        n[c] += a[k]["chars"]
        d[c] += a[k]["errors"] - b[k]["errors"]
    point = 100 * d.sum() / n.sum() if n.sum() else None
    out = {"delta": round(point, 4) if point is not None else None, "clusters": K, "chars": int(n.sum()), **boot}
    if K < boot["min_clusters"] or not n.sum():
        return dict(out, ci95=None)
    idx = np.random.default_rng(boot["seed"]).integers(0, K, (boot["reps"], K))
    tot = n[idx].sum(1)
    ok = tot > 0
    ds = d[idx].sum(1)[ok] / tot[ok]
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return dict(out, ci95=[round(100 * float(lo), 4), round(100 * float(hi), 4)])


def f1_diff(cand, base, boot=BOOT):
    """F1(cand) - F1(base) over frame counts {"unit", "vn", "tp", "fp", "fn"}, pooled, with a VN-cluster bootstrap
    (the same resamples as vn_diff)."""
    a, b = _paired(cand, base)
    vns, ci = _clusters(a.values())
    K = len(vns)
    m = np.zeros((2, 3, K))
    for k in a:
        c = ci[a[k]["vn"]]
        for s, u in enumerate((a[k], b[k])):
            m[s, :, c] += (u["tp"], u["fp"], u["fn"])

    def f1(t):  # t: (3, ...) sums -> F1
        den = 2 * t[0] + t[1] + t[2]
        return np.where(den > 0, 2 * t[0] / np.where(den > 0, den, 1), np.nan)

    point = float(f1(m[0].sum(1)) - f1(m[1].sum(1)))
    out = {"delta": round(point, 6), "clusters": K, **boot}
    if K < boot["min_clusters"]:
        return dict(out, ci95=None)
    idx = np.random.default_rng(boot["seed"]).integers(0, K, (boot["reps"], K))
    ds = f1(m[0][:, idx].sum(2)) - f1(m[1][:, idx].sum(2))
    ds = ds[~np.isnan(ds)]
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return dict(out, ci95=[round(float(lo), 6), round(float(hi), 6)])


# ----------------------------------------------------------------------------- the rule

def arm_summary(frames):
    """frames: [{"unit", "vn", "det": match_frame, "e2e": e2e_frame}] -> the units the rule compares."""
    counts = [{"unit": f["unit"], "vn": f["vn"], "tp": f["det"]["tp"], "fp": f["det"]["fp"], "fn": f["det"]["fn"]}
              for f in frames]
    u_all = [{"unit": f["unit"], "vn": f["vn"], "errors": f["e2e"]["all"][0], "chars": f["e2e"]["all"][1]}
             for f in frames]
    u_dlg = [{"unit": f["unit"], "vn": f["vn"], "errors": f["e2e"]["dialogue"][0], "chars": f["e2e"]["dialogue"][1]}
             for f in frames]
    return {"counts": counts, "all": u_all, "dialogue": u_dlg}


def compare(cand, base):
    """The rule's three comparisons of one candidate arm against one baseline arm (arm_summary results)."""
    return {"f1": f1_diff(cand["counts"], base["counts"]), "cer_all": vn_diff(cand["all"], base["all"]),
            "cer_dialogue": vn_diff(cand["dialogue"], base["dialogue"])}


def beats(cmp):
    """F1 resolved above, all-text CER resolved below, dialogue CER not resolved above."""
    f, e, d = cmp["f1"]["ci95"], cmp["cer_all"]["ci95"], cmp["cer_dialogue"]["ci95"]
    return bool(f and f[0] > 0 and e and e[1] < 0 and not (d and d[0] > 0))


def eligible(c, rule):
    return bool(c["arena_960"] <= rule["arena_max"] and c["est_ms_333_960"] <= rule["ms_max_333_960"]
                and c["c_equal"])


def size_key(c):
    return (c["blob_bytes"], c["macs_960"], c["decode"] != "det", c["name"])


def apply_rule(cands, baselines, rule):
    """cands: [{"name", "decode", "blob_bytes", "macs_960", "arena_960", "est_ms_333_960", "c_equal",
    "arm": arm_summary}], baselines: {name: arm_summary} -> {"eligible", "qualifying", "winner", "steps",
    "report_candidate", "comparisons"}."""
    comps, qual = {}, []
    elig = sorted((c for c in cands if eligible(c, rule)), key=size_key)
    for c in elig:
        comps[c["name"]] = {b: compare(c["arm"], arm) for b, arm in sorted(baselines.items())}
        if all(beats(x) for x in comps[c["name"]].values()):
            qual.append(c)
    winner, steps = (qual[0] if qual else None), []
    while winner is not None:
        nxt = None
        for c in qual:
            if size_key(c) <= size_key(winner):
                continue
            d = vn_diff(c["arm"]["all"], winner["arm"]["all"])
            if d["ci95"] and d["ci95"][1] < 0:
                nxt = c
                steps.append({"replaces": winner["name"], "by": c["name"], "cer_all": d})
                break
        if nxt is None:
            break
        winner = nxt
    report = winner
    if report is None and elig:
        report = min(elig, key=lambda c: (cer_of(c["arm"]["all"])["cer"], size_key(c)))
    return {"eligible": [c["name"] for c in elig], "qualifying": [c["name"] for c in qual],
            "winner": winner["name"] if winner else None, "steps": steps,
            "report_candidate": report["name"] if report else None, "comparisons": comps}
