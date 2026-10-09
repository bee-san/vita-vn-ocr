"""Line-level matching metrics for the dialogue line finder.

Boxes are (x0, y0, x1, y1) with exclusive ends, in region coordinates.

A predicted line matches a ground-truth line when their IoU is at least 0.5
(one-to-one, greedy by IoU). Ground-truth lines marked `ignore` (furigana, tiny UI
labels) can absorb predictions without counting as hits or false alarms.
Beyond plain matches the report counts the failure modes that matter for OCR:
  merged    one prediction covers most of two or more GT lines
  split     one GT line is covered by two or more predictions
  cut       a matched prediction misses part of its GT line horizontally
            (less than `cover` of the GT width inside the prediction)
"""


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def inter(a, b):
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a, b):
    i = inter(a, b)
    u = area(a) + area(b) - i
    return i / u if u else 0.0


def vfrac(p, g):
    """Fraction of g's height covered by p's rows."""
    h = g[3] - g[1]
    return max(0, min(p[3], g[3]) - max(p[1], g[1])) / h if h > 0 else 0.0


def hfrac(p, g):
    w = g[2] - g[0]
    return max(0, min(p[2], g[2]) - max(p[0], g[0])) / w if w > 0 else 0.0


def match(pred, gt, ignore=(), thr=0.5, cover=0.95):
    """pred, gt, ignore: lists of boxes. Returns a dict of counts and the pairs."""
    pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt)),
                   reverse=True)
    used_p, used_g, matched = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matched.append((i, j, v))
    fp = 0
    for i, p in enumerate(pred):
        if i in used_p:
            continue
        if any(iou(p, g) >= 0.3 or inter(p, g) >= 0.5 * area(p) for g in ignore):
            continue
        fp += 1
    merged = sum(1 for p in pred if sum(vfrac(p, g) >= 0.5 and hfrac(p, g) >= 0.5 for g in gt) >= 2)
    split = sum(1 for g in gt if sum(vfrac(p, g) >= 0.3 and hfrac(p, g) >= 0.3 for p in pred) >= 2)
    cut = sum(1 for i, j, _ in matched if hfrac(pred[i], gt[j]) < cover)
    return {"tp": len(matched), "fp": fp, "fn": len(gt) - len(matched), "merged": merged,
            "split": split, "cut": cut, "pairs": matched}


def summarize(rows):
    """rows: list of match() results (one per region). Line and region level totals."""
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    n_gt = tp + fn
    out = {
        "regions": len(rows),
        "gt_lines": n_gt,
        "line_recall": tp / n_gt if n_gt else 0.0,
        "line_precision": tp / (tp + fp) if tp + fp else 0.0,
        "merged": sum(r["merged"] for r in rows),
        "split": sum(r["split"] for r in rows),
        "cut": sum(r["cut"] for r in rows),
        # region ("box") level: every GT line found, nothing extra
        "region_exact": sum(r["fn"] == 0 and r["fp"] == 0 for r in rows) / len(rows) if rows else 0.0,
        # region level recall: at least one GT line found
        "region_any": sum(r["tp"] > 0 for r in rows) / len(rows) if rows else 0.0,
    }
    p, r = out["line_precision"], out["line_recall"]
    out["line_f1"] = 2 * p * r / (p + r) if p + r else 0.0
    return out
