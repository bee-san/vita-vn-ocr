#!/usr/bin/env python3
"""Measure the teacher on eval's hand-checked sets and choose the acceptance rule.

  measure.py rules [--set vita-test,psp-test] [--reads DIR] [--precision 0.99] [--out FILE]

label.py read --set vita-test / psp-test --no-cascade must have read the test screenshots
(every engine on every line). Each teacher line is matched to eval's ground truth: the
hand-checked lines whose box overlaps the teacher's crop box (vertically by at least half
the smaller height, horizontally at all), joined left to right. A teacher line that touches
an uncertain line is not scored, nor is one that matches nothing (menus and UI, which eval
leaves out). Then, for every rule in the grid (vote.accept parameters):

  precision  accepted scored lines whose label scores 0 errors against the truth
             (eval/metrics.norm on both sides, which vote.key reproduces)
  recall     scored ground-truth lines covered by an accepted, correct teacher line
  label CER  edit distance of accepted labels / truth characters
  accepted   lines accepted in all, including unscored ones (what the relabel would keep)

The chosen rule maximises recall with precision >= --precision on both sets pooled; it is
also chosen on each half of the games (2-fold, by VN) and scored on the other half, so the
reported held-out numbers are not tuned on the lines they describe. Per-engine accuracy on
the same matched lines is reported too ("pipeline" CER: each engine reading the teacher's
own view of the line).
"""
import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vote  # noqa: E402
from label import EVAL_SETS, OUT, eval_manifest  # noqa: E402

ENGINES = ("meiki", "meikiv", "v6s", "manga", "screenai", "v5s")


def gt_rows(name):
    rows = {}
    for line in eval_manifest(name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            x, y, w, h = r["box"]
            r["xyxy"] = [x, y, x + w, y + h]
            rows.setdefault(f"{r['vn']}_{r['url'].rsplit('/', 1)[-1].rsplit('.', 1)[0]}", []).append(r)
    return rows


def load_reads(reads_dir, name):
    out = {}
    for p in sorted(Path(reads_dir).glob(f"{name}.*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[r["shot"]] = r
    return out


def vov(a, b):
    inter = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, inter) / max(1, min(a[3] - a[1], b[3] - b[1]))


def match(shots, gts, name):
    """[(set, vn, teacher line, truth text or None, [gt ids])], and the scored gt ids."""
    items, scored_gt = [], {}
    for key, rows in gts.items():
        for g in rows:
            if not g.get("uncertain"):
                scored_gt[g["id"]] = g
    for key, r in shots.items():
        rows = gts.get(key, [])
        for ln in r["lines"]:
            box = ln.get("crop") or ln["ink"]
            hits = [g for g in rows if vov(box, g["xyxy"]) >= 0.5
                    and min(box[2], g["xyxy"][2]) - max(box[0], g["xyxy"][0]) > 0]
            if not hits or any(g.get("uncertain") for g in hits):
                truth = None
            else:
                truth = "".join(g["text"] for g in sorted(hits, key=lambda g: g["xyxy"][0]))
            items.append((name, r["vn"], ln, truth, [g["id"] for g in hits]))
    return items, scored_gt


def score(items, scored_gt, rule, vns=None):
    acc = acc_scored = correct = errs = chars = 0
    recalled = set()
    for _, vn, ln, truth, ids in items:
        if vns is not None and vn not in vns:
            continue
        lab, _ = vote.accept(ln, rule)
        if lab is None:
            continue
        acc += 1
        if truth is None:
            continue
        acc_scored += 1
        e = vote.lev(vote.eval_norm(lab), vote.eval_norm(truth))
        errs += e
        chars += len(vote.eval_norm(truth))
        if e == 0:
            correct += 1
            recalled.update(i for i in ids if i in scored_gt)
    n_gt = sum(1 for g in scored_gt.values() if vns is None or g["vn"] in vns)
    return {"accepted": acc, "accepted_scored": acc_scored, "correct": correct,
            "precision": correct / acc_scored if acc_scored else None,
            "recall": len(recalled) / n_gt if n_gt else None, "gt_lines": n_gt,
            "label_cer": errs / chars if chars else None}


def wilson(k, n, z=1.96):
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 4), round(c + h, 4)


def grid():
    voter_sets = []
    base = ("v6s", "manga", "screenai", "v5s")
    for n in range(1, len(base) + 1):
        voter_sets += list(itertools.combinations(base, n))
    for primary in ("meikiv", "meiki"):
        other = "meiki" if primary == "meikiv" else "meikiv"
        for min_conf in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95):      # MeikiOCR's confidence alone
            for det_conf in (0.0, 0.7, 0.8):
                yield {"primary": primary, "voters": (), "votes": 0, "min_conf": min_conf, "det_conf": det_conf}
        for vs in voter_sets:
            for with_other in (False, True):
                voters = vs + ((other,) if with_other else ())
                for votes in range(1, len(voters) + 1):
                    for min_conf in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9):
                        for det_conf in (0.0, 0.7, 0.8):
                            for veto in (False, True):
                                yield {"primary": primary, "voters": voters, "votes": votes,
                                       "min_conf": min_conf, "det_conf": det_conf, "veto": veto}


def choose(items, scored_gt, target, vns=None):
    best = None
    for rule in grid():
        s = score(items, scored_gt, rule, vns)
        if s["precision"] is None or s["precision"] < target:
            continue
        k = (s["recall"], s["precision"], -len(rule["voters"]))
        if best is None or k > best[0]:
            best = (k, rule, s)
    return best


def pipeline_engines(items):
    """Each engine's accuracy on the matched, scored teacher lines (its reading of the
    teacher's view; meiki = MeikiOCR on its own detector box)."""
    out = {}
    for e in ENGINES:
        n = exact = errs = chars = 0
        for _, _, ln, truth, _ in items:
            if truth is None:
                continue
            t = vote.reading(ln, e)
            if t is None:
                continue
            g = vote.eval_norm(truth)
            d = vote.lev(vote.key(t[0]), g)
            n, exact, errs, chars = n + 1, exact + (d == 0), errs + d, chars + len(g)
        out[e] = {"lines": n, "exact": round(exact / n, 4) if n else None,
                  "cer": round(errs / chars, 4) if chars else None}
    return out


def fold_of(vn):
    return int(hashlib.md5(f"teacher-fold:{vn}".encode()).hexdigest(), 16) % 2


def fmt(s):
    p, r, c = s["precision"], s["recall"], s["label_cer"]
    out = f"precision {100 * p:.2f}% ({s['correct']}/{s['accepted_scored']}) " if p is not None else "precision - "
    out += f"recall {100 * r:.2f}% of {s['gt_lines']} " if r is not None else ""
    out += f"label CER {100 * c:.3f}% " if c is not None else ""
    return out + f"accepted {s['accepted']}"


def cmd_rules(args):
    items, scored = [], {}
    sets = args.set.split(",")
    per_set = {}
    for name in sets:
        shots = load_reads(args.reads, name)
        gts = gt_rows(name)
        missing = [k for k in gts if k not in shots]
        if missing:
            print(f"WARNING {name}: {len(missing)} test screenshots not read yet", flush=True)
        it, sg = match(shots, gts, name)
        per_set[name] = (it, sg)
        items += it
        scored.update(sg)
        n_det = len({i for _, _, _, _, ids in it for i in ids if i in sg})
        print(f"{name}: {len(shots)} screenshots, {len(it)} teacher lines, {sum(t is not None for *_, t, _ in it)}"
              f" matched to truth; {len(sg)} scored truth lines, {n_det} overlapped by a teacher line "
              f"({100 * n_det / max(1, len(sg)):.1f}% detection recall)", flush=True)
    report = {"sets": sets, "engines": {n: pipeline_engines(it) for n, (it, _) in per_set.items()}}
    print("\nper-engine accuracy on the matched lines (each engine reading the teacher's view):")
    for name, eng in report["engines"].items():
        for e, s in eng.items():
            print(f"  {name:9s} {e:9s} lines {s['lines']:4d}  CER {s['cer']}  exact {s['exact']}")
    best = choose(items, scored, args.precision)
    if best is None:
        sys.exit(f"no rule reaches precision {args.precision}")
    _, rule, s = best
    report["chosen"] = {"rule": rule, "pooled": s,
                        "precision_ci95": wilson(s["correct"], s["accepted_scored"]),
                        "per_set": {n: score(it, sg, rule) for n, (it, sg) in per_set.items()}}
    print(f"\nchosen (recall-max at precision >= {args.precision}): {json.dumps(rule)}")
    print(f"  pooled: {fmt(s)}  95% CI of precision {report['chosen']['precision_ci95']}")
    for n, ss in report["chosen"]["per_set"].items():
        print(f"  {n}: {fmt(ss)}")
    # 2-fold by game: choose on one half, score on the other
    vns = {g["vn"] for g in scored.values()}
    folds = [{v for v in vns if fold_of(v) == f} for f in (0, 1)]
    held = []
    for f in (0, 1):
        b = choose(items, scored, args.precision, folds[f])
        if b is None:
            held.append(None)
            continue
        s_other = score(items, scored, b[1], folds[1 - f])
        held.append({"chosen_on": f, "rule": b[1], "train": b[2], "held_out": s_other})
        print(f"  fold {f} -> {1 - f}: {json.dumps(b[1])}\n      held out: {fmt(s_other)}")
    report["two_fold"] = held
    # a few reference rules
    refs = {"meikiv alone": {"voters": (), "votes": 0},
            "meiki alone": {"primary": "meiki", "voters": (), "votes": 0},
            "meikiv + 1 of v6s/manga/screenai": {"votes": 1},
            "meikiv + 2 of v6s/manga/screenai": {"votes": 2},
            "meikiv + 3 of v6s/manga/screenai": {"votes": 3},
            "meikiv + v6s + manga (no Screen AI)": {"voters": ("v6s", "manga"), "votes": 2},
            "meikiv + v5s + manga (label_real.py's engines)": {"voters": ("v5s", "manga"), "votes": 2}}
    report["reference"] = {}
    print("\nreference rules:")
    for name, rule in refs.items():
        s = score(items, scored, rule)
        report["reference"][name] = {"rule": rule, **s}
        print(f"  {name:48s} {fmt(s)}")
    # the precision/recall frontier of the grid (pooled)
    pts = []
    for rule in grid():
        s = score(items, scored, rule)
        if s["precision"] is not None:
            pts.append((s["recall"], s["precision"], rule, s))
    pts.sort(key=lambda t: (-t[0], -t[1]))
    front, best_p = [], -1.0
    for r, p, rule, s in pts:
        if p > best_p:
            front.append({"rule": rule, **s})
            best_p = p
    report["frontier"] = front
    print("\nfrontier (pooled):")
    for f in front[:25]:
        print(f"  recall {100 * f['recall']:.1f}%  precision {100 * f['precision']:.2f}%  {json.dumps(f['rule'])}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("rules")
    p.add_argument("--set", default=",".join(EVAL_SETS))
    p.add_argument("--reads", default=str(OUT / "reads"))
    p.add_argument("--precision", type=float, default=0.99)
    p.add_argument("--out", default="")
    args = ap.parse_args()
    cmd_rules(args)


if __name__ == "__main__":
    main()
