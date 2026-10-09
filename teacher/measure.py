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

The chosen rule maximises pooled recall with precision >= --precision (default 99.5%) on
each set and pooled; it is also chosen on each half of the games (2-fold, by VN) and scored
on the other half, so the reported held-out numbers are not tuned on the lines they describe. Per-engine accuracy on
the same matched lines is reported too ("pipeline" CER: each engine reading the teacher's
own view of the line).
"""
import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from teacher import vote  # noqa: E402
from teacher.label import EVAL_SETS, OUT, eval_manifest  # noqa: E402

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


COVER = 0.9          # a teacher crop must span this share of the truth's width to be scored


def _status(box, rows):
    hits = [g for g in rows if vov(box, g["xyxy"]) >= 0.5
            and min(box[2], g["xyxy"][2]) - max(box[0], g["xyxy"][0]) > 0]
    truth, status = None, "unmatched"
    if hits and any(g.get("uncertain") for g in hits):
        status = "uncertain"
    elif hits:
        gx0, gx1 = min(g["xyxy"][0] for g in hits), max(g["xyxy"][2] for g in hits)
        cover = (min(box[2], gx1) - max(box[0], gx0)) / max(1, gx1 - gx0)
        if cover < COVER:
            status = "partial"
        else:
            status = "scored"
            truth = "".join(g["text"] for g in sorted(hits, key=lambda g: g["xyxy"][0]))
    return truth, [g["id"] for g in hits], status


def match(shots, gts, name):
    """[(set, vn, teacher line, {primary: (truth or None, [gt ids], status)})], and the scored
    gt ids. Lines are first passed through geometry.refurigana (as build.py does), and each
    line is matched with the crop build.py would cut for it (build.crop_of, which depends on
    the rule's primary reading). status: scored | partial (the crop covers < COVER of the
    truth's width, so box matching can't tell whether a shorter label is right for it) |
    uncertain | unmatched (UI, menus and other text eval leaves out)."""
    from teacher import build
    from teacher import geometry as G
    items, scored_gt = [], {}
    for key, rows in gts.items():
        for g in rows:
            if not g.get("uncertain"):
                scored_gt[g["id"]] = dict(g, set=name)
    for key, r in shots.items():
        rows = gts.get(key, [])
        for ln in G.refurigana(r["lines"]):
            m = {p: _status(build.crop_of(ln, {"primary": p}), rows) for p in ("meiki", "meikiv")}
            items.append((name, r["vn"], ln, m))
    return items, scored_gt


def _m(item, rule):
    return item[3][(rule or {}).get("primary", vote.DEFAULT["primary"])]


def _aggregate(items, decisions, scored_gt, rule, keep):
    acc = acc_scored = correct = errs = chars = partial = 0
    recalled = set()
    for it, lab in zip(items, decisions):
        if lab is None or not keep(it):
            continue
        truth, ids, status = _m(it, rule)
        acc += 1
        partial += status == "partial"
        if truth is None:
            continue
        acc_scored += 1
        e = vote.lev(vote.eval_norm(lab), vote.eval_norm(truth))
        errs += e
        chars += len(vote.eval_norm(truth))
        if e == 0:
            correct += 1
            recalled.update(i for i in ids if i in scored_gt)
    n_gt = sum(1 for g in scored_gt.values() if keep((g["set"], g["vn"])))
    return {"accepted": acc, "accepted_scored": acc_scored, "accepted_partial": partial, "correct": correct,
            "precision": correct / acc_scored if acc_scored else None,
            "recall": len(recalled) / n_gt if n_gt else None, "gt_lines": n_gt,
            "label_cer": errs / chars if chars else None}


def score_all(items, scored_gt, rule, vns=None):
    """{"pooled": stats, <set>: stats, ...} for one rule (restricted to the VNs in vns)."""
    decisions = [vote.accept(it[2], rule)[0] for it in items]
    ok_vn = (lambda vn: True) if vns is None else (lambda vn: vn in vns)
    out = {"pooled": _aggregate(items, decisions, scored_gt, rule, lambda it: ok_vn(it[1]))}
    for name in sorted({it[0] for it in items}):
        out[name] = _aggregate(items, decisions, scored_gt, rule, lambda it, n=name: it[0] == n and ok_vn(it[1]))
    return out


def score(items, scored_gt, rule, vns=None):
    return score_all(items, scored_gt, rule, vns)["pooled"]


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
                                # trunc: any dissenting voter, or only Screen AI (it reads the whole
                                # screenshot, so it sees glyphs the line's own box left out)
                                for trunc in (False, True, ("screenai",)):
                                    yield {"primary": primary, "voters": voters, "votes": votes,
                                           "min_conf": min_conf, "det_conf": det_conf, "veto": veto,
                                           "trunc": trunc}


def choose(items, scored_gt, target, vns=None, cache=None):
    """The rule with the highest pooled recall whose precision reaches `target` on the
    pooled lines and on every set separately (so a rule must hold on Vita and PSP text
    alike); ties go to more required votes, then fewer voters (cheaper relabel)."""
    best = None
    for rule in grid():
        ss = score_all(items, scored_gt, rule, vns)
        if cache is not None:
            cache.append((rule, ss))
        if any(v["precision"] is None or v["precision"] < target for v in ss.values()):
            continue
        s = ss["pooled"]
        k = (round(s["recall"], 6), rule["votes"], s["precision"], -len(rule["voters"]))
        if best is None or k > best[0]:
            best = (k, rule, s)
    return best


def pipeline_engines(items):
    """Each engine's accuracy on the matched, scored teacher lines (its reading of the
    teacher's view; meiki = MeikiOCR on its own detector box)."""
    out = {}
    for e in ENGINES:
        n = exact = errs = chars = 0
        for it in items:
            ln, truth = it[2], _m(it, {"primary": "meiki"})[0]
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


def dump(items, scored_gt, rule, path):
    """Per-line diagnostics for one rule: every scored or partial teacher line with all
    readings, the decision and the truth, plus truth lines no teacher line covers."""
    from teacher import build
    rows, covered = [], set()
    for it in items:
        name, vn, ln, _ = it
        truth, ids, status = _m(it, rule)
        lab, how = vote.accept(ln, rule)
        covered.update(ids)
        if status == "unmatched" and lab is None:
            continue
        reads = {e: vote.reading(ln, e)[0] for e in ENGINES if vote.reading(ln, e) is not None}
        rows.append({"set": name, "vn": vn, "status": status, "gt": ids, "truth": truth, "label": lab,
                     "how": how, "ok": None if truth is None or lab is None else
                     vote.eval_norm(lab) == vote.eval_norm(truth), "crop": build.crop_of(ln, rule),
                     "det_conf": ln.get("det_conf"),
                     "conf": vote.low_conf(vote.reading(ln, rule.get("primary", "meikiv"))[1])
                     if vote.reading(ln, rule.get("primary", "meikiv")) else None, "reads": reads})
    missed = [{"gt": g["id"], "text": g["text"], "box": g["xyxy"]} for i, g in scored_gt.items() if i not in covered]
    Path(path).write_text(json.dumps({"rule": rule, "lines": rows, "uncovered_truth": missed}, ensure_ascii=False,
                                     indent=1) + "\n", encoding="utf-8")


def _rel(path):
    """A manifest path as the repo sees it: repo-relative, else under $TEACHER_OUT."""
    from teacher.label import ROOT
    path = Path(path).resolve()
    for base, name in ((ROOT.resolve(), ""), (OUT.resolve(), "$TEACHER_OUT/")):
        if base in path.parents:
            return name + str(path.relative_to(base))
    return str(path)


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
        n_det = len({i for x in it for i in _m(x, None)[1] if i in sg})
        st = Counter(_m(x, None)[2] for x in it)
        print(f"{name}: {len(shots)} screenshots, {len(it)} teacher lines ({dict(st)}); {len(sg)} scored truth "
              f"lines, {n_det} overlapped by a teacher line ({100 * n_det / max(1, len(sg)):.1f}%)", flush=True)
    report = {"sets": sets, "engines": {n: pipeline_engines(it) for n, (it, _) in per_set.items()}}
    print("\nper-engine accuracy on the matched lines (each engine reading the teacher's view):")
    for name, eng in report["engines"].items():
        for e, s in eng.items():
            print(f"  {name:9s} {e:9s} lines {s['lines']:4d}  CER {s['cer']}  exact {s['exact']}")
    cache = []
    best = choose(items, scored, args.precision, cache=cache)
    if best is None:
        sys.exit(f"no rule reaches precision {args.precision}")
    _, chosen, s = best
    rule = chosen
    all_sets = score_all(items, scored, rule)
    report["chosen"] = {"rule": rule, "pooled": s, "target_precision": args.precision,
                        "precision_ci95": wilson(s["correct"], s["accepted_scored"]),
                        "per_set": {n: all_sets[n] for n in sets if n in all_sets},
                        "per_set_ci95": {n: wilson(all_sets[n]["correct"], all_sets[n]["accepted_scored"])
                                         for n in sets if n in all_sets}}
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
            "meikiv + v5s + manga (label_real.py's engines)": {"voters": ("v5s", "manga"), "votes": 2},
            "meiki + 1 of v6s/screenai": {"primary": "meiki", "voters": ("v6s", "screenai"), "votes": 1},
            "meiki + v6s + screenai (both)": {"primary": "meiki", "voters": ("v6s", "screenai"), "votes": 2},
            "meiki + 2 of v6s/manga/screenai/v5s, veto, trunc": {
                "primary": "meiki", "voters": ("v6s", "manga", "screenai", "v5s"), "votes": 2, "veto": True,
                "trunc": True}}
    report["reference"] = {}
    print("\nreference rules:")
    for name, rule in refs.items():
        s = score(items, scored, rule)
        report["reference"][name] = {"rule": rule, **s}
        print(f"  {name:48s} {fmt(s)}")
    # the precision/recall frontier of the grid (pooled), from the scores choose() computed
    pts = []
    for rule, ss in cache:
        s = ss["pooled"]
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
    if args.rule_out:
        import hashlib
        mans = {n: str(eval_manifest(n)) for n in sets}
        Path(args.rule_out).write_text(json.dumps({
            "rule": chosen,
            "chosen_by": f"teacher/measure.py rules --set {args.set} --precision {args.precision}: the highest pooled "
                         f"recall whose precision is >= {args.precision} on {' and '.join(sets)} each and pooled",
            "measured": {"pooled": report["chosen"]["pooled"], "precision_ci95": report["chosen"]["precision_ci95"],
                         "per_set": report["chosen"]["per_set"], "per_set_ci95": report["chosen"]["per_set_ci95"],
                         "two_fold_held_out": [{"chosen_on_half": f["chosen_on"], "rule": f["rule"],
                                                "held_out": f["held_out"]} for f in held if f]},
            "eval_manifests": {n: {"path": _rel(m), "sha256": hashlib.sha256(Path(m).read_bytes()).hexdigest()}
                               for n, m in mans.items()},
        }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.rule_out}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        dump(items, scored, chosen, Path(args.out).with_suffix(".lines.json"))
        print(f"\nwrote {args.out} and {Path(args.out).with_suffix('.lines.json')}")


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / max(union, 1)


def cmd_hand(args):
    """Precision of the rule's accepted labels on lines realdata checked by hand (its
    manifests/handcheck.jsonl: training-side screenshots, dialogue and UI alike, so this is
    independent of eval's test games). A teacher line is compared with the hand line of the
    same screenshot whose ink box overlaps its crop most (IoU >= 0.5); hand lines marked
    unreadable (truth null) are not scored."""
    from teacher import build
    rule = json.loads(Path(args.rule).read_text(encoding="utf-8"))["rule"]
    hand = [json.loads(x) for x in Path(args.hand).read_text(encoding="utf-8").splitlines() if x.strip()]
    shots = build.load_reads(args.reads)
    by_shot = {}
    for h in hand:
        by_shot.setdefault(h["id"].rsplit("_", 1)[0], []).append(h)
    n = ok = acc = 0
    bad = []
    for sk, hs in by_shot.items():
        r = shots.get(sk)
        if r is None:
            continue
        for ln in r["lines"]:
            lab, how = vote.accept(ln, rule)
            if lab is None:
                continue
            box = ln.get("crop") or ln["ink"]
            h = max(hs, key=lambda h: iou(box, h["ink"]))
            if iou(box, h["ink"]) < 0.5:
                continue
            acc += 1
            if h["truth"] is None:
                continue
            n += 1
            good = vote.eval_norm(lab) == vote.eval_norm(h["truth"])
            ok += good
            if not good:
                bad.append((sk, lab, h["truth"], how))
    print(f"hand-checked lines (realdata dev set): {len(hand)} in {len(by_shot)} screenshots, "
          f"{sum(1 for s in by_shot if s in shots)} of them read; accepted {acc}, scored {n}, correct {ok} "
          f"(precision {100 * ok / max(1, n):.2f}%, 95% CI {wilson(ok, n)})")
    for b in bad:
        print("  wrong:", b)
    return {"hand_lines": len(hand), "accepted": acc, "scored": n, "correct": ok, "ci95": wilson(ok, n),
            "wrong": [list(b) for b in bad]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("hand")
    p.add_argument("--hand", default=str(OUT / "inputs" / "realdata_handcheck.jsonl"))
    p.add_argument("--rule", default=str(HERE / "rule.json"))
    p.add_argument("--reads", default=str(OUT / "reads"))
    p.add_argument("--out", default="")
    p = sub.add_parser("rules")
    p.add_argument("--set", default=",".join(EVAL_SETS))
    p.add_argument("--reads", default=str(OUT / "reads"))
    p.add_argument("--precision", type=float, default=0.995)
    p.add_argument("--out", default="")
    p.add_argument("--rule-out", default="", help="write the chosen rule and its measurements (teacher/rule.json)")
    args = ap.parse_args()
    if args.cmd == "hand":
        res = cmd_hand(args)
        if args.out:
            Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    else:
        cmd_rules(args)


if __name__ == "__main__":
    main()
