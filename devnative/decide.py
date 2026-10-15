#!/usr/bin/env python3
"""Decisions -> labels for the devnative cohort (and, with --mode train, for training pools).

  decide.py parse  --batch DIR                      validate DIR/decisions.jsonl -> DIR/decisions.parsed.jsonl
  decide.py queue  --pass1 DIR... --out Q.json      lines pass 2 must review (+ pass 1's text, unmarked)
  decide.py final  --pass1 DIR... [--pass2 DIR...] [--resolve R.jsonl] [--mode dev|train]
                   [--blind DIR...] [--confirm DIR...] [--auto-consensus] [--out $OUT/labels.jsonl]
  decide.py sample --labels L --n 300 --seed 20261012 --out B.json      random kept lines for the blind re-check
  decide.py blind  --labels L --batch DIR... --out REPORT.json          agreement of the blind pass

Every decision is the assistant's (Claude Opus 5.5 reading the contact sheets; reviewer_kind
"assistant"); no human checked these labels.

Pass 1 decides every candidate line on its sheet: pick a candidate, type the text, unsure, or
skip (ui / cut / bad). A pass-1 label is
  consensus  when both engine families read it (MeikiOCR: M or m; PP-OCRv6 small: V or v; the
             agreement key is eval's scoring normalisation, teacher/vote.key),
  single     when only one family read it,
  changed    when no engine read it (the reviewer corrected the text).
Development rule (--mode dev, devnative): consensus labels are kept; single and changed labels
are kept only when an independent second pass (other sessions, candidates shuffled and
unattributed, pass 1's text among them unmarked) decides the same key; anything else (unsure,
skipped, or the passes disagree) is dropped: nobody could settle it.
Training rule (--mode train, for realplus): a label needs at least one engine family after
review (changed labels are dropped; pass 2 optional, applied when given); --auto-consensus
also keeps, without review, lines all four readings agree on.
--resolve applies a third look after the blind re-check: {"id", "keep": bool, "text"?, "why" (a short
code), "note"?}.
--confirm: independent blind reads of the final crops (sheets.py mode blind with frames); a kept
line read there keeps its label only when the confirming read has the label's key, else it is
dropped (confirm-disagree / confirm-unsure / confirm-skip) unless a --resolve entry keeps it.
Every kept record counts its independent agreeing reads in "looks" (pass 1, an agreeing pass 2,
blind or confirm read, a keeping third look); --blind batches only count, they never change a label.
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from devnative import paths as P  # noqa: E402
from devnative import sheets as S  # noqa: E402
from teacher import vote  # noqa: E402

SKIPS = {"ui", "cut", "bad"}
ALIASES = {"p": "pick", "t": "text", "u": "unsure", "s": "skip"}


class DecisionError(ValueError):
    pass


# ----------------------------------------------------------------------------- parsing

def load_batch(d):
    d = Path(d)
    batch = json.loads((d / "batch.json").read_text(encoding="utf-8"))
    sheets = {}
    for p in sorted(d.glob("S*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        sheets[s["sheet"]] = {e["n"]: e for e in s["entries"]}
    return batch, sheets


def parse_decisions(text, sheets, mode, batch_name):
    """Validate a decisions.jsonl text against the batch's sheets. Returns one record per entry:
    {id, sheet, n, decision, text, skip, name, note}. Raises DecisionError listing every problem."""
    errs, got, defaults = [], {}, set()
    for i, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            d = json.loads(raw)
        except ValueError as e:
            errs.append(f"line {i}: not JSON ({e})")
            continue
        d = {ALIASES.get(k, k): v for k, v in d.items()}
        sid = d.get("sheet")
        if sid not in sheets:
            errs.append(f"line {i}: unknown sheet {sid!r}")
            continue
        if "default" in d:
            if mode not in ("pass1", "grad", "edge") or d["default"] != 1:
                errs.append(f"line {i}: default is only {{'default': 1}} in pass 1 and the GRAD / edge checks")
            defaults.add(sid)
            continue
        n = d.get("n")
        if n not in sheets[sid]:
            errs.append(f"line {i}: entry {n!r} is not on sheet {sid}")
            continue
        entry = sheets[sid][n]
        kinds = [k for k in ("pick", "text", "unsure", "skip") if k in d and d[k] not in (None, False, "")]
        if len(kinds) != 1:
            errs.append(f"line {i}: {sid} #{n} needs exactly one of pick / text / unsure / skip, got {kinds}")
            continue
        k = kinds[0]
        rec = {"id": entry["id"], "sheet": sid, "n": n, "decision": k, "text": None, "skip": None,
               "name": bool(d.get("name")), "note": d.get("note"), "batch": batch_name, "mode": mode,
               "reviewer_kind": "assistant"}
        if k == "pick":
            c = d["pick"]
            cands = {x["c"]: x["text"] for x in entry["candidates"]}
            if c not in cands:
                errs.append(f"line {i}: {sid} #{n} has no candidate {c!r}")
                continue
            rec["text"], rec["pick"] = vote.canon(cands[c]), c
        elif k == "text":
            t = vote.canon(str(d["text"]))
            if not vote.key(t):
                errs.append(f"line {i}: {sid} #{n} text is empty after normalisation")
                continue
            rec["text"] = t
        elif k == "skip":
            if d["skip"] not in SKIPS:
                errs.append(f"line {i}: {sid} #{n} skip must be one of {sorted(SKIPS)}")
                continue
            rec["skip"] = d["skip"]
        if entry["id"] in got and got[entry["id"]] != rec:
            errs.append(f"line {i}: {sid} #{n} decided twice, differently")
            continue
        got[entry["id"]] = rec
    for sid in defaults:
        for n, entry in sheets[sid].items():
            if entry["id"] in got:
                continue
            if len(entry["candidates"]) != 1:
                errs.append(f"{sid} #{n}: {len(entry['candidates'])} candidates, needs an explicit decision")
                continue
            got[entry["id"]] = {"id": entry["id"], "sheet": sid, "n": n, "decision": "pick",
                                "text": vote.canon(entry["candidates"][0]["text"]), "skip": None, "pick": 1,
                                "name": False, "note": None, "batch": batch_name, "mode": mode,
                                "reviewer_kind": "assistant", "via": "default"}
    missing = [f"{sid} #{n}" for sid, es in sheets.items() for n, e in es.items() if e["id"] not in got]
    if missing:
        errs.append(f"{len(missing)} entries without a decision: {', '.join(missing[:20])}"
                    + (" ..." if len(missing) > 20 else ""))
    if errs:
        raise DecisionError("\n".join(errs))
    return [got[e["id"]] for sid in sorted(sheets) for _, e in sorted(sheets[sid].items())]


def parsed(d):
    """Parsed decisions of a batch directory (parses and caches them)."""
    d = Path(d)
    batch, sheets = load_batch(d)
    recs = parse_decisions((d / "decisions.jsonl").read_text(encoding="utf-8"), sheets, batch["mode"], d.name)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    cache = d / "decisions.parsed.jsonl"
    if not cache.exists() or cache.read_text(encoding="utf-8") != text:    # frozen batches stay untouched
        cache.write_text(text, encoding="utf-8")
    return recs


def batches(dirs):
    """Batch directories: each given dir, or its b* sub-directories when it was chunked."""
    out = []
    for d in dirs:
        d = Path(d)
        out += sorted(x for x in d.glob("b*") if (x / "batch.json").exists()) if not (d / "batch.json").exists() \
            else [d]
    return out


def collect(dirs):
    out = {}
    for d in batches(dirs):
        for r in parsed(d):
            out[r["id"]] = r
    return out


# ----------------------------------------------------------------------------- rules

def classify(label, line):
    """(how, support): consensus / single / changed for a label against a line's readings."""
    sup = S.support(label, line)
    return ("consensus" if len(sup) == 2 else "single" if sup else "changed"), sup


def readers(label, line):
    """{engine: bool}: which readings have the label's key (no text)."""
    k = vote.key(label)
    return {e: vote.key(t) == k for e, t in S.readings(line).items()}


def needs_pass2(rec, line):
    return rec["decision"] in ("pick", "text") and classify(rec["text"], line)[0] != "consensus"


def decide_line(lid, line, p1, p2, mode, auto_consensus=False, resolution=None):
    """Final record of one line from its pass-1 and pass-2 decisions (either may be None)."""
    out = {"id": lid, "status": "dropped", "why": None, "label": None, "how": None, "support": [],
           "readers": {}, "name": bool(p1 and p1.get("name")), "reviewer_kind": "assistant",
           "pass1": None, "pass2": None}
    cands = S.candidates(line)
    if p1 is None:
        if auto_consensus and len(cands) == 1 and len(cands[0]["engines"]) == len(S.ENGINES):
            out.update(status="kept", label=cands[0]["text"], how="auto", support=["meiki", "v6s"],
                       readers=readers(cands[0]["text"], line))
        else:
            out["why"] = "undecided"
        return out
    out["pass1"] = {k: p1.get(k) for k in ("decision", "skip", "pick", "batch", "sheet", "n", "via") if p1.get(k)}
    if p1["decision"] == "skip":
        out["why"] = f"skip-{p1['skip']}"
        return out
    if p1["decision"] == "unsure":
        out["why"] = "pass1-unsure"
        return out
    label = p1["text"]
    how, sup = classify(label, line)
    out.update(how=how, support=sup, readers=readers(label, line))
    if p2 is not None:
        agree = p2["decision"] in ("pick", "text") and vote.key(p2["text"]) == vote.key(label)
        out["pass2"] = {"decision": p2["decision"], "skip": p2.get("skip"), "agree": agree, "batch": p2["batch"]}
    if mode == "train":
        if how == "changed":
            out["why"] = "no-engine-support"
        elif out["pass2"] is not None and not out["pass2"]["agree"]:
            out["why"] = "pass2-disagree"
        else:
            out.update(status="kept", label=label)
    elif how == "consensus":
        out.update(status="kept", label=label)
    elif out["pass2"] is None:
        out["why"] = "pass2-missing"
    elif out["pass2"]["agree"]:
        out.update(status="kept", label=label)
    else:
        out["why"] = {"unsure": "pass2-unsure", "skip": "pass2-skip"}.get(p2["decision"], "pass2-disagree")
    if resolution is not None and out["status"] == "kept":
        out["resolution"] = {k: resolution.get(k) for k in ("keep", "why", "note") if k in resolution}
        if not resolution.get("keep", True):
            out.update(status="dropped", why=f"blind-{resolution.get('why') or 'resolved'}", label=None)
        elif resolution.get("text"):
            t = vote.canon(resolution["text"])
            out.update(label=t, readers=readers(t, line))
            out["how"], out["support"] = classify(t, line)
            out["resolution"]["changed_label"] = True
    return out


def agrees(dec, label):
    return dec is not None and dec["decision"] in ("pick", "text") and vote.key(dec["text"]) == vote.key(label)


def apply_confirm(rec, conf, blind):
    """Apply an independent confirming read (or none) to a final record, and count its looks."""
    if rec["status"] != "kept":
        return rec
    if conf is not None:
        rec["confirm"] = {"decision": conf["decision"], "agree": agrees(conf, rec["label"]), "batch": conf["batch"]}
        if not rec["confirm"]["agree"] and not (rec.get("resolution") or {}).get("keep"):
            why = {"unsure": "confirm-unsure", "skip": "confirm-skip"}.get(conf["decision"], "confirm-disagree")
            rec.update(status="dropped", label=None, why=why)
            return rec
    if blind is not None:
        rec["blind"] = {"decision": blind["decision"], "agree": agrees(blind, rec["label"])}
    rec["looks"] = 1 + bool((rec.get("pass2") or {}).get("agree")) + bool((rec.get("blind") or {}).get("agree")) \
        + bool((rec.get("confirm") or {}).get("agree")) + bool((rec.get("resolution") or {}).get("keep"))
    return rec


def cmd_parse(args):
    try:
        recs = parsed(args.batch)
    except DecisionError as e:
        print(e, flush=True)
        sys.exit(1)
    print(json.dumps(dict(Counter(r["decision"] for r in recs))), flush=True)


def cmd_queue(args):
    shots = S.load_reads(args.reads)
    idx = S.line_index(shots)
    p1 = collect(args.pass1)
    ids, extra, why = [], {}, Counter()
    for lid, r in sorted(p1.items()):
        line = idx[lid][1]
        if needs_pass2(r, line):
            ids.append(lid)
            extra[lid] = r["text"]
            why[classify(r["text"], line)[0]] += 1
    Path(args.out).write_text(json.dumps({"ids": ids, "extra": extra}, ensure_ascii=False, indent=1) + "\n",
                              encoding="utf-8")
    print(f"{len(ids)} lines for pass 2: {dict(why)} -> {args.out}", flush=True)


def cmd_final(args):
    shots = S.load_reads(args.reads)
    idx = S.line_index(shots)
    p1 = collect(args.pass1)
    p2 = collect(args.pass2 or [])
    res = {}
    for f in args.resolve or []:
        for x in Path(f).read_text(encoding="utf-8").splitlines():
            if x.strip():
                r = json.loads(x)
                res[r["id"]] = r
    ids = sorted(set(p1) | ({lid for lid, (s, ln) in idx.items() if S.is_candidate(ln)[0]}
                            if args.auto_consensus else set()))
    blind, conf = collect(args.blind or []), collect(args.confirm or [])
    out = []
    for lid in ids:
        s, ln = idx[lid]
        rec = decide_line(lid, ln, p1.get(lid), p2.get(lid), args.mode, args.auto_consensus, res.get(lid))
        rec.update(shot=s["shot"], vn=s["vn"], subset=s["subset"], kind=s["kind"])
        apply_confirm(rec, conf.get(lid), blind.get(lid))
        out.append(rec)
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    st = Counter(f"{r['status']}:{r['how'] if r['status'] == 'kept' else r['why']}" for r in out)
    looks = Counter(r.get("looks") for r in out if r["status"] == "kept")
    print(json.dumps({"lines": len(out), "kept": sum(r["status"] == "kept" for r in out), "by": dict(st),
                      "kept_by_looks": {str(k): v for k, v in sorted(looks.items())}}, indent=1), flush=True)


def cmd_sample(args):
    rows = [json.loads(x) for x in Path(args.labels).read_text(encoding="utf-8").splitlines() if x.strip()]
    kept = sorted(r["id"] for r in rows if r["status"] == "kept")
    rng = random.Random(args.seed)
    ids = sorted(rng.sample(kept, min(args.n, len(kept))))
    Path(args.out).write_text(json.dumps({"ids": ids, "seed": args.seed, "from": len(kept)}, indent=1) + "\n")
    print(f"{len(ids)} of {len(kept)} kept lines -> {args.out}", flush=True)


def wilson95(a, n):
    """Wilson 95% interval of a rate a/n, rounded to 4 places."""
    import math
    p, z = a / n, 1.96
    den = 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / den, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def blind_agreement(labels, blind):
    """labels {id: final record}; blind {id: parsed blind decision}. Agreement = the blind pass
    reads the final label's key."""
    rows = []
    for lid, b in sorted(blind.items()):
        lab = labels[lid]
        agree = b["decision"] in ("pick", "text") and vote.key(b["text"]) == vote.key(lab["label"])
        rows.append({"id": lid, "agree": agree, "blind": b["decision"], "how": lab["how"], "subset": lab["subset"],
                     "kind": lab["kind"], "label": lab["label"], "blind_text": b.get("text"), "skip": b.get("skip")})
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        for k in ("all", f"subset:{r['subset']}", f"kind:{r['kind']}", f"how:{r['how']}"):
            by[k][0] += r["agree"]
            by[k][1] += 1
    summ = {k: {"agree": a, "lines": n, "rate": round(a / n, 4)} for k, (a, n) in sorted(by.items())}
    if rows:
        summ["all"]["wilson95"] = wilson95(*by["all"])
    return summ, [r for r in rows if not r["agree"]]


def cmd_blind(args):
    labels = {}
    for x in Path(args.labels).read_text(encoding="utf-8").splitlines():
        if x.strip():
            r = json.loads(x)
            labels[r["id"]] = r
    blind = collect(args.batch)
    summ, dis = blind_agreement(labels, blind)
    Path(args.out).write_text(json.dumps({"summary": summ, "disagreements": dis}, ensure_ascii=False, indent=1)
                              + "\n", encoding="utf-8")
    print(json.dumps(summ, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reads", default=str(P.OUT / "reads" / "reads.jsonl"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse")
    p.add_argument("--batch", required=True)
    p = sub.add_parser("queue")
    p.add_argument("--pass1", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("final")
    p.add_argument("--pass1", nargs="+", required=True)
    p.add_argument("--pass2", nargs="*")
    p.add_argument("--resolve", nargs="*")
    p.add_argument("--blind", nargs="*", help="blind re-check batches (counted in looks; never change a label)")
    p.add_argument("--confirm", nargs="*", help="independent confirming reads: a disagreeing one drops the line")
    p.add_argument("--mode", choices=("dev", "train"), default="dev")
    p.add_argument("--auto-consensus", action="store_true")
    p.add_argument("--out", default=str(P.OUT / "labels.jsonl"))
    p = sub.add_parser("sample")
    p.add_argument("--labels", default=str(P.OUT / "labels.jsonl"))
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=20261012)
    p.add_argument("--out", required=True)
    p = sub.add_parser("blind")
    p.add_argument("--labels", default=str(P.OUT / "labels.jsonl"))
    p.add_argument("--batch", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"parse": cmd_parse, "queue": cmd_queue, "final": cmd_final, "sample": cmd_sample,
     "blind": cmd_blind}[args.cmd](args)


if __name__ == "__main__":
    main()
