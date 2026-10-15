#!/usr/bin/env python3
"""realplus v3: Codex gpt-6-luna as the second, independent vote on every v3 pass-1 sheet.

The owner's rule (2026-10-13 05:40Z): a new training label needs the Claude reviewer and Luna to
agree; Luna's loose cut / ui calls only send a line back to Claude, never into the pool. The exact
rule is $REALPLUS/AGREEMENT.md, fixed before the first v3 batch was reviewed; `vote()` implements it.

  luna.py run   [--max 6] [--loop]      codex_sheet.sh on every v3 pass-1 sheet without a vote yet,
                                        at most --max Codex calls at a time -> $REALPLUS/luna/pass1/bNNNN/Sxxx.jsonl
  luna.py merge                         Claude's decisions x Luna's votes per line -> luna/pass1/bNNNN/votes.jsonl
  luna.py stats [--batches 10]          agreement on the first N v3 batches (reported in docs/realplus.md)

One fresh Codex session per contact sheet (README.md + Sxxx.json + Sxxx.png), so Luna never sees the
Claude decision; Claude never sees Luna's. Only public VNDB-sourced crops are sent (every realplus
screenshot is one). Votes carry reviewer_kind "codex:gpt-6-luna".
"""
import argparse
import collections
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import shots as SH  # noqa: E402

from devnative import decide as D  # noqa: E402
from teacher import vote as V  # noqa: E402

OUT = SH.realplus_dir()
LUNA = OUT / "luna"
HERD_DIR = Path(os.environ.get("HERD_DIR", "/local/home/skerraut/herd-synth3"))
SHEET_SH = HERD_DIR / "tools" / "codex_sheet.sh"
MODEL = os.environ.get("CODEX_MODEL", "gpt-6-luna")
KIND = f"codex:{MODEL}"
FIRST = int(os.environ.get("REALPLUS_V3_PASS1", "18"))     # pool.V3_FIRST["pass1"]
MAX_CALLS = 6                                               # the owner's limit on concurrent Codex calls
EDGE_PUNCT = set("「」『』。、！？")


def jl(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def v3_batches(kind="pass1", first=None):
    """The v3 batches of a kind: review/<kind>/bNNNN with NNNN >= first (default pool.V3_FIRST)."""
    first = FIRST if first is None else first
    d = OUT / "review" / kind
    return sorted(b for b in d.glob("b*") if (b / "batch.json").exists() and int(b.name[1:]) >= first) \
        if d.exists() else []


def sheets_of(b):
    return sorted(p.stem for p in Path(b).glob("S*.json"))


def luna_dir(b):
    return LUNA / Path(b).parent.name / Path(b).name


# ----------------------------------------------------------------------------- run

def run_sheet(b, sid):
    """One Codex call (codex_sheet.sh: 3 tries with backoff inside). Returns (sheet, ok, seconds)."""
    out = luna_dir(b) / f"{sid}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = out.with_suffix(".log")
    with open(log, "ab") as lf:
        rc = subprocess.run([str(SHEET_SH), str(b), sid, str(out)], stdin=subprocess.DEVNULL, stdout=lf,
                            stderr=subprocess.STDOUT, env=dict(os.environ, CODEX_MODEL=MODEL), timeout=2400).returncode
    ok = rc == 0 and out.exists() and out.stat().st_size > 0
    if not ok and out.exists() and out.stat().st_size == 0:
        out.unlink()
    with open(luna_dir(b) / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"sheet": sid, "ok": ok, "rc": rc, "seconds": round(time.time() - t0, 1),
                            "model": MODEL, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
    return sid, ok, time.time() - t0


def runs_of(b):
    c = collections.Counter()
    for r in jl(luna_dir(b) / "runs.jsonl"):
        c[r["sheet"]] += 1
    return c


def incomplete(b, sid, sheets):
    """True when the sheet's vote file lacks a decision for some entry (AGREEMENT.md rule 5: retried)."""
    p = luna_dir(b) / f"{sid}.jsonl"
    got, _ = parse_lenient(p.read_text(encoding="utf-8"), {sid: sheets[sid]}, "pass1", Path(b).name)
    return any(e["id"] not in got for e in sheets[sid].values())


def todo_sheets(batches, retries=2):
    """Sheets without a vote file, plus sheets whose vote misses entries and were run fewer than
    `retries` times."""
    out = []
    for b in batches:
        runs, sheets = runs_of(b), None
        for sid in sheets_of(b):
            if not (luna_dir(b) / f"{sid}.jsonl").exists():
                out.append((b, sid))
            elif runs[sid] < retries:
                sheets = sheets or D.load_batch(b)[1]
                if incomplete(b, sid, sheets):
                    (luna_dir(b) / f"{sid}.jsonl").rename(luna_dir(b) / f"{sid}.run{runs[sid]}.jsonl")
                    out.append((b, sid))
    return out


def cmd_run(args):
    """Vote on every v3 pass-1 sheet that has no vote file yet, --max calls at a time."""
    mx = min(args.max, MAX_CALLS)
    while True:
        todo = todo_sheets(v3_batches("pass1", args.first))
        if args.limit:
            todo = todo[:args.limit]
        if todo:
            print(f"{time.strftime('%H:%M:%S')} {len(todo)} sheets to vote on, {mx} at a time", flush=True)
            n_ok, n_bad, t0 = 0, 0, time.time()
            with cf.ThreadPoolExecutor(mx) as ex:
                for sid, ok, secs in ex.map(lambda bs: run_sheet(*bs), todo):
                    n_ok += ok
                    n_bad += not ok
                    if (n_ok + n_bad) % 20 == 0:
                        print(f"  {n_ok} ok, {n_bad} failed, {time.time() - t0:.0f}s", flush=True)
            print(f"{time.strftime('%H:%M:%S')} done: {n_ok} ok, {n_bad} failed", flush=True)
        if not args.loop:
            break
        time.sleep(args.every)


# ----------------------------------------------------------------------------- parse

def parse_lenient(text, sheets, mode, batch_name):
    """Luna's output, one record per entry it decided, through devnative's parser one line at a
    time (so a pick, a text, unsure and the skips mean exactly what they mean for Claude). Lines
    that do not parse are skipped and counted; {"default": 1} accepts candidate 1 of every
    single-candidate entry of the sheet that has no explicit line, as the README allows."""
    got, bad, defaults = {}, 0, set()
    for raw in text.splitlines():
        if not raw.strip():
            continue
        try:
            d = json.loads(raw)
            d = {D.ALIASES.get(k, k): v for k, v in d.items()}
            sid, n = d.get("sheet"), d.get("n")
            if "default" in d and sid in sheets:
                defaults.add(sid)
                continue
            entry = sheets[sid][n]
            # Luna sometimes writes {"pick": 1, "skip": "ui"} ("it reads X, but it is UI / cut"). That counts as
            # the skip: the line goes to a confirm look instead of into the pool (AGREEMENT.md, amendment 1).
            if ("skip" in d or d.get("unsure")) and ("pick" in d or "text" in d):
                d = {k: v for k, v in d.items() if k not in ("pick", "text")}
                d["note"] = (d.get("note") or "") + " [also read a candidate]"
            rec = D.parse_decisions(json.dumps(d, ensure_ascii=False), {sid: {n: entry}}, mode, batch_name)[0]
        except (ValueError, KeyError, TypeError, D.DecisionError):
            bad += 1
            continue
        rec["reviewer_kind"] = KIND
        got.setdefault(entry["id"], rec)
    for sid in defaults:
        for n, entry in sheets[sid].items():
            if entry["id"] not in got and len(entry["candidates"]) == 1:
                got[entry["id"]] = {"id": entry["id"], "sheet": sid, "n": n, "decision": "pick",
                                    "text": V.canon(entry["candidates"][0]["text"]), "skip": None, "pick": 1,
                                    "name": False, "note": None, "batch": batch_name, "mode": mode,
                                    "reviewer_kind": KIND, "via": "default"}
    return got, bad


def luna_votes(b):
    """{line id: Luna record} over a batch's vote files (the latest run of a sheet first, then earlier
    runs for entries it missed), plus the count of unparsable lines."""
    batch, sheets = D.load_batch(b)
    votes, bad = {}, 0
    for sid in sheets_of(b):
        files = [luna_dir(b) / f"{sid}.jsonl"] + sorted(luna_dir(b).glob(f"{sid}.run*.jsonl"), reverse=True)
        text = "\n".join(p.read_text(encoding="utf-8") for p in files if p.exists())
        if not text.strip():
            continue
        got, nb = parse_lenient(text, {sid: sheets[sid]}, batch["mode"], Path(b).name)
        votes.update(got)
        bad += nb
    return votes, bad


# ----------------------------------------------------------------------------- the rule

def read(dec):
    return dec is not None and dec["decision"] in ("pick", "text")


def edge_only(a, b):
    """True when two keys differ only by 「」『』。、！？ at either end."""
    def strip(k):
        i, j = 0, len(k)
        while i < j and k[i] in EDGE_PUNCT:
            i += 1
        while j > i and k[j - 1] in EDGE_PUNCT:
            j -= 1
        return k[i:j]
    return a != b and strip(a) == strip(b)


def vote(claude, luna):
    """AGREEMENT.md's rule for one line. Returns {"luna": status, "luna_vote": ..., "edge_only": bool}:
      agree        both read the same key (the line goes on to the training rule)
      disagree     both read, different keys: dropped
      skip-*/unsure  Claude read, Luna skipped or was unsure: a confirm look decides
      missing      no Luna vote for the entry (retry the sheet; dropped at the freeze)
      n/a          Claude skipped or was unsure: Luna's vote changes nothing"""
    out = {"luna": None, "luna_vote": None, "edge_only": False}
    if luna is not None:
        out["luna_vote"] = {k: luna.get(k) for k in ("decision", "skip", "pick", "text", "via") if luna.get(k)}
        out["luna_vote"]["reviewer_kind"] = luna.get("reviewer_kind", KIND)
    if not read(claude):
        out["luna"] = "n/a"
        return out
    if luna is None:
        out["luna"] = "missing"
    elif read(luna):
        a, b = V.key(claude["text"]), V.key(luna["text"])
        out["luna"] = "agree" if a == b else "disagree"
        out["edge_only"] = edge_only(a, b)
    elif luna["decision"] == "skip":
        out["luna"] = f"skip-{luna['skip']}"
    else:
        out["luna"] = "unsure"
    return out


def merge_batch(b):
    """[{id, claude: decision, luna: status, luna_vote, edge_only}] for a decided v3 batch, or None
    when Claude's decisions are not complete yet."""
    try:
        claude = {r["id"]: r for r in D.parsed(b)}
    except (D.DecisionError, ValueError, KeyError, FileNotFoundError):
        return None
    votes, bad = luna_votes(b)
    rows = []
    for lid, c in claude.items():
        rows.append({"id": lid, "claude": {k: c.get(k) for k in ("decision", "skip", "pick", "text", "name")
                                           if c.get(k) is not None}, **vote(c, votes.get(lid))})
    return rows, bad


def cmd_merge(args):
    """Write luna/pass1/bNNNN/votes.jsonl for every decided v3 batch (Claude done; Luna as far as voted)."""
    summ = {}
    for b in v3_batches("pass1", args.first):
        m = merge_batch(b)
        if m is None:
            continue
        rows, bad = m
        d = luna_dir(b)
        d.mkdir(parents=True, exist_ok=True)
        (d / "votes.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                       encoding="utf-8")
        summ[b.name] = dict(collections.Counter(r["luna"] for r in rows), unparsable=bad)
    print(json.dumps(summ, indent=1), flush=True)


def load_votes(first=None):
    """{line id: merged vote row} over every v3 batch with a votes.jsonl."""
    out = {}
    for b in v3_batches("pass1", first):
        for r in jl(luna_dir(b) / "votes.jsonl"):
            out[r["id"]] = r
    return out


# ----------------------------------------------------------------------------- stats

def agreement(rows):
    """Luna's agreement with Claude over lines Claude read (pick or text), and the rest."""
    read_ = [r for r in rows if r["luna"] != "n/a"]
    by = collections.Counter(r["luna"] for r in read_)
    n = len(read_)
    both = by["agree"] + by["disagree"]
    out = {"claude_read": n, "luna": dict(by),
           "agree_of_claude_read": [by["agree"], n, round(by["agree"] / n, 4) if n else None],
           "agree_of_both_read": [by["agree"], both, round(by["agree"] / both, 4) if both else None],
           "disagree_edge_only": sum(r["edge_only"] for r in read_ if r["luna"] == "disagree"),
           "luna_skipped_or_unsure": sum(r["luna"].startswith("skip") or r["luna"] == "unsure" for r in read_),
           "claude_skipped_or_unsure": len(rows) - n,
           "luna_read_where_claude_skipped": sum(1 for r in rows if r["luna"] == "n/a" and r.get("luna_vote")
                                                 and r["luna_vote"].get("decision") in ("pick", "text"))}
    if n:
        out["agree_of_claude_read_wilson95"] = D.wilson95(by["agree"], n)
    return out


def cmd_stats(args):
    bs = [b for b in v3_batches("pass1", args.first) if (luna_dir(b) / "votes.jsonl").exists()][:args.batches]
    rows = [r for b in bs for r in jl(luna_dir(b) / "votes.jsonl")]
    out = {"batches": [b.name for b in bs], "lines": len(rows), **agreement(rows)}
    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", type=int, default=FIRST, help="first v3 pass-1 batch number")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--max", type=int, default=MAX_CALLS)
    p.add_argument("--limit", type=int, default=0, help="at most this many sheets this round (0: all)")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--every", type=int, default=120)
    sub.add_parser("merge")
    p = sub.add_parser("stats")
    p.add_argument("--batches", type=int, default=10)
    p.add_argument("--out")
    args = ap.parse_args()
    {"run": cmd_run, "merge": cmd_merge, "stats": cmd_stats}[args.cmd](args)


if __name__ == "__main__":
    main()
