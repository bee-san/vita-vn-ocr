#!/usr/bin/env python3
"""Owner pool v2: the owner's console captures that no owner pool has read yet (docs/owner-pool.md, "Version 2").

Run with REALPLUS=NATIVE_POOL=<pool dir> NATIVE_OWNER_POOL=1 (as realplus/native.py), after owner_frames.py
pool-prep and devnative/read.py:

  owner_pool.py leak-drop LOG...                    leakcheck.py logs -> manifests/leak_drop.json, leak.json
  owner_pool.py batches [--n K] [--size 280]        pass-1 batches in pool.py plan's order -> review/pass1/bNNNN
  owner_pool.py final                               decide.py final (--mode train --auto-consensus, --confirm), then
                                                    the confirm rule (CONFIRM_RULE) -> labels.jsonl
  owner_pool.py confirm-batches [--n K] [--size 280] [--min 1]
                                                    confirm looks of the pending labels -> review/confirm/bNNNN
  owner_pool.py edge-batches [--n K] [--size 280] [--min 1]
                                                    edge checks of flagged final crops -> review/edge/bNNNN
  owner_pool.py helpers [--max 6] [--kinds check,confirm,edge,pass1] [--retry]
                                                    queue Claude review helpers (review-o2-N) for undecided batches
  owner_pool.py status                              batches made / decided / with a helper
  owner_pool.py cycle [--loop]                      final -> edge-boxes -> confirm / edge batches -> helpers, every
                                                    --every s (closes finished helpers, keeps them ahead in the queue)
  owner_pool.py merge --tag v2 --old DIR:TAG        one set of files: an earlier owner pool's lines plus this one's
  owner_pool.py publish --tag v2 [--check pre]      realplus/manifests/owner/stats_v2.json (aggregate only)

The labelling rule is fixed in the pool's AGREEMENT.md before any batch (docs/owner-pool.md quotes it):
  auto      the four readings agree: kept without review (devnative's training rule, --auto-consensus);
  pass 1    a Claude helper decides each other line; the label needs an engine family (decide.py --mode train);
  confirm   every pass-1 label needs an independent blind Claude read (another session; sheets.py mode blind framed
            on the final crop) with the same key (teacher/vote.key); anything else drops it. The frames are
            private, so the Codex Luna vote realplus v3 uses is not allowed: this second Claude look takes its place;
  edge      a flagged final crop needs an edge-check pick (as v1).
Nothing from these batches goes to Codex or any other model or third party: every helper is a Claude session
(reviewer_kind "assistant"), every README and launch message says so.
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pool as PL  # noqa: E402

from devnative import sheets as S  # noqa: E402
from teacher import vote  # noqa: E402

OUT, REVIEW = PL.OUT, PL.REVIEW
HELPER = os.environ.get("OWNER_HELPER", "review-o2-")
PRIVATE = ("These are the owner's PRIVATE console captures: never send anything from this batch to Codex or any other "
           "model or third party (its README says so).")
NOTE = ("\nNote (owner pool v2): these are the owner's own live console captures (PSP 480x272 and PS Vita 960x544\n"
        "framebuffers, lossless PNG), read at native size; decide them exactly as above. Speaker names: add\n"
        "\"name\": true. They are PRIVATE and copyrighted: never send any sheet, crop, frame or transcription\n"
        "from this batch to Codex, gpt-6-luna or any other model or third party. Images: view ONE sheet per tool\n"
        "call, each sheet once, at most 1280 px and ~60 images per session. Append the decisions of each sheet to\n"
        "decisions.jsonl in this directory as soon as that sheet is decided, so a fresh session can resume: if\n"
        "decisions.jsonl already exists, keep it and continue with the first sheet that has no decisions yet.\n")
CONFIRM_NOTE = ("This is a confirmation look (owner pool v2, AGREEMENT.md rule 3): another reviewer read these lines\n"
                "once. Decide each line again from the image alone, exactly as above: pick or type what the frame\n"
                "shows (a speaker name too, with \"name\": true); skip \"ui\", \"cut\" or \"bad\" only as defined\n"
                "above; unsure when you cannot tell.\n")
CONFIRM_RULE = "AGREEMENT.md rule 3: a pass-1 label is kept only with an agreeing confirm look"
KINDS = ("check", "confirm", "edge", "pass1")
MAX_TRIES = 3                      # helper sessions per batch (a re-queued one continues from decisions.jsonl)
HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))


def jl(path):
    return PL.jl(path)


def wjl(path, rows):
    PL.wjl(path, rows)


def note(out, *extra):
    with open(Path(out) / "README.md", "a", encoding="utf-8") as f:
        f.write(NOTE + "".join(extra))


# ----------------------------------------------------------------------------- pass 1

def leak_drop():
    """Shot keys the leak check dropped (manifests/leak_drop.json; native.py build leaves them out too)."""
    p = OUT / "manifests" / "leak_drop.json"
    return set(json.loads(p.read_text())) if p.exists() else set()


def leak_pairs(text):
    """[(corr, kind, test file name, pool shot key)] of a leakcheck.py log's pair lines, and its summary dict."""
    pairs, summ = [], {}
    for x in text.splitlines():
        p = x.split()
        if len(p) == 5 and p[3] == "~" and p[1] in ("same-vn", "cross-vn"):
            pairs.append((float(p[0]), p[1], p[2], Path(p[4]).stem))
        elif x.startswith("{"):
            summ = json.loads(x)
    return pairs, summ


def cmd_leak_drop(args):
    """manifests/leak_drop.json (every pool frame in a pair at or above the threshold: never labelled, never built)
    and manifests/leak.json (per log: the summary and the pair counts; no file names)."""
    drop, out = set(), {}
    for lp in args.logs:
        pairs, summ = leak_pairs(Path(lp).read_text(encoding="utf-8"))
        drop |= {k for _, _, _, k in pairs}
        out[Path(lp).stem] = {"summary": summ, "pool_frames_dropped": len({k for *_, k in pairs}),
                              "by_kind": dict(collections.Counter(kind for _, kind, _, _ in pairs))}
    (OUT / "manifests").mkdir(parents=True, exist_ok=True)
    (OUT / "manifests" / "leak_drop.json").write_text(json.dumps(sorted(drop)) + "\n")
    (OUT / "manifests" / "leak.json").write_text(json.dumps({"min_corr": 0.95, "logs": out,
                                                             "dropped": len(drop)}, indent=1) + "\n")
    print(json.dumps({"dropped": len(drop), "logs": out}, indent=1), flush=True)


def cmd_batches(args):
    """The next pass-1 batches of pool.py plan's review order (review/plan.jsonl), lines of one frame together;
    lines of a leak-dropped frame are never shown."""
    shots = PL.load_reads()
    idx = S.line_index(shots)
    drop = leak_drop()
    with PL.review_lock():
        done = PL.batched_ids()
        todo = [r["id"] for r in jl(REVIEW / "plan.jsonl")
                if r["id"] not in done and r["id"] in idx and idx[r["id"]][0]["shot"] not in drop]
        k = max([int(b.name[1:]) for b in PL.batch_dirs()] + [0])
        for _ in range(args.n):
            if not todo:
                break
            chunk, todo = todo[:args.size], todo[args.size:]
            chunk.sort(key=lambda lid: (idx[lid][0]["shot"], idx[lid][1]["ink"][1], idx[lid][1]["ink"][0]))
            k += 1
            out = REVIEW / "pass1" / f"b{k:04d}"
            entries = [(lid, idx[lid][0], idx[lid][1], S.candidates(idx[lid][1])) for lid in chunk]
            ns = S.make_batch(entries, out, "pass1", name=f"owner2/pass1/{out.name}")
            note(out)
            print(f"{out}: {len(chunk)} lines, {ns} sheets", flush=True)


# ----------------------------------------------------------------------------- final

def confirm_rule(labels):
    """AGREEMENT.md rule 3 on decide.py's records (it has already dropped a disagreeing confirm look): a kept label
    pass 1 decided needs an agreeing confirm look, else it waits ("pending", confirm-missing); an auto line needs
    none. Every record gets reviewer_kinds ["assistant"]. Returns the counts."""
    c = collections.Counter()
    for r in labels:
        r["reviewer_kinds"] = ["assistant"]
        if r["status"] == "kept" and r.get("pass1") is not None and not (r.get("confirm") or {}).get("agree"):
            r.update(status="pending", why="confirm-missing")
        c[f"{r['status']}:{r['how'] if r['status'] == 'kept' else r['why']}"] += 1
    return dict(c)


def cmd_final(args):
    """decide.py final (devnative's training rule, --mode train --auto-consensus, the confirm looks and any third
    looks), run in-process so it also works before the first pass-1 batch is decided (auto lines only), then
    CONFIRM_RULE."""
    from devnative import decide as D
    p1, conf = PL.done_batches("pass1"), PL.done_batches("confirm")
    res = sorted(REVIEW.glob("resolutions*.jsonl"))
    D.cmd_final(argparse.Namespace(reads=str(PL.reads_snapshot()), pass1=[str(b) for b in p1], pass2=[],
                                   resolve=[str(r) for r in res], blind=[], confirm=[str(b) for b in conf],
                                   mode="train", auto_consensus=True, out=str(OUT / "labels.jsonl")))
    labels = jl(OUT / "labels.jsonl")
    if res:
        n = PL.apply_auto_resolutions(labels, {r["id"]: r for f in res for r in jl(f)})
        print(f"third looks applied to {n} auto lines", flush=True)
    c = confirm_rule(labels)
    wjl(OUT / "labels.jsonl", labels)
    print(json.dumps({"rule": CONFIRM_RULE, "pass1_batches": len(p1), "confirm_batches": len(conf), "by": c},
                     indent=1), flush=True)


# ----------------------------------------------------------------------------- confirm looks and edge checks

class CapCount:
    """Lines per (VN, key) and per key against realplus's repeat caps (pool.PER_VN_CAP, GLOBAL_CAP): the build keeps at
    most that many copies of a label, so a look at a further copy cannot add a line."""
    def __init__(self, rows=()):
        self.vn, self.all = collections.Counter(), collections.Counter()
        for vn, label in rows:
            self.add(vn, label)

    def full(self, vn, label):
        k = vote.key(label)
        return self.vn[(vn, k)] >= PL.PER_VN_CAP or self.all[k] >= PL.GLOBAL_CAP

    def add(self, vn, label):
        k = vote.key(label)
        self.vn[(vn, k)] += 1
        self.all[k] += 1


def confirm_entries(labels, boxes, have, shots):
    """Blind sheet entries (candidates shuffled, unattributed; framed on the final crop) of the pending labels that
    need a confirm look and are in no confirm batch yet, VNs interleaved (pool.vn_rank). A label whose copies already
    fill the repeat caps (kept lines plus copies waiting in a confirm batch) is left for later: if a look drops a
    copy, the next run takes another."""
    pend = {r["id"]: r for r in labels if r["status"] == "pending" and r.get("why") == "confirm-missing"}
    cap = CapCount((r["vn"], r["label"]) for r in labels
                   if r["status"] == "kept" or (r["id"] in have and r["id"] in pend))
    ids = []
    for i in sorted((i for i in pend if i not in have and i in boxes), key=lambda i: (PL.vn_rank(i.split("_")[0]), i)):
        r = pend[i]
        if cap.full(r["vn"], r["label"]):
            continue
        cap.add(r["vn"], r["label"])
        ids.append(i)
    return S.review_entries(shots, ids, frames={i: boxes[i]["crop"] for i in ids})


def edge_todo(boxes, labels, have):
    """The flagged final crops of kept labels that need an edge check and are in no edge batch yet. A flagged line
    is skipped when unflagged kept copies of its label already fill the repeat caps (the build keeps those and drops
    the unchecked one); a pending label waits for its confirm look."""
    status = {r["id"]: r for r in labels}
    kept = [b for b in boxes if (status.get(b["id"]) or {}).get("status") == "kept"]
    # unflagged kept copies plus flagged copies already in an edge batch: what the build can keep of each label
    cap = CapCount((status[b["id"]]["vn"], b["label"]) for b in kept if not b["check"] or b["id"] in have)
    out = []
    for b in sorted(kept, key=lambda b: (PL.vn_rank(b["shot"].split("_")[0]), b["shot"], b["id"])):
        if not b["check"] or b["id"] in have:
            continue
        vn = status[b["id"]]["vn"]
        if cap.full(vn, b["label"]):
            continue
        cap.add(vn, b["label"])
        out.append(b)
    return out


def cmd_confirm_batches(args):
    labels = jl(OUT / "labels.jsonl")
    boxes = {b["id"]: b for b in jl(PL.EDGE / "boxes.jsonl")}
    with PL.review_lock():
        entries = confirm_entries(labels, boxes, PL.batched_ids("confirm"), PL.load_reads())
        start = len(PL.batch_dirs("confirm"))
        for k in range(args.n):
            part = entries[k * args.size:(k + 1) * args.size]
            if not part or len(part) < (args.min if args.min is not None else args.size):
                break
            out = REVIEW / "confirm" / f"b{start + k + 1:04d}"
            n = S.make_batch(part, out, "blind", name=f"owner2/confirm/{out.name}")
            note(out, CONFIRM_NOTE)
            print(f"{out}: {len(part)} lines, {n} sheets", flush=True)


def cmd_edge_batches(args):
    """pool.py's edge-check batches (final crop framed, the label the one candidate) with the owner note, for the
    flagged kept lines edge_todo picks."""
    boxes = jl(PL.EDGE / "boxes.jsonl")
    labels = jl(OUT / "labels.jsonl")
    with PL.review_lock():
        have = PL.batched_ids("edge")
        todo = {b["id"] for b in edge_todo(boxes, labels, have)}
        entries = PL.edge_entries([b for b in boxes if b["id"] in todo], have, PL.load_reads())
        start = len(PL.batch_dirs("edge"))
        for k in range(args.n):
            part = entries[k * args.size:(k + 1) * args.size]
            if not part or len(part) < (args.min if args.min is not None else args.size):
                break
            out = REVIEW / "edge" / f"b{start + k + 1:04d}"
            n = S.make_batch(part, out, "edge", name=f"owner2/edge/{out.name}")
            note(out, PL.EDGE_NOTE)
            print(f"{out}: {len(part)} lines, {n} sheets", flush=True)


# ----------------------------------------------------------------------------- helpers

def launch_text(kind, batch):
    if kind == "check":
        return (f"Your batch is {batch}. It is a blind label-precision check of the owner's PRIVATE console captures: "
                "no candidates are shown; type exactly the framed text for every entry, as its README says. Never "
                "send anything from this batch to Codex or any other model or third party.")
    return f"Your batch is {batch}. {PRIVATE}"


def queued_labels(herd_dir=PL.HERD_DIR):
    p = Path(herd_dir) / "queue.tsv"
    return {x.split("\t", 1)[0] for x in p.read_text(encoding="utf-8").splitlines() if x.strip()} if p.exists() \
        else set()


def cmd_helpers(args):
    """Queue one helper per undecided batch (kinds in --kinds order), at most --max outstanding (queued or
    working, batch not done, no final marker yet). --retry re-queues a batch whose helper ended without finishing
    it (the new session continues from its decisions.jsonl)."""
    with PL.review_lock():
        log, res = PL.helper_log(), PL.results()
        busy = PL.outstanding(log, res)
        n = max([int(h["label"][len(HELPER):]) for h in log if h["label"].startswith(HELPER)
                 and h["label"][len(HELPER):].isdigit()] + [0])
        queued = []
        for kind in args.kinds.split(","):
            for b in PL.batch_dirs(kind):
                if len(busy) + len(queued) >= args.max:
                    break
                if PL.batch_state(b)[0] == "done":
                    continue
                prior = [h for h in log if h.get("batch") == str(b)]
                if prior and (any(h in busy for h in prior) or not args.retry or len(prior) >= MAX_TRIES):
                    continue
                n += 1
                label = f"{HELPER}{n}"
                cmd = [str(PL.HERD_DIR / "enqueue.sh"), label, "-", str(PL.REVIEW_PROMPT), launch_text(kind, b)]
                subprocess.run(cmd, check=True, cwd=PL.HERD_DIR, env=dict(os.environ, HERD_DIR=str(PL.HERD_DIR)),
                               stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
                rec = {"label": label, "batch": str(b), "kind": kind,
                       "queued": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                log.append(rec)
                queued.append(rec)
        wjl(REVIEW / "helpers.jsonl", log)
    print(f"{len(busy)} helpers outstanding, {len(queued)} queued now: {[q['label'] for q in queued]}", flush=True)


# ----------------------------------------------------------------------------- the cycle (one loop keeps it moving)

def close_finished(log, res, herd_dir=PL.HERD_DIR):
    """Helpers of this pool whose batch parses as done but whose tab is still open and idle: close the tab and record
    REVIEW_DONE in results.txt and closed.txt (the orchestrator's format). A narrow pane can wrap the marker over two
    lines; the orchestrator then never sees it and resumes the finished session (realplus v3's monitor does the
    same for its helpers)."""
    H = Path(herd_dir)
    agents = {x.split()[0]: x.split()[1:] for x in (H / "agents.txt").read_text().splitlines() if x.strip()}
    closed = {x.strip() for x in (H / "closed.txt").read_text().splitlines()} if (H / "closed.txt").exists() else set()
    try:
        out = subprocess.run(["timeout", "20", "herdr", "pane", "list"], capture_output=True, text=True).stdout
        st = {q.get("label"): q["agent_status"] for q in json.loads(out)["result"]["panes"] if q.get("label")}
    except (ValueError, KeyError, OSError):
        return []
    done = []
    for h in log:
        lab = h["label"]
        if not lab.startswith(HELPER) or lab not in agents or lab in closed or f" {lab} REVIEW_DONE" in res:
            continue
        if st.get(lab) not in ("done", "idle") or PL.batch_state(Path(h["batch"]))[0] != "done":
            continue
        tab = agents[lab][1] if len(agents[lab]) > 1 else None
        if tab:
            subprocess.run(["timeout", "20", "herdr", "tab", "close", tab], capture_output=True)
        with open(H / "results.txt", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {lab} REVIEW_DONE {h['batch']} "
                    "(finished; marker missed by the orchestrator; tab closed by owner_pool.py cycle)\n")
        with open(H / "closed.txt", "a", encoding="utf-8") as f:
            f.write(lab + "\n")
        done.append(lab)
    return done


def queue_front(path=PL.HERD_DIR / "queue.tsv"):
    """Move this pool's queued helpers (HELPER*) to just before the first realplus-v3 review helper (review-v3-*),
    keeping every other line and its order. The orchestrator starts the queue's first line when a slot frees."""
    before = Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""
    lines = [x for x in before.splitlines() if x.strip()]
    mine = [x for x in lines if x.split("\t", 1)[0].startswith(HELPER)]
    rest = [x for x in lines if not x.split("\t", 1)[0].startswith(HELPER)]
    i = next((k for k, x in enumerate(rest) if x.split("\t", 1)[0].startswith("review-v3-")), len(rest))
    new = rest[:i] + mine + rest[i:]
    if new == lines:
        return False
    tmp = Path(str(path) + ".o2tmp")
    tmp.write_text("\n".join(new) + "\n", encoding="utf-8")
    if Path(path).read_text(encoding="utf-8") != before:
        tmp.unlink()
        return False
    os.replace(tmp, path)
    return True


def heavy(cmd, secs=3600):
    """One CPU-heavy step through heavy.sh (HEAVY_SLOTS=12), blocking until a slot is free."""
    return subprocess.run(["bash", "-c", f"HEAVY_SLOTS=12 {HERD}/heavy.sh timeout {secs} {cmd} < /dev/null"],
                          cwd=ROOT, env=dict(os.environ), capture_output=True, text=True)


def pass1_complete():
    """Every plan line is in a pass-1 batch and every pass-1 batch is decided."""
    drop = leak_drop()
    idx = S.line_index(PL.load_reads())
    have = PL.batched_ids()
    left = [r["id"] for r in jl(REVIEW / "plan.jsonl")
            if r["id"] in idx and idx[r["id"]][0]["shot"] not in drop and r["id"] not in have]
    return not left and all(PL.batch_state(b)[0] == "done" for b in PL.batch_dirs("pass1"))


def cmd_cycle(args):
    """One round (or --loop): close finished helpers; when a batch was decided since the last round, final ->
    native.py edge-boxes (heavy.sh) -> confirm batches (and edge batches of flagged kept crops) of at least --min
    lines, every line once pass 1 (and, for edge, the confirm looks) is complete; queue helpers up to --max (kinds in
    --kinds order, a stalled batch re-queued up to MAX_TRIES times); keep them ahead of realplus-v3's queue."""
    state_p = OUT / "logs" / "cycle_state.json"
    state_p.parent.mkdir(parents=True, exist_ok=True)
    while True:
        log, res = PL.helper_log(), PL.results()
        closed = close_finished(log, res)
        done = {k: len(PL.done_batches(k)) for k in KINDS}
        last = json.loads(state_p.read_text()) if state_p.exists() else {}
        made = []
        if done != last.get("done") or args.force:
            cmd_final(argparse.Namespace())
            r = heavy(f"{ROOT}/.venv-ci/bin/python {HERE}/native.py edge-boxes")
            print(r.stdout.strip()[-400:], r.stderr.strip()[-400:], flush=True)
            p1 = pass1_complete()
            with PL.review_lock():
                n0 = len(PL.batch_dirs("confirm"))
            cmd_confirm_batches(argparse.Namespace(n=10, size=args.size, min=1 if p1 else args.min))
            labels = jl(OUT / "labels.jsonl")
            have = PL.batched_ids("confirm")
            waiting = [x for x in labels if x["status"] == "pending" and x.get("why") == "confirm-missing"
                       and x["id"] in have]
            conf_done = p1 and all(PL.batch_state(b)[0] == "done" for b in PL.batch_dirs("confirm")) and not waiting
            n1 = len(PL.batch_dirs("edge"))
            cmd_edge_batches(argparse.Namespace(n=10, size=args.size, min=1 if conf_done else args.min))
            made = [f"confirm+{len(PL.batch_dirs('confirm')) - n0}", f"edge+{len(PL.batch_dirs('edge')) - n1}"]
            state_p.write_text(json.dumps({"done": done, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
        cmd_helpers(argparse.Namespace(max=args.max, kinds=args.kinds, retry=True))
        moved = queue_front()
        lab = collections.Counter(x["status"] for x in jl(OUT / "labels.jsonl"))
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} done={done} made={made} closed={closed} "
              f"queue_moved={moved} labels={dict(lab)}", flush=True)
        if not args.loop:
            break
        time.sleep(args.every)


def cmd_status(args):
    log, res = PL.helper_log(), PL.results()
    for kind in KINDS:
        bs = PL.batch_dirs(kind)
        if not bs:
            continue
        st = collections.Counter(PL.batch_state(b)[0] for b in bs)
        lines = sum(json.loads((b / "batch.json").read_text())["lines"] for b in bs)
        print(f"{kind}: {len(bs)} batches ({lines} lines): {dict(st)}", flush=True)
        for b in bs:
            s, err = PL.batch_state(b)
            if s == "bad":
                print(f"  {b.name}: {err}", flush=True)
    busy = PL.outstanding(log, res)
    q = queued_labels()
    print(f"helpers: {len(log)} in all, {len(busy)} outstanding: "
          f"{[(h['label'], 'queued' if h['label'] in q else 'running') for h in busy]}", flush=True)


# ----------------------------------------------------------------------------- one set of files

CROPS = ("ink", "grad", "ink_m4", "ink_m8")


def read_npz(path):
    """(crops, labels, metas) of a pool npz (devnative/build._save's format)."""
    import numpy as np
    with np.load(path) as z:
        data, offs, shapes = z["data"], z["offsets"], z["shapes"]
        labels, metas = json.loads(str(z["labels"])), json.loads(str(z["meta"]))
        crops = [data[offs[i]:offs[i + 1]].reshape(int(h), int(w)) for i, (h, w) in enumerate(shapes)]
    return crops, labels, metas


def merge_rows(parts, cap=PL.GLOBAL_CAP):
    """parts: [(pool name, crops, labels, metas)] in priority order -> (crops, labels, metas, dropped): every row,
    each meta tagged with its pool, the lines of a label past `cap` lines in all dropped (the global repeat cap over
    the union; the per-VN cap held inside each pool and the pools share no VN). A line keeps all its crop kinds:
    the cap counts line ids, not crops."""
    seen, keep_ids, dropped = collections.Counter(), set(), set()
    for name, _, labels, metas in parts:
        for lab, m in zip(labels, metas):
            lid = (name, m["id"])
            if lid in keep_ids or lid in dropped:
                continue
            k = vote.key(lab)
            if seen[k] >= cap:
                dropped.add(lid)
                continue
            seen[k] += 1
            keep_ids.add(lid)
    out_c, out_l, out_m = [], [], []
    for name, crops, labels, metas in parts:
        for c, lab, m in zip(crops, labels, metas):
            if (name, m["id"]) in keep_ids:
                out_c.append(c)
                out_l.append(lab)
                out_m.append(dict(m, owner_pool=name))
    return out_c, out_l, out_m, dropped


def cmd_merge(args):
    """owner_train_<crop>_<tag>.npz in this pool: an earlier pool's built lines (--old DIR:TAG, e.g. owner pool
    v1's clamped v2 files) then this pool's (--tag), under one global repeat cap; the line ids of the two pools
    never collide (shots <vn>_owner<n> and <vn>_p2f<n>)."""
    from devnative import build as B
    from devnative import paths as P
    old_dir, old_tag = args.old.split(":", 1)
    names = {"old": Path(old_dir).name, "new": OUT.name}
    stats = {"tag": args.tag, "old": {"dir": names["old"], "tag": old_tag}, "new": {"dir": names["new"],
                                                                                   "tag": args.tag}, "files": {}}
    ids = {}
    for crop in CROPS:
        parts = []
        for which, d, tag in (("old", Path(old_dir), old_tag), ("new", OUT, args.tag)):
            p = d / f"native_train_{crop}_{tag}.npz"
            if not p.exists():
                sys.exit(f"missing {p}")
            parts.append((names[which], *read_npz(p)))
        ids[crop] = [{(n, m["id"]) for m in ms} for n, _, _, ms in parts]
        crops, labels, metas, dropped = merge_rows(parts)
        out = OUT / f"owner_train_{crop}_{args.tag}.npz"
        if out.exists():
            sys.exit(f"refusing to overwrite {out}")
        B._save(out, crops, labels, metas)
        stats["files"][out.name] = {"crops": len(crops), "by_pool": dict(collections.Counter(
            m["owner_pool"] for m in metas)), "by_target": dict(collections.Counter(m["target"] for m in metas)),
            "capped_lines": len(dropped), "sha256": P.file_sha256(out)}
    if any(a & b for a, b in ids.values()):
        sys.exit("line ids collide across the pools")
    (OUT / "manifests" / f"merge_stats_{args.tag}.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1), flush=True)


# ----------------------------------------------------------------------------- publish (aggregate only)

NAME_RULE = ("a speaker name: a line a Claude look flagged (\"name\": true in pass 1, a confirm look, an edge check "
             "or the precision check), a line whose label has the key of a flagged line of the same VN, or a label "
             "that is one 【...】 bracket (the review conventions' 【name】)")
BRACKET_NAME = re.compile(r"【[^【】]+】")
MENU = ("title_screen", "settings_screen")          # the 2026-10-14 drop's frame categories that are not dialogue
MENU_RULE = ("a menu line: a line of a frame that the 2026-10-14 drop's own manifest calls a title or settings screen "
             "(owner-games/<folder>/manifest_20261014.jsonl \"category\"); the 2026-10-13 drop has no categories")


def name_ids(labels, kinds=KINDS):
    """Line ids of the speaker names among labels (NAME_RULE). Auto lines had no pass-1 look, and name plates repeat, so
    a flagged label marks its copies in the same VN too. The dialogue-only count is the built lines outside this set
    and outside the menu frames (menu_shots)."""
    from devnative import decide as D
    lab = {r["id"]: r for r in labels}
    flagged = {r["id"] for r in labels if r.get("name")}
    for kind in kinds:
        bs = PL.done_batches(kind)
        flagged |= {lid.split("@", 1)[0] for lid, d in (D.collect(bs) if bs else {}).items() if d.get("name")}
    keys = {(lab[i]["vn"], vote.key(lab[i]["label"])) for i in flagged if (lab.get(i) or {}).get("label")}
    return flagged | {r["id"] for r in labels if r.get("label") and ((r["vn"], vote.key(r["label"])) in keys
                                                                    or BRACKET_NAME.fullmatch(r["label"].strip()))}


def menu_shots(frames, root=None):
    """Shot keys of the frames whose drop manifest calls them a title or settings screen (MENU_RULE). Their lines are
    real console text and stay in the pool, but they are not dialogue."""
    import shots as SH
    root = Path(root) if root else SH.OWNER_GAMES
    cat = {}
    for folder in sorted({f["folder"] for f in frames}):
        for r in jl(root / folder / "manifest_20261014.jsonl"):
            cat[(folder, r["file"])] = r.get("category")
    return {f["shot"] for f in frames if cat.get((f["folder"], f["file"])) in MENU}


def per_game(frames, labels, man, drop_games=()):
    """{vn: {...}} for the table: platform, folders, frames read (of them menu frames), candidate lines, kept labels,
    built lines, menu lines, dialogue lines (built lines that are neither speaker names nor on a menu frame), by crop
    kind. Counts only. A frame or line row may carry "menu" (menu_shots)."""
    out = collections.defaultdict(lambda: collections.Counter())
    meta = collections.defaultdict(dict)
    for f in frames:
        out[f["vn"]]["frames"] += 1
        out[f["vn"]]["menu_frames"] += bool(f.get("menu"))
        meta[f["vn"]].setdefault("platform", f["kind"])
        meta[f["vn"]].setdefault("folders", set()).add(f["folder"])
    for r in labels:
        vn = r["id"].split("_")[0]
        out[vn]["candidates"] += 1
        out[vn]["kept_labels"] += r["status"] == "kept"
    for m in man:
        out[m["vn"]]["lines"] += 1
        out[m["vn"]]["menu_lines"] += bool(m.get("menu"))
        out[m["vn"]]["dialogue_lines"] += not m.get("name") and not m.get("menu")
        for c in m["crops"]:
            out[m["vn"]][f"crops_{c}"] += 1
    titles = {g["vn"]: g["title"] for g in drop_games}
    return {vn: {"platform": meta[vn].get("platform"), "folders": sorted(meta[vn].get("folders", ())),
                 "title": titles.get(vn), **dict(c)} for vn, c in sorted(out.items(), key=lambda kv: int(kv[0][1:]))}


def cmd_publish(args):
    """realplus/manifests/owner/stats_<tag>.json: the rule's sha256, frame and line counts per VN and stage, leak and
    hygiene results, the npz files' sha256 and the precision summary. No file name, hash, box or text of a private
    frame: VN ids, folder names, counts and the pool files' own sha256 only."""
    from devnative import paths as P
    frames = jl(OUT / "manifests" / "frames.jsonl")
    labels = jl(OUT / "labels.jsonl")
    man = jl(OUT / "manifests" / f"lines_{args.tag}.jsonl")
    # the build's line manifests carry no name flag: take it from the labels and every look (NAME_RULE); a line of a
    # title or settings screen (MENU_RULE) is not dialogue either
    names, pass1_names = name_ids(labels), {r["id"] for r in labels if r.get("name")}
    menu = menu_shots(frames)
    frames = [dict(f, menu=f["shot"] in menu) for f in frames]
    man = [dict(m, name=m["id"] in names, menu=m["shot"] in menu) for m in man]
    # the menu lines' ids, for a trainer that wants to leave them out (private, next to the npz files)
    wjl(OUT / "manifests" / f"menu_lines_{args.tag}.jsonl", [{"id": m["id"], "vn": m["vn"]} for m in man if m["menu"]])
    # titles: both drops' public tables (the 2026-10-13 drop's former -TEST folders train in this pool too)
    games = [g for d in ("20261013", "20261014") for g in json.loads(
        (ROOT / "data" / "manifests" / f"owner_drop_{d}.json").read_text(encoding="utf-8"))["games"]]
    build = json.loads((OUT / "manifests" / f"build_stats_{args.tag}.json").read_text())
    doc = {"kind": "owner-pool-stats", "tag": args.tag, "pool": "owner pool v2",
           "private_dir": f"shared/private/owner-games/_build2/{OUT.name}",
           "codex": "none: private frames; Claude (pass 1 + an independent confirm look) and MeikiOCR / PP-OCRv6 only",
           "rule_sha256": P.file_sha256(OUT / "AGREEMENT.md"), "rule": CONFIRM_RULE,
           "frames_read": len(jl(OUT / "reads" / "reads.jsonl")),
           "frames_by_kind": dict(collections.Counter(f["kind"] for f in frames)),
           "labels": dict(collections.Counter(f"{r['status']}:{r['how'] if r['status'] == 'kept' else r['why']}"
                                              for r in labels)),
           "build": build, "lines": len(man), "dialogue_lines": sum(not m["name"] and not m["menu"] for m in man),
           "dialogue_lines_by_kind": dict(collections.Counter(m["kind"] for m in man
                                                              if not m["name"] and not m["menu"])),
           "name_lines": sum(m["name"] for m in man), "name_rule": NAME_RULE,
           "name_lines_pass1_flag": sum(m["id"] in pass1_names for m in man),
           "menu_frames_read": len(menu), "menu_lines": sum(m["menu"] for m in man), "menu_rule": MENU_RULE,
           "menu_lines_by_how": dict(collections.Counter(m["how"] for m in man if m["menu"])),
           "per_game": per_game(frames, labels, man, games)}
    for k in ("pass1", "confirm", "edge", "check"):
        doc[f"batches_{k}"] = len(PL.batch_dirs(k))
    for name in ("leak", "hygiene"):
        p = OUT / "manifests" / f"{name}.json"
        if p.exists():
            doc[name] = json.loads(p.read_text())
    mp = OUT / "manifests" / f"merge_stats_{args.tag}.json"
    if mp.exists():
        doc["merged"] = json.loads(mp.read_text())
    cp = REVIEW / f"check_report_{args.check}.json"
    if cp.exists():
        d = json.loads(cp.read_text())
        doc["precision_check"] = {k: v for k, v in d.items() if k not in ("disagreements", "batches")}
    blob = json.dumps(doc, ensure_ascii=False)
    if ".png" in blob or '"label"' in blob or '"text"' in blob:
        raise ValueError("stats name a frame or hold text")
    dst = ROOT / "realplus" / "manifests" / "owner"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / f"stats_{args.tag}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"published {dst / f'stats_{args.tag}.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("batches", "confirm-batches", "edge-batches"):
        p = sub.add_parser(name)
        p.add_argument("--n", type=int, default=1)
        p.add_argument("--size", type=int, default=280)
        p.add_argument("--min", type=int, default=1, help="smallest batch to write")
    sub.add_parser("final")
    p = sub.add_parser("leak-drop")
    p.add_argument("logs", nargs="+", help="eval/tools/leakcheck.py logs of this pool's shots/")
    p = sub.add_parser("helpers")
    p.add_argument("--max", type=int, default=6)
    p.add_argument("--kinds", default="check,confirm,edge,pass1")
    p.add_argument("--retry", action="store_true")
    sub.add_parser("status")
    p = sub.add_parser("cycle")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--every", type=int, default=600)
    p.add_argument("--size", type=int, default=300)
    p.add_argument("--min", type=int, default=150, help="smallest confirm / edge batch before pass 1 is complete")
    p.add_argument("--max", type=int, default=6)
    p.add_argument("--kinds", default="check,confirm,edge,pass1")
    p.add_argument("--force", action="store_true", help="run final and make batches even if nothing new was decided")
    p = sub.add_parser("merge")
    p.add_argument("--tag", required=True)
    p.add_argument("--old", required=True, help="an earlier owner pool's DIR:TAG (its native_train_<crop>_TAG.npz)")
    p = sub.add_parser("publish")
    p.add_argument("--tag", required=True)
    p.add_argument("--check", default="pre", help="the build tag the precision sample was drawn from")
    a = ap.parse_args()
    {"batches": cmd_batches, "final": cmd_final, "leak-drop": cmd_leak_drop, "confirm-batches": cmd_confirm_batches,
     "edge-batches": cmd_edge_batches, "helpers": cmd_helpers, "status": cmd_status, "cycle": cmd_cycle,
     "merge": cmd_merge,
     "publish": cmd_publish}[a.cmd](a)


if __name__ == "__main__":
    main()
