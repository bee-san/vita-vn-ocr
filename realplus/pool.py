#!/usr/bin/env python3
"""realplus: label the realplus screenshots with devnative's frozen pipeline and build the pool.

The labelling pipeline is devnative's (devnative/README.md; read and sheets frozen at 35a86ca, GRAD
containment 0cb084e, final crops, edge check and looks f523e30): MeikiOCR finds and reads the lines,
PP-OCRv6 small reads them again (four readings per line), Claude adjudicates on contact sheets, and
the training rule decides (decide.py --mode train). This file only drives it over a large pool and
adds what a training pool of PC / console shots needs on top:

  pool.py prep                     shots.py's manifest -> $REALPLUS/reads/shots.jsonl (read.py's input)
  (read)  HEAVY_SLOTS=12 $HERD/heavy.sh timeout 14400 unshare -rn teacher/.venv/bin/python devnative/read.py \\
              run --shots $REALPLUS/reads/shots.jsonl --out $REALPLUS/reads --procs 5
  pool.py plan                     every candidate line: auto (all four readings agree) or a review tier
  pool.py batches [--n K]          the next K pass-1 batches (devnative/sheets.py) -> $REALPLUS/review/pass1/bNNNN
  pool.py helpers [--max 8]        queue review-rpN helper agents for undecided batches, at most --max at once
  pool.py status                   batches made / decided / with a helper
  pool.py reserve DIR...           batches this agent reviews itself (no helper is queued for them)
  pool.py feed [--loop]            keep reads, batches and helpers going (read when idle, --ready batches)
  pool.py final                    decide.py's training rule (--mode train --auto-consensus) -> labels.jsonl
  pool.py edge-boxes               final crops of the kept labels and which need an edge check
  pool.py edge-batches [--n K]     edge-check batches (sheets mode edge) -> $REALPLUS/review/edge/bNNNN
  pool.py edge-apply               edgecheck.py apply over the decided edge batches -> $REALPLUS/edge/edge.jsonl
  pool.py build                    the npz files (teacher_train's format) and the committed manifests
  pool.py blind [--n 540]          a blind re-check sample, framed on the final crops -> $REALPLUS/review/blind
  pool.py blind-score              devnative's blind agreement on the first 500 sampled lines in the pool
  pool.py publish                  the committed, text-free manifests: realplus/manifests/shots_NN.jsonl, stats.json

Each PC / console screenshot is read once, scaled to fit the Vita screen (960x544, cv2.INTER_AREA as
teacher scales its hires shots, devnative/paths.load_screen). A kept line becomes a Vita crop and,
when its ink is at least 10 px tall at PSP scale, a PSP crop of the same screenshot scaled to fit
480x272 (exactly half the Vita scale), each as the ink crop and, where the console's GRAD finder
returns a box holding the whole line and nothing else, as the GRAD crop. The ink crop is eval's and
teacher's (ink + furigana + 4 / 2 px; read_line), widened to the labelled characters' boxes
(lines.final_crop); a line whose final crop edgecheck.flags flags is kept only after an edge check.

Review tiers, in the order batches are made (most labels per sheet first):
  t1  the most supported candidate is read by both engine families (MeikiOCR and PP-OCRv6)
  t2  two candidates
  t3  three or more candidates, none read by both families
A review line is skipped (never shown) when every one of its candidates is a label its VN already
has twice among the auto lines: the per-VN repeat cap (devnative/build.py) would drop it anyway.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import shots as SH  # noqa: E402

from devnative import decide as D  # noqa: E402
from devnative import gradcheck as GC  # noqa: E402
from devnative import lines as L  # noqa: E402
from devnative import paths as P  # noqa: E402
from devnative import sheets as S  # noqa: E402
from teacher import vote  # noqa: E402

OUT = SH.realplus_dir()
READS = OUT / "reads" / "reads.jsonl"
REVIEW = OUT / "review"
HERD_DIR = Path(os.environ.get("HERD_DIR", "/local/home/skerraut/herd-synth3"))
REVIEW_PROMPT = HERD_DIR / "prompts" / "review.md"
FAMILY = {"meiki", "v6s"}
PER_VN_CAP, GLOBAL_CAP = 2, 8          # devnative/build.py's per-VN cap; teacher/build.py's overall cap
MIN_PSP_INK = L.MIN_INK                # a PSP crop needs 10 px of ink, as a native PSP line
NOTE = ("\nNote (realplus): these screenshots are PC / home-console visual-novel screenshots scaled to the "
        "PS Vita\nscreen (960 x 544); decide them exactly as above. Speaker names: add \"name\": true.\n"
        "Images: view ONE sheet per tool call, each sheet once, at most ~60 images per session. Append the\n"
        "decisions of each sheet to decisions.jsonl in this directory as soon as that sheet is decided, so a\n"
        "fresh session can resume: if decisions.jsonl already exists, keep it and continue with the first sheet\n"
        "that has no decisions yet.\n")
EDGE_NOTE = ("This check is about the frame only: menus, buttons, system text and speaker names are in scope\n"
             "like dialogue (do not skip them as \"ui\"); pick 1 when the frame shows exactly the label.\n")
BLIND_NOTE = ("This is a label-precision check: transcribe the framed text whatever it is (dialogue, a menu,\n"
              "a button, a name); do not use skip \"ui\". Skip \"cut\" or \"bad\" only as defined above.\n")


def jl(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def wjl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(p)


def load_reads(path=READS):
    """sheets.load_reads, skipping a last line read.py is still writing."""
    shots = {}
    for x in Path(path).read_text(encoding="utf-8").splitlines():
        if x.strip():
            try:
                r = json.loads(x)
            except ValueError:
                continue
            shots[r["shot"]] = r
    return shots


# ----------------------------------------------------------------------------- prep

def dups():
    p = OUT / "manifests" / "dups.json"
    return json.loads(p.read_text()) if p.exists() else {}


def read_rows(rows, dup):
    """read.py rows for every fetched, hashed, non-duplicate screenshot."""
    out = []
    for r in sorted(rows, key=lambda r: r["order"]):
        if not r.get("sha256") or not r.get("pixel_sha256") or str(r["image"]) in dup:
            continue
        out.append({"shot": f"{r['vn']}_{r['image']}", "vn": r["vn"], "kind": "vita", "url": r["url"],
                    "path": str(SH.shot_path(r)), "release": r["release"], "session": r["session"],
                    "subset": "realplus", "flags": [], "source_kind": r["kind"], "dims": r["dims"]})
    return out


def cmd_prep(args):
    rows = SH.load_shots()
    rr = read_rows(rows, dups())
    wjl(OUT / "reads" / "shots.jsonl", rr)
    print(f"{len(rr)} screenshots to read (of {len(rows)} selected)", flush=True)


# ----------------------------------------------------------------------------- plan

def families(engines):
    return {S.FAMILY[e] for e in engines}


def tier(cands):
    """auto, t1, t2 or t3 for a candidate line's distinct readings (sheets.candidates)."""
    if len(cands) == 1 and len(cands[0]["engines"]) == len(S.ENGINES):
        return "auto"
    if cands and families(cands[0]["engines"]) == FAMILY:
        return "t1"
    return "t2" if len(cands) == 2 else "t3"


def vn_rank(vn, seed=20261012):
    return hashlib.sha256(f"{seed}:{vn}".encode()).hexdigest()


def plan(shots):
    """{line id: {"tier", "vn", "shot", "keys"}} for every candidate line, and the review order: by
    tier, then round-robin over VNs (a VN's n-th screenshot after every VN's (n-1)-th), lines of
    one screenshot together, top to bottom."""
    lines, auto_keys = {}, collections.defaultdict(collections.Counter)
    by_vn = collections.defaultdict(list)
    for s in shots.values():
        by_vn[s["vn"]].append(s["shot"])
        for ln in s.get("lines", []):
            ok, _ = S.is_candidate(ln)
            if not ok:
                continue
            c = S.candidates(ln)
            t = tier(c)
            lid = f"{s['shot']}_{ln['k']}"
            lines[lid] = {"tier": t, "vn": s["vn"], "shot": s["shot"], "keys": [x["key"] for x in c],
                          "y": ln["ink"][1], "x": ln["ink"][0]}
            if t == "auto":
                auto_keys[s["vn"]][c[0]["key"]] += 1
    shot_rank = {}
    for vn, ss in by_vn.items():
        for i, sk in enumerate(sorted(ss, key=lambda k: vn_rank(vn, k))):
            shot_rank[sk] = (i, vn_rank(vn))
    order = []
    for lid, x in lines.items():
        if x["tier"] == "auto":
            continue
        if all(auto_keys[x["vn"]][k] >= PER_VN_CAP for k in x["keys"]):
            x["skip"] = "repeat-cap"
            continue
        order.append(lid)
    order.sort(key=lambda lid: (lines[lid]["tier"], shot_rank[lines[lid]["shot"]], lines[lid]["shot"],
                                lines[lid]["y"], lines[lid]["x"]))
    return lines, order


def cmd_plan(args):
    shots = load_reads()
    lines, order = plan(shots)
    c = collections.Counter(x["tier"] for x in lines.values())
    sk = sum(1 for x in lines.values() if x.get("skip"))
    wjl(REVIEW / "plan.jsonl", [{"id": lid, "tier": lines[lid]["tier"]} for lid in order])
    print(json.dumps({"screenshots": len(shots), "candidate_lines": len(lines), "tiers": dict(c),
                      "review": len(order), "skipped_repeat_cap": sk}, indent=1), flush=True)


# ----------------------------------------------------------------------------- batches

def batch_dirs(kind="pass1"):
    d = REVIEW / kind
    return sorted(x for x in d.glob("b*") if (x / "batch.json").exists()) if d.exists() else []


def batched_ids(kind="pass1"):
    out = set()
    for b in batch_dirs(kind):
        out |= {e["id"] for e in json.loads((b / "batch.json").read_text(encoding="utf-8"))["index"]}
    return out


def cmd_batches(args):
    shots = load_reads()
    idx = S.line_index(shots)
    plan_ids = [r["id"] for r in jl(REVIEW / "plan.jsonl")]
    done = batched_ids()
    todo = [lid for lid in plan_ids if lid not in done and lid in idx]
    have = batch_dirs()
    k = max([int(b.name[1:]) for b in have] + [0])
    made = 0
    while todo and made < args.n:
        chunk, todo = todo[:args.size], todo[args.size:]
        # lines of one screenshot together, so the reviewer sees a name next to its dialogue
        chunk.sort(key=lambda lid: (idx[lid][0]["shot"], idx[lid][1]["ink"][1], idx[lid][1]["ink"][0]))
        k += 1
        out = REVIEW / "pass1" / f"b{k:04d}"
        entries = [(lid, idx[lid][0], idx[lid][1], S.candidates(idx[lid][1])) for lid in chunk]
        ns = S.make_batch(entries, out, "pass1", name=f"realplus/pass1/{out.name}")
        with open(out / "README.md", "a", encoding="utf-8") as f:
            f.write(NOTE)
        print(f"{out}: {len(chunk)} lines, {ns} sheets", flush=True)
        made += 1


# ----------------------------------------------------------------------------- helpers

def helper_log():
    return jl(REVIEW / "helpers.jsonl")


class review_lock:
    """An exclusive lock on $REVIEW/.lock around helpers.jsonl and batch numbering (feed and reserve)."""
    def __enter__(self):
        import fcntl
        REVIEW.mkdir(parents=True, exist_ok=True)
        self.f = open(REVIEW / ".lock", "w")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        self.f.close()


def batch_state(b):
    """done (decisions parse), partial (valid so far, entries still undecided: a helper saves per
    sheet), bad (decisions.jsonl does not parse) or open."""
    if not (b / "decisions.jsonl").exists():
        return "open", None
    try:
        D.parsed(b)
        return "done", None
    except (D.DecisionError, ValueError, KeyError) as e:
        msg = str(e)
        return ("partial" if msg.split(":", 1)[0].endswith("entries without a decision") else "bad"), msg[:400]


def results():
    p = HERD_DIR / "results.txt"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def outstanding(log, res):
    """Helpers queued or working whose batch is not done and who have not printed a final marker."""
    out = []
    cancelled = {h["label"] for h in log if h.get("cancelled")}
    for h in log:
        if h.get("cancelled") or h["label"] in cancelled:
            continue
        b = Path(h["batch"])
        if h["label"].startswith("self") or batch_state(b)[0] == "done":
            continue
        if f" {h['label']} REVIEW_DONE" in res or f" {h['label']} REVIEW_BLOCKED" in res \
                or f" {h['label']} EXITED" in res or f" {h['label']} GONE" in res:
            continue
        out.append(h)
    return out


def cmd_helpers(args):
    log, res = helper_log(), results()
    busy = outstanding(log, res)
    n = max([int(h["label"].rsplit("rp", 1)[1].split("-")[0]) for h in log if "-rp" in h["label"]] + [0])
    queued = []
    for kind in args.kinds.split(","):
        for b in batch_dirs(kind):
            if len(busy) + len(queued) >= args.max:
                break
            st, _ = batch_state(b)
            if st == "done":
                continue
            prior = [h for h in log if h.get("batch") == str(b)]
            if prior and any(h in busy for h in prior):
                continue
            if prior and not args.retry:
                continue
            n += 1
            label = f"review-rp{n}"
            cmd = [str(HERD_DIR / "enqueue.sh"), label, "-", str(REVIEW_PROMPT), f"Your batch is {b}."]
            subprocess.run(cmd, check=True, cwd=HERD_DIR, env=dict(os.environ, HERD_DIR=str(HERD_DIR)),
                           stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
            rec = {"label": label, "batch": str(b), "kind": kind, "queued": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                            time.gmtime())}
            log.append(rec)
            queued.append(rec)
    wjl(REVIEW / "helpers.jsonl", log)
    print(f"{len(busy)} helpers outstanding, {len(queued)} queued now: {[q['label'] for q in queued]}", flush=True)


def cmd_reserve(args):
    """Mark batches as reviewed by this agent itself (no helper is queued for them)."""
    with review_lock():
        log = helper_log()
        for b in args.batch:
            log.append({"label": f"self-{Path(b).name}", "batch": str(Path(b)), "kind": Path(b).parent.name,
                        "queued": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        wjl(REVIEW / "helpers.jsonl", log)


def read_running():
    pid = OUT / "reads" / "read.pid"
    if not pid.exists():
        return False
    try:
        os.kill(int(pid.read_text().split()[0]), 0)
        return True
    except (OSError, ValueError):
        return False


def start_read():
    """dedupe + prep, then devnative/read.py on the screenshots not read yet, in one heavy.sh slot."""
    py = str(ROOT / ".venv" / "bin" / "python")
    subprocess.run([py, str(HERE / "shots.py"), "dedupe"], check=True, stdin=subprocess.DEVNULL)
    cmd_prep(None)
    log = OUT / "logs" / f"read_{time.strftime('%H%M%S')}.log"
    herd = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
    # REALPLUS_HEAVY_SLOTS=13: the conductor's 20:38Z allowance of one extra slot for this read only.
    slots = int(os.environ.get("REALPLUS_HEAVY_SLOTS", "12"))
    cmd = (f"HEAVY_SLOTS={slots} {herd}/heavy.sh timeout 14400 unshare -rn {ROOT}/teacher/.venv/bin/python "
           f"{ROOT}/devnative/read.py run --shots {OUT}/reads/shots.jsonl --out {OUT}/reads --procs 5 "
           f"> {log} 2>&1 < /dev/null")
    proc = subprocess.Popen(["bash", "-c", cmd], cwd=ROOT, start_new_session=True)
    (OUT / "reads" / "read.pid").write_text(f"{proc.pid}\n")
    return log


def unassigned(log):
    have = {h["batch"] for h in log if "batch" in h}
    return [b for b in batch_dirs("pass1") if str(b) not in have and batch_state(b)[0] != "done"]


def cmd_feed(args):
    """One round (or --loop forever): start a read of new screenshots when none runs, keep --ready
    pass-1 batches without a helper, and queue helpers up to --max."""
    while True:
        read = {json.loads(x)["shot"] for x in READS.read_text(encoding="utf-8").splitlines()[:-1] if x.strip()} \
            if READS.exists() else set()
        rows = [r for r in SH.load_shots() if r.get("pixel_sha256")]
        new = sum(1 for r in rows if f"{r['vn']}_{r['image']}" not in read)
        if not read_running() and new >= args.min_new:
            print(f"{time.strftime('%H:%M:%S')} reading {new} new screenshots -> {start_read()}", flush=True)
        with review_lock():
            if len(unassigned(helper_log())) < args.ready:
                cmd_plan(None)
                cmd_batches(argparse.Namespace(n=args.ready, size=args.size))
            cmd_helpers(argparse.Namespace(max=args.max, kinds=args.kinds, retry=False))
        if not args.loop:
            break
        time.sleep(args.every)


def cmd_status(args):
    log, res = helper_log(), results()
    for kind in ("edge", "pass1", "pass2", "blind"):
        bs = batch_dirs(kind)
        if not bs:
            continue
        st = collections.Counter(batch_state(b)[0] for b in bs)
        lines = sum(json.loads((b / "batch.json").read_text())["lines"] for b in bs)
        print(f"{kind}: {len(bs)} batches ({lines} lines): {dict(st)}", flush=True)
        for b in bs:
            s, err = batch_state(b)
            if s == "bad":
                print(f"  {b.name}: {err}", flush=True)
    busy = outstanding(log, res)
    print(f"helpers: {len(log)} queued in all, {len(busy)} outstanding: {[h['label'] for h in busy]}", flush=True)


# ----------------------------------------------------------------------------- final

def done_batches(kind):
    return [b for b in batch_dirs(kind) if batch_state(b)[0] == "done"]


def reads_snapshot():
    """READS without a partial last line (a read may be appending): devnative's loaders need whole
    lines. Returns the snapshot path."""
    snap = OUT / "reads" / "reads_snapshot.jsonl"
    text = READS.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        text = text[:text.rfind("\n") + 1]
    snap.write_text(text, encoding="utf-8")
    return snap


def cmd_final(args):
    p1 = done_batches("pass1")
    p2 = done_batches("pass2")
    cmd = [sys.executable, str(ROOT / "devnative" / "decide.py"), "--reads", str(reads_snapshot()), "final",
           "--pass1", *map(str, p1)]
    if p2:
        cmd += ["--pass2", *map(str, p2)]
    bl = done_batches("blind")
    if bl:
        cmd += ["--blind", *map(str, bl)]        # counted in looks; never changes a label
    res = sorted(REVIEW.glob("resolutions*.jsonl"))
    if res and not args.no_resolve:
        cmd += ["--resolve", *map(str, res)]
    cmd += ["--mode", "train", "--auto-consensus", "--out", str(OUT / "labels.jsonl")]
    if not p1:
        sys.exit("no decided pass-1 batch yet")
    subprocess.run(cmd, check=True)
    if res and not args.no_resolve:
        labels = jl(OUT / "labels.jsonl")
        n = apply_auto_resolutions(labels, {r["id"]: r for f in res for r in jl(f)})
        wjl(OUT / "labels.jsonl", labels)
        print(f"third looks applied to {n} auto lines", flush=True)


def apply_auto_resolutions(labels, res):
    """decide.py's --resolve rule for the auto lines: decide_line returns before its resolution block
    when a line has no pass-1 decision, so a third look on a 4-way auto line would be ignored. Same
    semantics: keep false drops the line (why blind-<why>); a text replaces the label only with
    engine support (classify), else the line is dropped. Returns the number of records changed."""
    n = 0
    for r in labels:
        x = res.get(r["id"])
        if x is None or r["status"] != "kept" or r.get("pass1") is not None or "resolution" in r:
            continue
        r["resolution"] = {k: x.get(k) for k in ("keep", "why", "note") if k in x}
        if not x.get("keep", True):
            r.update(status="dropped", why=f"blind-{x.get('why') or 'resolved'}", label=None)
        elif x.get("text"):
            r.update(status="dropped", why="blind-text-unsupported", label=None)  # auto lines: no engine read it
        n += 1
    return n


# ----------------------------------------------------------------------------- edge check

EDGE = OUT / "edge"
FROZEN_LABELS = "labels_frozen_2128.jsonl"      # labels.jsonl at the freeze: the blind sample's frame


def cmd_edge_boxes(args):
    """devnative/edgecheck.py boxes on realplus's crops: the final crop of every kept label and whether
    it needs a check (edge_flags; or a later look called the line cut)."""
    from common import to_gray
    from devnative import edgecheck as E
    labels = [r for r in jl(OUT / "labels.jsonl") if r["status"] == "kept"]
    shots = load_reads()
    idx = S.line_index(shots)
    ev = E.evidence(done_batches("blind"))     # a blind look that called the line cut
    out, cache = [], {}
    for r in sorted(labels, key=lambda r: (int(r["shot"].split("_")[0][1:]), r["id"])):
        s, ln = idx[r["id"]]
        if s["shot"] not in cache:
            cache.clear()
            rgb = screen(s, "vita")
            cache[s["shot"]] = (rgb, to_gray(rgb))
        rgb, gray = cache[s["shot"]]
        H, W = rgb.shape[:2]
        fc, rc, why = edge_flags(gray, ln, "vita", W, H, r["label"])
        why += ev.get(r["id"], [])
        out.append({"id": r["id"], "shot": s["shot"], "label": r["label"], "crop": fc, "reviewed": rc,
                    "check": bool(why), "why": why})
    EDGE.mkdir(parents=True, exist_ok=True)
    wjl(EDGE / "boxes.jsonl", out)
    print(json.dumps({"lines": len(out), "check": sum(x["check"] for x in out),
                      "why": dict(collections.Counter(w for x in out for w in x["why"]))}), flush=True)


def edge_entries(boxes, have, shots):
    """Sheet entries (sheets.make_batch, mode edge: the frame is the final crop, the one candidate
    its label) for flagged lines not in any edge batch yet, in review order."""
    idx = S.line_index(shots)
    todo = [b for b in boxes if b["check"] and b["id"] not in have and b["id"] in idx]
    todo.sort(key=lambda b: (vn_rank(b["shot"].split("_")[0]), b["shot"], b["id"]))
    return [(b["id"], idx[b["id"]][0], idx[b["id"]][1],
             [{"text": b["label"], "key": vote.key(b["label"]), "engines": []}], b["crop"]) for b in todo]


def cmd_edge_batches(args):
    """Disjoint edge-check batches of --size flagged lines (review/edge/bNNNN)."""
    boxes = jl(EDGE / "boxes.jsonl")
    with review_lock():
        have = batched_ids("edge")
        entries = edge_entries(boxes, have, load_reads())
        d = REVIEW / "edge"
        start = len(batch_dirs("edge"))
        for k in range(args.n):
            part = entries[k * args.size:(k + 1) * args.size]
            if len(part) < (args.min if args.min is not None else args.size):
                break
            out = d / f"b{start + k + 1:04d}"
            n = S.make_batch(part, out, "edge", name=f"realplus/edge/{out.name}")
            with open(out / "README.md", "a", encoding="utf-8") as f:
                f.write(NOTE + EDGE_NOTE)
            print(f"{out}: {len(part)} lines, {n} sheets", flush=True)


def cmd_edge_apply(args):
    """edgecheck.py apply over the decided edge batches -> edge/edge.jsonl (build reads it)."""
    done = done_batches("edge")
    cmd = [sys.executable, str(ROOT / "devnative" / "edgecheck.py"), "--reads", str(reads_snapshot()), "apply",
           "--boxes", str(EDGE / "boxes.jsonl"), "--out", str(EDGE / "edge.jsonl"), "--batch", *map(str, done)]
    subprocess.run(cmd, check=True)


# ----------------------------------------------------------------------------- build

def scale_line(ln, fx, fy):
    """A read line's geometry mapped from the Vita-scale screen to the PSP-scale screen."""
    def sb(b):
        return [int(round(b[0] * fx)), int(round(b[1] * fy)), int(round(b[2] * fx)), int(round(b[3] * fy))]

    def sb_out(b):          # ink boxes: never shrink below the ink at the new scale
        return [int(b[0] * fx), int(b[1] * fy), -int(-b[2] * fx // 1), -int(-b[3] * fy // 1)]
    out = dict(ln)
    out["ink"] = sb_out(ln["ink"])
    if ln.get("crop_ink"):
        out["crop"] = sb_out(ln["crop_ink"])
    out["det"] = sb(ln["det"])
    out["parts"] = [sb(b) for b in ln.get("parts") or []]
    if ln.get("view"):
        out["view"] = sb(ln["view"])
    if ln.get("furi"):
        out["furi"] = [sb_out(b) for b in ln["furi"]]
    return out


def scale_chars(chars, fx, fy):
    """MeikiOCR character boxes [c, x0, y0, x1, y1, conf] mapped to the PSP-scale screen, outward to
    whole pixels (crop boxes index the image)."""
    import math
    return [[c[0], math.floor(c[1] * fx), math.floor(c[2] * fy), math.ceil(c[3] * fx), math.ceil(c[4] * fy),
             *c[5:]] for c in chars] if chars else chars


def ink_mixed(box, line, dets):
    """True when a quarter or more of another detected text box lies inside the ink crop: the
    GRAD crops' "mixed" test (devnative/lines.grad_status), applied to the ink crop too. With
    tight line spacing (common in 640x480 PC games) the padded crop otherwise takes in half of
    the neighbouring line."""
    for o in L.others_of(line, dets):
        if L.area(o) and L.area(L.inter(o, box)) >= L.MIXED * L.area(o):
            return True
    return False


def read_line(ln):
    """A reads.jsonl line in the shape devnative/lines.py expects: "crop" = ink + furigana. read.py
    stores the padded crop under "crop" (and the unpadded box under "crop_ink"), so crop_box() on a
    raw reads line pads twice (ink + 8 / 4 px); realplus cuts ink + 4 / 2 px, eval's and teacher's crop."""
    return dict(ln, crop=ln.get("crop_ink") or ln["ink"])


def edge_flags(gray, ln, kind, W, H, label):
    """(final crop, reviewed frame, reasons) for one labelled reads line (raw, as in reads.jsonl).

    The crop realplus cuts is eval's (ink + furigana + PAD, read_line) widened to the labelled
    characters' boxes (lines.final_crop). Whether it needs an edge check is devnative's rule
    (edgecheck.flags) applied to what the reviewers saw, the sheet frame (crop_box of the raw line,
    which read.py pads twice): ink continuing past that frame, or labelled characters beyond it. A
    border or plate edge between the two paddings is not text the label lacks. A label no MeikiOCR
    reading spells has no character boxes to widen to: it is checked too ("no-char-boxes")."""
    from devnative import edgecheck as E
    chars = L.label_chars(ln, vote.key, label)
    fc = L.final_crop(read_line(ln), kind, W, H, chars)
    _, frame, why = E.flags(gray, ln, kind, W, H, label)
    if not chars:
        why = why + ["no-char-boxes"]
    return fc, frame, why


def load_vocab():
    return set((ROOT / "data" / "vocab.txt").read_text(encoding="utf-8").split("\n")) - {""}


def cap_repeats(kept):
    """devnative/build.cap_repeats (2 per VN) plus teacher/build.py's overall cap (8 per label)."""
    per_vn, overall, out, capped = collections.Counter(), collections.Counter(), [], []
    for r in sorted(kept, key=lambda r: (int(r["vn"][1:]), r["id"])):
        k = vote.key(r["label"])
        if per_vn[(r["vn"], k)] >= PER_VN_CAP or overall[k] >= GLOBAL_CAP:
            capped.append(r)
            continue
        per_vn[(r["vn"], k)] += 1
        overall[k] += 1
        out.append(r)
    return out, capped


def chars_in(line, label, box, fx=1.0, fy=1.0):
    """devnative/build.chars_in (MeikiOCR's character boxes relative to the crop), for a crop of
    the screen scaled by (fx, fy) from the Vita-scale screen the line was read on."""
    from devnative import build as B
    ch, ok = B.chars_in(line, label, [0, 0])
    return [[c[0], round(c[1] * fx - box[0], 1), round(c[2] * fy - box[1], 1), round(c[3] * fx - box[0], 1),
             round(c[4] * fy - box[1], 1), c[5]] for c in ch], ok


def save_npz(path, crops, labels, metas):
    from devnative import build as B
    B._save(Path(path), crops, labels, metas)


def screen(s, kind):
    return P.load_screen(dict(s, kind=kind))


def cut_shot(s, recs, idx, src=None, edge=None):
    """Cut one screenshot's kept lines (build_lines' work for one shot): ({line id: (manifest row,
    [((target, crop), img, meta)])}, drop counts)."""
    import numpy as np

    from common import to_gray
    from devnative import build as B
    src = src or {}
    drops = collections.Counter()
    cut = {}
    sk = s["shot"]
    recs = sorted(recs, key=lambda r: r["id"])
    lines_v = [read_line(idx[r["id"]][1]) for r in recs]
    rgb_v = np.ascontiguousarray(screen(s, "vita"))
    rgb_p = np.ascontiguousarray(screen(s, "psp"))
    gray_v = to_gray(rgb_v)
    Hv, Wv = rgb_v.shape[:2]
    Hp, Wp = rgb_p.shape[:2]
    fx, fy = Wp / Wv, Hp / Hv
    lines_p = [scale_line(ln, fx, fy) for ln in lines_v]
    dets_v = s.get("dets", [])
    dets_p = [dict(d, det=[int(round(d["det"][0] * fx)), int(round(d["det"][1] * fy)),
                           int(round(d["det"][2] * fx)), int(round(d["det"][3] * fy))]) for d in dets_v]
    grads = {"vita": L.grad_assign(rgb_v, lines_v, dets_v, "vita")}
    psp_ok = [L.size_ok(ln) for ln in lines_p]
    grads["psp"] = L.grad_assign(rgb_p, [ln for ln, ok in zip(lines_p, psp_ok) if ok],
                                 dets_p, "psp") if any(psp_ok) else []
    gp = iter(grads["psp"])
    gpsp = [next(gp) if ok else None for ok in psp_ok]
    m = src.get(sk, {})
    session, dims = m.get("session") or s.get("session"), m.get("dims")
    for r, lv, lp, g_v, g_p, pok in zip(recs, lines_v, lines_p, grads["vita"], gpsp, psp_ok):
        e = (edge or {}).get(r["id"])
        if e is not None:
            fc_v, flags, edge_why = e["crop"], e.get("flags") or [], e["edge_why"]
            if not e["edge_ok"]:
                drops[edge_why] += 1
                continue
        else:
            fc_v, _, flags = edge_flags(gray_v, idx[r["id"]][1], "vita", Wv, Hv, r["label"])
            edge_why = "unflagged"
            if flags:
                drops["edge-unchecked"] += 1
                continue
        chars_v = L.label_chars(lv, vote.key, r["label"])
        fc_p = L.final_crop(lp, "psp", Wp, Hp, scale_chars(chars_v, fx, fy))
        fr_v = L.crop_box(idx[r["id"]][1], "vita", Wv, Hv)          # the frame the reviewers saw
        fr_p = [max(0, int(fr_v[0] * fx)), max(0, int(fr_v[1] * fy)), min(Wp, -int(-fr_v[2] * fx // 1)),
                min(Hp, -int(-fr_v[3] * fy // 1))]
        row = {"id": r["id"], "shot": sk, "vn": s["vn"], "image": int(sk.split("_")[1]), "session": session,
               "release": s.get("release"), "how": r["how"], "support": r["support"], "readers": r["readers"],
               "pass1": r.get("pass1"), "pass2": r.get("pass2"), "blind": r.get("blind"),
               "looks": r.get("looks"), "name": r.get("name", False), "edge": {"flags": flags, "why": edge_why},
               "reviewer_kind": "assistant", "chars": len(vote.key(r["label"])), "crops": {}}
        out = []
        for target, ln, g, rgb, W, H, fc, fr in (("vita", lv, g_v, rgb_v, Wv, Hv, fc_v, fr_v),
                                                 ("psp", lp, g_p, rgb_p, Wp, Hp, fc_p, fr_p)):
            if target == "psp" and not pok:
                drops["psp-too-small"] += 1
                continue
            dets = dets_v if target == "vita" else dets_p
            for crop in ("ink", "grad"):
                if crop == "ink":
                    box = list(fc)
                    if ink_mixed(box, ln, dets):
                        drops[f"{target}-ink-mixed"] += 1
                        continue
                else:
                    if g is None or g["status"] != "ok":
                        drops[f"{target}-grad-{g['status'] if g else 'none'}"] += 1
                        continue
                    box = g["box"]
                    # devnative build.grad_default (0cb084e): a GRAD box that runs beyond the
                    # reviewed frame may hold text the label lacks; without a GRAD check it gets no
                    # crop. build.covers (f523e30): it must hold the final crop's labelled extent.
                    if GC.needs_check(box, fr, L.ink_height(ln)):
                        drops[f"{target}-grad-unchecked"] += 1
                        continue
                    if not B.covers(box, fc, target):
                        drops[f"{target}-grad-cuts-label"] += 1
                        continue
                img, dark, inverted = L.cut(rgb, box)
                if img.shape[0] < 4 or img.shape[1] < 4:
                    drops[f"{target}-{crop}-tiny"] += 1
                    continue
                ch, ch_ok = chars_in(lv, r["label"], box, *((1.0, 1.0) if target == "vita" else (fx, fy)))
                pol = "dark-on-light" if dark else "bright-on-dark"
                meta = {"real": True, "pool": "realplus", "id": r["id"], "vn": s["vn"],
                        "session": session, "release": s.get("release"), "kind": "hires",
                        "target": target, "crop": crop, "box": box, "dark": dark, "inverted": inverted,
                        "how": r["how"], "pol": pol, "support": r["support"], "readers": r["readers"],
                        "decision": (r.get("pass1") or {}).get("decision"), "looks": r.get("looks"),
                        "reviewer_kind": "assistant", "name": r.get("name", False), "chars": ch,
                        "chars_match": ch_ok, "src_dims": dims, "screen": [W, H]}
                out.append(((target, crop), img, meta))
                row["crops"][f"{target}_{crop}"] = box
        if out:
            cut[r["id"]] = (row, out)
        else:
            drops["no-crop"] += 1
    return cut, drops


_POOL = {}


def _cut_chunk(keys):
    c = _POOL
    out, drops = {}, collections.Counter()
    for sk in keys:
        cut, d = cut_shot(c["shots"][sk], c["by_shot"][sk], c["idx"], c["src"], c["edge"])
        out.update(cut)
        drops += d
    return out, drops


def build_lines(labels, shots, excluded_vns=frozenset(), excluded_shots=frozenset(), src=None, edge=None,
                procs=1):
    """Cut every kept line: {(target, crop): [(crop, label, meta)]}, manifest rows, drop counts.
    src: {shot key: shots.py manifest row} for the capture session and source size. edge: {line id:
    edgecheck.py apply record} for lines whose final crop was checked on a contact sheet.

    devnative f523e30's final crop: the ink crop widened to the labelled characters' boxes
    (lines.final_crop). A line edgecheck.flags flags (widened, or ink continuing past the final
    crop) keeps its label only when an edge check picked it; otherwise it is dropped (edge-unchecked).
    The repeat caps apply after that, so a dropped line does not use up a label's quota."""
    src = src or {}
    vocab = load_vocab()
    idx = S.line_index(shots)
    drops = collections.Counter()
    cands = []
    for r in labels:
        if r["status"] != "kept":
            continue
        if r["vn"] in excluded_vns or r["shot"] in excluded_shots:
            drops["excluded-late"] += 1
            continue
        if set(vote.key(r["label"])) - vocab or set(r["label"]) - vocab:
            drops["out-of-vocab"] += 1
            continue
        cands.append(r)
    by_shot = collections.defaultdict(list)
    for r in cands:
        by_shot[r["shot"]].append(r)
    keys = sorted(by_shot, key=lambda k: (int(k.split("_")[0][1:]), k))
    cut = {}
    if procs > 1 and len(keys) > 1:
        import multiprocessing as mp
        _POOL.update(shots=shots, by_shot=by_shot, idx=idx, src=src, edge=edge)
        chunks = [keys[i::procs * 8] for i in range(procs * 8)]
        try:
            with mp.get_context("fork").Pool(procs) as pool:
                for part, d in pool.imap_unordered(_cut_chunk, chunks):
                    cut.update(part)
                    drops += d
        finally:
            _POOL.clear()
    else:
        for sk in keys:
            part, d = cut_shot(shots[sk], by_shot[sk], idx, src, edge)
            cut.update(part)
            drops += d
    kept, capped = cap_repeats([r for r in cands if r["id"] in cut])
    drops["repeat-cap"] += len(capped)
    files = collections.defaultdict(list)
    man = []
    by_id = {r["id"]: r for r in kept}
    for lid in sorted(by_id, key=lambda i: (int(i.split("_")[0][1:]), i)):
        row, out = cut[lid]
        for key, img, meta in out:
            files[key].append((img, by_id[lid]["label"], meta))
        man.append(row)
    return files, man, drops, kept


def late_exclusions(src_rows, saved_vns=(), current_vns=(), dup_images=(), leak_shots=()):
    """VNs and shot keys to leave out at build time, whatever was decided earlier: the reserved VNs
    given (cmd_build: the reservation recomputed from today's inputs, shots.reserved_vns), screenshots
    dedupe marked as copies (of a pool image, a fresh native test image or another realplus image) and
    shots the leak check dropped. Shot keys are "<vn>_<image>"."""
    vns = set(saved_vns) | set(current_vns)
    dup = {str(i) for i in dup_images}
    shots_ = {f"{r['vn']}_{r['image']}" for r in src_rows if str(r["image"]) in dup} | set(leak_shots)
    return vns, shots_


def build_paths(tag=None):
    """The build's outputs. Built files are pinned by their users (VOCR3 pins the npz sha256s), so a
    build never replaces one: a new build needs a new --tag, whose files carry it in their names."""
    sfx = f"_{tag}" if tag else ""
    return {"ink": OUT / f"realplus_train_ink{sfx}.npz", "grad": OUT / f"realplus_train_grad{sfx}.npz",
            "lines": OUT / "manifests" / f"lines{sfx}.jsonl", "full": OUT / f"lines_full{sfx}.jsonl",
            "stats": OUT / "manifests" / f"build_stats{sfx}.json"}


def cmd_build(args):
    out = build_paths(args.tag)
    there = [str(v) for v in out.values() if v.exists()]
    if there:
        sys.exit(f"refusing to overwrite a built pool ({', '.join(there)}): its files are pinned; "
                 "build with a new --tag (new file names) and tell the conductor")
    labels = jl(OUT / "labels.jsonl")
    shots = load_reads()
    rows_ = SH.load_shots()
    # The reservation recomputed from today's dump and the committed eval/manifests/reserved.json (docs/
    # reservations.md). A reserved_vns.json saved at selection under the old rules (hash splits, development, fresh
    # native, owner test series) no longer excludes anything; select under the new rules rewrites it.
    current = SH.reserved_vns(SH.load_dump())
    leak = OUT / "manifests" / "leak_drop.json"
    late_vns, late_shots = late_exclusions(
        rows_, (), set(current), dups(), json.loads(leak.read_text()) if leak.exists() else ())
    src = {f"{r['vn']}_{r['image']}": r for r in rows_}
    edge_p = OUT / "edge" / "edge.jsonl"
    edge = {r["id"]: r for r in jl(edge_p)} if edge_p.exists() else None
    files, man, drops, kept = build_lines(labels, shots, late_vns, late_shots, src, edge, procs=args.procs)
    counts = {}
    for crop in ("ink", "grad"):
        items = files.get(("vita", crop), []) + files.get(("psp", crop), [])
        name = out[crop].name
        save_npz(out[crop], [i[0] for i in items], [i[1] for i in items], [i[2] for i in items])
        counts[name] = {"crops": len(items), "lines": len({i[2]["id"] for i in items}),
                        "vns": len({i[2]["vn"] for i in items}),
                        "by_target": dict(collections.Counter(i[2]["target"] for i in items)),
                        "chars": sum(len(vote.key(i[1])) for i in items), "sha256": P.file_sha256(out[crop])}
    wjl(out["lines"], man)
    full = {r["id"]: r for r in kept}
    wjl(out["full"], [dict(m, label=full[m["id"]]["label"]) for m in man])
    stats = {"lines": len(man), "vns": len({m["vn"] for m in man}), "screenshots": len({m["shot"] for m in man}),
             "how": dict(collections.Counter(m["how"] for m in man)), "names": sum(m["name"] for m in man),
             "files": counts, "dropped_at_build": dict(drops)}
    out["stats"].write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1), flush=True)
    print("BUILD_DONE", flush=True)


# ----------------------------------------------------------------------------- blind re-check

def cmd_blind(args):
    """A blind re-check sample: --n random kept lines of the frozen labels (labels.jsonl), in a
    pre-registered random order (blind_ids.json "order"), each framed on its final Vita ink crop
    (edge/boxes.jsonl: the crop the npz holds), candidates shuffled and unattributed (sheets mode
    blind). It can run while edge checks finish: blind-score keeps the first --score lines of the
    order that end up in the built pool (a line can still fail its edge check or a build rule)."""
    labels = jl(OUT / "labels.jsonl")
    boxes = {b["id"]: b for b in jl(EDGE / "boxes.jsonl")}
    spec_p = REVIEW / "blind_ids.json"
    if args.extend:
        # Continue the same sampling: more lines of the same frozen kept set, in a random order that
        # follows the registered one (seeded from the registered seed and the extension's number).
        spec = json.loads(spec_p.read_text())
        frozen = set(json.loads((REVIEW / "blind_frame.json").read_text())) if (REVIEW / "blind_frame.json").exists() \
            else None
        kept = sorted(frozen) if frozen is not None else \
            sorted(r["id"] for r in labels if r["status"] == "kept" and r["id"] in boxes)
        drawn = set(spec["order"])
        ext = spec.setdefault("extensions", [])
        rng = random.Random(f"{spec['seed']}:{len(ext) + 1}")
        new = rng.sample([i for i in kept if i not in drawn], args.extend)
        ext.append({"n": len(new), "seed": f"{spec['seed']}:{len(ext) + 1}"})
        spec["order"] += new
        spec_p.write_text(json.dumps(spec) + "\n")
        order, first = new, len(batch_dirs("blind")) + 1
    else:
        kept = sorted(r["id"] for r in labels if r["status"] == "kept" and r["id"] in boxes)
        if len(kept) < sum(r["status"] == "kept" for r in labels):
            sys.exit("edge-boxes first: every kept label needs its final crop")
        rng = random.Random(args.seed)
        order = rng.sample(kept, min(args.n, len(kept)))
        spec_p.write_text(json.dumps({"order": order, "seed": args.seed, "from": len(kept), "score": args.score})
                          + "\n")
        (REVIEW / "blind_frame.json").write_text(json.dumps(kept) + "\n")
        first = 1
    shots = load_reads()
    entries = S.review_entries(shots, order, frames={i: boxes[i]["crop"] for i in order})
    for k in range(0, len(entries), args.size):
        out = REVIEW / "blind" / f"b{first + k // args.size:04d}"
        ns = S.make_batch(entries[k:k + args.size], out, "blind", name=f"realplus/blind/{out.name}")
        with open(out / "README.md", "a", encoding="utf-8") as f:
            f.write(NOTE + BLIND_NOTE)
        print(f"{out}: {len(entries[k:k + args.size])} lines, {ns} sheets", flush=True)


def blind_scored(order, built, decided, n):
    """The first n ids of the pre-registered order that are in the built pool (with a Vita ink crop)
    and have a blind decision."""
    return [i for i in order if i in built and i in decided][:n]


def measured_pool(built, labels, frozen):
    """The pool as the blind check measured it: the built lines plus those a third look dropped
    afterwards (why "blind-*"), which take their label from the frozen labels the sample was drawn
    from. Re-scoring after the third looks then reproduces the measurement instead of replacing
    each dropped disagreement with the next line of the order."""
    pool, labs = set(built), dict(labels)
    for lid, r in labels.items():
        if r["status"] == "dropped" and str(r.get("why") or "").startswith("blind-") and lid in frozen:
            pool.add(lid)
            labs[lid] = frozen[lid]
    return pool, labs


def cmd_blind_score(args):
    """decide.py's blind agreement (devnative.decide.blind_agreement, Wilson 95% interval) on the
    pre-registered blind sample, restricted to lines of the pool as measured (measured_pool) ->
    review/blind_report.json."""
    spec = json.loads((REVIEW / "blind_ids.json").read_text())
    labels = {r["id"]: r for r in jl(OUT / "labels.jsonl")}
    frozen = {r["id"]: r for r in jl(OUT / FROZEN_LABELS)} if (OUT / FROZEN_LABELS).exists() else {}
    built = {m["id"] for m in jl(build_paths(args.tag)["lines"]) if "vita_ink" in m["crops"]}
    built, labels = measured_pool(built, labels, frozen)
    blind = D.collect(batch_dirs("blind"))
    ids = blind_scored(spec["order"], built, blind, spec.get("score", 500))
    summ, dis = D.blind_agreement(labels, {i: blind[i] for i in ids})
    first = len(spec["order"]) - sum(e["n"] for e in spec.get("extensions", []))
    summ["sample"] = {"drawn": len(spec["order"]), "decided": len(blind),
                      "in_pool": sum(i in built for i in spec["order"]), "scored": len(ids), "seed": spec["seed"],
                      "from": spec["from"], "scored_from_first_draw": sum(i in set(spec["order"][:first]) for i in ids),
                      "last_position": spec["order"].index(ids[-1]) + 1 if ids else 0}
    out = REVIEW / (f"blind_report_{args.tag}.json" if args.tag else "blind_report.json")
    out.write_text(json.dumps({"summary": summ, "disagreements": dis, "scored_ids": ids}, ensure_ascii=False, indent=1)
                   + "\n", encoding="utf-8")
    print(json.dumps(summ, indent=1), flush=True)


# ----------------------------------------------------------------------------- publish

MANIFESTS = HERE / "manifests"
MAX_FILE = 1_900_000                  # committed files stay under tests/test_repo_policy.py's 2 MiB


def write_split(rows, stem, out_dir=MANIFESTS, limit=MAX_FILE):
    """rows as JSON lines in <stem>_01.jsonl, <stem>_02.jsonl ..., each under `limit` bytes; stale
    files of the same stem are removed. Returns the file names."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parts, cur, size = [], [], 0
    for r in rows:
        line = json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
        if cur and size + len(line.encode()) > limit:
            parts.append(cur)
            cur, size = [], 0
        cur.append(line)
        size += len(line.encode())
    if cur:
        parts.append(cur)
    names = []
    for i, part in enumerate(parts, 1):
        name = f"{stem}_{i:02d}.jsonl"
        (out_dir / name).write_text("".join(part), encoding="utf-8")
        names.append(name)
    for old in out_dir.glob(f"{stem}_*.jsonl"):
        if old.name not in names:
            old.unlink()
    return names


def decision_counts(kind):
    c = collections.Counter()
    for b in done_batches(kind):
        for r in D.parsed(b):
            c[r["decision"] + (f"-{r['skip']}" if r["decision"] == "skip" else "")] += 1
    return dict(c)


def last_json(path):
    """The last JSON object line of a log (the leak check's summary), or None."""
    if not Path(path).exists():
        return None
    for x in reversed(Path(path).read_text(encoding="utf-8").splitlines()):
        if x.startswith("{"):
            return json.loads(x)
    return None


def cmd_publish(args):
    """The committed, text-free part of the pool: realplus/manifests/shots_NN.jsonl (every screenshot a
    built line comes from: URL, file and pixel sha256, VN, release, capture session, line and crop
    counts) and realplus/manifests/stats.json (counts at every stage, npz and local-manifest hashes,
    the blind re-check). Labels and crop boxes stay in the local lines manifests, pinned by hash."""
    bp = build_paths(args.tag)
    man = jl(bp["lines"])
    if not man:
        sys.exit("build first")
    build = json.loads(bp["stats"].read_text())
    rows = {f"{r['vn']}_{r['image']}": r for r in SH.load_shots()}
    nl, nc = collections.Counter(), collections.Counter()
    for m in man:
        nl[m["shot"]] += 1
        nc[m["shot"]] += len(m["crops"])
    shots_ = []
    for sk in sorted(nl, key=lambda k: (int(k.split("_")[0][1:]), int(k.split("_")[1]))):
        r = rows[sk]
        shots_.append({"shot": sk, "vn": r["vn"], "url": r["url"], "sha256": r["sha256"],
                       "pixel_sha256": r["pixel_sha256"], "dims": r["dims"], "release": r["release"],
                       "session": r["session"], "lines": nl[sk], "crops": nc[sk]})
    names = write_split(shots_, "shots")
    sel = json.loads((OUT / "manifests" / "selection.json").read_text())
    fetched = collections.Counter("ok" if r.get("sha256") else r.get("status", "not-fetched") for r in rows.values())
    dump = SH.realplus_dir() / "vndb" / f"vndb-db-{args.dump_date}.tar.zst"
    labels = jl(OUT / "labels.jsonl")
    blind = OUT / "review" / "blind_report_measured.json"      # as scored, before the third looks
    if not blind.exists():
        blind = OUT / "review" / "blind_report.json"
    looks3 = [x for f in sorted(REVIEW.glob("resolutions*.jsonl")) for x in jl(f)]
    stats = {
        "pool": "realplus",
        "pipeline": {"devnative": "f5a1dbd: read and sheets as frozen at 35a86ca, GRAD containment 0cb084e, "
                                  "final crops, edge check and looks f523e30",
                     "devnative_check": "every devnative function realplus calls has the same body at 14ae22d "
                                        "(edgecheck.flags gained crop=, default unchanged; decide.wilson95 factored "
                                        "out); edge flags recomputed at 14ae22d matched on 2,000 of 2,000 lines",
                     "code_sha256": {n: P.file_sha256(HERE / n) for n in ("shots.py", "pool.py")},
                     "published_at": args.commit},
        "vndb_dump": {"file": dump.name, "sha256": P.file_sha256(dump) if dump.exists() else None,
                      "bytes": dump.stat().st_size if dump.exists() else None, "timestamp": sel["dump"]},
        "selection": {k: sel[k] for k in ("kinds", "screenshots", "vns", "dropped", "reserved_vns",
                                          "reserved_screenshots")},
        "fetch": dict(fetched),
        "duplicates": len(dups()),
        "read": {"screenshots": len(load_reads())},
        "review": {"pass1": decision_counts("pass1"), "edge": decision_counts("edge"),
                   "pass2": decision_counts("pass2"), "reviewer_kind": "assistant"},
        "labels": dict(collections.Counter(f"{r['status']}:{r['how'] if r['status'] == 'kept' else r.get('why')}"
                                           for r in labels)),
        "edge_check": dict(collections.Counter(e["edge_why"] for e in jl(EDGE / "edge.jsonl"))),
        "leakcheck": {name: last_json(OUT / "logs" / f"{name}.log")
                      for name in ("leakcheck", "leakcheck_devnative", "leakcheck_teacher_val")},
        "leak_dropped_screenshots": len(json.loads((OUT / "manifests" / "leak_drop.json").read_text()))
        if (OUT / "manifests" / "leak_drop.json").exists() else 0,
        "blind_sample": {k: v for k, v in json.loads((REVIEW / "blind_ids.json").read_text()).items() if k != "order"}
        if (REVIEW / "blind_ids.json").exists() else None,
        "build": build,
        "screenshot_manifests": names,
        "local_manifests": {n: P.file_sha256(OUT / n) for n in (str(bp["lines"].relative_to(OUT)), "labels.jsonl",
                                                                bp["full"].name, "edge/boxes.jsonl",
                                                                "edge/edge.jsonl", "review/blind_ids.json",
                                                                "review/resolutions.jsonl",
                                                                "review/blind_report_measured.json",
                                                                FROZEN_LABELS, "logs/v2_vs_v1.json",
                                                                "logs/strict_edge.log", "logs/leakcheck.log",
                                                                "logs/leakcheck_devnative.log",
                                                                "logs/leakcheck_teacher_val.log",
                                                                "manifests/dups.json", "manifests/leak_drop.json")
                            if (OUT / n).exists()},
        "batches_used": {k: [b.name for b in done_batches(k)] for k in ("pass1", "pass2", "edge", "blind")},
        "blind": json.loads(blind.read_text())["summary"] if blind.exists() else None,
        "blind_on_this_build": json.loads((REVIEW / f"blind_report_{args.tag}.json").read_text())["summary"]["all"]
        if args.tag and (REVIEW / f"blind_report_{args.tag}.json").exists() else None,
        "build_tag": args.tag,
        "supersedes": {"build": "untagged", "files": {n: f["sha256"] for n, f in
                                                      json.loads(build_paths()["stats"].read_text())["files"].items()},
                       "lines": json.loads(build_paths()["stats"].read_text())["lines"]}
        if args.tag and build_paths()["stats"].exists() else None,
        "third_looks": {"lines": len(looks3), "kept": sum(bool(x.get("keep")) for x in looks3),
                        "why": dict(collections.Counter(x.get("why") for x in looks3)), "reviewer_kind": "assistant"},
    }
    (MANIFESTS / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"shots": len(shots_), "files": names, "lines": len(man)}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    sub.add_parser("plan")
    p = sub.add_parser("batches")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--size", type=int, default=200)
    p = sub.add_parser("helpers")
    p.add_argument("--max", type=int, default=8)
    p.add_argument("--kinds", default="pass1")
    p.add_argument("--retry", action="store_true", help="re-queue batches whose helper ended without decisions")
    sub.add_parser("status")
    p = sub.add_parser("feed")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--every", type=int, default=180)
    p.add_argument("--ready", type=int, default=3)
    p.add_argument("--size", type=int, default=200)
    p.add_argument("--max", type=int, default=8)
    p.add_argument("--min-new", type=int, default=400)
    p.add_argument("--kinds", default="edge,pass1", help="batch kinds helpers are queued for, in order")
    p = sub.add_parser("reserve")
    p.add_argument("batch", nargs="+")
    p = sub.add_parser("final")
    p.add_argument("--no-resolve", action="store_true")
    p = sub.add_parser("build")
    p.add_argument("--procs", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "1")),
                   help="worker processes (default: heavy.sh's 5 threads per slot)")
    p.add_argument("--tag", help="a new build's name suffix: built files are never overwritten")
    sub.add_parser("edge-boxes")
    p = sub.add_parser("edge-batches")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--size", type=int, default=200)
    p.add_argument("--min", type=int, help="smallest batch to write (default --size)")
    sub.add_parser("edge-apply")
    p = sub.add_parser("publish")
    p.add_argument("--dump-date", default="2026-10-12")
    p.add_argument("--tag", help="publish this tagged build")
    p.add_argument("--commit", default=subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                                      capture_output=True, text=True).stdout.strip())
    p = sub.add_parser("blind")
    p.add_argument("--n", type=int, default=540, help="lines drawn (a few may leave the pool at the build)")
    p.add_argument("--score", type=int, default=500, help="lines scored: the first of the order in the pool")
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--size", type=int, default=135)
    p.add_argument("--extend", type=int, default=0, help="continue the registered order with N more lines")
    p = sub.add_parser("blind-score")
    p.add_argument("--tag", help="score against this tagged build")
    args = ap.parse_args()
    {"prep": cmd_prep, "plan": cmd_plan, "batches": cmd_batches, "helpers": cmd_helpers, "status": cmd_status,
     "final": cmd_final, "build": cmd_build, "blind": cmd_blind, "reserve": cmd_reserve,
     "feed": cmd_feed, "edge-boxes": cmd_edge_boxes, "edge-batches": cmd_edge_batches,
     "edge-apply": cmd_edge_apply, "publish": cmd_publish, "blind-score": cmd_blind_score}[args.cmd](args)


if __name__ == "__main__":
    main()
