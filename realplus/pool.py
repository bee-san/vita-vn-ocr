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
  pool.py confirm-batches          v3: second, blind Claude looks at lines Luna did not confirm -> review/confirm/bNNNN
  pool.py check --tag T            v3: the pre-registered precision sample of build T (no candidates) -> review/check
  pool.py check-score --tag T      v3: the check's agreement with the labels -> review/check_report_T.json
  pool.py look3 --tag T [--resolve]  v3: third looks at the check's disagreements -> review/resolutions_v3.jsonl
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

import shots as SH  # noqa: E402

from devnative import decide as D  # noqa: E402
from devnative import gradcheck as GC  # noqa: E402
from devnative import lines as L  # noqa: E402
from devnative import paths as P  # noqa: E402
from devnative import sheets as S  # noqa: E402
from teacher import vote  # noqa: E402

OUT = SH.realplus_dir()
READS = OUT / "reads" / "reads.jsonl"
# The reclaim tier (owner / reserve-policy 2026-10-14): screenshots of VNs the old reservation reserved and the
# new FINAL policy frees, read by devnative/read.py into their own reads.jsonl (their rows carry an absolute
# "path", so the build loads them from the reserve-policy shots dir). When present they are planned, batched,
# reviewed and built with v2's already-read lines as one v3 pool; the plan tier (t1/t2/t3) applies to them too.
RECLAIM_READS = OUT / "reclaim" / "reads" / "reads.jsonl"
REVIEW = OUT / "review"
HERD_DIR = Path(os.environ.get("HERD_DIR", "/local/home/skerraut/herd-synth3"))
REVIEW_PROMPT = HERD_DIR / "prompts" / "review.md"
# Helper agents are labelled <HELPER><n>; v2 used review-rpN, v3 review-v3-N (results.txt keeps both).
HELPER = os.environ.get("REALPLUS_HELPER", "review-rp")
# v3 (docs/realplus.md, "Version 3"): pass-1 batches from b0018 and edge batches from b0020 are decided
# under the Claude + Luna agreement rule ($REALPLUS/AGREEMENT.md); earlier batches are v2's, frozen.
V3_FIRST = {"pass1": int(os.environ.get("REALPLUS_V3_PASS1", "18")),
            "edge": int(os.environ.get("REALPLUS_V3_EDGE", "20"))}
LUNA_KIND = "codex:gpt-6-luna"
# The reservation a build applies (build_exclusions): the owner's 2026-10-14 13:25Z FINAL policy, recorded in the
# published stats.json (v2's stats, pinned in git history, predate the field).
POLICY = "final-2026-10-14"
FAMILY = {"meiki", "v6s"}
PER_VN_CAP, GLOBAL_CAP = 2, 8          # devnative/build.py's per-VN cap; teacher/build.py's overall cap
MIN_PSP_INK = L.MIN_INK                # a PSP crop needs 10 px of ink, as a native PSP line
NOTE = ("\nNote (realplus): these screenshots are PC / home-console visual-novel screenshots scaled to the "
        "PS Vita\nscreen (960 x 544); decide them exactly as above. Speaker names: add \"name\": true.\n"
        "Images: view ONE sheet per tool call, each sheet once, at most ~60 images per session. Append the\n"
        "decisions of each sheet to decisions.jsonl in this directory as soon as that sheet is decided, so a\n"
        "fresh session can resume: if decisions.jsonl already exists, keep it and continue with the first sheet\n"
        "that has no decisions yet.\n")
# v3 (07:40Z, after batch b0019 showed Luna skipping wrapped lines as cut and name plates as ui): the
# conventions restated for both reviewers; appended to every batch made from b0022 on.
WRAP_NOTE = ("Conventions restated: a screen line is one line of the text box. A wrapped line that ends at the box's\n"
             "edge and continues on the next screen line is a complete line, not cut; \"cut\" is only for a frame\n"
             "that cuts a character or text of this line running on past the frame. Speaker names (name plates,\n"
             "【name】) are labelled with \"name\": true, not skipped as ui.\n")
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


def load_reads(path=None):
    """sheets.load_reads, skipping a last line read.py is still writing. When reading the default
    reads (path omitted) and the reclaim reads exist they are merged in (their shot keys are disjoint
    from v2's: the reclaim dedupe dropped any pixel copy of a pool image), so the plan and build see
    one combined set of read screenshots."""
    shots = {}
    paths = [READS] if path is None else [path]
    if (path is None or Path(path) == READS) and RECLAIM_READS.exists():
        paths.append(RECLAIM_READS)
    for pth in paths:
        for x in Path(pth).read_text(encoding="utf-8").splitlines():
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


RECLAIM_SHOTS = OUT / "reclaim" / "manifests" / "shots.jsonl"
RECLAIM_DUPS = OUT / "reclaim" / "manifests" / "dups.json"


def reclaim_shot_rows():
    """The reclaim tier's selection rows (shots.py select format), or [] when there is no reclaim tier.
    Preferred source: the reserve-policy manifest the fetcher wrote hashes into."""
    import shots as _SH
    for p in (RECLAIM_SHOTS, _SH.ART / "reserve-policy" / "realplus" / "manifests" / "shots.jsonl"):
        if p.exists():
            return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return []


def reclaim_dups():
    return json.loads(RECLAIM_DUPS.read_text()) if RECLAIM_DUPS.exists() else {}


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


def plan(shots, skip_vns=frozenset(), skip_shots=frozenset()):
    """{line id: {"tier", "vn", "shot", "keys"}} for every candidate line, and the review order: by
    tier, then round-robin over VNs (a VN's n-th screenshot after every VN's (n-1)-th), lines of
    one screenshot together, top to bottom. Screenshots of skip_vns (reserved VNs) and skip_shots (dedupe
    copies, leak drops: build_exclusions) get no line, but keep their place in the round-robin, so leaving
    them out never reorders the other lines."""
    lines, auto_keys = {}, collections.defaultdict(collections.Counter)
    by_vn = collections.defaultdict(list)
    for s in shots.values():
        by_vn[s["vn"]].append(s["shot"])
        if s["vn"] in skip_vns or s["shot"] in skip_shots:
            continue
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
    """plan.jsonl: the review order of every candidate line the build could keep. The build's exclusions apply
    here too (build_exclusions: reserved VNs, dedupe copies of v2's selection and the reclaim tier, among them the
    reclaim tier's PSP-scale reads, and leak drops), so no reviewer or Codex call sees such a line."""
    shots = load_reads()
    _, skip_vns, skip_shots, _ = build_exclusions()
    lines, order = plan(shots, skip_vns, skip_shots)
    c = collections.Counter(x["tier"] for x in lines.values())
    sk = sum(1 for x in lines.values() if x.get("skip"))
    out = [s for s in shots.values() if s["vn"] in skip_vns or s["shot"] in skip_shots]
    wjl(REVIEW / "plan.jsonl", [{"id": lid, "tier": lines[lid]["tier"]} for lid in order])
    print(json.dumps({"screenshots": len(shots), "candidate_lines": len(lines), "tiers": dict(c),
                      "review": len(order), "skipped_repeat_cap": sk,
                      "excluded_screenshots": {"reserved-vn": sum(s["vn"] in skip_vns for s in out),
                                               "dedupe-or-leak": sum(s["vn"] not in skip_vns for s in out)}},
                     indent=1), flush=True)


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
    tiers = set(args.tiers.split(",")) if getattr(args, "tiers", None) else None
    plan_ids = [r["id"] for r in jl(REVIEW / "plan.jsonl") if tiers is None or r["tier"] in tiers]
    done = batched_ids()
    todo = [lid for lid in plan_ids if lid not in done and lid in idx]
    have = batch_dirs()
    k = max([int(b.name[1:]) for b in have] + [0])
    made = 0
    while todo and made < args.n:
        chunk, todo = todo[:args.size], todo[args.size:]
        if len(chunk) < (getattr(args, "min", None) or 0):
            break
        # lines of one screenshot together, so the reviewer sees a name next to its dialogue
        chunk.sort(key=lambda lid: (idx[lid][0]["shot"], idx[lid][1]["ink"][1], idx[lid][1]["ink"][0]))
        k += 1
        out = REVIEW / "pass1" / f"b{k:04d}"
        entries = [(lid, idx[lid][0], idx[lid][1], S.candidates(idx[lid][1])) for lid in chunk]
        ns = S.make_batch(entries, out, "pass1", name=f"realplus/pass1/{out.name}")
        with open(out / "README.md", "a", encoding="utf-8") as f:
            f.write(NOTE + WRAP_NOTE)
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


def helper_number(label):
    """n of a helper label <HELPER><n> (0 for labels of another prefix or the agent's own self-* marks)."""
    if not label.startswith(HELPER):
        return 0
    tail = label[len(HELPER):].split("-")[0]
    return int(tail) if tail.isdigit() else 0


def live_helpers():
    """Helper labels (our prefix) the orchestrator counts as queued or running: in queue.tsv, or in
    agents.txt and not yet in closed.txt (a finished helper's tab stays open until its next poll)."""
    def labels(name, col=0):
        p = HERD_DIR / name
        return {x.split("\t")[col].split(" ")[0] for x in p.read_text(encoding="utf-8").splitlines() if x.strip()} \
            if p.exists() else set()
    live = labels("queue.tsv") | (labels("agents.txt") - labels("closed.txt"))
    return {x for x in live if x.startswith(HELPER)}


# v2 made edge batches b0001-b0030 and decided b0001-b0019; b0020-b0030 (2,091 kept v2 labels) were withdrawn at
# its freeze. v3 completes those after t2 (the plan's order). The edge batches v3 makes for its own new flagged
# lines (b0031 on) are part of labelling those lines and can go to helpers before the remaining pass-1 batches.
V2_EDGE_LAST = 30


def helper_batches(kind):
    """The batches a --kinds token names, in order: a batch kind (pass1, confirm, edge, check ...) or
    "edge-new" (only the edge batches after v2's withdrawn b0020-b0030)."""
    if kind == "edge-new":
        return [b for b in batch_dirs("edge") if int(b.name[1:]) > V2_EDGE_LAST]
    return batch_dirs(kind)


def cmd_helpers(args):
    log, res = helper_log(), results()
    busy = outstanding(log, res)
    live = live_helpers()
    n_busy = max(len(busy), len(live))
    n = max([helper_number(h["label"]) for h in log] + [0])
    queued = []
    for token in args.kinds.split(","):
        kind = token.split("-")[0]
        for b in helper_batches(token):
            if n_busy + len(queued) >= args.max:
                break
            st, _ = batch_state(b)
            if st == "done":
                continue
            prior = [h for h in log if h.get("batch") == str(b)]
            if prior and any(h in busy for h in prior):
                continue
            if any(h["label"].startswith("self") and not h.get("cancelled") for h in prior):
                continue                                  # reserved by this agent (pool.py reserve): no helper
            if prior and (not args.retry or len(prior) >= 3):     # a stalled or dead helper: at most 3 tries
                continue
            n += 1
            label = f"{HELPER}{n}"
            if os.environ.get("REALPLUS_HELPER_MODEL"):        # spawn.sh reads the per-label model from models.txt
                with open(HERD_DIR / "models.txt", "a", encoding="utf-8") as f:
                    f.write(f"{label} {os.environ['REALPLUS_HELPER_MODEL']}\n")
            cmd = [str(HERD_DIR / "enqueue.sh"), label, "-", str(REVIEW_PROMPT), f"Your batch is {b}."]
            subprocess.run(cmd, check=True, cwd=HERD_DIR, env=dict(os.environ, HERD_DIR=str(HERD_DIR)),
                           stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
            rec = {"label": label, "batch": str(b), "kind": kind, "queued": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                            time.gmtime()),
                   "model": os.environ.get("REALPLUS_HELPER_MODEL") or "default"}
            log.append(rec)
            queued.append(rec)
    wjl(REVIEW / "helpers.jsonl", log)
    print(f"{len(busy)} helpers outstanding ({len(live)} queued or with an open tab), {len(queued)} queued now: "
          f"{[q['label'] for q in queued]}", flush=True)


def cmd_reserve(args):
    """Mark batches as reviewed by this agent itself (no helper is queued for them; review.md's instructions, one
    sub-session per batch). --model records the reviewer's model, as helpers.jsonl does for helpers."""
    with review_lock():
        log = helper_log()
        for b in args.batch:
            log.append({"label": f"self-{Path(b).name}", "batch": str(Path(b)), "kind": Path(b).parent.name,
                        "queued": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "model": getattr(args, "model", None) or "default"})
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


def plan_left(tiers=None):
    """Plan lines (of these comma-separated tiers; default all) not in a pass-1 batch yet."""
    t = set(tiers.split(",")) if tiers else None
    done = batched_ids()
    return sum(1 for r in jl(REVIEW / "plan.jsonl") if (t is None or r["tier"] in t) and r["id"] not in done)


def make_batches(n, size, tiers):
    """cmd_batches for the feed. Rendering contact sheets is CPU work, so when the feed runs outside a
    heavy.sh slot (HERD set, HEAVY_SLOT unset) the batches are made in a slot of their own and the
    long-running loop does not hold one."""
    herd = os.environ.get("HERD")
    if herd and not os.environ.get("HEAVY_SLOT"):
        cmd = [f"{herd}/heavy.sh", "timeout", "3600", sys.executable, str(HERE / "pool.py"), "batches",
               "--n", str(n), "--size", str(size)] + (["--tiers", tiers] if tiers else [])
        subprocess.run(cmd, env=dict(os.environ, HEAVY_SLOTS=os.environ.get("HEAVY_SLOTS", "12")),
                       stdin=subprocess.DEVNULL, check=False, timeout=7200)
    else:
        cmd_batches(argparse.Namespace(n=n, size=size, tiers=tiers, min=None))


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
                tiers = getattr(args, "tiers", None)
                if not (REVIEW / "plan.jsonl").exists() or not tiers:
                    cmd_plan(None)
                if plan_left(tiers):
                    make_batches(args.ready, args.size, tiers)
            cmd_helpers(argparse.Namespace(max=args.max, kinds=args.kinds, retry=True))
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


RECLAIM_READS_ALL = RECLAIM_READS.with_name("reads.jsonl.prefilter")   # every reclaim read, before the 18 skipped
SHOT_KEY = re.compile(r'^\{"shot": "(v\d+_\d+)"', re.M)                # read.py writes "shot" first


def reads_snapshot():
    """READS without a partial last line (a read may be appending): devnative's loaders need whole
    lines. Returns the snapshot path. The reclaim reads are appended, and so are the reads of any
    screenshot that a batch holds a line of but that was taken out of the reclaim reads afterwards
    (2026-10-14: 18 screenshots of three VNs that became development-sample partners; one of their
    lines was already in pass-1 b0159), so that decide.py can resolve every decided line. Such lines
    stay out of the pool all the same: build_exclusions reserves their VNs."""
    snap = OUT / "reads" / "reads_snapshot.jsonl"
    text = READS.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        text = text[:text.rfind("\n") + 1]
    if RECLAIM_READS.exists():                 # decide.py resolves reclaim lines against their reads too
        rtext = RECLAIM_READS.read_text(encoding="utf-8")
        if rtext and not rtext.endswith("\n"):
            rtext = rtext[:rtext.rfind("\n") + 1]
        text += rtext
        if RECLAIM_READS_ALL.exists():
            have = set(SHOT_KEY.findall(text))
            need = {"_".join(lid.split("_")[:2]) for k in ("pass1", "confirm", "edge") for lid in batched_ids(k)}
            for x in RECLAIM_READS_ALL.read_text(encoding="utf-8").splitlines():
                m = SHOT_KEY.match(x)
                if m and m.group(1) in need and m.group(1) not in have:
                    text += x + "\n"
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
    conf = done_batches("confirm")
    if conf:
        cmd += ["--confirm", *map(str, conf)]    # v3: a disagreeing confirm look drops the line
    res = sorted(REVIEW.glob("resolutions*.jsonl"))
    if res and not args.no_resolve:
        cmd += ["--resolve", *map(str, res)]
    cmd += ["--mode", "train", "--auto-consensus", "--out", str(OUT / "labels.jsonl")]
    if not p1:
        sys.exit("no decided pass-1 batch yet")
    subprocess.run(cmd, check=True)
    labels = jl(OUT / "labels.jsonl")
    if res and not args.no_resolve:
        n = apply_auto_resolutions(labels, {r["id"]: r for f in res for r in jl(f)})
        print(f"third looks applied to {n} auto lines", flush=True)
    if v3_batch_names("pass1"):
        import luna as LU
        LU.cmd_merge(argparse.Namespace(first=V3_FIRST["pass1"]))      # Claude's decisions x Luna's votes
        c = apply_luna(labels, LU.load_votes(V3_FIRST["pass1"]))
        print(f"Luna rule applied to v3 lines: {json.dumps(c)}", flush=True)
    wjl(OUT / "labels.jsonl", labels)


def v3_batch_names(kind):
    """Names of the v3 batches of a kind (b-numbers from V3_FIRST[kind]); earlier ones are v2's, frozen."""
    return {b.name for b in batch_dirs(kind) if int(b.name[1:]) >= V3_FIRST.get(kind, 1)}


def apply_luna(labels, votes, v3=None):
    """AGREEMENT.md's rule on the lines a v3 pass-1 batch decided (votes: luna.load_votes). A line
    Claude read is kept only when Luna read the same key; a Luna skip or unsure sends it to a
    confirm look (status "pending" until the look exists; decide.py's --confirm has already dropped
    the lines whose look disagreed); a different Luna reading drops it; no vote drops it at the
    freeze. A label only one engine family read also needs the confirm look (v2's pass 2). Records
    get reviewer_kinds, luna and luna_vote. Returns the counts of what changed."""
    v3 = v3_batch_names("pass1") if v3 is None else set(v3)
    c = collections.Counter()
    for r in labels:
        b = (r.get("pass1") or {}).get("batch")
        if b not in v3:
            r.setdefault("reviewer_kinds", ["assistant"])
            continue
        v = votes.get(r["id"])
        r["reviewer_kinds"] = ["assistant", LUNA_KIND]
        r["luna"] = v["luna"] if v else "missing"
        r["luna_vote"] = v.get("luna_vote") if v else None
        if r["status"] != "kept":
            c[f"not-kept:{r['why']}"] += 1
            continue
        st = r["luna"]
        confirmed = (r.get("confirm") or {}).get("agree")
        if st == "missing":
            r.update(status="dropped", why="luna-missing", label=None)
        elif st == "disagree":
            r.update(status="dropped", why="luna-disagree", label=None)
        elif st != "agree" and not confirmed:                   # skip-ui / skip-cut / skip-bad / unsure
            r.update(status="pending", why="luna-unconfirmed")
        elif r["how"] == "single" and not confirmed and not (r.get("pass2") or {}).get("agree"):
            r.update(status="pending", why="confirm-missing")
        else:
            r["looks"] = (r.get("looks") or 1) + (st == "agree")
        c[f"{r['status']}:{r['why'] if r['status'] != 'kept' else r['luna']}"] += 1
    return dict(c)


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
    _, skip_vns, skip_shots, _ = build_exclusions()   # lines the build drops need no frame (and may have no read)
    shots = load_reads()
    idx = S.line_index(shots)
    labels = [r for r in jl(OUT / "labels.jsonl") if r["status"] in ("kept", "pending") and r["vn"] not in skip_vns
              and r["shot"] not in skip_shots and r["id"] in idx]
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
                    "check": bool(why), "why": why, "status": r["status"]})
    EDGE.mkdir(parents=True, exist_ok=True)
    wjl(EDGE / "boxes.jsonl", out)
    print(json.dumps({"lines": len(out), "pending": sum(x["status"] == "pending" for x in out),
                      "check": sum(x["check"] for x in out if x["status"] == "kept"),
                      "why": dict(collections.Counter(w for x in out for w in x["why"]))}), flush=True)


def edge_entries(boxes, have, shots, skip_vns=frozenset(), skip_shots=frozenset()):
    """Sheet entries (sheets.make_batch, mode edge: the frame is the final crop, the one candidate
    its label) for flagged kept lines not in any edge batch yet, in review order. Lines of skip_vns /
    skip_shots (build_exclusions: the build drops them) get no edge check."""
    idx = S.line_index(shots)
    todo = [b for b in boxes if b["check"] and b.get("status", "kept") == "kept" and b["id"] not in have
            and b["id"] in idx and b["shot"].split("_")[0] not in skip_vns and b["shot"] not in skip_shots]
    todo.sort(key=lambda b: (vn_rank(b["shot"].split("_")[0]), b["shot"], b["id"]))
    return [(b["id"], idx[b["id"]][0], idx[b["id"]][1],
             [{"text": b["label"], "key": vote.key(b["label"]), "engines": []}], b["crop"]) for b in todo]


def cmd_edge_batches(args):
    """Disjoint edge-check batches of --size flagged lines (review/edge/bNNNN)."""
    boxes = jl(EDGE / "boxes.jsonl")
    _, skip_vns, skip_shots, _ = build_exclusions()
    with review_lock():
        have = batched_ids("edge")
        entries = edge_entries(boxes, have, load_reads(), skip_vns, skip_shots)
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


# ----------------------------------------------------------------------------- v3: confirm looks

CONFIRM_NOTE = ("This is a confirmation look (v3, AGREEMENT.md rule 3): a second reviewer did not confirm the first\n"
                "reading of these lines. Decide each line from the image alone, exactly as above: pick or type what\n"
                "the frame shows; skip \"ui\", \"cut\" or \"bad\" only as defined above; unsure when you cannot\n"
                "tell.\n")


def cmd_confirm_batches(args):
    """Confirm-look batches (sheets mode blind: candidates shuffled and unattributed, framed on the
    final crop) of the pending v3 lines (labels.jsonl status "pending") not in a confirm batch yet."""
    _, skip_vns, skip_shots, _ = build_exclusions()          # lines the build drops get no confirm look
    labels = [r for r in jl(OUT / "labels.jsonl") if r["status"] == "pending" and r["vn"] not in skip_vns
              and r["shot"] not in skip_shots]
    boxes = {b["id"]: b for b in jl(EDGE / "boxes.jsonl")}
    with review_lock():
        have = batched_ids("confirm")
        ids = [r["id"] for r in labels if r["id"] not in have and r["id"] in boxes]
        ids.sort(key=lambda i: (vn_rank(i.split("_")[0]), i))
        shots = load_reads()
        entries = S.review_entries(shots, ids, frames={i: boxes[i]["crop"] for i in ids})
        start = len(batch_dirs("confirm"))
        for k in range(args.n):
            part = entries[k * args.size:(k + 1) * args.size]
            if not part or len(part) < (args.min if args.min is not None else args.size):
                break
            out = REVIEW / "confirm" / f"b{start + k + 1:04d}"
            n = S.make_batch(part, out, "blind", name=f"realplus/confirm/{out.name}")
            with open(out / "README.md", "a", encoding="utf-8") as f:
                f.write(NOTE + CONFIRM_NOTE)
            print(f"{out}: {len(part)} lines, {n} sheets", flush=True)


# ----------------------------------------------------------------------------- v3: independent precision check

CHECK_NOTE = ("This is a label-precision check (v3): transcribe the framed text whatever it is (dialogue, a menu,\n"
              "a button, a name), including every bracket and punctuation mark inside the frame; do not use skip\n"
              "\"ui\". Vita entries are at the Vita scale (960 x 544), PSP entries at the PSP scale (480 x 272).\n")
# The sample (pre-registered here before any check sheet existed): entries are (line, one of its crops).
# Amended 2026-10-15, still before any check sheet existed: the reclaim tier (2026-10-14) brought about 20k new
# auto lines, which would otherwise have fallen into new:edge; they get their own stratum, new:auto.
CHECK_STRATA = {"new:t1": 300, "new:t2": 100, "new:auto": 200, "new:edge": 100, "v2:auto": 120, "v2:reviewed": 80}
CROP_KINDS = ("vita_ink", "psp_ink", "vita_grad", "psp_grad")


def check_stratum(m, v2_ids, v3_pass1, tiers):
    """Stratum of a built line: new (not in v2) by its v3 review tier (plan tier of the line a v3 pass-1
    batch decided), "auto" (an auto line v2 did not build: the reclaim tier's, or a flagged one whose edge
    check v3 made) or "edge" (a reviewed v2 label whose withdrawn edge check v3 completed); v2 lines by auto /
    reviewed."""
    if m["id"] in v2_ids:
        return "v2:auto" if m["how"] == "auto" else "v2:reviewed"
    if (m.get("pass1") or {}).get("batch") in v3_pass1:
        return f"new:{tiers.get(m['id'], 't?')}"
    return "new:auto" if m["how"] == "auto" else "new:edge"


def check_sample(man, v2_ids, v3_pass1, tiers, seed, strata=CHECK_STRATA):
    """[{id, crop, stratum}]: `strata[s]` lines drawn at random from each stratum, each shown as one
    of its crops, chosen at random among the crops the npz holds for it (so PSP and GRAD crops are
    checked in the proportions the pool has them, and every entry is checked on exactly the pixels
    the pool holds)."""
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for m in man:
        by[check_stratum(m, v2_ids, v3_pass1, tiers)].append(m)
    out = []
    for s, n in strata.items():
        rows = sorted(by.get(s, []), key=lambda m: m["id"])
        for m in rng.sample(rows, min(n, len(rows))):
            crops = [c for c in CROP_KINDS if c in m["crops"]]
            out.append({"id": m["id"], "crop": rng.choice(crops), "stratum": s})
    return out


def check_entries(sample, man, shots):
    """Sheet entries for the check: the crop framed on its target's screen, no candidates."""
    idx = S.line_index(shots)
    rows = {m["id"]: m for m in man}
    out = []
    for e in sample:
        s, ln = idx[e["id"]]
        target = e["crop"].split("_")[0]
        box = rows[e["id"]]["crops"][e["crop"]]
        if target == "psp":
            rgb_v, rgb_p = screen(s, "vita"), screen(s, "psp")
            fx, fy = rgb_p.shape[1] / rgb_v.shape[1], rgb_p.shape[0] / rgb_v.shape[0]
            ln = scale_line(ln, fx, fy)
        row = dict(s, kind=target, shot=f"{s['shot']}@{target}")        # make_batch loads the screen by kind
        out.append((f"{e['id']}@{e['crop']}", row, ln, [], box))
    return out


def cmd_check(args):
    """The v3 precision sample (review/check_spec_<tag>.json, pre-registered) and its sheets
    (review/check/bNNNN, mode check: no candidates, the crop framed on its own screen)."""
    bp = build_paths(args.tag)
    man = jl(bp["full"])
    if not man:
        sys.exit("build first")
    spec_p = REVIEW / f"check_spec_{args.tag}.json"
    if spec_p.exists():
        sys.exit(f"{spec_p} exists: the sample is registered once")
    v2 = {m["id"] for m in jl(OUT / "manifests" / f"lines_{args.v2}.jsonl")}
    tiers = {r["id"]: r["tier"] for r in jl(REVIEW / "plan.jsonl")}
    sample = check_sample(man, v2, v3_batch_names("pass1"), tiers, args.seed)
    spec = {"tag": args.tag, "seed": args.seed, "strata": CHECK_STRATA, "from": len(man), "v2_lines": args.v2,
            "entries": sample, "counts": {"stratum": dict(collections.Counter(e["stratum"] for e in sample)),
                                          "crop": dict(collections.Counter(e["crop"] for e in sample))}}
    spec_p.write_text(json.dumps(spec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    shots = load_reads()
    order = list(range(len(sample)))
    random.Random(f"{args.seed}:order").shuffle(order)       # strata and crop kinds mixed on the sheets
    entries = check_entries([sample[i] for i in order], man, shots)
    for k in range(0, len(entries), args.size):
        out = REVIEW / "check" / f"b{k // args.size + 1:04d}"
        ns = S.make_batch(entries[k:k + args.size], out, "check", name=f"realplus/check/{out.name}")
        with open(out / "README.md", "a", encoding="utf-8") as f:
            f.write(NOTE + CHECK_NOTE)
        print(f"{out}: {len(entries[k:k + args.size])} entries, {ns} sheets", flush=True)
    print(json.dumps(spec["counts"]), flush=True)


def check_score(spec, man, decided):
    """Agreement of the independent transcriptions with the pool's labels: {stratum / crop / all:
    {agree, lines, rate, wilson95}} and the disagreements. A skip or unsure counts as a disagreement."""
    labels = {m["id"]: m["label"] for m in man}
    rows, by = [], collections.defaultdict(lambda: [0, 0])
    for e in spec["entries"]:
        d = decided.get(f"{e['id']}@{e['crop']}")
        if d is None:
            continue
        agree = d["decision"] in ("pick", "text") and vote.key(d["text"]) == vote.key(labels[e["id"]])
        rows.append({**e, "agree": agree, "decision": d["decision"], "text": d.get("text"), "skip": d.get("skip"),
                     "label": labels[e["id"]]})
        for k in ("all", f"stratum:{e['stratum']}", f"crop:{e['crop']}", f"side:{e['stratum'].split(':')[0]}"):
            by[k][0] += agree
            by[k][1] += 1
    summ = {k: {"agree": a, "lines": n, "rate": round(a / n, 4), "wilson95": D.wilson95(a, n)}
            for k, (a, n) in sorted(by.items()) if n}
    return summ, [r for r in rows if not r["agree"]], len(rows)


def cmd_check_score(args):
    spec = json.loads((REVIEW / f"check_spec_{args.tag}.json").read_text(encoding="utf-8"))
    man = jl(build_paths(args.tag)["full"])
    decided = D.collect(done_batches("check"))
    summ, dis, n = check_score(spec, man, decided)
    out = REVIEW / f"check_report_{args.tag}.json"
    out.write_text(json.dumps({"summary": summ, "scored": n, "of": len(spec["entries"]), "disagreements": dis,
                               "reviewer_kind": "assistant"}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"scored": n, "of": len(spec["entries"]), **{k: v for k, v in summ.items()
                                                                  if k == "all" or k.startswith("side")}},
                     indent=1), flush=True)
    for r in dis:
        print(json.dumps(r, ensure_ascii=False), flush=True)


# ----------------------------------------------------------------------------- v3: third looks at the check

LOOK3_NOTE = ("\nThis is a third look (v3) at lines where an independent transcription and the pool label differ, or\n"
              "the transcriber could not settle the line. The candidates are those readings, shuffled and\n"
              "unattributed (one candidate when the transcriber typed nothing). Decide from the image alone what\n"
              "the red frame shows, in the conventions above: pick the candidate that is exactly right, or type the\n"
              "text, or unsure, or skip \"cut\" / \"bad\" as defined above (never skip \"ui\": menus and names are in\n"
              "scope). Vita entries are at the Vita scale (960 x 544), PSP entries at the PSP scale (480 x 272).\n")


def look3_candidates(d):
    """The candidates a third look shows for a check disagreement: the pool label and, when the check typed
    another text, that text; shuffled per entry, unattributed."""
    cands = [{"text": d["label"], "key": vote.key(d["label"]), "engines": []}]
    if d["decision"] == "text" and d.get("text") and vote.key(d["text"]) != vote.key(d["label"]):
        cands.append({"text": d["text"], "key": vote.key(d["text"]), "engines": []})
    random.Random(f"look3:{d['id']}@{d['crop']}").shuffle(cands)
    return cands


def look3_resolutions(dis, decided, tag):
    """review/resolutions_v3.jsonl rows from the third looks (decided: {id@crop: decision}). A look that reads
    the pool label's key keeps the line (label-right); another reading drops it (label-wrong), and so do unsure
    and cut / bad: a label needs Claude and Luna to agree (AGREEMENT.md), so a third look never writes one."""
    out = []
    for d in dis:
        x = decided.get(f"{d['id']}@{d['crop']}")
        if x is None:
            raise KeyError(f"no third look for {d['id']}@{d['crop']}")
        if x["decision"] in ("pick", "text"):
            why = "label-right" if vote.key(x["text"]) == vote.key(d["label"]) else "label-wrong"
        else:
            why = "unsure" if x["decision"] == "unsure" else (x.get("skip") or "skip")
        third = x.get("text") if x["decision"] in ("pick", "text") else x["decision"]
        note = (f"third look (assistant) on {d['crop']} after the {tag} check (stratum {d['stratum']}, check "
                f"{d['decision']}): {json.dumps(third, ensure_ascii=False)}")
        if x.get("note"):
            note += f"; {x['note']}"
        out.append({"id": d["id"], "keep": why == "label-right", "why": why, "note": note})
    return out


def check_after_looks(spec, report, res):
    """The check after the third looks: per stratum / crop / side / target, lines whose label the check read
    (agree), or a third look read (look3-right), read another text (look3-wrong) or could not settle."""
    dis = {f"{d['id']}@{d['crop']}" for d in report["disagreements"]}
    by = collections.defaultdict(collections.Counter)
    for e in spec["entries"]:
        r = res.get(e["id"]) if f"{e['id']}@{e['crop']}" in dis else None
        st = "agree" if f"{e['id']}@{e['crop']}" not in dis else "look3-missing" if r is None else \
            "look3-right" if r["keep"] else "look3-wrong" if r["why"] == "label-wrong" else "look3-unsettled"
        for k in ("all", f"stratum:{e['stratum']}", f"crop:{e['crop']}", f"side:{e['stratum'].split(':')[0]}",
                  f"target:{e['crop'].split('_')[0]}"):
            by[k][st] += 1
    out = {}
    for k, c in sorted(by.items()):
        n, ok = sum(c.values()), c["agree"] + c["look3-right"]
        out[k] = {**c, "lines": n, "right": ok, "rate": round(ok / n, 4), "wilson95": D.wilson95(ok, n),
                  "wrong_wilson95": D.wilson95(c["look3-wrong"], n)}
    return out


def cmd_look3(args):
    """Third looks at the disagreements of check_report_<tag>.json: one batch review/look3/b0001 (sheets mode
    blind, framed on the crop the check showed), then --resolve: review/resolutions_v3.jsonl (final applies it)."""
    rep = json.loads((REVIEW / f"check_report_{args.tag}.json").read_text(encoding="utf-8"))
    dis = rep["disagreements"]
    b = REVIEW / "look3" / "b0001"
    if args.resolve:
        p = REVIEW / "resolutions_v3.jsonl"
        if p.exists():
            sys.exit(f"{p} exists")
        res = look3_resolutions(dis, D.collect([b]), args.tag)
        wjl(p, res)
        print(json.dumps(dict(collections.Counter((r["why"] for r in res)))), flush=True)
        return
    if b.exists():
        sys.exit(f"{b} exists")
    man = jl(build_paths(args.tag)["full"])
    ents = check_entries([{"id": d["id"], "crop": d["crop"], "stratum": d["stratum"]} for d in dis], man,
                         load_reads())
    by = {f"{d['id']}@{d['crop']}": d for d in dis}
    n = S.make_batch([(k, row, ln, look3_candidates(by[k]), box) for k, row, ln, _, box in ents], b, "blind",
                     name="realplus/look3/b0001")
    with open(b / "README.md", "a", encoding="utf-8") as f:
        f.write(NOTE + LOOK3_NOTE)
    print(f"{b}: {len(ents)} entries, {n} sheets", flush=True)


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
               "reviewer_kind": "assistant", "reviewer_kinds": r.get("reviewer_kinds", ["assistant"]),
               "luna": r.get("luna"), "confirm": r.get("confirm"),
               "chars": len(vote.key(r["label"])), "crops": {}}
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
                        "reviewer_kind": "assistant", "reviewer_kinds": r.get("reviewer_kinds", ["assistant"]),
                        "luna": r.get("luna"), "name": r.get("name", False), "chars": ch,
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


def build_exclusions():
    """What a build leaves out, whatever was decided earlier: (reserved VNs {vn: reason}, their set, shot keys,
    {shot key: selection row} of v2's selection and the reclaim tier). The reservation is recomputed from today's
    dump and unioned with the committed eval/manifests/reserved.json (shots.reserved_vns; docs/reservations.md;
    owner 2026-10-14 13:25Z: the test sets and their release partners, the CLANNAD family, the development
    sample). A reserved_vns.json saved at selection under the old rules (hash splits, development, fresh native,
    owner test series) no longer excludes anything. Shot keys: dedupe copies (v2's dups.json and the reclaim
    tier's) and the leak-check drops. hygiene.py checks the same pool."""
    rows_, rcl = SH.load_shots(), reclaim_shot_rows()
    current = SH.reserved_vns(SH.load_dump())
    leak = OUT / "manifests" / "leak_drop.json"
    vns, shots_ = late_exclusions(rows_, (), set(current), dups(),
                                  json.loads(leak.read_text()) if leak.exists() else ())
    rd = reclaim_dups()
    shots_ |= {f"{r['vn']}_{r['image']}" for r in rcl if str(r["image"]) in rd}
    return current, vns, shots_, {f"{r['vn']}_{r['image']}": r for r in rows_ + rcl}


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
    current, late_vns, late_shots, src = build_exclusions()
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
             "files": counts, "dropped_at_build": dict(drops), "policy": POLICY,
             "reserved_vns": dict(collections.Counter(current.values()))}
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


def luna_summary(d):
    """luna.py stats output without its list of batch names (a range instead)."""
    b = d.get("batches") or []
    return {**{k: v for k, v in d.items() if k != "batches"}, "batches": f"{b[0]}-{b[-1]} ({len(b)})" if b else None}


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
    rows = {f"{r['vn']}_{r['image']}": r for r in SH.load_shots() + reclaim_shot_rows()}
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
    sup = build_paths(args.supersedes)["stats"] if args.tag else None
    sup_stats = json.loads(sup.read_text()) if sup and sup.exists() else None
    sfx = f"_{args.tag}" if args.tag else ""
    luna_p = OUT / "luna" / "agreement_first10.json"
    luna_all = OUT / "luna" / "agreement_all.json"
    # the precision check runs on the build before the third looks (v3: check --tag v3pre); its third looks
    # (review/resolutions_v3.jsonl) are applied by final before this build
    csfx = f"_{args.check or args.tag}" if (args.check or args.tag) else ""
    check_p = REVIEW / f"check_report{csfx}.json"
    spec_p = REVIEW / f"check_spec{csfx}.json"
    res3_p = REVIEW / "resolutions_v3.jsonl"
    after3 = check_after_looks(json.loads(spec_p.read_text(encoding="utf-8")),
                               json.loads(check_p.read_text(encoding="utf-8")),
                               {r["id"]: r for r in jl(res3_p)}) if check_p.exists() and res3_p.exists() else None
    stats = {
        "pool": "realplus",
        "policy": POLICY,
        "pipeline": {"devnative": "f5a1dbd: read and sheets as frozen at 35a86ca, GRAD containment 0cb084e, "
                                  "final crops, edge check and looks f523e30",
                     "devnative_check": "every devnative function realplus calls has the same body at 14ae22d "
                                        "(edgecheck.flags gained crop=, default unchanged; decide.wilson95 factored "
                                        "out); edge flags recomputed at 14ae22d matched on 2,000 of 2,000 lines",
                     "code_sha256": {n: P.file_sha256(HERE / n) for n in ("shots.py", "pool.py", "luna.py")},
                     "published_at": args.commit},
        "vndb_dump": {"file": dump.name, "sha256": P.file_sha256(dump) if dump.exists() else None,
                      "bytes": dump.stat().st_size if dump.exists() else None, "timestamp": sel["dump"]},
        "selection": {k: sel[k] for k in ("kinds", "screenshots", "vns", "dropped", "reserved_vns",
                                          "reserved_screenshots")},
        "fetch": dict(fetched),
        "duplicates": len(dups()),
        "read": {"screenshots": len(load_reads())},
        "review": {"pass1": decision_counts("pass1"), "edge": decision_counts("edge"),
                   "pass2": decision_counts("pass2"), "confirm": decision_counts("confirm"),
                   "check": decision_counts("check"), "look3": decision_counts("look3"),
                   "reviewer_kind": "assistant",
                   "reviewer_kinds": {"v2 batches, auto lines": ["assistant"],
                                      "v3 pass-1 batches": ["assistant", LUNA_KIND]}},
        "labels": dict(collections.Counter(f"{r['status']}:{r['how'] if r['status'] == 'kept' else r.get('why')}"
                                           for r in labels)),
        "luna": {**json.loads(luna_p.read_text(encoding="utf-8")), "rule": "AGREEMENT.md",
                 "votes": dict(collections.Counter(r.get("luna") for r in labels if r.get("luna"))),
                 "all_batches": luna_summary(json.loads(luna_all.read_text(encoding="utf-8")))
                 if luna_all.exists() else None}
        if luna_p.exists() else None,
        "edge_check": dict(collections.Counter(e["edge_why"] for e in jl(EDGE / "edge.jsonl"))),
        "leakcheck": {name: last_json(OUT / "logs" / f"{name}.log")
                      for name in ("leakcheck", "leakcheck_devnative", "leakcheck_teacher_val", "leakcheck_reclaim")},
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
                                                                "review/resolutions_v2.jsonl",
                                                                f"review/resolutions{sfx}.jsonl",
                                                                "review/blind_report_measured.json",
                                                                f"review/check_spec{csfx}.json",
                                                                f"review/check_report{csfx}.json",
                                                                "review/look3/b0001/decisions.jsonl",
                                                                "luna/agreement_first10.json",
                                                                "luna/agreement_all.json", "AGREEMENT.md",
                                                                FROZEN_LABELS, "logs/v2_vs_v1.json",
                                                                "logs/strict_edge.log", "logs/leakcheck.log",
                                                                "logs/leakcheck_devnative.log",
                                                                "logs/leakcheck_teacher_val.log",
                                                                "logs/leakcheck_reclaim.log",
                                                                "logs/hygiene.json",
                                                                "manifests/dups.json", "manifests/leak_drop.json")
                            if (OUT / n).exists()},
        "batches_used": {k: [b.name for b in done_batches(k)] for k in ("pass1", "pass2", "edge", "blind", "confirm",
                                                                        "check", "look3")},
        "blind": json.loads(blind.read_text())["summary"] if blind.exists() else None,
        "blind_on_this_build": json.loads((REVIEW / f"blind_report_{args.tag}.json").read_text())["summary"]["all"]
        if args.tag and (REVIEW / f"blind_report_{args.tag}.json").exists() else None,
        "check": {**{k: v for k, v in json.loads(check_p.read_text(encoding="utf-8")).items() if k != "disagreements"},
                  "on_build": args.check or args.tag, "after_third_looks": after3}
        if check_p.exists() else None,
        "build_tag": args.tag,
        "supersedes": {"build": args.supersedes or "untagged",
                       "files": {n: f["sha256"] for n, f in sup_stats["files"].items()}, "lines": sup_stats["lines"]}
        if sup_stats else None,
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
    p.add_argument("--tiers", help="only plan lines of these tiers, e.g. t1,t2 (default: all)")
    p.add_argument("--min", type=int, help="smallest batch to write (default: any)")
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
    p.add_argument("--kinds", default="edge,pass1",
                   help="batch kinds helpers are queued for, in order (edge-new: v3's own edge batches only)")
    p.add_argument("--tiers", help="pass-1 batches only from these plan tiers (v3 Stage A: t1,t2)")
    p = sub.add_parser("reserve")
    p.add_argument("batch", nargs="+")
    p.add_argument("--model", help="the reviewer's model (recorded in helpers.jsonl)")
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
    p = sub.add_parser("confirm-batches")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--size", type=int, default=200)
    p.add_argument("--min", type=int, help="smallest batch to write (default --size)")
    p = sub.add_parser("check")
    p.add_argument("--tag", required=True, help="the build to sample (lines_full_<tag>.jsonl)")
    p.add_argument("--v2", default="v2", help="the earlier build's lines manifest tag (new = not in it)")
    p.add_argument("--seed", default="realplus-v3-check-20261013")
    p.add_argument("--size", type=int, default=120, help="entries per check batch")
    p = sub.add_parser("check-score")
    p.add_argument("--tag", required=True)
    p = sub.add_parser("look3")
    p.add_argument("--tag", required=True, help="the build whose check_report_<tag>.json disagreements get a look")
    p.add_argument("--resolve", action="store_true", help="write review/resolutions_v3.jsonl from the decided looks")
    p = sub.add_parser("publish")
    p.add_argument("--dump-date", default="2026-10-12")
    p.add_argument("--tag", help="publish this tagged build")
    p.add_argument("--supersedes", help="the tag of the build this one replaces (its stats stay pinned)")
    p.add_argument("--check", help="the build the precision check ran on (default --tag; v3: v3pre)")
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
     "edge-apply": cmd_edge_apply, "publish": cmd_publish, "blind-score": cmd_blind_score,
     "confirm-batches": cmd_confirm_batches, "check": cmd_check, "check-score": cmd_check_score,
     "look3": cmd_look3}[args.cmd](args)


if __name__ == "__main__":
    main()
