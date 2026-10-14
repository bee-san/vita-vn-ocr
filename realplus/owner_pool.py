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
    p1, conf = PL.done_batches("pass1"), PL.done_batches("confirm")
    if not p1:
        sys.exit("no decided pass-1 batch yet")
    cmd = [sys.executable, str(ROOT / "devnative" / "decide.py"), "--reads", str(PL.reads_snapshot()), "final",
           "--pass1", *map(str, p1)]
    if conf:
        cmd += ["--confirm", *map(str, conf)]
    res = sorted(REVIEW.glob("resolutions*.jsonl"))
    if res:
        cmd += ["--resolve", *map(str, res)]
    cmd += ["--mode", "train", "--auto-consensus", "--out", str(OUT / "labels.jsonl")]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
    labels = jl(OUT / "labels.jsonl")
    if res:
        n = PL.apply_auto_resolutions(labels, {r["id"]: r for f in res for r in jl(f)})
        print(f"third looks applied to {n} auto lines", flush=True)
    c = confirm_rule(labels)
    wjl(OUT / "labels.jsonl", labels)
    print(json.dumps({"rule": CONFIRM_RULE, "by": c}, indent=1), flush=True)


# ----------------------------------------------------------------------------- confirm looks and edge checks

def confirm_entries(labels, boxes, have, shots):
    """Blind sheet entries (candidates shuffled, unattributed; framed on the final crop) of the pending labels
    that need a confirm look and are in no confirm batch yet, VNs interleaved (pool.vn_rank)."""
    ids = [r["id"] for r in labels if r["status"] == "pending" and r.get("why") == "confirm-missing"
           and r["id"] not in have and r["id"] in boxes]
    ids.sort(key=lambda i: (PL.vn_rank(i.split("_")[0]), i))
    return S.review_entries(shots, ids, frames={i: boxes[i]["crop"] for i in ids})


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
    """pool.py's edge-check batches (final crop framed, the label the one candidate) with the owner note."""
    boxes = jl(PL.EDGE / "boxes.jsonl")
    with PL.review_lock():
        entries = PL.edge_entries(boxes, PL.batched_ids("edge"), PL.load_reads())
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
                if prior and (any(h in busy for h in prior) or not args.retry):
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

def per_game(frames, labels, man, drop_games=()):
    """{vn: {...}} for the table: platform, folders, frames read, candidate lines, kept labels, built lines, dialogue
    lines (built lines that are not speaker names), by crop kind. Counts only."""
    out = collections.defaultdict(lambda: collections.Counter())
    meta = collections.defaultdict(dict)
    for f in frames:
        out[f["vn"]]["frames"] += 1
        meta[f["vn"]].setdefault("platform", f["kind"])
        meta[f["vn"]].setdefault("folders", set()).add(f["folder"])
    for r in labels:
        vn = r["id"].split("_")[0]
        out[vn]["candidates"] += 1
        out[vn]["kept_labels"] += r["status"] == "kept"
    for m in man:
        out[m["vn"]]["lines"] += 1
        out[m["vn"]]["dialogue_lines"] += not m.get("name")
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
    names = {r["id"]: bool(r.get("name")) for r in jl(OUT / f"lines_full_{args.tag}.jsonl")}
    man = [dict(m, name=names.get(m["id"], False)) for m in man]
    drop = json.loads((ROOT / "data" / "manifests" / "owner_drop_20261014.json").read_text(encoding="utf-8"))
    build = json.loads((OUT / "manifests" / f"build_stats_{args.tag}.json").read_text())
    doc = {"kind": "owner-pool-stats", "tag": args.tag, "pool": "owner pool v2",
           "private_dir": f"shared/private/owner-games/_build2/{OUT.name}",
           "codex": "none: private frames; Claude (pass 1 + an independent confirm look) and MeikiOCR / PP-OCRv6 only",
           "rule_sha256": P.file_sha256(OUT / "AGREEMENT.md"), "rule": CONFIRM_RULE,
           "frames_read": len(jl(OUT / "reads" / "reads.jsonl")),
           "frames_by_kind": dict(collections.Counter(f["kind"] for f in frames)),
           "labels": dict(collections.Counter(f"{r['status']}:{r['how'] if r['status'] == 'kept' else r['why']}"
                                              for r in labels)),
           "build": build, "lines": len(man), "dialogue_lines": sum(not m["name"] for m in man),
           "name_lines": sum(m["name"] for m in man),
           "per_game": per_game(frames, labels, man, drop["games"])}
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
    p = sub.add_parser("merge")
    p.add_argument("--tag", required=True)
    p.add_argument("--old", required=True, help="an earlier owner pool's DIR:TAG (its native_train_<crop>_TAG.npz)")
    p = sub.add_parser("publish")
    p.add_argument("--tag", required=True)
    p.add_argument("--check", default="pre", help="the build tag the precision sample was drawn from")
    a = ap.parse_args()
    {"batches": cmd_batches, "final": cmd_final, "leak-drop": cmd_leak_drop, "confirm-batches": cmd_confirm_batches,
     "edge-batches": cmd_edge_batches, "helpers": cmd_helpers, "status": cmd_status, "merge": cmd_merge,
     "publish": cmd_publish}[a.cmd](a)


if __name__ == "__main__":
    main()
