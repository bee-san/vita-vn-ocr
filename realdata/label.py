#!/usr/bin/env python3
"""Find the text lines of every selected screenshot and read each with the OCR voters.

  label.py [--shard K --nshards N] [--procs P] [--voters v6s,meiki,manga,v5s] [--limit-shots M]
           [--sample-lines P --seed S] [--out FILE]

Reads realdata/manifests/shots.jsonl and the images that vndb.py fetched. For each
screenshot: PP-OCRv5 mobile det (PSP frames are upscaled 2x for detection only), rows
merged, ink boxes refined (lines.py), lines too small for every target dropped. Each
line's voter view is read by v6s and meiki, then by manga-ocr and v5s (the slow ones)
unless v6s reads nothing Japanese (English UI, copyright strings: never labelled). v5s
is also skipped where vote.v5s_can_matter() shows it cannot change the default rule's
decision; such rows list it under "skip".

Writes one JSON row per line to $REALDATA/labels/lines.<K>of<N>.jsonl:
  id, url, vn, kind, dims, ink [x0, y0, x1, y1] (source px), pol, r {voter: [text, score]}
and the screenshot URL to lines.<K>of<N>.done once all its lines are written. Every
lines.*.done counts, so a killed run resumes where it stopped, even with a different
sharding. build.py turns these rows into labelled crops.

--procs P splits this shard over P worker processes with OMP_NUM_THREADS / P threads each
(the readers scale poorly with threads on short lines: manga-ocr reads a line in 0.72 s on
one thread and 0.47 s on five), so one heavy.sh slot runs P single-thread workers.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lines as L  # noqa: E402
import voters as V  # noqa: E402
from vndb import load_shots, realdata_dir, shot_key, shot_path  # noqa: E402
from vote import japanese, v5s_can_matter  # noqa: E402

SLOW = ("manga", "v5s")


def shard_of(url, n):
    return int(hashlib.md5(url.encode()).hexdigest(), 16) % n


def line_candidates(det, bgr, kind):
    """[(ink box, polarity)] for one screenshot, top to bottom."""
    h, w = bgr.shape[:2]
    if kind == "psp":                     # 12-16 px glyphs: detect on a 2x upscale
        big = cv2.resize(bgr, (2 * w, 2 * h), interpolation=cv2.INTER_CUBIC)
        boxes = [[v / 2 for v in b] for b in L.detect(det, big)]
    else:
        boxes = L.detect(det, bgr)
    rgb = bgr[:, :, ::-1]
    targets = ["psp"] if kind == "psp" else ["vita"] if kind == "vita" else ["vita", "psp"]
    out = []
    for b in L.merge_rows(boxes):
        ink = L.ink_box(rgb, b)
        if ink is None:
            continue
        ih = ink[3] - ink[1]
        if not any(L.MIN_INK[t] <= ih * L.target_scale(w, h, t) <= L.MAX_INK for t in targets):
            continue
        x0, y0, x1, y1 = L.voter_box(ink, w, h, rgb)
        out.append((ink, L.polarity(rgb[ink[1]:ink[3], ink[0]:ink[2]]), (x0, y0, x1, y1)))
    out.sort(key=lambda t: (t[0][1], t[0][0]))
    return out


def spawn(args):
    """Run this shard as `procs` worker processes (sub-shards), each with an equal share of
    the slot's threads, and wait for all of them."""
    n = args.procs
    if args.out:
        sys.exit("--out cannot be combined with --procs (each worker writes its own shard file)")
    env = dict(os.environ)
    each = str(max(1, V.threads() // n))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCH_NUM_THREADS"):
        env[var] = each
    skip = {"--procs", "--shard", "--nshards"}
    rest, it = [], iter(sys.argv[1:])
    for a in it:
        if a.split("=")[0] in skip:
            if "=" not in a:
                next(it, None)
            continue
        rest.append(a)
    procs = []
    for i in range(n):
        cmd = [sys.executable, str(Path(__file__).resolve()), "--shard", str(args.shard * n + i),
               "--nshards", str(args.nshards * n), *rest]
        log = open(realdata_dir() / "labels" / f"worker.{args.shard * n + i}of{args.nshards * n}.log", "a")
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL))
    print(f"{n} workers with {each} thread(s) each: pids {[p.pid for p in procs]}", flush=True)
    rcs = [p.wait() for p in procs]
    print(f"workers done, exit codes {rcs}", flush=True)
    if any(rcs):
        sys.exit(1)
    print("LABEL_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--voters", default="v6s,meiki,manga,v5s")
    ap.add_argument("--kinds", default="vita,psp,hires")
    ap.add_argument("--limit-shots", type=int, default=0)
    ap.add_argument("--shot-frac", default="",
                    help="dev sample: kind=fraction,... keeps each screenshot with that probability")
    ap.add_argument("--sample-lines", type=float, default=1.0,
                    help="read only this fraction of lines (random, for a dev sample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-cascade", action="store_true", help="always run every voter")
    ap.add_argument("--procs", type=int, default=1)
    args = ap.parse_args()
    if args.procs > 1:
        return spawn(args)
    out = args.out or realdata_dir() / "labels" / f"lines.{args.shard}of{args.nshards}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done_path = out.with_suffix(".done")
    done = set()
    for path in [done_path] + ([] if args.out else sorted(out.parent.glob("lines.*.done"))):
        if path.exists():
            done |= set(path.read_text().split())

    kinds = set(args.kinds.split(","))
    rows = [r for r in load_shots() if r["kind"] in kinds and shard_of(r["url"], args.nshards) == args.shard
            and r["url"] not in done and r["sha256"] and shot_path(r).exists()]
    if args.shot_frac:
        frac = {k: float(v) for k, v in (x.split("=") for x in args.shot_frac.split(","))}
        rows = [r for r in rows if int(hashlib.sha1(f"{args.seed}:{r['url']}".encode()).hexdigest(), 16)
                % 10_000 < 10_000 * frac.get(r["kind"], 0.0)]
    if args.limit_shots:
        rows = rows[:args.limit_shots]
    import random
    rng = random.Random(args.seed)
    det = L.make_detector(V.threads())
    names = args.voters.split(",")
    voters = V.load(names)
    print(f"shard {args.shard}/{args.nshards}: {len(rows)} screenshots to do, voters {names}, "
          f"threads {V.threads()}", flush=True)
    t0, n_lines, n_slow = time.time(), 0, 0
    with open(out, "a", encoding="utf-8") as fo, open(done_path, "a") as fd:
        for i, r in enumerate(rows):
            bgr = cv2.imread(str(shot_path(r)), cv2.IMREAD_COLOR)
            if bgr is None or [bgr.shape[1], bgr.shape[0]] != r["dims"]:
                print(f"unreadable or wrong size: {shot_path(r)}", flush=True)
                fd.write(r["url"] + "\n")
                continue
            for k, (ink, pol, vb) in enumerate(line_candidates(det, bgr, r["kind"])):
                if rng.random() >= args.sample_lines:
                    continue
                view = bgr[vb[1]:vb[3], vb[0]:vb[2]]
                reads, skip = {}, []
                for name in names:
                    if name in SLOW and "v6s" in reads and not japanese(reads["v6s"][0]):
                        continue
                    if name == "v5s" and not args.no_cascade and \
                            not v5s_can_matter({k: v[0] for k, v in reads.items()}):
                        skip.append(name)
                        continue
                    text, score, _ = voters[name](view)
                    reads[name] = [text, round(score, 4)]
                    n_slow += name in SLOW
                rec = {"id": f"{shot_key(r)}_{k}", "url": r["url"], "vn": r["vn"], "kind": r["kind"],
                       "dims": r["dims"], "ink": [int(v) for v in ink], "view": [int(v) for v in vb],
                       "pol": pol, "r": reads}
                if skip:
                    rec["skip"] = skip
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_lines += 1
            fo.flush()
            fd.write(r["url"] + "\n")
            fd.flush()
            if i % 20 == 0:
                dt = time.time() - t0
                print(f"{i + 1}/{len(rows)} shots, {n_lines} lines ({n_slow} slow reads), {dt:.0f}s, "
                      f"{dt / max(i + 1, 1):.1f}s/shot", flush=True)
    print(f"done: {len(rows)} shots, {n_lines} lines in {time.time() - t0:.0f}s", flush=True)
    print("LABEL_DONE", flush=True)


if __name__ == "__main__":
    main()
