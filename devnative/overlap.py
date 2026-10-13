#!/usr/bin/env python3
"""Independent check of a cohort's screenshots against every eval set's screenshots.

  overlap.py [--cohort NAME SHOTS LINES ...] [--out devnative/manifests/overlap.json]
             [--require-clean NAME ...]

The eval sets are eval/manifests/vita-test.jsonl and psp-test.jsonl, and every eval/linefind/
manifests/*.jsonl file (the research, dev, holdout and candidate sets and the PSP pool). Each
screenshot of a cohort's SHOTS manifest (select.py's shots.jsonl; its local file under
$DEVNATIVE_OUT/shots) is compared with every eval screenshot (its local copy) by
  id     VNDB screenshot number,
  file   file sha256 (the eval manifest's, and its local copy's),
  pixel  decoded-pixel sha256 (paths.pixel_sha256) of the local copy,
  same   the same frame re-encoded (select.py's rule: thumbnail correlation >= 0.95 and a
         full-size grey p99 difference < 48),
  near   thumbnail correlation >= 0.95 without being the same frame.
This is a second implementation of select.py's duplicate checks (it does not reuse select.py's
reference list or role logic), for audits. The report holds ids, hashes, set names and counts
(no text, no pixels): every matching screenshot, its subset and the cohort lines on it (LINES,
build.py's lines.jsonl). --require-clean exits 1 if that cohort has any match.
Default: --cohort cohort devnative/manifests/shots.jsonl devnative/manifests/lines.jsonl.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from devnative import paths as P  # noqa: E402

NEAR = 0.95
SAME_P99 = 48
STRONG = ("id", "file", "pixel", "same")


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def eval_sets(root=ROOT):
    """{set name: [manifest rows with a url]}: eval's test sets and every linefind manifest."""
    out = {}
    for name in ("vita-test", "psp-test"):
        p = root / "eval" / "manifests" / f"{name}.jsonl"
        if p.exists():
            out[name] = [r for r in jsonl(p) if r.get("url")]
    for p in sorted((root / "eval" / "linefind" / "manifests").glob("*.jsonl")):
        out["linefind/" + p.name[:-len(".jsonl")]] = [r for r in jsonl(p) if r.get("url")]
    return out


def local_copy(set_name, row):
    """The local file of an eval screenshot, or None."""
    name = row.get("shot") or f"{row['vn']}_{row['url'].rsplit('/', 1)[-1]}"
    if set_name in ("vita-test", "psp-test"):
        cands = [P.ART / "eval" / set_name.split("-")[0] / "shots" / name]
    else:
        cands = [P.SHARED_DATA / "vndb" / "shots" / name, P.ART / "linefind" / "psp_dev" / "shots" / name,
                 P.ART / "synth2" / "vndb_psp" / "shots" / name]
    return next((c for c in cands if c.exists()), None)


def same_frame(a, b):
    """select.py's rule: grey p99 |a - b| at a's size below SAME_P99 (b resized with INTER_AREA)."""
    import cv2
    ga = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2GRAY).astype(np.int16)
    gb = cv2.cvtColor(np.ascontiguousarray(b), cv2.COLOR_RGB2GRAY)
    if gb.shape != ga.shape:
        gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.percentile(np.abs(ga - gb.astype(np.int16)), 99)) < SAME_P99


class EvalIndex:
    """Every eval screenshot: by number, file sha256, pixel sha256, and thumbnails."""

    def __init__(self, sets):
        self.sets = sets
        self.members = defaultdict(set)          # number -> {set names}
        self.by_sha, self.by_pix = defaultdict(set), defaultdict(set)
        self.rgb, nums, thumbs = {}, [], []
        self.missing = Counter()
        for name, rows in sets.items():
            for r in rows:
                n = P.image_num(r["url"])
                self.members[n].add(name)
                if r.get("sha256"):
                    self.by_sha[r["sha256"]].add(n)
                if n in self.rgb:
                    continue
                path = local_copy(name, r)
                rgb = P.read_rgb(path) if path else None
                if rgb is None:
                    self.missing[name] += 1
                    continue
                self.rgb[n] = rgb
                self.by_sha[P.file_sha256(path)].add(n)
                self.by_pix[P.pixel_sha256(rgb)].add(n)
                nums.append(n)
                thumbs.append(P.thumb(rgb))
        self.nums = nums
        self.thumbs = np.stack(thumbs) if thumbs else np.zeros((0, 576), np.float32)

    def match(self, num, sha, pix, rgb):
        """{eval number: sorted match kinds} for one screenshot."""
        how = defaultdict(set)
        if num in self.members:
            how[num].add("id")
        for n in self.by_sha.get(sha, ()):
            how[n].add("file")
        for n in self.by_pix.get(pix, ()):
            how[n].add("pixel")
        corr = self.thumbs @ P.thumb(rgb) if len(self.nums) else np.zeros(0)
        for j in np.flatnonzero(corr >= NEAR):
            n = self.nums[j]
            how[n].add("same" if same_frame(rgb, self.rgb[n]) else "near")
        return {n: sorted(h) for n, h in how.items()}


def check(name, shots, lines, index):
    """The report of one cohort: every screenshot that matches an eval screenshot."""
    by_shot = defaultdict(list)
    for ln in lines:
        by_shot[ln["shot"]].append(ln)
    hits = []
    for s in shots:
        path = P.OUT / "shots" / s["kind"] / f"{s['vn']}_{s['url'].rsplit('/', 1)[-1]}"
        rgb = P.read_rgb(path)
        if rgb is None:
            raise SystemExit(f"{name}: no local file for {s['shot']} ({path})")
        sha, pix = P.file_sha256(path), P.pixel_sha256(rgb)
        if sha != s["sha256"] or pix != s["pixel_sha256"]:
            raise SystemExit(f"{name}: {s['shot']} differs from its manifest hashes")
        m = index.match(P.image_num(s["url"]), sha, pix, rgb)
        if not m:
            continue
        sets = sorted({x for n in m for x in index.members[n]})
        lns = by_shot.get(s["shot"], [])
        hits.append({"shot": s["shot"], "id": s["id"], "vn": s["vn"], "kind": s["kind"], "subset": s["subset"],
                     "url": s["url"], "sha256": s["sha256"], "pixel_sha256": s["pixel_sha256"],
                     "flags": s.get("flags", []), "sets": sets,
                     "match": {f"sf{n}": kinds for n, kinds in sorted(m.items())},
                     "strong": any(set(k) & set(STRONG) for k in m.values()),
                     "lines": sorted(ln["id"] for ln in lns),
                     "lines_grad": sum(1 for ln in lns if (ln.get("grad") or {}).get("ok"))})
    summary = Counter()
    for h in hits:
        kind = "strong" if h["strong"] else "near-only"
        for st in h["sets"]:
            summary[f"{st}|{h['subset']}|{h['kind']}|{kind}|shots"] += 1
            summary[f"{st}|{h['subset']}|{h['kind']}|{kind}|lines"] += len(h["lines"])
    return {"screenshots": len(shots), "lines": len(lines), "matching_screenshots": len(hits),
            "matching_lines": sum(len(h["lines"]) for h in hits),
            "matching_lines_grad": sum(h["lines_grad"] for h in hits),
            "by_kind": dict(Counter(f"{h['subset']}/{h['kind']}" for h in hits)),
            "lines_by_kind": dict(Counter(f"{h['subset']}/{h['kind']}" for h in hits for _ in h["lines"])),
            "match_kinds": dict(Counter(k for h in hits for ks in h["match"].values() for k in ks)),
            "by_set": dict(sorted(summary.items())), "hits": hits}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", nargs=3, action="append", metavar=("NAME", "SHOTS", "LINES"))
    ap.add_argument("--out", default=str(P.MANIFESTS / "overlap.json"))
    ap.add_argument("--require-clean", nargs="*", default=[])
    args = ap.parse_args()
    cohorts = args.cohort or [["cohort", str(P.MANIFESTS / "shots.jsonl"), str(P.MANIFESTS / "lines.jsonl")]]
    sets = eval_sets()
    index = EvalIndex(sets)
    rep = {"kind": "devnative-eval-overlap", "rule": {"near": NEAR, "same_p99": SAME_P99},
           "eval_sets": {k: {"screenshots": len({P.image_num(r["url"]) for r in v})} for k, v in sets.items()},
           "eval_screenshots_local": len(index.nums), "eval_missing_local": dict(index.missing), "cohorts": {}}
    for name, shots, lines in cohorts:
        rep["cohorts"][name] = check(name, jsonl(shots), jsonl(lines), index)
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=True, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: {x: v[x] for x in ("screenshots", "lines", "matching_screenshots", "matching_lines",
                                            "by_kind", "match_kinds")} for k, v in rep["cohorts"].items()}, indent=1),
          flush=True)
    dirty = [n for n in args.require_clean if rep["cohorts"][n]["matching_screenshots"]]
    if args.require_clean and (dirty or index.missing):
        print(f"not clean: {dirty}; eval screenshots without a local copy: {dict(index.missing)}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
