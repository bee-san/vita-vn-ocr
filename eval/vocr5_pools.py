#!/usr/bin/env python3
"""VOCR5's fine-tune pools (eval/vocr5_protocol.json "data.real_pools"): a copy of eval/vocr4_pools.py under the
owner's 2026-10-14 13:25Z policy.

  vocr5_pools.py exclusions --out EXCLUSIONS.json
  vocr5_pools.py build --exclusions EXCLUSIONS.json --out DIR [--private-out PDIR] [--private NAME ...]
                       [--tag TAG] [--dev-npz COHORT.npz ...] NAME=POOL.npz [NAME=POOL.npz ...]
  vocr5_pools.py check --exclusions EXCLUSIONS.json [--dev-npz COHORT.npz]... POOL.npz ...
  vocr5_pools.py sources --pools-record RECORD.json [--refined NAME=REF.npz ...] --out FT_SOURCES.json
  vocr5_pools.py pin --sources FT_SOURCES.json --out POOLS.sha256

exclusions: every VN off limits to VOCR5 training: eval/manifests/reserved.json "test" (the CLANNAD family, vita-test,
  psp-test and their release partners) and "dev_sample" (the fixed development sample). Unlike VOCR4's STRICT rule,
  the old hash splits (is_test_vn, the val bucket), the former development sets and their links are training data.
build: each NAME=POOL.npz (train2's shard format) is copied to DIR/NAME.npz (PDIR for --private names: the owner's
  frames) without the rows of excluded VNs, without rows on a development cohort's screenshot (--dev-npz) and without
  rows on a vita-test / psp-test screenshot or capture session. A pool's group is its name less a crop suffix
  (_ink, _grad, _m4, _m8, _ref) and its crop style is ink (_ink, _m4, _m8, none), grad (_grad) or ref (_ref).
  The groups `tt` (teacher_train + teacher_val's train part) and `rd` (real_train + real_val's train part) are the
  control pools of VOCR3/VOCR4 and are never deduplicated; every other pool is deduplicated by line and crop style:
  a row is dropped when another group (tt, rd, or a pool earlier on the command line) holds a row of the same
  screenshot (screen_id, else the VNDB image id of its line id "<vn>_<image>_<k>") with the same normalised label
  (eval/metrics.norm) and the same crop style. So a GRAD crop stays when only its ink crop is elsewhere (finder-
  tighten: keep the real pools' GRAD crops). Rows are kept in order, byte for byte. DIR/pools_record_TAG.json records
  input and output sha256, rows dropped per reason and per VN, and the duplicates.
check: no row of a pool is on an excluded VN, on a development cohort's screenshot or VN, or on a vita-test / psp-test
  screenshot or session; exit status 1 on any hit. (eval/tools/reserve_check.py --dump --pixels runs on every pool too.)
sources: the fine-tune's real sources (protocol "recipe.fine_tune"): tt+tvt 0.15, rd+rvt 0.075, rp3 0.175, own 0.075,
  dn+nat 0.025; rp3, own and dn+nat split into an ink and a GRAD source by their built row counts; fa=0.3 on the ink
  sources (arm F only); each GRAD source lists its refined copies (arm F only).
pin: sha256 of every pool and refined copy the sources name (absolute paths), for eval/vocr5_run.sh.
"""
import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (HERE, ROOT / "realplus", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from metrics import norm  # noqa: E402
from synth3_filter_pools import keep_rows, save  # noqa: E402

LINE_ID = re.compile(r"^(v\d+)_(\d+)_\d+$")
IMAGE_ID = re.compile(r"/sf/\d\d/(\d+)\.jpg")
SUFFIX = re.compile(r"_(ink|grad|m4|m8|ref)$")
CONTROL = {"tt", "rd"}                                  # never deduplicated (VOCR3/VOCR4's control pools)
REAL_WEIGHTS = {"tt": 0.15, "rd": 0.075, "rp3": 0.175, "own": 0.075, "nt": 0.025}
FA_INK = 0.3


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def group_of(name):
    """tt / tvt -> tt, rd / rvt -> rd; otherwise the name less its crop suffix."""
    base = SUFFIX.sub("", name)
    return {"tvt": "tt", "rvt": "rd"}.get(base, base)


def style_of(name):
    m = SUFFIX.search(name)
    s = m.group(1) if m else "ink"
    return {"m4": "ink", "m8": "ink"}.get(s, s)


def screenshot_of(m):
    """The screenshot a row was cut from: its `screen_id` if it has one, else the VNDB image id of its line id."""
    if m.get("screen_id"):
        return str(m["screen_id"])
    g = LINE_ID.match(str(m.get("id", "")))
    return f"vndb:{g.group(2)}" if g else None


def reserved(path=None):
    p = Path(path) if path else HERE / "manifests" / "reserved.json"
    r = json.loads(p.read_text(encoding="utf-8"))
    ex = {v: f"test:{why}" for v, why in r["test"].items()}
    for v, why in r["dev_sample"].items():
        ex.setdefault(v, f"dev_sample:{why}")
    return ex, {"file": str(p), "sha256": sha256(p)}


def cmd_exclusions(args):
    ex, src = reserved(args.reserved)
    doc = {"kind": "vocr5-exclusions", "schema": 1, "reserved": src,
           "counts": dict(collections.Counter(r.split(":", 1)[0] for r in ex.values())),
           "vns": dict(sorted(ex.items()))}
    Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(ex)} VNs excluded: {doc['counts']}")


def cohort_vns_and_images(paths):
    vns, images = set(), set()
    for p in paths:
        meta = json.loads(str(np.load(p)["meta"]))
        for m in meta:
            vns.add(m["vn"])
            img = screenshot_of(m)
            if img:
                images.add(img)
    return vns, images


def test_set_screens():
    """Screenshot ids and capture sessions of eval's test sets."""
    imgs, sessions = set(), set(json.loads((HERE / "manifests" / "test_games.json").read_text())["sessions"])
    for s in ("vita-test", "psp-test"):
        for line in (HERE / "manifests" / f"{s}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            g = IMAGE_ID.search(r.get("url", ""))
            if g:
                imgs.add(f"vndb:{g.group(1)}")
            sessions.add(str(r.get("session")))
    return imgs, sessions


def sessions_of(m):
    return [s for s in str(m.get("release") or m.get("session") or "").split("/") if s]


def drop_reason(m, ex, dev_imgs, test_imgs, test_sessions):
    img = screenshot_of(m)
    if m["vn"] in ex:
        return ex[m["vn"]]
    if img and img in dev_imgs:
        return "dev-screenshot"
    if img and img in test_imgs:
        return "test-screenshot"
    if any(s in test_sessions for s in sessions_of(m)):
        return "test-session"
    return None


def cmd_build(args):
    ex = json.loads(Path(args.exclusions).read_text(encoding="utf-8"))["vns"]
    _, dev_imgs = cohort_vns_and_images(args.dev_npz)
    test_imgs, test_sessions = test_set_screens()
    out, pout = Path(args.out), Path(args.private_out) if args.private_out else None
    out.mkdir(parents=True, exist_ok=True)
    if pout:
        pout.mkdir(parents=True, exist_ok=True)
    seen = collections.defaultdict(set)              # group -> {(screenshot, normalised label, style)}
    recs = []
    for spec in args.pools:
        name, _, path = spec.partition("=")
        g, st = group_of(name), style_of(name)
        other = set().union(*(v for k, v in seen.items() if k != g)) if g not in CONTROL else set()
        z = np.load(path)
        meta, labels = json.loads(str(z["meta"])), json.loads(str(z["labels"]))
        keep = np.ones(len(meta), bool)
        dropped, dup = collections.Counter(), collections.Counter()
        dropped_vn = collections.Counter()
        for i, m in enumerate(meta):
            img = screenshot_of(m)
            key = (img, norm(labels[i] or ""), st) if img else None
            why = drop_reason(m, ex, dev_imgs, test_imgs, test_sessions)
            if why:
                keep[i] = False
                dropped[why] += 1
                dropped_vn[m["vn"]] += 1
                continue
            if key and key in other:
                keep[i] = False
                dup[m["vn"]] += 1
                continue
            if key:
                seen[g].add(key)
        dst = (pout if (pout and name in args.private) else out) / f"{name}.npz"
        save(dst, keep_rows(z, keep), compressed=True)
        rec = {"name": name, "group": g, "style": st, "pool": str(path), "sha256": sha256(path), "rows": len(meta),
               "out": str(dst), "out_sha256": sha256(dst), "out_rows": int(keep.sum()),
               "out_vns": len({m["vn"] for m, k in zip(meta, keep) if k}),
               "dropped_by_reason": dict(dropped.most_common()), "dropped_by_vn": dict(dropped_vn.most_common()),
               "duplicates_dropped": int(sum(dup.values())), "duplicates_by_vn": dict(dup.most_common()),
               "private": bool(pout and name in args.private)}
        recs.append(rec)
        print(f"{name}: {rec['rows']} -> {rec['out_rows']} rows ({rec['out_vns']} VNs); dropped {dict(dropped)}; "
              f"duplicates {rec['duplicates_dropped']}", flush=True)
    doc = {"kind": "vocr5-pools", "schema": 1,
           "exclusions": {"file": str(args.exclusions), "sha256": sha256(args.exclusions)},
           "dev_npz": [{"file": str(p), "sha256": sha256(p)} for p in args.dev_npz], "pools": recs}
    (out / f"pools_record_{args.tag}.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                                                       encoding="utf-8")


def check_pool(path, ex, dev_vns, dev_imgs, test_imgs, test_sessions):
    meta = json.loads(str(np.load(path)["meta"]))
    hits = collections.Counter()
    for m in meta:
        img = screenshot_of(m)
        if m["vn"] in ex:
            hits["excluded-vn"] += 1
        if m["vn"] in dev_vns:
            hits["dev-vn"] += 1
        if img and img in dev_imgs:
            hits["dev-screenshot"] += 1
        if img and img in test_imgs:
            hits["test-screenshot"] += 1
        if any(s in test_sessions for s in sessions_of(m)):
            hits["test-session"] += 1
    return len(meta), dict(hits)


def cmd_check(args):
    ex = json.loads(Path(args.exclusions).read_text(encoding="utf-8"))["vns"]
    dev_vns, dev_imgs = cohort_vns_and_images(args.dev_npz)
    test_imgs, test_sessions = test_set_screens()
    bad = 0
    for p in args.pools:
        n, hits = check_pool(p, ex, dev_vns, dev_imgs, test_imgs, test_sessions)
        bad += sum(hits.values())
        print(f"{Path(p).name}: {n} rows, hits {hits or 'none'}")
    print("CHECK_OK" if not bad else f"CHECK_FAIL {bad} hits")
    sys.exit(1 if bad else 0)


# the fine-tune's real sources: (source, protocol weight group, pools by built name, fa on arm F)
SOURCES = (("tt", "tt", ("tt", "tvt"), True), ("rd", "rd", ("rd", "rvt"), True),
           ("rpi", "rp3", ("rp3_ink",), True), ("rpg", "rp3", ("rp3_grad",), False),
           ("owi", "own", ("own_ink", "own_m4", "own_m8"), True), ("owg", "own", ("own_grad",), False),
           ("nti", "nt", ("nat_ink", "dntv_ink", "dnst_ink"), True),
           ("ntg", "nt", ("nat_grad", "dntv_grad", "dnst_grad"), False))


def ft_sources(record, refined):
    """The sources JSON: weights split within each weight group by the built row counts."""
    built = {p["name"]: p for p in record["pools"]}
    rows = {s: sum(built[n]["out_rows"] for n in names) for s, _, names, _ in SOURCES}
    group_rows = collections.Counter()
    for s, grp, _, _ in SOURCES:
        group_rows[grp] += rows[s]
    real = []
    for s, grp, names, ink in SOURCES:
        w = REAL_WEIGHTS[grp] * rows[s] / group_rows[grp]
        e = {"name": s, "weight": round(w, 6), "group": grp, "rows": rows[s],
             "files": [built[n]["out"] for n in names], "fa": FA_INK if ink else 0}
        if not ink:
            e["refined"] = [refined[n] for n in names if n in refined]
        real.append(e)
    return real


def cmd_sources(args):
    record = json.loads(Path(args.pools_record).read_text(encoding="utf-8"))
    refined = dict(spec.partition("=")[::2] for spec in args.refined)
    real = ft_sources(record, refined)
    total = sum(e["weight"] for e in real)
    doc = {"kind": "vocr5-ft-sources", "schema": 1, "pools_record": {"file": args.pools_record,
                                                                      "sha256": sha256(args.pools_record)},
           "real_half": round(total, 6), "real": real}
    Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for e in real:
        print(f"{e['name']}: weight {e['weight']} rows {e['rows']} fa {e['fa']} refined {len(e.get('refined', []))}")
    print(f"real half {total:.6f}")


def cmd_pin(args):
    doc = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    files = []
    for e in doc["real"]:
        files += e["files"] + e.get("refined", [])
    lines = [f"{sha256(f)}  {f}" for f in files]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} files pinned")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("exclusions")
    e.add_argument("--reserved", default=None, help="eval/manifests/reserved.json (default)")
    e.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--exclusions", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--private-out", default=None)
    b.add_argument("--private", nargs="*", default=[], help="pool names written to --private-out")
    b.add_argument("--dev-npz", action="append", default=[], help="a development cohort npz (repeat)")
    b.add_argument("--tag", default="build", help="the record is DIR/pools_record_TAG.json")
    b.add_argument("pools", nargs="+", help="NAME=POOL.npz, in precedence order")
    c = sub.add_parser("check")
    c.add_argument("--exclusions", required=True)
    c.add_argument("--dev-npz", action="append", default=[], help="a development cohort npz (repeat)")
    c.add_argument("pools", nargs="+")
    s = sub.add_parser("sources")
    s.add_argument("--pools-record", required=True)
    s.add_argument("--refined", action="append", default=[], help="NAME=REF.npz: a GRAD pool's refined copy")
    s.add_argument("--out", required=True)
    p = sub.add_parser("pin")
    p.add_argument("--sources", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"exclusions": cmd_exclusions, "build": cmd_build, "check": cmd_check, "sources": cmd_sources,
     "pin": cmd_pin}[args.cmd](args)


if __name__ == "__main__":
    main()
