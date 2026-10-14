#!/usr/bin/env python3
"""VOCR4's fine-tune pools (eval/vocr4_protocol.json "data"): the exclusion rule, the cleaned copies, the checks.

  vocr4_pools.py exclusions --dump DB --owner-vns OWNER_GAMES_VNS.json [--allow v34] \
                            [--dev-npz COHORT.npz]... --out EXCLUSIONS.json
  vocr4_pools.py build --exclusions EXCLUSIONS.json --out DIR [--new NAME ...] [--ref NAME=BUILT.npz ...] \
                       [--tag TAG] NAME=POOL.npz [NAME=POOL.npz ...]
  vocr4_pools.py check --exclusions EXCLUSIONS.json [--dev-npz COHORT.npz]... DIR/NAME.npz ...
  vocr4_pools.py vncheck --dump DB --vn v34 [--npz SET.npz]... [--manifest SET.jsonl]... --out CHECK.json

exclusions: every VN off limits to VOCR4 training, with its reason. realplus/shots.reserved_vns() on the given
VNDB dump decides, with these changes:
  - kept although reserved: VNs reserved only as val-bucket (the repo's validation hash split, not a set), and
    VNs reserved only as `family` or `shared-release` whose links (a VNDB release shared with another VN, or a
    FAMILY relation, either way) reach nothing but development or hash-held-out (test-vn) VNs: the pools that
    trained every released model were built with them, and no protected set holds those VNs;
  - excluded although not (or not strictly) reserved: every VN of the owner's test set (OWNER_GAMES_VNS.json
    "vns": Memories Off's series, Aiyoku no Eustia, 428 and their relations and release partners), and every VN
    of a development cohort given with --dev-npz (teacher_val, devnative), so no development VN trains;
  - --allow: VNs the owner released for training although a rule reserves them (planetarian v34, 2026-10-13
    20:47Z); never applied to a VN of the owner's test set or to CLANNAD itself.
  A reason of the form `shared-release:clannad+testset` names the protected classes a link reaches.
build: each NAME=POOL.npz (teacher's npz format) is copied to DIR/NAME.npz without the rows of excluded VNs. A pool
  named with --new is also deduplicated by line (protocol amendment 1): a row is dropped when another group (a --ref
  pool, or a pool earlier on the command line) holds a row of the same screenshot (its own `screen_id`, else the
  VNDB image id in its line id "<vn>_<image>_<k>") with the same normalised label (eval/metrics.norm). Other lines
  of a screenshot stay, and line ids are not compared: the pipelines number a screenshot's lines differently.
  NAME_ink and NAME_grad are one group (the same lines, two crops) and never deduplicate against each other.
  Every other row is kept in order, byte for byte (eval/synth3_filter_pools.keep_rows). DIR/pools_record_TAG.json
  records input and output sha256, rows dropped per reason and per VN, and the duplicates.
check: no row of a pool is on an excluded VN, on a screenshot or VN of a development cohort, or on a screenshot or
  capture session of eval's test sets (vita-test, psp-test: their screenshot ids and sessions, test_games.json
  "sessions"); exit status 1 on any hit.
vncheck: the rows of the given VNs and of their VNDB relations (every kind) in development and test sets, by VN id
  and by capture session (any release of those VNs): the check behind the owner's planetarian rule (2026-10-13
  20:47Z: planetarian lines in teacher_val, devnative, vita-test or psp-test leave those sets).
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

# reserved_vns() reasons that always exclude (the VN itself is protected, a test game, or a development VN)
STRICT = {"clannad", "clannad-family", "clannad-relation", "owner-test-series", "owner-test-relation",
          "owner-test-shared-release", "test-set", "test-vn", "fresh-native", "fresh-native-relation", "development"}
LINKED = {"family", "shared-release"}          # excluded iff a link reaches a protected class
PROTECTED_CLASSES = {"clannad", "owner", "testset", "fresh"}
NEVER_ALLOW = {"v4"}                           # CLANNAD itself
LINE_ID = re.compile(r"^(v\d+)_(\d+)_\d+$")
IMAGE_ID = re.compile(r"/sf/\d\d/(\d+)\.jpg")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def root_class(reason):
    """The protection class of a reserved_vns() reason, or None for the link-only and split reasons."""
    if reason.startswith("clannad"):
        return "clannad"
    if reason.startswith("owner"):
        return "owner"
    if reason == "test-set":
        return "testset"
    if reason == "test-vn":
        return "testvn"
    if reason.startswith("fresh"):
        return "fresh"
    if reason == "development":
        return "dev"
    return None


def link_classes(d, why):
    """{vn: protection classes of the VNs it shares a VNDB release with or is a FAMILY relation of}."""
    cls = {v: c for v, c in ((v, root_class(r)) for v, r in why.items()) if c}
    link = collections.defaultdict(set)
    for rv in d["rel_vns"].values():
        if len(rv) < 2:
            continue
        hit = {cls[w] for w in rv if w in cls}
        if not hit:
            continue
        for u in rv:
            link[u] |= {cls[w] for w in rv if w in cls and w != u}
    for u, rs in d["relations"].items():
        for w in rs:
            if w in cls:
                link[u].add(cls[w])
            if u in cls:
                link[w].add(cls[u])
    return link


def exclusions(why, link, owner_vns, dev_vns=(), allow=()):
    """{vn: reason} of the VNs VOCR4 never trains on (see the module docstring)."""
    out = {}
    for v, r in why.items():
        if r in STRICT:
            out[v] = r
        elif r in LINKED:
            hit = sorted(link.get(v, set()) & PROTECTED_CLASSES)
            if hit:
                out[v] = f"{r}:{'+'.join(hit)}"
    for v in owner_vns:
        out.setdefault(v, "owner-test-set")
    for v in dev_vns:
        out.setdefault(v, "development-cohort")
    for v in allow:
        if v in NEVER_ALLOW or v in set(owner_vns):
            raise SystemExit(f"{v} cannot be allowed")
        out.pop(v, None)
    return out


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


def screenshot_of(m):
    """The screenshot a row was cut from: its `screen_id` if it has one, else the VNDB image id of its line id."""
    if m.get("screen_id"):
        return str(m["screen_id"])
    g = LINE_ID.match(str(m.get("id", "")))
    return f"vndb:{g.group(2)}" if g else None


def cmd_exclusions(args):
    import shots
    d = shots.load_dump(Path(args.dump))
    why = shots.reserved_vns(d)
    link = link_classes(d, why)
    owner = json.loads(Path(args.owner_vns).read_text(encoding="utf-8"))["vns"]
    dev, _ = cohort_vns_and_images(args.dev_npz)
    ex = exclusions(why, link, owner, dev, args.allow)
    tables = {t: sha256(Path(args.dump) / t) for t in ("vn", "vn_relations", "releases_vn")}
    doc = {"kind": "vocr4-exclusions", "schema": 1, "dump": str(args.dump), "dump_tables_sha256": tables,
           "owner_vns": {"file": str(args.owner_vns), "sha256": sha256(args.owner_vns), "vns": len(owner)},
           "dev_npz": [{"file": str(p), "sha256": sha256(p)} for p in args.dev_npz],
           "allowed": sorted(args.allow), "allowed_reasons": {v: why.get(v) for v in sorted(args.allow)},
           "counts": dict(collections.Counter(ex.values()).most_common()), "vns": dict(sorted(ex.items()))}
    Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(ex)} VNs excluded: {doc['counts']}; allowed {doc['allowed_reasons']}")


def group_of(name):
    """A pool's group: its ink and GRAD files (NAME_ink, NAME_grad) hold the same lines and form one group."""
    return re.sub(r"_(ink|grad)$", "", name)


def cmd_build(args):
    ex = json.loads(Path(args.exclusions).read_text(encoding="utf-8"))["vns"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen = collections.defaultdict(set)                         # group -> {(screenshot, normalised label)}
    refs = []
    for spec in args.ref:                                       # already-built pools: deduplication only
        name, _, path = spec.partition("=")
        z = np.load(path)
        for m, lab in zip(json.loads(str(z["meta"])), json.loads(str(z["labels"]))):
            if screenshot_of(m):
                seen[group_of(name)].add((screenshot_of(m), norm(lab or "")))
        refs.append({"name": name, "pool": str(path), "sha256": sha256(path)})
    recs = []
    for spec in args.pools:
        name, _, path = spec.partition("=")
        g = group_of(name)
        other = set().union(*(v for k, v in seen.items() if k != g))
        z = np.load(path)
        meta, labels = json.loads(str(z["meta"])), json.loads(str(z["labels"]))
        keep = np.ones(len(meta), bool)
        dropped, dup = collections.Counter(), collections.Counter()
        for i, m in enumerate(meta):
            img = screenshot_of(m)
            key = (img, norm(labels[i] or "")) if img else None
            if m["vn"] in ex:
                keep[i] = False
                dropped[ex[m["vn"]]] += 1
                continue
            if name in args.new and key in other:
                keep[i] = False
                dup[m["vn"]] += 1
                continue
            if key:
                seen[g].add(key)
        dst = out / f"{name}.npz"
        save(dst, keep_rows(z, keep), compressed=True)
        by_vn = collections.Counter(m["vn"] for m, k in zip(meta, keep) if not k and m["vn"] in ex)
        rec = {"name": name, "pool": str(path), "sha256": sha256(path), "rows": len(meta), "out": str(dst),
               "out_sha256": sha256(dst), "out_rows": int(keep.sum()),
               "out_vns": len({m["vn"] for m, k in zip(meta, keep) if k}),
               "dropped_by_reason": dict(dropped.most_common()), "dropped_by_vn": dict(by_vn.most_common()),
               "duplicates_dropped": int(sum(dup.values())), "duplicates_by_vn": dict(dup.most_common())}
        recs.append(rec)
        print(f"{name}: {rec['rows']} -> {rec['out_rows']} rows ({rec['out_vns']} VNs); excluded {dict(dropped)}; "
              f"duplicates {rec['duplicates_dropped']}", flush=True)
    doc = {"kind": "vocr4-pools", "schema": 1, "exclusions": {"file": str(args.exclusions),
                                                                "sha256": sha256(args.exclusions)},
           "refs": refs, "new": list(args.new), "pools": recs}
    rec_path = out / f"pools_record_{args.tag}.json"
    rec_path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


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


def check_pool(path, ex, dev_vns, dev_imgs, test_imgs, test_sessions):
    meta = json.loads(str(np.load(path)["meta"]))
    hits = collections.Counter()
    for m in meta:
        img = screenshot_of(m)
        sess = str(m.get("release") or m.get("session") or "").split("/")
        if m["vn"] in ex:
            hits["excluded-vn"] += 1
        if m["vn"] in dev_vns:
            hits["dev-vn"] += 1
        if img and img in dev_imgs:
            hits["dev-screenshot"] += 1
        if img and img in test_imgs:
            hits["test-screenshot"] += 1
        if any(s in test_sessions for s in sess if s):
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


def cmd_vncheck(args):
    """Rows of the given VNs (with their VNDB relations of every kind) in development and test sets, by VN id and
    by capture session (any release of those VNs): the planetarian rule's check."""
    import shots
    d = shots.load_dump(Path(args.dump))
    vns = set(args.vn)
    for v in args.vn:
        vns |= set(d.get("any_relations", d["relations"]).get(v, ()))
    rels = {r for r, rv in d["rel_vns"].items() if rv & vns}
    sets = {}
    for p in args.npz:
        meta = json.loads(str(np.load(p)["meta"]))
        sess = [str(m.get("release") or m.get("session") or "").split("/") for m in meta]
        sets[Path(p).name if not args.label_parent else f"{Path(p).parent.name}/{Path(p).name}"] = {
            "file": str(p), "sha256": sha256(p), "rows": len(meta),
            "vn_rows": sum(m["vn"] in vns for m in meta),
            "session_rows": sum(any(s in rels for s in ss) for ss in sess)}
    for p in args.manifest:
        rows = [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
        sets[Path(p).name] = {"file": str(p), "sha256": sha256(p), "rows": len(rows),
                              "vn_rows": sum(r["vn"] in vns for r in rows),
                              "session_rows": sum(str(r.get("session")) in rels for r in rows)}
    doc = {"kind": "vocr4-vn-check", "schema": 1, "vns": sorted(vns), "releases": len(rels),
           "dump": str(args.dump), "sets": sets,
           "total_hits": sum(s["vn_rows"] + s["session_rows"] for s in sets.values())}
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    for k, s in sets.items():
        print(f"{k}: {s['rows']} rows, {s['vn_rows']} by VN, {s['session_rows']} by session")
    print(f"VNs {sorted(vns)}, {len(rels)} releases: {doc['total_hits']} hits")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("exclusions")
    e.add_argument("--dump", required=True)
    e.add_argument("--owner-vns", required=True)
    e.add_argument("--allow", nargs="*", default=[])
    e.add_argument("--dev-npz", action="append", default=[], help="a development cohort npz (repeat)")
    e.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--exclusions", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--new", nargs="*", default=[], help="pools deduplicated by line against the other groups")
    b.add_argument("--ref", action="append", default=[],
                   help="NAME=POOL.npz already built: its screenshots count for deduplication; not written")
    b.add_argument("--tag", default="build", help="the record is DIR/pools_record_TAG.json")
    b.add_argument("pools", nargs="+", help="NAME=POOL.npz, in precedence order")
    c = sub.add_parser("check")
    c.add_argument("--exclusions", required=True)
    c.add_argument("--dev-npz", action="append", default=[], help="a development cohort npz (repeat)")
    c.add_argument("pools", nargs="+")
    k = sub.add_parser("vncheck")
    k.add_argument("--dump", required=True)
    k.add_argument("--vn", action="append", required=True)
    k.add_argument("--npz", action="append", default=[])
    k.add_argument("--manifest", action="append", default=[])
    k.add_argument("--label-parent", action="store_true", help="name npz sets PARENT/FILE")
    k.add_argument("--out", required=True)
    args = ap.parse_args()
    {"exclusions": cmd_exclusions, "build": cmd_build, "check": cmd_check, "vncheck": cmd_vncheck}[args.cmd](args)


if __name__ == "__main__":
    main()
