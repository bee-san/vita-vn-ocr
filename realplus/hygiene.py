#!/usr/bin/env python3
"""realplus: hygiene of a pool against every protected set, from the raw inputs (not reserved_vns.json).

  hygiene.py [--labels $REALPLUS/labels.jsonl | --lines $REALPLUS/manifests/lines_v3.jsonl] [--out logs/hygiene.json]
             [--no-pixels]

The pool here is the set of screenshots (and their VNs, releases, file and pixel hashes) that the given lines come
from: before a build, the kept and pending labels minus what the build excludes (pool.build_exclusions; reported under
notes); after it, the lines manifest. Screenshots of v2's selection and of the reclaim tier are both covered.

Every check lists the pool's hits. Under the owner's 2026-10-14 13:25Z policy (docs/reservations.md: the only test
sets are vita-test, psp-test and the CLANNAD family, with their shared releases; a fixed development sample is held
out), these checks block (exit status 1 when any has a hit; FINAL_BLOCKERS):

  the committed eval/manifests/reserved.json: its test VNs and test capture sessions, and its development sample;
  CLANNAD (v4, the family list, every VN or release titled CLANNAD, v4's relations of every kind, the FAMILY
  relations of the family list, release partners of all of those);
  vita-test's and psp-test's games (test_games.json) and eval's two test manifests by VN, image id, file sha256 and
  capture session; decoded-pixel sha256 against eval's vita-test / psp-test screenshots on disk;
  the leak-check drops and dedupe copies.

Information only (reported so the policy change stays visible, not blocking): the old reservations, which are
training data now: test_games.json "all" (the set games' family), synth/common.is_test_vn, the validation bucket,
the fresh native textbox titles and their relations, the owner's former test series, teacher_val and
teacher_distill_val, synth3's exclude_dev_vns.txt, devnative (every version), eval/linefind, the eval candidates and
the other committed eval files; second-ring relations; FAMILY relations to test VNs. The 2026-10-12 audit of PR #57
ran these checks by hand (reports/audit-realplus-evidence/hygiene.py); eval/tools/reserve_check.py (PR #81) is the
canonical gate on the built npz files.
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import shots as SH  # noqa: E402

from common import is_test_vn  # noqa: E402

ART = SH.ART
VN_RE = re.compile(r'"vn":\s*"(v\d+)"')
SHOT_RE = re.compile(r"\b(v\d+)_(\d+)(?:\.(?:jpg|jpeg|png))?\b")
URL_RE = re.compile(r"sf/\d\d/(\d+)\.jpg")
SFID_RE = re.compile(r'"sf(\d+)"')
SHA_RE = re.compile(r'"(?:sha256|pixel_sha256|rgb_sha256)":\s*"([0-9a-f]{64})"')
CLANNAD_TITLE = re.compile(r"clannad|クラナド", re.I)
INFO = ("2nd ring", "family relation to test")


def jl(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def text_of(p):
    return Path(p).read_text(encoding="utf-8", errors="replace")


def ids_in_text(t):
    """(vn ids, image ids, hashes) quoted anywhere in a text."""
    vns = set(VN_RE.findall(t))
    imgs = {int(i) for i in URL_RE.findall(t)} | {int(i) for i in SFID_RE.findall(t)}
    for v, i in SHOT_RE.findall(t):
        vns.add(v)
        imgs.add(int(i))
    return vns, imgs, set(SHA_RE.findall(t))


def npz_meta(p):
    import numpy as np
    z = np.load(p)
    return json.loads(str(z["meta"]))


def tsv(db, name):
    with open(Path(db) / name, encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n").split("\t")


class Dump:
    """The dump tables the checks need: relations both ways with their kind, titles, releases."""

    def __init__(self, db):
        self.rel_any = collections.defaultdict(set)
        self.rel_kind = collections.defaultdict(dict)
        for a, b, kind, *_ in tsv(db, "vn_relations"):
            self.rel_any[a].add(b)
            self.rel_any[b].add(a)
            self.rel_kind[a][b] = kind
        self.titles = collections.defaultdict(list)
        for r in tsv(db, "vn_titles"):
            self.titles[r[0]] += [x for x in r[3:5] if x and x != "\\N"]
        self.rel_vns, self.vn_rels = collections.defaultdict(set), collections.defaultdict(set)
        for rid, vn, *_ in tsv(db, "releases_vn"):
            self.rel_vns[rid].add(vn)
            self.vn_rels[vn].add(rid)
        self.rtitles = collections.defaultdict(list)
        for r in tsv(db, "releases_titles"):
            self.rtitles[r[0]] += [x for x in r[3:5] if x and x != "\\N"]

    def as_shots_dict(self):
        """The tables as shots.load_dump() lays them out (for shots.owner_test_vns)."""
        fam = collections.defaultdict(set)
        for a, kinds in self.rel_kind.items():
            for b, k in kinds.items():
                if k in SH.FAMILY:
                    fam[a].add(b)
        return {"relations": fam, "any_relations": self.rel_any, "title": self.titles, "rel_vns": self.rel_vns}

    def relations(self, vns):
        return set().union(*(self.rel_any[v] for v in vns)) if vns else set()

    def family_relations(self, vns):
        out = set()
        for v in vns:
            out |= {u for u, k in self.rel_kind[v].items() if k in SH.FAMILY}
            out |= {u for u in self.rel_any[v] if self.rel_kind[u].get(v) in SH.FAMILY}
        return out

    def partners(self, vns):
        out = set()
        for v in vns:
            for rid in self.vn_rels.get(v, ()):
                out |= self.rel_vns[rid]
        return out - set(vns)

    def titled(self, rx):
        vns = {v for v, ts in self.titles.items() if any(rx.search(t) for t in ts)}
        vns |= {v for rid, ts in self.rtitles.items() if any(rx.search(t) for t in ts) for v in self.rel_vns[rid]}
        return vns


class Pool:
    """What the checked lines come from: shot keys, VNs, image ids, releases, hashes."""

    def __init__(self, shot_keys, rows):
        self.shots = set(shot_keys)
        self.vns = {s.split("_")[0] for s in self.shots}
        self.images = {int(s.split("_")[1]) for s in self.shots}
        self.rows = [rows[s] for s in self.shots if s in rows]
        self.releases = {r.get("release") for r in self.rows} - {None}
        self.file_sha = {r.get("sha256") for r in self.rows} - {None}
        self.pixel_sha = {r.get("pixel_sha256") for r in self.rows} - {None}
        self.rgb_sha = {r.get("rgb_sha256") for r in self.rows} - {None}

    def hit(self, vns=(), imgs=(), hashes=(), rels=()):
        h = {}
        if vns:
            h["vn"] = sorted(set(vns) & self.vns)
        if imgs:
            h["image"] = sorted(set(imgs) & self.images)
        if hashes:
            h["hash"] = sorted(set(hashes) & (self.file_sha | self.pixel_sha | self.rgb_sha))
        if rels:
            h["release"] = sorted(set(rels) & self.releases)
        h["sizes"] = {"vns": len(set(vns)), "images": len(set(imgs)), "hashes": len(set(hashes)),
                      "releases": len(set(rels))}
        return h


def pool_of(labels=None, lines=None):
    """Shot keys of the kept and pending labels (minus what the build excludes: pool.build_exclusions, the
    reservation recomputed from today's dump and the committed reserved.json, dedupe copies of v2's selection and
    the reclaim tier, leak-check drops), or of a lines manifest. Returns (shot keys, {excluded shot: reason})."""
    if lines:
        return {m["shot"] for m in jl(lines)}, {}
    import pool as PL
    current, vns, shots, _ = PL.build_exclusions()
    keep, out = set(), {}
    for r in jl(labels):
        if r["status"] not in ("kept", "pending"):
            continue
        if r["vn"] in vns:
            out[r["shot"]] = f"vn:{current.get(r['vn']) or 'reserved'}"
        elif r["shot"] in shots:
            out[r["shot"]] = "shot:dedupe-or-leak"
        else:
            keep.add(r["shot"])
    return keep, out


def checks(pool, dump, repo=ROOT, art=ART):
    """{check name: hits} over every protected set. Names in INFO are information only."""
    hits, notes = {}, {}
    tg = SH.test_games()
    hits["test_games.all"] = pool.hit(vns=tg["all"])
    hits["test_games.vita-test"] = pool.hit(vns=set(tg["vita-test"]))
    hits["test_games.psp-test"] = pool.hit(vns=set(tg["psp-test"]))
    hits["test_games.sessions (as releases)"] = pool.hit(rels=tg["sessions"])
    # Canonical reservation (eval/manifests/reserved.json, reserve-policy PR #81): the audited source of
    # truth under the FINAL policy. Blocks on test VNs and the development sample; everything else is info.
    try:
        import common as _C
        res = _C.reservation()
        hits["reserved.json test"] = pool.hit(vns=set(res["test"]), rels=set(res["sessions"]))
        hits["reserved.json dev-sample"] = pool.hit(vns=set(res["dev_sample"]))
    except Exception:  # noqa: BLE001  (reserved.json may be absent before #81)
        pass
    hits["is_test_vn"] = {"vn": sorted(v for v in pool.vns if is_test_vn(v))}
    hits["val-bucket"] = {"vn": sorted(v for v in pool.vns if SH.val_bucket(v))}
    # CLANNAD
    fam = set(SH.CLANNAD_FAMILY)
    titled = dump.titled(CLANNAD_TITLE)
    hits["CLANNAD v4"] = pool.hit(vns=SH.CLANNAD)
    hits["CLANNAD family list"] = pool.hit(vns=fam)
    hits["CLANNAD by VN or release title"] = pool.hit(vns=titled)
    v4rel = dump.relations(SH.CLANNAD)
    hits["CLANNAD relations of every kind"] = pool.hit(vns=v4rel)
    hits["CLANNAD family list: FAMILY relations"] = pool.hit(vns=dump.family_relations(fam) - fam - SH.CLANNAD)
    hits["CLANNAD + family: release partners"] = pool.hit(vns=dump.partners(fam | SH.CLANNAD | titled | v4rel))
    hits["CLANNAD 2nd ring"] = pool.hit(vns=dump.relations(v4rel | fam) - v4rel - fam - SH.CLANNAD)
    # fresh native
    fresh_vn, fresh_hash = set(SH.FRESH_NATIVE) | SH.fresh_native_vns(), SH.fresh_native_hashes()
    frel = dump.relations(fresh_vn)
    hits["fresh native titles"] = pool.hit(vns=fresh_vn, hashes=fresh_hash)
    hits["fresh native relations of every kind"] = pool.hit(vns=frel)
    hits["fresh native: release partners"] = pool.hit(vns=dump.partners(fresh_vn | frel))
    hits["fresh native 2nd ring"] = pool.hit(vns=dump.relations(frel) - frel - fresh_vn)
    # the owner's test series of 2026-10-13 (shots.owner_test_vns, PR #62 / #63: Memories Off, Aiyoku no Eustia,
    # 428: every series VN, their relations of any kind, their release partners)
    owner = SH.owner_test_vns(dump.as_shots_dict())
    hits["owner test series (Memories Off, Eustia, 428): series, relations, release partners"] = \
        pool.hit(vns=set(owner))
    notes["owner_test_set"] = {"vns": len(owner), "reasons": dict(collections.Counter(owner.values()))}
    nf = art / "native-frames" / "owner_games_vns.json"
    if nf.exists():
        theirs = set(json.loads(nf.read_text(encoding="utf-8")).get("vns", []))
        notes["owner_test_set"]["vs_native_frames"] = {"missing_here": sorted(theirs - set(owner)),
                                                       "extra_here": sorted(set(owner) - theirs)}
    # teacher_val, teacher_distill_val, exclude_dev_vns
    for name in ("teacher_val", "teacher_distill_val"):
        p = art / "teacher" / f"{name}.npz"
        if p.exists():
            meta = npz_meta(p)
            hits[name] = pool.hit(vns={m["vn"] for m in meta}, imgs={int(m["id"].split("_")[1]) for m in meta})
    ex = art / "synth3" / "mining" / "exclude_dev_vns.txt"
    if ex.exists():
        hits["synth3 exclude_dev_vns.txt"] = pool.hit(vns=set(ex.read_text().split()))
    hits["development VNs (shots.dev_vns)"] = pool.hit(vns=SH.dev_vns())
    # devnative, every version
    dn = art / "devnative"
    dvn, dimg, dhash = set(), set(), set()
    files = [p for p in dn.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl") and "refs" not in p.parts] \
        if dn.exists() else []
    files += sorted((repo / "devnative" / "manifests").glob("*"))
    for p in files:
        v, i, h = ids_in_text(text_of(p))
        dvn |= v
        dimg |= i
        dhash |= h
    for p in sorted(dn.rglob("*.npz")) if dn.exists() else []:
        if "refs" in p.parts:
            continue
        for m in npz_meta(p):
            dvn.add(m["vn"])
            for key in ("id", "tv_id"):
                if m.get(key):
                    dimg.add(int(m[key].split("_")[1]))
    for p in (list(dn.rglob("*.jpg")) + list(dn.rglob("*.png"))) if dn.exists() else []:
        m = SHOT_RE.match(p.name)
        if m and "review" not in p.parts:
            dvn.add(m.group(1))
            dimg.add(int(m.group(2)))
    hits["devnative, every version"] = pool.hit(vns=dvn, imgs=dimg, hashes=dhash)
    cohort = set()
    for p in [dn / "shots.jsonl", dn / "v1" / "shots.jsonl", *dn.rglob("lines_full.jsonl")] if dn.exists() else []:
        if p.exists():
            cohort |= ids_in_text(text_of(p))[0]
    hits["devnative cohort: release partners"] = pool.hit(vns=dump.partners(cohort))
    hits["devnative cohort: FAMILY relations"] = pool.hit(vns=dump.family_relations(cohort) - cohort)
    hits["devnative cohort: relations of every kind (2nd ring)"] = pool.hit(vns=dump.relations(cohort) - cohort)
    notes["devnative"] = {"files": len(files), "vns": len(dvn), "images": len(dimg), "hashes": len(dhash),
                          "cohort_vns": len(cohort)}
    # eval sets, linefind, candidates, every other eval file
    for s in ("vita-test", "psp-test"):
        rows = jl(repo / "eval" / "manifests" / f"{s}.jsonl")
        hits[f"eval {s}"] = pool.hit(vns={r["vn"] for r in rows}, imgs={int(URL_RE.search(r["url"]).group(1))
                                                                        for r in rows},
                                     hashes={r["sha256"] for r in rows}, rels={r["session"] for r in rows})
    lf_i, lf_h = set(), set()
    for p in sorted((repo / "eval" / "linefind" / "manifests").glob("*")):
        _, i, h = ids_in_text(text_of(p))
        lf_i |= i
        lf_h |= h
    hits["eval/linefind manifests (images, hashes)"] = pool.hit(imgs=lf_i, hashes=lf_h)
    ci, ch = set(), set()
    for k in ("vita", "psp"):
        p = art / "eval" / k / "shots.jsonl"
        if p.exists():
            _, i, h = ids_in_text(text_of(p))
            ci |= i
            ch |= h
    hits["eval candidates (images, hashes)"] = pool.hit(imgs=ci, hashes=ch)
    other = {}
    for p in sorted((repo / "eval").glob("*.json")) + sorted((repo / "eval").glob("*.jsonl")):
        v, i, h = ids_in_text(text_of(p))
        x = pool.hit(vns=v, imgs=i, hashes=h)
        if any(x.get(k) for k in ("vn", "image", "hash")):
            other[p.name] = x
    hits["other committed eval files"] = {"files": other}
    # leak drops and dedupe copies
    rp = SH.realplus_dir()
    leak = rp / "manifests" / "leak_drop.json"
    hits["leak-check drops"] = {"shot": sorted(set(json.loads(leak.read_text())) & pool.shots)} if leak.exists() \
        else {"shot": []}
    dups = rp / "manifests" / "dups.json"
    hits["dedupe copies"] = {"image": sorted(i for i in pool.images if str(i) in json.loads(dups.read_text()))} \
        if dups.exists() else {"image": []}
    # FAMILY relations to test VNs: information
    rel_test = {}
    for v in sorted(pool.vns):
        bad = [(u, k) for u, k in dump.rel_kind[v].items() if k in SH.FAMILY and (u in set(tg["all"]) or is_test_vn(u))]
        if bad:
            rel_test[v] = bad
    hits["family relation to test VN"] = {"vn": sorted(rel_test), "pairs": rel_test}
    return hits, notes


def pixel_hits(pool, dirs):
    """Pool screenshots whose decoded pixels equal a protected screenshot on disk."""
    protected = {}
    for name, d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png") and (p.is_file() or p.is_symlink()):
                try:
                    protected.setdefault(SH.pixel_sha256(p), []).append(f"{name}:{p.name}")
                except ValueError:
                    continue
    hit = {h: protected[h] for h in pool.pixel_sha if h in protected}
    return {"protected_images": sum(len(v) for v in protected.values()), "distinct_hashes": len(protected),
            "hits": hit}


# Checks that still block under the owner's 2026-10-14 FINAL policy: CLANNAD (every variant), the two
# set-game lists and their shared releases, the eval test-set manifests (vita-test, psp-test) and the
# leak / dedupe integrity checks. Everything else (is_test_vn, val-bucket, fresh native, owner series,
# teacher_val, devnative, dev VNs, linefind, eval candidates) becomes information only.
FINAL_BLOCKERS = ("CLANNAD", "test_games.vita-test", "test_games.psp-test", "eval vita-test", "eval psp-test",
                  "reserved.json test", "reserved.json dev-sample", "leak-check drops", "dedupe copies")


def blockers(hits, final=True):
    """Checks with a non-empty hit list, apart from the information-only ones. final (the policy every build
    applies): the owner's 2026-10-14 policy, where only the CLANNAD, vita-test and psp-test checks, the committed
    reservation and the integrity checks block; final=False lists what the old superset rules would have blocked."""
    out = {}
    for name, h in hits.items():
        if any(name.startswith(i) or i in name for i in INFO):
            continue
        if final and not any(name.startswith(b) for b in FINAL_BLOCKERS):
            continue
        bad = {k: v for k, v in h.items() if k not in ("sizes", "pairs") and v}
        if bad:
            out[name] = bad
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=str(SH.realplus_dir() / "labels.jsonl"))
    ap.add_argument("--lines", help="a built lines manifest instead of the labels")
    ap.add_argument("--out", default=str(SH.realplus_dir() / "logs" / "hygiene.json"))
    ap.add_argument("--no-pixels", action="store_true")
    args = ap.parse_args()
    import pool as PL
    # v2's selection and the reclaim tier (pool.reclaim_shot_rows): every screenshot's release, file and pixel
    # hashes, so the release, hash and pixel checks cover the reclaim screenshots too
    rows = {f"{r['vn']}_{r['image']}": r for r in SH.load_shots() + PL.reclaim_shot_rows()}
    keys, excluded = pool_of(args.labels, args.lines)
    pool = Pool(keys, rows)
    dump = Dump(SH.dump_dir())
    hits, notes = checks(pool, dump)
    notes["policy"] = PL.POLICY
    notes["rows"] = {"shots": len(pool.shots), "with_selection_row": len(pool.rows),
                     "with_pixel_hash": sum(1 for r in pool.rows if r.get("pixel_sha256"))}
    notes["build_time_exclusions"] = {"shots": len(excluded), "by": dict(collections.Counter(excluded.values())),
                                      "vns": sorted({k.split("_")[0] for k, v in excluded.items()
                                                     if v.startswith("vn:")})}
    res = {"source": args.lines or args.labels, "pool": {"shots": len(pool.shots), "vns": len(pool.vns),
                                                           "releases": len(pool.releases)},
           "hits": hits, "notes": notes}
    if not args.no_pixels:
        # Under the FINAL policy the only test screenshots on disk are eval's vita-test / psp-test sets (CLANNAD
        # is caught by VN id / title / relation above; it has no realplus screenshot). reserve_check.py --pixels
        # adds the owner's CLANNAD frames.
        dirs = [("eval-test", ART / "eval" / "vita" / "shots"), ("eval-test", ART / "eval" / "psp" / "shots")]
        res["pixels"] = pixel_hits(pool, dirs)
    bad = blockers(hits)
    if not args.no_pixels and res["pixels"]["hits"]:
        bad["pixel sha256"] = res["pixels"]["hits"]
    res["blockers"] = bad
    res["verdict"] = "HYGIENE_OK" if not bad else "HYGIENE_FAIL"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"pool": res["pool"], "blockers": bad, "pixels": {k: v for k, v in res.get("pixels", {}).items()
                                                                       if k != "hits"},
                      "sizes": {k: v.get("sizes") for k, v in hits.items() if v.get("sizes")},
                      "info": {k: {kk: len(vv) for kk, vv in v.items() if kk != "sizes"} for k, v in hits.items()
                               if any(i in k for i in INFO)}}, ensure_ascii=False, indent=1), flush=True)
    print(res["verdict"], flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
