#!/usr/bin/env python3
"""realplus/reclaim.py: what the 2026-10-14 reservation (docs/reservations.md) frees for training.

  reclaim.py [--dump DB] [--out DIR] [--manifest realplus/manifests/reclaim_20261014.json]
  reclaim.py --split NAME=NPZ [NAME=NPZ ...] [--out DIR]
                     a labelled pool (teacher_val, real_val, devnative, teacher_train, ...) cut by VN into
                     DIR/pools/NAME_train.npz (training data now) and DIR/pools/NAME_dev.npz (the development
                     sample's rows), same format (data, offsets, shapes, labels, meta); rows of a test VN
                     (reserved.json "test") go to neither and are counted -> DIR/pools/split.json

Compares the new rules (shots.reserved_vns: the test reservation and the development sample) with what the old
rules actually held back, from the files those rules wrote, and lists what becomes usable:

  realplus   VNDB screenshots shots.select now takes that the old selection ($ART/realplus/manifests/shots.jsonl,
             the 2026-10-12 dump) did not, each with the reason the old rules dropped it (old_why: the VN's reason in
             the old reserved_vns.json, or the old screenshot set) -> DIR/realplus_shots.jsonl, in shots.jsonl's
             format, so shots.py fetch / pool.py read it as a new batch tier (tier "reclaim-20261014")
  native     PSP / PS Vita frames native.classify now passes that the native pool's enumeration (2026-10-13 dump,
             $ART/native-frames/pool/manifests/enumeration_rows.jsonl) dropped for a VN or set reason
             -> DIR/native_rows.jsonl
  synth      VNDB character names (synth3_text names' VN gate) and on-disk scene candidates (synth3.scene_candidates)
             the old gates dropped and the new ones keep
  labelled   lines that already have labels and become training data: teacher_val, real_val, devnative outside the
             development sample, and the owner's frames by folder
  reserved   VNs reserved before and after, by reason

The lists stay under DIR (default $HERD/shared/artifacts/reserve-policy/reclaim); --manifest writes the counts
and the lists' sha256 (committed). Nothing is downloaded: shots.py fetch does that.
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import native as NA  # noqa: E402
import shots as SH  # noqa: E402

ART = SH.ART
TIER = "reclaim-20261014"
OLD_REALPLUS = ART / "realplus" / "manifests"
OLD_NATIVE = ART / "native-frames" / "pool" / "manifests" / "enumeration_rows.jsonl"
OLD_SETS = SH.UNRESERVED_SETS          # screenshot sets the old rules protected and the new ones do not


def jl(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def wjl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def old_image_sets():
    """{image id: set names} of the old screenshot sets that no longer protect (eval candidates, linefind), from
    their files and local copies, as the old shots.reserved_screenshots() read them."""
    out = collections.defaultdict(set)
    for name, path in SH.screenshot_sources():
        if name in OLD_SETS and Path(path).exists():
            for i in SH.image_ids_in(path):
                out[i].add(name)
    for name, d in SH.screenshot_dirs():
        if name in OLD_SETS and d.exists():
            for p in d.rglob("*"):
                m = SH.SHOT_FILE.match(p.name)
                if m:
                    out[int(m.group(1))].add(name)
    return out


def realplus_reclaim(d, reserved, rimg, sessions, old_reserved, old_rows, old_sets):
    """Rows shots.select keeps under the new rules that the old selection lacks, with old_why."""
    keep, why = SH.select(d, reserved, rimg, sessions)
    old_ids = {r["image"] for r in old_rows}
    out = []
    for r in keep:
        if r["image"] in old_ids:
            continue
        if r["vn"] in old_reserved:
            ow = "vn:" + old_reserved[r["vn"]]
        elif r["image"] in old_sets:
            ow = "in-set:" + "+".join(sorted(old_sets[r["image"]]))
        else:
            ow = "not-in-old-selection"          # e.g. new in the later dump
        out.append({**r, "old_why": ow, "tier": TIER})
    for i, r in enumerate(out):
        r["order"] = i
    return out, why


def native_reclaim(d, reserved, rimg, sessions, old_rows):
    """Native frames the old enumeration dropped for a VN or screenshot-set reason that the new rules make usable:
    'new' (on no disk: fetch, read and label), 'pooled' (on disk in realdata's / teacher's pools, read, lines left
    unlabelled) or 'pooled-other' (on disk in another pool). Frames the rating, language or original-language
    filters drop stay out."""
    old = {r["image"]: r["why"] for r in old_rows}
    new = NA.classify(d, reserved, rimg, sessions)
    out = []
    for r in new:
        ow = old.get(r["image"])
        if ow and ow.startswith(("vn:", "set:")) and (r["why"] in ("new", "pooled") or
                                                       r["why"].startswith("pooled-other")):
            out.append({**r, "old_why": ow, "tier": TIER})
    return out, collections.Counter(r["why"] for r in new)


def names_reclaim(chars_tsv, old_forbidden, new_forbidden):
    """Japanese VNDB characters synth3_text names' VN gate drops under the old and the new reservation."""
    p = Path(chars_tsv)
    if not p.exists():
        return None
    c = collections.Counter()
    with open(p, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        col = {k: i for i, k in enumerate(header)}
        for line in f:
            x = line.rstrip("\n").split("\t")
            if x[col["lang"]] != "ja":
                continue
            vns = x[col["vn_ids"]].split(",")
            c["ja_characters"] += 1
            o = any(old_forbidden(v) for v in vns)
            n = any(new_forbidden(v) for v in vns)
            c["dropped_old"] += o
            c["dropped_new"] += n
            c["reclaimed"] += o and not n
    return dict(c)


def scene_reclaim(old_forbidden, new_forbidden):
    """On-disk synth3 scene candidates (synth3.scene_candidates) the old and the new VN gates drop."""
    try:
        import synth3 as S3
        cands = S3.scene_candidates()
    except Exception as e:  # noqa: BLE001 - the shared scene lists are not on every machine
        return {"error": f"{type(e).__name__}: {e}"}
    out = {}
    for kind, paths in zip(("vita", "psp", "hires"), cands):
        vns = [p.name.split("_")[0] for p in paths]
        out[kind] = {"candidates": len(paths), "dropped_old": sum(map(old_forbidden, vns)),
                     "dropped_new": sum(map(new_forbidden, vns)),
                     "reclaimed": sum(1 for v in vns if old_forbidden(v) and not new_forbidden(v))}
    return out


def labelled_reclaim(dev, counts=None):
    """Labelled development lines by source and kind: in the development sample, or training data now."""
    counts = SH.dev_line_counts() if counts is None else counts
    out = {"dev_sample": collections.Counter(), "training": collections.Counter()}
    vns = {"dev_sample": set(), "training": set()}
    for v, c in counts.items():
        k = "dev_sample" if v in dev else "training"
        out[k].update(c)
        vns[k].add(v)
    return {k: {"lines": dict(sorted(out[k].items())), "vns": len(vns[k])} for k in out}


def owner_reclaim(reserved):
    """The owner's folders (shared/private/owner-games) with their frame counts and role under the new rules."""
    out = {}
    if not SH.OWNER_GAMES.exists():
        return out
    import re
    for p in sorted(SH.OWNER_GAMES.iterdir()):
        m = re.search(r"-(v\d+)(?:-TEST)?$", p.name)
        if not p.is_dir() or p.name.startswith("_") or not m:
            continue
        n = sum(1 for f in p.rglob("*") if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
        vn = m.group(1)
        out[p.name] = {"vn": vn, "frames": n, "role": "test" if vn in reserved and
                       not reserved[vn].startswith("dev-sample") else "training",
                       "reason": reserved.get(vn), "was_test_only": p.name.endswith("-TEST")}
    return out


def split_npz(src, out_train, out_dev, held, test=frozenset()):
    """Cut one npz pool by VN: rows of a held VN (the development sample) to out_dev, rows of a test VN nowhere
    (counted by VN), the rest to out_train. Returns {"train": {...}, "dev": {...}, "dropped_test": {vn: rows}}."""
    import numpy as np

    from devnative import build as B
    with np.load(src) as z:
        data, off, shapes = z["data"], z["offsets"], z["shapes"]
        labels, meta = json.loads(str(z["labels"])), json.loads(str(z["meta"]))
    parts = {"train": ([], [], []), "dev": ([], [], [])}
    dropped = collections.Counter()
    for i, m in enumerate(meta):
        if m.get("vn") in test:
            dropped[m["vn"]] += 1
            continue
        h, w = (int(x) for x in shapes[i])
        crop = data[off[i]:off[i] + h * w].reshape(h, w)
        k = "dev" if m.get("vn") in held else "train"
        parts[k][0].append(crop)
        parts[k][1].append(labels[i])
        parts[k][2].append(m)
    stats = {}
    for k, path in (("train", out_train), ("dev", out_dev)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        B._save(Path(path), *parts[k])
        stats[k] = {"rows": len(parts[k][2]), "vns": len({m.get("vn") for m in parts[k][2]}),
                    "file": Path(path).name, "sha256": sha256(path)}
    stats["dropped_test"] = dict(sorted(dropped.items()))
    return stats


def cmd_split(args):
    res = SH.committed_reservation()
    held, test = set(res["dev_sample"]), set(res["test"])
    out = Path(args.out) / "pools"
    sp = out / "split.json"
    doc = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {
        "kind": "reserve-policy-splits", "dev_sample": "eval/manifests/reserved.json dev_sample (with its release "
        "partners)", "test": "eval/manifests/reserved.json test: rows dropped", "sets": {}}
    for spec in args.split:
        name, _, path = spec.partition("=")
        doc["sets"][name] = {"source": path, "source_sha256": sha256(path),
                             **split_npz(path, out / f"{name}_train.npz", out / f"{name}_dev.npz", held, test)}
        print(name, json.dumps(doc["sets"][name]), flush=True)
    sp.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default=str(ART / "native-frames" / "vndb" / "dump" / "db"))
    ap.add_argument("--out", default=str(ART / "reserve-policy" / "reclaim"))
    ap.add_argument("--manifest", help="write the summary here too (committed: realplus/manifests/reclaim_*.json)")
    ap.add_argument("--chars", default=str(ART / "synth3" / "text" / "vndb_chars.tsv"))
    ap.add_argument("--split", nargs="+", metavar="NAME=NPZ", help="cut labelled development sets by VN instead")
    args = ap.parse_args()
    if args.split:
        return cmd_split(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = SH.load_dump(Path(args.dump))
    stamp_f = Path(args.dump).parent / "TIMESTAMP"
    stamp = stamp_f.read_text().strip() if stamp_f.exists() else None
    tg = SH.test_games()
    sessions = set(tg["sessions"])
    reserved = SH.reserved_vns(d)
    rimg = SH.reserved_screenshots()
    old_reserved = json.loads((OLD_REALPLUS / "reserved_vns.json").read_text())
    old_sets = old_image_sets()
    rp_rows, rp_why = realplus_reclaim(d, reserved, rimg, sessions, old_reserved, jl(OLD_REALPLUS / "shots.jsonl"),
                                       old_sets)
    wjl(out / "realplus_shots.jsonl", rp_rows)
    na_rows, na_why = native_reclaim(d, reserved, rimg, sessions, jl(OLD_NATIVE))
    wjl(out / "native_rows.jsonl", na_rows)
    old_all = set(tg.get("all_20261012", tg["all"]))
    dev_old = SH.dev_vns()

    def old_forbidden(v):            # synth3's old gate: test_games "all", the dev list, is_test_vn, CLANNAD
        from common import is_test_vn
        return v in old_all or v in dev_old or is_test_vn(v) or v in SH.CLANNAD

    def new_forbidden(v):
        return v in reserved

    dev = {v for v, r in reserved.items() if r == "dev-sample"}
    both = {r["image"] for r in rp_rows} | {r["image"] for r in na_rows}
    summary = {
        "kind": "reserve-policy-reclaim", "tier": TIER, "dump": stamp, "code": "realplus/reclaim.py",
        "reserved": {"before_realplus_2026_10_12_file": len(old_reserved),
                     "before_by_reason": dict(collections.Counter(old_reserved.values()).most_common()),
                     "after": len(reserved), "after_test": sum(1 for r in reserved.values()
                                                               if not r.startswith("dev-sample")),
                     "after_dev_sample": sum(1 for r in reserved.values() if r.startswith("dev-sample")),
                     "after_by_reason": dict(collections.Counter(reserved.values()).most_common()),
                     "newly_reserved": {v: r for v, r in sorted(reserved.items()) if v not in old_reserved},
                     "freed_vns": len(set(old_reserved) - set(reserved))},
        "distinct_vndb_screenshots": {"realplus_or_native": len(both),
                                      "vns": len({r["vn"] for r in rp_rows} | {r["vn"] for r in na_rows})},
        "realplus": {
            "screenshots": len(rp_rows), "vns": len({r["vn"] for r in rp_rows}),
            "by_kind": {k: {"screenshots": sum(r["kind"] == k for r in rp_rows),
                            "vns": len({r["vn"] for r in rp_rows if r["kind"] == k})} for k in SH.KINDS},
            "by_old_why": dict(collections.Counter(r["old_why"] for r in rp_rows).most_common()),
            "new_selection_total": sum(rp_why.values()) and None,
            "new_selection_dropped": dict(rp_why.most_common()),
            "file": "realplus_shots.jsonl", "sha256": sha256(out / "realplus_shots.jsonl")},
        "native": {
            "frames": len(na_rows), "vns": len({r["vn"] for r in na_rows}),
            "by_kind": dict(collections.Counter(r["kind"] for r in na_rows)),
            "by_new_class": dict(collections.Counter(r["why"] for r in na_rows).most_common()),
            "by_old_why": dict(collections.Counter(r["old_why"] for r in na_rows).most_common()),
            "enumeration_now": dict(na_why.most_common()),
            "file": "native_rows.jsonl", "sha256": sha256(out / "native_rows.jsonl")},
        "synth": {"names": names_reclaim(args.chars, old_forbidden, new_forbidden),
                  "scene_candidates_on_disk": scene_reclaim(old_forbidden, new_forbidden)},
        "labelled": labelled_reclaim(dev),
        "owner_frames": owner_reclaim(reserved),
    }
    summary["realplus"].pop("new_selection_total")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if args.manifest:
        Path(args.manifest).write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("reserved", "realplus", "native", "synth", "labelled")},
                     ensure_ascii=False, indent=1)[:6000], flush=True)


if __name__ == "__main__":
    main()
