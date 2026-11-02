#!/usr/bin/env python3
"""textspot-desktop round 2 (#173): a training source of dense PC pages, for the lead to oversample: desktop-data's
1b v3 frames (realplus PC screenshots at their own size, #171) with at least 8 labelled horizontal text lines (the
dense sets' rule, eval/tsdesk_dense.py), of every VN outside the held-out and test sets.

  tsdesk_dense_source.py build --out DIR [--min-lines 8] [--shard-frames 200] [--manifest-out M] [--verify-npy]
      DIR/desk_dense_v1/dense_NNN.{npy,json}: the frames as 1b v3 stores them (#97's frames: each sample unchanged
      but for off, h, w), DIR/manifest.json (= M when given): the shards' names, counts and sha256, every frame's
      VNDB ids and screenshot sha256 and its pixels' sha256, the exclusion check; no pixels, no text

The source, as built:
  candidates  1b v3's frames (data/manifests/desktop_data/realplus_native_v3.json; every shard's json sha256, and with
              --verify-npy its npy sha256, checked) with >= min-lines horizontal positives of class text
  excluded    every frame of a VN of, or of a screenshot (file sha256, and VNDB screenshot id where the set has one)
              in: the dense sets (skip_vns.txt: their 200 VNs; tsdesk_dense_v1.json: their 240 screenshots), dev-v2
              (vocr-desktop's frames dev.pkl and pc.pkl), the PC test set (eval/manifests/desktop-pc-test-v1.jsonl),
              confirm-v2 (frames confirm.pkl), the hooked test titles (PCSG00807, NPJH50882: b2-ingest2's VNDB map)
              and the reservation's test VNs (eval/manifests/reserved.json "test": the CLANNAD family, vita-test,
              psp-test and their release partners). The sets' VN ids and hashes are read for this only; no frame of
              theirs is opened
  check       after the build, every written frame is checked again against every set, by VN, by sha256 and by
              screenshot id, and its pixels against the 1b frame it was copied from; the counts (all 0) go into the
              manifest, and the build stops without a manifest when one is not 0
Load it as --extra NAME=MANIFEST:ROOT in train/tsdesk_det.py (its --skip-vns then drops nothing from it).
"""
import argparse
import hashlib
import json
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth", ROOT / "teacher", ROOT / "eval", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

HERD = Path("/local/home/skerraut/herd-vita-ocr")
ART, PV = HERD / "shared" / "artifacts", HERD / "shared" / "private"
MAN = ROOT / "data" / "manifests" / "desktop_data" / "realplus_native_v3.json"
SHARDS = ART / "desktop-data" / "pages" / "realplus_v3"
DENSE_MAN = ROOT / "data" / "manifests" / "tsdesk_dense_v1.json"
SKIP_VNS = PV / "textspot-desktop" / "frames" / "skip_vns.txt"
FRAMES = PV / "vocr-desktop" / "frames"
PCTEST = ROOT / "eval" / "manifests" / "desktop-pc-test-v1.jsonl"
RESERVED = ROOT / "eval" / "manifests" / "reserved.json"
VNDB_MAP = ART / "b2-ingest2" / "vndb_map_v2.json"
TEST_TITLES = ("PCSG00807", "NPJH50882")
DIR_NAME = "desk_dense_v1"
SHOT_ID = re.compile(r"(v\d+_\d+)")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def text_lines(smp):
    return sum(1 for ln in smp["lines"] if not ln[4] and ln[5] == "text")


def shot_id(s):
    m = SHOT_ID.search(Path(str(s)).name) if s else None
    return m.group(1) if m else None


def _frames_ids(path):
    """(VNs, screenshot sha256, VNDB screenshot ids) of an eval/desktop.py frames file (its metadata only)."""
    with open(path, "rb") as f:
        frames = pickle.load(f)["frames"]
    vns = {fr["cluster"] for fr in frames}
    shas = {fr["src"]["sha256"] for fr in frames if fr["src"].get("sha256")}
    ids = {i for i in (shot_id(fr["src"].get("path")) for fr in frames) if i}
    return vns, shas, ids


def exclusion_sets():
    """{name: {"vns", "sha256", "shots", "source"}}: what the source must not hold (module docstring)."""
    out = {}
    skip = {v.strip() for v in SKIP_VNS.read_text(encoding="utf-8").split() if v.strip()}
    dm = json.loads(DENSE_MAN.read_text(encoding="utf-8"))
    d_vns, d_sha, d_ids = set(), set(), set()
    for s in dm["sets"].values():
        d_vns |= set(s["vns"])
        for r in s["shots"]:
            d_sha.add(r["sha256"])
            d_ids.add(r["shot"])
    out["dense_sets"] = {"vns": skip | d_vns, "sha256": d_sha, "shots": d_ids,
                         "source": {"skip_vns.txt": sha256_file(SKIP_VNS), DENSE_MAN.name: sha256_file(DENSE_MAN)}}
    vns, shas, ids = set(), set(), set()
    src = {}
    for n in ("dev.pkl", "pc.pkl"):
        a, b, c = _frames_ids(FRAMES / n)
        vns, shas, ids = vns | a, shas | b, ids | c
        src[n] = sha256_file(FRAMES / n)
    out["dev_v2"] = {"vns": vns, "sha256": shas, "shots": ids, "source": src}
    vns, shas, ids = set(), set(), set()
    for line in PCTEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            vns.add(r["vn"])
            for k in ("sha256", "file_sha256"):
                if r.get(k):
                    shas.add(r[k])
            for k in ("shot", "screenshot", "id"):
                i = shot_id(r.get(k))
                if i:
                    ids.add(i)
    out["pc_test"] = {"vns": vns, "sha256": shas, "shots": ids, "source": {PCTEST.name: sha256_file(PCTEST)}}
    a, b, c = _frames_ids(FRAMES / "confirm.pkl")
    out["confirm_v2"] = {"vns": a, "sha256": b, "shots": c,
                         "source": {"confirm.pkl": sha256_file(FRAMES / "confirm.pkl")}}
    m = json.loads(VNDB_MAP.read_text(encoding="utf-8"))["titles"]
    out["hooked_test_titles"] = {"vns": {v for t in TEST_TITLES for v in (m.get(t) or {}).get("vns") or []},
                                 "sha256": set(), "shots": set(), "titles": list(TEST_TITLES),
                                 "source": {VNDB_MAP.name: sha256_file(VNDB_MAP)}}
    res = json.loads(RESERVED.read_text(encoding="utf-8"))["test"]
    out["reserved_test"] = {"vns": set(res) if not isinstance(res, dict) else set(res.keys()), "sha256": set(),
                            "shots": set(), "source": {RESERVED.name: sha256_file(RESERVED)}}
    return out


def hits(rec, ex):
    """The exclusion sets a frame record {vn, sha256, shot} is in, by what."""
    out = []
    for name, e in ex.items():
        if rec["vn"] in e["vns"]:
            out.append(f"{name}:vn")
        if rec["sha256"] in e["sha256"]:
            out.append(f"{name}:sha256")
        if rec["shot"] in e["shots"]:
            out.append(f"{name}:shot")
    return out


def candidates(man, min_lines, verify_npy):
    """[(shard entry, sample index, sample)] of 1b v3's frames with >= min_lines text lines, and the threshold
    counts (frames and VNs at 6 / 8 / 10 / 12 lines, names counted or not)."""
    out, levels = [], Counter()
    vns = {}
    for s in man["shards"]:
        base = SHARDS / s["dir"] / s["name"]
        if sha256_file(base.with_suffix(".json")) != s["json_sha256"]:
            raise SystemExit(f"{base}.json: sha256 differs from the manifest's")
        samples = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))["samples"]
        used = False
        for i, smp in enumerate(samples):
            n_text = text_lines(smp)
            n_all = sum(1 for ln in smp["lines"] if not ln[4])
            for t in (6, 8, 10, 12):
                if n_text >= t:
                    levels[f"text>={t}"] += 1
                    vns.setdefault(f"text>={t}", set()).add(smp["vn"])
                if n_all >= t:
                    levels[f"text+name>={t}"] += 1
            if n_text >= min_lines:
                out.append((s, i, smp))
                used = True
        if used and verify_npy and sha256_file(base.with_suffix(".npy")) != s["npy_sha256"]:
            raise SystemExit(f"{base}.npy: sha256 differs from the manifest's")
    return out, {k: {"frames": levels[k], "vns": len(vns.get(k, ()))} if k in vns else {"frames": levels[k]}
                 for k in sorted(levels)}


def cmd_build(a):
    from desktopdata import shards as SH
    t0 = time.time()
    man = json.loads(MAN.read_text(encoding="utf-8"))
    ex = exclusion_sets()
    cands, levels = candidates(man, a.min_lines, a.verify_npy)
    hooked = set()
    m = json.loads(VNDB_MAP.read_text(encoding="utf-8"))["titles"]
    for t, e in m.items():
        hooked |= set(e.get("vns") or [])
    keep, dropped = [], Counter()
    for s, i, smp in cands:
        rec = {"vn": smp["vn"], "sha256": smp["meta"]["sha256"], "shot": smp["shot"]}
        h = hits(rec, ex)
        if h:
            for x in h:
                dropped[x] += 1
            dropped["frames"] += 1
            continue
        keep.append((s, i, smp))
    # the dense sets' own frames are 1b frames: they must be among the candidates with the same sha256 (the hash
    # kinds agree), and all of them dropped
    dense_ids = ex["dense_sets"]["shots"]
    c_ids = {smp["shot"]: smp["meta"]["sha256"] for _, _, smp in cands}
    dense_seen = sum(1 for r in json.loads(DENSE_MAN.read_text(encoding="utf-8"))["sets"].values() for x in r["shots"]
                     if c_ids.get(x["shot"]) == x["sha256"])
    keep.sort(key=lambda t: (t[2]["vn"], t[2]["shot"]))
    out = Path(a.out)
    (out / DIR_NAME).mkdir(parents=True, exist_ok=True)
    cache = {}

    def grey(s, i):
        key = (s["dir"], s["name"])
        if key not in cache:
            cache.clear()
            cache[key] = SH.read_shard(SHARDS / s["dir"] / s["name"])
        data, samples = cache[key]
        return SH.canvas(data, samples[i]).copy()
    frames, shards = [], []
    for k in range(0, len(keep), a.shard_frames):
        part = keep[k:k + a.shard_frames]
        samples, greys = [], []
        for s, i, smp in part:
            g = grey(s, i)
            smp = json.loads(json.dumps(smp))
            samples.append(smp)
            greys.append(g)
            frames.append({"shot": smp["shot"], "vn": smp["vn"], "sha256": smp["meta"]["sha256"],
                           "size": [int(g.shape[1]), int(g.shape[0])], "text_lines": text_lines(smp),
                           "lines": len(smp["lines"]), "pixels_sha256": hashlib.sha256(g.tobytes()).hexdigest(),
                           "from": f"{s['dir']}/{s['name']}:{i}", "hooked_vn": smp["vn"] in hooked})
        name = f"dense_{k // a.shard_frames:03d}"
        SH.write_shard(out / DIR_NAME / name, samples, greys)
        shards.append(SH.shard_record(out / DIR_NAME / name, DIR_NAME))
    # the check, on what was written (read back from the shards, the sets loaded again, compared without hits()):
    # every frame against every set by VN, sha256 and screenshot id, and its pixels against the 1b frame
    ex2 = exclusion_sets()
    check = {name: {"vn": 0, "sha256": 0, "shot": 0, "vns_in_set": len(e["vns"]), "sha256_in_set": len(e["sha256"]),
                    "shots_in_set": len(e["shots"]), "source": e["source"]} for name, e in ex2.items()}
    n_pix_bad = n_frames = 0
    src_of = {f["shot"]: f["from"] for f in frames}
    for r in shards:
        data, samples = SH.read_shard(out / DIR_NAME / r["name"])
        for smp in samples:
            n_frames += 1
            for name, e in ex2.items():
                check[name]["vn"] += smp["vn"] in e["vns"]
                check[name]["sha256"] += smp["meta"]["sha256"] in e["sha256"]
                check[name]["shot"] += smp["shot"] in e["shots"]
            sd, si = src_of[smp["shot"]].rsplit(":", 1)
            d0, s0 = SH.read_shard(SHARDS / sd)
            if not np.array_equal(SH.canvas(data, smp), SH.canvas(d0, s0[int(si)])):
                n_pix_bad += 1
    bad = sum(v[w] for v in check.values() for w in ("vn", "sha256", "shot")) + n_pix_bad
    tot = SH.totals(shards)["all"] if shards else {}
    vns = sorted({f["vn"] for f in frames})
    rec = {"kind": "textspot-desktop-dense-source", "schema": 1, "issue": 173, "created": SH.utc(),
           "format": "#97's frames (synth/textdet_data.py): train/tsdesk_det.py --extra NAME=MANIFEST:ROOT reads it "
                     "(desktop_det.ExtraPool); ROOT is the directory holding " + DIR_NAME,
           "rule": {"source": "desktop-data 1b v3 frames", "min_text_lines": a.min_lines,
                    "counted": "horizontal positives of class text (eval/tsdesk_dense.py's dense rule)",
                    "per_vn": "every dense frame of an eligible VN", "excluded": sorted(ex),
                    "hooked_vns": "kept (training VNs)"},
           "source": {"manifest": str(MAN.relative_to(ROOT)), "manifest_sha256": sha256_file(MAN),
                      "plan_sha256": man["plan_sha256"], "json_checked": True, "npy_checked": bool(a.verify_npy)},
           "candidates": {"frames": len(cands), "vns": len({smp["vn"] for _, _, smp in cands}),
                          "dropped": dict(sorted(dropped.items())), "dense_set_frames_found": dense_seen,
                          "dense_set_frames": len(dense_ids), "thresholds": levels},
           "check": {"frames_checked": n_frames, "by_set": check, "pixels_differ_from_source": n_pix_bad,
                     "passed": bad == 0},
           "totals": {"frames": len(frames), "vns": len(vns), "text_lines": sum(f["text_lines"] for f in frames),
                      "lines": tot.get("lines", 0), "ignore": tot.get("ignore", 0), "pixels": tot.get("pixels", 0),
                      "by_class": tot.get("by_class", {}),
                      "sizes": dict(Counter(f"{f['size'][0]}x{f['size'][1]}" for f in frames).most_common(20)),
                      "frames_of_hooked_vns": sum(f["hooked_vn"] for f in frames),
                      "max_frames_per_vn": max(Counter(f["vn"] for f in frames).values()) if frames else 0,
                      "npy_bytes": sum((out / DIR_NAME / f"{r['name']}.npy").stat().st_size for r in shards)},
           "code_sha256": SH.code_sha256(ROOT, ["train/tsdesk_dense_source.py", "eval/tsdesk_dense.py",
                                                "desktopdata/shards.py"]),
           "seconds": round(time.time() - t0, 1), "vns": vns, "frames": frames, "shards": shards,
           "note": "shards hold copyrighted VNDB screenshots and stay local; this manifest holds names, counts, "
                   "sha256 and VNDB ids"}
    if bad:
        print(json.dumps({"check": rec["check"]}, indent=1))
        raise SystemExit(f"the exclusion check failed ({bad}): no manifest written")
    txt = json.dumps(rec, ensure_ascii=False, indent=1) + "\n"
    (out / "manifest.json").write_text(txt, encoding="utf-8")
    if a.manifest_out:
        Path(a.manifest_out).write_text(txt, encoding="utf-8")
    print(json.dumps({"frames": len(frames), "vns": len(vns), "text_lines": rec["totals"]["text_lines"],
                      "lines": rec["totals"]["lines"], "shards": len(shards), "check_passed": True,
                      "dropped": rec["candidates"]["dropped"], "seconds": rec["seconds"]}), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--out", required=True)
    p.add_argument("--min-lines", type=int, default=8)
    p.add_argument("--shard-frames", type=int, default=200)
    p.add_argument("--manifest-out")
    p.add_argument("--verify-npy", action="store_true", help="check the npy sha256 of every 1b shard read from")
    a = ap.parse_args(argv)
    {"build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
