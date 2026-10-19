#!/usr/bin/env python3
"""The real half of the full-frame spotter's training data (issue #104): #97's pinned real canvases
(data/manifests/textdet_shards_v1.json: realplus, teacher, devnative, the owner's), measured from their JSON, with
their exclusion record checked and the canvases the textspot protocol's exclusions would still remove listed.

  textspot_real_half.py --out JSON [--manifest data/manifests/textdet_shards_v1.json]

Reads shard JSON, #97's hygiene rows and its reserve_check record only (no image, no detector):
  stats        per source: canvases by size and kind, lines per canvas, vertical lines, ignore areas, the share of
               lines under 6 px and over 72 px across, whole screenshots against cut regions
  reservation  eval/manifests/reserved.json against the record #97's hygiene wrote (reserve_check.json, RESERVE_OK,
               dev_sample blocked), and an independent check: every canvas's shot has a hygiene row, and no row is of
               a reserved test or development-sample VN or a test session
  skip         canvases of a screenshot that an eval/manifests or eval/linefind/manifests list holds (by VNDB image id
               or file sha256; #97 excluded vita-test's, psp-test's and the six scored linefind sets' only, so the
               later vertical sets and linefind's unscored candidate lists and psp_pool remain), and, a wider reading
               of "every linefind VN", canvases of a VN on any eval/linefind list. The textspot sets are never opened
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
SHARDS = {"realplus": HERD / "shared/artifacts/textdet/shards/realplus",
          "teacher": HERD / "shared/artifacts/textdet/shards/teacher",
          "devnative": HERD / "shared/artifacts/textdet/shards/devnative",
          "owner": HERD / "shared/private/textdet/shards/owner"}
ROWS = {"realplus": HERD / "shared/artifacts/textdet/hygiene/rows_realplus.jsonl",
        "teacher": HERD / "shared/artifacts/textdet/hygiene/rows_teacher.jsonl",
        "devnative": HERD / "shared/artifacts/textdet/hygiene/rows_devnative.jsonl",
        "owner": HERD / "shared/private/textdet/hygiene/rows_owner.jsonl"}
CHECK = HERD / "shared/private/textdet/hygiene/reserve_check.json"
URL_ID = re.compile(r"/(\d+)\.(?:jpg|png)$")
SHOT_ID = re.compile(r"^v\d+_(\d+)$")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    for x in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            yield json.loads(x)
        except json.JSONDecodeError:
            continue


def eval_lists():
    """{image id: lists}, {sha256: lists}, {vn: linefind lists} of every eval/manifests and eval/linefind/manifests
    jsonl but the textspot sets."""
    ids, shas, vns = {}, {}, {}
    files = sorted((ROOT / "eval/manifests").glob("*.jsonl")) + \
        sorted((ROOT / "eval/linefind/manifests").glob("*.jsonl"))
    for f in files:
        if f.name.startswith("textspot"):
            continue
        for r in jsonl(f):
            m = URL_ID.search(r.get("url") or "")
            if m:
                ids.setdefault(m.group(1), set()).add(f.name)
            if r.get("sha256"):
                shas.setdefault(r["sha256"], set()).add(f.name)
            if f.parent.parent.name == "linefind" and r.get("vn"):
                vns.setdefault(r["vn"], set()).add(f.name)
    return ids, shas, vns


def shard_jsons(source):
    return [Path(p) for p in sorted(glob.glob(str(SHARDS[source] / "*.json")))
            if not p.endswith(".tmp.json") and Path(p).with_suffix(".npy").exists()]


def main():
    import textspot_data as TS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(ROOT / "data/manifests/textdet_shards_v1.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    pinned = {(s["dir"], s["name"]): s for s in man["shards"]}
    res_path = ROOT / "eval/manifests/reserved.json"
    res = json.loads(res_path.read_text(encoding="utf-8"))
    test, dev, sessions = set(res["test"]), set(res["dev_sample"]), set(res["sessions"])
    ids, shas, lf_vns = eval_lists()
    stats = TS.real_stats([p for s in SHARDS for p in shard_jsons(s)])
    check = json.loads(CHECK.read_text(encoding="utf-8"))
    reservation = {"reserved_sha256": sha256(res_path), "reserved_dump": res["dump"],
                   "textdet_manifest_reserved_sha256": man["reserved_sha256"],
                   "record": {"verdict": check["verdict"], "reservation_dump": check["reservation_dump"],
                              "dev_sample_mode": check["dev_sample_mode"],
                              "rows": {k: v["rows"] for k, v in check["pools"].items()},
                              "blockers": {k: len(v["blockers"]) for k, v in check["pools"].items()}},
                   "sources": {}}
    skip = {"eval_screenshot": {}, "linefind_list_vn": {}}
    for src in SHARDS:
        rows = {r["shot"]: r for r in jsonl(ROWS[src])}
        shots, canvases, match = {}, 0, 0
        for p in shard_jsons(src):
            pin = pinned.get((src, p.stem))
            match += bool(pin and sha256(p) == pin["json_sha256"])
            for s in json.loads(p.read_text(encoding="utf-8"))["samples"]:
                canvases += 1
                shots[s["shot"]] = shots.get(s["shot"], 0) + 1
        vns = {rows[s]["vn"] for s in shots if s in rows}
        in_sessions = 0
        for s in shots:
            r = rows.get(s) or {}
            if (r.get("release") or "") in sessions or any(q in sessions for q in
                                                            re.findall(r"r\d+", r.get("session") or "")):
                in_sessions += 1
        reservation["sources"][src] = {
            "canvases": canvases, "shots": len(shots), "vns": len(vns),
            "shard_json_match_pin": f"{match}/{sum(1 for k in pinned if k[0] == src)}",
            "shots_without_row": sum(1 for s in shots if s not in rows),
            "test_vns": len(vns & test), "dev_sample_vns": len(vns & dev), "test_session_shots": in_sessions}
        for shot, n in sorted(shots.items()):
            r = rows.get(shot) or {}
            m = URL_ID.search(r.get("url") or "") or SHOT_ID.match(shot)
            why = set(ids.get(m.group(1), ())) if m else set()
            why |= shas.get(r.get("sha256"), set()) if r.get("sha256") else set()
            if why:
                skip["eval_screenshot"][shot] = {"source": src, "canvases": n, "lists": sorted(why),
                                                 "vn": r.get("vn")}
            elif r.get("vn") in lf_vns:
                skip["linefind_list_vn"][shot] = {"source": src, "canvases": n, "lists": sorted(lf_vns[r["vn"]]),
                                                  "vn": r["vn"]}
    for k in skip:
        rows = skip[k]
        skip[k] = {"shots": len(rows), "canvases": sum(v["canvases"] for v in rows.values()),
                   "by_source": {s: sum(v["canvases"] for v in rows.values() if v["source"] == s) for s in SHARDS},
                   "lists": sorted({lst for v in rows.values() for lst in v["lists"]})}
        if k == "eval_screenshot":      # public VNDB screenshots: the shot keys (vn_imageid) the trainer skips
            skip[k]["skip_shots"] = sorted(rows)
        else:                           # by VN (owner frame keys stay private)
            skip[k]["skip_vns"] = sorted({v["vn"] for v in rows.values()})
    doc = {"kind": "textspot-real-half", "schema": 1, "textdet_manifest": str(Path(args.manifest).name),
           "textdet_manifest_sha256": sha256(args.manifest), "code_sha256": {
               "synth/textspot_real_half.py": sha256(Path(__file__)), "synth/textspot_data.py": sha256(TS.__file__)},
           "stats": stats, "reservation": reservation, "skip": skip}
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"reservation": reservation, "skip": {k: {kk: vv for kk, vv in v.items()
                                                               if kk not in ("skip_shots", "skip_vns")}
                                                           for k, v in skip.items()}}, indent=1))


if __name__ == "__main__":
    main()
