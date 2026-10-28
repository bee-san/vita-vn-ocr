#!/usr/bin/env python3
"""VOCR6-V (issue #159): #103's vertical test set, read once after the last decision (eval/vocr6v_protocol.json
"scoring.report_only"). Report only: it decides nothing.

  vocr6v_test_once.py --sets DIR --private DIR --blob-old VOCR5_V.vocr --blob-new VOCR6_V.vocr \\
      --rubi RUBI_CLONE --tool vjo-page-ocr --models MODELS_DIR --cli runtime/build/vocr_cli \\
      --attempts DIR --work DIR --out eval/vocr6v_test_<date>.json

Before anything is read it writes DIR/vocr6v-test.json (the attempt record: time, git head, the inputs' sha256); an
existing record refuses the run. Then, for the incumbent and the new blob:
  - Rubi's job (Rubi tools/vertical_eval.py ext-units + run, `--vertical on --line-fix on`, the shipped 1 MB LM:
    the measurement build that takes the blob in VOCR5_V_w8.vocr's place) on every test region (tight);
  - #103's fixed column crops (eval/vertical.py columns, arm rot, the C runtime W8I greedy).
Dialogue CER per cohort (VN: the public VNDB pages and the owner's private Trick x Logic frames, apart and pooled;
manga: Black Jack ni Yoroshiku vol. 13), intervals by VN series cluster / manga book page (10,000 resamples, seed 0),
and new - old. The output holds counts, rates and intervals only (no text of any frame).
Seen flags: Yunohana SpRING! (v17343, PS Vita, one test page) is the hooked title both new arms trained on, and the
shipped LM's text holds its train pages' script.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import vocr6v  # noqa: E402

SEEN = {"v17343": "Yunohana SpRING! (PCSG00653): the new arms trained on its hooked train pages; the shipped LM's "
                  "text holds its train pages' script"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_paths(a):
    """The read's seven inputs, each under its own key: the public and the private sets share their file names.
    (The one read of 2026-10-28 keyed the four sets by file name, so its attempt record holds the private sets'
    hashes and not the public ones', which were lost to that collision: docs/vocr6v.md, "The test set, once".)"""
    return {"public/columns_test.jsonl": Path(a.sets) / "columns_test.jsonl",
            "public/regions_test.jsonl": Path(a.sets) / "regions_test.jsonl",
            "private/columns_test.jsonl": Path(a.private) / "columns_test.jsonl",
            "private/regions_test.jsonl": Path(a.private) / "regions_test.jsonl",
            "blob_old": Path(a.blob_old), "blob_new": Path(a.blob_new), "tool": Path(a.tool)}


def attempt_doc(a, head, start):
    """The attempt record main() writes before anything is read: every input's sha256 and path under its own key."""
    files = input_paths(a)
    return {"kind": "vocr6v-test-attempt", "start": start, "head": head,
            "inputs": {key: sha(path) for key, path in files.items()},
            "paths": {key: str(path) for key, path in files.items()}}


def models_dir(base, blob, out):
    out.mkdir(parents=True, exist_ok=True)
    for f in Path(base).iterdir():
        dst = out / f.name
        if not dst.exists():
            os.symlink(os.path.realpath(f), dst)
    v = out / "VOCR5_V_w8.vocr"
    if v.exists() or v.is_symlink():
        v.unlink()
    os.symlink(os.path.realpath(blob), v)
    return out


def job(a, units, blob, name, work):
    m = models_dir(a.models, blob, work / f"models_{name}")
    res = work / f"{units.stem}.{name}.results.jsonl"
    subprocess.run([sys.executable, str(Path(a.rubi) / "tools" / "vertical_eval.py"), "run", "--units", str(units),
                    "--pages", "/", "--tool", a.tool, "--models", str(m), "--tool-args", "--vertical on --line-fix on",
                    "--workers", "5", "--out", str(res)], check=True, stdout=subprocess.DEVNULL)
    return res


def columns(a, manifest, blob, name, work):
    out = work / f"{Path(manifest).stem}.{name}.cols.json"
    subprocess.run([sys.executable, str(HERE / "vertical.py"), "columns", "--manifest", str(manifest), "--arm", "rot",
                    "--model", f"cmd:{a.cli} {blob} --mode w8i --stdin", "--out", str(out)], check=True,
                   stdout=subprocess.DEVNULL)
    return json.loads(out.read_text(encoding="utf-8"))["scores"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("sets", "private", "blob-old", "blob-new", "rubi", "tool", "models", "cli", "attempts", "work", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    rec = Path(a.attempts) / "vocr6v-test.json"
    if rec.exists():
        raise SystemExit(f"{rec} exists: the test set has been read already")
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).decode().strip()
    doc = attempt_doc(a, head, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    Path(a.attempts).mkdir(parents=True, exist_ok=True)
    rec.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    out = {"attempt": doc, "seen": SEEN, "job": {}, "fixed_columns": {}}
    parts = {"public": Path(a.sets) / "regions_test.jsonl", "private": Path(a.private) / "regions_test.jsonl"}
    regions = {}
    units = {}
    for part, path in parts.items():
        regions.update(vocr6v.region_index(path))
        u = work / f"units_{part}.jsonl"
        subprocess.run([sys.executable, str(Path(a.rubi) / "tools" / "vertical_eval.py"), "ext-units", "--vdev",
                        str(path), "--out", str(u)], check=True, stdout=subprocess.DEVNULL)
        units[part] = [x for x in vocr6v.read_jsonl(u) if x["kind"] == "tight"]
        tight = work / f"units_{part}_tight.jsonl"
        tight.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in units[part]), encoding="utf-8")
        units[part + "_path"] = tight
    per = {}
    for name, blob in (("old", a.blob_old), ("new", a.blob_new)):
        rows = []
        for part in parts:
            rows += vocr6v.read_jsonl(job(a, units[part + "_path"], blob, name, work))
        allu = units["public"] + units["private"]
        per[name] = {"vn": vocr6v.per_cluster(allu, rows, "cluster", regions, "vn"),
                     "vn_public": vocr6v.per_cluster(units["public"], rows, "cluster", regions, "vn"),
                     "vn_private": vocr6v.per_cluster(units["private"], rows, "cluster", regions, "vn"),
                     "manga": vocr6v.per_cluster(allu, rows, "book", regions, "manga")}
        seen_free = [u for u in allu if regions[u["id"].rsplit(":", 1)[0]]["vn"] not in SEEN]
        per[name]["vn_unseen"] = vocr6v.per_cluster(seen_free, rows, "cluster", regions, "vn")
    for coh in ("vn", "vn_public", "vn_private", "vn_unseen", "manga"):
        o, n = per["old"][coh], per["new"][coh]
        d, ci = vocr6v.boot_diff([o], [n])
        out["job"][coh] = {"old": round(vocr6v.cer(o), 2), "old_ci95": vocr6v.boot_cer(o),
                           "new": round(vocr6v.cer(n), 2), "new_ci95": vocr6v.boot_cer(n),
                           "diff": d, "ci95": ci, "clusters": len(o), "chars": sum(v[1] for v in o.values())}
    for name, blob in (("old", a.blob_old), ("new", a.blob_new)):
        pub = columns(a, Path(a.sets) / "columns_test.jsonl", blob, name, work)
        pri = columns(a, Path(a.private) / "columns_test.jsonl", blob, name, work)
        vn_e = pub["vn"]["errs"] + pri["vn"]["errs"]
        vn_c = pub["vn"]["chars"] + pri["vn"]["chars"]
        out["fixed_columns"][name] = {"vn": round(100 * vn_e / vn_c, 2), "vn_chars": vn_c,
                                      "vn_public": round(100 * pub["vn"]["errs"] / pub["vn"]["chars"], 2),
                                      "vn_private": round(100 * pri["vn"]["errs"] / pri["vn"]["chars"], 2),
                                      "manga": round(100 * pub["manga"]["errs"] / pub["manga"]["chars"], 2),
                                      "manga_chars": pub["manga"]["chars"]}
    doc["end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    out["attempt"] = doc
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"job": out["job"], "fixed_columns": out["fixed_columns"]}, indent=1))


if __name__ == "__main__":
    main()
