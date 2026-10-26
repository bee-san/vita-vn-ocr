#!/usr/bin/env python3
"""hval units that arm A's init has seen (issue #125, audit-vocr6-lead F3): the region units of an hval plan holding
a line whose eval/metrics.norm text equals the label of a row of the same VN in VOCR5's own fine-tune pools (the
owner's console captures and teacher_train, as pinned by eval/vocr5_pools.sha256). VOCR5's fine-tune, and so every
arm that starts from it (stage 1's A), trained on those rows; a fresh fine-tune from the synthetic pretrain (B) did
not. eval/vocr6_guard.py takes the same rows out of the screening copies, so no VOCR6 run's new steps see them.

  vocr6_exposed.py --plan HVAL_PLAN.pkl --vn-of POOLS_HVAL_DIR --pool FT_POOL.npz ... --out UNITS.json

Unit ids are region ids (title, page, the region's lines, variant), so the list holds for any plan built on the same
page pool, whatever finder cut its crops.
"""
import argparse
import glob
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from metrics import norm  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def title_vns(vn_of):
    """title id -> VNDB id, from the meta of vocr6-ingest's hval pools."""
    t2v = {}
    for f in glob.glob(str(Path(vn_of) / "*" / "*.npz")):
        for m in json.loads(str(np.load(f)["meta"])):
            if m.get("vn"):
                t2v[m["title_id"]] = m["vn"]
    return t2v


def exposed_units(units, t2v, pool_rows):
    """units: the plan's units; pool_rows: [(label, vn)] of the fine-tune pools -> sorted exposed unit ids."""
    seen = {}
    for lab, vn in pool_rows:
        if lab and vn:
            seen.setdefault(vn, set()).add(norm(lab))
    out = set()
    for u in units:
        texts = seen.get(t2v.get(u["title"]))
        if texts and any(norm(ln["text"]) in texts for ln in u["lines"]):
            out.add(u["unit"])
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--vn-of", required=True, help="vocr6-ingest's pools/<v>/hval directory (title -> VNDB id)")
    ap.add_argument("--pool", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plan = pickle.load(open(a.plan, "rb"))
    rows = []
    for p in a.pool:
        z = np.load(p)
        rows += [(lab, m.get("vn")) for lab, m in zip(json.loads(str(z["labels"])), json.loads(str(z["meta"])))]
    units = exposed_units(plan["units"], title_vns(a.vn_of), rows)
    doc = {"kind": "vocr6-hval-exposed-units", "plan_sha256": sha256(a.plan),
           "rule": "units holding a line whose norm'd text equals a same-VN row label of VOCR5's fine-tune pools "
                   "(arm A's init trained on them; audit-vocr6-lead F3)",
           "pools": {p: sha256(p) for p in a.pool}, "units": units}
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    print(f"exposed units: {len(units)} of {len(plan['units'])} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
