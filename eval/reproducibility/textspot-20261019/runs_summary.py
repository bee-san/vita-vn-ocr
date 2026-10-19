#!/usr/bin/env python3
"""Summarise the textspot family B sweep (sweep.sh's OUT directory) for the report.

  runs_summary.py OUT [--json FILE] [--md FILE]

Per run of the protocol's 22 (eval/textspot_protocol.json "candidates.family_b"): started and finished (UTC), wall
time, attempts (sweep.sh's moved failed attempts and crashes.jsonl), the loss curve (the logged total, #97's
k_bce + dice + 0.5 o + a, at steps 200, 2,000, 10,000, 20,000, 24,000 = the last float record, the first record
after QAT starts, and 30,000; the four parts averaged over the last five records) and the blobs' bytes and sha256.
Across the finished runs it checks that every run recorded the same code, manifests, protocol and commit, and
lists what differs. Reads only the runs' own files; scores nothing.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PARTS = ("k_bce", "dice", "o", "a")
AT = (200, 2000, 10000, 20000)


def protocol_runs():
    fb = json.loads((ROOT / "eval" / "textspot_protocol.json").read_text())["candidates"]["family_b"]
    return [f"{sc}_{sh}_seed{n}" for sc, shapes in fb["shapes"].items() for sh in shapes for n in fb["seeds"]]


def total(rec):
    return rec["k_bce"] + rec["dice"] + 0.5 * rec["o"] + rec["a"]


def jsonl(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def curve(log, qat_start):
    recs = [r for r in log if "lr" in r and all(k in r for k in PARTS)]
    by = {r["step"]: r for r in recs}
    out = {f"total_{s}": round(total(by[s]), 4) for s in AT if s in by}
    pre = [r for r in recs if r["step"] <= qat_start]
    post = [r for r in recs if r["step"] > qat_start]
    if pre:
        out["total_last_float"] = round(total(pre[-1]), 4)
        out["step_last_float"] = pre[-1]["step"]
    if post:
        out["total_first_qat"] = round(total(post[0]), 4)
        out["step_first_qat"] = post[0]["step"]
    if recs:
        out["total_final"] = round(total(recs[-1]), 4)
        out["step_final"] = recs[-1]["step"]
        tail = recs[-5:]
        out["parts_last5"] = {k: round(sum(r[k] for r in tail) / len(tail), 5) for k in PARTS}
        out["ms_per_step"] = round(1000 * recs[-1]["sec"] / recs[-1]["step"], 1)
    return out


def summarise(out_dir):
    out_dir = Path(out_dir)
    crashes = jsonl(out_dir / "crashes.jsonl")
    restarts = jsonl(out_dir / "restarts.jsonl")
    rows = []
    for name in protocol_runs():
        d = out_dir / name
        failed = sorted(p.name for p in out_dir.glob(f"{name}.failed*") if p.is_dir())
        row = {"run": name, "state": "missing", "failed_attempts": failed,
               "crashes": [c for c in crashes if c.get("run") == name],
               "restarts": [r for r in restarts if r.get("run") == name]}
        if (d / "run.json").exists():
            info = json.loads((d / "run.json").read_text())
            sk = info.get("skip") or {}
            row.update(state="running", started=info["started"], steps=info["steps"], qat_start=info["qat_start"],
                       git=info.get("git"), code_sha256=info["code_sha256"],
                       manifests_sha256=info["manifests_sha256"], protocol_sha256=info.get("protocol_sha256"),
                       skip={"dropped": sk.get("dropped"), "dropped_total": sk.get("dropped_total"),
                             "lists": [{k: v for k, v in r.items() if k.endswith("sha256") or k.startswith("skip")}
                                       for r in sk.get("lists") or []]},
                       curve=curve(jsonl(d / "log.jsonl"), info["qat_start"]))
        if (d / "done.json").exists():
            done = json.loads((d / "done.json").read_text())
            row.update(state="finished", finished=done["finished"], wall_seconds=done["wall_seconds"],
                       train_seconds=done["seconds"], vtd_bytes=done["vtd_bytes"], vtd_sha256=done["vtd_sha256"],
                       vtsp_bytes=done["vtsp_bytes"], vtsp_sha256=done["vtsp_sha256"], pt_sha256=done["pt_sha256"])
        rows.append(row)
    fin = [r for r in rows if r["state"] == "finished"]
    same = {}
    for key in ("code_sha256", "manifests_sha256", "protocol_sha256", "git", "skip"):
        vals = {json.dumps(r.get(key), sort_keys=True) for r in fin}
        same[key] = len(vals) <= 1
    return {"kind": "textspot-train-runs", "runs": rows, "finished": len(fin), "of": len(rows),
            "consistent": same, "crashes": len(crashes), "restarts": len(restarts)}


def markdown(doc):
    lines = ["| run | started | finished | wall h | attempts | loss 200 / 24k / first QAT / 30k | final k_bce / dice / "
             "o / a (last 5) | VTSP B | final.vtsp sha256 |", "|---|---|---|---:|---:|---|---|---:|---|"]
    for r in doc["runs"]:
        c = r.get("curve", {})
        p = c.get("parts_last5", {})
        loss = " / ".join(f"{c[k]:.3f}" if k in c else "-" for k in ("total_200", "total_last_float",
                                                                      "total_first_qat", "total_final"))
        parts = " / ".join(f"{p[k]:.3f}" for k in PARTS) if p else "-"
        wall = f"{r['wall_seconds'] / 3600:.2f}" if "wall_seconds" in r else "-"
        lines.append(f"| {r['run']} | {r.get('started', '-')} | {r.get('finished', r['state'])} | {wall} | "
                     f"{1 + len(r['failed_attempts'])} | {loss} | {parts} | {r.get('vtsp_bytes', '-')} | "
                     f"`{r.get('vtsp_sha256', '-')}` |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--json")
    ap.add_argument("--md")
    args = ap.parse_args()
    doc = summarise(args.out)
    if args.json:
        Path(args.json).write_text(json.dumps(doc, indent=1) + "\n")
    md = markdown(doc)
    if args.md:
        Path(args.md).write_text(md)
    print(md)
    print(f"finished {doc['finished']}/{doc['of']}; crashes {doc['crashes']}, restarts {doc['restarts']}; "
          f"consistent {doc['consistent']}")


if __name__ == "__main__":
    main()
