#!/usr/bin/env python3
"""Text-free summary of how the devnative cohort was selected and reviewed.

  report.py [--out devnative/manifests/review.json]

Collects counts from the local artifacts ($DEVNATIVE_OUT): the selection report, the readings,
every review stage's parsed decisions (pass 1, pass 2, GRAD check, blind re-check, edge check,
confirming reads), the third-look resolutions, the final labels and the built files. Line ids,
reason codes and counts only: no label text, so it can be committed.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from devnative import decide as D  # noqa: E402
from devnative import paths as P  # noqa: E402
from devnative import sheets as S  # noqa: E402
from teacher import vote  # noqa: E402

STAGES = (("pass1", "pass1"), ("pass2", "pass2"), ("gradcheck", "grad"), ("blind", "blind"),
          ("edgecheck", "edge"), ("confirm", "confirm1"), ("confirm2", "confirm2"))


def jl(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def decisions_summary(recs):
    c = Counter(r["decision"] if r["decision"] != "skip" else f"skip-{r['skip']}" for r in recs)
    return {"lines": len(recs), "decisions": dict(sorted(c.items())), "names": sum(r["name"] for r in recs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(P.MANIFESTS / "review.json"))
    args = ap.parse_args()
    out_dir = P.OUT
    sel = json.loads((out_dir / "select_report.json").read_text(encoding="utf-8"))
    shots = S.load_reads(out_dir / "reads" / "reads.jsonl")
    lines = [ln for s in shots.values() for ln in s.get("lines", [])]
    cand = Counter((s["subset"], S.is_candidate(ln)[1] or "candidate") for s in shots.values() for ln in s["lines"])
    out = {"kind": "devnative-review-summary", "reviewer_kind": "assistant",
           "reviewer": "Claude Opus 5.5 sessions reading contact sheets (no human review)",
           "selection": {k: sel[k] for k in ("listed", "excluded", "selected", "by_subset_kind", "vns_by_subset",
                                             "flags", "clannad_family", "fresh_test_family")},
           "listing": {k: sel["listing"].get(k) for k in ("api", "fetched_utc", "reserved", "reserved_sha256",
                                                         "vns_returned", "screenshots", "queries_sha256",
                                                         "responses_sha256")},
           "engines": json.loads((out_dir / "reads" / "engines.json").read_text()),
           "reads": {"screenshots": len(shots), "lines": len(lines),
                     "by_subset_status": {f"{a}:{b}": n for (a, b), n in sorted(cand.items())}},
           "stages": {}}
    for d, name in STAGES:
        base = out_dir / "review" / d
        if base.exists():
            recs = [r for b in D.batches([base]) for r in D.parsed(b)]
            out["stages"][name] = decisions_summary(recs) | {"batches": len(D.batches([base]))}
    labels = jl(out_dir / "labels.jsonl")
    p1, idx = D.collect([out_dir / "review" / "pass1"]), S.line_index(shots)
    out["pass1_label_support"] = dict(Counter(D.classify(r["text"], idx[i][1])[0] for i, r in p1.items()
                                              if r["decision"] in ("pick", "text")))
    rep = json.loads((out_dir / "review" / "blind_report.json").read_text(encoding="utf-8"))
    sample = json.loads((out_dir / "review" / "blind_ids.json").read_text())
    final = {r["id"] for r in jl(P.MANIFESTS / "lines.jsonl")}
    dropped = {r["id"]: r["why"] for r in jl(P.MANIFESTS / "dropped.jsonl")}
    blind = D.collect([out_dir / "review" / "blind"])
    lab_by = {r["id"]: r for r in labels}
    dis = []
    for x in rep["disagreements"]:
        kind = ("unsure" if x["blind"] == "unsure" else f"skip-{x['skip']}" if x["blind"] == "skip"
                else "end-punctuation" if vote.key(x["label"]).rstrip("。、") == vote.key(x["blind_text"] or "")
                else "text")
        dis.append({"id": x["id"], "subset": x["subset"], "kind": kind, "in_final": x["id"] in final,
                    "final_why": None if x["id"] in final else dropped.get(x["id"])})
    in_final = [i for i in sample["ids"] if i in final]
    agree_final = sum(1 for i in in_final if D.agrees(blind.get(i), lab_by[i]["label"]))
    # the measurement as it was made (labels after passes 1 and 2), on the sampled lines whose
    # screenshot the selection still holds (cohort v2 dropped eval/linefind's screenshots)
    selected = {r["shot"] for r in jl(P.MANIFESTS / "shots.jsonl")}
    on_sel = [i for i in sample["ids"] if i.rsplit("_", 1)[0] in selected]
    dis_sel = sum(1 for x in rep["disagreements"] if x["id"].rsplit("_", 1)[0] in selected)
    out["blind_recheck"] = {"sample": {"n": len(sample["ids"]), "seed": sample["seed"], "from_kept": sample["from"]},
                            "agreement": rep["summary"], "disagreements": dis,
                            "on_selected_shots": {"lines": len(on_sel), "agree": len(on_sel) - dis_sel,
                                                  "rate": round((len(on_sel) - dis_sel) / len(on_sel), 4),
                                                  "wilson95": D.wilson95(len(on_sel) - dis_sel, len(on_sel))},
                            "sampled_lines_in_final": len(in_final), "agree_in_final": agree_final}
    res = jl(out_dir / "review" / "resolutions.jsonl")
    out["third_look"] = [{k: r.get(k) for k in ("id", "keep", "why", "note")} | {"text_changed": bool(r.get("text"))}
                         for r in res]
    out["final_labels"] = {"by": dict(Counter(f"{r['status']}:{r['how'] if r['status'] == 'kept' else r['why']}"
                                              for r in labels)),
                           "kept_by_looks": dict(Counter(str(r.get("looks")) for r in labels if r["status"] == "kept"))}
    out["grad"] = dict(Counter(r["grad_why"] for r in jl(out_dir / "grad" / "grad.jsonl")))
    out["edge"] = dict(Counter(r["edge_why"] for r in jl(out_dir / "edge" / "edge.jsonl")))
    out["built"] = json.loads((P.MANIFESTS / "stats.json").read_text(encoding="utf-8"))
    Path(args.out).write_text(json.dumps(out, ensure_ascii=True, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("final_labels", "grad", "edge")}, indent=1), flush=True)
    print(json.dumps({k: out["blind_recheck"][k] for k in ("on_selected_shots", "sampled_lines_in_final",
                                                           "agree_in_final")}), flush=True)


if __name__ == "__main__":
    main()
