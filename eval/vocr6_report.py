#!/usr/bin/env python3
"""VOCR6 T1's results records and the tables docs/vocr6.md shows (#125).

  vocr6_report.py records --out eval/vocr6_results_20261027.json
      collects, from the lead's artifacts, every decision of stages 1-4 (eval/vocr6_select.py compare outputs: totals,
      seed pairs, checks, adopt) and the summary of every run they compare (score files' `summary`: CER only, no text,
      no rows), plus the shipped blobs' sha256 / bytes / arena. Counts and rates only.
  vocr6_report.py tables [--records FILE]
      renders the markdown tables of docs/vocr6.md from the records (tests/test_vocr6_report.py checks that the
      document shows them)."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = Path("/local/home/skerraut/herd-vita-ocr/shared/artifacts")
RECORDS = ROOT / "eval/vocr6_results_20261027.json"
SCORE_DIRS = {"v1": [A / "vocr6-lead/scores"], "v2": [A / "vocr6-lead/scores_v2", A / "vocr6-s1-a/scores",
                                                       A / "vocr6-s1-b/scores", A / "vocr6-s2-a/scores",
                                                       A / "vocr6-s2-b/scores", A / "vocr6-s3-a/scores"],
              "shipped": [A / "vocr6-lead/scores4"]}
# decision file -> (stage, arm, plan, what)
DECISIONS = {
    "s1_A15vsV5N": ("1", "A15", "v2", "fine-tune VOCR5-N's checkpoint, 15k steps (vs VOCR5-N)"),
    "s1_BvsA_excl": ("1", "B15", "v2", "VOCR5-N's 150k synthetic pretrain, then fine-tune (vs A15, exposed units out)"),
    "s2_S2cil": ("2", "S2cil", "v2", "console crops at 0.10 of the hooked 0.25"),
    "s2_S2t5": ("2", "S2t5", "v2", "per-title temperature 0.5"),
    "s2_S2x2": ("2", "S2x2", "v2", "PSP lines in the Adrenaline 2x view"),
    "s2_S2s6": ("2", "S2s6", "v2", "synth6 v3 (game fonts) at 0.15"),
    "s2_S2r70": ("2", "S2r70", "v2", "real share 70%"),
    "s2_S2r30": ("2", "S2r30", "v2", "real share 30%"),
    "s2_S3attn": ("3", "S3attn", "v2", "one global attention layer (#130)"),
    "s2_S3rep": ("3", "S3rep", "v2", "RepCBR branches, training only"),
    "s2_S3long": ("3", "S3long", "v2", "A15 at 45,000 steps"),
    "s4_S": ("4a", "R3-S", "shipped", "A15 + S2cil at Small vs the released VOCR5-S"),
    "s4_N": ("4a", "R3-N", "shipped", "A15 + S2cil at Normal vs the released VOCR5-N"),
    "s4_L": ("4a", "R3-L", "shipped", "A15 + S2cil at Large vs the released VOCR5-L"),
    "s4_HN": ("4b", "4b-N", "shipped", "4b (hval folded in) vs its 4a sibling"),
    "s4_HL": ("4b", "4b-L", "shipped", "4b (hval folded in) vs its 4a sibling"),
    # stage 4c (amendment 6, its open cases audit-vocr6-lead-h H1) and 4d (amendment 7): the follow-up, v0.4.1-vocr6.2
    "s4c_N": ("4c", "4c-N", "shipped", "4b's recipe at 45,000 steps, hval in, into 4c-L (vs the shipped 4b-N into "
                                       "4b-L)"),
    "s4c_L": ("4c", "4c-L", "shipped", "4b's recipe at 45,000 steps, hval in (vs the shipped 4b-L)"),
    "s4d_r": ("4d", "4d-r", "shipped", "Small with VOCR5's real pools back at full weight (mix r3all_v5r) vs the "
                                       "released VOCR5-S"),
    "s4d_l": ("4d", "4d-l", "shipped", "Small, R3 at half the learning rate (6e-4) vs the released VOCR5-S"),
    "s4d_lh": ("4d-fold", "4d-l fold-in", "shipped", "4d-l with hval folded in (into the shipped 4b-N) vs its 4a "
                                                     "sibling seed 1 (into v6s2cil_N_seed1)"),
    "s4d_report_SvsV5S_up4cN": ("4d-report", "Small in the follow-up's cascade", "shipped",
                                "report only: the new Small vs VOCR5-S, both cascading into 4c-N"),
}
EXTRA_RUNS = {"shipped": ["VOCR5-S_up6N", "v6f4h_S_seed1"]}
# the shipped plain blobs of each release: v0.4.0-vocr6.1 (stage 4b's N and L; Small stayed VOCR5-S) and the
# follow-up v0.4.1-vocr6.2 (stage 4c's N and L, stage 4d's Small)
SHIPPED_V040 = {"N": A / "vocr6-final-train/blobs/v6f4h_N_seed1_w8.vocr",
                "L": A / "vocr6-final-train/blobs/v6f4h_L_seed1_w8.vocr"}
SHIPPED = {"S": A / "vocr6-lead/blobs/v6f4dlh_S_seed1_w8.vocr",
           "N": A / "vocr6-lead/blobs/v6f4c_N_seed1_w8.vocr",
           "L": A / "vocr6-lead/blobs/v6f4c_L_seed1_w8.vocr"}
# the follow-up's gate fits (eval/vocr6_gate.sh on main, vocr6-followup): the shipped blob of each tier
GATE_RESULT = A / "vocr6-followup/gate/result.json"
COMPONENTS = ("hval_page", "hval_clean", "hval_page_all", "hval_page_seen", "hval_page_nomerge", "tv", "dn", "wd",
              "dev")
SUMMARY_KEYS = ("hval_page", "hval_page_vita", "hval_page_psp", "hval_clean_dlg", "hval_page_all", "hval_page_seen",
                "tv", "dn", "wd", "dev_score")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# the hval plan each record plan was read on (eval/vocr6_protocol.json: scoring.hval_plan_v1, scoring.hval_plan,
# stages.4.plans.hval): a score file of the same name read on another plan is skipped, never mixed in
PLAN_SHA256 = {"v1": "dfefdd96c0f4c6596f964189a996dd719e35f7dde941fabc61f15cf0e743c8ff",
               "v2": "3a6f80e1dfe4e4f5cf61d2b56aef4253e2039939a76231d344e17a4d351d97ba",
               "shipped": "17b054496a612df9b152a16309e7e21237ba86d27f566e7d1980bdde945d063a"}


def find_summary(plan, run):
    for d in SCORE_DIRS[plan]:
        p = d / f"score_{run}.json"
        if p.exists():
            doc = json.loads(p.read_text(encoding="utf-8"))
            if doc.get("hval_plan_sha256") != PLAN_SHA256[plan]:
                continue
            s = doc["summary"]["rubi"]
            return {"summary": {k: s[k] for k in SUMMARY_KEYS if k in s}, "blob_sha256": doc.get("blob_sha256"),
                    "hval_plan_sha256": doc.get("hval_plan_sha256"), "dev_plan_sha256": doc.get("dev_plan_sha256"),
                    "cascade": doc.get("cascade")}
    raise SystemExit(f"no score file for {run} on plan {plan}")


def records(out):
    dec, runs = {}, {}
    for name, (stage, arm, plan, what) in DECISIONS.items():
        d = json.loads((A / "vocr6-lead/decisions" / f"{name}.json").read_text(encoding="utf-8"))
        tot = {k: {"delta_points": d["total"][k]["delta_points"], "ci95": d["total"][k]["ci95"],
                   **({"cer_x": d["total"][k]["cer_x"], "cer_y": d["total"][k]["cer_y"]} if "cer_x" in d["total"][k]
                      else {})}
               for k in COMPONENTS if k in d["total"]}
        dec[name] = {"stage": stage, "arm": arm, "plan": plan, "what": what, "x": d["x"], "y": d["y"], "total": tot,
                     "pairs": d["pairs"], "checks": d.get("checks"), "adopt_y": d["adopt_y"]}
        for r in d["x"] + d["y"]:
            runs.setdefault(plan, {})[r] = find_summary(plan, r)
    for plan, names in EXTRA_RUNS.items():
        for r in names:
            runs.setdefault(plan, {})[r] = find_summary(plan, r)
    shipped = {}
    for t, p in SHIPPED.items():
        shipped[t] = {"run": p.name.replace("_w8.vocr", ""), "bytes": p.stat().st_size, "sha256": sha256(p)}
    shipped_v040 = {}
    for t, p in SHIPPED_V040.items():
        shipped_v040[t] = {"run": p.name.replace("_w8.vocr", ""), "bytes": p.stat().st_size, "sha256": sha256(p)}
    gate = None
    if GATE_RESULT.exists():
        g = json.loads(GATE_RESULT.read_text(encoding="utf-8"))
        gate = {"protocol_sha256": g["protocol_sha256"], "repo_head": g["repo_head"], "tiers": {}}
        for t, s in g["tiers"].items():
            if s["input"]["sha256"] != shipped[t]["sha256"]:
                raise SystemExit(f"the gate result's {t} input is not the follow-up's shipped plain blob")
            fin, today = s["estimate"]["final"], s["estimate"]["today"]
            gate["tiers"][t] = {"gate": s["gate_adopted"], "two_level": s["two_level_adopted"],
                                "final_arm": s["final_arm"], "blob": {k: s["final_blob"][k] for k in
                                                                      ("name", "sha256", "bytes")},
                                "arena_w8i_1024": fin["arena_w8i_1024"],
                                "est_box_ms": {"plain_333": today["pld0"]["nominal_box_ms_333"],
                                               "plain_444": today["pld0"]["nominal_box_ms_444"],
                                               "shipped_333": fin["pld0"]["nominal_box_ms_333"],
                                               "shipped_444": fin["pld0"]["nominal_box_ms_444"]}}
    doc = {"kind": "vocr6-results", "schema": 1, "issue": 125, "protocol": "eval/vocr6_protocol.json",
           "metric": "dialogue CER (%), Rubi's per-line decode; 'shipped' = Rubi's shipped path (plan v3 + "
                     "screening LM for hval, dev plan v2 + shipped 1 MB LM, alpha 600 / beta 2500 / BOS, cascade into "
                     "the same seed's next tier); v1 / v2 = the screening plans, no cascade",
           "intervals": "95% cluster bootstrap, 10,000 resamples: titles (hval), VNs (dev)",
           "decisions": dec, "runs": runs, "shipped_blobs": shipped, "shipped_blobs_v0.4.0": shipped_v040,
           "follow_up_gate": gate,
           "measured_on": "the host (W8I int8 blobs through the C runtime); nothing ran on a console"}
    Path(out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("records:", len(dec), "decisions,", sum(len(v) for v in runs.values()), "run summaries ->", out)


def iv(t, k):
    x = t[k]
    return f"{x['delta_points']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"


def stage4_rows(doc):
    """Table 1 rows of docs/vocr6.md's stage 4 section: (tier, model, values...)."""
    R = doc["runs"]["shipped"]
    out = []
    for t in ("S", "N", "L"):
        d = doc["decisions"][f"s4_{t}"]
        rows = [(f"VOCR5-{t} (released)", R[d["x"][0]]["summary"])] + [(y, R[y]["summary"]) for y in d["y"]]
        for label, s in rows:
            out.append((t, label, s))
        m = {k: sum(R[y]["summary"][k] for y in d["y"]) / len(d["y"]) for k in SUMMARY_KEYS}
        out.append((t, "**R3 mean of 3**", m))
    return out


def tables(doc):
    cols = [("hval_page", "hval page"), ("hval_clean_dlg", "hval clean"), ("dev_score", "dev score"), ("wd", "dev wd"),
            ("tv", "tv"), ("dn", "dn")]
    md = ["| tier | model | " + " | ".join(c for _, c in cols) + " |", "|---|---|" + "---:|" * len(cols)]
    for t, label, s in stage4_rows(doc):
        b = "**" if label.startswith("**") else ""
        md.append(f"| {t} | {label} | " + " | ".join(f"{b}{s[k]:.3f}{b}" for k, _ in cols) + " |")
    md += ["", "| tier | Δ hval page (R3 − VOCR5) | Δ hval clean | Δ dev score | Δ dev wd | seed pairs (hval page) | "
           "adopted |", "|---|---|---|---|---|---|---|"]
    for t in ("S", "N", "L"):
        d = doc["decisions"][f"s4_{t}"]
        tt = d["total"]
        pairs = ", ".join(f"{v['hval_page']:+.3f}" for v in d["pairs"].values())
        md.append(f"| {t} | {iv(tt, 'hval_page')} | {iv(tt, 'hval_clean')} | {iv(tt, 'dev')} | {iv(tt, 'wd')} | "
                  f"{pairs} | {'**yes**' if d['adopt_y'] else 'no'} |")
    md += ["", "| tier | 4b (hval folded in) vs its 4a sibling | Δ dev score | Δ dev wd | "
           "hval page, 4b (in training) | check |", "|---|---|---|---|---:|---|"]
    R = doc["runs"]["shipped"]
    for t in ("N", "L"):
        d = doc["decisions"][f"s4_H{t}"]
        ok = d["total"]["dev"]["ci95"][0] <= 0
        md.append(f"| {t} | {d['y'][0]} vs {d['x'][0]} | {iv(d['total'], 'dev')} | {iv(d['total'], 'wd')} | "
                  f"{R[d['y'][0]]['summary']['hval_page']:.3f} | {'passes' if ok else 'FAILS: the 4a sibling ships'} |")
    return "\n".join(md) + "\n"


def follow_up_tables(doc):
    """docs/vocr6.md's stage 4c / 4d tables (the follow-up, v0.4.1-vocr6.2), from the records."""
    D, R = doc["decisions"], doc["runs"]["shipped"]
    md = ["| tier | 4c (45,000 steps, hval in) vs the shipped 4b | Δ dev score | Δ tv | Δ dn | Δ dev wd | "
          "Δ hval page (both in training) | rule (ii) |", "|---|---|---|---|---|---|---|---|"]
    for t, what in (("N", "v6f4c_N_seed1 (into 4c-L) vs v6f4h_N_seed1 (into 4b-L)"),
                    ("L", "v6f4c_L_seed1 vs v6f4h_L_seed1")):
        d = D[f"s4c_{t}"]
        ok = not d["total"]["dev"]["ci95"][0] > 0
        md.append(f"| {t} | {what} | {iv(d['total'], 'dev')} | {iv(d['total'], 'tv')} | {iv(d['total'], 'dn')} | "
                  f"{iv(d['total'], 'wd')} | {iv(d['total'], 'hval_page')} | {'passes' if ok else 'fails'} |")
    cols = [("hval_page", "hval page"), ("hval_clean_dlg", "hval clean"), ("dev_score", "dev score"), ("wd", "dev wd"),
            ("tv", "tv"), ("dn", "dn")]
    md += ["", "| arm | model | " + " | ".join(c for _, c in cols) + " |", "|---|---|" + "---:|" * len(cols)]
    md.append("| - | VOCR5-S (released) | " + " | ".join(f"{R['VOCR5-S']['summary'][k]:.3f}" for k, _ in cols) + " |")
    for name in ("s4d_r", "s4d_l"):
        d = D[name]
        for y in d["y"]:
            md.append(f"| {d['arm']} | {y} | " + " | ".join(f"{R[y]['summary'][k]:.3f}" for k, _ in cols) + " |")
        m = {k: sum(R[y]["summary"][k] for y in d["y"]) / len(d["y"]) for k, _ in cols}
        md.append(f"| {d['arm']} | **mean of 3** | " + " | ".join(f"**{m[k]:.3f}**" for k, _ in cols) + " |")
    md += ["", "| arm | Δ hval page (4d − VOCR5-S) | Δ hval clean | Δ dev score | Δ dev wd | seed pairs (hval page) | "
           "seed pairs (dev) | adopted |", "|---|---|---|---|---|---|---|---|"]
    for name in ("s4d_r", "s4d_l"):
        d = D[name]
        tt = d["total"]
        hp = ", ".join(f"{v['hval_page']:+.3f}" for v in d["pairs"].values())
        dv = ", ".join(f"{v['dev']:+.3f}" for v in d["pairs"].values())
        md.append(f"| {d['arm']} | {iv(tt, 'hval_page')} | {iv(tt, 'hval_clean')} | {iv(tt, 'dev')} | {iv(tt, 'wd')} "
                  f"| {hp} | {dv} | {'**yes**' if d['adopt_y'] else 'no'} |")
    md += ["", "| check | compared | Δ dev score | Δ tv | Δ dn | Δ dev wd | Δ hval page | result |",
           "|---|---|---|---|---|---|---|---|"]
    d = D["s4d_lh"]
    ok = not d["total"]["dev"]["ci95"][0] > 0
    md.append(f"| 4d-l's fold-in vs its 4a sibling | v6f4dlh_S_seed1 (into 4b-N) vs v6f4dl_S_seed1 (into "
              f"v6s2cil_N_seed1) | {iv(d['total'], 'dev')} | {iv(d['total'], 'tv')} | {iv(d['total'], 'dn')} | "
              f"{iv(d['total'], 'wd')} | {iv(d['total'], 'hval_page')} (hval in the fold-in's training) | "
              f"{'passes' if ok else 'fails'} |")
    d = D["s4d_report_SvsV5S_up4cN"]
    md.append(f"| Small in the follow-up's cascade | v6f4dlh_S_seed1 vs VOCR5-S, both into 4c-N | "
              f"{iv(d['total'], 'dev')} | {iv(d['total'], 'tv')} | {iv(d['total'], 'dn')} | {iv(d['total'], 'wd')} | "
              f"{iv(d['total'], 'hval_page')} (hval in the new Small's training) | report only |")
    ds = lambda r: f"{R[r]['summary']['dev_score']:.3f}"  # noqa: E731
    md += ["", "| tier | VOCR5 (released) | v0.4.0-vocr6.1 | v0.4.1-vocr6.2 (the follow-up) |", "|---|---|---|---|",
           f"| Small | VOCR5-S into VOCR5-N: {ds('VOCR5-S')} | VOCR5-S into 4b-N: {ds('VOCR5-S_up6N')} | "
           f"v6f4dlh_S_seed1 into 4c-N: **{ds('v6f4dlh_S_seed1_up4cN')}** (VOCR5-S into 4c-N: "
           f"{ds('VOCR5-S_up4cN')}) |",
           f"| Normal | VOCR5-N into VOCR5-L: {ds(D['s4_N']['x'][0])} | 4b-N into 4b-L: {ds('v6f4h_N_seed1')} | "
           f"4c-N into 4c-L: **{ds('v6f4c_N_seed1')}** |",
           f"| Large | VOCR5-L: {ds(D['s4_L']['x'][0])} | 4b-L: {ds('v6f4h_L_seed1')} | "
           f"4c-L: **{ds('v6f4c_L_seed1')}** |"]
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("records")
    r.add_argument("--out", default=str(RECORDS))
    t = sub.add_parser("tables")
    t.add_argument("--records", default=str(RECORDS))
    a = ap.parse_args()
    if a.cmd == "records":
        records(a.out)
    else:
        doc = json.loads(Path(a.records).read_text(encoding="utf-8"))
        print(tables(doc) + "\n" + follow_up_tables(doc), end="")


if __name__ == "__main__":
    main()
