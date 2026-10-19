"""Tests of eval/textspot_tables.py (the markdown tables for docs/textspot.md) on fake units: the pinned
eval/textspot.py baseline and select commands write the records from them, then the tables are rendered from those
records. Checks the rule's size order, the winner's marks, the report recognizers' units being of the same boxes,
the per-part rows and the warning when select had no refinement arena. No screenshots, models or network.
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "synth", ROOT / "train", ROOT / "eval", ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot as TS  # noqa: E402
import textspot_metrics as M  # noqa: E402
import textspot_tables as TT  # noqa: E402

import textdet_model as TM  # noqa: E402

MAN = "m" * 64
PARTS = ("vita", "psp", "pc")


def units(err, tp, fp, fn, chars=40, h_in=None, read=True, n_vns=10):
    zero = dict.fromkeys(M.CLASSES, 0)
    out = []
    for v in range(n_vns):
        for part in PARTS:
            u = {"unit": f"{part}{v}", "vn": f"v{v}", "part": part, "text_free": False, "boxes": tp + fp,
                 "vertical": 0, "truncated": 0, "left_out": 0, "tp": tp, "fp": fp, "fn": fn,
                 "tp_cls": dict(zero, dialogue=tp), "fn_cls": dict(zero, dialogue=fn), "pairs": [[0.8, 0.95]] * tp,
                 "e2e": {"all": [err, chars], "dialogue": [err // 2, chars]} if read else None,
                 "junk": 0 if read else None,
                 "pick": {"dialogue": True, "chars": chars, "iou": 0.6, "hit": err < 8,
                          "errors": err if read else None},
                 "c_equal": True, "h_in": h_in}
            out.append(u)
    for t in range(2):
        out.append({"unit": f"tf{t}", "vn": f"v{t}", "part": "text-free", "text_free": True, "boxes": t,
                    "vertical": 0, "truncated": 0, "left_out": 0, "tp": 0, "fp": t, "fn": 0, "tp_cls": dict(zero),
                    "fn_cls": dict(zero), "pairs": [], "e2e": {"all": [2 * t, 0], "dialogue": [0, 0]} if read else None,
                    "junk": 2 * t if read else None, "pick": {"dialogue": False}, "c_equal": True, "h_in": h_in})
    return out


def write_arm(d, name, meta, models, err, tp, fp, fn, h_in=None):
    a = d / name
    a.mkdir(parents=True)
    us = units(err, tp, fp, fn, h_in=h_in, read=bool(models))
    frames = {u["unit"]: {"boxes": [], "blocks": [], "pick": -1, "truncated": 0, "h_in": h_in, "c_equal": True,
                          "part": u["part"]} for u in us}
    (a / "boxes.json").write_text(json.dumps({"arm": name, "spec_meta": meta, "frames": frames,
                                              "manifest": {"sha256": MAN}}))
    bsha = hashlib.sha256((a / "boxes.json").read_bytes()).hexdigest()
    for k, m in enumerate(models):
        um = units(err + k, tp, fp, fn, h_in=h_in)
        meta_u = {"manifest": {"set": "dev", "sha256": MAN, "frames": len(um)}, "boxes_sha256": bsha,
                  "recognizers_sha256": {m: "r" * 64}, "reader": None}
        (a / f"units_{m}.json").write_text(json.dumps({"meta": meta_u, "units": um, "summary": TS.summarize(um)}))
    if not models:
        (a / "units_detection.json").write_text(json.dumps({"units": us, "summary": TS.summarize(us)}))


def cmeta(kind, shape, half, nbytes, name):
    return {"kind": kind, "decode": {"det": "det", "detref": "det+refine"}[kind], "shape": shape, "half_above": half,
            "blob_bytes": nbytes, "blob_sha256": hashlib.sha256(name.encode()).hexdigest(), "family": "A",
            "run": name, "spec": name}


def a9_doc(refine=True):
    doc = {"summary": [
        {"shape": "s8_c8", "scale": "h2", "half_above": 600, "vtsp_bytes": 1916, "ms_333_960": 31.3,
         "arena_up_to_960": 28551, "macs_960": TM.cost(TM.SHAPES["s8_c8"], 480, 272)["macs"]},
        {"shape": "s8_c24", "scale": "h2", "half_above": 600, "vtsp_bytes": 5732, "ms_333_960": 60.9,
         "arena_up_to_960": 63103, "macs_960": TM.cost(TM.SHAPES["s8_c24"], 480, 272)["macs"]},
        {"shape": "s8_c24", "scale": "f1", "half_above": 0, "vtsp_bytes": 5732, "ms_333_960": 195.0,
         "arena_up_to_960": 93503, "macs_960": TM.cost(TM.SHAPES["s8_c24"], 960, 544)["macs"]}]}
    if refine:
        doc["refine_arena_960x544"] = 72391
    return doc


@pytest.fixture()
def records(tmp_path):
    bdir, cdir = tmp_path / "base", tmp_path / "cand"
    full = list(TS.H_MODELS)
    for k, b in enumerate(TS.BASELINES):
        write_arm(bdir, b, {"kind": "baseline"}, full, 10 + k % 3, 5, 5, 5)
    write_arm(cdir, "small_det", cmeta("det", "s8_c8", 600, 1916, "small"), full, 4, 8, 2, 2)
    write_arm(cdir, "small_detref", cmeta("detref", "s8_c8", 600, 1916, "small"), full, 4, 8, 2, 2, h_in=7)
    write_arm(cdir, "big_det", cmeta("det", "s8_c24", 600, 5732, "big"), full, 1, 9, 1, 1)
    write_arm(cdir, "slow_det", cmeta("det", "s8_c24", 0, 5732, "slow"), [], 0, 9, 1, 1)
    (tmp_path / "a9.json").write_text(json.dumps(a9_doc()))
    TS.main(["baseline", "--units", str(bdir), "--out", str(tmp_path / "base.json")])
    TS.main(["select", "--baseline", str(tmp_path / "base.json"), "--units", str(cdir), "--a9",
             str(tmp_path / "a9.json"), "--out", str(tmp_path / "sel.json")])
    base = json.loads((tmp_path / "base.json").read_text())
    sel = json.loads((tmp_path / "sel.json").read_text())
    return tmp_path, base, sel, cdir


def blocks(text):
    """The markdown tables of a text: lists of their row lines."""
    out, cur = [], []
    for line in text.splitlines():
        if line.startswith("|"):
            cur.append(line)
        elif cur:
            out.append(cur)
            cur = []
    return out + ([cur] if cur else [])


def test_tables_from_the_pinned_records(records):
    tmp, base, sel, cdir = records
    rep = TT.report_units([cdir], sel)
    text = TT.render(base, sel, rep)
    tabs = blocks(text)
    assert len(tabs) == 5
    for t in tabs:                                        # every row has its header's columns
        assert len({row.count("|") for row in t}) == 1, t[0]
    cand = [r.split("|")[1].strip().strip("`") for r in tabs[1][2:]]
    assert cand == ["small_det", "small_detref", "big_det", "slow_det"]           # the rule's size order
    row = {r.split("|")[1].strip().strip("`"): [c.strip() for c in r.split("|")[1:-1]] for r in tabs[1][2:]}
    assert row["big_det"][-1] == "winner" and row["small_det"][-1] == "yes" and row["slow_det"][-2] == "no"
    assert row["small_det"][5] == f"{31.3:.1f} / {31.3 * 333 / 444:.1f}"
    assert row["small_detref"][4] == "72,391" and row["small_detref"][5] == f"{31.3 + 9.8:.1f} / {41.1 * 0.75:.1f}"
    s, ll = (rep["small_det"][m]["cer_all"]["cer"] for m in ("VOCR5-S", "VOCR5-L"))
    assert row["small_det"][12] == f"{s:.2f} / {ll:.2f}"
    assert row["slow_det"][9] != "–" and row["slow_det"][10] == "–"               # detection only: no CER
    assert "winner: big_det" in text and "WARNING" not in text
    assert tabs[2][0].startswith("| `big_det` minus") and len(tabs[2]) == 2 + len(TS.BASELINES)
    assert all(r.rstrip(" |").endswith("yes") for r in tabs[2][2:])
    parts = [r.split("|")[1].strip() for r in tabs[3][2:]]
    assert parts == ["pc", "psp", "vita"] and "Text-free frames (2)" in text
    assert tabs[4][2].startswith("| `big_det` |") and len(tabs[4]) == 3 + len(TS.BASELINES)
    bs = {a: base["arms"][a]["VOCR5-N"]["summary"] for a in TS.BASELINES}
    f1 = max(TS.BASELINES, key=lambda a: (bs[a]["detection"]["f1"], a))
    ce = min(TS.BASELINES, key=lambda a: (bs[a]["cer_all"]["cer"], a))
    assert TT.default_part_baselines(base) == [f1, ce] and tabs[3][0].count(" F1 |") == 3


def test_report_units_must_score_the_same_boxes(records):
    tmp, base, sel, cdir = records
    f = cdir / "big_det" / "units_VOCR5-S.json"
    orig = f.read_text()
    doc = json.loads(orig)
    doc["meta"]["boxes_sha256"] = "0" * 64
    f.write_text(json.dumps(doc))
    with pytest.raises(SystemExit):                       # S units of other boxes
        TT.report_units([cdir], sel)
    f.write_text(orig)
    assert set(TT.report_units([cdir], sel)) == {"small_det", "small_detref", "big_det"}
    n = cdir / "small_det" / "units_VOCR5-N.json"
    other = json.loads(n.read_text())
    other["summary"]["boxes"] += 1
    n.write_text(json.dumps(other))
    with pytest.raises(SystemExit):                       # N units the selection record did not score
        TT.report_units([cdir], sel)


def test_warning_without_the_refinement_arena(records):
    tmp, base, _, cdir = records
    (tmp / "a9_norefine.json").write_text(json.dumps(a9_doc(refine=False)))
    TS.main(["select", "--baseline", str(tmp / "base.json"), "--units", str(cdir), "--a9",
             str(tmp / "a9_norefine.json"), "--out", str(tmp / "sel2.json")])
    sel = json.loads((tmp / "sel2.json").read_text())
    assert not sel["candidates"]["small_detref"]["eligible"]
    text = TT.render(base, sel, TT.report_units([cdir], sel))
    assert "WARNING" in text


def test_baselines_only_and_cli(records, capsys):
    tmp, base, sel, cdir = records
    TT.main(["--baseline", str(tmp / "base.json"), "--out", str(tmp / "b.md")])
    text = (tmp / "b.md").read_text()
    assert len(blocks(text)) == 2 and "Candidates" not in text
    TT.main(["--baseline", str(tmp / "base.json"), "--dev", str(tmp / "sel.json"), "--units", str(cdir),
             "--eligible-only", "--out", str(tmp / "d.md")])
    text = (tmp / "d.md").read_text()
    assert not any("`slow_det`" in r for r in blocks(text)[1]) and "3 candidates, 3 eligible" in text
