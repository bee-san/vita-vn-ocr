"""Tests of the full-frame text spotter's evaluation harness (eval/textspot.py) and of its protected read
(eval/textspot_protected.py): arm construction on synthetic frames, the union rule of grad_refine_columns,
det+refine keeping vertical boxes, the reading cache's key, the unit and summary schema, select's eligibility inputs
on fake tables (through textspot_metrics.apply_rule), and the protected script's attempt record refusing a second
run. Synthetic frames and random int8 networks only: no screenshots, models or network.
"""
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "synth", ROOT / "train", ROOT / "eval", ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot as TS  # noqa: E402
import textspot_metrics as M  # noqa: E402
import textspot_protected as TP  # noqa: E402

import lines_refine as LR  # noqa: E402
import textdet_export as E  # noqa: E402
import textdet_model as TM  # noqa: E402
import textdet_twin as TW  # noqa: E402
import textspot_twin as S  # noqa: E402
from common import find_lines_ex  # noqa: E402

_BLOBS = {}


def vtdt(shape="s8_c8", seed=1):
    if (shape, seed) not in _BLOBS:
        _BLOBS[shape, seed] = E.pack(*E.random_int_layers(shape, seed))
    return _BLOBS[shape, seed]


def synthetic_frame(w=960, h=544, seed=0, columns=True):
    """A dark frame with a light dialogue box of stroke rows at the bottom, a UI label top right and (optionally)
    two vertical columns of strokes on the left."""
    rng = np.random.default_rng(seed)
    g = np.full((h, w), 30, np.uint8)
    g[h - 150:h - 10, 20:w - 20] = 200
    for k in range(3):
        y0 = h - 140 + 40 * k
        for x0 in range(40, w - 120, 24):
            if rng.random() < 0.85:
                g[y0:y0 + 22, x0:x0 + 16:3] = 20
                g[y0 + 10:y0 + 12, x0:x0 + 16] = 20
    for x0 in range(w - 200, w - 40, 22):
        g[16:36, x0:x0 + 15:2] = 235
    if columns:
        for cx in (60, 110):
            for y0 in range(40, 300, 26):
                g[y0:y0 + 18, cx:cx + 20:4] = 240
                g[y0 + 8:y0 + 10, cx:cx + 20] = 240
    return g


def frame_of(g, fid="f1", vn="v900001", part="vita", refs=None, text_free=False):
    rgb = np.repeat(g[..., None], 3, 2)
    return TS.Frame({"id": fid, "vn": vn, "part": part, "refs": refs or [], "text_free": text_free}, g,
                    TS.pixel_sha256(rgb))


@pytest.fixture(scope="module")
def nets():
    return {"s8c24": TW.Model(vtdt("s8_c8", 3)), "s8c24_spot": S.Model(S.pack(vtdt("s8_c8", 3), 0, 1)),
            "gate": TW.Model(vtdt("s8_c8", 4))}


def check_arm(a, w, h):
    assert set(a) >= {"boxes", "blocks", "pick", "truncated"}
    for b in a["boxes"]:
        x, y, bw, bh, v, conf, dark, block = b
        assert 0 <= x and 0 <= y and bw > 0 and bh > 0 and x + bw <= w and y + bh <= h
        assert v in (0, 1) and dark in (0, 1) and 0 <= block < len(a["blocks"])
    assert sum(bk[6] for bk in a["blocks"]) == len(a["boxes"])
    assert (a["pick"] == -1) == (not a["boxes"])


# ---------------------------------------------------------------- arms

def test_every_baseline_on_a_synthetic_frame(nets):
    g = synthetic_frame()
    f = frame_of(g)
    tw = TS.Twin(f, nets, (4, -8))
    arms = {b: tw.arm(TS.Spec(b, "baseline", base=b)) for b in TS.BASELINES}
    for a in arms.values():
        check_arm(a, 960, 544)
    gb, gd, gt = find_lines_ex(g, "grad")
    assert sorted(tuple(b[:4]) for b in arms["grad"]["boxes"]) == sorted(tuple(map(int, b)) for b in gb)
    assert all(b[4] == 0 for b in arms["grad"]["boxes"]) and gb
    rb, rd, _ = LR.refine_lines(g, gb, gd, LR.DEFAULT)
    assert sorted(tuple(b[:4]) for b in arms["grad_refine"]["boxes"]) == sorted(tuple(map(int, b)) for b in rb)
    lines, tr = TW.find_lines(nets["s8c24"], g)
    assert sorted(b[:5] for b in arms["s8c24_textdet"]["boxes"]) == sorted((x, y, w, h, v) for x, y, w, h, _, v in
                                                                          lines)
    r = S.spot(nets["s8c24_spot"], g)
    assert arms["s8c24_spot"]["boxes"] == [(b.x, b.y, b.w, b.h, b.vertical, b.conf, b.dark, b.block) for b in r.boxes]
    assert arms["s8c24_spot"]["pick"] == r.pick
    assert all(b[4] == 1 for b in arms["columns"]["boxes"])
    # every arm's boxes are textspot_twin.group_and_order's: the same blocks and pick for the same boxes
    for name in ("grad", "grad_refine_columns", "s8c24_textdet_refine"):
        a = arms[name]
        again = S.group_and_order([b[:7] for b in a["boxes"]], 544)
        assert (again[0], again[1], again[2]) == (a["boxes"], a["blocks"], a["pick"])


def test_gate_arm_follows_the_gate(nets):
    g = synthetic_frame(seed=1)
    f = frame_of(g)
    spec = TS.Spec("grad_refine_gate", "baseline", base="grad_refine_gate")
    open_ = TS.Twin(f, nets, (4, -129)).arm(spec)
    shut = TS.Twin(f, nets, (4, 128)).arm(spec)
    ref = TS.Twin(f, nets, (4, -8)).arm(TS.Spec("grad_refine", "baseline", base="grad_refine"))
    assert open_["boxes"] == ref["boxes"] and open_["boxes"]
    assert shut["boxes"] == [] and shut["pick"] == -1 and shut["truncated"] == 0


def test_union_rule_of_grad_refine_columns():
    h = [(100, 100, 400, 30, 0, 0, 0)]
    inside = (110, 90, 30, 50, 1, 0, 0)            # 900 of its 1,500 px inside the box: not added
    half = (110, 115, 30, 30, 1, 0, 0)             # 450 of 900: exactly half is not "less than half": not added
    outside = (600, 50, 30, 200, 1, 0, 0)          # nothing inside: added
    split = (480, 95, 40, 40, 1, 0, 0)             # 600 of 1,600 inside: added
    assert TS.union_columns(h, [inside, half, outside, split]) == h + [outside, split]
    two = [(0, 0, 50, 100, 0, 0, 0), (50, 0, 50, 100, 0, 0, 0)]
    assert TS.union_columns(two, [(25, 0, 50, 100, 1, 0, 0)]) == two      # half in each box: not added
    assert TS.union_columns(two, [(10, 0, 90, 100, 1, 0, 0)]) == two      # 5,000 of 9,000 in the second box
    wide = [(0, 0, 50, 100, 0, 0, 0), (60, 0, 50, 100, 0, 0, 0)]
    gap = (30, 0, 50, 100, 1, 0, 0)                # 2,000 of 5,000 in each box (80% in their union): added
    assert TS.union_columns(wide, [gap]) == wide + [gap]


def test_det_refine_keeps_vertical_boxes():
    g = synthetic_frame(seed=2)
    boxes = [(30, 384, 900, 40, 0, 90, 0, 0), (30, 424, 900, 40, 0, 80, 0, 0), (50, 30, 40, 290, 1, 70, 0, 1),
             (100, 30, 40, 290, 1, 60, 1, 1)]
    out, n = TS.refine_spot_boxes(boxes, g)
    assert n == 2
    rb, rd, org = LR.refine_lines(g, [b[:4] for b in boxes[:2]], [b[6] for b in boxes[:2]], LR.DEFAULT)
    assert [o[:4] for o in out if not o[4]] == [tuple(map(int, b)) for b in rb]
    assert [o[5] for o in out if not o[4]] == [boxes[i][5] for i in org]          # a refined box keeps its conf
    assert [o for o in out if o[4]] == [b[:7] for b in boxes[2:]]                 # vertical boxes unchanged


def test_detref_arm_and_family_specs(tmp_path, nets):
    run = tmp_path / "sweep" / "s8_c8_seed9"
    run.mkdir(parents=True)
    (run / "final.vtd").write_bytes(vtdt("s8_c8", 5))
    specs = TS.parse_arms("A:s8_c8_seed9@600,famA", fam_a={"s8_c8_seed9": run / "final.vtd"})
    assert [s.name for s in specs] == ["A_s8_c8_seed9_h600_det", "A_s8_c8_seed9_h600_detref",
                                       "A_s8_c8_seed9_h0_det", "A_s8_c8_seed9_h0_detref"]
    meta = specs[0].meta()
    assert (meta["half_above"], meta["blob_bytes"], meta["shape"]) == (600, len(vtdt("s8_c8", 5)) + 32, "s8_c8")
    g = synthetic_frame(seed=3)
    f = frame_of(g)
    allnets = dict(nets, **TS.load_nets(specs))
    tw = TS.Twin(f, allnets)
    det, ref = tw.arm(specs[2]), tw.arm(specs[3])
    check_arm(det, 960, 544)
    check_arm(ref, 960, 544)
    assert sorted(b[:7] for b in ref["boxes"] if b[4]) == sorted(b[:7] for b in det["boxes"] if b[4])
    assert ref["h_in"] == sum(1 for b in det["boxes"] if not b[4])


# ---------------------------------------------------------------- reading cache

def test_cache_key_and_persistence(tmp_path):
    k = TS.Cache.key("ab" * 32, (1, 2, 3, 4, 0, 9, 1), 1, 0, "cd" * 32)
    assert k == f"{'ab' * 32}|1,2,3,4|1|0|{'cd' * 32}"
    assert len({TS.Cache.key("f", (1, 2, 3, 4), d, v, r) for d in (0, 1) for v in (0, 1) for r in ("a", "b")}) == 8
    c = TS.Cache(tmp_path / "cache")
    c.put(k, "テキスト")
    c.close()
    (tmp_path / "cache" / "torn.jsonl").write_text('{"k": "x", "t": "y"}\n{"k": "z", "t"', encoding="utf-8")
    c2 = TS.Cache(tmp_path / "cache")
    assert c2.get(k) == "テキスト" and c2.get("x") == "y" and c2.get("z") is None


class FakeReader:
    """Reads a crop as one character per 16 px of its width (a vertical crop arrives turned: width = height)."""

    def __init__(self, path):
        self.path = path

    def read(self, g, invert):
        return "あ" * max(1, g.shape[1] // 16)


def fake_readers(tmp_path, models=("VOCR5-N", "VOCR5-S", "VOCR5-L", "VOCR5-V")):
    paths = {}
    for m in models:
        p = tmp_path / "rec" / f"{m}.bin"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(m.encode())
        paths[m] = p
    return TS.Readers(paths, FakeReader)


def test_reading_shares_boxes_between_arms(tmp_path):
    g = synthetic_frame(seed=4)
    f = frame_of(g)
    box = (40, 400, 400, 30, 0, 0, 0, 0)
    arms = [{"a": {"boxes": [box]}, "b": {"boxes": [box, (60, 40, 30, 200, 1, 0, 1, 1)]}}]
    rd = fake_readers(tmp_path)
    cache = TS.Cache(tmp_path / "cache")
    assert TS.read_all([f], arms, ["VOCR5-N"], rd, cache, workers=2) == 2
    assert TS.read_all([f], arms, ["VOCR5-N"], rd, cache, workers=2) == 0
    assert TS.texts_of(f, arms[0]["b"]["boxes"], "VOCR5-N", rd, cache) == ["あ" * 25, "あ" * 12]


# ---------------------------------------------------------------- units and summaries

def ref(box, cls="dialogue", text="あいう", vertical=False, ignore=False):
    return {"box": list(box), "cls": cls, "text": text, "vertical": vertical, "ignore": ignore}


def test_unit_and_summary_schema():
    refs = [ref([100, 400, 300, 24], text="あいうえお"), ref([100, 440, 300, 24], text="かきくけこ"),
            ref([800, 20, 60, 16], cls="ui", text="SAVE")]
    f = frame_of(np.zeros((544, 960), np.uint8), refs=refs)
    boxes = [(96, 396, 308, 32, 0, 50, 0), (96, 436, 308, 32, 0, 50, 0), (500, 100, 40, 40, 0, 9, 0)]
    ordered, blocks, pick = S.group_and_order(boxes, 544)
    arm = {"boxes": ordered, "blocks": blocks, "pick": pick, "truncated": 0, "h_in": None}
    texts = {(96, 396): "あいうえお", (96, 436): "かきくけ", (500, 100): "ゴミ"}
    t = [texts[b[:2]] for b in ordered]
    u = TS.frame_unit(f, arm, t, True)
    assert set(u) >= {"unit", "vn", "part", "tp", "fp", "fn", "tp_cls", "fn_cls", "e2e", "junk", "pick", "c_equal"}
    assert (u["tp"], u["fp"], u["fn"]) == (2, 1, 1)
    assert u["e2e"]["all"] == [1 + 4 + 2, 14] and u["e2e"]["dialogue"] == [1, 10] and u["junk"] == 2
    assert u["pick"]["dialogue"] and u["pick"]["hit"] and u["pick"]["errors"] == 1 and u["pick"]["chars"] == 10
    tf = frame_of(np.zeros((544, 960), np.uint8), fid="f2", vn="v900002", part="text_free", text_free=True)
    u2 = TS.frame_unit(tf, {"boxes": [(5, 5, 20, 20, 0, 1, 0, 0)], "blocks": [(5, 5, 20, 20, 0, 0, 1)], "pick": 0,
                            "truncated": 0}, ["x"], None)
    s = TS.summarize([u, u2])
    assert s["detection"]["tp"] == 2 and s["detection"]["fp"] == 2
    assert s["cer_all"] == {"errors": 8, "chars": 14, "cer": round(100 * 8 / 14, 4)}
    assert s["recall_by_class"]["dialogue"]["recall"] == 1.0 and s["recall_by_class"]["ui"]["recall"] == 0.0
    assert s["text_free"]["frames"] == 1 and s["text_free"]["share_with_any_box"] == 1.0
    assert s["pick"]["frames_with_dialogue"] == 1 and s["pick"]["hit_share"] == 1.0
    assert s["c_equal"] is None and s["c_unchecked_frames"] == 1
    assert set(s["per_part"]) == {"vita", "text_free"} and s["per_part"]["vita"]["frames"] == 1
    assert s["tightness"]["pairs"] == 2
    ra = TS.rule_arm([u, u2])
    assert ra["all"][0] == {"unit": "f1", "vn": "v900001", "errors": 7, "chars": 14}
    det_only = TS.frame_unit(f, arm, None)
    assert det_only["e2e"] is None and det_only["pick"]["errors"] is None and "cer_all" not in TS.summarize([det_only])


# ---------------------------------------------------------------- select on fake tables

def units_for(n_vns, err, tp, fp, fn, chars=40, part="vita", h_in=None):
    out = []
    for v in range(n_vns):
        for k in range(3):
            out.append({"unit": f"u{v}_{k}", "vn": f"v{v}", "part": part, "text_free": False, "boxes": tp + fp,
                        "vertical": 0, "truncated": 0, "left_out": 0, "tp": tp, "fp": fp, "fn": fn,
                        "tp_cls": dict.fromkeys(M.CLASSES, 0), "fn_cls": dict.fromkeys(M.CLASSES, 0), "pairs": [],
                        "e2e": {"all": [err, chars], "dialogue": [err, chars]}, "junk": 0,
                        "pick": {"dialogue": False}, "c_equal": True, "h_in": h_in})
    return out


def write_arm(d, name, meta, units, h_in=None):
    a = d / name
    a.mkdir(parents=True)
    frames = {u["unit"]: {"boxes": [], "blocks": [], "pick": -1, "truncated": 0, "h_in": h_in, "c_equal": True,
                          "part": u["part"]} for u in units}
    (a / "boxes.json").write_text(json.dumps({"arm": name, "spec_meta": meta, "frames": frames,
                                              "manifest": {"sha256": "m" * 64}}))
    doc = {"meta": {"manifest": {"sha256": "m" * 64}}, "units": units, "summary": TS.summarize(units)}
    (a / f"units_{TS.RULE_MODEL}.json").write_text(json.dumps(doc))
    return a / f"units_{TS.RULE_MODEL}.json"


def test_select_eligibility_inputs_and_rule(tmp_path):
    bdir, cdir = tmp_path / "base", tmp_path / "cand"
    base_doc = {"manifest": {"sha256": "m" * 64}, "arms": {}}
    for b in TS.BASELINES:
        f = write_arm(bdir, b, {"kind": "baseline"}, units_for(10, 10, 5, 5, 5))
        base_doc["arms"][b] = {TS.RULE_MODEL: {"units": str(f), "units_sha256": TS.sha256(f)}}

    def meta(kind, shape, half, nbytes, name):
        return {"kind": kind, "decode": {"det": "det", "detref": "det+refine"}[kind], "shape": shape,
                "half_above": half, "blob_bytes": nbytes, "blob_sha256": hashlib.sha256(name.encode()).hexdigest(),
                "family": "A", "run": name, "spec": name}
    write_arm(cdir, "small_det", meta("det", "s8_c8", 600, 1916, "small"), units_for(10, 4, 8, 2, 2))
    write_arm(cdir, "small_detref", meta("detref", "s8_c8", 600, 1916, "small"), units_for(10, 4, 8, 2, 2, h_in=7),
              h_in=7)
    write_arm(cdir, "slow_det", meta("det", "s8_c24", 0, 5732, "slow"), units_for(10, 1, 9, 1, 1))
    write_arm(cdir, "big_det", meta("det", "s8_c24", 600, 5732, "big"), units_for(10, 1, 9, 1, 1))
    a9 = {"refine_arena_960x544": 70000,
          "table": [{"shape": "s8_c8", "half_above": 600, "size": [960, 544], "arena_bytes": 50000,
                     "ms_per_frame": {"333": {"low": 10.0, "high": 20.0}}},
                    {"shape": "s8_c24", "half_above": 0, "size": [960, 544], "arena_bytes": 92639,
                     "ms_per_frame": {"333": {"low": 150.0, "high": 191.0}}},
                    {"shape": "s8_c24", "scale": "h2", "size": [960, 544], "arena_bytes": 40000,
                     "ms_per_frame": {"333": {"high": 48.0, "calibrated": 48.0336}}}]}
    rule = json.loads((ROOT / "eval" / "textspot_protocol.json").read_text())["rule"]
    result, report, extra = TS.select(base_doc, [cdir], a9, rule)
    assert report.pop("_refine_arena") == {"bytes_960x544": 70000, "source": "a9:refine_arena_960x544"}
    sm = report["small_det"]
    assert sm["macs_960"] == TM.cost(TM.SHAPES["s8_c8"], 480, 272)["macs"]
    assert sm["est_ms_333_960"] == round(20.0 * TS.A9_CAL, 3) and sm["arena_960"] == 50000
    sr = report["small_detref"]
    assert sr["median_h_boxes_vita"] == statistics.median([7] * 30)
    assert sr["est_ms_333_960"] == round(round(20.0 * TS.A9_CAL, 3) + 1.4 * 7, 3)
    assert sr["arena_960"] == 70000                                         # the larger of the two arenas
    assert report["slow_det"]["macs_960"] == TM.cost(TM.SHAPES["s8_c24"], 960, 544)["macs"]
    assert not report["slow_det"]["eligible"] and "slow_det" in extra       # 191 ms: not eligible
    assert result["eligible"] == ["small_det", "small_detref", "big_det"]
    assert result["qualifying"] == ["small_det", "small_detref", "big_det"]
    assert result["winner"] == "big_det" and result["steps"][0]["replaces"] == "small_det"
    assert set(result["comparisons"]["small_det"]) == set(TS.BASELINES)
    # the frozen baseline's units are checked against their sha256
    f = bdir / "grad" / f"units_{TS.RULE_MODEL}.json"
    f.write_text(f.read_text() + " ")
    with pytest.raises(SystemExit):
        TS.select(base_doc, [cdir], a9, rule)


def test_pinned_sha_reads_amendments():
    p = {"amendments": [{"n": 1, "manifests": {"eval/manifests/textspot-dev.jsonl": "a" * 64}},
                        {"n": 2, "pins": [{"path": "eval/manifests/textspot-dev.jsonl", "sha256": "b" * 64}]}]}
    assert TS.pinned_sha(p, "eval/manifests/textspot-dev.jsonl") == "b" * 64
    assert TS.pinned_sha(p, "eval/manifests/textspot-test.jsonl") is None
    assert TS.pinned_sha({"amendments": []}, "x") is None
    # eval/textspot_amend.py --pin NAME=PATH: the path as typed (relative, ./ or a clone's absolute path)
    for path in ("eval/manifests/textspot-dev.jsonl", "./eval/manifests/textspot-dev.jsonl",
                 "/somewhere/clones/textspot/eval/manifests/textspot-dev.jsonl"):
        q = {"amendments": [{"id": 2, "pins": {"dev_manifest": {"path": path, "sha256": "c" * 64, "bytes": 1}}}]}
        assert TS.pinned_sha(q, "eval/manifests/textspot-dev.jsonl") == "c" * 64
    bak = {"path": "eval/manifests/textspot-dev.jsonl.bak", "sha256": "d" * 64}
    q = {"amendments": [{"id": 2, "pins": {"x": bak}}]}
    assert TS.pinned_sha(q, "eval/manifests/textspot-dev.jsonl") is None


# ---------------------------------------------------------------- the protected script (pseudo data only)

def write_image(path, g):
    import cv2
    rgb = np.repeat(g[..., None], 3, 2)
    ok, buf = cv2.imencode(".png", rgb[:, :, ::-1])
    assert ok
    Path(path).write_bytes(buf.tobytes())


def fake_test_set(tmp_path):
    shots = tmp_path / "shots"
    shots.mkdir()
    rows = []
    for i, part in enumerate(("vita", "psp", "adrenaline", "vita", "vita", "psp")):
        dims = (480, 272) if part == "psp" else (960, 544)
        g = synthetic_frame(*dims, seed=10 + i, columns=part == "vita")
        vn = f"v99{i % 5:04d}"
        url = f"https://t.vndb.org/sf/0{i}/90{i}.jpg"
        p = shots / TS.shot_file_name(vn, url)
        write_image(p, g)
        rows.append({"id": f"t{i}", "vn": vn, "part": part, "url": url, "sha256": TS.sha256(p),
                     "src_dims": list(dims), "scale": None, "dims": list(dims), "text_free": False,
                     "refs": [ref([40, dims[1] - 140, dims[0] - 160, 22], text="あいうえおかきくけこ")]})
    man = tmp_path / "textspot-test.jsonl"
    man.write_text("".join(json.dumps(r) + "\n" for r in rows))
    proto = json.loads((ROOT / "eval" / "textspot_protocol.json").read_text())
    proto["amendments"] = [{"n": 1, "manifests": {TP.TEST_MANIFEST: TS.sha256(man)}}]
    pp = tmp_path / "protocol.json"
    pp.write_text(json.dumps(proto))
    return man, pp, shots


@pytest.fixture()
def fake_blobs(tmp_path, monkeypatch):
    s8 = tmp_path / "s8c24.vtd"
    s8.write_bytes(vtdt("s8_c8", 3))
    gate = tmp_path / "gate.vtd"
    gate.write_bytes(vtdt("s8_c8", 4))
    monkeypatch.setattr(TS, "S8C24", s8)
    monkeypatch.setattr(TS, "GATE", gate)
    monkeypatch.setattr(TS, "check_baseline_blobs", lambda proto: None)     # random blobs, not the pinned ones
    cand = tmp_path / "cand.vtsp"
    cand.write_bytes(S.pack(vtdt("s8_c8", 6), 600, 1))
    return cand


def args_for(**kw):
    import argparse
    d = {"decode": "", "cbin": "", "models": "N,S,L", "workers": 1, "compiler": "gcc", "data": "", "shots": [],
         "report_parts": list(TP.REPORT_PARTS), "manifest": ""}
    d.update(kw)
    return argparse.Namespace(**d)


def test_protected_tests_once_and_public(tmp_path, fake_blobs):
    man, pp, shots = fake_test_set(tmp_path)
    out = tmp_path / "out"
    args = args_for(cand=f"det:cand={fake_blobs}", out=str(out), cache=str(tmp_path / "pcache"),
                    protocol=str(pp), manifest=str(man), shots=[str(shots)])
    res = TP.cmd_tests(args, readers=fake_readers(tmp_path), log=lambda s: None)
    assert (out / "tests.attempt.json").exists() and (out / "tests.json").exists()
    assert res["report_only_frames"] == 1 and res["scored_frames"] == 5
    assert set(res["arms"]) == {"cand_det", *TS.BASELINES}
    assert set(res["comparisons"]["VOCR5-N"]) == set(TS.BASELINES)
    assert res["arms"]["cand_det"]["VOCR5-N"]["frames"] == 5
    assert "adrenaline" in res["arms"]["cand_det"]["VOCR5-N"]["all_frames_per_part"]
    TP.assert_public(json.loads((out / "tests.json").read_text()))
    with pytest.raises(FileExistsError):                  # the attempt record refuses a second run
        TP.cmd_tests(args, readers=fake_readers(tmp_path), log=lambda s: None)


def test_protected_tests_refuses_an_unpinned_manifest(tmp_path, fake_blobs):
    man, pp, shots = fake_test_set(tmp_path)
    man.write_text(man.read_text() + "\n")
    args = args_for(cand=f"det:cand={fake_blobs}", out=str(tmp_path / "out"), cache=str(tmp_path / "pcache"),
                    protocol=str(pp), manifest=str(man), shots=[str(shots)])
    with pytest.raises(SystemExit):
        TP.cmd_tests(args, readers=fake_readers(tmp_path), log=lambda s: None)
    assert not (tmp_path / "out" / "tests.attempt.json").exists()


def fake_clannad(tmp_path):
    session = tmp_path / "session"
    (session / "frames").mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    roi = [0, 300, 960, 244]
    lines, refs = [], []
    for p in range(3):
        g = synthetic_frame(seed=20 + p, columns=False)
        path = session / "frames" / f"p{p:05d}.png"
        write_image(path, g)
        src = {"path": f"frames/p{p:05d}.png", "sha256": TS.sha256(path)}
        gb = find_lines_ex(g[roi[1]:, :], "grad")[0]
        for i, b in enumerate(gb if p < 2 else gb[:1]):
            lid = f"p{p:05d}-{i}"
            lines.append({"id": lid, "page": f"p{p:05d}", "seq": p, "i": i, "source": src, "roi": roi,
                          "box": list(map(int, b)), "dark": False})
            refs.append({"id": lid, "page": f"p{p:05d}", "status": "ref" if p < 2 else "notext",
                         "text": "あいう" if p < 2 else None, "name": i == 0})
    lp = work / "lines.jsonl"
    lp.write_text("".join(json.dumps(r) + "\n" for r in lines))
    (work / "lines.meta.json").write_text(json.dumps({"session": str(session), "lines_sha256": TS.sha256(lp)}))
    rp = work / "refs.jsonl"
    rp.write_text("".join(json.dumps(r) + "\n" for r in refs))
    (work / "refs.freeze.json").write_text(json.dumps({"refs_sha256": TS.sha256(rp)}))
    return work


def test_protected_clannad_once_and_public(tmp_path, fake_blobs):
    work = fake_clannad(tmp_path)
    out = tmp_path / "cout"
    args = args_for(cand=f"det:cand={fake_blobs}", out=str(out), cache=str(tmp_path / "pcache"),
                    protocol=str(ROOT / "eval" / "textspot_protocol.json"), work=str(work), models="N")
    res = TP.cmd_clannad(args, readers=fake_readers(tmp_path), grad_fn=TP.twin_grad, log=lambda s: None)
    assert res["counts"]["pages"] == 3 and res["counts"]["no_text_pages"] == 1 and res["counts"]["unmatched"] == 0
    assert res["scored_lines"] == res["counts"]["matched"] > 0
    dl = res["scopes"]["dialogue"]
    assert dl["lines"] < res["scopes"]["all"]["lines"]               # speaker-name lines are left out of dialogue
    assert set(dl["pairs"]) == {f"cand_det/VOCR5-N:{b}/VOCR5-N" for b in TS.BASELINES}
    TP.assert_public(json.loads((out / "clannad.json").read_text()))
    with pytest.raises(FileExistsError):
        TP.cmd_clannad(args, readers=fake_readers(tmp_path), grad_fn=TP.twin_grad, log=lambda s: None)


def test_run_refuses_protected_and_unpinned_sets(tmp_path):
    import argparse
    proto = {"amendments": []}
    with pytest.raises(SystemExit):
        TS.load_set(argparse.Namespace(set="test", manifest="", limit=0), proto)
    m = tmp_path / "dev.jsonl"
    m.write_text(json.dumps({"id": "a", "vn": "v900001"}) + "\n")
    with pytest.raises(SystemExit):                       # dev: not pinned by an amendment
        TS.load_set(argparse.Namespace(set="dev", manifest=str(m), limit=0), proto)
    pinned = {"amendments": [{"manifests": {"eval/manifests/textspot-dev.jsonl": TS.sha256(m)}}]}
    rows, rec = TS.load_set(argparse.Namespace(set="dev", manifest=str(m), limit=0), pinned)
    assert rec["sha256"] == TS.sha256(m) and len(rows) == 1
    from common import test_set_vns
    tv = sorted(test_set_vns())[0]
    m.write_text(json.dumps({"id": "a", "vn": tv}) + "\n")
    with pytest.raises(SystemExit):                       # never a protected test VN
        TS.load_set(argparse.Namespace(set="pseudo", manifest=str(m), limit=0), proto)


def test_scale_frame_steps():
    import cv2
    rgb = np.random.default_rng(0).integers(0, 256, (272, 480, 3)).astype(np.uint8)
    row = {"id": "x", "dims": [960, 544], "scale": {"method": "cv2.INTER_LINEAR", "factor": 2}}
    want = cv2.resize(rgb, (960, 544), interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(TS.scale_frame(row, rgb), want)
    calls = []

    def build(img, dims, scale):                       # the sets' own function, when it exists, does the step
        calls.append((tuple(dims), scale))
        return cv2.resize(img, tuple(dims), interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(TS.scale_frame(row, rgb, build), want) and calls == [((960, 544), row["scale"])]
    assert TS.scale_frame({"id": "n", "dims": [480, 272], "scale": None}, rgb, lambda i, d, s: i) is not None
    row2 = {"id": "y", "dims": [960, 544], "scale": [{"method": "INTER_AREA", "dims": [240, 136]},
                                                     {"method": "INTER_LINEAR", "dims": [960, 544]}]}
    assert TS.scale_frame(row2, rgb).shape == (544, 960, 3)
    with pytest.raises(ValueError):
        TS.scale_frame({"id": "z", "dims": [100, 100], "scale": None}, rgb)


def test_decode_suffix_and_a9_numeric_rows(tmp_path):
    run = tmp_path / "sweep" / "s8_c8_seed9"
    run.mkdir(parents=True)
    (run / "final.vtd").write_bytes(vtdt("s8_c8", 5))
    fam = {"s8_c8_seed9": run / "final.vtd"}
    assert [s.name for s in TS.parse_arms("A:s8_c8_seed9@600:detref", fam_a=fam)] == ["A_s8_c8_seed9_h600_detref"]
    assert [s.name for s in TS.parse_arms("famA:det", fam_a=fam)] == ["A_s8_c8_seed9_h0_det", "A_s8_c8_seed9_h600_det"]
    # eval/textspot_a9_<date>.json's form: ms_per_frame["333"] is already the high bound x 1.0007 at 333 MHz
    rows = TS.a9_rows({"table": [{"shape": "s8_c8", "scale": "h2", "half_above": 600, "size": [960, 544],
                                  "arena_bytes": 28551, "ms_per_frame": {"333": 31.32, "444": 23.49},
                                  "cycles_per_frame": {"333": [1.0, 2.0]}},
                                 {"shape": "s8_c8", "scale": "f1", "size": [960, 544], "arena_bytes": 42391,
                                  "cycles_per_frame": {"333": [5.0e6, 2.95e7]}}]})
    assert TS.a9_lookup(rows, "s8_c8", 600)["ms_333"] == 31.32 and TS.a9_lookup(rows, "s8_c8", 600)["arena"] == 28551
    assert TS.a9_lookup(rows, "s8_c8", 0)["ms_333"] == pytest.approx(2.95e7 * 1.0007 / 333e3)
    assert TS.a9_lookup(rows, "s8_c8", 600, (480, 272)) is None


def test_parity_record_overrides_unchecked_frames():
    doc = {"arm": "x_det", "_sha256": "s" * 64,
           "frames": {"a": {"c_equal": None}, "b": {"c_equal": True}, "c": {"c_equal": None}}}
    rec = {"arms": {"x_det": {"boxes_sha256": "s" * 64, "frames": {"a": True, "b": True, "c": False}}}}
    out = TS.apply_parity(doc, rec)
    assert [out["frames"][k]["c_equal"] for k in "abc"] == [True, True, False]
    assert TS.apply_parity(doc, {"arms": {}}) is doc
    with pytest.raises(SystemExit):
        TS.apply_parity(doc, {"arms": {"x_det": dict(rec["arms"]["x_det"], boxes_sha256="t" * 64)}})


def test_a9_summary_rows_first_and_cross_checks():
    doc = {"summary": [{"shape": "s8_c8", "scale": "h2", "half_above": 600, "vtsp_bytes": 1916, "ms_333_960": 31.3,
                        "arena_up_to_960": 28551, "macs_960": TM.cost(TM.SHAPES["s8_c8"], 480, 272)["macs"]}],
           "table": [{"shape": "s8_c8", "scale": "h2", "half_above": 600, "size": [960, 544], "arena_bytes": 1,
                      "ms_per_frame": {"333": 1.0}}]}
    rows = TS.a9_rows(doc)
    hit = TS.a9_lookup(rows, "s8_c8", 600)
    assert (hit["from"], hit["ms_333"], hit["arena"]) == ("summary", 31.3, 28551)
    meta = {"shape": "s8_c8", "half_above": 600, "blob_bytes": 1916, "blob_sha256": "x" * 64, "decode": "det"}
    bd = {"arm": "a_det", "frames": {"f": {"c_equal": True, "h_in": None, "part": "vita"}}}
    e = TS.eligibility(meta, bd, rows, None)
    assert (e["est_ms_333_960"], e["arena_960"], e["a9_row"]) == (31.3, 28551, "summary")
    bad = TS.a9_rows({"summary": [dict(doc["summary"][0], macs_960=1)]})
    with pytest.raises(SystemExit):                       # a table row of another shape or scale
        TS.eligibility(meta, bd, bad, None)
    with pytest.raises(SystemExit):
        TS.eligibility(dict(meta, blob_bytes=2000), bd, rows, None)


def test_public_guard():
    TS.assert_public({"arms": {"grad": {"summary": {"cer_all": {"errors": 1, "chars": 2}}}}, "note": "counts"})
    for bad in ({"units": [{"text": "あ"}]}, {"x": [{"y": {"label": "x"}}]}, [{"readings": []}]):
        with pytest.raises(ValueError):
            TS.assert_public(bad)
