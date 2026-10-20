"""Tests of textspot2's scoring wrapper (eval/textspot2.py) and its protected read (eval/textspot2_protected.py): the
run names, the version-2 dispatch, the cache seed, finished runs, scoring on synthetic frames, the median-seed
grouping with r0's textspot seeds, the eligibility of a new shape, textspot2_rule.apply_rule2 called through select
on synthetic units, the ceiling's pad and vertical rule, the secondary column, and the protected script's once-only
refusal on pseudo frames. Synthetic frames, random int8 networks and fake readers only: no screenshots, no models,
no network (CI).
"""
import argparse
import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "synth", ROOT / "train", ROOT / "eval", ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot as TS  # noqa: E402
import textspot2 as T2E  # noqa: E402
import textspot2_protected as TP2  # noqa: E402
import textspot2_rule as R  # noqa: E402
import textspot_metrics as M  # noqa: E402
import textspot_protected as TP  # noqa: E402

import textdet_export as E  # noqa: E402
import textdet_model as TM  # noqa: E402
import textspot2_shapes as S2  # noqa: E402
import textspot_twin as S  # noqa: E402

_BLOBS = {}


@pytest.fixture(autouse=True)
def _model_shapes_restored():
    """train/textdet_model.SHAPES as it was: registering textspot2's shapes stays inside a test (#97's tests compare
    that registry with its own sweep)."""
    saved = dict(TM.SHAPES)
    yield
    TM.SHAPES.clear()
    TM.SHAPES.update(saved)


def vtdt(shape="s8_c8", seed=1):
    if (shape, seed) not in _BLOBS:
        S2.register()
        _BLOBS[shape, seed] = E.pack(*E.random_int_layers(shape, seed))
    return _BLOBS[shape, seed]


def vtsp(shape="s4p_c8", seed=1, half_above=600):
    return S.pack(vtdt(shape, seed), half_above, 1)


def as_v2(blob):
    """A version-1 blob relabelled as version 2 (the header's version field and CRC): what the dispatch must send to
    textspot2_twin."""
    b = bytearray(blob)
    struct.pack_into("<I", b, 4, 2)
    struct.pack_into("<I", b, 12, zlib.crc32(bytes(b[16:])) & 0xFFFFFFFF)
    return bytes(b)


def synthetic_frame(w=960, h=544, seed=0):
    """A dark frame with a light dialogue box of stroke rows at the bottom and a UI label top right."""
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
    return g


def frame_of(g, fid="f1", vn="v900001", part="vita", refs=None):
    rgb = np.repeat(g[..., None], 3, 2)
    return TS.Frame({"id": fid, "vn": vn, "part": part, "refs": refs or [], "text_free": False}, g,
                    TS.pixel_sha256(rgb))


class FakeReader:
    """Reads a crop as one character per 16 px of its width (a vertical crop arrives turned)."""

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


def write_run(runs, name, blob, listed=True, steps=30000):
    d = runs / name
    d.mkdir(parents=True)
    (d / "final.vtsp").write_bytes(blob)
    sha = hashlib.sha256(blob).hexdigest()
    (d / "done.json").write_text(json.dumps({"steps": steps, "vtsp_bytes": len(blob), "vtsp_sha256": sha}))
    if listed:
        with open(runs / "finished.txt", "a") as f:
            f.write(f"{name} {sha} {len(blob)} {steps} 1.0\n")
    return d


# ----------------------------------------------------------------------------- run names and stage 1

def test_stage1_names_and_run_parsing():
    p = json.loads(T2E.PROTOCOL2.read_text(encoding="utf-8"))
    assert len(T2E.STAGE1) == 38 and len(set(T2E.STAGE1)) == 38
    for n in T2E.STAGE1:
        r = T2E.parse_run(n)
        assert r["shape"] in p["shapes"] and r["recipe"] in p["recipes"] and r["seed"] in (1, 2, 3)
    assert sum(1 for n in T2E.STAGE1 if n.startswith("r0_")) == 2
    assert T2E.parse_run("kd_h2_s8_c24n4x2_seed3") == {"run": "kd_h2_s8_c24n4x2_seed3", "recipe": "kd",
                                                       "shape": "s8_c24n4x2", "seed": 3,
                                                       "arm_id": "kd_h2_s8_c24n4x2"}
    for bad in ("data_h2_bogus_seed1", "data_s4p_c8_seed1", "Data_h2_s4p_c8_seed1", "data_h2_s4p_c8"):
        with pytest.raises(ValueError):
            T2E.parse_run(bad)


# ----------------------------------------------------------------------------- the version-2 dispatch

def test_blob_version_and_v2_dispatch(tmp_path, monkeypatch):
    v1 = vtsp("s4p_c8", 3)
    v2 = as_v2(v1)
    assert (T2E.blob_version(v1), T2E.blob_version(v2)) == (1, 2)
    with pytest.raises(ValueError):
        T2E.blob_version(b"VTDT" + bytes(8))
    calls = []

    class Fake:                          # a stand-in for synth/textspot2_twin: version 2 decoded as its version 1
        class Model:
            def __init__(self, blob):
                b = bytearray(blob)
                struct.pack_into("<I", b, 4, 1)
                struct.pack_into("<I", b, 12, zlib.crc32(bytes(b[16:])) & 0xFFFFFFFF)
                self.inner = S.Model(bytes(b))
                for k in ("name", "stride", "half_above", "conf_min", "min_cells", "blob_bytes", "net"):
                    setattr(self, k, getattr(self.inner, k))

        @staticmethod
        def spot(model, img):
            calls.append(model.inner.name)
            return S.spot(model.inner, img)

    monkeypatch.setitem(sys.modules, "textspot2_twin", Fake)
    monkeypatch.setattr(T2E, "TWIN2", T2E.PROTOCOL2)            # "exists": the fake module is imported
    g = synthetic_frame(seed=5)
    f = frame_of(g)
    s1 = TS.Spec("a_det", "det", blob=v1, half_above=600)
    s2 = TS.Spec("b_det", "det", blob=v2, half_above=600)
    with pytest.raises(ValueError):                              # textspot's twin refuses version 2, unchanged
        s2.meta()
    want = TS.Twin(f, TS.load_nets([s1])).arm(s1)
    assert not calls
    with T2E.v2_dispatch(True):
        assert sys.modules["textspot_twin"].TEXTSPOT2_SHIM and TS.CDrivers is T2E.CDrivers2
        assert s2.meta()["shape"] == "s4p_c8"
        nets = TS.load_nets([s1, s2])
        tw = TS.Twin(f, nets)
        a1, a2 = tw.arm(s1), tw.arm(s2)
        assert calls == ["s4p_c8"] and a2["boxes"] == a1["boxes"] == want["boxes"] and a2["pick"] == want["pick"]
        assert T2E.CDrivers2(tmp_path).spot([g], v2) is None    # no version-2 driver built: unchecked
    assert sys.modules["textspot_twin"] is S and TS.CDrivers is not T2E.CDrivers2
    with T2E.v2_dispatch(False):
        assert sys.modules["textspot_twin"] is S


def test_v2_needs_the_runtime_twin(monkeypatch, tmp_path):
    monkeypatch.setattr(T2E, "TWIN2", tmp_path / "missing.py")
    with pytest.raises(SystemExit):
        T2E.model_of(as_v2(vtsp()))
    assert T2E.model_of(vtsp()).name == "s4p_c8"


def test_a_d2s_run_through_the_runtime_twin_and_its_driver(tmp_path):
    """A random VTSP version-2 blob (train/textspot2_export) as a finished run: scored through synth/textspot2_twin
    and checked against runtime/vocr_textspot2_test (textspot's batch protocol) on every frame."""
    import shutil
    import subprocess

    import textspot2_export as X
    import textspot2_twin as TT
    if shutil.which("gcc") is None:
        pytest.skip("no gcc")
    blob, _ = X.random_blob("s8_c24n4x2", 3)
    runs = tmp_path / "runs"
    runs.mkdir()
    write_run(runs, "data_h2_s8_c24n4x2_seed1", blob)
    found, _ = T2E.finished_runs(runs)
    assert found["data_h2_s8_c24n4x2_seed1"]["version"] == 2
    cdir = tmp_path / "cbuild"
    cdir.mkdir()
    subprocess.run(["gcc", *TS.C_FLAGS, "-o", str(cdir / T2E.DRIVER2), *T2E.DRIVER2_SOURCE_SETS[0]], check=True,
                   cwd=ROOT / "runtime")
    frames = [frame_of(synthetic_frame(seed=40 + i), fid=f"f{i}", vn=f"v90020{i}") for i in range(3)]
    manifest = {"set": "pseudo", "path": "x", "sha256": "p" * 64, "frames": 3, "vns": 3}
    out = tmp_path / "units"
    todo = T2E.pending(out, found, manifest, [f.id for f in frames], ["VOCR5-N"], True)
    T2E.score_runs(frames, manifest, found, todo, out, ["VOCR5-N"], fake_readers(tmp_path), TS.Cache(tmp_path / "c"),
                   cdir, 1, log=lambda s: None)
    bd = json.loads((out / "T2_data_h2_s8_c24n4x2_seed1_det" / "boxes.json").read_text())
    for f in frames:
        r = TT.spot(TT.Model(blob), f.grey)
        assert bd["frames"][f.id]["boxes"] == [[b.x, b.y, b.w, b.h, b.vertical, b.conf, b.dark, b.block]
                                               for b in r.boxes]
        assert bd["frames"][f.id]["c_equal"] is True                          # the C driver gave the twin's boxes
    assert bd["spec_meta"]["shape"] == "s8_c24n4x2" and bd["spec_meta"]["stride"] == 4
    side = json.loads((out / "T2_data_h2_s8_c24n4x2_seed1_det" / "textspot2.json").read_text())
    assert side["blob_version"] == 2 and side["driver2_sha256"] == TS.sha256(cdir / T2E.DRIVER2)
    assert sys.modules["textspot_twin"] is S                                    # the dispatch is undone


# ----------------------------------------------------------------------------- the cache seed

def test_seed_cache_once_and_read_only_sources(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.jsonl").write_text('{"k": "x", "t": "y"}\n')
    (src / "b.jsonl").write_text('{"k": "z", "t": "w"}\n')
    dst = tmp_path / "dst"
    rec = T2E.seed_cache(dst, src)
    assert sorted(rec["files"]) == ["a.jsonl", "b.jsonl"] and (dst / "SEEDED.json").exists()
    assert TS.Cache(dst).get("z") == "w"
    (src / "a.jsonl").write_text('{"k": "x", "t": "changed"}\n')
    assert T2E.seed_cache(dst, src) == rec                       # seeded once: later calls change nothing
    assert TS.Cache(dst).get("x") == "y"
    monkeypatch.setattr(T2E, "T", tmp_path / "textspot")
    with pytest.raises(SystemExit):                              # never written under textspot's artifacts
        T2E.seed_cache(tmp_path / "textspot" / "eval" / "cache", src)
    with pytest.raises(SystemExit):
        T2E.check_writable(tmp_path / "textspot" / "x.json")
    with pytest.raises(SystemExit):
        T2E.seed_cache(tmp_path / "dst2", tmp_path / "empty")


# ----------------------------------------------------------------------------- finished runs

def test_finished_runs_and_specs(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    write_run(runs, "data_h2_s4p_c8_seed1", vtsp("s4p_c8", 1))
    write_run(runs, "data_h2_s4p_c8_seed2", vtsp("s4p_c8", 2), listed=False)      # not in finished.txt yet
    write_run(runs, "probe_s4p_c8", vtsp("s4p_c8", 4))                             # not a run name: skipped
    found, skipped = T2E.finished_runs(runs)
    assert sorted(found) == ["data_h2_s4p_c8_seed1"] and skipped == ["probe_s4p_c8"]
    r = found["data_h2_s4p_c8_seed1"]
    assert (r["recipe"], r["shape"], r["seed"], r["version"], r["half_above"]) == ("data", "s4p_c8", 1, 1, 600)
    specs = T2E.run_specs(found)
    assert [s.name for s in specs] == ["T2_data_h2_s4p_c8_seed1_det", "T2_data_h2_s4p_c8_seed1_detref"]
    m = specs[0].meta()
    assert (m["family"], m["run"], m["decode"], m["spec"]) == ("T2", "data_h2_s4p_c8_seed1", "det",
                                                               "T2:data_h2_s4p_c8_seed1")
    with pytest.raises(SystemExit):
        T2E.finished_runs(runs, ["data_h2_s4p_c8_seed2"])                          # asked for, but not listed
    (runs / "finished.txt").unlink()
    found, skipped = T2E.finished_runs(runs)
    assert sorted(found) == ["data_h2_s4p_c8_seed1", "data_h2_s4p_c8_seed2"] and skipped == ["probe_s4p_c8"]
    write_run(runs, "data_h2_s4p_c12_seed1", vtsp("s4p_c8", 5), listed=False)      # the blob is another shape
    with pytest.raises(SystemExit):
        T2E.finished_runs(runs)
    (runs / "data_h2_s4p_c12_seed1" / "final.vtsp").unlink()
    (runs / "data_h2_s4p_c12_seed1" / "done.json").unlink()
    (runs / "data_h2_s4p_c8_seed1" / "final.vtsp").write_bytes(vtsp("s4p_c8", 7))  # not done.json's blob
    with pytest.raises(SystemExit):
        T2E.finished_runs(runs)


def test_score_runs_on_synthetic_frames(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    write_run(runs, "data_h2_s4p_c8_seed1", vtsp("s4p_c8", 1))
    found, _ = T2E.finished_runs(runs)
    frames = [frame_of(synthetic_frame(seed=i), fid=f"f{i}", vn=f"v90000{i}") for i in range(3)]
    manifest = {"set": "pseudo", "path": "x", "sha256": "p" * 64, "frames": 3, "vns": 3}
    out, ids = tmp_path / "units", [f.id for f in frames]
    todo = T2E.pending(out, found, manifest, ids, ["VOCR5-N"], False)
    assert [s.name for s in todo] == ["T2_data_h2_s4p_c8_seed1_det", "T2_data_h2_s4p_c8_seed1_detref"]
    rd, cache = fake_readers(tmp_path), TS.Cache(tmp_path / "cache")
    T2E.score_runs(frames, manifest, found, todo, out, ["VOCR5-N"], rd, cache, None, 1, log=lambda s: None)
    for s in todo:
        side = json.loads((out / s.name / "textspot2.json").read_text())
        assert (side["arm_id"], side["seed"], side["blob_version"]) == ("data_h2_s4p_c8", 1, 1)
        u = json.loads((out / s.name / "units_VOCR5-N.json").read_text())
        assert u["meta"]["spec_meta"]["family"] == "T2" and len(u["units"]) == 3
    assert T2E.pending(out, found, manifest, ids, ["VOCR5-N"], False) == []          # scored: skipped next time
    assert len(T2E.pending(out, found, manifest, ids, ["VOCR5-N", "VOCR5-S"], False)) == 2
    doc = T2E.record_frames(tmp_path / "frames.json", frames, manifest)
    assert doc["frames"]["f1"]["sha"] == frames[1].sha
    bad = [frame_of(synthetic_frame(seed=9), fid="f1", vn="v900001")]
    with pytest.raises(SystemExit):                                                  # other pixels, same frame id
        T2E.record_frames(tmp_path / "frames.json", bad, manifest)


# ----------------------------------------------------------------------------- a synthetic development world

MAN, RN, RV = "m" * 64, "n" * 64, "v" * 64
W, H = 960, 544
DLG, NAME, UI = "あいうえおかきくけこ", "名前", "セーブ"
REFS = [{"box": [100, 400, 300, 24], "cls": "dialogue", "text": DLG, "vertical": False, "ignore": False},
        {"box": [100, 360, 80, 20], "cls": "name", "text": NAME, "vertical": False, "ignore": False},
        {"box": [800, 20, 60, 16], "cls": "ui", "text": UI, "vertical": False, "ignore": False}]
# readings of the dialogue, name and UI references (None: not found) and the class's own box shift (so that one box
# always has one reading); all-text errors per frame: base 10, ref 4 (dialogue 2), fast_ref 5, r0 4 (dialogue 3),
# perfect 0, one 1, two 2 (dialogue 1), of 15 characters
QUALITY = {"base": ((DLG[:5], None, None), (0, 0)), "ref": ((DLG[:8], NAME, UI[:1]), (1, 0)),
           "fast_ref": ((DLG[:7], NAME, UI[:1]), (2, 0)), "r0": ((DLG[:7], NAME, UI[:2]), (3, 0)),
           "perfect": ((DLG, NAME, UI), (0, 1)), "one": ((DLG, NAME, UI[:2]), (1, 1)),
           "two": ((DLG[:9], NAME, UI[:2]), (2, 1))}
NBYTES = {"s4p_c8": 2732, "s4_c24n4": 5732}
TS_A9_FAKE = {"summary": [{"shape": "s4p_c8", "scale": "h2", "half_above": 600, "ms_333_960": 71.854,
                           "arena_up_to_960": 37767, "ms_333_480": 65.25},
                          {"shape": "s4_c24n4", "scale": "h2", "half_above": 600, "ms_333_960": 130.69,
                           "arena_up_to_960": 80799, "ms_333_480": 121.17}]}
RULE = json.loads(T2E.PROTOCOL2.read_text(encoding="utf-8"))["rule"]


def world(n_vns=10, per_vn=3):
    rows = [{"id": f"f{v}_{k}", "vn": f"v{v:03d}", "part": "vita", "dims": [W, H], "text_free": False, "refs": REFS}
            for v in range(n_vns) for k in range(per_vn)]
    return rows, {r["id"]: hashlib.sha256(r["id"].encode()).hexdigest() for r in rows}


def make_arm(d, name, meta, rows, shas, cache, quality, h_in=None):
    """An arm directory as eval/textspot.py writes one (boxes.json, units_VOCR5-N.json): its units are frame_unit's
    for its boxes and readings, and the readings go into the cache."""
    texts_q, (sx, sw) = QUALITY[quality]
    fr, units = {}, []
    for row in rows:
        by_box = {}
        for r, t in zip(row["refs"], texts_q):
            if t is not None:
                x, y, w, h = M.pad_box(r["box"], False, W, H)
                by_box[(x + sx, y, w + sw, h)] = t
        ordered, blocks, pick = S.group_and_order([(*b, 0, 50, 0) for b in by_box], H)
        texts = [by_box[tuple(b[:4])] for b in ordered]
        for b, t in zip(ordered, texts):
            cache.put(TS.Cache.key(shas[row["id"]], b[:4], b[6], b[4], RN), t)
        arm = {"boxes": [tuple(b) for b in ordered], "blocks": blocks, "pick": pick, "truncated": 0, "h_in": h_in}
        units.append(TS.frame_unit(TS.Frame(row, np.zeros((H, W), np.uint8), shas[row["id"]]), arm, texts, True))
        fr[row["id"]] = {"boxes": [list(b) for b in ordered], "blocks": [list(b) for b in blocks], "pick": pick,
                         "truncated": 0, "h_in": h_in, "c_equal": True, "part": row["part"]}
    a = Path(d) / name
    a.mkdir(parents=True)
    (a / "boxes.json").write_text(json.dumps({"kind": "textspot-boxes", "arm": name, "spec_meta": meta,
                                              "manifest": {"sha256": MAN}, "frames": fr}))
    doc = {"meta": {"manifest": {"sha256": MAN}, "boxes_sha256": TS.sha256(a / "boxes.json"), "model": "VOCR5-N",
                    "recognizers_sha256": {"VOCR5-N": RN, "VOCR5-V": RV}, "spec_meta": meta},
           "units": units, "summary": TS.summarize(units)}
    (a / "units_VOCR5-N.json").write_text(json.dumps(doc))
    return a


def cand_meta(name, kind, shape, run=None, family=None):
    return {"arm": name, "spec": name, "kind": kind, "baseline": None, "decode": T2E.DECODE[kind], "family": family,
            "run": run, "blob_sha256": hashlib.sha256(name.rsplit("_", 1)[0].encode()).hexdigest(),
            "blob_bytes": NBYTES[shape], "shape": shape, "stride": 4, "half_above": 600, "conf_min": 1,
            "source": None}


def t2_arms(units, run, quality, rows, shas, cache):
    shape = T2E.parse_run(run)["shape"]
    for kind in T2E.KINDS:
        name = f"T2_{run}_{kind}"
        make_arm(units, name, cand_meta(name, kind, shape, run, "T2"), rows, shas, cache, quality,
                 3 if kind == "detref" else None)


TS_QUALITY = {("s4_c24n4", 1): "r0", ("s4_c24n4", 2): "ref", ("s4p_c8", 1): "fast_ref", ("s4p_c8", 2): "r0"}


def fake_textspot(tmp_path, rows, shas, cache):
    """textspot's r0 seeds 1-2 (famB, famC) and its selection record's summaries of them."""
    t_dir, dev = tmp_path / "textspot", {"candidates": {}}
    for arm_id, seeds in T2E.R0_TEXTSPOT.items():
        shape = T2E.parse_run(arm_id + "_seed1")["shape"]
        for seed, (fam, prefix) in seeds.items():
            for kind in T2E.KINDS:
                name = f"{prefix}_{kind}"
                a = make_arm(t_dir / "eval" / fam, name, cand_meta(name, kind, shape), rows, shas, cache,
                             TS_QUALITY[shape, seed], 3 if kind == "detref" else None)
                dev["candidates"][name] = {"summary": json.loads((a / "units_VOCR5-N.json").read_text())["summary"]}
    ts_dev = tmp_path / "textspot_dev.json"
    ts_dev.write_text(json.dumps(dev))
    return t_dir, ts_dev


# ----------------------------------------------------------------------------- seeds

def test_median_seed_grouping_with_r0_textspot_seeds(tmp_path):
    rows, shas = world()
    cache = TS.Cache(tmp_path / "cache")
    t_dir, ts_dev = fake_textspot(tmp_path, rows, shas, cache)
    units = tmp_path / "units"
    for shape in ("s4_c24n4", "s4p_c8"):
        t2_arms(units, f"r0_h2_{shape}_seed3", "two", rows, shas, cache)
    t2_arms(units, "data_h2_s4p_c8_seed1", "perfect", rows, shas, cache)
    t2_arms(units, "data_h2_s4p_c8_seed3", "two", rows, shas, cache)            # seed 2 crashed: two finished
    seeds = T2E.t2_seeds([units]) + T2E.textspot_r0_seeds(t_dir, ts_dev)
    reps = T2E.representatives(seeds)
    g = reps["r0_h2_s4_c24n4", "det+refine"]
    assert [(s["seed"], s["source"]) for s in g["seeds"]] == [(1, "textspot"), (2, "textspot"), (3, "textspot2")]
    # all-text CER: seed 3 (2 errors a frame), seed 2 (4; dialogue 2), seed 1 (4; dialogue 3): the median is seed 2,
    # textspot's winner itself
    assert g["order"] == [3, 2, 1] and g["index"] == 1 and g["rep"]["name"] == "B_h2_s4_c24n4_seed2_detref"
    g = reps["r0_h2_s4p_c8", "det"]
    assert g["order"] == [3, 2, 1] and g["rep"]["name"] == "C_h2_s4p_c8_seed2_det"
    g = reps["data_h2_s4p_c8", "det"]
    assert g["order"] == [1, 3] and g["rep"]["seed"] == 3                       # two seeds: the worse one
    rep, info = R.median_seed([{k: s[k] for k in ("seed", "cer_all", "cer_dialogue", "f1")} for s in g["seeds"]])
    assert (rep["seed"], info["order"]) == (3, [1, 3])                          # the rule's own function agrees
    dev = json.loads(ts_dev.read_text())
    dev["candidates"]["C_h2_s4p_c8_seed1_det"]["summary"]["frames"] += 1
    ts_dev.write_text(json.dumps(dev))
    with pytest.raises(SystemExit):                                             # not the units textspot scored
        T2E.textspot_r0_seeds(t_dir, ts_dev)


# ----------------------------------------------------------------------------- eligibility

def test_eligibility_of_a_new_shape():
    S2.register()
    shape = "s4p_c10n4"
    macs = TM.cost(TM.SHAPES[shape], 480, 272)["macs"]
    t2 = {"summary": [{"shape": shape, "scale": "h2", "half_above": 600, "vtsp_bytes": 4700, "macs_960": macs,
                       "ms_333_960": 110.0, "arena_up_to_960": 50000, "ms_333_480": 101.0},
                      {"shape": "s4p_c8", "scale": "h2", "half_above": 600, "ms_333_960": 72.5,
                       "arena_up_to_960": 37767}]}
    d2s = {"summary": [{"shape": "s8_c24n4x2", "scale": "h2", "half_above": 600,
                        "macs_960": T2E.d2s_macs("s8_c24n4x2", 600), "ms_333_960": 70.0, "arena_up_to_960": 60000}]}
    rows = T2E.a9_rows_all(t2, d2s, TS_A9_FAKE)
    frames = {f"f{i}": {"c_equal": True, "h_in": 4 + i % 3, "part": "vita" if i < 5 else "psp"} for i in range(8)}
    bd = {"arm": "x", "frames": frames}
    meta = {"shape": shape, "half_above": 600, "blob_bytes": 4700, "blob_sha256": "b" * 64, "decode": "det+refine"}
    e = T2E.eligibility2(meta, bd, rows, 72391)
    # the median of the Vita frames' horizontal boxes (4, 5, 6, 4, 5) is 5: + 7.0 ms of refinement
    assert (e["macs_960"], e["est_ms_333_960"], e["arena_960"], e["c_equal"]) == (macs, 117.0, 72391, True)
    assert R.eligible(T2E.light(e), RULE) and not R.eligible(T2E.light(e), RULE, RULE["fast_ms_max_333_960"])
    d = T2E.eligibility2(dict(meta, decode="det"), bd, rows, 72391)
    assert (d["est_ms_333_960"], d["arena_960"]) == (110.0, 50000)
    assert T2E.report_costs(d, rows) == {"est_ms_444_960": round(110.0 * 333 / 444, 3), "ms_333_480_spot": 101.0}
    p = T2E.eligibility2(dict(meta, shape="s4p_c8", blob_bytes=2732, decode="det"), bd, rows, 72391)
    assert p["est_ms_333_960"] == 71.854                                         # textspot's table for its shapes
    v = T2E.eligibility2(dict(meta, shape="s8_c24n4x2", blob_bytes=9000, decode="det"), bd, rows, 72391)
    assert (v["macs_960"], v["macs_source"], v["est_ms_333_960"]) == (12305280, "train/textspot2_export.costs", 70.0)
    assert TS.macs_960 is not None and TS.macs_960(shape, 600) == macs           # the harness is restored
    off = T2E.a9_rows_all(t2, {"summary": [dict(d2s["summary"][0], macs_960=123456)]}, TS_A9_FAKE)
    with pytest.raises(SystemExit):                                              # the d2s row's MACs are checked
        T2E.eligibility2(dict(meta, shape="s8_c24n4x2", blob_bytes=9000, decode="det"), bd, off, 72391)
    bad = T2E.a9_rows_all({"summary": [dict(t2["summary"][0], macs_960=macs + 1)]}, None, TS_A9_FAKE)
    with pytest.raises(SystemExit):
        T2E.eligibility2(meta, bd, bad, 72391)
    frames["f0"]["c_equal"] = False
    assert not R.eligible(T2E.light(T2E.eligibility2(meta, bd, rows, 72391)), RULE)
    proto2 = json.loads(T2E.PROTOCOL2.read_text())
    assert T2E.refine_arena2(None, None, proto2)[0] == 72391
    assert T2E.refine_arena2({"refine_arena_960x544": 72391}, None, proto2) == (72391, "a9:refine_arena_960x544")
    with pytest.raises(SystemExit):
        T2E.refine_arena2({"refine_arena_960x544": 70000}, None, proto2)


# ----------------------------------------------------------------------------- the ceiling

def test_ceiling_pad_and_vertical_rule():
    raw = [[100, 200, 400, 230], [50, 50, 80, 250], [10, 10, 30, 30], [950, 530, 960, 544], [5, 5, 5, 40],
           [300, 300, 340, 320], [400, 400, 500, 420]]
    scores = [0.5, 0.9, 0.7, 0.6, 0.99, 0.4999, float("nan")]
    assert T2E.ceiling_boxes(raw, scores, 960, 544) == [
        (95, 195, 310, 40, 0, 0),        # 300 x 30: horizontal, pad max(2, (30 18 + 50) / 100) = 5
        (45, 45, 40, 210, 1, 0),         # 30 x 200: vertical (taller than wide), pad across its width
        (6, 6, 28, 28, 0, 0),            # square: horizontal (not taller than wide), pad 4
        (948, 528, 12, 16, 1, 0)]        # at the frame's corner: pad 2, clipped
    got = T2E.map_320(np.array([[100.4, 50.2, 200.9, 60.6], [-3.0, 10.0, 330.0, 200.0]], np.float32), 640, 384)
    assert got.tolist() == [[200, 100, 401, 121], [0, 20, 640, 384]]          # int(v / ratio), then clipped
    g = synthetic_frame(seed=6)
    f = frame_of(g)
    boxes = [(30, 384, 900, 40, 0, 0), (30, 424, 900, 40, 0, 0), (50, 30, 40, 290, 1, 0)]
    det, ref = T2E.ceiling_arm(f, boxes, "det"), T2E.ceiling_arm(f, boxes, "detref")
    assert det["h_in"] is None and ref["h_in"] == 2 and det["c_equal"] is None and ref["refine_c_equal"] is None
    assert [b[:5] for b in det["boxes"] if b[4]] == [b[:5] for b in ref["boxes"] if b[4]] == [(50, 30, 40, 290, 1)]
    hz = TS.refine_spot_boxes([(*b, 0) for b in boxes], g)[0]
    assert sorted(b[:4] for b in ref["boxes"] if not b[4]) == sorted(b[:4] for b in hz if not b[4])
    assert sum(bk[6] for bk in det["blocks"]) == len(det["boxes"]) and det["pick"] >= 0


def test_ceiling_arms_scored_as_candidates_never_one(tmp_path):
    pairs = []
    for i in range(3):
        g = synthetic_frame(seed=20 + i)
        rgb = np.repeat(g[..., None], 3, 2)
        pairs.append((TS.Frame({"id": f"f{i}", "vn": f"v90010{i}", "part": "vita", "refs": [
            {"box": [40, 404, 800, 22], "cls": "dialogue", "text": "あいうえお", "vertical": False, "ignore": False}],
            "text_free": False}, g, TS.pixel_sha256(rgb)), rgb))
    det = {"model": "fake", "model_sha256": "f" * 64,
           "frames": {f.id: {"sha": f.sha, "boxes": [[40, 404, 840, 426], [60, 30, 80, 300], [0, 0, 3, 3]],
                             "scores": [0.8, 0.6, 0.3]} for f, _ in pairs}}
    proto2 = json.loads(T2E.PROTOCOL2.read_text())
    manifest = {"set": "pseudo", "path": "x", "sha256": "p" * 64, "frames": 3, "vns": 3}
    rec = T2E.run_ceiling(pairs, manifest, tmp_path / "ceiling", ["960x544", "320x192"], tmp_path, proto2,
                          ["VOCR5-N"], fake_readers(tmp_path), TS.Cache(tmp_path / "cache"), workers=1,
                          log=lambda s: None, detections={"960x544": det, "320x192": det})
    assert sorted(rec["arms"]) == ["meiki320_det", "meiki320_detref", "meiki960_det", "meiki960_detref"]
    s = rec["arms"]["meiki960_det"]["VOCR5-N"]
    assert s["boxes"] == 6 and s["vertical_boxes"] == 3 and s["detection"]["tp"] == 3   # score 0.3: dropped
    TS.assert_public(json.loads((tmp_path / "ceiling" / "ceiling.json").read_text()))
    assert TS.load_candidate_dirs([tmp_path / "ceiling"]) == {}                         # never a candidate
    c = T2E.ceiling_of(tmp_path / "ceiling")
    assert sorted(c) == sorted(rec["arms"]) and c["meiki960_detref"]["boxes"]["spec_meta"]["decode"] == "det+refine"


# ----------------------------------------------------------------------------- the secondary column

def test_dnc_units_recomputed_and_checked(tmp_path):
    refs = [{"box": [100, 400, 300, 24], "cls": "dialogue", "text": "あいうえお", "vertical": False, "ignore": False},
            {"box": [100, 360, 80, 20], "cls": "name", "text": "名前", "vertical": False, "ignore": False},
            {"box": [100, 300, 60, 20], "cls": "choice", "text": "はい", "vertical": False, "ignore": False},
            {"box": [800, 20, 60, 16], "cls": "ui", "text": "SAVE", "vertical": False, "ignore": False},
            {"box": [100, 380, 300, 10], "cls": "dialogue", "text": None, "vertical": False, "ignore": True}]
    row = {"id": "f1", "vn": "v1", "part": "vita", "dims": [W, H], "refs": refs}
    shas = {"f1": "s" * 64}
    reads = {(96, 396, 308, 32): "あいうえ", (96, 296, 68, 28): "はい", (797, 17, 66, 22): "SAV",
             (500, 200, 40, 30): "ゴミ"}
    cache = TS.Cache(tmp_path / "cache")
    ordered, blocks, pick = S.group_and_order([(*b, 0, 50, 0) for b in reads], H)
    texts = [reads[tuple(b[:4])] for b in ordered]
    for b, t in zip(ordered, texts):
        cache.put(TS.Cache.key("s" * 64, b[:4], b[6], b[4], RN), t)
    arm = {"boxes": [tuple(b) for b in ordered], "blocks": blocks, "pick": pick, "truncated": 0}
    unit = TS.frame_unit(TS.Frame(row, np.zeros((H, W), np.uint8), "s" * 64), arm, texts, True)
    assert unit["e2e"]["all"] == [1 + 2 + 0 + 1 + 2, 13] and unit["junk"] == 2
    bd = {"arm": "x", "frames": {"f1": {"boxes": [list(b) for b in ordered]}}}
    # dialogue 1 error, the missed name 2, the choice 0; the UI label and the junk are not in the column
    assert T2E.dnc_of(bd, [row], shas, cache, (RN, RV), [unit]) == [{"unit": "f1", "vn": "v1", "errors": 3,
                                                                      "chars": 9}]
    bad = json.loads(json.dumps(unit))
    bad["e2e"]["all"][0] += 1
    with pytest.raises(SystemExit):                                  # the recomputation must give the units' errors
        T2E.dnc_of(bd, [row], shas, cache, (RN, RV), [bad])
    with pytest.raises(SystemExit):                                  # every box needs its cached reading
        T2E.dnc_of(bd, [row], shas, cache, ("x" * 64, RV), None)


# ----------------------------------------------------------------------------- select (textspot2_rule.apply_rule2)

def select_world(tmp_path):
    rows, shas = world()
    cache = TS.Cache(tmp_path / "cache")
    t_dir, ts_dev = fake_textspot(tmp_path, rows, shas, cache)
    units = tmp_path / "units"
    for shape in ("s4_c24n4", "s4p_c8"):
        t2_arms(units, f"r0_h2_{shape}_seed3", "two", rows, shas, cache)
    for seed, q in ((1, "perfect"), (2, "one"), (3, "two")):
        t2_arms(units, f"data_h2_s4p_c8_seed{seed}", q, rows, shas, cache)
        t2_arms(units, f"data_h2_s4_c24n4_seed{seed}", "perfect", rows, shas, cache)
    base_doc = {"manifest": {"sha256": MAN}, "arms": {}}
    for b in TS.BASELINES:
        a = make_arm(tmp_path / "baseline", b, {"kind": "baseline", "arm": b}, rows, shas, cache, "base")
        base_doc["arms"][b] = {"VOCR5-N": {"units": str(a / "units_VOCR5-N.json"),
                                           "units_sha256": TS.sha256(a / "units_VOCR5-N.json"),
                                           "boxes_sha256": TS.sha256(a / "boxes.json")}}
    ceil = tmp_path / "ceiling"
    make_arm(ceil, "meiki960_det", {"kind": "ceiling", "decode": "det", "arm": "meiki960_det"}, rows, shas, cache,
             "perfect")
    proto2 = json.loads(T2E.PROTOCOL2.read_text(encoding="utf-8"))
    proto2["reused"]["dev_manifest"]["sha256"] = MAN
    for k, fam, name in (("ref", "famB", "B_h2_s4_c24n4_seed2_detref"),
                         ("fast_ref", "famC", "C_h2_s4p_c8_seed1_detref")):
        d = t_dir / "eval" / fam / name
        proto2["references"][k].update(name=name, units=str(d / "units_VOCR5-N.json"), boxes=str(d / "boxes.json"),
                                       units_sha256=TS.sha256(d / "units_VOCR5-N.json"),
                                       boxes_sha256=TS.sha256(d / "boxes.json"))
    return rows, shas, cache, t_dir, ts_dev, units, base_doc, ceil, proto2


def test_select_calls_rule2_on_synthetic_units(tmp_path):
    rows, shas, cache, t_dir, ts_dev, units, base_doc, ceil, proto2 = select_world(tmp_path)
    doc = T2E.select([units], {"summary": []}, proto2, base_doc, rows, shas, cache, None, TS_A9_FAKE, None, None,
                     ceil, t_dir, ts_dev, log=lambda s: None)
    res = doc["result"]
    p8, p24 = "T2_data_h2_s4p_c8_seed2", "T2_data_h2_s4_c24n4_seed2"
    assert sorted(doc["candidates"]) == sorted([f"{p8}_det", f"{p8}_detref", f"{p24}_det", f"{p24}_detref"])
    assert res["eligible"] == [f"{p8}_det", f"{p8}_detref", f"{p24}_det", f"{p24}_detref"]       # cost order
    assert res["bar"] == res["eligible"] and res["main"]["comparator"] == "B_h2_s4_c24n4_seed2_detref"
    # the cheapest qualifier (s4p_c8's median seed, 1 error a frame) is replaced by s4_c24n4's (0), resolved
    # better; det+refine reads the same and is not better than det
    assert res["main"]["winner"] == f"{p24}_det"
    assert [(s["replaces"], s["by"]) for s in res["main"]["steps"]] == [(f"{p8}_det", f"{p24}_det")]
    assert (res["fast"]["tier"], res["fast"]["winner"], res["fast"]["pool"]) == (1, f"{p8}_det",
                                                                                  [f"{p8}_det", f"{p8}_detref"])
    a = doc["arms"]["data_h2_s4p_c8|det"]
    assert (a["order"], a["index"], a["representative"], a["role"]) == ([1, 2, 3], 1, f"{p8}_det", "candidate")
    r0 = doc["arms"]["r0_h2_s4_c24n4|det+refine"]
    assert (r0["role"], r0["representative"]) == ("control", "B_h2_s4_c24n4_seed2_detref")
    assert set(doc["report_comparisons"]) == {r0["representative"], *(doc["arms"][k]["representative"] for k in (
        "r0_h2_s4_c24n4|det", "r0_h2_s4p_c8|det", "r0_h2_s4p_c8|det+refine"))}
    assert doc["pareto"]["front"] == [f"{p24}_det", f"{p8}_det"]
    assert doc["pareto"]["points"]["C_h2_s4p_c8_seed1_detref"]["role"] == "fast_ref"
    rows_ = {(b["what"], b["decode"]): b for b in doc["bought"]}
    assert rows_["lab - r0", "det"]["missing"] and rows_["data s4p_c12 - data s4_c24n4", "det+refine"]["missing"]
    b = rows_["data s4p_c8 - data s4_c24n4", "det"]
    assert (b["minuend"], b["subtrahend"]) == (f"{p8}_det", f"{p24}_det")
    assert b["comparison"]["cer_all"]["delta"] == round(100 / 15, 4) and b["dnc"]["delta"] == 0.0
    c = rows_["meiki960_det - ref", "det"]
    assert c["comparison"]["cer_all"]["delta"] == round(-400 / 15, 4)
    assert c["dnc"]["delta"] == round(-200 / 12, 4)                       # dialogue + name: 2 of 12 characters
    assert rows_["main pick - ref", "det"]["minuend"] == f"{p24}_det"
    s = {x["name"]: x for x in doc["seeds"]}
    assert len(s) == 24 and s[f"{p8}_det"]["eligible_fast"] and not s[f"{p24}_det"]["eligible_fast"]
    assert s[f"{p8}_detref"]["est_ms_333_960"] == round(71.854 + 1.4 * 3, 3)
    assert s[f"{p8}_det"]["est_ms_444_960"] == round(71.854 * 333 / 444, 3)
    assert s["B_h2_s4_c24n4_seed2_detref"]["dnc"]["cer"] == round(100 * 2 / 12, 4)
    assert doc["ceiling"]["meiki960_det"]["secondary"]["dnc"]["cer"] == 0.0
    st = doc["stage2_inputs"]
    assert (st["ema_kept"], st["cls_kept"], st["fast_shape"]["shape"], st["main_shape"]["shape"],
            st["kd_teacher"]["by"], st["long_on"]) == (None, None, "s4p_c8", "s4_c24n4", f"{p24}_det", [])
    TS.assert_public(doc)
    n = T2E.write_compact(tmp_path / "dev.json", doc)
    assert n < T2E.MAX_RECORD and json.loads((tmp_path / "dev.json").read_text()) == json.loads(json.dumps(doc))
    with pytest.raises(SystemExit):
        T2E.write_compact(tmp_path / "big.json", doc, limit=1000)
    md = T2E.render(json.loads((tmp_path / "dev.json").read_text()))
    assert f"**{p24}_det**" in md and "not run" in md and "`meiki960_det`" in md
    s_units = json.loads((units / f"{p24}_det" / "units_VOCR5-N.json").read_text())
    s_units["summary"]["cer_all"]["cer"] = 1.5
    (units / f"{p24}_det" / "units_VOCR5-S.json").write_text(json.dumps(s_units))
    sl = T2E.report_sl(doc, [units])
    assert list(sl) == [f"{p24}_det"] and sl[f"{p24}_det"]["VOCR5-S"]["cer_all"]["cer"] == 1.5
    assert "| VOCR5-S | " in T2E.render(doc, sl=sl)
    s_units["meta"]["boxes_sha256"] = "z" * 64
    (units / f"{p24}_det" / "units_VOCR5-S.json").write_text(json.dumps(s_units))
    with pytest.raises(SystemExit):                                       # S / L units of other boxes
        T2E.report_sl(doc, [units])
    seeds, _, _ = T2E.gather([units], proto2, rows, shas, cache, None, None, TS_A9_FAKE, t_dir=t_dir, ts_dev=ts_dev)
    hl = T2E.headline_md(seeds, RULE)
    assert f"`{p8}_det`" in hl and "Stage 1: 8 of 38 runs scored" in hl     # r0 seed 3 twice, data x 2 x 3
    f = Path(proto2["references"]["ref"]["units"])
    f.write_text(f.read_text() + "\n")
    with pytest.raises(SystemExit):                                       # the protocol's sha256 of ref's units
        T2E.reference(proto2, "ref")


def test_select_refuses_stale_and_foreign_units(tmp_path):
    rows, shas, cache, t_dir, ts_dev, units, base_doc, ceil, proto2 = select_world(tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    write_run(runs, "data_h2_s4p_c8_seed1", vtsp("s4p_c8", 1))                # another blob than the units scored
    with pytest.raises(SystemExit):
        T2E.t2_seeds([units], runs)
    bad = dict(base_doc, manifest={"sha256": "x" * 64})
    with pytest.raises(SystemExit):
        T2E.select([units], {"summary": []}, proto2, bad, rows, shas, cache, None, TS_A9_FAKE, None, None, None,
                   t_dir, ts_dev, log=lambda s: None)


# ----------------------------------------------------------------------------- the protected script (pseudo data)

def write_image(path, g):
    import cv2
    ok, buf = cv2.imencode(".png", np.repeat(g[..., None], 3, 2)[:, :, ::-1])
    assert ok
    Path(path).write_bytes(buf.tobytes())


@pytest.fixture()
def fake_blobs(tmp_path, monkeypatch):
    s8, gate = tmp_path / "s8c24.vtd", tmp_path / "gate.vtd"
    s8.write_bytes(vtdt("s8_c8", 3))
    gate.write_bytes(vtdt("s8_c8", 4))
    monkeypatch.setattr(TS, "S8C24", s8)
    monkeypatch.setattr(TS, "GATE", gate)
    monkeypatch.setattr(TS, "check_baseline_blobs", lambda proto: None)          # random blobs, not the pinned ones
    ref, cand = tmp_path / "ref.vtsp", tmp_path / "cand.vtsp"
    ref.write_bytes(S.pack(vtdt("s8_c8", 6), 600, 1))
    cand.write_bytes(S.pack(vtdt("s8_c8", 7), 600, 1))
    return ref, cand


def fake_test_set(tmp_path):
    shots = tmp_path / "shots"
    shots.mkdir()
    rows = []
    for i, part in enumerate(("vita", "psp", "adrenaline", "vita", "vita", "psp")):
        dims = (480, 272) if part == "psp" else (960, 544)
        g = synthetic_frame(*dims, seed=10 + i)
        vn, url = f"v99{i % 5:04d}", f"https://t.vndb.org/sf/0{i}/90{i}.jpg"
        p = shots / TS.shot_file_name(vn, url)
        write_image(p, g)
        refs = [{"box": [40, dims[1] - 140, dims[0] - 160, 22], "cls": "dialogue", "text": "あいうえおかきくけこ",
                 "vertical": False, "ignore": False},
                {"box": [40, dims[1] - 100, 200, 22], "cls": "name", "text": "なまえ", "vertical": False,
                 "ignore": False}]
        rows.append({"id": f"t{i}", "vn": vn, "part": part, "url": url, "sha256": TS.sha256(p), "src_dims": list(dims),
                     "scale": None, "dims": list(dims), "text_free": False, "refs": refs})
    man = tmp_path / "textspot-test.jsonl"
    man.write_text("".join(json.dumps(r) + "\n" for r in rows))
    proto = json.loads((ROOT / "eval" / "textspot_protocol.json").read_text())
    proto["amendments"] = [{"n": 1, "manifests": {TP.TEST_MANIFEST: TS.sha256(man)}}]
    pp = tmp_path / "protocol.json"
    pp.write_text(json.dumps(proto))
    proto2 = json.loads(T2E.PROTOCOL2.read_text())
    proto2["reused"]["test_manifest"]["sha256"] = TS.sha256(man)
    pp2 = tmp_path / "protocol2.json"
    pp2.write_text(json.dumps(proto2))
    return man, pp, pp2, shots


def test_protected_tests_once_against_stored_units(tmp_path, fake_blobs):
    refb, candb = fake_blobs
    man, pp, pp2, shots = fake_test_set(tmp_path)
    rd = fake_readers(tmp_path)
    ts_out = tmp_path / "textspot_tests"        # textspot's own read of ref: how the stored files were made
    TP.cmd_tests(argparse.Namespace(cand=f"detref:B_h2_s4_c24n4_seed2={refb}", decode="", out=str(ts_out),
                                    cache=str(tmp_path / "pcache"), protocol=str(pp), manifest=str(man),
                                    shots=[str(shots)], cbin="", models="N,S,L", workers=1, compiler="gcc", data="",
                                    report_parts=list(TP.REPORT_PARTS)), readers=rd, log=lambda s: None)
    out = tmp_path / "t2" / "tests"
    args = argparse.Namespace(cand=f"det:cand={candb}", decode="", out=str(out), runs=str(tmp_path / "runs"), cbin="",
                              models="N,S,L", workers=1, cache=str(tmp_path / "p2cache"),
                              seed_from=str(tmp_path / "pcache"), compiler="gcc", protocol=str(pp),
                              protocol2=str(pp2), stored=str(ts_out / "private"), record=str(ts_out / "tests.json"),
                              manifest=str(man), data="", shots=[str(shots)], report_parts=list(TP.REPORT_PARTS))
    res = TP2.cmd_tests(args, readers=rd, log=lambda s: None)
    assert (out / "tests.attempt.json").exists() and (out / "tests.json").exists()
    assert set(res["arms"]) == {"cand_det"} and (res["scored_frames"], res["report_only_frames"]) == (5, 1)
    assert sorted(p.name for p in (out / "private").iterdir()) == ["cand_det"]       # only the pick ran
    assert set(res["comparisons"]["VOCR5-N"]) == set(TP2.STORED_ARMS) == {TP2.REF_ARM, *TS.BASELINES}
    assert res["stored"]["reproduces_record"] and res["stored"]["crops_read_now"] == 0
    assert set(res["secondary"]["VOCR5-L"]["dnc"]) == {"cand_det", *TP2.STORED_ARMS}
    assert res["cache_seed"]["files"] and (tmp_path / "p2cache" / "SEEDED.json").exists()
    TS.assert_public(json.loads((out / "tests.json").read_text()))
    with pytest.raises(FileExistsError):                                              # once per command
        TP2.cmd_tests(args, readers=rd, log=lambda s: None)
    f = ts_out / "private" / "grad" / "units_VOCR5-N.json"
    doc = json.loads(f.read_text())
    doc["units"][0]["tp"] += 1
    f.write_text(json.dumps(doc))
    args2 = argparse.Namespace(**dict(vars(args), out=str(tmp_path / "t2b" / "tests")))
    with pytest.raises(SystemExit):           # stored units that do not reproduce the record: refused before reading
        TP2.cmd_tests(args2, readers=rd, log=lambda s: None)
    assert not (tmp_path / "t2b" / "tests" / "tests.attempt.json").exists()


def fake_clannad(tmp_path):
    from common import find_lines_ex
    session, work = tmp_path / "session", tmp_path / "work"
    (session / "frames").mkdir(parents=True)
    work.mkdir()
    roi, lines, refs = [0, 300, 960, 244], [], []
    for p in range(3):
        g = synthetic_frame(seed=30 + p)
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


def test_protected_clannad_once_against_stored_boxes(tmp_path, fake_blobs):
    refb, candb = fake_blobs
    work = fake_clannad(tmp_path)
    rd = fake_readers(tmp_path)
    proto = str(ROOT / "eval" / "textspot_protocol.json")
    ts_out = tmp_path / "textspot_clannad"
    TP.cmd_clannad(argparse.Namespace(cand=f"detref:B_h2_s4_c24n4_seed2={refb}", decode="", out=str(ts_out),
                                      cache=str(tmp_path / "pcache"), protocol=proto, work=str(work), cbin="",
                                      models="N", workers=1, compiler="gcc"),
                   readers=rd, grad_fn=TP.twin_grad, log=lambda s: None)
    out = tmp_path / "t2" / "clannad"
    args = argparse.Namespace(cand=f"det:cand={candb}", decode="", out=str(out), runs=str(tmp_path / "runs"), cbin="",
                              models="N", workers=1, cache=str(tmp_path / "p2cache"),
                              seed_from=str(tmp_path / "pcache"), compiler="gcc", protocol=proto,
                              protocol2=str(T2E.PROTOCOL2), stored=str(ts_out / "private"),
                              record=str(ts_out / "clannad.json"), work=str(work))
    res = TP2.cmd_clannad(args, readers=rd, grad_fn=TP.twin_grad, log=lambda s: None)
    assert res["stored"]["reproduces_record"] and res["stored"]["differences"] == []
    assert set(res["arms"]) == {"cand_det"} and sorted(p.name for p in (out / "private").iterdir()) == ["cand_det"]
    dl = res["scopes"]["dialogue"]
    assert set(dl["pairs"]) == {f"cand_det/VOCR5-N:{b}/VOCR5-N" for b in TP2.STORED_ARMS}
    assert dl["lines"] < res["scopes"]["all"]["lines"] and res["counts"]["no_text_pages"] == 1
    TS.assert_public(json.loads((out / "clannad.json").read_text()))
    with pytest.raises(FileExistsError):
        TP2.cmd_clannad(args, readers=rd, grad_fn=TP.twin_grad, log=lambda s: None)


def test_cand_spec_forms(tmp_path, fake_blobs):
    refb, candb = fake_blobs
    runs = tmp_path / "runs"
    runs.mkdir()
    write_run(runs, "data_h2_s4p_c8_seed2", vtsp("s4p_c8", 2))
    s, v = TP2.cand_spec("T2:data_h2_s4p_c8_seed2", "det+refine", runs)
    assert (s.name, s.decode, v) == ("T2_data_h2_s4p_c8_seed2_detref", "det+refine", 1)
    s, v = TP2.cand_spec(f"detref:x={candb}")
    assert (s.name, v) == ("x_detref", 1)
    for bad in ("T2:data_h2_s4p_c8_seed2", "B:h2_s4_c24n4_seed2", f"det{candb}"):
        with pytest.raises(SystemExit):
            TP2.cand_spec(bad, None, runs)
