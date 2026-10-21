"""Tests of textspot3's scoring wrapper (eval/textspot3.py) and its protected read (eval/textspot3_protected.py): the
run names and the registered shapes, the VTSP version dispatch (version 2 through textspot2's path, version 3
through a fake twin and, once textspot3-runtime's files exist, the real twin and C driver), the cache seed, finished
runs, scoring on synthetic frames, the median-seed grouping, the eligibility of a T and an M shape,
textspot3_rule.apply_rule3 called through select on synthetic units, the recall table, UI recall against bytes, the
bought pairs, the stage-2 inputs, and the protected script's once-only refusal on pseudo frames. Synthetic frames,
random int8 networks and fake readers only: no screenshots, no models, no network (CI).
"""
import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import types
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
import textspot3 as T3E  # noqa: E402
import textspot3_protected as TP3  # noqa: E402
import textspot3_rule as R  # noqa: E402
import textspot_metrics as M  # noqa: E402
import textspot_protected as TP  # noqa: E402

import textdet_export as E  # noqa: E402
import textdet_model as TM  # noqa: E402
import textspot2_export as X  # noqa: E402
import textspot2_shapes as S2  # noqa: E402
import textspot2_twin as TT2  # noqa: E402
import textspot_twin as S  # noqa: E402

PROTO3 = json.loads(T3E.PROTOCOL3.read_text(encoding="utf-8"))
T_SHAPE = "s8_c24n4x2"                       # textspot2's d2s shape, registered here as a family-T shape
M_SHAPES = {"mS": 100000, "mM": 250000, "mL": 500000}
NBYTES = {T_SHAPE: 7416, "s4p_c8": 2732, **M_SHAPES}
M_MACS = {"mS": 40_000_000, "mM": 90_000_000, "mL": 160_000_000, "s4p_c8": 17_886_720}
_BLOBS = {}


@pytest.fixture(autouse=True)
def _model_shapes_restored():
    """train/textdet_model.SHAPES as it was (textspot2's register() adds its shapes)."""
    saved = dict(TM.SHAPES)
    yield
    TM.SHAPES.clear()
    TM.SHAPES.update(saved)


PROTO2 = json.loads(T2E.PROTOCOL2.read_text(encoding="utf-8"))


def t_entry():
    return dict(PROTO2["shapes"][T_SHAPE], family="T")


def fake_shapes(points=True):
    """A registry: textspot2's s8_c24n4x2 as family T, three family-M shapes (with or without a point field)."""
    out = {T_SHAPE: t_entry()}
    for (n, _), p in zip(M_SHAPES.items(), T3E.POINTS):
        out[n] = {"family": "M", "stride": 8, "head": "ms_x2", **({"point": p} if points else {})}
    return out


@pytest.fixture()
def proto3_file(tmp_path, monkeypatch):
    """textspot3's protocol with fake registered shapes, as the module's PROTOCOL3."""
    doc = json.loads(json.dumps(PROTO3))
    doc["shapes"] = fake_shapes()
    p = tmp_path / "textspot3_protocol.json"
    p.write_text(json.dumps(doc))
    monkeypatch.setattr(T3E, "PROTOCOL3", p)
    return p


class FakeModel3:
    """The cost functions train/textspot3_model.py gives (exact MACs per frame size, VTSP bytes; textspot3_shapes
    .costs calls costs(shape, sizes)) for the fake M shapes: the MACs of the 480 x 272 network input."""

    @staticmethod
    def costs(shape, sizes):
        assert all(w <= 960 and h <= 544 for w, h in sizes)
        return {"macs": {f"{w}x{h}": M_MACS[shape["name"]] for w, h in sizes}, "vtsp_bytes": NBYTES[shape["name"]]}


class FakeModel3Cost:
    """Only cost(shape, w, h) at the network's input."""

    @staticmethod
    def cost(shape, w, h):
        assert (w, h) == (480, 272)
        return {"macs": M_MACS[shape["name"]], "vtsp_bytes": NBYTES[shape["name"]]}


def vtdt(shape="s4p_c8", seed=1):
    if (shape, seed) not in _BLOBS:
        S2.register()
        _BLOBS[shape, seed] = E.pack(*E.random_int_layers(shape, seed))
    return _BLOBS[shape, seed]


def vtsp(shape="s4p_c8", seed=1, half_above=600):
    return S.pack(vtdt(shape, seed), half_above, 1)


def as_version(blob, v):
    """A blob relabelled as VTSP version v (the header's version field and CRC)."""
    b = bytearray(blob)
    struct.pack_into("<I", b, 4, v)
    struct.pack_into("<I", b, 12, zlib.crc32(bytes(b[16:])) & 0xFFFFFFFF)
    return bytes(b)


def d2s_blob(seed=1, half_above=480):
    return X.random_blob(T_SHAPE, seed, half_above=half_above)[0]


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


def write_run(runs, name, blob, listed=True, steps=30000, extras=None):
    d = runs / name
    d.mkdir(parents=True)
    (d / "final.vtsp").write_bytes(blob)
    sha = hashlib.sha256(blob).hexdigest()
    (d / "done.json").write_text(json.dumps({"steps": steps, "vtsp_bytes": len(blob), "vtsp_sha256": sha}))
    if extras is not None:
        (d / "run.json").write_text(json.dumps({"recipe": name.split("_")[0], "extras": extras}))
    if listed:
        with open(runs / "finished.txt", "a") as f:
            f.write(f"{name} {sha} {len(blob)} {steps} 1.0\n")
    return d


def fake_twin3(calls):
    """A stand-in for synth/textspot3_twin: a version-3 blob decoded as its version-1 relabelling."""
    mod = types.ModuleType("textspot3_twin")

    class Model:
        def __init__(self, blob):
            if T3E.blob_version(blob) != 3:
                raise ValueError("version 3 only")
            self.inner = S.Model(as_version(blob, 1))
            for k in ("name", "stride", "half_above", "conf_min", "min_cells", "blob_bytes", "net"):
                setattr(self, k, getattr(self.inner, k))
            self.half_above_h = 272 if self.half_above == 480 else 0

    def spot(model, img):
        calls.append(model.inner.name)
        return S.spot(model.inner, img)

    mod.Model, mod.spot = Model, spot
    return mod


@pytest.fixture()
def twin3(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "textspot3_twin", fake_twin3(calls))
    monkeypatch.setattr(T3E, "TWIN3", T3E.PROTOCOL3)               # "exists": the fake module is imported
    return calls


# ----------------------------------------------------------------------------- names, shapes, stage 1

def test_run_names_and_registry(proto3_file, monkeypatch):
    reg = T3E.registry()
    assert reg[T_SHAPE]["family"] == "T" and reg["mM"]["family"] == "M" and reg["mS"]["name"] == "mS"
    assert T3E.parse_run(f"base_{T_SHAPE}_seed2") == {"run": f"base_{T_SHAPE}_seed2", "recipe": "base",
                                                      "shape": T_SHAPE, "seed": 2, "arm_id": f"base_{T_SHAPE}",
                                                      "family": "T"}
    r = T3E.parse_run("kd+size_mS_seed3")
    assert (r["recipe"], r["shape"], r["family"], r["arm_id"]) == ("kd+size", "mS", "M", "kd+size_mS")
    for bad in ("base_s4p_c8_seed1", "base_mS", "Base_mS_seed1", "base_mX_seed1", "base__seed1"):
        with pytest.raises(ValueError):
            T3E.parse_run(bad)
    assert T3E.size_points(reg) == ({"S": "mS", "M": "mM", "L": "mL"}, "the registry's point field")
    plain = {k: {kk: vv for kk, vv in v.items() if kk != "point"} for k, v in reg.items()}
    assert T3E.size_points(plain, {"mS": 3, "mM": 1, "mL": 2}) == ({"S": "mM", "M": "mL", "L": "mS"},
                                                                   "VTSP-byte order")
    monkeypatch.setattr(T3E, "m_model", lambda: (_ for _ in ()).throw(SystemExit("no model")))
    assert T3E.size_points(plain)[0] == {}                       # no bytes known and no cost function
    names = T3E.stage1_runs(reg, T3E.size_points(reg)[0])
    assert len(names) == 3 * 7 and len(set(names)) == 21
    assert {n.rsplit("_seed", 1)[0] for n in names} == {f"base_{T_SHAPE}", "base_mS", "base_mM", "base_mL",
                                                        "size_mM", "clean_mM", "long_mM"}
    bad = json.loads(proto3_file.read_text())
    bad["shapes"]["odd"] = {"head": "kroa"}
    with pytest.raises(SystemExit):
        T3E.registry(bad)
    am = json.loads(proto3_file.read_text())                      # an amendment's own shapes object; a list: names
    am["amendments"] = [{"shapes": {"mX": {"family": "M", "head": "ms_x2"}}}, {"shapes": ["mX"]},
                        {"shapes": {"tX": dict(t_entry(), half_above=None) | {"half_above": 480}}}]
    reg2 = T3E.registry(am)
    assert reg2["mX"]["family"] == "M" and reg2["tX"]["half_above"] == 480 and set(reg) < set(reg2)
    no_half = {k: v for k, v in t_entry().items() if k != "half_above"}
    assert T3E.registry({"shapes": {"tY": no_half}})["tY"]["half_above"] == 480


def test_the_registered_t_shapes_cost_as_their_pinned_rows():
    """Every T shape the real protocol registers: the wrapper's MACs are train/textspot3_shapes.costs' and the
    pinned A9 table's (eval/textspot3_a9_t_20261021.json, amendment 1), and the table is the pinned file."""
    import textspot3_shapes as S3
    reg = T3E.registry(PROTO3)
    t = {n: s for n, s in reg.items() if s["family"] == "T"}
    if not t:
        pytest.skip("no T shape registered")
    table = ROOT / "eval" / "textspot3_a9_t_20261021.json"
    T3E.check_pinned(table, PROTO3)
    rows = {r["shape"]: r for r in json.loads(table.read_text())["summary"] if r["half_above"] == 480}
    for n, s in t.items():
        assert s["half_above"] == 480
        assert T3E.macs_of(n, reg)[0] == S3.costs(n)["macs"]["960x544"] == rows[n]["macs_960"]
        assert S3.costs(n)["vtsp_bytes"] == rows[n]["vtsp_bytes"]


def test_m_input_and_macs(proto3_file, monkeypatch):
    assert T3E.m_input(960, 544) == (480, 272) and T3E.m_input(480, 272) == (480, 272)
    assert T3E.m_input(728, 272) == (364, 136) and T3E.m_input(480, 300) == (240, 150)
    reg = T3E.registry()
    macs, src = T3E.macs_of(T_SHAPE, reg)
    assert (macs, src) == (T2E.d2s_macs(T_SHAPE, 600), "train/textspot2_export.cost")   # 960 x 544 -> 480 x 272
    monkeypatch.setattr(T3E, "m_model", lambda: FakeModel3)
    assert T3E.macs_of("mL", reg) == (M_MACS["mL"], "train/textspot3_model.costs")
    assert T3E.m_vtsp_bytes(reg["mM"]) == 250000
    monkeypatch.setattr(T3E, "m_model", lambda: FakeModel3Cost)
    assert T3E.macs_of("mL", reg) == (M_MACS["mL"], "train/textspot3_model.cost")
    plain = {k: {kk: vv for kk, vv in v.items() if kk != "point"} for k, v in reg.items()}
    assert T3E.size_points(plain) == ({"S": "mS", "M": "mM", "L": "mL"}, "VTSP-byte order")


# ----------------------------------------------------------------------------- the version dispatch

def test_blob_versions_and_dispatch(tmp_path, twin3):
    v1 = vtsp("s4p_c8", 3)
    v2, v3 = d2s_blob(3), as_version(v1, 3)
    assert [T3E.blob_version(b) for b in (v1, v2, v3)] == [1, 2, 3]
    g = synthetic_frame(seed=5)
    f = frame_of(g)
    s1 = TS.Spec("a_det", "det", blob=v1, half_above=600)
    s2 = TS.Spec("b_det", "det", blob=v2, half_above=480)
    s3 = TS.Spec("c_det", "det", blob=v3, half_above=600)
    want1 = TS.Twin(f, TS.load_nets([s1])).arm(s1)
    want2 = TT2.spot(TT2.Model(v2), g)
    for s in (s2, s3):
        with pytest.raises(ValueError):                           # textspot's twin refuses versions 2 and 3
            s.meta()
    with T3E.dispatch({1, 2}):                                    # versions 1-2: textspot2's v2_dispatch exactly
        assert sys.modules["textspot_twin"].TEXTSPOT2_SHIM and TS.CDrivers is T2E.CDrivers2
        assert not getattr(sys.modules["textspot_twin"], "TEXTSPOT3_SHIM", False)
    assert sys.modules["textspot_twin"] is S and TS.CDrivers is not T2E.CDrivers2
    with T3E.dispatch({1}):
        assert sys.modules["textspot_twin"] is S                  # version 1 alone: textspot's path
    with T3E.dispatch({1, 2, 3}):
        assert sys.modules["textspot_twin"].TEXTSPOT3_SHIM and TS.CDrivers is T3E.CDrivers3
        with T3E.dispatch({3}):                                   # nested: unchanged
            assert TS.CDrivers is T3E.CDrivers3
        with T2E.v2_dispatch(True):                               # textspot2's own dispatch inside: a no-op
            assert TS.CDrivers is T3E.CDrivers3
        assert s3.meta()["shape"] == "s4p_c8" and s2.meta()["shape"] == T_SHAPE
        nets = TS.load_nets([s1, s2, s3])
        tw = TS.Twin(f, nets)
        a1, a2, a3 = tw.arm(s1), tw.arm(s2), tw.arm(s3)
        assert twin3 == ["s4p_c8"]                                # only the version-3 blob went to textspot3_twin
        assert a3["boxes"] == a1["boxes"] == want1["boxes"] and a3["pick"] == want1["pick"]
        assert a2["boxes"] == [(b.x, b.y, b.w, b.h, b.vertical, b.conf, b.dark, b.block) for b in want2.boxes]
        assert T3E.CDrivers3(tmp_path).spot([g], v3) is None      # no version-3 driver built: unchecked
        assert T3E.CDrivers3(tmp_path).spot([g], v2) is None      # nor a version-2 one
    assert sys.modules["textspot_twin"] is S and TS.CDrivers is not T3E.CDrivers3
    with T2E.v2_dispatch(True):
        with pytest.raises(RuntimeError):                         # a version-3 batch never nests in textspot2's
            with T3E.dispatch({3}):
                pass


def test_v3_needs_the_runtime_twin(monkeypatch, tmp_path):
    monkeypatch.setattr(T3E, "TWIN3", tmp_path / "missing.py")
    with pytest.raises(SystemExit):
        T3E.model_of(as_version(vtsp(), 3))
    assert T3E.model_of(vtsp()).name == "s4p_c8" and T3E.model_of(d2s_blob()).name == T_SHAPE


def test_driver_spot_batch_protocol(tmp_path):
    """CDrivers3's version-3 batch reads textspot's driver protocol (here a script standing in for the driver)."""
    drv = tmp_path / T3E.DRIVER3
    drv.write_text("#!/bin/sh\necho '0 2 1 0 0 1 2 3 4 0 50 1 0 5 6 7 8 1 60 0 0 1 2 3 4 0 0 2'\n")
    drv.chmod(0o755)
    got = T3E.CDrivers3(tmp_path).spot([np.zeros((4, 4), np.uint8)], as_version(vtsp(), 3))
    assert got == [([(1, 2, 3, 4, 0, 50, 1, 0), (5, 6, 7, 8, 1, 60, 0, 0)], [(1, 2, 3, 4, 0, 0, 2)], 0, 0)]
    drv.write_text("#!/bin/sh\necho '3 0 0 -1 0'\n")
    with pytest.raises(RuntimeError):
        T3E.CDrivers3(tmp_path).spot([np.zeros((4, 4), np.uint8)], as_version(vtsp(), 3))


# ----------------------------------------------------------------------------- the cache seed and write guard

def test_seed_cache_once_and_read_only_sources(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.jsonl").write_text('{"k": "x", "t": "y"}\n')
    (src / "b.jsonl").write_text('{"k": "z", "t": "w"}\n')
    (src / "SEEDED.json").write_text("{}")                           # textspot2's own marker is not copied
    dst = tmp_path / "dst"
    rec = T3E.seed_cache(dst, src)
    assert sorted(rec["files"]) == ["a.jsonl", "b.jsonl"] and rec["kind"] == "textspot3-cache-seed"
    assert TS.Cache(dst).get("z") == "w"
    (src / "a.jsonl").write_text('{"k": "x", "t": "changed"}\n')
    assert T3E.seed_cache(dst, src) == rec                           # seeded once: later calls change nothing
    assert TS.Cache(dst).get("x") == "y"
    for name in ("T", "P", "T2", "P2"):
        monkeypatch.setattr(T3E, name, tmp_path / name.lower())
        with pytest.raises(SystemExit):                              # never written under textspot's or textspot2's
            T3E.seed_cache(tmp_path / name.lower() / "eval" / "cache", src)
        with pytest.raises(SystemExit):
            T3E.check_writable(tmp_path / name.lower() / "x.json")
    T3E.check_writable(tmp_path / "t3" / "x.json")
    with pytest.raises(SystemExit):
        T3E.seed_cache(tmp_path / "dst2", tmp_path / "empty")
    assert T3E.T2_CACHE == T2E.T2 / "eval" / "cache" and T3E.DEFAULT["cache"] == T3E.T3 / "eval" / "cache"


# ----------------------------------------------------------------------------- finished runs and scoring

def test_finished_runs_and_specs(tmp_path, proto3_file, twin3):
    runs = tmp_path / "runs"
    runs.mkdir()
    write_run(runs, f"base_{T_SHAPE}_seed1", d2s_blob(1), extras=[])
    write_run(runs, f"base_{T_SHAPE}_seed2", d2s_blob(2), listed=False)            # not in finished.txt yet
    write_run(runs, "probe_mS", d2s_blob(4))                                        # not a run name: skipped
    found, skipped = T3E.finished_runs(runs)
    assert sorted(found) == [f"base_{T_SHAPE}_seed1"] and skipped == ["probe_mS"]
    r = found[f"base_{T_SHAPE}_seed1"]
    assert (r["recipe"], r["shape"], r["family"], r["seed"], r["version"], r["half_above"], r["extras"]) == \
        ("base", T_SHAPE, "T", 1, 2, 480, [])
    specs = T3E.run_specs(found)
    assert [s.name for s in specs] == [f"T3_base_{T_SHAPE}_seed1_det", f"T3_base_{T_SHAPE}_seed1_detref"]
    with T3E.dispatch({2}):
        m = specs[0].meta()
    assert (m["family"], m["run"], m["decode"], m["spec"], m["half_above"]) == \
        ("T3", f"base_{T_SHAPE}_seed1", "det", f"T3:base_{T_SHAPE}_seed1", 480)
    with pytest.raises(SystemExit):
        T3E.finished_runs(runs, [f"base_{T_SHAPE}_seed2"])                          # asked for, but not listed
    (runs / "finished.txt").unlink()
    write_run(runs, "kd_mS_seed1", as_version(vtsp("s4p_c8", 5), 3), listed=False, extras="size,clean")
    with pytest.raises(SystemExit):                                                 # an M blob named s4p_c8
        T3E.finished_runs(runs)
    shutil.rmtree(runs / "kd_mS_seed1")
    write_run(runs, "base_mS_seed1", d2s_blob(5), listed=False)                     # an M run with a v2 blob
    with pytest.raises(SystemExit):
        T3E.finished_runs(runs)
    shutil.rmtree(runs / "base_mS_seed1")
    write_run(runs, f"base_{T_SHAPE}_seed3", d2s_blob(3, half_above=600), listed=False)   # T at half_above 600
    with pytest.raises(SystemExit):
        T3E.finished_runs(runs)
    shutil.rmtree(runs / f"base_{T_SHAPE}_seed3")
    d = write_run(runs, f"base_{T_SHAPE}_seed4", d2s_blob(4), listed=False)
    (d / "run.json").write_text(json.dumps({"family": "M", "extras": []}))         # run.json: another family
    with pytest.raises(SystemExit):
        T3E.finished_runs(runs)
    shutil.rmtree(d)
    found, _ = T3E.finished_runs(runs)
    assert sorted(found) == [f"base_{T_SHAPE}_seed1", f"base_{T_SHAPE}_seed2"]
    (runs / f"base_{T_SHAPE}_seed1" / "final.vtsp").write_bytes(d2s_blob(7))        # not done.json's blob
    with pytest.raises(SystemExit):
        T3E.finished_runs(runs)


def test_run_settings(tmp_path):
    assert T3E.run_settings(tmp_path) == {"extras": [], "run_family": None}
    (tmp_path / "run.json").write_text(json.dumps({"extras": "size,clean", "family": "M"}))
    assert T3E.run_settings(tmp_path) == {"extras": ["clean", "size"], "run_family": "M"}
    (tmp_path / "run.json").write_text(json.dumps({"settings": {"extras": ["size"]}}))
    assert T3E.run_settings(tmp_path)["extras"] == ["size"]
    (tmp_path / "run.json").write_text(json.dumps({"extras": "none"}))
    assert T3E.run_settings(tmp_path)["extras"] == []


def test_score_runs_v2_and_v3_on_synthetic_frames(tmp_path, proto3_file, twin3, monkeypatch):
    """A T run (VTSP version 2 at half_above 480, the real d2s twin) and an M run (version 3, the fake twin) in one
    batch: boxes, units, sidecars; the T arm's boxes are textspot2_twin's own."""
    doc = json.loads(proto3_file.read_text())
    doc["shapes"]["s4p_c8"] = {"family": "M", "stride": 8, "head": "ms_x2"}          # the fake M blob's shape
    proto3_file.write_text(json.dumps(doc))
    runs = tmp_path / "runs"
    runs.mkdir()
    tb = d2s_blob(1)
    write_run(runs, "base_s4p_c8_seed2", as_version(vtsp("s4p_c8", 2), 3))             # M at half_above 600
    with pytest.raises(SystemExit):                                                 # family M's scale is 480 x 272
        T3E.finished_runs(runs)
    shutil.rmtree(runs / "base_s4p_c8_seed2")
    (runs / "finished.txt").unlink()
    write_run(runs, f"base_{T_SHAPE}_seed1", tb)
    write_run(runs, "base_s4p_c8_seed1", as_version(vtsp("s4p_c8", 1, half_above=480), 3))
    found, _ = T3E.finished_runs(runs)
    assert {r["version"] for r in found.values()} == {2, 3} and found["base_s4p_c8_seed1"]["half_above"] == 480
    frames = [frame_of(synthetic_frame(seed=i), fid=f"f{i}", vn=f"v90000{i}") for i in range(3)]
    manifest = {"set": "pseudo", "path": "x", "sha256": "p" * 64, "frames": 3, "vns": 3}
    out, ids = tmp_path / "units", [f.id for f in frames]
    todo = T3E.pending(out, found, manifest, ids, ["VOCR5-N"], None)
    assert len(todo) == 4
    rd, cache = fake_readers(tmp_path), TS.Cache(tmp_path / "cache")
    T3E.score_runs(frames, manifest, found, todo, out, ["VOCR5-N"], rd, cache, None, 1, log=lambda s: None)
    assert sys.modules["textspot_twin"] is S                                        # the dispatch is undone
    bd = json.loads((out / f"T3_base_{T_SHAPE}_seed1_det" / "boxes.json").read_text())
    for f in frames:
        r = TT2.spot(TT2.Model(tb), f.grey)
        assert bd["frames"][f.id]["boxes"] == [[b.x, b.y, b.w, b.h, b.vertical, b.conf, b.dark, b.block]
                                               for b in r.boxes]
    side = json.loads((out / "T3_base_s4p_c8_seed1_detref" / "textspot3.json").read_text())
    assert (side["family"], side["blob_version"], side["arm_id"], side["decode"]) == ("M", 3, "base_s4p_c8",
                                                                                     "det+refine")
    assert side["twin_sha256"] == TS.sha256(T3E.TWIN3)
    u = json.loads((out / f"T3_base_{T_SHAPE}_seed1_detref" / "units_VOCR5-N.json").read_text())
    assert u["meta"]["spec_meta"]["family"] == "T3" and len(u["units"]) == 3
    assert T3E.pending(out, found, manifest, ids, ["VOCR5-N"], None) == []          # scored: skipped next time
    assert len(T3E.pending(out, found, manifest, ids, ["VOCR5-N", "VOCR5-S"], None)) == 4
    monkeypatch.setattr(T3E, "twin_sha", lambda v: "changed" if v == 3 else T3E.TS.sha256(ROOT / "synth" /
                                                                                         "textspot2_twin.py"))
    assert len(T3E.pending(out, found, manifest, ids, ["VOCR5-N"], None)) == 2      # another v3 twin: re-score
    doc = T3E.record_frames(tmp_path / "frames.json", frames, manifest)
    assert doc["frames"]["f1"]["sha"] == frames[1].sha and doc["kind"] == "textspot3-frames"
    with pytest.raises(SystemExit):                                                 # other pixels, same frame id
        T3E.record_frames(tmp_path / "frames.json", [frame_of(synthetic_frame(seed=9), fid="f1")], manifest)


# ----------------------------------------------------------------------------- a synthetic development world

MAN, RN, RV = "m" * 64, "n" * 64, "v" * 64
W, H = 960, 544
DLG, NAME, UI = "あいうえおかきくけこ", "名前", "セーブ"
REFS = [{"box": [100, 400, 300, 24], "cls": "dialogue", "text": DLG, "vertical": False, "ignore": False},
        {"box": [100, 360, 80, 20], "cls": "name", "text": NAME, "vertical": False, "ignore": False},
        {"box": [800, 20, 60, 16], "cls": "ui", "text": UI, "vertical": False, "ignore": False}]
# readings of the dialogue, name and UI references (None: not found) and the box shift of the quality (so that one
# box always has one reading); all-text errors per frame of 15 characters: base 10, ref 4 (dialogue 2), perfect 0,
# one 1, two 2 (dialogue 1), noui 3 (the UI label missed)
QUALITY = {"base": ((DLG[:5], None, None), (0, 0)), "ref": ((DLG[:8], NAME, UI[:1]), (1, 0)),
           "perfect": ((DLG, NAME, UI), (0, 1)), "one": ((DLG, NAME, UI[:2]), (1, 1)),
           "two": ((DLG[:9], NAME, UI[:2]), (2, 1)), "noui": ((DLG, NAME, None), (3, 1))}
A9 = {"summary": [{"shape": T_SHAPE, "scale": "h2", "half_above": 480, "vtsp_bytes": 7416,
                   "macs_960": T2E.d2s_macs(T_SHAPE, 600), "ms_333_960": 73.19, "arena_up_to_960": 79647,
                   "ms_333_480": 63.59, "arena_up_to_480": 66671}]
      + [{"shape": n, "scale": "h2", "half_above": 480, "vtsp_bytes": b, "macs_960": M_MACS[n], "arena_up_to_480": 1,
          "ms_333_960": {"mS": 150.0, "mM": 220.0, "mL": 299.0}[n],
          "arena_up_to_960": {"mS": 150000, "mM": 200000, "mL": 250000}[n]} for n, b in M_SHAPES.items()]}
RULE = PROTO3["rule"]


def world(n_vns=10, per_vn=3):
    rows = [{"id": f"f{v}_{k}", "vn": f"v{v:03d}", "part": "vita" if k else "psp", "dims": [W, H],
             "text_free": False, "refs": REFS} for v in range(n_vns) for k in range(per_vn)]
    return rows, {r["id"]: hashlib.sha256(r["id"].encode()).hexdigest() for r in rows}


def make_arm(d, name, meta, rows, shas, cache, quality, h_in=None, side=None):
    """An arm directory as eval/textspot.py writes one (boxes.json, units_VOCR5-N.json; textspot3.json when side):
    its units are frame_unit's for its boxes and readings, and the readings go into the cache."""
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
    if side is not None:
        (a / "textspot3.json").write_text(json.dumps(side))
    return a


def blob_sha(run):
    return hashlib.sha256(run.encode()).hexdigest()


def t3_arms(units, run, quality, rows, shas, cache, extras=()):
    info = T3E.parse_run(run, T3E.registry(dict(PROTO3, shapes=fake_shapes())))
    for kind in T3E.KINDS:
        name = f"T3_{run}_{kind}"
        meta = {"arm": name, "spec": f"T3:{run}", "kind": kind, "baseline": None, "decode": T3E.DECODE[kind],
                "family": "T3", "run": run, "blob_sha256": blob_sha(run), "blob_bytes": NBYTES[info["shape"]],
                "shape": info["shape"], "stride": 4, "half_above": 480, "conf_min": 1, "source": None}
        side = {"run": run, "blob_sha256": blob_sha(run), "blob_version": 2 if info["family"] == "T" else 3,
                "extras": list(extras)}
        make_arm(units, name, meta, rows, shas, cache, quality, 3 if kind == "detref" else None, side)


PLAN = {f"base_{T_SHAPE}": ("two", "one", "perfect"), "base_mS": ("perfect",) * 3, "base_mM": ("perfect",) * 3,
        "base_mL": ("perfect",) * 3, "size_mM": ("one",) * 3, "clean_mM": ("perfect",) * 3,
        f"kd_{T_SHAPE}": ("perfect",) * 3, "size+clean_mL": ("perfect",) * 3}


def select_world(tmp_path, plan=PLAN):
    rows, shas = world()
    cache = TS.Cache(tmp_path / "cache")
    units = tmp_path / "units"
    for aid, qs in plan.items():
        for seed, q in enumerate(qs, 1):
            t3_arms(units, f"{aid}_seed{seed}", q, rows, shas, cache,
                    ("clean", "size") if aid.startswith("size+clean") else ())
    base_doc = {"manifest": {"sha256": MAN}, "arms": {}}
    for b in TS.BASELINES:
        a = make_arm(tmp_path / "baseline", b, {"kind": "baseline", "arm": b}, rows, shas, cache, "base")
        base_doc["arms"][b] = {"VOCR5-N": {"units": str(a / "units_VOCR5-N.json"),
                                           "units_sha256": TS.sha256(a / "units_VOCR5-N.json"),
                                           "boxes_sha256": TS.sha256(a / "boxes.json")}}
    proto3 = json.loads(json.dumps(PROTO3))
    proto3["shapes"] = fake_shapes()
    proto3["reused"]["dev_manifest"]["sha256"] = MAN
    name = proto3["references"]["ref"]["name"]
    d = make_arm(tmp_path / "textspot" / "famB", name, {"kind": "detref", "arm": name}, rows, shas, cache, "ref", 3)
    proto3["references"]["ref"].update(units=str(d / "units_VOCR5-N.json"), boxes=str(d / "boxes.json"),
                                       units_sha256=TS.sha256(d / "units_VOCR5-N.json"),
                                       boxes_sha256=TS.sha256(d / "boxes.json"))
    return rows, shas, cache, units, base_doc, proto3


def test_median_seed_grouping(tmp_path):
    rows, shas = world(5, 2)
    cache = TS.Cache(tmp_path / "cache")
    units = tmp_path / "units"
    for seed, q in ((1, "two"), (2, "one"), (3, "perfect")):
        t3_arms(units, f"base_{T_SHAPE}_seed{seed}", q, rows, shas, cache)
    for seed, q in ((1, "perfect"), (3, "two")):                      # seed 2 crashed: two finished seeds
        t3_arms(units, f"base_mS_seed{seed}", q, rows, shas, cache)
    reg = T3E.registry(dict(PROTO3, shapes=fake_shapes()))
    seeds = T3E.t3_seeds([units], reg=reg)
    assert len(seeds) == 10 and {s["source"] for s in seeds} == {"textspot3"}
    reps = T3E.representatives(seeds)
    g = reps[f"base_{T_SHAPE}", "det"]
    # dialogue CER: perfect 0, one 0 (all-text 1 a frame), two 1 a frame: perfect, one, two; the median is "one"
    assert g["order"] == [3, 2, 1] and g["index"] == 1 and g["rep"]["seed"] == 2
    g = reps["base_mS", "det+refine"]
    assert g["order"] == [1, 3] and g["rep"]["seed"] == 3                       # two seeds: the worse one
    rep, info = R.median_seed([{k: s[k] for k in ("seed", "cer_all", "cer_dialogue", "f1")} for s in g["seeds"]])
    assert (rep["seed"], info["order"]) == (3, [1, 3])                          # the rule's own function agrees
    side = units / "T3_base_mS_seed1_det" / "textspot3.json"
    s = json.loads(side.read_text())
    side.write_text(json.dumps(dict(s, blob_sha256="x" * 64)))
    with pytest.raises(SystemExit):                                             # a sidecar of another blob
        T3E.t3_seeds([units], reg=reg)
    side.unlink()
    with pytest.raises(SystemExit):                                             # no sidecar
        T3E.t3_seeds([units], reg=reg)


# ----------------------------------------------------------------------------- eligibility

def test_eligibility_of_a_t_and_an_m_shape(monkeypatch):
    monkeypatch.setattr(T3E, "m_model", lambda: FakeModel3)
    reg = T3E.registry(dict(PROTO3, shapes=fake_shapes()))
    rows = T3E.a9_rows3(A9)
    frames = {f"f{i}": {"c_equal": True, "h_in": 4 + i % 3, "part": "vita" if i < 5 else "psp"} for i in range(8)}
    bd = {"arm": "x", "frames": frames}
    meta = {"shape": T_SHAPE, "half_above": 480, "blob_bytes": 7416, "blob_sha256": "b" * 64, "decode": "det+refine"}
    e = T3E.eligibility3(meta, bd, rows, 72391, reg)
    # the median of the Vita frames' horizontal boxes (4, 5, 6, 4, 5) is 5: + 7.0 ms of refinement
    assert (e["macs_960"], e["est_ms_333_960"], e["arena_960"], e["c_equal"], e["family"]) == \
        (T2E.d2s_macs(T_SHAPE, 600), 80.19, 79647, True, "T")
    assert e["macs_source"] == "train/textspot2_export.cost" and R.eligible(T3E.light(e))
    d = T3E.eligibility3(dict(meta, decode="det"), bd, rows, 72391, reg)
    assert (d["est_ms_333_960"], d["arena_960"]) == (73.19, 79647)
    assert T3E.report_costs(d, rows) == {"est_ms_444_960": round(73.19 * 333 / 444, 3), "ms_333_480_spot": 63.59,
                                         "ms_444_480_spot": round(63.59 * 333 / 444, 3), "arena_up_to_480": 66671}
    m = T3E.eligibility3(dict(meta, shape="mL", blob_bytes=500000), bd, rows, 72391, reg)
    assert (m["macs_960"], m["est_ms_333_960"], m["arena_960"], m["family"]) == (M_MACS["mL"], 306.0, 250000, "M")
    assert not R.eligible(T3E.light(m))                                          # 299 + 7 ms: over 300 ms
    m = T3E.eligibility3(dict(meta, shape="mL", blob_bytes=500000, decode="det"), bd, rows, 72391, reg)
    assert R.eligible(T3E.light(m)) and m["est_ms_333_960"] == 299.0
    assert TS.macs_960.__name__ == "macs_960"                                    # the harness is restored
    with pytest.raises(SystemExit):                                              # the table's MACs are checked
        T3E.eligibility3(dict(meta, shape="mS", blob_bytes=100000), bd,
                         T3E.a9_rows3({"summary": [dict(A9["summary"][1], macs_960=1)]}), 72391, reg)
    with pytest.raises(SystemExit):                                              # and its VTSP bytes
        T3E.eligibility3(dict(meta, shape="mS", blob_bytes=100001), bd, rows, 72391, reg)
    none = T3E.eligibility3(dict(meta, shape="mM", blob_bytes=250000), bd, T3E.a9_rows3({}), 72391, reg)
    assert none["est_ms_333_960"] is None and not R.eligible(T3E.light(none))     # no A9 row: never eligible
    frames["f0"]["c_equal"] = False
    assert not R.eligible(T3E.light(T3E.eligibility3(meta, bd, rows, 72391, reg)))
    assert T2E.refine_arena2(None, None, PROTO3)[0] == 72391


# ----------------------------------------------------------------------------- the report's pieces

def test_spearman_ranks_and_pareto():
    assert T3E.ranks([3, 1, 2, 2]) == [4.0, 1.0, 2.5, 2.5]
    assert T3E.spearman([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.5]) == 1.0
    assert T3E.spearman([1, 2, 3, 4], [0.4, 0.3, 0.2, 0.1]) == -1.0
    assert T3E.spearman([1, 2], [1, 2]) is None and T3E.spearman([1, 2, 3], [5, 5, 5]) is None
    p = {"a": {"blob_bytes": 1, "est_ms_333_960": 1, "est_ms_444_960": 1, "arena_960": 1, "f1": 0.5, "cer_all": 30,
               "cer_dialogue": 2, "ui_recall": 0.3},
         "b": {"blob_bytes": 2, "est_ms_333_960": 1, "est_ms_444_960": 1, "arena_960": 1, "f1": 0.5, "cer_all": 30,
               "cer_dialogue": 2, "ui_recall": 0.3},
         "c": {"blob_bytes": 3, "est_ms_333_960": 1, "est_ms_444_960": 1, "arena_960": 1, "f1": 0.5, "cer_all": 30,
               "cer_dialogue": 2, "ui_recall": 0.6}}
    assert T3E.pareto_front(p) == ["a", "c"]                      # b: dominated by a; c: better UI recall


def test_recall_by_class_of_units(tmp_path):
    rows, shas = world(5, 2)
    cache = TS.Cache(tmp_path / "cache")
    a = make_arm(tmp_path, "x", {"kind": "det", "arm": "x"}, rows, shas, cache, "noui")
    units = json.loads((a / "units_VOCR5-N.json").read_text())
    r = T3E.recall_of(units)
    assert r["ui"] == {"tp": 0, "n": 10, "recall": 0.0} and r["dialogue"] == {"tp": 10, "n": 10, "recall": 1.0}
    assert r["choice"]["recall"] is None
    assert {c: r[c]["recall"] for c in M.CLASSES} == {c: v["recall"] for c, v in
                                                      units["summary"]["recall_by_class"].items()}


def test_select_calls_rule3_on_synthetic_units(tmp_path, monkeypatch):
    monkeypatch.setattr(T3E, "m_model", lambda: FakeModel3)
    rows, shas, cache, units, base_doc, proto3 = select_world(tmp_path)
    doc = T3E.select([units], A9, proto3, base_doc, log=lambda s: None)
    res = doc["result"]
    t = f"T3_base_{T_SHAPE}_seed2"
    assert len(doc["candidates"]) == 16 and len(doc["seeds"]) == 48
    assert res["comparator"] == proto3["references"]["ref"]["name"]
    # mL's det+refine (299 + 4.2 ms) is over the budget; everything else is eligible, in cost order
    assert res["ineligible"] == ["T3_base_mL_seed2_detref", "T3_size+clean_mL_seed2_detref"]
    assert res["eligible"][:4] == [f"{t}_det", f"T3_kd_{T_SHAPE}_seed2_det", f"{t}_detref",
                                   f"T3_kd_{T_SHAPE}_seed2_detref"]          # bytes, then est. ms
    # every candidate's dialogue is better than ref's: all qualify; the cheapest (T base, 1 error a frame) is
    # replaced by the next cheaper perfect one (kd on T), resolved better on all-text CER; nothing replaces it
    assert set(res["qualifying"]) == set(res["eligible"])
    assert res["winner"] == f"T3_kd_{T_SHAPE}_seed2_det"
    assert [(s["replaces"], s["by"]) for s in res["steps"]] == [(f"{t}_det", f"T3_kd_{T_SHAPE}_seed2_det")]
    assert res["bar"] == res["eligible"] and doc["pick"]["run"] == f"kd_{T_SHAPE}_seed2"
    assert set(doc["report_comparisons"]) == set(res["ineligible"])
    assert doc["report_comparisons"]["T3_base_mL_seed2_detref"]["qualifies_if_eligible"]
    a = doc["arms"][f"base_{T_SHAPE}|det"]
    assert (a["order"], a["index"], a["representative"], a["family"]) == ([3, 2, 1], 1, f"{t}_det", "T")
    assert doc["size_points"] == {"points": {"S": "mS", "M": "mM", "L": "mL"}, "source": "the registry's point field"}
    assert doc["representatives"][f"{t}_det"]["recall_by_class"]["ui"]["recall"] == 1.0
    assert doc["references"][res["comparator"]]["recall_by_class"]["ui"]["recall"] == 1.0
    assert set(doc["baselines"]) == set(TS.BASELINES) and doc["baselines"]["grad"]["recall_by_class"]["ui"]["tp"] == 0
    s = {x["name"]: x for x in doc["seeds"]}
    assert s[f"{t}_detref"]["est_ms_333_960"] == round(73.19 + 1.4 * 3, 3)
    assert s[f"{t}_det"]["est_ms_444_960"] == round(73.19 * 333 / 444, 3) and s[f"{t}_det"]["summary"]["frames"] == 30
    assert doc["ui_recall_vs_bytes"]["by_decode"]["det"]["rises"] is None        # one UI recall everywhere: no rank
    pts = doc["pareto"]["points"]
    assert pts[res["comparator"]]["role"] == "ref" and pts[res["comparator"]]["blob_bytes"] == 5732
    assert doc["pareto"]["front"] == [f"T3_kd_{T_SHAPE}_seed2_det"]
    rows_ = {(b["what"], b["decode"]): b for b in doc["bought"]}
    assert rows_["long M - base M", "det"]["missing"]
    b = rows_["size M - base M", "det"]
    assert (b["minuend"], b["subtrahend"]) == ("T3_size_mM_seed2_det", "T3_base_mM_seed2_det")
    assert b["comparison"]["cer_all"]["delta"] == round(100 / 15, 4)
    assert rows_["base M - base S", "det+refine"]["comparison"]["cer_all"]["delta"] == 0.0
    assert rows_[f"base S - base T {T_SHAPE}", "det"]["comparison"]["cer_all"]["delta"] == round(-100 / 15, 4)
    assert rows_[f"kd {T_SHAPE} - base {T_SHAPE}", "det"]["comparison"]["cer_dialogue"]["delta"] == 0.0
    assert rows_["size+clean L - base L (the kept extras)", "det"]["minuend"] == "T3_size+clean_mL_seed2_det"
    assert rows_["clean_mM - ref", "det"]["comparison"]["cer_dialogue"]["delta"] == -20.0
    assert rows_["pick - ref", "det"]["minuend"] == res["winner"]
    st = doc["stage2_inputs"]
    assert (st["size_kept"], st["clean_kept"], st["extras_on_L"]["runs"]) == (False, False, False)
    assert st["kd_teacher"]["name"] == "T3_base_mL_seed2_det"
    assert [x["shape"] for x in st["kd_students"]] == [T_SHAPE, "mS"] and st["kd_students"][0]["base_qualifies"]
    TS.assert_public(doc)
    n = T3E.write_compact(tmp_path / "dev.json", doc)
    assert n < T3E.MAX_RECORD and json.loads((tmp_path / "dev.json").read_text()) == json.loads(json.dumps(doc))
    md = T3E.render(json.loads((tmp_path / "dev.json").read_text()))
    assert f"**{res['winner']}**" in md and "not run" in md and "| `grad` (baseline) |" in md
    assert "No VOCR5-S / VOCR5-L units of the pick yet" in md
    w = res["winner"]
    sl_units = json.loads((units / w / "units_VOCR5-N.json").read_text())
    sl_units["summary"]["cer_all"]["cer"] = 1.5
    (units / w / "units_VOCR5-S.json").write_text(json.dumps(sl_units))
    sl = T3E.report_sl(doc, [units])
    assert list(sl) == [w] and sl[w]["VOCR5-S"]["cer_all"]["cer"] == 1.5 and "| VOCR5-S | " in T3E.render(doc, sl=sl)
    sl_units["meta"]["boxes_sha256"] = "z" * 64
    (units / w / "units_VOCR5-S.json").write_text(json.dumps(sl_units))
    with pytest.raises(SystemExit):                                       # S / L units of other boxes
        T3E.report_sl(doc, [units])
    reg = T3E.registry(proto3)
    seeds, _, _ = T3E.gather([units], proto3, A9, reg=reg)
    hl = T3E.headline_md(seeds, RULE, reg, T2E.reference(proto3, "ref")["arm"])
    assert f"`{t}_det`" in hl and "Stage 1: 18 of 21 runs scored; not yet: long_mM_seed1" in hl
    assert "Stage-2 inputs" in hl
    seeds2 = [x for x in seeds if x["shape"] != "mS"]                     # a table without a seed's row: refused
    with pytest.raises(SystemExit):
        T3E.select([units], {"summary": [r for r in A9["summary"] if r["shape"] != "mS"]}, proto3, base_doc,
                   log=lambda s: None)
    assert seeds2 and T3E.a9_rows3([A9, None, {"summary": []}]) == T3E.a9_rows3(A9)
    f = Path(proto3["references"]["ref"]["units"])
    f.write_text(f.read_text() + "\n")
    with pytest.raises(SystemExit):                                       # the protocol's sha256 of ref's units
        T3E.select([units], A9, proto3, base_doc, log=lambda s: None)
    other = tmp_path / "a9.json"
    other.write_text(json.dumps(A9))
    with pytest.raises(SystemExit):                                       # an A9 table the protocol does not pin
        T3E.check_pinned(other, PROTO3)


def test_select_no_pick_and_ui_trend(tmp_path, monkeypatch):
    """No candidate's dialogue is resolved better than ref's: no pick; and UI recall rising with bytes."""
    monkeypatch.setattr(T3E, "m_model", lambda: FakeModel3)
    plan = {f"base_{T_SHAPE}": ("ref",) * 3, "base_mS": ("ref",) * 3, "base_mM": ("ref",) * 3}
    rows, shas, cache, units, base_doc, proto3 = select_world(tmp_path, plan)
    doc = T3E.select([units], A9, proto3, base_doc, log=lambda s: None)
    assert doc["result"]["winner"] is None and doc["result"]["qualifying"] == [] and doc["pick"] is None
    assert "No qualifier: the verdict is negative" in T3E.render(doc)
    assert "No pick: no VOCR5-S / VOCR5-L report reading." in T3E.render(doc)
    st = doc["stage2_inputs"]
    assert st["kd_teacher"]["shape"] == "mM" and [x["shape"] for x in st["kd_students"]] == [T_SHAPE, "mS"]
    assert not any(x["base_qualifies"] for x in st["kd_students"])            # the smallest eligible make up two
    plan = {f"base_{T_SHAPE}": ("noui",) * 3, "base_mS": ("noui",) * 3, "base_mM": ("perfect",) * 3,
            "base_mL": ("perfect",) * 3}
    rows, shas, cache, units, base_doc, proto3 = select_world(tmp_path / "b", plan)
    doc = T3E.select([units], A9, proto3, base_doc, log=lambda s: None)
    t = doc["ui_recall_vs_bytes"]["by_decode"]["det"]
    assert t["rises"] and t["spearman_rho"] > 0 and t["m_base_non_decreasing"]
    assert [b["ui_recall"] for b in t["m_base"]] == [0.0, 1.0, 1.0]
    assert "det: UI recall rises with VTSP bytes" in T3E.render(doc)


def test_select_refuses_stale_and_foreign_units(tmp_path, proto3_file, monkeypatch):
    monkeypatch.setattr(T3E, "m_model", lambda: FakeModel3)
    rows, shas, cache, units, base_doc, proto3 = select_world(tmp_path, {f"base_{T_SHAPE}": ("one",)})
    runs = tmp_path / "runs"
    runs.mkdir()
    write_run(runs, f"base_{T_SHAPE}_seed1", d2s_blob(1))                    # another blob than the units scored
    with pytest.raises(SystemExit):
        T3E.t3_seeds([units], runs)
    with pytest.raises(SystemExit):
        T3E.select([units], A9, proto3, dict(base_doc, manifest={"sha256": "x" * 64}), log=lambda s: None)
    with pytest.raises(SystemExit):
        T3E.select([tmp_path / "empty"], A9, proto3, base_doc, log=lambda s: None)


def test_select_reads_every_finished_run(tmp_path, monkeypatch):
    """A finished run without its two scored arms stops select (every finished run or none); the report's
    per-part, text-free and dialogue-pick figures are each representative's summary's."""
    monkeypatch.setattr(T3E, "m_model", lambda: FakeModel3)
    rows, shas, cache, units, base_doc, proto3 = select_world(tmp_path, {f"base_{T_SHAPE}": ("one",)})
    reg = T3E.registry(proto3)
    runs = tmp_path / "runs"
    runs.mkdir()
    run = f"base_{T_SHAPE}_seed1"
    d = runs / run
    d.mkdir()
    (d / "done.json").write_text(json.dumps({"vtsp_sha256": blob_sha(run), "vtsp_bytes": 7416}))
    doc = T3E.select([units], A9, proto3, base_doc, runs_dir=runs, reg=reg, log=lambda s: None)
    rep = doc["representatives"][f"T3_{run}_det"]
    s = next(x for x in doc["seeds"] if x["name"] == f"T3_{run}_det")["summary"]
    assert rep["per_part"]["vita"] == {"frames": 20, "f1": s["per_part"]["vita"]["detection"]["f1"],
                                       "cer_all": s["per_part"]["vita"]["cer_all"]["cer"],
                                       "cer_dialogue": s["per_part"]["vita"]["cer_dialogue"]["cer"]}
    assert rep["text_free"]["frames"] == 0 and rep["dialogue_pick"]["frames_with_dialogue"] == 30
    for name in (f"base_{T_SHAPE}_seed2", "probe_x"):                         # a second finished run; a non-run
        (runs / name).mkdir()
        (runs / name / "done.json").write_text("{}")
    assert T3E.unscored_runs(runs, T3E.t3_seeds([units], reg=reg), reg) == [f"base_{T_SHAPE}_seed2"]
    with pytest.raises(SystemExit):
        T3E.select([units], A9, proto3, base_doc, runs_dir=runs, reg=reg, log=lambda s: None)
    (runs / "finished.txt").write_text(f"{run} {blob_sha(run)} 7416 30000 1.0\n")   # seed 2 not listed yet
    assert T3E.unscored_runs(runs, T3E.t3_seeds([units], reg=reg), reg) == []


def test_read_pick_needs_the_record_pick(tmp_path):
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps({"result": {"winner": "T3_a_mS_seed2_det"},
                               "seeds": [{"name": "T3_a_mS_seed2_det", "run": "a_mS_seed2"}]}))
    args = argparse.Namespace(dev=str(dev), names=["a_mS_seed1"], models="S,L")
    with pytest.raises(SystemExit):
        T3E.cmd_read_pick(args)
    with pytest.raises(SystemExit):
        T3E.cmd_read_pick(argparse.Namespace(dev="", names=["a_mS_seed2"], models="N"))
    dev.write_text(json.dumps({"result": {"winner": None}, "seeds": []}))
    with pytest.raises(SystemExit):
        T3E.cmd_read_pick(args)


# ----------------------------------------------------------------------------- pins

def test_pinned_files_unchanged():
    """The files textspot3's protocol pins, which this wrapper imports and uses unchanged, are byte for byte as
    frozen (also checked by tests/test_textspot3_rule.py over every pin)."""
    pins = dict(PROTO3["sources_sha256"])
    for a in PROTO3.get("amendments", []):
        pins.update(a.get("sources_sha256", {}))
    for f in ("eval/textspot.py", "eval/textspot_metrics.py", "eval/textspot2.py", "eval/textspot3_rule.py",
              "eval/textspot_protected.py", "eval/textspot2_protected.py", "synth/textspot_twin.py",
              "synth/textspot2_twin.py", "train/textspot2_export.py", "eval/textspot_baseline_20261019.json"):
        assert TS.sha256(ROOT / f) == pins[f], f"{f} changed after textspot3's protocol pinned it"
    T3E.check_baseline_pin(T3E.BASELINE, PROTO3)


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
    ref = tmp_path / "ref.vtsp"
    ref.write_bytes(S.pack(vtdt("s8_c8", 6), 600, 1))
    return ref


def fake_test_set(tmp_path, proto3_file):
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
    doc = json.loads(proto3_file.read_text())
    doc["reused"]["test_manifest"]["sha256"] = TS.sha256(man)
    proto3_file.write_text(json.dumps(doc))
    return man, pp, shots


def pick_record(tmp_path, name, blob):
    dev = tmp_path / "textspot3_dev.json"
    dev.write_text(json.dumps({"kind": "textspot3-selection", "result": {"winner": name},
                               "seeds": [{"name": name, "blob_sha256": TS.sha256_bytes(blob)}]}))
    return dev


def t_run(tmp_path, seed=7):
    runs = tmp_path / "runs"
    runs.mkdir(exist_ok=True)
    run = f"base_{T_SHAPE}_seed{seed}"
    blob = d2s_blob(seed)
    write_run(runs, run, blob)
    return runs, run, blob


def test_protected_tests_once_against_stored_units(tmp_path, fake_blobs, proto3_file):
    refb = fake_blobs
    man, pp, shots = fake_test_set(tmp_path, proto3_file)
    rd = fake_readers(tmp_path)
    ts_out = tmp_path / "textspot_tests"        # textspot's own read of ref: how the stored files were made
    TP.cmd_tests(argparse.Namespace(cand=f"detref:B_h2_s4_c24n4_seed2={refb}", decode="", out=str(ts_out),
                                    cache=str(tmp_path / "pcache"), protocol=str(pp), manifest=str(man),
                                    shots=[str(shots)], cbin="", models="N,S,L", workers=1, compiler="gcc", data="",
                                    report_parts=list(TP.REPORT_PARTS)), readers=rd, log=lambda s: None)
    runs, run, blob = t_run(tmp_path)
    name = f"T3_{run}_detref"
    out = tmp_path / "t3" / "tests"
    args = argparse.Namespace(cand=f"T3:{run}", decode="det+refine", dev=str(pick_record(tmp_path, name, blob)),
                              out=str(out), runs=str(runs), cbin="", models="N,S,L", workers=1,
                              cache=str(tmp_path / "p3cache"), seed_from=str(tmp_path / "pcache"), compiler="gcc",
                              protocol=str(pp), protocol3=str(proto3_file), stored=str(ts_out / "private"),
                              record=str(ts_out / "tests.json"), manifest=str(man), data="", shots=[str(shots)],
                              report_parts=list(TP.REPORT_PARTS))
    res = TP3.cmd_tests(args, readers=rd, log=lambda s: None)
    assert (out / "tests.attempt.json").exists() and (out / "tests.json").exists()
    assert set(res["arms"]) == {name} and (res["scored_frames"], res["report_only_frames"]) == (5, 1)
    assert res["blob_version"] == 2 and res["candidate"]["half_above"] == 480
    assert sorted(p.name for p in (out / "private").iterdir()) == [name]            # only the pick ran
    assert set(res["comparisons"]["VOCR5-N"]) == set(TP3.STORED_ARMS) == {TP3.REF_ARM, *TS.BASELINES}
    assert set(res["recall_by_class"]["VOCR5-L"]) == {name, TP3.REF_ARM} and res["stored"]["reproduces_record"]
    assert res["cache_seed"]["files"] and (tmp_path / "p3cache" / "SEEDED.json").exists()
    assert res["pick"]["winner"] == name
    TS.assert_public(json.loads((out / "tests.json").read_text()))
    with pytest.raises(FileExistsError):                                            # once per command
        TP3.cmd_tests(args, readers=rd, log=lambda s: None)
    (tmp_path / "x").mkdir()
    for bad in ({"decode": "det"}, {"dev": str(pick_record(tmp_path / "x", name, d2s_blob(8)))}):
        a2 = argparse.Namespace(**dict(vars(args), out=str(tmp_path / "t3c" / "tests"), **bad))
        with pytest.raises(SystemExit):              # not the record's pick (another decode, another blob): refused
            TP3.cmd_tests(a2, readers=rd, log=lambda s: None)
        assert not (tmp_path / "t3c" / "tests" / "tests.attempt.json").exists()
    nopick = tmp_path / "nopick.json"
    nopick.write_text(json.dumps({"kind": "textspot3-selection", "result": {"winner": None}, "seeds": []}))
    with pytest.raises(SystemExit):                  # no pick: nothing is read
        TP3.cmd_tests(argparse.Namespace(**dict(vars(args), dev=str(nopick), out=str(tmp_path / "t3d"))),
                      readers=rd, log=lambda s: None)
    f = ts_out / "private" / "grad" / "units_VOCR5-N.json"
    doc = json.loads(f.read_text())
    doc["units"][0]["tp"] += 1
    f.write_text(json.dumps(doc))
    args2 = argparse.Namespace(**dict(vars(args), out=str(tmp_path / "t3b" / "tests")))
    with pytest.raises(SystemExit):           # stored units that do not reproduce the record: refused before reading
        TP3.cmd_tests(args2, readers=rd, log=lambda s: None)
    assert not (tmp_path / "t3b" / "tests" / "tests.attempt.json").exists()


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


def test_protected_clannad_once_against_stored_boxes(tmp_path, fake_blobs, proto3_file, twin3):
    refb = fake_blobs
    work = fake_clannad(tmp_path)
    rd = fake_readers(tmp_path)
    proto = str(ROOT / "eval" / "textspot_protocol.json")
    ts_out = tmp_path / "textspot_clannad"
    TP.cmd_clannad(argparse.Namespace(cand=f"detref:B_h2_s4_c24n4_seed2={refb}", decode="", out=str(ts_out),
                                      cache=str(tmp_path / "pcache"), protocol=proto, work=str(work), cbin="",
                                      models="N", workers=1, compiler="gcc"),
                   readers=rd, grad_fn=TP.twin_grad, log=lambda s: None)
    cb = tmp_path / "cand.vtsp"
    blob = as_version(S.pack(vtdt("s8_c8", 7), 600, 1), 3)                  # a version-3 pick (the fake twin)
    cb.write_bytes(blob)
    out = tmp_path / "t3" / "clannad"
    args = argparse.Namespace(cand=f"det:cand={cb}", decode="", dev=str(pick_record(tmp_path, "cand_det", blob)),
                              out=str(out), runs=str(tmp_path / "runs"), cbin="", models="N", workers=1,
                              cache=str(tmp_path / "p3cache"), seed_from=str(tmp_path / "pcache"), compiler="gcc",
                              protocol=proto, protocol3=str(proto3_file), stored=str(ts_out / "private"),
                              record=str(ts_out / "clannad.json"), work=str(work))
    res = TP3.cmd_clannad(args, readers=rd, grad_fn=TP.twin_grad, log=lambda s: None)
    assert twin3 and res["blob_version"] == 3
    assert res["stored"]["reproduces_record"] and res["stored"]["differences"] == []
    assert set(res["arms"]) == {"cand_det"} and sorted(p.name for p in (out / "private").iterdir()) == ["cand_det"]
    dl = res["scopes"]["dialogue"]
    assert set(dl["pairs"]) == {f"cand_det/VOCR5-N:{b}/VOCR5-N" for b in TP3.STORED_ARMS}
    assert dl["lines"] < res["scopes"]["all"]["lines"] and res["counts"]["no_text_pages"] == 1
    TS.assert_public(json.loads((out / "clannad.json").read_text()))
    with pytest.raises(FileExistsError):
        TP3.cmd_clannad(args, readers=rd, grad_fn=TP.twin_grad, log=lambda s: None)
    rec = json.loads((ts_out / "clannad.json").read_text())
    rec["arms"]["grad"]["boxes"] += 1
    bad = tmp_path / "bad_record.json"
    bad.write_text(json.dumps(rec))
    a2 = argparse.Namespace(**dict(vars(args), out=str(tmp_path / "t3b" / "clannad"), record=str(bad)))
    with pytest.raises(SystemExit):           # stored boxes that do not reproduce the record's counts: refused first
        TP3.cmd_clannad(a2, readers=rd, grad_fn=TP.twin_grad, log=lambda s: None)
    assert not (tmp_path / "t3b" / "clannad" / "clannad.attempt.json").exists()


def test_cand_spec_forms_and_seed_source(tmp_path, proto3_file):
    runs, run, blob = t_run(tmp_path, 2)
    s, v = TP3.cand_spec(f"T3:{run}", "det+refine", runs)
    assert (s.name, s.decode, v) == (f"T3_{run}_detref", "det+refine", 2)
    cb = tmp_path / "c.vtsp"
    cb.write_bytes(vtsp())
    s, v = TP3.cand_spec(f"detref:x={cb}")
    assert (s.name, v) == ("x_detref", 1)
    for bad in (f"T3:{run}", "B:h2_s4_c24n4_seed2", f"det{cb}"):
        with pytest.raises(SystemExit):
            TP3.cand_spec(bad, None, runs)
    a, b = tmp_path / "p2cache", tmp_path / "pcache"
    a.mkdir()
    b.mkdir()
    (b / "x.jsonl").write_text("")
    assert TP3.seed_source((a, b)) == b                       # textspot2's private cache is empty: textspot's
    (a / "y.jsonl").write_text("")
    assert TP3.seed_source((a, b)) == a and TP3.seed_source((tmp_path / "none",)) is None
    assert TP3.CACHE_SOURCES == (T2E.P2 / "cache", TS.P / "cache")
    with pytest.raises(SystemExit):
        TP3.check_pick("", s)


# ----------------------------------------------------------------------------- the real runtime (once it exists)

def have_runtime3():
    return (ROOT / "synth" / "textspot3_twin.py").exists() and (ROOT / "train" / "textspot3_model.py").exists() and \
        (ROOT / "runtime" / "vocr_textspot3_test.c").exists()


@pytest.mark.skipif(not have_runtime3(), reason="textspot3-runtime's twin, model and driver are not merged yet")
def test_a_v3_run_through_the_real_twin_and_driver(tmp_path, monkeypatch):
    """A random VTSP version-3 blob of a family-M shape (train/textspot3_model.random_blob) as a finished run: scored
    through synth/textspot3_twin.py and checked against runtime/vocr_textspot3_test on every frame."""
    import textspot3_model as TM3
    import textspot3_twin as TT3
    if shutil.which("gcc") is None:
        pytest.skip("no gcc")
    probes = TM3.PROBES                                     # the runtime's probe shapes (not registered)
    name = min(probes, key=lambda n: (TM3.costs(probes[n])["macs"]["960x544"], n))      # the cheapest
    shape = probes[name]
    doc = json.loads(json.dumps(PROTO3))
    doc["shapes"] = {name: dict(shape, family="M")}
    p = tmp_path / "proto3.json"
    p.write_text(json.dumps(doc))
    monkeypatch.setattr(T3E, "PROTOCOL3", p)
    blob = TM3.random_blob(name, 3)
    blob = blob[0] if isinstance(blob, tuple) else blob
    runs = tmp_path / "runs"
    runs.mkdir()
    write_run(runs, f"base_{name}_seed1", blob)
    found, _ = T3E.finished_runs(runs)
    assert found[f"base_{name}_seed1"]["version"] == 3
    cdir = tmp_path / "cbuild"
    cdir.mkdir()
    srcs = next(s for s in T3E.DRIVER3_SOURCE_SETS if all((ROOT / "runtime" / x).exists() for x in s))
    subprocess.run(["gcc", *TS.C_FLAGS, "-o", str(cdir / T3E.DRIVER3), *srcs], check=True, cwd=ROOT / "runtime")
    frames = [frame_of(synthetic_frame(seed=40 + i), fid=f"f{i}", vn=f"v90020{i}") for i in range(3)]
    manifest = {"set": "pseudo", "path": "x", "sha256": "p" * 64, "frames": 3, "vns": 3}
    out = tmp_path / "units"
    todo = T3E.pending(out, found, manifest, [f.id for f in frames], ["VOCR5-N"], cdir)
    T3E.score_runs(frames, manifest, found, todo, out, ["VOCR5-N"], fake_readers(tmp_path), TS.Cache(tmp_path / "c"),
                   cdir, 1, log=lambda s: None)
    bd = json.loads((out / f"T3_base_{name}_seed1_det" / "boxes.json").read_text())
    for f in frames:
        r = TT3.spot(TT3.Model(blob), f.grey)
        assert bd["frames"][f.id]["boxes"] == [[b.x, b.y, b.w, b.h, b.vertical, b.conf, b.dark, b.block]
                                               for b in r.boxes]
        assert bd["frames"][f.id]["c_equal"] is True                          # the C driver gave the twin's boxes
    side = json.loads((out / f"T3_base_{name}_seed1_det" / "textspot3.json").read_text())
    assert side["blob_version"] == 3 and side["driver_sha256"] == TS.sha256(cdir / T3E.DRIVER3)
    assert sys.modules["textspot_twin"] is S


def test_a_v2_run_through_the_d2s_driver(tmp_path, proto3_file):
    """A family-T run (VTSP version 2 at half_above 480) checked against runtime/vocr_textspot2_test on every frame,
    through textspot2's path."""
    if shutil.which("gcc") is None:
        pytest.skip("no gcc")
    runs, run, blob = t_run(tmp_path, 3)
    found, _ = T3E.finished_runs(runs)
    cdir = tmp_path / "cbuild"
    cdir.mkdir()
    subprocess.run(["gcc", *TS.C_FLAGS, "-o", str(cdir / T2E.DRIVER2), *T2E.DRIVER2_SOURCE_SETS[0]], check=True,
                   cwd=ROOT / "runtime")
    subprocess.run(["gcc", *TS.C_FLAGS, "-fPIC", "-shared", "-o", str(cdir / "libvocr_refine.so"),
                    *TS.LIBS["libvocr_refine.so"]], check=True, cwd=ROOT / "runtime")     # det+refine's C path
    frames = [frame_of(synthetic_frame(seed=50 + i), fid=f"f{i}", vn=f"v90030{i}") for i in range(2)]
    frames.append(frame_of(synthetic_frame(480, 272, seed=59), fid="f9", vn="v900309"))
    manifest = {"set": "pseudo", "path": "x", "sha256": "p" * 64, "frames": 3, "vns": 3}
    out = tmp_path / "units"
    todo = T3E.pending(out, found, manifest, [f.id for f in frames], ["VOCR5-N"], cdir)
    T3E.score_runs(frames, manifest, found, todo, out, ["VOCR5-N"], fake_readers(tmp_path), TS.Cache(tmp_path / "c"),
                   cdir, 1, log=lambda s: None)
    for k in T3E.KINDS:
        bd = json.loads((out / f"T3_{run}_{k}" / "boxes.json").read_text())
        assert all(fr["c_equal"] is True for fr in bd["frames"].values())
        side = json.loads((out / f"T3_{run}_{k}" / "textspot3.json").read_text())
        assert side["driver_sha256"] == TS.sha256(cdir / T2E.DRIVER2) and side["blob_version"] == 2
    assert T3E.pending(out, found, manifest, [f.id for f in frames], ["VOCR5-N"], cdir) == []
