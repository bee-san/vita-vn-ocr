"""textspot3's A9 table of record (eval/textspot3_a9_20261021.json, eval/textspot3_a9r.py): family M's registered
shapes (qS4, qM4, qL5) on runtime/vocr_textspot3.c with the default call and VOCR_TEXTSPOT_NO_DARK at 960x544 and
480x272, their blobs train/textspot3_model.random_blob's, their MACs and bytes exact, within textspot3's budget; family
T's registered shapes and textspot2's d2s shapes measured again on runtime/vocr_textspot2.c, every row identical to
the pinned tables; code sizes and stack peaks recorded. No toolchain, qemu or data needed."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "train", ROOT / "synth", ROOT / "vita", ROOT / "psp"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot3_a9r as R  # noqa: E402

import textspot3_model as M3  # noqa: E402
import textspot3_shapes as S3  # noqa: E402

TABLE = ROOT / "eval" / "textspot3_a9_20261021.json"
M_SHAPES = ("qS4", "qM4", "qL5")
T_SHAPES = ("t64n4c16", "t48n8c16d8", "t64n8c16")
COLUMNS = ("ms_333_960", "ms_444_960", "ms_333_960_nodark", "ms_444_960_nodark", "ms_333_480", "ms_444_480",
           "ms_333_480_nodark", "arena_up_to_960", "arena_frame_960", "arena_up_to_480")


@pytest.fixture(scope="module")
def table():
    if not TABLE.exists():
        pytest.skip("the table of record is not committed yet")
    return json.loads(TABLE.read_text(encoding="utf-8"))


def test_m_rows_of_record(table):
    b = table["textspot3"]
    assert b["runtime"] == "v3" and table["calibration"] == 1.0007
    assert set(table["sources_sha256"]) == {"runtime/vocr_textspot3.c", "runtime/vocr_lines.c",
                                            "runtime/vocr_lines_rows.c", "runtime/vocr_textspot3_test.c",
                                            "runtime/vocr_textspot3.h"}
    rows = {r["shape"]: r for r in b["rows"]}
    assert set(rows) == set(M_SHAPES) == set(S3.M)
    points = {"qS4": "S", "qM4": "M", "qL5": "L"}
    for name, r in rows.items():
        c = M3.costs(S3.M[name])
        assert r["point"] == points[name] and r["vtsp_bytes"] == c["vtsp_bytes"]
        assert r["macs_960"] == c["macs"]["960x544"] and r["weights_int8"] == c["weights_int8"]
        for k in COLUMNS:
            assert r[k] is not None, (name, k)
        assert r["ms_444_960"] == pytest.approx(r["ms_333_960"] * 333 / 444)
        assert r["within_budget"] and r["vtsp_bytes"] <= 512000 and r["ms_333_960"] <= 300
        assert r["arena_up_to_960"] <= 262144 and r["arena_frame_960"] <= r["arena_up_to_960"]
        sp = r["split_960"]
        assert set(sp) >= {"stream", "map", "decode", "dark", "other", "cache"} and sp["other"]["ms_333"] < 1.0
    for t in table["table"][:3]:
        blob, src = M3.random_blob(S3.M[t["shape"]], 1)
        assert t["blob_sha256"] == hashlib.sha256(blob).hexdigest() and t["blob"] == src
    nd = [a for a in table["also"] if all(t["flags"] == "VOCR_TEXTSPOT_NO_DARK" for t in a["table"])]
    assert len(nd) == 1 and len(nd[0]["table"]) == len(table["table"]) == 2 * len(M_SHAPES)


def test_t_rows_reproduce_the_pinned_tables(table):
    t = table["textspot3"]["t"]
    assert t["runtime"] == "v2"
    assert {r["shape"] for r in t["rows"]} == set(T_SHAPES) | {"s8_c24n4x2", "s8_c48n4x2"}
    reps = {r["against"]: r for r in t["reproduces"]}
    assert set(reps) == {"eval/textspot3_a9_t_20261021.json", "eval/textspot2_a9_d2s_20261020.json"}
    for path, r in reps.items():
        assert r["all_identical"] and r["identical"] == r["rows"] and not r["differ"], path
        assert r["sha256"] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    assert reps["eval/textspot3_a9_t_20261021.json"]["rows"] == 20
    assert reps["eval/textspot2_a9_d2s_20261020.json"]["rows"] == 8


def test_code_size_and_stack_recorded(table):
    b = table["textspot3"]
    cs = b["code_size"]["vita"]["builds"]
    assert set(cs) == {"-O2", "-Os", "-O2 -DVOCR_NO_NEON", "-Os -DVOCR_NO_NEON"}
    assert all(v["text"] > 0 for v in cs.values()) and set(b["code_size"]["psp"]["object_text"]) == {
        "-O2 -G0", "-Os -G0"}
    st = b["stack"]["builds"]
    assert set(st) == {"Vita -O2", "Vita -Os", "PSP -O2 -G0", "PSP -Os -G0"}
    for build in st.values():
        assert {"vocr_textspot3_find", "vocr_textspot3_find_rows", "vocr_textspot3_load"} <= set(build["peaks"])
        assert build["peaks"]["vocr_textspot3_find"]["bytes"] < 8192


def test_part_table_and_no_local_paths(table):
    parts = R.parts()
    assert "pw_sums" in parts["map"] and "dw_row" in parts["map"] and R.part_of("dw_row.isra.0", set()) == "map"
    assert R.part_of("vocr_line_dark", {"vocr_line_dark"}) == "dark" and R.part_of("main", set()) == "other"
    raw = TABLE.read_text(encoding="utf-8")
    assert "/home/" not in raw and "/local/" not in raw
