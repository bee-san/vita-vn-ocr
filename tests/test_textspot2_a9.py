"""textspot2's A9 table (eval/textspot2_a9.py, eval/textspot2_a9_20261020.json): the wrapper's helpers, and the
committed table covers every "kroa" shape of eval/textspot2_protocol.json as the protocol's estimates require, with
textspot's own shapes reproducing eval/textspot_a9_20261019.json row for row. No toolchain, qemu or data needed."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "train", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot2_a9 as W  # noqa: E402

import textspot2_shapes as S2  # noqa: E402

TABLE = ROOT / "eval" / "textspot2_a9_20261020.json"
TEXTSPOT = ROOT / "eval" / "textspot_a9_20261019.json"
RULE_KEYS = ("shape", "scale", "half_above", "vtsp_bytes", "macs_960", "ms_333_960", "ms_444_960",
             "ms_333_960_nodark", "arena_up_to_960", "arena_frame_960", "ms_333_480", "ms_444_480",
             "arena_up_to_480", "arena_frame_480", "within_limits")


@pytest.fixture(scope="module")
def table():
    return json.loads(TABLE.read_text(encoding="utf-8"))


def test_protocol_shapes():
    kroa, d2s = W.protocol_shapes(), W.protocol_shapes("kroa_x2")
    assert kroa == ["s4p_c8", "s4p_c6n4", "s4p_c8n4", "s4p_c10n4", "s4p_c12", "s4_c24n4", "s4_c32n3"]
    assert d2s == ["s8_c24n4x2", "s8_c48n4x2"] and set(d2s) == set(S2.D2S)


def test_portable():
    assert W.portable(None) is None
    p = W.portable(["--out", f"{W.HERD}/shared/artifacts/textspot2/a9/x", str(W.ROOT / "eval" / "a.json")])
    assert p == ["--out", "$HERD/shared/artifacts/textspot2/a9/x", "eval/a.json"]


def test_compare_rows():
    row = {"shape": "s", "scale": "h2", "size": [960, 544], "format": "grey", "flags": "0", "insns_per_frame": 5.0,
           "arena_bytes": 7, "blob_sha256": "ab"}
    same = W.compare_rows([row], [dict(row)], "main")
    assert same == [{"table": "main", "shape": "s", "scale": "h2", "size": [960, 544], "format": "grey",
                     "flags": "0", "identical": True, "diffs": {}}]
    diff = W.compare_rows([row], [dict(row, insns_per_frame=6.0)], "main")
    assert not diff[0]["identical"] and diff[0]["diffs"] == {"insns_per_frame": [5.0, 6.0]}
    assert W.compare_rows([row], [dict(row, flags="VOCR_TEXTSPOT_NO_DARK")], "main") == []


def test_table_covers_every_kroa_shape(table):
    """Every kroa shape at h2: a summary row with the rule's columns, the 960x544 and 480x272 rows of the default call
    and of VOCR_TEXTSPOT_NO_DARK, grey, MACs as train/textdet_model.cost gives them."""
    shapes = W.protocol_shapes()
    summ = {r["shape"]: r for r in table["summary"]}
    assert set(summ) == set(shapes) and all(r["scale"] == "h2" and r["half_above"] == 600 for r in summ.values())
    for name in shapes:
        r = summ[name]
        assert all(k in r and r[k] is not None for k in RULE_KEYS), name
        assert r["macs_960"] == S2.costs(name)["macs"]["960x544"]
        assert r["within_limits"] == (r["ms_333_960"] <= 150 and r["arena_up_to_960"] <= 131072)
        assert r["ms_444_960"] == pytest.approx(r["ms_333_960"] * 333 / 444)
        assert r["arena_frame_960"] <= r["arena_up_to_960"]
    main = {(t["shape"], tuple(t["size"])) for t in table["table"] if t["flags"] == "0" and t["format"] == "grey"}
    assert main == {(s, z) for s in shapes for z in ((960, 544), (480, 272))}
    nd = [a for a in table["also"] if all(t["flags"] == "VOCR_TEXTSPOT_NO_DARK" for t in a["table"])]
    assert len(nd) == 1 and {(t["shape"], tuple(t["size"])) for t in nd[0]["table"]} == main
    assert table["calibration"] == 1.0007 and table["textspot2"]["runtime"] == "v1"
    # the protocol file changes with every amendment; the table records the sha256 it was measured under, and
    # what must still hold is that it covers the protocol's shapes (above) under the same rule columns
    assert len(table["textspot2"]["protocol"]["sha256"]) == 64


def test_textspot_rows_reproduce(table):
    """s4p_c8 and s4_c24n4 (textspot's shapes, measured again in the same batch) equal textspot's rows field for
    field: the same code, frames, blobs and tools give the same counts."""
    ref = json.loads(TEXTSPOT.read_text(encoding="utf-8"))
    rows = W.compare_rows(table["table"], ref["table"], "main")
    nd = next(a for a in table["also"] if a["table"][0]["flags"] != "0")
    rnd = next(a for a in ref["also"] if a["table"][0]["flags"] == nd["table"][0]["flags"])
    rows += W.compare_rows(nd["table"], rnd["table"], "nodark")
    assert {(r["shape"], tuple(r["size"]), r["table"]) for r in rows} == \
        {(s, z, t) for s in ("s4p_c8", "s4_c24n4") for z in ((960, 544), (480, 272)) for t in ("main", "nodark")}
    assert all(r["identical"] for r in rows), [r for r in rows if not r["identical"]]
    rep = table["textspot2"]["reproduces"]
    assert rep["all_identical"] and len(rep["rows"]) == 8 and rep["sha256"] == W.sha256(TEXTSPOT)


def test_code_size_and_no_local_paths(table):
    cs = table["textspot2"]["code_size"]["vita"]["builds"]
    assert {k: v["text"] for k, v in cs.items()} == {"-O2": 11440, "-Os": 9466, "-O2 -DVOCR_NO_NEON": 10396,
                                                      "-Os -DVOCR_NO_NEON": 8058}   # textspot's figures
    for path in (TABLE, D2S):
        raw = path.read_text(encoding="utf-8")
        assert "/home/" not in raw and "/local/" not in raw


# ----------------------------------------------------------------------------- the d2s table (runtime v2)

D2S = ROOT / "eval" / "textspot2_a9_d2s_20261020.json"
OUTPUT_FIELDS = ("blob_sha256", "blob_bytes", "frame_sha256", "batch_sha256", "rc", "boxes", "blocks", "pick",
                 "truncated", "frame_arena_bytes", "arena_bytes", "macs", "net_input")


@pytest.fixture(scope="module")
def d2s():
    return json.loads(D2S.read_text(encoding="utf-8"))


def test_d2s_table_covers_the_d2s_shapes_on_the_new_runtime(d2s):
    import textspot2_export as X
    b = d2s["textspot2"]
    assert b["runtime"] == "v2" and set(d2s["sources_sha256"]) == {
        "runtime/vocr_textspot2.c", "runtime/vocr_lines.c", "runtime/vocr_lines_rows.c",
        "runtime/vocr_textspot2_test.c", "runtime/vocr_textspot2.h"}
    shapes = [*W.protocol_shapes("kroa_x2"), "s4_c24n4"]
    summ = {r["shape"]: r for r in d2s["summary"]}
    assert set(summ) == set(shapes)
    for name in shapes:
        r = summ[name]
        assert all(k in r and r[k] is not None for k in RULE_KEYS), name
        want = X.costs(name) if name in S2.D2S else S2.costs(name)
        assert r["macs_960"] == want["macs"]["960x544"]
        assert r["within_limits"] == (r["ms_333_960"] <= 150 and r["arena_up_to_960"] <= 131072)
    for t in d2s["table"]:
        if t["shape"] in S2.D2S:
            blob, src = X.random_blob(t["shape"], 1, half_above=600)
            assert t["blob_sha256"] == hashlib.sha256(blob).hexdigest() and t["blob_bytes"] == len(blob)
            assert t["blob"] == src
    nd = [a for a in d2s["also"] if all(t["flags"] == "VOCR_TEXTSPOT_NO_DARK" for t in a["table"])]
    assert len(nd) == 1 and len(nd[0]["table"]) == len(d2s["table"]) == 2 * len(shapes)


def test_version_1_through_the_new_runtime_costs_what_textspot_costs(d2s, table):
    """s4_c24n4 (version 1) through runtime/vocr_textspot2.c: the same blob, frames, boxes, blocks, pick and arenas
    as through runtime/vocr_textspot.c (eval/textspot2_a9_20261020.json), and the estimate within 0.1% (the extra
    d2s code moves the code's layout, not its work)."""
    rep = d2s["textspot2"]["reproduces"]
    assert rep["sha256"] == W.sha256(TABLE) and len(rep["rows"]) == 4
    for r in rep["rows"]:
        assert r["shape"] == "s4_c24n4" and not set(r["diffs"]) & set(OUTPUT_FIELDS), r["diffs"]
    ours = {(t["shape"], tuple(t["size"]), t["flags"]): t for t in d2s["table"] + next(
        a for a in d2s["also"] if a["table"][0]["flags"] != "0")["table"]}
    theirs = {(t["shape"], tuple(t["size"]), t["flags"]): t for t in table["table"] + next(
        a for a in table["also"] if a["table"][0]["flags"] != "0")["table"]}
    for k, t in ours.items():
        if k[0] != "s4_c24n4":
            continue
        o = theirs[k]
        assert all(t[f] == o[f] for f in OUTPUT_FIELDS if f in t)
        assert abs(t["ms_per_frame"]["333"] - o["ms_per_frame"]["333"]) < 1e-3 * o["ms_per_frame"]["333"]
