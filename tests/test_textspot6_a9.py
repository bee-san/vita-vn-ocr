"""textspot6's A9 table (eval/textspot6_a9_20261026.json, eval/textspot6_a9.py): every tagged row (4 tags) of the
issue's shapes and the added ones, measured on runtime/vocr_textspot6.c as committed, with the default call and
VOCR_TEXTSPOT_NO_DARK at 960x544 and 480x272, -O2 and -Os (and the T rows on the -DVOCR_TEXTSPOT6_NO_MS build); its
blobs train/textspot6_model.random_blob's, its MACs and bytes exact; the same shapes' version-2/3 blobs beside them and
the tags' extra time against textspot3's rows of record; code sizes and stack peaks with and without the option; every
time marked as an estimate. No toolchain, qemu or data needed."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "train", ROOT / "synth", ROOT / "vita", ROOT / "psp"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot6_a9 as A6  # noqa: E402

import textspot6_model as M6  # noqa: E402

TABLE = ROOT / "eval" / "textspot6_a9_20261026.json"
ISSUE_T = ("t64n8c16", "t48n8c16d8", "t64n4c16", "t32n4c16", "t32n6c16", "t24n6c16", "t48n4c16")
ISSUE_M = ("qS4", "qM4", "qL5")


@pytest.fixture(scope="module")
def table():
    return json.loads(TABLE.read_text(encoding="utf-8"))


def test_the_rows_cover_the_issue(table):
    rows = {r["row"]: r for r in table["summary"]}
    want = {f"{n}_t4" for n in (*ISSUE_T, *ISSUE_M, *A6.T_FULL_ADDED)} | set(ISSUE_T) | set(ISSUE_M) | \
        set(A6.T_FULL_ADDED)
    assert set(rows) == want
    for name, r in rows.items():
        f = A6.figures(name)
        assert {k: r[k] for k in ("vtsp_bytes", "weights_int8", "macs", "tags", "version")} == \
            {k: f[k] for k in ("vtsp_bytes", "weights_int8", "macs", "tags", "version")}, name
        builds = ("O2", "Os") if r["tags"] else ("O2",)
        for b in builds:
            for mode in ("dark", "nodark"):
                for size in ("960x544", "480x272"):
                    c = r["ms"][b][mode][size]
                    assert c["444"] == pytest.approx(c["333"] * 333 / 444) and 20 < c["333"] < 1000, (name, b, mode)
        if r["tags"] and r["family"] == "T":
            assert set(r["ms"]["Os no_ms"]) == {"dark", "nodark"}
        assert r["arena"]["960x544"]["up_to"] >= r["arena"]["480x272"]["up_to"] > 0
        assert set(r["split_333"]["960x544"]) == {"stream", "map", "decode", "dark", "other", "cache"}
    for n in (*ISSUE_T, *ISSUE_M):
        e = rows[f"{n}_t4"]["extra_of_the_tags"]["dark"]["960x544"]
        assert 0 < e["tags_ms_333"] < 25
        if n in ("t64n8c16", "t48n8c16d8", "t64n4c16", *ISSUE_M):        # textspot3 has a row of record
            assert e["vs_textspot3_row_ms_333"] == pytest.approx(e["tags_ms_333"] + e["runtime_ms_333"])
    assert {r["row"] for r in table["probes"]["rows"]} == {f"{n}_t4" for n in A6.T_FULL_PROBE}


def test_blobs_frames_and_sources(table):
    """Every row's blob is the generator's (random_blob seed 1), every frame textspot's pinned one, every directory
    measured on the committed runtime (one source hash), every table of the report one of its measured directories."""
    assert table["calibration"] == 1.0007 and "estimates" in table["note"] and "nothing ran on a console" in \
        table["note"]
    for f, digest in table["sources_sha256"].items():
        assert hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == digest, f
    seen = set()
    for t in table["table"]:
        if t["shape"] in seen:
            continue
        seen.add(t["shape"])
        if t["shape"].endswith("_t4") and t["shape"].split("_t4")[0] in ("t32n4c16", "qS4", "t64n4c16"):
            shape, _, tags = A6.row_shape(t["shape"])
            blob, _ = M6.random_blob(shape, 1, tags)
            assert t["blob_sha256"] == hashlib.sha256(blob).hexdigest() and t["blob_bytes"] == len(blob)
    labels = {tb["label"] for tb in table["tables"]}
    assert labels == {"main", "nodark", "base", "base_nodark", "Os", "Os_nodark", "Os_noms", "Os_noms_nodark"}
    assert {t["table"] for t in table["table"]} == labels


def test_code_size_and_stack(table):
    cs = table["code_size"]["vita"]["builds"]
    keys = {f"{lvl}{neon}{ms}" for lvl in ("-O2", "-Os") for neon in ("", " -DVOCR_NO_NEON")
            for ms in ("", " -DVOCR_TEXTSPOT6_NO_MS")}
    assert set(cs) == keys
    for k in keys:
        if "NO_MS" in k:                        # the option drops the map ops: at most 60% of the full build's code
            full = cs[k.replace(" -DVOCR_TEXTSPOT6_NO_MS", "")]["text"]
            assert 0 < cs[k]["text"] < 0.6 * full, k
    psp = table["code_size"]["psp"]["object_text"]
    assert set(psp) == {"-O2 -G0", "-Os -G0", "-O2 -G0 -DVOCR_TEXTSPOT6_NO_MS", "-Os -G0 -DVOCR_TEXTSPOT6_NO_MS"}
    st = table["stack"]["builds"]
    assert len(st) == 8
    for b in st.values():
        assert set(A6.STACK_ROOTS) <= set(b["peaks"]) and b["peaks"]["vocr_textspot6_find"]["bytes"] < 4096


def test_no_local_paths():
    raw = TABLE.read_text(encoding="utf-8")
    assert "/home/" not in raw and "/local/" not in raw
