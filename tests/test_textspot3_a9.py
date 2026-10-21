"""textspot3's A9 tables (eval/textspot3_a9.py): the wrapper's helpers, and the committed family-T table
(eval/textspot3_a9_t_20261021.json): every T candidate at half_above 480 with the default call and
VOCR_TEXTSPOT_NO_DARK at 960x544 and 480x272, its blobs textspot2_export's, and textspot2's two d2s shapes measured
again at half_above 600 equal to eval/textspot2_a9_d2s_20261020.json row for row. No toolchain, qemu or data."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "train", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot3_a9 as W  # noqa: E402
import textspot_a9 as A9  # noqa: E402

import textspot2_export as X  # noqa: E402
import textspot3_shapes as S3  # noqa: E402

T_TABLE = ROOT / "eval" / "textspot3_a9_t_20261021.json"
D2S = ROOT / "eval" / "textspot2_a9_d2s_20261020.json"


@pytest.fixture(scope="module")
def table():
    return json.loads(T_TABLE.read_text(encoding="utf-8"))


def test_scales():
    assert W.scale_name(600) == "h2" and W.scale_name(0) == "f1" and W.scale_name(480) == "h480"
    assert "h480" not in A9.SCALES
    with W.scales(480) as name:
        assert name == "h480" and A9.SCALES["h480"] == 480
    assert "h480" not in A9.SCALES
    with W.scales(600) as name:
        assert name == "h2" and A9.SCALES["h2"] == 600
    assert A9.SCALES["h2"] == 600


def test_within_budget():
    r = {"vtsp_bytes": 512000, "ms_333_960": 300.0, "arena_up_to_960": 262144}
    assert W.within(r)
    assert not W.within(dict(r, vtsp_bytes=512001)) and not W.within(dict(r, ms_333_960=300.01))
    assert not W.within(dict(r, arena_up_to_960=262145))


def test_t_table_covers_every_candidate(table):
    b = table["textspot3"]
    assert b["runtime"] == "v2" and table["calibration"] == 1.0007
    assert set(table["sources_sha256"]) == {"runtime/vocr_textspot2.c", "runtime/vocr_lines.c",
                                            "runtime/vocr_lines_rows.c", "runtime/vocr_textspot2_test.c",
                                            "runtime/vocr_textspot2.h"}
    rows = {(r["shape"], r["half_above"]): r for r in b["rows"]}
    want = {(n, 480) for n in S3.CANDIDATES} | {("s8_c24n4x2", 600), ("s8_c48n4x2", 600)}
    assert set(rows) == want
    for (name, ha), r in rows.items():
        for k in ("ms_333_960", "ms_444_960", "ms_333_960_nodark", "ms_333_480", "ms_333_480_nodark",
                  "ms_444_480_nodark", "arena_up_to_960", "arena_frame_960", "arena_up_to_480"):
            assert r[k] is not None, (name, k)
        assert r["ms_444_960"] == pytest.approx(r["ms_333_960"] * 333 / 444)
        assert r["arena_frame_960"] <= r["arena_up_to_960"]
        assert r["within_budget"] == (r["vtsp_bytes"] <= 512000 and r["ms_333_960"] <= 300 and
                                      r["arena_up_to_960"] <= 262144)
        if ha == 480:
            c = S3.costs(name)
            assert r["vtsp_bytes"] == c["vtsp_bytes"] and r["macs_960"] == c["macs"]["960x544"]
            assert r["weights_int8"] == c["weights_int8"] and r["receptive_field"] == c["receptive_field"]
    # the measured blobs are textspot2_export.random_blob's, seed 1, at the row's half_above
    for t in table["table"][:4]:
        with S3.inject(S3.CANDIDATES):
            blob, src = X.random_blob(t["shape"], 1, half_above=t["half_above"])
        assert t["blob_sha256"] == hashlib.sha256(blob).hexdigest() and t["blob"] == src
    # the 64-channel d8 trunks need a 17-row ring of 64 channels: over the 256 KiB arena
    assert {n for (n, ha), r in rows.items() if not r["within_budget"]} == {"t64n8c16d8", "t64n8c24d8"}


def test_textspot2_rows_reproduce(table):
    rep = table["textspot3"]["reproduces"]
    assert len(rep) == 1 and rep[0]["against"] == "eval/textspot2_a9_d2s_20261020.json"
    assert rep[0]["sha256"] == W.sha256(D2S) and rep[0]["all_identical"]
    assert {(r["shape"], tuple(r["size"]), r["flags"]) for r in rep[0]["rows"]} == {
        (s, z, f) for s in ("s8_c24n4x2", "s8_c48n4x2") for z in ((960, 544), (480, 272))
        for f in ("0", "VOCR_TEXTSPOT_NO_DARK")}


def test_no_local_paths():
    raw = T_TABLE.read_text(encoding="utf-8")
    assert "/home/" not in raw and "/local/" not in raw
