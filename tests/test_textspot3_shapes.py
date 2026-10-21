"""textspot3's shape registry (train/textspot3_shapes.py): the T candidates, the injection into textspot2's frozen
code, and the exact figures against real blobs. No data needed."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot2_export as X  # noqa: E402
import textspot2_shapes as S2  # noqa: E402
import textspot3_shapes as S3  # noqa: E402


def test_candidates_are_the_t_rows_grid():
    c = S3.CANDIDATES
    assert len(c) == 14
    for name, s in c.items():
        assert s["family"] == "T" and s["block"] == "sep" and s["head"] == "kroa_x2" and s["stride"] == 8
        assert s["c0"] == 8 and s["c1"] in (16, 24) and s["half_above"] == 480
        n = len(s["chans"])
        assert s["chans"] == [s["c2"]] * n and len(s["dil"]) == n
        assert (s["c2"], n) in {(64, 4), (64, 6), (64, 8), (48, 6), (48, 8)}
        assert s["dil"] == list(S3.DIL8 if name.endswith("d8") else S3.DIL[n])
        assert name == f"t{s['c2']}n{n}c{s['c1']}" + ("d8" if name.endswith("d8") else "")
    assert sum(name.endswith("d8") for name in c) == 4
    if not S3.REGISTERED:
        assert S3.SHAPES == c and S3.T == c and not S3.M


def test_inject_adds_and_removes():
    assert not set(S3.CANDIDATES) & set(S2.SHAPES)
    with S3.inject(S3.CANDIDATES):
        assert set(S3.CANDIDATES) <= set(S2.SHAPES) and set(S3.CANDIDATES) <= set(S2.D2S)
        assert X.shape("t64n4c16")["chans"] == [64] * 4
    assert not set(S3.CANDIDATES) & set(S2.SHAPES) and not set(S3.CANDIDATES) & set(S2.D2S)
    bad = {"s8_c24n4x2": S3.t_shape("s8_c24n4x2", 16, 24, 4, (1, 2, 4, 1))}   # textspot2's name, other dilations
    with pytest.raises(ValueError):
        with S3.inject(bad):
            pass
    assert S2.SHAPES["s8_c24n4x2"]["dil"] == [1, 2, 4, 2]
    same = {"s8_c24n4x2": S3.t_shape("s8_c24n4x2", 16, 24, 4, (1, 2, 4, 2))}  # the same shape: left alone
    with S3.inject(same):
        pass
    assert "s8_c24n4x2" in S2.SHAPES


@pytest.mark.parametrize("name", ["t48n6c16", "t64n4c24"])
def test_costs_match_real_blobs(name):
    c = S3.costs(name)
    with S3.inject(S3.CANDIDATES):
        blob, src = X.random_blob(name, 1, half_above=480)
        ref = X.cost(X.shape(name), 480, 272)
    assert c["vtsp_bytes"] == len(blob) and "half_above 480" in src
    assert c["macs"]["960x544"] == c["macs"]["480x272"] == ref["macs"]
    assert c["weights_int8"] == ref["weights_int8"] and c["params"] == ref["params"]


def test_vtdt_bytes_of_textspot2_shapes():
    for name, size in (("s8_c24n4x2", 7416), ("s8_c48n4x2", 18568)):
        assert 32 + S3.vtdt_bytes(X.layers(X.shape(name))) == size


def test_receptive_fields_and_figures():
    rf = {n: S3.costs(n)["receptive_field"] for n in S3.CANDIDATES}
    assert rf["t64n4c16"] == 143 and rf["t64n6c16"] == 239 and rf["t64n8c16"] == 287 and rf["t64n8c16d8"] == 383
    c = S3.costs("t64n8c24")
    assert c["weights_int8"] == 40488 and c["vtsp_bytes"] == 51800 and c["macs"]["960x544"] == 86414400
    assert max(S3.costs(n)["vtsp_bytes"] for n in S3.CANDIDATES) < 512000
