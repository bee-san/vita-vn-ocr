"""textspot2's shape registry (train/textspot2_shapes.py) against the protocol and train/textdet_model."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
import textdet_model as TM  # noqa: E402
import textspot2_shapes as S2  # noqa: E402


def test_register_is_idempotent_and_matches_textspot():
    S2.register()
    S2.register()
    for name in S2.VTDT:
        assert TM.SHAPES[name]["name"] == name
    # the two textspot shapes are textdet_model's own, unchanged
    assert TM.SHAPES["s4p_c8"]["block"] == "plain" and TM.SHAPES["s4p_c8"]["chans"] == [8, 8, 8]
    assert TM.SHAPES["s4_c24n4"]["dil"] == [1, 2, 2, 1]
    assert not set(S2.D2S) & set(TM.SHAPES)


def test_costs_of_textspot_shapes_are_textspots():
    # eval/textspot_dev_20261019.json: macs_960 of the two references at half_above 600
    assert S2.costs("s4p_c8")["macs"]["960x544"] == 17886720
    assert S2.costs("s4_c24n4")["macs"]["960x544"] == 28462080
    c = S2.costs("s4p_c10n4")
    assert c["weights_int8"] == 4036 and c["receptive_field"] == 127
    assert c["macs"]["960x544"] == c["macs"]["480x272"]      # a Vita frame is read at the PSP's size


def test_every_layer_chain_is_valid():
    for name in S2.VTDT:
        layers = TM.layers(TM.SHAPES[name])
        assert layers[0]["cin"] == 1 and layers[-1]["cout"] == TM.N_OUT and layers[-1]["head"]
        for a, b in zip(layers, layers[1:]):
            assert a["cout"] == b["cin"]
        assert max(L["cout"] for L in layers) <= 64          # the runtime's MAXCH
        assert sum(1 for L in layers) <= 32                 # VOCR_TEXTDET_MAX_LAYERS
