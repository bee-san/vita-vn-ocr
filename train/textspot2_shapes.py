"""textspot2's network shapes (issue #106, docs/textspot2.md "Shapes", eval/textspot2_protocol.json "shapes").

    import textspot2_shapes as S2
    S2.register()                  # adds the protocol's VTDT shapes to train/textdet_model.SHAPES (idempotent)
    S2.SHAPES["s4p_c8n4"]          # every shape the protocol names, with its head ("kroa" or "kroa_x2")
    S2.D2S                         # the contingent d2s shapes (head "kroa_x2"): never registered in TM.SHAPES

train/textdet_model.py is pinned by both protocols and stays unchanged: its layer grammar (stem, down, down8, blocks,
a 1x1 head) builds every "kroa" shape here, and register() only adds entries to its SHAPES dict at import time of
the caller. A "kroa_x2" shape (a 2 x 2 sub-pixel head: 16 head channels on the stride-8 trunk) needs the d2s
extension's own layer list and export (docs/textspot2.md "d2s"), so it is listed in D2S and not registered.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "train") not in sys.path:
    sys.path.insert(0, str(ROOT / "train"))
import textdet_model as TM  # noqa: E402

PROTOCOL = ROOT / "eval" / "textspot2_protocol.json"
FIELDS = ("stride", "c0", "c1", "c2", "chans", "dil", "block", "head")


def _load(path=PROTOCOL):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for name, s in doc["shapes"].items():
        if not isinstance(s, dict):
            continue
        out[name] = {"name": name, **{k: s[k] for k in FIELDS}}
        out[name]["chans"], out[name]["dil"] = list(s["chans"]), list(s["dil"])
        if s.get("contingent"):
            out[name]["contingent"] = s["contingent"]
    return out


SHAPES = _load()
D2S = {k: v for k, v in SHAPES.items() if v["head"] == "kroa_x2"}
VTDT = {k: v for k, v in SHAPES.items() if v["head"] == "kroa"}


def _plain(s):
    return {k: s[k] for k in ("name", "stride", "c0", "c1", "c2", "chans", "dil", "block", "head")}


def register():
    """Add every "kroa" shape of the protocol to train/textdet_model.SHAPES. A name textdet_model already defines
    must be the same shape (textspot's s4p_c8 and s4_c24n4 are); anything else is an error."""
    for name, s in VTDT.items():
        want = _plain(s)
        have = TM.SHAPES.get(name)
        if have is None:
            TM.SHAPES[name] = want
        elif _plain(have) != want:
            raise ValueError(f"{name}: train/textdet_model.SHAPES defines another shape under this name")
    return dict(TM.SHAPES)


def costs(name, sizes=((960, 544), (480, 272)), half_above=600):
    """Exact MACs per frame at the network's input size (a frame wider than half_above is halved), parameters and
    int8 weights (train/textdet_model.cost) of a registered shape."""
    register()
    shape = TM.SHAPES[name]
    out = {}
    for w, h in sizes:
        iw, ih = ((w + 1) // 2, (h + 1) // 2) if half_above and w > half_above else (w, h)
        out[f"{w}x{h}"] = TM.cost(shape, iw, ih)["macs"]
    c = TM.cost(shape, 480, 272)
    return {"macs": out, "params": c["params"], "weights_int8": c["weights_int8"],
            "receptive_field": TM.receptive_field(shape)}
