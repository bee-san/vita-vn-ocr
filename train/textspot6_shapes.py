"""textspot6's shapes (issue #128, docs/textspot6.md "Model"): textspot3's family-T trunks and family-M multi-scale
networks with the tagged head (train/textspot6_model.py), and family-T trunks read at full resolution (the
input-scale arm). Each shape names its scale: half_above (a frame wider than this is halved; 0: never) and, for M,
half_above_h.

The registry is CANDIDATES until eval/textspot6_protocol.json registers shapes by an amendment (REGISTERED then
lists them and SHAPES holds only those).
"""
import json
from pathlib import Path

import textspot3_model as M3
import textspot3_shapes as S3

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "eval" / "textspot6_protocol.json"


def t_shape(name, c2, chans, dil, half_above, c0=8, c1=16):
    return {"name": name, "family": "T", "stride": 8, "c0": c0, "c1": c1, "c2": c2, "chans": list(chans),
            "dil": list(dil), "block": "sep", "head": "kroa_x2", "half_above": half_above}


CANDIDATES = {}
# family T at textspot3's scale (a frame wider than 480 px is halved): textspot3's registered shapes
for _n in ("t64n8c16", "t48n8c16d8", "t64n4c16"):
    CANDIDATES[_n] = dict(S3.SHAPES[_n])
# family T at full resolution (half_above 0): the input-scale arm; narrower trunks so a 960 x 544 frame fits the
# time budget
CANDIDATES["t32n4c16_f1"] = t_shape("t32n4c16_f1", 32, [32] * 4, [1, 2, 4, 1], 0)
CANDIDATES["t32n6c16_f1"] = t_shape("t32n6c16_f1", 32, [32] * 6, [1, 2, 4, 1, 2, 4], 0)
CANDIDATES["t24n6c16_f1"] = t_shape("t24n6c16_f1", 24, [24] * 6, [1, 2, 4, 1, 2, 4], 0)
CANDIDATES["t48n4c16_f1"] = t_shape("t48n4c16_f1", 48, [48] * 4, [1, 2, 4, 1], 0)
# family M: textspot3's size points S, M, L
for _n in ("qS4", "qM4", "qL5"):
    _s = dict(M3.shape_of(_n))
    _s["half_above"], _s["half_above_h"] = 480, 272
    CANDIDATES[_n] = _s


def _registered():
    if not PROTOCOL.exists():
        return None
    doc = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    names = doc.get("shapes", {}).get("registered")
    return list(names) if names else None


REGISTERED = _registered()
SHAPES = {k: CANDIDATES[k] for k in REGISTERED} if REGISTERED else dict(CANDIDATES)
