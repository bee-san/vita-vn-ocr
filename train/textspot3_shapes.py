"""textspot3's network shapes (issue #108, docs/textspot3.md "Two families", eval/textspot3_protocol.json "shapes").

    import textspot3_shapes as S3
    S3.SHAPES                  # every shape by name: the protocol's registered shapes, else the candidates below
    S3.REGISTERED              # True once eval/textspot3_protocol.json registers shapes (an amendment)
    S3.T, S3.M                 # the shapes of family T (VTSP version 2, d2s) and family M (VTSP version 3)
    S3.costs(name)             # MACs per 960x544 and 480x272 frame, parameters, int8 weights, VTSP bytes, RF
    S3.inject()                # context: the T shapes visible to train/textspot2_shapes (S2.SHAPES, S2.D2S), so
                               # train/textspot2_export's frozen layers, costs and random blobs build them

Family T is the d2s trunk scaled up (docs/textspot3.md "T, the trunk"): train/textdet_model.layers' grammar (stem
3x3 stride 2 1 -> c0, down sep c0 -> c1, down8 sep c1 -> c2, separable blocks chans / dil) with the 1x1 head to 16
channels (head "kroa_x2"), VTSP version 2, runtime/vocr_textspot2.c as merged, half_above 480. Its layers, costs,
quantisation and blobs are train/textspot2_export.py's, unchanged: textspot2_export.shape(name) reads
train/textspot2_shapes.SHAPES, so inject() adds the T shapes there for the duration of a call (and removes them).

Family M is the multi-scale VTSP version 3 (train/textspot3_model.py's grammar, synth/textspot3_twin.py's format);
its probe shapes come with the version-3 runtime (textspot3_model.PROBES) and are measured, not registered.

Until the conductor registers shapes by an amendment (the protocol's "shapes", or an amendment's), SHAPES holds
CANDIDATES: **candidates, not registered**. They are the grid the conductor asked for (T_ROWS): every one "sep",
head kroa_x2, c0 8, c1 16 or 24, a stride-8 trunk of n blocks of 64 channels (n 4, 6, 8; dilations 1,2,4,1 /
1,2,4,1,2,4 / 1,2,4,1,2,4,1,2 and, for n = 8, 1,2,4,8,1,2,4,1) or of 48 channels (n 6, 8).
"""
import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PROTOCOL = ROOT / "eval" / "textspot3_protocol.json"
T_FIELDS = ("stride", "c0", "c1", "c2", "chans", "dil", "block", "head")
T_HALF_ABOVE = 480                 # docs/textspot3.md: a frame wider than 480 px is halved
SIZES = ((960, 544), (480, 272))
DIL = {4: (1, 2, 4, 1), 6: (1, 2, 4, 1, 2, 4), 8: (1, 2, 4, 1, 2, 4, 1, 2)}
DIL8 = (1, 2, 4, 8, 1, 2, 4, 1)    # n = 8 only: a dilation of 8 (a larger receptive field, a 17-row ring)


def t_shape(name, c1, c2, n, dil, c0=8):
    """A family-T shape dict: textspot2_shapes' fields, the family, the scale."""
    return {"name": name, "family": "T", "stride": 8, "c0": c0, "c1": c1, "c2": c2, "chans": [c2] * n,
            "dil": list(dil), "block": "sep", "head": "kroa_x2", "half_above": T_HALF_ABOVE}


def _candidates():
    out = {}
    for c2, ns in ((64, (4, 6, 8)), (48, (6, 8))):
        for c1 in (16, 24):
            for n in ns:
                for suffix, dil in (("", DIL[n]), ("d8", DIL8)):
                    if suffix and n != 8:
                        continue
                    name = f"t{c2}n{n}c{c1}{suffix}"
                    out[name] = t_shape(name, c1, c2, n, dil)
    return out


# candidates, not registered: the T_ROWS grid (eval/textspot3_a9.py measures them; the conductor registers shapes
# by a protocol amendment, and then SHAPES below is the protocol's)
CANDIDATES = _candidates()


def _from_protocol(path=PROTOCOL):
    """The shapes eval/textspot3_protocol.json registers: its "shapes" and every amendment's "shapes" (later ones
    win), each a dict with "family" "T" or "M" ({} when none is registered yet)."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    found = {}
    for block in [doc.get("shapes")] + [a.get("shapes") for a in doc.get("amendments", []) if isinstance(a, dict)]:
        if not isinstance(block, dict):
            continue
        for name, s in block.items():
            if isinstance(s, dict) and s.get("family") in ("T", "M"):
                found[name] = _normal(name, s)
    return found


def _normal(name, s):
    s = json.loads(json.dumps(s))      # a deep, plain copy
    s["name"] = name
    if s["family"] == "T":
        for k in T_FIELDS:
            if k not in s:
                raise ValueError(f"{name}: a T shape needs {k}")
        s["chans"], s["dil"] = list(s["chans"]), list(s["dil"])
        s.setdefault("half_above", T_HALF_ABOVE)
        if s["head"] != "kroa_x2" or s["stride"] != 8:
            raise ValueError(f"{name}: a T shape is a stride-8 trunk with head kroa_x2")
    return s


_REG = _from_protocol()
REGISTERED = bool(_REG)
SHAPES = _REG if REGISTERED else dict(CANDIDATES)
T = {k: v for k, v in SHAPES.items() if v["family"] == "T"}
M = {k: v for k, v in SHAPES.items() if v["family"] == "M"}


def t_plain(s):
    """The textspot2_shapes form of a T shape (S2.SHAPES' entries; textspot2_export.shape reads S2.FIELDS)."""
    return {"name": s["name"], **{k: (list(s[k]) if isinstance(s[k], list) else s[k]) for k in T_FIELDS}}


@contextlib.contextmanager
def inject(shapes=None):
    """train/textspot2_shapes.SHAPES and .D2S see the T shapes (default: every T shape here) inside the block, so
    train/textspot2_export's layers / cost / random_blob and eval/textspot2_a9's v2 runtime build them with their
    frozen code; a name textspot2 defines already must be that same shape. Removed again on exit."""
    import textspot2_shapes as S2
    shapes = T if shapes is None else shapes
    added = []
    try:
        for name, s in shapes.items():
            want = t_plain(s)
            have = S2.SHAPES.get(name)
            if have is not None:
                if {k: have[k] for k in T_FIELDS} != {k: want[k] for k in T_FIELDS}:
                    raise ValueError(f"{name}: train/textspot2_shapes defines another shape under this name")
                continue
            S2.SHAPES[name] = want
            S2.D2S[name] = want
            added.append(name)
        yield
    finally:
        for name in added:
            S2.SHAPES.pop(name, None)
            S2.D2S.pop(name, None)


def vtdt_bytes(layers):
    """The exact VTDT size of a network of these layers (train/textdet_export.pack's layout: a 64-byte header,
    32-byte records, per layer int8 weights, int32 bias, int32 multiplier, u8 shift, each section 4-aligned)."""
    def a4(n):
        return (n + 3) & ~3
    total = 64 + 32 * len(layers)
    for L in layers:
        per = 1 if L["kind"] == "dw" else L["cin"]
        total += a4(L["cout"] * per * L["k"] * L["k"]) + 8 * L["cout"] + a4(L["cout"])
    return total


def costs(name, sizes=SIZES):
    """Exact figures of a shape: MACs per frame at the network's input (a T frame wider than half_above is halved;
    an M frame by version 3's rule), parameters, int8 weights, the VTSP bytes and the receptive field (input px).
    A name the protocol does not register is looked up among the T_ROWS candidates."""
    s = SHAPES[name] if name in SHAPES else CANDIDATES[name]
    if s["family"] == "T":
        import textdet_model as TM
        import textspot2_export as X
        ha = s.get("half_above", T_HALF_ABOVE)
        spec = X.layers(t_plain(s))
        macs = {}
        for w, h in sizes:
            iw, ih = ((w + 1) // 2, (h + 1) // 2) if ha and w > ha else (w, h)
            macs[f"{w}x{h}"] = X.cost(t_plain(s), iw, ih)["macs"]
        c = X.cost(t_plain(s), 480, 272)
        return {"family": "T", "macs": macs, "params": c["params"], "weights_int8": c["weights_int8"],
                "vtsp_bytes": 32 + vtdt_bytes(spec), "receptive_field": TM.receptive_field(t_plain(s)),
                "layers": len(spec), "half_above": ha}
    import textspot3_model as M3
    return M3.costs(s, sizes)
