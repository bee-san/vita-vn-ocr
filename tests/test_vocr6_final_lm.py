"""vocr6-final-lm (#125, #127): eval/vocr6_final_lm.py's selection, rule and counter-set helpers, and the protocol's
consistency, on generated fixtures only (no hooked, development or protected data is read here)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "eval", ROOT / "runtime", ROOT / "lm"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import vocr6_final_lm as F  # noqa: E402
import vocr6_lm as V  # noqa: E402

PROTO = json.loads((ROOT / "eval" / "vocr6_final_lm_protocol.json").read_text(encoding="utf-8"))


def test_first_min_ties_go_to_the_earlier_option():
    v = np.array([3.0, 1.0, 1.0, 2.0])
    assert F.first_min(v, [0, 1, 2, 3]) == 1
    assert F.first_min(v, [2, 0, 1, 3]) == 2           # the shipped option listed first wins a tie
    assert F.first_min(v + np.array([0, 1e-13, 0, 0]), [2, 1]) == 2


def test_cv_choose_and_errors():
    obj0, obj1, obj = np.array([1.0, 0.5, 0.7]), np.array([0.4, 0.9, 0.4]), np.array([0.7, 0.7, 0.55])
    c0, c1, ca = F.cv_choose(obj0, obj1, obj, [0, 1, 2])
    assert (c0, c1, ca) == (1, 0, 2)
    e = np.arange(12).reshape(3, 4)
    fold = np.array([0, 1, 0, 1])
    # fold 0's units take the choice made on fold 1 (c1 = 0), fold 1's the choice made on fold 0 (c0 = 1)
    assert F.cv_errors(fold, e, c0, c1).tolist() == [0, 5, 2, 7]


def test_cs_compare():
    cl = ["a", "a", "b", "b", "c", "c"]
    ch = [10] * 6
    ex = np.array([1, 0, 2, 0, 0, 1])
    kx = np.array([1, 1, 0, 1, 1, 0], bool)
    same = F.cs_compare(cl, ch, ex, ex, kx, kx, n_boot=200)
    assert same["cer"] == {"delta_points": 0.0, "ci95": [0.0, 0.0]}
    assert same["kept_share"]["ci95"] == [0.0, 0.0] and same["kept"] == [4, 4] and same["clusters"] == 3
    ey = ex + 1                                        # one more error on every crop: +10 points everywhere
    ky = np.zeros(6, bool)                             # every counter lost
    d = F.cs_compare(cl, ch, ex, ey, kx, ky, n_boot=200)
    assert d["cer"]["delta_points"] == pytest.approx(10.0) and d["cer"]["ci95"] == [pytest.approx(10.0)] * 2
    assert d["kept_share"]["delta_points"] == pytest.approx(-100 * 4 / 6, abs=1e-3)
    assert d["kept_share"]["ci95"][1] < 0 and d["errors"] == [4, 10] and d["crops"] == 6 and d["chars"] == 60


def _cmp(obj, hp=(-1, 1), hc=(-1, 1), dev=(-1, 1)):
    return {"obj": {"ci95": list(obj)}, "hp": {"ci95": list(hp)}, "hc": {"ci95": list(hc)}, "dev": {"ci95": list(dev)}}


def _cs(cer=(-1, 1), kept=(-1, 1)):
    return {"cer": {"ci95": list(cer)}, "kept_share": {"ci95": list(kept)}}


def test_apply_rule():
    assert F.apply_rule(_cmp((-0.3, -0.01)), _cs(), True)["passes"]
    assert not F.apply_rule(_cmp((-0.3, 0.01)), _cs(), True)["passes"]            # objective not resolved
    assert not F.apply_rule(_cmp((-0.3, -0.01), dev=(0.001, 0.2)), _cs(), True)["passes"]
    assert not F.apply_rule(_cmp((-0.3, -0.01), hc=(0.001, 0.2)), _cs(), True)["passes"]
    assert not F.apply_rule(_cmp((-0.3, -0.01), hp=(0.001, 0.2)), _cs(), True)["passes"]
    assert not F.apply_rule(_cmp((-0.3, -0.01)), _cs(cer=(0.01, 0.5)), True)["passes"]
    assert not F.apply_rule(_cmp((-0.3, -0.01)), _cs(kept=(-2.0, -0.1)), True)["passes"]
    assert F.apply_rule(_cmp((-0.3, -0.01)), _cs(kept=(-2.0, 0.0)), True)["passes"]   # 0 inside: not lower
    assert not F.apply_rule(_cmp((-0.3, -0.01)), _cs(), False)["passes"]


def test_decide():
    yes, no = {"passes": True}, {"passes": False}
    assert F.decide(no, no, -1, -1) == "keep"
    assert F.decide(yes, no, -0.1, -0.5) == "b"
    assert F.decide(no, yes, -0.1, -0.5) == "c"
    assert F.decide(yes, yes, -0.1, -0.5) == "c" and F.decide(yes, yes, -0.5, -0.1) == "b"
    assert F.decide(yes, yes, -0.2, -0.2) == "b"


def test_admissible():
    cfg = {"gate": 8000}
    assert F.admissible(cfg, (6000, 4000), 41632, 49152)
    assert F.admissible(cfg, (8000, 5000), 41632, 49152)          # at the gate: allowed (protocol arms.c.note)
    assert not F.admissible({"gate": 4000}, (6000, 4000), 41632, 49152)
    assert not F.admissible(cfg, (6000, 4000), 49153, 49152)


def test_kept_test():
    from metrics import norm
    for s, want in (("九ヶ月です", True), ("「九月です」", False), ("ーヶ月", False), ("１ヶ所", True),
                    ("何ヵ国", True), ("数ケ年", True), ("十箇条", True), ("ルッ月", False), ("三ヶ日", False)):
        assert (F.SITE.search(norm(s)) is not None) == want, s


def test_protocol_grid_and_pins():
    assert PROTO["kind"] == "vocr6-final-lm-protocol"
    for lt in ("500k", "1000k", "2000k", "4000k"):
        cfgs, ship, cells, thrs = F.protocol_grid(PROTO, lt)
        assert ship in cfgs and len(cfgs) == 352 and cfgs.count(ship) == 1
        assert PROTO["lm_tiers"][lt]["cascade_conf"] in thrs and thrs == [500, 600, 700]
        assert len(cells) == 15 and (6000, 4000) in cells
        for src in ("dev", "hval"):
            pin = PROTO["lm_tiers"][lt][src]
            assert len(pin["sha256"]) == 64 and int(pin["sha256"], 16) >= 0 and pin["path"].endswith(".vlm")
        assert all(c["gate"] == 8000 and c["beam"] == 8 and c["topk"] == 8 for c in cfgs)
        assert all(F.admissible(c, (6000, 4000), 0, 1) for c in cfgs)
    assert PROTO["lm_tiers"]["1000k"]["shipped"]["flags"] == 1 and PROTO["lm_tiers"]["500k"]["shipped"]["flags"] == 2
    assert PROTO["counter_shipped"]["tau"] == 6000 and PROTO["counter_shipped"]["open"] == 4000
    # the game-text tiers read hval with their screening builds, the ctclm.2 tiers with their own bytes
    assert PROTO["lm_tiers"]["1000k"]["dev"]["sha256"] != PROTO["lm_tiers"]["1000k"]["hval"]["sha256"]
    assert PROTO["lm_tiers"]["500k"]["dev"] == PROTO["lm_tiers"]["500k"]["hval"]


def test_option_order_and_paths():
    order = F.option_order(3, 1, [500, 600, 700], 700)
    assert order[0] == (1, 700) and len(order) == 9 and len(set(order)) == 9
    assert F.option_order(2, 0, ["none"], "none") == [(0, "none"), (1, "none")]
    assert F.path_keys([500, 700]) == [("S", "none"), ("S", 500), ("S", 700), ("N", "none"), ("N", 500),
                                       ("N", 700), ("L", "none")]


def test_job_texts_cascade():
    class T:
        def __init__(self, conf):
            self.conf = conf
    tiers = {"S": T({"a": 650, "b": 900}), "N": T({"a": 10, "b": 10})}
    tx = {"S": {"a": "s1", "b": "s2"}, "N": {"a": "n1", "b": "n2"}}
    assert F.job_texts(tx, tiers, "S", "none") == {"a": "s1", "b": "s2"}
    assert F.job_texts(tx, tiers, "S", 700) == {"a": "n1", "b": "s2"}
    assert F.job_texts(tx, tiers, "S", 600) == {"a": "s1", "b": "s2"}


def test_counter_units_and_kept(tmp_path):
    """counter_units on a generated set in kanum_dev's file format: one unit per crop, cluster = line // 3; cs_kept
    on raw texts (no display rules)."""
    meta, imgs = [], []
    texts = ["「九ヶ月です」", "九ヶ月です", "九ヶ月です」", "「三ヵ所」", "三ヵ所", "三ヵ所」"]
    for i, t in enumerate(texts):
        for cut in (3, 0):
            meta.append({"id": f"sy{i:04d}c{cut}", "line": i, "source": "x.dev.txt", "text": t, "font": "f",
                         "size": 24, "cut": cut})
            imgs.append(np.full((4, 6), i, np.uint8))
    offs = np.cumsum([0] + [x.size for x in imgs])
    p = tmp_path / "synth.npz"
    np.savez_compressed(p, data=np.concatenate([x.reshape(-1) for x in imgs]), offsets=offs,
                        shapes=np.array([x.shape for x in imgs], np.int32), meta=json.dumps(meta, ensure_ascii=False))
    units, rows = F.counter_units(p)
    assert len(units) == 12 and {u.cluster for u in units} == {"cs000", "cs001"}
    assert all(u.part == "cs" and u.segs[0][1] == (("cs", u.uid),) for u in units)
    pred = {("cs", u.uid): ("九月です」" if u.uid.startswith("sy0001") else r["text"]) for u, r in zip(units, rows)}
    kept, err = F.cs_kept(units, pred, None)
    assert kept.sum() == 10 and err.sum() == 2 * 2         # sy0001 (2 crops): 九月です」 against 九ヶ月です


def test_lev_matches_metrics():
    from metrics import lev
    rng = np.random.default_rng(0)
    al = "あいうヶ月九"
    for _ in range(200):
        a = "".join(rng.choice(list(al), rng.integers(0, 8)))
        b = "".join(rng.choice(list(al), rng.integers(0, 8)))
        assert V.lev(a, b) == lev(a, b)
