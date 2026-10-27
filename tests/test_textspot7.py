"""Tests of textspot7's fallback (F) and post-pass (D): synth/textspot7_twin.py on hand-made regions shaped like the
diagnosis's cases (docs/textspot7.md), runtime/vocr_textspot7.c against the twin on random inputs (every step set,
both modes, the grid's parameters), the fit's grid order, eval/textspot7.py's batch records, and the confirmation
builder's pure helpers. The C parts compile the host driver with the system compiler (cc, or $CC) and are skipped
without one. No screenshots, fonts or models; no network.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "synth", ROOT / "eval", ROOT / "teacher"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot7 as E  # noqa: E402

import textspot7_twin as T7  # noqa: E402

CC = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
SRC = ("vocr_textspot7_test.c", "vocr_textspot7.c", "vocr_lines_refine.c", "vocr_lines.c")
P_FIT = T7.Params("S", 16, 750, 1000)


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    if CC is None:
        pytest.skip("no C compiler")
    d = tmp_path_factory.mktemp("textspot7")
    exe = d / "vocr_textspot7_test"
    subprocess.run([CC, "-std=c99", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                    f"-I{ROOT / 'runtime'}", "-o", str(exe), *[str(ROOT / "runtime" / s) for s in SRC]], check=True)
    return exe


# ----------------------------------------------------------------------------- F on the diagnosis's shapes

CASES = {   # name: (spotter boxes (x, y, w, h, conf), GRAD boxes, the answer under P_FIT)
    "merged by GRAD (v27449): the spotter's two rows": ([(10, 12, 552, 32, 116), (2, 42, 560, 32, 98)],
                                                          [(0, 11, 561, 69)], "S"),
    "merged by the spotter (v3496): GRAD's two rows": ([(8, 10, 408, 30, 95)], [(8, 0, 404, 35), (9, 29, 121, 28)],
                                                       "G"),
    "no spotter box (v8410)": ([], [(0, 8, 237, 32)], "G"),
    "unsure spotter (v20524)": ([(146, 2, 32, 28, 11), (178, 2, 24, 28, 4)], [(6, 8, 205, 19)], "G"),
    "side-by-side choices (v1971): GRAD reads one": ([(0, 0, 289, 39, 109), (470, 0, 300, 41, 91)],
                                                     [(485, 4, 278, 32)], "S"),
    "a thin GRAD junk row above the lines": ([(20, 11, 434, 35, 120), (12, 49, 554, 36, 110)],
                                             [(0, 0, 376, 18), (29, 12, 430, 31), (12, 51, 547, 31)], "S"),
    "both empty": ([], [], "S"),
}


@pytest.mark.parametrize("name", list(CASES))
def test_choose_diagnosis_cases(name):
    S, G, want = CASES[name]
    assert T7.choose(S, G, P_FIT) == want


def test_choose_modes_and_switches():
    S, G, _ = CASES["merged by GRAD (v27449): the spotter's two rows"]
    assert T7.choose(S, G, T7.Params("G", 16, 0, 0)) == "S"          # GRAD merged two spotter rows
    assert T7.choose(S, G, T7.Params("G", 120, 0, 0)) == "G"         # ... but the spotter is not sure enough
    S2, G2, _ = CASES["merged by the spotter (v3496): GRAD's two rows"]
    assert T7.choose(S2, G2, T7.Params("G", 0, 0, 0)) == "G"
    assert T7.choose(S2, G2, T7.Params("S", 0, 0, 0)) == "S"          # lam off: no structure test
    assert T7.choose([], [(0, 0, 10, 10)], T7.Params("G", 0, 0, 0)) == "G"
    assert T7.choose([(0, 0, 10, 10, 1)], [], T7.Params("G", 0, 0, 0)) == "S"
    # S_short: a GRAD line reaching a whole height beyond the spotter's box of its row
    assert T7.choose([(40, 0, 200, 30, 100)], [(0, 0, 240, 30)], T7.Params("S", 0, 0, 1000)) == "G"
    assert T7.choose([(40, 0, 200, 30, 100)], [(20, 0, 220, 30)], T7.Params("S", 0, 0, 1000)) == "S"
    with pytest.raises(ValueError):
        T7.choose([], [], T7.Params("X", 0, 0, 0))


def test_rows_and_medians():
    b = [(0, 0, 10, 10), (20, 4, 10, 10), (0, 30, 10, 10), (40, 34, 10, 10), (80, 2, 10, 10)]
    assert T7.rows_of(b) == [0, 0, 2, 2, 0]
    assert T7.lower_median([5, 1, 9, 3]) == 3
    assert T7.lower_median([7]) == 7


def test_grid_order_and_params():
    g = E.grid()
    assert len(g) == 9 * 4 * 3 + 9
    assert g[0] == T7.Params("S", 96, 600, 500) and g[-1] == T7.Params("G", 0, 0, 0)
    assert E.parse_params("S,8,900,0") == T7.Params("S", 8, 900, 0)
    assert T7.Params("S", 8, 900, 0).name() == "S_t8_l900_k0"


# ----------------------------------------------------------------------------- D

def test_assign_centre_and_overlap():
    region = (494, 492, 738, 540)
    fb = [(34, 495, 700, 43, 104, 0), (500, 500, 100, 30, 90, 1), (480, 530, 100, 30, 50, 0)]
    plain = T7.assign(fb, region, overlap=False)
    assert plain == [(6, 8, 100, 30, 90, 1)]                            # only the centre rule
    over = T7.assign(fb, region, overlap=True)
    assert over[0] == (0, 3, 240, 43, 104, 0) and over[1] == (6, 8, 100, 30, 90, 1)
    assert len(over) == 2                                               # 10 of 30 rows inside: not enough


def test_join():
    b = [(0, 0, 100, 30, 90, 0), (110, 2, 60, 28, 70, 1), (400, 0, 50, 30, 80, 0), (0, 40, 100, 30, 60, 0)]
    j5 = T7.join(b, 5)
    assert j5[0] == (0, 0, 170, 30, 70, 0) and len(j5) == 3            # gap 10 <= 0.5 x 30; dark of the wider
    j10 = T7.join(b, 10)
    assert len(j10) == 3                                                # gap 230 > 30: never joined
    assert T7.join([(0, 0, 50, 30, 9, 0), (20, 0, 50, 30, 5, 1)], 5) == [(0, 0, 70, 30, 5, 0)]


def text_image(h=60, w=400, x0=30, x1=330, seed=0):
    rng = np.random.default_rng(seed)
    g = np.full((h, w), 40, np.uint8)
    for x in range(x0, x1, 12):
        g[20:40, x:x + 8] = 220 if rng.random() > 0.1 else 40
    return g


def test_extend_grows_over_ink_only():
    g = text_image()
    box = (60, 16, 200, 28, 100, 0)                                    # cuts the text at both ends
    (x, y, w, h, c, d), = T7.extend([box], g)
    assert x < 60 and x + w > 260 and (y, h, c, d) == (16, 28, 100, 0)
    assert x >= 30 - 2 - 5 and x + w <= 330 + 2 + 5
    blank = np.full((60, 400), 40, np.uint8)
    assert T7.extend([box], blank) == [box]                             # nothing to grow over
    other = (280, 16, 60, 28, 90, 0)                                    # a box of the same row stops it
    e = T7.extend([box, other], g)
    assert e[0][0] + e[0][2] <= 280


def test_post_steps_and_errors():
    g = text_image()
    fb = [(100 + 60, 200 + 16, 200, 28, 100, 0)]
    region = (100, 200, 500, 260)
    assert T7.post(fb, region, None, []) == [(60, 16, 200, 28, 100, 0)]
    out = T7.post(fb, region, g, ["extend", "tighten"])
    assert len(out) == 1 and out[0][2] > 200
    with pytest.raises(ValueError):
        T7.post(fb, region, g, ["nope"])


# ----------------------------------------------------------------------------- C == twin

def rand_boxes(rng, n, W, H, conf=True):
    out = []
    for _ in range(n):
        w, h = int(rng.integers(4, W // 2)), int(rng.integers(4, 60))
        x, y = int(rng.integers(-20, W)), int(rng.integers(-20, H))
        out.append((x, y, w, h, int(rng.integers(0, 128)), int(rng.integers(0, 2))) if conf else (x, y, w, h))
    return out


def test_c_choose_equals_twin(driver):
    rng = np.random.default_rng(7)
    recs, want = [], []
    params = E.grid()
    for k in range(1500):
        S = [b[:5] for b in rand_boxes(rng, int(rng.integers(0, 7)), 500, 200)]
        G = [b[:4] for b in rand_boxes(rng, int(rng.integers(0, 6)), 500, 200, conf=False)]
        if k % 3 == 0 and S:                                    # stacked rows inside a GRAD box: G_merge cases
            G = [(0, 0, 500, 200)]
        p = params[int(rng.integers(0, len(params)))]
        recs.append(E.rec_choose(S, G, p))
        want.append("C 0" if T7.choose(S, G, p) == "S" else "C 1")
    assert E.run_c(driver, recs) == want


def test_c_post_equals_twin(driver):
    rng = np.random.default_rng(11)
    recs, want = [], []
    combos = [[], ["assign"], ["join5"], ["join10"], ["extend"], ["tighten"], ["assign", "join5", "extend"],
              ["assign", "join10", "extend", "tighten"], ["join5", "tighten"]]
    for k in range(400):
        W, H = int(rng.integers(40, 420)), int(rng.integers(20, 160))
        X0, Y0 = int(rng.integers(0, 200)), int(rng.integers(0, 100))
        region = (X0, Y0, X0 + W, Y0 + H)
        g = text_image(H, W, int(rng.integers(0, W // 3)), int(rng.integers(W // 2, W)), seed=k)
        if k % 2:
            g = (g.astype(np.int16) + rng.integers(-30, 30, g.shape)).clip(0, 255).astype(np.uint8)
        fb = [(x + X0, y + Y0, w, h, c, d) for x, y, w, h, c, d in rand_boxes(rng, int(rng.integers(0, 8)), W, H)]
        steps = combos[k % len(combos)]
        recs.append(E.rec_post(fb, region, g, steps))
        want.append(T7.post(fb, region, g, steps))
    got = [E.parse_post(line) for line in E.run_c(driver, recs)]
    assert got == [[tuple(b) for b in w] for w in want]


def test_c_post_rows_equals_post(driver):
    """vocr_textspot7_post_rows (the row-streamed form, kind 3) gives vocr_textspot7_post's boxes."""
    rng = np.random.default_rng(13)
    recs2, recs3 = [], []
    combos = [["extend"], ["join10", "extend"], ["tighten"], ["assign", "join5", "extend", "tighten"]]
    for k in range(240):
        W, H = int(rng.integers(40, 420)), int(rng.integers(20, 160))
        X0, Y0 = int(rng.integers(0, 200)), int(rng.integers(0, 100))
        region = (X0, Y0, X0 + W, Y0 + H)
        g = text_image(H, W, int(rng.integers(0, W // 3)), int(rng.integers(W // 2, W)), seed=k)
        g = (g.astype(np.int16) + rng.integers(-30, 30, g.shape)).clip(0, 255).astype(np.uint8)
        fb = [(x + X0, y + Y0, w, h, c, d) for x, y, w, h, c, d in rand_boxes(rng, int(rng.integers(0, 8)), W, H)]
        r = E.rec_post(fb, region, g, combos[k % len(combos)])
        recs2.append(r)
        recs3.append(E.i32(3) + r[4:])
    assert E.run_c(driver, recs3) == E.run_c(driver, recs2)


def test_c_rejects_bad_input(driver):
    recs = [E.i32(1, 7, 0, 0, 0, 0, 0)]                          # an unknown mode, no boxes
    assert E.run_c(driver, recs) == ["C E-1"]


# ----------------------------------------------------------------------------- the confirmation builder

def test_confirm_helpers():
    import textspot7_confirm as C
    assert C.review_key("a") == C.review_key("a") != C.review_key("b")
    g = np.full((60, 200), 30, np.uint8)
    g[8:24, 10:190] = 230
    g[34:50, 10:120] = 230
    parts = C.split_rows(g, [0, 0, 200, 60], 2)
    assert len(parts) == 2 and parts[0][3] == parts[1][1]
    assert 24 <= parts[0][3] <= 34


def test_protocol_records_are_counts_only():
    doc = json.loads((ROOT / "eval" / "textspot7_protocol.json").read_text(encoding="utf-8"))
    assert doc["rule"]["confirmation"].startswith("a qualifier must also be not resolved worse")
    fr = json.loads((ROOT / "data" / "manifests" / "textspot7_confirm_frames_v1.json").read_text(encoding="utf-8"))
    kept = [f for f in fr["frames"] if "dropped" not in f]
    assert len(kept) == fr["kept"] and len({f["vn"] for f in kept}) >= 40
    assert all(set(f) <= {"id", "source", "vn", "platform", "url", "sha256", "dims", "release", "review_rank",
                          "dropped"} for f in fr["frames"])
    fit = ROOT / "eval" / "textspot7_fit_20261027.json"
    if fit.exists():
        assert "text" not in fit.read_text(encoding="utf-8").lower().replace("textspot", "")


# ----------------------------------------------------------------------------- the fit of D and the selection

def fake_result(path, errors_by_uid, chars=20, cluster_of=lambda u: u.split(":")[0]):
    units = [{"uid": u, "set": "x", "cluster": cluster_of(u), "platform": "vita", "variant": "base", "errors": e,
              "chars": chars, "lines": 1, "boxes": 1, "whole": 1, "cut": 0, "missed": 0, "noink": 0, "merged": 0,
              "junk": 0, "frames": 10} for u, e in errors_by_uid.items()]
    Path(path).write_text(json.dumps({"units": units}), encoding="utf-8")
    return str(path)


def test_d_fit_rule(tmp_path):
    uids = [f"t{i}:{k}" for i in range(3) for k in range(4)]
    base = {u: 2 for u in uids}
    for r in ("runA", "runB"):
        (tmp_path / "mine" / r).mkdir(parents=True)
        (tmp_path / "stored" / r).mkdir(parents=True)
        fake_result(tmp_path / "stored" / r / "hval_det.json", base)
        fake_result(tmp_path / "mine" / r / "hval_D_plain.json", base)
        for v, e in (("assign", 2), ("join5", 1), ("join10", 1), ("extend", 3), ("tighten", 1)):
            fake_result(tmp_path / "mine" / r / f"hval_D_{v}.json", {u: e for u in uids})
    out = tmp_path / "D.json"
    E.main(["d-fit", "--runs", "runA,runB", "--dir", str(tmp_path / "mine"), "--stored", str(tmp_path / "stored"),
            "--out", str(out)])
    d = json.loads(out.read_text())
    assert d["kept"] == ["join5", "tighten"] and d["D"] == "join5,tighten"     # a join tie goes to join5
    fake_result(tmp_path / "mine" / "runB" / "hval_D_plain.json", {**base, uids[0]: 5})
    with pytest.raises(SystemExit):
        E.main(["d-fit", "--runs", "runA,runB", "--dir", str(tmp_path / "mine"), "--stored",
                str(tmp_path / "stored"), "--out", str(out)])


def test_select_two_phases(tmp_path):
    import textspot7_select as SEL
    vns = [f"v{i}" for i in range(12)]
    uid = [f"{v}:{k}" for v in vns for k in range(3)]
    g = fake_result(tmp_path / "grad.json", {u: 4 for u in uid})
    fix = fake_result(tmp_path / "fix.json", {u: 3 for u in uid})
    good = fake_result(tmp_path / "good.json", {u: 1 for u in uid})
    meh = fake_result(tmp_path / "meh.json", {u: (1 if i % 2 else 9) for i, u in enumerate(uid)})
    cg = fake_result(tmp_path / "cg.json", {u: 2 for u in uid})
    cf = fake_result(tmp_path / "cf.json", {u: 2 for u in uid})
    cand = {"arm": "a", "run": "a_s1", "vtsp_bytes": 52176, "est_ms_333": 280.0, "est_ms_444": 210.0, "arena": 244607,
            "c_equal": True}
    tab = {"baseline": {"dev": g, "hval": g}, "rubi_next": {"name": "rubi,fix=1", "dev": fix, "hval": fix,
                                                            "confirm": cf},
           "seen_dev_clusters": [],
           "candidates": [dict(cand, name="a|F", decode="F", dev=good, hval=good, confirm=cg),
                          dict(cand, name="a|D", decode="D", dev=meh, hval=meh),
                          dict(cand, name="b|D", decode="D", dev=good, hval=good, est_ms_333=301.0)]}
    tp = tmp_path / "tab.json"
    tp.write_text(json.dumps(tab))
    dev = tmp_path / "dev.json"
    SEL.main(["--table", str(tp), "--phase", "dev", "--out", str(dev)])
    d = json.loads(dev.read_text())
    assert d["qualifiers"] == ["a|F"]                                    # b|D is not eligible (301 ms)
    conf = tmp_path / "conf.json"
    SEL.main(["--table", str(tp), "--phase", "confirm", "--dev-record", str(dev), "--out", str(conf)])
    c = json.loads(conf.read_text())
    assert c["pick"] == "a|F" and c["rubi_default"]["finder"] == "spotter"
    tab["candidates"][0]["confirm"] = fake_result(tmp_path / "cbad.json", {u: 9 for u in uid})
    tp.write_text(json.dumps(tab))
    SEL.main(["--table", str(tp), "--phase", "confirm", "--dev-record", str(dev), "--out", str(conf)])
    assert json.loads(conf.read_text())["pick"] is None
