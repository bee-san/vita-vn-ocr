"""Tests for devnative/ (the native development cohort): ids and hashes, the selection rules,
line geometry and the GRAD crops, contact sheets, the decision rules and the npz format.
No network, screenshots or shared artifacts are needed."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "synth"))

from common import is_test_vn  # noqa: E402
from devnative import build as B  # noqa: E402
from devnative import decide as D  # noqa: E402
from devnative import lines as L  # noqa: E402
from devnative import paths as P  # noqa: E402
from devnative import select as SEL  # noqa: E402
from devnative import sheets as S  # noqa: E402

# ----------------------------------------------------------------------------- ids, hashes

def test_image_ids():
    url = "https://t.vndb.org/sf/46/97646.jpg"
    for ref in (url, "v36_97646.jpg", "v36_97646", "sf97646", "/x/shots/vita/v36_97646.jpg"):
        assert P.image_num(ref) == 97646
    assert P.image_id(url) == "sf97646"
    assert P.shot_key("v36", url) == "v36_97646"
    assert P.vndb_url(97646) == url
    assert P.vndb_url(8367) == "https://t.vndb.org/sf/67/8367.jpg"
    with pytest.raises(ValueError):
        P.image_num("no-number-here")


def test_pixel_sha256():
    a = np.zeros((4, 6, 3), np.uint8)
    b = a.copy()
    assert P.pixel_sha256(a) == P.pixel_sha256(b)
    b[1, 2, 0] = 1
    assert P.pixel_sha256(a) != P.pixel_sha256(b)
    assert P.pixel_sha256(np.zeros((6, 4, 3), np.uint8)) != P.pixel_sha256(a)   # shape is hashed


def test_thumb_unit_norm():
    rng = np.random.default_rng(0)
    t = P.thumb(rng.integers(0, 255, (272, 480, 3), dtype=np.uint8))
    assert t.shape == (576,) and abs(float(t @ t) - 1) < 1e-3


# ----------------------------------------------------------------------------- selection rules

def _ctx(**kw):
    test_vn = next(f"v{i}" for i in range(1, 10000) if is_test_vn(f"v{i}"))
    ok = [f"v{i}" for i in range(100, 10000) if not is_test_vn(f"v{i}")][:6]
    ctx = {"reserved": set(ok) | {test_vn, "v4"}, "test_vns": set(), "test_sessions": {"r9"},
           "clannad": {"v4"}, "label_vns": {ok[1]}, "label_releases": {"r8"}, "scene_vns": {ok[2]},
           "ref_roles": {}, "tv_shots": {}}
    ctx.update(kw)
    return ctx, test_vn, ok


def _row(vn, num=1000, kind="psp", langs=("ja",), release="r1", sexual=0.0, violence=0.0):
    plat, dims = (["psp"], [480, 272]) if kind == "psp" else (["psv"], [960, 544])
    return {"vn": vn, "url": P.vndb_url(num), "dims": dims, "platforms": plat, "langs": list(langs),
            "release": release, "sexual": sexual, "violence": violence}


def test_native_kind():
    assert SEL.native_kind(_row("v1", kind="vita")) == "vita"
    assert SEL.native_kind(_row("v1", kind="psp")) == "psp"
    r = _row("v1", kind="psp")
    r["dims"] = [960, 544]
    assert SEL.native_kind(r) is None                    # a PSP release at Vita size is not native
    r = _row("v1", kind="vita")
    r["platforms"] = ["win"]
    assert SEL.native_kind(r) is None


def test_shot_exclusions():
    ctx, test_vn, ok = _ctx()
    assert SEL.shot_exclusion(_row("v99999"), ctx) == "not-reserved"
    assert SEL.shot_exclusion(_row(test_vn), ctx) == "test-vn"
    assert SEL.shot_exclusion(_row("v4"), ctx) == "clannad-family"
    assert SEL.shot_exclusion(_row(ok[1]), ctx) == "label-pool-vn"
    assert SEL.shot_exclusion(_row(ok[0], langs=("en", "ja")), ctx) == "language"
    assert SEL.shot_exclusion(_row(ok[0], sexual=0.5), ctx) == "rating"
    assert SEL.shot_exclusion(_row(ok[0], release="r9"), ctx) == "test-session"
    assert SEL.shot_exclusion(_row(ok[0], release="r8"), ctx) == "label-release"
    ctx["ref_roles"] = {1000: {"eval-test"}, 1001: {"label-train"}}
    assert SEL.shot_exclusion(_row(ok[0], 1000), ctx) == "eval-shot"
    assert SEL.shot_exclusion(_row(ok[0], 1001), ctx) == "label-shot"
    assert SEL.shot_exclusion(_row(ok[0], 1002), ctx) is None
    ctx["fresh_vns"] = {ok[3]}
    assert SEL.shot_exclusion(_row(ok[3], 1003), ctx) == "fresh-test-vn"
    ctx["test_vns"] = {ok[0]}
    assert SEL.shot_exclusion(_row(ok[0], 1002), ctx) == "test-vn"


def test_fresh_native_titles_protected():
    """The fresh native textbox test's eight titles (PR #49) are excluded with their relations."""
    assert set(SEL.FRESH_NATIVE) == {"v14909", "v25459", "v29454", "v21289", "v21976", "v20519", "v16509", "v19140"}
    if SEL.FRESH_PROVENANCE.exists():
        vns, shas, pix = SEL.fresh_sources()
        assert vns <= set(SEL.FRESH_NATIVE) and shas and pix
    ctx = SEL.context([], {"relations": [{"id": "v77777"}]}, {"v21976": {"relations": [{"id": "v88888"}]}})
    assert {"v21976", "v88888"} <= ctx["fresh_vns"] and "v77777" in ctx["clannad"]


def test_subsets():
    ctx, _, ok = _ctx(ref_roles={2000: {"teacher-val", "scene", "eval-linefind"}, 2001: {"eval-general-vn"},
                                 2002: {"scene"}})
    sub, flags = SEL.subset_of(_row(ok[0], 2000), ctx)
    assert sub == "tv_native" and set(flags) == {"bg_shot", "linefind"}
    assert SEL.subset_of(_row(ok[0], 2001), ctx) == (None, "other-set-shot")
    assert SEL.subset_of(_row(ok[2], 2002), ctx) == (None, "scene-shot")
    assert SEL.subset_of(_row(ok[2], 2003), ctx) == ("scene_vn", ["scene_vn"])
    assert SEL.subset_of(_row(ok[0], 2004), ctx) == ("strict", [])
    ctx["tv_shots"] = {P.shot_key(ok[3], P.vndb_url(2005)): ["x_0"]}
    assert SEL.subset_of(_row(ok[3], 2005), ctx)[0] == "tv_native"


def test_same_frame():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 255, (272, 480, 3), dtype=np.uint8)
    b = np.clip(a.astype(np.int16) + rng.integers(-3, 4, a.shape), 0, 255).astype(np.uint8)
    assert SEL.same_frame(a, b)
    c = a.copy()
    c[200:260, 20:460] = 255 - c[200:260, 20:460]        # other dialogue in the box
    assert not SEL.same_frame(a, c)


# ----------------------------------------------------------------------------- geometry

def _line(ink, k=0, furi=None):
    ln = {"k": k, "ink": list(ink), "crop": list(ink), "parts": [list(ink)], "hint": {"bright_frac": 0.9}}
    if furi:
        ln["furi"] = [list(furi)]
        ln["crop"] = L.union([ink, furi])
    return ln


def test_crop_box_and_regions():
    a, b, c = _line([20, 200, 300, 220]), _line([20, 228, 280, 248], 1), _line([20, 20, 120, 40], 2)
    assert L.crop_box(a, "psp", 480, 272) == [18, 198, 302, 222]
    assert L.crop_box(a, "vita", 960, 544) == [16, 196, 304, 224]
    regs = L.regions([a, b, c], "psp", 480, 272)
    assert [m for _, m in regs] == [[2], [0, 1]]           # top to bottom; a and b sit together
    assert regs[1][0] == [14, 194, 306, 254]                # union + 6 px on PSP
    f = _line([30, 100, 200, 120], furi=[60, 90, 90, 98])
    assert L.text_box(f) == [30, 90, 200, 120] and L.crop_box(f, "psp", 480, 272) == [28, 88, 202, 122]


def test_match_and_status():
    ln = _line([20, 200, 300, 220])
    assert L.match([[16, 196, 305, 224]], [L.text_box(ln)]) == {0: 0}
    assert L.match([[16, 150, 305, 170]], [L.text_box(ln)]) == {}
    assert L.grad_status([16, 196, 305, 224], ln, []) == "ok"
    assert L.grad_status([40, 196, 305, 224], ln, []) == "partial"        # cuts the left edge
    assert L.grad_status([16, 205, 305, 224], ln, []) == "partial"        # cuts the top
    assert L.grad_status([16, 196, 305, 224], ln, [[250, 198, 330, 222]]) == "mixed"
    assert L.grad_status([16, 196, 305, 224], ln, [[16, 150, 100, 170]]) == "ok"
    dets = [{"det": [20, 200, 300, 220]}, {"det": [60, 188, 90, 198]}, {"det": [16, 150, 100, 170]}]
    f = _line([20, 200, 300, 220], furi=[60, 188, 90, 198])
    assert L.others_of(f, dets) == [[16, 150, 100, 170]]


def _glyph_rows(img, top, x0, x1, h, value):
    for x in range(x0, x1, 14):
        img[top:top + h, x:x + 10] = value
        img[top + h // 2 - 1:top + h // 2 + 1, x - 2:x + 12] = value


def test_grad_assign_and_polarity():
    img = np.full((272, 480, 3), 30, np.uint8)
    img[180:262, 10:470] = 60
    _glyph_rows(img, 196, 30, 400, 16, 235)
    _glyph_rows(img, 226, 30, 300, 16, 235)
    a, b = _line([28, 196, 402, 212]), _line([28, 226, 302, 242], 1)
    dets = [{"det": a["ink"]}, {"det": b["ink"]}]
    res = L.grad_assign(img, [a, b], dets, "psp")
    assert [r["status"] for r in res] == ["ok", "ok"]
    for r, ln in zip(res, (a, b)):
        g = r["box"]
        assert g[0] <= ln["ink"][0] and g[2] >= ln["ink"][2] and r["dark"] == 0
    crop, dark, inv = L.cut(img, L.crop_box(a, "psp", 480, 272))
    assert dark == 0 and not inv and crop.dtype == np.uint8
    light = 255 - img
    crop2, dark2, inv2 = L.cut(light, L.crop_box(a, "psp", 480, 272))
    assert dark2 == 1 and inv2 and int(np.median(crop2[0])) < 128    # stored inverted: dark ground


# ----------------------------------------------------------------------------- sheets

def _read_line(meiki, meikiv, v6s, v6sc, ink=(20, 200, 300, 220), edge=False):
    return {"k": 0, "ink": list(ink), "crop": list(ink), "edge": edge, "hint": {},
            "reads": {"meiki": [meiki, []], "meikiv": [meikiv, 0.9, 0.8, []], "v6s": [v6s, 0.9, 0.8],
                      "v6sc": [v6sc, 0.9, 0.8]}}


def test_candidates_support():
    ln = _read_line("「そうか……」", "「そうか‥‥」", "そうか……」", "「そうか……」")   # ‥‥ is two cells too
    c = S.candidates(ln)
    assert [x["text"] for x in c] == ["「そうか……」", "そうか……」"]
    assert c[0]["engines"] == ["meiki", "meikiv", "v6sc"] and c[1]["engines"] == ["v6s"]
    assert S.support("「そうか……」", ln) == ["meiki", "v6s"]
    assert S.support("そうか……」", ln) == ["v6s"]
    assert S.support("「そうか…」", ln) == []
    assert S.is_candidate(ln) == (True, None)
    assert S.is_candidate(_read_line("abc", "abc", "abc", "abc"))[1] == "not-japanese"
    assert S.is_candidate(_read_line("あ", "あ", "あ", "あ", ink=(0, 0, 10, 5)))[1] == "size"
    assert S.is_candidate(_read_line("あい", "あい", "あい", "あい", edge=True))[1] == "screen-edge"


def test_pack_limits():
    from PIL import Image
    blocks = [Image.new("RGB", (w, h)) for w, h in [(1280, 100), (300, 90), (400, 120), (700, 80)] * 12]
    sheets, sizes = S.pack(blocks)
    assert sum(len(s) for s in sheets) == len(blocks)
    for (w, h), members in zip(sizes, sheets):
        assert w <= S.SHEET and h <= S.SHEET
        for i, x, y in members:
            assert x + blocks[i].width <= S.SHEET and y + blocks[i].height <= S.SHEET


def test_shuffled_deterministic():
    c = [{"text": t} for t in "あいうえお"]
    assert S.shuffled(c, "v1_1_0") == S.shuffled(c, "v1_1_0")
    assert sorted(x["text"] for x in S.shuffled(c, "v1_1_0")) == list("あいうえお")


def _fake_batch(tmp_path, mode="pass1"):
    import cv2
    img = np.full((272, 480, 3), 40, np.uint8)
    _glyph_rows(img, 196, 30, 400, 16, 230)
    shot = tmp_path / "v100_5.png"
    cv2.imwrite(str(shot), img)
    s = {"shot": "v100_5", "vn": "v100", "kind": "psp", "url": P.vndb_url(5), "path": str(shot), "subset": "strict",
         "dets": [{"det": [28, 196, 402, 212]}]}
    entries = []
    for k, reads in enumerate([("あいう", "あいう", "あいう", "あいう"), ("かきく", "かきぐ", "かきく", "かきぐ")]):
        ln = _read_line(*reads, ink=(28, 196, 402, 212))
        ln["k"] = k
        entries.append((f"v100_5_{k}", s, ln, S.candidates(ln)))
    out = tmp_path / "batch"
    n = S.make_batch(entries, out, mode)
    return out, n, s, entries


def test_make_batch(tmp_path):
    out, n, _, _ = _fake_batch(tmp_path)
    from PIL import Image
    assert n == 1 and (out / "README.md").exists() and (out / "batch.json").exists()
    im = Image.open(out / "S001.png")
    assert im.width <= 1280 and im.height <= 1280
    sheet = json.loads((out / "S001.json").read_text(encoding="utf-8"))
    assert [e["n"] for e in sheet["entries"]] == [1, 2]
    assert sheet["entries"][1]["candidates"][0]["engines"]
    readme = (out / "README.md").read_text(encoding="utf-8")
    for key in ('"pick"', '"text"', '"unsure"', '"skip"', '"default"', '"name"'):
        assert key in readme


def test_parse_decisions(tmp_path):
    out, _, _, _ = _fake_batch(tmp_path)
    batch, sheets = D.load_batch(out)
    ok = '{"sheet": "S001", "default": 1}\n{"sheet": "S001", "n": 2, "t": "かきく"}\n'
    recs = D.parse_decisions(ok, sheets, "pass1", "b")
    assert [(r["decision"], r["text"]) for r in recs] == [("pick", "あいう"), ("text", "かきく")]
    assert recs[0]["via"] == "default" and all(r["reviewer_kind"] == "assistant" for r in recs)
    with pytest.raises(D.DecisionError, match="needs an explicit decision"):
        D.parse_decisions('{"sheet": "S001", "default": 1}\n', sheets, "pass1", "b")
    with pytest.raises(D.DecisionError, match="without a decision"):
        D.parse_decisions('{"sheet": "S001", "n": 1, "pick": 1}\n', sheets, "pass1", "b")
    with pytest.raises(D.DecisionError, match="no candidate"):
        D.parse_decisions('{"sheet": "S001", "n": 1, "pick": 3}\n{"sheet": "S001", "n": 2, "unsure": true}\n',
                          sheets, "pass1", "b")
    with pytest.raises(D.DecisionError, match="skip must be"):
        D.parse_decisions('{"sheet": "S001", "n": 1, "skip": "menu"}\n{"sheet": "S001", "n": 2, "unsure": true}\n',
                          sheets, "pass1", "b")
    with pytest.raises(D.DecisionError, match="default is only"):
        D.parse_decisions('{"sheet": "S001", "default": 1}\n', sheets, "pass2", "b")
    recs = D.parse_decisions('{"sheet": "S001", "n": 1, "s": "ui"}\n{"sheet": "S001", "n": 2, "u": true, '
                             '"name": true}\n', sheets, "pass1", "b")
    assert [(r["decision"], r["skip"], r["name"]) for r in recs] == [("skip", "ui", False), ("unsure", None, True)]


def _p(decision, text=None, skip=None, batch="b"):
    return {"decision": decision, "text": text, "skip": skip, "batch": batch, "name": False}


def test_decide_rules():
    ln = _read_line("かきく", "かきぐ", "かきく", "かきぐ")
    r = D.decide_line("x", ln, _p("pick", "かきく"), None, "dev")
    assert (r["status"], r["how"], r["support"]) == ("kept", "consensus", ["meiki", "v6s"])
    assert r["readers"] == {"meiki": True, "meikiv": False, "v6s": True, "v6sc": False}
    single = _read_line("かきく", "かきく", "かぎく", "かぎく")
    r = D.decide_line("x", single, _p("pick", "かきく"), None, "dev")
    assert (r["status"], r["how"], r["why"]) == ("dropped", "single", "pass2-missing")
    r = D.decide_line("x", single, _p("pick", "かきく"), _p("text", "かきく"), "dev")
    assert r["status"] == "kept" and r["pass2"]["agree"]
    r = D.decide_line("x", single, _p("pick", "かきく"), _p("pick", "かぎく"), "dev")
    assert (r["status"], r["why"]) == ("dropped", "pass2-disagree")
    r = D.decide_line("x", single, _p("pick", "かきく"), _p("unsure"), "dev")
    assert r["why"] == "pass2-unsure"
    r = D.decide_line("x", single, _p("text", "がきく"), _p("text", "がきく"), "dev")
    assert (r["status"], r["how"]) == ("kept", "changed")
    r = D.decide_line("x", single, _p("text", "がきく"), _p("text", "がきく"), "train")
    assert (r["status"], r["why"]) == ("dropped", "no-engine-support")
    r = D.decide_line("x", single, _p("pick", "かきく"), None, "train")
    assert r["status"] == "kept"
    assert D.decide_line("x", ln, _p("skip", skip="ui"), None, "dev")["why"] == "skip-ui"
    assert D.decide_line("x", ln, _p("unsure"), None, "dev")["why"] == "pass1-unsure"
    agree4 = _read_line("あいう", "あいう", "あいう", "あいう")
    assert D.decide_line("x", agree4, None, None, "train", auto_consensus=True)["how"] == "auto"
    assert D.decide_line("x", agree4, None, None, "train")["why"] == "undecided"
    r = D.decide_line("x", ln, _p("pick", "かきく"), None, "dev", resolution={"keep": False, "why": "misread"})
    assert (r["status"], r["why"]) == ("dropped", "blind-misread")
    r = D.decide_line("x", ln, _p("pick", "かきく"), None, "dev", resolution={"keep": True, "text": "かきぐ"})
    assert r["label"] == "かきぐ" and r["status"] == "kept" and r["resolution"]["changed_label"]


def test_needs_pass2():
    ln = _read_line("かきく", "かきく", "かぎく", "かぎく")
    assert D.needs_pass2(_p("pick", "かきく"), ln)
    both = _read_line("かきく", "かきく", "かきく", "かぎく")
    assert not D.needs_pass2(_p("pick", "かきく"), both)
    assert not D.needs_pass2(_p("skip", skip="ui"), ln)


def test_blind_agreement():
    labels = {f"x{i}": {"label": "あいう", "how": "consensus", "subset": "strict", "kind": "psp"} for i in range(10)}
    blind = {f"x{i}": {"decision": "pick", "text": "あいう" if i < 9 else "あいお"} for i in range(10)}
    summ, dis = D.blind_agreement(labels, blind)
    assert summ["all"]["agree"] == 9 and summ["all"]["lines"] == 10 and len(dis) == 1
    lo, hi = summ["all"]["wilson95"]
    assert lo < 0.9 < hi


# ----------------------------------------------------------------------------- build

def test_build_npz_format(tmp_path):
    out, _, s, entries = _fake_batch(tmp_path)
    shots = {s["shot"]: dict(s, lines=[e[2] for e in entries])}
    labels = []
    for lid, _, ln, _ in entries:
        rec = D.decide_line(lid, ln, _p("pick", S.candidates(ln)[0]["text"]), None, "dev")
        rec.update(shot=s["shot"], vn=s["vn"], subset=s["subset"], kind=s["kind"])
        labels.append(rec)
    stats = B.build(labels, shots, tmp_path / "out", manifests=tmp_path / "man", tvl={})
    z = np.load(tmp_path / "out" / "devnative_strict_ink.npz")
    assert set(z.files) == {"data", "offsets", "shapes", "labels", "meta"}
    labs, meta = json.loads(str(z["labels"])), json.loads(str(z["meta"]))
    assert len(labs) == len(meta) == stats["kept"] == 2
    for i, (lab, m) in enumerate(zip(labs, meta)):
        h, w = (int(v) for v in z["shapes"][i])
        g = z["data"][z["offsets"][i]:z["offsets"][i + 1]].reshape(h, w)
        assert g.dtype == np.uint8 and lab
        for k in ("id", "vn", "session", "kind", "target", "box", "dark", "inverted", "how", "pol", "chars",
                  "subset", "crop", "reviewer_kind"):
            assert k in m
        assert m["target"] == m["kind"] == "psp" and m["crop"] == "ink" and m["reviewer_kind"] == "assistant"
    man = (tmp_path / "man" / "lines.jsonl").read_text(encoding="utf-8")
    for lab in labs:
        assert lab not in man                                # no text in the committed manifest
    rows = [json.loads(x) for x in man.splitlines()]
    assert {"id", "url", "ink_box", "grad", "readers", "how", "reviewer_kind"} <= set(rows[0])


def test_cap_repeats():
    rows = [{"id": f"v1_{i}_0", "vn": "v1", "label": "千鶴"} for i in range(4)] + [{"id": "v2_1_0", "vn": "v2",
                                                                                     "label": "千鶴"}]
    kept, capped = B.cap_repeats(rows)
    assert len(kept) == 3 and len(capped) == 2


# ----------------------------------------------------------------------------- GRAD containment

def test_grad_extension_rules():
    from devnative import gradcheck as GC
    crop = [100, 50, 300, 70]
    assert GC.extension([98, 49, 303, 71], crop) == [2, 1, 3, 1]
    assert not GC.needs_check([98, 49, 303, 71], crop, 16)
    assert GC.needs_check([60, 49, 303, 71], crop, 16)          # 40 px to the left
    assert GC.needs_check([98, 30, 303, 71], crop, 16)          # 20 px above (> 0.6 ink heights)
    assert not GC.needs_check([96, 44, 304, 76], crop, 16)      # within max(3, 0.3 h) and 0.6 h
    rec = {"status": "ok", "check": True, "label": "あいう"}
    assert GC.accept(rec, None) == (False, "grad-unchecked")
    assert GC.accept(rec, {"decision": "pick", "pick": 1}) == (True, "checked")
    assert GC.accept(rec, {"decision": "text", "text": "あいう"}) == (True, "checked")
    assert GC.accept(rec, {"decision": "text", "text": "あいう67"}) == (False, "grad-extra-text")
    assert GC.accept(rec, {"decision": "skip", "skip": "cut"}) == (False, "grad-cut")
    assert GC.accept({"status": "ok", "check": False}, None) == (True, "inside")
    assert GC.accept({"status": "mixed"}, None) == (False, "grad-mixed")
    assert GC.region_member({"decision": "skip", "skip": "cut"}) and not GC.region_member(
        {"decision": "skip", "skip": "ui"})


def test_grad_check_batch(tmp_path):
    out, _, s, entries = _fake_batch(tmp_path)
    lid, s, ln, _ = entries[0]
    from devnative import gradcheck  # noqa: F401  (the mode exists)
    n = S.make_batch([(lid, s, ln, [{"text": "あいう", "key": "あいう", "engines": []}], [20, 190, 420, 216])],
                     tmp_path / "g", "grad")
    assert n == 1
    readme = (tmp_path / "g" / "README.md").read_text(encoding="utf-8")
    assert "GRAD-crop check" in readme and '"default"' in readme
    batch, sheets = D.load_batch(tmp_path / "g")
    recs = D.parse_decisions('{"sheet": "S001", "default": 1}\n', sheets, "grad", "g")
    assert recs[0]["decision"] == "pick" and recs[0]["pick"] == 1
