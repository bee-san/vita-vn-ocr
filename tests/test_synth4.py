"""synth/synth4.py: font manifest and licence profiles, curriculum quotas, contexts, style corrections.

CI has no fonts, pools or scenes: these tests exercise the pieces that need none, and the owner-font
checks run only where the private files exist (never committed)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synth"))
import synth4 as S4  # noqa: E402

MANIFEST = ROOT / "data" / "manifests" / "synth4_fonts.json"


def test_manifest_adds_only_the_owner_fonts():
    m4 = json.loads(MANIFEST.read_text(encoding="utf-8"))
    m3 = json.loads((ROOT / "data" / "manifests" / "synth3_fonts.json").read_text(encoding="utf-8"))
    f3 = {(f["family"], x["sha256"]) for f in m3["families"] for x in f["files"]}
    f4 = {(f["family"], x["sha256"]) for f in m4["families"] for x in f["files"]}
    assert f3 <= f4
    extra = [f for f in m4["families"] if (f["family"], f["files"][0]["sha256"]) not in f3]
    assert {f["family"] for f in extra} == {"owner_cinecaption", "owner_nds12"}
    for f in extra:
        assert f["licence_class"] == "owner" and f["root"] == "synth4"
        assert "non-commercial private OCR project" in f["licence_id"]
    nds = next(f for f in extra if f["family"] == "owner_nds12")
    assert nds["category"] == "pixel" and nds["files"][0]["pixel_em"] == 10
    assert m4["counts"]["owner_families"] == ["owner_cinecaption", "owner_nds12"]
    assert m4["base_manifest"]["file"] == "data/manifests/synth3_fonts.json"


def test_owner_fonts_only_in_the_personal_profile():
    S4.patch()
    import synth3 as S3
    assert "owner" in S3.LICENCE_PROFILES["personal"]
    assert S3.LICENCE_PROFILES["open"] == {"open"}


def test_cfg_patch_keys_exist_in_synth3():
    import synth3 as S3
    S3.merge_cfg(S3.CFG, S4.CFG_PATCH)          # raises on a key synth3 does not have


def test_shard_targets_sum_to_the_deficit_and_are_deterministic():
    deficit = np.array([0, 1, 50, 99, 144, 7], np.float64)
    tot = sum(S4.shard_targets(i, 20261013, deficit, 100) for i in range(100))
    assert np.all(np.abs(tot - deficit) <= np.maximum(3, 0.35 * deficit))
    a = S4.shard_targets(5, 20261013, deficit, 100)
    b = S4.shard_targets(5, 20261013, deficit, 100)
    assert np.array_equal(a, b)
    assert S4.shard_targets(5, 1, deficit, 100)[0] == 0


def test_build_curriculum():
    vocab = ["あ", "鬱", "璽", "々"]
    exposures = {"あ": 5000, "鬱": 3, "璽": 0}
    jw = {"あ": 1.0, "鬱": 0.001}
    real = {"々": 4}
    words = {"鬱": [("憂鬱", 0.01), ("鬱", 0.002)]}
    mined = {"鬱": (3, 10), "あ": (1, 1000)}
    base = {"あ": {"real": 0.1, "jiten_vn": 0.2, "hard": 0.1}, "鬱": {"real": 0.0, "jiten_vn": 0.0, "hard": 0.0}}
    cur = S4.build_curriculum(vocab, exposures, jw, real, words, mined, base)
    k_vn, k_other = S4.CFG4["curriculum"]["k_vn"], S4.CFG4["curriculum"]["k_other"]
    assert cur["k"] == {"あ": k_vn, "鬱": k_vn, "璽": k_other, "々": k_vn}
    assert cur["deficit"]["あ"] == 0 and cur["deficit"]["鬱"] == k_vn - 3 and cur["deficit"]["璽"] == k_other
    assert cur["words"]["鬱"][0][0] == "憂鬱"
    assert cur["targets"]["鬱"]["hard"] > cur["targets"]["あ"]["hard"] * 0 and 0 <= cur["targets"]["鬱"]["hard"] <= 1


def test_carriers_hold_the_word():
    rng = np.random.default_rng(0)
    for _ in range(200):
        for kind in ("dialogue", "narration"):
            s = S4.carrier(rng, "憂鬱", kind, other=None)
            assert "憂鬱" in s and "{" not in s
        s = S4.carrier(rng, "言う", "dialogue")
        assert "言う" in s
    s = S4.carrier(np.random.default_rng(3), "猫", "dialogue", other="犬")
    assert "猫" in s


def test_name_text_keeps_the_character():
    rng = np.random.default_rng(1)
    import synth3 as S3
    assert "space" in S3.CFG["names"]
    for _ in range(50):
        assert "鴇" in S4.name_text(("鴇田", "優"), "鴇", rng)
        assert "優" in S4.name_text(("鴇田", "優"), "優", rng)


def test_style_corrections_stay_in_range():
    S4.patch()
    import render as S2
    rng = np.random.default_rng(7)
    seen_white = 0
    for _ in range(400):
        st = S2.sample_style(rng, "vita")
        for k in ("box", "fill"):
            a = np.asarray(st[k])
            assert a.shape == (3,) and float(a.min()) >= 0 and float(a.max()) <= 255
        assert st["on_dark"] == (S2.luma(st["fill"]) > S2.luma(st["box"]))
        seen_white += int(np.all(np.asarray(st["fill"]) == 255))
    assert seen_white > 20


def test_accent_redraws_green():
    S4.patch()
    import colorsys

    import render as S2
    rng = np.random.default_rng(11)
    lo, hi = S4.CFG4["green_band"]
    green = sum(lo <= colorsys.rgb_to_hsv(*(np.asarray(S2.accent(rng, True)) / 255))[0] <= hi for _ in range(2000))
    rng = np.random.default_rng(11)
    base = sum(lo <= colorsys.rgb_to_hsv(*(np.asarray(S4._ORIG["accent"](rng, True)) / 255))[0] <= hi
               for _ in range(2000))
    assert green < 0.6 * base


def test_px_fits_lets_10px_fonts_onto_the_vita_at_2x():
    class F:
        pixel_em = 10
    assert S4.px_fits4(F, "vita") and S4.px_fits4(F, "psp")
    F.pixel_em = 18
    assert S4.px_fits4(F, "vita") and not S4.px_fits4(F, "psp")


NDS = S4.S4_ART / "fonts" / "owner" / "nds12" / "NDS12.ttf"


@pytest.mark.skipif(not NDS.exists(), reason="owner font is private (never committed)")
def test_nds12_is_pixel_exact_at_its_10_ppem_strike():
    from PIL import Image, ImageDraw, ImageFont
    m4 = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rec = next(f for f in m4["families"] if f["family"] == "owner_nds12")["files"][0]
    assert S4.sha256(NDS) == rec["sha256"]
    f10 = ImageFont.truetype(str(NDS), 10)
    assert f10.getmetrics() == (11, 1)                   # a 12-px line cell

    def draw(font, text, mode1=True, k=1):
        im = Image.new("L", (200 * k, 16 * k), 0)
        d = ImageDraw.Draw(im)
        if mode1:
            d.fontmode = "1"
        d.text((0, 0), text, font=font, fill=255)
        return np.asarray(im) > 127
    text = "あいうえおカタカナ「」！"
    small = draw(f10, text)
    big = draw(ImageFont.truetype(str(NDS), 40), text, mode1=False, k=4)      # 4 px per brick
    h, w = small.shape
    pooled = big[:h * 4, :w * 4].reshape(h, 4, w, 4).mean(axis=(1, 3)) > 0.5
    assert (pooled != small).mean() < 0.002                      # every brick is one pixel at 10 ppem


def test_ablation_decision_and_curriculum_tags():
    import sys
    sys.path.insert(0, str(ROOT / "eval"))
    import synth4_ablation as AB
    tail = {"鬱", "璽"}
    fl = {"あ", "鬱"}
    assert AB.curriculum_tags("あ鬱", tail, fl) == ["tail-kanji", "tail-any"]
    assert AB.curriculum_tags("あ璽", tail, fl) == ["tail-kanji", "tail-any", "fl10-unk"]
    assert AB.curriculum_tags("ああ", tail, fl) == []
    better = {"all_vn_cluster": {"ci95": [-0.8, -0.2]}, "per_seed": {"a": {"delta_points": -0.5},
                                                                     "b": {"delta_points": -0.3}}}
    guard_ok = {"all_vn_cluster": {"ci95": [-0.4, 0.2]}}
    d = AB.decide(better, guard_ok)
    assert d["synth4_better"] and not d["synth4_better_on_both"]
    worse = {"all_vn_cluster": {"ci95": [0.1, 0.5]}, "per_seed": {"a": {"delta_points": 0.3}}}
    assert not AB.decide(worse, guard_ok)["synth4_better"]
    guard_bad = {"all_vn_cluster": {"ci95": [0.1, 0.5]}}
    assert not AB.decide(better, guard_bad)["devnative_not_worse"]
    both = {"all_vn_cluster": {"ci95": [-0.5, -0.1]}}
    assert AB.decide(better, both)["synth4_better_on_both"]


def test_committed_ablation_result_holds_no_prediction_text():
    f = ROOT / "eval" / "synth4_ablation_20261013.json"
    if not f.exists():
        return
    import json
    r = json.loads(f.read_text(encoding="utf-8"))
    assert r["kind"] == "synth4-ablation-result"
    assert "pred" not in json.dumps(r) and "lines" not in r["teacher_val"]
    assert set(r["decision"]) >= {"teacher_val_better", "devnative_not_worse", "synth4_better"}
