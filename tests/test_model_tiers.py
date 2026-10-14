"""vita/model_tiers.py and eval/model_gaps.py: the tier sizing (random-init blobs, the free-memory
rule, the runtime selection rule) and the development-data error taxonomy. None needs VitaSDK, qemu,
Vita3K or any protected set; the sizes and the memory arithmetic are exact.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "vita"), str(ROOT / "eval"), str(ROOT / "train")]
import model_gaps as MG  # noqa: E402
import model_tiers as MT  # noqa: E402

# ---------------------------------------------------------------- model_tiers: sizes

@pytest.mark.parametrize("name, blob, arena, stride, height", [
    ("s_w100v3500", 657_616, 124_832, 8, 32),      # FL10's shape (its own vocab blob is 657,664 B)
    ("n_w200", 1_558_016, 241_056, 8, 32),         # W20 / F20's shape, exactly
    ("n_w200v5631", 1_587_680, 241_056, 8, 32),
    ("n_w250h48s16v5631", 2_287_712, 698_848, 16, 48),
])
def test_candidate_blobs_have_the_expected_sizes(name, blob, arena, stride, height):
    import export
    data, meta = MT.export_blob(MT.CANDIDATES[name])
    info = MT.BS.blob_info(data)
    assert len(data) == blob
    assert info["arena"]["t4h16"]["w8i"] == arena
    assert meta["stride"] == stride
    assert info["height"] == height
    assert export.plan_arena(export.read_blob(data), mode=export.MODE_W8I) == arena


def test_frame_stride_16_halves_the_frames():
    assert MT.frames_per_line(32, 8) == 80
    assert MT.frames_per_line(32, 16) == 40
    assert MT.frames_per_line(48, 8) == 120        # a taller line is more frames at the same stride


def test_collapse_rows_matches_the_runtime():
    assert MT.collapse_rows(32) == 2               # 32 -> 16 -> 8 -> 4 -> 2
    assert MT.collapse_rows(24) == 2               # the model.py comment: 24 -> 12 -> 6 -> 3 -> 2
    assert MT.collapse_rows(40) == 3
    assert MT.collapse_rows(48) == 3


# ---------------------------------------------------------------- model_tiers: memory

def test_job_need_is_the_overlay_layout():
    # FL10 at the overlay's figures: weights 659,456 B resident, workspace 188,416 B (API 19, no region plane).
    j = MT.job_need(657_664, 124_832)
    assert j["weights"] == MT.page(657_664) == 659_456
    assert j["capture"] == MT.OVERLAY["capture_block_max"] == 61_440
    assert j["line_block"] == MT.OVERLAY["line_block_max"]
    # workspace = page(max(finder, recognizer arena) rounded to 16 + runtime structs)
    scratch = -(-max(MT.OVERLAY["finder_arena_960x544"], 124_832) // 16) * 16
    assert j["workspace"] == MT.page(scratch + MT.OVERLAY["runtime_structs"])
    assert j["total"] == j["weights"] + j["workspace"] + j["capture"] + j["line_block"]
    # resident weights are not re-counted
    assert MT.job_need(657_664, 124_832, resident=True)["weights"] == 0


def test_free_needed_adds_the_reserve():
    for t in MT.TIERS:
        need = MT.free_needed(900_000, 131_072, t["reserve"])
        assert need == MT.job_need(900_000, 131_072)["total"] + t["reserve"]


def test_select_tier_picks_the_largest_that_fits():
    models = [{"name": "Large", "blob": 3_689_840, "arena": 1_176_576, "reserve": 16 << 20},
              {"name": "Normal", "blob": 1_587_680, "arena": 241_056, "reserve": 8 << 20},
              {"name": "Small", "blob": 819_792, "arena": 124_832, "reserve": 4 << 20}]
    assert MT.select_tier(90 << 20, models) == "Large"          # ~87 MiB free in CLANNAD: Large fits
    assert MT.select_tier(14 << 20, models) == "Normal"         # Normal needs ~10.3 MiB free
    assert MT.select_tier(6 << 20, models) == "Small"           # Small needs ~5.5 MiB free
    assert MT.select_tier(3 << 20, models) is None              # nothing fits
    assert MT.select_tier(-1, models) is None                   # the query failed


def test_select_tier_counts_a_resident_model_as_free():
    models = [{"name": "Normal", "blob": 1_587_680, "arena": 241_056, "reserve": 8 << 20},
              {"name": "Small", "blob": 819_792, "arena": 124_832, "reserve": 4 << 20}]
    free = 10_000_000  # Normal from scratch ~10.3 MiB; resident ~8.8 MiB; Small ~5.5 MiB
    assert MT.select_tier(free, models) == "Small"             # Normal does not fit from scratch
    res = {"name": "Normal", "blob": 1_587_680}                # but if Normal's weights are resident,
    assert MT.select_tier(free, models, resident=res) == "Normal"  # its block is released and counts as free


def test_tier_fit_checks_weights_arena_and_box_time():
    small, normal, large = MT.TIERS
    assert small["box_ms"] == 1000 and normal["box_ms"] == 2000 and large["box_ms"] == 3000   # per 3-line box @444
    assert all(t["mhz"] == 444 for t in MT.TIERS)

    def info(blob, a4, a16=None):
        return {"blob_bytes": blob, "arena": {"t4h16": {"w8i": a4}, "t16h64": {"w8i": a16 or a4 * 3}}}

    def run(box4=None, box16=None):
        ms = {}
        if box4 is not None:
            ms["l22048"] = {"444": {"box3x640": [box4 / 2, box4]}, "333": {"box3x640": [box4, box4 * 1.3]}}
        if box16 is not None:
            ms["l22048_t16h64"] = {"444": {"box3x640": [box16 / 2, box16]}, "333": {"box3x640": [box16, box16]}}
        return {"ms": ms}

    # Small: strict < 1 MB weights, 128 KiB arena, <= 1 s box; only the default tiling
    assert MT.tier_fit(small, info(999_999, 131_072), run(box4=900))[0] is True
    assert MT.tier_fit(small, info(1_000_000, 131_072), run(box4=900))[0] is False     # weights not < 1 MB
    assert MT.tier_fit(small, info(999_999, 131_073), run(box4=900))[0] is False       # arena over the ceiling
    assert MT.tier_fit(small, info(999_999, 131_072), run(box4=1001))[0] is False      # over the box budget
    assert MT.tier_fit(small, info(999_999, 131_072), run())[0] is None                # no a9 run: unmeasured
    # Large: tries 16-frame tiles when the default tiling is over 3 s but the t16 arena fits
    fits, tiling, arena, _ = MT.tier_fit(large, info(3_000_000, 360_000, 1_200_000), run(box4=3400, box16=2900))
    assert fits is True and tiling == "t16h64" and arena == 1_200_000
    # the default tiling is preferred when it already fits (less arena)
    fits, tiling, _, _ = MT.tier_fit(large, info(1_558_016, 241_056, 775_296), run(box4=1234, box16=1216))
    assert fits is True and tiling == "t4h16"
    # over the arena ceiling at every tiling: a definite miss, not unmeasured
    assert MT.tier_fit(large, info(3_000_000, 1_400_000, 1_400_000), run(box4=2900, box16=2900))[0] is False


# ---------------------------------------------------------------- model_gaps

def test_align_counts_edits_like_levenshtein():
    from metrics import lev
    for ref, pred in [("abc", "abc"), ("abc", "abd"), ("abc", "ac"), ("ab", "abc"), ("", "x"), ("x", "")]:
        ops = MG.align(ref, pred)
        assert sum(op != "=" for op, *_ in ops) == lev(ref, pred)
        # the kept reference characters, in order, are the reference
        assert "".join(r for op, i, r, p in ops if op in ("=", "s", "d")) == ref


def test_char_class_places_each_kind():
    assert MG.char_class("あ") == "hiragana"
    assert MG.char_class("が") == "hira_voiced"
    assert MG.char_class("ぁ") == "hira_small"
    assert MG.char_class("ア") == "katakana"
    assert MG.char_class("パ") == "kata_voiced"
    assert MG.char_class("ー") == "long_vowel"
    assert MG.char_class("漢") == "kanji"
    assert MG.char_class("A") == "latin"
    assert MG.char_class("7") == "digit"
    assert MG.char_class("。") == "punct"
    assert MG.char_class("酒", vocab=set("あ")) == "oov"       # not in the model's vocab


def test_pair_kind_names_the_confusion_families():
    assert MG.pair_kind("が", "か") == "dakuten"
    assert MG.pair_kind("パ", "バ") == "dakuten"
    assert MG.pair_kind("ゃ", "や") == "small_kana"
    assert MG.pair_kind("ー", "一") == "dash_like"
    assert MG.pair_kind("。", "、") == "punct_punct"
    assert MG.pair_kind("蓮", "達") == "kanji_kanji"


def test_kanji_band_and_exposure_band():
    ranks = {"人": 1, "恋": 1500, "璧": 9000}
    assert MG.kanji_band("人", ranks) == "kanji_r500"
    assert MG.kanji_band("恋", ranks) == "kanji_r2000"
    assert MG.kanji_band("璧", ranks) == "kanji_rare"
    assert MG.kanji_band("丼", ranks) == "kanji_rare"          # unranked = rare
    assert MG.exposure_band(0) == "real_0"
    assert MG.exposure_band(5) == "real_1-9"
    assert MG.exposure_band(50) == "real_10-99"
    assert MG.exposure_band(500) == "real_100+"


# ---------------------------------------------------------------- the committed report

def test_report_json_is_well_formed():
    """The committed docs/model-tiers.json: every model has the fields the doc cites, and the tiers
    and overlay constants match vita-jp-overlay's."""
    p = ROOT / "docs/model-tiers.json"
    if not p.exists():
        pytest.skip("docs/model-tiers.json not written yet")
    d = json.loads(p.read_text())
    assert [t["name"] for t in d["tiers"]] == ["Small", "Normal", "Large"]
    assert d["tiers"][0]["weights_lt"] == 1_000_000
    assert d["tiers"][1]["weights_le"] == 4_000_000
    assert d["tiers"][2]["weights_le"] == 9_000_000
    # per-3-line-box time at 444 MHz (owner, final 2026-10-14 07:33Z/08:01Z)
    assert [t["box_ms"] for t in d["tiers"]] == [1000, 2000, 3000]
    assert all(t["mhz"] == 444 for t in d["tiers"])
    assert [t["arena_le"] for t in d["tiers"]] == [131_072, 262_144, 1_310_720]
    assert d["tiers"][2]["tiling"] == "t16h64"
    assert d["overlay"]["reserve"] == 4 * 1024 * 1024          # the overlay's own per-job reserve
    assert d["overlay"]["capture_block_max"] == 61_440
    for r in d["models"]:
        assert r["blob_bytes"] > 0 and r["arena_w8i"] > 0 and r["macs_per_frame"] > 0
        assert set(r["tiers"]) == {"Small", "Normal", "Large"}
        assert set(r["free_needed"]) == {"Small", "Normal", "Large"}


def test_capacity_curve_is_matched_recipe():
    p = ROOT / "docs/model-tiers.json"
    if not p.exists() or "capacity" not in json.loads(p.read_text()):
        pytest.skip("capacity not written yet")
    cap = json.loads(p.read_text())["capacity"]
    assert set(cap["arms"]) >= {"C_mix"}
    # the matched curve is first-step-steeper: FL10->H15 pays more per MB than H15->W20
    steps = {s["from"] + "->" + s["to"]: s for s in cap["arms"]["C_mix"]["steps"]}
    assert steps["FL10->H15"]["points_per_MB"] > steps["H15->W20"]["points_per_MB"]
    # the FL10 point must be a 32k-pretrain run, not the 128k selection (the confound the audit flagged)
    fl10 = next(p for p in cap["arms"]["C_mix"]["points"] if p["size"] == "FL10")
    assert all("pre128k" not in r for r in fl10["runs"])


def test_recommendations_json_names_one_model_per_tier():
    p = ROOT / "docs/model-tiers.json"
    if not p.exists() or "recommendations" not in json.loads(p.read_text()):
        pytest.skip("recommendations not written yet")
    rec = json.loads(p.read_text())["recommendations"]
    assert set(rec) == {"Small", "Normal", "Large"}
    for name, r in rec.items():
        assert r["weights_ceiling"] and r["arena_ceiling"] and r["box_ms_ceiling_444"]
        # both the trained-now and the next-round model must fit the tier, and their recorded free_needed
        # must equal recomputation from the arena at the tiling the entry records (audit R2)
        for which in ("train_now", "next_round"):
            e = r[which]
            assert e["fits"] is True, (name, which)
            assert e["free_needed"] == MT.free_needed(e["blob_bytes"], e["arena_w8i"], r["reserve"])
