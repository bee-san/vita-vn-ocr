"""synth/synth6*.py: the harvested-font reader and drawing, the font inventory's exclusions, the style measurement,
layout, decontamination, compositing and the shard format, on small synthetic fixtures.

CI has no font pack, corpus or text pools: every fixture is built here, from the pinned OFL font (conftest.font)
where glyphs are needed; the world the renderer reads is a fake title with one generated page and a stub text
sampler."""
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synth"))
sys.path.insert(0, str(ROOT / "teacher"))
import synth6 as S6  # noqa: E402
import synth6_assets as A6  # noqa: E402
import synth6_fonts as F6  # noqa: E402

CHARS = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん「」、。！？…" \
        "一二三日本人今何見思言気天でがだ・"


def atlas_font(dirpath, font_path, title, fid, px=24, chars=CHARS, verdict="verified", vertical=False,
               glyph_count=None):
    """A harvest-format bitmap font: every char drawn by FreeType into a cell of an atlas, [x, y, w, h, 0, 0, adv]."""
    from PIL import Image, ImageDraw, ImageFont
    d = Path(dirpath) / title / fid
    d.mkdir(parents=True, exist_ok=True)
    f = ImageFont.truetype(str(font_path), px)
    cell = px + 2
    cols = 16
    rows = math.ceil(len(chars) / cols)
    atlas = Image.new("L", (cols * cell, rows * cell), 0)
    dr = ImageDraw.Draw(atlas)
    table = {}
    for i, c in enumerate(chars):
        x, y = (i % cols) * cell, (i // cols) * cell
        if vertical and c in "、。":
            dr.text((x + px // 2 + 2, y - px // 2 + 1), c, font=f, fill=255)       # tategaki forms: top-right
        else:
            dr.text((x + 1, y + 1), c, font=f, fill=255)
        table[c] = [x, y, cell, cell, 0, 0, px]
    atlas.save(d / "atlas.png")
    (d / "glyphs.json").write_text(json.dumps({"atlas": "atlas.png", "cell": [cell, cell], "chars": table},
                                              ensure_ascii=False), encoding="utf-8")
    meta = {"schema": "vn-font-harvest/1", "title_id": title, "font_id": fid, "kind": "bitmap", "format": "test-atlas",
            "face": f"{title} face", "glyph_count": glyph_count or 4000, "pixel_size": px, "cell": [cell, cell],
            "line_height": cell, "content_sha256": F6.sha256_file(d / "atlas.png"), "file": "atlas.png",
            "glyphs_file": "glyphs.json", "coverage": {"vocr5_kanji": 12},
            "fidelity_by_title": {title: {"verdict": verdict, "median_glyph_corr": 0.95, "native_size_or_scale": 1.0,
                                          "native_spacing": 0.5, "native_px": float(px)}}}
    (d / "font.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    (d / "chars.txt").write_text(chars, encoding="utf-8")
    return d


# ----------------------------------------------------------------------------- fonts

def test_atlas_draw_is_exact_at_scale_one(tmp_path, font):
    d = atlas_font(tmp_path, font, "TITLEA", "bm")
    f = F6.open_font(d)
    assert f.kind == "bitmap" and f.has("あ") and not f.has("鬱") and f.has("\u3000")
    cov, cells = F6.draw_line(f, "あい", scale=1.0, spacing=0.0, x=3, y=2)
    g, _, _, adv = f.glyph("あ")
    assert np.allclose(cov[2:2 + g.shape[0], 3:3 + g.shape[1]], g)
    assert cells[0][:3] == ("あ", 3.0, 3.0 + adv) and cells[1][1] == 3.0 + adv
    # scale and spacing move the pen as the engines do
    _, cells2 = F6.draw_line(f, "あいう", scale=0.5, spacing=2.0)
    assert cells2[2][1] == pytest.approx(2 * (adv * 0.5 + 2.0))
    assert F6.line_advance(f, "あいう", scale=0.5, spacing=2.0) == pytest.approx(3 * adv * 0.5 + 4.0)
    f.deny = {"い"}
    assert not f.has("い")


def test_vector_font_draws(tmp_path, font):
    d = tmp_path / "TITLEB" / "vec"
    d.mkdir(parents=True)
    shutil.copy(font, d / "font.ttf")
    (d / "font.json").write_text(json.dumps({"title_id": "TITLEB", "font_id": "vec", "kind": "vector",
                                             "file": "font.ttf", "pixel_size": None}), encoding="utf-8")
    f = F6.open_font(d, size=20)
    assert f.kind == "vector" and f.has("あ")
    cov, cells = F6.draw_line(f, "あいう", spacing=1.0)
    assert cov.max() > 0.9 and len(cells) == 3
    assert not F6.vertical_forms(f)


def test_inventory_exclusions(tmp_path, font):
    atlas_font(tmp_path, font, "TRAIN1", "good")
    t = atlas_font(tmp_path, font, "PCSG00415", "testfont", chars=CHARS[::-1])        # a test title's font
    cp = tmp_path / "OTHER" / "copy"
    shutil.copytree(t, cp)                                                            # byte-identical copy
    m = json.loads((cp / "font.json").read_text(encoding="utf-8"))
    m.update(title_id="OTHER", font_id="copy", fidelity_by_title={})
    (cp / "font.json").write_text(json.dumps(m), encoding="utf-8")
    atlas_font(tmp_path, font, "TRAIN2", "bad", verdict="mismatch", chars=CHARS[1:])
    atlas_font(tmp_path, font, "TRAIN3", "tiny", glyph_count=100, chars=CHARS[2:])
    atlas_font(tmp_path, font, "TRAIN4", "vert", vertical=True, chars=CHARS[3:])
    rows = {r["font"]: r for r in F6.inventory(tmp_path, test_titles=("PCSG00415",), train_titles={"TRAIN1"})}
    assert rows["TRAIN1/good"]["excluded"] == [] and rows["TRAIN1/good"]["trusted_map"]
    assert rows["TRAIN1/good"]["farm_train_titles"] == ["TRAIN1"]
    assert "test-title font" in rows["PCSG00415/testfont"]["excluded"]
    assert "byte-identical to a test-title font" in rows["OTHER/copy"]["excluded"]
    assert any("mismatch" in w for w in rows["TRAIN2/bad"]["excluded"])
    assert any("glyphs" in w for w in rows["TRAIN3/tiny"]["excluded"])
    assert any("vertical" in w for w in rows["TRAIN4/vert"]["excluded"])
    best = F6.dialogue_fonts(list(rows.values()), {"TRAIN1", "PCSG00415"})
    assert set(best) == {"TRAIN1"} and best["TRAIN1"][0] == "TRAIN1/good"


# ----------------------------------------------------------------------------- style measurement and layout

def styled_line(outline=0, shadow=None, fill=(250, 250, 250), box=(60, 60, 110), dark=False, font=None):
    """A real-looking line crop: glyph mask (FreeType) with an optional hard outline and drop shadow."""
    from PIL import Image, ImageDraw, ImageFont
    f = ImageFont.truetype(str(font), 26)
    m = Image.new("L", (360, 40), 0)
    ImageDraw.Draw(m).text((10, 4), "今日はいい天気ですね。", font=f, fill=255)
    cov = np.asarray(m, np.float32) / 255.0
    img = np.zeros((40, 360, 3), np.float32) + np.asarray(box, np.float32)
    if shadow:
        sm = S6.shift(cov, *shadow)[..., None]
        img = img * (1 - sm) + np.asarray((0, 0, 0), np.float32) * sm
    if outline:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outline + 1, 2 * outline + 1))
        om = cv2.dilate(cov, k)[..., None]
        img = img * (1 - om) + np.asarray((0, 0, 0) if not dark else (255, 255, 255), np.float32) * om
    a = cov[..., None]
    img = img * (1 - a) + np.asarray(fill, np.float32) * a
    return np.clip(img, 0, 255).astype(np.uint8)


def test_measure_finds_outline_shadow_and_polarity(font):
    ms = [A6.measure_line(styled_line(outline=2, font=font), (8, 6, 340, 30)) for _ in range(4)]
    st = A6.summarise_styles(ms)
    assert st["dark_share"] == 0 and st["halo"] and st["halo"]["hard"] and st["halo"]["radius"] in (1, 2)
    assert max(st["fill_rgb"]) >= 240
    ms = [A6.measure_line(styled_line(shadow=(2, 2), font=font), (8, 6, 340, 30)) for _ in range(4)]
    st = A6.summarise_styles(ms)
    assert st.get("shadow") and st["shadow"]["offset"] in ([2, 2], [1, 1], [3, 3], [1, 2], [2, 1])
    ms = [A6.measure_line(styled_line(fill=(20, 20, 20), box=(230, 225, 215), font=font), (8, 6, 340, 30))
          for _ in range(4)]
    st = A6.summarise_styles(ms)
    assert st["dark_share"] == 1 and st.get("halo") is None and max(st["fill_rgb"]) < 60


def test_rows_and_free_rows():
    boxes = [(100, 400, 500, 30), (100, 440, 300, 30), (130, 401, 400, 30), (100, 480, 200, 30)] * 8
    rows = A6.cluster_rows(boxes)
    assert [r["y0"] for r in sorted(rows, key=lambda r: r["y0"])] == [400, 440, 480]
    lay = A6.layout_of([{"kind": "line", "box": list(b)} for b in boxes])
    assert len(lay["line"]) == 3 and lay["name"] == []
    free = S6.free_rows(lay["line"], [[100, 400, 500, 30]], "vita")
    assert [r["y0"] for r in free] == [440, 480]
    assert S6.free_rows(lay["line"], [[100, 400, 500, 30], [150, 445, 50, 20], [100, 479, 30, 30]], "vita") == []


# ----------------------------------------------------------------------------- text

def test_test_filter_blocks_test_title_text():
    import synth6_text as T6
    tf = T6.TestFilter(["「そんなこと、絶対に許さないんだからね！」", "はい"])
    assert not tf.ok("そんなこと、絶対に許さない")               # an 8-character run of a test line
    assert not tf.ok("「そんなこと、絶対に許さないんだからね！」")
    assert tf.ok("はい")                                         # short generic lines stay (under 4 characters)
    assert tf.ok("今日はいい天気ですね。")


def test_word_lines_skip_exclusion_markers():
    import synth6_text as T6
    t = T6.Text6.__new__(T6.Text6)
    t.R = type("R", (), {"exclusions": {"markers": {"有紀寧", "朋也"}}})()
    t.words = {"寧": [("有紀寧", 1.0), ("丁寧", 0.5)], "也": [("朋也", 1.0)]}
    rng = np.random.default_rng(0)
    assert all("有紀寧" not in t.word_line(rng, "寧") for _ in range(50))
    assert t.word_line(rng, "也") is None


def test_wrap_hangs_closing_marks(tmp_path, font):
    f = F6.open_font(atlas_font(tmp_path, font, "T", "bm"))
    adv = f.glyph("あ")[3]
    lines = S6.wrap(f, "あいうえお。かきく", 1.0, 0.0, 5 * adv, np.random.default_rng(0))
    assert lines == ["あいうえお。", "かきく"]
    spaced = S6.add_spaces(np.random.default_rng(1), "えっ！そうなの？うん")
    assert spaced.replace("\u3000", "") == "えっ！そうなの？うん"


def test_align_and_feasibility():
    assert S6.align("あいう", "あいう") == ["あ", "い", "う"]
    assert S6.align("あいう", "あえう") == ["あ", "え", "う"]
    assert S6.align("あいう", "あう") == ["あ", None, "う"]
    assert S6.crop_feasible((32, 80), "あいう") and not S6.crop_feasible((32, 16), "あいうえお")


# ----------------------------------------------------------------------------- compositing

def test_composite_and_legibility(tmp_path, font):
    f = F6.open_font(atlas_font(tmp_path, font, "T", "bm"))
    cov, _ = F6.draw_line(f, "今日は", x=4, y=4)
    img = np.full((cov.shape[0] + 8, cov.shape[1] + 8, 3), 90.0, np.float32)
    style = {"fill": np.array([250, 250, 250], np.float32), "dark": False,
             "halo": {"radius": 1, "width": 1, "hard": True, "alpha": 1.0, "rgb": np.zeros(3, np.float32)},
             "shadow": None}
    total = S6.composite(img, cov, 2, 2, style)
    core = cov >= 0.999
    assert np.allclose(img[2:2 + cov.shape[0], 2:2 + cov.shape[1]][core], 250.0, atol=0.5)
    assert total.max() >= 1.0 - 1e-6 and (total > cov + 0.5).any()             # the halo drew outside the glyphs
    assert S6.legibility(img, cov, total, 2, 2)[0] > 150
    flat = np.full_like(img, 245.0)
    t2 = S6.composite(flat, cov, 2, 2, {"fill": np.array([250, 250, 250], np.float32), "halo": None, "shadow": None})
    assert S6.legibility(flat, cov, t2, 2, 2)[0] < S6.CFG["min_contrast"]


# ----------------------------------------------------------------------------- one region and a shard

class StubText:
    def __init__(self):
        self.R = type("R", (), {"text_identity": {"stub": True}})()

    def utterance(self, rng, mix, title=None, font=None, narration=0.18):
        return "「今日は、いい天気ですね。」", "pool:stub", None

    def name(self, rng):
        return "見本", "name:stub"

    def clean_label(self, label, source=""):
        return "天気" not in label or "いい" in label


def fake_world(tmp_path, font):
    harvest = tmp_path / "harvest"
    atlas_font(harvest, font, "TITLEV", "bm", px=24)
    assets = tmp_path / "assets"
    pages = tmp_path / "pages" / "TITLEV" / "pages"
    pages.mkdir(parents=True)
    rng = np.random.default_rng(0)
    img = np.zeros((544, 960, 3), np.uint8)
    img[:] = rng.integers(40, 200, (1, 1, 3), dtype=np.uint8)
    img[380:530, 40:920] = (30, 30, 70)                     # the dialogue box
    cv2.imwrite(str(pages / "000001.png"), img[..., ::-1])
    rows = [{"y0": 400, "h": 26, "x0": [[80, 10]], "x1": 880, "count": 10},
            {"y0": 440, "h": 26, "x0": [[80, 10]], "x1": 880, "count": 10}]
    style = {"lines": 10, "dark_share": 0.0, "fill_rgb": [250, 250, 250], "fill_plain_share": 1.0, "accent_fills": [],
             "advance_p50": 24.0, "bg_rgb_p50": [30, 30, 70], "halo": None, "shadow": None}
    titles = {"kind": "synth6-titles", "schema": 1, "splits_sha256": "0" * 64, "test_titles": [],
              "pages_root": str(tmp_path / "pages"),
              "titles": {"TITLEV": {"title": "TITLEV", "platform": "vita", "dims": [960, 544],
                                    "layout": {"line": rows, "name": []}, "style": {"line": style},
                                    "box": {"alpha": 0.8, "rgb": [30, 30, 70]},
                                    "font": {"font": "TITLEV/bm", "size_or_scale": 1.0, "spacing": 0.5, "corr": 0.95},
                                    "pages": {"000001": [[80, 400, 300, 26]]}}}}
    assets.mkdir()
    (assets / "titles.json").write_text(json.dumps(titles), encoding="utf-8")
    inv = F6.inventory(harvest, test_titles=("PCSG00415",), train_titles={"TITLEV"})
    (assets / "fonts_inventory.json").write_text(json.dumps(inv), encoding="utf-8")
    (assets / "hooktext.jsonl").write_text("", encoding="utf-8")
    (assets / "real_counts.json").write_text("{}", encoding="utf-8")
    W = S6.World(assets, harvest, text=False)
    W.text = StubText()
    S6.W = W
    return W


def test_render_region_and_shard_format(tmp_path, font, monkeypatch):
    fake_world(tmp_path, font)
    monkeypatch.setitem(S6.CFG, "placement", {"rows": 1.0, "scene": 0.0, "scene_box": 0.0})
    monkeypatch.setitem(S6.CFG, "matched", 1.0)
    rng = np.random.default_rng(3)
    got = None
    for _ in range(20):
        got = S6.render_region(rng, "vita")
        if got:
            break
    img, lines, meta = got
    assert meta["title"] == "TITLEV" and meta["mode"] == "matched" and meta["placement"] == "rows"
    # the page's own text row (y 400) is masked: lines go to the free row only
    assert all(ln["cell"][1] >= 430 for ln in lines if ln["cell"])
    out = S6.extract(rng, img, lines, meta)
    assert out and all(lab and set(lab) <= set(S6.canon("「今日は、いい天気ですね。」")) for _, lab, _ in out)
    crop, lab, m = out[0]
    assert crop.dtype == np.uint8 and crop.ndim == 2 and m["crop"] in ("cell", "ink", "grad")
    # a shard: data / offsets / shapes / labels / meta, every label canon and in the vocab
    monkeypatch.setitem(S6.CFG, "placement", {"rows": 0.6, "scene": 0.2, "scene_box": 0.2})
    idx, n, regions, errors, _ = S6.run_shard((0, 12, 1, "vita", str(tmp_path), "t6"))
    assert n == 12 and errors == 0
    z = np.load(tmp_path / "t6_00000.npz")
    labels, metas = json.loads(str(z["labels"])), json.loads(str(z["meta"]))
    assert len(labels) == len(metas) == len(z["shapes"]) == 12
    assert z["offsets"][-1] == len(z["data"]) == int((z["shapes"][:, 0] * z["shapes"][:, 1]).sum())
    assert all(lab == S6.canon(lab) for lab in labels)
    assert all("real_boxes" not in mm for mm in metas)
    # determinism: the same shard index and seed render the same bytes
    (tmp_path / "again").mkdir()
    again = S6.run_shard((0, 12, 1, "vita", str(tmp_path / "again"), "t6"))
    z2 = np.load(tmp_path / "again" / "t6_00000.npz")
    assert again[1] == 12 and np.array_equal(z["data"], z2["data"]) and str(z["labels"]) == str(z2["labels"])


def test_psp2x_view_doubles_the_frame():
    img = np.zeros((272, 480, 3), np.float32)
    out, k, filt = S6.to_view(np.random.default_rng(0), img, "psp2x")
    assert out.shape == (544, 960, 3) and k == 2 and filt in S6.CFG["adrenaline"]
    same, k1, f1 = S6.to_view(np.random.default_rng(0), img, "psp")
    assert same is img and k1 == 1 and f1 is None


# ----------------------------------------------------------------------------- the audit's defects (B1-B4, F2)

def test_row_chain_never_stacks_rows():
    rows = [{"y0": y, "h": 29, "x0": [[100, 9]], "x1": 800, "count": 9} for y in (64, 69, 80, 99, 110, 115, 145)]
    rng = np.random.default_rng(0)
    for _ in range(200):
        ch = S6.row_chain(rng, rows, 4)
        assert ch and all(b["y0"] >= a["y0"] + a["h"] and b["y0"] - a["y0"] <= 2.5 * a["h"] for a, b in zip(ch, ch[1:]))


def test_stacked_and_overlaps():
    assert S6.stacked([0, 100, 200, 30], [10, 105, 200, 30])            # drawn on top of each other
    assert not S6.stacked([0, 100, 200, 30], [0, 127, 200, 30])         # neighbours whose halos touch (3 px)
    assert not S6.stacked([0, 100, 200, 30], [300, 100, 200, 30])       # side by side
    assert S6.overlaps([0, 0, 10, 10], [5, 5, 10, 10]) and not S6.overlaps([0, 0, 10, 10], [10, 0, 5, 5])


def test_wrap_caps_the_overhang(tmp_path, font):
    f = F6.open_font(atlas_font(tmp_path, font, "T", "bm"))
    adv = f.glyph("あ")[3]
    lines = S6.wrap(f, "あいうえお……！？」かき", 1.0, 0.0, 5 * adv, np.random.default_rng(0))
    assert lines[0] == "あいうえお…"                 # one closing mark hangs, the next breaks the line
    assert all(S6.F6.line_advance(f, ln) <= 6 * adv for ln in lines)


def test_lines_off_the_frame_or_stacked_are_not_cropped(tmp_path, font, monkeypatch):
    fake_world(tmp_path, font)
    img = np.zeros((544, 960, 3), np.float32)
    ok = {"text": "今日は", "kind": "line", "ink": [100, 400, 80, 26], "cell": [100, 400, 80, 26], "legible": True}
    cut = {"text": "今日は", "kind": "line", "ink": [900, 400, 80, 26], "cell": [900, 400, 80, 26], "legible": True}
    low = {"text": "いい天気", "kind": "line", "ink": [100, 530, 80, 20], "cell": [100, 530, 80, 20], "legible": True}
    meta = {"view": "vita", "title": "TITLEV", "real_boxes": [], "mode": "matched", "placement": "rows"}
    monkeypatch.setitem(S6.CFG, "crop", {"ink": 1.0})
    got = S6.extract(np.random.default_rng(0), img, [ok, cut, low], dict(meta))
    assert [m["ink"] for _, _, m in got] == [[100, 400, 80, 26]]
    a = dict(ok)
    b = dict(ok, ink=[110, 405, 80, 26], cell=[110, 405, 80, 26])
    assert S6.extract(np.random.default_rng(0), img, [a, b], dict(meta)) == []


def test_misplaced_glyphs(tmp_path, font):
    d = atlas_font(tmp_path, font, "T", "bm")
    f = F6.open_font(d)
    x, y, w, h, ox, oy, adv = f.chars["・"]
    f.chars["・"] = [x, y, w, h, 6, oy, 6]                 # a full cell at +6 with a half-width advance (B3)
    f.chars["一"] = [x, y, w, h, ox, oy, adv]              # a blank cell would be "no visible ink"
    f.atlas[y:y + h, x:x + w] = 0
    f._cache.clear()
    bad = set(F6.misplaced(f, "あ・一「」。"))
    assert bad == {"・", "一"}


def test_near_identical_copies_of_a_test_font_are_excluded(tmp_path, font):
    atlas_font(tmp_path, font, "PCSG00415", "gothp32")
    cp = atlas_font(tmp_path, font, "OTHER", "copy")                       # the same glyph bitmaps, re-encoded:
    from PIL import Image
    Image.open(cp / "atlas.png").save(cp / "atlas.png", compress_level=1)  # other PNG bytes, same pixels
    g = json.loads((cp / "glyphs.json").read_text(encoding="utf-8"))
    (cp / "glyphs.json").write_text(json.dumps(g, ensure_ascii=False, indent=2), encoding="utf-8")
    m = json.loads((cp / "font.json").read_text(encoding="utf-8"))
    m.update(fidelity_by_title={}, content_sha256="x" * 64)
    (cp / "font.json").write_text(json.dumps(m), encoding="utf-8")
    atlas_font(tmp_path, font, "TRAIN1", "own", px=20)                     # another size: other bitmaps
    old = F6.NEAR_MIN_COMMON
    F6.NEAR_MIN_COMMON = 10
    try:
        rows = {r["font"]: r for r in F6.inventory(tmp_path, test_titles=("PCSG00415",), train_titles=set())}
    finally:
        F6.NEAR_MIN_COMMON = old
    assert any("near-identical" in w for w in rows["OTHER/copy"]["excluded"])
    assert rows["OTHER/copy"]["near_test_font"]["font"] == "PCSG00415/gothp32"
    assert not rows["TRAIN1/own"]["excluded"]


def test_legibility_catches_a_partly_invisible_line(tmp_path, font):
    f = F6.open_font(atlas_font(tmp_path, font, "T", "bm"))
    cov, cells = F6.draw_line(f, "今日はいい天気", x=4, y=4)
    img = np.full((cov.shape[0] + 8, cov.shape[1] + 8, 3), 40.0, np.float32)
    half = cov.shape[1] // 2
    img[:, half + 2:] = 245.0                                           # the right half over a bright patch
    st = {"fill": np.array([250, 250, 250], np.float32), "halo": None, "shadow": None}
    total = S6.composite(img, cov, 2, 2, st)
    contrast, busy, worst = S6.legibility(img, cov, total, 2, 2, cells)
    assert worst < S6.CFG["min_char_contrast"] <= contrast or contrast < S6.CFG["min_contrast"]


def test_shifted_decoder_tables_are_found(tmp_path, font):
    ref = F6.OutlineFont(font, size=48)
    letters = "ABCDEFGHIJKLMNOPQRST"
    good = F6.open_font(atlas_font(tmp_path, font, "T1", "good", px=32, chars=letters))
    assert F6.swapped(good, letters, [ref]) == []
    # the same atlas, but every letter from E on shows the glyph of the letter before it (a table off by one)
    bad = F6.open_font(atlas_font(tmp_path, font, "T2", "bad", px=32, chars=letters))
    rects = {c: list(bad.chars[c]) for c in letters}
    for prev, c in zip(letters[3:], letters[4:]):
        bad.chars[c] = rects[prev]
    bad._cache.clear()
    got = set(F6.swapped(bad, letters, [ref]))
    assert len(got & set(letters[4:])) >= 10 and not got & set(letters[:4])
    one = F6.open_font(atlas_font(tmp_path, font, "T3", "one", px=32, chars=letters))
    one.chars["K"] = list(one.chars["J"])                         # a single odd glyph is not a shifted table
    one._cache.clear()
    assert F6.swapped(one, letters, [ref]) == []


def test_grad_boxes_must_cover_the_line(monkeypatch):
    grey = np.zeros((200, 400), np.uint8)
    line = {"ink_v": [50, 100, 250, 30], "text": "x"}             # region: x 38-312, y 88-142
    monkeypatch.setattr(S6, "find_lines_ex", lambda reg, mode: ([(9, 15, 256, 15)], [0], False))
    assert S6.grad_boxes(grey, [line], [], 12, 1) == {}              # a 15 px band through a 30 px line: refused
    monkeypatch.setattr(S6, "find_lines_ex", lambda reg, mode: ([(9, 9, 256, 36)], [0], False))
    assert S6.grad_boxes(grey, [line], [], 12, 1) == {0: (47, 97, 256, 36)}
    # a line that starts at x 2, at the region's edge: a box holding only its right part is refused (F7)
    edge = {"ink_v": [2, 100, 250, 30], "text": "x"}                # region: x 0-264, y 88-142
    monkeypatch.setattr(S6, "find_lines_ex", lambda reg, mode: ([(120, 9, 135, 36)], [0], False))
    assert S6.grad_boxes(grey, [edge], [], 12, 1) == {}
    monkeypatch.setattr(S6, "find_lines_ex", lambda reg, mode: ([(0, 9, 255, 36)], [0], False))
    assert S6.grad_boxes(grey, [edge], [], 12, 1) == {0: (0, 97, 255, 36)}
    # the mirror case: a line that ends at the frame's right edge, and a box that holds only its left part
    right = {"ink_v": [148, 100, 250, 30], "text": "x"}             # region: x 136-400, y 88-142
    monkeypatch.setattr(S6, "find_lines_ex", lambda reg, mode: ([(9, 9, 135, 36)], [0], False))
    assert S6.grad_boxes(grey, [right], [], 12, 1) == {}
    monkeypatch.setattr(S6, "find_lines_ex", lambda reg, mode: ([(9, 9, 255, 36)], [0], False))
    assert S6.grad_boxes(grey, [right], [], 12, 1) == {0: (145, 97, 255, 36)}


def test_shift_steps_over_unassigned_codes():
    assert F6.cp932_shift("◯", -1) == "¶"          # 0x81F8-0x81FB are unassigned: ◯'s previous assigned code is ¶
    assert F6.cp932_shift("」", -1) == "「" and F6.cp932_shift("B", -1) == "A" and F6.cp932_shift("A", 1) == "B"
    assert F6.cp932_shift("é", 1) is None             # not in cp932


def test_manual_denials_reach_the_renderer(tmp_path, font):
    W = fake_world(tmp_path, font)
    assert W.font("TITLEV/bm").has("今")
    (tmp_path / "assets" / "glyph_geometry.json").write_text(json.dumps(
        {"fonts": {"TITLEV/bm": {"misplaced": "", "swapped": "", "manual": "今"}}}), encoding="utf-8")
    W2 = S6.World(tmp_path / "assets", tmp_path / "harvest", text=False)
    assert not W2.font("TITLEV/bm").has("今") and W2.font("TITLEV/bm").has("日")


def test_raster_share_ignores_the_palette(tmp_path, font):
    import shutil

    from PIL import Image
    a = atlas_font(tmp_path, font, "PCSG00415", "ref", px=30)
    # a 4-bit reference, levels k x 17 (as CLANNAD's fonts store them)
    q = np.round(np.asarray(Image.open(a / "atlas.png"), np.float32) / 255.0 * 15).astype(int)
    Image.fromarray((q * 17).astype(np.uint8)).save(a / "atlas.png")
    ref = F6.open_font(a)
    # the same raster, re-quantised through a gamma-like palette (as Little Busters' LUCA fonts)
    b = tmp_path / "OTHER" / "copy"
    shutil.copytree(a, b)
    pal = np.array([0] + [round(255 * (k / 15) ** 0.6) for k in range(1, 16)], np.uint8)
    Image.fromarray(pal[q]).save(b / "atlas.png")
    copy = F6.open_font(b)
    assert F6.raster_share(copy, ref) > 0.9
    other = F6.open_font(atlas_font(tmp_path, font, "THIRD", "bold", px=30, chars=CHARS))
    other.atlas = np.roll(other.atlas, 2, axis=1)                       # another raster: shifted strokes
    assert F6.raster_share(other, ref) < 0.5
    # another raster of the same typeface: the same font rasterised at 28 px instead of 30
    other2 = F6.open_font(atlas_font(tmp_path, font, "FOURTH", "px28", px=28))
    assert F6.raster_share(other2, ref) < 0.5
    # the tolerance's edge: 15% of the pixels at level 3 or more moved down by exactly 2 levels, at random. No tone
    # map absorbs a per-pixel change, and a glyph with one such pixel fails the one-level tolerance (a two-level
    # tolerance would accept it)
    edge_dir = tmp_path / "EDGE" / "copy"
    shutil.copytree(a, edge_dir)
    rng = np.random.default_rng(0)
    noisy = np.where((q >= 3) & (rng.random(q.shape) < 0.15), q - 2, q)
    Image.fromarray((noisy * 17).astype(np.uint8)).save(edge_dir / "atlas.png")
    assert F6.raster_share(F6.open_font(edge_dir), ref) < 0.5
    # a two-level (1-bit) font gives no evidence: 0, not 1
    bit = tmp_path / "BIT" / "copy"
    shutil.copytree(a, bit)
    Image.fromarray(((q > 7) * 255).astype(np.uint8)).save(bit / "atlas.png")
    assert F6.raster_share(F6.open_font(bit), ref) == 0.0
