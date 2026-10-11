"""Source pinning, full-label fallback and bitmap renderer dispatch."""
import hashlib
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

import native_bitmap_font as N
import native_font_manifest as M
import render


@pytest.fixture
def pinned(tmp_path):
    codes = sorted(int.from_bytes(c.encode("cp932"), "big") for c in "A　あっ")
    blob = (struct.pack("<HHI", 8, 8, len(codes)) + bytes(8)
            + struct.pack(f"<{len(codes)}H", *codes)
            + bytes([0xf0] * 32 * len(codes)) + bytes([0x0f] * 32 * len(codes)))
    count = max(N.metric_index(code) for code in codes) + 1
    pro = struct.pack("<I", count) + bytes(12) + bytes([0, 0, 8] * count)
    record = {"family": "generated_bitmap", "category": "gothic"}
    for suffix, data in (("fnt", blob), ("pro", pro)):
        path = tmp_path / f"fixture.{suffix}"
        path.write_bytes(data)
        record[suffix] = {"file": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps({"schema": 1, "redistribute": False, "fonts": [record]}))
    return manifest, tmp_path, record


def test_loads_only_pinned_local_sources_and_binds_coverage(pinned):
    manifest, root, record = pinned
    faces, rows, binding = M.load_manifest(manifest, root)
    assert rows[0][3] == set("A　あっ")
    assert binding["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert binding["redistribute"] is False
    assert binding["fonts"][0]["source_cell"] == [8, 8]
    assert list(faces.values())[0].glyphs.sha256 == record["fnt"]["sha256"]


@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("bytes", 1),
                                         ("file", "../fixture.fnt"), ("file", "/tmp/fixture.fnt")])
def test_rejects_unpinned_or_escaping_sources(pinned, field, value):
    manifest, root, _ = pinned
    data = json.loads(manifest.read_text())
    data["fonts"][0]["fnt"][field] = value
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="mismatch|relative path"):
        M.load_manifest(manifest, root)


def test_symlink_must_resolve_inside_source_root(pinned, tmp_path):
    manifest, root, record = pinned
    outside = root.parent / f"{root.name}-outside.fnt"
    outside.write_bytes((root / record["fnt"]["file"]).read_bytes())
    (root / "link.fnt").symlink_to(outside)
    data = json.loads(manifest.read_text())
    data["fonts"][0]["fnt"]["file"] = "link.fnt"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="escapes"):
        M.load_manifest(manifest, root)


@pytest.mark.parametrize("value", [-.01, 1.01, float("nan"), float("inf")])
def test_probability_cannot_hide_nonfinite_or_out_of_range_sampling(value):
    with pytest.raises(ValueError, match="probability"):
        M.probability(value)


def test_missing_glyph_uses_one_covered_fallback_for_the_whole_region(pinned, monkeypatch):
    faces, rows, _ = M.load_manifest(*pinned[:2])
    fallback = ("gothic", "fixture_ofl", pinned[1] / "fixture.ttf", set("A　あっ土"))
    resources = SimpleNamespace(native_font_probability=1, native_faces=faces, native_fonts=rows,
                                by_cat={"gothic": [fallback]}, fonts=[fallback])
    monkeypatch.setattr(render, "R", resources)
    rng = np.random.default_rng(42)
    assert render.pick_font(rng, set("あっ"), "vita")[1] == "generated_bitmap"
    assert render.pick_font(rng, set("あ土"), "psp") == fallback
    assert render.pick_font(rng, {"龜"}, "psp") is None


def test_disabled_native_fonts_preserve_rng_sequence(pinned, monkeypatch):
    faces, rows, _ = M.load_manifest(*pinned[:2])
    fallback = ("gothic", "fixture_ofl", pinned[1] / "fixture.ttf", set("A　あっ"))
    resources = SimpleNamespace(native_font_probability=0, native_faces=faces, native_fonts=rows,
                                by_cat={"gothic": [fallback]}, fonts=[fallback])
    monkeypatch.setattr(render, "R", resources)
    rng, expected = np.random.default_rng(42), np.random.default_rng(42)
    render.pick_font(rng, set("あっ"), "vita", cat="gothic")
    expected.integers(1)
    assert rng.random() == expected.random()


def test_bitmap_font_dispatch_keeps_strikes_separate_and_supports_strokes(pinned, monkeypatch):
    faces, _, _ = M.load_manifest(*pinned[:2])
    monkeypatch.setattr(render, "R", SimpleNamespace(native_faces=faces))
    monkeypatch.setattr(render, "_FONTS", {})
    path = next(iter(faces))
    fonts = [render.get_font(path, 16, strike) for strike in (0, 1)]
    assert [font.strike for font in fonts] == [0, 1]
    assert fonts[0].glyphs is fonts[1].glyphs
    image = Image.new("L", (100, 40))
    ImageDraw.Draw(image).text((4, 25), "あっ", font=fonts[0], anchor="ls", fill=255,
                              stroke_width=1, stroke_fill=180)
    assert image.getbbox() is not None
    assert render.pick_weight(np.random.default_rng(10), path) in (0, 1)


def test_font_cache_also_pins_metric_bytes(pinned, monkeypatch):
    manifest, root, record = pinned
    faces, _, _ = M.load_manifest(manifest, root)
    resources = SimpleNamespace(native_faces=faces)
    monkeypatch.setattr(render, "R", resources)
    monkeypatch.setattr(render, "_FONTS", {})
    path = next(iter(faces))
    old_font = render.get_font(path, 16, 0)
    pro = root / record["pro"]["file"]
    data = bytearray(pro.read_bytes())
    data[16 + N.metric_index("あ") * 3 + 2] = 4
    pro.write_bytes(data)
    pin = json.loads(manifest.read_text())
    pin["fonts"][0]["pro"]["sha256"] = hashlib.sha256(data).hexdigest()
    manifest.write_text(json.dumps(pin))
    resources.native_faces, _, _ = M.load_manifest(manifest, root)
    new_font = render.get_font(path, 16, 0)
    assert old_font.getlength("あ") == 16 and new_font.getlength("あ") == 8
    assert old_font is not new_font


def test_native_baseline_keeps_top_row_strokes_inside_declared_line_band(pinned, monkeypatch):
    faces, _, _ = M.load_manifest(*pinned[:2])
    monkeypatch.setattr(render, "R", SimpleNamespace(native_faces=faces))
    monkeypatch.setattr(render, "_FONTS", {})
    path = next(iter(faces))
    font = render.get_font(path, 16, 0)
    top = 12
    baseline = render.text_baseline(top, 16, path)
    image = Image.new("L", (64, 64))
    ImageDraw.Draw(image).text((4, baseline), "あっ", font=font, anchor="ls", fill=255)
    assert image.getbbox()[1] == top
    assert image.getbbox()[3] <= top + 16
    assert render.text_baseline(top, 16, path.with_suffix(".ttf")) == top + .88 * 16


def test_native_band_includes_pro_metric_descenders_and_negative_bearings(pinned):
    faces, _, _ = M.load_manifest(*pinned[:2])
    original = next(iter(faces.values()))
    payload = bytearray(original.metrics._payload)
    index = N.metric_index("あ") * 3
    payload[index + 1] = 2  # positive y bearing extends ink below the nominal cell
    metrics = N.ProTable(original.metrics.count, bytes(payload), "generated-positive")
    font = N.NativeBitmapFont(original.glyphs, metrics, size=16)
    assert render.native_vertical_band(font, "あ", 12, 16) == (12, 32)
    payload[index + 1] = 254  # signed -2 extends above the nominal cell
    metrics = N.ProTable(original.metrics.count, bytes(payload), "generated-negative")
    font = N.NativeBitmapFont(original.glyphs, metrics, size=16)
    assert render.native_vertical_band(font, "あ", 12, 16) == (8, 28)
