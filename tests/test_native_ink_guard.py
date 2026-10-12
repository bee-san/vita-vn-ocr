"""Generated bitmap glyphs prove finder crops preserve labelled core strokes."""
import math
import struct
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

import native_bitmap_font as N
import render


@pytest.fixture
def bitmap():
    codes = sorted(int.from_bytes(c.encode("cp932"), "big") for c in "AB　")
    patterns = {}
    for char in "AB　":
        pixels = [0] * 64
        if char == "A":
            pixels[0] = 1  # antialias fringe below the substantial-core threshold
            pixels[2 * 8 + 2] = pixels[5 * 8 + 5] = 15
        elif char == "B":
            pixels[0] = pixels[7 * 8 + 7] = 15
        patterns[char] = bytes((pixels[i] << 4) | pixels[i + 1] for i in range(0, 64, 2))
    bank = b"".join(patterns[code.to_bytes(2 if code > 255 else 1, "big").decode("cp932")]
                    for code in codes)
    fnt = (struct.pack("<HHI", 8, 8, len(codes)) + bytes(8)
           + struct.pack(f"<{len(codes)}H", *codes) + bank * 2)
    count = max(N.metric_index(code) for code in codes) + 1
    metrics = bytearray([0, 0, 8] * count)
    index = N.metric_index("B") * 3
    metrics[index:index + 3] = bytes([254, 2, 8])  # signed negative x bearing + descender
    pro = struct.pack("<I", count) + bytes(12) + metrics
    return N.NativeBitmapFont(N.parse_fnt(fnt), N.parse_pro(pro), size=8)


@pytest.mark.parametrize("x,base", [(6.6, 16.4), (6.4, 16.6), (-.6, 16.5)])
@pytest.mark.parametrize("char", ["A", "B"])
def test_fractional_bounds_match_pillow_thresholded_pixels(bitmap, x, base, char):
    image = Image.new("L", (40, 40))
    ImageDraw.Draw(image).text((x, base), char, font=bitmap, anchor="ls", fill=255)
    core = image.point(lambda value: 255 if value >= 128 else 0).getbbox()
    expected = render.native_glyph_core_box(bitmap, char, x, base)
    # The negative-origin B draw is clipped by the canvas; its unclipped bound
    # remains outside the image and must cause a native crop to be rejected.
    if expected[0] < 0:
        assert not render.preserves_core((0, 0, 40, 40), expected)
        assert core[0] >= 0
    else:
        assert expected == core


def test_threshold_excludes_fringe_and_empty_spacing_cells(bitmap):
    assert bitmap.getbbox("A", anchor="ls") == (0, -8, 6, -2)
    assert render.native_glyph_core_box(bitmap, "A", 0, 8) == (2, 2, 6, 6)
    assert render.native_glyph_core_box(bitmap, "　", 0, 8) is None
    assert render.union_core_box(None, None) is None
    assert render.union_core_box((2, 2, 6, 6), None) == (2, 2, 6, 6)


@pytest.mark.parametrize("track", [-1.2, 0, 3.4])
def test_tracking_spaces_and_metric_bearings_match_per_character_draws(bitmap, track):
    image = Image.new("L", (60, 40))
    draw = ImageDraw.Draw(image)
    x, base, box = 12.6, 16.4, None
    for char in "A　B":
        if char != "　":
            draw.text((x, base), char, font=bitmap, anchor="ls", fill=255)
            box = render.union_core_box(box, render.native_glyph_core_box(bitmap, char, x, base))
        x += bitmap.getlength(char) + track
    assert box == image.point(lambda value: 255 if value >= 128 else 0).getbbox()


def test_supersample_coordinates_are_rounded_outward(bitmap):
    font = bitmap.font_variant(size=16)
    high = render.native_glyph_core_box(font, "B", 9.4, 31.6)
    low = render.native_glyph_core_box(font, "B", 4.7, 15.8, ss=2)
    assert low == (math.floor(high[0] / 2), math.floor(high[1] / 2),
                   math.ceil(high[2] / 2), math.ceil(high[3] / 2))


def extraction_setup(monkeypatch, bitmap, *, native=True, nominal=(8, 23), detector=(8, 6, 18, 18)):
    image = Image.new("L", (40, 40))
    draw = ImageDraw.Draw(image)
    core = None
    for char, x, base in (("A", 10, 15), ("B", 16, 20)):
        draw.text((x, base), char, font=bitmap, anchor="ls", fill=255)
        core = render.union_core_box(core, render.native_glyph_core_box(bitmap, char, x, base))
    # Extra unlabelled distractor ink must not become part of the preservation box.
    draw.rectangle((32, 1, 35, 4), fill=255)
    line = {"text": "AB", "kind": "dialogue", "fs": 8,
            "y0": nominal[0], "y1": nominal[1], "ix0": 12, "ix1": 22}
    if native:
        line["native_core_box"] = core
    monkeypatch.setattr(render, "find_lines_ex", lambda *args: (np.array([detector]), np.array([0])))
    rgb = np.repeat(np.asarray(image)[:, :, None], 3, axis=2)
    return rgb, line, core


@pytest.mark.parametrize("detector", [(8, 10, 18, 14), (8, 6, 18, 15)])
def test_original_finder_crop_rejects_cut_top_or_descender(monkeypatch, bitmap, detector):
    rgb, line, core = extraction_setup(monkeypatch, bitmap, detector=detector)
    assert not render.preserves_core(detector, core)
    assert render.extract(rgb, [line], finder="grad") == []


def test_nominal_whitespace_slack_does_not_reject_preserved_strokes(monkeypatch, bitmap):
    rgb, line, core = extraction_setup(monkeypatch, bitmap, nominal=(0, 28))
    got = render.extract(rgb, [line], finder="grad")
    assert len(got) == 1
    metadata = got[0][3]
    assert metadata["native_core_box"] == list(core)
    assert metadata["native_core_alpha_threshold"] == 128
    assert metadata["native_label_ink_preserved"] is True
    assert metadata["native_core_jitter_reverted"] is False


class AlwaysJitter:
    def __init__(self):
        self.calls = 0

    def random(self):
        self.calls += 1
        return 0


def test_clipping_jitter_reverts_to_original_and_reports_actual_change(monkeypatch, bitmap):
    rgb, line, _ = extraction_setup(monkeypatch, bitmap)
    monkeypatch.setattr(render, "edge_box", lambda *args: (8, 10, 18, 14))
    rng = AlwaysJitter()
    got = render.extract(rgb, [line], finder="grad", rng=rng, edge_jitter=True)
    assert rng.calls == 1
    assert len(got) == 1
    assert got[0][2] == (8, 6, 18, 18)
    metadata = got[0][3]
    assert metadata["original_crop_box"] == metadata["crop_box"] == [8, 6, 18, 18]
    assert metadata["crop_edge_jitter_applied"] is True
    assert metadata["crop_edge_jitter_changed"] is False
    assert metadata["native_core_jitter_reverted"] is True


def test_original_rejection_consumes_no_jitter_rng(monkeypatch, bitmap):
    rgb, line, _ = extraction_setup(monkeypatch, bitmap, detector=(8, 10, 18, 14))
    rng = AlwaysJitter()
    assert render.extract(rgb, [line], finder="grad", rng=rng, edge_jitter=True) == []
    assert rng.calls == 0


def test_ofl_crop_behavior_rng_and_metadata_stay_unchanged(monkeypatch, bitmap):
    rgb, line, _ = extraction_setup(monkeypatch, bitmap, native=False)
    monkeypatch.setattr(render, "edge_box", lambda *args: (8, 10, 18, 14))
    rng = AlwaysJitter()
    got = render.extract(rgb, [line], finder="grad", rng=rng, edge_jitter=True)
    assert rng.calls == 1 and len(got) == 1
    assert got[0][2] == (8, 10, 18, 14)
    assert got[0][3]["crop_edge_jitter_changed"] is True
    assert "native_label_ink_preserved" not in got[0][3]


def test_crop_metadata_keeps_native_guard_evidence(monkeypatch):
    monkeypatch.setattr(render, "R", SimpleNamespace(crop_edge_jitter=True, policy_sha256=None))
    line = {"kind": "dialogue", "original_crop_box": [1, 2, 20, 16], "crop_box": [1, 2, 20, 16],
            "crop_edge_jitter_applied": True, "crop_edge_jitter_changed": False,
            "native_core_box": [4, 5, 18, 14], "native_core_alpha_threshold": 128,
            "native_label_ink_preserved": True, "native_core_jitter_reverted": True}
    metadata = render.crop_meta({"native_font_used": True}, line)
    for key in ("native_core_box", "native_core_alpha_threshold", "native_label_ink_preserved",
                "native_core_jitter_reverted"):
        assert metadata[key] == line[key]
