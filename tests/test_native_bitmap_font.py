"""Native font parsing and Pillow rendering using generated 4bpp fixtures only."""
import hashlib
import math
import struct

import pytest
from PIL import Image, ImageDraw, ImageFont

import native_bitmap_font as N


def _nibbles(values):
    assert len(values) % 2 == 0
    return bytes((values[i] << 4) | values[i + 1] for i in range(0, len(values), 2))


def _fnt(codes=(0x41, 0x42), width=2, height=2, banks=None):
    """Two sequential glyph banks; this is generated test data, not a font extract."""
    if banks is None:
        banks = [[list(range(width * height)) for _ in codes] for _ in range(2)]
    return (struct.pack("<HHI", width, height, len(codes)) + bytes(8)
            + struct.pack("<" + "H" * len(codes), *codes)
            + b"".join(_nibbles(pixels) for bank in banks for pixels in bank))


def _pro(count=67, records=None):
    records = records or {}
    triplets = [records.get(i, (0, 0, 2)) for i in range(count)]
    return (struct.pack("<I", count) + bytes(12)
            + bytes(value & 255 for triplet in triplets for value in triplet))


def _font(codes=(0x41,), *, size=2, strike=0, records=None, banks=None):
    if banks is None:
        banks = [[[15] * 4 for _ in codes] for _ in range(2)]
    glyphs = N.parse_fnt(_fnt(codes, banks=banks))
    count = max(N.metric_index(code) for code in codes) + 1
    return N.NativeBitmapFont(glyphs, N.parse_pro(_pro(count, records)),
                              size=size, strike=strike)


def test_two_sequential_banks_decode_high_nibble_first_and_keep_source_hash():
    banks = [[[0, 1, 14, 15], [2, 3, 4, 5]],
             [[6, 7, 8, 9], [10, 11, 12, 13]]]
    blob = _fnt(banks=banks)
    glyphs = N.parse_fnt(blob)
    assert (glyphs.width, glyphs.height, glyphs.codes) == (2, 2, (0x41, 0x42))
    assert glyphs.sha256 == hashlib.sha256(blob).hexdigest()
    for strike in (0, 1):
        for index, code in enumerate(glyphs.codes):
            image = glyphs.glyph(code, strike=strike)
            assert image.mode == "L" and image.size == (2, 2)
            assert list(image.tobytes()) == [v * 17 for v in banks[strike][index]]
    # A caller may edit the returned image without changing the source bitmap.
    image = glyphs.glyph(0x41)
    image.putpixel((0, 0), 255)
    assert glyphs.glyph(0x41).getpixel((0, 0)) == 0


@pytest.mark.parametrize("code,text,index", [
    (0x41, "A", 65), (0x42, "B", 66), (0x82A0, "あ", 539),
    (0x8141, "、", 257), (0x8740, "①", 1384),
])
def test_metric_index_maps_cp932_to_jis_grid_including_nec_extension(code, text, index):
    assert N.metric_index(code) == index
    assert N.metric_index(text) == index
    if text in ("あ", "、"):
        row, column = text.encode("euc_jp")
        assert index == 256 + (row - 0xA1) * 94 + column - 0xA1


@pytest.mark.parametrize("value", ["", "AB", "☃", "😀", "〜", -1, 0x10000, 0x817F, 0xFFFF])
def test_invalid_or_noncanonical_codes_cannot_be_metric_indices(value):
    with pytest.raises((ValueError, TypeError)):
        N.metric_index(value)


def test_pro_keeps_raw_triplets_and_exposes_signed_bearing_hints():
    metrics = N.parse_pro(_pro(records={65: (-1, -128, 3)}))
    assert metrics.count == 67
    record = metrics.record(65)
    assert record.raw == (255, 128, 3)
    assert record.x_bearing_hint == -1
    assert record.y_bearing_hint == -128
    assert record.advance_hint == 3


def test_training_ambiguity_requires_identical_nonblank_ink_and_metrics():
    assert _font(codes=(0x41, 0x42)).ambiguous_glyph_groups() == ("AB",)
    assert _font(codes=(0x41, 0x42), records={66: (0, 0, 3)}).ambiguous_glyph_groups() == ()
    blank = [[[0] * 4, [0] * 4] for _ in range(2)]
    assert _font(codes=(0x41, 0x42), banks=blank).ambiguous_glyph_groups() == ()
    one_bank = [[[15] * 4, [15] * 4], [[15] * 4, [0, 15, 15, 15]]]
    assert _font(codes=(0x41, 0x42), banks=one_bank).ambiguous_glyph_groups() == ("AB",)


@pytest.mark.parametrize("index", [-1, 67, 1000000])
def test_pro_record_bounds_do_not_wrap(index):
    with pytest.raises((IndexError, ValueError)):
        N.parse_pro(_pro()).record(index)


@pytest.mark.parametrize("blob", [
    b"", bytes(15),
    struct.pack("<HHI", 0, 2, 1) + bytes(8),
    struct.pack("<HHI", 2, 0, 1) + bytes(8),
    struct.pack("<HHI", 3, 3, 1) + bytes(8),
    struct.pack("<HHI", 129, 2, 1) + bytes(8),
    struct.pack("<HHI", 2, 129, 1) + bytes(8),
    struct.pack("<HHI", 2, 2, 0) + bytes(8),
    struct.pack("<HHI", 2, 2, 16385) + bytes(8),
    struct.pack("<HHI", 65535, 65535, 0xFFFFFFFF) + bytes(8),
    _fnt()[:17], _fnt()[:-1], _fnt() + b"\0",
    _fnt((0x42, 0x41)), _fnt((0x41, 0x41)), _fnt((0x817F,)), _fnt((0xFFFF,)),
])
def test_fnt_rejects_malformed_dimensions_lengths_and_codes(blob):
    with pytest.raises(N.FontFormatError):
        N.parse_fnt(blob)


@pytest.mark.parametrize("offset", range(8, 16))
def test_fnt_reserved_header_bytes_are_checked(offset):
    blob = bytearray(_fnt())
    blob[offset] = 1
    with pytest.raises(N.FontFormatError):
        N.parse_fnt(bytes(blob))


@pytest.mark.parametrize("blob", [
    b"", bytes(15), struct.pack("<I", 0) + bytes(12),
    struct.pack("<I", 9093) + bytes(12),
    struct.pack("<I", 0xFFFFFFFF) + bytes(12),
    _pro()[:-1], _pro() + b"\0",
])
def test_pro_rejects_malformed_count_and_exact_length(blob):
    with pytest.raises(N.FontFormatError):
        N.parse_pro(blob)


@pytest.mark.parametrize("offset", range(4, 16))
def test_pro_reserved_header_bytes_are_checked(offset):
    blob = bytearray(_pro())
    blob[offset] = 1
    with pytest.raises(N.FontFormatError):
        N.parse_pro(bytes(blob))


def test_coverage_uses_canonical_unicode_and_rejects_absent_or_alias_characters():
    font = _font((0x41, 0x8160, 0x82A0, 0x8740))
    assert isinstance(font, ImageFont.FreeTypeFont)
    assert font.coverage == {ord("A"), ord("～"), ord("あ"), ord("①")}
    assert font.getlength("Aあ①～") == 8
    for text in ("B", "☃", "😀", "〜"):
        with pytest.raises(N.UnsupportedGlyphError):
            font.getlength(text)
        with pytest.raises(N.UnsupportedGlyphError):
            font.getmask2(text)
    with pytest.raises(N.UnsupportedGlyphError):
        N.parse_fnt(_fnt((0x41,))).glyph(0x42)


def test_distinct_cp932_byte_aliases_remain_parsed_but_use_one_canonical_glyph():
    # Both CP932 byte codes decode to Roman numeral one; Unicode encodes as 8754.
    glyphs = N.parse_fnt(_fnt((0x8754, 0xFA4A),
                              banks=[[[15] * 4, [0] * 4]] * 2))
    assert glyphs.codes == (0x8754, 0xFA4A)
    assert glyphs.glyph(0xFA4A).getbbox() is None
    metrics = N.parse_pro(_pro(N.metric_index(0x8754) + 1))
    font = N.NativeBitmapFont(glyphs, metrics, size=2)
    assert font.coverage == {ord("Ⅰ")}
    assert list(font.getmask("Ⅰ")) == [255] * 4
    with pytest.raises(N.UnsupportedGlyphError):
        font.getlength(bytes.fromhex("fa4a"))


def test_metrics_must_cover_present_glyphs_and_zero_advance_excludes_coverage():
    glyphs = N.parse_fnt(_fnt((0x82A0,)))
    with pytest.raises((ValueError, N.FontFormatError)):
        N.NativeBitmapFont(glyphs, N.parse_pro(_pro(539)), size=2)
    font = N.NativeBitmapFont(glyphs, N.parse_pro(_pro(540, {539: (0, 0, 0)})), size=2)
    assert ord("あ") not in font.coverage
    with pytest.raises(N.UnsupportedGlyphError):
        font.getlength("あ")
    with pytest.raises(N.UnsupportedGlyphError):
        font.getmask2("あ")
    with pytest.raises(TypeError):
        N.NativeBitmapFont(glyphs, None, size=2)


def test_fractional_size_scales_advances_and_font_variants_select_the_other_bank():
    banks = [[[15] * 4], [[0] * 4]]
    font = _font(size=3.5, records={65: (0, 0, 3)}, banks=banks)
    assert font.getlength("AA") == pytest.approx(10.5)
    larger = font.font_variant(size=7)
    assert larger.getlength("AA") == pytest.approx(21)
    assert larger.strike == 0
    assert larger.glyphs is font.glyphs and larger.metrics is font.metrics
    other = font.font_variant(strike=1)
    assert other.size == 3.5 and other.strike == 1
    assert not any(other.getmask("A"))
    assert any(font.getmask("A"))
    assert font.size == 3.5 and font.strike == 0
    same = font.font_variant()
    assert same.getlength("A") == font.getlength("A")


def test_signed_bearings_advance_and_horizontal_baseline_anchors():
    font = _font(records={65: (-1, 1, 3)})
    assert font.getlength("AA") == 6
    assert font.getbbox("AA", anchor="ls") == (-1, -1, 4, 1)
    assert font.getbbox("AA", anchor="ms") == (-4, -1, 1, 1)
    assert font.getbbox("AA", anchor="rs") == (-7, -1, -2, 1)
    mask, offset = font.getmask2("AA", mode="L", anchor="ls")
    assert offset == (-1, -1) and mask.size == (5, 2)
    assert list(mask) == [255, 255, 0, 255, 255] * 2
    assert list(font.getmask("AA", mode="L", anchor="ls")) == list(mask)


def test_imagedraw_places_bitmap_at_baseline_and_expands_stroke():
    font = _font(records={65: (-1, 1, 3)})
    image = Image.new("L", (24, 24))
    draw = ImageDraw.Draw(image)
    assert draw.textbbox((8, 8), "AA", font=font, anchor="ls") == (7, 7, 12, 9)
    draw.text((8, 8), "AA", font=font, anchor="ls", fill=255)
    assert image.getbbox() == (7, 7, 12, 9)
    assert list(image.crop((7, 7, 12, 9)).tobytes()) == [255, 255, 0, 255, 255] * 2
    outlined = Image.new("L", (24, 24))
    draw = ImageDraw.Draw(outlined)
    draw.text((8, 8), "AA", font=font, anchor="ls", fill=255,
              stroke_width=1, stroke_fill=128)
    assert outlined.getbbox() == (6, 6, 13, 10)
    assert font.getbbox("AA", anchor="ls", stroke_width=1) == (-2, -2, 5, 2)
    mask, offset = font.getmask2("AA", anchor="ls", stroke_width=1)
    assert offset == (-2, -2) and mask.size == (7, 4)


def test_default_top_anchor_and_empty_text_are_pillow_compatible():
    font = _font()
    assert font.getbbox("A") == (0, 0, 2, 2)
    assert font.getbbox("A", anchor="ls") == (0, -2, 2, 0)
    assert font.getlength("") == 0
    assert font.getbbox("") == (0, 0, 0, 0)
    mask, offset = font.getmask2("")
    assert mask.size == (0, 0) and offset == (0, 0)


def test_load_native_font_uses_sibling_pro_and_explicit_override(tmp_path):
    fnt_path = tmp_path / "generated.fnt"
    pro_path = fnt_path.with_suffix(".pro")
    fnt_path.write_bytes(_fnt((0x41,), banks=[[[15] * 4], [[0] * 4]]))
    pro_path.write_bytes(_pro(66, {65: (0, 0, 3)}))
    font = N.load_native_font(fnt_path, size=2, strike=1)
    assert font.getlength("A") == 3 and not any(font.getmask("A"))
    override = tmp_path / "different.pro"
    override.write_bytes(_pro(66, {65: (0, 0, 4)}))
    assert N.load_native_font(fnt_path, override, size=2).getlength("A") == 4


@pytest.mark.parametrize("size", [0, -1, math.nan, math.inf, -math.inf, 1025])
def test_bad_sizes_are_rejected_before_rendering(size):
    with pytest.raises((ValueError, TypeError)):
        _font(size=size)


@pytest.mark.parametrize("strike", [-1, 2, 100])
def test_invalid_strikes_are_rejected(strike):
    with pytest.raises((ValueError, TypeError)):
        _font(strike=strike)
    with pytest.raises((ValueError, TypeError)):
        N.parse_fnt(_fnt()).glyph(0x41, strike=strike)


@pytest.mark.parametrize("kwargs", [
    {"anchor": "xx"}, {"direction": "rtl"}, {"language": "ja"},
    {"features": ["kern"]}, {"stroke_width": -1}, {"stroke_width": math.inf},
    {"mode": "RGB"},
])
def test_unsupported_render_arguments_raise(kwargs):
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        _font().getmask2("A", **kwargs)


def test_text_and_mask_limits_prevent_large_allocations(monkeypatch):
    font = _font()
    large = font.font_variant(size=1024)

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("a rejected mask must not allocate a Pillow image")

    monkeypatch.setattr(Image, "new", unexpected_allocation)
    with pytest.raises(ValueError):
        font.getmask2("A" * 4097)
    with pytest.raises(ValueError):
        large.getmask2("A" * 32)
