"""Read the measured CLANNAD .fnt/.pro format without redistributing its bytes.

.fnt has a <HHI header, eight zero reserved bytes, sorted CP932 uint16 codes,
then two sequential banks of high-nibble-first 4bpp glyph cells. .pro has a
uint32 count, twelve zero reserved bytes and three bytes per JIS-grid entry.
The banks are named strike0/1; their purpose has not been established.

PRO values are exposed unchanged. The adapter *assumes* signed x/y bearings
and an unsigned advance, and defines the baseline as the bottom of the cell.
These layout assumptions are not evidence of the game's own layout algorithm.
Unsupported characters and non-round-tripping CP932 aliases raise explicitly.
"""
from __future__ import annotations

import bisect
import collections
import hashlib
import math
import struct
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageFont

MAX_FONT_BYTES = 64 * 1024 * 1024
MAX_DIMENSION = 128
MAX_GLYPHS = 16384
MAX_PRO_RECORDS = 256 + 94 * 94
MAX_SIZE = 1024
MAX_TEXT_LENGTH = 4096
MAX_MASK_PIXELS = 16 * 1024 * 1024
MAX_CACHE_PIXELS = 2 * 1024 * 1024


class FontFormatError(ValueError):
    """Malformed or unsupported binary font structure."""


class UnsupportedGlyphError(ValueError):
    """The requested character has no unambiguous supported glyph/metric."""


def _blob(data):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("font data must be bytes-like")
    if memoryview(data).nbytes > MAX_FONT_BYTES:
        raise FontFormatError("font data exceeds the size limit")
    return bytes(data)


def _code_bytes(code):
    if type(code) is not int or not 0 <= code <= 0xffff:
        raise UnsupportedGlyphError("CP932 code must be an unsigned 16-bit integer")
    raw = bytes([code]) if code <= 0xff else code.to_bytes(2, "big")
    try:
        char = raw.decode("cp932", errors="strict")
    except UnicodeDecodeError as exc:
        raise UnsupportedGlyphError(f"invalid CP932 code 0x{code:04x}") from exc
    if len(char) != 1:
        raise UnsupportedGlyphError("a CP932 code must encode exactly one character")
    return raw, char


def _char_code(char):
    if not isinstance(char, str) or len(char) != 1:
        raise UnsupportedGlyphError("a glyph request must be one Unicode character")
    try:
        raw = char.encode("cp932", errors="strict")
    except UnicodeEncodeError as exc:
        raise UnsupportedGlyphError(f"character U+{ord(char):04X} is not CP932 encodable") from exc
    if raw.decode("cp932", errors="strict") != char:
        raise UnsupportedGlyphError(f"ambiguous CP932 alias U+{ord(char):04X}")
    return int.from_bytes(raw, "big")


def metric_index(code):
    """Index single bytes directly; map double bytes onto the 94-column JIS grid.

    Direct byte-coordinate conversion also handles CP932's NEC row13 symbols,
    which Python's euc_jp Unicode codec does not encode. The resulting grid
    coordinates correspond to the measured EUC/JIS formula; no Unicode
    normalization or substitution is performed.
    """
    if isinstance(code, str):
        code = _char_code(code)
    raw, _ = _code_bytes(code)
    if len(raw) == 1:
        return code
    lead, trail = raw
    if not (0x81 <= lead <= 0x9f or 0xe0 <= lead <= 0xef):
        raise UnsupportedGlyphError("CP932 extension lies outside the supported JIS grid")
    row = (lead - (0x81 if lead <= 0x9f else 0xc1)) * 2 + 0x21
    if trail >= 0x9f:
        row, column = row + 1, trail - 0x7e
    else:
        column = trail - (0x1f if trail < 0x7f else 0x20)
    if not (0x21 <= row <= 0x7e and 0x21 <= column <= 0x7e):
        raise UnsupportedGlyphError("CP932 glyph lies outside the supported JIS grid")
    return 256 + (row - 0x21) * 94 + column - 0x21


@dataclass(frozen=True)
class GlyphBank:
    width: int
    height: int
    codes: tuple[int, ...]
    _payload: bytes
    sha256: str

    def glyph(self, code, strike=0):
        """Return a new grayscale cell; neither source bytes nor cached data change."""
        if type(strike) is not int or strike not in (0, 1):
            raise ValueError("strike must be 0 or 1")
        _code_bytes(code)
        index = bisect.bisect_left(self.codes, code)
        if index == len(self.codes) or self.codes[index] != code:
            raise UnsupportedGlyphError(f"missing CP932 glyph 0x{code:04x}")
        cell_bytes = self.width * self.height // 2
        offset = (strike * len(self.codes) + index) * cell_bytes
        raw = self._payload[offset:offset + cell_bytes]
        pixels = bytearray(self.width * self.height)
        for i, byte in enumerate(raw):
            pixels[2 * i], pixels[2 * i + 1] = (byte >> 4) * 17, (byte & 15) * 17
        return Image.frombytes("L", (self.width, self.height), bytes(pixels))


@dataclass(frozen=True)
class MetricTriplet:
    raw: tuple[int, int, int]

    @property
    def x_bearing_hint(self):
        return self.raw[0] if self.raw[0] < 128 else self.raw[0] - 256

    @property
    def y_bearing_hint(self):
        return self.raw[1] if self.raw[1] < 128 else self.raw[1] - 256

    @property
    def advance_hint(self):
        return self.raw[2]


@dataclass(frozen=True)
class ProTable:
    count: int
    _payload: bytes
    sha256: str

    def record(self, index):
        if type(index) is not int or not 0 <= index < self.count:
            raise UnsupportedGlyphError("PRO metric index is out of range")
        return MetricTriplet(tuple(self._payload[index * 3:index * 3 + 3]))


def parse_fnt(data):
    data = _blob(data)
    if len(data) < 16:
        raise FontFormatError("truncated FNT header")
    width, height, count = struct.unpack_from("<HHI", data)
    if (not 1 <= width <= MAX_DIMENSION or not 1 <= height <= MAX_DIMENSION
            or width * height % 2 or not 1 <= count <= MAX_GLYPHS):
        raise FontFormatError("invalid/oversize FNT dimensions or glyph count")
    if data[8:16] != bytes(8):
        raise FontFormatError("unsupported nonzero FNT reserved header bytes")
    expected = 16 + 2 * count + count * width * height
    if len(data) != expected:
        raise FontFormatError("FNT size does not exactly match two glyph banks")
    codes = struct.unpack_from(f"<{count}H", data, 16)
    if any(a >= b for a, b in zip(codes, codes[1:])):
        raise FontFormatError("FNT codes must be sorted and unique")
    try:
        for code in codes:
            _code_bytes(code)
    except UnsupportedGlyphError as exc:
        raise FontFormatError("FNT contains an invalid CP932 character code") from exc
    return GlyphBank(width, height, codes, data[16 + count * 2:], hashlib.sha256(data).hexdigest())


def parse_pro(data):
    data = _blob(data)
    if len(data) < 16:
        raise FontFormatError("truncated PRO header")
    count, = struct.unpack_from("<I", data)
    if not 1 <= count <= MAX_PRO_RECORDS:
        raise FontFormatError("invalid/oversize PRO record count")
    if data[4:16] != bytes(12):
        raise FontFormatError("unsupported nonzero PRO reserved header bytes")
    if len(data) != 16 + count * 3:
        raise FontFormatError("PRO size does not match its metric count")
    return ProTable(count, data[16:], hashlib.sha256(data).hexdigest())


class NativeBitmapFont(ImageFont.FreeTypeFont):
    """Pillow adapter with explicit metric assumptions and bounded mask allocation.

    FreeTypeFont is the interface base because current ImageDraw only applies
    getmask2's anchor offsets to that class. No FreeType parser is used. Horizontal
    anchors and strokes work; vertical/RTL shaping and OpenType features do not.
    Arbitrary positive sizes up to MAX_SIZE are supported; size32 retains source
    pixels. Variants share immutable parsed source bytes.
    """

    def __init__(self, glyphs, metrics, size=32, strike=0):
        if not isinstance(glyphs, GlyphBank) or not isinstance(metrics, ProTable):
            raise TypeError("parsed GlyphBank and ProTable are required")
        if isinstance(size, bool) or not isinstance(size, Real) or not math.isfinite(size) or not 0 < size <= MAX_SIZE:
            raise ValueError("font size must be finite, positive and within the size limit")
        if type(strike) is not int or strike not in (0, 1):
            raise ValueError("strike must be 0 or 1")
        self.glyphs, self.metrics, self.size, self.strike = glyphs, metrics, float(size), strike
        self.path = None
        self._scale = self.size / glyphs.height
        self._cell = (max(1, round(glyphs.width * self._scale)), max(1, round(self.size)))
        self._records, coverage = {}, set()
        for code in glyphs.codes:
            _, char = _code_bytes(code)
            if int.from_bytes(char.encode("cp932"), "big") != code:
                continue  # distinct byte aliases remain in the parser, not the text adapter
            try:
                record = metrics.record(metric_index(code))
            except UnsupportedGlyphError as exc:
                raise FontFormatError("glyph table exceeds the available PRO metric grid") from exc
            if record.advance_hint:
                self._records[code] = record
                coverage.add(ord(char))
        self._coverage = frozenset(coverage)
        self._cache = collections.OrderedDict()
        self._cache_pixels = 0
        self._descent = max((max(0, r.y_bearing_hint) for r in self._records.values()), default=0) * self._scale

    @property
    def coverage(self):
        """Unicode code points with canonical CP932 glyphs and positive metrics."""
        return set(self._coverage)

    def _text(self, text):
        if isinstance(text, (bytes, bytearray)):
            raw = bytes(text)
            try:
                text = raw.decode("cp932", errors="strict")
            except UnicodeDecodeError as exc:
                raise UnsupportedGlyphError("text is not valid CP932") from exc
            if text.encode("cp932") != raw:
                raise UnsupportedGlyphError("text contains ambiguous CP932 byte aliases")
        if not isinstance(text, str):
            raise TypeError("text must be Unicode or CP932 bytes")
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError("text exceeds the length limit")
        result = []
        for char in text:
            code = _char_code(char)
            if ord(char) not in self._coverage:
                raise UnsupportedGlyphError(f"unsupported glyph U+{ord(char):04X}")
            result.append((code, self._records[code]))
        return result

    @staticmethod
    def _options(mode="", direction=None, features=None, language=None):
        if mode not in ("", "L", "1") or direction not in (None, "ltr") or features or language:
            raise ValueError("only grayscale horizontal text without shaping features is supported")

    def getlength(self, text, mode="", direction=None, features=None, language=None, *args, **kwargs):
        self._options(mode, direction, features, language)
        return sum(record.advance_hint for _, record in self._text(text)) * self._scale

    def getmetrics(self):
        return math.ceil(self.size), math.ceil(self._descent)

    def getname(self):
        return "NativeBitmap", f"strike{self.strike}"

    def font_variant(self, *, size=None, strike=None, **kwargs):
        if kwargs:
            raise TypeError("only size and strike bitmap font variants are supported")
        result = type(self)(self.glyphs, self.metrics, self.size if size is None else size,
                            self.strike if strike is None else strike)
        result.path = self.path
        return result

    def _glyph(self, code):
        image = self._cache.get(code)
        if image is not None:
            self._cache.move_to_end(code)
            return image
        image = self.glyphs.glyph(code, self.strike)
        if image.size != self._cell:
            image = image.resize(self._cell, Image.Resampling.LANCZOS)
        pixels = image.width * image.height
        while self._cache and (len(self._cache) >= 128 or self._cache_pixels + pixels > MAX_CACHE_PIXELS):
            _, old = self._cache.popitem(last=False)
            self._cache_pixels -= old.width * old.height
        self._cache[code] = image
        self._cache_pixels += pixels
        return image

    def _layout(self, text, anchor, stroke_width):
        if (isinstance(stroke_width, bool) or not isinstance(stroke_width, Real)
                or not math.isfinite(stroke_width) or not 0 <= stroke_width <= MAX_SIZE):
            raise ValueError("stroke width must be finite, nonnegative and bounded")
        anchor = "la" if anchor is None else anchor
        if not isinstance(anchor, str) or len(anchor) != 2 or anchor[0] not in "lmr" or anchor[1] not in "atsmbd":
            raise ValueError("unsupported horizontal text anchor")
        entries = self._text(text)
        advance = sum(r.advance_hint for _, r in entries) * self._scale
        # Check a conservative mask budget before decoding a potentially long string.
        xs = [r.x_bearing_hint for _, r in entries] or [0]
        ys = [r.y_bearing_hint for _, r in entries] or [0]
        extent = advance + self._cell[0] + (max(xs) - min(xs)) * self._scale + 2 * math.ceil(stroke_width)
        height = self._cell[1] + (max(ys) - min(ys)) * self._scale + 2 * math.ceil(stroke_width)
        if entries and extent * height > MAX_MASK_PIXELS:
            raise ValueError("text mask exceeds the pixel allocation limit")
        items, boxes, pen = [], [], 0.0
        for code, record in entries:
            image = self._glyph(code)
            x, y = round(pen + record.x_bearing_hint * self._scale), round(record.y_bearing_hint * self._scale)
            ink = image.getbbox()
            if ink:
                boxes.append((x + ink[0], y + ink[1], x + ink[2], y + ink[3]))
                items.append((image, x, y))
            pen += record.advance_hint * self._scale
        if not boxes:
            return (0, 0, math.ceil(advance), 0), items
        left, top = min(b[0] for b in boxes), min(b[1] for b in boxes)
        right, bottom = max(b[2] for b in boxes), max(b[3] for b in boxes)
        ax = {"l": 0, "m": advance / 2, "r": advance}[anchor[0]]
        ay = {"a": 0, "t": top, "s": self.size, "m": (self.size + self._descent) / 2,
              "b": bottom, "d": self.size + self._descent}[anchor[1]]
        stroke = math.ceil(stroke_width)
        ax, ay = round(ax), round(ay)
        box = (left - ax - stroke, top - ay - stroke, right - ax + stroke, bottom - ay + stroke)
        # Bitmap anchors round to the nearest pixel; every item uses the same offset.
        positioned = [(image, x - ax, y - ay) for image, x, y in items]
        return box, positioned

    def getbbox(self, text, mode="", direction=None, features=None, language=None, stroke_width=0, anchor=None):
        self._options(mode, direction, features, language)
        return self._layout(text, anchor, stroke_width)[0]

    def getmask2(self, text, mode="", direction=None, features=None, language=None, stroke_width=0,
                 anchor=None, ink=0, start=None, *args, **kwargs):
        self._options(mode, direction, features, language)
        if args or set(kwargs) - {"stroke_filled"}:
            raise TypeError("unsupported bitmap mask arguments")
        if start is None:
            start = (0, 0)
        if (not isinstance(start, (tuple, list)) or len(start) != 2
                or any(not isinstance(v, Real) or not math.isfinite(v) or not -1 < v < 1 for v in start)):
            raise ValueError("fractional text start must contain two finite offsets within one pixel")
        box, items = self._layout(text, anchor, stroke_width)
        left, top, right, bottom = box
        width, height = right - left, bottom - top
        if width * height > MAX_MASK_PIXELS:
            raise ValueError("text mask exceeds the pixel allocation limit")
        image = Image.new("L", (width, height), 0)
        for glyph, x, y in items:
            gx, gy = x - left, y - top
            area = (max(0, gx), max(0, gy), min(width, gx + glyph.width), min(height, gy + glyph.height))
            if area[0] < area[2] and area[1] < area[3]:
                source = glyph.crop((area[0] - gx, area[1] - gy, area[2] - gx, area[3] - gy))
                image.paste(ImageChops.lighter(image.crop(area), source), area)
        if stroke_width and width and height:
            image = image.filter(ImageFilter.MaxFilter(2 * math.ceil(stroke_width) + 1))
        if mode == "1":
            image = image.point(lambda value: 255 if value >= 128 else 0, mode="1")
        return image.im, (left + round(start[0]), top + round(start[1]))

    def getmask(self, text, mode="", *args, **kwargs):
        return self.getmask2(text, mode, *args, **kwargs)[0]


def _read_bounded(path):
    with Path(path).open("rb") as source:
        data = source.read(MAX_FONT_BYTES + 1)
    if len(data) > MAX_FONT_BYTES:
        raise FontFormatError("font file exceeds the size limit")
    return data


def load_native_font(fnt_path, pro_path=None, size=32, strike=0):
    """Read private local files; default metrics path is the sibling .pro file."""
    fnt_path = Path(fnt_path)
    pro_path = Path(pro_path) if pro_path is not None else fnt_path.with_suffix(".pro")
    font = NativeBitmapFont(parse_fnt(_read_bounded(fnt_path)), parse_pro(_read_bounded(pro_path)), size, strike)
    font.path = str(fnt_path)
    return font
