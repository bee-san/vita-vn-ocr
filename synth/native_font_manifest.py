"""Bind optional, privately owned bitmap fonts to immutable source identities.

This loader never downloads font files. The manifest records source URLs and
hashes; the caller supplies a local directory containing their own game assets.
"""
import hashlib
import json
import math
from pathlib import Path

from native_bitmap_font import MAX_FONT_BYTES, load_native_font


def checked_path(root, record, suffix):
    relative = Path(record["file"])
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != suffix:
        raise ValueError("native font must use a relative path with the expected suffix")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("native font path escapes its local source directory")
    size = record["bytes"]
    if type(size) is not int or not 0 < size <= MAX_FONT_BYTES or path.stat().st_size != size:
        raise ValueError("native font source size mismatch")
    digest = record["sha256"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("native font requires a lowercase SHA256 identity")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("native font source SHA256 mismatch")
    return path


def load_manifest(manifest_path, root):
    root = Path(root).resolve()
    raw = Path(manifest_path).read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema") != 1 or manifest.get("redistribute") is not False:
        raise ValueError("native font manifest must explicitly prohibit redistribution")
    records = manifest.get("fonts")
    if not isinstance(records, list) or not records or len(records) > 32:
        raise ValueError("native font manifest requires a bounded nonempty font list")
    faces, rows, identities = {}, [], []
    names = set()
    for record in records:
        name, category = record["family"], record["category"]
        if (not isinstance(name, str) or not name or name in names
                or category not in ("gothic", "rounded", "mincho", "pen", "pixel", "display")):
            raise ValueError("native font families must be unique and have a known category")
        names.add(name)
        path = checked_path(root, record["fnt"], ".fnt")
        pro = checked_path(root, record["pro"], ".pro")
        if path in faces:
            raise ValueError("native font source is repeated")
        font = load_native_font(path, pro)
        # Check the bytes actually parsed, in addition to the path verification.
        if font.glyphs.sha256 != record["fnt"]["sha256"] or font.metrics.sha256 != record["pro"]["sha256"]:
            raise ValueError("native font changed while being loaded")
        coverage = {chr(code) for code in font.coverage}
        if "　" not in coverage:
            raise ValueError("native renderer fonts require a full-width space advance")
        faces[path] = font
        ambiguities = font.ambiguous_glyph_groups()
        training_coverage = coverage - set("".join(ambiguities))
        rows.append((category, name, path, training_coverage))
        identities.append({"family": name, "category": category,
                           "fnt": record["fnt"], "pro": record["pro"],
                           "canonical_unicode_coverage": len(coverage),
                           "unambiguous_training_coverage": len(training_coverage),
                           "visually_ambiguous_groups": list(ambiguities),
                           "source_cell": [font.glyphs.width, font.glyphs.height]})
    binding = {"manifest_sha256": hashlib.sha256(raw).hexdigest(), "fonts": identities,
               "redistribute": False, "sampling_unit": "eligible rendered region",
               "strike_names": ["strike0", "strike1"],
               "metrics_status": "layout bearings/advances are inferred; see native_bitmap_font.py"}
    return faces, rows, binding


def probability(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("native font probability must be finite and between 0 and 1")
    return value
