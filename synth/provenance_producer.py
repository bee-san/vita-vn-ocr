#!/usr/bin/env python3
"""Opt-in synthetic crops with pinned, per-crop font/text/background ownership.

This wrapper does not download assets or change render.py's default behavior.
Its registry is input to the consumption policy, not an exposure-cap guarantee.
"""
import argparse
import collections
import contextlib
import hashlib
import json
import math
import multiprocessing as mp
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import fontTools
import numpy as np
import PIL
from fontTools.ttLib import TTFont
from PIL import Image, features

import render
from kanji_policy import allowed_label, is_han, load_policy, name_weights
from native_bitmap_font import MAX_FONT_BYTES, load_native_font

ROOT = Path(__file__).resolve().parents[1]
SHA = re.compile(r"[0-9a-f]{64}\Z")
VN = re.compile(r"v[1-9][0-9]*\Z")
RELEASE = re.compile(r"r[1-9][0-9]*\Z")
MAX_TEXT_BYTES = 64 * 1024 * 1024
STATE = None


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def identifier(value):
    return isinstance(value, str) and 0 < len(value) <= 256 and not any(ord(c) < 32 for c in value)


def parse_manifest(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate producer identity/role field")
            result[key] = value
        return result
    result = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(result, dict):
        raise ValueError("producer manifest must be an object")
    return result


def checked_file(root, record, maximum=MAX_TEXT_BYTES):
    """Validate role/ownership separately, then pin bounded local bytes before decoding."""
    if not isinstance(record, dict):
        raise ValueError("producer asset file pin must be an object")
    relative = Path(record.get("file", ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("producer asset requires a contained relative file")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("producer asset path escapes its local root")
    size, sha = record.get("bytes"), record.get("sha256")
    if type(size) is not int or not 0 < size <= maximum or path.stat().st_size != size:
        raise ValueError("producer asset size mismatch")
    if not isinstance(sha, str) or not SHA.fullmatch(sha):
        raise ValueError("producer asset needs a lowercase SHA256")
    raw = path.read_bytes()
    if digest(raw) != sha:
        raise ValueError("producer asset SHA256 mismatch")
    return path, raw


def ownership(row):
    kind, origin = row.get("kind"), row.get("origin")
    if kind not in ("font", "text", "background") or origin not in ("game", "ofl", "non_game", "unknown"):
        raise ValueError("producer input needs an explicit supported kind and origin")
    result = {key: row.get(key) for key in ("kind", "origin", "sha256")}
    if not isinstance(result["sha256"], str) or not SHA.fullmatch(result["sha256"]):
        raise ValueError("producer registry requires SHA256 identities")
    if origin == "game":
        if not VN.fullmatch(str(row.get("vn", ""))) or not RELEASE.fullmatch(str(row.get("release", ""))):
            raise ValueError("game assets require canonical VNDB VN and release IDs")
        result.update(vn=row["vn"], release=row["release"])
    elif row.get("vn") is not None or row.get("release") is not None:
        raise ValueError("non-game/unknown assets must not carry a conflicting game owner")
    if origin == "ofl":
        if kind != "font" or row.get("license") != "OFL-1.1":
            raise ValueError("OFL origin requires an OFL-1.1 font")
        result["license"] = "OFL-1.1"
    if origin == "non_game":
        if kind != "text" or row.get("game_free_reviewed") is not True:
            raise ValueError("non-game text needs explicit game-free review")
        result["game_free_reviewed"] = True
    return result


class SourceText(str):
    """Carry all contributing source chunks through the renderer's string operations."""
    def __new__(cls, value, asset_ids=()):
        obj = super().__new__(cls, value)
        obj.asset_ids = tuple(sorted(set(asset_ids)))
        return obj

    def _with(self, value, other=None):
        refs = self.asset_ids + getattr(other, "asset_ids", ())
        return SourceText(value, refs)

    def __getitem__(self, key):
        return self._with(super().__getitem__(key))

    def __add__(self, other):
        return self._with(super().__add__(other), other)

    def __radd__(self, other):
        return self._with(str(other) + str(self), other)

    def replace(self, old, new, count=-1):
        return self._with(super().replace(old, new, count), new)

    def strip(self, chars=None):
        return self._with(super().strip(chars))

    def lstrip(self, chars=None):
        return self._with(super().lstrip(chars))

    def rstrip(self, chars=None):
        return self._with(super().rstrip(chars))


class SourceBackground(np.ndarray):
    def __new__(cls, pixels, asset_id):
        obj = np.asarray(pixels).view(cls)
        obj.asset_id = asset_id
        return obj

    def __array_finalize__(self, original):
        self.asset_id = getattr(original, "asset_id", None)


class Producer:
    def __init__(self, manifest, asset_root, vocab, *, split, source, seed, shards, per_shard,
                 finder="grad", invert=True, edge_jitter=True, broad=False, target_kana="",
                 correction_pattern_id=None, kanji_policy=None, kanji_pairs=False, native_probability=0):
        if split not in ("train", "dev"):
            raise ValueError("producer split must be train or dev")
        if not identifier(source) or not re.fullmatch(r"[A-Za-z0-9_-]+", source):
            raise ValueError("producer source needs a stable simple name")
        if type(shards) is not int or shards < 1 or type(per_shard) is not int or per_shard < 1:
            raise ValueError("producer needs positive shard and crop counts")
        if type(seed) is not int or seed < 0 or finder not in ("mask", "grad", "legacy"):
            raise ValueError("invalid producer seed or finder")
        native_probability = render.probability(native_probability)
        if not isinstance(target_kana, str):
            raise ValueError("kana targets must be an explicit character set")
        target_kana = "".join(sorted(set(target_kana)))
        canonical_pattern = "kana:" + target_kana if target_kana else None
        if correction_pattern_id is not None and correction_pattern_id != canonical_pattern:
            raise ValueError("kana correction pattern ID must match its canonical target set")
        correction_pattern_id = canonical_pattern
        if broad and (target_kana or correction_pattern_id or kanji_pairs or native_probability):
            raise ValueError("broad constructor disables native fonts and deliberate correction")
        if target_kana and kanji_pairs:
            raise ValueError("kana and pair targeting are separate")
        if correction_pattern_id is not None and (not identifier(correction_pattern_id)
                                                 or not correction_pattern_id.strip()):
            raise ValueError("correction pattern ID must be a nonempty stable string")
        if split == "dev" and (target_kana or kanji_pairs):
            raise ValueError("development producer cannot deliberately target corrections")
        self.manifest = Path(manifest).resolve()
        self.asset_root = Path(asset_root).resolve()
        raw = self.manifest.read_bytes()
        doc = parse_manifest(raw)
        if type(doc.get("schema")) is not int or doc["schema"] != 1 or doc.get("kind") != "synthetic-producer-inputs":
            raise ValueError("producer requires a schema-1 synthetic-producer-inputs manifest")
        inputs = doc.get("assets")
        if not isinstance(inputs, dict) or not inputs:
            raise ValueError("producer requires an explicit asset registry")
        self.registry, self.by_sha = {}, {}
        self.file_pins = {}
        self.source, self.split, self.broad = source, split, broad
        self.pattern_id, self.font_refs, self.background_refs = correction_pattern_id, {}, {}
        self.current_font, self.current_background, self.glyphs = (), None, []
        self.shard_index, self.crop_index = None, 0
        vocab_path = Path(vocab).resolve()
        vocab_values = render.load_vocab(vocab_path)
        if (vocab_values != vocab_path.read_text(encoding="utf-8").splitlines() or not vocab_values
                or len(set(vocab_values)) != len(vocab_values) or any(len(c) != 1 for c in vocab_values)):
            raise ValueError("producer vocabulary needs unique single-character classes in original order")
        policy = load_policy(kanji_policy) if kanji_policy else None
        vocab_sha = digest(vocab_path.read_bytes())
        if policy and policy["vocab_sha256"] != vocab_sha:
            raise ValueError("producer kanji policy and vocabulary differ")
        if kanji_pairs and not policy:
            raise ValueError("corrective kanji producer requires a pinned policy")
        if kanji_pairs and per_shard % 2:
            raise ValueError("paired kanji producer needs an even per-shard crop count")
        self.recipe = {"schema": 1, "input_manifest_sha256": digest(raw), "vocab_sha256": vocab_sha,
                       "input_asset_sha256": {key: {name: row.get(name) for name in ("sha256", "split")}
                                              for key, row in inputs.items() if isinstance(row, dict)},
                       "kanji_policy_sha256": digest(Path(kanji_policy).read_bytes()) if policy else None,
                       "split": split, "source": source, "seed": seed, "shards": shards,
                       "per_shard": per_shard, "finder": finder, "invert_dark": bool(invert),
                       "crop_edge_jitter": bool(edge_jitter), "broad": bool(broad), "target_kana": target_kana,
                       "correction_pattern_id": correction_pattern_id, "kanji_pairs": bool(kanji_pairs),
                       "native_probability": native_probability,
                       "code_sha256": {name: digest((ROOT / name).read_bytes()) for name in (
                           "synth/provenance_producer.py", "synth/render.py", "synth/common.py",
                           "synth/native_bitmap_font.py", "synth/native_font_manifest.py", "synth/kanji_policy.py",
                           "synth/bg_exclude.py", "synth/mask_finder.py")},
                       "dependency_sha256": digest((ROOT / "requirements-ci.lock").read_bytes()),
                       "exclusions_sha256": {name: digest((ROOT / name).read_bytes())
                                             if (ROOT / name).is_file() else None for name in (
                                                 "data/manifests/synth2_bg_exclude.txt",
                                                 "eval/manifests/test_games.json")},
                       "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                                       "pillow": PIL.__version__, "opencv": cv2.__version__,
                                       "fonttools": fontTools.__version__, "freetype": features.version("freetype2"),
                                       "opencv_build_sha256": digest(cv2.getBuildInformation().encode())}}
        self.recipe_sha = digest(canonical(self.recipe))
        self.procedural_text = self.register("procedural-text:" + self.recipe_sha,
            {"kind": "text", "origin": "procedural", "sha256": digest(canonical({
                "kind": "text", "recipe": self.recipe_sha, "algorithm": "renderer dressing/names/ruby"})),
             "game_free_reviewed": True, "recipe_sha256": self.recipe_sha,
             "generator_sha256": self.recipe["code_sha256"]["synth/render.py"],
             "review": "Names use reviewed non-game frequency inputs or Jiten/vocabulary priors only."})
        self.procedural_background = self.register("procedural-background:" + self.recipe_sha,
            {"kind": "background", "origin": "procedural", "sha256": digest(canonical({
                "kind": "background", "recipe": self.recipe_sha, "algorithm": "render.background"})),
             "recipe_sha256": self.recipe_sha,
             "generator_sha256": self.recipe["code_sha256"]["synth/render.py"]})
        r = SimpleNamespace(vocab=vocab_values, vocab_set=set(vocab_values), kanji_policy=policy,
                            policy_sha256=self.recipe["kanji_policy_sha256"], kanji_pairs=kanji_pairs,
                            crop_edge_jitter=edge_jitter, active_kanji=None, target_kana=set(target_kana),
                            pools={}, target_pools={}, fonts=[], by_cat={}, native_faces={}, native_fonts=[],
                            native_font_probability=native_probability, native_font_binding=None,
                            bg_vita=[], bg_psp=[], bg_boxes={}, kanji_contexts={}, pair_bands={})
        declared_forbidden = doc.get("forbidden_vns", [])
        if not isinstance(declared_forbidden, list) or any(not isinstance(v, str) or not VN.fullmatch(v)
                                                        for v in declared_forbidden):
            raise ValueError("producer forbidden VNs must be canonical VNDB IDs")
        forbidden = set(render.load_exclusions()) | set(declared_forbidden)
        self.forbidden_vns = sorted(v for v in forbidden if VN.fullmatch(str(v)))
        texts, name_frequency, roles = [], collections.Counter(), {}
        prepared_inputs, declared_owners = [], {}
        # Inspect the entire role/ownership inventory before opening eligible assets.
        # A later train/dev alias must not be discovered after the shared bytes were read.
        for asset_id, row in inputs.items():
            if not identifier(asset_id) or not isinstance(row, dict):
                raise ValueError("producer asset IDs/records must be nonempty strings/objects")
            owner = ownership(row)
            if broad and owner["kind"] == "font" and owner["origin"] == "ofl" and row.get(
                    "format", "truetype") != "truetype":
                raise ValueError("broad constructor cannot load a native bitmap font")
            declared = [owner]
            if row.get("format") == "native_bitmap":
                metrics = row.get("metrics")
                if not isinstance(metrics, dict) or not SHA.fullmatch(str(metrics.get("sha256", ""))):
                    raise ValueError("native producer metrics need a lowercase SHA256")
                declared.append({**owner, "sha256": metrics["sha256"]})
            for identity in declared:
                sha = identity["sha256"]
                fields = tuple(identity.get(key) for key in
                               ("kind", "origin", "vn", "release", "license", "game_free_reviewed"))
                if sha in declared_owners and declared_owners[sha] != fields:
                    raise ValueError("same declared asset SHA256 has conflicting ownership/kind")
                declared_owners[sha] = fields
            if row.get("split") is not None and row["split"] not in ("train", "dev"):
                raise ValueError("producer input has an invalid split")
            if owner["kind"] in ("text", "background") and row.get("split") not in ("train", "dev"):
                raise ValueError("text/background assets require an explicit split")
            if owner["kind"] in ("text", "background"):
                sha, role = owner["sha256"], row["split"]
                if sha in roles and roles[sha] != role:
                    raise ValueError("identical text/background bytes cannot appear in both train and dev")
                roles[sha] = role
            prepared_inputs.append((asset_id, row, owner))
        for asset_id, row, owner in prepared_inputs:
            if row.get("split", split) != split:
                continue  # other roles are never decoded or read as rendering inputs
            if broad and not (owner["kind"] == "font" and owner["origin"] == "ofl"
                              or owner["kind"] == "text" and owner["origin"] == "non_game"):
                continue  # broad resources cannot access game/unknown/native/screenshot bytes
            if broad and owner["kind"] == "font" and row.get("format", "truetype") != "truetype":
                raise ValueError("broad constructor cannot load a native bitmap font")
            if owner["origin"] == "game" and (owner["vn"] in forbidden or render.is_reserved_vn(owner["vn"])):
                raise ValueError("reserved test VN is not a producer asset")
            path, content = self.checked_input(row)
            if owner["kind"] == "font":
                self.load_font(r, asset_id, row, owner, path)
            elif owner["kind"] == "text":
                if row.get("pool") not in render.POOL_WEIGHT:
                    raise ValueError("text source requires a renderer pool name")
                chunks = content.decode("utf-8").splitlines()
                if not chunks or any(not chunk.strip() or not set(chunk) <= r.vocab_set
                                     or policy and not allowed_label(chunk, policy) for chunk in chunks):
                    raise ValueError("producer text chunks must be nonempty and vocabulary/policy bound")
                if policy is None and any(is_han(c) for chunk in chunks for c in chunk):
                    raise ValueError("producer Han text requires a pinned Jiten kanji policy")
                texts.append((asset_id, row, owner, chunks))
                if owner["origin"] == "non_game":
                    name_frequency.update("".join(chunks))
            else:
                self.load_background(r, asset_id, row, owner, path)
        if not r.fonts or not any(self.registry[refs[0]]["origin"] == "ofl" for refs in self.font_refs.values()):
            raise ValueError("producer requires a covered OFL fallback font")
        for asset_id, row, owner, chunks in texts:
            for index, text in enumerate(chunks):
                chunk_sha = digest(text.encode())
                identity = digest(canonical({"source_sha256": row["sha256"], "line": index,
                                             "chunk_sha256": chunk_sha}))
                ref = self.register(f"text-chunk:{identity}", {**owner, "sha256": identity,
                    "source_asset_id": asset_id, "source_sha256": row["sha256"],
                    "chunk_sha256": chunk_sha, "chunk_index": index, "split": split})
                r.pools.setdefault(row["pool"], []).append(SourceText(text, (ref, self.procedural_text)))
        if not r.pools:
            raise ValueError("producer has no eligible text pool")
        if not {"dialogue", "narration"} <= r.pools.keys():
            raise ValueError("producer needs both dialogue and narration pools for choice layouts")
        for cat in render.CATEGORY:
            faces = [font for font in r.fonts if font[0] == cat]
            if faces:
                r.by_cat[cat] = faces
        r.katakana = [c for c in r.vocab if 0x30A1 <= ord(c) <= 0x30FA and c not in "ヵヶヮ"]
        r.hiragana = [c for c in r.vocab if 0x3041 <= ord(c) <= 0x3093 and c not in "ゎゐゑ"]
        if not r.hiragana or not r.katakana:
            raise ValueError("producer vocabulary needs hiragana and katakana for generated names")
        r.name_kanji = [c for c, _ in name_frequency.most_common() if is_han(c)
                        and (not policy or policy["ranks"].get(c, math.inf) <= 2000)][:1200]
        if not r.name_kanji and policy:
            r.name_kanji = [c for c in policy["allowed_kanji"] if policy["ranks"][c] <= 2000][:1200]
        if r.name_kanji:
            counts = np.array([name_frequency[c] for c in r.name_kanji], dtype=np.float64)
            r.name_probs = counts / counts.sum() if counts.sum() else name_weights(r.name_kanji, policy)
        else:
            r.name_kanji = r.hiragana  # kana-only fixture/generated names, never unknown corpus priors
            r.name_probs = np.full(len(r.name_kanji), 1 / len(r.name_kanji))
        for character in sorted(r.target_kana):
            if not "ぁ" <= character <= "ゖ" or character not in r.vocab_set:
                raise ValueError("producer target must be model-supported hiragana")
            pools = {name: [s for s in items if character in s] for name, items in r.pools.items()}
            r.target_pools[character] = {name: items for name, items in pools.items() if items}
            if not r.target_pools[character]:
                raise ValueError("producer target has no natural source context")
        if kanji_pairs:
            connected = set("".join(policy["pairs"]))
            for pool, chunks in r.pools.items():
                for text in chunks:
                    for character in set(text) & connected:
                        r.kanji_contexts.setdefault(character, {}).setdefault(pool, []).append(text)
            for pair in policy["pairs"]:
                if all(c in r.kanji_contexts for c in pair):
                    rank = max(policy["ranks"][c] for c in pair)
                    band = 0 if rank <= 2000 else 1 if rank <= 3000 else 2 if rank <= 4000 else 3
                    r.pair_bands.setdefault(band, []).append(pair)
            if not r.pair_bands.get(0):
                raise ValueError("producer needs common similar-kanji pairs with natural contexts")
        self.resources = r

    def checked_input(self, row, maximum=MAX_TEXT_BYTES):
        path, content = checked_file(self.asset_root, row, maximum)
        self.file_pins[path] = (row["bytes"], row["sha256"])
        return path, content

    def verify_inputs(self):
        """Catch changed lazy-loaded fonts/backgrounds before or during a fresh render."""
        for path, (size, sha) in self.file_pins.items():
            if path.stat().st_size != size or digest(path.read_bytes()) != sha:
                raise ValueError("producer rendering input changed after its SHA256 was bound")

    def register(self, asset_id, row):
        """One immutable identity; equal bytes cannot receive conflicting owner labels."""
        fields = ("kind", "origin", "vn", "release", "license", "game_free_reviewed")
        previous_id = self.by_sha.get(row["sha256"])
        if previous_id is not None:
            previous = self.registry[previous_id]
            if any(previous.get(key) != row.get(key) for key in fields):
                raise ValueError("same asset SHA256 has conflicting ownership/kind")
            return previous_id
        if asset_id in self.registry:
            raise ValueError("producer registry ID was reused for different asset bytes")
        self.registry[asset_id] = dict(row)
        self.by_sha[row["sha256"]] = asset_id
        return asset_id

    def load_font(self, r, asset_id, row, owner, path):
        if (row.get("category") not in render.CATEGORY or not isinstance(row.get("family"), str)
                or not row["family"].strip()):
            raise ValueError("producer font requires a family and known category")
        license_pin = {}
        if owner["origin"] == "ofl":
            _, license_bytes = self.checked_input(row.get("license_file", {}), 1024 * 1024)
            license_text = license_bytes.decode("utf-8").upper()
            if "SIL OPEN FONT LICENSE" not in license_text or "1.1" not in license_text:
                raise ValueError("producer OFL font needs pinned OFL-1.1 licence text")
            license_pin = {"license_sha256": row["license_file"]["sha256"],
                           "license_bytes": row["license_file"]["bytes"]}
        ref = self.register(asset_id, {**owner, "family": row["family"], "category": row["category"],
                                      "bytes": row["bytes"], **license_pin})
        if row.get("format", "truetype") == "native_bitmap":
            if owner["origin"] == "ofl":
                raise ValueError("private native bitmap assets cannot masquerade as OFL fallback fonts")
            pro, _ = self.checked_input(row.get("metrics", {}), MAX_FONT_BYTES)
            font = load_native_font(path, pro)
            if font.glyphs.sha256 != row["sha256"] or font.metrics.sha256 != row["metrics"]["sha256"]:
                raise ValueError("native producer source changed while parsing")
            if path in r.native_faces:
                raise ValueError("native font source path was repeated")
            r.native_faces[path] = font
            coverage = {chr(c) for c in font.coverage} - set("".join(font.ambiguous_glyph_groups()))
            metric_ref = self.register(asset_id + ":metrics", {**owner, "sha256": row["metrics"]["sha256"],
                                                               "component": "metrics"})
            refs = (ref, metric_ref)
            r.native_fonts.append((row["category"], row["family"], path, coverage))
            r.native_font_binding = {"manifest_sha256": self.recipe["input_manifest_sha256"]}
        elif row.get("format", "truetype") == "truetype" and path.suffix.lower() in (".ttf", ".otf"):
            with TTFont(str(path), lazy=True) as font:
                coverage = {chr(c) for c in (font.getBestCmap() or {})}
            refs = (ref,)
            r.fonts.append((row["category"], row["family"], path, coverage))
        else:
            raise ValueError("unsupported producer font format")
        if "　" not in coverage:
            raise ValueError("producer fonts need a full-width spacing glyph")
        self.font_refs[path] = refs

    def load_background(self, r, asset_id, row, owner, path):
        platform = row.get("platform")
        native = (480, 272) if platform == "psp" else (960, 544) if platform == "psv" else None
        box = row.get("scenery_box")
        if (native is None or row.get("scenery_reviewed") is not True or not isinstance(box, list)
                or len(box) != 4 or any(type(v) is not int for v in box)):
            raise ValueError("background needs reviewed native-platform scenery geometry")
        with Image.open(path) as image:
            if image.size != native:
                raise ValueError("producer background has non-native dimensions")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= native[0] and 0 <= y0 < y1 <= native[1]):
            raise ValueError("producer scenery patch escapes screenshot")
        ref = self.register(asset_id, {**owner, "split": self.split, "platform": platform,
                                       "scenery_box": box, "bytes": row["bytes"]})
        self.background_refs[path] = ref
        (r.bg_psp if platform == "psp" else r.bg_vita).append(path)
        r.bg_boxes[path] = tuple(box)

    def validate_references(self, refs):
        if not refs or any(ref not in self.registry for ref in refs):
            raise ValueError("crop has unbound producer asset references")
        rows = [self.registry[ref] for ref in refs]
        if not {"font", "text", "background"} <= {row["kind"] for row in rows}:
            raise ValueError("crop must identify its font, text and background")
        if self.broad and any(not (row["kind"] == "font" and row["origin"] == "ofl"
                                  or row["kind"] == "text" and row["origin"] in ("non_game", "procedural")
                                  and row.get("game_free_reviewed") is True
                                  or row["kind"] == "background" and row["origin"] == "procedural")
                              for row in rows):
            raise ValueError("broad crop used a game, unknown or unreviewed asset")


def core_box(font, character, x, base, ss, mode):
    px, py = x * ss, base * ss
    mask, offset = font.getmask2(character, mode=mode, anchor="ls",
                               start=(math.modf(px)[0], math.modf(py)[0]))
    if not all(mask.size):
        return None
    values = np.frombuffer(bytes(mask), np.uint8).reshape(mask.size[1], mask.size[0])
    ys, xs = np.where(values >= 128)
    if not len(xs):
        return None
    ox, oy = int(px) + offset[0], int(py) + offset[1]
    return (math.floor((ox + xs.min()) / ss), math.floor((oy + ys.min()) / ss),
            math.ceil((ox + xs.max() + 1) / ss), math.ceil((oy + ys.max() + 1) / ss))


@contextlib.contextmanager
def installed(producer):
    """All tracing and crop-safety hooks are scoped to this opt-in wrapper."""
    global STATE
    names = ("clean", "add_spaces", "random_name", "pick_font", "load_bg", "background", "render_region",
             "extract", "crop_meta")
    original = {name: getattr(render, name) for name in names}
    old_r, old_state, old_glyph = render.R, STATE, render.Layer.glyph
    old_caches = render._FONTS, render._AXES, render._BG
    STATE, render.R = producer, producer.resources
    render._FONTS, render._AXES, render._BG = {}, {}, {}

    def clean(text):
        return SourceText(original["clean"](text), getattr(text, "asset_ids", ()))

    def spaces(text, rng):
        return SourceText(original["add_spaces"](text, rng), getattr(text, "asset_ids", ()))

    def name(rng):
        display, label = original["random_name"](rng)
        return SourceText(display, (producer.procedural_text,)), SourceText(label, (producer.procedural_text,))

    def font(*args, **kwargs):
        result = original["pick_font"](*args, **kwargs)
        if result is not None:
            producer.current_font = producer.font_refs[result[2]]
            producer.current_font_path = result[2]
        return result

    def load_bg(path, *args, **kwargs):
        return SourceBackground(original["load_bg"](path, *args, **kwargs), producer.background_refs[path])

    def background(*args, **kwargs):
        pixels = original["background"](*args, **kwargs)
        producer.current_background = getattr(pixels, "asset_id", None) or producer.procedural_background
        return np.asarray(pixels)

    def glyph(layer, x, base, character, face, k=0, small=False):
        old_glyph(layer, x, base, character, face, k, small)
        if not small:
            box = core_box(face, character, x, base, layer.ss, layer.d[k].fontmode)
            if box:
                producer.glyphs.append((base, box))

    def region(*args, **kwargs):
        producer.current_font, producer.current_background, producer.glyphs = (), None, []
        result = original["render_region"](*args, **kwargs)
        if result:
            _, layout, _, meta = result
            scale = 2 if meta["platform"] == "psp2x" else 1
            for line in layout:
                baseline = render.text_baseline(line["top"], line["fs"], producer.current_font_path)
                core = None
                for base, box in producer.glyphs:
                    if abs(base * scale - baseline) < 1e-6:
                        core = render.union_core_box(core, tuple(v * scale for v in box))
                line["producer_core_box"] = core
            meta["producer_font_asset_ids"] = list(producer.current_font)
            meta["producer_background_asset_id"] = producer.current_background
        return result

    def extract(*args, **kwargs):
        got = original["extract"](*args, **kwargs)
        return None if got is None else [item for item in got if item[3].get("producer_core_box")
            and render.preserves_core(item[3]["original_crop_box"], item[3]["producer_core_box"])
            and render.preserves_core(item[3]["crop_box"], item[3]["producer_core_box"])]

    def crop_meta(meta, line, *args, **kwargs):
        bound = original["crop_meta"](meta, line, *args, **kwargs)
        refs = sorted(set(meta["producer_font_asset_ids"] + [meta["producer_background_asset_id"]]
                          + list(getattr(line["text"], "asset_ids", ()))))
        producer.validate_references(refs)
        if producer.shard_index is None:
            raise ValueError("producer sample identity requires an active immutable shard")
        patterns = [producer.pattern_id] if producer.pattern_id else []
        bound.update(split=producer.split, source_kind="synthetic", producer_recipe_sha256=producer.recipe_sha,
                     producer_core_box=list(line["producer_core_box"]), producer_core_alpha_threshold=128,
                     producer_label_ink_preserved=True,
                     source_provenance={"schema": 1,
                         "sample_id": f"synth:{producer.recipe_sha}:{producer.shard_index}:{producer.crop_index}",
                         "asset_ids": refs, "correction_pattern_ids": patterns,
                         "kanji_corrective": bool(producer.resources.kanji_pairs),
                         "broad_sampling": not bool(producer.pattern_id or producer.resources.kanji_pairs)})
        producer.crop_index += 1
        bound.pop("producer_font_asset_ids")
        bound.pop("producer_background_asset_id")
        return bound

    hooks = dict(clean=clean, add_spaces=spaces, random_name=name, pick_font=font, load_bg=load_bg,
                 background=background, render_region=region, extract=extract, crop_meta=crop_meta)
    try:
        for key, value in hooks.items():
            setattr(render, key, value)
        render.Layer.glyph = glyph
        yield
    finally:
        for key, value in original.items():
            setattr(render, key, value)
        render.R, STATE, render.Layer.glyph = old_r, old_state, old_glyph
        render._FONTS, render._AXES, render._BG = old_caches


def produce_shard(job):
    STATE.shard_index, STATE.crop_index = job[0], 0
    result = render.run_shard(job)
    path = Path(job[3]) / f"synth_{job[0]:05d}.npz"
    with np.load(path, allow_pickle=False) as shard:
        arrays = {key: shard[key] for key in shard.files}
    metas = json.loads(str(arrays["meta"]))
    for meta in metas:
        if meta["source_provenance"]["kanji_corrective"]:
            meta["source_provenance"]["correction_pattern_ids"] = ["kanji-pair:" + "".join(sorted(meta["kanji_pair"]))]
    arrays["meta"] = np.array(json.dumps(metas, ensure_ascii=False))
    temp = path.with_name(path.name + ".bound.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(path)
    return result


def produce(producer, out, workers=1):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("producer snapshots are immutable; choose a new output directory")
    if type(workers) is not int or workers < 1:
        raise ValueError("producer workers must be positive")
    producer.verify_inputs()
    out.mkdir(parents=True)
    recipe = producer.recipe
    jobs = [(i, recipe["per_shard"], recipe["seed"], str(out), recipe["finder"], recipe["invert_dark"])
            for i in range(recipe["shards"])]
    with installed(producer):
        if workers == 1:
            results = [produce_shard(job) for job in jobs]
        else:
            with mp.get_context("fork").Pool(workers) as pool:
                results = list(pool.imap_unordered(produce_shard, jobs))
    producer.verify_inputs()
    shards, used, seen = [], set(), set()
    for job in jobs:
        path = out / f"synth_{job[0]:05d}.npz"
        with np.load(path, allow_pickle=False) as shard:
            metas = json.loads(str(shard["meta"]))
        for meta in metas:
            provenance = meta["source_provenance"]
            if meta["split"] != producer.split or provenance["sample_id"] in seen:
                raise ValueError("producer role changed or sample identity was reused")
            producer.validate_references(provenance["asset_ids"])
            if producer.broad and (provenance["correction_pattern_ids"] or provenance["kanji_corrective"]
                                   or not provenance["broad_sampling"]):
                raise ValueError("broad output acquired a deliberate correction tag")
            seen.add(provenance["sample_id"])
            used.update(provenance["asset_ids"])
        shards.append({"file": path.name, "sha256": digest(path.read_bytes()), "bytes": path.stat().st_size,
                       "samples": len(metas), "source": producer.source, "split": producer.split})
    assets = {key: producer.registry[key] for key in sorted(used)}
    manifest = {"schema": 1, "kind": "synthetic-producer-snapshot", "assets": assets, "shards": shards,
                "recipe": recipe, "recipe_sha256": producer.recipe_sha, "render_results": sorted(results),
                "forbidden_vns": producer.forbidden_vns,
                "exposure_caps_enforced": False, "note": "Caps apply to actually consumed training rows."}
    (out / "source-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if producer.broad and producer.split == "train":
        policy = {"schema": 1, "kind": "general-vn-source-policy", "assets": assets, "shards": shards,
                  "fallback_source": producer.source, "forbidden_vns": producer.forbidden_vns}
        (out / "policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "asset-root", "vocab", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--per-shard", type=int, default=500)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--finder", choices=("grad", "mask", "legacy"), default="grad")
    parser.add_argument("--keep-polarity", action="store_true")
    parser.add_argument("--crop-edge-jitter", action="store_true")
    parser.add_argument("--broad", action="store_true")
    parser.add_argument("--kanji-policy", type=Path)
    correction = parser.add_mutually_exclusive_group()
    correction.add_argument("--target-kana", default="")
    correction.add_argument("--kanji-pairs", action="store_true")
    parser.add_argument("--correction-pattern-id")
    parser.add_argument("--native-probability", type=render.probability, default=0)
    args = parser.parse_args()
    producer = Producer(args.manifest, args.asset_root, args.vocab, split=args.split, source=args.source,
        seed=args.seed, shards=args.shards, per_shard=args.per_shard, finder=args.finder,
        invert=not args.keep_polarity, edge_jitter=args.crop_edge_jitter, broad=args.broad,
        target_kana=args.target_kana, correction_pattern_id=args.correction_pattern_id,
        kanji_policy=args.kanji_policy, kanji_pairs=args.kanji_pairs, native_probability=args.native_probability)
    result = produce(producer, args.out, args.workers)
    print(json.dumps({"recipe_sha256": producer.recipe_sha, "shards": len(result["shards"]),
                      "samples": sum(s["samples"] for s in result["shards"]), "exposure_caps_enforced": False}))


if __name__ == "__main__":
    main()
