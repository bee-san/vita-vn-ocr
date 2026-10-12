"""New producer inputs and glyph/provenance checks use local OFL and generated fixtures."""
import copy
import json
import shutil
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageFont

import fonts
import kanji_policy as KP
import native_bitmap_font as N
import provenance_producer as P
import render
import sandbox

ROOT = Path(__file__).resolve().parents[1]
VOCAB = ROOT / "data/vocab.txt"


def file_record(root, name, **fields):
    path = root / name
    return dict(file=name, bytes=path.stat().st_size, sha256=P.digest(path.read_bytes()), **fields)


def setup_inputs(root, font):
    root.mkdir()
    for name in fonts.FILES:
        shutil.copy2(font.parent / name, root / name)
    assets = {"font:fixture": file_record(root, fonts.FONT_FILE, kind="font", origin="ofl",
        license="OFL-1.1", license_file=file_record(root, "OFL.txt"), category="gothic", family=fonts.FAMILY)}
    for pool, lines in sandbox.fixture_lines().items():
        name = pool + ".txt"
        (root / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        assets["text:" + pool] = file_record(root, name, kind="text", origin="non_game",
            game_free_reviewed=True, split="train", pool=pool)
    return {"schema": 1, "kind": "synthetic-producer-inputs", "assets": assets}


@pytest.fixture(scope="module")
def producer_policy(tmp_path_factory):
    """Generated ranks/graph; real-policy bytes and hashes are never rewritten."""
    root = tmp_path_factory.mktemp("generated-policy")
    characters = sorted({c for lines in sandbox.fixture_lines().values() for line in lines for c in line
                         if KP.is_han(c)} | set("人入"))
    jiten, similar = root / "kanji.csv", root / "similar.txt"
    jiten.write_text("Kanji,Rank\n" + "".join(f"{c},{i + 1}\n" for i, c in enumerate(characters)), encoding="utf-8")
    similar.write_text("人/入/\n", encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(KP, "JITEN_SHA256", KP.sha256(jiten))
        patch.setattr(KP, "SIMILAR_SHA256", KP.sha256(similar))
        path = root / "policy.json"
        path.write_text(json.dumps(KP.build_policy(jiten, similar, VOCAB), ensure_ascii=False), encoding="utf-8")
        yield path


@pytest.fixture
def inputs(tmp_path, font, producer_policy):
    root, manifest = tmp_path / "assets", tmp_path / "inputs.json"
    doc = setup_inputs(root, font)

    def create(**kwargs):
        manifest.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        return P.Producer(manifest, root, VOCAB, split="train", source="fixture_broad",
            seed=1234, shards=1, per_shard=8, broad=True, kanji_policy=producer_policy, **kwargs)

    return SimpleNamespace(root=root, doc=doc, create=create, manifest=manifest, policy=producer_policy)


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory, font, producer_policy):
    folder = tmp_path_factory.mktemp("producer")
    root, manifest = folder / "assets", folder / "inputs.json"
    doc = setup_inputs(root, font)
    manifest.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    producer = P.Producer(manifest, root, VOCAB, split="train", source="fixture_broad",
        seed=1234, shards=2, per_shard=12, broad=True, kanji_policy=producer_policy)
    first, second = folder / "first", folder / "second"
    result = P.produce(producer, first, workers=1)
    P.produce(producer, second, workers=2)
    return producer, first, second, result


def metadata(path):
    with np.load(path, allow_pickle=False) as shard:
        return json.loads(str(shard["labels"])), json.loads(str(shard["meta"]))


def test_chunk_sources_survive_wrapping_and_inline_combination():
    name = P.SourceText("「あかり」", ("text:name",))
    body = P.SourceText("ずっと、そこにいます。ちょっと、まってね。", ("text:body",))
    combined = "　" + name + "　" + body
    lines = render.wrap(combined, 8, 8, np.random.default_rng(9))
    assert lines and all(line.asset_ids == ("text:body", "text:name") for line in lines)
    assert combined.replace("　", "").strip("「」").asset_ids == combined.asset_ids
    assert combined[2:].lstrip().rstrip().asset_ids == combined.asset_ids


@pytest.mark.parametrize("fields", [
    {"origin": "game"}, {"origin": "game", "vn": "123", "release": "r123"},
    {"origin": "game", "vn": "v123", "release": "123"},
    {"origin": "unknown", "vn": "v123", "release": "r123"},
    {"origin": "non_game", "game_free_reviewed": False}, {"origin": "procedural"},
])
def test_input_cannot_infer_ownership_or_claim_unreviewed_game_free_text(fields):
    with pytest.raises(ValueError):
        P.ownership({"kind": "text", "sha256": "1" * 64, **fields})


def test_duplicate_role_fields_and_invalid_source_identifiers_fail_closed(inputs):
    with pytest.raises(ValueError, match="duplicate"):
        P.parse_manifest('{"assets":{},"assets":{}}')
    with pytest.raises(ValueError, match="object"):
        P.parse_manifest("[]")
    row = {"kind": "font", "origin": "unknown"}
    with pytest.raises(ValueError, match="SHA256"):
        P.ownership(row)
    inputs.doc["assets"]["bad\nidentity"] = inputs.doc["assets"]["font:fixture"]
    with pytest.raises(ValueError, match="IDs"):
        inputs.create()


def test_unknown_is_retained_and_game_ids_are_explicit():
    assert P.ownership({"kind": "font", "origin": "unknown", "sha256": "1" * 64})["origin"] == "unknown"
    game = P.ownership({"kind": "font", "origin": "game", "sha256": "2" * 64,
                        "vn": "v300000", "release": "r300000"})
    assert (game["vn"], game["release"]) == ("v300000", "r300000")


def test_broad_and_other_role_assets_are_excluded_before_file_reads(inputs):
    inputs.doc["assets"].update({
        "font:private": dict(kind="font", origin="game", vn="v300000", release="r300000",
                             sha256="1" * 64, file="missing.fnt", format="native_bitmap",
                             metrics=dict(file="missing.pro", bytes=16, sha256="4" * 64)),
        "background:private": dict(kind="background", origin="unknown", split="train",
                                   sha256="2" * 64, file="missing.png"),
        "text:dev": dict(kind="text", origin="non_game", game_free_reviewed=True, split="dev",
                         sha256="3" * 64, file="never_read.txt"),
    })
    producer = inputs.create()
    assert not producer.resources.bg_vita and not producer.resources.bg_psp
    assert not producer.resources.native_fonts and producer.resources.native_font_probability == 0
    assert all("missing" not in str(path) and "never_read" not in str(path) for path in producer.file_pins)


def test_text_bytes_cannot_be_aliased_across_train_and_dev_before_any_asset_read(inputs, monkeypatch):
    inputs.doc["assets"]["text:dev-alias"] = {**inputs.doc["assets"]["text:dialogue"], "split": "dev"}
    monkeypatch.setattr(P.Producer, "checked_input", lambda *args: pytest.fail("asset bytes read before role checks"))
    with pytest.raises(ValueError, match="both train and dev"):
        inputs.create()


def test_late_invalid_role_fails_before_any_asset_read(inputs, monkeypatch):
    inputs.doc["assets"]["text:invalid-role"] = {**inputs.doc["assets"]["text:dialogue"], "split": "test"}
    monkeypatch.setattr(P.Producer, "checked_input", lambda *args: pytest.fail("asset bytes read before role checks"))
    with pytest.raises(ValueError, match="split"):
        inputs.create()


@pytest.mark.parametrize("kind,origin", [("font", "game"), ("font", "unknown"), ("text", "game")])
def test_broad_exclusions_cannot_hide_conflicting_raw_asset_owners(inputs, monkeypatch, kind, origin):
    original = inputs.doc["assets"]["font:fixture" if kind == "font" else "text:dialogue"]
    alias = {**original, "origin": origin}
    alias.pop("license", None)
    alias.pop("game_free_reviewed", None)
    if origin == "game":
        alias.update(vn="v300000", release="r300000")
    inputs.doc["assets"][kind + ":excluded-alias"] = alias
    monkeypatch.setattr(P.Producer, "checked_input", lambda *args: pytest.fail("asset bytes read before owner checks"))
    with pytest.raises(ValueError, match="conflicting ownership"):
        inputs.create()


def test_excluded_native_metrics_cannot_alias_an_ofl_font_identity(inputs, monkeypatch):
    inputs.doc["assets"]["font:native"] = dict(kind="font", origin="game", vn="v300000",
        release="r300000", sha256="1" * 64, file="missing.fnt", format="native_bitmap",
        metrics={**inputs.doc["assets"]["font:fixture"], "file": "missing.pro"})
    monkeypatch.setattr(P.Producer, "checked_input", lambda *args: pytest.fail("asset bytes read before owner checks"))
    with pytest.raises(ValueError, match="conflicting ownership"):
        inputs.create()


def test_changed_vocabulary_order_cannot_use_the_original_jiten_policy(inputs):
    values = VOCAB.read_text(encoding="utf-8").splitlines()
    changed = inputs.root / "vocab-reordered.txt"
    changed.write_text("\n".join(reversed(values)) + "\n", encoding="utf-8")
    inputs.manifest.write_text(json.dumps(inputs.doc, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="vocabulary differ"):
        P.Producer(inputs.manifest, inputs.root, changed, split="train", source="reordered", seed=1,
                   shards=1, per_shard=1, broad=True, kanji_policy=inputs.policy)


@pytest.mark.parametrize("kwargs", [
    {"native_probability": 1}, {"target_kana": "ず", "correction_pattern_id": "kana:ず"},
    {"kanji_pairs": True},
])
def test_broad_constructor_disables_deliberate_correction_and_native_selection(inputs, kwargs):
    with pytest.raises(ValueError, match="broad constructor"):
        inputs.create(**kwargs)


def test_native_cannot_be_claimed_as_ofl_to_enter_broad_constructor(inputs):
    inputs.doc["assets"]["font:fixture"].update(format="native_bitmap", file="missing.fnt")
    with pytest.raises(ValueError, match="broad constructor"):
        inputs.create()


def test_text_ownership_role_vocab_and_raw_hash_are_checked(inputs):
    original = copy.deepcopy(inputs.doc)
    text = inputs.doc["assets"]["text:dialogue"]
    for mutation, message in [({"split": "test"}, "split"), ({"sha256": "0" * 64}, "SHA256"),
                              ({"game_free_reviewed": False}, "game-free")]:
        text.update(mutation)
        with pytest.raises(ValueError, match=message):
            inputs.create()
        inputs.doc.clear()
        inputs.doc.update(copy.deepcopy(original))
        text = inputs.doc["assets"]["text:dialogue"]
    (inputs.root / "dialogue.txt").write_text("\U00020000\n", encoding="utf-8")
    text.update(file_record(inputs.root, "dialogue.txt"))
    with pytest.raises(ValueError, match="vocabulary/policy"):
        inputs.create()


def test_han_including_extensions_requires_jiten_before_sampling(inputs):
    for character in ("人", "\U00020000"):
        (inputs.root / "han.txt").write_text(character + "\n", encoding="utf-8")
        inputs.doc["assets"] = {"font:fixture": inputs.doc["assets"]["font:fixture"],
            "text:han": file_record(inputs.root, "han.txt", kind="text", origin="non_game",
                                    game_free_reviewed=True, split="train", pool="dialogue")}
        vocab = inputs.root / "vocab.txt"
        vocab.write_text(character + "\nあ\nア\n　\n", encoding="utf-8")
        inputs.manifest.write_text(json.dumps(inputs.doc, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(ValueError, match="Jiten"):
            P.Producer(inputs.manifest, inputs.root, vocab, split="train", source="han", seed=1,
                       shards=1, per_shard=1, broad=True)


def test_file_paths_hashes_and_bounded_bytes(inputs, tmp_path):
    path = inputs.root / "dialogue.txt"
    row = file_record(inputs.root, "dialogue.txt")
    assert P.checked_file(inputs.root, row)[1] == path.read_bytes()
    for change in ({"bytes": row["bytes"] + 1}, {"sha256": "0" * 64}, {"file": "../outside.txt"},
                   {"file": str(path)}, {"bytes": True}):
        with pytest.raises(ValueError):
            P.checked_file(inputs.root, {**row, **change})
    outside = tmp_path / "outside.txt"
    outside.write_text("あ")
    (inputs.root / "escape.txt").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        P.checked_file(inputs.root, {**row, "file": "escape.txt"})


def test_asset_aliases_cannot_change_owner_and_independent_chunks_do_not_collide(inputs):
    producer = inputs.create()
    row = producer.registry[producer.font_refs[next(iter(producer.font_refs))][0]]
    with pytest.raises(ValueError, match="conflicting"):
        producer.register("font:wrong-owner", {**row, "origin": "game", "vn": "v300000", "release": "r300000"})
    assert producer.register("font:alias", row) == "font:fixture"
    sha = P.digest("ずっと".encode())
    assert P.digest(P.canonical(dict(source_sha256="1" * 64, line=0, chunk_sha256=sha))) != P.digest(
        P.canonical(dict(source_sha256="2" * 64, line=0, chunk_sha256=sha)))


def test_changed_lazy_input_prevents_publishing_snapshot(inputs, tmp_path):
    producer = inputs.create()
    (inputs.root / fonts.FONT_FILE).write_bytes(b"changed")
    out = tmp_path / "never-created"
    with pytest.raises(ValueError, match="changed"):
        P.produce(producer, out)
    assert not out.exists()


def test_scoped_hooks_restore_defaults_even_when_render_fails(inputs):
    producer = inputs.create()
    saved = (render.R, render.Layer.glyph, render.pick_font, render.crop_meta, render._FONTS, render._BG, P.STATE)
    with pytest.raises(RuntimeError), P.installed(producer):
        assert render.R is producer.resources
        raise RuntimeError("generated failure")
    assert saved == (render.R, render.Layer.glyph, render.pick_font, render.crop_meta,
                     render._FONTS, render._BG, P.STATE)


def test_actual_successful_background_only_and_procedural_when_patch_does_not_fit(inputs):
    path = inputs.root / "generated.png"
    Image.new("RGB", (480, 272), (15, 25, 35)).save(path)
    inputs.doc["assets"]["background:generated"] = file_record(inputs.root, path.name, kind="background",
        origin="game", vn="v300000", release="r300000", split="train", platform="psp",
        scenery_reviewed=True, scenery_box=[20, 30, 460, 250])
    inputs.manifest.write_text(json.dumps(inputs.doc, ensure_ascii=False), encoding="utf-8")
    producer = P.Producer(inputs.manifest, inputs.root, VOCAB, split="train",
        source="mixed", seed=3, shards=1, per_shard=1, kanji_policy=inputs.policy)
    with P.installed(producer):
        for seed in range(10):
            render.background(np.random.default_rng(seed), 32, 32, "psp")
            if producer.current_background == "background:generated":
                break
        else:
            pytest.fail("seeded generated source never selected")
        render.background(np.random.default_rng(seed), 1000, 1000, "psp")
        assert producer.current_background == producer.procedural_background
        assert render.R.bg_boxes[path] == (20, 30, 460, 250)


def test_owned_native_source_and_metrics_are_bound_and_ambiguous_glyphs_fall_back(inputs):
    codes = (0x41, 0x42, 0x8140, 0x82A0)
    bank = bytes((255, 255, 255, 255, 0, 0, 15, 240))  # identical A/B, empty space, distinct あ
    fnt = struct.pack("<HHI", 2, 2, len(codes)) + bytes(8) + struct.pack("<4H", *codes) + bank * 2
    count = N.metric_index("あ") + 1
    pro = struct.pack("<I", count) + bytes(12) + bytes((0, 0, 2)) * count
    (inputs.root / "generated.fnt").write_bytes(fnt)
    (inputs.root / "generated.pro").write_bytes(pro)
    inputs.doc["assets"]["font:generated"] = file_record(inputs.root, "generated.fnt", kind="font",
        origin="game", vn="v300000", release="r300000", format="native_bitmap", category="pixel",
        family="generated_fixture", metrics=file_record(inputs.root, "generated.pro"))
    inputs.manifest.write_text(json.dumps(inputs.doc, ensure_ascii=False), encoding="utf-8")
    producer = P.Producer(inputs.manifest, inputs.root, VOCAB, split="train", source="owned_native",
        seed=3, shards=1, per_shard=1, kanji_policy=inputs.policy, native_probability=1)
    native_path = inputs.root / "generated.fnt"
    assert producer.font_refs[native_path] == ("font:generated", "font:generated:metrics")
    assert producer.registry["font:generated"]["sha256"] == P.digest(fnt)
    assert producer.registry["font:generated:metrics"]["sha256"] == P.digest(pro)
    assert not set("AB") & producer.resources.native_fonts[0][3]
    with P.installed(producer):
        assert render.pick_font(np.random.default_rng(1), set("あ"), "vita")[2] == native_path
        assert producer.current_font == ("font:generated", "font:generated:metrics")
        assert render.pick_font(np.random.default_rng(1), set("AB"), "vita")[2] != native_path
        assert producer.current_font == ("font:fixture",)


def test_core_box_contains_actual_fractional_ofl_ink(font):
    face = ImageFont.truetype(str(font), 24)
    core = P.core_box(face, "ず", 3.4, 29.7, 2, "L")
    assert core and core[0] >= 0 and core[1] < core[3]
    assert render.preserves_core((0, 0, 100, 100), core)
    assert not render.preserves_core((core[0] + 1, core[1], 100, 100), core)
    assert P.core_box(face, "　", 3.4, 29.7, 2, "L") is None


def test_broad_snapshot_is_fully_verified_and_ordinary_kana_has_no_correction_tags(snapshot):
    producer, first, _, result = snapshot
    seen, natural = set(), []
    for shard in result["shards"]:
        path = first / shard["file"]
        assert shard["sha256"] == P.digest(path.read_bytes())
        assert shard["samples"] == 12 and shard["split"] == "train"
        labels, metas = metadata(path)
        for label, meta in zip(labels, metas, strict=True):
            row = meta["source_provenance"]
            assert meta["split"] == "train" and meta["producer_label_ink_preserved"] is True
            assert render.preserves_core(meta["original_crop_box"], meta["producer_core_box"])
            assert render.preserves_core(meta["crop_box"], meta["producer_core_box"])
            assert row["schema"] == 1 and row["broad_sampling"] is True
            assert row["correction_pattern_ids"] == [] and row["kanji_corrective"] is False
            producer.validate_references(row["asset_ids"])
            assert row["sample_id"] not in seen
            seen.add(row["sample_id"])
            natural.append("ず" in label or "っ" in label)
    assert len(seen) == 24 and any(natural)
    assert result["exposure_caps_enforced"] is False
    policy = json.loads((first / "policy.json").read_text())
    assert policy["fallback_source"] == "fixture_broad" and policy["shards"] == result["shards"]
    assert policy["assets"] == result["assets"]


def test_fixed_recipe_seed_and_rows_are_identical_across_worker_counts(snapshot):
    producer, first, second, result = snapshot
    for entry in result["shards"]:
        with np.load(first / entry["file"], allow_pickle=False) as a, np.load(second / entry["file"],
                                                                            allow_pickle=False) as b:
            assert a.files == b.files
            for key in a.files:
                assert np.array_equal(a[key], b[key]), key
    assert json.loads((second / "source-manifest.json").read_text())["recipe_sha256"] == producer.recipe_sha
    with pytest.raises(FileExistsError, match="immutable"):
        P.produce(producer, first)


def test_deliberate_kana_stream_tags_only_its_explicit_pattern(inputs, tmp_path):
    inputs.manifest.write_text(json.dumps(inputs.doc, ensure_ascii=False), encoding="utf-8")
    producer = P.Producer(inputs.manifest, inputs.root, VOCAB, split="train", source="kana",
        seed=5, shards=1, per_shard=4, kanji_policy=inputs.policy,
        target_kana="ず", correction_pattern_id="kana:ず")
    out = tmp_path / "targeted"
    result = P.produce(producer, out)
    labels, metas = metadata(out / result["shards"][0]["file"])
    assert all("ず" in label for label in labels)
    assert all(meta["source_provenance"]["correction_pattern_ids"] == ["kana:ず"] and
               meta["source_provenance"]["kanji_corrective"] is False and
               meta["source_provenance"]["broad_sampling"] is False for meta in metas)
    assert not (out / "policy.json").exists()


def test_kana_target_order_duplicates_and_optional_id_share_one_recipe(inputs):
    inputs.manifest.write_text(json.dumps(inputs.doc, ensure_ascii=False), encoding="utf-8")
    common = dict(split="train", source="kana", seed=5, shards=1, per_shard=4, kanji_policy=inputs.policy)
    a = P.Producer(inputs.manifest, inputs.root, VOCAB, target_kana="ずっ", **common)
    b = P.Producer(inputs.manifest, inputs.root, VOCAB, target_kana="っずっず",
                   correction_pattern_id="kana:ずっ", **common)
    assert a.recipe_sha == b.recipe_sha
    assert a.pattern_id == b.pattern_id == "kana:ずっ"


@pytest.mark.parametrize("pattern", ["kana:ず:shard-1", "kana:ず:shard-2", "kana:っず", "unrelated"])
def test_kana_pattern_aliases_cannot_split_exposure_caps(inputs, monkeypatch, pattern):
    inputs.manifest.write_text(json.dumps(inputs.doc, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(P.Producer, "checked_input", lambda *args: pytest.fail("asset bytes read before target checks"))
    with pytest.raises(ValueError, match="canonical target set"):
        P.Producer(inputs.manifest, inputs.root, VOCAB, split="train", source="kana", seed=5,
                   shards=1, per_shard=4, kanji_policy=inputs.policy,
                   target_kana="ず", correction_pattern_id=pattern)


def test_deliberate_pair_stream_uses_one_canonical_pattern_for_both_endpoint_orders(inputs, tmp_path):
    inputs.manifest.write_text(json.dumps(inputs.doc, ensure_ascii=False), encoding="utf-8")
    producer = P.Producer(inputs.manifest, inputs.root, VOCAB, split="train", source="pair_fixture",
        seed=5, shards=1, per_shard=4, kanji_policy=inputs.policy, kanji_pairs=True)
    out = tmp_path / "pairs"
    result = P.produce(producer, out)
    labels, metas = metadata(out / result["shards"][0]["file"])
    assert set(meta["kanji_target"] for meta in metas) == set("人入")
    for label, meta in zip(labels, metas, strict=True):
        row = meta["source_provenance"]
        assert meta["kanji_target"] in label and row["kanji_corrective"] is True
        assert row["broad_sampling"] is False and row["correction_pattern_ids"] == ["kanji-pair:人入"]
    assert not (out / "policy.json").exists()


def test_development_generation_keeps_role_and_cannot_be_a_training_fallback(inputs, tmp_path):
    for row in inputs.doc["assets"].values():
        if row["kind"] == "text":
            row["split"] = "dev"
    inputs.manifest.write_text(json.dumps(inputs.doc, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="development"):
        P.Producer(inputs.manifest, inputs.root, VOCAB, split="dev", source="dev_fixture", seed=5,
            shards=1, per_shard=2, kanji_policy=inputs.policy, target_kana="ず", correction_pattern_id="kana:ず")
    producer = P.Producer(inputs.manifest, inputs.root, VOCAB, split="dev", source="dev_fixture", seed=5,
        shards=1, per_shard=2, kanji_policy=inputs.policy, broad=True)
    out = tmp_path / "development"
    result = P.produce(producer, out)
    _, metas = metadata(out / result["shards"][0]["file"])
    assert all(meta["split"] == "dev" for meta in metas)
    assert result["shards"][0]["split"] == "dev" and not (out / "policy.json").exists()
