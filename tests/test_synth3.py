"""synth3 (docs/synth3.md): the generator, its text pools and names, the blank-glyph guard, the
length gate, the pinned font manifest and the ablation statistics. Everything runs from the
pinned CI font and the fixture corpus in ci/fixtures; no shared data, screenshots or network
beyond ci/fonts.py."""
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synth"))
sys.path.insert(0, str(ROOT / "eval"))

import synth3_ablation as ABL  # noqa: E402

import fonts  # noqa: E402
import synth3  # noqa: E402
import synth3_text as T  # noqa: E402
from common import load_vocab  # noqa: E402

VOCAB = load_vocab()
FIXTURES = ROOT / "ci" / "fixtures" / "corpus"


def fixture_lines(name):
    return [s for s in (FIXTURES / f"{name}.txt").read_text(encoding="utf-8").split("\n") if s]


@pytest.fixture(scope="module")
def assets(font, tmp_path_factory):
    """A synth3 asset root holding only CI fixtures: the pinned OFL font, the fixture corpus as
    every pool, a few names, context indexes, targets and lists; no screenshots."""
    root = tmp_path_factory.mktemp("synth3") / "assets"
    fam = root / "fonts" / fonts.FAMILY
    fam.mkdir(parents=True)
    for name in fonts.FILES:
        shutil.copy2(Path(font).parent / name, fam / name)
    f = fam / fonts.FONT_FILE
    manifest = {"kind": "synth3-fonts", "schema": 1, "families": [{
        "family": fonts.FAMILY, "category": "gothic", "root": "synth3", "licence_id": "OFL-1.1",
        "licence_class": "open", "licence_file": f"{fonts.FAMILY}/OFL.txt",
        "files": [{"file": f"{fonts.FAMILY}/{fonts.FONT_FILE}", "bytes": f.stat().st_size,
                   "sha256": synth3.sha256(f), "category": "gothic", "weight": 400, "pixel_em": None}]}]}
    (root / "fonts_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pools = root / "text" / "pools3"
    pools.mkdir(parents=True)
    meta = {}
    src = {"aozora_dialogue": fixture_lines("dialogue"), "aozora_narration": fixture_lines("narration"),
           "wiki": fixture_lines("wiki"), "tatoeba": [s.strip("「」") for s in fixture_lines("dialogue")]}
    for name, lines in src.items():
        for split in ("train", "dev"):
            path = pools / f"{name}.{split}.txt"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            meta[f"{name}.{split}"] = {"file": path.name, "lines": len(lines), "class": "open"}
    (pools / "pools.json").write_text(json.dumps({"pools": meta}), encoding="utf-8")
    names = "vndb\tprimary\t佐藤\t花子\nvndb\tside\tアリス\nsyosetu\t-\t健太\n"
    for split in ("train", "dev"):
        (pools / f"names.{split}.tsv").write_text(names, encoding="utf-8")
        T.cmd_index(type("A", (), {"vocab": ROOT / "data" / "vocab.txt", "pools": str(pools), "split": split,
                                   "k": 8})())
    chars = {c: {"real": 1e-3, "jiten_vn": 1e-3, "hard": 0.0} for c in "".join(sum(src.values(), []))}
    (pools / "targets.json").write_text(json.dumps({"chars": chars}, ensure_ascii=False), encoding="utf-8")
    lists = {"interjections": ["あ", "えっ"], "stammers": ["あ、あの……"], "onomatopoeia": ["ドキドキ"],
             "honorifics": ["さん", "先輩"], "elongations": ["ーっ"], "endings": ["だよ"]}
    (root / "text" / "lists.json").write_text(json.dumps(lists, ensure_ascii=False), encoding="utf-8")
    (root / "mining").mkdir()
    (root / "mining" / "exclude_dev_vns.txt").write_text("v1\n")
    return root


@pytest.fixture(scope="module")
def shard(assets, tmp_path_factory):
    synth3.init(assets=assets, font_manifest=assets / "fonts_manifest.json", scenes=False)
    out = tmp_path_factory.mktemp("shards")
    idx, n, regions, errors, _ = synth3.run_shard((0, 60, 7, str(out), "synth3"))
    assert n == 60 and errors == 0
    return out / "synth3_00000.npz"


def test_shard_format_labels_and_meta(shard):
    z = np.load(shard)
    labels = json.loads(str(z["labels"]))
    meta = json.loads(str(z["meta"]))
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    assert len(labels) == len(meta) == len(shapes) == 60
    assert (np.diff(offs) == shapes[:, 0] * shapes[:, 1]).all() and offs[-1] == data.size
    assert all(lab and set(lab) <= set(VOCAB) for lab in labels)
    assert not any(set(lab) <= set("「」『』（）()、。・") for lab in labels)
    for g, lab in zip(shapes, labels):
        assert synth3.S2.crop_ctc_feasible(tuple(g), lab)
    for m, (h, w) in zip(meta, shapes):
        assert m["crop"] in ("gt", "grad") and m["platform"] in synth3.CFG["platform"]
        assert m["inverted"] == m["dark"]            # stored in console polarity
        x0, y0, x1, y1 = m["ink_box"]
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h   # the labelled ink is inside the crop


def test_labels_come_from_the_fixture_text_and_names(shard):
    z = np.load(shard)
    labels = json.loads(str(z["labels"]))
    meta = json.loads(str(z["meta"]))
    corpus = "\n".join(sum((fixture_lines(n) for n in ("dialogue", "narration", "wiki")), []))
    text = [lab for lab, m in zip(labels, meta) if m["kind"] in ("dialogue", "narration")]

    def from_fixture(lab):
        core = lab.strip("「」『』")
        return core in corpus or any(core[i:i + 4] in corpus for i in range(len(core) - 3))
    assert text and sum(map(from_fixture, text)) >= 2 * len(text) // 3, text


def test_render_is_deterministic(assets, tmp_path):
    """A shard depends on (seed, shard index) only: the same job renders the same bytes."""
    synth3.init(assets=assets, font_manifest=assets / "fonts_manifest.json", scenes=False)
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        synth3.run_shard((3, 12, 11, str(tmp_path / d), "synth3"))
    a, b = (np.load(tmp_path / d / "synth3_00003.npz") for d in ("a", "b"))
    assert all(np.array_equal(a[k], b[k]) for k in ("data", "offsets", "shapes"))
    assert str(a["labels"]) == str(b["labels"]) and str(a["meta"]) == str(b["meta"])


def test_drawn_chars_ignore_mapped_but_empty_glyphs(tmp_path):
    """A cmap entry whose glyph has no outline would render a blank under its label."""
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder([".notdef", "full", "empty"])
    fb.setupCharacterMap({ord("あ"): "full", ord("い"): "empty", 0x3000: "empty"})
    pen = TTGlyphPen(None)
    pen.moveTo((100, 100))
    pen.lineTo((900, 100))
    pen.lineTo((900, 900))
    pen.closePath()
    glyphs = {".notdef": TTGlyphPen(None).glyph(), "full": pen.glyph(), "empty": TTGlyphPen(None).glyph()}
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics({g: (1000, 0) for g in glyphs})
    fb.setupHorizontalHeader(ascent=880, descent=-120)
    fb.setupNameTable({"familyName": "T", "styleName": "R"})
    fb.setupOS2()
    fb.setupPost()
    path = tmp_path / "t.ttf"
    fb.save(str(path))
    assert synth3.drawn_chars(path, ["あ", "い", "\u3000", "う"]) == ["あ", "\u3000"]


def test_gt_box_pads_ink_and_keeps_neighbours_out():
    ink = np.zeros((80, 200), bool)
    ink[10:30, 20:120] = True              # line 1
    ink[34:54, 20:150] = True              # line 2, 4 px below
    lay = [{"y0": 9, "y1": 31, "fs": 20, "ix0": 20, "ix1": 120},
           {"y0": 33, "y1": 55, "fs": 20, "ix0": 20, "ix1": 150}]
    rng = np.random.default_rng(0)
    for _ in range(50):
        box, ib = synth3.gt_box(lay[0], lay, ink, np.zeros_like(ink), 4, rng, ink.shape)
        assert ib == (20, 10, 120, 30)
        x, y, w, h = box
        assert x <= 16 and y <= 6 and x + w >= 124 and y + h >= 30   # at least eval's 4 px pad above
        assert y + h <= 33 + 0.25 * 22 + 1                          # at most a quarter of line 2


def test_length_gate_moves_toward_the_target():
    """Proposals with every length 1-40 equally likely (far too few short lines): the kept crops'
    length histogram is much closer to the target. The gate assumes every target bin gets
    proposals, as the renderer's do; a bin that never did would pull acceptance to the floor."""
    gate = synth3.LengthGate()
    rng = np.random.default_rng(1)
    proposals = rng.integers(1, 41, size=30000)
    kept = np.array([n for n in proposals if gate.accept(rng, int(n))])
    edges, target = synth3.CFG["length_edges"], np.array(synth3.CFG["length_target"])

    def hist(x):
        h = np.histogram(x, bins=edges)[0].astype(float)
        return h / h.sum()
    assert np.abs(hist(kept) - target).sum() < 0.5 * np.abs(hist(proposals) - target).sum()
    assert 0.2 < len(kept) / len(proposals) < 1.0


def test_text_normalisation_and_document_split():
    assert T.norm_vn("えっ!? そ、そんな...") == "えっ！？そ、そんな…"
    assert T.norm_vn("ﾃｽﾄ〜") == "テスト～"
    vocab = set(VOCAB)
    assert T.keep("「おはよう」", vocab) and not T.keep("abc", vocab) and not T.keep("お" * 61, vocab)
    assert all(len(p) <= T.MAX_LEN for p in T.sentences("あ。" * 80))
    splits = [T.split_of("tatoeba", i) for i in range(4000)]
    assert splits == [T.split_of("tatoeba", i) for i in range(4000)]
    assert 0.03 < splits.count("dev") / len(splits) < 0.07


def test_names_drop_reserved_and_development_vns(tmp_path):
    reserved = json.loads((ROOT / "eval" / "manifests" / "test_games.json").read_text())["all"][0]
    allowed = next(f"v{i}" for i in range(100000, 101000) if not T.is_test_vn(f"v{i}"))
    tsv = tmp_path / "chars.tsv"
    tsv.write_text("char_id\tlang\tname\tlatin\tsex\tvn_ids\troles\tmain_role\thas_kana\n"
                   f"c1\tja\t佐藤 花子\t\tf\t{allowed}\tprimary\tprimary\t0\n"
                   f"c2\tja\t鈴木 一郎\t\tm\t{allowed},{reserved}\tprimary\tprimary\t0\n"
                   f"c3\tja\t田中 太郎\t\tm\tv777777\tside\tside\t0\n", encoding="utf-8")
    dev = tmp_path / "dev.txt"
    dev.write_text("v777777\n")
    T.cmd_names(type("A", (), {"vocab": ROOT / "data" / "vocab.txt", "vndb_chars": tsv, "syosetu": None,
                               "dev_vns": dev, "out": str(tmp_path)})())
    rows = (tmp_path / "names.train.tsv").read_text(encoding="utf-8") + \
        (tmp_path / "names.dev.tsv").read_text(encoding="utf-8")
    assert "佐藤\t花子" in rows and "鈴木" not in rows and "田中" not in rows


def test_font_manifest_pins_every_file():
    doc = json.loads((ROOT / "data" / "manifests" / "synth3_fonts.json").read_text(encoding="utf-8"))
    seen = set()
    for fam in doc["families"]:
        assert fam["licence_class"] in ("open", "nc") and fam["root"] in ("pool", "pool_extra", "synth3")
        for f in fam["files"]:
            assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) and f["bytes"] > 0 and f["sha256"] not in seen
            seen.add(f["sha256"])
            assert f["category"] != "pixel" or f["pixel_em"]
    assert doc["counts"]["files"] == len(seen)


def test_ablation_bootstrap_and_decision(tmp_path):
    rng = np.random.default_rng(0)
    lines = [{"id": f"l{i}", "vn": f"v{i % 20}", "ref": "あいうえお", "tags": ["kind:psp"]} for i in range(400)]
    base = rng.integers(0, 3, 400)
    models = {}
    for run, shift in (("a1", 0), ("a2", 0), ("b1", -1), ("b2", -1)):
        e = np.clip(base + (shift * (rng.random(400) < 0.5)), 0, None).tolist()
        models[run] = {"blob_sha256": "0" * 64, "blob_bytes": 1, "arena_bytes": 1, "cer": {},
                       "errors": {"rv": e, "rvn": e}}
    (tmp_path / "s.json").write_text(json.dumps({"dev": {}, "lines": lines, "models": models}))
    proto = {"views": ["rv", "rvn"], "bootstrap": {"resamples": 2000, "seed": 0}, "decision": "test",
             "arms": {"A": ["a1", "a2"], "B": ["b1", "b2"]},
             "comparisons": {"B_vs_A": ["A", "B"], "A_vs_A": ["A", "A"]}}
    (tmp_path / "p.json").write_text(json.dumps(proto))
    ABL.cmd_compare(type("A", (), {"scores": [tmp_path / "s.json"], "protocol": tmp_path / "p.json",
                                   "out": tmp_path / "r.json"})())
    r = json.loads((tmp_path / "r.json").read_text())["comparisons"]
    assert r["B_vs_A"]["decision"]["b_better"] and r["B_vs_A"]["all_vn_cluster"]["ci95"][1] < 0
    assert r["A_vs_A"]["all_vn_cluster"]["delta_points"] == 0 and not r["A_vs_A"]["decision"]["b_better"]


def test_protocol_is_consistent():
    p = json.loads((ROOT / "eval" / "synth3_ablation_protocol.json").read_text(encoding="utf-8"))
    assert set(p["arms"]) == set(p["arm_sources"])
    for a, b in p["comparisons"].values():
        assert a in p["arms"] and b in p["arms"]
    for srcs in p["arm_sources"].values():
        assert abs(sum(srcs.values()) - 1.0) < 1e-9 and set(srcs) <= set(p["sources"])
    assert p["recipe"]["steps"] == 16000 and p["checkpoint"].startswith("ckpt_step16000.pt")
