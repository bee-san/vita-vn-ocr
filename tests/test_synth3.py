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


def test_grad_crops_need_one_hugged_line(font):
    """A GRAD box is kept only when it holds one labelled line and hugs its ink: unlabelled text
    in the same band (screen text, a neighbour) makes the box wider and drops the crop."""
    from PIL import Image, ImageDraw, ImageFont
    f = ImageFont.truetype(str(font), 24)

    def region(extra):
        im = Image.new("RGB", (520, 60), (20, 20, 30))
        d = ImageDraw.Draw(im)
        d.text((16, 40), "あいうえお", font=f, fill=(240, 240, 240), anchor="ls")
        if extra:
            d.text((150, 40), "かきくけこさしすせそ", font=f, fill=(240, 240, 240), anchor="ls")
        img = np.asarray(im)
        m = Image.new("L", (520, 60), 0)
        ImageDraw.Draw(m).text((16, 40), "あいうえお", font=f, fill=255, anchor="ls")
        ink = np.asarray(m) >= 128
        cols = np.flatnonzero(ink.any(axis=0))
        lay = [{"text": "あいうえお", "top": 40 - 0.88 * 24, "fs": 24, "y0": 40 - 0.88 * 24 - 1,
                "y1": 40 + 0.12 * 24 + 1, "ix0": int(cols[0]), "ix1": int(cols[-1]) + 1, "kind": "dialogue"}]
        return img, lay
    img, lay = region(False)
    got = synth3.grad_crops(img, lay)
    assert len(got) == 1 and got[0][1] == "あいうえお"
    img, lay = region(True)
    assert synth3.grad_crops(img, lay) == []


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


def test_study2_protocol_is_consistent():
    p = json.loads((ROOT / "eval" / "synth3_study2_protocol.json").read_text(encoding="utf-8"))
    for arm, runs in p["arms"].items():
        arch, src = arm.split("_", 1)
        assert arch in p["architectures"] and src in p["arm_sources"]
        assert runs == [f"ft_{arch}_{src}_seed{s}" for s in p["seeds"][arch]]
    for a, b in p["comparisons"].values():
        assert a in p["arms"] and b in p["arms"] and a.split("_")[0] == b.split("_")[0]
        assert len(p["arms"][a]) == len(p["arms"][b])
    for stages in p["arm_sources"].values():
        for mix in stages.values():
            assert abs(sum(mix.values()) - 1.0) < 1e-9 and set(mix) <= set(p["sources"]) - {"targets"}
    assert {a["pretrain_steps"] for a in p["architectures"].values()} == {32000, 40000}
    assert "ckpt_step15000.pt" in p["checkpoint"] and p["views"] == ["rv", "rvn"]
    run = (ROOT / "eval" / "synth3_study2_run.sh").read_text(encoding="utf-8")
    assert 'PRUN=pre_${ARCH}_${ARM}${TAG}_seed${SEED}' in run and 'FRUN=ft_${ARCH}_${ARM}${TAG}_seed${SEED}' in run
    assert "kd=0" in run and "kd=1" not in run and "--steps 15000" in run


@pytest.mark.parametrize("script", ["synth3_study2_run.sh", "synth3_curve_run.sh", "synth3_run_arm.sh",
                                    "synth3_run_ft.sh"])
def test_run_scripts_parse(script):
    import subprocess
    subprocess.run(["bash", "-n", str(ROOT / "eval" / script)], check=True)


def _run_json(errors, vns):
    lines = [{"id": f"x{i}", "vn": vns[i], "styles": ["name"] if i % 2 else ["outlined"], "text": "あいう", "pred": "",
              "errors": int(e), "n": 3} for i, e in enumerate(errors)]
    sets = {s: {"lines": lines} for s in ("vita-test", "psp-test")}
    return {"model_info": {"sha256": "0" * 64, "blob_bytes": 1, "arena_bytes": 1}, "env": {"git": "x"}, "sets": sets}


def test_test_report_arms_pairs_and_styles(tmp_path):
    import synth3_test_report as TR
    rng = np.random.default_rng(1)
    vns = [f"v{i % 10}" for i in range(60)]
    base = rng.integers(0, 3, 60)
    for run, shift in (("a1", 0), ("a2", 0), ("b1", 1), ("b2", 1), ("ref", 0)):
        for pol in ("console", "native"):
            (tmp_path / f"{run}_{pol}.json").write_text(json.dumps(_run_json(base + shift, vns)))
    proto = {"bootstrap": {"resamples": 500, "seed": 0}, "arms": {"A": ["a1", "a2"], "B": ["b1", "b2"]},
             "comparisons": {"B_vs_A": ["A", "B"]}}
    (tmp_path / "p.json").write_text(json.dumps(proto))
    args = type("A", (), {"protocol": tmp_path / "p.json", "dir": tmp_path, "pair": ["a1=b1"], "ref": ["ref"]})()
    res = TR.build(args)
    s = res["polarities"]["console"]["sets"]["vita-test"]
    assert s["arms"]["A"]["errors"] == 2 * int(base.sum()) and s["arms"]["A"]["chars"] == 2 * 180
    c = s["comparisons"]["B_vs_A"]
    assert c["vn_cluster"]["delta_points"] == pytest.approx(100 / 3, abs=1e-3) and c["vn_cluster"]["ci95"][0] > 0
    assert s["pairs"]["a1->b1"]["line"]["delta_points"] == pytest.approx(100 / 3, abs=1e-3)
    assert set(c["styles"]) == {"name", "outlined"} and "ref" in s["runs"]
    pooled = res["polarities"]["native"]["sets"]["pooled"]["arms"]["B"]
    assert pooled["chars"] == 2 * 2 * 180 and "B_vs_A" in TR.markdown(res)


def _scores(path, names_shift, n=300, vns=20, seed=0, tag=""):
    rng = np.random.default_rng(seed)
    lines = [{"id": f"{tag}l{i}", "vn": f"v{i % vns}", "ref": "あいうえお", "tags": ["kind:psp"]} for i in range(n)]
    base = rng.integers(0, 3, n)
    models = {}
    for run, shift in names_shift.items():
        e = np.clip(base + (shift * (rng.random(n) < 0.5)), 0, None).tolist()
        models[run] = {"blob": "x", "blob_sha256": "0" * 64, "blob_bytes": 1, "arena_bytes": 1,
                       "pred": {"rv": [""] * n, "rvn": [""] * n}, "errors": {"rv": e, "rvn": e}}
    for m in models.values():
        m["cer"] = {v: sum(e) / (5 * n) for v, e in m["errors"].items()}
    Path(path).write_text(json.dumps({"dev": {"lines": n}, "lines": lines, "models": models}))
    return path


def test_study2_result_applies_the_frozen_rule(tmp_path):
    import synth3_study2 as S2
    runs_a = {f"ft_FL10_A_s2_seed{s}": 0 for s in (1, 2, 3)}
    better = {f"ft_FL10_C_mix_seed{s}": -1 for s in (1, 2, 3)}
    tv = _scores(tmp_path / "tv.json", {**runs_a, **better})
    dn1 = _scores(tmp_path / "devnative_strict_ink.json", {**runs_a, **better}, n=100, seed=1, tag="s")
    dn2 = _scores(tmp_path / "devnative_tv_native_grad.json", {**runs_a, **better}, n=80, seed=2, tag="t")
    merged = S2.merge([dn1, dn2])
    assert merged["dev"]["lines"] == 180 and merged["lines"][0]["id"].startswith("devnative_strict_ink:")
    assert len(merged["models"]["ft_FL10_A_s2_seed1"]["errors"]["rv"]) == 180
    (tmp_path / "dn.json").write_text(json.dumps(merged))
    out = tmp_path / "r.json"
    args = type("A", (), {"protocol": ROOT / "eval" / "synth3_study2_protocol.json", "tv": [tv],
                          "dn": str(tmp_path / "dn.json"), "dn_commit": "abc", "reported": "",
                          "reported_scores": [], "out": out})()
    S2.cmd_result(args)
    r = json.loads(out.read_text())
    assert list(r["comparisons"]) == ["FL10_C_vs_A"]           # H15 runs are not scored
    d = r["comparisons"]["FL10_C_vs_A"]["decision"]
    assert d["better"] == "C_mix" and d["c_better"]["devnative_ci_side"] and not d["a_better"]["holds"]
    same = {f"ft_FL10_C_mix_seed{s}": 0 for s in (1, 2, 3)}
    tv0 = _scores(tmp_path / "tv0.json", {**runs_a, **same})
    args.tv, args.dn, args.out = [tv0], "", tmp_path / "r0.json"
    S2.cmd_result(args)
    d0 = json.loads((tmp_path / "r0.json").read_text())["comparisons"]["FL10_C_vs_A"]["decision"]
    assert d0["better"] is None and d0["recommended"] == "A_s2"


def test_study2_reported_spec_is_consistent():
    rep = json.loads((ROOT / "eval" / "synth3_study2_reported.json").read_text(encoding="utf-8"))
    p = json.loads((ROOT / "eval" / "synth3_study2_protocol.json").read_text(encoding="utf-8"))
    for arm, runs in p["arms"].items():
        assert rep["arms"][arm] == runs
    for a, b in rep["comparisons"].values():
        assert a in rep["arms"] and b in rep["arms"] and len(rep["arms"][a]) == len(rep["arms"][b])
    assert rep["arms"]["curve_1M"] == ["B_s3_seed1", "B_s3_seed2"]


# ----------------------------------------------------------------------------- version 3

def test_version_3_cfg_and_calibration_overrides():
    assert synth3.CFG["version"] == 3
    for key in ("platform", "layout", "crop"):
        assert abs(sum(synth3.CFG[key].values()) - 1.0) < 1e-9
    for group in synth3.CFG["box_alpha"].values():
        assert set(group) == {"on_dark", "light", "on_colour"}
    merged = synth3.merge_cfg(synth3.CFG, {"platform": {"vita": 0.5}, "latin_keep": 0.0})
    assert merged["platform"]["vita"] == 0.5 and merged["platform"]["psp"] == synth3.CFG["platform"]["psp"]
    assert merged["latin_keep"] == 0.0 and synth3.CFG["latin_keep"] != 0.0       # the base is not modified
    with pytest.raises(KeyError):
        synth3.merge_cfg(synth3.CFG, {"platfrom": {}})


def test_pages_show_any_page_of_overflowing_text():
    """Text longer than the box continues on later pages; a render may show any of them (v2 only
    ever showed the first, so a long utterance never reached its closing 」)."""
    lines = [f"l{i}" for i in range(7)]
    assert synth3.paginate(lines[:3], 3, np.random.default_rng(0)) == (lines[:3], 0, 1)
    seen = set()
    for seed in range(200):
        page, k, n = synth3.paginate(lines, 3, np.random.default_rng(seed))
        assert n == 3 and page == lines[3 * k:3 * k + 3]
        seen.add(k)
    assert seen == {0, 1, 2}


def test_pool_lines_with_latin_are_mostly_redrawn(assets, tmp_path):
    synth3.init(assets=assets, font_manifest=assets / "fonts_manifest.json", scenes=False)
    path = tmp_path / "pool.txt"
    path.write_text("".join(("ＯＫだよ" if i % 2 else "そうだね") + "\n" for i in range(400)), encoding="utf-8")
    synth3.R.pools["mixed"] = synth3.TextPool(path)
    saved = dict(synth3.CFG)
    try:
        def latin_share(keep):
            synth3.CFG["latin_keep"] = keep
            rng = np.random.default_rng(3)
            return np.mean([bool(synth3.LATIN.search(synth3.pool_line(rng, "mixed"))) for _ in range(2000)])
        assert latin_share(1.0) > 0.4
        assert latin_share(0.0) < 0.01
        assert 0.1 < latin_share(0.25) < 0.3          # kept 1/4 of the time, else redrawn
    finally:
        synth3.CFG.clear()
        synth3.CFG.update(saved)
        del synth3.R.pools["mixed"]


def test_ui_words_split_by_latin():
    assert set(synth3.UI_LATIN) | set(synth3.UI_JAPANESE) == set(synth3.UI_WORDS)
    assert not set(synth3.UI_LATIN) & set(synth3.UI_JAPANESE)
    assert "SAVE" in synth3.UI_LATIN and "Q.セーブ" in synth3.UI_LATIN and "セーブ" in synth3.UI_JAPANESE
    assert all(not synth3.LATIN.search(w) for w in synth3.UI_PLACES)
    assert synth3.LATIN.search("ＣＧ") and synth3.LATIN.search("３") and not synth3.LATIN.search("」")


def _region(font, extra):
    """A one-line region (and its labelled layout); `extra` adds unlabelled text in the same band,
    which makes the GRAD finder's box wider than the line, so no GRAD crop may be taken."""
    from PIL import Image, ImageDraw, ImageFont
    f = ImageFont.truetype(str(font), 24)
    im = Image.new("RGB", (520, 60), (20, 20, 30))
    d = ImageDraw.Draw(im)
    d.text((16, 40), "あいうえお", font=f, fill=(240, 240, 240), anchor="ls")
    if extra:
        d.text((150, 40), "かきくけこさしすせそ", font=f, fill=(240, 240, 240), anchor="ls")
    m = Image.new("L", (520, 60), 0)
    ImageDraw.Draw(m).text((16, 40), "あいうえお", font=f, fill=255, anchor="ls")
    ink = np.asarray(m) >= 128
    cols = np.flatnonzero(ink.any(axis=0))
    lay = [{"text": "あいうえお", "top": 40 - 0.88 * 24, "fs": 24, "y0": 40 - 0.88 * 24 - 1,
            "y1": 40 + 0.12 * 24 + 1, "ix0": int(cols[0]), "ix1": int(cols[-1]) + 1, "kind": "dialogue"}]
    return np.asarray(im), lay, ink


def test_grad_mode_falls_back_to_the_gt_crop(font, monkeypatch):
    """Version 3: a line whose GRAD box fails the one-line check still yields its gt crop (kind
    "gt"), so lines beside clutter are not dropped from GRAD regions; v2 behaviour without it."""
    img, lay, ink = _region(font, False)
    got = synth3.extract(img, lay, ink, np.random.default_rng(0), "grad", 4)
    assert [(g[1], g[5]) for g in got] == [("あいうえお", "grad")]
    img, lay, ink = _region(font, True)
    got = synth3.extract(img, lay, ink, np.random.default_rng(0), "grad", 4)
    assert [(g[1], g[5]) for g in got] == [("あいうえお", "gt")]
    x, y, w, h = got[0][2]
    assert x + w <= lay[0]["ix1"] + 4 + synth3.CFG["pad_extra"]["vita"]["right"] * 24 + 1   # not the merged band
    monkeypatch.setitem(synth3.CFG, "grad_fallback", False)
    assert synth3.extract(img, lay, ink, np.random.default_rng(0), "grad", 4) == []


def test_gt_box_pads_the_fill_of_outlined_white_text():
    """eval's and teacher's boxes come from refine(), whose white-text mask finds the glyph fill
    (+1 px): the outline of white text sits inside the pad. A line flagged fillbox gets that box."""
    ink = np.zeros((60, 200), bool)
    ink[10:40, 20:120] = True                 # fill plus a 3 px outline
    fill = np.zeros_like(ink)
    fill[13:37, 23:117] = True                # the fill, dilated by 1 px
    lay = [{"y0": 9, "y1": 41, "fs": 26, "ix0": 20, "ix1": 120, "fillbox": True}]
    zero = {"p": 0.0, "top": 0, "bottom": 0, "left": 0, "right": 0}
    box, ib = synth3.gt_box(lay[0], lay, ink, np.zeros_like(ink), 4, np.random.default_rng(0), ink.shape,
                            extra=zero, fill=fill)
    assert ib == (23, 13, 117, 37) and box == (19, 9, 102, 32)
    lay[0]["fillbox"] = False                 # coloured text: refine() keeps the outline
    box, ib = synth3.gt_box(lay[0], lay, ink, np.zeros_like(ink), 4, np.random.default_rng(0), ink.shape,
                            extra=zero, fill=fill)
    assert ib == (20, 10, 120, 40) and box == (16, 6, 108, 38)


def test_exclusion_manifest_covers_the_fresh_native_titles():
    doc = json.loads((ROOT / "data" / "manifests" / "synth3_exclude_fresh_native.json").read_text(encoding="utf-8"))
    protected = set(doc["protected"]["psp"]) | set(doc["protected"]["vita"]) | set(doc["protected"]["reserves"])
    assert protected == {"v14909", "v25459", "v29454", "v21289", "v21976", "v20519", "v16509", "v19140"}
    related = {r["id"] for rs in doc["relations"].values() for r in rs}
    assert protected | related <= set(doc["vns"]) and set(doc["relations"]) == protected
    hexes = doc["screenshots"]["sha256"] + doc["screenshots"]["pixel_sha256"] + doc["names"]["name_sha256"]
    assert hexes and all(re.fullmatch(r"[0-9a-f]{64}", h) for h in hexes)
    assert len(doc["screenshots"]["pixel_sha256"]) == doc["screenshots"]["sources"] == 157
    assert re.fullmatch(r"[0-9a-f]{40}", doc["source"]["commit"])
    assert re.fullmatch(r"[0-9a-f]{64}", doc["source"]["sha256"])
    ex = synth3.load_exclusions([ROOT / "data" / "manifests" / "synth3_exclude_fresh_native.json"])
    assert ex["vns"] == set(doc["vns"]) and len(ex["pixel_sha256"]) == 157


def test_excluded_names_are_dropped(assets, tmp_path):
    """A name whose characters belong to an excluded VN never reaches a crop (by name hash)."""
    doc = {"kind": "synth3-exclusion", "schema": 1, "vns": ["v999999"],
           "screenshots": {"sha256": [], "pixel_sha256": []},
           "names": {"name_sha256": [synth3.name_hash(["佐藤", "花子"])]}}
    path = tmp_path / "exclude.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    r = synth3.Resources(assets=assets, font_manifest=assets / "fonts_manifest.json", scenes=False, exclude=[path])
    assert r.names_dropped == 1 and ("佐藤", "花子") not in r.names["vndb"] and ("アリス",) in r.names["vndb"]
    assert r.text_identity["names_dropped_excluded"] == 1


def test_pixel_hash_matches_the_fresh_harness_definition(tmp_path):
    from PIL import Image
    rgb = np.random.default_rng(0).integers(0, 255, (12, 16, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(tmp_path / "a.png")
    import hashlib
    assert synth3.pixel_sha256(tmp_path / "a.png") == hashlib.sha256(rgb.tobytes()).hexdigest()


def test_stats_measure_ink_the_same_way_for_every_source(tmp_path):
    import synth3_stats as S
    g = np.full((30, 80), 200, np.uint8)
    g[8:22, 10:70] = 30                                   # dark "text" 14 px tall
    assert S.px_ink(g) == (10, 8, 70, 22)
    busy, edge_ink, tex, tb = S.edge_parts(g)
    assert edge_ink == 0 and tex == 0 and busy == 0 and tb == 0
    g2 = g.copy()
    g2[0, 10:40] = 30                                     # a neighbour's ink in the top row
    assert S.edge_parts(g2)[1] > 0
    crops = [g, g2]
    shapes = np.array([c.shape for c in crops], np.int32)
    np.savez(tmp_path / "s.npz", data=np.concatenate([c.ravel() for c in crops]),
             offsets=np.concatenate([[0], np.cumsum(shapes[:, 0] * shapes[:, 1])]), shapes=shapes,
             labels=np.array(json.dumps(["あいう", "かきく"], ensure_ascii=False)),
             meta=np.array(json.dumps([{"real": True, "kind": "vita", "target": "vita", "vn": "v1"},
                                       {"real": True, "kind": "hires", "target": "vita", "vn": "v14909"}])))
    out = S.measure([str(tmp_path / "s.npz")], 0, np.random.default_rng(0), reserved={"v14909"})
    assert out["lines"] == 1 and out["reserved_rows_dropped"] == 1
    assert out["geometry"]["vita:vita"]["px"]["ink_h"][1] == 14 and "vita:hires" not in out["geometry"]
    assert "v14909" in S.reserved_vns(S.DEFAULT_EXCLUDE)
