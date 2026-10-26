"""T4 (vocr6-lm, #127): lm/build.py's VOCR6 options, lm/hook_text.py and eval/vocr6_lm.py's helpers, on generated
fixtures only (no hooked, development or protected data is read here)."""
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "lm", ROOT / "eval", ROOT / "runtime", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import hook_text as H  # noqa: E402
import memorization as M  # noqa: E402
import vlm  # noqa: E402
import vocr6_lm as V  # noqa: E402
from metrics import lev as lev_ref  # noqa: E402

import test_ctclm as T  # noqa: E402


def _load(name, path):
    """lm/build.py under its own module name: realdata/ has a build.py too (tests share sys.modules)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load("vocr6_lm_build", ROOT / "lm" / "build.py")

HOOK = ["かきくけこ、さし", "すせそ。", "ねえおけ、そう", "たちつてと", "すせそ。"]   # ねえおけ: a protected name


def sources_with_hook(tmp):
    """test_ctclm's generated sources plus a visual-lines corpus and an extra vocabulary."""
    src_path, held = T.sources(tmp)
    src = json.loads(src_path.read_text(encoding="utf-8"))
    hook = tmp / "hook.txt"
    hook.write_text("\n".join(HOOK * 50) + "\n", encoding="utf-8")
    src["corpora"]["hook"] = {"path": str(hook), "sha256": T.sha(hook), "licence": "test", "attribution": "generated",
                              "visual_lines": True}
    extra = tmp / "extra.txt"
    extra.write_text("ゑ\nゐ\n", encoding="utf-8")
    src["vocab_extra"] = {"x": {"path": str(extra), "sha256": T.sha(extra)}}
    tg = json.loads(Path(src["exclusions"]["test_games"]["path"]).read_text(encoding="utf-8"))
    tg.update({"vita-test": {"v1": {}}, "psp-test": {}})
    tgp = Path(src["exclusions"]["test_games"]["path"])
    tgp.write_text(json.dumps(tg), encoding="utf-8")
    src["exclusions"]["test_games"]["sha256"] = T.sha(tgp)
    src_path.write_text(json.dumps(src), encoding="utf-8")
    return src_path, held


def run_build(tmp, out, *extra, hook=True):
    src, held = sources_with_hook(tmp) if hook else T.sources(tmp)
    cmd = [sys.executable, str(ROOT / "lm/build.py"), "--out", str(out), "--manifest", str(out) + ".json",
           "--sources", str(src), "--jiten", "vn=0.2", "--order", "3", "--bytes", "60000",
           "--heldout", str(held), "--heldout-lines", "100", *extra]
    subprocess.run(cmd, check=True, capture_output=True, cwd=tmp)
    return out.read_bytes(), json.loads(Path(str(out) + ".json").read_text(encoding="utf-8"))


def test_defaults_are_the_old_build(tmp_path):
    """The new options at their defaults: the same bytes and manifest as test_ctclm's build of the same sources."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    old, man_old = T.build(tmp_path / "a", tmp_path / "a" / "lm.vlm")
    new, man_new = run_build(tmp_path / "b", tmp_path / "b" / "lm.vlm", "--corpora", "gen", "--protect", "ctclm2",
                             "--recut", "none", hook=False)
    assert new == old
    for k in ("recut", "weight", "visual_lines"):
        assert k not in json.dumps(man_new["corpora"]) and k not in man_new
    assert "protect" not in man_new["exclusion"] and "extra" not in man_new["vocab"]


def test_weights_hook_text_and_vocab(tmp_path):
    """A visual-lines corpus at weight 4 is counted 4 times and never filtered: its protected name ねえおけ stays (the
    public corpus' pieces holding it are dropped); the extra vocabulary's characters become tokens."""
    _, man = run_build(tmp_path, tmp_path / "lm.vlm", "--corpora", "gen,hook:4", "--protect", "vocr6",
                       "--vocab-extra", "x")
    lm = vlm.LM((tmp_path / "lm.vlm").read_bytes())
    hook = next(c for c in man["corpora"] if c["name"] == "hook")
    assert hook["weight"] == 4 and hook["visual_lines"] is True and hook["lines"] == 250
    assert man["exclusion"]["protect"] == "vocr6" and man["exclusion"]["visual_lines_exempt"] == 250
    assert man["exclusion"]["corpus_pieces_dropped"] >= 6
    assert lm.find(1, lm.token_of("ね"), lm.token_of("え")) != vlm.NONE       # from the exempt hook text
    assert lm.token_of("ゑ") != vlm.UNK and man["vocab"]["extra"] == {"x": T.sha(tmp_path / "extra.txt")}
    pub = sum(c["chars"] for c in man["corpora"] if c["name"] == "gen")
    kept_pub = man["exclusion"]["corpus_chars_kept"] - 4 * sum(len(s) for s in HOOK * 50)
    assert 0 < kept_pub <= pub


def test_protect_rules(tmp_path):
    src, _ = sources_with_hook(tmp_path)
    s = B.load_sources(src)
    assert B.protected_vns(s, "ctclm2") == {"v1", "v2"}
    assert B.protected_vns(s, "vocr6", ["v7"]) == {"v1", "v2", "v7"}       # vita-test v1, CLANNAD's v2, an extra
    tg = Path(s["exclusions"]["test_games"]["path"])
    tg.write_text(json.dumps({"all": ["v1", "v5"], "vita-test": {}, "psp-test": {}}), encoding="utf-8")
    s["exclusions"]["test_games"]["sha256"] = T.sha(tg)
    assert B.protected_vns(s, "ctclm2") == {"v1", "v2", "v5"} and B.protected_vns(s, "vocr6") == {"v2"}
    cl = Path(s["exclusions"]["clannad"]["path"])
    cl.write_text(json.dumps({"vns": ["v2", "v34"], "text_markers": {"strings": []}}), encoding="utf-8")
    s["exclusions"]["clannad"]["sha256"] = T.sha(cl)
    assert "v34" not in B.protected_vns(s, "vocr6") and "v34" in B.protected_vns(s, "ctclm2")
    with pytest.raises(SystemExit):
        B.protected_vns(s, "other")


def test_wrap_and_recut():
    s = "あいうえお。かきくけこ「さしすせそ」たちつてと"
    for w in (3, 4, 5, 7, 10, 30):
        parts = B.wrap(s, w)
        assert "".join(parts) == s
        assert all(len(p) <= w + B.HANG for p in parts)
        assert all(p[0] not in B.NO_START for p in parts[1:] if len(p) > 0) or w <= 2
        assert all(p[-1] not in B.NO_END for p in parts[:-1])
    assert B.wrap("あいう。」えお", 3) == ["あいう。」", "えお"]                # hung punctuation
    assert B.wrap("あい「うえお", 3) == ["あい", "「うえ", "お"]                # an opener never ends a line
    lines, w = B.recut(["あ" * 50, "い" * 5], [2, 3], [16], "lines")
    assert lines == ["あ" * 16] * 3 + ["あ" * 2, "い" * 5] and w == [2, 2, 2, 2, 3]
    lines, w = B.recut(["あ" * 20], [1], [16], "both")
    assert lines == ["あ" * 20, "あ" * 16, "あ" * 4] and w == [1, 1, 1]
    assert B.recut(["x"], [1], [16], "none") == (["x"], [1])
    a = B.recut(["あいうえおかきくけこさしすせそ" * 3], [1], [8, 12, 16], "lines")
    assert a == B.recut(["あいうえおかきくけこさしすせそ" * 3], [1], [8, 12, 16], "lines")   # deterministic


def test_hook_text_select():
    recs = [{"id": "T:000002:line1", "page": "000002", "kind": "line", "text": "かきく"},
            {"id": "T:000001:line1", "page": "000001", "kind": "line", "text": "あいう"},
            {"id": "T:000001:name2", "page": "000001", "kind": "name", "text": "さし"},
            {"id": "T:000003:line1", "page": "000003", "kind": "line", "text": "あいう"},
            {"id": "T:000004:line1", "page": "000004", "kind": "line", "text": "ゑゑ"},
            {"id": "T:000005:line1", "page": "000005", "kind": "line", "text": "たち"},
            {"id": "T:000006:line1", "page": "000006", "kind": "ruby", "text": "つ"}]
    assign = {"000001": "train", "000002": "train", "000003": "train", "000004": "train", "000005": "hval",
              "000006": "train"}
    vocab = set("あいうかきくさしたちつ")
    got, c = H.select(recs, assign, "train", vocab)
    assert got == ["あいう", "かきく"]                          # page order, the repeat once, name/ruby/oov out
    assert (c["kind"], c["duplicate"], c["oov"], c["page"], c["kept"]) == (2, 1, 1, 1, 2)
    assert H.select(recs, assign, "hval", vocab)[0] == ["たち"]
    assert H.select(recs, assign, "final", vocab)[0] == ["あいう", "かきく", "たち"]
    assign["000005"] = "guard"
    assert H.select(recs, assign, "train", vocab)[0] == ["あいう", "かきく"]
    assert "たち" in H.select(recs, assign, "final", vocab)[0]


def test_fast_lev_equals_metrics_lev():
    rng = random.Random(0)
    alpha = "あいう。、「」漢字ーx"
    for _ in range(20000):
        a = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 14)))
        b = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 14)))
        assert V.lev(a, b) == lev_ref(a, b)
    for _ in range(200):
        a = "".join(rng.choice(alpha) for _ in range(rng.randint(60, 130)))
        b = list(a)
        for _ in range(rng.randint(0, 12)):
            i = rng.randint(0, max(0, len(b) - 1))
            r = rng.random()
            if r < 0.33 and b:
                b[i] = rng.choice(alpha)
            elif r < 0.66:
                b.insert(i, rng.choice(alpha))
            elif b:
                del b[i]
        b = "".join(b)
        assert V.lev(a, b) == lev_ref(a, b)


def test_units_cascade_and_bootstrap():
    texts = {"a": "あいう", "b": "かき", "c": "x", "d": ""}
    u = V.Unit("wd", "u1", "v1", False, [("あいうかき", ["a", "b"])])
    assert u.chars == 5 and u.errors(texts) == 0
    texts["b"] = "かく"
    assert u.errors(texts) == 1 and len(u.memo) == 2
    two = V.Unit("tv", "u2", "v2", False, [("x", ["c"]), ("x", ["d"])])
    assert two.chars == 2 and two.errors(texts) == 1
    conf = {"a": 699, "b": 700}
    assert V.cascade({"a": "1", "b": "2"}, conf, {"a": "U", "b": "V"}, 700) == {"a": "U", "b": "2"}
    assert V.cascade({"a": "1"}, conf, None, 700) == {"a": "1"}
    assert V.line_conf({"text": "", "conf_min": 0.1}) == 1000
    assert V.line_conf({"text": "a", "conf_min": 0.7}) == 700
    units = [V.Unit("wd", f"w{i}", f"v{i % 7}", False, [("あいうえお", [f"k{i}"])]) for i in range(70)]
    ex = np.array([1] * 70)
    ey = np.array([0] * 70)
    cmp = V.compare(units, ex, ey, n_boot=500)
    assert cmp["wd"]["delta_points"] == -20.0 and cmp["wd"]["ci95"] == [-20.0, -20.0] and cmp["wd"]["clusters"] == 7
    assert "dev" not in cmp
    assert V.grid_configs({"b": [1, 2], "a": [3]}) == [{"a": 3, "b": 1}, {"a": 3, "b": 2}]
    assert V.fold_of("v12") in (0, 1) and V.fold_of("v12") == int(hashlib.sha256(b"v12").hexdigest(), 16) % 2



def test_memorisation_argmax_and_recovery(tmp_path):
    """Next.argmax is the brute-force argmax of LM.score over every token; a line the LM was built from many times
    is recovered further by greedy continuation than random text; kept_ngrams lists every entry."""
    blob, _ = T.build(tmp_path, tmp_path / "lm.vlm")
    lm = vlm.LM(blob)
    nx = M.Next(lm)
    st = lm.initial(vlm.FLAG_BOS)
    for ch in "あいうかき":
        toks = [t for t in range(vlm.FIRST, lm.V)] + [vlm.EOS]
        brute = max(toks, key=lambda t: (lm.score(st, t)[0], -t))
        assert nx.argmax(st) == brute
        st = lm.score(st, lm.token_of(ch))[1]
    kept = M.kept_ngrams(lm)
    assert [len(kept[k]) for k in range(1, lm.order + 1)] == lm.n
    r = M.recovery(lm, ["あいうかきく", "ほへせそ"])
    assert r["lines"] == 2 and 0 <= r["next_char_exact_share"] <= 1 and r["longest_run"]["max"] >= 1
    assert sum(r["longest_run_by_length"].values()) == 2


def test_display_rules_filter():
    assert V.is_japanese_char("あ") and V.is_japanese_char("ア") and V.is_japanese_char("漢")
    assert not V.is_japanese_char("1") and not V.is_japanese_char("Ａ") and not V.is_japanese_char("」")


RUBI = Path(os.environ.get("RUBI_DIR", "/home/skerraut/herd-synth3/clones/vocr6-lm-rubi/Rubi_for_vita"))


@pytest.mark.skipif(not (RUBI / "core" / "textnorm.c").exists(), reason="no Rubi_for_vita checkout (RUBI_DIR)")
def test_rubi_show(tmp_path):
    try:
        sh = V.RubiShow(RUBI, tmp_path)
    except SystemExit:
        pytest.skip("the Rubi checkout is not at the pinned textnorm.c")
    assert sh(["ーニーーーーニーニーーーー"]) == []                       # the border rule (R08d)
    assert sh(["テキスト", "ーニーーーニー"]) == ["テキスト"]
    assert sh(["Ab12Cd34Ef56", "「これはテスト"]) == ["「これはテスト"]      # the Shell's non-Japanese filter
    assert sh(["", "あ", ""]) == ["あ"] and sh(["コ一一"]) == ["コーー"] and sh(["一二三"]) == ["一二三"]


def test_sub_compare_is_compare_on_the_kept_units():
    """checks' sensitivity helper: dropping units equals comparing the kept units alone."""
    rng = np.random.default_rng(0)
    units = []
    for i in range(60):
        part = ("tv", "dn", "wd", "hc", "hp")[i % 5]
        u = V.Unit(part, f"u{i}", f"c{i % 12}", False, [], "vita")
        u.chars = 10
        units.append(u)
    x, y = rng.integers(0, 4, 60), rng.integers(0, 4, 60)
    drop = np.zeros(60, bool)
    drop[::7] = True
    keep = np.flatnonzero(~drop)
    assert V._sub_compare(units, x, y, drop) == V.compare([units[i] for i in keep], x[keep], y[keep])


def _fake_tune(path, units, configs, errs, lm="x", lm_path=None, cascade=(500, 600, 700)):
    meta = {"lm": lm, "lm_path": lm_path, "lm_sha256": None, "configs": configs, "cascade": list(cascade),
            "units": units}
    np.savez_compressed(path, meta=json.dumps(meta), **{k: np.asarray(v, np.int16) for k, v in errs.items()})


def test_select_cross_validates(tmp_path):
    """Two recipes, two settings: recipe b's setting 1 has fewer errors on every unit, so both folds choose it, and
    the cross-validated errors are its errors; against an incumbent with more errors the rule's (a)-(c) hold."""
    import argparse
    units = []
    for i in range(80):
        part = ("tv", "dn", "wd", "hp")[i % 4]
        units.append([part, f"u{i}", f"c{i % 16}", False, 10, "vita"])
    n = len(units)
    base = np.full((1, n), 3)
    keys = [f"{m}|{t}" for m in ("S", "N") for t in (500, 600, 700, 800, 900)] + ["S|none", "N|none", "L|none"]
    _fake_tune(tmp_path / "inc.npz", units, [{"alpha": 500, "beta": 3500}], {k: base for k in keys}, "inc")
    _fake_tune(tmp_path / "greedy.npz", units, [{"alpha": 0, "beta": 0}], {k: base + 1 for k in keys}, "greedy")
    a = np.stack([np.full(n, 2), np.full(n, 2)])
    b = np.stack([np.full(n, 2), np.full(n, 1)])
    _fake_tune(tmp_path / "a.npz", units, [{"alpha": 400, "beta": 2000}, {"alpha": 400, "beta": 2500}],
               {k: a for k in keys}, "a", cascade=(500, 600, 700, 800, 900))
    _fake_tune(tmp_path / "b.npz", units, [{"alpha": 400, "beta": 2000}, {"alpha": 400, "beta": 2500}],
               {k: b for k in keys}, "b", cascade=(500, 600, 700, 800, 900))
    out = tmp_path / "sel.json"
    V.cmd_select(argparse.Namespace(protocol=str(ROOT / "eval" / "vocr6_lm_protocol.json"), tier="1000k",
                                    incumbent=str(tmp_path / "inc.npz"), greedy=str(tmp_path / "greedy.npz"),
                                    tune=[str(tmp_path / "a.npz"), str(tmp_path / "b.npz")], path="shown",
                                    out=str(out)))
    r = json.loads(out.read_text(encoding="utf-8"))
    for f in ("fold0", "fold1", "in_sample"):
        assert r["chosen"][f]["lm"] == "b" and r["chosen"][f]["config"]["beta"] == 2500
        assert r["chosen"][f]["cascade_conf"] == 500                           # ties go to the first threshold
    n_cmp = r["models"]["N"]["cv_minus_incumbent"]
    assert n_cmp["hp"]["delta_points"] == -20.0 and n_cmp["hp"]["ci95"] == [-20.0, -20.0]
    assert r["rule"]["a_hp_better"] and r["rule"]["b_dev_not_worse"] and r["rule"]["c_S_L_not_worse"]
    assert r["rule"]["d_bytes"] is None and not r["rule"]["adopted"]          # no blob: (d) cannot be checked


def test_doc_tables_match_the_committed_result(tmp_path):
    """docs/vocr6-lm.md's tables are eval/vocr6_lm.py tables of eval/vocr6_lm_dev_20261026.json, and what ships fits
    the incumbents' bytes and Rubi's decoder workspace."""
    import argparse
    doc = ROOT / "docs" / "vocr6-lm.md"
    res = ROOT / "eval" / "vocr6_lm_dev_20261026.json"
    copy = tmp_path / "doc.md"
    copy.write_text(doc.read_text(encoding="utf-8"), encoding="utf-8")
    V.cmd_tables(argparse.Namespace(json=str(res), doc=str(copy)))
    assert copy.read_text(encoding="utf-8") == doc.read_text(encoding="utf-8")
    j = json.loads(res.read_text(encoding="utf-8"))
    proto = json.loads((ROOT / "eval" / "vocr6_lm_protocol.json").read_text(encoding="utf-8"))
    assert [x["n"] for x in proto["amendments"]] == [1, 2]
    for tier, s in j["shipped"].items():
        assert s["bytes"] <= proto["incumbent"]["lms"][tier]["bytes"]
    for tier in ("1000k", "4000k"):
        key = "B3" if "B3" in j["tiers"][tier] else "B"
        rule = j["tiers"][tier][key]["rule"]
        assert rule["adopted"] and max(rule["workspace_bytes"].values()) <= 49152
        chosen = j["tiers"][tier][key]["chosen"]["in_sample"]["config"]
        assert j["shipped"][tier]["config"] == dict({"beam": 8, "topk": 8}, **chosen)
    for tier in ("500k", "2000k"):
        assert j["tiers"][tier]["AB"]["rule"]["ships"] == "incumbent" and j["shipped"][tier]["family"] == "incumbent"

def test_settings_report_picks_the_seed_mean_best():
    """Another recognizer's settings: setting 1 has fewer errors on every unit for both seeds, so every fold and the
    in-sample choice pick it; its CV errors are its errors; the ref (setting 0) comparison is the paired difference."""
    units = []
    for i in range(60):
        part = ("tv", "dn", "wd", "hc", "hp")[i % 5]
        units.append([part, f"u{i}", f"c{i % 12}", False, 10, "vita"])
    n = len(units)
    cfgs = [{"alpha": 500, "beta": 3500, "flags": 2, "gate": 8000, "bracket": 8000},
            {"alpha": 600, "beta": 2500, "flags": 1, "gate": 8000, "bracket": 8000}]
    errs = {s: np.stack([np.full(n, 3), np.full(n, 1)]) for s in ("1", "3")}
    greedy = {s: np.full(n, 4) for s in ("1", "3")}
    r = V.settings_report(units, greedy, errs, cfgs, ref=cfgs[0])
    assert r["chosen"] == {"fold0": cfgs[1], "fold1": cfgs[1], "in_sample": cfgs[1]}
    for s in ("1", "3"):
        ps = r["per_seed"][s]
        assert ps["cv"]["hc"] == 10.0 and ps["greedy"]["hc"] == 40.0 and ps["vocr5_setting"]["hc"] == 30.0
        assert ps["cv_minus_vocr5_setting"]["hc"]["delta_points"] == -20.0
    assert r["by_alpha"]["600"]["config"] == cfgs[1]
