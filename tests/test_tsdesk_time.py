"""textspot-desktop round 2's goal timing (eval/tsdesk_time.py goal, issue #173): the interleaved schedule (every
pipeline on a page before the next page, shuffled, seeded), the goal blocks eval/tsdesk_goal.py reads (R's and the
ONNX page's budget variant per view, the page path the goal takes) and the TIMING JSON's keys. No data and no model:
stub pipelines, synthetic frames and a fake clock."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def TT():
    return _load("tsdesk_time", ROOT / "eval" / "tsdesk_time.py")


@pytest.fixture(scope="module")
def G():
    return _load("tsdesk_goal", ROOT / "eval" / "tsdesk_goal.py")


class FakeClock:
    """A clock the stub pipelines advance: each call takes a set number of ms."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def take(self, ms):
        self.t += ms / 1e3


# ----------------------------------------------------------------------------- the schedule

def test_interleave_is_round_robin_per_page_and_seeded(TT):
    names = ["a", "b", "c", "d", "e"]
    n_pages, passes, warmup = 9, 3, 1
    clock = FakeClock()
    calls = []

    def call(n, i):
        calls.append((n, i))
        clock.take(10.0 * (1 + names.index(n)) + i)        # a's time 10 + i ms, b's 20 + i, ...
        return i % 3

    times, items, sched = TT.interleave(n_pages, names, call, passes, warmup, seed=7, clock=clock)
    assert [(n, i) for _, i, n in sched] == calls
    assert len(sched) == (warmup + passes) * n_pages * len(names)
    page_orders = []
    for p in range(warmup + passes):
        rows = [(i, n) for q, i, n in sched if q == p]
        pages = [rows[k][0] for k in range(0, len(rows), len(names))]
        assert sorted(pages) == list(range(n_pages))      # every page once per pass
        page_orders.append(pages)
        for k in range(0, len(rows), len(names)):         # a page's pipelines run together, each once
            block = rows[k:k + len(names)]
            assert {i for i, _ in block} == {block[0][0]} and sorted(n for _, n in block) == names
    assert len({tuple(o) for o in page_orders}) > 1       # shuffled per pass
    blocks = [tuple(n for _, _, n in sched[k:k + len(names)]) for k in range(0, len(sched), len(names))]
    assert len(set(blocks)) > 1                           # the pipelines shuffled per page
    for n in names:                                       # the warm-up pass is not recorded
        assert sorted(times[n]) == list(range(n_pages)) and all(len(v) == passes for v in times[n].values())
        for i, v in times[n].items():
            assert v == pytest.approx([10.0 * (1 + names.index(n)) + i] * passes)
            assert items[n][i] == i % 3
    again = TT.interleave(n_pages, names, lambda n, i: clock.take(1.0), passes, warmup, seed=7, clock=clock)[2]
    assert again == sched                                 # the same seed, the same schedule
    other = TT.interleave(n_pages, names, lambda n, i: clock.take(1.0), passes, warmup, seed=8, clock=clock)[2]
    assert other != sched


def test_interleave_keeps_run_s_schedule(TT):
    """cmd_run's loop before it used interleave(): the same shuffles in the same order (random.Random(seed + threads):
    the pages once per pass, the pipelines once per page)."""
    import random
    names, n_pages, seed = ["x|t1", "y|t1", "z|t1"], 6, 20261102
    rng = random.Random(seed)
    order, nm, want = list(range(n_pages)), list(names), []
    for p in range(4):
        rng.shuffle(order)
        for i in order:
            rng.shuffle(nm)
            want += [(p, i, n) for n in nm]
    assert TT.interleave(n_pages, names, lambda n, i: 0, 3, 1, seed)[2] == want


def test_view_stats(TT):
    times = {0: [3.0, 1.0, 2.0], 1: [5.0, 5.0, 4.0], 2: [9.0, 9.0, 9.0]}
    items = {0: 1, 1: 3, 2: 2}
    st = TT.view_stats(times, items, ["vita", "vita", "psp"])
    assert st["vita"]["n"] == 2 and st["vita"]["p50"] == pytest.approx(3.5) and st["vita"]["items_mean"] == 2.0
    assert st["psp"]["p50"] == 9.0 and st["psp"]["p95"] == 9.0


# ----------------------------------------------------------------------------- the goal blocks

def _st(p50, p95=None):
    return {"p50": p50, "p95": p50 * 1.1 if p95 is None else p95, "mean": p50, "n": 60, "items_mean": 3.0}


def _stats(name="q8b", c=(18.0, 15.0, 21.0), o=(16.0, 14.0, 40.0)):
    views = ("vita", "psp", "pc1080", "vita1080")
    s = {"det1:meiki": {v: _st(105.0) for v in views},
         f"det1:cand:{name}": {v: _st(14.0 + k) for k, v in enumerate(views)}}
    for k, base in ((1, 140.0), (2, 100.0), (4, 74.0)):            # R: the 4-thread variant fastest, but not on psp
        s[f"page4:R#d{k}"] = {v: _st(base + (40.0 if (k == 4 and v == "psp") else 0.0)) for v in views}
    s[f"page4:c:{name}"] = {"vita": _st(c[0]), "psp": _st(c[1]), "pc1080": _st(c[2]), "vita1080": _st(c[2])}
    for k, extra in ((1, 3.0), (2, 0.0), (4, 1.0)):
        s[f"page4:onnx:{name}#d{k}"] = {"vita": _st(o[0] + extra), "psp": _st(o[1] + extra),
                                        "pc1080": _st(o[2] + extra), "vita1080": _st(o[2] + extra)}
    return s


def test_goal_block_keys_variants_and_path(TT, G, tmp_path):
    blk = TT.goal_block(_stats(), "q8b")
    assert set(blk["det_1t"]) == {"cand", "meiki"}
    for who in ("cand", "meiki"):
        assert {"vita", "psp", "pc1080"} <= set(blk["det_1t"][who]) and set(blk["det_1t_p95"][who]) == \
            set(blk["det_1t"][who])
    assert blk["det_1t"]["cand"]["vita"] == 14.0 and blk["det_1t"]["meiki"]["vita"] == 105.0
    assert set(blk["page_4t"]) == {"cand", "R"} and {"vita", "psp", "pc"} <= set(blk["page_4t"]["R"])
    assert blk["page_4t_R_variant"] == {"vita": "d4", "psp": "d2", "pc": "d4", "vita1080": "d4"}
    assert blk["page_4t"]["R"]["psp"] == 100.0 and blk["page_4t"]["R"]["vita"] == 74.0
    assert blk["page_4t_onnx_variant"]["vita"] == "d2" and blk["page_4t_paths"]["onnx"]["vita"] == 16.0
    # worst view fraction of R: C max(18/74, 15/100, 21/74) = 0.2838; ONNX max(16/74, 14/100, 40/74) = 0.5405
    assert blk["page_4t_path"] == "c" and blk["page_4t"]["cand"] == blk["page_4t_paths"]["c"]
    assert blk["page_4t_paths_worst_frac_of_R"]["c"] == pytest.approx(21 / 74, abs=1e-4)
    assert blk["interleaved"] is True and blk["ratios"]["det_1t_meiki_over_cand"]["vita"] == 7.5
    # the ONNX page when it is the faster path everywhere
    o = TT.goal_block(_stats(c=(30.0, 30.0, 30.0), o=(12.0, 12.0, 12.0)), "q8b")
    assert o["page_4t_path"] == "onnx" and o["page_4t"]["cand"]["pc"] == 12.0
    # the rule as code reads the block: G2 (14 <= 105 / 5) and G3 (every view <= R / 3) pass
    p = tmp_path / "t.json"
    p.write_text(json.dumps(blk))
    g2, g3 = G.speed(str(p))
    assert g2["pass"] and g3["pass"] and g2["ratio"] == 7.5
    slow = TT.goal_block(_stats(c=(30.0, 15.0, 21.0), o=(40.0, 40.0, 40.0)), "q8b")
    p.write_text(json.dumps(slow))
    assert not G.speed(str(p))[1]["pass"]                 # C 30 > 74 / 3 on vita, ONNX slower: G3 fails


def test_goal_block_unread_parts(TT):
    """A run without the pages (e.g. --views vita only) leaves those keys out: the rule reads them as unread."""
    st = {k: {"vita": v["vita"]} for k, v in _stats().items()}
    blk = TT.goal_block(st, "q8b")
    assert blk["page_4t_path"] is None and blk["page_4t"]["cand"] == {} and blk["det_1t"]["cand"] == {"vita": 14.0}


def test_parse_cand_and_pipelines(TT):
    c = TT.parse_cand("q8b=/b/q8b.vtsd,/b/q8b.onnx", "long:960", "k6v4")
    assert c == {"name": "q8b", "blob": "/b/q8b.vtsd", "onnx": "/b/q8b.onnx", "policy": "long:960", "rule": "k6v4"}
    c2 = TT.parse_cand("x=/b/x.vtsd,/b/x.onnx,fit:960,h0.45v0.1", "long:960", "k6v4")
    assert c2["policy"] == "fit:960" and c2["rule"] == "h0.45v0.1"
    for bad in ("q8b", "q8b=/b/q8b.vtsd", "=a,b", "a b=x,y", "q=x,y,z,w,v"):
        with pytest.raises(ValueError):
            TT.parse_cand(bad, "long:960", "k6v4")
    jobs = TT.goal_pipelines([c, c2])
    keys = [k for k, *_ in jobs]
    assert len(keys) == len(set(keys)) and keys[:4] == ["det1:meiki", "page4:R#d1", "page4:R#d2", "page4:R#d4"]
    d = {k: (spec, th, pad) for k, spec, th, pad in jobs}
    assert d["det1:meiki"] == ("meikidet", 1, "auto") and d["page4:R#d2"] == ("rpage#d2", 4, "auto")
    assert d["det1:cand:x"] == ("cbox:/b/x.vtsd@fit:960+refine", 1, "h0.45v0.1")
    assert d["page4:c:q8b"] == ("cpage:/b/q8b.vtsd@long:960+refine+c:L", 4, "k6v4")
    assert d["page4:onnx:q8b#d4"] == ("opage:/b/q8b.onnx@long:960@exact+refine+L#d4", 4, "k6v4")
    for spec, _, _ in d.values():                          # every spec parses
        base = spec.partition("#d")[0]
        if base not in ("meikidet", "rpage"):
            TT.split_spec(base)


# ----------------------------------------------------------------------------- the TIMING JSON

class StubPipe:
    """A stand-in for tsdesk_time.Pipeline: a fixed time per view (ms) on the fake clock and a fixed item count."""
    clock = None
    ms = {}

    def __init__(self, spec, threads, pad="auto", lib=None, onnx_dir=None, **kw):
        self.spec, self.threads, self.pad = spec, threads, pad

    def __call__(self, rgb, i=None):
        w = rgb.shape[1]
        view = {960: "vita", 480: "psp", 1920: "pc1080"}[w]
        kind = self.spec.split(":")[0].split("#")[0]
        k = int(self.spec.rpartition("#d")[2]) if "#d" in self.spec else 4
        base = {"meikidet": 105.0, "rpage": {1: 140.0, 2: 100.0, 4: 74.0}[k], "cbox": 14.0, "cpage": 20.0,
                "opage": 25.0 + k}[kind]
        StubPipe.clock.take(base * (1.3 if view == "pc1080" else 1.0))
        return 2


def _fake_pages(n=(4, 3, 2)):
    pages = []
    for (view, (w, h)), k in zip((("vita", (960, 544)), ("psp", (480, 272)), ("pc1080", (1920, 1080))), n):
        pages += [{"key": f"{view}:{j}", "view": view, "W": w, "H": h} for j in range(k)]
    return pages


@pytest.fixture
def stubbed(TT, monkeypatch):
    clock = FakeClock()
    StubPipe.clock = clock
    monkeypatch.setattr(TT, "Pipeline", StubPipe)
    monkeypatch.setattr(TT, "load_pages", lambda path, random_n=0: ({"source": "test"}, _fake_pages()))
    monkeypatch.setattr(TT, "load_rgb", lambda p: np.zeros((p["H"], p["W"], 3), np.uint8))
    monkeypatch.setattr(TT, "host", lambda: {"cpu": "test", "load_1min_at_start": 1.5})
    monkeypatch.setattr(TT, "_lib_record", lambda lib: {"lib": lib, "simd": 1, "build": {"arch": "-march=x86-64-v3"}})
    monkeypatch.setattr(TT.D, "sha256_file", lambda p: "0" * 64)
    monkeypatch.setattr(TT.D, "git_head", lambda: "f" * 40)
    real = TT.interleave
    monkeypatch.setattr(TT, "interleave", lambda *a, **k: real(*a, **dict(k, clock=clock)))
    return clock


def test_goal_json_one_candidate(TT, G, stubbed, tmp_path):
    out = tmp_path / "goal.json"
    TT.main(["goal", "--pages", str(tmp_path / "pages.pkl"), "--cand", "q4b=/x/q4b.vtsd,/x/q4b.onnx", "--lib",
             "/x/lib.so", "--mem-views", "", "--out", str(out)])
    t = json.loads(out.read_text())
    assert t["interleaved"] is True and t["schema"] == TT.GOAL_SCHEMA
    for k in ("det_1t", "det_1t_p95", "page_4t", "page_4t_p95", "page_4t_path", "page_4t_paths", "page_4t_R_variant",
              "page_4t_onnx_variant", "host", "load_1min_before", "load_1min_after", "c", "passes", "warmup", "seed",
              "pages", "pages_sha256", "repo_head", "candidate", "policy", "rule", "blob_sha256", "onnx_sha256",
              "results", "pipelines", "what", "timing_files"):
        assert k in t, k
    assert t["det_1t"]["cand"] == {"vita": 14.0, "psp": 14.0, "pc1080": 18.2}
    assert t["det_1t"]["meiki"]["vita"] == 105.0
    assert t["page_4t"]["R"] == {"vita": 74.0, "psp": 74.0, "pc": 96.2}
    assert t["page_4t_path"] == "c" and t["page_4t"]["cand"]["vita"] == 20.0
    assert t["page_4t_onnx_variant"]["vita"] == "d1" and t["page_4t_paths"]["onnx"]["vita"] == 26.0
    assert t["pages"] == {"pc1080": 2, "psp": 3, "vita": 4} and t["passes"] == 3 and t["warmup"] == 1
    assert t["c"]["simd"] == 1 and t["policy"] == "long:960" and t["rule"] == "k6v4"
    assert t["schedule"]["calls"] == 4 * 9 * 9           # 4 passes x 9 pages x 9 pipelines
    assert t["timing_files"] == {"q4b": str(out)}
    g2, g3 = G.speed(str(out))
    assert g2["pass"] and g3["pass"] and g2["interleaved"] and g3["interleaved"]


def test_goal_json_several_candidates(TT, G, stubbed, tmp_path):
    out = tmp_path / "run.json"
    TT.main(["goal", "--pages", str(tmp_path / "pages.pkl"), "--cand", "a=/x/a.vtsd,/x/a.onnx",
             "--cand", "b=/x/b.vtsd,/x/b.onnx,fit:960,h0.45v0.1", "--lib", "/x/lib.so", "--mem-views", "",
             "--passes", "2", "--warmup", "1", "--out", str(out)])
    run = json.loads(out.read_text())
    assert "det_1t" not in run and set(run["candidates"]) == {"a", "b"}
    assert run["schedule"]["calls"] == 3 * 9 * 14          # 4 shared + 5 per candidate pipelines
    for name in ("a", "b"):
        f = Path(run["timing_files"][name])
        assert f == tmp_path / f"run.{name}.json"
        t = json.loads(f.read_text())
        assert t["candidate"] == name and t["interleaved"] is True and t["det_1t"]["meiki"]["vita"] == 105.0
        assert G.speed(str(f))[0]["pass"]
    assert json.loads((tmp_path / "run.b.json").read_text())["rule"] == "h0.45v0.1"
    with pytest.raises(SystemExit):
        TT.main(["goal", "--pages", "p", "--cand", "a=x,y", "--cand", "a=z,w", "--lib", "l", "--out",
                 str(tmp_path / "x.json")])


# ----------------------------------------------------------------------------- the options (protocol v2 amendment 4)

C6 = "+dec:vote_pct=70+post:merge_pct=70,rows_pct=65,valley_pct=0,ext_pct=1200+drop:nojp=2,nojp_gap_pct=200"


def test_options_in_the_specs(TT):
    det, all_ = TT.opts_parts(C6)
    assert det == "+dec:vote_pct=70+post:merge_pct=70,rows_pct=65,valley_pct=0,ext_pct=1200" and all_ == C6
    assert TT.opts_parts("") == ("", "") and TT.opts_parts(None) == ("", "")
    for bad in ("dec:vote_pct=70", "+refine", "+zap:x=1", "+post:merge_pct"):
        with pytest.raises(ValueError):
            TT.opts_parts(bad)
    c = TT.parse_cand("q=/b/q.vtsd,/b/q.onnx", "long:960", "c0.45v0.1")
    c["opts"] = C6
    d = {k: spec for k, spec, *_ in TT.goal_pipelines([c])}
    assert d["det1:cand:q"] == "cbox:/b/q.vtsd@long:960+refine" + det          # the detector: no reading filter
    assert d["page4:c:q"] == "cpage:/b/q.vtsd@long:960+refine" + C6 + "+c:L"
    assert d["page4:onnx:q#d2"] == "opage:/b/q.onnx@long:960@exact+refine" + C6 + "+L#d2"
    off = {k: spec for k, spec, *_ in TT.goal_pipelines([c], c_opts=False)}
    assert off["det1:cand:q"] == "cbox:/b/q.vtsd@long:960+refine" and off["page4:onnx:q#d2"] == d["page4:onnx:q#d2"]
    kind, path, size, rec, refine, o = TT.split_spec(d["page4:c:q"])
    assert (kind, path, size, rec, refine) == ("cpage", "/b/q.vtsd", "long:960", "L", True)
    assert o["decode"] == {"vote_pct": 70.0} and o["post"]["ext_pct"] == 1200 and o["drop"] == {"nojp": 2,
                                                                                                 "nojp_gap_pct": 200}
    with pytest.raises(ValueError):
        TT.split_spec("cbox:/b/q.vtsd@long:960+refine+drop:nojp=1")              # a detector has no reading filter
    with pytest.raises(ValueError):
        TT.split_spec("cdet:/b/q.vtsd+dec:vote_pct=70")


def test_c_options_need_their_twins(TT, monkeypatch):
    import tsdesk_c as TC

    assert TT.c_options({"decode": {}, "post": {}, "drop": {}}, "detect") == {}
    if "decode" not in __import__("inspect").signature(TC.CModel.detect).parameters:
        with pytest.raises(ValueError, match="C twins"):
            TT.c_options({"decode": {"vote_pct": 70}, "post": {}, "drop": {}}, "detect")

    class Twin:                                    # a binding with the twins: the options pass through
        def detect(self, img, size=None, refine=None, decode=None, post=None):
            pass

        def __init__(self, det=None, rec=None, decode=None, post=None, drop=None):
            pass
    monkeypatch.setattr(TC, "CModel", Twin)
    monkeypatch.setattr(TC, "CPage", Twin)
    o = {"decode": {"vote_pct": 70}, "post": {"merge_pct": 70}, "drop": {"nojp": 2}}
    assert TT.c_options(o, "detect") == {"decode": o["decode"], "post": o["post"]} and TT.c_options(o, "page") == o


def test_goal_refuses_c_options_without_twins(TT, G, stubbed, tmp_path, monkeypatch):
    """With options the C path needs their C twins: the goal stops, unless --c-without-options (recorded)."""
    seen = []

    class Strict(StubPipe):
        def __init__(self, spec, threads, pad="auto", lib=None, onnx_dir=None, **kw):
            super().__init__(spec, threads, pad, lib, onnx_dir)
            seen.append(spec)
            if spec.startswith(("cbox", "cpage")) and "+dec:" in spec:
                raise ValueError("tools/tsdesk_c has no ['decode'] options (their C twins)")
    monkeypatch.setattr(TT, "Pipeline", Strict)
    args = ["goal", "--pages", "p", "--cand", "q=/x/q.vtsd,/x/q.onnx", "--lib", "/x/lib.so", "--mem-views", "",
            "--opts", C6, "--out", str(tmp_path / "t.json")]
    with pytest.raises(SystemExit, match="c-without-options"):
        TT.main(args)
    TT.main(args + ["--c-without-options"])
    t = json.loads((tmp_path / "t.json").read_text())
    assert t["options"] == C6 and t["c_options"].startswith("off")
    assert any(s.startswith("opage") and "+drop:nojp=2" in s for s in seen)


def test_onnx_page_reading_filter(TT):
    """The ONNX page drops the readings tools/tsdesk.drop_set rejects (nojp 1: no Japanese character), after reading
    every horizontal line, and reads at the clip rule's crops."""
    import tsdesk as T
    import tsdesk_c as TC

    boxes = [T.Box(10.0, 10.0, 200.0, 20.0, 0, 3.0, 0, 1.0, 9), T.Box(10.0, 50.0, 80.0, 20.0, 0, 3.0, 0, 1.0, 4),
             T.Box(100.0, 50.0, 60.0, 20.0, 0, 2.0, 0, 1.0, 3), T.Box(300.0, 10.0, 20.0, 90.0, 1, 2.0, 0, 1.0, 5)]

    class Det:
        def detect(self, g):
            return list(boxes)

    class Rec:
        def read(self, crop, invert=False):
            class R:
                pass
            r = R()
            r.text, r.conf_mean = ("こんにちは" if crop.shape[1] > 150 else "ABC"), 0.9
            return r

    pp = TT.Pipeline.__new__(TT.Pipeline)
    pp.kind, pp.T, pp.TC, pp.det, pp.rec, pp.pool = "opage", T, TC, Det(), Rec(), None
    pp.V = type("V", (), {"line_dark": staticmethod(lambda g, cb: 0)})
    pp.rule = TC.rule_of("c0.45v0.1")
    frame = np.zeros((120, 400, 3), np.uint8)
    pp.dropo = None
    assert pp._opage(frame) == ["こんにちは", "ABC", "ABC"]                    # the vertical box is not read
    pp.dropo = T.drop_options({"nojp": 1})
    assert pp._opage(frame) == ["こんにちは"]
