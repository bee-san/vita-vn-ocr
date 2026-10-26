"""eval/vocr6_score.py, eval/vocr6_guard.py and eval/vocr6_exposed.py on synthetic data (no game data, no blob):
Rubi's line filter, the rubi text path (beam only on a non-empty greedy line, the filter after textnorm), the finder
bookkeeping, the hval rows and summaries, the dev-score weights, the guard's drop rule and the exposed-unit rule."""
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import vocr6_exposed as X  # noqa: E402
import vocr6_score as V  # noqa: E402
import vocr6_select as S  # noqa: E402


def test_has_japanese_is_rubis_ranges():
    assert V.has_japanese("あ") and V.has_japanese("カ") and V.has_japanese("漢")
    assert not V.has_japanese("ABC 123 …「」") and not V.has_japanese("")
    assert V.has_japanese("\u3040") and V.has_japanese("\u30ff") and V.has_japanese("\u9faf")
    assert not V.has_japanese("\u9fb0") and not V.has_japanese("\u303f")


def test_rubi_path_beams_only_non_empty_greedy_and_filters_after_textnorm():
    dump = {"crops": [{"g": "", "b": "ゴミ"}, {"g": "あ", "b": "いう"}, {"g": "x", "b": "ab"}]}
    seen = []

    def norm_stub(ts):
        seen.append(list(ts))
        return [t.replace("う", "u") for t in ts]   # a textnorm that turns the only kana of crop 2 into Latin

    assert V.texts(dump, "greedy", norm_stub) == ["", "あ", "x"]
    out = V.texts(dump, "rubi", norm_stub)
    assert seen[-1] == ["", "いう", "ab"]            # crop 0: empty greedy -> no beam text
    assert out == ["", "いu", ""]                     # crop 2 has no Japanese after textnorm -> dropped


def test_covers_needs_half_the_height_and_some_width():
    lb = [10, 10, 110, 30]                            # x0, y0, x1, y1: 20 px high
    assert V.covers((0, 15, 200, 20), lb)            # 15 px of 20 vertically, overlaps horizontally
    assert not V.covers((0, 25, 200, 20), lb)        # 5 px of 20
    assert not V.covers((120, 10, 50, 20), lb)       # no horizontal overlap


def plan_and_dump():
    """Two units (one clean, one with a junk box) and two clean lines, with the crops a dump would read."""
    plan = {"fixed": [{"crop": 0, "id": "T1:l1", "title": "T1", "platform": "vita", "kind": "line", "ref": "あいう"},
                      {"crop": 1, "id": "T2:l1", "title": "T2", "platform": "psp", "kind": "line", "ref": "かき"}],
            "units": [{"unit": "T1:p1:a:base", "title": "T1", "platform": "vita", "variant": "base", "junk": 0,
                       "merged": 0, "missed": 0, "lines": [{"id": "a", "text": "あいう", "styles": []}],
                       "finder": [2]},
                      {"unit": "T2:p1:b:base", "title": "T2", "platform": "psp", "variant": "base", "junk": 1,
                       "merged": 0, "missed": 0, "lines": [{"id": "b", "text": "かき", "styles": []}],
                       "finder": [3, 4]}]}
    dump = {"crops": [{"g": "あいう", "b": "あいう"}, {"g": "かき", "b": "かけ"}, {"g": "あい", "b": "あい"},
                      {"g": "かき", "b": "かき"}, {"g": "そし", "b": "そし"}]}
    return plan, dump


def test_score_hval_rows_and_the_summary(monkeypatch):
    plan, dump = plan_and_dump()
    sc = V.score_hval(plan, dump, lambda ts: list(ts))
    c, g = sc["hval_clean"], sc["hval_page"]
    assert c["errors"]["rubi"] == [0, 1] and c["chars"] == [3, 2]
    assert g["junk"] == [0, 1] and g["chars"] == [3, 2]
    assert g["errors"]["rubi"][0] == 1               # one deletion
    assert g["errors"]["rubi"][1] == 2               # the junk box's text is two insertions
    monkeypatch.setitem(S._SEEN, "units", frozenset())
    s = V.summarize(sc, paths=("rubi",))["rubi"]
    assert s["hval_page"] == round(100 * 1 / 3, 4)            # junk-free units only
    assert s["hval_page_all"] == round(100 * 3 / 5, 4)
    assert s["hval_clean_dlg"] == round(100 * 1 / 5, 4)
    monkeypatch.setitem(S._SEEN, "units", frozenset({"T1:p1:a:base"}))
    s = V.summarize(sc, paths=("rubi",))["rubi"]
    assert s["hval_page_seen"] == round(100 * 1 / 3, 4)       # seen kept
    assert s["hval_page"] == 0.0                              # no unit left


def test_dev_score_weights():
    rows = {"errors": {"rubi": [1, 0]}, "chars": [10, 10], "name": [False, True]}
    wd = {"errors": {"rubi": [4]}, "chars": [100]}
    s = V.summarize({"tv": rows, "dn": rows, "wd": wd}, paths=("rubi",))["rubi"]
    assert s["tv"] == 10.0 and s["dn"] == 10.0 and s["wd"] == 4.0
    assert s["dev_score"] == round(0.25 * 10 + 0.25 * 10 + 0.5 * 4, 4)


def shard(path, labels, meta):
    imgs = [np.full((4, 3 + i), i, np.uint8) for i in range(len(labels))]
    np.savez(path, data=np.concatenate([g.ravel() for g in imgs]),
             offsets=np.concatenate([[0], np.cumsum([g.size for g in imgs])]).astype(np.int64),
             shapes=np.asarray([g.shape for g in imgs], np.int32),
             labels=np.asarray(json.dumps(labels, ensure_ascii=False)),
             meta=np.asarray(json.dumps(meta, ensure_ascii=False)))


def test_guard_drops_same_vn_hval_lines_only(tmp_path):
    hv = tmp_path / "hval" / "line"
    hv.mkdir(parents=True)
    shard(hv / "T1.npz", ["あいう"], [{"title_id": "T1", "vn": "v1"}])
    plan = {"fixed": [{"title": "T1", "ref": "あいう", "crop": 0}]}
    pp = tmp_path / "plan.pkl"
    pp.write_bytes(pickle.dumps(plan))
    src, dst = tmp_path / "own.npz", tmp_path / "own_g.npz"
    shard(src, ["あいう", "あいう", "えお", ""], [{"vn": "v1"}, {"vn": "v2"}, {"vn": "v1"}, {"vn": "v1"}])
    rec = tmp_path / "rec.json"
    subprocess.run([sys.executable, str(ROOT / "eval/vocr6_guard.py"), "--hval-plan", str(pp), "--vn-of",
                    str(tmp_path / "hval"), "--pool", f"own={src}={dst}", "--record", str(rec)], check=True,
                   capture_output=True)
    z, z0 = np.load(dst), np.load(src)
    labels = json.loads(str(z["labels"]))
    assert labels == ["あいう", "えお", ""]               # only v1's copy of v1's hval line leaves
    assert z["shapes"].tolist() == z0["shapes"][[1, 2, 3]].tolist()
    keep = np.concatenate([z0["data"][z0["offsets"][i]:z0["offsets"][i + 1]] for i in (1, 2, 3)])
    assert np.array_equal(z["data"], keep)               # the other rows byte for byte
    r = json.loads(rec.read_text())
    assert r["pools"]["own"]["rows_in"] == 4 and r["pools"]["own"]["rows_out"] == 3


def test_exposed_units_are_same_vn_label_matches():
    units = [{"unit": "u1", "title": "T1", "lines": [{"text": "あいう"}]},
             {"unit": "u2", "title": "T2", "lines": [{"text": "あいう"}]},
             {"unit": "u3", "title": "T1", "lines": [{"text": "かき"}, {"text": "くけ"}]}]
    t2v = {"T1": "v1", "T2": "v2"}
    rows = [("あいう", "v1"), ("くけ", "v1"), ("あいう", "v9"), ("", "v2")]
    assert X.exposed_units(units, t2v, rows) == ["u1", "u3"]


def test_cascade_reads_low_confidence_lines_from_the_next_tier():
    dump = {"crops": [{"g": "あい", "b": "あい", "cmin": 0.9}, {"g": "かき", "b": "かき", "cmin": 0.5},
                      {"g": "", "b": "さ", "cmin": 0.1}, {"g": "た", "b": "た", "cmin": 0.6995}]}
    up = {"crops": [{"g": "x", "b": "あう", "cmin": 1}, {"g": "け", "b": "かけ", "cmin": 1},
                    {"g": "し", "b": "し", "cmin": 1}, {"g": "", "b": "ち", "cmin": 1}]}
    ident = lambda ts: list(ts)   # noqa: E731
    assert V.texts(dump, "rubi", ident) == ["あい", "かき", "", "た"]
    # crop 1 (0.5) and crop 3 (0.6995 -> 700 after rounding? int(699.5 + 0.5) = 700: not under) ; crop 2 has no greedy
    assert V.texts(dump, "rubi", ident, up) == ["あい", "かけ", "", "た"]
    up["crops"][3]["g"] = "ち"
    dump["crops"][3]["cmin"] = 0.69
    assert V.texts(dump, "rubi", ident, up)[3] == "ち"


def test_lm_opts_override():
    o = V.lm_opts_of("alpha=600,beta=2500,flags=1")
    assert o["alpha"] == 600 and o["beta"] == 2500 and o["flags"] == 1 and o["gate"] == V.LM_OPTS["gate"]
    import pytest
    with pytest.raises(SystemExit):
        V.lm_opts_of("nonsense=1")
