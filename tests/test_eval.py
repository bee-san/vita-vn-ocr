"""Fast checks of the eval harness (eval/) and its committed manifests.

  python -m pytest tests/test_eval.py      (or without pytest: python tests/test_eval.py)

No network, no screenshots, no model downloads.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL))
import backends  # noqa: E402
import finders  # noqa: E402
import metrics  # noqa: E402
import sets  # noqa: E402


def test_norm_folds_width_ellipsis_and_dashes():
    n = metrics.norm
    assert n("「え？　そう！」") == n("「え?そう!」")
    assert n("……") == "……" and len(n("……")) == 2
    assert n("...") == "…" and n("・・・") == "…"
    assert n("〜") == n("～") == n("~")
    assert n("――") == n("——") == n("--")
    assert n("ＡＢ１") == "AB1"
    assert n(" a\tb\n") == "ab"


def test_norm_keeps_real_errors():
    n = metrics.norm
    assert n("ー") != n("―") and n("ー") != n("一")
    assert n("っ") != n("つ") and n("ィ") != n("イ")
    assert n("バ") != n("パ") and n("ロ") != n("口")


def test_lev_and_scores():
    assert metrics.lev("abc", "abc") == 0
    assert metrics.lev("abc", "abd") == 1
    assert metrics.lev("", "abc") == 3
    assert metrics.lev("kitten", "sitting") == 3
    rows = [{"pred": "あいう", "text": "あいう", "styles": ["name"], "vn": "v1"},
            {"pred": "かき", "text": "かきく", "styles": [], "vn": "v2"}]
    res = metrics.score_lines(rows)
    assert res["all"]["errors"] == 1 and res["all"]["chars"] == 6
    assert abs(res["all"]["cer"] - 1 / 6) < 1e-4
    assert res["all"]["exact"] == 0.5
    assert res["styles"]["name"]["cer"] == 0.0
    assert abs(res["game_macro_cer"] - (0 + 1 / 3) / 2) < 1e-4


def test_oov_chars():
    assert metrics.oov_chars(["あい！"], {"あ", "!"}) == {"い": 1}


def test_ctc_greedy():
    vocab = ["a", "b", "c"]
    assert backends.ctc_greedy([0, 1, 1, 0, 1, 2, 2, 3, 0], vocab) == "aabc"
    assert backends.ctc_greedy([], vocab) == ""


def test_tiny_prep_shape():
    x, t = backends.TinyTorch.prep(np.zeros((40, 300, 3), np.uint8))
    assert x.shape[:3] == (1, 1, 32) and x.shape[3] % 8 == 0
    assert t == -(-round(300 * 32 / 40) // 8)
    assert float(x.min()) == -1.0
    x, t = backends.TinyTorch.prep(np.zeros((40, 300, 3), np.uint8), 24)
    assert x.shape[:3] == (1, 1, 24) and t == -(-round(300 * 24 / 40) // 8)


def test_tiny_checkpoint_fields(tmp_path):
    """The checkpoint's height and bottleneck are honoured; a multi-character class such as
    "<unk>" decodes to one placeholder character."""
    import torch

    from model import TinyVNOCR
    vocab = ["あ", "い", "<unk>"]
    torch.manual_seed(0)
    m = TinyVNOCR(len(vocab) + 1, width=0.25, bottleneck=48)
    path = tmp_path / "ck.pt"
    torch.save({"model": m.state_dict(), "vocab": vocab, "width": 0.25, "bottleneck": 48, "step": 1}, path)
    rec = backends.TinyTorch(path)
    assert rec.H == 32 and rec.info["bottleneck"] == 48 and rec.vocab[-1] == backends.UNK
    assert len(rec.read([np.full((30, 90, 3), 128, np.uint8)])) == 1
    assert backends.ctc_greedy([3, 1], rec.vocab) == backends.UNK + "あ"


def test_finder_precision_ignores_uncertain_lines():
    import run
    img = np.zeros((100, 200, 3), np.uint8)
    rows = [{"id": "a", "box": [10, 10, 100, 20], "text": "あいう", "uncertain": False},
            {"id": "b", "box": [10, 50, 100, 20], "text": "", "uncertain": True}]

    class Rec:
        def read(self, crops):
            return ["あいう"] * len(crops)

    def find(region):   # one detection per line, plus a false alarm
        return [(0, 8, 200, 24), (0, 48, 200, 24), (0, 80, 200, 15)]

    class Args:
        match = 0.5

    saved = run.sets.regions, run.sets.ImageCache.get
    run.sets.regions = lambda name, data: [("x.png", [0, 0, 200, 100], rows)]
    run.sets.ImageCache.get = lambda self, path: img
    try:
        f = run.finder_mode(Rec(), find, "vita-test", None, Args())["finder"]
    finally:
        run.sets.regions, run.sets.ImageCache.get = saved
    assert f["detections"] == 2 and f["line_precision"] == 0.5 and f["line_recall"] == 1.0
    assert f["pipeline_cer"] == 1.0   # the false alarm's 3 characters against 3 scored ones


def test_console_polarity_inverts_dark_text_only():
    import run
    dark = np.full((24, 80, 3), 230, np.uint8)
    dark[8:16, 10:70:3] = 20                      # dark strokes on a light box
    light = 255 - dark                            # light strokes on a dark box
    out, flags = run.console_polarity([dark, light])
    assert flags == [True, False]
    assert out[0].shape == dark.shape and (out[0] == 255 - dark[..., :1]).all()
    assert out[1] is light
    same, none = run.prepare([dark], "native")
    assert same[0] is dark and none == [False]


def test_paddle_prep_width():
    p = object.__new__(backends.PaddleOnnx)
    p.max_w = 1024
    assert p.prep(np.zeros((30, 300, 3), np.uint8)).shape == (1, 3, 48, 480)
    assert p.prep(np.zeros((10, 2000, 3), np.uint8)).shape == (1, 3, 48, 1024)
    assert p.prep(np.zeros((40, 2, 3), np.uint8)).shape == (1, 3, 48, 8)


def test_reference_weights_are_pinned(tmp_path):
    assert set(backends.PPOCR_FILES) == set(backends.PPOCR.values())
    assert all(re.fullmatch(r"[0-9a-f]{64}", sha) for _, sha in backends.PPOCR_FILES.values())
    assert re.fullmatch(r"[0-9a-f]{40}", backends.MANGA_OCR[1])
    (tmp_path / backends.PPOCR["ppocr-v5-mobile"]).write_bytes(b"not the pinned export")
    saved, backends.MODEL_DIRS = backends.MODEL_DIRS, [tmp_path]
    try:
        backends.make_recognizer("ppocr-v5-mobile-jis")
        raise AssertionError("a preset file with another sha256 was accepted")
    except ValueError as e:
        assert "sha256" in str(e)
    finally:
        backends.MODEL_DIRS = saved


def test_cmd_backend_protocols(tmp_path):
    crops = [np.full((8, 12, 3), 200, np.uint8), np.zeros((9, 5, 3), np.uint8)]
    batch = backends.Cmd("sed 's#.*/##'").read(crops)
    assert batch == ["000000.ppm", "000001.ppm"]
    per = backends.Cmd("head -c 2 {} && echo").read(crops)
    assert per == ["P6", "P6"]


def test_gradient_finder_and_match():
    img = np.full((60, 200, 3), 30, np.uint8)
    img[8:22, 10:180] = 230
    img[34:50, 10:120] = 230
    for x in range(10, 180, 6):              # glyph-like vertical strokes
        img[8:22, x:x + 2] = 30
    for x in range(10, 120, 6):
        img[34:50, x:x + 2] = 30
    dets = finders.gradient(img)
    assert len(dets) == 2
    gts = [{"box_r": (10, 8, 170, 14)}, {"box_r": (10, 34, 110, 16)}]
    pairs, unmatched = finders.match(dets, gts)
    assert sorted(j for _, j in pairs) == [0, 1] and not unmatched


def test_console_finder_is_the_runtime_twin():
    from common import find_lines
    img = np.full((60, 200, 3), 225, np.uint8)     # dark strokes on a light box: both polarities work
    img[20:40, 20:180:4] = 20
    assert finders.make_finder("console") is finders.console
    assert finders.console(img) == find_lines(img, "grad") and len(finders.console(img)) == 1


def test_overlay_finder_rejects_light_box():
    img = np.full((40, 200, 3), 240, np.uint8)
    img[10:25, 10:150] = 20
    assert finders.overlay(img) is None


def load_games():
    path = sets.MANIFESTS / "test_games.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def test_manifests_schema():
    hexsha = re.compile(r"^[0-9a-f]{64}$")
    games = load_games()
    reserved = set(games.get("all", []))
    sessions = set(games.get("sessions", []))
    for name in ("vita-test", "psp-test"):
        path = sets.MANIFESTS / f"{name}.jsonl"
        if not path.exists():
            continue
        rows = sets.read_manifest(name)
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids)), f"{name}: duplicate ids"
        for r in rows:
            assert hexsha.match(r["sha256"]) and r["url"].startswith("https://t.vndb.org/")
            W, H = r["dims"]
            x, y, w, h = r["box"]
            rx, ry, rw, rh = r["region"]
            assert w > 0 and h > 0 and 0 <= x and 0 <= y and x + w <= W and y + h <= H, r["id"]
            assert rx <= x and ry <= y and x + w <= rx + rw and y + h <= ry + rh, r["id"]
            assert set(r["styles"]) <= set(sets.STYLES), (r["id"], r["styles"])
            assert isinstance(r["uncertain"], bool)
            assert r["uncertain"] or r["text"], r["id"]
            assert (W, H) == ((960, 544) if name == "vita-test" else (480, 272))
            assert re.match(r"^v\d+$", r["vn"]) and re.match(r"^r\d+$", r["session"]), r["id"]
            if reserved:
                assert r["vn"] in reserved, f"{r['vn']} not in test_games.json all"
                assert r["session"] in sessions, f"{r['session']} not in test_games.json sessions"
                assert r["vn"] in games[name], f"{r['vn']} missing from test_games.json {name}"


def test_manifests_have_no_duplicate_lines():
    """Two lines of one screenshot may not share more than half of the smaller box."""
    for name in ("vita-test", "psp-test"):
        if not (sets.MANIFESTS / f"{name}.jsonl").exists():
            continue
        by_url = {}
        for r in sets.read_manifest(name):
            by_url.setdefault(r["url"], []).append(r)
        for rows in by_url.values():
            for i, a in enumerate(rows):
                for b in rows[i + 1:]:
                    (ax, ay, aw, ah), (bx, by, bw, bh) = a["box"], b["box"]
                    iw = min(ax + aw, bx + bw) - max(ax, bx)
                    ih = min(ay + ah, by + bh) - max(ay, by)
                    inter = max(0, iw) * max(0, ih)
                    assert inter <= 0.5 * min(aw * ah, bw * bh), (a["id"], b["id"])


def test_test_games_reservations():
    games = load_games()
    if not games:
        return
    reserved = set(games["all"])
    for name in ("vita-test", "psp-test"):
        for vn, g in games.get(name, {}).items():
            assert vn in reserved and set(g["sessions"]) <= set(games["sessions"]), vn
    for vn, ids in games.get("family", {}).items():
        assert vn in reserved and set(ids) <= reserved, vn
    # A game may be in both sets (its Vita and PSP releases are separate captures, and both sets
    # are held out), but one capture session may not be split across them.
    sess = {k: {s for g in v.values() for s in g["sessions"]} for k, v in games.items() if k.endswith("-test")}
    assert not (sess.get("vita-test", set()) & sess.get("psp-test", set())), "a capture session in two sets"


def test_test_games_disjoint_from_real_train():
    """No test game may have lines in data/real/real_train.npz (checked when present)."""
    real = EVAL.parent / "data" / "real" / "lines.jsonl"
    tg = sets.MANIFESTS / "test_games.json"
    if not real.exists() or not tg.exists():
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location("label_real", EVAL.parent / "data" / "label_real.py")
    lr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lr)
    train = {r["vn"] for r in map(json.loads, real.read_text(encoding="utf-8").splitlines())
             if r["split"] == "train" and lr.decide(r["a"], r["b"])}
    games = json.loads(tg.read_text(encoding="utf-8"))
    used = set(games.get("vita-test", {})) | set(games.get("psp-test", {}))
    assert not used & train, sorted(used & train)


if __name__ == "__main__":
    import inspect
    import tempfile
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                if "tmp_path" in inspect.signature(fn).parameters:
                    with tempfile.TemporaryDirectory() as d:
                        fn(Path(d))
                else:
                    fn()
                print(f"ok   {name}")
            except Exception as e:  # report every failure, then exit non-zero
                failed += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if failed else 0)
