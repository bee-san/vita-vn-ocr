"""textspot6's data and trainer pieces (synth/textspot6_pages.py, train/textspot6_train.py), no data files: the page
labels from a hand-made pool record, the window geometry (scale rule, Adrenaline view, zoom), the tag targets and
loss, and a short training run on tiny synthetic shards with the export reading back as VTSP version 4."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "synth", ROOT / "train", ROOT / "tests", ROOT / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot6_pages as PG  # noqa: E402
import textspot6_train as TR  # noqa: E402

torch = pytest.importorskip("torch")


def record():
    return {"title_id": "T1", "page": "000001", "image": "T1/pages/000001.png", "image_sha256": "0" * 64,
            "dims": [960, 544], "platform": "vita", "split": "train", "complete": False, "untrusted": False,
            "uncovered": [3], "roi": [0, 0, 960, 544],
            "boxes": [{"index": 1, "xyxy": [100.0, 400.0, 500.5, 425.2], "kind": "line"},
                      {"index": 2, "xyxy": [100.0, 360.0, 160.0, 380.0], "kind": "line"},
                      {"index": 3, "xyxy": [700.0, 20.0, 800.0, 40.0], "kind": "line"}],
            "lines": [{"id": "T1:000001:line1", "kind": "line", "vertical": False, "box": [100, 400, 401, 26],
                       "box_main": [100, 405, 401, 21], "furigana": True, "box_index": [1], "drop": None},
                      {"id": "T1:000001:name2", "kind": "name", "vertical": False, "box": [100, 360, 60, 20],
                       "box_main": None, "furigana": False, "box_index": [2], "drop": None},
                      {"id": "T1:000001:ruby3", "kind": "ruby", "vertical": False, "box": [120, 398, 30, 8],
                       "box_main": None, "furigana": False, "box_index": [], "drop": None}]}


def test_build_labels():
    rec = record()
    refined = {"T1:000001:line1": ([101, 406, 499, 425], "ink"), "T1:000001:name2": ([100, 360, 160, 380], "quad")}
    meiki = [{"raw": [700, 20, 800, 40], "score": 0.9, "box": [701, 21, 799, 39], "vertical": False},   # meets box 3
             {"raw": [10, 10, 60, 30], "score": 0.9, "box": [11, 12, 58, 29], "vertical": False},     # other text
             {"raw": [10, 100, 30, 200], "score": 0.3, "box": [12, 100, 28, 198], "vertical": True},  # low: ignore
             {"raw": [110, 401, 490, 424], "score": 0.95, "box": [110, 401, 490, 424], "vertical": False}]  # meets 1
    lines, ignore, c = PG.build_labels(rec, refined, meiki, 960, 544)
    assert [ln[5] for ln in lines] == ["dialogue", "name", "notdlg"]
    assert lines[0][:4] == [101, 406, 499, 425] and lines[2][:4] == [11, 12, 58, 29]
    assert lines[2][6] == 1.0 and lines[0][6] == 1.0
    assert c["ruby_ignore"] == 1 and c["uncovered_ignore"] == 1 and c["meiki_meets"] == 2
    assert c["meiki_other"] == 1 and c["meiki_ignore"] == 1
    assert [119, 397, 151, 407] in ignore                     # the ruby box grown by 1
    assert [698, 18, 802, 42] in ignore                       # the uncovered page box grown by 2
    assert PG.meiki_weight(0.5) == 0.5 and PG.meiki_weight(0.95) == 1.0


def test_dropped_lines_and_refine_check():
    rec = record()
    rec["lines"][1]["drop"] = "geometry"
    refined = {"T1:000001:line1": ([101, 406, 499, 425], "ink"), "T1:000001:name2": ([100, 360, 160, 380], "quad")}
    lines, ignore, c = PG.build_labels(rec, refined, [], 960, 544)
    assert [ln[5] for ln in lines] == ["dialogue"] and c["drop_ignore"] == 1
    # refined_line keeps the quad when the ink box leaves it
    rgb = np.full((60, 200, 3), 200, np.uint8)
    rgb[20:40, 20:180] = 30
    box, how = PG.refined_line(rgb, [18, 18, 182, 42], False)
    assert how == "ink" and box[1] >= 18 and box[3] <= 42


def test_window_geometry_and_targets():
    rng = np.random.default_rng(0)
    canvas = np.full((272, 480), 200, np.uint8)
    canvas[200:216, 40:300] = 20
    smp = {"lines": [[40, 200, 300, 216, 0, "dialogue", 1.0]], "ignore": [], "kind": "psp"}
    f_of = TR.scale_rule({"name": "t", "family": "T", "half_above": 480, "head": "kroa_x2"})
    saved = dict(TR.AUG)
    for _ in range(20):
        img, pos, ign = TR.make_window(canvas, smp, "hook", rng, f_of, None, aug=False)
        assert img.shape == (256, 480)            # PSP read as it is: 272 rows -> a 256-row window
        assert len(pos) <= 1
    # the Adrenaline view: upscaled 2x, then halved by the scale rule: about the native box again
    TR.AUG["adrenaline_p"], TR.AUG["zoom_p"] = 1.0, 0.0
    try:
        for k in TR.AUG:
            if k.endswith("_p") and k not in ("adrenaline_p", "zoom_p"):
                TR.AUG[k] = 0.0
        img, pos, ign = TR.make_window(canvas, smp, "hook", rng, f_of, None, aug=True)
        assert img.shape == (256, 480)
        if pos:
            x0, y0, x1, y1 = pos[0][:4]
            assert abs(x0 - 40) <= 1 and abs(x1 - 300) <= 1
    finally:
        TR.AUG.update(saved)
    lines = [[40, 100, 300, 116, 0, 1.0, TR.CLASS_MASK["dialogue"]],
             [40, 140, 100, 156, 0, 0.7, TR.CLASS_MASK["notdlg"]],
             [40, 180, 300, 196, 0, 1.0, TR.CLASS_MASK["text"]]]
    K, O, A1, A2, v, k, Wt, Mk = TR.targets(lines, [], 480, 256)
    assert set(np.unique(Mk[k])) <= {0, 1, 30}
    assert (Mk[~k] == 0).all() and (Mk[k] == 1).any() and (Mk[k] == 30).any()
    assert (Wt[Mk == 30] == np.float32(0.7)).all()


def test_tag_loss_values():
    t = torch.zeros(1, 4, 2, 2)
    Mk = torch.tensor([[[1, 30], [0, 2]]])
    Wt = torch.ones(1, 2, 2)
    loss, parts = TR.tag_loss(t, Mk, Wt)
    # uniform logits over 5 classes: class 0 alone -> log 5; set {1,2,3,4} -> log 5/4; class 1 -> log 5
    want = (np.log(5) + np.log(5 / 4) + np.log(5)) / 3
    assert abs(float(loss) - want) < 1e-5 and parts["tag_cells"] == 3
    t2 = torch.zeros(1, 4, 2, 2, requires_grad=True)
    loss2, _ = TR.tag_loss(t2, Mk, Wt)
    loss2.backward()
    assert t2.grad is not None and float(t2.grad.abs().sum()) > 0


def _tiny_shards(tmp, n=6):
    """A synthetic-format shard (src synth) and a hook-format shard, tiny canvases."""
    rng = np.random.default_rng(1)
    d = tmp / "synth_frames"
    d.mkdir()
    samples, blobs, off = [], [], 0
    for i in range(n):
        h, w = (544, 960) if i % 2 else (272, 480)
        g = rng.integers(100, 140, (h, w)).astype(np.uint8)
        g[200:224, 100:600 if w > 480 else 400] = 20
        lines = [[100, 200, 600 if w > 480 else 400, 224, 0, ["dialogue", "ui"][i % 2]]] if i % 3 else []
        samples.append({"off": off, "h": h, "w": w, "src": "synth:vita", "kind": "vita", "vn": None, "shot": None,
                        "lines": lines, "ignore": []})
        blobs.append(g.tobytes())
        off += h * w
    np.save(d / "synth_frames_0000.npy", np.frombuffer(b"".join(blobs), np.uint8))
    (d / "synth_frames_0000.json").write_text(json.dumps({"samples": samples}))
    import hashlib
    man = {"kind": "test-shards", "shards": [{"dir": "synth_frames", "name": "synth_frames_0000",
                                                   "npy_sha256": hashlib.sha256((d / "synth_frames_0000.npy")
                                                                                .read_bytes()).hexdigest(),
                                                   "json_sha256": hashlib.sha256((d / "synth_frames_0000.json")
                                                                                 .read_bytes()).hexdigest()}]}
    (tmp / "synth_manifest.json").write_text(json.dumps(man))
    hd = tmp / "hook" / "train"
    hd.mkdir(parents=True)
    hs, hb, off = [], [], 0
    for i in range(n):
        h, w = (544, 960) if i % 2 else (272, 480)
        g = rng.integers(150, 200, (h, w)).astype(np.uint8)
        g[300 if h > 300 else 200:316 if h > 300 else 216, 50:400] = 10
        y0 = 300 if h > 300 else 200
        hs.append({"off": off, "h": h, "w": w, "src": "hook", "kind": "vita" if w > 480 else "psp", "vn": "v1",
                   "shot": f"T{i % 2}:{i:06d}", "title": f"T{i % 2}", "page": f"{i:06d}", "split": "train",
                   "lines": [[50, y0, 400, y0 + 16, 0, "dialogue", 1.0], [50, y0 - 30, 120, y0 - 14, 0, "name", 1.0]],
                   "ignore": [], "meta": {"counts": {}}})
        hb.append(g.tobytes())
        off += h * w
    np.save(hd / "T_000.npy", np.frombuffer(b"".join(hb), np.uint8))
    (hd / "T_000.json").write_text(json.dumps({"samples": hs}))
    hman = {"kind": "textspot6-pages", "shards": [{"dir": "train", "name": "T_000", "split": "train",
                                                   "npy_sha256": "", "json_sha256": ""}]}
    (tmp / "hook_manifest.json").write_text(json.dumps(hman))
    return tmp


def test_short_run_exports_v4(tmp_path):
    import textspot6_twin as T6
    _tiny_shards(tmp_path)
    shape = {"name": "tiny6", "family": "T", "stride": 8, "c0": 4, "c1": 8, "c2": 8, "chans": [8], "dil": [1],
             "block": "sep", "head": "kroa_x2", "half_above": 480}
    out = tmp_path / "run"
    done = TR.main(["--recipe", "base", "--shape", "tiny6", "--shape-json", json.dumps(shape), "--seed", "1",
                    "--out", str(out), "--hook-manifest", str(tmp_path / "hook_manifest.json"),
                    "--hook-shards", str(tmp_path / "hook"), "--manifest", str(tmp_path / "synth_manifest.json"),
                    "--shards", str(tmp_path / "synth_frames"), "--steps", "10", "--log-every", "5",
                    "--cal-batches", "1", "--bn-batches", "1", "--threads", "1", "--workers", "0",
                    "--mix-json", json.dumps({"hook": 0.5, "synth": 0.5})])
    m = T6.Model((out / "final.vtsp").read_bytes())
    assert m.version == 4 and m.tags == 4 and m.form == "trunk" and done["vtsp_version"] == 4
    log = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
    assert any("tag" in r for r in log) and any(r.get("qat") for r in log)
