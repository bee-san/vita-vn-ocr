"""eval/vocr5_refine.py: VOCR5's refined copies of the GRAD pools (prompts/vocr5-aug.md part B). The keep rule (one
line, differing from the crop, at least 8 px tall), the cut in both polarities, the meta it adds, deterministic bytes,
the guards (no overwrite, no development file, the owner's frames stay private), train2 reading the copy, the
reserve_check call, and the devnative agreement test on drawn and procedural regions. No screenshots."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for d in ("eval", "synth", "tests", "train"):
    sys.path.insert(0, str(ROOT / d))

import vocr5_refine as V  # noqa: E402

import lines_refine as R  # noqa: E402
from common import find_lines_ex, grey_of  # noqa: E402
from linefind_parity import make_cases  # noqa: E402

VOCAB = ["あ", "い", "う", "え", "お"]


def glyph_row(img, top, height, x0, x1, ink):
    """A row of dense block glyphs (tests/test_lines_refine.py's), `height` rows tall, from column x0 to x1."""
    for x in range(x0, x1 - height + 1, height + 4):
        g = np.zeros((height, height), bool)
        g[:, ::3] = True
        g[::4, :] = True
        g[0, :] = g[-1, :] = True
        img[top:top + height, x:x + height][g] = ink
    return img


def screen(h, w, paper=40):
    return np.full((h, w), paper, np.uint8)


def drawn_cases():
    """(name, screen grey, expected status): bright text on a dark box."""
    kept = glyph_row(screen(40, 200), 12, 16, 4, 196, 230)
    kept[2, :] = 230                                                  # the box's border above the line
    return [("kept", kept, "kept"),
            ("unchanged", glyph_row(screen(32, 200), 2, 28, 2, 198, 230), "unchanged"),
            ("split", glyph_row(glyph_row(screen(40, 200), 4, 12, 4, 196, 230), 22, 12, 4, 196, 230), "split"),
            ("too_small", glyph_row(screen(30, 200), 12, 5, 4, 196, 230), "too_small")]


def write_pool(path, rows):
    """rows: [(stored crop, label, meta)] in train2's shard format."""
    crops = [c for c, _, _ in rows]
    np.savez_compressed(path, data=np.concatenate([c.ravel() for c in crops]).astype(np.uint8),
                        offsets=np.cumsum([0] + [c.size for c in crops]).astype(np.int64),
                        shapes=np.array([c.shape for c in crops], np.int32),
                        labels=np.array(json.dumps([lab for _, lab, _ in rows], ensure_ascii=False)),
                        meta=np.array(json.dumps([m for _, _, m in rows], ensure_ascii=False)))


@pytest.fixture
def pool(tmp_path):
    """Every drawn case twice: bright on dark as it is (dark 0), and dark on light (dark 1), stored inverted as the
    console hands it over. Frame boxes at an offset, so refine_box is checkable."""
    rows, want = [], []
    for k, (name, scr, status) in enumerate(drawn_cases()):
        for dark in (0, 1):
            s = 255 - scr if dark else scr                           # what the screen shows
            stored = 255 - s if dark else s
            h, w = s.shape
            x0, y0 = 30 + k, 100 + 2 * k
            m = {"id": f"v1_{k}{dark}_0", "vn": "v1", "crop": "grad", "box": [x0, y0, x0 + w, y0 + h], "dark": dark,
                 "inverted": bool(dark), "target": "psp" if k % 2 else "vita", "chars": [["あ", 1, 2, 3, 4, 0.9]]}
            rows.append((stored, "あいう" if k % 2 else "えお", m))
            want.append((name, status, s))
    src = tmp_path / "pool_grad.npz"
    write_pool(src, rows)
    return src, rows, want


def load(path):
    z = np.load(path)
    labels, meta = json.loads(str(z["labels"])), json.loads(str(z["meta"]))
    out = []
    for i in range(len(labels)):
        h, w = (int(v) for v in z["shapes"][i])
        out.append((z["data"][z["offsets"][i]:z["offsets"][i + 1]].reshape(h, w), labels[i], meta[i]))
    return out


def test_drawn_cases_refine_as_named_in_both_polarities():
    for name, scr, status in drawn_cases():
        for dark in (0, 1):
            s = 255 - scr if dark else scr
            stored = 255 - s if dark else s
            boxes, darks = V.refine_stored(stored, bool(dark), dark)
            assert V.classify(boxes, stored.shape) == status, (name, dark)
            assert darks == [dark] * len(boxes)                      # the refinement keeps the line's flag


def test_build_keeps_exactly_the_single_tighter_lines(pool, tmp_path):
    src, rows, want = pool
    out = tmp_path / "pool_grad_ref.npz"
    rep = V.build(src, out, name="t_grad", reserve=False)
    assert rep["counts"] == {"kept": 2, "unchanged": 2, "split": 2, "too_small": 2}
    assert rep["split_lines"] == {"2": 2} and rep["rows"] == 8
    assert rep["out_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    got = load(out)
    src_sha = hashlib.sha256(src.read_bytes()).hexdigest()
    kept = [(i, r) for i, (r, w) in enumerate(zip(rows, want)) if w[1] == "kept"]
    assert len(got) == len(kept) == 2
    for (crop, lab, meta), (i, (stored, lab0, meta0)) in zip(got, kept):
        s = want[i][2]
        (x, y, w, h), = R.refine_lines(s, [(0, 0, s.shape[1], s.shape[0])], [meta0["dark"]], R.DEFAULT)[0]
        cut = s[y:y + h, x:x + w]
        assert np.array_equal(crop, 255 - cut if meta0["inverted"] else cut)   # stored polarity, as the source
        assert np.array_equal(crop, stored[y:y + h, x:x + w])
        assert h < stored.shape[0] and h >= V.MIN_H
        assert lab == lab0                                                         # the label is unchanged
        b = meta0["box"]
        assert meta == dict(meta0, refined_from={"file": src.name, "sha256": src_sha, "row": i},
                            refine_box=[b[0] + x, b[1] + y, b[0] + x + w, b[1] + y + h])


def test_output_bytes_are_deterministic(pool, tmp_path):
    src = pool[0]
    a, b = tmp_path / "a" / "x_ref.npz", tmp_path / "b" / "x_ref.npz"
    V.build(src, a, reserve=False)
    V.build(src, b, reserve=False)
    assert a.read_bytes() == b.read_bytes()


def test_guards(pool, tmp_path, monkeypatch):
    src = pool[0]
    out = tmp_path / "o_ref.npz"
    V.build(src, out, reserve=False)
    with pytest.raises(SystemExit, match="exists"):
        V.build(src, out, reserve=False)                             # a pinned copy is never replaced silently
    V.build(src, out, reserve=False, force=True)
    dev = tmp_path / "devnative_tv_native_grad_dev.npz"
    dev.write_bytes(src.read_bytes())
    with pytest.raises(SystemExit, match="development"):
        V.build(dev, tmp_path / "d_ref.npz", reserve=False)
    with pytest.raises(SystemExit, match="overwrite"):
        V.build(src, src, reserve=False, force=True)
    priv = tmp_path / "shared" / "private"
    (priv / "owner").mkdir(parents=True)
    own = priv / "owner" / "owner_grad.npz"
    own.write_bytes(src.read_bytes())
    monkeypatch.setattr(V, "PRIVATE", priv)
    with pytest.raises(SystemExit, match="private"):
        V.build(own, tmp_path / "own_ref.npz", reserve=False)        # the owner's frames never leave private/
    V.build(own, priv / "aug" / "own_ref.npz", reserve=False)


def test_build_rejects_a_meta_box_that_is_not_the_crops(pool, tmp_path):
    src, rows, _ = pool
    rows = [(c, lab, dict(m, box=[0, 0, 5, 5])) for c, lab, m in rows]
    bad = tmp_path / "bad_grad.npz"
    write_pool(bad, rows)
    with pytest.raises(ValueError, match="frame box"):
        V.build(bad, tmp_path / "bad_ref.npz", reserve=False)


def test_a_refined_copy_is_not_refined_again(pool, tmp_path):
    out = tmp_path / "once_ref.npz"
    V.build(pool[0], out, reserve=False)
    with pytest.raises(SystemExit, match="refined copy already"):
        V.build(out, tmp_path / "twice_ref.npz", reserve=False)


def test_rows_only_skips_copies_whose_columns_were_cut(tmp_path):
    """The opt-in rule: a next-page cursor after the text is a column cut (kept by default, columns_cut with
    rows_only); a copy that only loses rows is kept either way."""
    cursor = glyph_row(screen(40, 220), 12, 16, 4, 150, 230)
    cursor[15:25, 185:195] = 230                                       # a solid cursor beyond the text
    rows_cut = drawn_cases()[0][1]
    pool = []
    for k, s in enumerate((cursor, rows_cut)):
        h, w = s.shape
        pool.append((s, "あい", {"id": f"v1_{k}_0", "vn": "v1", "box": [0, 0, w, h], "dark": 0, "inverted": False,
                                 "target": "vita"}))
    src = tmp_path / "c_grad.npz"
    write_pool(src, pool)
    boxes = V.refine_stored(cursor, False, 0)[0]
    assert len(boxes) == 1 and boxes[0][2] < cursor.shape[1]           # the cursor's columns are cut
    a = V.build(src, tmp_path / "a_ref.npz", reserve=False)
    b = V.build(src, tmp_path / "b_ref.npz", reserve=False, rows_only=True)
    assert a["counts"]["kept"] == 2 and "columns_cut" not in a["counts"] and a["rows_only"] is False
    assert b["counts"]["kept"] == 1 and b["counts"]["columns_cut"] == 1 and b["rows_only"] is True
    (crop, _, meta), = load(tmp_path / "b_ref.npz")
    assert meta["id"] == "v1_1_0" and crop.shape[1] == rows_cut.shape[1]


def test_summary_collects_counts_and_holds_no_label_text(pool, tmp_path):
    out = tmp_path / "s_ref.npz"
    V.build(pool[0], out, name="s_grad", reserve=False)
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps({"verdict": "DEVNATIVE_AGREE", "lines": 3, "screenshots": 2, "totals": {"x": 1},
                               "files": [], "flags": 31, "code": {}, "git": None, "disagree": []}))
    doc = V.summary([tmp_path / "s_ref.report.json"], dev)
    assert doc["totals"] == {"kept": 2, "unchanged": 2, "split": 2, "too_small": 2}
    assert doc["pools"][0]["name"] == "s_grad" and doc["devnative_agreement"]["disagree"] == 0
    text = json.dumps(doc, ensure_ascii=False)
    assert "あいう" not in text and "えお" not in text


def test_train2_reads_the_refined_copy(pool, tmp_path):
    from metrics import norm

    import train2 as T2
    from train import encoder
    out = tmp_path / "r_ref.npz"
    V.build(pool[0], out, reserve=False)
    items, stats = T2.load_shard(out, encoder(VOCAB), {}, norm)
    assert stats["kept"] == 2 and all(it.img.shape[0] == 32 for it in items)


def test_reserve_check_runs_on_the_output_with_dump_and_pixels(pool, tmp_path, monkeypatch):
    seen = {}
    real_run = V.subprocess.run

    class Done:
        returncode, stdout, stderr = 0, "t: 2 rows\nRESERVE_OK\n", ""

    def fake_run(cmd, **kw):
        if not str(cmd[1]).endswith("reserve_check.py"):
            return real_run(cmd, **kw)                                # git rev-parse
        seen["cmd"] = cmd
        Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps({"verdict": "RESERVE_OK"}))
        return Done()
    monkeypatch.setattr(V.subprocess, "run", fake_run)
    out = tmp_path / "c_ref.npz"
    rep = V.build(pool[0], out, name="c_ref", dump="/x/db")
    cmd = seen["cmd"]
    assert cmd[1].endswith("eval/tools/reserve_check.py") and cmd[2] == f"c_ref={out}"
    assert cmd[cmd.index("--dump") + 1] == "/x/db" and "--pixels" in cmd
    assert cmd[cmd.index("--out") + 1] == str(tmp_path / "c_ref.reserve_check.json")   # next to the output
    assert rep["reserve_check"]["verdict"] == "RESERVE_OK"
    assert json.loads((tmp_path / "c_ref.report.json").read_text())["reserve_check"]["verdict"] == "RESERVE_OK"


def test_pool_names_and_where_their_copies_go():
    p = V.pools("v3i7")
    assert set(p) == {"rp3_grad", "own_grad", "nat_grad", "dntv_grad", "dnst_grad"}
    assert p["rp3_grad"].name == "realplus_train_grad_v3i7.npz"
    assert not any(f.name.endswith("_dev.npz") for f in p.values())
    assert V.is_private(p["own_grad"]) and not any(V.is_private(f) for n, f in p.items() if n != "own_grad")
    assert V.out_path(p["own_grad"], True).parent == V.PRIVATE_OUT_DIR
    assert V.out_path(p["nat_grad"], False) == V.OUT_DIR / "native_train_grad_v1_ref.npz"


def test_ctc_frames_short():
    assert not V.ctc_frames_short("あい", (32, 24))                # 3 frames for 2 classes
    assert V.ctc_frames_short("ああ", (32, 16))                    # a repeat needs a blank: 3 > 2 frames
    assert not V.ctc_frames_short("ああ", (16, 12))                # 24 px at height 32: 3 frames


# ----------------------------------------------------------------------------- devnative agreement

def stored_of(region, box, dark):
    x, y, w, h = box
    c = region[y:y + h, x:x + w]
    return 255 - c if dark else c


def test_frame_agreement_on_a_drawn_dialogue_region():
    reg = screen(120, 260)
    reg[3, 4:256] = 230                                               # the box's border
    glyph_row(reg, 14, 16, 8, 250, 230)                               # line 1
    glyph_row(reg, 46, 12, 8, 250, 230)                               # line 2: two rows the finder may merge
    glyph_row(reg, 62, 12, 8, 250, 230)
    glyph_row(reg, 92, 16, 8, 120, 230)                               # line 3, short
    boxes, dark, _ = find_lines_ex(reg, "grad")
    assert len(boxes) >= 2
    for b, d in zip(boxes, dark):
        r = V.frame_agreement(reg, b, stored_of(reg, b, d), d, bool(d))
        assert r["box_reproduced"] and r["crop_reproduced"] and r["dark_equal"] and r["refine_agree"], (b, r)


def test_frame_agreement_on_procedural_regions_and_its_negative_cases():
    checked = 0
    for img in make_cases(5, 64):
        reg = grey_of(img)
        boxes, dark, _ = find_lines_ex(reg, "grad")
        alone = [V.refine_stored(stored_of(reg, b, d), bool(d), d)[0] for b, d in zip(boxes, dark)]
        if not boxes or sum(map(len, alone)) > 16:                    # the 16-line cap on splits can bind
            continue
        for b, d in zip(boxes, dark):
            r = V.frame_agreement(reg, b, stored_of(reg, b, d), d, bool(d))
            assert all(r[k] for k in ("box_reproduced", "crop_reproduced", "dark_equal", "refine_agree")), (b, r)
            checked += 1
    assert checked >= 20
    reg = grey_of(make_cases(5, 64)[5])
    boxes, dark, _ = find_lines_ex(reg, "grad")
    b, d = boxes[0], dark[0]
    other = stored_of(reg, b, d).copy()
    other[0, 0] ^= 0xFF
    assert not V.frame_agreement(reg, b, other, d, bool(d))["crop_reproduced"]
    assert not V.frame_agreement(reg, b, stored_of(reg, b, d), 1 - d, bool(1 - d))["dark_equal"]
    moved = (b[0] + 1, b[1], b[2] - 1, b[3])
    assert not V.frame_agreement(reg, moved, stored_of(reg, moved, d), d, bool(d))["box_reproduced"]
