"""data/console_crops.py (issue #129, VOCR6 #125 T3): console crops from a procedural page pool, end to end.

A tiny page pool is written to a temp dir: two pages of one title, a train page and an hval page. It has two dialogue
lines, renderer cells and a region, plus a frame strip on the region's edge, which the frozen finder cuts as a line
of its own. The build runs through eval/pagepath.py's finder driver, and the test checks:
- whole crops carry their line's text, pixel for pixel the region's box (inverted when dark);
- the strip is an empty crop;
- jittered crops hold the line's whole ink and are seeded (the same crops twice);
- a label outside the vocab goes to drop_*;
- hval stays in its own split;
- the manifest records every file's sha256;
- train/train2.py's shard reader takes the files.

No screenshots, fonts or models.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "data", ROOT / "eval", ROOT / "synth"):
    sys.path.insert(0, str(p))
import console_crops as CC  # noqa: E402

CCOMP = shutil.which("gcc") or shutil.which("cc")
PAPER, INK = 30, 225
TEXT1 = "あいうえおかきくけこさしすせそ"          # in data/vocab_vn_5631.txt
TEXT2 = "たちつてとなにぬねの"


def glyph(g, x0, y0, s):
    for y in range(s):
        for x in range(s):
            if x % 4 < 2 or y % 6 == 0 or y + 1 == s:
                g[y0 + y, x0 + x] = INK


def page(tid, n, split, text2, dark=False):
    """A 960x544 Vita frame: a frame strip on the region's top edge, two text rows; the record as the pool has it."""
    g = np.full((544, 960, 3), PAPER, np.uint8)
    lines = []
    for li, (text, y0) in enumerate(((TEXT1, 420), (text2, 452))):  # region-relative rows 16 and 48
        cells = []
        for i in range(len(text)):
            x0 = 40 + i * 24
            glyph(g, x0, y0, 20)
            cells.append([x0, y0, x0 + 22, y0 + 20])
        lines.append({"id": f"{tid}:{n}:line{li + 1}", "kind": "line", "vertical": False,
                      "box": [40, y0, 24 * len(text) - 2, 20], "box_main": None, "pad": 4, "text": text,
                      "cell_boxes": cells})
    for y in range(404, 416):                                          # the frame strip: 12 rows the finder keeps
        for x in range(30, 930):
            if x % 8 < 3 or (y - 404) % 5 == 0:
                g[y, x] = INK
    if dark:
        g = 255 - g
    region = [30, 404, 930, 480]
    rec = {"title_id": tid, "page": n, "image": f"{tid}/pages/{n}.png", "dims": [960, 544], "platform": "vita",
           "split": split, "lines": lines, "boxes": [{"xyxy": [40, 420, 400, 440]}, {"xyxy": [40, 452, 280, 472]}],
           "regions": [{"variant": "base", "region": region, "lines": [ln["id"] for ln in lines]}]}
    return g, rec


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if CCOMP is None:
        pytest.skip("no C compiler")
    cv2 = pytest.importorskip("cv2")
    d = tmp_path_factory.mktemp("cc")
    pool, pages = d / "pagepool", d / "pages"
    (pages / "T" / "pages").mkdir(parents=True)
    pool.mkdir()
    rows = []
    for n, split, t2, dark in (("000001", "train", TEXT2, False), ("000002", "train", "たちつ€てと", True),
                               ("000003", "hval", TEXT2, False)):
        img, rec = page("T", n, split, t2, dark)
        cv2.imwrite(str(pages / "T" / "pages" / f"{n}.png"), img[..., ::-1])
        rec["image_sha256"] = hashlib.sha256((pages / "T" / "pages" / f"{n}.png").read_bytes()).hexdigest()
        rows.append(rec)
    (pool / "T.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (pool / "manifest.json").write_text(json.dumps({"titles": {"T": {}}}))
    out = d / "out"
    args = ["build", "--pagepool", str(pool), "--pages", str(pages), "--finder", "rubi", "--sets", "train,hval",
            "--variants", "base", "--views", "console", "--jitter", "2", "--procs", "1", "--work", str(d / "work"),
            "--out", str(out)]
    CC.main(args)
    return d, out, pages, rows, args


def read(out, rel):
    p = out / rel
    return CC.read_npz(p) if p.exists() else []


def test_whole_crops_are_labelled_and_exact(built):
    d, out, pages, rows, _ = built
    cons = read(out, "train/console/T.npz")
    labels = sorted(lab for m, lab, c in cons)
    assert labels == sorted([TEXT1, TEXT2, TEXT1])                     # page 2's second line is out of vocab
    for m, lab, c in cons:
        assert m["kind"] == "whole" and m["view"] == "console" and m["variant"] == "base"
        img = CC.PP.Frames(pages, "console").grey("T", m["page"], "vita")
        x0, y0 = 30, 404
        x, y, w, h = m["box"]
        want = img[y0 + y:y0 + y + h, x0 + x:x0 + x + w]
        assert np.array_equal(c, 255 - want if m["inverted"] else want)
        assert m["inverted"] == (m["page"] == "000002")                 # the dark page's crops are inverted


def test_oov_label_goes_to_drop(built):
    d, out, *_ = built
    drop = read(out, "train/drop_console/T.npz")
    assert [lab for m, lab, c in drop] == ["たちつ€てと"] and drop[0][0]["oov"] == ["€"]


def test_frame_strip_is_an_empty_crop(built):
    d, out, *_ = built
    empty = read(out, "train/console_empty/T.npz")
    assert empty and all(lab == "" and m["kind"] == "empty" for m, lab, c in empty)
    assert all(m["box"][1] == 0 and m["box"][2] == 900 for m, lab, c in empty)   # the strip at the region's top


def test_jitter_holds_the_ink_and_is_seeded(built):
    d, out, pages, rows, args = built
    jit = read(out, "train/console_jitter/T.npz")
    assert jit and all(m["jitter"] in (1, 2) and lab in (TEXT1, TEXT2) for m, lab, c in jit)
    by_line = {}
    for r in rows:
        for ln in r["lines"]:
            by_line[ln["id"]] = ln
    for m, lab, c in jit:
        x, y, w, h = m["box"]
        ln = by_line[m["line"]]
        x0, y0 = 30, 404
        bx, by, bw, bh = ln["box"]
        assert x <= bx - x0 and y <= by - y0 and x + w >= bx - x0 + bw - 2 and y + h >= by - y0 + bh - 1
        assert m["box"] != m["finder_box"]
    again = d / "again"
    CC.main([*args[:-1], str(again)])
    a, b = read(out, "train/console_jitter/T.npz"), read(again, "train/console_jitter/T.npz")
    assert [(m["box"], lab) for m, lab, c in a] == [(m["box"], lab) for m, lab, c in b]


def test_splits_and_manifest(built):
    d, out, *_ = built
    hv = read(out, "hval/console/T.npz")
    assert hv and all(m["page"] == "000003" for m, lab, c in hv)
    assert all(m["page"] != "000003" for m, lab, c in read(out, "train/console/T.npz"))
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert man["schema"] == CC.SCHEMA and man["finder"] == "rubi" and man["sets"] == ["hval", "train"]
    for rel, v in man["files"].items():
        assert hashlib.sha256((out / rel).read_bytes()).hexdigest() == v["sha256"]
        assert len(CC.read_npz(out / rel)) == v["crops"]
    assert sorted(str(p.relative_to(out)) for p in out.rglob("*.npz")) == sorted(man["files"])
    assert man["counts"]["console:console"] == 5 and man["counts"]["console:drop_console"] == 1   # 3 train + 2 hval
    assert man["counts"]["console:box:whole"] == 6 and man["counts"]["console:console_empty"] == 3


def test_train2_reads_the_shards(built):
    d, out, *_ = built
    sys.path.insert(0, str(ROOT / "train"))
    pytest.importorskip("torch")
    import train2
    from common import load_vocab
    vocab = load_vocab(ROOT / "data" / "vocab_vn_5631.txt")
    items, st = train2.load_shard(str(out / "train/console/T.npz"), train2.encoder(vocab), {}, train2.norm)
    assert st["kept"] == 3 and st["dropped"] == 0 and all(it.img.shape[0] == 32 for it in items)
