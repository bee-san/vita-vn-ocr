"""data/console_crops.py (issue #129, VOCR6 #125 T3): console crops from a procedural page pool, end to end.

A tiny page pool is written to a temp dir: pages of one title, train and hval. Each has two dialogue lines,
renderer cells and a region, plus a frame strip on the region's top edge, which the frozen finder cuts as a line of
its own. The build runs through eval/pagepath.py's finder driver (built once, with two worker processes), and the
test checks:
- whole crops carry their line's text, pixel for pixel the region's box (inverted when dark);
- the strip is an empty crop, unless a renderer text box overlaps it;
- a line whose first word is drawn too dim for the finder is never labelled whole;
- the same crop from two region variants is stored once;
- jittered crops hold the line's whole ink and are seeded (the same crops twice);
- a label outside the vocab goes to drop_*;
- hval stays in its own split, and the test split is refused;
- the manifest records every file's sha256;
- train/train2.py's shard reader takes the files.

The labelling rules (classify, required_cells, jitter_boxes) are also unit-tested on small arrays. No screenshots,
fonts or models.
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
PAPER, INK, DIM = 30, 225, 90       # DIM: a contrast of 60, under the line's threshold (78) and over half of it
TEXT1 = "あいうえおかきくけこさしすせそ"          # in data/vocab_vn_5631.txt
TEXT2 = "たちつてとなにぬねの"
REGION = [30, 404, 930, 480]
GROW = {"base": (0, 0), "expanded-8": (8, 4)}       # (left/right/bottom, top) growth of a region variant


def glyph(g, x0, y0, s, v=INK):
    for y in range(s):
        for x in range(s):
            if x % 4 < 2 or y % 6 == 0 or y + 1 == s:
                g[y0 + y, x0 + x] = v


def origin(variant):
    gx, gy = GROW[variant]
    return REGION[0] - gx, REGION[1] - gy


def page(tid, n, split, text2, dark=False, dim=0, strip_text=False, variants=("base",)):
    """A 960x544 Vita frame: a frame strip on the region's top edge and two text rows (the first `dim` glyphs of the
    first drawn dim); the record as the pool has it (with strip_text, a renderer text box over the strip)."""
    g = np.full((544, 960, 3), PAPER, np.uint8)
    lines = []
    for li, (text, y0) in enumerate(((TEXT1, 420), (text2, 452))):  # region-relative rows 16 and 48
        cells = []
        for i in range(len(text)):
            x0 = 40 + i * 24
            dimmed = i in dim if isinstance(dim, tuple) else i < dim
            glyph(g, x0, y0, 20, DIM if li == 0 and dimmed else INK)
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
    boxes = [{"xyxy": [40, 420, 400, 440]}, {"xyxy": [40, 452, 280, 472]}]
    if strip_text:
        boxes.append({"xyxy": [600, 400, 700, 414]})                   # e.g. a name plate's box over the strip
    regions = []
    for v in variants:
        gx, gy = GROW[v]
        regions.append({"variant": v, "region": [REGION[0] - gx, REGION[1] - gy, REGION[2] + gx, REGION[3] + gx],
                        "lines": [ln["id"] for ln in lines]})
    rec = {"title_id": tid, "page": n, "image": f"{tid}/pages/{n}.png", "dims": [960, 544], "platform": "vita",
           "split": split, "lines": lines, "boxes": boxes, "regions": regions}
    return g, rec


PAGES = (("000001", "train", TEXT2, False, {}), ("000002", "train", "たちつ€てと", True, {}),
         ("000003", "hval", TEXT2, False, {}), ("000004", "train", TEXT2, False, {"dim": 3}),
         ("000005", "train", TEXT2, False, {"strip_text": True}),
         ("000006", "train", TEXT2, False, {"variants": ("base", "expanded-8")}),
         ("000007", "train", TEXT2, False, {"dim": (7,)}))


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
    for n, split, t2, dark, kw in PAGES:
        img, rec = page("T", n, split, t2, dark, **kw)
        cv2.imwrite(str(pages / "T" / "pages" / f"{n}.png"), img[..., ::-1])
        rec["image_sha256"] = hashlib.sha256((pages / "T" / "pages" / f"{n}.png").read_bytes()).hexdigest()
        rows.append(rec)
    (pool / "T.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (pool / "manifest.json").write_text(json.dumps({"titles": {"T": {}}}))
    out = d / "out"
    args = ["build", "--pagepool", str(pool), "--pages", str(pages), "--finder", "rubi", "--sets", "train,hval",
            "--variants", "base,expanded-8", "--views", "console", "--jitter", "2", "--procs", "2", "--work",
            str(d / "work"), "--out", str(out)]
    CC.main(args)
    return d, out, pages, rows, args


def read(out, rel):
    p = out / rel
    return CC.read_npz(p) if p.exists() else []


def test_whole_crops_are_labelled_and_exact(built):
    d, out, pages, rows, _ = built
    cons = read(out, "train/console/T.npz")
    by_page = {}
    for m, lab, c in cons:
        by_page.setdefault(m["page"], []).append(lab)
    assert sorted(by_page["000001"]) == sorted([TEXT1, TEXT2])
    assert by_page["000002"] == [TEXT1]                                # its second line is out of vocab
    assert by_page["000004"] == [TEXT2]                                # its first line's dim word: never whole
    assert sorted(by_page["000005"]) == sorted([TEXT1, TEXT2])
    assert sorted(by_page["000007"]) == sorted([TEXT1, TEXT2])          # a dim glyph mid-line, inside the box
    for m, lab, c in cons:
        assert m["kind"] == "whole" and m["view"] == "console"
        img = CC.PP.Frames(pages, "console").grey("T", m["page"], "vita")
        x0, y0 = origin(m["variant"])
        x, y, w, h = m["box"]
        want = img[y0 + y:y0 + y + h, x0 + x:x0 + x + w]
        assert np.array_equal(c, 255 - want if m["inverted"] else want)
        assert m["inverted"] == (m["page"] == "000002")                 # the dark page's crops are inverted


def test_dim_glyph_inside_the_box_stays_whole(built):
    d, out, pages, rows, _ = built
    cell = next(r for r in rows if r["page"] == "000007")["lines"][0]["cell_boxes"][7]
    jit = [m for m, lab, c in read(out, "train/console_jitter/T.npz") if m["page"] == "000007" and lab == TEXT1]
    assert jit
    for m in jit:                                                       # every jittered box still covers it
        x0, y0 = origin(m["variant"])
        x, y, w, h = m["box"]
        assert CC.covers((x + x0, y + y0, w, h), cell)


def test_dim_word_makes_the_box_partial(built):
    d, out, *_ = built
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert man["counts"]["console:box:partial"] >= 1                  # page 4: the finder's box leaves the dim word out
    assert all(not (m["page"] == "000004" and lab == TEXT1) for m, lab, c in read(out, "train/console_jitter/T.npz"))


def test_variants_are_deduplicated(built):
    d, out, *_ = built
    cons = [m for m, lab, c in read(out, "train/console/T.npz") if m["page"] == "000006"]
    assert len(cons) == 2 and {m["variant"] for m in cons} == {"base"}   # expanded-8's identical crops: not again
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert man["counts"]["console:console:duplicate"] >= 2


def test_oov_label_goes_to_drop(built):
    d, out, *_ = built
    drop = read(out, "train/drop_console/T.npz")
    assert [lab for m, lab, c in drop] == ["たちつ€てと"] and drop[0][0]["oov"] == ["€"]


def test_frame_strip_is_an_empty_crop(built):
    d, out, *_ = built
    empty = read(out, "train/console_empty/T.npz")
    assert empty and all(lab == "" and m["kind"] == "empty" for m, lab, c in empty)
    assert all(m["box"][1] == 0 and m["box"][2] == 900 for m, lab, c in empty if m["variant"] == "base")
    assert "000005" not in {m["page"] for m, lab, c in empty}          # a renderer text box overlaps it there
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert man["counts"]["console:empty_on_text"] >= 1


def test_jitter_holds_the_ink_and_is_seeded(built):
    d, out, pages, rows, args = built
    jit = read(out, "train/console_jitter/T.npz")
    assert jit and all(m["jitter"] in (1, 2) and lab in (TEXT1, TEXT2) for m, lab, c in jit)
    by_line = {ln["id"]: ln for r in rows for ln in r["lines"]}
    for m, lab, c in jit:
        x, y, w, h = m["box"]
        x0, y0 = origin(m["variant"])
        bx, by, bw, bh = by_line[m["line"]]["box"]
        assert x <= bx - x0 and y <= by - y0 and x + w >= bx - x0 + bw - 2 and y + h >= by - y0 + bh - 1
        assert m["box"] != m["finder_box"]
    again = d / "again"
    CC.main([*args[:-1], str(again)])
    a, b = read(out, "train/console_jitter/T.npz"), read(again, "train/console_jitter/T.npz")
    assert [(m["box"], lab) for m, lab, c in a] == [(m["box"], lab) for m, lab, c in b]


def test_splits_and_manifest(built):
    d, out, pages, rows, args = built
    hv = read(out, "hval/console/T.npz")
    assert hv and all(m["page"] == "000003" for m, lab, c in hv)
    assert all(m["page"] != "000003" for m, lab, c in read(out, "train/console/T.npz"))
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert man["schema"] == CC.SCHEMA and man["finder"] == "rubi" and man["sets"] == ["hval", "train"]
    for rel, v in man["files"].items():
        assert hashlib.sha256((out / rel).read_bytes()).hexdigest() == v["sha256"]
        assert len(CC.read_npz(out / rel)) == v["crops"]
    assert sorted(str(p.relative_to(out)) for p in out.rglob("*.npz")) == sorted(man["files"])
    # whole crops stored: pages 1, 5, 6, 7 two each, 2 and 4 one each (train, 10), page 3 two (hval)
    assert man["counts"]["console:console"] == 12 and man["counts"]["console:drop_console"] == 1
    with pytest.raises(SystemExit):
        CC.main([*args[:8], "train,test", *args[9:-1], str(d / "never")])  # the test split is refused
    assert args[7] == "--sets" and not (d / "never").exists()


def test_train2_reads_the_shards(built):
    d, out, *_ = built
    sys.path.insert(0, str(ROOT / "train"))
    pytest.importorskip("torch")
    import train2
    from common import load_vocab
    vocab = load_vocab(ROOT / "data" / "vocab_vn_5631.txt")
    items, st = train2.load_shard(str(out / "train/console/T.npz"), train2.encoder(vocab), {}, train2.norm)
    assert st["kept"] == 10 and st["dropped"] == 0 and all(it.img.shape[0] == 32 for it in items)


# ----------------------------------------------------------------------------- the rules, on small arrays

def test_classify_kinds():
    pts = [[(10, 10), (20, 10), None, (40, 10)], [(10, 30), (20, 30)]]
    assert CC.classify((0, 0, 50, 20), pts)[:2] == ("whole", 0)
    assert CC.classify((0, 0, 25, 20), pts)[:2] == ("partial", 0)
    assert CC.classify((0, 0, 50, 45), pts)[0] == "merged"
    assert CC.classify((60, 0, 20, 20), pts)[0] == "empty"
    # line 0's third cell holds a character too dim to ink: the box must cover half of it to be whole
    need = [[[30.0, 0.0, 36.0, 20.0]], []]
    assert CC.classify((0, 0, 50, 20), pts, need)[:2] == ("whole", 0)
    assert CC.classify((0, 0, 32, 20), pts, need)[0] == "partial"       # 2 of its 6 columns
    assert CC.classify((25, 0, 30, 20), pts, need)[:2] == ("partial", 0)
    assert CC.classify((29, 0, 6, 20), [[None, None, None, None], []], need)[:2] == ("whole", 0)  # the dim cell alone
    assert CC.classify((0, 0, 50, 45), pts, need)[0] == "merged"
    assert CC.covers((0, 0, 10, 10), [5, 0, 15, 10]) and not CC.covers((0, 0, 10, 10), [6, 0, 16, 10])


def test_required_cells():
    img = np.full((24, 80), PAPER, np.uint8)
    for x0, v in ((2, INK), (22, DIM), (62, INK)):                     # 3 glyphs; cell 2 (x 40-60) blank: a space
        for y in range(2, 22):
            for x in range(x0, x0 + 16):
                if x % 4 < 2 or y % 6 == 0:
                    img[y, x] = v
    cells = [[[0, 0, 20, 24], [20, 0, 40, 24], [40, 0, 60, 24], [60, 0, 80, 24]]]
    boxes = [[0, 0, 80, 24]]
    pts = CC.PP.ink_points(img, cells, boxes)
    assert [p is None for p in pts[0]] == [False, True, True, False]  # the dim glyph and the space have no ink point
    dim = [[20.0, 0.0, 40.0, 24.0]]
    assert CC.required_cells(img, cells, boxes, ["あいうえ"], pts) == [dim + [[40.0, 0.0, 60.0, 24.0]]]
    assert CC.required_cells(img, cells, boxes, ["あい\u3000え"], pts) == [dim]        # a space needs no cover
    # one cell more than characters: an inkless cell counts only with weak ink (the dim glyph, not the blank cell),
    # and only when there are more such cells than extra cells (then they cannot all be the line's blank extras)
    assert CC.required_cells(img, cells, boxes, ["あいう"], pts) == [[]]
    two = [cells[0] + [[80, 0, 100, 24]]]                               # one more cell: still one extra, now blank
    img2 = np.concatenate([img, np.full((24, 20), PAPER, np.uint8)], axis=1)
    pts2 = CC.PP.ink_points(img2, two, [[0, 0, 100, 24]])
    assert CC.required_cells(img2, two, [[0, 0, 100, 24]], ["あいうえ"], pts2) == [[]]
    tex = img2.copy()
    tex[2:22:3, 82:98:3] = PAPER + 50                                   # texture: weak "ink" in the blank extra cell
    pts3 = CC.PP.ink_points(tex, two, [[0, 0, 100, 24]])
    assert CC.required_cells(tex, two, [[0, 0, 100, 24]], ["あいうえ"], pts3) == [dim + [[80.0, 0.0, 100.0, 24.0]]]


def test_jitter_avoids_other_lines_and_keeps_required_cells():
    box, ink = (10, 10, 100, 20), [12, 12, 108, 28]
    above = [(60, 4)]                                                   # another line's inked cell just above
    jb = CC.jitter_boxes(box, ink, above, 200, 60, 6, 7)
    assert jb and all(not CC.PP.inside(b, above[0]) for b in jb)
    assert all(b[0] <= ink[0] and b[1] <= ink[1] and b[0] + b[2] >= ink[2] and b[1] + b[3] >= ink[3] for b in jb)
    assert jb == CC.jitter_boxes(box, ink, above, 200, 60, 6, 7)        # seeded
    need = [[108.0, 10.0, 130.0, 30.0]]                                 # a dim character after the inked ones
    jn = CC.jitter_boxes(box, ink, [], 200, 60, 6, 7, need)
    assert jn and all(CC.covers(b, need[0]) for b in jn)
    other = [[100.0, 30.0, 120.0, 40.0]]                                # another line's dim cell below
    free = CC.jitter_boxes(box, ink, [], 200, 60, 30, 7)
    assert any(CC.covers(b, other[0]) for b in free)                    # some boxes would reach it ...
    jo = CC.jitter_boxes(box, ink, [], 200, 60, 30, 7, (), other)
    assert jo and all(not CC.covers(b, other[0]) for b in jo)           # ... and are not kept
