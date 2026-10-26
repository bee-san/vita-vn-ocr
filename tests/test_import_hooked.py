"""data/import_hooked.py: splits, crops, duplicate clusters, re-flow and the whole import on a synthetic mirror."""
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
import import_hooked as IH  # noqa: E402

HAVE_ZSTD = shutil.which("zstd") is not None and shutil.which("tar") is not None


# ----------------------------------------------------------------------------- split

def ids(n):
    return [f"{i:06d}" for i in range(1, n + 1)]


def test_split_test_title_is_whole_and_small_title_is_train():
    assign, blocks = IH.split_title(ids(300), "test", "T1")
    assert set(assign.values()) == {"test"} and blocks == []
    assign, blocks = IH.split_title(ids(150), "train", "T2")
    assert set(assign.values()) == {"train"} and blocks == []
    with pytest.raises(ValueError):
        IH.split_title(ids(300), "dev", "T3")


@pytest.mark.parametrize("n", [271, 1000, 1640, 2047])
def test_split_blocks_guards_and_determinism(n):
    pages = ids(n)
    assign, blocks = IH.split_title(pages, "train", "PCSG00001", share=0.04, guard=15)
    assert IH.split_title(pages[::-1], "train", "PCSG00001", share=0.04, guard=15) == (assign, blocks)
    other = IH.split_title(pages, "train", "PCSG00002", share=0.04, guard=15)[1]
    assert other != blocks or n < 300                     # the title id seeds the placement
    order = [assign[p] for p in pages]
    H = round(0.04 * n)
    assert order.count("hval") == H
    assert len(blocks) == -(-H // 40)
    for first, last in blocks:                            # contiguous hval runs, a guard band either side
        a, b = pages.index(first), pages.index(last)
        assert all(order[i] == "hval" for i in range(a, b + 1))
        for i in range(max(0, a - 15), a):
            assert order[i] == "guard"
        for i in range(b + 1, min(n, b + 16)):
            assert order[i] == "guard"
    runs = [i for i in range(1, n) if order[i] == "hval" and order[i - 1] != "hval"] + \
        ([0] if order[0] == "hval" else [])
    assert len(runs) == len(blocks)
    # every guard page is within 15 pages of an hval page; nothing else leaves train
    hv = [i for i, s in enumerate(order) if s == "hval"]
    for i, s in enumerate(order):
        if s == "guard":
            assert min(abs(i - j) for j in hv) <= 15


def test_split_blocks_spread_over_segments():
    pages = ids(2047)
    _, blocks = IH.split_title(pages, "train", "PCSG00469")
    starts = [pages.index(b[0]) for b in blocks]
    for j, s in enumerate(starts):
        assert j * 2047 // 3 <= s < (j + 1) * 2047 // 3


# ----------------------------------------------------------------------------- crops and polarity

def text_crop(h=24, w=120, dark_text=False, seed=0, noise=0.0, bg=None):
    rng = np.random.default_rng(seed)
    base = 30 if bg is None else bg
    g = np.full((h, w), 225 if dark_text else base, np.uint8)
    ink = 20 if dark_text else 235
    for x in range(4, w - 8, 14):                         # glyph-like blocks
        g[5:h - 5, x:x + 3] = ink
        g[h // 2 - 1:h // 2 + 1, x:x + 9] = ink
    if noise:
        g = np.clip(g + rng.normal(0, noise, g.shape), 0, 255).astype(np.uint8)
    return g


def test_crop_rect_clips_at_the_page_edges():
    assert IH.crop_rect([0, 2, 10, 5], 4, 100, 50) == (0, 0, 14, 11)
    assert IH.crop_rect([90, 40, 10, 10], 4, 100, 50) == (86, 36, 100, 50)


def test_polarity_store_inverts_dark_text_only():
    light = text_crop(dark_text=False)
    dark = text_crop(dark_text=True)
    s1, d1 = IH.polarity_store(light)
    s2, d2 = IH.polarity_store(dark)
    assert d1 == 0 and np.array_equal(s1, light)
    assert d2 == 1 and np.array_equal(s2, 255 - dark)


def test_turn_box_matches_rot90():
    col = np.zeros((40, 10), np.uint8)
    col[12:20, 3:7] = 255                                 # a box x 3..7, y 12..20 in the column
    turned = np.rot90(col, 1)
    ys, xs = np.nonzero(turned)
    x0, y0, x1, y1 = IH.turn_box(10)([3, 12, 7, 20])
    assert (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) == (x0, y0, x1, y1)


def test_chars_in_needs_cells_that_spell_the_label():
    rec = {"cell_boxes": [[10, 5, 20, 25], [20, 5, 30, 25]], "cell_chars": ["あ", "い"]}
    assert IH.chars_in(rec, "あい", 8, 3) == [["あ", 2.0, 2.0, 12.0, 22.0], ["い", 12.0, 2.0, 22.0, 22.0]]
    assert IH.chars_in(rec, "あう", 8, 3) is None


def test_reflow_lays_cells_left_to_right():
    grey = np.full((120, 60), 40, np.uint8)
    cells = []
    for i in range(3):                                    # three 20 x 20 cells stacked in a column at x 20..40
        y = 10 + 22 * i
        grey[y + 4:y + 16, 24:36] = 200 + i
        cells.append([20.0, float(y), 40.0, float(y + 20)])
    rec = {"box": [20, 10, 20, 64], "cell_boxes": cells}
    strip, chars = IH.reflow(grey, rec, "あいう", pad=0)
    assert strip.shape[1] == 3 * 20 + 2 * IH.REFLOW_GAP
    assert [c[0] for c in chars] == list("あいう")
    for i, (_, x0, _, x1, _) in enumerate(chars):         # each cell's ink sits in its own slot
        part = strip[:, int(x0):int(x1)]
        assert part.max() == 200 + i
    assert IH.reflow(grey, rec, "あい", pad=0) == (None, None)


def test_label_doubt_flags_image_rows_labels_whose_first_character_lies_outside_the_crop():
    lead = {"source": "image_rows", "checks": {"count": {"uncited_lead_cells": 1}}}
    plain = {"source": "image_rows", "checks": {"count": {}}}
    cells = {"source": "glyph_cells"}
    assert IH.label_doubt(lead, "「あいうえお")             # the bracket is drawn left of the row's box
    assert not IH.label_doubt(lead, "あいうえお")
    assert not IH.label_doubt(plain, "「あいう」")
    assert IH.label_doubt(plain, "』かきくけこ")             # a closing bracket ends the row above
    assert not IH.label_doubt(cells, "」")                   # the renderer's own cells: a lone 」 line is real


# ----------------------------------------------------------------------------- duplicates

def test_clusters_exact_near_conflict_and_split_priority():
    a = text_crop(seed=1)
    a_near = np.clip(a.astype(int) + 3, 0, 255).astype(np.uint8)
    b = text_crop(w=150, seed=2)
    c = text_crop(seed=3, bg=120)                         # same label, another background: not near
    items = [("h1|line", "あいう", a, "hval", 0),         # an hval copy first in page order
             ("t1|line", "あいう", a.copy(), "train", 5),
             ("t2|line", "あいう", a_near, "train", 6),
             ("t3|line", "あいう", c, "train", 7),
             ("t4|line", "かき", b, "train", 8),
             ("x1|line", "さし", b.copy(), "train", 9)]  # b's pixels under another label
    cl = IH.clusters(items)
    assert cl["t1|line"] == ("t1|line", None)              # the train copy represents the cluster
    assert cl["h1|line"] == ("t1|line", "exact")
    assert cl["t2|line"] == ("t1|line", "near")
    assert cl["t3|line"] == ("t3|line", None)
    assert cl["t4|line"] == (None, "conflict") and cl["x1|line"] == (None, "conflict")


def test_near_needs_similar_shape():
    a = text_crop()
    assert IH.near(a, a.copy())
    assert not IH.near(a, text_crop(w=130))
    assert not IH.near(a, 255 - a)


# ----------------------------------------------------------------------------- the whole import

def zst(path, data):
    subprocess.run(["zstd", "-q", "-f", "-o", str(path)], input=data, check=True)


def fake_page(W, H, lines, dark_text=False):
    """An RGB page with a glyph-like stroke pattern per character (it depends on the character) in each box."""
    img = np.full((H, W, 3), 225 if dark_text else 30, np.uint8)
    ink = 20 if dark_text else 235
    for text, (x, y, w, h) in lines:
        cw = max(4, w // max(1, len(text)))
        for i, ch in enumerate(text):
            gx, k = x + i * cw, ord(ch)
            img[y + 2:y + h - 2, gx + 1 + k % 3:gx + 3 + k % 3] = ink
            r = y + 3 + (k // 3) % max(1, h - 8)
            img[r:r + 2, gx + 1:gx + max(2, cw - 2)] = ink
    return img


def build_mirror(root):
    """A one-title PSP mirror: 6 pages, a dark-on-light page, an OOV line, a CTC-infeasible line, a duplicate."""
    vocab = IH.load_vocab_set(IH.VOCAB)
    oov = "\U00020B9F"
    assert oov not in vocab and all(c in vocab for c in "あいうえおかきくけこ")
    tid, W, H = "NPJH99999", 480, 272
    folder = root / "psp" / f"{tid} Fake"
    folder.mkdir(parents=True)
    pages, lines, members = [], [], {}
    specs = {1: [("あいうえお", [40, 200, 90, 18])],
             2: [("かきくけこ", [40, 200, 90, 18]), ("あい", [40, 222, 40, 18])],
             3: [("あいうえお", [40, 200, 90, 18])],                      # page 1's line again: an exact duplicate
             4: [("あ" + oov + "う", [40, 200, 54, 18])], 5: [("あいうえおかきくけこ", [40, 200, 30, 18])],
             6: [("えおかき", [40, 200, 72, 18])]}
    for n, spec in specs.items():
        pid = f"{n:06d}"
        img = fake_page(W, H, spec, dark_text=n == 6)
        ok, png = cv2.imencode(".png", img[..., ::-1])
        members[f"pages/{pid}.png"] = png.tobytes()
        sha = hashlib.sha256(png.tobytes()).hexdigest()
        boxes = [{"xyxy": [b[0], b[1], b[0] + b[2], b[1] + b[3]], "kind": "line"} for _, b in spec]
        pages.append({"image": f"pages/{pid}.png", "image_sha256": sha, "boxes": boxes,
                      "calibration": {"dialogue_roi": [0, 180, W, H]}, "text_clean": "\n".join(t for t, _ in spec)})
        for i, (text, box) in enumerate(spec):
            lines.append({"id": f"{tid}:{pid}:line{i + 1}", "kind": "line", "title_id": tid, "platform": "psp",
                          "page": pid, "image": f"pages/{pid}.png", "image_sha256": sha, "dimensions": [W, H],
                          "box": box, "pad": 2, "vertical": False, "text": text, "text_hook": text,
                          "cell_boxes": [], "cell_chars": [], "source": "glyph_cells",
                          "source_check": "cell_count_match", "box_index": [i + 1], "split": "train",
                          "text_source": "stored"})
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    zst(folder / "pages.tar.zst", buf.getvalue())
    zst(folder / "pages.jsonl.zst", "".join(json.dumps(p) + "\n" for p in pages).encode())
    zst(folder / "lines.jsonl.zst", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lines).encode())
    files = {f: {"bytes": (folder / f).stat().st_size, "sha256": IH.sha256_file(folder / f)}
             for f in ("pages.tar.zst", "pages.jsonl.zst", "lines.jsonl.zst")}
    meta = {"title_id": tid, "title": "Fake", "platform": "psp", "resolution": [W, H], "pages": len(pages),
            "split": "train", "vndb": {"vn": "v999999"}, "files": files}
    (folder / "metadata.json").write_text(json.dumps(meta))
    return tid, specs, oov


@pytest.mark.skipif(not HAVE_ZSTD, reason="needs the zstd and tar binaries")
def test_import_end_to_end(tmp_path):
    mirror, art, man = tmp_path / "mirror", tmp_path / "art", tmp_path / "manifests"
    tid, specs, oov = build_mirror(mirror)
    IH.main(["extract", "--root", str(mirror), "--out", str(art / "pages"), "--procs", "1"])
    assert len(list((art / "pages" / tid / "pages").glob("*.png"))) == 6
    IH.main(["split", "--root", str(mirror), "--out", str(art / "splits.json"), "--min-pages", "1000"])
    splits = json.loads((art / "splits.json").read_text())
    assert set(splits["titles"][tid]["assign"].values()) == {"train"}
    IH.main(["lines", "--root", str(mirror), "--pages", str(art / "pages"), "--splits", str(art / "splits.json"),
             "--out", str(art / "pools"), "--procs", "1"])
    pools = json.loads((art / "pools" / "manifest.json").read_text())
    c = pools["titles"][tid]["counts"]
    assert c["records"] == 7
    assert c["line:oov"] == 1 and c["line:ctc"] == 1
    assert c["dup:train:line:exact"] == 1                  # page 3 repeats page 1
    rows = IH.read_npz(art / "pools" / "train" / "line" / f"{tid}.npz")
    labels = [lab for _, lab, _ in rows]
    assert sorted(labels) == sorted(["あいうえお", "かきくけこ", "あい", "えおかき"])
    for m, lab, g in rows:                                 # the crop is the page's box + pad, stored polarity
        x, y, w, h = m["box"]
        assert g.shape == (h + 4, w + 4) and m["crop_box"] == [x - 2, y - 2, x + w + 2, y + h + 2]
        assert m["inverted"] == (m["page"] == "000006")
        assert m["platform"] == "psp" and m["vn"] == "v999999" and m["hand_checked"] is False
    x2 = IH.read_npz(art / "pools" / "train" / "line_x2" / f"{tid}.npz")
    assert sorted(lab for _, lab, _ in x2) == sorted(labels)
    for m, lab, g in x2:
        x0, y0, x1, y1 = m["crop_box"]
        assert g.shape == (y1 - y0, x1 - x0) and m["scale"] == 2
    oov_rows = IH.read_npz(art / "pools" / "drop" / "oov" / "line" / f"{tid}.npz")
    assert [lab for _, lab, _ in oov_rows] == ["あ" + oov + "う"]
    # the shards load through train2 as a training source
    sys.path.insert(0, str(ROOT / "train"))
    import train2
    from common import load_vocab
    from train import encoder
    enc = encoder(load_vocab(IH.VOCAB))
    items, stats = train2.load_shard(art / "pools" / "train" / "line" / f"{tid}.npz", enc, {}, lambda s: s)
    assert stats["kept"] == 4 and all(it.img.shape[0] == 32 for it in items)
    IH.main(["pages", "--root", str(mirror), "--pages", str(art / "pages"), "--splits", str(art / "splits.json"),
             "--pools", str(art / "pools"), "--out", str(art / "pagepool"), "--procs", "1"])
    pp = IH.read_jsonl(art / "pagepool" / f"{tid}.jsonl")
    assert [p["complete"] for p in pp] == [True] * 6       # filters leave the text known
    p2 = pp[1]
    base = [g for g in p2["regions"] if g["variant"] == "base"]
    assert len(base) == 1 and base[0]["lines"] == [f"{tid}:000002:line1", f"{tid}:000002:line2"]
    assert base[0]["region"] == [34, 194, 136, 246]       # the union plus PSP's 6 px
    IH.main(["manifests", "--splits", str(art / "splits.json"), "--pools", str(art / "pools"),
             "--pagepool", str(art / "pagepool"), "--out", str(man)])
    text = "".join(p.read_text(encoding="utf-8") for p in man.iterdir())
    for spec in specs.values():                            # ids, hashes and codes only: no hooked text
        for label, _ in spec:
            assert label not in text
    rows = IH.read_jsonl(man / f"{tid}.jsonl")
    assert rows[0]["lines"] == {"line1": "line:train+line_x2:train"}
    assert rows[2]["lines"] == {"line1": "line:dup-exact+line_x2:dup-exact"}
    assert rows[3]["lines"] == {"line1": "line:o"}         # an OOV line keeps its native crop only (drop/oov)
    assert rows[4]["lines"]["line1"].startswith("line:c")


def write_title(root, tid, pages_spec, platform="psp", split="train"):
    """A one-title mirror from {page number: ([(text, box, extra record fields)], [renderer xyxy boxes] or None)};
    renderer boxes default to the records' boxes, each record citing its own index."""
    W, H = (480, 272) if platform == "psp" else (960, 544)
    folder = root / platform / f"{tid} Fake"
    folder.mkdir(parents=True)
    pages, lines, members = [], [], {}
    for n, (spec, rboxes) in sorted(pages_spec.items()):
        pid = f"{n:06d}"
        img = fake_page(W, H, [(t, b) for t, b, _ in spec])
        for x0, y0, x1, y1 in rboxes or []:                  # draw unlabelled text in boxes no record covers
            if not any(b[:2] == [x0, y0] for _, b, _ in spec):
                img[y0 + 2:y1 - 2, x0 + 2:x1 - 2:6] = 235
        ok, png = cv2.imencode(".png", img[..., ::-1])
        members[f"pages/{pid}.png"] = png.tobytes()
        sha = hashlib.sha256(png.tobytes()).hexdigest()
        rb = rboxes or [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for _, b, _ in spec]
        pages.append({"image": f"pages/{pid}.png", "image_sha256": sha,
                      "boxes": [{"xyxy": list(b), "kind": "line"} for b in rb]})
        for i, (text, box, extra) in enumerate(spec):
            rec = {"id": f"{tid}:{pid}:line{i + 1}", "kind": "line", "title_id": tid, "platform": platform,
                   "page": pid, "image": f"pages/{pid}.png", "image_sha256": sha, "dimensions": [W, H], "box": box,
                   "pad": 2, "vertical": False, "text": text, "text_hook": text, "cell_boxes": [], "cell_chars": [],
                   "source": "glyph_cells", "source_check": "cell_count_match", "box_index": [i + 1],
                   "split": split, "text_source": "stored"}
            rec.update(extra)
            lines.append(rec)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    zst(folder / "pages.tar.zst", buf.getvalue())
    zst(folder / "pages.jsonl.zst", "".join(json.dumps(p) + "\n" for p in pages).encode())
    zst(folder / "lines.jsonl.zst", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lines).encode())
    files = {f: {"bytes": (folder / f).stat().st_size, "sha256": IH.sha256_file(folder / f)}
             for f in ("pages.tar.zst", "pages.jsonl.zst", "lines.jsonl.zst")}
    (folder / "metadata.json").write_text(json.dumps({
        "title_id": tid, "title": "Fake", "platform": platform, "resolution": [W, H], "pages": len(pages),
        "split": split, "vndb": {"vn": "v999998"}, "files": files}))
    return folder


@pytest.mark.skipif(not HAVE_ZSTD, reason="needs the zstd and tar binaries")
def test_hval_targets_guard_copies_and_the_tall_band(tmp_path):
    """The screening set never holds a line with a train copy; hval_fold holds each hval cluster's representative; an
    hval copy of a guard line is screened but in no final pool; a tall furigana band is cut off and its page gets no
    region, while a normal furigana line keeps its band and gets a furimain copy."""
    a, b, c = [40, 200, 90, 18], [40, 200, 90, 18], [40, 200, 90, 18]
    tall = {"furigana": True, "box_main": [40, 182, 90, 18]}             # band 22 px over an 18-px line: 1.2x
    small = {"furigana": True, "box_main": [40, 208, 90, 18]}            # band 8 px: 0.44x, real ruby
    spec = {1: ([("あいうえお", a, {})], None),                           # train
            2: ([("かきくけこ", b, {})], None),                           # guard
            3: ([("あいうえお", a, {})], None),                           # hval, a copy of a train line
            4: ([("かきくけこ", b, {})], None),                           # hval, a copy of a guard line
            5: ([("さしすせそ", c, {})], None),                           # hval, new
            6: ([("さしすせそ", c, {})], None),                           # hval, a copy of page 5
            7: ([("たちつてと", [40, 160, 90, 40], tall)], [[40, 160, 130, 180], [40, 182, 130, 200]]),  # test
            8: ([("なにぬねの", [40, 200, 90, 26], small)], None)}        # train
    tid = "NPJH99998"
    folder = write_title(tmp_path / "mirror", tid, spec)
    pages_dir, out = tmp_path / "pages", tmp_path / "pools"
    IH.main(["extract", "--root", str(tmp_path / "mirror"), "--out", str(pages_dir), "--procs", "1"])
    assign = {"000001": "train", "000002": "guard", "000003": "hval", "000004": "hval", "000005": "hval",
              "000006": "hval", "000007": "test", "000008": "train"}
    vocab, vocab_s = IH.load_vocab_set(IH.VOCAB), IH.load_vocab_set(IH.VOCAB_S)
    _, counts, files = IH.import_title((folder, pages_dir, assign, out, vocab, vocab_s, IH.NEAR_MAD))

    def pages_of(split, pool="line"):
        f = out / split / pool / f"{tid}.npz"
        return sorted(m["page"] for m, _, _ in IH.read_npz(f)) if f.exists() else []
    assert pages_of("train") == ["000001", "000008"]
    assert pages_of("guard") == ["000002"]
    assert pages_of("hval") == ["000004", "000005"]          # screening: no train copy, one per cluster
    assert pages_of("hval_fold") == ["000005"]               # representatives only (page 4's is the guard copy)
    assert pages_of("test") == ["000007"]
    dec = {r["id"]: r for r in IH.read_jsonl(out / "decisions" / f"{tid}.jsonl")}
    seen = {k.split(":")[1]: v["pools"]["line"].get("seen_train") for k, v in dec.items() if k.split(":")[1] in
            ("000003", "000004", "000005", "000006")}
    assert seen == {"000003": True, "000004": False, "000005": False, "000006": False}
    assert dec[f"{tid}:000006:line1"]["pools"]["line"]["to"] == []          # a second hval copy: in no pool
    # the tall band: cut at box_main (+ pad), flagged, no furimain copy; the small band keeps its furimain copy
    (m7, lab7, g7), = IH.read_npz(out / "test" / "line" / f"{tid}.npz")
    assert m7["tall_band"] is True and m7["crop_box"] == [38, 180, 132, 202] and g7.shape == (22, 94)
    assert "furimain" not in dec[f"{tid}:000007:line1"]["pools"]
    assert pages_of("train", "furimain") == ["000008"]
    m8 = [m for m, _, _ in IH.read_npz(out / "train" / "line" / f"{tid}.npz") if m["page"] == "000008"][0]
    assert not m8.get("tall_band") and m8["crop_box"] == [38, 198, 132, 228]   # the band stays on a ruby line
    # the page pool: page 7 is incomplete and has no region; page 8 has one
    pp_dir = tmp_path / "pagepool"
    IH.import_pages((folder, assign, out, pp_dir))
    pp = {r["page"]: r for r in IH.read_jsonl(pp_dir / f"{tid}.jsonl")}
    assert pp["000007"]["complete"] is False and pp["000007"]["regions"] == []
    assert pp["000007"]["lines"][0]["tall_band"] is True
    assert pp["000008"]["complete"] is True and [g["variant"] for g in pp["000008"]["regions"]][0] == "base"
