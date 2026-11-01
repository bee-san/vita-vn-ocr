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
    # band 22 px over an 18-px line (1.2x); the record cites both renderer boxes, so only the tall-band rules can
    # make its page incomplete and keep it out of the regions
    tall = {"furigana": True, "box_main": [40, 182, 90, 18], "box_index": [1, 2]}
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


# ----------------------------------------------------------------------------- capture runs (pools v2)

def test_moved_page_and_record():
    assert IH.moved_page("000123", 0) == "000123" and IH.moved_page("000123", 1) == "100123"
    assert IH.moved_page("003292", 2) == "203292"
    with pytest.raises(ValueError):
        IH.moved_page("100000", 1)                         # outside the scheme: would collide with run 1's ids
    line = {"id": "T1:000007:line2", "page": "000007", "image": "pages/000007.png", "kind": "line"}
    m = IH.move_record(line, 1, "T1")
    assert (m["id"], m["page"], m["image"]) == ("T1:100007:line2", "100007", "pages/100007.png")
    assert (m["src_id"], m["src_page"], m["src_image"], m["run_index"]) == \
        ("T1:000007:line2", "000007", "pages/000007.png", 1)
    page = {"id": "000007", "image": "pages/000007.png"}
    assert IH.move_record(page, 2, "T1")["id"] == "200007" and IH.move_record(page, 2, "T1")["src_id"] == "000007"
    assert line["id"] == "T1:000007:line2"                  # the input is not changed
    ruby = {"id": "T1:000007:ruby1", "page": "000007", "image": "pages/000007.png", "host_line": "T1:000007:line2"}
    assert IH.move_record(ruby, 1, "T1")["host_line"] == "T1:100007:line2"      # the host line moves with the page


def test_split_runs_one_run_is_split_title():
    pages = ids(1000)
    texts = {p: f"t{p}" * 3 for p in pages}
    assert IH.split_runs([pages], "train", "PCSG00001", texts, {}) == \
        (*IH.split_title(pages, "train", "PCSG00001"), [])


def test_cross_run_guard_moves_repeats_of_held_out_pages():
    run0 = ids(1000)
    run1 = [IH.moved_page(p, 1) for p in ids(100)]          # a re-capture of the opening
    assign, blocks = IH.split_title(run0, "train", "PCSG00001")
    for p in run1:
        assign[p] = "train"
    hv = [p for p in run0 if assign[p] == "hval"]
    gd = [p for p in run0 if assign[p] == "guard"]
    texts = {p: f"page text {p}" for p in run0}
    texts.update({p: f"other text {p}" for p in run1})      # by default no run 1 page repeats a run 0 page
    t_h, t_g = hv[0], gd[0]
    texts[run1[10]] = texts[t_h]                           # repeats an hval page: guard, with a 15-page band
    texts[run1[80]] = texts[t_g]                           # repeats a guard page: guard, no band
    texts[run1[50]] = "short"                              # under CROSS_MIN characters: never matched
    labels = {run1[60]: {"あいうえおかきく"}, hv[1]: {"あいうえおかきく"}}   # a shared line label with an hval page
    a = dict(assign)
    moved = IH.cross_run_guard([run0, run1], a, texts, labels)
    band = set(run1[max(0, 10 - 15):10 + 16]) | set(run1[60 - 15:60 + 16])
    assert set(moved) == band | {run1[80]}
    assert all(a[p] == "guard" for p in moved) and all(a[p] == assign[p] for p in run0)   # run 0 unchanged
    assert a[run1[50]] == "train" or run1[50] in band


def write_run(root, folder_rel, tid, n_pages, texts, run, packaged, split="train", platform="vita"):
    """A one-run folder at root/folder_rel: n_pages pages, page k showing texts[k] as one line."""
    W, H = (480, 272) if platform == "psp" else (960, 544)
    folder = root / folder_rel
    folder.mkdir(parents=True)
    pages, lines, members = [], [], {}
    for k in range(1, n_pages + 1):
        pid = f"{k:06d}"
        text = texts[k]
        box = [40, 400, 20 * len(text), 24]
        img = fake_page(W, H, [(text, box)])
        img[0, 0, 0] = run                                  # the runs' frames differ by a pixel
        ok, png = cv2.imencode(".png", img[..., ::-1])
        members[f"pages/{pid}.png"] = png.tobytes()
        sha = hashlib.sha256(png.tobytes()).hexdigest()
        pages.append({"id": pid, "image": f"pages/{pid}.png", "image_sha256": sha, "text_clean": text,
                      "boxes": [{"xyxy": [box[0], box[1], box[0] + box[2], box[1] + box[3]], "kind": "line"}]})
        lines.append({"id": f"{tid}:{pid}:line1", "kind": "line", "title_id": tid, "platform": platform,
                      "page": pid, "image": f"pages/{pid}.png", "image_sha256": sha, "dimensions": [W, H],
                      "box": box, "pad": 4, "vertical": False, "text": text, "text_hook": text, "cell_boxes": [],
                      "cell_chars": [], "source": "glyph_cells", "source_check": "cell_count_match",
                      "box_index": [1], "split": split, "text_source": "stored"})
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    zst(folder / "pages.tar.zst", buf.getvalue())
    zst(folder / "pages.jsonl.zst", "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pages).encode())
    zst(folder / "lines.jsonl.zst", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lines).encode())
    files = {f: {"bytes": (folder / f).stat().st_size, "sha256": IH.sha256_file(folder / f)}
             for f in ("pages.tar.zst", "pages.jsonl.zst", "lines.jsonl.zst")}
    meta = {"title_id": tid, "title": "Fake", "platform": platform, "resolution": [W, H], "pages": n_pages,
            "characters": sum(len(texts[k]) for k in range(1, n_pages + 1)), "split": split,
            "vndb": {"vn": None}, "lines": {"verified_lines": n_pages}, "run": run, "packaged": packaged,
            "files": files}
    (folder / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False))
    return meta


@pytest.mark.skipif(not HAVE_ZSTD, reason="needs the zstd and tar binaries")
def test_two_run_title_end_to_end(tmp_path):
    """A title of two runs: the later run uploaded into the earlier run's folder (its first run recovered under
    versions/ and listed in index.json), a third recovered export that index.json does not list (left out). Run 1's
    pages and records are moved by 100000 and keep their own ids; run 0 splits as a title of its own; run 1's
    repeats of held-out pages are guard; the pools carry the run and the VN from the map."""
    mirror = tmp_path / "mirror"
    tid = "PCSG99999"
    kana = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
    texts0 = {k: kana[k % 40] + kana[(k // 40) % 40] + kana[(k // 1600) % 40] + "のこと" for k in range(1, 301)}
    assign0, blocks0 = IH.split_title([f"{k:06d}" for k in range(1, 301)], "train", tid)
    first = int(blocks0[0][0])
    src = {k: min(300, max(1, first - 10 + k)) for k in range(1, 41)}   # run 1 page k shows run 0 page src[k]
    texts1 = {k: texts0[src[k]] for k in range(1, 41)}       # a re-capture of the stretch around the hval block
    m0 = write_run(mirror, f"versions/vita/{tid} Fake/20261025T000000Z", tid, 300, texts0, 0, "2026-10-25T00:00:00Z")
    m1 = write_run(mirror, f"vita/{tid} Fake", tid, 40, texts1, 1, "2026-10-27T00:00:00Z")
    write_run(mirror, f"versions/vita/{tid} Fake/20261024T000000Z", tid, 2, texts0, 2, "2026-10-24T00:00:00Z")
    (mirror / "index.json").write_text(json.dumps({"titles": [
        {"title_id": tid, "pages": m["pages"], "characters": m["characters"],
         "lines": {"verified_lines": m["lines"]["verified_lines"]}} for m in (m0, m1)]}))
    runs = IH.title_runs(mirror)
    assert [f.name for f in runs[tid]] == ["20261025T000000Z", f"{tid} Fake"]   # packaged order; the export left out
    art = tmp_path / "art"
    IH.main(["extract", "--root", str(mirror), "--out", str(art / "pages"), "--procs", "1"])
    got = sorted(p.stem for p in (art / "pages" / tid / "pages").glob("*.png"))
    assert got == [f"{k:06d}" for k in range(1, 301)] + [f"{100000 + k:06d}" for k in range(1, 41)]
    IH.main(["split", "--root", str(mirror), "--out", str(art / "splits.json")])
    sp = json.loads((art / "splits.json").read_text())["titles"][tid]
    assert sp["hval_blocks"] == blocks0 and all(sp["assign"][p] == s for p, s in assign0.items())
    s0 = {k: assign0[f"{src[k]:06d}"] for k in range(1, 41)}
    hv1 = [k for k in range(1, 41) if s0[k] == "hval"]
    assert hv1                                               # run 1 repeats hval pages
    for k in range(1, 41):
        want = "guard" if s0[k] in ("hval", "guard") or any(abs(k - j) <= 15 for j in hv1) else "train"
        assert sp["assign"][f"{100000 + k:06d}"] == want, k
    assert sorted(sp["cross_run_guard"]) == sorted(f"{100000 + k:06d}" for k in range(1, 41)
                                                   if sp["assign"][f"{100000 + k:06d}"] == "guard")
    assert [r["index"] for r in sp["runs"]] == [0, 1] and sp["runs"][1]["page_offset"] == 100000
    vmap = tmp_path / "vndb_map.json"
    vmap.write_text(json.dumps({"titles": {tid: {"vns": ["v1234"], "source": "vndb-dump"}}}))
    IH.main(["lines", "--root", str(mirror), "--pages", str(art / "pages"), "--splits", str(art / "splits.json"),
             "--out", str(art / "pools"), "--procs", "1", "--vndb-map", str(vmap)])
    man = json.loads((art / "pools" / "manifest.json").read_text())
    assert man["pool_tag"] == "hooked-v2" and man["titles"][tid]["counts"]["records"] == 340
    rows = [r for s in ("train", "guard", "hval_fold") for f in [art / "pools" / s / "line" / f"{tid}.npz"]
            if f.exists() for r in IH.read_npz(f)]
    assert all(m["vn"] == "v1234" and m["pool"] == "hooked-v2" for m, _, _ in rows)
    later = [m for m, _, _ in rows if int(m["page"]) > 100000]
    assert later and all(m["run"] == 1 and m["src_id"] == f"{tid}:{m['src_page']}:line1" and
                         m["id"] == f"{tid}:{m['page']}:line1" for m in later)
    assert not any("run" in m for m, _, _ in rows if int(m["page"]) < 100000)    # run 0's rows as in v1
    IH.main(["pages", "--root", str(mirror), "--pages", str(art / "pages"), "--splits", str(art / "splits.json"),
             "--pools", str(art / "pools"), "--out", str(art / "pagepool"), "--procs", "1"])
    pp = {r["page"]: r for r in IH.read_jsonl(art / "pagepool" / f"{tid}.jsonl")}
    assert len(pp) == 340 and pp["100007"]["src_page"] == "000007" and pp["100007"]["run"] == 1
    assert "run" not in pp["000007"]
    IH.main(["manifests", "--splits", str(art / "splits.json"), "--pools", str(art / "pools"),
             "--pagepool", str(art / "pagepool"), "--out", str(tmp_path / "man")])
    mrows = {r["page"]: r for r in IH.read_jsonl(tmp_path / "man" / f"{tid}.jsonl")}
    assert mrows["100007"]["src"] == "1:000007" and "src" not in mrows["000007"]
    idx = json.loads((tmp_path / "man" / "index.json").read_text())
    assert idx["titles"][tid]["runs"][1]["run"] == 1 and idx["pool_tag"] == "hooked-v2"
    text = "".join(p.read_text(encoding="utf-8") for p in (tmp_path / "man").iterdir())
    assert not any(t in text for t in texts0.values())       # ids, hashes and codes only


# ----------------------------------------------------------------------------- pools v2's label rules

def test_doubled_cells_and_remapped():
    pair = [[0, 0, 36, 36], [2, 2, 34, 34]]                 # one glyph drawn twice under two keys
    rec = {"source": "glyph_cells", "vertical": False, "cell_boxes": pair * 3}
    rec["cell_boxes"] = [[x + 36 * (i // 2), y, xx + 36 * (i // 2), yy] for i, (x, y, xx, yy) in
                         enumerate(rec["cell_boxes"])]
    assert IH.doubled_cells(rec)
    plain = {"source": "glyph_cells", "vertical": False,
             "cell_boxes": [[30 * i, 0, 30 * i + 32, 32] for i in range(6)]}  # neighbours overlap 2 px: normal
    assert not IH.doubled_cells(plain)
    assert not IH.doubled_cells(dict(rec, source="image_rows"))
    assert IH.remapped("NPJH50882", "「あⅥⅧッ！」") and not IH.remapped("NPJH50882", "「あーーッ！」")
    assert not IH.remapped("PCSG00807", "ⅥⅧ")               # the classes are per title


def test_undrawn_check_strips_a_blank_code_and_refuses_a_shifted_one():
    grey = np.full((60, 200), 20, np.uint8)
    cells = [[10 + 30 * i, 10, 40 + 30 * i, 40] for i in range(5)]
    for i in (1, 2, 3):                                     # glyphs in cells 1-3; cells 0 and 4 blank
        x0 = 10 + 30 * i
        grey[16:34, x0 + 8:x0 + 22] = 230
    rec = {"text": "^あいう", "cell_chars": ["^", "あ", "い", "う"], "cell_boxes": cells[:4],
           "box": [10, 10, 150, 30], "source": "glyph_cells"}
    out, label = IH.undrawn_check(rec, grey, "^")
    assert label == "あいう" and out["cell_chars"] == ["あ", "い", "う"] and out["undrawn_removed"] == 1
    assert out["cell_boxes"] == cells[1:4] and rec["cell_chars"][0] == "^"     # the input is not changed
    trailing = dict(rec, text="^あいう^", cell_chars=["^", "あ", "い", "う", "^"], cell_boxes=cells)
    assert IH.undrawn_check(trailing, grey, "^") is None    # a code after the line's start: dropped, even on a blank
    on_ink = dict(rec, text="^いう", cell_chars=["^", "い", "う"], cell_boxes=cells[1:4])   # a code on a drawn glyph
    assert IH.undrawn_check(on_ink, grey, "^") is None
    blank = dict(rec, text="^あいうえ", cell_chars=["^", "あ", "い", "う", "え"], cell_boxes=cells)  # え on a blank
    assert IH.undrawn_check(blank, grey, "^") is None


@pytest.mark.skipif(not HAVE_ZSTD, reason="needs the zstd and tar binaries")
def test_label_rules_in_the_import(tmp_path, monkeypatch):
    """v2's rules in import_title: a record of a remapped codepoint and one of doubled cells are dropped as
    label_doubt (their pages lose their regions); --label-rules v1 keeps both, as v1 did."""
    tid = "NPJH99997"
    monkeypatch.setitem(IH.REMAPPED, tid, "Ⅷ")
    a, b = [40, 200, 90, 18], [40, 200, 72, 18]
    dbl = {"cell_boxes": [[40 + 24 * (i // 2) + (i % 2), 200, 64 + 24 * (i // 2) + (i % 2), 218] for i in range(6)],
           "cell_chars": list("「ふーん…」")}
    spec = {1: ([("あいうえお", a, {})], None), 2: ([("かⅧき", b, {})], None), 3: ([("「ふーん…」", b, dbl)], None)}
    folder = write_title(tmp_path / "mirror", tid, spec)
    pages_dir = tmp_path / "pages"
    IH.main(["extract", "--root", str(tmp_path / "mirror"), "--out", str(pages_dir), "--procs", "1"])
    assign = {"000001": "train", "000002": "train", "000003": "train"}
    vocab, vocab_s = IH.load_vocab_set(IH.VOCAB), IH.load_vocab_set(IH.VOCAB_S)
    vocab = vocab | {"Ⅷ"}                                   # let the remapped character pass the vocab filter
    for rules, out in (("v2", tmp_path / "v2"), ("v1", tmp_path / "v1")):
        _, counts, _ = IH.import_title((folder, pages_dir, assign, out, vocab, vocab_s, IH.NEAR_MAD, "t", None, rules))
        dec = {r["id"]: r for r in IH.read_jsonl(out / "decisions" / f"{tid}.jsonl")}
        got = {k.split(":")[1]: (v.get("drop"), v.get("label_rule")) for k, v in dec.items()}
        if rules == "v2":
            assert got == {"000001": (None, None), "000002": ("label_doubt", "remapped"),
                           "000003": ("label_doubt", "doubled_cells")}
            assert counts["label_rule:remapped"] == 1 and counts["record:label_doubt"] == 2
            IH.import_pages((folder, assign, out, tmp_path / "pp"))
            pp = {r["page"]: r for r in IH.read_jsonl(tmp_path / "pp" / f"{tid}.jsonl")}
            assert pp["000002"]["regions"] == [] and pp["000002"]["untrusted"] and pp["000001"]["regions"]
        else:
            assert all(d is None and r is None for d, r in got.values())
