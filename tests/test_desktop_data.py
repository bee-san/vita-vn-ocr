"""desktopdata/ (docs/desktop-data.md): the capture model's exact box maps, the 1a plan and label maps, the 1b labels,
the 1c renderer's boxes, the PC test set's rules and review plumbing, the shard format, the stats, and the committed
manifests. Fast, no network, no screenshots (the 1c test renders one line with CI's font when it is cached)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "realplus"), str(ROOT / "synth")):
    if p not in sys.path:
        sys.path.insert(0, p)

from desktopdata import capture as C  # noqa: E402
from desktopdata import hooked as HK  # noqa: E402
from desktopdata import pctest as PT  # noqa: E402
from desktopdata import realplus as RP  # noqa: E402
from desktopdata import shards as SH  # noqa: E402
from desktopdata import stats as ST  # noqa: E402

MAN = ROOT / "data" / "manifests" / "desktop_data"


# ----------------------------------------------------------------------------- the capture model

def ink_box(img, thr=127):
    g = img.max(2) if img.ndim == 3 else img
    ys, xs = np.nonzero(g > thr)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


BOXES = ([200, 100, 520, 132], [40, 400, 900, 430], [700, 20, 760, 70])


def frame(w=960, h=544, boxes=BOXES):
    img = np.zeros((h, w, 3), np.uint8)
    for x0, y0, x1, y1 in boxes:
        img[y0:y1, x0:x1] = 255
    return img


@pytest.mark.parametrize("resampler", ["nearest", "bilinear", "bicubic", "lanczos", "sharp_bilinear",
                                       "bilinear_sharpen"])
@pytest.mark.parametrize("screen,place", [((1920, 1080), None), ((2560, 1600), None), ((3440, 1440), None),
                                          ((1024, 768), None), ((800, 600), None), ((1600, 1200), "stretch")])
def test_capture_maps_every_box_to_its_ink(resampler, screen, place):
    w, h = 960, 544
    pl = (0, 0, *screen) if place == "stretch" else C.fit_place(w, h, *screen)
    cap = C.Capture(w, h, "stretch" if place else "fullscreen", screen, pl, resampler,
                    sharpen=(0.6, 1.0) if resampler == "bilinear_sharpen" else None)
    for b in BOXES:
        out = cap.apply(frame(boxes=[b]))
        got = ink_box(out)
        want = cap.box(b)
        assert all(abs(g - x) <= 1 for g, x in zip(got, want)), (resampler, screen, b, got, want)


def test_capture_bars_dpi_downscale_and_jpeg_compose_exactly():
    w, h = 480, 272
    rng = np.random.default_rng(5)
    for _ in range(40):
        cap = C.sample(rng, w, h, "psp")
        cap.jpeg = None
        img = frame(w, h, boxes=[[100, 50, 300, 80]])
        out = cap.apply(img)
        assert out.shape == (cap.FH, cap.FW, 3)
        want = cap.box([100, 50, 300, 80])
        got = ink_box(out)
        assert all(abs(g - x) <= 1 for g, x in zip(got, want)), (cap.record(), got, want)
    cap = C.Capture(960, 544, "fullscreen", (1920, 1200), C.fit_place(960, 544, 1920, 1200), "bilinear",
                    bars=(20, 20, 20))
    out = cap.apply(np.full((544, 960, 3), 200, np.uint8))
    x0, y0, x1, y1 = cap.content_rect()
    assert (x0, x1) == (0, 1920) and y0 > 0 and y1 < 1200          # letterbox: bars above and below
    assert (out[0, 0] == 20).all() and (out[600, 960] == 200).all()
    pill = C.Capture(960, 544, "fullscreen", (3440, 1440), C.fit_place(960, 544, 3440, 1440), "nearest")
    a, b, c, d = pill.content_rect()
    assert a > 0 and c < 3440 and (b, d) == (0, 1440)             # 21:9: pillarbox
    dpi = C.Capture(960, 544, "window_free", (1500, 850), C.fit_place(960, 544, 1500, 850), "bilinear", dpi=1.5,
                    down=0.6, jpeg=(85, 420))
    out = dpi.apply(frame())
    assert out.shape == (dpi.FH, dpi.FW, 3) == (round(850 * 0.6), round(1500 * 0.6), 3)
    assert dpi.record()["affine"][0] == pytest.approx(dpi.FW / 1500 * 1500 / dpi.LW * dpi.logical_place()[2] / 960)


def test_capture_mix_is_deterministic_and_covers_the_setups():
    a = [C.sample(np.random.default_rng(3), 960, 544, "vita").record() for _ in range(2)]
    assert a[0] == a[1]
    rng = np.random.default_rng(0)
    seen = {C.sample(rng, 960, 544, "vita").setup for _ in range(400)}
    assert seen == {"fullscreen", "window_multiple", "window_free", "stretch"}


# ----------------------------------------------------------------------------- 1a

def test_allocate_spreads_pages_by_title_weight_and_caps_at_the_title():
    sizes = {"A": 10000, "B": 100, "C": 5, "D": 2500}
    a = HK.allocate(sizes, 1000)
    assert sum(a.values()) == 1000 and a["C"] == 5 and a["A"] > a["D"] > a["B"]
    assert HK.allocate(sizes, 10**6) == sizes
    assert HK.allocate(sizes, 1000) == a


def test_hooked_labels_follow_the_capture():
    cap = C.Capture(960, 544, "fullscreen", (1920, 1080), C.fit_place(960, 544, 1920, 1080), "bilinear")
    lines = HK.map_lines(cap, [[100, 400, 500, 430, 0, "dialogue", 1.0], [10, 10, 40, 30, 0, "notdlg", 0.7]])
    ax, bx, ay, by = cap.affine()
    assert lines[0][:4] == [int(np.floor(ax * 100 + bx)), int(np.floor(ay * 400 + by)),
                            int(np.ceil(ax * 500 + bx)), int(np.ceil(ay * 430 + by))]
    assert lines[0][4:] == [0, "dialogue", 1.0] and lines[1][5:] == ["notdlg", 0.7]
    assert HK.map_ignore(cap, [[0, 0, 5, 5]])[0][:2] == [int(bx), int(by)]


# ----------------------------------------------------------------------------- 1b

def test_realplus_labels_scale_refine_and_ignore_the_rest():
    W, H = 1920, 1080
    s = RP.fit_scale(W, H)
    rgb = np.full((H, W, 3), 30, np.uint8)
    rgb[600:640, 200:900] = 240           # the kept line (native px)
    rgb[700:740, 200:700] = 240           # a line the reads found but realplus did not keep
    rgb[50:70, 1700:1800] = 240           # UI text only MeikiOCR's 0.2 pass found
    fit = lambda b: [int(v * s) for v in b]  # noqa: E731
    rd = {"lines": [{"k": 0, "ink": fit([200, 600, 900, 640]), "crop_ink": fit([200, 600, 900, 640])},
                    {"k": 1, "ink": fit([200, 700, 700, 740]), "crop_ink": fit([200, 700, 700, 740])}],
          "dets": [{"det": fit([198, 598, 902, 642])}, {"det": fit([198, 698, 702, 742])}]}
    row = {"lines": [{"k": 0, "name": False, "vita_ink": [v + d for v, d in zip(fit([200, 600, 900, 640]),
                                                                              (-4, -4, 4, 4))]}]}
    lines, ignore, c = RP.labels_of(rgb, row, rd, [[1698, 48, 1802, 72, 0.3], [199, 599, 901, 641, 0.9]])
    assert len(lines) == 1 and lines[0][4:] == [0, "text", 1.0]
    assert all(abs(a - b) <= 2 for a, b in zip(lines[0][:4], [200, 600, 900, 640]))
    assert c["positive_ink"] == 1 and c["other_line_ignore"] == 1 and c["det02_ignore"] == 1
    assert c["det05_ignore_dup"] == 1                     # the same unkept line again: one area
    assert any(b[0] <= 200 and b[1] <= 700 and b[2] >= 700 and b[3] >= 740 for b in ignore)
    assert any(b[0] <= 1700 and b[2] >= 1800 for b in ignore)
    assert not any(RP.iou(b, lines[0][:4]) > 0.2 for b in ignore)


# ----------------------------------------------------------------------------- the PC test set

def fake_dump():
    rel = {f"r{i}": {"engine": "KiriKiri", "reso": [1280, 720], "released": "20250101"} for i in range(1, 12)}
    return {
        "olang": {"v1": "ja", "v2": "ja", "v3": "ja", "v4": "en", "v5": "ja", "v6": "ja", "v7": "ja", "v8": "ja",
                  "v9": "ja", "v10": "ja"},
        "title": {f"v{i}": [f"title {i}"] for i in range(1, 11)},
        "img": {i: (1920, 1080, 0.0, 0.0) for i in range(100, 130)} | {120: (1920, 1080, 0.9, 0.0),
                                                                        121: (500, 400, 0.0, 0.0),
                                                                        122: (1280, 720, 0.0, 0.0)},
        "plat": {"r1": {"win"}, "r2": {"win"}, "r3": {"win"}, "r4": {"win"}, "r5": {"swi"}, "r6": {"win"},
                 "r7": {"win"}, "r8": {"psv"}, "r9": {"win"}, "r10": {"win"}, "r11": {"win"}},
        "langs": {f"r{i}": {"ja"} for i in range(1, 12)} | {"r7": {"en"}},
        "rel_vns": {"r1": {"v1"}, "r2": {"v2", "v9"}, "r3": {"v3"}, "r4": {"v4"}, "r5": {"v5"}, "r6": {"v6"},
                    "r7": {"v7"}, "r8": {"v8"}, "r9": {"v9"}, "r10": {"v10", "v9"}, "r11": {"v10"}},
        "relations": {"v6": {"v9"}},
        "release": rel,
        # (vn, image, release)
        "shots": [("v1", 100, "r1"), ("v2", 101, "r2"), ("v3", 102, "r3"), ("v4", 103, "r4"), ("v5", 104, "r5"),
                  ("v6", 105, "r6"), ("v7", 106, "r7"), ("v8", 107, "r8"), ("v9", 108, "r9"), ("v3", 120, "r3"),
                  ("v3", 121, "r3"), ("v1", 122, "r1"), ("v10", 109, "r10"), ("v10", 110, "r11")],
    }


def test_pctest_candidates_follow_the_rules():
    d = fake_dump()
    pool, reserved = {"v9": ["realplus"]}, {"v8": "test-set"}
    rows, why = PT.candidates(d, pool, reserved)
    got = {r["shot"]: r for r in rows}
    assert set(got) == {"v1_100", "v3_102", "v5_104", "v6_105", "v1_122", "v10_110"}
    assert why["pool"] == 1 and why["reserved"] == 1 and why["not-japanese-original"] == 1
    assert why["shared-release"] == 2           # v2's and v10's screenshot on a release holding pool VN v9
    assert why["rating"] == 1 and why["size"] == 1 and why["no-japanese-release"] == 1
    assert got["v5_104"]["group"] == "console-hd" and got["v1_100"]["group"] == "pc"
    assert got["v6_105"]["family_of"] == ["v9"] and got["v1_100"]["family_of"] == []
    assert got["v10_110"]["release_partner_of"] == ["v9"]
    assert sorted(r["order"] for r in rows) == list(range(len(rows)))
    rows2, why2 = PT.candidates(d, pool, reserved, family="drop")
    assert "v6_105" not in {r["shot"] for r in rows2} and why2["family"] == 1


def test_last_decision_line_wins():
    t = '{"sheet": "S001", "n": 1, "pick": 1}\n{"sheet": "S001", "n": 2, "skip": "ui"}\n' \
        '{"sheet": "S001", "n": 1, "text": "「はい」"}\n{"sheet": "S001", "default": 1}\n'
    out = [json.loads(x) for x in PT.last_wins(t).splitlines()]
    assert out == [{"sheet": "S001", "n": 2, "skip": "ui"}, {"sheet": "S001", "n": 1, "text": "「はい」"},
                   {"sheet": "S001", "default": 1}]


def read_row(shot, vn, lines, dims=(1920, 1080)):
    return {"shot": shot, "vn": vn, "dims": list(dims), "fit": PT.fit_scale(*dims),
            "lines": [{"k": k, "ink": b, "edge": False, "reads": {"meiki": [t, []]}} for k, (b, t) in
                      enumerate(lines)]}


def test_select_takes_japanese_screens_in_rank_order_with_caps():
    jp, en = "「今日はいい天気ですね」", "Hello world"
    reads = {"v1_1": read_row("v1_1", "v1", [([100, 900, 900, 940], jp), ([100, 950, 900, 990], jp)]),
             "v1_2": read_row("v1_2", "v1", [([100, 900, 900, 940], en)]),
             "v1_3": read_row("v1_3", "v1", [([100, 900, 900, 902], jp), ([100, 10, 900, 300], jp)]),
             "v2_1": read_row("v2_1", "v2", [([100, 900, 900, 924], jp)], dims=(1280, 720))}
    cands = [{"shot": s, "rank": i, "order": i} for i, s in enumerate(reads)]
    picked, rec, why = PT.select_lines(reads, cands)
    assert set(picked) == {"v1_1_0", "v1_1_1", "v2_1_0"}
    assert rec == {"v1": {"shots": 1, "lines": 2}, "v2": {"shots": 1, "lines": 1}}
    assert why["shot-not-japanese"] == 1 and why["line-size"] == 2 and why["shot-no-line"] == 1
    assert PT.native_pad(1920, 1080) == 8 and PT.native_pad(640, 480) == 4 and PT.native_margin(3840, 2160) == 48


def test_units_group_lines_and_only_clean_ones_score():
    W, H = 1920, 1080
    kept = [{"id": "a", "text_box": [100, 800, 900, 840]}, {"id": "b", "text_box": [100, 860, 800, 900]},
            {"id": "c", "text_box": [100, 100, 600, 140]}]
    us = PT.units_of({}, kept, [[1000, 820, 1100, 840], [1500, 50, 1600, 80]], W, H)
    assert [len(u["lines"]) for u in us] == [1, 2]                 # top to bottom: c, then a + b
    assert us[0]["clean"] and us[1]["clean"]
    us = PT.units_of({}, kept, [[850, 820, 905, 838]], W, H)      # a piece of line a: still clean
    assert us[1]["clean"]
    us = PT.units_of({}, kept, [[905, 842, 921, 858]], W, H)      # unlabelled text in the a + b region's margin
    assert not us[1]["clean"] and us[0]["clean"]


# ----------------------------------------------------------------------------- shards and stats

def test_shard_roundtrip_and_record(tmp_path):
    g1 = np.arange(12, dtype=np.uint8).reshape(3, 4)
    g2 = np.full((2, 5), 7, np.uint8)
    samples = [{"src": "x", "kind": "pc", "lines": [[0, 0, 2, 2, 0, "text", 1.0]], "ignore": [], "meta": {}},
               {"src": "x", "kind": "pc", "lines": [], "ignore": [[1, 1, 2, 2]], "meta": {}}]
    SH.write_shard(tmp_path / "s0", samples, [g1, g2])
    data, back = SH.read_shard(tmp_path / "s0")
    assert (SH.canvas(data, back[0]) == g1).all() and (SH.canvas(data, back[1]) == g2).all()
    rec = SH.shard_record(tmp_path / "s0", "d")
    assert rec["samples"] == 2 and rec["lines"] == 1 and rec["ignore"] == 1 and rec["by_size"] == {"4x3": 1, "5x2": 1}
    assert SH.totals([rec, rec])["all"]["samples"] == 4
    assert (SH.to_gray(np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], np.uint8)) == [[76, 149, 28]]).all()


def test_stats_classes_and_fit():
    assert ST.res_class(480) == "<=640" and ST.res_class(1920) == "1920-2559" and ST.res_class(3840) == ">=3840"
    assert ST.fit960(1920, 1080) == 0.5 and ST.fit960(640, 480) == 1.0
    d = ST.dist([10, 20, 30, 40])
    assert d["n"] == 4 and d["p50"] == 25.0 and sum(d["hist"].values()) == 4


# ----------------------------------------------------------------------------- 1c

def ci_font():
    import os
    cache = Path(os.environ.get("VOCR_FONT_CACHE", Path.home() / ".cache" / "vita-vn-ocr" / "fonts"))
    hits = sorted(cache.rglob("MPLUS1p-Regular.ttf"))
    return hits[0] if hits else None


def test_synth_line_box_is_its_coverage():
    font_path = ci_font()
    if font_path is None:
        pytest.skip("CI's font is not cached (ci/fonts.py)")
    from PIL import ImageFont

    import textspot_data as TSD
    from desktopdata import synth as SY
    SY.TS = TSD
    fr = TSD.Frame("pc", 1280, 720, "vita", 720 / 544, None)
    fr.img = np.zeros((720, 1280, 3), np.float32)
    font = ImageFont.truetype(str(font_path), 40)
    st = {"fill": np.array([255, 255, 255], np.float32), "stroke": 3, "stroke_c": np.zeros(3, np.float32),
          "shadow": {"dx": 2, "dy": 2, "blur": 0.0, "c": np.zeros(3, np.float32), "a": 1.0}}
    bb = SY.draw_text(fr, np.random.default_rng(0), 100, 300, "「こんにちは、世界」", font, st, "dialogue")
    assert bb is not None and len(fr.inst) == 1
    lines, ignore = SY.instance_boxes(fr)
    x0, y0, x1, y1 = lines[0][:4]
    assert lines[0][4:] == [0, "dialogue"] and 30 <= y1 - y0 <= 60
    assert abs(x0 - bb[0]) <= 1 and abs(y0 - bb[1]) <= 1 and x1 >= bb[2] and y1 >= bb[3]   # + the hard shadow
    lit = np.nonzero(fr.img.max(2) > 127)
    assert lit[1].min() >= x0 and lit[1].max() < x1 and lit[0].min() >= y0 and lit[0].max() < y1


# ----------------------------------------------------------------------------- the owner's desktop captures

def test_owner_desktop_capture_import_end_to_end(tmp_path):
    import hashlib
    import io
    import shutil
    import tarfile

    import cv2
    if not (shutil.which("zstd") and shutil.which("tar")):
        pytest.skip("needs the zstd and tar binaries")
    sys.path.insert(0, str(ROOT / "data"))
    import import_hooked as IH
    import test_import_hooked as TIH
    from desktopdata import owner as OW
    W, H, tid = 1920, 1080, "PCDT00001"
    folder = tmp_path / "bucket" / "pc" / f"{tid} Fake desktop capture"
    folder.mkdir(parents=True)
    spec = {1: [("あいうえお", [200, 900, 400, 48]), ("かきくけこ", [200, 960, 400, 48])],
            2: [("さしすせそ", [300, 880, 400, 48])]}
    pages, lines, members = [], [], {}
    for n, sp in spec.items():
        pid = f"{n:06d}"
        ok, png = cv2.imencode(".png", TIH.fake_page(W, H, sp)[..., ::-1])
        members[f"pages/{pid}.png"] = png.tobytes()
        sha = hashlib.sha256(png.tobytes()).hexdigest()
        pages.append({"image": f"pages/{pid}.png", "image_sha256": sha,
                      "boxes": [{"xyxy": [b[0], b[1], b[0] + b[2], b[1] + b[3]], "kind": "line"} for _, b in sp]})
        for i, (text, box) in enumerate(sp):
            lines.append({"id": f"{tid}:{pid}:line{i + 1}", "kind": "line", "title_id": tid, "platform": "pc",
                          "page": pid, "image": f"pages/{pid}.png", "image_sha256": sha, "dimensions": [W, H],
                          "box": box, "pad": 4, "vertical": False, "text": text, "text_hook": text, "cell_boxes": [],
                          "cell_chars": [], "source": "glyph_cells", "source_check": "cell_count_match",
                          "box_index": [i + 1], "split": "train", "text_source": "stored"})
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    TIH.zst(folder / "pages.tar.zst", buf.getvalue())
    TIH.zst(folder / "pages.jsonl.zst", "".join(json.dumps(p) + "\n" for p in pages).encode())
    TIH.zst(folder / "lines.jsonl.zst", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lines).encode())
    files = {f: {"bytes": (folder / f).stat().st_size, "sha256": IH.sha256_file(folder / f)}
             for f in ("pages.tar.zst", "pages.jsonl.zst", "lines.jsonl.zst")}
    (folder / "metadata.json").write_text(json.dumps({
        "title_id": tid, "title": "Fake", "platform": "pc", "resolution": [W, H], "pages": len(pages),
        "split": "train", "vndb": {"vn": "v999997"}, "files": files}))
    work, out = tmp_path / "work", tmp_path / "shards"
    OW.main(["import", "--root", str(tmp_path / "bucket"), "--work", str(work), "--procs", "1", "--min-pages", "1000"])
    pp = IH.read_jsonl(work / "pagepool" / f"{tid}.jsonl")
    assert [p["dims"] for p in pp] == [[W, H]] * 2 and all(p["platform"] == "pc" for p in pp)
    base = [g for g in pp[0]["regions"] if g["variant"] == "base"]
    m = OW.margin_for(H)
    assert m == 24 and base[0]["region"] == [200 - m, 900 - m, 600 + m, 1008 + m]
    OW.main(["shards", "--work", str(work), "--root", str(tmp_path / "bucket"), "--out", str(out), "--workers", "1",
             "--allow-missing-meiki"])
    samples = json.loads((out / "train" / f"{tid}_000.json").read_text(encoding="utf-8"))["samples"]
    assert [s["kind"] for s in samples] == ["pc", "pc"] and all((s["w"], s["h"]) == (W, H) for s in samples)
    assert samples[0]["vn"] == "v999997" and len(samples[0]["lines"]) == 2
    for ln, (_, (x, y, w, h)) in zip(samples[0]["lines"], spec[1]):
        assert ln[5] == "dialogue" and abs(ln[0] - x) <= 4 and abs(ln[1] - y) <= 4 and abs(ln[2] - (x + w)) <= 4


# ----------------------------------------------------------------------------- the committed manifests

def test_committed_training_manifests_are_consistent():
    hk = json.loads((MAN / "hooked_desktop_v1.json").read_text(encoding="utf-8"))
    assert hk["kind"] == "textspot6-pages" and hk["desktop"]["kind"] == "desktop"
    assert {s["split"] for s in hk["shards"]} == {"train"} and all(s["dir"] == "train" for s in hk["shards"])
    assert sum(s["samples"] for s in hk["shards"]) == hk["desktop"]["plan"]["pages"] == hk["totals"]["train"]["samples"]
    assert hk["desktop"]["capture_mix"] == json.loads(json.dumps(C.MIX))
    rp = json.loads((MAN / "realplus_native_v1.json").read_text(encoding="utf-8"))
    assert rp["kind"] == "desktop-frames" and {s["dir"] for s in rp["shards"]} == {RP.DIR}
    assert sum(s["samples"] for s in rp["shards"]) == rp["plan"]["frames"] == rp["totals"]["all"]["samples"]
    assert set(rp["totals"]["all"]["by_class"]) <= {"text", "name"}
    sy = json.loads((MAN / "synth_desktop_v1.json").read_text(encoding="utf-8"))
    assert sy["kind"] == "desktop-frames" and sum(s["samples"] for s in sy["shards"]) == sy["totals"]["all"]["samples"]
    assert set(sy["totals"]["all"]["by_class"]) <= {"dialogue", "name", "choice", "ui"}
    st = json.loads((MAN / "stats_v1.json").read_text(encoding="utf-8"))
    for name in ("1a hooked desktop", "1b realplus native", "1c synthetic desktop", "hooked Vita (native)",
                 "hooked PSP (native)"):
        assert st["sets"][name]["frames"] > 0
    assert st["sets"]["1a hooked desktop"]["frames"] == hk["desktop"]["plan"]["pages"]
    names = [s["name"] for s in hk["shards"] + rp["shards"] + sy["shards"]]
    assert len(names) == len(set(names))


def test_pc_test_set_is_frozen_reserved_and_in_no_pool():
    import hashlib
    import re

    import shots as RS

    import common
    man = ROOT / "eval" / "manifests" / "desktop-pc-test-v1.jsonl"
    rows = [json.loads(x) for x in man.read_text(encoding="utf-8").splitlines() if x.strip()]
    summ = json.loads((ROOT / "eval" / "desktop_pc_test_v1.json").read_text(encoding="utf-8"))
    assert summ["manifest_sha256"] == hashlib.sha256(man.read_bytes()).hexdigest()
    dlg = [r for r in rows if not r["name"]]
    assert len(rows) == summ["lines"] and len(dlg) == summ["dialogue_lines"]
    assert len({r["vn"] for r in dlg}) == summ["vns"] >= 60
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert r["reviewer_kind"] == "assistant" and r["text"] and r["how"] in ("consensus", "single", "changed")
        assert re.fullmatch(r"https://t\.vndb\.org/sf/\d\d/\d+\.jpg", r["url"]) and len(r["sha256"]) == 64
        x, y, w, h = r["box"]
        assert w > 0 and h > 0 and x + w <= r["dims"][0] and y + h <= r["dims"][1]
        assert r["group"] in ("pc", "console-hd")
        # a kept label: both engine families read it, or the independent pass 2 agreed, or (the one corrected label)
        # the hand check's third look set it
        assert r["how"] == "consensus" or (r["pass2"] or {}).get("agree") or (r["hand_checked"] or {}).get("agree")
    assert sum(1 for r in rows if r["hand_checked"]) == summ["hand_checked"] >= 200
    res = json.loads((ROOT / "eval" / "manifests" / "reserved.json").read_text(encoding="utf-8"))
    vns = {r["vn"] for r in rows}
    assert set(res["desktop_pc_test"]) == vns and set(res["desktop_pc_test"].values()) == {"desktop-pc-test"}
    assert res["desktop_pc_test_record"]["manifest_sha256"] == summ["manifest_sha256"]
    assert not vns & (set(res["test"]) | set(res["dev_sample"]))
    assert not vns & set(RS.training_pool_vns())                 # teacher, realdata, realplus, native, owner
    hooked = json.loads((ROOT / "data" / "manifests" / "vocr6_hooked_v2" / "vndb_map.json").read_text())
    assert not vns & set(re.findall(r'"(v\d+)"', json.dumps(hooked)))
    v = sorted(vns)[0]
    assert common.is_reserved_vn(v) and common.is_reserved_vn(v, dev=False)
    d = {"olang": {}, "title": {}, "relations": {}, "any_relations": {}, "rel_vns": {}}
    assert RS.reserved_vns(d, sets=set(), dev=set())[v] == "desktop-pc-test"


def test_docs_tables_match_the_stats():
    doc = ROOT / "docs" / "desktop-data.md"
    st = json.loads((MAN / "stats_v1.json").read_text(encoding="utf-8"))
    text = doc.read_text(encoding="utf-8")
    assert ST.render_doc(text, st) == text
