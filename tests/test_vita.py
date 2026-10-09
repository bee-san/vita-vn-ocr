"""The Vita harness (vita/): the bundle format round trip through the app's own C parser,
the Cortex-A9 cost model, the generated LiveArea images, the screen font and compare.py.
None of it needs VitaSDK, Vita3K or qemu."""
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
VITA = ROOT / "vita"
sys.path[:0] = [str(VITA)]
import a9_cycles  # noqa: E402
import bundle  # noqa: E402
import compare  # noqa: E402

CC = shutil.which("cc") or shutil.which("gcc")


def fake_cases():
    rng = np.random.default_rng(0)
    grey = rng.integers(0, 256, (31, 362), dtype=np.uint8)
    line = rng.integers(0, 256, (40, 300, 4), dtype=np.uint8)
    region = rng.integers(0, 256, (120, 500, 4), dtype=np.uint8)
    cases = [dict(id="synth-00000", kind=bundle.KIND_LINE, flags=0, px=grey, gt="「おいおい」"),
             dict(id="vita-0001", kind=bundle.KIND_LINE, flags=bundle.FLAG_AUTO | bundle.FLAG_DARK, px=line,
                  gt="暗い文字"),
             dict(id="vita-0002@region", kind=bundle.KIND_REGION, flags=0, px=region, gt="一行目\n二行目")]
    exps = [{"w8i": dict(text="「おいおい」", crc=1, frames=46, boxes=[], max_line_frames=46),
             "f32": dict(text="「おいおい", crc=2, frames=46, boxes=[], max_line_frames=46)},
            {"w8i": dict(text="暗い文字", crc=0xFFFFFFFF, frames=30, boxes=[], max_line_frames=30),
             "f32": dict(text="", crc=0, frames=30, boxes=[], max_line_frames=30)},
            {"w8i": dict(text="一行目\n二行目\\x", crc=77, frames=90, boxes=[[4, 5, 300, 30, 0], [4, 50, 280, 31, 1]],
                         max_line_frames=50),
             "f32": dict(text="一行目", crc=78, frames=45, boxes=[[4, 5, 300, 30, 0]], max_line_frames=45)}]
    models = [dict(name="w8i", file="v1_w8.vocr", mode=3, bytes=859904, crc=711465887, arena=124832),
              dict(name="f32", file="v1_f32.vocr", mode=1, bytes=3067840, crc=1107696978, arena=116384)]
    return models, cases, exps


@pytest.fixture(scope="module")
def cases_dump(tmp_path_factory):
    if not CC:
        pytest.skip("no C compiler")
    exe = tmp_path_factory.mktemp("vita") / "cases_dump"
    subprocess.run([CC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-Wpedantic", "-o", str(exe),
                    str(VITA / "tests" / "cases_dump.c"), str(VITA / "app" / "cases.c")], check=True)
    return exe


def test_cases_bin_round_trip_through_the_apps_parser(tmp_path, cases_dump):
    models, cases, exps = fake_cases()
    sizes = bundle.write_cases_bin(tmp_path / "cases.bin", models, cases, exps, 5355)
    assert sizes == dict(max_frames=50, max_lines=2, max_region=(500, 120))
    out = subprocess.run([str(cases_dump), str(tmp_path / "cases.bin")], check=True, capture_output=True,
                         text=True).stdout.splitlines()
    recs = [x.split("\t") for x in out]
    header = recs[0]
    assert header[0] == "header"
    assert [int(v) for v in header[1:]] == [2, 3, largest_record(tmp_path / "cases.bin", 3), 2, 50, 5355, 500, 120]
    assert recs[1][1:] == ["w8i", "v1_w8.vocr", "3", "859904", "711465887", "124832"]
    assert recs[2][1:] == ["f32", "v1_f32.vocr", "1", "3067840", "1107696978", "116384"]
    rows = recs[3:]
    for i, (case, e) in enumerate(zip(cases, exps)):
        c = rows[3 * i]
        h, w = case["px"].shape[:2]
        ch = 1 if case["px"].ndim == 2 else 4
        assert c[0] == "case" and c[1] == case["id"]
        assert [int(v) for v in c[2:7]] == [case["kind"], case["flags"], w, h, ch]
        assert compare.unescape(c[7]) == case["gt"]
        assert int(c[8]) == int(case["px"].astype(np.uint64).sum()) & 0xFFFFFFFF
        for m, name in enumerate(("w8i", "f32")):
            x = rows[3 * i + 1 + m]
            want = e[name]
            assert x[0] == "expect" and int(x[1]) == m
            assert compare.unescape(x[2]) == want["text"]
            assert [int(v) for v in x[3:6]] == [want["crc"], want["frames"], len(want["boxes"])]
            assert [int(v) for v in x[6:]] == [v for b in want["boxes"] for v in b]


def largest_record(path, n_cases, n_models=2):
    """The largest case record of a cases.bin, walking the records by their lengths."""
    data = Path(path).read_bytes()
    off, biggest = 64 + 96 * n_models, 0
    for _ in range(n_cases):
        n = struct.unpack_from("<I", data, off)[0]
        biggest = max(biggest, n)
        off += n
    assert off == len(data)
    return biggest


def test_the_apps_parser_rejects_a_truncated_bundle(tmp_path, cases_dump):
    models, cases, exps = fake_cases()
    bundle.write_cases_bin(tmp_path / "cases.bin", models, cases, exps, 5355)
    data = (tmp_path / "cases.bin").read_bytes()
    (tmp_path / "cut.bin").write_bytes(data[:len(data) - 1000])
    p = subprocess.run([str(cases_dump), str(tmp_path / "cut.bin")], capture_output=True, text=True)
    assert p.returncode != 0 and "bad record 2" in p.stderr
    bad = bytearray(data)
    bad[8] = 2                                                     # version 2
    (tmp_path / "v2.bin").write_bytes(bytes(bad))
    assert subprocess.run([str(cases_dump), str(tmp_path / "v2.bin")], capture_output=True).returncode != 0


@pytest.mark.parametrize("mn,ops,unit,issue,ls", [
    # Cortex-A9 NEON MPE TRM (ARM DDI 0409I) tables 3-2 to 3-9
    ("vmlal.s16", "q15, d18, d7[0]", "N", 1, 0),
    ("vmlal.s32", "q15, d18, d7[0]", "N", 2, 0),
    ("vmla.f32", "q8, q9, d0[1]", "N", 2, 0),
    ("vmla.f32", "d16, d17, d0[1]", "N", 1, 0),
    ("vmul.f32", "q8, q8, q9", "N", 2, 0),
    ("vcvt.f32.s32", "q9, q9", "N", 2, 0),
    ("vmovl.s8", "q9, d18", "N", 1, 0),
    ("vqmovn.s32", "d16, q8", "N", 1, 0),
    ("vadd.i32", "q8, q8, q9", "N", 1, 0),
    ("vld1.8", "{d18}, [r1]!", "N", 2, 1),
    ("vld1.32", "{d16-d17}, [r3]", "N", 2, 2),
    ("vld1.64", "{d18-d19}, [r2 :64]!", "N", 1, 2),
    ("vld1.16", "{d16[]}, [r1]", "N", 2, 1),
    ("vst1.32", "{d16-d17}, [r3]", "N", 2, 2),
    ("vstr", "d24, [sp, #320]", "L", 1, 1),
    ("vldmia", "r3, {d8-d11}", "L", 4, 4),
    ("vpush", "{d8-d15}", "L", 8, 8),
    ("vadd.f32", "s14, s14, s15", "V", 1, 0),
    ("vmul.f64", "d16, d17, d18", "V", 2, 0),
    ("vdiv.f32", "s0, s1, s2", "V", 10, 0),
    ("vcmpe.f32", "s15, s14", "V", 1, 0),
    ("vmrs", "APSR_nzcv, fpscr", "V", 1, 0),
    ("vmov", "r0, r1, d16", "V", 1, 0),
    ("add.w", "r3, r3, #8", "I", 1, 0),
    ("ldr", "r3, [sp, #4]", "I", 1, 1),
    ("push", "{r4, r5, r6, r7, lr}", "I", 3, 3),
    ("bne.n", "11eac <mm_i16+0x544>", "I", 1, 0),
    ("itt", "ne", "I", 0, 0),
])
def test_a9_instruction_costs(mn, ops, unit, issue, ls):
    got = a9_cycles.classify(mn, ops)
    assert got[:3] == (unit, issue, ls)


def test_a9_neon_to_core_transfer_is_flagged():
    assert "n2c" in a9_cycles.classify("vmov.32", "r0, d16[1]")[3]
    assert "n2c" not in a9_cycles.classify("vmov.32", "d16[0], r3")[3]
    assert a9_cycles.classify("bic", "r3, r3, #7")[3] == {"alu"}


def test_a9_bounds_on_a_synthetic_mix():
    rows = {0x10: ("mm_i16", "vmlal.s16", "q0, d1, d2[0]"), 0x14: ("mm_i16", "vstr", "d24, [sp, #8]"),
            0x18: ("mm_i16", "add.w", "r1, r1, #8"), 0x1c: ("main", "add.w", "r1, r1, #8")}
    counts = {0x10: 800, 0x14: 100, 0x18: 400, 0x1c: 10**6}         # main is the CLI: not counted
    extra = {"switches_l_free": 0, "switches_l_vfp": 200}
    cache = {"l1d_misses": 10, "l2_misses": 1}
    r = a9_cycles.model_mode(rows, counts, extra, {"mm_i16"}, frames=10, cache=cache)
    assert r["insns_per_frame"] == 130
    lo, hi = r["cycles_per_frame"][444]
    # low: NEON + VLDR/VSTR issue (800 + 100) beats dual issue (1300 / 2), LS (100) and refill (87)
    assert lo == pytest.approx(900 / 10)
    # high: max(900, 1300 instructions) + 200 switches * 10 + 10 L1 misses * 30 + 1 L2 miss * (30 + 110 ns)
    assert hi == pytest.approx((1300 + 2000 + 300 + 30 + 110 * 0.444) / 10)


def test_mkicon_writes_palette_pngs_of_the_livearea_sizes(tmp_path):
    from PIL import Image
    subprocess.run([sys.executable, str(VITA / "mkicon.py"), str(tmp_path)], check=True)
    for name, size in {"icon0.png": (128, 128), "bg.png": (840, 500), "startup.png": (280, 158)}.items():
        with Image.open(tmp_path / name) as im:
            assert im.mode == "P" and im.size == size
            im.verify()


def test_screen_font_matches_the_apps_reader(tmp_path, font):
    texts = ["「ホーホケキョ」と大きな声で", "abc 123!?"]
    n, _ = bundle.write_font(tmp_path, texts, vocab=["吾", "輩"])
    data = (tmp_path / "font.bin").read_bytes()
    assert data[:4] == b"VFNT"
    version, count, height = struct.unpack_from("<3I", data, 4)
    assert (version, count, height) == (1, n, 16)
    cps = []
    for i in range(count):                                       # the checks scr_load_font() makes
        cp, w, pad, off = struct.unpack_from("<IHHI", data, 16 + 12 * i)
        assert w in (8, 16) and pad == 0 and off + 16 * (w // 8) <= len(data)
        cps.append(cp)
    assert cps == sorted(set(cps))
    assert {ord(c) for c in "ホケキョ吾輩a1!"} <= set(cps)
    glyph = dict(zip(cps, range(count)))[ord("ホ")]
    cp, w, _, off = struct.unpack_from("<IHHI", data, 16 + 12 * glyph)
    assert w == 16 and any(data[off:off + 32])                   # full width, not blank
    assert (tmp_path / "font-OFL.txt").read_text().startswith("Copyright")


def write_results(path, rows):
    cols = ["id", "kind", "model", "status", "rc", "dark", "width", "height", "frames", "lines", "ms_min",
            "ms_median", "us_per_frame", "cycles", "insns", "neon", "dmiss", "dstall", "nodispatch", "crc_host",
            "crc_vita", "text_host", "text_vita"]
    lines = ["# vocr_test version=1 git=abc built=x", "# clock arm_mhz=444 bus_mhz=222",
             "# stack probe_used=4100 probe_stack=32768 kernel_free=1", "#cols " + "\t".join(cols)]
    lines += ["\t".join(str(r[c]) for c in cols) for r in rows]
    lines.append("# done cases=2 of 2 total_s=1.0")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_compare_counts_agreement_cer_and_time(tmp_path):
    cases = [dict(id="a", kind=0, gt="あいう", expected={"w8i": dict(text="あいう", crc=1, frames=10, boxes=[])}),
             dict(id="b", kind=0, gt="かきく", expected={"w8i": dict(text="かきけ", crc=2, frames=20, boxes=[])}),
             dict(id="c", kind=0, gt="x", expected={"w8i": dict(text="x", crc=3, frames=5, boxes=[])})]
    (tmp_path / "cases.json").write_text(json.dumps(dict(models=[dict(name="w8i")], cases=cases)))
    base = dict(kind="line", model="w8i", rc=0, dark=0, width=100, height=32, lines=1, ms_min=1, cycles=0,
                insns=0, neon=0, dmiss=0, dstall=0, nodispatch=0)
    write_results(tmp_path / "results.tsv", [
        dict(base, id="a", status="MATCH", frames=10, ms_median=2.0, us_per_frame=200.0, crc_host="00000001",
             crc_vita="00000001", text_host="あいう", text_vita="あいう"),
        dict(base, id="b", status="DIFF", frames=20, ms_median=8.0, us_per_frame=400.0, crc_host="00000002",
             crc_vita="0000000f", text_host="かきけ", text_vita="かきこ")])
    rep = compare.summarise(tmp_path, tmp_path / "results.tsv", emulator=True)
    m = rep["models"]["w8i"]
    assert (m["MATCH"], m["DIFF"], m["missing"]) == (1, 1, ["c"])
    assert m["accuracy"]["lines"]["cer"] == pytest.approx(1 / 6)       # かきこ against かきく
    assert m["us_per_frame"]["median"] == pytest.approx(300.0)
    assert m["ms_per_640px"] == pytest.approx(24.0)
    assert not m["bundle_mismatch"]


def test_scripts_parse():
    for sh in sorted((VITA / "scripts").glob("*.sh")):
        subprocess.run(["bash", "-n", str(sh)], check=True)
