"""c-speed's tools (#165, docs/c-speed.md): the spotter's parity records (vita/ts_speed_parity.py), the Vita3K parity
driver's jobs and region records (vita/vita3k_parity.py), vita/a9_layers.py's --vocr-cflags and the parity app's
argument splitting. None of it needs VitaSDK, qemu, Vita3K or screenshots."""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VITA = ROOT / "vita"
sys.path[:0] = [str(VITA)]
import ts_speed_parity as TS  # noqa: E402
import vita3k_parity as V3  # noqa: E402


def read_records(path):
    data, off, out = Path(path).read_bytes(), 0, []
    while off < len(data):
        w, h, bpp, flags = struct.unpack_from("<4I", data, off)
        off += 16
        out.append((w, h, bpp, flags, data[off:off + w * h * bpp]))
        off += w * h * bpp
    return out


def test_records_round_trip_in_the_drivers_format(tmp_path):
    recs = [(3, 2, 1, TS.ROWS | TS.NO_DARK, bytes(range(6))), (2, 1, 4, TS.SCALE2, bytes(range(8)))]
    TS.write_records(tmp_path / "r.bin", recs)
    assert read_records(tmp_path / "r.bin") == recs


def test_grey_is_the_spotters_luma():
    px = np.array([[[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [10, 20, 30, 0]]], np.uint8)
    assert TS.grey_of(px).tolist() == [[(77 * 255) >> 8, (150 * 255) >> 8, (29 * 255) >> 8,
                                        (77 * 10 + 150 * 20 + 29 * 30) >> 8]]


def write_regions(path, regions):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(regions)))
        for px in regions:
            f.write(struct.pack("<II", px.shape[1], px.shape[0]))
            f.write(px.tobytes())


def test_region_records_are_rubis_crop_decode_and_the_default_call(tmp_path):
    rng = np.random.default_rng(1)
    regions = [rng.integers(0, 256, (5, 7, 4), dtype=np.uint8), rng.integers(0, 256, (3, 2, 4), dtype=np.uint8)]
    write_regions(tmp_path / "reg.bin", regions)
    recs = V3.region_records(tmp_path / "reg.bin")
    assert [(w, h, b, f) for w, h, b, f, _ in recs] == [
        (7, 5, 4, TS.ROWS | TS.NO_DARK | TS.SCALE2), (7, 5, 4, 0), (2, 3, 4, TS.ROWS | TS.NO_DARK | TS.SCALE2),
        (2, 3, 4, 0)]
    assert recs[0][4] == regions[0].tobytes() and recs[3][4] == regions[1].tobytes()
    # ts_speed_parity's records: per region the crop decode, the dark pass and grey
    trs = TS.records(tmp_path / "reg.bin", 0)
    assert [(w, h, b, f) for w, h, b, f, _ in trs][:3] == [
        (7, 5, 4, TS.ROWS | TS.NO_DARK | TS.SCALE2), (7, 5, 4, TS.SCALE2), (7, 5, 1, 0)]
    assert trs[2][4] == TS.grey_of(regions[0]).tobytes()


def test_procedural_records_cover_both_scales_and_the_flags():
    recs = TS.records(None, len(TS.SIZES))
    flags = [f for _, _, _, f, _ in recs]
    assert any(f & TS.SCALE1 for f in flags) and any(f & TS.SCALE2 for f in flags)
    assert any(f & TS.PADDED for f in flags) and any(f & TS.ODD for f in flags) and any(f & TS.UPTO for f in flags)
    assert {(w, h) for w, h, _, _, _ in recs} == set(TS.SIZES)
    assert all(len(px) == w * h * b for w, h, b, _, px in recs)
    # a 1x1 or 2-row frame is never asked to halve
    assert not any((f & TS.SCALE2) and (w <= 2 or h <= 2) for w, h, _, f, _ in recs)


def test_vita3k_jobs_read_the_full_head_and_rubis_options():
    args = argparse.Namespace(blob=["S=/x/S.vocr", "N=/x/N.vocr:--head-gate on --two-level"], modes="w8i,w8f")
    jobs = V3.jobs_of(args, None)
    cmds = [f"{t} {' '.join(a)}" for t, a, _ in jobs]
    assert cmds[0] == "cli {D}S.vocr --mode w8i --pack {D}sub.bin --logits {D}logits_S_w8i.bin"
    assert cmds[1] == "cli {D}S.vocr --mode w8f --pack {D}sub.bin --logits {D}logits_S_w8f.bin"
    assert cmds[2] == "cli {D}S.vocr --mode w8i --pack {D}pack.bin --topk 8 --topk-out {D}topk_S.bin"
    # the logits jobs never carry the head options (they read every frame through the full head anyway); the top-K
    # job does, as Rubi runs the blob
    assert "--head-gate" not in cmds[3] and "--head-gate" not in cmds[4]
    assert cmds[5].endswith("--topk-out {D}topk_N.bin --head-gate on --two-level")
    assert cmds[-2:] == ["ts6 --batch {D}spot.vtsp {D}regions.bin", "ts6 --maps {D}spot.vtsp {D}regions.bin"]
    assert [f for _, _, f in jobs][:3] == [["logits_S_w8i.bin"], ["logits_S_w8f.bin"], ["topk_S.bin"]]
    # every job line fits the app's limits (vita/par/main.c: 8,192-byte lines, 64 arguments)
    for t, a, _ in jobs:
        line = f"{t} {' '.join(a).replace('{D}', V3.UX)}"
        assert len(line) < 8192 and len(line.split(" ")) <= 64 and "  " not in line


def test_a9_layers_measure_takes_the_vocr_cflags():
    src = (VITA / "a9_layers.py").read_text()
    assert '"--vocr-cflags", default="-O2"' in src
    assert "build(out, vitasdk, arm_cc, runtime, prof, shlex.split(args.vocr_cflags))" in src


def test_the_parity_app_resets_the_cli_state():
    """vita/par runs vocr_cli's main once per job in one process: every file-scope option must start from 0."""
    cli = (ROOT / "runtime/tools/vocr_cli.c").read_text()
    main = cli[cli.index("int main(int argc, char **argv)"):]
    for name in ("invert_flag", "topk_n", "conf_flag", "topk_out", "lm_dec", "post_flag", "redecode_on",
                 "gated_total"):
        assert f"\n    {name} = " in main[:main.index("for (i = 2; i < argc; i++)")], name
    assert "memset(&redecode_o, 0, sizeof redecode_o);" in main
