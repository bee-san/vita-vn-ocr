"""c-speed's tools and record (#165, docs/c-speed.md): the spotter's parity records (vita/ts_speed_parity.py), the
Vita3K parity driver's jobs and region records (vita/vita3k_parity.py), vita/a9_layers.py's --vocr-cflags, vocr_cli's
reset for the parity app, the runtime re-pins (eval/runtime_repins.json) and eval/c_speed_20261101.json against the
files and docs/c-speed.md. None of it needs VitaSDK, qemu, Vita3K or screenshots."""
import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

import repins

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


# ----------------------------------------------------------------------------- the re-pins and the record

def test_a_split_parity_run_is_one_run(tmp_path):
    """eval/c_speed_record.py reads a speed_parity run split per build and mode as the one run would have written it:
    the rows in its order, every build's new hashes against the first build's base hashes per mode."""
    sys.path[:0] = [str(ROOT / "eval")]
    import c_speed_record as R

    def row(build, mode, base, new):
        h = {"logits_sha256": "", "texts_sha256": "", "logit_bytes": 0, "cli": "", "topk_sha256": ""}
        return {"blob": "b.vocr", "mode": mode, "build": build, "base": h | {"texts_sha256": base},
                "new": h | {"texts_sha256": new}, "new_equals_base": base == new, "equals_other_builds": True}

    def part(name, *rows):
        (tmp_path / name).write_text(json.dumps({"ok": True, "pack_sha256": "p", "cli_opts": "--topk 8",
                                                 "logits": False, "results": list(rows)}))

    part("sp_L_rubi.vita-neon.w8i.json", row("vita-neon", "w8i", "a", "a"))
    part("sp_L_rubi.host-gcc.w8i.json", row("host-gcc", "w8i", "a", "a"))
    part("sp_L_rubi.host-gcc.w8f.json", row("host-gcc", "w8f", "f", "f"))
    d = R.sp_run(tmp_path, "sp_L_rubi")
    assert d["ok"] and d["split_runs"] == 3 and d["cli_opts"] == "--topk 8" and d["logits"] is False
    assert [(r["mode"], r["build"]) for r in d["results"]] == [("w8i", "host-gcc"), ("w8i", "vita-neon"),
                                                               ("w8f", "host-gcc")]
    part("sp_L_rubi.vita-neon.w8i.json", row("vita-neon", "w8i", "b", "b"))   # equal to its base, not to host-gcc's
    d = R.sp_run(tmp_path, "sp_L_rubi")
    assert not d["ok"] and [r["equals_other_builds"] for r in d["results"]] == [True, False, True]
    part("sp_L_rubi.arm-neon.w8i.json", row("arm-neon", "w8i", "a", "a"))
    (tmp_path / "sp_L_rubi.arm-neon.w8i.json").write_text(
        (tmp_path / "sp_L_rubi.arm-neon.w8i.json").read_text().replace("--topk 8", "--topk 3"))
    try:
        R.sp_run(tmp_path, "sp_L_rubi")
        raise AssertionError("split runs with different options were merged")
    except SystemExit as e:
        assert "cli_opts" in str(e)


RECORD = ROOT / "eval" / "c_speed_20261101.json"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_repins_reach_the_files_and_name_their_records():
    reg = json.loads(repins.REGISTRY.read_text(encoding="utf-8"))["repins"]
    for r in reg:
        assert set(r) >= {"file", "from", "to", "issue", "record", "what"} and r["from"] != r["to"]
        assert (ROOT / r["record"]).exists() and (ROOT / r["file"]).exists()
    for f in {r["file"] for r in reg}:
        firsts = [r["from"] for r in reg if r["file"] == f and not any(q["to"] == r["from"] for q in reg
                                                                        if q["file"] == f)]
        assert len(firsts) == 1
        assert sha256(ROOT / f) in repins.reachable(f, firsts[0])
    assert repins.reachable("runtime/none.c", "x") == {"x"}


def test_the_record_holds_and_the_doc_quotes_it():
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    after, before = rec["sources"]["after"], rec["sources"]["before"]
    for f in ("vocr.c", "vocr_textspot6.c"):
        assert sha256(ROOT / "runtime" / f) == after[f]
        assert after[f] in repins.reachable(f"runtime/{f}", before[f])
    doc = (ROOT / "docs" / "c-speed.md").read_text(encoding="utf-8")
    for t, row in rec["recognizer"].items():
        b, a = row["before"], row["after"]
        assert b["texts_sha256"] == a["texts_sha256"] and b["frames"] == a["frames"] == 1609
        assert a["vocr_c_sha256"] == after["vocr.c"] and b["vocr_c_sha256"] == before["vocr.c"]
        for pld in ("nopld", "pld"):
            for mhz in ("333", "444"):
                x, y = b[pld][mhz], a[pld][mhz]
                for z in (x, y):   # the record keeps ms per frame to 9 places and the box to 0.1 ms
                    box = rec["box"]["frames"] * z["ms_per_frame"] + rec["box"]["finder_ms"]
                    assert abs(z["box_ms"] - box) <= 0.05 + 1e-6
                assert y["ms_per_frame"] < x["ms_per_frame"]
        n0, n1 = b["nopld"]["333"]["ms_per_frame"], a["nopld"]["333"]["ms_per_frame"]
        assert f"{n0:.3f} → **{n1:.3f}**" in doc, t
        na = row["after_no_attn"]   # Rubi's build: the same arithmetic without the attention op, the same texts
        assert na["texts_sha256"] == b["texts_sha256"] and na["vocr_cflags"] == "-O2 -DVOCR_NO_ATTN"
        assert na["vocr_c_sha256"] == after["vocr.c"] and na["nopld"]["333"]["ms_per_frame"] < n0
        b_s, a_s = (round(x["nopld"]["333"]["box_ms"]) / 1000 for x in (b, a))   # whole ms, then seconds
        assert f"{b_s:.3f} → **{a_s:.3f} s**" in doc, t
    # the other pages that quote the recognizer's estimates (runtime/README.md, CHANGELOG): the record's figures
    rt_readme = (ROOT / "runtime" / "README.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    ms = {t: (row["before"]["nopld"]["333"]["ms_per_frame"], row["after"]["nopld"]["333"]["ms_per_frame"])
          for t, row in rec["recognizer"].items()}
    assert ("VOCR6.2 S / N / L as Rubi runs them: est. " + ", ".join(
        f"{b:.3f} → {a:.3f}" for b, a in (ms["S"], ms["N"])) + f" and {ms['L'][0]:.3f} → {ms['L'][1]:.3f}"
        ) in " ".join(rt_readme.split()), "runtime/README.md quotes other estimates than the record"
    box = {t: tuple(round(row[v]["nopld"]["333"]["box_ms"]) / 1000 for v in ("before", "after"))
           for t, row in rec["recognizer"].items()}
    assert (f"VOCR6.2 S / N / L {box['S'][0]:.3f} → {box['S'][1]:.3f}, {box['N'][0]:.3f} → {box['N'][1]:.3f}, "
            f"{box['L'][0]:.3f} → {box['L'][1]:.3f} s") in " ".join(changelog.split()), "CHANGELOG's boxes"
    gains = [100 * (1 - v[m]["after"]["ms_333"] / v[m]["before"]["ms_333"])
             for size, v in rec["spotter"].items() if size != "sources" for m in ("dark", "no_dark")]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"and {int(min(gains))}-{int(max(gains) + 0.5)}% faster" in readme, (min(gains), max(gains))
    for size in ("900x150", "960x544"):
        d = rec["spotter"][size]["dark"]
        assert f"{d['before']['ms_333']:.1f} → **{d['after']['ms_333']:.1f}**" in doc, size
    # the host table: main's figures, and how far this change's builds (the same x86 code) differ from them
    hr, hs = rec["host_ms"]["recognizer_ms_per_line"], rec["host_ms"]["spotter_ms_per_region"]
    row = " | ".join(" / ".join(f"{hr[f'{t}_base_{fl}']:.1f}" for t in "SNL") for fl in ("O2", "O3n"))
    assert f"S / N / L | {row} |" in doc, row
    assert f"| {hs['base_O2']:.2f} | {hs['base_O3n']:.2f} |" in doc
    noise = max([abs(hr[k.replace("_base_", "_new_")] / hr[k] - 1) for k in hr if "_base_" in k] +
                [abs(hs["new" + k[4:]] / hs[k] - 1) for k in hs if k.startswith("base")])
    assert f"agree with `main`'s to {math.ceil(noise * 1000) / 10:.1f}%" in doc, noise
    for size, v in rec["spotter"].items():
        if size != "sources":
            for mode in ("dark", "no_dark"):
                assert v[mode]["after"]["ms_333"] < v[mode]["before"]["ms_333"]
    p = rec["parity"]
    for run in p["recognizer"] + p["recognizer_vs_rubi_pin"]:
        assert run["ok"] and all(r["new_equals_base"] for r in run["rows"])
    assert {(r["tier"], r["run"]) for r in p["recognizer"]} == {(t, k) for t in "SNL" for k in ("logits", "rubi")}
    for run in p["recognizer"] + p["recognizer_vs_rubi_pin"]:   # Rubi's options: also the Vita objects as Rubi builds
        if run.get("run", "rubi") == "rubi":
            assert "vita-neon-noattn" in {r["build"] for r in run["rows"]}, run["tier"]
    assert all(r["equals_other_builds"] for run in p["recognizer"] for r in run["rows"])
    assert p["spotter"]["ok"] and len(p["spotter"]["rows"]) == 10
    assert all(r["new_equals_base"] and r["equals_other_builds"] for r in p["spotter"]["rows"])
    v3 = p["vita3k"]
    assert v3["ok"] and v3["outputs"] and all(o["all_equal"] for o in v3["outputs"])
    assert v3["versions"]["new"]["vocr_c_sha256"] == after["vocr.c"]
    assert v3["versions"]["new"]["vocr_textspot6_c_sha256"] == after["vocr_textspot6.c"]
    assert v3["versions"]["base"]["vocr_c_sha256"] == before["vocr.c"]
    assert p["spotter_twin_neon"]["ok"] and p["check_arm"] and p["parity_arm"]
    for k, v in rec["memory"]["arena"].items():
        assert v["before"] == v["after"] > 0, k
    code = rec["memory"]["vita_code"]
    rubi = "rubi -O2 -ffunction-sections -fdata-sections: "
    r0 = code[rubi + "vocr.c"]["before"], code[rubi + "vocr_textspot6.c -DVOCR_TEXTSPOT6_NO_MS"]["before"]
    r1 = code[rubi + "vocr.c -DVOCR_NO_ATTN"]["after"], code[rubi + "vocr_textspot6.c -DVOCR_TEXTSPOT6_NO_MS"]["after"]
    d = sum(r1) - sum(r0)
    assert d < 0 and f"{r0[0]:,} + {r0[1]:,} →\n  {r1[0]:,} + {r1[1]:,} B (−{-d:,} B)" in doc
    v = code["vita -O2: vocr.c"]
    assert f"`vocr.o` {v['before']:,} → {v['after']:,} B" in doc
    st = rec["memory"]["vocr_stack"]["vita -O2: vocr.c"]
    assert f"2,536 → {st['after']['vocr_run']['bytes']:,} B" in doc and st["before"]["vocr_run"]["bytes"] == 2536
    assert rec["machine_code_without_neon"] and all(v["identical"] for v in rec["machine_code_without_neon"].values())
