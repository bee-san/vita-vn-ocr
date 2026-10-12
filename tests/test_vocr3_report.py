"""eval/vocr3_report.py on generated fixtures only: random-init blobs, drawn captures, synthetic counts.

No protected set, screenshot or trained model is read: every image here is drawn by
tests/vocr3_fixtures.py and every blob is a random-init export.
"""
import json
import struct
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "vita"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sets  # noqa: E402
import vocr3_report as R  # noqa: E402

import vocr3_fixtures as F  # noqa: E402

# ---------------------------------------------------------------- bootstraps

def boot_reference(a, b, n=10000, seed=0):
    """The train2 review's boot.py (paired bootstrap of CER differences), verbatim in substance."""
    ids = sorted(set(a) & set(b))
    ea = np.array([a[i][0] for i in ids], float)
    eb = np.array([b[i][0] for i in ids], float)
    nn = np.array([a[i][1] for i in ids], float)
    d = (ea.sum() - eb.sum()) / nn.sum()
    idx = np.random.default_rng(seed).integers(0, len(ids), (n, len(ids)))
    ds = (ea[idx].sum(1) - eb[idx].sum(1)) / nn[idx].sum(1)
    return 100 * d, 100 * np.percentile(ds, 2.5), 100 * np.percentile(ds, 97.5)


def synthetic_lines(n=120, seed=3):
    rng = np.random.default_rng(seed)
    chars = rng.integers(1, 30, n)
    a = {f"l{i:03d}": (int(rng.integers(0, 4)), int(chars[i])) for i in range(n)}
    b = {f"l{i:03d}": (int(rng.integers(0, 5)), int(chars[i])) for i in range(n)}
    return a, b


def test_line_bootstrap_equals_train2_boot():
    a, b = synthetic_lines()
    got = R.paired_bootstrap(a, b, reps=10000, seed=0)
    d, lo, hi = boot_reference(a, b)
    assert got["delta"] == pytest.approx(d, abs=1e-5)
    assert got["ci95"] == pytest.approx([lo, hi], abs=1e-5)
    assert (got["candidate_errors"], got["baseline_errors"]) == (sum(v[0] for v in a.values()),
                                                                 sum(v[0] for v in b.values()))
    assert got["units"] == 120 and got["dropped_resamples"] == 0


def test_bootstrap_rejects_unpaired_units_and_handles_empty_strata():
    a, b = synthetic_lines(10)
    with pytest.raises(ValueError, match="identical units"):
        R.paired_bootstrap(a, dict(list(b.items())[:9]))
    b2 = dict(b)
    k = next(iter(b2))
    b2[k] = (b2[k][0], b2[k][1] + 1)
    with pytest.raises(ValueError, match="denominators"):
        R.paired_bootstrap(a, b2)
    empty = R.paired_bootstrap({"x": (0, 0)}, {"x": (0, 0)})
    assert empty["delta"] is None and empty["ci95"] is None


def test_cluster_bootstrap_resamples_whole_vns():
    a, b = synthetic_lines(60)
    few = {k: f"v{int(k[1:]) % 3}" for k in a}
    assert R.cluster_bootstrap(a, b, few, min_clusters=5)["ci95"] is None
    many = {k: f"v{int(k[1:]) % 6}" for k in a}
    got = R.cluster_bootstrap(a, b, many, reps=2000, min_clusters=5)
    sums = {}
    for k, vn in many.items():
        s = sums.setdefault(vn, [0, 0, 0])
        s[0], s[1], s[2] = s[0] + a[k][0], s[1] + b[k][0], s[2] + a[k][1]
    ref = R.paired_bootstrap({v: (s[0], s[2]) for v, s in sums.items()}, {v: (s[1], s[2]) for v, s in sums.items()},
                             reps=2000)
    assert got["clusters"] == 6 and got["ci95"] == ref["ci95"]
    same = R.cluster_bootstrap(a, a, many, reps=500)
    assert same["ci95"] == [0.0, 0.0]


# ---------------------------------------------------------------- gates

def tallies(errors, hira, chars=(2564, 2510), hira_chars=(1417, 1382)):
    return {s: {"lines": 1, "chars": c, "errors": e, "hira_chars": hc, "hira_errors": h}
            for s, e, h, c, hc in zip(("vita-test", "psp-test"), errors, hira, chars, hira_chars)}


def test_fixed_line_gate_on_the_published_vocr2_counts():
    """docs/vocr2-qualification.md: 61 -> 59 and 133 -> 119 full, 15 -> 13 and 30 -> 25 hiragana."""
    gate = R.fixed_line_gate(tallies((59, 119), (13, 25)), tallies((61, 133), (15, 30)))
    assert gate["passed"] and gate["pooled_hiragana"] == {"chars": 2799, "baseline_errors": 45,
                                                          "candidate_errors": 38}


@pytest.mark.parametrize("cand,reason", [
    (((62, 119), (13, 25)), "vita-test: full-text errors increase"),
    (((59, 119), (13, 31)), "psp-test: hiragana errors increase"),
    (((61, 133), (15, 30)), "pooled hiragana errors do not decrease"),
])
def test_fixed_line_gate_rejects_each_failure(cand, reason):
    gate = R.fixed_line_gate(tallies(*cand), tallies((61, 133), (15, 30)))
    assert not gate["passed"] and any(reason in r for r in gate["reasons"])


def test_fixed_line_gate_needs_identical_denominators():
    gate = R.fixed_line_gate(tallies((59, 119), (13, 25), chars=(2563, 2510)), tallies((61, 133), (15, 30)))
    assert not gate["passed"] and any("denominators" in r for r in gate["reasons"])


def group(platform, vn, errs, chars=100, ref="r"):
    full, hira = errs
    return {"platform": platform, "vn": vn, "reference_sha256": ref + vn,
            "all": {"full": {"chars": chars, "errors": full}, "hira": {"chars": chars // 2, "errors": hira}},
            "body": {"full": {"chars": chars - 10, "errors": full}, "hira": {"chars": chars // 2 - 5, "errors": hira}}}


def guard_case(cand_errs):
    titles = [("psp", "v1"), ("psp", "v2"), ("vita", "v3"), ("vita", "v4")]
    base = {v: {f"{p}:{n}": group(p, n, (10, 6)) for p, n in titles} for v in R.VARIANTS}
    cand = {v: {f"{p}:{n}": group(p, n, cand_errs.get((v, n), (9, 5))) for p, n in titles} for v in R.VARIANTS}
    return base, cand


def test_dialogue_guard_passes_and_counts_checks():
    base, cand = guard_case({})
    g = R.dialogue_guard(base, cand)
    assert g["passed"] and g["title_checks"] == 3 * 4 * 4 and not g["failed_cells"]
    assert all(len(t) == 4 for t in g["gain_titles"].values())


def test_dialogue_guard_records_every_regressing_cell():
    base, cand = guard_case({("expanded-8", "v3"): (11, 5)})
    g = R.dialogue_guard(base, cand)
    assert not g["passed"]
    assert [(c["variant"], c["group"], c["metric"]) for c in g["failed_cells"]] == [
        ("expanded-8", "vita:v3", "full"), ("expanded-8", "vita:v3", "full")]


def test_dialogue_guard_needs_two_gain_titles_and_a_balanced_gain():
    base, cand = guard_case({(v, n): (10, 6) for v in R.VARIANTS for n in ("v1", "v2", "v3")})
    g = R.dialogue_guard(base, cand)
    assert not g["passed"] and any("fewer than two titles" in r for r in g["reasons"])
    base, cand = guard_case({(v, n): (10, 6) for v in R.VARIANTS for n in ("v1", "v2", "v3", "v4")})
    g = R.dialogue_guard(base, cand)
    assert any("no strict console-balanced" in r for r in g["reasons"])


def test_dialogue_guard_rejects_reference_and_console_mismatch():
    base, cand = guard_case({})
    cand["base"]["psp:v1"]["reference_sha256"] = "other"
    assert any("references differ" in r for r in R.dialogue_guard(base, cand)["reasons"])
    base, cand = guard_case({})
    for v in R.VARIANTS:
        for d in (base, cand):
            d[v] = {k: x for k, x in d[v].items() if x["platform"] == "vita"}
    assert any("both consoles" in r for r in R.dialogue_guard(base, cand)["reasons"])


# ---------------------------------------------------------------- spec and arms

def test_parse_and_validate_spec(tmp_path):
    a = R.parse_arm("VOCR3=dir/x_w8.vocr@textbox#" + "a" * 64)
    assert a == {"name": "VOCR3", "file": "dir/x_w8.vocr", "finder": "textbox", "sha256": "a" * 64}
    spec = R.build_spec(arms=["A=a.vocr", "B=b.vocr@textbox"], pairs=["B:A"], gates=["B:A"])
    assert spec["primary_polarity"] == "console" and spec["bootstrap"] == R.DEFAULT_BOOTSTRAP
    bad = [dict(arms=["A=a.vocr@legacy"]), dict(arms=["A=a.vocr", "A=b.vocr"]), dict(arms=["A=a.vocr"], pairs=["B:A"]),
           dict(arms=["A=a.vocr", "B=b.vocr"], pairs=["A:A"]), dict(arms=["A=a.vocr#XYZ"]),
           dict(arms=["A=a.vocr", "B=b.vocr"], bootstrap={"reps": 10}),
           dict(arms=["A=a.vocr"], sets_=["clannad-test"]), dict(arms=[])]
    for kw in bad:
        with pytest.raises(ValueError):
            R.build_spec(**kw)
    stored = R.build_spec(arms=["A=a.vocr"], stored=["bar:native:vita-test=x.json"], pairs=["A:bar"])
    assert stored["stored"][0] == {"name": "bar", "polarity": "native", "set": "vita-test", "path": "x.json"}
    with pytest.raises(ValueError, match="blob arms only"):
        R.build_spec(arms=["A=a.vocr"], stored=["bar:native:vita-test=x.json"], gates=["A:bar"])
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"schema": 1, "kind": "vocr3-report-spec", "arms": [{"name": "A", "file": "a"}],
                                "unexpected": 1}))
    with pytest.raises(ValueError, match="unknown spec fields"):
        R.build_spec(path)


def test_released_spec_pins_the_three_published_blobs():
    spec = R.build_spec(ROOT / "eval" / "vocr3_harness_released.json")
    pins = {a["name"]: (a["file"], a["finder"], a["sha256"][:8]) for a in spec["arms"]}
    assert pins == {"VOCR1": ("FL10_w8.vocr", "grad", "500e0b84"), "VOCR_H15": ("H15_w8.vocr", "grad", "316ebc37"),
                    "VOCR2": ("VOCR2_FL10_w8.vocr", "textbox", "d3adc52e")}
    assert spec["gates"] == [{"candidate": "VOCR2", "baseline": "VOCR1"}]


def test_arm_checks_hash_crc_and_int8(tmp_path):
    path, sha = F.fixture_blob(tmp_path / "a.vocr")
    arm = R.Arm({"name": "A", "file": "a.vocr", "finder": "grad", "sha256": sha}, tmp_path)
    assert arm.identity()["bytes"] == path.stat().st_size and arm.blob["int8"]
    with pytest.raises(ValueError, match="pins"):
        R.Arm({"name": "A", "file": "a.vocr", "finder": "grad", "sha256": "0" * 64}, tmp_path)
    data = bytearray(path.read_bytes())
    data[200] ^= 1
    (tmp_path / "bad.vocr").write_bytes(bytes(data))
    with pytest.raises(ValueError, match="CRC"):
        R.Arm({"name": "B", "file": "bad.vocr", "finder": "grad"}, tmp_path)
    f32 = bytearray(path.read_bytes())
    struct.pack_into("<I", f32, 12, 0)                        # clear the int8 flag (CRC covers bytes 64+)
    (tmp_path / "f32.vocr").write_bytes(bytes(f32))
    with pytest.raises(ValueError, match="int8"):
        R.Arm({"name": "C", "file": "f32.vocr", "finder": "grad"}, tmp_path)


def test_public_outputs_never_hold_text():
    R.assert_public({"a": [{"errors": 1, "chars": 2}], "b": {"n": 1}})
    for bad in ({"lines": [{"text": "x"}]}, {"x": {"pred": "y"}}, [{"predictions": []}]):
        with pytest.raises(ValueError, match="text field"):
            R.assert_public(bad)


def test_compact_drops_record_lists_but_keeps_counts():
    rep = {"shared": {"fixed_line": {"console": {"A": {"lines": {"s": [{"errors": 1}]}, "summary": {"x": 1}}}},
                      "dialogue": {"units": {"A": [{"u": 1}]}, "totals": {"A": 1}}}}
    small = R.compact(rep)
    assert small == {"shared": {"fixed_line": {"console": {"A": {"summary": {"x": 1}}}},
                                "dialogue": {"totals": {"A": 1}}}}


# ---------------------------------------------------------------- overlay model table

OVERLAY_C = """static const VjoVocrModel models[] = {
    {"H15_w8.vocr", VJO_VOCR_H15_BYTES,
     {0x31,0x6e,0xbc,0x37,0x7e,0xc3,0x45,0x20,0x44,0x09,0x91,0xc5,0x08,0x51,0x99,0xd8,
      0xe7,0x40,0x31,0xf0,0x1a,0x86,0x79,0x38,0xbe,0x90,0xa6,0x0c,0x1e,0x3f,0x03,0xcd},
     182944, VOCR_LINES_GRAD},
#if VJO_VOCR2_ENABLED
    {VJO_VOCR2_ARTIFACT_NAME, VJO_VOCR2_BYTES,
     {0xd3,0xad,0xc5,0x2e,0x5c,0xf6,0x2b,0x54,0x8a,0xf0,0x4c,0x5a,0xb0,0x7a,0x2f,0x57,
      0xcf,0x04,0x19,0x38,0x6b,0xbd,0xf4,0x9b,0xa1,0x9c,0x97,0x03,0xbe,0x6e,0x17,0x66},
     124832, VOCR_LINES_TEXTBOX},
#endif
};
#define N_MODELS (sizeof(models) / sizeof(models[0]))
"""
SIZES_H = '#define VJO_VOCR2_ARTIFACT_NAME "VOCR2_FL10_w8.vocr"\n'
H15 = "316ebc377ec34520440991c5085199d8e74031f01a867938be90a60c1e3f03cd"
VOCR2 = "d3adc52e5cf62b548af04c5ab07a2f57cf0419386bbdf49ba19c9703be6e1766"


def test_overlay_table_parsing_and_scratch_patch():
    table = R.overlay_table(OVERLAY_C, SIZES_H)
    assert table == {("H15_w8.vocr", H15): (182944, "VOCR_LINES_GRAD"),
                     ("VOCR2_FL10_w8.vocr", VOCR2): (124832, "VOCR_LINES_TEXTBOX")}
    assert R.overlay_table_accepts(OVERLAY_C, "H15_w8.vocr", H15, SIZES_H)
    assert not R.overlay_table_accepts(OVERLAY_C, "H15_w8.vocr", VOCR2, SIZES_H)      # name/hash must pair
    entry = dict(name="VOCR3_FL10_w8.vocr", sha256="ab" * 32, bytes=657664, finder="grad", arena_w8i=124832)
    patched = R.overlay_table_patch(OVERLAY_C, [entry])
    assert R.overlay_table(patched, SIZES_H)[("VOCR3_FL10_w8.vocr", "ab" * 32)] == (124832, "VOCR_LINES_GRAD")
    assert patched.index("VOCR3_FL10_w8.vocr") < patched.index("#define N_MODELS")
    with pytest.raises(ValueError, match="no textbox_recovery"):
        R.overlay_table_patch(OVERLAY_C, [dict(entry, finder="textbox_recovery")])
    with pytest.raises(ValueError, match="layout"):
        R.overlay_table_patch("static const VjoVocrModel models[] = {};", [entry])


# ---------------------------------------------------------------- end to end on a generated private-style set

@pytest.fixture
def local_registry(monkeypatch):
    monkeypatch.setattr(sets, "SETS", dict(sets.SETS))
    monkeypatch.setattr(sets, "LOCAL_MANIFESTS", dict(sets.LOCAL_MANIFESTS))


@pytest.fixture
def fixture_run(tmp_path, local_registry):
    _, sha_a = F.fixture_blob(tmp_path / "A.vocr", seed=0)
    _, sha_b = F.fixture_blob(tmp_path / "B.vocr", seed=1)
    manifest, rows = F.local_set(tmp_path / "set", n_regions=6)
    spec = R.build_spec(arms=[f"A=A.vocr#{sha_a}", f"B=B.vocr@textbox#{sha_b}"], pairs=["B:A"], gates=["B:A"],
                        local_sets=[f"fixture-test={manifest}"], sets_=[],
                        bootstrap={"reps": 500, "min_clusters": 2})
    rep = R.shared(spec, tmp_path, tmp_path / "unused-data", tmp_path / "out")
    return types.SimpleNamespace(tmp=tmp_path, spec=spec, rep=rep, rows=rows, manifest=manifest)


def test_shared_report_counts_add_up_and_stay_public(fixture_run):
    rep, rows = fixture_run.rep, fixture_run.rows
    chars = sum(len(R.metrics.norm(r["text"])) for r in rows)
    for pol in ("console", "native"):
        for arm in ("A", "B"):
            entry = rep["fixed_line"][pol][arm]
            lines = entry["lines"]["fixture-test"]
            t = entry["summary"]["fixture-test"]["all"]
            assert t["lines"] == len(rows) == len(lines) and t["chars"] == chars
            assert t["errors"] == sum(x["errors"] for x in lines)
            assert sum(entry["summary"]["fixture-test"]["styles"]["name"]["lines"] for _ in [0]) == 6
            assert all(isinstance(x["errors"], int) and isinstance(x["hira_errors"], int) for x in lines)
    inverted = sum(x["inverted"] for x in rep["fixed_line"]["console"]["A"]["lines"]["fixture-test"])
    assert inverted == 9 and not any(x["inverted"] for x in rep["fixed_line"]["native"]["A"]["lines"]["fixture-test"])
    acc = rep["dialogue"]["accounting"]
    assert acc["sets"]["fixture-test"]["regions"] == 6 and acc["sets"]["fixture-test"]["cases"] == 18
    for arm in ("A", "B"):
        tot = rep["dialogue"]["totals"][arm]["base"]["vita"]
        assert tot["all"]["full"]["chars"] == chars and tot["regions"] == 6
        assert rep["forwards"][arm] == 2 * len(rows) + acc["detections"][arm]
    assert rep["resources"]["A"]["arena_w8i"] > rep["resources"]["A"]["arena_w8f"] > 0
    assert set(rep["gates"]["B:A"]) >= {"fixed_line", "dialogue", "passed"}
    assert rep["pairs"]["B:A"]["console"]["fixture-test"]["all"]["full"]["lines"]["units"] == len(rows)
    assert rep["spec"]["arms"][0]["file"] == "A.vocr" and str(fixture_run.tmp) not in json.dumps(rep["spec"])
    R.assert_public(rep)
    out = fixture_run.tmp / "out"
    assert (out / "shared.md").read_text().startswith("# VOCR3 locked report")
    private = json.loads((out / "private" / "fixed_A_console_fixture-test.json").read_text())
    assert len(private["lines"]) == len(rows) and "pred" in private["lines"][0]


def test_shared_report_matches_eval_run_recognizer_mode(fixture_run, tmp_path):
    """The fixed-line numbers are eval/run.py's own: its vocr: backend gives the same totals."""
    import run as eval_run
    blob = fixture_run.tmp / "A.vocr"
    res = eval_run.main(["--model", f"vocr:{blob}", "--set", "fixture-test", "--polarity", "console",
                         "--data", str(tmp_path)])
    got = fixture_run.rep["fixed_line"]["console"]["A"]["summary"]["fixture-test"]["all"]
    want = res["sets"]["fixture-test"]
    assert (got["errors"], got["chars"], got["exact"]) == (want["all"]["errors"], want["all"]["chars"],
                                                           round(want["all"]["exact"] * want["all"]["lines"]))
    assert (got["hira_errors"], got["hira_chars"]) == (want["hiragana"]["errors"], want["hiragana"]["chars"])


def test_reports_are_never_overwritten_and_render_combines_parts(fixture_run):
    with pytest.raises(FileExistsError):
        R.shared(fixture_run.spec, fixture_run.tmp, fixture_run.tmp, fixture_run.tmp / "out")
    combined = R.render(fixture_run.tmp / "out")
    assert combined["parts"] == ["shared"]
    summary = json.loads((fixture_run.tmp / "out" / "report_summary.json").read_text())
    assert "lines" not in summary["shared"]["fixed_line"]["console"]["A"]
    assert summary["shared"]["fixed_line"]["console"]["A"]["summary"] == \
        combined["shared"]["fixed_line"]["console"]["A"]["summary"]
    with pytest.raises(FileExistsError):
        R.render(fixture_run.tmp / "out")


def test_stored_results_are_rescored_on_the_manifest(tmp_path, local_registry):
    manifest, rows = F.local_set(tmp_path / "set", n_regions=3)
    sets.register_local("fixture-test", manifest)
    stored = {"polarity": "native", "sets": {"fixture-test": {"lines": [{"id": r["id"], "pred": r["text"]}
                                                                        for r in rows]}}}
    path = tmp_path / "stored.json"
    path.write_text(json.dumps(stored, ensure_ascii=False))
    recs, pol = R.score_stored(path, "fixture-test")
    assert pol == "native" and len(recs) == len(rows) and sum(x["errors"] for x in recs) == 0
    stored["sets"]["fixture-test"]["lines"].pop()
    path.write_text(json.dumps(stored, ensure_ascii=False))
    with pytest.raises(ValueError, match="no stored prediction"):
        R.score_stored(path, "fixture-test")


def test_dialogue_regions_drop_regions_with_an_uncertain_line(tmp_path, local_registry):
    manifest, rows = F.local_set(tmp_path / "set", n_regions=3)
    rows[0]["uncertain"] = True
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    sets.register_local("fixture-test", manifest)
    regions, excluded = R.dialogue_regions("fixture-test", tmp_path)
    assert len(regions) == 2 and excluded["regions"] == 1 and excluded["certain_lines"] == 2
    assert excluded["chars"] == sum(len(R.metrics.norm(r["text"])) for r in rows[1:3])
    assert R.expand([5, 5, 10, 10], 8, [20, 18]) == [0, 0, 20, 18]


def test_finder_driver_batches_match_single_calls(tmp_path):
    native = R.build_native(tmp_path / "native")
    imgs = [np.concatenate([F.screen((300, 120), [0, 0, 300, 120], [(10, 10 + 30 * i, 200, 20) for i in range(k)],
                                     dark=bool(k % 2), seed=k), np.full((120, 300, 1), 255, np.uint8)], axis=2)
            for k in (1, 2, 3)]
    batch = R.run_finder(native["finder"]["path"], imgs, 0)
    single = [R.run_finder(native["finder"]["path"], [im], 0)[0] for im in imgs]
    assert batch == single and [len(b[1]) for b in batch] == [1, 2, 3]
    assert [set(b[2]) for b in batch] == [{1}, {0}, {1}]            # k odd: dark ink on a light box


# ---------------------------------------------------------------- A9 wiring (no VitaSDK needed)

def test_a9_wiring_passes_the_blob_and_checks_its_identity(tmp_path, monkeypatch):
    _, sha = F.fixture_blob(tmp_path / "A.vocr")
    spec = R.build_spec(arms=[f"A=A.vocr#{sha}"])
    calls = []

    def runner(cmd, **kw):
        calls.append(cmd)
        if cmd[2] == "measure":
            d = Path(cmd[cmd.index("--out") + 1])
            d.mkdir(parents=True, exist_ok=True)
            (d / "meta.json").write_text(json.dumps({"lines": 4, "blob_sha256": sha,
                                                     "modes": {"w8i": {"lines": 4, "frames": 80}}}))
            (d / "lines.bin").write_bytes(b"pack")
            (d / "vocr_cli_vitacc").write_bytes(b"cli")
        else:
            (Path(cmd[3]) / "report.json").write_text(json.dumps({"lines": 4, "modes": {"w8i": {
                "cycles_per_frame": {"444": [1.0e5, 2.0e5], "333": [1.0e5, 2.0e5]},
                "ms_640px": {"444": [18.0, 36.0], "333": [24.0, 48.0]}, "insns_per_frame": 9.0e4}}}))
    synth = tmp_path / "synth_val.npz"
    synth.write_bytes(b"generated")
    monkeypatch.setenv("VOCR_SYNTH_VAL", str(synth))
    res = R.a9(spec, tmp_path, tmp_path / "out", vitasdk=str(tmp_path), lines=4, modes="w8i", runner=runner)
    assert res["synth_val"] == {"file": "synth_val.npz", "sha256": R.sha256(synth)}
    assert calls[0][calls[0].index("--blob") + 1] == str(tmp_path / "A.vocr")
    assert res["estimate"] is True and res["arms"]["A"]["modes"]["w8i"]["ms_640px_line"]["444"] == [18.0, 36.0]

    def other(cmd, **kw):
        runner(cmd, **kw)
        if cmd[2] == "measure":
            p = Path(cmd[cmd.index("--out") + 1]) / "meta.json"
            p.write_text(json.dumps(dict(json.loads(p.read_text()), blob_sha256="0" * 64)))
    with pytest.raises(ValueError, match="another blob"):
        R.a9(spec, tmp_path, tmp_path / "out2", vitasdk=str(tmp_path), lines=4, modes="w8i", runner=other)


def test_a9_cycles_given_blob_is_int8_only(tmp_path):
    import a9_cycles
    path, sha = F.fixture_blob(tmp_path / "A.vocr")
    blobs, got = a9_cycles.given_blob(tmp_path, path, ["w8i", "w8f"])
    assert got == sha and blobs["w8"].read_bytes() == path.read_bytes()
    with pytest.raises(SystemExit, match="f32"):
        a9_cycles.given_blob(tmp_path, path, ["w8i", "f32"])
    assert a9_cycles.QEMU_GUEST_ENV == {}
