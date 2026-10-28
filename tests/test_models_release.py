"""Release v0.5.0's asset table (docs/models.py, docs/models.json, docs/models.md) and the other readers' public record
(eval/other_readers_20261028.json): the table is consistent with itself, with Rubi's pins and with the documents
that cite it; the record holds counts and hashes only and was scored under the registered protocol.

No network, no release download, no private data.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("models_release", ROOT / "docs" / "models.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

DOC = json.loads(M.MANIFEST.read_text(encoding="utf-8"))
ASSETS = DOC["assets"]
BY = {x["name"]: x for x in ASSETS}
READERS = ROOT / "eval" / "other_readers_20261028.json"


def test_the_manifest_is_the_table():
    """docs/models.json is docs/models.py's ASSETS with each file's bytes and sha256 added (`manifest`)."""
    assert [{k: v for k, v in x.items() if k not in ("bytes", "sha256")} | {"from": {
        k: v for k, v in x["from"].items() if k != "in_release_sha256sums"}} for x in ASSETS] == M.ASSETS
    assert DOC["rubi"] == M.RUBI_PINS and DOC["left_out"] == M.LEFT_OUT and DOC["release"] == "v0.5.0"
    assert DOC["files"] == len(ASSETS) + 2


def test_every_asset_has_one_name_one_byte_string_and_a_known_status():
    names = [x["name"] for x in ASSETS]
    assert len(set(names)) == len(names)
    assert M.LIST_NAME not in names and "SHA256SUMS" not in names
    hashes = [x["sha256"] for x in ASSETS]
    assert len(set(hashes)) == len(hashes), "the same bytes under two names"
    for x in ASSETS:
        assert re.fullmatch(r"[0-9a-f]{64}", x["sha256"]) and x["bytes"] > 0, x["name"]
        assert x["status"] in DOC["statuses"], x["name"]
        assert re.fullmatch(r"v\d+\.\d+\.\d+(-[0-9a-z.-]+)?", x["from"]["tag"]), x["name"]
        assert x["why"] and x["licence"], x["name"]
        if x["attribution"] is not None:
            assert x["attribution"] in BY and BY[x["attribution"]]["kind"] == "attribution", x["name"]


def test_the_rubi_statuses_are_rubis_pins():
    shipped, tools = DOC["rubi"]["shipped"], DOC["rubi"]["host_tools_and_tests"]
    for name, sha in {**shipped, **tools}.items():
        assert BY[name]["sha256"] == sha, name
    assert {x["name"] for x in ASSETS if x["status"] == M.RUBI} == set(shipped)
    assert {x["name"] for x in ASSETS if x["status"] == M.RUBI_TOOLS} == set(tools)
    assert re.fullmatch(r"[0-9a-f]{40}", DOC["rubi"]["commit"])


def test_rubis_three_sizes_and_spotter_are_vocr6_and_textspot8():
    assert {BY[n]["model"] for n in ("VOCR6.2_S_w8.vocr", "VOCR6.2_N_w8g2.vocr", "VOCR6.2_L_w8_2l.vocr")} == \
        {"VOCR6-S", "VOCR6-N", "VOCR6-L"}
    assert all(BY[n]["status"] == M.RUBI for n in ("VOCR6.2_S_w8.vocr", "VOCR6.2_N_w8g2.vocr",
                                                     "VOCR6.2_L_w8_2l.vocr", "textspot8_t64n8c16_w8.vtsp"))
    assert BY["textspot8_t64n8c16_w8.vtsp"]["sha256"] == \
        "e129d41adfa02c658a7d13d6f98064e6ab6ce9a2bc8c7a56da700a21357b5e1c"


def test_the_models_trained_on_game_data_are_non_commercial():
    for x in ASSETS:
        if x["from"]["tag"] == "v0.2.3-vocr2.1":      # before the personal synthetic profile; no attribution file
            assert x["licence"] == M.VOCR2_LIC and x["attribution"] is None, x["name"]
            continue
        if x["kind"] in ("recognizer", "line spotter", "recognizer (PSP candidate)", "recognizer (vertical)",
                         "text spotter (whole frame)", "text-presence gate", "text detector"):
            assert x["licence"].startswith("non-commercial use only"), x["name"]
        if x["from"]["tag"] in ("v0.4.1-vocr6.2", "v0.4.0-vocr6.1", "v0.3.10-textspot8.1", "v0.3.9-textspot7.1",
                                "v0.3.8-textspot6.1", "v0.3.7-modelb-test.1") and x["kind"] != "attribution":
            assert "game" in x["licence"], x["name"]
        if x["kind"] == "character LM":
            assert x["licence"].startswith("CC BY-SA 4.0"), x["name"]
            assert ("hooked" in x["licence"]) == ("lm6" in x["name"]), x["name"]


def test_every_attribution_file_of_the_source_releases_is_included():
    tags = {x["from"]["tag"] for x in ASSETS}
    have = {x["from"]["tag"] for x in ASSETS if x["kind"] == "attribution"}
    no_attribution = {"v0.2.5-vocr3.1", "v0.2.5-vocr4.1", "v0.2.3-vocr2.1"}       # those releases published none
    assert have == tags - no_attribution | {"v0.3.7-lm6.1"}
    assert {BY[n]["kind"] for n in ("VOCR6-ATTRIBUTION.md", "MODEL-ATTRIBUTION.md", "LM-ATTRIBUTION.md",
                                     "SPOTTER-ATTRIBUTION.md", "GATE-ATTRIBUTION.md", "VERTICAL-ATTRIBUTION.md")} == \
        {"attribution"}


def test_the_research_snapshots_before_vocr3_are_left_out_and_said_so():
    tags = {x["from"]["tag"] for x in ASSETS}
    for t in ("v0.2.2-meiki.1", "v0.2.2-fl10-training.1", "v0.2.2-fl10-bitmap.1", "v0.2.2-fl10-general.1",
              "v0.2.4-fl10-head.1"):
        assert t not in tags
        assert any(t in x["tag"] for x in DOC["left_out"]), t
    # VOCR2's small int8 blob and its notes are in; its checkpoint and result files are left out, and said so
    assert {x["name"] for x in ASSETS if x["from"]["tag"] == "v0.2.3-vocr2.1"} == {"VOCR2_FL10_w8.vocr",
                                                                                   "vocr2-qualification.md"}
    assert BY["VOCR2_FL10_w8.vocr"]["status"] == M.RESEARCH
    assert any(x["tag"] == "v0.2.3-vocr2.1" and "vocr2-quarter-checkpoint.pt" in x["asset"] for x in DOC["left_out"])


def test_the_md_tables_are_generated_from_the_manifest():
    text = (ROOT / "docs" / "models.md").read_text(encoding="utf-8")
    assert M.render_md(text, DOC) == text, "run `python docs/models.py md`"
    for x in ASSETS:
        assert f"| `{x['name']}` |" in text and x["sha256"] in text, x["name"]


def flat(text):
    """Text with its line breaks and indents folded to single spaces (wrapped markdown)."""
    return re.sub(r"\s*\n\s*", " ", text)


def test_the_counts_the_documents_give_are_the_manifests():
    md = flat((ROOT / "docs" / "models.md").read_text(encoding="utf-8"))
    readme = flat((ROOT / "README.md").read_text(encoding="utf-8"))
    n, tags = len(ASSETS), len({x["from"]["tag"] for x in ASSETS})
    listed = sum(x["from"]["in_release_sha256sums"] for x in ASSETS)
    assert f"{n} files come from {tags} releases; with `SHA256SUMS` and `v0.5.0-assets.json`" in md
    assert f"the release has {n + 2}." in md
    assert f"for the {listed} files its release's `SHA256SUMS` lists" in md
    assert f"`SHA256SUMS` lists the other {n + 1} files" in md, "SHA256SUMS lists every file but itself"
    assert f"({n} files from {tags} releases, plus `SHA256SUMS` and `v0.5.0-assets.json`" in readme
    left = sum(int(b.replace(",", "")) for b in re.findall(r"(\d[\d,]+) B together", json.dumps(DOC["left_out"])))
    assert f"({round(left / 1e6)} MB together)" in readme


def test_stage_and_verify_round_trip(tmp_path):
    """stage copies each source file (checked against the manifest) and writes SHA256SUMS; verify accepts the result
    and refuses a changed byte. Made with fake sources and a fake manifest, so no release is needed."""
    src = tmp_path / "src"
    fake = []
    for i, x in enumerate(ASSETS[:4]):
        p = src / x["from"]["tag"] / x["from"]["name"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"asset {i}".encode())
        fake.append({**x, "bytes": p.stat().st_size, "sha256": M.sha256(p)})
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps({**DOC, "assets": fake}), encoding="utf-8")
    old = M.MANIFEST
    M.MANIFEST = manifest
    try:
        out = tmp_path / "out"
        M.cmd_stage(type("A", (), {"out": str(out), "src": str(src)}))
        sums = M.release_sums(out)
        assert set(sums) == {x["name"] for x in fake} | {M.LIST_NAME}
        M.cmd_verify(type("A", (), {"dir": str(out)}))
        (out / fake[0]["name"]).write_bytes(b"changed")
        with pytest.raises(SystemExit):
            M.cmd_verify(type("A", (), {"dir": str(out)}))
    finally:
        M.MANIFEST = old


# ----------------------------------------------------------------------------- the other readers' record

def record():
    return json.loads(READERS.read_text(encoding="utf-8"))


def test_the_readers_record_is_public_and_registered():
    text = READERS.read_text(encoding="utf-8")
    for word in ('"text"', '"pred"', '"ref"', "/local/home", "/home/"):
        assert word not in text, word
    r = record()
    sys.path.insert(0, str(ROOT / "eval"))
    import other_readers as OR
    assert r["protocol_sha256"] == OR.sha256_file(ROOT / "eval" / "other_readers_protocol.json")
    assert r["training_or_selection_use"] is False
    assert r["checks"]["compared"] == r["checks"]["equal"] == 28
    assert set(r["sets"]) == {"clannad", "vita-test", "psp-test", "pcsg00807", "vita+psp"}
    assert r["sets"]["pcsg00807"]["values"].get("Lens") is None, "Lens never reads private game data"


def test_the_readers_tables_add_up():
    r = record()
    for name, t in r["sets"].items():
        for k, v in t["values"].items():
            assert v["chars"] == t["chars"] and v["lines"] == t["units"], (name, k)
            assert v["cer"] == pytest.approx(100 * v["errors"] / v["chars"], abs=5e-5), (name, k)
            assert 0 <= v["exact_lines"] <= v["lines"]
        for k, p in t["pairs"].items():
            a, b = k.split("-", 1)
            want = 100 * (t["values"][a]["errors"] - t["values"][b]["errors"]) / t["chars"]
            assert p["delta"] == pytest.approx(want, abs=1e-5), (name, k)
            assert p["errors"] == [t["values"][a]["errors"], t["values"][b]["errors"]], (name, k)
            ci = p["ci95"] or p["unit_ci95"]
            assert ci[0] <= p["delta"] <= ci[1], (name, k)
    pooled = r["sets"]["vita+psp"]
    for k, v in pooled["values"].items():
        parts = [r["sets"][s]["values"][k]["errors"] for s in ("vita-test", "psp-test")]
        assert v["errors"] == sum(parts), k


def test_the_readers_table_and_pair_helpers():
    sys.path.insert(0, str(ROOT / "eval"))
    import other_readers as OR
    units = [{"id": f"u{i}", "cluster": f"c{i % 6}", "chars": 10,
              "errors": {"VOCR6_S": 0, "VOCR6_N": 0, "VOCR6_L": 0, "VOCR5_S": 1, "VOCR5_N": 1, "VOCR5_L": 1,
                         "X": 2}} for i in range(30)]
    t = OR.table(units, list(OR.VOCR) + ["X"], "vn")
    assert t["values"]["X"]["cer"] == 20.0 and t["values"]["VOCR6_L"]["exact_lines"] == 30
    p = t["pairs"]["VOCR6_L-X"]
    assert p["delta"] == -20.0 and p["ci95"] == [-20.0, -20.0] and p["clusters"] == 6
    assert t["pairs"]["VOCR6_N-VOCR5_N"]["delta"] == -10.0
    assert OR.neutral(str(OR.ROOT / "eval" / "x.py")) == "eval/x.py"
    assert OR.neutral(str(OR.HERD / "a")) == "$HERD/a"


# ----------------------------------------------------------------------------- VOCR6's numbers in the documents

FOLLOWUP = ROOT / "eval" / "vocr6_final_test_followup_20261027.json"
RESULTS = ROOT / "eval" / "vocr6_results_20261027.json"


def followup():
    return json.loads(FOLLOWUP.read_text(encoding="utf-8"))


def shipped(node, fam, tier):
    return node["rates"][f"{fam}:{tier}:shipped"]["cer"]


def test_the_vocr6_rows_of_docs_models_are_the_records():
    """docs/models.md's VOCR6.2 rows: the follow-up's read-once numbers (Rubi's path and greedy) and its gate fits'
    arenas and estimated box times; the held-out and dev numbers carry their caveats."""
    md = (ROOT / "docs" / "models.md").read_text(encoding="utf-8")
    fu = followup()
    b, a = fu["path_b"], fu["path_a"]
    gate = json.loads(RESULTS.read_text(encoding="utf-8"))["follow_up_gate"]["tiers"]
    sec = md[md.index("## The recognizers"):md.index("## The line finders")]
    rows = {m.group(1): m.group(0) for m in re.finditer(r"^\| `(VOCR6\.2_[SNL]_w8[_a-z0-9]*\.vocr)` \|.*$", sec, re.M)}
    names = {"S": "VOCR6.2_S_w8.vocr", "N": "VOCR6.2_N_w8g2.vocr", "L": "VOCR6.2_L_w8_2l.vocr"}
    for t, name in names.items():
        row = rows[name]
        g = gate[t]
        assert f"| {g['arena_w8i_1024']:,} |" in row, t
        assert f"| {g['est_box_ms']['shipped_333'] / 1000:.2f} / {g['est_box_ms']['shipped_444'] / 1000:.2f} |" in row
        cl_b = shipped(b["clannad_lines"]["dialogue"], "VOCR6", t)
        cl_a = a["clannad_lines"]["dialogue"]["rates"][f"VOCR6:{t}:greedy"]["cer"]
        vp = shipped(b["vita_psp_fixed_lines"]["pooled|dialogue"], "VOCR6", t)
        dd = shipped(b["pcsg00807_clean"], "VOCR6", t)
        for x in (cl_b, cl_a, vp, dd):
            assert f"{x:.2f}%" in row, (t, x)
        assert "3 seeds" in row and "as it ships" in row, f"{t}: the held-out and dev caveats"
        # the held-out pages (the recipe's 3-seed means) and the dev score (the set as it ships), VOCR5's beside
        held, v5_held, dev, v5_dev = vocr6_recipe_numbers(t)
        assert f"{held:.3f}% (VOCR5-{t} {v5_held:.3f}%)" in row, (t, held)
        assert f"{dev:.3f} (" in row and f"{v5_dev:.3f}" in row, (t, dev)
    assert "+0.041 [−0.016, +0.099]" in md and "not held out" in md


# The records of VOCR6's held-out and dev numbers: eval/vocr6_results_20261027.json. Held-out: the decision of the
# recipe each tier ships (stage 4d-l for Small, stage 4a for Normal and Large), 3 seeds; dev: the shipped set's runs.
RECIPE = {"S": "s4d_l", "N": "s4_N", "L": "s4_L"}
SHIPPED_DEV = {"S": ("v6f4dlh_S_seed1_up4cN", "VOCR5-S"), "N": ("v6f4c_N_seed1", "VOCR5-N"),
               "L": ("v6f4c_L_seed1", "VOCR5-L")}


def vocr6_recipe_numbers(tier):
    r = json.loads(RESULTS.read_text(encoding="utf-8"))
    hp = r["decisions"][RECIPE[tier]]["total"]["hval_page"]
    runs = r["runs"]["shipped"]
    new, old = SHIPPED_DEV[tier]
    return hp["cer_y"], hp["cer_x"], runs[new]["summary"]["dev_score"], runs[old]["summary"]["dev_score"]


def test_the_held_out_and_dev_numbers_are_the_vocr6_records():
    """CHANGELOG's VOCR6 held-out and dev numbers, from the decisions and runs they come from."""
    entry = flat((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))
    entry = entry[entry.index("## [0.5.0]"):entry.index("## Earlier releases")]
    nums = {t: vocr6_recipe_numbers(t) for t in "SNL"}
    held = ", ".join(f"{nums[t][1]:.3f} → {nums[t][0]:.3f}%" for t in "SNL")
    dev = ", ".join(f"{nums[t][3]:.3f} → {nums[t][2]:.3f}" for t in "SNL")
    assert held in entry, held
    assert dev in entry, dev
    r = json.loads(RESULTS.read_text(encoding="utf-8"))["decisions"]
    for t in "SNL":
        assert r[RECIPE[t]]["total"]["hval_page"]["ci95"][1] < 0, f"{t}: the held-out gain is resolved"
        assert len(r[RECIPE[t]]["y"]) == 3, f"{t}: three seeds"
    rep = r["s4d_report_SvsV5S_up4cN"]["total"]["dev"]
    assert f"+{rep['delta_points']:.3f} [−{abs(rep['ci95'][0]):.3f}, +{rep['ci95'][1]:.3f}]" in entry


def test_the_changelog_vocr6_numbers_are_the_records():
    ch = flat((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))
    entry = ch[ch.index("## [0.5.0]"):ch.index("## Earlier releases")]
    b = followup()["path_b"]
    node = b["clannad_lines"]["dialogue"]
    trio = lambda fam: " / ".join(f"{shipped(node, fam, t):.2f}" for t in "SNL") + "%"   # noqa: E731
    assert f"CLANNAD's dialogue lines {trio('VOCR5')} → {trio('VOCR6')}" in entry
    assert all(node["pairs"][f"{t}:shipped"]["ci95"][1] < 0 for t in "SNL"), "CLANNAD resolved better at every tier"
    for words in ("the mean of three seeds", "as it ships", "+0.041 [−0.016, +0.099]", "not resolved",
                  "glyph raster", "Est. 0.63 / 0.77 / 1.88 s", "since Rubi 0.10.0"):
        assert words in entry, words
    n, tags = len(ASSETS), len({x["from"]["tag"] for x in ASSETS})
    assert f"({n} files from {tags} releases, plus" in entry
    rec = record()
    assert all(rec["sets"][s]["values"]["VOCR6_L"]["cer"] == min(v["cer"] for v in rec["sets"][s]["values"].values())
               for s in ("clannad", "vita-test", "psp-test", "pcsg00807")), "Large lowest on all four sets"
