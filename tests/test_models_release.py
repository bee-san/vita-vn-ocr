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
    no_attribution = {"v0.2.5-vocr3.1", "v0.2.5-vocr4.1"}       # those releases published none
    assert have == tags - no_attribution | {"v0.3.7-lm6.1"}
    assert {BY[n]["kind"] for n in ("VOCR6-ATTRIBUTION.md", "MODEL-ATTRIBUTION.md", "LM-ATTRIBUTION.md",
                                     "SPOTTER-ATTRIBUTION.md", "GATE-ATTRIBUTION.md", "VERTICAL-ATTRIBUTION.md")} == \
        {"attribution"}


def test_the_research_snapshots_before_vocr3_are_left_out_and_said_so():
    tags = {x["from"]["tag"] for x in ASSETS}
    for t in ("v0.2.2-meiki.1", "v0.2.2-fl10-training.1", "v0.2.2-fl10-bitmap.1", "v0.2.2-fl10-general.1",
              "v0.2.3-vocr2.1", "v0.2.4-fl10-head.1"):
        assert t not in tags
        assert any(t in x["tag"] for x in DOC["left_out"]), t


def test_the_md_tables_are_generated_from_the_manifest():
    text = (ROOT / "docs" / "models.md").read_text(encoding="utf-8")
    assert M.render_md(text, DOC) == text, "run `python docs/models.py md`"
    for x in ASSETS:
        assert f"| `{x['name']}` |" in text and x["sha256"] in text, x["name"]


def test_the_counts_the_documents_give_are_the_manifests():
    md = (ROOT / "docs" / "models.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    n, tags = len(ASSETS), len({x["from"]["tag"] for x in ASSETS})
    listed = sum(x["from"]["in_release_sha256sums"] for x in ASSETS)
    assert f"{n} files come from {tags} releases; with `SHA256SUMS` and `v0.5.0-assets.json`" in md
    assert f"the release has {n + 2}." in md
    assert f"for the {listed} files its release's `SHA256SUMS` lists" in md
    assert f"lists all {n + 2} files" in md
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
