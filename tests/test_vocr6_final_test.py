"""eval/vocr6_final_test.py's pure parts and its protocol (eval/vocr6_final_test_protocol.json): Rubi's line-path
rules (confidence, re-read, cascade, filter), the parsing of Rubi's settings, the statistics and the public-output
guard. No protected set, model or screenshot is read here."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "train", ROOT / "runtime", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import vocr6_final_test as F  # noqa: E402

PROTO = json.loads((ROOT / "eval" / "vocr6_final_test_protocol.json").read_text(encoding="utf-8"))
SHA = re.compile(r"^[0-9a-f]{64}$")

RUBI_LMS_API42 = """
static const VjoVocrLm lms[VJO_VOCR_LM_TIERS] = {
    {VJO_VOCR_LM1_NAME, VJO_VOCR_LM1_BYTES,
     {0x38,0xea,0x94,0xe0,0xc4,0xec,0x93,0x86,0x34,0x54,0x38,0x7e,0x7c,0x2c,0x46,0x9d,
      0x20,0x9a,0xaf,0x21,0xf2,0xf6,0xee,0x5d,0x04,0x0e,0x75,0xa0,0x9d,0x00,0xd2,0x86}, 3000, 1},
    {VJO_VOCR_LM2_NAME, VJO_VOCR_LM2_BYTES,
     {0x99,0x30,0x00,0x61,0x15,0x55,0x17,0x1d,0xaf,0x63,0x61,0xe2,0x9a,0x50,0x3b,0x08,
      0x71,0xa5,0xf3,0xdd,0x24,0x0a,0x19,0x55,0x5d,0x55,0xf1,0x6a,0xaa,0x3a,0xb3,0xd9}, 3500, 2},
};
static void lm_opts(const VjoVocrLm *lm, vocr_lm_opts *o)
{
    memset(o, 0, sizeof(*o));
    o->beam = 8;
    o->topk = VJO_VOCR_LM_TOPK;
    o->max_frames = 0;  /* the model's clamp: 1,024 px / 8 */
    o->alpha = 500;
    o->beta = lm->beta;
    o->gate = 8000;
    o->flags = VOCR_LM_EOS;
    o->bracket = 8000;
}
"""

RUBI_LMS_306 = """
    {VJO_VOCR_LM2_NAME, VJO_VOCR_LM2_BYTES,
     {0x41,0x59,0x1a,0xcc,0xf8,0x49,0x9f,0x1c,0x34,0x55,0x7e,0x58,0x5c,0xca,0x38,0x2d,
      0xe2,0xc4,0xb5,0xca,0x15,0xf2,0x70,0xc9,0x85,0x61,0x2c,0x32,0x48,0x7e,0x25,0x1f}, 2500, 2, 600, VOCR_LM_BOS},
static void lm_opts(const VjoVocrLm *lm, vocr_lm_opts *o)
{
    memset(o, 0, sizeof(*o));
    o->beam = 8;
    o->topk = VJO_VOCR_LM_TOPK;
    o->max_frames = 0;
    o->alpha = lm->alpha;
    o->beta = lm->beta;
    o->gate = 8000;
    o->flags = lm->flags;
    o->bracket = 8000;
}
"""


def test_protocol_shape():
    assert PROTO["kind"] == "vocr6-final-test-protocol" and PROTO["schema"] == 1
    assert set(PROTO["sets"]) >= {"vita-test", "psp-test", "clannad", "pcsg00807"}
    assert set(PROTO["paths"]) == {"a", "b"}
    for fam in ("path_a", "gated"):
        for t in "SNL":
            assert SHA.match(PROTO["models"]["VOCR5"][fam][t]["sha256"])
    for c in PROTO["paths"]["b"]["lm_candidates"].values():
        assert SHA.match(c["sha256"])
        assert set(c["settings"]) == {"beam", "topk", "alpha", "beta", "gate", "flags", "bracket"}
    s = PROTO["sets"]
    for k in ("refs_sha256", "lines_sha256", "refs_freeze_sha256", "lines_meta_sha256", "clannad_refs_protocol_sha256"):
        assert SHA.match(s["clannad"][k])
    for k in ("pools_manifest_sha256", "pagepool_manifest_sha256", "clean_npz_sha256", "pagepool_title_sha256"):
        assert SHA.match(s["pcsg00807"][k])
    # the manifests of the shared sets are pinned as committed
    for name in ("vita-test", "psp-test"):
        assert F.sha256(ROOT / s[name]["manifest"]) == s[name]["manifest_sha256"]
    assert F.sha256(ROOT / "eval" / "clannad_refs_protocol.json") == s["clannad"]["clannad_refs_protocol_sha256"]
    # the validation targets name VOCR5's locked numbers
    t = PROTO["validation"]["targets"]
    assert "75 / 53 / 37" in t["fixed_lines"] and "54 / 34 / 25" in t["fixed_lines_lm"]
    assert "385 / 370 / 328" in t["whole_dialogue"] and "463 / 322 / 246" in t["clannad"]


def test_line_conf_is_rubis_float_rule():
    assert F.line_conf("", 0.1) == 1000
    assert F.line_conf("あ", 0.0) == 0
    assert F.line_conf("あ", 1.0) == 1000
    assert F.line_conf("あ", 0.699) == 699
    # float32 arithmetic, then truncation (core/vocr_ocr.c line_conf)
    for c in (0.6994999, 0.6995, 0.69950001, 0.12345):
        want = int(np.float32(np.float32(c) * np.float32(1000.0) + np.float32(0.5)))
        assert F.line_conf("x", c) == want


def test_kana_counter_and_read_again():
    assert F.kana_counter("三カ月") and F.kana_counter("五ケ所") and not F.kana_counter("三ヶ月")
    assert not F.kana_counter("カメラ")
    o = {"gated": 0, "escaped": 10, "frames": 10}
    assert not F.read_again(o, 1, False, "x")                   # nothing skipped: never read again
    assert F.read_again(dict(o, gated=2), 1, False, "x")        # the gate skipped frames, a counter context
    assert not F.read_again(dict(o, gated=2), 0, False, "三カ月")  # gate only: no kana rule
    assert F.read_again(dict(o, escaped=3), 0, True, "三カ月")    # two-level: frames ran the frequent rows only
    assert not F.read_again(dict(o, escaped=3), 0, True, "カメラ")
    assert not F.read_again(o, 0, True, "三カ月")                # two-level, nothing skipped


def test_compose_cascade():
    own = {"g": "abc", "b": "ABC", "conf": 650}
    up = {"g": "abd", "b": "ABD"}
    assert F.compose(own, up, 700) == "ABD"
    assert F.compose(dict(own, conf=700), up, 700) == "ABC"
    assert F.compose(own, None, 700) == "ABC"                     # Large: no cascade
    assert F.compose(own, {"g": "", "b": ""}, 700) == ""          # the next tier reads it as empty
    assert F.compose({"g": "", "b": "", "conf": 1000}, up, 700) == ""


def test_has_japanese_matches_rubi():
    assert F.has_japanese("あ") and F.has_japanese("ア") and F.has_japanese("漢")
    assert not F.has_japanese("1010…") and not F.has_japanese("ABC") and not F.has_japanese("")
    assert not F.has_japanese("「」")                             # U+300C/D are not in vjo_is_japanese_char


def test_page_reference_rule():
    ref, nt, cut = {"status": "ref"}, {"status": "excluded", "notext": True}, {"status": "excluded"}
    assert F.page_reference([ref, nt]) == "complete"
    assert F.page_reference([ref]) == "complete"
    assert F.page_reference([nt, nt]) == "no_reference_line"
    assert F.page_reference([ref, cut]) == "incomplete_reference"


def test_parse_rubi_lm_rows():
    sha, beta, tier, alpha, flags = F.parse_lm_row(RUBI_LMS_API42)
    assert sha == PROTO["paths"]["b"]["lm_candidates"]["ctclm2_1000k"]["sha256"]
    assert (beta, tier, alpha, flags) == (3500, 2, None, None)
    opts = F.parse_lm_opts(RUBI_LMS_API42)
    assert opts["alpha"] == 500 and opts["flags"] == 2 and opts["beta"] == "lm->beta" and opts["topk"] == 8
    sha, beta, tier, alpha, flags = F.parse_lm_row(RUBI_LMS_306)
    assert sha == PROTO["paths"]["b"]["lm_candidates"]["lm6_1000k"]["sha256"]
    assert (beta, tier, alpha, flags) == (2500, 2, 600, 1)
    opts = F.parse_lm_opts(RUBI_LMS_306)
    assert opts["alpha"] == "lm->alpha" and opts["flags"] == "lm->flags"


def test_resolve_lm_matches_a_candidate(tmp_path, monkeypatch):
    proto = json.loads(json.dumps(PROTO))
    for name in ("ctclm2_1000k", "lm6_1000k"):
        f = tmp_path / f"{name}.vlm"
        f.write_bytes(name.encode())
        proto["paths"]["b"]["lm_candidates"][name]["file"] = str(f)
        proto["paths"]["b"]["lm_candidates"][name]["sha256"] = F.sha256(f)
    src = RUBI_LMS_API42
    want = proto["paths"]["b"]["lm_candidates"]["ctclm2_1000k"]["sha256"]
    hexb = ",".join(f"0x{want[i:i + 2]}" for i in range(0, 64, 2))
    src = re.sub(r"(\{VJO_VOCR_LM2_NAME, VJO_VOCR_LM2_BYTES,\s*)\{[^}]*\}",
                 lambda m: m.group(1) + "{" + hexb + "}", src)
    got = F.resolve_lm(src, proto)
    assert got["candidate"] == "ctclm2_1000k" and got["settings"]["alpha"] == 500 and got["settings"]["flags"] == 2
    bad = src.replace("o->gate = 8000;", "o->gate = 7000;")
    with pytest.raises(SystemExit):
        F.resolve_lm(bad, proto)
    other = re.sub(r"(\{VJO_VOCR_LM2_NAME, VJO_VOCR_LM2_BYTES,\s*)\{[^}]*\}",
                   lambda m: m.group(1) + "{" + ",".join(["0x00"] * 32) + "}", src)
    with pytest.raises(SystemExit):
        F.resolve_lm(other, proto)


def test_pinned_lm_amendment_2(tmp_path):
    """Path (b)'s LM is pinned (amendment 2): the candidate's file and the release's settings.json must agree."""
    proto = json.loads(json.dumps(PROTO))
    pin = proto["paths"]["b"]["lm"]
    assert pin["candidate"] == "lm6_1000k" and pin["settings_tier"] == "1000k"
    c = proto["paths"]["b"]["lm_candidates"]["lm6_1000k"]
    assert c["settings"] == {"beam": 8, "topk": 8, "alpha": 600, "beta": 2500, "gate": 8000, "flags": 1,
                             "bracket": 8000}
    lm = tmp_path / "lm.vlm"
    lm.write_bytes(b"lm")
    c["file"], c["sha256"] = str(lm), F.sha256(lm)
    st = {"tiers": {"1000k": {"sha256": c["sha256"], "config": dict(c["settings"]), "cascade_conf": 700}}}
    sj = tmp_path / "settings.json"
    sj.write_text(json.dumps(st))
    pin["settings_json"], pin["settings_json_sha256"] = str(sj), F.sha256(sj)
    got = F.pinned_lm(proto)
    assert got["sha256"] == c["sha256"] and got["settings"]["flags"] == 1 and got["settings"]["alpha"] == 600
    st["tiers"]["1000k"]["config"]["beta"] = 3500
    sj.write_text(json.dumps(st))
    pin["settings_json_sha256"] = F.sha256(sj)
    with pytest.raises(SystemExit):
        F.pinned_lm(proto)
    sha, settings = F.rubi_lm(RUBI_LMS_306)
    assert sha == PROTO["paths"]["b"]["lm_candidates"]["lm6_1000k"]["sha256"] and settings == {
        "beam": 8, "topk": 8, "gate": 8000, "bracket": 8000, "beta": 2500, "alpha": 600, "flags": 1}


def test_vocr_pins_and_defines():
    cmake = ("set(VJO_VOCR_COMMIT " + "a" * 40 + ")\nset(_vocr_pins\n  vocr.c " + "b" * 64 + "\n  vocr_lines_fix.c "
             + "c" * 64 + ")\n")
    commit, pins = F.vocr_pins(cmake)
    assert commit == "a" * 40 and pins == {"vocr.c": "b" * 64, "vocr_lines_fix.c": "c" * 64}
    assert F.c_define("#define VJO_VOCR_CASCADE_CONF 700u\n", "VJO_VOCR_CASCADE_CONF") == 700
    assert F.c_define("#define VJO_VOCR_COUNTER_TAU 6000\n", "VJO_VOCR_COUNTER_TAU") == 6000


def test_assert_public():
    F.assert_public({"rates": {"VOCR6:S:shipped": {"cer": 1.0, "errors": 1, "chars": 100}}, "ids": ["a"]})
    for k in ("text", "pred", "preds", "g", "b", "ref"):
        with pytest.raises(ValueError):
            F.assert_public({"x": [{k: "あ"}]})


def test_statistics_agree_with_the_tools():
    import vocr3_report as R
    rng = np.random.default_rng(3)
    rows = []
    for i in range(60):
        n = int(rng.integers(5, 30))
        rows.append({"id": f"l{i}", "vn": f"v{i % 7}", "page": f"p{i // 3}", "chars": n,
                     "errors": {"VOCR5:S:x": int(rng.integers(0, 3)), "VOCR6:S:x": int(rng.integers(0, 3))}})
    t = F.table(rows, ["VOCR5:S:x", "VOCR6:S:x"], exact=True)
    r5 = t["rates"]["VOCR5:S:x"]
    assert r5["errors"] == sum(r["errors"]["VOCR5:S:x"] for r in rows) and r5["chars"] == sum(r["chars"] for r in rows)
    assert r5["clusters"] == 7 and r5["ci95"][0] <= r5["cer"] <= r5["ci95"][1]
    a = {r["id"]: (r["errors"]["VOCR6:S:x"], r["chars"]) for r in rows}
    b = {r["id"]: (r["errors"]["VOCR5:S:x"], r["chars"]) for r in rows}
    want = R.cluster_bootstrap(a, b, {r["id"]: r["vn"] for r in rows}, 10000, 0, 5)
    assert t["pairs"]["S:x"]["ci95"] == want["ci95"]
    assert t["pairs"]["S:x"]["unit_ci95"] == R.paired_bootstrap(a, b, 10000, 0)["ci95"]
    # page clusters: the same as clannad_refs' paired page bootstrap over page sums
    tp = F.table(rows, ["VOCR5:S:x", "VOCR6:S:x"], cluster=lambda r: r["page"], min_clusters=2)
    pa, pb = {}, {}
    for r in rows:
        x = pa.setdefault(r["page"], [0, 0])
        x[0] += r["errors"]["VOCR6:S:x"]
        x[1] += r["chars"]
        y = pb.setdefault(r["page"], [0, 0])
        y[0] += r["errors"]["VOCR5:S:x"]
        y[1] += r["chars"]
    assert tp["pairs"]["S:x"]["ci95"] == R.paired_bootstrap({k: tuple(v) for k, v in pa.items()},
                                                             {k: tuple(v) for k, v in pb.items()}, 10000, 0)["ci95"]


def test_first_diff_and_strip_timing():
    assert F.first_diff({"a": [1, {"b": 2}]}, {"a": [1, {"b": 2}]}) is None
    assert "b" in F.first_diff({"a": [1, {"b": 2}]}, {"a": [1, {"b": 3}]})
    assert F.strip_timing({"seconds": 3, "x": {"seconds": 1, "y": 2}}) == {"x": {"y": 2}}


def test_markdown_measure_and_headline_shapes():
    rates = {f"{f}:{t}:shipped": {"cer": 1.0, "errors": 10, "chars": 1000, "ci95": [0.5, 1.5], "clusters": 9}
             for f in ("VOCR5", "VOCR6") for t in "SNL"}
    pairs = {f"{t}:shipped": {"delta": -0.1, "ci95": [-0.3, 0.1]} for t in "SNL"}
    md = F.md_measure("x", {"rates": rates, "pairs": pairs}, "shipped")
    assert len([m for m in md if m.startswith("| S |") or m.startswith("| N |") or m.startswith("| L |")]) == 3
    assert "1.00% [0.50, 1.50]" in md[4] and "-0.10 [-0.30, +0.10]" in md[4]


def test_stage_texts_and_failed_units():
    def crop(g, b, conf, err=None):
        c = {"g": g, "b": b, "conf": conf, "reread": 0}
        if err:
            c["error"] = err
        return c
    dumps = {("VOCR6", "N"): {"crops": [crop("ab", "AB", 900), crop("cd", "CD", 600), crop("", "", 1000),
                                        crop("", "", 1000, "VocrError")]},
             ("VOCR6", "L"): {"crops": [crop("ab", "AX", 950), crop("cd", "CE", 990, None), crop("x", "X", 900),
                                        crop("", "", 1000)]}}
    st = F.stage_texts(dumps, "VOCR6", "N", 700, lambda texts: list(texts))
    assert st["greedy"] == ["ab", "cd", "", ""]
    assert st["rubi_tier"] == ["AB", "CD", "", ""]
    assert st["rubi_cascade"] == ["AB", "CE", "", ""]            # the doubtful line shows Large's reading
    assert st["failed"] == {3}
    # a cascade read that failed fails the line it replaces
    dumps[("VOCR6", "L")]["crops"][1]["error"] = "VocrError"
    assert F.stage_texts(dumps, "VOCR6", "N", 700, lambda t: list(t))["failed"] == {1, 3}
    # Large: no cascade
    assert F.stage_texts(dumps, "VOCR6", "L", 700, lambda t: list(t))["rubi_cascade"] == ["AX", "CE", "X", ""]
    u = {"lines": [{"text": "あい", "styles": []}, {"text": "う", "styles": ["name"]}], "crops": [0, 1]}
    texts = ["あい", "う"]
    assert F.unit_score(u, texts) == (0, 2, 0, 3)
    assert F.unit_score(u, texts, failed={1}) == (2, 2, 3, 3)    # the job fails: nothing shown


def test_driver_script_is_safe():
    sh = (ROOT / "eval" / "reproducibility" / "vocr6-final" / "report.sh").read_text(encoding="utf-8")
    assert "vocr6_final_test_protocol.json" in sh and "set -euo pipefail" in sh
    # no completion-style token is printed by the driver itself (agent panes treat them as markers)
    assert not re.search(r"[A-Z0-9]+_(DONE|BLOCKED)\b", sh)
    assert not re.search(r"[A-Z0-9]+_(DONE|BLOCKED)\b", (ROOT / "eval" / "vocr6_final_test.py").read_text())
