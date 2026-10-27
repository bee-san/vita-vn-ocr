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


def test_c_function_extraction():
    src = """static int a(int x)
{
    if (x) { return 1; }
    return 0;
}

static __attribute__((noinline)) int probe(Stream *s, VocrLineSet *set)
{
    while (1) { break; }
    return 2;
}
"""
    assert F.c_function(src, "a").startswith("static int a(int x)") and F.c_function(src, "a").endswith("}")
    assert "while (1) { break; }" in F.c_function(src, "probe")
    assert F.c_function(src, "missing") is None
    ref = PROTO["paths"]["b"]["rubi_functions"]["sha256"]
    assert set(ref) >= {"finish_line", "read_again", "probe_short_lines", "textfilter.c:line_has_japanese"}
    assert all(SHA.match(v) for v in ref.values())


def test_rubi_job_defaults(tmp_path):
    r = tmp_path / "rubi"
    for d in ("core", "game_ocr"):
        (r / d).mkdir(parents=True)
    (r / "core" / "vocr_ocr.c").write_text("#ifndef VJO_VOCR_FINDER\n"
                                           "#define VJO_VOCR_FINDER VOCR_LINES_TEXTBOX_RECOVERY\n")
    cfg = ("    c->non_japanese_filter = VJO_FILTER_LINES;\n    c->ocr_skip_blank_lines = 0;\n"
           "    c->ocr_cascade_ms = 2000;\n    c->ocr_text_gate = VJO_TEXT_GATE_DEFAULT;\n")
    (r / "core" / "config.c").write_text(cfg)
    (r / "core" / "config.h").write_text("#define VJO_TEXT_GATE_DEFAULT 0\n")
    (r / "game_ocr" / "main.c").write_text("#define VJO_GAME_CASCADE_CONF 0u\n")
    got = F.rubi_job_defaults(r)
    assert all(ok for _, ok in got.values()) and got["line_fix_default"][0] is None
    # #330's form: the fix on by default, mapped to VOCR_FIX_DEFAULT
    (r / "core" / "config.h").write_text("#define VJO_TEXT_GATE_DEFAULT 0\n#define VJO_LINE_FIX_DEFAULT 1\n")
    (r / "game_ocr" / "main.c").write_text("#define VJO_GAME_CASCADE_CONF 0u\np->line_fix = (r->speed & "
                                           "VJO_GAME_OCR_S_LINE_FIX ? (unsigned)VOCR_FIX_DEFAULT : 0u);\n")
    assert F.rubi_job_defaults(r)["line_fix_default"] == (1, True)
    # a default that path (b) does not model is refused
    (r / "core" / "config.h").write_text("#define VJO_TEXT_GATE_DEFAULT 1\n#define VJO_LINE_FIX_DEFAULT 2\n")
    got = F.rubi_job_defaults(r)
    assert not got["ocr_text_gate"][1] and not got["line_fix_default"][1]
    (r / "core" / "config.c").write_text(cfg.replace("2000", "0").replace("VJO_FILTER_LINES", "VJO_FILTER_NONE"))
    got = F.rubi_job_defaults(r)
    assert not got["ocr_cascade_ms"][1] and not got["non_japanese_filter"][1]


def test_filter_floor():
    rows = [{"chars": 10, "nojp_chars": 0}, {"chars": 6, "nojp_chars": 6}, {"chars": 4}]
    assert F.filter_floor(rows) == {"lines": 1, "chars": 6, "points": 30.0}
    units = [{"chars": 20, "nojp_lines": 2, "nojp_chars": 5}, {"chars": 30, "nojp_lines": 0, "nojp_chars": 0}]
    assert F.filter_floor(units) == {"lines": 2, "chars": 5, "points": 10.0}
    assert F.filter_floor([])["points"] is None


def test_rows_vertical_matches_rubis_rule_cases():
    """Rubi host/tests/test_vocr_vertical.c test_rows_vertical_rule's cases, on the port."""
    two = [(12, 11, 59, 28), (14, 36, 57, 26), (12, 59, 59, 28), (13, 83, 58, 27), (12, 130, 59, 27),
           (12, 154, 59, 27), (14, 179, 57, 25), (44, 201, 26, 18)]
    three = [(12, 12, 89, 27), (12, 36, 90, 27), (12, 59, 90, 27), (15, 83, 87, 27), (16, 106, 86, 28),
             (12, 131, 91, 24), (12, 154, 89, 27), (45, 202, 57, 25)]
    psp = [(10, 8, 306, 38), (11, 44, 309, 48)]
    vita = [(30, 13, 510, 34), (12, 48, 279, 34)]
    one = [(14, 8, 831, 46)]
    screen = [(196, 0, 764, 34), (196, 28, 764, 25), (196, 48, 220, 20), (196, 87, 475, 28), (0, 358, 960, 74),
              (251, 426, 164, 28)]
    rv = F.rows_vertical
    assert rv(two, 83, 287) and rv(three, 114, 358) and rv(two, 300, 287) and rv(three, 300, 358)
    assert not rv(two[:4], 300, 287) and rv(two[:2], 83, 287) and not rv(two[:1], 83, 287)
    assert not rv(psp, 358, 94) and not rv(vita, 552, 95) and not rv(one, 880, 62) and not rv(screen, 960, 544)
    assert not rv(psp, 94, 358) and not rv([], 83, 287)
    edge = [(0, 0, 40, 10), (0, 10, 40, 10), (0, 20, 40, 10), (0, 30, 41, 10), (0, 40, 41, 10)]
    assert rv(edge, 100, 100) and not rv(edge, 101, 100)
    fewer = [(0, 0, 40, 10), (0, 10, 40, 10), (0, 20, 41, 10), (0, 30, 41, 10), (0, 40, 41, 10)]
    assert not rv(fewer, 100, 100)
    b = [[0, 10 * i, 35, 10] for i in range(10)]
    assert rv([tuple(x) for x in b], 500, 100)
    for i in range(3):
        b[i][2] = 36
    assert rv([tuple(x) for x in b], 500, 100)
    b[3][2] = 36
    assert not rv([tuple(x) for x in b], 500, 100)
    b = [[0, 10 * i, 30, 10] for i in range(10)]
    for i in range(3):
        b[i][0] = 300
    assert rv([tuple(x) for x in b], 500, 100)
    b[3][0] = 300
    assert not rv([tuple(x) for x in b], 500, 100)
    b = [[0, 10 * i, 30, 10] for i in range(10)]
    for i in (9, 8, 7, 6):
        b[i][0] = 7
    assert not rv([tuple(x) for x in b], 500, 100)
    for i in (9, 8, 7, 6):
        b[i][0] = 6
    assert rv([tuple(x) for x in b], 500, 100)
    assert not rv([(0, 0, 10, 10)] * 17, 100, 300)          # more than 16 rows: Rubi's guard


def test_stage_texts_columns():
    """A crop of a region auto read as columns shows the vertical model's reading, at every tier, never cascaded."""
    def crop(g, b, conf):
        return {"g": g, "b": b, "conf": conf, "reread": 0}
    dumps = {("VOCR6", "N"): {"crops": [crop("ab", "AB", 600), None]},
             ("VOCR6", "L"): {"crops": [crop("ab", "AX", 950), None]},
             ("V", "V"): {"crops": [None, crop("cd", "CD", 300)]}}
    st = F.stage_texts(dumps, "VOCR6", "N", 700, lambda t: list(t), columns=[1])
    assert st["rubi_cascade"] == ["AX", "CD"] and st["greedy"] == ["ab", "cd"] and st["rubi_tier"] == ["AB", "CD"]


def test_job_crops_frozen_check_for_tall_regions(monkeypatch):
    """Rubi API 50 + 47 (recognize: frozen_check): with a fix and the auto check, a region at least as tall as wide is
    checked on the frozen finder's rows (fix 0), a wider one on the fixed finder's; a region the check calls vertical
    is read as columns, any other through the line path at the fix (audit N16)."""
    calls = []
    vertical_rows = [(0, 10 * i, 30, 10) for i in range(8)]       # rows of one character: vertical text
    wide_row = [(0, 0, 400, 30)]

    def fake_find(exe, imgs, platforms, fix, probe=1, refine=None):
        calls.append((len(imgs), fix, probe, refine))
        out = []
        for img in imgs:
            tall = img.shape[0] >= img.shape[1]
            rows = vertical_rows if (tall and fix == 0) else wide_row   # the fix joins the columns' parts
            out.append((rows, [0] * len(rows), {"found": len(rows), "probed": 0, "rc": 0}))
        return out

    monkeypatch.setattr(F, "find_boxes", fake_find)
    monkeypatch.setattr(F, "region_columns", lambda g: ([((1, 2, 3, 4), 1, np.zeros((3, 4), np.uint8))], 0))
    tall, wide = np.zeros((200, 100), np.uint8), np.zeros((100, 400), np.uint8)
    rp = {"finder": {"driver": "x", "fix": 1}, "vertical_check": True}
    got = F.job_crops(rp, [tall, wide], ["vita", "vita"])
    assert got[0][4] is True and got[0][0] == [[1, 2, 3, 4]]          # tall: frozen rows look vertical -> columns
    assert got[1][4] is False and got[1][0] == [list(wide_row[0])]     # wide: the line path at the fix
    assert (1, 0, 0, 0) in calls and (1, 1, 0, 0) in calls             # the check: frozen for tall, fixed for wide
    assert (2, 1, 1, None) in calls                                    # the line path: both regions, fix 1
    # without the fix the check runs at fix 0 for every region (Rubi before API 50)
    calls.clear()
    F.job_crops({"finder": {"driver": "x", "fix": 0}, "vertical_check": True}, [tall, wide], ["vita", "vita"])
    assert (2, 0, 0, 0) in calls
    # without auto there is no check
    calls.clear()
    got = F.job_crops({"finder": {"driver": "x", "fix": 1}, "vertical_check": False}, [tall], ["vita"])
    assert calls == [(1, 1, 1, None)] and got[0][4] is False


def test_vocr_pins_and_defines():
    cmake = ("set(VJO_VOCR_COMMIT " + "a" * 40 + ")\nset(_vocr_pins\n  vocr.c " + "b" * 64 + "\n  vocr_lines_fix.c "
             + "c" * 64 + ")\n")
    commit, pins = F.vocr_pins(cmake)
    assert commit == "a" * 40 and pins == {"vocr.c": "b" * 64, "vocr_lines_fix.c": "c" * 64}
    assert F.c_define("#define VJO_VOCR_CASCADE_CONF 700u\n", "VJO_VOCR_CASCADE_CONF") == 700
    assert F.c_define("#define VJO_VOCR_COUNTER_TAU 6000\n", "VJO_VOCR_COUNTER_TAU") == 6000


def test_committed_records_are_public():
    """Every committed record of this harness holds counts, rates, intervals, ids and hashes only."""
    recs = sorted(p for p in (ROOT / "eval").glob("vocr6_final_test_*.json")
                  if p.name != "vocr6_final_test_protocol.json")
    assert len(recs) >= 5
    for p in recs:
        F.assert_public(json.loads(p.read_text(encoding="utf-8")))


def test_final_record_is_the_one_read():
    """The final read's record: one final attempt, on the committed protocol (unchanged since the read), the blobs
    vocr6-lead's FINAL READ named, VOCR5's path (a) equal to its published records, no read or finder error."""
    import hashlib
    rec = json.loads((ROOT / "eval" / "vocr6_final_test_20261027.json").read_text(encoding="utf-8"))
    # the protocol as it was at that read: amendment 9 (the follow-up's finalist set) was appended after it
    a9 = [x for x in PROTO["amendments"] if x["n"] == 9]
    proto = a9[0]["protocol_before_sha256"] if a9 else \
        hashlib.sha256((ROOT / "eval" / "vocr6_final_test_protocol.json").read_bytes()).hexdigest()
    assert rec["attempt"]["mode"] == "final" and rec["protocol_sha256"] == rec["attempt"]["protocol_sha256"] == proto
    assert rec["attempt"]["models"] == PROTO["finalist_sets"]["v0.4.0-vocr6.1"]["models"]
    assert {t: h[:8] for t, h in rec["attempt"]["models"].items()} == {"S": "76a1bc5d", "N": "929652e3",
                                                                       "L": "b051fd4f"}
    assert rec["validation"]["all_equal"] is True
    for s, r in rec["path_b"]["reads"].items():
        assert r["finder_errors"] == 0 and not any(r["read_errors"].values()), s


def test_follow_up_record_is_its_one_read():
    """The follow-up's read (amendment 9): one final attempt of the registered set v0.4.1-vocr6.2, on the committed
    protocol (no amendment since the read), at the registered Rubi head, VOCR5's path (a) equal to its records, no
    read or finder error; its compare pairs it with the earlier read unit for unit (VOCR5 alike in every unit), and
    every compared rate is the one each read's own report shows."""
    import hashlib
    rec = json.loads((ROOT / "eval" / "vocr6_final_test_followup_20261027.json").read_text(encoding="utf-8"))
    cmp_ = json.loads((ROOT / "eval" / "vocr6_final_test_followup_compare_20261027.json").read_text(encoding="utf-8"))
    old = json.loads((ROOT / "eval" / "vocr6_final_test_20261027.json").read_text(encoding="utf-8"))
    fs = PROTO["finalist_sets"]["v0.4.1-vocr6.2"]
    proto = hashlib.sha256((ROOT / "eval" / "vocr6_final_test_protocol.json").read_bytes()).hexdigest()
    assert rec["attempt"]["mode"] == "final" and rec["attempt"]["models"] == fs["models"]
    assert rec["protocol_sha256"] == rec["attempt"]["protocol_sha256"] == proto
    assert rec["rubi_path"]["rubi_head"] == fs["rubi_head"] and rec["validation"]["all_equal"] is True
    for s, r in rec["path_b"]["reads"].items():
        assert r["finder_errors"] == 0 and not any(r["read_errors"].values()), s
    F.assert_public(rec)
    F.assert_public(cmp_)
    assert (cmp_["this"]["set"], cmp_["earlier"]["set"]) == ("v0.4.1-vocr6.2", "v0.4.0-vocr6.1")
    assert cmp_["this"]["attempt"] == rec["attempt"] and cmp_["earlier"]["attempt"] == old["attempt"]
    keys = {"b|clannad_lines": "clannad_lines", "b|clannad_pages": "clannad_pages",
            "b|vita_psp_fixed_lines": "vita_psp_fixed_lines",
            "b|vita_psp_whole_dialogue_base": "vita_psp_whole_dialogue_base", "b|pcsg00807_clean": "pcsg00807_clean",
            "b|pcsg00807_page_base": "pcsg00807_page", "a|clannad_lines": "clannad_lines",
            "a|vita_psp_fixed_lines": "vita_psp_fixed_lines", "a|vita_psp_fixed_lines_lm": "vita_psp_fixed_lines_lm",
            "a|vita_psp_whole_dialogue_base": "vita_psp_whole_dialogue_base", "a|pcsg00807_clean": "pcsg00807_clean",
            "a|pcsg00807_page_refine_base": "pcsg00807_page"}
    assert set(cmp_["measures"]) == set(keys)
    for name, m in cmp_["measures"].items():
        assert m["same_units"] and m["vocr5_units_differ"] == 0, name
        path = "path_" + name[0]
        for t in "SNL":
            h, ho, e = rec["headline"][path][keys[name]][t], old["headline"][path][keys[name]][t], m["tiers"][t]
            assert (e["this"]["errors"], e["this"]["chars"]) == (h["VOCR6"]["errors"], h["VOCR6"]["chars"]), name
            assert e["vocr5"]["errors"] == h["VOCR5"]["errors"] == ho["VOCR5"]["errors"], name
            assert e["earlier"]["errors"] == ho["VOCR6"]["errors"], name
            assert e["this_minus_vocr5"]["ci95"] == h["VOCR6_minus_VOCR5"]["ci95"], name


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
    # a cascade read that failed leaves the job tier's own reading (Rubi's finish_line), counted
    dumps[("VOCR6", "L")]["crops"][1]["error"] = "VocrError"
    st2 = F.stage_texts(dumps, "VOCR6", "N", 700, lambda t: list(t))
    assert st2["failed"] == {3} and st2["rubi_cascade"][1] == "CD" and st2["cascade_failed"] == 1
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
    # amendment 9: a final read goes through the finalist-set guard and publishes to its set's directory
    assert '"$H" finalist --models "$MODELS" --priv "$PRIV"' in sh
    assert 'cp "$OUT/report.json" "$OUT/report.md" "$PUBLISH/"' in sh and '"$A/vocr6-final-test/"' not in sh
    assert 'OUT is the finalist set\'s publish directory' in sh and "names other blobs than" in sh    # the review's
    assert "exists and is not empty: give a new OUT" in sh                                     # audit n4


# ----------------------------------------------------------------------------- amendment 9

def test_finalist_sets_and_runtime_differences_are_registered():
    """Amendment 9: two registered finalist sets with distinct blobs (the follow-up's names the earlier set, its
    publish directory and the Rubi head it reads at); the runtime differences rubi-path accepts are exactly the
    runtimecheck record's, on the files this checkout holds, with every read equal for the follow-up's staged blobs."""
    fs = {k: v for k, v in PROTO["finalist_sets"].items() if k != "rule"}
    assert set(fs) == {"v0.4.0-vocr6.1", "v0.4.1-vocr6.2"}
    for v in fs.values():
        assert set(v["models"]) == set("SNL") and all(SHA.match(h) for h in v["models"].values())
    assert fs["v0.4.0-vocr6.1"]["models"] != fs["v0.4.1-vocr6.2"]["models"]
    f = fs["v0.4.1-vocr6.2"]
    assert f["earlier"] in fs and re.fullmatch(r"[0-9a-f]{40}", f["rubi_head"]) and "vocr6-followup" in f["publish"]
    assert f["earlier_private"].endswith("/final-20261027")
    assert "vocr6-final-test" in fs["v0.4.0-vocr6.1"]["publish"]
    rd = PROTO["paths"]["b"]["runtime_differences"]
    rec_path = ROOT / rd["check"]["record"]
    assert F.sha256(rec_path) == rd["check"]["record_sha256"]
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    F.assert_public(rec)
    assert rec["all_equal"] is True and rec["rubi_head"] == f["rubi_head"]
    assert {k: [v["rubi_pin"], v["checkout"]] for k, v in rd["files"].items()} == rec["files"]
    for k, v in rd["files"].items():
        assert F.sha256(ROOT / "runtime" / k) == v["checkout"]
    for t in "SNL":
        e = rec["blobs"][f"staged_{t}"]
        assert e["sha256"] == f["models"][t] and e["raw_equal"] == e["line_equal"] == e["crops"] >= 4000
    assert rec["libraries"]["checkout"] != rec["libraries"]["rubi"]
    assert F.runtime_differences({k: v["rubi_pin"] for k, v in rd["files"].items()},
                                 {k: v["checkout"] for k, v in rd["files"].items()}, rd["files"], rec)["refused"] == []


def _models(s, n, large):
    return {"S": s * 64, "N": n * 64, "L": large * 64}


def test_finalist_of_and_same_set_attempts(tmp_path):
    proto = {"finalist_sets": {"rule": "x", "one": {"models": _models("a", "b", "c")},
                               "two": {"models": _models("a", "d", "e"), "publish": "/p", "rubi_head": "f" * 40}}}
    assert F.finalist_of(proto, _models("a", "d", "e"))[0] == "two"
    with pytest.raises(SystemExit):
        F.finalist_of(proto, _models("a", "b", "e"))            # not a registered set
    root = tmp_path / "priv"
    for d, rec in (("old", {"mode": "final", "models": _models("a", "b", "c")}),
                   ("val", {"mode": "validate"})):
        (root / d).mkdir(parents=True)
        (root / d / "attempt.json").write_text(json.dumps(rec))
    me = root / "new"
    me.mkdir()
    assert F.same_set_attempts(root, me, _models("a", "d", "e")) == []
    assert F.same_set_attempts(root, me, _models("a", "b", "c")) == [str(root / "old" / "attempt.json")]
    assert F.same_set_attempts(root, root / "old", _models("a", "b", "c")) == []    # its own record (a resume)
    # a final record without models cannot be told apart: it counts as the same set
    (root / "bare").mkdir()
    (root / "bare" / "attempt.json").write_text(json.dumps({"mode": "final"}))
    assert F.same_set_attempts(root, me, _models("a", "d", "e")) == [str(root / "bare" / "attempt.json")]


def test_rubi_head_and_registered_record(tmp_path):
    """Amendment 9's two rubi-path guards (audit-vocr6-followup n3): the registered Rubi head, and the runtimecheck
    record accepted only as the registered file for Rubi's pinned commit."""
    F.check_rubi_head("a" * 40, "")
    F.check_rubi_head("a" * 40, "-")
    F.check_rubi_head("a" * 40, "a" * 40)
    with pytest.raises(SystemExit):
        F.check_rubi_head("a" * 40, "b" * 40)
    rec = tmp_path / "eval" / "rec.json"
    rec.parent.mkdir()
    rec.write_text(json.dumps({"all_equal": True, "rubi_vocr_commit": "c" * 40}))
    rd = {"check": {"record": "eval/rec.json", "record_sha256": F.sha256(rec)}}
    assert F.registered_record(rd, "c" * 40, tmp_path)[1]["all_equal"] is True
    assert F.registered_record(rd, "d" * 40, tmp_path)[1] is None                  # made for another Rubi pin
    assert F.registered_record(dict(rd, check=dict(rd["check"], record_sha256="0" * 64)), "c" * 40, tmp_path)[1] \
        is None                                                                         # not the registered file
    assert F.registered_record({}, "c" * 40, tmp_path) == (None, None)
    assert F.registered_record({"check": {"record": "eval/none.json"}}, "c" * 40, tmp_path)[1] is None


def test_finalist_needs_a_publish_directory(tmp_path, monkeypatch):
    """audit-vocr6-followup n4: a registered set without its own publish directory is refused (it would have
    published into the first read's directory)."""
    models = tmp_path / "models"
    models.mkdir()
    (models / "models.json").write_text(json.dumps({"tiers": {t: {"sha256": h} for t, h in _models("e", "f", "9")
                                                              .items()}}))
    proto = {"kind": "vocr6-final-test-protocol", "schema": 1,
             "finalist_sets": {"rule": "x", "new": {"models": _models("e", "f", "9")}}}
    monkeypatch.setattr(F, "protocol", lambda: proto)
    args = type("A", (), {"models": str(models), "priv": str(tmp_path / "priv" / "new")})
    with pytest.raises(SystemExit):
        F.cmd_finalist(args)
    proto["finalist_sets"]["new"]["publish"] = str(tmp_path / "pub")
    F.cmd_finalist(args)


def test_runtime_differences():
    pins = {"vocr.c": "1" * 64, "vocr.h": "2" * 64, "vocr_lm.c": "3" * 64}
    checkout = {"vocr.c": "4" * 64, "vocr.h": "2" * 64, "vocr_lm.c": "3" * 64}
    reg = {"vocr.c": {"rubi_pin": "1" * 64, "checkout": "4" * 64}}
    rec = {"all_equal": True, "files": {"vocr.c": ["1" * 64, "4" * 64]}}
    assert F.runtime_differences(pins, checkout, reg, rec) == {"accepted": ["vocr.c"], "refused": []}
    assert F.runtime_differences(pins, checkout, reg, dict(rec, all_equal=False))["refused"] == ["vocr.c"]
    assert F.runtime_differences(pins, checkout, reg, None)["refused"] == ["vocr.c"]
    assert F.runtime_differences(pins, dict(checkout, **{"vocr.c": "5" * 64}), reg, rec)["refused"] == ["vocr.c"]
    assert F.runtime_differences(pins, dict(checkout, **{"vocr_lm.c": None}), reg, rec)["refused"] == ["vocr_lm.c"]
    assert F.runtime_differences(pins, dict(pins), {}, None) == {"accepted": [], "refused": []}


def test_compare_reads_pairs_the_same_units():
    import vocr3_report as R
    rng = np.random.default_rng(5)
    old, new = [], []
    for i in range(40):
        n = int(rng.integers(5, 30))
        e5 = {f"VOCR5:{t}:shipped": int(rng.integers(0, 3)) for t in "SNL"}
        o = {"id": f"u{i}", "vn": f"v{i % 6}", "chars": n,
             "errors": {**e5, **{f"VOCR6:{t}:shipped": int(rng.integers(0, 3)) for t in "SNL"}}}
        m = {"id": f"u{i}", "vn": f"v{i % 6}", "chars": n,
             "errors": {**e5, **{f"VOCR6:{t}:shipped": int(rng.integers(0, 3)) for t in "SNL"}}}
        old.append(o)
        new.append(m)
    c = F.compare_reads(new, old[::-1], "shipped", "vn", 5)
    assert c["same_units"] and c["vocr5_units_differ"] == 0 and c["units"] == 40 and c["clusters"] == 6
    a = {r["id"]: (r["errors"]["VOCR6:N:shipped"], r["chars"]) for r in new}
    b = {r["id"]: (r["errors"]["VOCR6:N:shipped"], r["chars"]) for r in old}
    want = R.cluster_bootstrap(a, b, {r["id"]: r["vn"] for r in new}, 10000, 0, 5)
    assert c["tiers"]["N"]["this_minus_earlier"]["ci95"] == want["ci95"]
    assert c["tiers"]["N"]["earlier"]["errors"] == sum(r["errors"]["VOCR6:N:shipped"] for r in old)
    old[3]["errors"]["VOCR5:S:shipped"] += 1
    assert F.compare_reads(new, old, "shipped", "vn", 5)["vocr5_units_differ"] == 1
    assert F.compare_reads(new[:-1], old, "shipped", "vn", 5)["same_units"] is False
    other = [dict(r) for r in old]
    other[0]["chars"] += 1                                       # the same ids, another reference length
    assert F.compare_reads(new, other, "shipped", "vn", 5)["same_units"] is False
    F.assert_public(c)
