"""Synthetic, offline checks of eval/clannad_meiki.py: the pins (protocol, lock, code), the package check (CRLF
read as LF), reading order and the ROI rule, and one end-to-end run on a tiny fake session with a fake engine:
arm (a) in clannad_refs's extra-arm format scored by `clannad_refs.py score --suffix meiki`, arm (b) on the fake
frames, and the one whole-dialogue score. No CLANNAD data, no meikiocr, no onnxruntime."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "eval" / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


CM = module("clannad_meiki_tested", "clannad_meiki.py")
CR = CM.CR


# ----------------------------------------------------------------------------- pins

def test_protocol_lock_and_code_pin_the_same_engine():
    p = CM.protocol()
    assert p["split"] == "test" and p["amendments"] == [] and p["arm_a"]["name"] == CM.ARM_A
    assert p["arm_b"]["name"] == CM.ARM_B
    models = {m["role"]: (m["file"], m["repo"], m["revision"], m["bytes"], m["sha256"]) for m in p["engine"]["models"]}
    assert models == CM.MODEL_PINS
    assert [tuple(x) for x in p["whole_dialogue"]["pairs"]] == list(CM.PAIRS)
    for pin in (CM.WHEEL_SHA256, CM.COMMIT, *CM.BLOBS.values()):
        assert pin in p["engine"]["package"]
    assert "--pair MeikiOCR:VOCR3_W20 --pair MeikiOCR:Lens" in p["arm_a"]["score"]
    ref = json.loads((ROOT / "eval" / "clannad_refs_protocol.json").read_text())
    assert CR.sha256_file(CR.PROTOCOL_PATH) in p["references"]["freeze"]       # the frozen protocol, unchanged
    assert {"VOCR3_W20", "VOCR_H15"} <= {x["name"] for x in ref["arms"]} >= set(CM.SPEED_VOCR)
    # meikiocr 0.3.5's own defaults (run_ocr / run_recognition / CLI) and the teacher's model files
    assert (CM.DET_THRESHOLD, CM.REC_THRESHOLD, CM.PUNCT_FACTOR, CM.MAX_BATCH) == (0.5, 0.1, 1.0, 8)
    teacher = (ROOT / "teacher" / "engines.py").read_text()
    for role, (fname, _, rev, _, sha) in CM.MODEL_PINS.items():
        assert fname in teacher and rev in teacher and sha in teacher, role


def test_the_lock_pins_the_wheel_with_hashes_and_the_teachers_versions():
    assert CM.WHEEL_SHA256 in CM.lock_hashes("meikiocr")
    text = CM.LOCK_PATH.read_text()
    teacher = (ROOT / "teacher" / "requirements.lock").read_text()
    for pin in ("meikiocr==0.3.5", "onnxruntime==1.30.0", "numpy==2.4.6", "opencv-python-headless==5.0.0.93",
                "pillow==12.3.0", "huggingface-hub==0.36.2"):
        assert pin in text and pin in teacher, pin
    reqs = [x for x in text.splitlines() if x and not x[0].isspace() and not x.startswith("#")]
    assert reqs and all("==" in x for x in reqs)
    assert all(CM.lock_hashes(x.split("==")[0].lower()) for x in reqs)           # every requirement hashed


def test_verify_files_reads_crlf_as_lf_and_record_hashes(tmp_path):
    lf = b"import os\n\nx = 1\n"
    (tmp_path / "a.py").write_bytes(lf.replace(b"\n", b"\r\n"))
    (tmp_path / "b.py").write_bytes(b"y = 2\n")
    blobs = {"a.py": CM.git_blob_sha1(lf), "b.py": CM.git_blob_sha1(b"y = 3\n")}
    got = CM.verify_files(tmp_path, blobs)
    assert got["a.py"]["matches_commit"] and got["a.py"]["crlf"] == 3 and not got["b.py"]["matches_commit"]
    assert CM.git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"     # git's empty blob
    d = tmp_path / "x.dist-info"
    d.mkdir()
    digest = hashlib.sha256(b"abc").digest()
    import base64
    b64 = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    (d / "RECORD").write_text(f"pkg/m.py,sha256={b64},3\npkg/RECORD,,\n")
    assert CM.record_hashes(d) == {"pkg/m.py": digest.hex()}


# ----------------------------------------------------------------------------- geometry

def test_reading_order_rows_then_left_to_right():
    boxes = [[400, 402, 740, 428],    # right half of the dialogue line
             [44, 366, 140, 392],     # name plate
             [44, 401, 398, 429],     # left half of the same line
             [300, 460, 340, 490],    # the next line
             [600, 360, 900, 394]]    # a wide crop beside the name plate (same row)
    assert CM.reading_order(boxes) == [1, 4, 2, 0, 3]
    assert CM.reading_order([]) == []
    assert CM.reading_order([[0, 0, 10, 10], [20, 9, 30, 19]]) == [0, 1]     # 1 px of overlap: two rows
    assert CM.reading_order([[50, 0, 60, 10], [0, 4, 10, 14]]) == [1, 0]      # 6 of 10 px: one row


def test_roi_rule_and_overlap():
    roi = [32, 360, 896, 184]
    assert CM.in_roi([32, 360, 34, 362], roi) and CM.in_roi([900, 500, 926, 540], roi)
    assert not CM.in_roi([0, 360, 60, 380], roi)                                 # centre x 30 < 32
    assert not CM.in_roi([100, 20, 300, 60], roi) and not CM.in_roi([900, 540, 960, 550], roi)   # cy 545 >= 544
    assert CM.overlaps([0, 0, 10, 10], [9, 9, 20, 20]) and not CM.overlaps([0, 0, 10, 10], [10, 0, 20, 10])


def test_network_interfaces_and_the_offline_guard(monkeypatch):
    names = CM.network_interfaces()
    assert isinstance(names, list) and all(isinstance(x, str) for x in names)
    monkeypatch.setattr(CM, "network_interfaces", lambda: ["eth0", "lo"])
    with pytest.raises(SystemExit, match="unshare -rn"):
        CM.require_offline()
    monkeypatch.setattr(CM, "network_interfaces", lambda: ["lo"])
    assert CM.require_offline() == ["lo"]


def test_timed_and_the_speed_sample_are_deterministic():
    rows = [{"id": f"p{p:05d}-{i}", "page": f"p{p:05d}"} for p in range(300) for i in range(2)]
    a, b = CM.speed_sample(rows), CM.speed_sample(list(reversed(rows)))
    assert a == b and len(a[0]) == CM.SPEED_LINES and len(a[1]) == CM.SPEED_PAGES
    t = CM.timed(lambda x: x * 2, list(range(10)), repeats=2, warmup=3)
    assert t["calls"] == 20 and t["items"] == 10 and len(t["repeat_medians_ms"]) == 2


# ----------------------------------------------------------------------------- end to end (fake session)

ROI = [32, 360, 896, 184]
# id, page, ROI-relative crop box, Lens, R1 MeikiOCR / PP-OCRv6 candidates, fake arm (a) reading, reads (r1, r2, r3)
LINES = [
    ("p00001-0", "p00001", [10, 5, 100, 28], "智代", ("智代", "智伐"), "智代",
     ({"decision": "text", "text": "智代", "name": True}, {"decision": "text", "text": "智代", "name": True}, None)),
    ("p00001-1", "p00001", [10, 40, 700, 30], "「はい」", ("「はい」", "「はい」"), "「はい」",
     ({"decision": "pick", "text": "「はい」"}, {"decision": "text", "text": "「はい」"}, None)),
    ("p00002-0", "p00002", [10, 40, 700, 30], "「まあ、そう」", ("「まあ、そう」", "「まぁ、そう」"), "「まあ、そう」",
     ({"decision": "text", "text": "「まぁ、そう」"}, {"decision": "text", "text": "「まあ、そう」"},
      {"decision": "text", "text": "「まぁ、そう」"})),
    ("p00002-1", "p00002", [600, 0, 280, 35], "", ("", ""), "",
     ({"decision": "skip", "skip": "notext"}, {"decision": "skip", "skip": "notext"}, None)),
    ("p00003-0", "p00003", [10, 40, 300, 30], "「あ」", ("「ぁ」", "「あ」"), "「あ」",
     ({"decision": "text", "text": "「あ」"}, {"decision": "unsure"}, {"decision": "skip", "skip": "cut"})),
    ("p00004-0", "p00004", [10, 0, 200, 30], "", ("", ""), "",
     ({"decision": "skip", "skip": "notext"}, {"decision": "skip", "skip": "notext"}, None)),
]
SCREENS = {  # page: [(native box, text)] as the fake detector returns them (meikiocr sorts by y0 itself)
    "p00001": [([100, 20, 300, 60], "メニュー"), ([44, 366, 140, 392], "智代"), ([44, 401, 300, 429], "「はい」")],
    "p00002": [([400, 402, 740, 428], "そう」"), ([44, 401, 398, 429], "「まぁ、"), ([300, 460, 340, 490], "あ")],
    "p00003": [([44, 401, 300, 429], "「あ」")],
    "p00004": [([50, 365, 80, 390], "ー")],
}


class FakeMeiki:
    """MeikiOCR's call surface; the fill value of a crop or frame says which line or page it is."""

    def __init__(self):
        self.lines = {10 + k: x[5] for k, x in enumerate(LINES)}
        self.pages = {100 + k: SCREENS[p] for k, p in enumerate(sorted(SCREENS))}

    def run_recognition(self, images, conf_threshold=0.1, punct_conf_factor=1.0):
        assert (conf_threshold, punct_conf_factor) == (0.1, 1.0)
        return [{"text": self.lines[int(im[0, 0, 0])], "chars": [], "is_vertical": im.shape[0] > im.shape[1]}
                for im in images]

    def run_detection(self, image, conf_threshold=0.5):
        return [{"bbox": list(b)} for b, _ in self.pages[int(image[0, 0, 0])]]

    def run_ocr(self, image, det_threshold=0.5, rec_threshold=0.1, punct_conf_factor=1.0):
        assert (det_threshold, rec_threshold, punct_conf_factor) == (0.5, 0.1, 1.0)
        boxes = self.run_detection(image, det_threshold)                         # as meikiocr: one detection call
        texts = dict((tuple(b), t) for b, t in self.pages[int(image[0, 0, 0])])
        return [{"text": texts[tuple(b["bbox"])], "chars": [], "is_vertical": False} for b in boxes]


def png(path, array, mode):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(array, mode)
    im.save(path)
    return {"path": str(path.relative_to(path.parents[1])), "sha256": CR.sha256_file(path), "mode": mode,
            "size": [im.width, im.height], "pixels_sha256": hashlib.sha256(im.tobytes()).hexdigest()}


def collect(work, rnd, recs):
    sha = CR.write_jsonl(work / f"{rnd}.jsonl", recs)
    info = {"round": rnd, "lines": len(recs), "sha256": sha, "decision_files_sha256": {"b01": "0" * 64}}
    CR.write_json(work / f"{rnd}.collect.json", info)
    return info


def fake_work(tmp_path):
    """A frozen clannad_refs work dir over a fake session: lines, engines, protocol arms, rounds, references."""
    session, work = tmp_path / "session", tmp_path / "work"
    pages = sorted({x[1] for x in LINES})
    frames = {p: png(session / "frames" / f"{p}.png", np.full((544, 960, 3), 100 + k, np.uint8), "RGB")
              for k, p in enumerate(pages)}
    rows = []
    for k, (lid, page, box, lens, _, _, _) in enumerate(LINES):
        line = png(session / "lines" / f"{lid}.png", np.full((box[3], box[2]), 10 + k, np.uint8), "L")
        rows.append({"id": lid, "page": page, "seq": int(page[1:]), "i": int(lid[-1]), "line": line,
                     "source": frames[page], "roi": ROI, "box": box, "dark": False, "lens": lens, "cached": lens})
    sha = CR.write_jsonl(work / "lines.jsonl", rows)
    CR.write_json(work / "lines.meta.json", {"lines_sha256": sha, "pages": len(pages), "excluded_pages": 0,
                                             "lines": len(rows), "lock_sha256": "l", "captures_sha256": "c",
                                             "comparisons_sha256": "m", "roi": ROI, "session": str(session)})
    esha = CR.write_jsonl(work / "engines.jsonl", [{"id": x[0], "meiki": x[4][0], "v6s": x[4][1]} for x in LINES])
    CR.write_json(work / "engines.meta.json", {"engines_sha256": esha, "lines": len(LINES)})
    rounds = {}
    for k, rnd in enumerate(CR.ROUNDS):
        recs = [{"id": x[0], "round": rnd, "text": None, "skip": None, "name": False, **x[6][k]}
                for x in LINES if x[6][k] is not None]
        rounds[rnd] = collect(work, rnd, recs)
    d = [CR.load_round(work, r) for r in CR.ROUNDS]
    refs = CR.build_refs(rows, *d)
    rsha = CR.write_jsonl(work / "refs.jsonl", refs)
    CR.write_json(work / "refs.freeze.json", {
        "refs_sha256": rsha, "lines_sha256": sha, "rounds": rounds, "protocol_sha256": CR.sha256_file(CR.PROTOCOL_PATH),
        "counts": {}, "scored_lines": sum(r["status"] == "ref" for r in refs), "frozen": "2026-10-14T03:04:39Z"})
    ref_text = {r["id"]: r["text"] or "" for r in refs}
    names = [x["name"] for x in CR.protocol()["arms"]]
    arm_rows = [{"id": i, **{n: ("" if n == "v1" else "ー" if n == "VOCR3_H15" and i == "p00002-1" else ref_text[i])
                             for n in names}} for i in ref_text]
    asha = CR.write_jsonl(work / "arms.jsonl", arm_rows)
    CR.write_json(work / "arms.meta.json", {
        "arms_sha256": asha, "arms": {x["name"]: {"sha256": x["sha256"]} for x in CR.protocol()["arms"]},
        "resources": {}, "native": {}, "cache_check": {}, "code_sha256": {},
        "environment": {"python": "3", "system": "Linux", "machine": "x86_64", "libc": "glibc", "packages": {}}})
    return work, {r["id"]: r for r in refs}


def fake_identity():
    return {"package": "meikiocr", "version": "0.3.5", "commit": CM.COMMIT, "modules_match_commit": True,
            "installed_files_match_record": True, "lock_pins_wheel": True,
            "models": {r: {"file": f, "bytes": b, "sha256": s} for r, (f, _, _, b, s) in CM.MODEL_PINS.items()}}


def test_end_to_end_both_arms_and_the_scores(tmp_path, monkeypatch):
    work, refs = fake_work(tmp_path)
    into = tmp_path / "meiki"
    before = sorted(p.name for p in work.iterdir())
    monkeypatch.setattr(CM, "engine_identity", fake_identity)
    monkeypatch.setattr(CM, "require_offline", lambda: ["lo"])
    monkeypatch.setitem(CM._W, "m", FakeMeiki())
    assert refs["p00002-0"]["why"] == "two of three" and refs["p00002-1"]["notext"]
    assert refs["p00003-0"]["status"] == "excluded" and refs["p00004-0"]["notext"] and refs["p00001-0"]["name"]

    unfrozen = tmp_path / "unfrozen"                                             # MeikiOCR reads only after the freeze
    unfrozen.mkdir()
    with pytest.raises(SystemExit, match="refs.freeze.json is missing"):
        CM.main(["lines", "--work", str(unfrozen), "--into", str(into), "--procs", "0"])
    CM.main(["lines", "--work", str(work), "--into", str(into), "--procs", "0"])
    with pytest.raises(SystemExit, match="runs once"):
        CM.main(["lines", "--work", str(work), "--into", str(into), "--procs", "0"])
    arm_a = {r["id"]: r for r in CR.read_jsonl(into / "arms.meiki.jsonl")}
    assert {i: r[CM.ARM_A] for i, r in arm_a.items()} == {x[0]: x[5] for x in LINES}
    assert all(set(r) == {"id", CM.ARM_A} for r in arm_a.values())

    # arm (a), line by line, through clannad_refs's own extra-arm score
    out = tmp_path / "lines.json"
    CR.main(["score", "--work", str(work), "--suffix", "meiki", "--into", str(into), "--pair", "MeikiOCR:VOCR3_W20",
             "--pair", "MeikiOCR:Lens", "--out", str(out)])
    res = json.loads(out.read_text())
    CR.assert_public(res)
    a = res["scores"]["systems"][CM.ARM_A]
    assert (a["all"]["lines"], a["all"]["errors"], a["all"]["exact_lines"]) == (3, 1, 2)       # まあ for まぁ
    assert a["name"]["errors"] == 0 and a["not_name"]["errors"] == 1
    assert set(res["scores"]["pairs"]) == {"MeikiOCR:VOCR3_W20", "MeikiOCR:Lens"}
    assert res["scores"]["pairs"]["MeikiOCR:Lens"]["cer_delta"] == 0.0                         # Lens: the same miss
    assert res["notext_output"][CM.ARM_A] == {"lines": 2, "lines_with_output": 0, "output_chars": 0}
    assert res["extra_arms"][CM.ARM_A]["engine"] == "meikiocr 0.3.5"

    # arm (b): its own detector on the frames
    CM.main(["screens", "--work", str(work), "--into", str(into), "--procs", "0"])
    smeta = json.loads((into / "screens.meiki.meta.json").read_text())
    assert smeta["pages"] == 4 and smeta["detections"] == {"all": 8, "in_roi": 7, "vertical_in_roi": 0,
                                                           "frame_sizes": {"960x544": 4}}
    scr = {r["page"]: r for r in CR.read_jsonl(into / "screens.meiki.jsonl")}
    assert [d["text"] for d in scr["p00002"]["detections"]] == ["そう」", "「まぁ、", "あ"]       # as detected

    so = tmp_path / "screens.json"
    CM.main(["score-screens", "--work", str(work), "--into", str(into), "--out", str(so)])
    r = json.loads(so.read_text())
    CR.assert_public(r)
    blob = so.read_text(encoding="utf-8")
    for t in ("智代", "まぁ", "そう", "メニュー"):
        assert t not in blob                                                     # no text of any line or screen
    assert r["screens"] == {"pages": 4, "scored": 2, "empty": 1, "excluded": 1,
                            "excluded_pages_by_reason": {"two of three non-text": 1}, "excluded_reference_lines": 0,
                            "reference_lines": 3, "crops": 4, "empty_crops": 1}
    s = r["scores"]["systems"]
    assert s[CM.ARM_B]["chars"] == 13 and s[CM.ARM_B]["errors"] == 1 and s[CM.ARM_B]["exact_screens"] == 1
    assert s[CM.ARM_B]["dialogue_chars"] == 11 and s[CM.ARM_B]["dialogue_errors"] == 1        # the insertion: body
    assert s["VOCR1"]["errors"] == 0 and s["VOCR1"]["exact"] == 100.0
    assert s["VOCR3_H15"]["errors"] == 1                                         # ー written on the empty crop
    assert s["v1"]["errors"] == 13 and s[CM.ARM_A]["errors"] == 1 and s["Lens"]["errors"] == 1
    p = r["scores"]["pairs"]
    assert set(p) == {"MeikiOCR_screen:VOCR3_W20", "MeikiOCR_screen:Lens", "MeikiOCR_screen:MeikiOCR"}
    assert p["MeikiOCR_screen:VOCR3_W20"]["cer"]["delta"] == pytest.approx(100 / 13, abs=1e-4)
    assert p["MeikiOCR_screen:Lens"]["cer"]["delta"] == 0 and p["MeikiOCR_screen:VOCR3_W20"]["exact"]["delta"] == -50
    assert r["empty_screens_output"][CM.ARM_B] == {"screens": 1, "screens_with_output": 1, "output_chars": 1}
    assert r["empty_screens_output"]["VOCR3_H15"]["output_chars"] == 0
    d = r["meiki_screen_detections"]
    assert (d["detections"], d["in_roi"], d["outside_roi"]) == (8, 7, 1)
    assert d["scored_screens"] == {"screens": 2, "kept_detections": 5, "screens_without_detection": 0,
                                   "detections_outside_crops": 1, "chars_outside_crops": 1, "chars_written": 14,
                                   "vertical": 0}
    e = r["entanglement"]
    assert (e["arm_a_equals_r1_meiki"], e["scored_arm_a_equals_r1_meiki"]) == (5, 3)
    assert (e["scored_ref_key_equals_r1_meiki"], e["scored_ref_key_equals_r1_v6s"]) == (2, 2)
    assert (e["lens_disagree_lines"], e["lens_disagree_ref_key_equals_r1_meiki"],
            e["lens_disagree_ref_key_equals_r1_v6s"]) == (1, 0, 1)
    private = CR.read_jsonl(into / "screens.score.private.jsonl")
    assert [x["preds"][CM.ARM_B] for x in private] == ["智代「はい」", "「まぁ、そう」あ"]        # reading order
    with pytest.raises(FileExistsError):                                         # one attempt
        CM.main(["score-screens", "--work", str(work), "--into", str(into), "--out", str(tmp_path / "again.json")])
    assert sorted(p.name for p in work.iterdir()) == before                     # WORK is only read


def test_score_screens_refuses_arms_from_another_protocol(tmp_path, monkeypatch):
    work, _ = fake_work(tmp_path)
    into = tmp_path / "meiki"
    monkeypatch.setattr(CM, "engine_identity", fake_identity)
    monkeypatch.setattr(CM, "require_offline", lambda: ["lo"])
    monkeypatch.setitem(CM._W, "m", FakeMeiki())
    for cmd in ("lines", "screens"):
        CM.main([cmd, "--work", str(work), "--into", str(into), "--procs", "0"])
    changed = tmp_path / "protocol.json"
    changed.write_text(CM.PROTOCOL_PATH.read_text(encoding="utf-8").replace('"amendments": []', '"amendments": [1]'))
    monkeypatch.setattr(CM, "PROTOCOL_PATH", changed)
    with pytest.raises(SystemExit, match="another eval/clannad_meiki_protocol.json"):
        CM.main(["score-screens", "--work", str(work), "--into", str(into), "--out", str(tmp_path / "x.json")])
    assert not (into / "screens.score.attempt.json").exists()                   # a refusal uses no attempt
