"""The VOCR3 private-set adapters on generated fixtures only (never the owner's private inputs).

Fresh native textbox: tests/test_fresh_native_textbox.py's generated cohort (six titles, two consoles)
with drawn native frames. CLANNAD long run: a generated session written through
eval/clannad_longrun.py's own capture and hash-chained journal functions.
"""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT / "eval", ROOT / "runtime", ROOT / "synth", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import clannad_longrun as LR  # noqa: E402
import vocr3_clannad as VC  # noqa: E402
import vocr3_fresh as VF  # noqa: E402
import vocr3_report as R  # noqa: E402

import test_fresh_native_textbox as TF  # noqa: E402
import vocr3_fixtures as F  # noqa: E402


def write_spec(tmp_path, pairs=("B:A",), gates=("B:A",)):
    _, sha_a = F.fixture_blob(tmp_path / "A.vocr", seed=0)
    _, sha_b = F.fixture_blob(tmp_path / "B.vocr", seed=1)
    spec = {"schema": 1, "kind": "vocr3-report-spec",
            "arms": [{"name": "A", "file": "A.vocr", "finder": "grad", "sha256": sha_a},
                     {"name": "B", "file": "B.vocr", "finder": "textbox", "sha256": sha_b}],
            "pairs": [R.parse_pair(p) for p in pairs], "gates": [R.parse_pair(g) for g in gates],
            "bootstrap": {"reps": 200, "seed": 0, "min_clusters": 2}}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path, sha_a, sha_b


def authorize(prepared, kind, tmp_path):
    auth = tmp_path / "authorization.json"
    auth.write_text(json.dumps(R.expected_authorization(prepared, kind)))
    return auth


# ---------------------------------------------------------------- fresh native textbox

@pytest.fixture
def fresh_inputs(tmp_path):
    """The existing generated cohort, with a drawn native frame behind every source."""
    m, a = TF.cohort()
    for i, source in enumerate(m["sources"]):
        w, h = source["dims"]
        rgb = F.screen((w, h), source["base_roi"], [(20, 120, 300, 24)], dark=bool(i % 3 == 0), seed=i)
        Image.fromarray(rgb).save(tmp_path / source["image"])
        source["sha256"] = hashlib.sha256((tmp_path / source["image"]).read_bytes()).hexdigest()
        source["pixel_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
        for variant in source["variants"].values():
            x, y, vw, vh = variant["roi"]
            roi = np.ascontiguousarray(rgb[y:y + vh, x:x + vw])
            variant["pixel_sha256"] = hashlib.sha256(roi.tobytes()).hexdigest()
    manifest, annotations = TF.documents(tmp_path, m, a)
    spec, _, _ = write_spec(tmp_path)
    return SimpleNamespace(tmp=tmp_path, manifest=manifest, annotations=annotations, spec=spec, m=m, a=a)


def test_fresh_prepare_is_frozen_and_forward_free(fresh_inputs, monkeypatch):
    t = fresh_inputs
    monkeypatch.setattr(R.CReader, "text", lambda *a: pytest.fail("prepare ran a model forward"))
    prepared = VF.prepare(t.manifest, t.annotations, t.spec, t.tmp, t.tmp / "audit")
    assert prepared["finder_calls"] == 0 and prepared["model_forwards"] == 0
    assert prepared["readiness"]["ready"] and prepared["readiness"]["selected_sources"] == 36
    assert set(prepared["arms"]) == {"A", "B"} and prepared["resources"]["B"]["finder"] == "textbox"
    with pytest.raises(FileExistsError):
        VF.prepare(t.manifest, t.annotations, t.spec, t.tmp, t.tmp / "audit")
    VF.verify(t.tmp / "audit" / "prepared.json", images=True)
    t.a["lines"][0]["text"] += "あ"
    t.annotations.write_text(json.dumps(t.a, ensure_ascii=False))
    with pytest.raises(ValueError, match="changed"):
        VF.verify(t.tmp / "audit" / "prepared.json")


def test_fresh_metadata_and_pins_are_checked_before_pixels(fresh_inputs, monkeypatch):
    t = fresh_inputs
    monkeypatch.setattr(VF.fresh, "image_rgb", lambda s: pytest.fail("invalid inputs reached pixels"))
    bad = json.loads(t.manifest.read_text())
    bad["sources"][0]["split"] = "dev"
    t.manifest.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        VF.prepare(t.manifest, t.annotations, t.spec, t.tmp, t.tmp / "audit")
    assert not (t.tmp / "audit").exists()
    spec = json.loads(t.spec.read_text())
    del spec["arms"][0]["sha256"]
    t.spec.write_text(json.dumps(spec))
    t.manifest.write_text(json.dumps(t.m))
    with pytest.raises(ValueError, match="pinned by sha256"):
        VF.prepare(t.manifest, t.annotations, t.spec, t.tmp, t.tmp / "audit")


def test_fresh_one_authorized_attempt_scores_every_arm(fresh_inputs):
    t = fresh_inputs
    VF.prepare(t.manifest, t.annotations, t.spec, t.tmp, t.tmp / "audit")
    prepared = t.tmp / "audit" / "prepared.json"
    wrong = t.tmp / "wrong.json"
    wrong.write_text(json.dumps({"authorization": "no"}))
    with pytest.raises(ValueError, match="authorization"):
        VF.compare(prepared, wrong)
    assert not (t.tmp / "audit" / "attempt.json").exists()
    result = VF.compare(prepared, authorize(prepared, VF.KIND, t.tmp))
    R.assert_public(result)
    per_title = {}
    for line in t.a["lines"]:
        source = next(s for s in t.m["sources"] if s["id"] == line["source_id"])
        key = source["platform"] + ":" + source["vn"]
        per_title[key] = per_title.get(key, 0) + len(VF.norm(line["text"]))
    for arm in ("A", "B"):
        for variant in VF.fresh.VARIANTS:
            groups = result["groups"][arm][variant]
            assert len(groups) == 6 and sum(g["frames"] for g in groups.values()) == 36
            assert all(type(g["all"]["full"]["errors"]) is int for g in groups.values())
            assert {k: g["all"]["full"]["chars"] for k, g in groups.items()} == per_title
        assert result["accounting"][arm]["forwards"] == result["accounting"][arm]["detections"]
    pair = result["pairs"]["B:A"]["base"]["pooled"]["all/full"]
    assert pair["regions"]["units"] == 36 and pair["vn_clusters"]["clusters"] == 6
    assert result["gates"]["B:A"]["title_checks"] == 3 * 6 * 4
    assert result["one_attempt"] and not result["retry"] and len(result["source_manifest"]) == 36
    rows = [json.loads(x) for x in (t.tmp / "audit" / "private-predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 36 * 3 * 2 and all(r["status"] == "ok" for r in rows)
    assert result["raw_predictions_sha256"] == R.sha256(t.tmp / "audit" / "private-predictions.jsonl")
    with pytest.raises(FileExistsError, match="no retries"):
        VF.compare(prepared, t.tmp / "authorization.json")


def test_fresh_failure_keeps_a_receipt_and_blocks_retry(fresh_inputs, monkeypatch):
    t = fresh_inputs
    VF.prepare(t.manifest, t.annotations, t.spec, t.tmp, t.tmp / "audit")
    prepared = t.tmp / "audit" / "prepared.json"
    auth = authorize(prepared, VF.KIND, t.tmp)

    def boom(*a, **k):
        raise RuntimeError("generated failure")
    monkeypatch.setattr(VF, "score", boom)
    with pytest.raises(RuntimeError, match="generated failure"):
        VF.compare(prepared, auth)
    receipt = json.loads((t.tmp / "audit" / "failure.json").read_text())
    assert receipt["retry_allowed"] is False and receipt["error_type"] == "RuntimeError"
    with pytest.raises(FileExistsError, match="no retries"):
        VF.compare(prepared, auth)


# ---------------------------------------------------------------- CLANNAD long run

REFS = ["あいうえお", "かきくけこさし", "たちつてと", "なにぬねの", "はひふへほ", "まみむめも"]


def clannad_session(tmp_path, blob, pages=5):
    """A generated long-run session in the frozen harness's own formats (lock, journals, PNGs)."""
    from pyvocr import Vocr
    session = tmp_path / "session"
    sha = R.sha256(blob)
    lock_sha = LR.frozen_file(session / "lock.json", {"schema": 1, "split": "test", "runtime": {"model_sha256": sha},
                                                      "sources": {}, "host_dependencies": {}})
    native = R.build_native(tmp_path / "fixture-native")
    reader = Vocr(blob, mode="w8i", max_width=1024, lib=R.load_library(native))
    roi, rows = [32, 360, 896, 184], [(20, 30, 520, 26), (20, 74, 420, 26)]
    for i in range(pages):
        x0, y0 = roi[:2]
        rgb = F.screen((960, 544), roi, [(x0 + x, y0 + y, w, h) for x, y, w, h in rows], seed=i)

        def crops(region):
            grey = LR.to_gray(region)
            return [(grey[y:y + h, x:x + w].copy(), [x, y, w, h], False) for x, y, w, h in rows], False, 1234
        event = LR.capture_event(session, rgb, roi, SimpleNamespace(crops=crops), f"signature-{i}", lock_sha,
                                 "generated", {"page": i, "method": "operator"})
        capture = LR.append_event(session / "captures.jsonl", event)
        lines, refs = [], []
        for j, entry in enumerate(capture["lines"]):
            ref = REFS[(i + j) % len(REFS)]
            refs.append(ref)
            cached = reader.text(LR.verified_image(session, entry), invert=False)
            lines.append({"input_sha256": entry["sha256"], "pixels_sha256": entry["pixels_sha256"], "box": entry["box"],
                          "dark": entry["dark"], "google": {"ok": True, "text": ref, "host_seconds": 0.1},
                          "vocr": {"ok": True, "text": cached, "host_seconds": 0.01}})
        whole_text = "" if i == pages - 1 else "".join(refs)            # the last page has an empty reference
        whole = {"ok": True, "text": whole_text, "lines": [], "host_seconds": 0.2}
        LR.append_event(session / "comparisons.jsonl", {
            "kind": "comparison", "split": "test", "capture_sha256": capture["event_sha256"], "compared_at_unix": 0.0,
            "lock_sha256": lock_sha, "ok": bool(LR.norm(whole_text)), "lines": lines, "whole_roi_google": whole})
    expect = {"lock_sha256": R.sha256(session / "lock.json"), "captures_sha256": R.sha256(session / "captures.jsonl"),
              "comparisons_sha256": R.sha256(session / "comparisons.jsonl"), "baseline_model_sha256": sha,
              "counts": None}
    return session, expect


@pytest.fixture
def clannad_inputs(tmp_path):
    spec, sha_a, _ = write_spec(tmp_path, gates=())
    session, expect = clannad_session(tmp_path, tmp_path / "A.vocr")
    return SimpleNamespace(tmp=tmp_path, spec=spec, session=session, expect=expect, sha_a=sha_a)


def test_clannad_published_pins_are_required_before_pixels(clannad_inputs, monkeypatch):
    t = clannad_inputs
    monkeypatch.setattr(LR, "verified_image", lambda *a: pytest.fail("unpinned session reached pixels"))
    with pytest.raises(ValueError, match="expected pin"):
        VC.prepare(t.session, t.spec, t.tmp, t.tmp / "audit")
    assert not (t.tmp / "audit").exists()
    assert VC.PUBLISHED["counts"]["paired_lines"] == 4271
    assert VC.PUBLISHED["baseline_model_sha256"].startswith("500e0b84")


def test_clannad_binding_rejects_retried_or_unbound_comparisons(clannad_inputs):
    t = clannad_inputs
    rows = LR.read_journal(t.session / "comparisons.jsonl")
    LR.append_event(t.session / "comparisons.jsonl", {k: v for k, v in rows[0].items()
                                                       if k not in ("sequence", "previous_sha256", "event_sha256")})
    expect = dict(t.expect, comparisons_sha256=R.sha256(t.session / "comparisons.jsonl"))
    with pytest.raises(ValueError, match="retried"):
        VC.bound_session(t.session, expect)


def test_clannad_prepare_and_one_attempt(clannad_inputs):
    t = clannad_inputs
    prepared = VC.prepare(t.session, t.spec, t.tmp, t.tmp / "audit", pairs=["A:cached", "B:cached"], expect=t.expect)
    assert prepared["counts"]["captured"] == 5 and prepared["counts"]["successful_pages"] == 4
    assert prepared["counts"]["excluded_failed_pages"] == 1 and prepared["counts"]["paired_lines"] == 8
    assert prepared["model_forwards"] == 0
    path = t.tmp / "audit" / "prepared.json"
    result = VC.compare(path, authorize(path, VC.KIND, t.tmp))
    R.assert_public(result)
    assert result["published_session"] is False
    a, b = result["arms"]["A"], result["arms"]["B"]
    assert a["cache_agreement"]["paired_lines_identical"] == a["cache_agreement"]["paired_lines"] == 8
    assert "cache_agreement" not in b
    assert a["paired_line_proxy"] == result["cached_baseline"]["paired_line_proxy"]
    assert a["paired_line_proxy"]["reference_chars"] == sum(len(REFS[(i + j) % 6]) for i in range(4) for j in range(2))
    assert set(result["pairs"]) == {"B:A", "A:cached", "B:cached"}
    assert result["pairs"]["A:cached"]["paired_line"]["lines"]["full"]["delta"] == 0.0
    assert result["pairs"]["B:A"]["whole_roi"]["pages"]["full"]["units"] == 4
    assert result["promotion_decision"].startswith("none")
    with pytest.raises(FileExistsError, match="no retries"):
        VC.compare(path, t.tmp / "authorization.json")


def test_clannad_verify_detects_changed_pixels(clannad_inputs):
    t = clannad_inputs
    VC.prepare(t.session, t.spec, t.tmp, t.tmp / "audit", expect=t.expect)
    capture = [r for r in LR.read_journal(t.session / "captures.jsonl") if r["kind"] == "capture"][0]
    png = t.session / capture["lines"][0]["path"]
    pixels = np.asarray(Image.open(png)).copy()
    pixels[0, 0] ^= 1
    Image.fromarray(pixels).save(png)
    with pytest.raises(ValueError, match="corruption"):
        VC.verify(t.tmp / "audit" / "prepared.json", images=True)


def test_clannad_pairs_need_known_arms():
    with pytest.raises(ValueError, match="candidate must be a blob arm"):
        VC.check_pairs(["cached:A"], ["A"])
    assert VC.check_pairs(["A:cached"], ["A"]) == [{"candidate": "A", "baseline": "cached"}]
