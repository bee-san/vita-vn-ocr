"""Fresh-test contracts use procedural text/pixels and stub readers, never a real model."""

import copy
import hashlib
import json
import struct
import sys
import types
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import fresh_native_textbox as fresh  # noqa: E402


def pin(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def cohort(frames=6):
    sources, lines = [], []
    for console, first_vn in (("psp", 1), ("vita", 4)):
        for vn in range(first_vn, first_vn + 3):
            for frame in range(frames):
                sid = f"{console}-v{vn}-{frame:02}"
                lid = sid + "-body"
                dims = fresh.DIMENSIONS[console]
                roi = [12, 100, 400, 80]
                sources.append({
                    "id": sid, "split": "test", "platform": console, "vn": f"v{vn}",
                    "release": f"r{100 + vn}", "session": f"r{100 + vn}", "image": sid + ".png",
                    "url": "https://example.test/" + sid, "sha256": pin(sid), "pixel_sha256": pin(sid + "RGB"),
                    "dims": dims, "base_roi": roi, "line_ids": [lid], "hand_checked": True,
                    "reviewer_kind": "assistant", "complete_eligible_text": True, "uncertain": False,
                    "scope_exclusions": [], "eligibility": dict.fromkeys(fresh.ELIGIBILITY, False),
                    "variants": {v: {"roi": box, "pixel_sha256": pin(sid + v)}
                                 for v, box in fresh.roi_variants(roi, dims).items()},
                })
                lines.append({"id": lid, "source_id": sid, "split": "test", "content_kind": "body",
                              "box": [20, 120, 300, 24], "text": "あ" * 50 + "漢" * 50 + f"{vn}:{frame}",
                              "uncertain": False})
    manifest = {"schema": 1, "split": "test", "purpose": "fresh-native-textbox",
                "original_training_history": "unknown", "sources": sources,
                "exclusions": {"schema": 1, "evidence_file": "exposure.json", "evidence_sha256": pin("evidence"),
                               "vns": ["v900"], "release_ids": ["r900"], "urls": [], "sha256": []}}
    return manifest, {"schema": 1, "split": "test", "lines": lines, "supplementary": []}


def documents(tmp_path, manifest, annotations):
    evidence = {"excluded": {k: manifest["exclusions"][k] for k in ("vns", "release_ids", "urls", "sha256")}}
    (tmp_path / "exposure.json").write_text(json.dumps(evidence))
    manifest["exclusions"]["evidence_sha256"] = fresh.sha256(tmp_path / "exposure.json")
    paths = (tmp_path / "manifest.json", tmp_path / "annotations.json")
    for p, value in zip(paths, (manifest, annotations)):
        p.write_text(json.dumps(value, ensure_ascii=False))
    return paths


def test_valid_cohort_has_native_minima_and_explicit_review_limits():
    m, a = cohort()
    selected, readiness = fresh.checked_cohort(m, a)
    assert readiness["ready"] and len(selected) == 36
    assert all(g["frames"] == 6 and g["body_chars"] >= 600 and g["body_hiragana"] == 300
               for g in readiness["groups"].values())
    assert m["original_training_history"] == "unknown"
    assert all(s["reviewer_kind"] == "assistant" for s in selected)


@pytest.mark.parametrize("mutation", ["top-role", "source-role", "exposed", "unknown-history", "wrong-native",
                                      "family", "variant", "source-hash", "duplicate-source"])
def test_invalid_metadata_is_rejected_before_annotations_images_or_model(tmp_path, monkeypatch, mutation):
    m, a = cohort()
    if mutation == "top-role":
        m["split"] = "dev"
    elif mutation == "source-role":
        m["sources"][-1]["split"] = "dev"
    elif mutation == "exposed":
        m["exclusions"]["vns"].append(m["sources"][-1]["vn"])
    elif mutation == "unknown-history":
        m["original_training_history"] = "proven_clean"
    elif mutation == "wrong-native":
        m["sources"][-1]["dims"] = [960, 540]
    elif mutation == "family":
        m["sources"][-1]["eligibility"]["related_to_used_title"] = True
    elif mutation == "variant":
        m["sources"][-1]["variants"]["expanded-4"]["roi"][0] += 1
    elif mutation == "source-hash":
        m["sources"][-1]["sha256"] = "not-a-hash"
    else:
        m["sources"][-1]["id"] = m["sources"][0]["id"]
    manifest, _ = documents(tmp_path, m, a)
    monkeypatch.setattr(fresh, "image_rgb", lambda _: pytest.fail("invalid metadata reached image decoding"))
    with pytest.raises(ValueError):
        fresh.prepare(manifest, tmp_path / "ANNOTATIONS-MUST-NOT-OPEN", tmp_path / "MODEL-MUST-NOT-OPEN",
                      tmp_path / "LIBRARY-MUST-NOT-OPEN", tmp_path / "audit")
    assert not (tmp_path / "audit").exists()


@pytest.mark.parametrize("mutation", ["line-role", "ruby-main", "unbound", "source-binding", "uncertain", "clipped"])
def test_reference_roles_and_complete_geometry(mutation):
    m, a = cohort()
    line = a["lines"][-1]
    if mutation == "line-role":
        line["split"] = "dev"
    elif mutation == "ruby-main":
        line["content_kind"] = "ruby"
    elif mutation == "unbound":
        m["sources"][-1]["line_ids"] = [m["sources"][0]["line_ids"][0]]
    elif mutation == "source-binding":
        line["source_id"] = "another-source"
    elif mutation == "uncertain":
        line["uncertain"] = True
    else:
        line["box"] = [0, 120, 300, 24]
    with pytest.raises(ValueError):
        fresh.checked_cohort(m, a)


@pytest.mark.parametrize("field", ["frames", "body_chars", "body_hiragana", "titles"])
def test_minima_cannot_be_met_by_undersized_cohorts(field):
    m, a = cohort(4 if field == "frames" else 6)
    if field == "body_chars":
        for line in a["lines"]:
            line["text"] = "あ" * 50 + line["id"]
    elif field == "body_hiragana":
        for line in a["lines"]:
            line["text"] = "漢" * 100 + line["id"]
    elif field == "titles":
        m["sources"] = [s for s in m["sources"] if s["vn"] != "v6"]
        a["lines"] = [line for line in a["lines"] if not line["source_id"].startswith("vita-v6-")]
    with pytest.raises(ValueError, match="ineligible"):
        fresh.checked_cohort(m, a)


def test_repeated_body_and_identical_pixels_are_removed_before_readiness():
    m, a = cohort()
    # One duplicate does not break the six-frame title; it is retained only as an exclusion receipt.
    a["lines"][1]["text"] = a["lines"][0]["text"]
    selected, readiness = fresh.checked_cohort(m, a)
    assert len(selected) == 35 and readiness["groups"]["psp:v1"]["frames"] == 5
    assert readiness["deduplicated_sources"][0]["reason"] == "repeated_body"
    m["sources"][1]["pixel_sha256"] = m["sources"][0]["pixel_sha256"]
    _, readiness = fresh.checked_cohort(m, a)
    assert readiness["deduplicated_sources"][0]["reason"] == "duplicate_rgb"
    a["lines"][2]["text"] = a["lines"][0]["text"]
    with pytest.raises(ValueError, match="ineligible"):
        fresh.checked_cohort(m, a)


def test_repeated_lines_do_not_inflate_distinct_character_quota():
    m, a = cohort()
    for i, source in enumerate(m["sources"]):
        a["lines"][i]["text"] = "あ" * 100 + source["vn"]
        extra = dict(a["lines"][i], id=source["id"] + "-short", text=f"あ{i}", box=[20, 150, 100, 20])
        a["lines"].append(extra)
        source["line_ids"].append(extra["id"])
    with pytest.raises(ValueError, match="body_chars"):
        fresh.checked_cohort(m, a)


def test_rgb_duplicate_with_conflicting_annotations_is_rejected():
    m, a = cohort()
    m["sources"][1]["pixel_sha256"] = m["sources"][0]["pixel_sha256"]
    with pytest.raises(ValueError, match="conflicting"):
        fresh.checked_cohort(m, a)


def test_identical_dialogue_in_distinct_titles_preserves_font_coverage():
    m, a = cohort()
    for frame in range(6):
        a["lines"][6 + frame]["text"] = a["lines"][frame]["text"]
    selected, readiness = fresh.checked_cohort(m, a)
    assert len(selected) == 36 and readiness["deduplicated_sources"] == []
    assert readiness["groups"]["psp:v1"]["body_chars"] == readiness["groups"]["psp:v2"]["body_chars"]
    assert readiness["groups"]["psp:v2"]["body_hiragana"] == 300


@pytest.mark.parametrize("conflict", ["title", "box"])
def test_identical_pixels_require_consistent_title_and_ink_geometry(conflict):
    m, a = cohort()
    index = 6 if conflict == "title" else 1
    m["sources"][index]["pixel_sha256"] = m["sources"][0]["pixel_sha256"]
    a["lines"][index]["text"] = a["lines"][0]["text"]
    if conflict == "box":
        a["lines"][index]["box"][0] += 1
    with pytest.raises(ValueError, match="conflicting"):
        fresh.checked_cohort(m, a)


def test_exposure_evidence_hash_and_lists_are_bound(tmp_path):
    m, a = cohort()
    paths = documents(tmp_path, m, a)
    fresh.checked_inputs(*paths)
    (tmp_path / "exposure.json").write_text('{"excluded":{}}')
    with pytest.raises(ValueError, match="evidence changed"):
        fresh.checked_inputs(*paths)
    m["exclusions"]["evidence_sha256"] = fresh.sha256(tmp_path / "exposure.json")
    paths[0].write_text(json.dumps(m))
    with pytest.raises(ValueError, match="exclusion lists"):
        fresh.checked_inputs(*paths)


def generated_image_source(tmp_path):
    m, a = cohort()
    source = dict(m["sources"][0])
    rgb = np.full((272, 480, 3), 30, np.uint8)
    rgb[120:144, 24:324:4] = 240
    path = tmp_path / "procedural.png"
    Image.fromarray(rgb).save(path)
    source.update(_image=str(path), sha256=fresh.sha256(path),
                  pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                  _lines=[dict(a["lines"][0], styles=[])], _reference_sha256=pin("synthetic-reference"))
    for variant in source["variants"].values():
        x, y, w, h = variant["roi"]
        variant["pixel_sha256"] = hashlib.sha256(np.ascontiguousarray(rgb[y:y + h, x:x + w]).tobytes()).hexdigest()
    return source


def test_native_image_rgb_and_roi_pins(tmp_path):
    source = generated_image_source(tmp_path)
    assert fresh.image_rgb(source).shape == (272, 480, 3)
    source["variants"]["expanded-8"]["pixel_sha256"] = pin("different-ROI")
    with pytest.raises(ValueError, match="ROI pixel"):
        fresh.image_rgb(source)


def generated_blob():
    data = bytearray(657664)
    data[:4] = b"VOCR"
    struct.pack_into("<HHII", data, 4, 1, 64, len(data), 1)
    struct.pack_into("<H", data, 18, 1024)
    struct.pack_into("<I", data, 24, 3501)
    struct.pack_into("<I", data, 40, zlib.crc32(data[64:]))
    return data


@pytest.mark.parametrize("offset,value", [(None, None), (100, 1), (24, 0), (19, 0), (12, 0)])
def test_static_blob_contract_checks_crc_width_classes_and_mode(offset, value):
    data = generated_blob()
    if offset is None:
        fresh.validate_blob_bytes(data)
    else:
        data[offset] = value
        with pytest.raises(ValueError):
            fresh.validate_blob_bytes(data)


def test_native_reader_explicit_library_crc_and_memory(monkeypatch):
    calls = []

    class FakeVocr:
        blob_bytes, arena_bytes, n_classes = 657664, 124832, 3501

        def __init__(self, path, **kw):
            calls.append((path, kw))

    monkeypatch.setitem(sys.modules, "pyvocr", types.SimpleNamespace(
        Vocr=FakeVocr, load_lib=lambda path: "explicit:" + path))
    monkeypatch.setattr(fresh, "sha256", lambda path: fresh.MODEL_SHA if path == "model" else fresh.LIBRARY_SHA)
    reader = fresh.NativeReader({"model": "model", "library": "library"})
    assert calls == [("model", {"mode": "w8i", "max_width": 1024, "lib": "explicit:library", "check_crc": True})]
    reader.close()
    FakeVocr.arena_bytes += 1
    with pytest.raises(ValueError, match="memory/class"):
        fresh.NativeReader({"model": "model", "library": "library"})


def test_transcript_scoring_charges_all_insertions_to_body_and_keeps_kana_distinct():
    lines = [{"text": "名", "content_kind": "name", "styles": ["name"]},
             {"text": "あい", "content_kind": "body", "styles": []}]
    score = fresh.transcript_score(lines, ["X名あいう"])
    assert score["all"]["full"] == {"chars": 3, "errors": 2}
    assert score["body"]["full"] == {"chars": 2, "errors": 2}
    assert score["body"]["hira"] == {"chars": 2, "errors": 1}
    line = [{"text": "つば未", "content_kind": "body", "styles": []}]
    score = fresh.transcript_score(line, ["っは末"])
    assert score["all"]["full"]["errors"] == 3 and score["all"]["hira"]["errors"] == 2
    dots = [{"text": ".", "content_kind": "body", "styles": []}] * 2
    dot_score = fresh.transcript_score(dots, [".."])
    assert dot_score["all"]["full"]["errors"] == 2


def passing_scores():
    result = {side: {v: {} for v in fresh.VARIANTS} for side in fresh.MODES}
    for side, variants in result.items():
        for groups in variants.values():
            for i in range(6):
                console = "psp" if i < 3 else "vita"
                vn = f"v{i + 1}"
                errors = 10 if side == "C4" else 20
                groups[console + ":" + vn] = {
                    "platform": console, "vn": vn, "frames": 6, "reference_sha256": pin(vn),
                    "detections": 12, "blank_outputs": 0, "zero_detection_frames": 0,
                    "all": {"full": {"chars": 600, "errors": errors}, "hira": {"chars": 230, "errors": errors}},
                    "body": {"full": {"chars": 500, "errors": errors}, "hira": {"chars": 200, "errors": errors}},
                }
    return result


def test_fixed_gate_pass_and_exact_fraction_boundary():
    result = passing_scores()
    assert fresh.qualify(result)["passed"]
    for g in result["C4"]["base"].values():
        g["all"]["full"]["errors"] = 18
    assert fresh.qualify(result)["passed"]  # 108 == 9/10 * 120
    result["C4"]["base"]["psp:v1"]["all"]["full"]["errors"] += 1
    decision = fresh.qualify(result)
    assert not decision["passed"] and any("10 percent" in s for s in decision["reasons"])


@pytest.mark.parametrize("mutation", ["title-regression", "pooled-regression", "breadth", "balanced", "zero-anchor",
                                      "reference", "variant", "counter"])
def test_strong_fresh_gate_rejects_each_failure(mutation):
    result = passing_scores()
    if mutation == "title-regression":
        result["C4"]["expanded-8"]["vita:v6"]["all"]["full"]["errors"] = 21
    elif mutation == "pooled-regression":
        for g in result["pure_C3"]["expanded-4"].values():
            g["body"]["full"]["errors"] = 0
    elif mutation == "breadth":
        for title in ("psp:v1", "psp:v2"):
            result["C4"]["base"][title]["body"]["hira"]["errors"] = 20
    elif mutation == "balanced":
        for g in result["C4"]["base"].values():
            g["body"]["hira"]["errors"] = 20
    elif mutation == "zero-anchor":
        for side in ("pure_C3", "C4"):
            for g in result[side]["base"].values():
                g["all"]["full"]["errors"] = 0
    elif mutation == "reference":
        result["C4"]["base"]["psp:v1"]["reference_sha256"] = pin("other")
    elif mutation == "variant":
        del result["C4"]["expanded-8"]
    else:
        result["C4"]["base"]["psp:v1"]["body"]["hira"]["errors"] = -1
    assert not fresh.qualify(result)["passed"]


class FakeFinder:
    def __init__(self, mode, failure=None):
        self.mode, self.failure = mode, failure
        self.last_dark, self.last_truncated = [], False

    def __call__(self, image):
        self.last_truncated = self.failure == "truncated"
        if self.failure == "exception":
            raise RuntimeError("procedural finder failure")
        if self.failure == "none":
            return None
        if self.mode == 0:
            self.last_dark = []
            return []
        self.last_dark = [0, 1]
        return [(0, 0, 12, 12), (20, 0, 12, 12)]


def test_zero_detections_blanks_all_crops_polarities_and_insertions_are_retained(tmp_path):
    source = generated_image_source(tmp_path)
    source["_lines"] = [{"text": "あ", "content_kind": "body", "styles": []}]
    calls, events = [], []

    class Reader:
        def __init__(self, prepared):
            pass

        def text(self, crop, dark):
            calls.append(dark)
            return "あ余" if dark else ""

        def close(self):
            calls.append("closed")

    result, forwards = fresh.score_family({}, [source], Reader, FakeFinder, events.append)
    assert forwards == {"pure_C3": 6, "C4": 6} and calls[-1] == "closed"
    assert calls[:-1] == [False, True] * 6
    for variant in fresh.VARIANTS:
        zero = result["original_GRAD"][variant]["psp:v1"]
        assert zero["zero_detection_frames"] == 1 and zero["all"]["full"]["errors"] == 1
        got = result["C4"][variant]["psp:v1"]
        assert got["detections"] == 2 and got["blank_outputs"] == 1 and got["body"]["full"]["errors"] == 1
    assert len(events) == 9 and all(e["status"] == "ok" for e in events)


@pytest.mark.parametrize("failure", ["none", "truncated", "exception", "reader", "non-string"])
def test_failed_finder_or_reader_is_visible_and_never_coerced_to_success(tmp_path, failure):
    source = generated_image_source(tmp_path)
    events, closed = [], []

    class Reader:
        def __init__(self, prepared):
            pass

        def text(self, image, dark):
            if failure == "reader":
                raise RuntimeError("procedural reader failure")
            return None

        def close(self):
            closed.append(True)

    def finder(mode):
        return FakeFinder(mode, failure if failure in ("none", "truncated", "exception") else None)

    with pytest.raises(RuntimeError):
        fresh.score_family({}, [source], Reader, finder, events.append)
    assert closed == [True] and events[-1]["status"] == "failed"
    assert any(events[-1].get(k) for k in ("finder_error", "truncated", "reader_error"))


@pytest.mark.parametrize("bad", ["box", "polarity"])
def test_invalid_finder_geometry_or_polarity_records_failure_before_forward(tmp_path, bad):
    events = []
    source = generated_image_source(tmp_path)

    class BadFinder(FakeFinder):
        def __call__(self, image):
            self.last_dark = [9 if bad == "polarity" else 0]
            return [(-1 if bad == "box" else 0, 0, 12, 12)]

    class Reader:
        def __init__(self, prepared):
            pass

        def text(self, crop, dark):
            pytest.fail("invalid geometry reached a model forward")

        def close(self):
            pass

    with pytest.raises((RuntimeError, ValueError)):
        fresh.score_family({}, [source], Reader, BadFinder, events.append)
    assert events[-1]["finder_error"] and events[-1]["status"] == "failed"


def test_wrong_authorization_and_existing_attempt_block_before_verification(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared.json"
    fresh.freeze(prepared, {"protocol_sha256": pin("protocol"), "source_fingerprint": pin("sources")})
    auth = tmp_path / "authorization.json"
    fresh.freeze(auth, {"authorization": "not-approved"})
    monkeypatch.setattr(fresh, "verify", lambda *a, **k: pytest.fail("unauthorized execution reached verification"))
    with pytest.raises(ValueError, match="authorization"):
        fresh.compare(prepared, auth)
    auth.unlink()
    fresh.freeze(auth, fresh.expected_authorization(prepared))
    fresh.freeze(tmp_path / "attempt.json", {"status": "previous-attempt"})
    with pytest.raises(FileExistsError, match="no retries"):
        fresh.compare(prepared, auth)
    assert not (tmp_path / "private-predictions.jsonl").exists()


def test_atomic_snapshots_and_changed_prepared_authorization(tmp_path):
    path = tmp_path / "prepared.json"
    fresh.freeze(path, {"protocol_sha256": pin("protocol"), "source_fingerprint": pin("sources")})
    auth = fresh.expected_authorization(path)
    with pytest.raises(FileExistsError):
        fresh.freeze(path, {})
    old = copy.deepcopy(fresh.read(path))
    old["source_fingerprint"] = pin("changed-cohort")
    path.write_text(json.dumps(old))
    assert auth != fresh.expected_authorization(path)


def test_no_prepare_forward_after_valid_metadata_and_quotas(tmp_path, monkeypatch):
    m, a = cohort()
    manifest, annotations = documents(tmp_path, m, a)
    visited = []

    def image_boundary(source):
        visited.append(source["id"])
        raise RuntimeError("procedural image boundary")

    monkeypatch.setattr(fresh, "image_rgb", image_boundary)
    monkeypatch.setattr(fresh, "NativeReader", lambda _: pytest.fail("prepare constructed a model"))
    with pytest.raises(RuntimeError, match="image boundary"):
        fresh.prepare(manifest, annotations, tmp_path / "unopened-model", tmp_path / "unopened-library",
                      tmp_path / "audit")
    assert len(visited) == 1 and not (tmp_path / "audit").exists()


def test_verify_changed_protocol_or_reference_before_model_access(tmp_path):
    m, a = cohort()
    manifest, annotations = documents(tmp_path, m, a)
    _, sources, ready = fresh.checked_inputs(manifest, annotations)
    audit = tmp_path / "audit"
    audit.mkdir()
    fresh.freeze(audit / "protocol.json", fresh.PROTOCOL)
    prepared = {
        "schema": 1, "kind": "fresh-native-textbox-prepared", "directory": str(audit),
        "manifest": str(manifest), "annotations": str(annotations),
        "protocol_sha256": fresh.sha256(audit / "protocol.json"), "code_sha256": fresh.code_identity(),
        "manifest_sha256": fresh.sha256(manifest), "annotations_sha256": fresh.sha256(annotations),
        "readiness": ready,
        "source_fingerprint": fresh.digest([{k: v for k, v in s.items() if k != "_lines"} for s in sources]),
        "reference_fingerprint": fresh.digest([s["_reference_sha256"] for s in sources]),
        "exposure_evidence_sha256": m["exclusions"]["evidence_sha256"],
        "model": str(tmp_path / "MODEL-MUST-NOT-OPEN"), "library": str(tmp_path / "LIBRARY-MUST-NOT-OPEN"),
    }
    fresh.freeze(audit / "prepared.json", prepared)
    changed = copy.deepcopy(fresh.PROTOCOL)
    changed["variants"]["expanded-4"] = 2
    (audit / "protocol.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="frozen protocol"):
        fresh.verify(audit / "prepared.json")
    protocol_text = json.dumps(fresh.PROTOCOL, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    (audit / "protocol.json").write_text(protocol_text)
    a["lines"][0]["text"] += "あ"
    annotations.write_text(json.dumps(a))
    with pytest.raises(ValueError, match="frozen protocol"):
        fresh.verify(audit / "prepared.json")


def test_failed_attempt_has_immutable_receipt_and_cannot_run_again(tmp_path, monkeypatch):
    # Isolate one-shot orchestration. Real metadata/reference validators are exercised above;
    # this stub represents their successful return, not a replacement test-role validator.
    prepared = tmp_path / "prepared.json"
    fresh.freeze(prepared, {"protocol_sha256": pin("protocol"), "source_fingerprint": pin("sources")})
    auth = tmp_path / "authorization.json"
    fresh.freeze(auth, fresh.expected_authorization(prepared))
    verified = []

    def verified_fixture(path, images=False):
        verified.append(images)
        return {}, []

    def failure(*args, **kwargs):
        raise RuntimeError("procedural execution failure")

    monkeypatch.setattr(fresh, "verify", verified_fixture)
    monkeypatch.setattr(fresh, "score_family", failure)
    with pytest.raises(RuntimeError, match="execution failure"):
        fresh.compare(prepared, auth)
    receipt = fresh.read(tmp_path / "failure.json")
    assert receipt["status"] == "failed" and receipt["retry_allowed"] is False
    assert receipt["partial_predictions_retained"] and receipt["error_type"] == "RuntimeError"
    before = fresh.sha256(tmp_path / "attempt.json")
    with pytest.raises(FileExistsError, match="no retries"):
        fresh.compare(prepared, auth)
    assert fresh.sha256(tmp_path / "attempt.json") == before and verified == [False, True]
