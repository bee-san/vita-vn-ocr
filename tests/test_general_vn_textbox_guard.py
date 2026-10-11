"""Generated metadata/pixels and fake transcripts; no real VN or model scoring."""

import copy
import fcntl
import json
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
import pytest

import general_vn_textbox_guard as T
from test_general_vn_select import package  # noqa: F401
from test_general_vn_select import prepare as prepare_lines
from test_linefind import lines_image


def plan_for(generated):
    sources = []
    for row in generated["rows"]:
        source = {key: row[key] for key in ("id", "split", "platform", "vn", "session", "release", "dims", "sha256")}
        source.update(image=str(generated["root"] / row["image"]), line_ids=[row["id"]], base_roi=[4, 4, 44, 28],
                      hand_checked=True, complete_eligible_text=True, uncertain=False, scope_exclusions=[])
        sources.append(source)
    return {"schema": 1, "split": "dev", "sources": sources}


@pytest.fixture
def roi_package(request, monkeypatch):
    generated = request.getfixturevalue("package")
    for row in generated["rows"]:
        row["content_kind"] = "body"
    generated["manifest"].write_text("".join(json.dumps(r) + "\n" for r in generated["rows"]))
    line_identity = prepare_lines(generated)
    monkeypatch.setattr(T, "BASELINE_SHA", T.sha256(generated["baseline"]))
    monkeypatch.setattr(T.selector, "validate_blob", lambda *args: None)
    plan = generated["root"] / "roi-plan.json"
    plan.write_text(json.dumps(plan_for(generated)))
    manifest = generated["root"] / "roi-manifest.json"
    T.materialize_plan(generated["out"], plan, manifest)
    return {**generated, "line_identity": line_identity, "roi_plan": plan, "roi_manifest": manifest,
            "guard_out": generated["root"] / "guard.json", "result_out": generated["root"] / "result.json"}


def prepare_guard(p):
    return T.prepare(p["out"], p["roi_manifest"], p["guard_out"], p["result_out"], compiler=str(p["compiler"]))


def line(text, kind="body"):
    return {"text": text, "content_kind": kind, "styles": ["name"] if kind == "name" else []}


def test_full_alignment_preserves_small_kana_and_does_not_charge_missed_names_to_body():
    rows = [line("あい", "name"), line("っが"), line("うえ")]
    missing_name = T.transcript_score(rows, "っがうえ")
    assert missing_name["all"]["full"] == {"chars": 6, "errors": 2}
    assert missing_name["all"]["hira"] == {"chars": 6, "errors": 2}
    assert missing_name["body"]["full"] == {"chars": 4, "errors": 0}
    missing_body = T.transcript_score(rows, "あいっが")
    assert missing_body["body"]["full"] == {"chars": 4, "errors": 2}
    assert missing_body["body"]["hira"] == {"chars": 4, "errors": 2}
    small_kana = T.transcript_score(rows, "あいつがうえ")
    assert small_kana["body"]["hira"]["errors"] == 1


def test_extra_ui_and_unmatched_detections_are_charged_to_body_even_before_names():
    rows = [line("あい", "name"), line("っが")]
    scored = T.transcript_score(rows, "めにゅーあいっが設定")
    assert scored["all"]["full"] == {"chars": 4, "errors": 6}
    assert scored["body"]["full"] == {"chars": 2, "errors": 6}
    assert scored["body"]["hira"] == {"chars": 2, "errors": 3}
    assert T.transcript_score(rows, "")['body']['full']['errors'] == 2


def test_full_alignment_ties_are_frozen_and_not_optimized_for_body_score():
    # Diagonal tie gives the retained あ to the body, so the missed glyph belongs to the name.
    result = T.transcript_score([line("あ", "name"), line("あ")], "あ")
    assert result["all"]["full"]["errors"] == 1
    assert result["body"]["full"]["errors"] == 0
    assert T.transcript_score([line("っが！")], "つか!")["body"]["hira"]["errors"] == 2


def test_local_alignment_has_the_existing_deterministic_ties_after_normalization():
    from metrics import align_text, norm

    texts = ["", "っが！", "…", "．．"] + ["".join(chars) for n in (1, 2)
                                      for chars in product("あっ.!", repeat=n)]
    for reference, prediction in product(texts, repeat=2):
        assert T.align_normalized(norm(reference), norm(prediction)) == align_text(reference, prediction)


@pytest.mark.parametrize("punctuation", [".", "・"])
def test_punctuation_cannot_collapse_across_reference_or_detection_boundaries(punctuation):
    rows = [line(punctuation, "name"), line(punctuation)]
    exact = T.transcript_score(rows, [punctuation, punctuation])
    assert exact["all"]["full"] == {"chars": 2, "errors": 0}
    assert exact["body"]["full"] == {"chars": 1, "errors": 0}
    # Within one detection the existing normalization still applies, with fixed body attribution.
    merged_detection = T.transcript_score(rows, [punctuation * 2])
    assert merged_detection["all"]["full"] == {"chars": 2, "errors": 2}
    assert merged_detection["body"]["full"] == {"chars": 1, "errors": 1}


def test_ellipsis_normalization_is_per_line_and_per_detection():
    exact = T.transcript_score([line("...", "name"), line("．．．っ")], ["…", "・・・っ"])
    assert exact["all"]["full"] == {"chars": 3, "errors": 0}
    assert exact["body"]["full"] == {"chars": 2, "errors": 0}
    assert T.transcript_score([line("...", "name"), line("．．．っ")], ["…", "・・・つ"])["body"]["hira"] == {
        "chars": 1, "errors": 1}


@pytest.mark.parametrize("role", ["train", "test", None])
@pytest.mark.parametrize("where", ["manifest", "source"])
def test_role_rejection_precedes_source_bytes_pixels_and_models(tmp_path, monkeypatch, role, where):
    doc = {"schema": 1, "split": "dev", "sources": [{"split": "dev"}]}
    if where == "manifest":
        doc["split"] = role
    else:
        doc["sources"][0]["split"] = role
    manifest = tmp_path / "roles.json"
    manifest.write_text(json.dumps(doc))
    monkeypatch.setattr(T.selector, "verify_lock", lambda *a: pytest.fail("wrong role reached line sources"))
    monkeypatch.setattr(T.Image, "open", lambda *a: pytest.fail("wrong role reached pixels"))
    monkeypatch.setattr(T, "NativeReader", lambda *a: pytest.fail("wrong role reached model"))
    with pytest.raises(ValueError, match="development-only"):
        T.prepare(tmp_path / "never-read.json", manifest, tmp_path / "out.json", tmp_path / "result.json")
    assert not (tmp_path / "out.json").exists()


def test_variants_are_fixed_native_clamped_and_do_not_mutate_original():
    base = [1, 2, 10, 11]
    assert T.roi_variants(base, [20, 20]) == {
        "base": base, "expanded-4": [0, 0, 15, 17], "expanded-8": [0, 0, 19, 20]}
    assert base == [1, 2, 10, 11]
    for invalid in ([-1, 0, 10, 10], [0, 0, 21, 10], [True, 0, 10, 10], [1.0, 0, 10, 10]):
        with pytest.raises(ValueError, match="integers|bounds"):
            T.roi_variants(invalid, [20, 20])


@pytest.mark.parametrize("change", ["clipped", "variant", "missing", "reordered_duplicate", "incomplete", "game"])
def test_reviewed_metadata_scope_cannot_be_weakened(roi_package, monkeypatch, change):
    p = roi_package
    doc = json.loads(p["roi_manifest"].read_text())
    source = doc["sources"][0]
    if change == "clipped":
        source["base_roi"] = [10, 10, 10, 10]
        source["variants"] = {n: {"roi": r, "pixel_sha256": "1" * 64}
                              for n, r in T.roi_variants(source["base_roi"], source["dims"]).items()}
    elif change == "variant":
        source["variants"]["expanded-4"]["roi"][0] += 1
    elif change == "missing":
        doc["sources"].pop()
    elif change == "reordered_duplicate":
        source["line_ids"] *= 2
    elif change == "incomplete":
        source["complete_eligible_text"] = False
    else:
        source["vn"] = "v99"
    p["roi_manifest"].write_text(json.dumps(doc))
    monkeypatch.setattr(T.Image, "open", lambda *a: pytest.fail("invalid metadata reached pixels"))
    with pytest.raises(ValueError, match="clips|fixed symmetric|every ready|unique bound|complete|provenance"):
        prepare_guard(p)
    assert not p["guard_out"].exists()


def test_prepare_freezes_actual_builds_without_pixels_and_refuses_overwrite(roi_package, monkeypatch):
    p = roi_package
    monkeypatch.setattr(T.Image, "open", lambda *a: pytest.fail("preparation decoded pixels"))
    monkeypatch.setattr(T, "NativeReader", lambda *a: pytest.fail("preparation executed model"))
    lock = prepare_guard(p)
    assert lock["model_forwards"] == 0 and lock["readiness"]["ready"]
    assert lock["protocol"] == T.PROTOCOL
    assert lock["finder"]["binary_sha256"] == T.sha256(lock["finder"]["binary"])
    assert lock["runtime"]["library_sha256"] == T.sha256(lock["runtime"]["library"])
    assert "-DVOCR_NO_NEON" in lock["finder"]["argv"]
    assert T.verify_prepared(p["guard_out"])[0] == lock
    raw = p["guard_out"].read_bytes()
    with pytest.raises(FileExistsError):
        prepare_guard(p)
    assert raw == p["guard_out"].read_bytes()


@pytest.mark.parametrize("change", ["image", "manifest", "code", "environment", "finder", "compiler"])
def test_frozen_input_or_execution_drift_refuses_before_pixels(roi_package, monkeypatch, change):
    p = roi_package
    lock = prepare_guard(p)
    if change == "image":
        image = Path(json.loads(p["roi_manifest"].read_text())["sources"][0]["image"])
        image.write_bytes(b"corrupt")
    elif change == "manifest":
        p["roi_manifest"].write_text(p["roi_manifest"].read_text() + " ")
    elif change == "code":
        original = T.sha256
        monkeypatch.setattr(T, "sha256", lambda path: "0" * 64 if str(path).endswith("textbox_guard.py")
                            else original(path))
    elif change == "environment":
        monkeypatch.setattr(T.selector, "environment_identity", lambda: {})
    elif change == "finder":
        Path(lock["finder"]["binary"]).write_bytes(b"stale finder")
    else:
        p["compiler"].write_text(p["compiler"].read_text() + "\n# drift\n")
    monkeypatch.setattr(T.Image, "open", lambda *a: pytest.fail("drift reached decoded pixels"))
    with pytest.raises(ValueError, match="changed|mismatch"):
        T.verify_prepared(p["guard_out"])


def test_changed_finder_build_cannot_hide_behind_old_cached_binary(roi_package):
    p = roi_package
    lock = prepare_guard(p)
    p["variation"].write_bytes(b"different compiler output")
    # Existing original bytes still match, but a deterministic rebuild must also match.
    T.verify_finder(lock["finder"])
    with pytest.raises(ValueError, match="rebuild differs"):
        T.build_finder(lock["finder"]["directory"], p["compiler"], expected=lock["finder"])
    assert T.sha256(lock["finder"]["binary"]) == lock["finder"]["binary_sha256"]


def test_decoded_source_and_roi_hashes_are_verified_before_forward(roi_package):
    p = roi_package
    _, rows = T.selector.verify_lock(p["out"])
    sources = T.checked_sources(p["roi_manifest"], rows)
    bad = copy.deepcopy(sources)
    bad[-1]["variants"]["expanded-8"]["pixel_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="ROI pixel identity"):
        T.load_cases(bad)
    bad = copy.deepcopy(sources)
    bad[-1]["pixel_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="decoded RGB"):
        T.load_cases(bad)


def test_source_roi_detections_are_reused_for_both_models_and_never_filtered_by_gt():
    rgba = np.zeros((24, 40, 4), np.uint8)
    rgba[..., 0] = np.arange(40, dtype=np.uint8)
    rgba[..., 3] = 255
    source = {"platform": "psp", "vn": "v1", "_lines": [line("あい")]}
    calls = []

    class GeneratedFinder:
        last_dark = [1, 0]
        last_truncated = False

        def __call__(self, image):
            calls.append(image.copy())
            return [(3, 2, 5, 6), (20, 10, 7, 8)]

    cases = T.find_crops([{"rgba": rgba, "source": source, "variant": "base"}], GeneratedFinder())
    observed = []

    class GeneratedReader:
        def __init__(self, *args):
            self.n = 0

        def text(self, crop, dark):
            observed.append((crop.copy(), dark))
            self.n += 1
            return "あい" if self.n == 1 else "余計"

        def close(self):
            pass

    first = T.score_cases("generated", cases, {}, GeneratedReader)
    second = T.score_cases("generated2", cases, {}, GeneratedReader)
    assert len(calls) == 1 and first == second
    assert first["base"]["psp:v1"]["body"]["full"]["errors"] == 2
    assert [x[1] for x in observed] == [True, False, True, False]
    assert np.array_equal(observed[0][0], rgba[2:8, 3:8])
    assert np.array_equal(observed[2][0], observed[0][0])


@pytest.mark.parametrize("failure", ["missing", "truncated", "badbox", "polarity", "reader"])
def test_engine_errors_abort_and_are_not_blank_success(failure):
    class GeneratedFinder:
        last_dark = [2 if failure == "polarity" else 0]
        last_truncated = failure == "truncated"

        def __call__(self, image):
            return None if failure == "missing" else [(99 if failure == "badbox" else 0, 0, 4, 4)]

    cases = [{"rgba": np.zeros((8, 8, 4), np.uint8), "source": {
        "platform": "psp", "vn": "v1", "_lines": [line("あ")]}, "variant": "base"}]
    if failure != "reader":
        with pytest.raises((ValueError, RuntimeError)):
            T.find_crops(cases, GeneratedFinder())
    else:
        T.find_crops(cases, GeneratedFinder())

        class FailedReader:
            def __init__(self, *args):
                pass

            def text(self, *args):
                raise RuntimeError("generated inference error")

            def close(self):
                pass

        with pytest.raises(RuntimeError, match="inference error"):
            T.score_cases("generated", cases, {}, FailedReader)


def aggregate(errors=10, chars=100):
    return {variant: {f"{platform}:{vn}": {"platform": platform, "vn": vn, "frames": 5, "detections": 10,
                "reference_sha256": "a" * 64,
                **{part: {script: {"chars": chars, "errors": errors} for script in ("full", "hira")}
                   for part in ("all", "body")}}
             for platform in ("psp", "vita") for vn in ("v1", "v2", "v3")} for variant in T.MARGINS}


@pytest.mark.parametrize("part,script", [("all", "full"), ("all", "hira"), ("body", "full"), ("body", "hira")])
def test_one_vn_one_variant_regression_rejects_despite_every_other_gain(part, script):
    baseline, candidate = aggregate(), aggregate(errors=1)
    candidate["expanded-8"]["vita:v3"][part][script]["errors"] = 11
    result = T.qualify(baseline, candidate)
    assert not result["passed"] and any("expanded-8/vita:v3" in r for r in result["reasons"])
    assert T.qualify(baseline, aggregate(errors=9))["passed"]


def test_each_variant_requires_strict_body_gain_and_two_distinct_vns():
    baseline, candidate = aggregate(), aggregate(errors=9)
    candidate["base"] = copy.deepcopy(baseline["base"])
    assert not T.qualify(baseline, candidate)["passed"]
    candidate = copy.deepcopy(baseline)
    for groups in candidate.values():
        for platform in ("psp", "vita"):
            for part in ("all", "body"):
                groups[platform + ":v1"][part]["hira"]["errors"] -= 1
    result = T.qualify(baseline, candidate)
    assert not result["passed"] and result["gain_vns"] == ["v1"]


def test_gains_cannot_be_pooled_across_roi_variants():
    baseline = aggregate()
    candidate = copy.deepcopy(baseline)
    expected = dict(zip(T.MARGINS, (["v1"], ["v2"], ["v3"])))
    for variant, vns in expected.items():
        for platform in ("psp", "vita"):
            for part in ("all", "body"):
                candidate[variant][f"{platform}:{vns[0]}"][part]["hira"]["errors"] -= 1
        for part in ("all", "body"):
            assert T.balanced_rate(candidate[variant], part) < T.balanced_rate(baseline[variant], part)
    result = T.qualify(baseline, candidate)
    assert not result["passed"]
    assert result["gain_vns"] == ["v1", "v2", "v3"]
    assert result["gain_vns_by_variant"] == expected
    assert all(any(r.startswith(variant + ": fewer than two") for r in result["reasons"])
               for variant in T.MARGINS)


def test_balanced_ranking_uses_exact_fractions_not_pixel_or_character_weighting():
    groups = aggregate()["base"]
    groups = {k: v for k, v in groups.items() if k in ("psp:v1", "psp:v2", "vita:v3")}
    groups["vita:v3"]["body"]["hira"] = {"chars": 100000, "errors": 1}
    assert T.balanced_rate(groups, "body") == Fraction(1, 20) + Fraction(1, 200000)


def selected_fixture(p, lock):
    candidate = p["root"] / "generated-chosen.vocr"
    candidate.write_bytes(b"generated selected export; never a model")
    selection = {"schema": 1, "protocol_lock_sha256": T.sha256(p["out"]),
                 "strict_general_dev_improvement": True, "test_or_google_evaluated": False,
                 "runtime_promotion": False, "protocol": T.selector.PROTOCOL,
                 "environment": p["line_identity"]["environment"], "runtime": p["line_identity"]["runtime"],
                 "baseline": {"blob_sha256": lock["baseline_sha256"]},
                 "chosen": {"blob": str(candidate), "blob_sha256": T.sha256(candidate)},
                 "candidates": [{"blob": str(candidate), "blob_sha256": T.sha256(candidate),
                                 "eligible": True, "status": "scored"}]}
    path = Path(p["line_identity"]["selection_out"])
    path.write_text(json.dumps(selection))
    return candidate, path, selection


def test_single_chosen_export_binding_cannot_reopen_selection_or_accept_test_scores(roi_package):
    p = roi_package
    lock = prepare_guard(p)
    candidate, selection_path, selection = selected_fixture(p, lock)
    binding_path = Path(lock["candidate_out"])
    binding = T.bind_selection(p["guard_out"], selection_path, binding_path)
    assert T.verify_binding(binding_path)[0] == binding
    with pytest.raises(FileExistsError):
        T.bind_selection(p["guard_out"], selection_path, binding_path)
    candidate.write_bytes(b"changed selected export")
    with pytest.raises(ValueError, match="changed"):
        T.verify_binding(binding_path)
    selection["test_or_google_evaluated"] = True
    selection_path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="preregistered destination"):
        T.bind_selection(p["guard_out"], selection_path, p["root"] / "another.json")


def test_real_c_finder_reads_generated_rgba_and_preserves_crop_and_polarity(tmp_path):
    identity = T.build_finder(tmp_path / "private-native-build")
    T.verify_finder(identity)
    for ink, paper, dark in ((30, 230, True), (240, 30, False)):
        rgb = lines_image(ink, paper)
        rgba = np.concatenate([rgb, np.full((*rgb.shape[:2], 1), 255, np.uint8)], axis=2)
        finder = T.Finder("grad", binary=identity["binary"])
        cases = T.find_crops([{"rgba": rgba}], finder)
        assert len(cases[0]["crops"]) == 2
        assert all(polarity is dark for _, polarity in cases[0]["crops"])
        for (crop, _), detection in zip(cases[0]["crops"], cases[0]["detections"]):
            x, y, w, h = detection["box"]
            assert np.array_equal(crop, rgba[y:y + h, x:x + w])
        assert not finder.last_truncated


def test_generated_end_to_end_comparison_is_fixed_and_publishes_only_aggregates(roi_package, monkeypatch):
    p = roi_package
    lock = prepare_guard(p)
    candidate, selection_path, _ = selected_fixture(p, lock)
    binding_path = Path(lock["candidate_out"])
    T.bind_selection(p["guard_out"], selection_path, binding_path)

    class GeneratedFinder:
        last_dark = [0]
        last_truncated = False

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, rgba):
            return [(0, 0, 2, 2)]

    observed = []

    class GeneratedReader:
        def __init__(self, blob, runtime):
            self.is_candidate = Path(blob) == candidate

        def text(self, rgba, dark):
            observed.append((self.is_candidate, rgba.tobytes(), dark))
            return ("あ" if self.is_candidate else "い") + "あ" * 39 + "漢" * 60

        def close(self):
            pass

    monkeypatch.setattr(T, "Finder", GeneratedFinder)
    monkeypatch.setattr(T, "NativeReader", GeneratedReader)
    result = T.compare(binding_path, p["result_out"])
    assert result["decision"]["passed"] and result["decision"]["gain_vns"] == ["v1", "v2", "v3"]
    assert result["successful_model_forwards"] == {"baseline": 90, "candidate": 90}
    assert len(result["cases"]) == 90
    assert all(not got[0] for got in observed[:90]) and all(got[0] for got in observed[90:])
    assert [got[1:] for got in observed[:90]] == [got[1:] for got in observed[90:]]
    raw = p["result_out"].read_text()
    assert "あ" not in raw and "漢" not in raw and '"pred"' not in raw and '"text"' not in raw
    with pytest.raises(FileExistsError):
        T.compare(binding_path, p["result_out"])
    assert len(observed) == 180


def test_comparison_has_exclusive_writer_before_any_pixels_or_forwards(roi_package, monkeypatch):
    p = roi_package
    lock = prepare_guard(p)
    _, selection_path, _ = selected_fixture(p, lock)
    binding_path = Path(lock["candidate_out"])
    T.bind_selection(p["guard_out"], selection_path, binding_path)
    writer_path = p["result_out"].with_name(p["result_out"].name + ".writer.lock")
    monkeypatch.setattr(T, "load_cases", lambda *a: pytest.fail("second writer reached pixels"))
    monkeypatch.setattr(T, "NativeReader", lambda *a: pytest.fail("second writer reached model"))
    with writer_path.open("a") as writer:
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            T.compare(binding_path, p["result_out"])
    assert not p["result_out"].exists()


@pytest.mark.parametrize("change", ["protected", "fallback", "another_selection"])
def test_binding_rejects_wrong_role_fallback_or_other_output_before_candidate_blob(roi_package, monkeypatch, change):
    p = roi_package
    lock = prepare_guard(p)
    _, selection_path, selection = selected_fixture(p, lock)
    if change == "protected":
        selection["test_or_google_evaluated"] = True
    elif change == "fallback":
        selection["strict_general_dev_improvement"] = False
    else:
        selection_path = p["root"] / "unregistered-selection.json"
    selection_path.write_text(json.dumps(selection))
    monkeypatch.setattr(T.selector.export, "read_blob", lambda *a: pytest.fail("rejected selection reached blob"))
    with pytest.raises(ValueError, match="completed improved|preregistered output"):
        T.bind_selection(p["guard_out"], selection_path, Path(lock["candidate_out"]))
