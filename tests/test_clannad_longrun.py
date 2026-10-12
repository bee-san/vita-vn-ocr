"""Synthetic, offline checks for immutable long-run test capture and proxy scoring."""

import ctypes as C
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "eval" / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


LR = module("clannad_longrun_tested", "clannad_longrun.py")
BRIDGE = module("owocr_bridge_tested", "owocr_bridge.py")


@pytest.mark.parametrize("split", ["dev", "train", None])
def test_non_test_role_rejected_before_capture_or_pixel_sideeffects(tmp_path, monkeypatch, split):
    def forbidden(*args, **kwargs):
        pytest.fail("a non-test role reached runtime construction or image access")

    monkeypatch.setattr(LR, "Native", forbidden)
    monkeypatch.setattr(LR, "verified_image", forbidden)
    output = tmp_path / "must-not-be-created"
    with pytest.raises(ValueError, match="split=test"):
        LR.collect(SimpleNamespace(split=split, session_dir=output))
    with pytest.raises(ValueError, match="split=test"):
        LR.verify_capture(tmp_path, {"split": split}, "a" * 64)
    assert not output.exists()


def test_journal_resume_preserves_prefix_and_continues_hash_chain(tmp_path):
    path = tmp_path / "nested" / "captures.jsonl"
    first = LR.append_event(path, {"kind": "capture", "split": "test", "value": "っ"})
    old = path.read_bytes()
    second = LR.append_event(path, {"kind": "pause", "split": "test", "value": "retry"})
    assert path.read_bytes().startswith(old)
    assert first["sequence"] == 0 and first["previous_sha256"] is None
    assert second["sequence"] == 1 and second["previous_sha256"] == first["event_sha256"]
    assert LR.read_journal(path) == [first, second]
    assert LR.append_event(path, {"kind": "resume", "split": "test"})["sequence"] == 2


def test_journal_concurrent_append_and_follow_keep_one_complete_chain(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "captures.jsonl"
    LR.append_event(path, {"kind": "seed", "value": -1})
    original = path.read_bytes()

    def append(value):
        event = LR.append_event(path, {"kind": "generated", "value": value})
        assert event in LR.read_journal(path)

    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(append, range(24)))
    rows = LR.read_journal(path)
    assert path.read_bytes().startswith(original)
    assert len(rows) == 25 and {row["value"] for row in rows} == set(range(-1, 24))
    assert [row["sequence"] for row in rows] == list(range(25))


@pytest.mark.parametrize("corruption", ["payload", "sequence", "previous", "reorder", "partial"])
def test_journal_rejects_tamper_and_partial_append_without_rewriting(tmp_path, corruption):
    path = tmp_path / "captures.jsonl"
    rows = [LR.append_event(path, {"kind": "capture", "value": i}) for i in range(2)]
    if corruption == "payload":
        rows[0]["value"] = "changed"
    elif corruption == "sequence":
        rows[1]["sequence"] = 3
    elif corruption == "previous":
        rows[1]["previous_sha256"] = "0" * 64
    elif corruption == "reorder":
        rows.reverse()
    raw = b"".join(LR.canonical(row) + b"\n" for row in rows)
    if corruption == "partial":
        raw += b'{"kind":"capture"'
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="journal"):
        LR.read_journal(path)
    with pytest.raises(ValueError, match="journal"):
        LR.append_event(path, {"kind": "resume"})
    assert path.read_bytes() == raw


def test_frozen_protocol_is_idempotent_and_rejects_changed_identity(tmp_path):
    path = tmp_path / "lock.json"
    protocol = {"split": "test", "model_sha256": "a" * 64, "roi": [8, 10, 40, 25]}
    expected = LR.frozen_file(path, protocol)
    original = path.read_bytes()
    assert LR.frozen_file(path, dict(protocol)) == expected
    with pytest.raises(ValueError, match="frozen protocol mismatch"):
        LR.frozen_file(path, dict(protocol, model_sha256="b" * 64))
    assert path.read_bytes() == original


@pytest.mark.parametrize("changed", ["model", "source", "dependency", "role"])
def test_locked_baseline_sources_dependencies_and_role_are_verified(tmp_path, monkeypatch, changed):
    source = tmp_path / "runtime" / "vocr.c"
    source.parent.mkdir()
    source.write_text("generated code identity\n")
    session = tmp_path / "session"
    lock = {
        "schema": 1,
        "split": "test",
        "runtime": {"model_sha256": LR.BASELINE_SHA},
        "sources": {"runtime/vocr.c": LR.file_sha(source)},
        "host_dependencies": {"numpy": "pinned"},
    }
    LR.frozen_file(session / "lock.json", lock)
    monkeypatch.setattr(LR, "ROOT", tmp_path)
    monkeypatch.setattr(LR.importlib.metadata, "version", lambda name: "pinned")
    loaded, lock_sha = LR.validate_lock(session)
    assert loaded == lock and lock_sha == LR.file_sha(session / "lock.json")
    if changed == "model":
        lock["runtime"]["model_sha256"] = "0" * 64
        (session / "lock.json").write_text(json.dumps(lock))
    elif changed == "source":
        source.write_text("changed code identity\n")
    elif changed == "dependency":
        monkeypatch.setattr(LR.importlib.metadata, "version", lambda name: "changed")
    else:
        lock["split"] = "dev"
        (session / "lock.json").write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="model|source|dependency|split=test"):
        LR.validate_lock(session)


@pytest.mark.parametrize(
    "field,value", [("sha256", "0" * 64), ("pixels_sha256", "0" * 64), ("mode", "RGB"), ("size", [12, 8])]
)
def test_verified_image_checks_encoded_and_decoded_identity(tmp_path, field, value):
    pixels = np.arange(8 * 11, dtype=np.uint8).reshape(8, 11)
    entry = LR.store_image(tmp_path, "inputs/line.png", pixels)
    np.testing.assert_array_equal(LR.verified_image(tmp_path, entry), pixels)
    with pytest.raises(ValueError, match="corruption"):
        LR.verified_image(tmp_path, dict(entry, **{field: value}))
    with pytest.raises(ValueError, match="already exists"):
        LR.store_image(tmp_path, entry["path"], 255 - pixels)
    np.testing.assert_array_equal(LR.verified_image(tmp_path, entry), pixels)


def test_verified_image_rejects_path_escape_before_file_access(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("escaping capture path reached file access")

    monkeypatch.setattr(LR, "file_sha", forbidden)
    with pytest.raises(ValueError, match="escapes session"):
        LR.verified_image(tmp_path, {"path": "../outside.png"})


def test_dialogue_mask_ignores_scenery_and_cursor_but_keeps_new_ink():
    image = np.full((60, 100, 3), 40, np.uint8)
    image[18:28, 20:45:3] = 245
    image[24:30, 65:70] = 250  # blinking advance marker within ignored rectangle
    roi, ignored = [10, 10, 65, 30], [[63, 20, 10, 15]]
    signature, ink = LR.dialogue_signature(image, roi, ignored, 200)
    scenery = np.random.default_rng(4).integers(0, 190, image.shape, dtype=np.uint8)
    changed = np.where((image >= 200), image, scenery)
    changed[20:35, 63:73] = 0
    changed[:10] = 255  # scenery outside ROI also has no effect
    assert LR.dialogue_signature(changed, roi, ignored, 200) == (signature, ink)
    changed[30:35, 45:48] = 245
    assert LR.dialogue_signature(changed, roi, ignored, 200)[0] != signature


def test_stability_requires_completion_and_persists_accepted_signatures():
    tracker = LR.Stability(seconds=2, samples=3)
    for now in (0, 1, 2):
        assert not tracker.observe("unfinished", now, complete=False)
    assert not tracker.observe("unfinished", 3, complete=True, kind="choice")
    assert not tracker.observe("finished", 4, complete=True)
    assert not tracker.observe("finished", 5, complete=True)
    assert tracker.observe("finished", 6, complete=True)
    tracker.accept("finished")
    assert not tracker.observe("finished", 7, complete=True)
    resumed = LR.Stability(seconds=2, samples=3, seen=tracker.seen)
    for now in (10, 11, 12, 13):
        assert not resumed.observe("finished", now, complete=True)


@pytest.fixture(scope="module")
def c_finder(tmp_path_factory):
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
        pytest.skip("a C compiler is required for the actual finder contract")
    library = tmp_path_factory.mktemp("longrun-c") / "finder.so"
    subprocess.run(
        [
            compiler,
            "-std=c99",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fPIC",
            "-shared",
            str(ROOT / "runtime/vocr_lines.c"),
            "-o",
            str(library),
        ],
        check=True,
    )
    lib = C.CDLL(str(library))
    lib.vocr_lines_arena_bytes.argtypes = [C.c_uint, C.c_uint]
    lib.vocr_lines_arena_bytes.restype = C.c_size_t
    lib.vocr_find_lines.argtypes = [C.POINTER(LR._LinesImage), C.c_int, C.c_void_p, C.c_size_t, C.POINTER(LR._LineSet)]
    native = LR.Native.__new__(LR.Native)
    native.lib = lib
    return native


def test_native_c_finder_returns_full_ink_and_shared_polarity(c_finder):
    image = np.full((80, 320, 3), 30, np.uint8)
    yy, xx = np.mgrid[0:20, 0:244]
    cell = xx % 24
    ys, xs = np.nonzero((cell < 18) & ((cell % 6 < 3) | (yy % 8 < 3)))
    for top in (12, 46):
        image[top + ys, xs + 16] = 240
    light, truncated, arena = c_finder.crops(image)
    dark, dark_truncated, dark_arena = c_finder.crops(255 - image)
    assert not truncated and not dark_truncated and arena == dark_arena
    assert len(light) == len(dark) == 2
    for (crop, box, inverted), (dark_crop, dark_box, dark_inverted), top in zip(light, dark, (12, 46)):
        assert not inverted and dark_inverted and box == dark_box
        x, y, w, h = box
        assert x <= 16 and x + w >= 254 and y <= top and y + h >= top + 20
        np.testing.assert_array_equal(crop, LR.to_gray(image)[y : y + h, x : x + w])
        np.testing.assert_array_equal(crop, dark_crop)


def synthetic_capture(directory):
    rgb = np.arange(40 * 70 * 3, dtype=np.uint32).reshape(40, 70, 3).astype(np.uint8)
    crops = [np.arange(6 * 7, dtype=np.uint8).reshape(6, 7), np.arange(5 * 9, dtype=np.uint8).reshape(5, 9) + 100]
    native = SimpleNamespace(
        crops=lambda region: ([(crops[0], [2, 3, 7, 6], False), (crops[1], [4, 13, 9, 5], True)], False, 1234)
    )
    event = LR.capture_event(
        directory,
        rgb,
        [8, 10, 40, 25],
        native,
        "signature",
        "a" * 64,
        "generated-session",
        {"page": "page-1", "complete": True},
    )
    return LR.append_event(directory / "captures.jsonl", event), crops


def test_interrupted_capture_does_not_block_resume_or_overwrite_saved_inputs(tmp_path):
    rgb = np.full((40, 70, 3), 37, np.uint8)

    def interrupted(region):
        raise RuntimeError("generated interruption before capture journal append")

    with pytest.raises(RuntimeError, match="interruption"):
        LR.capture_event(
            tmp_path,
            rgb,
            [8, 10, 40, 25],
            SimpleNamespace(crops=interrupted),
            "interrupted-signature",
            "a" * 64,
            "generated-session",
            {"page": "unfinished"},
        )
    assert not (tmp_path / "captures.jsonl").exists()
    saved = {path: LR.file_sha(path) for path in (tmp_path / "inputs").glob("*.png")}
    assert len(saved) == 2
    capture, _ = synthetic_capture(tmp_path)
    assert LR.read_journal(tmp_path / "captures.jsonl") == [capture]
    assert tmp_path / capture["source"]["path"] not in saved
    for path, expected in saved.items():
        assert LR.file_sha(path) == expected


def test_comparison_engines_consume_exact_same_frozen_line_pixels(tmp_path):
    capture, expected = synthetic_capture(tmp_path)
    native_inputs, google_inputs = [], []

    def text(pixels):
        native_inputs.append(pixels.copy())
        return "っあ"

    def request(directory, entry):
        pixels = LR.verified_image(directory, entry)
        google_inputs.append(pixels)
        return {"ok": True, "text": "っあ", "lines": [], "host_seconds": 0.1}

    result = LR.compare_capture(tmp_path, capture, SimpleNamespace(text=text), SimpleNamespace(request=request))
    assert result["ok"] and len(result["lines"]) == 2 and len(google_inputs) == 3
    assert result["capture_sha256"] == capture["event_sha256"]
    for i, entry in enumerate(capture["lines"]):
        assert entry["mode"] == "L"
        np.testing.assert_array_equal(native_inputs[i], expected[i])
        np.testing.assert_array_equal(google_inputs[i], native_inputs[i])
        assert result["lines"][i]["input_sha256"] == entry["sha256"]
        assert result["lines"][i]["pixels_sha256"] == entry["pixels_sha256"]
    assert google_inputs[-1].ndim == 3  # separate whole-ROI finder diagnostic


def test_comparison_records_engine_and_transport_failures_without_losing_pairs(tmp_path):
    capture, _ = synthetic_capture(tmp_path)
    calls = []

    def text(pixels):
        raise RuntimeError("synthetic native failure")

    def request(directory, entry):
        calls.append(entry["path"])
        if len(calls) == 1:
            return {"ok": False, "error": "synthetic service rejection", "host_seconds": 0.1}
        raise RuntimeError("synthetic worker transport failure")

    result = LR.compare_capture(tmp_path, capture, SimpleNamespace(text=text), SimpleNamespace(request=request))
    assert not result["ok"] and len(result["lines"]) == 2 and len(calls) == 3
    assert all(not row["vocr"]["ok"] for row in result["lines"])
    assert all(not row["google"]["ok"] for row in result["lines"])
    assert "transport failure" in result["lines"][1]["google"]["error"]
    assert not result["whole_roi_google"]["ok"]
    summary = LR.aggregate([capture], [result])
    assert summary["counts"]["failed_captures"] == 1
    assert summary["counts"]["line_google_failures"] == summary["counts"]["line_vocr_failures"] == 2
    assert summary["counts"]["roi_google_failures"] == 1
    assert summary["paired_line_proxy"]["reference_chars"] == 0
    assert summary["paired_line_proxy"]["disagreement_rate"] is None


def comparison(capture_sha, reference, prediction, ok=True):
    return {
        "kind": "comparison",
        "split": "test",
        "capture_sha256": capture_sha,
        "ok": ok,
        "lines": [
            {
                "google": {"ok": True, "text": reference, "host_seconds": 0.2},
                "vocr": {"ok": True, "text": prediction, "host_seconds": 0.1},
            }
        ],
        "whole_roi_google": {"ok": True, "text": reference, "lines": [{"text": reference}], "host_seconds": 0.3},
    }


def test_aggregate_accounts_failures_pending_retries_and_repeated_proxy_pages():
    captures = [
        {"kind": "capture", "split": "test", "event_sha256": str(i), "signature": f"unique-{i}"} for i in range(6)
    ]
    captures[-1]["signature"] = captures[0]["signature"]
    comparisons = [
        comparison("0", "っあ", "", ok=False),
        comparison("0", "っあ", "つあ"),
        comparison("1", "っあ", "つつ"),
        comparison("2", "いいう", "い"),
        comparison("3", "失敗", "", ok=False),
        comparison("5", "重複", "除外"),
    ]
    result = LR.aggregate(captures, comparisons)
    assert result["counts"]["captures"] == 6
    assert result["counts"]["failed_captures"] == result["counts"]["pending_captures"] == 1
    assert result["counts"]["successful_paired_lines"] == 3
    assert result["counts"]["successful_unique_pages"] == 3
    assert result["counts"]["duplicate_dialogue_signatures"] == 1
    assert result["paired_line_proxy"]["units"] == 3
    assert result["paired_line_proxy"]["reference_chars"] == 7
    assert result["paired_line_proxy"]["edit_disagreements"] == 5
    assert result["whole_roi_vocr_proxy"]["reference_chars"] == 7
    assert result["whole_roi_vs_google_finder_proxy"]["edit_disagreements"] == 0
    assert result["timing"]["vocr"]["samples"] == 3
    assert "proxy" in result["reference"] and "not hand-checked" in result["reference"]
    assert not result["target_met"]


def test_proxy_normalization_preserves_small_kana_and_dakuten():
    capture = [{"kind": "capture", "split": "test", "event_sha256": "one", "signature": "ink-one"}]
    result = LR.aggregate(capture, [comparison("one", "っが！", "つか!")])
    paired = result["paired_line_proxy"]
    assert paired["reference_chars"] == 3 and paired["edit_disagreements"] == 2
    assert paired["hiragana"]["chars"] == paired["hiragana"]["errors"] == 2
    assert "cer" not in paired["hiragana"] and paired["hiragana"]["disagreement_rate"] == 1
    empty = LR.aggregate([], [])
    assert empty["paired_line_proxy"]["disagreement_rate"] is None
    with pytest.raises(ValueError, match="split=test"):
        LR.aggregate([dict(capture[0], split="dev")], [])


def test_google_bridge_preserves_word_separators_without_importing_service():
    box = SimpleNamespace(left=1, top=2, right=11, bottom=12)
    words = [SimpleNamespace(text="あ", separator=" "), SimpleNamespace(text="い", separator=None)]
    result = SimpleNamespace(
        paragraphs=[
            SimpleNamespace(
                lines=[
                    SimpleNamespace(text=None, words=words, bounding_box=box),
                    SimpleNamespace(text="っ！", words=[], bounding_box=None),
                ]
            )
        ]
    )
    text, boxes = BRIDGE.result_text(result)
    assert text == "あ い\nっ！"
    assert boxes == [{"text": "あ い", "box": [1, 2, 11, 12]}, {"text": "っ！", "box": None}]


def test_collector_resume_appends_new_pages_without_recapturing_old_signatures(tmp_path, monkeypatch):
    from PIL import Image

    def image(extra):
        pixels = np.full((544, 960, 3), 30, np.uint8)
        pixels[410:425, 40:75:3] = 245
        if extra:
            pixels[410:425, 80 : 80 + extra] = 245
        return pixels

    images = {"a": image(0), "b": image(3), "c": image(6)}
    sequence = ["a", "a", "b", "b"]
    cursor = {"sample": 0, "clock": 0}

    class Native:
        def __init__(self, *args):
            self.info = {}

        def crops(self, pixels):
            return [(LR.to_gray(pixels)[5:25, 5:50].copy(), [5, 5, 45, 20], False)], False, 1234

    def control(path):
        assert cursor["sample"] < len(sequence), "collector failed to accept generated stable pages"
        return {"kind": "dialogue", "complete": True, "page": sequence[cursor["sample"]]}

    def screenshot(command, **kwargs):
        assert command[:2] == ["import", "-window"]
        Image.fromarray(images[sequence[cursor["sample"]]]).save(command[-1])
        cursor["sample"] += 1
        return SimpleNamespace(returncode=0)

    def monotonic():
        cursor["clock"] += 1
        return cursor["clock"]

    monkeypatch.setattr(LR, "Native", Native)
    monkeypatch.setattr(
        LR,
        "protocol",
        lambda args, native: {
            "split": "test",
            "session": args.session,
            "ignore_boxes": args.ignore_box,
            "native_size": [960, 544],
        },
    )
    monkeypatch.setattr(LR, "read_control", control)
    monkeypatch.setattr(LR.subprocess, "run", screenshot)
    monkeypatch.setattr(LR.time, "monotonic", monotonic)
    monkeypatch.setattr(LR.time, "sleep", lambda seconds: None)
    args = SimpleNamespace(
        split="test",
        count=2,
        interval=0.25,
        stable_samples=2,
        stable_seconds=1,
        session_dir=tmp_path / "session",
        session="generated",
        model=tmp_path / "unused",
        display=":generated",
        window="generated",
        control_file=tmp_path / "unused-control",
        roi=[20, 400, 100, 40],
        ignore_box=[],
        ink_threshold=200,
        complete_box=None,
        max_lines=5,
        advance_key=None,
        idle_timeout=120,
    )
    LR.collect(args)
    journal = args.session_dir / "captures.jsonl"
    original = journal.read_bytes()
    prior_rows = LR.read_journal(journal)
    first = [row for row in prior_rows if row["kind"] == "capture"]
    assert len(first) == 2 and [row["completion"]["page"] for row in first] == ["a", "b"]
    first_counts = LR.aggregate(prior_rows, [])["counts"]
    assert first_counts["sampled_frames"] == 4 and first_counts["incomplete_observations"] == 0
    input_hashes = {
        entry["path"]: LR.file_sha(args.session_dir / entry["path"])
        for row in first
        for entry in [row["source"], row["roi"], *row["lines"]]
    }
    sequence[:] = ["a", "a", "b", "b", "c", "c"]
    cursor["sample"] = 0
    args.count = 1
    LR.collect(args)
    assert journal.read_bytes().startswith(original)
    resumed_rows = LR.read_journal(journal)
    resumed = [row for row in resumed_rows if row["kind"] == "capture"]
    assert len(resumed) == 3 and resumed[-1]["completion"]["page"] == "c"
    assert resumed[-1]["sequence"] == len(prior_rows)
    assert resumed[-1]["previous_sha256"] == prior_rows[-1]["event_sha256"]
    assert len({row["signature"] for row in resumed}) == 3
    counts = LR.aggregate(resumed_rows, [])["counts"]
    assert counts["sampled_frames"] == 10
    assert counts["incomplete_observations"] == counts["duplicate_signature_observations"] == 4
    for name, expected in input_hashes.items():
        assert LR.file_sha(args.session_dir / name) == expected


@pytest.mark.parametrize("reveal", [None, "x"])
def test_verified_controller_sequence_releases_keys_and_does_not_repeat_advance(monkeypatch, reveal):
    calls, sleeps = [], []
    monkeypatch.setattr(LR.subprocess, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(LR.time, "sleep", sleeps.append)
    args = SimpleNamespace(window="77", advance_key="c", key_hold_seconds=0.3, reveal_key=reveal, reveal_delay=0.5)
    LR.advance(args, {"DISPLAY": ":test"})
    assert calls[:3] == [["xdotool", "windowfocus", "77"], ["xdotool", "keydown", "c"], ["xdotool", "keyup", "c"]]
    assert sleeps == ([0.3] if reveal is None else [0.3, 0.5, 0.3])
    if reveal:
        assert calls[3:] == [["xdotool", "windowfocus", "77"], ["xdotool", "keydown", "x"], ["xdotool", "keyup", "x"]]
    else:
        assert len(calls) == 3


def test_automatic_controller_requires_explicit_operator_screen_kind():
    args = SimpleNamespace(split="test", complete_box=[0, 0, 10, 10], control_file=None, advance_key="c")
    with pytest.raises(ValueError, match="screen-kind control"):
        LR.protocol(args, None)
