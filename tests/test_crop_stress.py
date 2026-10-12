"""Generated development captures verify bounded crop and ROI diagnostics."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import crop_stress as CS  # noqa: E402

import export  # noqa: E402
from model import TinyVNOCR  # noqa: E402


@pytest.fixture
def capture(tmp_path):
    pixels = np.full((32, 64, 3), 220, np.uint8)
    pixels[8:20, 10:38:3] = 30
    image = tmp_path / "capture.png"
    Image.fromarray(pixels).save(image)
    row = {"id": "dev-1", "split": "dev", "session": "capture-block-1", "hand_checked": True,
           "image": image.name, "sha256": CS.sha256(image), "text": "あい", "dims": [64, 32],
           "box": [10, 8, 28, 12], "region": [4, 4, 44, 22]}
    manifest = tmp_path / "dev.jsonl"
    manifest.write_text(json.dumps(row))
    return manifest, row, pixels


@pytest.mark.parametrize("role", ["test", "train", None])
def test_non_dev_role_is_rejected_before_pixels_or_model(capture, monkeypatch, tmp_path, role):
    manifest, row, _ = capture
    row.update(split=role, image="absent.png")
    manifest.write_text(json.dumps(row))
    monkeypatch.setattr(CS, "NativeVocr", lambda _: pytest.fail("non-dev manifest reached model"))
    with pytest.raises(ValueError, match="development-only"):
        CS.run(manifest, "absent.vocr", tmp_path / "result.json")


@pytest.mark.parametrize("change,message", [
    ({"box": [-1, 8, 28, 12]}, "outside"),
    ({"box": [10, 8, 100, 12]}, "outside"),
    ({"box": [10, 8, 0, 12]}, "outside"),
    ({"box": [10.0, 8, 28, 12]}, "integer"),
    ({"dims": [128, 64]}, "dimensions"),
    ({"sha256": "0" * 64}, "hash mismatch"),
    ({"hand_checked": False}, "hand_checked"),
    ({"text": ""}, "nonempty"),
])
def test_frame_coordinate_hash_and_label_guards(capture, change, message):
    manifest, row, _ = capture
    manifest.write_text(json.dumps({**row, **change}))
    with pytest.raises(ValueError, match=message):
        CS.load_development(manifest)


def test_every_margin_case_preserves_full_ink_and_reports_boundary_padding():
    box = [0, 1, 15, 12]
    for case in CS.CASES:
        bounds, actual = CS.crop_bounds(box, case, 16, 16)
        assert bounds[0] <= 0 and bounds[1] <= 1 and bounds[2] >= 15 and bounds[3] >= 13
        assert actual["left"] == 0
        assert actual["top"] == 1
        assert actual["bottom"] == 3
        assert actual["right"] == min(1, case["right"])
    left = CS.CASES[-2]
    right = CS.CASES[-1]
    assert CS.crop_bounds([10, 8, 28, 12], left, 64, 32)[0] == (10, 4, 46, 24)
    assert CS.crop_bounds([10, 8, 28, 12], right, 64, 32)[0] == (2, 4, 38, 24)


def test_negative_padding_is_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        CS.crop_bounds([1, 1, 10, 10], {"left": -1, "right": 0, "top": 4, "bottom": 4}, 32, 32)


class PerfectRecognizer:
    def __init__(self):
        self.crops = []

    def read(self, crops):
        self.crops.extend(crops)
        return ["あい"] * len(crops)


def test_line_scoring_uses_console_polarity_and_aggregate_only(capture):
    rows, images, _ = CS.load_development(capture[0])
    rec = PerfectRecognizer()
    result = CS.line_case(rec, rows, images, CS.CASES[2])
    expected = 255 - CS.to_gray(capture[2][4:24, 6:42])
    np.testing.assert_array_equal(rec.crops[0], expected)
    assert result["full"]["errors"] == result["hiragana"]["errors"] == 0
    assert result["full"]["chars"] == result["hiragana"]["chars"] == 2
    assert result["fully_visible_ink_lines"] == 1 and result["padding_clamped_lines"] == 0
    assert result["inverted_lines"] == 1
    assert '"text"' not in json.dumps(result)
    assert '"pred"' not in json.dumps(result)


def test_roi_shrink_skips_any_region_that_would_clip_reference_ink(capture, monkeypatch):
    rows, images, _ = CS.load_development(capture[0])
    monkeypatch.setattr(CS.finders, "console", lambda _: pytest.fail("clipped region reached finder"))
    result = CS.roi_case(PerfectRecognizer(), rows, images, -8)
    assert result["regions_skipped_to_preserve_ink"] == 1
    assert result["regions"] == result["fully_visible_gt_lines"] == result["full"]["chars"] == 0


def test_roi_finder_miss_is_charged_to_full_and_hiragana_errors(capture, monkeypatch):
    rows, images, _ = CS.load_development(capture[0])
    monkeypatch.setattr(CS.finders, "console", lambda _: [])
    result = CS.roi_case(PerfectRecognizer(), rows, images, 0)
    assert result["regions_skipped_to_preserve_ink"] == 0
    assert result["full"]["errors"] == result["hiragana"]["errors"] == 2
    assert result["full"]["chars"] == result["hiragana"]["chars"] == 2
    assert result["line_recall"] == 0 and result["detections"] == 0


def test_roi_expansion_clamps_frame_padding_and_retains_ink(capture, monkeypatch):
    rows, images, _ = CS.load_development(capture[0])
    monkeypatch.setattr(CS.finders, "console", lambda _: [(10, 8, 28, 12)])
    result = CS.roi_case(PerfectRecognizer(), rows, images, 10)
    assert result["regions_skipped_to_preserve_ink"] == 0
    assert result["padding_clamped_regions"] == 1
    assert result["fully_visible_gt_lines"] == 1 and result["full"]["errors"] == 0
    assert result["line_recall"] == result["line_precision"] == 1


def test_native_int8_c_run_records_identity_memory_and_refuses_overwrite(capture, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = TinyVNOCR(4, width=.25, bottleneck=16).eval()
    raw, _ = export.build_blob(export.model_ops(model, 32), ["あ", "い", "〓"], 32, int8=True)
    blob = tmp_path / "generated.vocr"
    blob.write_bytes(raw)
    out = tmp_path / "result.json"
    result = CS.run(capture[0], blob, out)
    assert result["dataset"]["manifest_sha256"] == CS.sha256(capture[0])
    assert result["dataset"]["split"] == "dev"
    assert result["model_info"]["sha256"] == CS.sha256(blob)
    assert result["model_info"]["mode"] == "w8i" and result["model_info"]["blob_bytes"] == len(raw)
    assert result["model_info"]["arena_bytes"] == export.plan_arena(export.read_blob(raw), mode=export.MODE_W8I)
    assert len(result["cases"]) == 6 and not result["heldout_test_evaluated"]
    assert {case["full"]["chars"] for case in result["cases"]} == {2}
    assert json.loads(out.read_text()) == result
    assert '"text"' not in out.read_text() and '"pred"' not in out.read_text()
    with pytest.raises(FileExistsError, match="already exists"):
        CS.run(capture[0], blob, out)
