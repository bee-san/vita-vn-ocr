"""Scenery patches cannot leak neighbouring source text into synthetic crops."""
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import render as R


def test_background_load_crops_before_scaling(tmp_path):
    pixels = np.zeros((100, 120, 3), np.uint8)
    pixels[:40] = (20, 40, 60)
    pixels[40:] = (240, 240, 240)  # stand-in for source dialogue/title ink
    path = tmp_path / "source.png"
    Image.fromarray(pixels).save(path)
    patch = R.load_bg(path, 1.4, (0, 0, 120, 40))
    assert patch.shape == (56, 168, 3)
    assert np.array_equal(patch, np.broadcast_to((20, 40, 60), patch.shape))


@pytest.mark.parametrize("reviewed", [False, True])
def test_tall_background_region_uses_fallback_instead_of_source_text(monkeypatch, reviewed):
    path = "generated.png"
    resource = SimpleNamespace(bg_psp=[], bg_vita=[path], bg_boxes={path: (0, 0, 100, 40)} if reviewed else {})
    monkeypatch.setattr(R, "R", resource)
    monkeypatch.setattr(R, "load_bg", lambda *_: np.full((100, 120, 3), 255, np.uint8))
    height = 110 if reviewed else 70
    result = R.background(np.random.default_rng(1), 40, height, "vita")
    assert result.shape == (height, 40, 3)
    assert np.any(result != 255)


def test_repeated_ctc_labels_and_tall_detections_are_rejected():
    assert R.crop_ctc_feasible((32, 16), "あい")
    assert not R.crop_ctc_feasible((32, 16), "ああ")
    assert R.crop_ctc_feasible((32, 17), "ああ")
    assert not R.crop_ctc_feasible((85, 716), "あい" * 18 + "あ")
    assert not R.crop_ctc_feasible((32, 1024), "あ" * 65)
