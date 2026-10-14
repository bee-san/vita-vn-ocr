"""The C runtime arena and train/export.py plan_arena agree for tail_conf > 0 too.

CI (runtime/tests/parity.py) only checks the arena for tail_conf == 0, so this test
covers the enabled path: it must grow the arena by exactly one acc_frames-float
array, and plan_arena must predict it. It also re-checks that tail_conf == 0 is
byte-identical to a plain context.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))

import export  # noqa: E402  (conftest puts train/ on the path)
from model import TinyVNOCR  # noqa: E402


def _blob():
    import torch
    torch.manual_seed(3)
    model = TinyVNOCR(4, width=0.25, bottleneck=16).eval()
    raw, _ = export.build_blob(export.model_ops(model, 32), ["あ", "い", "〓"], 32, int8=True)
    return raw


@pytest.mark.parametrize("mode", ["w8i", "w8f"])
@pytest.mark.parametrize("width", [0, 256, 1024])
@pytest.mark.parametrize("tail", [0, 1, 500, 1000])
def test_c_arena_matches_plan_arena_for_tail_conf(mode, width, tail):
    pyvocr = pytest.importorskip("pyvocr")
    blob = _blob()
    parsed = export.read_blob(blob)
    mode_id = {"w8i": export.MODE_W8I, "w8f": export.MODE_W8F}[mode]
    want = export.plan_arena(parsed, max_width=width, mode=mode_id, tail_conf=tail)
    got = pyvocr.Vocr(blob, mode=mode, max_width=width, tail_conf=tail, check_crc=False).arena_bytes
    assert got == want


def test_tail_conf_zero_does_not_grow_the_arena():
    pyvocr = pytest.importorskip("pyvocr")
    blob = _blob()
    base = pyvocr.Vocr(blob, mode="w8i", check_crc=False).arena_bytes
    off = pyvocr.Vocr(blob, mode="w8i", tail_conf=0, check_crc=False).arena_bytes
    on = pyvocr.Vocr(blob, mode="w8i", tail_conf=500, check_crc=False).arena_bytes
    assert off == base          # the disabled path is identical to a plain context
    assert on > base            # the enabled path reserves the softmax denominator
    assert on - base == export.plan_arena(export.read_blob(blob), mode=export.MODE_W8I,
                                          tail_conf=500) - base


def test_tail_conf_above_1000_is_rejected():
    pyvocr = pytest.importorskip("pyvocr")
    blob = _blob()
    with pytest.raises(ValueError):
        pyvocr.Vocr(blob, mode="w8i", tail_conf=1001, check_crc=False)
