"""Reject incomplete or mismatching PSP profiles before reporting performance."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "psp"))
spec = importlib.util.spec_from_file_location("psp_profile", ROOT / "psp/profile.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def fixture(app):
    stats = {"lines": 2, "frames": 20, "chars": 0, "text_ok": 2, "logits_ok": 2,
             "expect": True, "mode": "w8f", "tile": 4, "head": 16, "max_width": 1024,
             "invert": 0, "cpu_mhz": 333, "repeat": 3,
             "stack_bytes": 4096, "stack_used": 2992,
             "blob_bytes": 657664, "classes": 3501, "arena_bytes": 118432}
    (app / "stats.json").write_text(json.dumps(stats))
    (app / "results.txt").write_text("0\t8\t0\t80\t1\t1\t\n1\t12\t0\t120\t1\t1\t\n# done: 2 lines\n")
    return stats


def test_blank_text_can_pass_numerical_profile(tmp_path):
    expected = fixture(tmp_path)
    stats, cases = profile.checked_run(tmp_path, 2, expected, "w8f", 4, False)
    assert stats["text_ok"] == 2 and [r["frames"] for r in cases] == [8, 12]


@pytest.mark.parametrize("field,value", [("logits_ok", 1), ("text_ok", 1), ("expect", False),
                                         ("cpu_mhz", 222), ("mode", "w8i"), ("arena_bytes", 353536),
                                         ("repeat", 1), ("classes", 1), ("stack_bytes", 16384),
                                         ("stack_used", 4096), ("stack_used", 0)])
def test_mismatched_profile_cannot_be_a_performance_result(tmp_path, field, value):
    expected = fixture(tmp_path)
    changed = dict(expected, **{field: value})
    (tmp_path / "stats.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        profile.checked_run(tmp_path, 2, expected, "w8f", 4, False)


@pytest.mark.parametrize("rows", ["0\t8\t0\t80\t1\t1\t\n1\t12\t0\t120\t1\t1\t\n",
                                  "1\t8\t0\t80\t1\t1\t\n0\t12\t0\t120\t1\t1\t\n# done: 2 lines\n",
                                  "0\t8\t0\t80\t1\t1\t\n1\t12\t0\t120\t1\t0\t\n# done: 2 lines\n"])
def test_incomplete_reordered_or_mismatched_cases_are_rejected(tmp_path, rows):
    expected = fixture(tmp_path)
    (tmp_path / "results.txt").write_text(rows)
    with pytest.raises(ValueError):
        profile.checked_run(tmp_path, 2, expected, "w8f", 4, False)
