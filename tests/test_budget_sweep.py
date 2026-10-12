"""vita/budget_sweep.py (docs/vita-memory-budget.md): the random-init grid, its exact sizes, the cache
model's parameters, the pack format and the tables. None of it needs VitaSDK, qemu or Vita3K."""
import json
import struct
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "vita"), str(ROOT / "train"), str(ROOT / "eval")]
import a9_cycles  # noqa: E402
import budget_sweep as BS  # noqa: E402


def test_vocab_of_extends_the_vocab_with_new_ideographs():
    base = (ROOT / "data/vocab.txt").read_text(encoding="utf-8").splitlines()
    assert BS.vocab_of(3501) == base[:3500]
    assert BS.vocab_of(len(base) + 1) == base
    big = BS.vocab_of(11001)
    assert len(big) == 11000 and len(set(big)) == 11000 and big[:len(base)] == base
    assert all(0x4E00 <= ord(c) < 0xA000 for c in big[len(base):])
    assert sum(len(c.encode()) for c in big) < 65536          # the blob's 16-bit vocab offsets
    with pytest.raises(ValueError):
        BS.vocab_of(40000)


@pytest.mark.parametrize("name, blob, arena_w8i, macs", [
    ("w100", 859_904, 124_832, 1_002_720),           # v1's shape: runtime/README.md
    ("w100v3500", 657_616, 124_832, 824_736),        # FL10's shape (FL10's own vocab differs: 657,664 B)
    ("w150", 1_155_968, 182_944, 1_564_992),         # H15's shape: docs/train2.md
])
def test_grid_blobs_have_the_documented_sizes(name, blob, arena_w8i, macs):
    import export
    width, bn, depth, ctx, ctx_k, n_classes, height = BS.GRID[name]
    model = BS.random_model(width, bn, depth, ctx, ctx_k, n_classes)
    data, _ = export.build_blob(export.model_ops(model, height), BS.vocab_of(n_classes), height, int8=True)
    info = BS.blob_info(data)
    assert info["blob_bytes"] == len(data) == blob
    assert sum(info["split"].values()) == blob
    assert info["arena"]["t4h16"]["w8i"] == arena_w8i
    assert info["macs_per_frame"] == macs
    # bigger tiles and head batches never shrink the arena, and read each weight fewer times
    for key in ("t8h32", "t4h64", "t16h64"):
        assert info["arena"][key]["w8i"] >= arena_w8i
        assert info["weight_bytes_per_frame"][key] < info["weight_bytes_per_frame"]["t4h16"]
    t, h = BS.DEFAULT_TILE, BS.DEFAULT_HEAD
    assert info["weight_bytes_per_frame"]["t4h16"] == pytest.approx(
        info["body_weight_bytes"] / t + info["head_weight_bytes"] / h)


def test_random_models_are_deterministic():
    import export
    a, b = (export.build_blob(export.model_ops(BS.random_model(0.5, 48, 1, 2, 5, 101), 32), BS.vocab_of(101), 32,
                              int8=True)[0] for _ in range(2))
    assert a == b


def test_cache_args_default_is_a9_cycles_exactly():
    assert BS.cache_args(2 * 1024 * 1024) == a9_cycles.CACHE_ARGS
    assert "evict" not in a9_cycles.CACHE_ARGS                       # qemu's default: LRU
    s = BS.cache_args(1024 * 1024, ways=16)
    assert "l2cachesize=1048576,l2assoc=16," in s and s.replace("1048576", "2097152").replace(
        "l2assoc=16", "l2assoc=8") == a9_cycles.CACHE_ARGS
    assert BS.cache_args(2 * 1024 * 1024, evict="rand") == a9_cycles.CACHE_ARGS + ",evict=rand"
    with pytest.raises(ValueError):
        BS.cache_args(1000)
    with pytest.raises(ValueError):
        BS.cache_args(2 * 1024 * 1024, evict="mru")


def test_run_names_of_the_default_setup_are_unchanged():
    assert BS.run_name("H15", 2048, 0, 0) == "H15_l22048"
    assert BS.run_name("H15", 512, 16, 64) == "H15_l2512_t16h64"
    assert BS.run_name("H15", 2048, 0, 0, 16) == "H15_l22048_w16"
    assert BS.run_name("H15", 2048, 0, 0, 16, "rand", 2) == "H15_l22048_w16_rand_r2"


def test_synth_pack_matches_a9_cycles_pack_format(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    greys = [rng.integers(0, 256, (32, w), dtype=np.uint8) for w in (40, 77)]
    lines = [types.SimpleNamespace(crop_rgb=np.repeat(g[:, :, None], 3, axis=2)) for g in greys]
    import sets
    monkeypatch.setattr(sets, "load_synth_val", lambda path, limit: lines[:limit])
    n = BS.write_synth_pack(tmp_path / "lines.bin", 2)
    raw = (tmp_path / "lines.bin").read_bytes()
    assert n == 2 and struct.unpack_from("<I", raw)[0] == 2
    off = 4
    for g in greys:
        assert struct.unpack_from("<III", raw, off) == (32, g.shape[1], 1)
        off += 12
        assert raw[off:off + g.size] == g.tobytes()
        off += g.size
    assert off == len(raw)


def _run(name, l2, cyc, misses, tile=4, head=16, ways=8, evict="lru", rep=0):
    ms = {str(m): [c * 80 / (m * 1000) for c in cyc] for m in (444, 333)}
    return {"name": name, "l2_kib": l2, "l2_ways": ways, "evict": evict, "rep": rep, "tile": tile, "head": head,
            "l2_misses_per_frame": misses, "l1d_misses_per_frame": 1.0, "cycles_per_frame": {"444": cyc, "333": cyc},
            "ms_640px": ms}


def test_ceiling_status_line_is_the_documented_one():
    line = BS.status_line()
    assert line == "BUDGET_READY weights<=1600000 arena<=262144 ms<=550"
    assert line in (ROOT / "docs/vita-memory-budget.md").read_text(encoding="utf-8")


def test_verdict_checks_all_three_limits():
    info = {"blob_bytes": 1_558_016, "arena": {"t4h16": {"w8i": 241_056}}}
    run = _run("F20", 2048, [1_296_081, 2_282_290], 34)
    rows = BS.verdict(info, run)
    assert [r[3] for r in rows] == [True, True, True]
    assert rows[2][1] == pytest.approx(548.3, abs=0.05)              # 2,282,290 x 80 / 333,000
    big = {"blob_bytes": 1_600_001, "arena": {"t4h16": {"w8i": 262_145}}}
    slow = _run("x", 2048, [1, 2_300_000], 1)
    assert [r[3] for r in BS.verdict(big, slow)] == [False, False, False]


def test_tables_from_synthetic_runs(tmp_path):
    sizes = {"w100": {"params": 1, "blob_bytes": 859_904, "head_weight_bytes": 557_440, "split": {},
                      "arena": {k: {"w8i": 124_832 + i, "w8f": 1} for i, k in enumerate(("t4h16", "t16h64"))},
                      "macs_per_frame": 1_000_000, "weight_bytes_per_frame": {"t4h16": 100.0, "t16h64": 25.0}}}
    (tmp_path / "a9").mkdir()
    runs = (_run("w100", 2048, [600_000, 1_200_000], 20), _run("w100", 512, [600_000, 1_300_000], 2000),
            _run("w100", 512, [590_000, 1_250_000], 900, 16, 64), _run("w100", 2048, [600_000, 1_210_000], 25, ways=16),
            _run("w100", 2048, [600_000, 1_190_000], 15, ways=16, evict="rand", rep=1))
    for i, r in enumerate(runs):
        (tmp_path / "a9" / f"run{i}.json").write_text(json.dumps(r))
    text, summary = BS.tables(sizes, BS.load_runs(tmp_path))
    assert "| w100 | 1 | 859,904 | 557,440 | 124,832 | 1,000,000 | 600,000-1,200,000 | 0.600-1.200 | 108-216 |" \
        in text
    assert "| w100 | 512 | 16 / 64 | 124,833 | 25 | 900 | 142-300 | 2,000; 144-312 |" in text
    assert summary["w100"]["l2"][512]["l2_misses_per_frame"] == 2000
    assert summary["w100"]["cache_model"]["l22048_w16_lru_r0"]["l2_misses_per_frame"] == 25
    assert summary["w100"]["cache_model"]["l22048_w16_rand_r1"]["l2_misses_per_frame"] == 15
    assert "| w100 | 2048 | 16 | rand | 1 | 1 | 15 |" in text
    assert "| w100 | 2048 | 16 | lru | 0 | 1 | 25 |" in text
