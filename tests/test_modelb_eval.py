"""Model B (#126): the native-PSP scorer (eval/modelb_eval.py) and the pool builder (data/modelb_pools.py) on
generated data: the console-crop pairing rule, train2's shard format, per-component totals, the multi-seed paired
comparison, the protocol's consistency with the launcher. No development, test or private data is read."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("eval", "data", "train", "runtime", "synth"):
    if str(ROOT / p) not in sys.path:
        sys.path.insert(0, str(ROOT / p))

import modelb_eval as ME  # noqa: E402

import modelb_pools as MP  # noqa: E402


def test_pairing_keeps_whole_lines_only():
    # two lines of a region, a finder box per line
    lines = [((10, 10, 100, 18), (10, 10, 100, 18)), ((10, 40, 80, 18), (10, 40, 80, 18))]
    cells = [[(10 + 16 * i, 26 + 16 * i) for i in range(6)], [(10 + 16 * i, 26 + 16 * i) for i in range(5)]]
    boxes = [(8, 8, 104, 22, 0), (8, 38, 84, 22, 1)]
    pairs, why = MP.pair_lines(boxes, lines, (), cells)
    assert pairs == [(0, 0), (1, 1)] and why["kept"] == 2
    # a box that leaves out the last whole glyph: cut
    pairs, why = MP.pair_lines([(8, 8, 80, 22, 0)], lines[:1], (), cells[:1])
    assert pairs == [] and why["cuts_text"] == 1
    # half of a punctuation cell outside (its ink is the left half): kept
    pairs, _ = MP.pair_lines([(8, 8, 98, 22, 0)], lines[:1], (), cells[:1])
    assert pairs == [(0, 0)]
    # one box over both rows: a merged pair is never kept
    pairs, why = MP.pair_lines([(8, 8, 104, 52, 0)], lines, (), cells)
    assert pairs == [] and why["merged_rows"] == 1
    # without cells the fallback needs the main text's columns less a quarter of its height
    pairs, _ = MP.pair_lines([(14, 8, 92, 22, 0)], lines[:1], (), None)
    assert pairs == [(0, 0)]
    pairs, _ = MP.pair_lines([(20, 8, 86, 22, 0)], lines[:1], (), None)
    assert pairs == []


def test_shard_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 255, (int(rng.integers(10, 30)), int(rng.integers(20, 200)))).astype(np.uint8)
             for _ in range(7)]
    labels = [f"あ{i}" for i in range(7)]
    meta = [{"id": i, "kind": "psp"} for i in range(7)]
    sha = MP.write_shard(tmp_path / "x.npz", crops, labels, meta)
    c, lab, m = MP.load_shard(tmp_path / "x.npz")
    assert len(sha) == 64 and lab == labels and m == meta and all(np.array_equal(a, b) for a, b in zip(c, crops))
    import train2
    items, stats = train2.load_shard(tmp_path / "x.npz", lambda t: [1, 2], {}, lambda t: t)
    assert stats["kept"] == 7 and all(it.img.shape[0] == train2.H for it in items)


def _rows(errors, comp="wd_grad", name=False, vn_of=lambda i: f"v{i % 6}"):
    return [{"comp": comp, "unit": f"{comp}|{i}", "vn": vn_of(i), "name": name, "errors": int(e), "chars": 20,
             "frames": 30, "blank": 20, "gated": 0, "escaped": 0, "lines_out": 1} for i, e in enumerate(errors)]


def test_totals_and_components():
    rows = _rows([1, 2, 0, 3]) + _rows([0, 1], comp="tv") + _rows([5], comp="tv", name=True) + _rows([2], comp="dn")
    t = ME.totals(rows)
    assert t["wd_grad"]["errors"] == 6 and t["wd_grad"]["chars"] == 80 and t["wd_grad"]["cer"] == 7.5
    assert t["fixed"]["errors"] == 3 and t["fixed"]["units"] == 3      # tv + dn dialogue, names out
    assert t["names"]["errors"] == 5 and t["wd_grad"]["blank_share"] == round(20 / 30, 4)


def test_compare_mean_over_seeds(tmp_path):
    rng = np.random.default_rng(1)
    base = rng.integers(0, 6, 60)
    cands = [np.maximum(base - rng.integers(0, 3, 60), 0) for _ in range(3)]
    files = []
    for k, e in enumerate([base] + cands):
        p = tmp_path / f"r{k}.json"
        p.write_text(json.dumps({"blob_sha256": f"{k}" * 64, "rows": _rows(e)}))
        files.append(str(p))

    class A:
        cand, base, out = files[1:], files[:1], str(tmp_path / "cmp.json")
    ME.compare(A)
    c = json.loads(Path(A.out).read_text())["components"]["wd_grad"]
    want = 100 * (np.mean(cands, axis=0).sum() - base.sum()) / (20 * 60)
    assert abs(c["delta"] - round(want, 6)) < 1e-6 and c["vn_ci95"][1] < 0 and c["clusters"] == 6
    assert len(c["per_seed_delta"]) == 3
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"blob_sha256": "x" * 64, "rows": _rows(base[:-1])}))

    class B:
        cand, base, out = [str(bad)], files[:1], str(tmp_path / "cmp2.json")
    with pytest.raises(SystemExit):
        ME.compare(B)


def test_namelike_rule():
    assert ME.namelike("佐藤") and not ME.namelike("「佐藤」") and not ME.namelike("こんにちはさようなら")


def test_protocol_and_launcher_agree():
    proto = json.loads((ROOT / "eval/vocr6_psp_protocol.json").read_text())
    import vocr6_psp_run as R
    w = proto["data"]["sources"]
    assert abs(sum(s["weight"] for s in w.values()) - 1.0) < 1e-9
    assert abs(w["hk"]["weight"] + w["hc"]["weight"] + w["npi"]["weight"] + w["npg"]["weight"] - 0.5) < 1e-9
    for name, arm in proto["arms"].items():
        if name == "Q":
            continue
        assert arm["init"] in R.INIT and "--height" in arm["train2_args"] and arm["seeds"] in ([1, 2], [1])
    assert set(R.NATIVE_INK) | set(R.NATIVE_GRAD) == set(MP.NATIVE)
    assert proto["budgets"]["memory_bytes_le"] == 4000000 and proto["budgets"]["time_s_le"] == 1.0
