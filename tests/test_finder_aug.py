"""train/finder_aug.py and train2's fa= option (VOCR5 arm F; prompts/vocr5-aug.md part A): shape and height,
determinism, the min_width guard, labels untouched, the spec's feature rates, and fa=0 bit for bit equal to train2
before the option (a frozen copy of its make_batch, run on the same seeds). Procedural crops only, CPU only."""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for d in ("train", "synth", "eval"):
    sys.path.insert(0, str(ROOT / d))

import finder_aug as FA  # noqa: E402
import train2 as T2  # noqa: E402

torch.set_num_threads(1)


def text_crop(w=120, seed=0, bg=30.0, ink=220.0, h=32):
    """A line-like crop: strokes of 'glyphs' every 11-17 px in rows 6-26 of an h-row background."""
    r = np.random.default_rng(seed)
    g = np.full((h, w), bg, np.float32)
    x = 4
    while x + 14 < w - 4:
        gw = int(r.integers(8, 15))
        for _ in range(3):
            if r.random() < 0.5:
                y = int(r.integers(6 * h // 32, 24 * h // 32))
                g[y:y + 2, x:x + gw] = ink
            else:
                x0 = x + int(r.integers(0, gw - 2))
                g[6 * h // 32:26 * h // 32, x0:x0 + 2] = ink
        x += gw + 3
    return g


# ----------------------------------------------------------------------------- finder_aug


def test_levels_are_the_border_median_and_the_farther_percentile():
    g = np.full((32, 50), 40.0, np.float32)
    g[10:20, 5:45] = 230.0                                  # bright text on a dark box
    assert FA.levels(g) == (40.0, 230.0)
    assert FA.levels(255.0 - g) == (215.0, 25.0)            # dark text on a light box


@pytest.mark.parametrize("seed", range(40))
def test_output_is_a_height_32_float_crop_in_range(seed):
    g = text_crop(int(np.random.default_rng(seed).integers(16, 400)), seed)
    log = []
    out = FA.finder_aug(g, np.random.default_rng(seed), text_crop(90, seed + 1000), log=log)
    assert out.dtype == np.float32 and out.ndim == 2 and out.shape[0] == T2.H
    assert 8 <= out.shape[1] <= T2.MAX_W
    assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 255.0
    t, b = (int(v) for v in log[0].split("-")[1:])
    assert 3 <= t + b <= 19                                 # E = round(32 U(0.1, 0.6))
    if not any(f.startswith(("cursor", "portrait")) for f in log):  # rows only: the line keeps its aspect
        assert out.shape[1] == round(g.shape[1] * 32 / (32 + t + b))


def test_same_generator_state_gives_the_same_crop():
    g, donor = text_crop(200, 1), text_crop(150, 2)
    outs = []
    for _ in range(2):
        rng, log = np.random.default_rng(7), []
        outs.append((FA.finder_aug(g, rng, donor, log=log), log, rng.bit_generator.state))
    assert np.array_equal(outs[0][0], outs[1][0]) and outs[0][1] == outs[1][1] and outs[0][2] == outs[1][2]
    other = [FA.finder_aug(g, np.random.default_rng(s), donor) for s in range(8, 16)]
    assert any(o.shape != outs[0][0].shape or not np.array_equal(o, outs[0][0]) for o in other)


def test_input_and_donor_are_not_modified():
    g, donor = text_crop(200, 3), text_crop(120, 4).astype(np.uint8)
    g0, d0 = g.copy(), donor.copy()
    for s in range(30):
        FA.finder_aug(g, np.random.default_rng(s), donor)
    assert np.array_equal(g, g0) and np.array_equal(donor, d0)


def test_min_width_guard_returns_the_crop_itself():
    g = text_crop(64, 5)
    for s in range(20):
        log = []
        out = FA.finder_aug(g, np.random.default_rng(s), text_crop(64, 6), min_width=10_000, log=log)
        assert out is g and log[-1] == "unchanged-min-width"


def test_width_is_capped_at_max_w():
    g = text_crop(1024, 7)
    for s in range(10):
        assert FA.finder_aug(g, np.random.default_rng(s), None).shape[1] <= T2.MAX_W


def test_without_a_donor_nothing_is_borrowed():
    seen = Counter()
    for s in range(400):
        log = []
        FA.finder_aug(text_crop(100, s), np.random.default_rng(s), None, log=log)
        seen.update(log[1:])
    assert not [f for f in seen if f.startswith(("neighbour", "furigana")) or f.endswith("-donor")]
    assert any(f.startswith("portrait") for f in seen) and any(f.startswith("cursor") for f in seen)


def test_a_donor_without_glyphs_adds_nothing():
    blank = np.full((32, 80), 100.0, np.float32)
    for s in range(200):
        log = []
        FA.finder_aug(text_crop(100, s), np.random.default_rng(s), blank, log=log)
        assert not [f for f in log if f.startswith(("neighbour", "furigana"))]


def test_feature_rates_follow_the_spec():
    """3,000 draws with a donor: each feature's share among the cases it can occur in, against the spec's p."""
    n, c, sides = 3000, Counter(), Counter()
    rng = np.random.default_rng(2026)
    for s in range(n):
        log = []
        FA.finder_aug(text_crop(160, s % 50), rng, text_crop(120, 50 + s % 50), log=log)
        t, b = (int(v) for v in log[0].split("-")[1:])
        sides.update({"top": t > 0, "bottom": b > 0, "top2": t >= 2, "bottom2": b >= 2})
        c.update(f.rsplit("-", 1)[0] if f.startswith("portrait") else f for f in log[1:])
        c["clutter"] += any(f.startswith(("cursor", "portrait")) for f in log)
    tol = 0.035
    for side in ("top", "bottom"):
        assert abs(c[f"frame-{side}"] / sides[side] - 0.35) < tol
        assert abs(c[f"neighbour-{side}"] / sides[f"{side}2"] - 0.35) < tol
        assert abs(c[f"scenery-{side}"] / sides[side] - 0.20) < tol
    assert abs(c["furigana"] / sides["top"] - 0.15) < tol
    assert abs(c["clutter"] / n - 0.5) < tol
    cursor = sum(v for k, v in c.items() if k.startswith("cursor"))
    assert abs(cursor / c["clutter"] - 0.6) < tol
    starts = sum(v for k, v in c.items() if k.startswith("portrait-start"))
    assert abs(starts / (c["clutter"] - cursor) - 0.5) < 0.06
    assert {k for k in c if k.startswith("cursor")} == {f"cursor-{k}" for k in FA.SHAPES}


def test_sheet_of_a_synthetic_shard(tmp_path):
    crops = [text_crop(40 + 3 * i, i, h=24 + i % 20).astype(np.uint8) for i in range(60)]
    shard = tmp_path / "synth.npz"
    np.savez(shard, data=np.concatenate([c.ravel() for c in crops]),
             offsets=np.cumsum([0] + [c.size for c in crops]).astype(np.int64),
             shapes=np.array([c.shape for c in crops], np.int32), labels=json.dumps(["あ"] * 60))
    out = tmp_path / "sheet.png"
    FA.main(["sheet", "--shard", str(shard), "--out", str(out), "--n", "40"])
    import cv2
    im = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    assert im is not None and max(im.shape) <= 1280
    rows = json.loads(out.with_suffix(".json").read_text())["rows"]
    assert len(rows) == 40 and all(r["features"] for r in rows)


# ----------------------------------------------------------------------------- train2 fa=


def make_batch_before_fa(items, rng, src_opts):
    """train2.make_batch as it was before fa= (origin/main 1e02a77), verbatim but for the module prefix."""
    H, STRIDE, augment = T2.H, T2.STRIDE, T2.augment
    gs = []
    for it in items:
        o = src_opts[it.src]
        if rng is not None and o.get("aug", "1") != "0":
            frames = 1 if it.ids is None else len(it.ids) + sum(
                a == b for a, b in zip(it.ids, it.ids[1:]))
            min_width = max(8, (frames - 1) * STRIDE + 1)
            g = augment(it.img, rng, geo=it.kid is None, inv=float(o.get("inv", 0.2)),
                        min_width=min_width)
        else:
            g = it.img.astype(np.float32)
        gs.append(g)
    W = (max(g.shape[1] for g in gs) + STRIDE - 1) // STRIDE * STRIDE
    x = np.empty((len(gs), 1, H, W), np.float32)
    for j, g in enumerate(gs):
        w = g.shape[1]
        x[j, 0, :, :w] = g
        x[j, 0, :, w:] = g[:, -1:]
    x = torch.from_numpy(x / 127.5 - 1.0).contiguous(memory_format=torch.channels_last)
    in_len = torch.tensor([math.ceil(g.shape[1] / STRIDE) for g in gs], dtype=torch.long)
    lab = [j for j, it in enumerate(items) if it.ids is not None]
    tgt = torch.from_numpy(np.concatenate([items[j].ids for j in lab])) if lab else torch.zeros(0, dtype=torch.long)
    tgt_len = torch.tensor([len(items[j].ids) for j in lab], dtype=torch.long)
    kd = None
    kd_rows = [j for j, it in enumerate(items) if it.kid is not None]
    if kd_rows:
        Tm = W // STRIDE
        k = items[kd_rows[0]].kid.shape[1]
        kid = np.zeros((len(items), Tm, k), np.int64)
        kp = np.zeros((len(items), Tm, k), np.float32)
        mask = np.zeros((len(items), Tm), np.float32)
        for j in kd_rows:
            t = items[j].kid.shape[0]
            kid[j, :t], kp[j, :t], mask[j, :t] = items[j].kid, items[j].kp, 1.0
        kd = (torch.from_numpy(kid), torch.from_numpy(kp), torch.from_numpy(mask))
    return x, in_len, torch.tensor(lab, dtype=torch.long), tgt, tgt_len, kd


class ListSource:
    """RamSource's sampling over a fixed item list."""

    def __init__(self, name, items, seed):
        self.name, self.items, self.rng = name, items, np.random.default_rng(seed)
        self.order, self.pos = self.rng.permutation(len(items)), 0

    sample = T2.RamSource.sample


def items_of(n, seed, kd_every=0):
    r = np.random.default_rng(seed)
    out = []
    for i in range(n):
        w = int(r.integers(24, 300))
        img = np.clip(text_crop(w, seed * 1000 + i) + r.normal(0, 3, (32, w)), 0, 255).astype(np.uint8)
        k = int(r.integers(1, max(2, min(30, w // 8))))
        ids = r.integers(1, 6, k).astype(np.int64)
        kid = kp = None
        if kd_every and i % kd_every == 0:
            t = (w + 7) // 8
            kid, kp = np.zeros((t, 4), np.int16), np.full((t, 4), 0.25, np.float16)
        out.append(T2.Item(img, ids, kid, kp))
    return out


def batches(fn, opts, steps=25, seed=3, batch=12):
    sources = [ListSource("a", items_of(40, 1, kd_every=7), 11), ListSource("b", items_of(30, 2), 12),
               ListSource("c", items_of(20, 3), 13)]
    rng = np.random.default_rng(seed)
    mixer = T2.Mixer(sources, [0.5, 0.3, 0.2], batch, rng, bucket=3)
    out = []
    for _ in range(steps):
        items = mixer.next()
        out.append((fn(items, rng, opts), rng.bit_generator.state))
    return out


def same_batch(a, b):
    (xa, la, laba, ta, tla, kda), (xb, lb, labb, tb, tlb, kdb) = a, b
    if not all(torch.equal(u, v) for u, v in ((xa, xb), (la, lb), (laba, labb), (ta, tb), (tla, tlb))):
        return False
    return (kda is None) == (kdb is None) and (kda is None or all(torch.equal(u, v) for u, v in zip(kda, kdb)))


@pytest.mark.parametrize("fa", [None, "0", "0.0"])
def test_fa0_batches_are_bit_identical_to_train2_before_the_option(fa):
    base = [{"inv": "0.5"}, {"inv": "0.2"}, {"aug": "0"}]
    opts = [dict(o, **({} if fa is None else {"fa": fa})) for o in base]
    before, after = batches(make_batch_before_fa, base), batches(T2.make_batch, opts)
    assert len(before) == len(after) == 25
    for (b0, s0), (b1, s1) in zip(before, after):
        assert same_batch(b0, b1) and s0 == s1               # same tensors, same generator state after


def test_fa_changes_crops_but_never_labels():
    items = items_of(12, 9)
    snap = [(it.img.copy(), it.ids.copy()) for it in items]
    plain = T2.make_batch(items, np.random.default_rng(5), [{"inv": "0"}])
    aug = T2.make_batch(items, np.random.default_rng(5), [{"inv": "0", "fa": "1"}])
    for k in (2, 3, 4):                                       # lab, tgt, tgt_len
        assert torch.equal(plain[k], aug[k])
    assert not torch.equal(plain[1], aug[1]) or not torch.equal(plain[0], aug[0])
    assert all(np.array_equal(it.img, a) and np.array_equal(it.ids, b) for it, (a, b) in zip(items, snap))


def test_fa_keeps_every_ctc_label_reachable():
    """The min_width guard: a crop never ends up narrower than its label's frames need."""
    for s in range(30):
        items = items_of(10, 100 + s)
        for it in items:                                      # tight labels: one frame per 8 px
            it.ids = np.resize(np.arange(1, 6), max(1, it.img.shape[1] // 8 - 1)).astype(np.int64)
        _, in_len, _, _, tgt_len, _ = T2.make_batch(items, np.random.default_rng(s), [{"fa": "1", "inv": "0"}])
        need = [len(it.ids) + sum(a == b for a, b in zip(it.ids, it.ids[1:])) for it in items]
        assert all(int(n_in) >= n for n_in, n in zip(in_len, need))


def test_fa_picks_another_item_as_donor_and_skips_frame_target_lines(monkeypatch):
    calls = []

    def spy(g, rng, donor, **kw):
        calls.append(donor)
        return FA.finder_aug(g, rng, donor, **kw)
    monkeypatch.setattr(T2, "finder_aug", spy)
    items = items_of(9, 4, kd_every=3)                        # items 0, 3, 6 carry frame targets
    T2.make_batch(items, np.random.default_rng(0), [{"fa": "1"}])
    assert len(calls) == 6                                    # every line without frame targets, once
    own = [it.img for it in items if it.kid is None]
    assert all(d is not None and any(d is it.img for it in items) for d in calls)
    assert all(d is not g for d, g in zip(calls, own))        # never the line itself
    calls.clear()
    T2.make_batch(items[1:2], np.random.default_rng(0), [{"fa": "1"}])
    assert calls == [None]                                    # a one-line batch: no donor


def test_fa_only_applies_to_its_source():
    items = items_of(8, 6)
    for it in items[4:]:
        it.src = 1
    calls = []
    orig = T2.finder_aug
    try:
        T2.finder_aug = lambda g, rng, donor, **kw: calls.append(1) or orig(g, rng, donor, **kw)
        T2.make_batch(items, np.random.default_rng(1), [{"fa": "1"}, {}])
    finally:
        T2.finder_aug = orig
    assert len(calls) == 4
