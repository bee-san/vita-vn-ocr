"""Fast tests of train2's pure parts: the frame-level KD loss, batching with KD targets, the
shard loader's label / confidence / KD filters, the reduced-vocab KD remap, the streaming
loader's error path, the teacher's class map and frame pooling (a fake teacher, no ONNX model)
and bar.py's scoring. CPU only; the shards are tiny files in tmp_path."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for d in ("train", "synth", "eval"):
    sys.path.insert(0, str(ROOT / d))
sys.path.insert(0, str(ROOT))

from metrics import norm  # noqa: E402

import bar  # noqa: E402
import teacher_targets as TT  # noqa: E402
import train2 as T2  # noqa: E402
from train import encoder  # noqa: E402

torch.set_num_threads(1)
VOCAB = ["あ", "い", "う", "え", "お"]
K = 4


# ----------------------------------------------------------------------------- KD loss

def kd_targets(N, T, C, seed):
    rng = np.random.default_rng(seed)
    kid = np.stack([np.stack([rng.choice(C, K, replace=False) for _ in range(T)]) for _ in range(N)])
    kp = rng.random((N, T, K)).astype(np.float32) + 0.05
    kp /= kp.sum(2, keepdims=True)
    return torch.from_numpy(kid.astype(np.int64)), torch.from_numpy(kp)


@pytest.mark.parametrize("tau", [1.0, 2.0])
def test_kd_loss_is_zero_when_the_student_matches_the_targets(tau):
    N, T, C = 2, 6, 20
    kid, kp = kd_targets(N, T, C, seed=0)
    logits = torch.full((T, N, C), -40.0)
    for n in range(N):
        for t in range(T):
            logits[t, n, kid[n, t]] = kp[n, t].log()
    loss = T2.kd_loss(logits, (kid, kp, torch.ones(N, T)), tau)
    assert float(loss) == pytest.approx(0.0, abs=1e-4)


def test_kd_loss_is_the_mean_kl_over_target_frames_only():
    N, T, C = 3, 6, 20
    kid, kp = kd_targets(N, T, C, seed=1)
    torch.manual_seed(0)
    logits = torch.randn(T, N, C, requires_grad=True)
    mask = torch.zeros(N, T)
    mask[0, :4] = 1          # row 0: targets for its first 4 frames
    mask[2, :] = 1           # row 2: every frame; row 1 has no targets at all
    loss = T2.kd_loss(logits, (kid, kp, mask), 1.0)
    lp = torch.log_softmax(logits.detach(), 2).permute(1, 0, 2).numpy()
    ref = [float((kp[n, t].numpy() * (np.log(kp[n, t].numpy()) - lp[n, t, kid[n, t].numpy()])).sum())
           for n in range(N) for t in range(T) if mask[n, t]]
    assert float(loss) == pytest.approx(np.mean(ref), rel=1e-5)
    loss.backward()
    g = logits.grad
    assert torch.all(g[:, 1] == 0)            # a line without targets gets no KD gradient
    assert torch.all(g[4:, 0] == 0)           # nor do frames past a line's targets
    assert torch.any(g[:4, 0] != 0) and torch.any(g[:, 2] != 0)


# ----------------------------------------------------------------------------- batching

def test_make_batch_pads_and_builds_ctc_and_kd_inputs():
    def item(w, ids, t_kd=None):
        img = np.full((32, w), 200, np.uint8)
        img[:, -1] = 7                                   # the last column, which padding repeats
        kid = kp = None
        if t_kd:
            kid, kp = np.full((t_kd, K), 3, np.int16), np.full((t_kd, K), 0.25, np.float16)
        return T2.Item(img, None if ids is None else np.asarray(ids, np.int64), kid, kp, 0)

    items = [item(40, [1, 2]), item(64, None, t_kd=8), item(17, [3], t_kd=3)]
    x, in_len, lab, tgt, tgt_len, kd = T2.make_batch(items, None, [{}])
    assert x.shape == (3, 1, 32, 64)
    assert in_len.tolist() == [5, 8, 3]
    assert lab.tolist() == [0, 2]                         # the KD-only line has no CTC label
    assert tgt.tolist() == [1, 2, 3] and tgt_len.tolist() == [2, 1]
    assert torch.allclose(x[0, 0, :, 40:], torch.full((32, 24), 7 / 127.5 - 1.0))
    kid, kp, mask = kd
    assert kid.shape == (3, 8, K) and kp.shape == (3, 8, K)
    assert mask.tolist() == [[0] * 8, [1] * 8, [1] * 3 + [0] * 5]


# ----------------------------------------------------------------------------- shard loader

def write_shard(path, crops, labels):
    np.savez(path, data=np.concatenate([c.ravel() for c in crops]).astype(np.uint8),
             offsets=np.cumsum([0] + [c.size for c in crops]).astype(np.int64),
             shapes=np.array([c.shape for c in crops], np.int32),
             labels=json.dumps(labels, ensure_ascii=False))


def write_targets(path, texts, confs, ts, ok):
    off = np.concatenate([[0], np.cumsum(ts)]).astype(np.int64)
    n = int(off[-1])
    np.savez(path, text=json.dumps(texts, ensure_ascii=False), conf=np.asarray(confs, np.float32),
             T=np.asarray(ts, np.int32), off=off, kid=np.tile(np.arange(K, dtype=np.int16), (n, 1)),
             kp=np.full((n, K), 1 / K, np.float16), soft_ok=np.asarray(ok, bool))


@pytest.fixture
def shard(tmp_path):
    """Six 32 x 64 crops (8 student frames each):
    0 labelled, teacher agrees            3 unlabelled, teacher unsure (conf 0.3)
    1 unlabelled, teacher confident       4 labelled, frame targets unusable (soft_ok False)
    2 labelled, teacher disagrees         5 labelled with characters outside the vocab"""
    crops = [np.full((32, 64), 40 * i, np.uint8) for i in range(6)]
    write_shard(tmp_path / "s.npz", crops, ["あい", "", "えお", "", "あ", "かき"])
    tdir = tmp_path / "targets"
    tdir.mkdir()
    write_targets(tdir / "s.v6s.npz", ["あい", "う", "えあ", "おお", "あ", "か"],
                  [[0.9, 0.95], [0.95, 0.97], [0.9, 0.9], [0.3, 0.6], [0.9, 0.9], [0.9, 0.9]],
                  [8] * 6, [True, True, True, True, False, True])
    return tmp_path / "s.npz", tdir


def load(path, **opts):
    return T2.load_shard(path, encoder(VOCAB), opts, norm)


def test_load_shard_gt_labels_only(shard):
    items, st = load(shard[0])
    assert st == {"kept": 3, "dropped": 3, "kd": 0, "kdonly": 0}     # 1, 3 unlabelled; 5 OOV
    assert all(it.kid is None for it in items)
    assert [it.ids.tolist() for it in items] == [[1, 2], [4, 5], [1]]


def test_load_shard_kd_only_where_the_teacher_agrees_and_its_frames_are_usable(shard):
    items, st = load(shard[0], targets=str(shard[1]), kd="1")
    assert st["kept"] == 3 and st["kd"] == 1
    assert items[0].kid is not None and items[0].kid.shape == (8, K)
    assert items[1].kid is None and items[2].kid is None              # disagrees; not soft_ok


def test_load_shard_teacher_fills_missing_labels_above_the_confidence(shard):
    items, st = load(shard[0], targets=str(shard[1]), label="gt|teacher", kd="1")
    assert st["kept"] == 4 and st["kd"] == 2                          # line 3 is under conf 0.5
    assert items[1].ids.tolist() == [3] and items[1].kid is not None


def test_load_shard_kdonly_keeps_unsure_lines_without_a_ctc_label(shard):
    items, st = load(shard[0], targets=str(shard[1]), label="teacher", conf="0.5", kd="1", kdonly="1")
    assert st == {"kept": 5, "dropped": 1, "kd": 4, "kdonly": 1}
    assert items[2].ids.tolist() == [4, 1]                            # the teacher's reading
    assert items[3].ids is None and items[3].kid is not None


def test_load_shard_needs_targets_for_kd(shard, tmp_path):
    with pytest.raises(SystemExit):
        load(shard[0], targets=str(tmp_path / "nowhere"), kd="1")


def test_kd_remap_maps_onto_a_reduced_vocab():
    full = T2.load_vocab(T2.DATA / "vocab.txt")
    red = full[:10] + [T2.UNK]
    m = T2.kd_remap(red)
    assert len(m) == len(full) + 1 and m[0] == 0                       # blank stays blank
    assert m[1:11].tolist() == list(range(1, 11))
    assert set(m[11:].tolist()) == {11}                               # the rest go to <unk>
    assert T2.kd_remap(full) is None
    with pytest.raises(SystemExit):
        T2.kd_remap(full[:10])                                        # no <unk> class


# ----------------------------------------------------------------------------- streaming

def test_stream_source_samples_its_shards(tmp_path):
    for k in range(2):
        write_shard(tmp_path / f"s{k}.npz", [np.full((32, 64), 50, np.uint8)] * 3, ["あい", "う", "お"])
    src = T2.StreamSource("s", [str(tmp_path / f"s{k}.npz") for k in range(2)], encoder(VOCAB),
                          {"pool": "2"}, norm, np.random.default_rng(0))
    got = [src.sample() for _ in range(10)]
    assert all(it.ids is not None and it.img.shape == (32, 64) for it in got)


def test_stream_source_raises_instead_of_hanging_when_a_shard_fails(tmp_path):
    write_shard(tmp_path / "s.npz", [np.full((32, 64), 50, np.uint8)], ["あい"])
    with pytest.raises(RuntimeError, match="shard loader failed"):
        T2.StreamSource("s", [str(tmp_path / "s.npz")], encoder(VOCAB),
                        {"targets": str(tmp_path / "nowhere"), "kd": "1"}, norm, np.random.default_rng(0))


def test_stream_source_raises_when_no_shard_has_a_usable_line(tmp_path):
    write_shard(tmp_path / "s.npz", [np.full((32, 64), 50, np.uint8)] * 2, ["", ""])
    with pytest.raises(RuntimeError, match="no usable lines"):
        T2.StreamSource("s", [str(tmp_path / "s.npz")], encoder(VOCAB), {}, norm, np.random.default_rng(0))


# ----------------------------------------------------------------------------- teacher targets

def fake_teacher(vocab, chars):
    t = TT.Teacher.__new__(TT.Teacher)                # no ONNX session
    t.set_classes(chars, vocab)
    return t


def test_teacher_classes_map_onto_the_student_vocab_through_canon():
    t = fake_teacher(["あ", "い", "！", "1"], ["あ", "い", "!", "１", "鬱"])
    # blank, あ, い, ! -> ！, １ -> 1, 鬱 (not in the vocab) -> no class, space -> blank
    assert t.map.tolist() == [0, 1, 2, 3, 4, -1, 0]


def test_student_input_is_train_resize_plus_last_column_padding():
    g = (np.arange(20 * 47) % 251).astype(np.uint8).reshape(20, 47)
    x, T = TT.student_input(g)
    w32 = round(47 * 32 / 20)                          # 75 px at height 32
    assert T == (w32 + 7) // 8 and x.shape == (32, 8 * T)
    np.testing.assert_array_equal(x[:, :w32], T2.resize_h(g, 32))
    assert (x[:, w32:] == x[:, w32 - 1:w32]).all()


def test_aligned_pools_two_teacher_frames_per_student_frame():
    t = fake_teacher(VOCAB[:4], VOCAB[:4])            # teacher classes: blank, あいうえ, space
    y = np.zeros((4, 6), np.float32)                   # 4 teacher frames for 2 student frames
    y[0, [0, 1]] = 0.9, 0.1
    y[1, [0, 1, 2]] = 0.2, 0.7, 0.1
    y[2, [0, 1, 2]] = 0.1, 0.8, 0.1
    y[3, [0, 5]] = 0.9, 0.1                            # space counts as blank
    seen = {}
    t._run = lambda x: seen.setdefault("shape", x.shape) and y
    T, kid, kp, seq = t.aligned(np.full((32, 16), 128, np.uint8))
    assert T == 2 and seen["shape"] == (48, 32, 3)     # the student's 16 px, upscaled to 48 x 16T
    assert kid.dtype == np.int16 and kp.dtype == np.float16 and kid.shape == (2, K)
    np.testing.assert_allclose(kp.astype(np.float32).sum(1), 1.0, atol=2e-3)
    q0 = dict(zip(kid[0].tolist(), kp[0].astype(np.float64).tolist()))
    assert q0[0] == pytest.approx(0.9 * 0.2, abs=1e-3)                     # both frames blank
    assert q0[1] == pytest.approx((1 - 0.9 * 0.3) * 0.82 / 0.83, abs=1e-3)   # noisy-or, renormalised
    assert q0[2] == pytest.approx(0.1 * 0.82 / 0.83, abs=1e-3)
    q1 = dict(zip(kid[1].tolist(), kp[1].astype(np.float64).tolist()))
    assert q1[0] == pytest.approx(0.1, abs=1e-3) and q1[1] == pytest.approx(0.8, abs=1e-3)
    assert seq == [1]                    # あ in both frames, no blank between: one あ (see soft_ok)


# ----------------------------------------------------------------------------- bar.py

def test_bar_scores_per_style_and_marks_the_rules_criteria():
    man = {"a": {"text": "あいう", "styles": ["name"]}, "b": {"text": "えお", "styles": []}}
    sc = bar.score({"a": "あいう", "b": "え"}, man)
    assert sc["all"] == pytest.approx((1 / 5, 0.5, 2))
    assert sc["name"] == pytest.approx((0.0, 1.0, 1))
    assert bar.mark(0.020, 0.023, overall=True) == "*"
    assert bar.mark(0.023, 0.023, overall=True) == "="                 # a tie does not beat the bar
    assert bar.mark(0.030, 0.023, overall=True) == ""
    assert bar.mark(0.023, 0.023, overall=False) == "*"                # a style may tie
    assert bar.missing({"a": "x"}, man) == ["b"]
