"""train/train.py helpers: resize, CTC feasibility, batching, greedy decode and edit distance."""
import numpy as np
import torch

import train as T


def test_resize32_keeps_aspect_and_clamps_width():
    assert T.resize32(np.zeros((48, 300), np.uint8)).shape == (32, 200)
    assert T.resize32(np.zeros((16, 2), np.uint8)).shape == (32, 8)          # at least 8 wide
    assert T.resize32(np.zeros((10, 1000), np.uint8)).shape == (32, 1024)    # at most MAX_W


def test_ctc_ok_needs_a_frame_per_label_and_per_repeat():
    assert T.ctc_ok([1, 2, 3], 24)          # 3 frames for 3 labels
    assert not T.ctc_ok([1, 2, 3], 16)      # 2 frames
    assert not T.ctc_ok([5, 5], 16)         # a repeat needs a blank between: 3 frames
    assert T.ctc_ok([5, 5], 17)             # ceil(17 / 8) = 3


def test_make_batch_pads_to_the_stride_and_scales_to_unit_range():
    imgs = [np.full((32, 20), 255, np.uint8), np.zeros((32, 9), np.uint8)]
    labels = [np.array([1, 2]), np.array([3])]
    x, in_len, tgt, tgt_len = T.make_batch(imgs, labels, [0, 1])
    assert tuple(x.shape) == (2, 1, 32, 24)                 # widest line rounded up to 8
    assert float(x[0].min()) == 1.0 and float(x[1].max()) == -1.0
    assert bool(x[1, 0, :, 9:].eq(-1.0).all())              # right edge replicated into the pad
    assert in_len.tolist() == [3, 2]                        # ceil(w / 8) per line
    assert tgt.tolist() == [1, 2, 3] and tgt_len.tolist() == [2, 1]


def test_greedy_collapses_repeats_and_drops_blanks():
    frames = [0, 1, 1, 0, 1, 2, 2]                          # -> 1 1 2
    logits = torch.full((len(frames), 1, 3), -5.0)          # T x N x C
    for t, c in enumerate(frames):
        logits[t, 0, c] = 5.0
    assert T.greedy(logits, torch.tensor([7])) == [[1, 1, 2]]
    assert T.greedy(logits, torch.tensor([3])) == [[1]]     # only the first in_len frames count


def test_edit_distance():
    assert T.edit_distance([], []) == 0
    assert T.edit_distance([1, 2, 3], [1, 2, 3]) == 0
    assert T.edit_distance([1, 2, 3], [1, 3]) == 1
    assert T.edit_distance([1, 2], [3, 4, 5]) == 3
    assert T.edit_distance("kitten", "sitting") == 3
