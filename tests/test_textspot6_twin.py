"""The tagged spotter's twin (synth/textspot6_twin.py) and its exporter (train/textspot6_model.py), no data and no C:
versions 1-3 give textspot3_twin's results with every tag 0; version 4's header, both network forms, the head's
channel order, the tag decode (scores, ties, tconf), tag-aware blocks, the dialogue pick and presence on hand-made
maps; the loader's refusals; random blobs of both families round-trip."""
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "synth", ROOT / "train", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_parity as TP  # noqa: E402
import textspot2_export as X  # noqa: E402
import textspot2_shapes as S2  # noqa: E402
import textspot2_twin as T2  # noqa: E402
import textspot3_model as M3  # noqa: E402
import textspot3_twin as T3  # noqa: E402
import textspot6_model as M6  # noqa: E402
import textspot6_twin as T6  # noqa: E402
import textspot_parity as P  # noqa: E402
import textspot_twin as S  # noqa: E402

TINY_M = M3.m_shape("tinym", 4, 8, 8, [8], [1], [(16, [1]), (16, [2])], 8, 8, 1)
TINY_T = {"name": "tinyt", "family": "T", "stride": 8, "c0": 4, "c1": 8, "c2": 16, "chans": [16, 16], "dil": [1, 2],
          "block": "sep", "head": "kroa_x2", "half_above": 480}


def recrc(b):
    b = bytearray(b)
    if len(b) > 48 and b[32:36] == b"VTMS":
        struct.pack_into("<I", b, 44, zlib.crc32(bytes(b[48:])) & 0xFFFFFFFF)
    struct.pack_into("<I", b, 12, zlib.crc32(bytes(b[16:])) & 0xFFFFFFFF)
    return bytes(b)


def strip(r):
    """A textspot6 Result's fields that textspot3_twin's Result has."""
    return ([(b.x, b.y, b.w, b.h, b.vertical, b.conf, b.dark, b.block) for b in r.boxes],
            [(b.x, b.y, b.w, b.h, b.vertical, b.first, b.count) for b in r.blocks], r.pick, r.truncated)


def strip3(r):
    return ([(b.x, b.y, b.w, b.h, b.vertical, b.conf, b.dark, b.block) for b in r.boxes],
            [(b.x, b.y, b.w, b.h, b.vertical, b.first, b.count) for b in r.blocks], r.pick, r.truncated)


# ---------------------------------------------------------------- versions 1-3 are textspot3_twin's

def test_versions_1_to_3_are_textspot3_twin():
    rng = np.random.default_rng(5)
    S2.register()
    blobs = [S.pack(TP.random_blob("s4_c24n4", 41, 2), 600, 1), S.pack(TP.random_blob("s8_c24n4", 42, 2), 0, 3)]
    blobs += [T2.pack(X.random_vtdt(s, 7, 2), 600, 2) for s in S2.D2S]
    blobs += [M3.random_blob(TINY_M, 3, n_cal=2)[0]]
    for blob in blobs:
        a, b = T6.Model(blob), T3.Model(blob)
        assert a.tags == 0 and a.version == b.version and a.head == b.head
        for g in (P.tiled(rng, 960, 544), P.tiled(rng, 480, 272), P.tiled(rng, 37, 61)):
            img = TP.to_rgba(g, rng) if rng.random() < 0.5 else g
            r6, r3 = T6.spot(a, img), T3.spot(b, img)
            assert strip(r6) == strip3(r3)
            assert all(x.tag == 0 and x.tconf == 0 for x in r6.boxes) and all(x.tag == 0 for x in r6.blocks)
            assert r6.presence == max([x.conf for x in r3.boxes], default=0)
            assert r6 == T6.from_v3(r3)


def test_group_and_order_with_one_tag_is_textspot_twin():
    rng = np.random.default_rng(9)
    for _ in range(200):
        n = int(rng.integers(0, 14))
        boxes = set()
        while len(boxes) < n:
            v = int(rng.random() < 0.3)
            x, y = int(rng.integers(0, 900)), int(rng.integers(0, 500))
            w, h = (int(rng.integers(4, 30)), int(rng.integers(20, 200))) if v else \
                (int(rng.integers(20, 300)), int(rng.integers(4, 30)))
            boxes.add((x, y, w, h, v, int(rng.integers(1, 128)), int(rng.integers(0, 2))))
        boxes = list(boxes)
        ob, bl, pk = S.group_and_order(boxes, 544)
        ob6, bl6, pk6 = T6.group_and_order([(*b, 0, 0) for b in boxes], 544)
        assert [b[:7] + (b[9],) for b in ob6] == [tuple(b) for b in ob]
        assert [b[:7] for b in bl6] == [tuple(b) for b in bl] and pk6 == pk


# ---------------------------------------------------------------- version 4: blobs, refusals

def test_random_blobs_round_trip():
    for shape, tags in ((TINY_T, 1), (TINY_T, 3), (TINY_T, 4), (TINY_M, 2), (TINY_M, 4)):
        blob, _ = M6.random_blob(shape, 2, tags, n_cal=2)
        m = T6.Model(blob)
        assert m.version == 4 and m.tags == tags and m.form == ("trunk" if shape is TINY_T else "ms")
        assert (m.half_above, m.half_above_h) == M6.scales_of(shape)
        g = P.tiled(np.random.default_rng(1), 960, 544)
        x, f = T6.net_input(m, g)
        head = T6.run_head(m, x)
        assert head.shape[0] == 16 + 4 * tags
        maps = T6.run_maps(m, x)
        assert maps.shape == (4 + tags, -(-x.shape[0] // 4), -(-x.shape[1] // 4))
        assert (T6.maps_to_head(maps, *x.shape, tags)[:, :maps.shape[1] // 2] ==
                head[:, :maps.shape[1] // 2]).all()
        r = T6.spot(m, g)
        assert all(0 <= b.tag <= tags for b in r.boxes)


def test_head_channel_order():
    tags, h, w = 3, 24, 40
    rng = np.random.default_rng(3)
    maps = rng.integers(-128, 128, (4 + tags, 6, 10))
    head = T6.maps_to_head(maps, h, w, tags)
    for dy in (0, 1):
        for dx in (0, 1):
            for m in range(4):
                assert (head[T2.channel(dy, dx, m)] == maps[m, dy::2, dx::2]).all()
            for t in range(tags):
                assert (head[T6.tag_channel(dy, dx, t, tags)] == maps[4 + t, dy::2, dx::2]).all()
    assert (T6.head_to_maps(head, h, w, tags) == maps).all()


def good_v4():
    return [M6.random_blob(TINY_T, 1, 4, n_cal=2)[0], M6.random_blob(TINY_M, 1, 4, n_cal=2)[0]]


def test_refusals():
    t, m = good_v4()
    T6.Model(t)
    T6.Model(m)
    bad = []
    b = bytearray(t)
    b[19] = 0                                   # tags 0
    bad.append(recrc(b))
    b = bytearray(t)
    b[19] = 5                                   # tags 5
    bad.append(recrc(b))
    b = bytearray(t)
    b[19] = 3                                   # header says 3 tags, the network has 32 outputs
    bad.append(recrc(b))
    b = bytearray(t)
    b[30] = 1                                   # reserved
    bad.append(recrc(b))
    b = bytearray(t)
    b[18] = 0                                   # conf_min 0
    bad.append(recrc(b))
    b = bytearray(t)
    b[40] ^= 1                                  # CRC
    bad.append(bytes(b))
    b = bytearray(t)
    b[32:36] = b"XXXX"                          # neither VTDT nor VTMS
    bad.append(recrc(b))
    b = bytearray(m)
    b[19] = 2                                   # VTMS with 32 outputs under a 2-tag header
    bad.append(recrc(b))
    for blob in bad:
        with pytest.raises(ValueError):
            T6.Model(blob)
    # a multi-scale network must have 16 + 4 T a multiple of 8
    with pytest.raises(ValueError):
        M6.random_blob(TINY_M, 1, 3, n_cal=2)
    # old loaders refuse version 4
    for blob in (t, m):
        with pytest.raises(ValueError):
            T3.Model(blob)
        with pytest.raises(ValueError):
            T2.Model(blob)


# ---------------------------------------------------------------- version 4: the decode on hand-made maps

def _maps(h, w, tags):
    hd, wd = -(-h // 4), -(-w // 4)
    mp = np.zeros((4 + tags, hd, wd), np.int64)
    mp[0] = -100
    mp[2] = mp[3] = 4                          # A1 = A2 = 2 px: lines 4 px across at the input
    return mp


def _line(mp, row, c0, c1, tag_codes, k=64):
    mp[0, row, c0:c1] = k
    for t, v in enumerate(tag_codes):
        mp[4 + t, row, c0:c1] = v


def _model(tags):
    blob, _ = M6.random_blob(TINY_T, 1, tags, n_cal=2)
    m = T6.Model(blob)
    return m


def test_tag_decode_scores_ties_and_tconf():
    m = _model(4)
    h, w = 272, 480
    mp = _maps(h, w, 4)
    _line(mp, 10, 5, 40, [-16, -16, -16, -16])        # class 0 (all others below 0)
    _line(mp, 20, 5, 40, [32, -16, 0, 0])            # class 1, tconf = 32 - 0 = 32
    _line(mp, 30, 5, 40, [16, 16, 0, 0])             # tie 1 vs 2 -> 1, tconf 0
    _line(mp, 40, 5, 40, [0, 0, 0, 0])               # all 0 -> class 0, tconf 0
    _line(mp, 50, 5, 40, [-8, -8, 8, 24])            # class 4, tconf 16
    g = np.full((h, w), 128, np.uint8)
    r = T6.spot_maps(m, mp, 1, g)
    by_y = {b.y: b for b in r.boxes}
    tags = [by_y[k].tag for k in sorted(by_y)]
    tconfs = [by_y[k].tconf for k in sorted(by_y)]
    assert tags == [0, 1, 1, 0, 4]
    assert tconfs == [16, 32, 0, 0, 16]


def test_blocks_join_one_tag_only_and_pick_dialogue():
    m = _model(4)
    h, w = 272, 480
    mp = _maps(h, w, 4)
    # a name line above two dialogue lines, all close enough to join by geometry
    _line(mp, 50, 10, 20, [40, -10, -10, -10])       # name
    _line(mp, 53, 10, 60, [-10, -10, -10, -10])      # dialogue
    _line(mp, 56, 10, 60, [-10, -10, -10, -10])      # dialogue
    # a UI label in the lower half: big score under the old rule, but not dialogue
    _line(mp, 60, 70, 119, [-10, -10, 40, -10])      # ui (class 3)
    _line(mp, 63, 70, 119, [-10, -10, 40, -10])
    _line(mp, 66, 70, 119, [-10, -10, 40, -10])
    g = np.full((h, w), 128, np.uint8)
    r = T6.spot_maps(m, mp, 1, g)
    assert sorted(b.tag for b in r.boxes) == [0, 0, 1, 3, 3, 3]
    for blk in r.blocks:
        members = r.boxes[blk.first:blk.first + blk.count]
        assert {b.tag for b in members} == {blk.tag}
    assert r.pick >= 0 and r.blocks[r.pick].tag == 0 and r.blocks[r.pick].count == 2
    assert r.presence == 64
    # without any dialogue box: no pick, presence 0
    mp2 = _maps(h, w, 4)
    _line(mp2, 30, 10, 60, [40, -10, -10, -10])
    r2 = T6.spot_maps(m, mp2, 1, g)
    assert len(r2.boxes) == 1 and r2.boxes[0].tag == 1 and r2.pick == -1 and r2.presence == 0


def test_cap_key_and_truncation():
    m = _model(2)
    h, w = 272, 480
    mp = _maps(h, w, 2)
    for i in range(70):                               # 70 single-row lines of 3 cells
        row, col = 2 + 2 * (i // 8), 2 + 14 * (i % 8)
        _line(mp, row, col, col + 3, [(-1) ** i * 20, 0], k=1 + i)
    g = np.full((h, w), 128, np.uint8)
    r = T6.spot_maps(m, mp, 1, g)
    assert len(r.boxes) == 64 and r.truncated == 1
    assert min(b.conf for b in r.boxes) == 7         # the 6 least confident went


def test_cost_and_sizes():
    c = M6.cost(TINY_T, 480, 272, 4)
    blob, _ = M6.random_blob(TINY_T, 1, 4, n_cal=2)
    assert c["weights_int8"] > 0 and len(blob) > c["weights_int8"]
    assert M6.head_scales(4)[16:] == (1 / 16,) * 16
