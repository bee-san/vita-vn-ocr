"""Memory ownership and UTF-8 boundaries of the existing-Meiki C bindings."""
import ctypes as C
import random
import shutil
import subprocess
from pathlib import Path

import pytest

PORT = Path(__file__).resolve().parents[1] / "ports/meiki"


class Output(C.Structure):
    _fields_ = [("codes", C.c_int32 * 48), ("boxes", C.c_float * 192), ("scores", C.c_float * 48)]


@pytest.fixture(scope="module")
def binding(tmp_path_factory):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("C compiler unavailable")
    out = tmp_path_factory.mktemp("meiki-c") / "binding.so"
    subprocess.run([cc, "-std=c99", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                    "-fPIC", "-shared", str(PORT / "arena.c"), str(PORT / "decode.c"), "-o", str(out)], check=True)
    lib = C.CDLL(str(out))
    lib.meiki_arena_begin.argtypes = [C.c_void_p, C.c_size_t]
    lib.meiki_arena_alloc.argtypes = [C.c_size_t]
    lib.meiki_arena_alloc.restype = C.c_void_p
    lib.meiki_arena_free.argtypes = [C.c_void_p]
    lib.meiki_arena_used.restype = lib.meiki_arena_peak.restype = C.c_size_t
    lib.meiki_decode.argtypes = [C.POINTER(Output), C.c_uint, C.c_uint, C.c_void_p, C.c_size_t]
    return lib


@pytest.mark.parametrize("seed", [17, 81])
def test_arena_preserves_live_allocations_and_reclaims_whole_pool(binding, seed):
    size = 32768
    backing = C.create_string_buffer(b"\xa5" * (size + 192))
    address = (C.addressof(backing) + 127) & ~63
    before = C.string_at(address - 32, 32)
    after = C.string_at(address + size, 32)
    rng, live = random.Random(seed), []
    assert binding.meiki_arena_begin(address, size) == 0
    try:
        assert binding.meiki_arena_begin(address, size) == -1
        for _ in range(4000):
            if live and (rng.random() < .5 or len(live) >= 64):
                pointer, expected = live.pop(rng.randrange(len(live)))
                assert C.string_at(pointer, len(expected)) == expected
                binding.meiki_arena_free(pointer)
            else:
                n = rng.randrange(1, 1537)
                pointer = binding.meiki_arena_alloc(n)
                if pointer:
                    assert pointer % 64 == 0
                    assert address <= pointer and pointer + n <= address + size
                    expected = bytes([rng.randrange(256)]) * n
                    C.memmove(pointer, expected, n)
                    live.append((pointer, expected))
        for pointer, expected in live:
            assert C.string_at(pointer, len(expected)) == expected
            binding.meiki_arena_free(pointer)
        assert binding.meiki_arena_used() == 0
        pointer = binding.meiki_arena_alloc(size - 64)
        assert pointer
        assert not binding.meiki_arena_alloc(1)
        assert binding.meiki_arena_failed()
        binding.meiki_arena_free(pointer)
        assert binding.meiki_arena_used() == 0
        assert C.string_at(address - 32, 32) == before
        assert C.string_at(address + size, 32) == after
    finally:
        binding.meiki_arena_end()
    assert binding.meiki_arena_begin(address, size) == 0
    assert not binding.meiki_arena_failed()
    binding.meiki_arena_end()


def raw_text(text):
    raw = Output()
    for i, char in enumerate(text):
        raw.codes[i] = ord(char)
        raw.scores[i] = .9
        raw.boxes[4*i], raw.boxes[4*i+2] = i * 10, (i + 1) * 10
    return raw


@pytest.mark.parametrize("wrong,correct", [
    ("儡傀", "傀儡"), ("談冗", "冗談"), ("汰淘", "淘汰"), ("沱滂", "滂沱"),
    ("攣痙", "痙攣"), ("酊酩", "酩酊"), ("麭麺", "麺麭"), ("哭慟", "慟哭"),
])
def test_decoder_preserves_upstream_first_occurrence_pair_rule(binding, wrong, correct):
    raw = raw_text(wrong + wrong)
    text = C.create_string_buffer(193)
    assert binding.meiki_decode(C.byref(raw), 100, 100, text, len(text)) == 4
    assert text.value.decode() == correct + wrong


def test_decoder_does_not_write_a_partial_utf8_character(binding):
    raw = raw_text("あ")
    text = C.create_string_buffer(b"\xa5" * 4)
    assert binding.meiki_decode(C.byref(raw), 100, 100, text, 3) == -3
    assert text[0] == b"\0"
    assert text[3] == b"\xa5"
    assert binding.meiki_decode(C.byref(raw), 100, 100, text, 4) == 1
    assert text.value.decode() == "あ"
