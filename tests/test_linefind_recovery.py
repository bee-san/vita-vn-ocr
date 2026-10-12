"""Procedural semantic checks for optional textbox recovery; no screenshots or OCR."""
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval" / "linefind"))
sys.path.insert(0, str(ROOT / "tests"))
from cfinder import Finder  # noqa: E402

import linefind_parity  # noqa: E402

BASE = "2c028bf655af08b6a864630671c78c917fbadb62"
CC = shutil.which("cc")


@pytest.fixture(scope="module")
def recovery_driver(tmp_path_factory):
    if not CC:
        pytest.skip("no C compiler")
    out = tmp_path_factory.mktemp("recovery") / "lines"
    subprocess.run([CC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                    str(ROOT / "runtime/vocr_lines.c"), str(ROOT / "runtime/vocr_lines_test.c"),
                    "-o", str(out)], check=True)
    return out


def draw_row(image, top, left, height, width, ink, outline=None, sparse=False):
    """Glyph-like strokes, independent of fonts or a recognizer's labels."""
    yy, xx = np.mgrid[:height, :width]
    pitch = max(12, height + 4)
    cell = xx % pitch
    stroke = (cell % 4 == 0) | (yy % 4 == 0)
    if sparse:
        stroke = (cell == 2) | (yy == height // 2)
    mask = (cell < height) & stroke
    ys, xs = np.nonzero(mask)
    if outline is not None:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                y, x = top + ys + dy, left + xs + dx
                ok = (y >= 0) & (y < image.shape[0]) & (x >= 0) & (x < image.shape[1])
                image[y[ok], x[ok]] = outline
    image[top + ys, left + xs] = ink


def contains(box, ink_box):
    x, y, w, h = box
    a, b, c, d = ink_box
    return x <= a and y <= b and x + w >= a + c and y + h >= b + d


def scene_and_body(height=20, fraction=90, margin=0, dark=True):
    width = 21 * height
    header = 12 * height // 5
    top = header + 7 * height // 5 + margin
    paper, ink = (230, 20) if dark else (30, 240)
    image = np.full((top + 3 * height, width), paper, np.uint8)
    image[:header, :width * fraction // 100:4] = ink
    left, body_width = 3 * height, 14 * height
    draw_row(image, top, left, height, body_width, ink)
    return image, (left, top, body_width - 4, height)


def neighbour_fragment_image(height, margin, dark, fragment):
    image, body = scene_and_body(height, 90, margin, dark)
    left, top, _, _ = body
    ink = 20 if dark else 240
    if fragment == "stroke":
        image[top - 4 - height:top - 4, left + 4:left + 6] = ink
    else:
        ruby_height = max(8, height // 3)
        draw_row(image, top - 5 - ruby_height, left + 4, ruby_height, 6, ink)
    return image, body


def neighbour_tail_image(dark, side):
    image, body = scene_and_body(20, 90, 4, dark)
    left, top, _, height = body
    ink = 20 if dark else 240
    if side == "above":
        draw_row(image, top - 29, left + 4, height, 12, ink)
        image[top - 9:top - 4, left + 4:left + 6] = ink
    else:
        draw_row(image, top + height + 5, left + 4, 8, 6, ink)
    return image, body


def tiny_neighbour_glyph_image(dark, margin):
    image, body = scene_and_body(28, 90, margin, dark)
    left, top, _, _ = body
    draw_row(image, top - 13, left + 4, 8, 4, 20 if dark else 240)
    return image, body


def scaled_ring_image(dark):
    image, body = scene_and_body(28, 90, 4, dark)
    left, top, _, _ = body
    ink, paper = (20, 230) if dark else (240, 30)
    image[top - 7:top - 2, left + 3:left + 8] = ink
    image[top - 6:top - 3, left + 4:left + 7] = paper
    return image, body


@pytest.mark.parametrize("height", [12, 20, 28])
@pytest.mark.parametrize("fraction", [85, 89, 90, 92, 100])
@pytest.mark.parametrize("margin", [0, 4, 8])
@pytest.mark.parametrize("dark", [False, True])
def test_recovery_does_not_depend_on_scene_width_cutoff(recovery_driver, height, fraction, margin, dark):
    image, body = scene_and_body(height, fraction, margin, dark)
    old = Finder("textbox", binary=recovery_driver)
    expected = old(image)
    assert expected and not any(contains(box, body) for box in expected)
    finder = Finder("textbox_recovery", binary=recovery_driver)
    boxes = finder(image)
    assert len(boxes) == len(expected) + 1 and not finder.last_truncated
    assert all(box in boxes for box in expected)
    recovered = next(box for box in boxes if contains(box, body))
    assert finder.last_dark[boxes.index(recovered)] == int(dark)


@pytest.mark.parametrize("length", [1, 2])
@pytest.mark.parametrize("dark", [False, True])
def test_short_continuation_borrows_supported_row_scale(recovery_driver, length, dark):
    paper, ink = (230, 20) if dark else (30, 240)
    image = np.full((178, 520), paper, np.uint8)
    image[:70, ::4] = ink
    draw_row(image, 94, 70, 20, 280, ink)
    draw_row(image, 138, 70, 20, 24 * length, ink)
    image[146:150, 80 + 24 * length:83 + 24 * length] = ink
    old = Finder("textbox", binary=recovery_driver)
    expected = old(image)
    finder = Finder("textbox_recovery", binary=recovery_driver)
    boxes = finder(image)
    assert all(box in boxes for box in expected) and not finder.last_truncated
    assert sum(contains(box, (70, 94, 272, 20)) for box in boxes) == 1
    assert sum(contains(box, (70, 138, 24 * length - 4, 20)) for box in boxes) == 1
    assert len(boxes) == len(set(boxes))


@pytest.mark.parametrize("height", [8, 10, 12])
def test_detached_ruby_or_small_art_does_not_create_a_new_row(recovery_driver, height):
    image = np.full((116, 420), 30, np.uint8)
    draw_row(image, 45, 30, 26, 336, 240)
    draw_row(image, 12, 36, height, 140, 240)
    draw_row(image, 92, 46, 8, 112, 240)
    old = Finder("textbox", binary=recovery_driver)
    expected = old(image)
    finder = Finder("textbox_recovery", binary=recovery_driver)
    assert finder(image) == expected and finder.last_dark == old.last_dark


@pytest.mark.parametrize("width", [8, 18])
def test_old_single_narrow_glyph_remains_exact(recovery_driver, width):
    image = np.full((64, 80), 30, np.uint8)
    draw_row(image, 18, 30, 26, width, 240)
    old = Finder("textbox", binary=recovery_driver)
    expected = old(image)
    finder = Finder("textbox_recovery", binary=recovery_driver)
    assert expected and finder(image) == expected and finder.last_dark == old.last_dark


@pytest.mark.parametrize("gap", [0, 1, 2, 3, 4])
def test_close_rows_preserve_existing_geometry_and_do_not_duplicate(recovery_driver, gap):
    image = np.full((84, 320), 30, np.uint8)
    for top in (12, 32 + gap):
        draw_row(image, top, 24, 20, 260, 240)
    old = Finder("textbox", binary=recovery_driver)
    expected = old(image)
    finder = Finder("textbox_recovery", binary=recovery_driver)
    boxes = finder(image)
    assert boxes == expected and finder.last_dark == old.last_dark
    assert len(boxes) == len(set(boxes))


@pytest.mark.parametrize("side", ["left", "right", "top", "bottom"])
@pytest.mark.parametrize("margin", [0, 1, 2])
def test_boundary_contact_preserves_credible_old_crop(recovery_driver, side, margin):
    image = np.full((76, 320), 230, np.uint8)
    draw_row(image, 24, 20, 24, 264, 20)
    x1, x2, y1, y2 = 12, 290, 18, 56
    if side == "left":
        x1 = 20 - margin
    elif side == "right":
        x2 = 280 + margin
    elif side == "top":
        y1 = 24 - margin
    else:
        y2 = 48 + margin
    cropped = image[y1:y2, x1:x2]
    old = Finder("textbox", binary=recovery_driver)
    expected = old(cropped)
    finder = Finder("textbox_recovery", binary=recovery_driver)
    assert expected and finder(cropped) == expected and finder.last_dark == old.last_dark


def test_outlined_text_and_pixel_formats_remain_equivalent(recovery_driver):
    image = np.full((120, 360), 180, np.uint8)
    for top in (20, 65):
        draw_row(image, top, 28, 24, 300, 255, outline=20)
    old = Finder("textbox", binary=recovery_driver)
    expected = old(image)
    finder = Finder("textbox_recovery", binary=recovery_driver)
    assert finder(image) == expected and finder.last_dark == old.last_dark
    assert finder.last_dark == [0, 0]
    rgba = np.repeat(image[..., None], 4, axis=2)
    rgba[..., 3] = 7
    assert finder(rgba) == expected and finder.last_dark == [0, 0]


@pytest.mark.parametrize("sparse", [False, True])
def test_glyph_stroke_density_does_not_delete_old_rows(recovery_driver, sparse):
    image = np.full((120, 360), 30, np.uint8)
    for top in (20, 65):
        draw_row(image, top, 28, 24, 300, 240, sparse=sparse)
    old = Finder("textbox", binary=recovery_driver)
    expected = old(image)
    finder = Finder("textbox_recovery", binary=recovery_driver)
    boxes = finder(image)
    assert all(box in boxes for box in expected)
    assert len(boxes) == len(set(boxes)) and not finder.last_truncated


def test_inseparable_frame_and_conflicting_scales_fall_back(recovery_driver):
    image = np.full((160, 420), 230, np.uint8)
    image[:3] = 20
    image[:, :36:4] = 20
    draw_row(image, 62, 28, 20, 244, 20)
    old = Finder("textbox", binary=recovery_driver)
    expected = old(image)
    finder = Finder("textbox_recovery", binary=recovery_driver)
    assert finder(image) == expected and finder.last_dark == old.last_dark
    # Similar aspect support at incompatible heights gives no evidence for
    # replacing the old scale with the smaller lower component.
    image = np.full((164, 520), 30, np.uint8)
    draw_row(image, 12, 24, 36, 480, 240)
    draw_row(image, 94, 24, 16, 220, 240)
    expected = old(image)
    assert finder(image) == expected and finder.last_dark == old.last_dark


def test_many_fragments_share_one_physical_row(recovery_driver):
    image = np.full((120, 620), 230, np.uint8)
    for top in (20, 70):
        for left in range(28, 570, 36):
            draw_row(image, top, left, 24, 18, 20)
    old = Finder("textbox", binary=recovery_driver)
    expected = old(image)
    finder = Finder("textbox_recovery", binary=recovery_driver)
    boxes = finder(image)
    assert boxes == expected and len(boxes) == 2 and not finder.last_truncated


def test_existing_modes_match_exact_published_source_on_generated_inputs(recovery_driver, tmp_path):
    for name in ("vocr_lines.c", "vocr_lines.h", "vocr_lines_test.c"):
        data = subprocess.run(["git", "show", f"{BASE}:runtime/{name}"], cwd=ROOT,
                              check=True, capture_output=True).stdout
        (tmp_path / name).write_bytes(data)
    previous = tmp_path / "previous"
    subprocess.run([CC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                    str(tmp_path / "vocr_lines.c"), str(tmp_path / "vocr_lines_test.c"),
                    "-o", str(previous)], check=True)
    images = linefind_parity.make_cases(412, 320)
    images += [scene_and_body(h, f, m, d)[0] for h in (12, 20, 28)
               for f in (85, 89, 90, 92, 100) for m in (0, 4, 8) for d in (False, True)]
    for height in (12, 20, 28):
        outlined = np.full((144, 420), 180, np.uint8)
        for top in (20, 76):
            draw_row(outlined, top, 32, height, 330, 255, outline=20)
        images.append(outlined)
    for side in ("left", "right", "top", "bottom"):
        for margin in (0, 1, 2):
            boundary = np.full((76, 320), 230, np.uint8)
            draw_row(boundary, 24, 20, 24, 264, 20)
            x1, x2, y1, y2 = 12, 290, 18, 56
            if side == "left":
                x1 = 20 - margin
            elif side == "right":
                x2 = 280 + margin
            elif side == "top":
                y1 = 24 - margin
            else:
                y2 = 48 + margin
            images.append(boundary[y1:y2, x1:x2])
    images += [neighbour_fragment_image(h, m, d, f)[0] for h in (12, 20, 28)
               for m in (0, 4, 8) for d in (False, True) for f in ("stroke", "ruby")]
    images += [neighbour_tail_image(d, s)[0] for d in (False, True) for s in ("above", "below")]
    images += [tiny_neighbour_glyph_image(d, m)[0] for d in (False, True) for m in (0, 4, 8)]
    images += [scaled_ring_image(d)[0] for d in (False, True)]
    batch = tmp_path / "generated.bin"
    with batch.open("wb") as stream:
        for image in images:
            h, w = image.shape[:2]
            if image.ndim == 2:
                bpp, pixels = 1, image
            else:
                bpp = 4
                pixels = image if image.shape[2] == 4 else np.concatenate(
                    [image, np.full((h, w, 1), 255, np.uint8)], axis=2)
            for mode in (0, 1, 2):
                stream.write(struct.pack("<5I", w, h, bpp, mode, w * bpp))
                stream.write(np.ascontiguousarray(pixels).tobytes())
    actual = subprocess.run([str(recovery_driver), str(batch)], check=True, capture_output=True).stdout
    expected = subprocess.run([str(previous), str(batch)], check=True, capture_output=True).stdout
    assert len(actual.splitlines()) == len(images) * 3
    assert actual == expected


@pytest.mark.parametrize("height", [12, 20])
@pytest.mark.parametrize("mark", ["pair", "ring"])
def test_recovered_row_keeps_detached_small_marks(recovery_driver, height, mark):
    image, body = scene_and_body(height, 90, 4)
    left, top, _, _ = body
    x, y = left + 3, top - 6
    if mark == "pair":
        image[y, x] = image[y, x + 3] = 20
        mark_box = (x, y, 4, 1)
    else:
        image[y:y + 3, x:x + 3] = 20
        image[y + 1, x + 1] = 230
        mark_box = (x, y, 3, 3)
    finder = Finder("textbox_recovery", binary=recovery_driver)
    boxes = finder(image)
    cropped = next(box for box in boxes if contains(box, body))
    assert contains(cropped, mark_box)


def test_recovered_row_keeps_small_kana_and_detached_quotes(recovery_driver):
    image, body = scene_and_body(20, 90, 4)
    left, top, width, height = body
    draw_row(image, top + 9, left + width + 12, 10, 8, 20)
    image[top + 1:top + 7, left - 14:left - 12] = 20
    image[top + 1:top + 3, left - 14:left - 7] = 20
    image[top + 12:top + 14, left + width + 34:left + width + 36] = 20
    finder = Finder("textbox_recovery", binary=recovery_driver)
    boxes = finder(image)
    cropped = next(box for box in boxes if contains(box, body))
    assert contains(cropped, (left - 14, top + 1, 7, 6))
    assert contains(cropped, (left + width + 12, top + 9, 8, 10))
    assert contains(cropped, (left + width + 34, top + 12, 2, 2))


def test_unaligned_small_art_cannot_borrow_a_dialogue_row(recovery_driver):
    image = np.full((144, 960), 30, np.uint8)
    draw_row(image, 36, 160, 24, 740, 240)
    draw_row(image, 90, 16, 24, 18, 240)
    old = Finder("textbox", binary=recovery_driver)
    expected = old(image)
    finder = Finder("textbox_recovery", binary=recovery_driver)
    assert len(expected) == 1 and finder(image) == expected and finder.last_dark == old.last_dark


def test_recovered_outlined_text_keeps_light_polarity(recovery_driver):
    image = np.full((154, 520), 180, np.uint8)
    image[:60, ::4] = 20
    draw_row(image, 96, 64, 24, 390, 255, outline=20)
    old = Finder("textbox", binary=recovery_driver)
    expected = old(image)
    finder = Finder("textbox_recovery", binary=recovery_driver)
    boxes = finder(image)
    assert all(box in boxes for box in expected)
    recovered = next(box for box in boxes if contains(box, (64, 96, 384, 24)))
    assert finder.last_dark[boxes.index(recovered)] == 0


def test_c_recovery_exact_arena_safety(recovery_driver, tmp_path):
    safety = tmp_path / "recovery_safety"
    subprocess.run([CC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-Wpedantic",
                    "-Wshadow", "-Wstrict-prototypes", "-Wmissing-prototypes", "-Wcast-align",
                    "-Wpointer-arith", str(ROOT / "runtime/vocr_lines.c"),
                    str(ROOT / "runtime/tests/test_vocr_lines_recovery.c"), "-o", str(safety)], check=True)
    result = subprocess.run([str(safety)], check=True, capture_output=True, text=True)
    assert "RECOVERY_SAFETY_OK 2400 generated images" in result.stdout


@pytest.mark.parametrize("height", [12, 20, 28])
@pytest.mark.parametrize("margin", [0, 4, 8])
@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("fragment", ["stroke", "ruby"])
def test_incomplete_neighbour_fragment_does_not_expand_recovered_body(
        recovery_driver, height, margin, dark, fragment):
    image, body = scene_and_body(height, 90, margin, dark)
    finder = Finder("textbox_recovery", binary=recovery_driver)
    baseline = next(box for box in finder(image) if contains(box, body))
    image, body = neighbour_fragment_image(height, margin, dark, fragment)
    old = Finder("textbox", binary=recovery_driver)
    expected = old(image)
    boxes = finder(image)
    assert all(box in boxes for box in expected)
    recovered = next(box for box in boxes if contains(box, body))
    assert recovered == baseline
    assert len(boxes) == len(set(boxes)) and not finder.last_truncated


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("side", ["above", "below"])
def test_neighbour_tail_and_next_short_row_cannot_be_attached(recovery_driver, dark, side):
    image, body = scene_and_body(20, 90, 4, dark)
    finder = Finder("textbox_recovery", binary=recovery_driver)
    baseline = next(box for box in finder(image) if contains(box, body))
    # A neighbouring glyph's tail cannot change the scale or enlarge this row.
    image, body = neighbour_tail_image(dark, side)
    old = Finder("textbox", binary=recovery_driver)
    expected = old(image)
    boxes = finder(image)
    assert all(box in boxes for box in expected)
    recovered = next(box for box in boxes if contains(box, body))
    assert recovered == baseline
    assert len(boxes) == len(set(boxes)) and not finder.last_truncated


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("margin", [0, 4, 8])
def test_a_complete_tiny_glyph_is_too_tall_to_be_an_attached_mark(recovery_driver, dark, margin):
    image, body = scene_and_body(28, 90, margin, dark)
    finder = Finder("textbox_recovery", binary=recovery_driver)
    baseline = next(box for box in finder(image) if contains(box, body))
    # Its width is below the component run minimum; completeness alone does not
    # make an eight-pixel high neighbouring glyph a detached accent stroke.
    image, body = tiny_neighbour_glyph_image(dark, margin)
    boxes = finder(image)
    recovered = next(box for box in boxes if contains(box, body))
    assert recovered == baseline


@pytest.mark.parametrize("dark", [False, True])
def test_scaled_complete_ring_mark_is_preserved(recovery_driver, dark):
    image, body = scaled_ring_image(dark)
    left, top, _, _ = body
    finder = Finder("textbox_recovery", binary=recovery_driver)
    recovered = next(box for box in finder(image) if contains(box, body))
    assert contains(recovered, (left + 3, top - 7, 5, 5))
