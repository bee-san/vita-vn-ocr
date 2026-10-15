"""eval/lens_lines.py: the Lens reader's cache, retries, resume and the cmd: hand-over to eval/run.py.

A stand-in bridge replaces eval/owocr_bridge.py: no network, no screenshots, no Google request.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

EVAL = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL))
import backends  # noqa: E402
import lens_lines  # noqa: E402
import sets  # noqa: E402

# Answers "L<first pixel>-<width>" for a crop; a crop whose first pixel is 7 fails its first attempt, one whose
# first pixel is 9 always fails, one whose first pixel is 5 reads as two Lens lines.
FAKE_BRIDGE = r'''
import json, sys
from PIL import Image
if "--info" in sys.argv:
    print(json.dumps({"dependencies": {"owocr": "1.26.7"}, "bridge_sha256": "fake"}))
    sys.exit(0)
tries = {}
for raw in sys.stdin:
    req = json.loads(raw)
    with Image.open(req["path"]) as im:
        im.load()
        first, w = im.getpixel((0, 0))[0], im.size[0]
    tries[req["pixels_sha256"]] = tries.get(req["pixels_sha256"], 0) + 1
    if first == 9 or (first == 7 and tries[req["pixels_sha256"]] == 1):
        out = {"id": req["id"], "ok": False, "error": "RuntimeError: fake failure"}
    else:
        text = f"L{first}-{w}" + ("\nsecond" if first == 5 else "")
        out = {"id": req["id"], "ok": True, "text": text, "lines": [{"text": t, "box": None} for t in text.split("\n")]}
    out["requested_at_unix"] = 0.0
    out["host_seconds"] = 0.0
    print(json.dumps(out), flush=True)
'''


def line(i, first, w=12, h=6):
    rgb = np.full((h, w, 3), 100, np.uint8)
    rgb[0, 0] = first
    return sets.Line(id=f"t_{i}", set="vita-test", vn="v1", crop_rgb=rgb)


def bridge(tmp_path):
    fake = tmp_path / "fake_bridge.py"
    fake.write_text(FAKE_BRIDGE)
    log = open(tmp_path / "bridge.log", "a")
    return lens_lines.Bridge([sys.executable, str(fake)], log), log


def read(tmp_path, lines, out):
    b, log = bridge(tmp_path)
    try:
        return lens_lines.read_lines(lines, out, b, "1.26.7", "fake", sleep_s=0, backoff_s=0, retries=3)
    finally:
        b.close()
        log.close()


def test_crop_key_is_stable_and_shape_aware():
    a = np.zeros((2, 6, 3), np.uint8)
    assert lens_lines.crop_key(a) == lens_lines.crop_key(a.copy())
    assert lens_lines.crop_key(a) != lens_lines.crop_key(np.zeros((3, 4, 3), np.uint8))   # same bytes, other shape


def test_read_retries_resumes_and_never_rereads(tmp_path):
    out = tmp_path / "cache.jsonl"
    lines = [line(0, 1), line(1, 7), line(2, 9), line(3, 5, w=20)]
    req, ok, failed = read(tmp_path, lines, out)
    assert (ok, failed) == (3, 1) and req == 1 + 2 + 3 + 1
    rows = lens_lines.load_cache(out)
    assert rows["t_0"]["ok"] and rows["t_0"]["text"] == "L1-12" and rows["t_0"]["attempts"] == 1
    assert rows["t_1"]["ok"] and rows["t_1"]["attempts"] == 2                   # retried after one failure
    assert not rows["t_2"]["ok"] and rows["t_2"]["attempts"] == 3 and rows["t_2"]["text"] is None
    assert rows["t_3"]["lines"] == [{"text": "L5-20", "box": None}, {"text": "second", "box": None}]
    assert rows["t_0"]["crop_sha256"] == lens_lines.crop_key(lines[0].crop_rgb)
    assert rows["t_0"]["crop_wh"] == [12, 6] and rows["t_0"]["owocr"] == "1.26.7"
    # a resumed run sends only the failed line again; its new row supersedes the old one
    req, ok, failed = read(tmp_path, lines, out)
    assert (req, ok, failed) == (3, 0, 1)
    assert len(out.read_text().splitlines()) == 5 and not lens_lines.load_cache(out)["t_2"]["ok"]


def write_ppm(path, rgb):
    path.write_bytes(b"P6\n%d %d\n255\n" % (rgb.shape[1], rgb.shape[0]) + np.ascontiguousarray(rgb).tobytes())


def serve(cache, paths, *extra):
    return subprocess.run([sys.executable, str(EVAL / "lens_lines.py"), "serve", "--cache", str(cache), *extra],
                          input="".join(f"{p}\n" for p in paths), capture_output=True, text=True)


def test_serve_answers_in_order_by_crop_hash(tmp_path):
    out = tmp_path / "cache.jsonl"
    lines = [line(0, 1), line(1, 5, w=20), line(2, 3)]
    read(tmp_path, lines, out)
    paths = []
    for ln in reversed(lines):                                  # order follows stdin, not the cache
        p = tmp_path / f"{ln.id}.ppm"
        write_ppm(p, ln.crop_rgb)
        paths.append(p)
    r = serve(out, paths)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split("\n") == ["L3-12", "L5-20 second", "L1-12", ""]   # a two-line reading joins with a space


def test_serve_refuses_crops_without_a_reading(tmp_path):
    out = tmp_path / "cache.jsonl"
    read(tmp_path, [line(0, 1), line(1, 9)], out)
    p0, p1 = tmp_path / "a.ppm", tmp_path / "b.ppm"
    write_ppm(p0, line(0, 1).crop_rgb)
    write_ppm(p1, line(1, 9).crop_rgb)                          # failed in the cache
    r = serve(out, [p0, p1])
    assert r.returncode == 3 and "1 crops have no OK Lens reading" in r.stderr and r.stdout == ""
    r = serve(out, [p0, p1], "--missing-empty")
    assert r.returncode == 0 and r.stdout.split("\n") == ["L1-12", "", ""]


def test_eval_cmd_backend_reads_the_cache(tmp_path):
    """The exact hand-over eval/run.py uses: backends.Cmd writes binary PPMs and reads one line per crop."""
    out = tmp_path / "cache.jsonl"
    lines = [line(0, 1), line(1, 4, w=16)]
    read(tmp_path, lines, out)
    rec = backends.make_recognizer(f"cmd:{sys.executable} {EVAL / 'lens_lines.py'} serve --cache {out}")
    assert rec.read([ln.crop_rgb for ln in lines]) == ["L1-12", "L4-16"]


def test_read_only_allows_the_public_sets():
    assert lens_lines.SETS == ("vita-test", "psp-test")
    r = subprocess.run([sys.executable, str(EVAL / "lens_lines.py"), "read", "--set", "clannad-test", "--out", "x",
                        "--owocr-python", sys.executable], capture_output=True, text=True)
    assert r.returncode == 2 and "invalid choice" in r.stderr


@pytest.mark.parametrize("name", ["vita-test", "psp-test"])
def test_pad_matches_eval_run(name):
    """read() crops through sets.load_real with the default pad, as eval/run.py's recognizer mode does."""
    assert sets.PAD[sets.SETS[name]] == {"vita-test": 4, "psp-test": 2}[name]
    assert json.loads((EVAL / "manifests" / f"{name}.jsonl").read_text().splitlines()[0])["box"]
