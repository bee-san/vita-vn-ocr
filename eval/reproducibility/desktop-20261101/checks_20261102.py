"""Checks behind docs/desktop.md's C-path and luma statements (audit-vocr-desktop F3, N7, N8), on dev-v2 only (never a
protected set). Writes counts and timings only.

  checks_20261102.py --out CHECKS.json [--det-onnx M.onnx --det-vdsk M.vdsk --rec-vocr L.vocr --frames dev.pkl
                                        --c-boxes dev_final3c_long640.jsonl]

1. luma: tools/vocr_desktop.grey_of on 960 x 544 and 3840 x 2160 RGB frames (ms per call, with the 1-minute load
   before and after).
2. resampler: the release model's ONNX network and Python decode run on the C path's own resize of each frame
   (vocr_desk_resize) instead of OpenCV's; its crops against the C detector's crops (the c_parity run's boxes). If
   they are equal on every frame, the resampler alone explains the frames whose ONNX and C crops differ.
3. page: the whole C page (vocr_desk_page_run_rgb: detector, reading order, crops, vocr_line_dark, runtime/vocr.c W8I
   with the blob's head sections) against the harness's composition of the same parts (the C detector's boxes, the
   crop rule of eval/desktop.py with no pad, vocr_onnx.line_dark, eval/mida_det.CRec), frame by frame.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "tools", ROOT / "eval", ROOT / "synth", ROOT / "teacher", ROOT / "runtime", ROOT / "train"):
    sys.path.insert(0, str(_p))

HERD = os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr")
A = f"{HERD}/shared/artifacts/vocr-desktop"
PV = f"{HERD}/shared/private/vocr-desktop"


def luma(reps=50):
    import vocr_desktop as VD
    out = {}
    rng = np.random.default_rng(0)
    for w, h in ((960, 544), (3840, 2160)):
        rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        for _ in range(3):
            VD.grey_of(rgb)
        l0 = os.getloadavg()[0]
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            VD.grey_of(rgb)
            ts.append((time.perf_counter() - t) * 1e3)
        out[f"{w}x{h}"] = {"reps": reps, "ms_median": round(float(np.median(ts)), 2), "ms_min": round(min(ts), 2),
                           "ms_max": round(max(ts), 2), "load_1min_before": round(l0, 1),
                           "load_1min_after": round(os.getloadavg()[0], 1)}
    return out


def onnx_on_c_resize(det, cm, g):
    """Detector.detect with the C path's resize in place of OpenCV's (boxes in frame px)."""
    import vocr_desktop as VD
    H, W = g.shape
    f = VD.scale_for(det.size, W, H)
    nw, nh = max(1, int(round(W * f))), max(1, int(round(H * f)))
    r = cm.resize(g, nw, nh) if (nw, nh) != (W, H) else g
    x = np.pad(r, ((0, -nh % 32), (0, -nw % 32)), mode="edge")[None, None]
    boxes = VD.decode(det.maps(np.ascontiguousarray(x)), nw, nh, det.min_cells, det.conf_min_logit, det.pad_pct,
                      det.pad_min, det.min_across, det.tags)
    fx, fy = nw / W, nh / H
    fp = float(max(det.pad_frame_min, round(det.pad_frame * H / 544.0))) if det.pad_frame else 0.0
    out = []
    for b in boxes:
        x0, y0 = max(0.0, b.x / fx - fp), max(0.0, b.y / fy - fp)
        x1, y1 = min(float(W), (b.x + b.w) / fx + fp), min(float(H), (b.y + b.h) / fy + fp)
        if x1 > x0 and y1 > y0:
            out.append([x0, y0, x1, y1, b.vertical])
    return out


def crops(boxes, W, H):
    import vocr_desktop as VD
    return sorted(cb + (int(b[4]),) for b in boxes
                  for cb in [VD.crop_of((b[0], b[1], b[2] - b[0], b[3] - b[1]), W, H)] if cb)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--det-onnx", default=f"{A}/models/final3/vocr_desktop_det.onnx")
    ap.add_argument("--det-vdsk", default=f"{A}/models/final3/vocr_desktop_det.vdsk")
    ap.add_argument("--rec-vocr", default=f"{A}/models/VOCR6.2_L_w8_2l.vocr")
    ap.add_argument("--frames", default=f"{PV}/frames/dev.pkl")
    ap.add_argument("--c-boxes", default=f"{PV}/boxes/dev_final3c_long640.jsonl")
    ap.add_argument("--lib", default=None, help="desktop/build/libvocrdesk.so (default)")
    a = ap.parse_args(argv)
    import mida_det as MD
    import vocr_desktop as VD
    import vocr_onnx as V

    import desktop as D
    res = {"kind": "vocr-desktop-checks/1", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "repo_head": D.git_head(), "det_onnx_sha256": D.sha256_file(a.det_onnx),
           "det_vdsk_sha256": D.sha256_file(a.det_vdsk), "rec_sha256": D.sha256_file(a.rec_vocr),
           "frames_sha256": D.sha256_file(a.frames), "c_boxes_sha256": D.sha256_file(a.c_boxes),
           "luma": luma()}
    frames, _ = D.load_frames(a.frames)
    cboxes, _ = D.load_boxes(a.c_boxes)
    det = VD.Detector(a.det_onnx, threads=1)
    cm = VD.CModel(a.det_vdsk, lib=a.lib)
    page = VD.CDesk(a.det_vdsk, a.rec_vocr, threads=1, lib=a.lib)
    crec = MD.CRec(a.rec_vocr)
    same_resize = same_page = same_order = n = 0
    resize_diff_frames = []
    page_lines = harness_lines = 0
    for fr in frames:
        rgb = D.load_rgb(fr)
        g = VD.grey_of(rgb)
        H, W = g.shape
        n += 1
        cb = cboxes[fr["key"]]
        if crops(onnx_on_c_resize(det, cm, g), W, H) == crops(cb, W, H):
            same_resize += 1
        else:
            resize_diff_frames.append(n - 1)
        got = page.page(rgb)
        mine = [(tuple(box), dark, text) for text, box, _tag, dark, vertical in got if not vertical]
        want = []
        for b in cb:
            if b[4]:
                continue
            c = D.crop_box(b, 0, W, H)
            if c is None:
                continue
            x, y, w, h = c
            dark = int(V.line_dark(g, c))
            want.append((c, dark, crec.read(g[y:y + h, x:x + w], invert=bool(dark)).text))
        page_lines += len(mine)
        harness_lines += len(want)
        same_page += sorted(mine) == sorted(want)
        hor = [b for b in cb if not b[4] and D.crop_box(b, 0, W, H) is not None]
        order = VD.reading_order([(b[0], b[1], b[2] - b[0], b[3] - b[1]) for b in hor])
        same_order += [m[0] for m in mine] == [D.crop_box(hor[i], 0, W, H) for i in order]
    res["resampler"] = {"frames": n, "frames_same_crops_as_c": same_resize,
                        "note": "the ONNX network and Python decode on vocr_desk_resize's input, against the C "
                                "detector's crops"}
    res["page"] = {"frames": n, "frames_same_lines": same_page, "frames_same_order": same_order,
                   "lines_page": page_lines, "lines_harness": harness_lines,
                   "note": "vocr_desk_page_run_rgb's horizontal lines (crop, dark flag, text) against the harness's "
                           "composition of the C detector's boxes, eval/desktop.py's crop (no pad), "
                           "vocr_onnx.line_dark and eval/mida_det.CRec (runtime/vocr.c W8I, head sections)"}
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("luma", "resampler", "page")}), flush=True)


if __name__ == "__main__":
    main()
