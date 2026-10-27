#!/usr/bin/env python3
"""textspot8's animated-icon check (report only; docs/textspot8.md "Spinners"): does a finder box a spinner or a page
cursor drawn next to the dialogue, which makes Rubi's line checksum unstable (a user's report: a spinner inside the
OCR region kept the text from stabilising), and what does the icon cost the page path?

  textspot8_spinner.py build --out DIR [--seed 0]
      every frame of #119's whole-dialogue dev units with one icon drawn after the end of the frame's last dialogue
      line of each region (the bottom line, at 0.35 of the line height after its end, its size 0.8 of the line
      height, vertically centred; the text's polarity), in two styles: "trained" (train/textspot8_train.ICON_KINDS,
      the shapes textspot8's student saw as background) and "unseen" (three shapes no spotter trained on: a
      four-blade pinwheel, a double ring, a three-dot bar). A unit enters the set when the icon lies inside its
      region. Writes DIR/<style>/units.pkl (every unit of #119's file in its order, the set's units with their
      modified images: eval/pagepath.py's uids pair with the stored results),
      DIR/<style>/icons.json ({uid: [x0, y0, x1, y1] in the unit's region px at its view scale}) and
      DIR/<style>/frames.npz (the modified console-view frames, keyed as eval/textspot6.dev_frames)
  textspot8_spinner.py spot --blob B.vtsp --dir DIR/<style> --out SPOT.jsonl [--workers 5]
      the spotter (the twin) on the modified frames: eval/textspot6.py's spot file
  textspot8_spinner.py overlap --rows ROWS.jsonl --icons DIR/<style>/icons.json --out RES.json
      per unit of a page-path rows file: does any box cover at least half of the icon's box (the icon inside a line
      box: a checksum over the line boxes changes as it spins), and how many boxes hold nothing but the icon

Everything here is private (game frames); icons.json and RES.json hold coordinates and counts only.
"""
import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "train", ROOT / "teacher"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

UNSEEN = ("pinwheel", "double_ring", "dot_bar")


def draw_unseen(img, kind, cx, cy, size, value, rng):
    import cv2
    r = max(2.0, size / 2.0)
    v = int(value)
    if kind == "pinwheel":
        a0 = rng.uniform(0, 2 * math.pi)
        for k in range(4):
            a = a0 + k * math.pi / 2
            pts = np.array([[cx, cy], [cx + r * math.cos(a), cy + r * math.sin(a)],
                            [cx + r * math.cos(a + 0.6), cy + r * math.sin(a + 0.6)]])
            cv2.fillPoly(img, [np.round(pts).astype(np.int32)], v, lineType=cv2.LINE_AA)
    elif kind == "double_ring":
        cv2.circle(img, (int(round(cx)), int(round(cy))), max(2, int(r * 0.85)), v, 1, lineType=cv2.LINE_AA)
        cv2.circle(img, (int(round(cx)), int(round(cy))), max(1, int(r * 0.45)), v, 1, lineType=cv2.LINE_AA)
    elif kind == "dot_bar":
        for k in (-1, 0, 1):
            cv2.circle(img, (int(round(cx + k * 0.6 * r)), int(round(cy))), max(1, int(r * 0.22)), v, -1,
                       lineType=cv2.LINE_AA)
    else:
        raise ValueError(kind)


def icon_for(style, rng):
    import textspot8_train as T8T
    kinds = T8T.ICON_KINDS if style == "trained" else UNSEEN
    return kinds[int(rng.integers(len(kinds)))]


def draw(img, style, kind, cx, cy, size, value, rng):
    import textspot8_train as T8T
    if style == "trained":
        T8T.draw_icon(img, kind, cx, cy, size, value, rng)
    else:
        draw_unseen(img, kind, cx, cy, size, value, rng)


def native_grey_frames():
    """{frame key: (key dict, native grey frame, scale s of its console view)} of #119's dev units, rebuilt and
    checked as eval/textspot6.dev_frames does, but at native size (a PSP frame not upscaled)."""
    import glob

    import cv2
    import geometry as G
    import textspot6 as E6

    from common import to_gray
    units = E6.dev_units()
    by = {}
    for u in units:
        by.setdefault(E6.dev_frame_key(u), []).append(u)
    out = {}
    for key, us in sorted(by.items()):
        u0 = us[0]
        g = None
        for p in [p for d in E6.SHOT_DIRS[u0["src_kind"]] for p in glob.glob(f"{d}/{u0['shot']}.*")]:
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            if u0["src_kind"] == "hires":
                rgb = G.scale_image(rgb, float(u0["scale"]))
            g = to_gray(np.ascontiguousarray(rgb))
            break
        if g is None:
            raise FileNotFoundError(u0["shot"])
        for u in us:
            x0, y0, x1, y1 = u["region"]
            if not (g[y0:y1, x0:x1] == u["image"]).all():
                raise SystemExit(f"{u['unit']}: the rebuilt frame's region is not the unit's image")
        out[key] = ({"frame": key, "view": "psp2x" if u0["kind"] == "psp" else "vita", "shot": u0["shot"],
                     "vn": u0["vn"]}, g, 2 if u0["kind"] == "psp" else 1)
    return out


def cmd_build(args):
    import cv2
    import textspot6 as E6

    from common import line_dark
    units = pickle.load(open(E6.DEV_UNITS, "rb"))
    frames = native_grey_frames()
    by_frame = {}
    for i, u in enumerate(units["units"]):
        by_frame.setdefault(E6.dev_frame_key(u), []).append((i, u))
    for style in ("trained", "unseen"):
        rng = np.random.default_rng(args.seed)
        out_units, icons, det_frames = list(units["units"]), {}, []
        for fkey, us in sorted(by_frame.items()):
            k, g0, s = frames[fkey]
            g = np.array(g0, copy=True)
            drawn = {}
            for i, u in us:                     # one icon per region (its base unit), after its bottom line's end
                if u["variant"] != "base":
                    continue
                x0r, y0r = u["region"][0], u["region"][1]
                last = max(u["lines"], key=lambda ln: (ln["box"][1] + ln["box"][3]) / 2)
                bx0, by0, bx1, by1 = last["box"]                           # region px, native
                lh = by1 - by0
                size = 0.8 * lh
                cx = x0r + bx1 + 0.35 * lh + size / 2
                cy = y0r + (by0 + by1) / 2
                ib = [int(cx - size / 2) - 1, int(cy - size / 2) - 1, int(cx + size / 2) + 2, int(cy + size / 2) + 2]
                if ib[0] < 0 or ib[1] < 0 or ib[2] > g.shape[1] or ib[3] > g.shape[0]:
                    continue
                crop = g0[y0r + by0:y0r + by1, x0r + bx0:x0r + bx1]
                dark = line_dark(crop, (0, 0, crop.shape[1], crop.shape[0])) if crop.size else 0
                kind = icon_for(style, rng)
                draw(g, style, kind, cx, cy, size, 25 if dark else 235, rng)
                drawn[u["unit"]] = (ib, kind)
            gv = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR) if s == 2 else g
            det_frames.append((k, np.ascontiguousarray(gv)))
            for i, u in us:
                if u["unit"] not in drawn:
                    continue
                ib, kind = drawn[u["unit"]]
                x0, y0, x1, y1 = u["region"]
                if not (ib[0] >= x0 and ib[1] >= y0 and ib[2] <= x1 and ib[3] <= y1):
                    continue                    # the icon is outside this unit's region: not in the set
                out_units[i] = dict(u, image=np.ascontiguousarray(g[y0:y1, x0:x1]), icon={"kind": kind, "style": style})
                icons[f"dev:{i}:{u['unit']}:{u['variant']}"] = [(ib[0] - x0) * s, (ib[1] - y0) * s, (ib[2] - x0) * s,
                                                               (ib[3] - y0) * s]
        d = Path(args.out) / style
        d.mkdir(parents=True, exist_ok=True)
        pickle.dump({"fixed": units.get("fixed"), "units": out_units}, open(d / "units.pkl", "wb"))
        (d / "icons.json").write_text(json.dumps(icons) + "\n", encoding="utf-8")
        np.savez_compressed(d / "frames.npz", **{f"f{j}": gv for j, (k, gv) in enumerate(det_frames)},
                            keys=np.array([json.dumps(k) for k, gv in det_frames]))
        print(f"textspot8 spinner set written: {style} {len(icons)} of {len(out_units)} units with an icon, "
              f"{len(det_frames)} frames -> {d}",
              flush=True)


def load_frames(d):
    z = np.load(Path(d) / "frames.npz")
    keys = [json.loads(k) for k in z["keys"]]
    return [(k, z[f"f{j}"]) for j, k in enumerate(keys)]


def cmd_spot(args):
    import textspot6 as E6
    jobs = load_frames(args.dir)
    E6.write_spot(args.blob, jobs, args.out, "dev-spinner", args.workers)
    print(f"textspot8 spinner spot written: {len(jobs)} frames -> {args.out}", flush=True)


def cmd_overlap(args):
    icons = json.loads(Path(args.icons).read_text(encoding="utf-8"))
    n = covered = junk = errors = chars = 0
    for line in open(args.rows, encoding="utf-8"):
        r = json.loads(line)
        ib = icons.get(r["uid"])
        if ib is None:
            continue
        n += 1
        ia = max(1, (ib[2] - ib[0]) * (ib[3] - ib[1]))
        cov = False
        for x, y, w, h in r["boxes_xywh"]:
            ox = max(0, min(ib[2], x + w) - max(ib[0], x))
            oy = max(0, min(ib[3], y + h) - max(ib[1], y))
            if ox * oy >= 0.5 * ia:
                cov = True
        covered += cov
        junk += int(r.get("junk", 0))
        errors += int(r["errors"])
        chars += int(r["chars"])
    out = {"rows": str(args.rows), "units": n, "icon_inside_a_box": covered,
           "share": round(covered / n, 4) if n else None, "junk_boxes": junk, "errors": errors, "chars": chars,
           "cer": round(100 * errors / chars, 4) if chars else None}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"textspot8 spinner overlap: {covered}/{n} units with the icon inside a box", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("spot")
    s.add_argument("--blob", required=True)
    s.add_argument("--dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=5)
    o = sub.add_parser("overlap")
    o.add_argument("--rows", required=True)
    o.add_argument("--icons", required=True)
    o.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    t0 = time.time()
    {"build": cmd_build, "spot": cmd_spot, "overlap": cmd_overlap}[args.cmd](args)
    print(f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
