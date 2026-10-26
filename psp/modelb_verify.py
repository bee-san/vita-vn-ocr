#!/usr/bin/env python3
"""Model B (vita-vn-ocr #126): the PSP build of a model B blob in PPSSPPHeadless against the host runtime.

  modelb_verify.py run --blob B.vocr --out DIR --eboot scalar=E.PBP [--eboot vfpu=V.PBP] [--plan dev=P.pkl ...]
                       [--regions dev|hval ...] [--mode w8f] [--head-gate on] [--two-level] [--tile 16] [--head 64]
                       [--chunk 2500]
      lines: every crop of each plan (eval/modelb_eval.py), unique by content, as grey crops in the console's view
             (a crop the plan reads inverted is stored as 255 - grey, which the runtime's VOCR_INVERT reads the same),
             in packs of --chunk crops; the host's expectations (psp/pack.py expected: the CRC-32 of every logit of
             the first run, the text of the timed runs, i.e. gated / two-level); one PPSSPP boot per pack and build
      regions: the plans' whole-dialogue regions (modelb_eval dev_regions / hval_regions) through the app's finder
             mode (VOCR_LINES_GRAD + vocr_refine_lines(VOCR_REFINE_DEFAULT), then every line), checked against the
             host's C finder + runtime: the page path end to end
      DIR/verify.json: per build and set, lines / regions, text and logit matches, emulated us and frames, the
      finder's emulated us, the arena and the stack the app measured. Exit status 1 if anything differs.
  modelb_verify.py budget --verify DIR/verify.json --blob B.vocr --size SIZE.txt --height H --out BUDGET.json
      the protocol's budgets: memory (blob + arena + code from `make -C psp size` + stack + the finder's arena for a
      480 x 272 screen) and the est. console seconds per nominal box (1.45 x (per-frame emulated time on the dev
      whole-dialogue finder crops x the nominal box's frames + the finder's emulated time per pixel x 480 x 90))

Everything here is emulated time (PPSSPP charges about one cycle per instruction, no cache), not console time; the
1.45 factor is psp/README.md's upper console bound for scalar W8F, an estimate. Packs hold copyrighted crops: keep DIR
local.
"""
import argparse
import hashlib
import json
import math
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (HERE, ROOT / "eval", ROOT / "runtime", ROOT / "synth", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import pack  # noqa: E402

FACTOR_HIGH = 1.45            # psp/README.md: console <= 1.45 x emulated (scalar W8F), an estimate
NOMINAL = {"lines": 3, "width": 320, "crop_height": 19, "region": (480, 90)}
FINDER_ARENA_480x272 = 25355  # vocr_lines_arena_bytes(480, 272), runtime/vocr_lines.h


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def nominal_frames(height, stride=8):
    w = int(round(NOMINAL["width"] * height / NOMINAL["crop_height"]))
    return NOMINAL["lines"] * math.ceil(w / stride)


def plan_crops(path):
    """Unique crops of a plan in the console's view (inverted crops stored as 255 - grey), with a tag per crop
    (its first row's component)."""
    with open(path, "rb") as f:
        plan = pickle.load(f)
    seen, crops, comps = set(), [], []
    for r in plan["rows"]:
        for g, inv in r["crops"]:
            c = np.ascontiguousarray(255 - g if inv else g, np.uint8)
            k = hashlib.sha256(c.tobytes() + repr(c.shape).encode()).digest()
            if k in seen:
                continue
            seen.add(k)
            crops.append(c)
            comps.append(r["comp"])
    return crops, comps


def boot(bundle_dir, stats, ppsspp, timeout, keep):
    cmd = [sys.executable, str(HERE / "ppsspp_cli.py"), "--bundle", str(bundle_dir), "--stats", str(stats),
           "--timeout", str(timeout), "--keep", str(keep)]
    if ppsspp:
        cmd += ["--ppsspp", ppsspp]
    p = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=timeout + 300)
    if not Path(stats).exists():
        raise SystemExit(f"PPSSPP run failed: {p.stdout[-1500:]} {p.stderr[-1500:]}")
    return json.loads(Path(stats).read_text()), p.returncode


def bundle(work, eboot, blob, crops, a, finder=-1):
    pk = work / "pack"
    pk.mkdir(parents=True, exist_ok=True)
    pack.write_pack(pk / "lines.bin", crops)
    args = ["bundle", "--blob", str(blob), "--mode", a.mode, "--pack", str(pk), "--out", str(work / "stick"),
            "--eboot", str(eboot), "--tile", str(a.tile), "--head", str(a.head), "--repeat", "1",
            "--head-gate", a.head_gate, "--finder", str(finder)]
    if a.two_level:
        args.append("--two-level")
    subprocess.run([sys.executable, str(HERE / "pack.py"), *args], check=True, cwd=ROOT, capture_output=True)
    return work / "stick"


def per_line(results, finder):
    """(frames, us, finder_us) per row of the app's results.txt."""
    out = []
    for ln in Path(results).read_text(encoding="utf-8").splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.split("\t")
        out.append((int(f[1]), int(f[3]), int(f[7]) if finder else 0))
    return out


def cmd_run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eboots = dict(e.split("=", 1) for e in a.eboot)
    rec = {"blob": str(a.blob), "blob_sha256": sha256(a.blob), "mode": a.mode, "head_gate": a.head_gate,
           "two_level": bool(a.two_level), "tile": a.tile, "head": a.head,
           "eboots": {k: sha256(v) for k, v in eboots.items()}, "sets": {}}
    jobs = []
    for spec in a.plan or []:
        name, path = spec.split("=", 1)
        crops, comps = plan_crops(path)
        if a.limit:
            crops, comps = crops[:a.limit], comps[:a.limit]
        rec["sets"][f"lines:{name}"] = {"plan": path, "plan_sha256": sha256(path), "crops": len(crops),
                                        "by_component": {c: comps.count(c) for c in sorted(set(comps))}}
        for k in range(0, len(crops), a.chunk):
            jobs.append((f"lines:{name}", k // a.chunk, crops[k:k + a.chunk], [c for c in comps[k:k + a.chunk]], -1))
    for name in a.regions or []:
        import modelb_eval as ME
        regs = ME.dev_regions() if name == "dev" else ME.hval_regions(a.pagepool, a.pages)
        imgs = [np.ascontiguousarray(r["image"], np.uint8) for r in regs][:a.limit or None]
        rec["sets"][f"regions:{name}"] = {"regions": len(imgs)}
        for k in range(0, len(imgs), a.chunk):
            jobs.append((f"regions:{name}", k // a.chunk, imgs[k:k + a.chunk], None, 0))
    bad = 0
    for build, eboot in eboots.items():
        for set_name, part, crops, comps, finder in jobs:
            work = out / build / f"{set_name.replace(':', '_')}_{part:03d}"
            if work.exists():
                shutil.rmtree(work)
            stick = bundle(work, eboot, a.blob, crops, a, finder)
            st, rc = boot(stick, work / "stats.json", a.ppsspp, a.timeout, work / "emu")
            res = work / "emu" / "vocr-emulator-app" / "results.txt"
            rows = per_line(res, finder >= 0) if res.exists() else []
            if res.exists():
                shutil.copyfile(res, work / "results.txt")
            ok = st["lines"] == len(crops) and st["text_ok"] == len(crops) and st["logits_ok"] == len(crops)
            bad += not ok
            e = rec["sets"][set_name].setdefault(build, {"parts": 0, "items": 0, "text_ok": 0, "logits_ok": 0,
                                                         "frames": 0, "us": 0, "finder_us": 0, "pixels": 0,
                                                         "by_component": {}, "arena_bytes": st["arena_bytes"],
                                                         "stack_used_max": 0, "gated": 0, "escaped": 0})
            e["parts"] += 1
            e["items"] += st["lines"]
            e["text_ok"] += st["text_ok"]
            e["logits_ok"] += st["logits_ok"]
            e["frames"] += st["frames"]
            e["us"] += st["us_total"]
            e["finder_us"] += st.get("finder_us_total", 0)
            e["gated"] += st.get("gated", 0)
            e["escaped"] += st.get("escaped", 0)
            e["stack_used_max"] = max(e["stack_used_max"], st["stack_used"])
            if finder >= 0:
                e["pixels"] += int(sum(c.shape[0] * c.shape[1] for c in crops))
            if comps:
                for c, (fr, us, _) in zip(comps, rows):
                    b = e["by_component"].setdefault(c, {"frames": 0, "us": 0})
                    b["frames"] += fr
                    b["us"] += us
            print(f"{build} {set_name} part {part}: {st['lines']} items, text {st['text_ok']}, logits "
                  f"{st['logits_ok']}, {st['us_total'] / max(st['frames'], 1):.1f} us/frame emulated "
                  f"(ppsspp rc {rc})", flush=True)
            (out / "verify.json").write_text(json.dumps(rec, indent=1) + "\n")
            if not a.keep:
                for d in ("stick", "pack", "emu"):
                    shutil.rmtree(work / d, ignore_errors=True)
    rec["all_identical"] = bad == 0
    (out / "verify.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"all_identical": rec["all_identical"]}))
    sys.exit(0 if bad == 0 else 1)


def code_bytes(size_txt):
    """text + data + bss of vocr.o, vocr_lines.o and vocr_lines_refine.o from `make -C psp size`'s psp-size table."""
    out = {}
    for ln in Path(size_txt).read_text().splitlines():
        f = ln.split()
        if len(f) >= 6 and f[-1] in ("vocr.o", "vocr_lines.o", "vocr_lines_refine.o") and f[0].isdigit():
            out[f[-1]] = int(f[0]) + int(f[1]) + int(f[2])
    if len(out) != 3:
        raise SystemExit(f"{size_txt}: no psp-size rows for the three objects")
    return out


def cmd_budget(a):
    v = json.loads(Path(a.verify).read_text())
    blob_bytes = Path(a.blob).stat().st_size
    if sha256(a.blob) != v["blob_sha256"]:
        raise SystemExit("the verify record is of another blob")
    proto = json.loads((ROOT / "eval/vocr6_psp_protocol.json").read_text())
    code = code_bytes(a.size)
    res = {"blob": a.blob, "blob_sha256": v["blob_sha256"], "settings": {k: v[k] for k in
                                                                         ("mode", "head_gate", "two_level", "tile",
                                                                          "head")}, "builds": {}}
    for build in v["eboots"]:
        wd = v["sets"]["lines:dev"][build]["by_component"]["wd_grad"]
        us_frame = wd["us"] / wd["frames"]
        regs = v["sets"].get("regions:dev", {}).get(build)
        finder_us_px = regs["finder_us"] / regs["pixels"] if regs and regs["pixels"] else None
        frames = nominal_frames(a.height)
        rx, ry = NOMINAL["region"]
        rec_us = us_frame * frames
        fin_us = finder_us_px * rx * ry if finder_us_px is not None else None
        emu = (rec_us + (fin_us or 0)) / 1e6
        arena = v["sets"]["lines:dev"][build]["arena_bytes"]
        stack = max(s[build]["stack_used_max"] for s in v["sets"].values() if build in s)
        stack_r = int(math.ceil(stack / 4096) * 4096)
        mem = blob_bytes + arena + sum(code.values()) + stack_r + FINDER_ARENA_480x272
        res["builds"][build] = {
            "us_per_frame_dev_wd": round(us_frame, 2), "nominal_frames": frames,
            "finder_us_per_pixel": round(finder_us_px, 5) if finder_us_px else None,
            "emulated_s_box": round(emu, 4), "est_console_s_box_high": round(FACTOR_HIGH * emu, 4),
            "est_console_s_box_low": round(emu, 4),
            "memory": {"blob": blob_bytes, "arena": arena, "code": code, "stack_used": stack, "stack": stack_r,
                       "finder_arena": FINDER_ARENA_480x272, "total": mem},
            "time_ok": FACTOR_HIGH * emu <= proto["budgets"]["time_s_le"],
            "memory_ok": mem <= proto["budgets"]["memory_bytes_le"]}
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--blob", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--eboot", action="append", required=True, help="NAME=PATH")
    r.add_argument("--plan", action="append", help="NAME=PLAN.pkl")
    r.add_argument("--regions", action="append", choices=["dev", "hval"])
    r.add_argument("--pagepool", default="")
    r.add_argument("--pages", default="")
    r.add_argument("--mode", default="w8f", choices=["f32", "w8f", "w8i"])
    r.add_argument("--head-gate", default="off", choices=["off", "on"])
    r.add_argument("--two-level", action="store_true")
    r.add_argument("--tile", type=int, default=16)
    r.add_argument("--head", type=int, default=64)
    r.add_argument("--chunk", type=int, default=2500)
    r.add_argument("--ppsspp", default="")
    r.add_argument("--timeout", type=int, default=7200)
    r.add_argument("--keep", action="store_true")
    r.add_argument("--limit", type=int, default=0, help="the first N crops / regions of each set (a smoke test)")
    b = sub.add_parser("budget")
    b.add_argument("--verify", required=True)
    b.add_argument("--blob", required=True)
    b.add_argument("--size", required=True, help="`make -C psp size` output")
    b.add_argument("--height", type=int, required=True)
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    {"run": cmd_run, "budget": cmd_budget}[a.cmd](a)


if __name__ == "__main__":
    main()
