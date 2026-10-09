#!/usr/bin/env python3
"""Run the PSP test app (psp/EBOOT.PBP) in PPSSPPHeadless with vocr_cli's command line.

  ppsspp_cli.py MODEL.vocr [opts] IMAGE.pgm|ppm...  one text line per image on stdout
  ppsspp_cli.py MODEL.vocr [opts] --stdin           image paths on stdin, one per line
  ppsspp_cli.py MODEL.vocr [opts] --pack LINES.bin  many crops in one file

vocr_cli's options, with the same meaning and output formats (runtime/tools/vocr_cli.c):
  --mode auto|f32|w8f|w8i  --tile N  --head N  --max-width N  --invert  --crc
  --logits OUT.bin  --input-out OUT.bin  --repeat N  --arena W[,W...]
so it drops in wherever vocr_cli does:
  runtime/tests/parity.py --ckpt CKPT --set psp-test --cli "python psp/ppsspp_cli.py"
  eval/run.py --model 'cmd:python psp/ppsspp_cli.py v1_w8.vocr --mode w8i --stdin' --set psp-test
Own options: --ppsspp PATH ($PPSSPP, default psp/tools/ppsspp/PPSSPPHeadless), --eboot PATH
($PSP_EBOOT, default psp/EBOOT.PBP), --timeout S ($PPSSPP_TIMEOUT, default 3600, wall clock),
--cpu jit|ir|interp (PPSSPP's CPU core, default jit), --clock MHz (default 333), --stack-kb N,
--stats OUT.json (the app's stats.json), --keep DIR (keep the app directory there).

Every call boots the EBOOT once, in a fresh memory stick ($HOME/.ppsspp of a temporary HOME),
from ms0:/PSP/GAME/VOCR/, exactly the layout on a PSP. The texts, logits and preprocessed
inputs are what the PSP build of vocr.c computed. The ms/line it prints is PPSSPP's emulated
time: PPSSPP charges about one cycle per MIPS instruction, with no cache or pipeline model, so
it is not the console's speed.
"""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPU = {"jit": "-j", "ir": "--ir", "interp": "-i"}


def read_pnm(path):
    """Binary PGM (P5) or PPM (P6), maxval 255 -> (h, w, channels, bytes); PPM becomes RGBA
    with alpha 255, as vocr_cli does."""
    raw = Path(path).read_bytes()
    if raw[:2] not in (b"P5", b"P6"):
        raise ValueError(f"{path}: not a binary PGM/PPM")
    vals, i = [], 2
    while len(vals) < 3:
        while raw[i:i + 1].isspace() or raw[i:i + 1] == b"#":
            if raw[i:i + 1] == b"#":
                while raw[i:i + 1] not in (b"\n", b""):
                    i += 1
            else:
                i += 1
        j = i
        while raw[j:j + 1].isdigit():
            j += 1
        vals.append(int(raw[i:j]))
        i = j
    w, h, maxval = vals
    if maxval != 255:
        raise ValueError(f"{path}: maxval {maxval}")
    px = raw[i + 1:]
    if raw[:2] == b"P5":
        return h, w, 1, px[:w * h]
    rgb = px[:3 * w * h]
    out = bytearray(4 * w * h)
    out[0::4], out[1::4], out[2::4] = rgb[0::3], rgb[1::3], rgb[2::3]
    out[3::4] = b"\xff" * (w * h)
    return h, w, 4, bytes(out)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    ap.add_argument("model", nargs="?")
    ap.add_argument("images", nargs="*")
    ap.add_argument("--bundle", help="run a pack.py bundle (its vocr.cfg and expect.txt) instead")
    ap.add_argument("--mode", default="auto", choices=["auto", "f32", "w8f", "w8i"])
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--max-width", type=int, default=0)
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--crc", action="store_true")
    ap.add_argument("--logits")
    ap.add_argument("--input-out")
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--arena")
    ap.add_argument("--pack")
    ap.add_argument("--stdin", action="store_true")
    ap.add_argument("--ppsspp", default=os.environ.get("PPSSPP", str(HERE / "tools" / "ppsspp" / "PPSSPPHeadless")))
    ap.add_argument("--eboot", default=os.environ.get("PSP_EBOOT", str(HERE / "EBOOT.PBP")))
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("PPSSPP_TIMEOUT", "3600")))
    ap.add_argument("--cpu", default=os.environ.get("PPSSPP_CPU", "jit"), choices=sorted(CPU))
    ap.add_argument("--clock", type=int, default=333)
    ap.add_argument("--stack-kb", type=int, default=16)
    ap.add_argument("--stats")
    ap.add_argument("--keep")
    return ap.parse_args(argv)


def write_images_pack(path, images):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(images)))
        for p in images:
            h, w, ch, px = read_pnm(p)
            f.write(struct.pack("<III", h, w, ch))
            f.write(px)


def make_cfg(a):
    """The app's vocr.cfg (psp/main.c, cfg_set) for vocr_cli-style options."""
    cfg = {"mode": a.mode, "tile": a.tile, "head": a.head, "max_width": a.max_width,
           "invert": int(a.invert), "crc": int(a.crc), "repeat": a.repeat, "clock": a.clock,
           "stack_kb": a.stack_kb, "wait": 0, "expect": "", "pack": "lines.bin"}
    if a.logits:
        cfg["logits"] = "logits.bin"
    if a.input_out:
        cfg["input_out"] = "input.bin"
    if a.arena:
        cfg["arena"] = a.arena
    return cfg


def boot(a, home, app):
    """Boot app/EBOOT.PBP in PPSSPPHeadless with HOME=home; returns (results.txt, log)."""
    env = dict(os.environ, HOME=str(home))
    cmd = [a.ppsspp, CPU[a.cpu], f"--timeout={a.timeout:g}", str(app / "EBOOT.PBP")]
    p = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                       timeout=a.timeout + 120)
    log = p.stdout.decode("utf-8", "replace") + p.stderr.decode("utf-8", "replace")
    res = (app / "results.txt").read_text("utf-8", "replace") if (app / "results.txt").exists() else ""
    if "# ERROR" in res or p.returncode:
        raise SystemExit(f"PSP app failed (PPSSPP rc {p.returncode}):\n{res[-2000:]}\n{log[-3000:]}")
    return res, log


def check_results(res, st):
    """results.txt against stats.json: one row per crop, and the per-line frames and chars (the
    columns a console run reports back) adding up to the app's totals. EBOOTs built before
    psp/main.c kept the first run's extras wrote 0 for both whenever repeat >= 1: a warning."""
    rows = [ln.split("\t") for ln in res.splitlines() if ln and not ln.startswith("#")]
    frames, chars = sum(int(r[1]) for r in rows), sum(int(r[2]) for r in rows)
    if len(rows) == st["lines"] and (frames, chars) == (st["frames"], st["chars"]):
        return
    if len(rows) == st["lines"] and frames == chars == 0 and st.get("repeat", 0) > 0:
        print("warning: results.txt has no per-line frames or chars (an EBOOT from before the fix; "
              "rebuild it with make -C psp)", file=sys.stderr)
        return
    raise SystemExit(f"results.txt ({len(rows)} rows, {frames} frames, {chars} chars) disagrees with "
                     f"stats.json ({st['lines']} lines, {st['frames']} frames, {st['chars']} chars)")


def run_bundle(a):
    """A pack.py bundle as the console would run it (its own vocr.cfg and expect.txt; only
    wait is turned off). Prints the app's summary; exit status 1 if a line differs from host."""
    games = sorted(d for d in (Path(a.bundle) / "PSP" / "GAME").glob("*") if (d / "EBOOT.PBP").exists())
    if len(games) != 1:
        raise SystemExit(f"{a.bundle}: expected one PSP/GAME/<name>/EBOOT.PBP, found {len(games)}")
    tmp = tempfile.TemporaryDirectory(prefix="vocr_psp_")
    home = Path(a.keep).resolve() if a.keep else Path(tmp.name)
    app = home / ".ppsspp" / "PSP" / "GAME" / games[0].name
    if app.exists():
        shutil.rmtree(app)
    shutil.copytree(games[0], app)
    cfg = [ln for ln in (app / "vocr.cfg").read_text().splitlines() if not ln.startswith("wait=")]
    (app / "vocr.cfg").write_text("\n".join(cfg + ["wait=0"]) + "\n")
    res, log = boot(a, home, app)
    if "# done" not in res:
        raise SystemExit(f"PSP app did not finish (timeout {a.timeout:g} s?):\n{log[-3000:]}")
    st = json.loads((app / "stats.json").read_text())
    check_results(res, st)
    if a.stats:
        Path(a.stats).write_text(json.dumps(st, indent=1) + "\n")
    if log.strip():
        print(log.strip())
    # PPSSPPHeadless shows the app's own stdout only with -l, so the summary comes from stats.json
    n = st["lines"]
    print(f"{n} lines, {st['frames']} frames; same as host: text {st['text_ok']}/{n}, logits {st['logits_ok']}/{n}; "
          f"{st['us_total'] / 1e3 / max(n, 1):.3f} ms/line, {st['us_total'] / max(st['frames'], 1):.1f} us/frame "
          f"(PPSSPP emulated time, not console time); arena {st['arena_bytes']} B, stack used {st['stack_used']} B")
    bad = st["expect"] and (st["text_ok"] != st["lines"] or st["logits_ok"] != st["lines"])
    sys.exit(1 if bad else 0)


def run(a):
    if a.bundle:
        return run_bundle(a)
    if not a.model:
        raise SystemExit("MODEL.vocr is required (or --bundle DIR)")
    tmp = tempfile.TemporaryDirectory(prefix="vocr_psp_")
    home = Path(a.keep).resolve() if a.keep else Path(tmp.name)
    app = home / ".ppsspp" / "PSP" / "GAME" / "VOCR"
    if app.exists():
        shutil.rmtree(app)
    app.mkdir(parents=True)
    shutil.copyfile(a.eboot, app / "EBOOT.PBP")
    shutil.copyfile(a.model, app / "model.vocr")
    images = list(a.images)
    if a.stdin:
        images += [ln.rstrip("\r\n") for ln in sys.stdin if ln.strip()]
    n_lines = 0
    if a.pack:
        shutil.copyfile(a.pack, app / "lines.bin")
        n_lines = struct.unpack("<I", (app / "lines.bin").read_bytes()[:4])[0]
    elif images:
        write_images_pack(app / "lines.bin", images)
        n_lines = len(images)
    elif not a.arena:
        raise SystemExit("no input: give --pack, --stdin or image paths")
    (app / "vocr.cfg").write_text("".join(f"{k}={v}\n" for k, v in make_cfg(a).items()))
    res, log = boot(a, home, app)
    if a.arena:
        sys.stdout.write((app / "arena.txt").read_text())
        return
    if "# done" not in res:
        raise SystemExit(f"PSP app did not finish (timeout {a.timeout:g} s?):\n{log[-3000:]}")
    texts = (app / "texts.txt").read_bytes().decode("utf-8").split("\n")
    if texts and texts[-1] == "":
        texts.pop()
    if len(texts) != n_lines:
        raise SystemExit(f"PSP app wrote {len(texts)} texts for {n_lines} crops")
    if a.logits:
        shutil.copyfile(app / "logits.bin", a.logits)
    if a.input_out:
        shutil.copyfile(app / "input.bin", a.input_out)
    st = json.loads((app / "stats.json").read_text())
    check_results(res, st)
    if a.stats:
        Path(a.stats).write_text(json.dumps(st, indent=1) + "\n")
    sys.stdout.write("".join(t + "\n" for t in texts))
    sys.stdout.flush()
    lines, frames, us = st["lines"], st["frames"], st["us_total"]
    print(f"blob {st['blob_bytes']} bytes, arena {st['arena_bytes']} bytes, mode {st['mode']}, "
          f"tile {st['tile']}, head {st['head']} (PSP build in PPSSPP, {st['cpu_mhz']} MHz)\n"
          f"{lines} lines, {frames} frames, {us / 1e3 / max(lines, 1):.3f} ms/line, "
          f"{us / max(frames, 1):.1f} us/frame (PPSSPP emulated time, not console time)",
          file=sys.stderr)


def main():
    run(parse_args(sys.argv[1:]))


if __name__ == "__main__":
    main()
