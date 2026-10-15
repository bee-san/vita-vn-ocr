#!/usr/bin/env python3
"""Google Lens readings of the hand-checked line crops of vita-test and psp-test, for a report-only Lens arm.

`read` sends every scored line's crop, exactly as eval/run.py's recognizer mode cuts it (the manifest box plus
eval's pad: 4 px Vita, 2 px PSP; uncertain lines skipped; screen polarity, RGB), to Google Lens through
eval/owocr_bridge.py (owocr 1.26.7), one request per crop, and appends one JSON line per crop to a private
cache. It scores nothing and never prints a reading. `serve` hands the cached readings to eval/run.py as a
`cmd:` recognizer, so the Lens arm is scored by the same code and metrics as every other arm:

  eval/run.py --set vita-test --polarity native \\
      --model 'cmd:.venv-ci/bin/python eval/lens_lines.py serve --cache LENS_DIR/vita-test.jsonl'

Only these two sets are allowed: their screenshots are public VNDB images (prompts/_rules.md lets Lens see
public VNDB images only). Never point this at private captures.

The Lens environment (the CLANNAD long run's, eval/requirements-owocr-longrun.txt; its own venv):

  UV=~/.local/share/mise/installs/uv/0.11.28/uv-x86_64-unknown-linux-musl/uv
  $UV venv --python 3.11.15 .venv-owocr
  $UV pip install --python .venv-owocr/bin/python --no-deps -r eval/requirements-owocr-longrun.txt

Then (resumable: rows already read are skipped, failed ones retried):

  .venv-ci/bin/python eval/lens_lines.py read --set vita-test --data $VOCR_EVAL_DATA \\
      --owocr-python .venv-owocr/bin/python --out $HERD/shared/private/vocr5-train/lens/vita-test.jsonl

Cache rows: {id, set, vn, crop_sha256, crop_wh, ok, text, lines, error, attempts, owocr, bridge_sha256,
requested_at_unix, host_seconds}; `crop_sha256` is sha256(b"<w>x<h>\\n" + RGB bytes) of the crop. The
first run also writes <out>.meta.json: the bridge's environment (--info), the manifest sha256 and the pad.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
SETS = ("vita-test", "psp-test")
RETRIES = 5          # attempts per crop
BACKOFF_S = 4.0      # first wait after a failed attempt; doubles each time


def crop_key(rgb):
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    return hashlib.sha256(b"%dx%d\n" % (w, h) + rgb.tobytes()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_png(rgb, path):
    """A lossless PNG and the request fields eval/owocr_bridge.py verifies."""
    from PIL import Image
    im = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), "RGB")
    im.save(path, format="PNG")
    return {"path": str(path), "sha256": file_sha(path), "pixels_sha256": hashlib.sha256(im.tobytes()).hexdigest(),
            "mode": "RGB"}


class Bridge:
    """One persistent eval/owocr_bridge.py process (JSON lines in and out), restarted if it dies."""

    def __init__(self, argv, log):
        self.argv, self.log, self.proc, self.n = argv, log, None, 0

    def info(self):
        return json.loads(subprocess.check_output([*self.argv, "--info"], text=True, stderr=self.log,
                                                  stdin=subprocess.DEVNULL))

    def request(self, entry):
        if self.proc is None or self.proc.poll() is not None:
            self.proc = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                         text=True)
        self.n += 1
        self.proc.stdin.write(json.dumps(dict(entry, id=self.n)) + "\n")
        self.proc.stdin.flush()
        raw = self.proc.stdout.readline()
        if not raw:
            self.proc = None
            raise RuntimeError("bridge exited without a result")
        answer = json.loads(raw)
        if answer.get("id") != self.n:
            raise RuntimeError("bridge answer id mismatch")
        return answer

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.terminate()


def load_cache(path):
    """The last row per line id (a resumed run appends retries after earlier failures)."""
    rows = {}
    if Path(path).exists():
        for raw in Path(path).read_text(encoding="utf-8").splitlines():
            if raw.strip():
                r = json.loads(raw)
                rows[r["id"]] = r
    return rows


def read_lines(lines, out, bridge, owocr_version, bridge_sha, sleep_s=1.0, backoff_s=BACKOFF_S, retries=RETRIES):
    """Read every line not yet read OK into `out` (appending). Returns (requests, ok, failed)."""
    done = {k for k, r in load_cache(out).items() if r.get("ok")}
    requests = ok = failed = 0
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lens_lines_") as tmp, open(out, "a", encoding="utf-8") as f:
        for ln in lines:
            if ln.id in done:
                continue
            key = crop_key(ln.crop_rgb)
            entry = write_png(ln.crop_rgb, Path(tmp) / "crop.png")
            answer, attempts, wait = None, 0, backoff_s
            while attempts < retries:
                attempts += 1
                requests += 1
                try:
                    answer = bridge.request(entry)
                except Exception as exc:  # a dead or confused bridge counts as a failed attempt
                    answer = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                if answer.get("ok"):
                    break
                if attempts < retries:
                    time.sleep(wait)
                    wait *= 2
            h, w = ln.crop_rgb.shape[:2]
            row = {"id": ln.id, "set": ln.set, "vn": ln.vn, "crop_sha256": key, "crop_wh": [w, h],
                   "ok": bool(answer.get("ok")), "text": answer.get("text") if answer.get("ok") else None,
                   "lines": answer.get("lines") if answer.get("ok") else None,
                   "error": None if answer.get("ok") else answer.get("error"), "attempts": attempts,
                   "owocr": owocr_version, "bridge_sha256": bridge_sha,
                   "requested_at_unix": answer.get("requested_at_unix"), "host_seconds": answer.get("host_seconds")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            ok += row["ok"]
            failed += not row["ok"]
            if sleep_s:
                time.sleep(sleep_s)
    return requests, ok, failed


def cmd_read(a):
    import sets
    from fetch import data_dir
    if a.set not in SETS:
        raise SystemExit(f"only {SETS} (public VNDB screenshots)")
    lines, _ = sets.load_real(a.set, data_dir(a.data))
    log = open(str(a.out) + ".bridge.log", "a")
    argv = [str(a.owocr_python), str(HERE / "owocr_bridge.py")] if not a.bridge_cmd else json.loads(a.bridge_cmd)
    bridge = Bridge(argv, log)
    info = bridge.info()
    meta = Path(str(a.out) + ".meta.json")
    if not meta.exists():
        meta.write_text(json.dumps({"kind": "lens-lines-meta", "schema": 1, "set": a.set,
                                    "manifest_sha256": file_sha(sets.MANIFESTS / f"{a.set}.jsonl"),
                                    "pad_px": sets.PAD[sets.SETS[a.set]], "polarity": "screen (native), RGB",
                                    "lines": len(lines), "bridge_info": info}, indent=1, sort_keys=True) + "\n")
    version = (info.get("dependencies") or {}).get("owocr")
    try:
        req, ok, failed = read_lines(lines, a.out, bridge, version, info.get("bridge_sha256"), sleep_s=a.sleep)
    finally:
        bridge.close()
        log.close()
    rows = load_cache(a.out)
    print(f"{a.set}: {len(lines)} lines, {sum(r['ok'] for r in rows.values())} read OK, "
          f"{sum(not r['ok'] for r in rows.values())} failed; this run {req} requests ({ok} OK, {failed} failed)")


def parse_ppm(path):
    data = Path(path).read_bytes()
    parts = data.split(b"\n", 3)
    if parts[0] != b"P6" or parts[2] != b"255":
        raise ValueError(f"{path}: not eval's binary PPM")
    w, h = map(int, parts[1].split())
    return np.frombuffer(parts[3][:w * h * 3], dtype=np.uint8).reshape(h, w, 3)


def cmd_serve(a):
    """eval/run.py's cmd: protocol: crop paths on stdin, one reading per line on stdout, in order."""
    by_key = {}
    for r in load_cache(a.cache).values():
        if r.get("ok"):
            by_key.setdefault(r["crop_sha256"], r["text"] or "")
    out, missing = [], 0
    for raw in sys.stdin:
        p = raw.strip()
        if not p:
            continue
        text = by_key.get(crop_key(parse_ppm(p)))
        if text is None:
            missing += 1
            text = ""
        out.append(" ".join(text.splitlines()))
    if missing and not a.missing_empty:
        print(f"{missing} crops have no OK Lens reading in {a.cache} (polarity must be native; "
              "--missing-empty scores them as empty output)", file=sys.stderr)
        raise SystemExit(3)
    sys.stdout.write("".join(t + "\n" for t in out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("--set", required=True, choices=SETS)
    r.add_argument("--data", default=None, help="screenshot dir (default $VOCR_EVAL_DATA or data/eval)")
    r.add_argument("--out", required=True, type=Path, help="private JSONL cache (appended)")
    r.add_argument("--owocr-python", type=Path, help="python of the owocr 1.26.7 venv")
    r.add_argument("--bridge-cmd", help=argparse.SUPPRESS)  # JSON argv of a stand-in bridge (tests)
    r.add_argument("--sleep", type=float, default=1.0, help="seconds between crops")
    s = sub.add_parser("serve")
    s.add_argument("--cache", required=True, type=Path)
    s.add_argument("--missing-empty", action="store_true", help="score crops without a reading as empty output")
    a = ap.parse_args()
    if a.cmd == "read" and not (a.owocr_python or a.bridge_cmd):
        ap.error("read needs --owocr-python")
    {"read": cmd_read, "serve": cmd_serve}[a.cmd](a)


if __name__ == "__main__":
    main()
