#!/usr/bin/env python3
"""MeikiOCR on the CLANNAD long run, against clannad_refs's frozen references (eval/clannad_meiki_protocol.json).

  .venv-meiki: clannad_meiki.py verify    --into DIR
               clannad_meiki.py lines     --work WORK --into DIR [--procs 5]     (offline: unshare -rn; heavy.sh)
               clannad_meiki.py screens   --work WORK --into DIR [--procs 5]     (offline: unshare -rn; heavy.sh)
               clannad_meiki.py resources --into DIR
               clannad_meiki.py speed     --engine meiki --work WORK --into DIR
  .venv-ci:    clannad_meiki.py speed     --engine vocr --work WORK --into DIR --blobs BLOBS
               clannad_refs.py  score     --work WORK --suffix meiki --into DIR --pair MeikiOCR:VOCR3_W20 \
                                          --pair MeikiOCR:Lens --out eval/clannad_meiki_lines_<date>.json   (once)
               clannad_meiki.py score-screens --work WORK --into DIR --out eval/clannad_meiki_screens_<date>.json (once)
               clannad_meiki.py report    --into DIR --out eval/clannad_meiki_resources_<date>.json

MeikiOCR is rtr46/meikiocr 0.3.5 (PyPI; GitHub 52aa607), run with its package defaults through its own MeikiOCR
methods; only the onnxruntime sessions are made here (one thread, the pinned model files). Two arms:

lines    arm (a), "MeikiOCR": run_recognition on every stored line crop (the C GRAD finder's crop, the pixels of
         every protocol arm and of Lens) -> DIR/arms.meiki.jsonl / .meta.json in clannad_refs's extra-arm format,
         so clannad_refs.py score --suffix meiki scores it line by line exactly as the protocol's arms.
screens  arm (b), "MeikiOCR_screen": run_ocr (its own detector, then its recognizer) on every page's stored native
         frame -> DIR/screens.meiki.jsonl / .meta.json: every detection's box and text (private).
score-screens  once: whole dialogue per screen. The screens whose every crop is a reference or empty; per system
         the page's texts in reading order against the page's references (general_vn_textbox_guard.transcript_score),
         page bootstrap, the protocol's pairs, the MeikiOCR_screen detection counts and the entanglement counts.
resources, speed, report   model files, parameters and host times (never console numbers) -> one public JSON.

WORK (clannad_refs's work dir) is only read. Everything under DIR is private (copyrighted CLANNAD frames, readings,
references): never commit it, never send it to Codex or any third party. Public JSON holds counts, rates, intervals,
hashes, sizes and times only.
"""
import argparse
import base64
import collections
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "train", ROOT / "synth", ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import clannad_refs as CR  # noqa: E402

PROTOCOL_PATH = ROOT / "eval" / "clannad_meiki_protocol.json"
LOCK_PATH = ROOT / "eval" / "clannad_meiki_requirements.lock"
SUFFIX = "meiki"
ARM_A, ARM_B = "MeikiOCR", "MeikiOCR_screen"
HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
MODELS = Path(os.environ.get("TEACHER_MODELS") or HERD / "shared" / "artifacts" / "teacher" / "models")
VERSION, COMMIT = "0.3.5", "52aa607ecdc1038c7bf390f37cd9a8b306e8a08a"
WHEEL_SHA256 = "5d1ede4bcf46ecf17ef9a59db26edbc3e622aa3ed3c27bb64317077abbe2fccb"
BLOBS = {"__init__.py": "ad94604546562f32428428450fecc24b931bdd66",        # git blob sha1 at COMMIT
         "cli.py": "d8234e41628dd76785cc8771b207d1e3a31476b7",
         "ocr.py": "2fd63c6716fdc6b64b0c76a8ea3852abf9b49427"}
MODEL_PINS = {   # role: (file, Hugging Face repo, revision, bytes, sha256); teacher/engines.PINS' files
    "det": ("meiki.text.detect.v0.1.960x544.onnx", "rtr46/meiki.text.detect.v0",
            "a9cffa4f60cbf72ddb87edf19c6f98a01cd042e6", 14503825,
            "40b6a016667745cae7d3055929ae3b8b1e7716aac795f5904cd3c2c7c3b8404b"),
    "rec": ("meiki.text.rec.v0.960x32.onnx", "rtr46/meiki.txt.recognition.v0",
            "a28cf5874dc2438ebb1c86336be26bcec51e3375", 18593254,
            "3e96bc772fbee9717e536a6353032bb944c3382dd2f6960ef4890decda43b000"),
    "vrec": ("meiki.text.rec.v0.vertical.32x480.onnx", "rtr46/meiki.txt.recognition.v0",
             "a28cf5874dc2438ebb1c86336be26bcec51e3375", 12872961,
             "2c2a83a23bc3b7e6c63962175f507ecc6c5e85cc174f17bdec37d9bbd0bf895a"),
}
# meikiocr 0.3.5's defaults (MeikiOCR.run_ocr / run_recognition / run_detection and its CLI)
DET_THRESHOLD, REC_THRESHOLD, PUNCT_FACTOR, MAX_BATCH = 0.5, 0.1, 1.0, 8
PAIRS = ((ARM_B, "VOCR3_W20"), (ARM_B, CR.LENS), (ARM_B, ARM_A))
SPEED_LINES, SPEED_PAGES, SPEED_REPEATS, SPEED_WARMUP, SPEED_SEED = 200, 50, 3, 5, 0
SPEED_VOCR = ("VOCR3_W20", "VOCR_H15")


def now():
    return CR.now()


def protocol(path=PROTOCOL_PATH):
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    if p.get("kind") != "clannad-meiki-protocol" or p.get("schema") != 1:
        raise ValueError("not the CLANNAD MeikiOCR protocol")
    return p


def code_identity():
    files = ("eval/clannad_meiki.py", "eval/clannad_meiki_protocol.json", "eval/clannad_meiki_requirements.lock",
             "eval/clannad_refs.py", "eval/clannad_refs_protocol.json", "eval/clannad_longrun.py", "eval/metrics.py",
             "teacher/vote.py", "train/general_vn_textbox_guard.py", "eval/vocr3_report.py")
    return {f: CR.sha256_file(ROOT / f) for f in files if (ROOT / f).is_file()}


def git_state():
    def git(*a):
        try:
            return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"head": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def environment(packages=("meikiocr", "onnxruntime", "numpy", "opencv-python-headless", "pillow", "onnx")):
    import importlib.metadata as md
    pk = {}
    for name in packages:
        try:
            pk[name] = md.version(name)
        except md.PackageNotFoundError:
            pk[name] = None
    return {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
            "libc": list(platform.libc_ver()), "cpu": cpu_model(), "packages": pk, "git": git_state(),
            "heavy_slot": os.environ.get("HEAVY_SLOT"), "offline_env": os.environ.get("HF_HUB_OFFLINE") == "1"}


def cpu_model():
    try:
        for x in Path("/proc/cpuinfo").read_text().splitlines():
            if x.startswith("model name"):
                return x.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def network_interfaces():
    """The interfaces of this process's network namespace (only 'lo' inside `unshare -rn`); no traffic."""
    import socket
    return sorted(name for _, name in socket.if_nameindex())


def require_offline():
    """MeikiOCR reads CLANNAD only in a namespace without a network."""
    names = network_interfaces()
    if names != ["lo"]:
        raise SystemExit(f"network interfaces {names}: run MeikiOCR on CLANNAD inside `unshare -rn`")
    return names


def frozen(work):
    """The frozen references' record; extra arms run only after it exists."""
    f = Path(work) / "refs.freeze.json"
    if not f.exists():
        raise SystemExit("refs.freeze.json is missing: MeikiOCR reads CLANNAD only after the references are frozen")
    freeze = json.loads(f.read_text(encoding="utf-8"))
    if CR.sha256_file(Path(work) / "refs.jsonl") != freeze["refs_sha256"]:
        raise SystemExit("refs.jsonl differs from the freeze record")
    return freeze


# ----------------------------------------------------------------------------- the package and the models

def git_blob_sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify_files(pkg_dir, blobs=BLOBS):
    """Each module of the installed package against its git blob at COMMIT, reading CRLF as LF (the 0.3.5 wheel
    was built with Windows line endings)."""
    out = {}
    for name, want in blobs.items():
        raw = (Path(pkg_dir) / name).read_bytes()
        lf = raw.replace(b"\r\n", b"\n")
        out[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "crlf": raw.count(b"\r\n"),
                     "git_blob_sha1_lf": git_blob_sha1(lf), "matches_commit": git_blob_sha1(lf) == want}
    return out


def record_hashes(dist_info):
    """{relative path: sha256 hex} from a dist-info RECORD (urlsafe base64 digests)."""
    out = {}
    for row in (Path(dist_info) / "RECORD").read_text(encoding="utf-8").splitlines():
        parts = row.rsplit(",", 2)
        if len(parts) == 3 and parts[1].startswith("sha256="):
            d = parts[1][7:]
            out[parts[0]] = base64.urlsafe_b64decode(d + "=" * (-len(d) % 4)).hex()
    return out


def lock_hashes(name, lock=LOCK_PATH):
    """The --hash values the lock pins for one package."""
    lines, out, on = Path(lock).read_text(encoding="utf-8").splitlines(), [], False
    for x in lines:
        if x and not x[0].isspace():
            on = x.split("==")[0].strip().lower() == name
        elif on and "--hash=sha256:" in x:
            out.append(x.split("--hash=sha256:")[1].split()[0].rstrip("\\").strip())
    return out


def model_path(role):
    return MODELS / MODEL_PINS[role][0]


def checked_models():
    """{role: Path} after checking every pinned file's size and sha256."""
    out = {}
    for role, (fname, _, _, size, sha) in MODEL_PINS.items():
        p = model_path(role)
        if p.stat().st_size != size or CR.sha256_file(p) != sha:
            raise SystemExit(f"{p}: not the pinned {fname} ({size} B, sha256 {sha[:12]}…)")
        out[role] = p
    return out


def engine_identity():
    """What runs: package version, commit check, wheel pin, model pins, session options (public)."""
    import importlib.metadata as md
    import importlib.util
    spec = importlib.util.find_spec("meikiocr")
    pkg = Path(spec.origin).parent
    files = verify_files(pkg)
    dist = Path(md.distribution("meikiocr")._path)
    rec = record_hashes(dist)
    installed = {n: rec.get(f"meikiocr/{n}") == files[n]["sha256"] for n in BLOBS}
    return {"package": "meikiocr", "version": md.version("meikiocr"), "commit": COMMIT,
            "modules_match_commit": all(f["matches_commit"] for f in files.values()), "modules": files,
            "installed_files_match_record": all(installed.values()),
            "wheel_sha256_pinned": WHEEL_SHA256, "lock_pins_wheel": WHEEL_SHA256 in lock_hashes("meikiocr"),
            "lock_sha256": CR.sha256_file(LOCK_PATH),
            "models": {r: {"file": f, "repo": repo, "revision": rev, "bytes": size, "sha256": sha}
                       for r, (f, repo, rev, size, sha) in MODEL_PINS.items()},
            "defaults": {"det_threshold": DET_THRESHOLD, "rec_threshold": REC_THRESHOLD,
                         "punct_conf_factor": PUNCT_FACTOR, "max_batch_size": MAX_BATCH},
            "sessions": {"provider": "CPUExecutionProvider", "graph_optimization": "ORT_ENABLE_ALL",
                         "intra_op_spinning": 0, "inter_op_spinning": 0, "intra_op_threads": 1,
                         "inter_op_threads": 1}}


def cmd_verify(a):
    ident = engine_identity()
    checked_models()
    ok = ident["version"] == VERSION and ident["modules_match_commit"] and ident["installed_files_match_record"] \
        and ident["lock_pins_wheel"]
    out = {"engine": ident, "models_checked": True, "ok": ok, "environment": environment(), "created": now()}
    CR.write_json(Path(a.into) / "verify.json", out)
    print(json.dumps({"ok": ok, "version": ident["version"], "modules_match_commit": ident["modules_match_commit"],
                      "installed_files_match_record": ident["installed_files_match_record"],
                      "lock_pins_wheel": ident["lock_pins_wheel"]}))
    if not ok:
        raise SystemExit("VERIFY_FAILED")


def session(path, n_threads=1):
    """An onnxruntime session with meikiocr 0.3.5's own options, plus a fixed thread count."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.add_session_config_entry("session.inter_op.allow_spinning", "0")
    so.intra_op_num_threads = n_threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def load_meiki(n_threads=1):
    """meikiocr.MeikiOCR without its __init__ (which downloads through the Hub and takes every core): the same
    attributes, with the pinned files and n_threads."""
    import onnxruntime as ort
    from meikiocr import MeikiOCR
    ort.set_default_logger_severity(3)
    paths = checked_models()
    m = MeikiOCR.__new__(MeikiOCR)
    m.det_session = session(paths["det"], n_threads)
    m.rec_session = session(paths["rec"], n_threads)
    m.vrec_session = session(paths["vrec"], n_threads)
    m.active_provider = m.det_session.get_providers()[0]
    m.max_batch_size = MAX_BATCH
    if m.active_provider != "CPUExecutionProvider":
        raise SystemExit(f"provider {m.active_provider}, want CPUExecutionProvider")
    return m


def read_line(m, grey):
    """Arm (a): (text, vertical) of one stored grey crop, three identical channels, meikiocr's defaults."""
    bgr = np.ascontiguousarray(np.repeat(grey[:, :, None], 3, axis=2))
    r = m.run_recognition([bgr], conf_threshold=REC_THRESHOLD, punct_conf_factor=PUNCT_FACTOR)[0]
    return r["text"], bool(r["is_vertical"])


def frame_bgr(rgb):
    """A stored frame (RGB, RGBA or grey) as the BGR uint8 image meikiocr reads (cv2.imread's layout)."""
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    return np.ascontiguousarray(rgb[:, :, :3][:, :, ::-1])


def read_screen(m, bgr):
    """Arm (b): run_ocr with its defaults; the line boxes are the return value of the run_detection call run_ocr
    itself makes (a pass-through wrapper on the instance; no second detection pass)."""
    seen = []
    original = m.run_detection

    def observed(image, conf_threshold=DET_THRESHOLD):
        boxes = original(image, conf_threshold)
        seen.append([list(map(int, b["bbox"])) for b in boxes])
        return boxes
    m.run_detection = observed
    try:
        results = m.run_ocr(bgr, det_threshold=DET_THRESHOLD, rec_threshold=REC_THRESHOLD,
                            punct_conf_factor=PUNCT_FACTOR)
    finally:
        del m.run_detection
    if len(seen) != 1 or len(seen[0]) != len(results):
        raise RuntimeError(f"run_ocr made {len(seen)} detection calls / {len(results)} results")
    return [{"box": b, "text": r["text"], "vertical": bool(r["is_vertical"])} for b, r in zip(seen[0], results)]


# ----------------------------------------------------------------------------- arm (a): the stored line crops

_W = {}


def _worker_init(n_threads):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _W["m"] = load_meiki(n_threads)


def _line_job(job):
    session_dir, row = job
    g = CR.load_grey(session_dir, row["line"])
    t0 = time.perf_counter()
    text, vertical = read_line(_W["m"], g)
    return {"id": row["id"], ARM_A: text}, {"id": row["id"], "vertical": vertical,
                                            "ms": round(1000 * (time.perf_counter() - t0), 3)}


def _screen_job(job):
    session_dir, page, src = job
    import clannad_longrun as LR
    rgb = LR.verified_image(Path(session_dir), src)
    t0 = time.perf_counter()
    dets = read_screen(_W["m"], frame_bgr(rgb))
    return {"page": page, "size": [int(rgb.shape[1]), int(rgb.shape[0])], "detections": dets,
            "ms": round(1000 * (time.perf_counter() - t0), 3)}


def run_pool(fn, jobs, procs, n_threads=1):
    """In order; procs 0 runs in this process (tests, or a fake engine already in _W)."""
    if procs <= 0:
        return [fn(j) for j in jobs]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(procs, initializer=_worker_init, initargs=(n_threads,)) as pool:
        return pool.map(fn, jobs, chunksize=8)


def time_stats(ms):
    ms = sorted(ms)
    if not ms:
        return None
    return {"calls": len(ms), "median_ms": round(statistics.median(ms), 3),
            "p10_ms": round(float(np.percentile(ms, 10)), 3), "p90_ms": round(float(np.percentile(ms, 90)), 3),
            "mean_ms": round(statistics.fmean(ms), 3), "total_s": round(sum(ms) / 1000, 3)}


def run_meta(a, freeze, meta, extra):
    proto_sha = CR.sha256_file(PROTOCOL_PATH)
    return {"suffix": SUFFIX, "lines_sha256": meta["lines_sha256"], "after_freeze": freeze["refs_sha256"],
            "protocol_sha256": proto_sha, "refs_protocol_sha256": freeze.get("protocol_sha256"),
            "procs": a.procs, "threads_per_proc": 1, "network_interfaces": network_interfaces(),
            "code_sha256": code_identity(), "environment": environment(),
            "created": now(), **extra}


def pinned_engine():
    ident = engine_identity()
    if not (ident["version"] == VERSION and ident["modules_match_commit"] and ident["lock_pins_wheel"]):
        raise SystemExit("meikiocr is not the pinned 0.3.5 / 52aa607: run verify")
    return ident


def cmd_lines(a):
    work, into = Path(a.work), Path(a.into)
    offline = require_offline()
    freeze = frozen(work)
    rows, meta = CR.load_lines(work)
    ident = pinned_engine()
    for p in (into / f"arms.{SUFFIX}.jsonl", into / f"arms.{SUFFIX}.meta.json"):
        if p.exists():
            raise SystemExit(f"{p} exists: arm (a) runs once")
    t0 = time.time()
    out = run_pool(_line_job, [(meta["session"], r) for r in rows], a.procs)
    texts, calls = [x[0] for x in out], [x[1] for x in out]
    sha = CR.write_jsonl(into / f"arms.{SUFFIX}.jsonl", texts, exclusive=True)
    models = {r: ident["models"][r] for r in ("rec", "vrec")}
    CR.write_json(into / f"arms.{SUFFIX}.meta.json", run_meta(a, freeze, meta, {
        "lines": len(texts), "arms_sha256": sha,
        "arms": {ARM_A: {"name": ARM_A, "engine": f"meikiocr {ident['version']}", "commit": COMMIT,
                         "call": f"run_recognition(conf_threshold={REC_THRESHOLD}, punct_conf_factor={PUNCT_FACTOR})",
                         "input": "stored GRAD line crop, grey as three identical channels", "models": models,
                         "finder": "grad (stored crop)"}},
        "resources": {ARM_A: {"model_bytes": sum(m["bytes"] for m in models.values()), "models": models}},
        "native": {"engine": ident, "onnxruntime": environment()["packages"]["onnxruntime"]},
        "calls": {"vertical": sum(c["vertical"] for c in calls), "time": time_stats([c["ms"] for c in calls]),
                  "note": f"{a.procs or 1} concurrent single-thread processes; not the speed measurement"},
        "offline": offline, "wall_s": round(time.time() - t0, 1)}), exclusive=True)
    print(json.dumps({"suffix": SUFFIX, "lines": len(texts), "arms_sha256": sha,
                      "vertical": sum(c["vertical"] for c in calls)}))


# ----------------------------------------------------------------------------- arm (b): the full screens

def page_sources(rows):
    """[(page, source entry)] in page order; every line of a page must name the same frame."""
    out = {}
    for r in rows:
        src = out.setdefault(r["page"], r["source"])
        if src != r["source"]:
            raise ValueError(f"{r['page']}: lines name different source frames")
    return list(out.items())


def cmd_screens(a):
    work, into = Path(a.work), Path(a.into)
    offline = require_offline()
    freeze = frozen(work)
    rows, meta = CR.load_lines(work)
    ident = pinned_engine()
    for p in (into / f"screens.{SUFFIX}.jsonl", into / f"screens.{SUFFIX}.meta.json"):
        if p.exists():
            raise SystemExit(f"{p} exists: arm (b) runs once")
    t0 = time.time()
    pages = page_sources(rows)
    out = run_pool(_screen_job, [(meta["session"], p, s) for p, s in pages], a.procs)
    sha = CR.write_jsonl(into / f"screens.{SUFFIX}.jsonl", out, exclusive=True)
    roi = meta["roi"]
    dets = [d for s in out for d in s["detections"]]
    kept = [d for d in dets if in_roi(d["box"], roi)]
    CR.write_json(into / f"screens.{SUFFIX}.meta.json", run_meta(a, freeze, meta, {
        "pages": len(out), "screens_sha256": sha, "roi": roi,
        "arm": {"name": ARM_B, "engine": f"meikiocr {ident['version']}", "commit": COMMIT,
                "call": f"run_ocr(det_threshold={DET_THRESHOLD}, rec_threshold={REC_THRESHOLD}, "
                        f"punct_conf_factor={PUNCT_FACTOR})", "input": "stored native frame, RGB to BGR",
                "models": ident["models"]},
        "native": {"engine": ident},
        "detections": {"all": len(dets), "in_roi": len(kept), "vertical_in_roi": sum(d["vertical"] for d in kept),
                       "frame_sizes": dict(collections.Counter("x".join(map(str, s["size"])) for s in out))},
        "time": time_stats([s["ms"] for s in out]),
        "time_note": f"{a.procs or 1} concurrent single-thread processes; not the speed measurement",
        "offline": offline, "wall_s": round(time.time() - t0, 1)}), exclusive=True)
    print(json.dumps({"suffix": SUFFIX, "pages": len(out), "screens_sha256": sha, "detections": len(dets),
                      "in_roi": len(kept)}))


# ----------------------------------------------------------------------------- whole dialogue per screen

def in_roi(box, roi):
    """The box's centre lies in roi [x, y, w, h] (left and top inclusive)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return roi[0] <= cx < roi[0] + roi[2] and roi[1] <= cy < roi[1] + roi[3]


def overlaps(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def reading_order(boxes):
    """Indices of boxes [x0, y0, x1, y1] in reading order: sorted by centre y, a box joins the current row when
    its vertical overlap with the row's span is at least half the smaller height; rows top to bottom, boxes
    left to right within a row."""
    order = sorted(range(len(boxes)), key=lambda i: ((boxes[i][1] + boxes[i][3]) / 2, boxes[i][0], i))
    rows, span = [], None
    for i in order:
        y0, y1 = boxes[i][1], boxes[i][3]
        if rows:
            ov = min(y1, span[1]) - max(y0, span[0])
            if ov > 0 and ov >= 0.5 * min(y1 - y0, span[1] - span[0]):
                rows[-1].append(i)
                span = (min(span[0], y0), max(span[1], y1))
                continue
        rows.append([i])
        span = (y0, y1)
    return [i for r in rows for i in sorted(r, key=lambda i: (boxes[i][0], boxes[i][1], i))]


def screen_sets(rows, refs):
    """(eligible {page: [line rows in reading order]}, empty [pages], excluded {page: [reasons]}) by the protocol:
    a page is scored when each crop is a reference or a two-read empty crop and at least one is a reference."""
    by = {r["id"]: r for r in refs}
    pages = collections.defaultdict(list)
    for r in rows:
        pages[r["page"]].append(r)
    eligible, empty, excluded = {}, [], {}
    for page in sorted(pages):
        prs = pages[page]
        bad = [by[r["id"]]["why"] for r in prs if by[r["id"]]["status"] != "ref" and not by[r["id"]]["notext"]]
        if bad:
            excluded[page] = bad
            continue
        boxes = [CR.native_box(r) for r in prs]
        ordered = [prs[i] for i in reading_order(boxes)]
        if any(by[r["id"]]["status"] == "ref" for r in prs):
            eligible[page] = ordered
        else:
            empty.append(page)
    return eligible, empty, excluded


def kept_detections(screen, roi):
    """MeikiOCR_screen's detections in the ROI, in reading order."""
    dets = [d for d in screen["detections"] if in_roi(d["box"], roi)]
    return [dets[i] for i in reading_order([d["box"] for d in dets])]


def screen_unit(ref_lines, preds):
    """One page: transcript_score counts and the exact flag. ref_lines [{'text', 'name'}], preds [str]."""
    import metrics

    from general_vn_textbox_guard import transcript_score
    lines = [{"text": x["text"], "styles": ["name"] if x["name"] else []} for x in ref_lines]
    s = transcript_score(lines, list(preds))
    exact = "".join(metrics.norm(p) for p in preds) == "".join(metrics.norm(x["text"]) for x in ref_lines)
    return {"all_errors": s["all"]["full"]["errors"], "all_chars": s["all"]["full"]["chars"],
            "body_errors": s["body"]["full"]["errors"], "body_chars": s["body"]["full"]["chars"],
            "hira_errors": s["all"]["hira"]["errors"], "hira_chars": s["all"]["hira"]["chars"], "exact": int(exact)}


def screen_summary(units, ids, idx):
    if not ids:
        return {"screens": 0, "chars": 0, "errors": 0, "exact_screens": 0, "cer": None, "cer_ci95": None,
                "exact": None, "exact_ci95": None, "dialogue_chars": 0, "dialogue_errors": 0, "dialogue_cer": None,
                "dialogue_cer_ci95": None, "hiragana_errors": 0, "hiragana_chars": 0, "hiragana_cer": None}

    def col(k):
        return [units[p][k] for p in ids]
    e, n, be, bn, x = col("all_errors"), col("all_chars"), col("body_errors"), col("body_chars"), col("exact")
    he, hn = sum(col("hira_errors")), sum(col("hira_chars"))
    m = [1] * len(ids)
    return {"screens": len(ids), "chars": sum(n), "errors": sum(e), "exact_screens": sum(x),
            "cer": round(100 * sum(e) / sum(n), 4), "cer_ci95": CR.rate_ci(e, n, idx),
            "exact": round(100 * sum(x) / len(ids), 4), "exact_ci95": CR.rate_ci(x, m, idx),
            "dialogue_chars": sum(bn), "dialogue_errors": sum(be),
            "dialogue_cer": round(100 * sum(be) / sum(bn), 4) if sum(bn) else None,
            "dialogue_cer_ci95": CR.rate_ci(be, bn, idx) if sum(bn) else None,
            "hiragana_errors": he, "hiragana_chars": hn, "hiragana_cer": round(100 * he / hn, 4) if hn else None}


def score_screens(rows, refs, roi, texts, screens, systems, pairs=PAIRS, reps=CR.REPS, seed=CR.SEED):
    """Public scores and private per-page rows. texts: {line id: {system: text}} for the GRAD-crop systems;
    screens: {page: arm (b) record}."""
    import metrics
    import vocr3_report as R
    by = {r["id"]: r for r in refs}
    eligible, empty, excluded = screen_sets(rows, refs)
    ids = sorted(eligible)
    units, private = {s: {} for s in systems + [ARM_B]}, []
    for page in ids:
        crops = eligible[page]
        ref_lines = [{"text": by[r["id"]]["text"], "name": bool(by[r["id"]]["name"])}
                     for r in crops if by[r["id"]]["status"] == "ref"]
        preds = {s: [texts[r["id"]][s] for r in crops] for s in systems}
        preds[ARM_B] = [d["text"] for d in kept_detections(screens[page], roi)]
        for s, p in preds.items():
            units[s][page] = screen_unit(ref_lines, p)
        private.append({"page": page, "ref": "".join(x["text"] for x in ref_lines),
                        "preds": {s: "".join(p) for s, p in preds.items()},
                        "errors": {s: units[s][page]["all_errors"] for s in preds}})
    idx = CR.boot_idx(len(ids), reps, seed)
    out = {"systems": {s: screen_summary(units[s], ids, idx) for s in units}, "pairs": {},
           "bootstrap": {"unit": "screen", "screens": len(ids), "reps": reps, "seed": seed}}
    for cand, base in pairs:
        p = {}
        scopes = {"cer": ("all_errors", "all_chars"), "dialogue_cer": ("body_errors", "body_chars")}
        for scope, (ek, nk) in scopes.items():
            r = R.paired_bootstrap({q: (units[cand][q][ek], units[cand][q][nk]) for q in ids},
                                   {q: (units[base][q][ek], units[base][q][nk]) for q in ids}, reps, seed)
            p[scope] = {"delta": r["delta"], "ci95": r["ci95"], "candidate_errors": r["candidate_errors"],
                        "baseline_errors": r["baseline_errors"], "chars": r["chars"]}
        xa, xb = [units[cand][q]["exact"] for q in ids], [units[base][q]["exact"] for q in ids]
        p["exact"] = {"delta": round(100 * (sum(xa) - sum(xb)) / len(ids), 4),
                      "ci95": CR.diff_ci(xa, xb, [1] * len(ids), idx)}
        out["pairs"][f"{cand}:{base}"] = p
    # screens whose crops are all empty: what each system writes there (never in CER)
    pages = collections.defaultdict(list)
    for r in rows:
        pages[r["page"]].append(r)
    empty_out = {}
    for s in systems + [ARM_B]:
        n = [sum(len(metrics.norm(texts[r["id"]][s])) for r in pages[p]) if s != ARM_B else
             sum(len(metrics.norm(d["text"])) for d in kept_detections(screens[p], roi)) for p in empty]
        empty_out[s] = {"screens": len(empty), "screens_with_output": sum(x > 0 for x in n), "output_chars": sum(n)}
    # MeikiOCR_screen's detections
    det = {"all_pages": len(screens), "detections": 0, "in_roi": 0, "outside_roi": 0, "vertical_in_roi": 0}
    for s in screens.values():
        k = [d for d in s["detections"] if in_roi(d["box"], roi)]
        det["detections"] += len(s["detections"])
        det["in_roi"] += len(k)
        det["outside_roi"] += len(s["detections"]) - len(k)
        det["vertical_in_roi"] += sum(d["vertical"] for d in k)
    sc = {"screens": len(ids), "kept_detections": 0, "screens_without_detection": 0,
          "detections_outside_crops": 0, "chars_outside_crops": 0, "chars_written": 0, "vertical": 0}
    for page in ids:
        k = kept_detections(screens[page], roi)
        crops = [CR.native_box(r) for r in pages[page]]
        sc["kept_detections"] += len(k)
        sc["screens_without_detection"] += not k
        sc["vertical"] += sum(d["vertical"] for d in k)
        for d in k:
            n = len(metrics.norm(d["text"]))
            sc["chars_written"] += n
            if not any(overlaps(d["box"], c) for c in crops):
                sc["detections_outside_crops"] += 1
                sc["chars_outside_crops"] += n
    det["scored_screens"] = sc
    reasons = collections.Counter(w for ws in excluded.values() for w in set(ws))
    account = {"pages": len(pages), "scored": len(ids), "empty": len(empty), "excluded": len(excluded),
               "excluded_pages_by_reason": dict(sorted(reasons.items())),
               "excluded_reference_lines": sum(by[r["id"]]["status"] == "ref" for p in excluded for r in pages[p]),
               "reference_lines": sum(by[r["id"]]["status"] == "ref" for p in ids for r in eligible[p]),
               "crops": sum(len(eligible[p]) for p in ids),
               "empty_crops": sum(bool(by[r["id"]]["notext"]) for p in ids for r in eligible[p])}
    return out, empty_out, det, account, private


def entanglement(rows, refs, engines, arm_a):
    """Counts of how MeikiOCR's readings relate to the references it helped to build (public counts only)."""
    key = CR.key
    scored = [x for x in refs if x["status"] == "ref"]
    dis = [x for x in scored if x["lens_disagree"]]

    def same_key(xs, eng):
        return sum(key(x["text"]) == key(engines[x["id"]].get(eng) or "") for x in xs)
    return {"lines": len(rows), "arm_a_equals_r1_meiki": sum(arm_a[r["id"]] == engines[r["id"]]["meiki"] for r in rows),
            "scored_lines": len(scored),
            "scored_arm_a_equals_r1_meiki": sum(arm_a[x["id"]] == engines[x["id"]]["meiki"] for x in scored),
            "scored_ref_key_equals_r1_meiki": same_key(scored, "meiki"),
            "scored_ref_key_equals_r1_v6s": same_key(scored, "v6s"),
            "lens_disagree_lines": len(dis), "lens_disagree_ref_key_equals_r1_meiki": same_key(dis, "meiki"),
            "lens_disagree_ref_key_equals_r1_v6s": same_key(dis, "v6s"),
            "note": "R1's MeikiOCR candidate is teacher/engines.Meiki (punct_conf_factor 0.2, owocr's setting); "
                    "arm (a) is meikiocr's default (1.0); keys are teacher/vote.key"}


def load_screens(into):
    into = Path(into)
    meta = json.loads((into / f"screens.{SUFFIX}.meta.json").read_text(encoding="utf-8"))
    if CR.sha256_file(into / f"screens.{SUFFIX}.jsonl") != meta["screens_sha256"]:
        raise ValueError(f"screens.{SUFFIX}.jsonl changed since it was written")
    return {r["page"]: r for r in CR.read_jsonl(into / f"screens.{SUFFIX}.jsonl")}, meta


def cmd_score_screens(a):
    work, into = Path(a.work), Path(a.into)
    freeze = frozen(work)
    proto_sha = CR.sha256_file(PROTOCOL_PATH)
    if freeze.get("protocol_sha256") != CR.sha256_file(CR.PROTOCOL_PATH):
        raise SystemExit("eval/clannad_refs_protocol.json differs from the one the references were frozen under")
    extra, emeta = CR.load_extra(into, SUFFIX)
    screens, smeta = load_screens(into)
    for name, m in (("arms", emeta), ("screens", smeta)):
        if m["after_freeze"] != freeze["refs_sha256"]:
            raise SystemExit(f"{name}.{SUFFIX}: not run against these frozen references")
        if m["protocol_sha256"] != proto_sha:
            raise SystemExit(f"{name}.{SUFFIX}: ran under another eval/clannad_meiki_protocol.json")
    attempt = into / "screens.score.attempt.json"
    CR.write_json(attempt, {"started": now(), "refs_sha256": freeze["refs_sha256"], "protocol_sha256": proto_sha},
                  exclusive=True)                                                  # once
    rows, meta = CR.load_lines(work)
    arms, ameta = CR.load_arms(work)
    refs = CR.read_jsonl(work / "refs.jsonl")
    eng_meta = json.loads((work / "engines.meta.json").read_text(encoding="utf-8"))
    if CR.sha256_file(work / "engines.jsonl") != eng_meta["engines_sha256"]:
        raise SystemExit("engines.jsonl changed since `engines`")
    engines = {r["id"]: r for r in CR.read_jsonl(work / "engines.jsonl")}
    if sorted(screens) != sorted({r["page"] for r in rows}) or set(extra) != {r["id"] for r in rows}:
        raise SystemExit("the MeikiOCR arms do not cover every line and page of lines.jsonl")
    names = [x["name"] for x in CR.protocol()["arms"]]
    systems = [ARM_A] + names + [CR.LENS]
    texts = {r["id"]: {**{n: arms[r["id"]][n] for n in names}, ARM_A: extra[r["id"]][ARM_A], CR.LENS: r["lens"]}
             for r in rows}
    scores, empty_out, det, account, private = score_screens(rows, refs, meta["roi"], texts, screens, systems)
    result = {"schema": 1, "kind": "clannad-meiki-screens-result", "protocol_sha256": proto_sha,
              "refs_protocol_sha256": freeze["protocol_sha256"],
              "references": {"refs_sha256": freeze["refs_sha256"], "frozen": freeze["frozen"],
                             "scored_lines": freeze["scored_lines"], "reviewer_kind": "assistant"},
              "screens": account, "scores": scores, "empty_screens_output": empty_out,
              "meiki_screen_detections": det,
              "entanglement": entanglement(rows, refs, engines, {i: extra[i][ARM_A] for i in extra}),
              "arm_a": {"arms_sha256": emeta["arms_sha256"], "arms": emeta["arms"], "calls": emeta.get("calls")},
              "arm_b": {"screens_sha256": smeta["screens_sha256"], "arm": smeta["arm"], "pages": smeta["pages"],
                        "time": smeta.get("time"), "time_note": smeta.get("time_note")},
              "engine": smeta["native"]["engine"], "lines_sha256": meta["lines_sha256"],
              "arms_sha256": ameta["arms_sha256"], "engines_sha256": eng_meta["engines_sha256"],
              "code_sha256": code_identity(), "environment": {k: v for k, v in environment(()).items()
                                                               if k in ("python", "system", "machine", "git")},
              "run_environment": {k: smeta["environment"].get(k) for k in ("python", "cpu", "packages", "git")},
              "training_or_selection_use": False, "created": now()}
    CR.assert_public(result)
    CR.write_jsonl(into / "screens.score.private.jsonl", private, exclusive=True)
    CR.write_json(Path(a.out), result, exclusive=True)
    md = screens_markdown(result)
    CR.write_json(into / "screens.score.done.json", {"finished": now(), "result_sha256": CR.sha256_file(a.out)},
                  exclusive=True)
    print(md)


def screens_markdown(result):
    s = result["scores"]["systems"]
    md = ["| system | screens | ref chars | errors | screen CER % [95% CI] | dialogue CER % [95% CI] | "
          "exact screens % [95% CI] |", "|---|---:|---:|---:|---|---|---|"]
    for name, e in s.items():
        d = e["dialogue_cer_ci95"] or [0, 0]
        md.append(f"| {name} | {e['screens']:,} | {e['chars']:,} | {e['errors']:,} | {e['cer']:.2f} "
                  f"[{e['cer_ci95'][0]:.2f}, {e['cer_ci95'][1]:.2f}] | {e['dialogue_cer']:.2f} "
                  f"[{d[0]:.2f}, {d[1]:.2f}] | "
                  f"{e['exact']:.1f} [{e['exact_ci95'][0]:.1f}, {e['exact_ci95'][1]:.1f}] |")
    md += ["", "| pair | screen CER delta [95% CI] | dialogue CER delta [95% CI] | exact delta [95% CI] |",
           "|---|---|---|---|"]
    for k, p in result["scores"]["pairs"].items():
        md.append(f"| {k} | {p['cer']['delta']:+.2f} [{p['cer']['ci95'][0]:+.2f}, {p['cer']['ci95'][1]:+.2f}] | "
                  f"{p['dialogue_cer']['delta']:+.2f} [{p['dialogue_cer']['ci95'][0]:+.2f}, "
                  f"{p['dialogue_cer']['ci95'][1]:+.2f}] | {p['exact']['delta']:+.1f} "
                  f"[{p['exact']['ci95'][0]:+.1f}, {p['exact']['ci95'][1]:+.1f}] |")
    return "\n".join(md)


# ----------------------------------------------------------------------------- size and host speed

def onnx_params(path):
    """Parameters of an ONNX file: initializers plus Constant node tensors, by data type."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(str(path), load_external_data=False)
    by = collections.Counter()
    for t in m.graph.initializer:
        by[onnx.TensorProto.DataType.Name(t.data_type)] += int(np.prod(t.dims)) if len(t.dims) else 1
    for node in m.graph.node:
        if node.op_type == "Constant":
            for at in node.attribute:
                if at.type == onnx.AttributeProto.TENSOR:
                    arr = numpy_helper.to_array(at.t)
                    by[onnx.TensorProto.DataType.Name(at.t.data_type)] += int(arr.size)
    inputs = [{"name": i.name, "shape": [d.dim_value if d.HasField("dim_value") else d.dim_param
                                         for d in i.type.tensor_type.shape.dim]} for i in m.graph.input]
    return {"params": sum(by.values()), "params_by_dtype": dict(sorted(by.items())), "nodes": len(m.graph.node),
            "opset": [{"domain": o.domain or "ai.onnx", "version": o.version} for o in m.opset_import],
            "inputs": inputs}


def cmd_resources(a):
    checked_models()
    out = {"models": {}}
    for role, (fname, repo, rev, size, sha) in MODEL_PINS.items():
        out["models"][role] = {"file": fname, "repo": repo, "revision": rev, "bytes": size, "sha256": sha,
                               "dtype_note": "fp32 ONNX as published", **onnx_params(model_path(role))}
    out["arm_a_models"] = ["rec", "vrec"]
    out["arm_b_models"] = ["det", "rec", "vrec"]
    out["environment"] = environment()
    out["created"] = now()
    CR.write_json(Path(a.into) / "resources.meiki.json", out)
    print(json.dumps({r: {"bytes": m["bytes"], "params": m["params"]} for r, m in out["models"].items()}))


def speed_sample(rows):
    ids = sorted(r["id"] for r in rows)
    pages = sorted({r["page"] for r in rows})
    rng = random.Random(SPEED_SEED)
    lid = rng.sample(ids, min(SPEED_LINES, len(ids)))
    pid = rng.sample(pages, min(SPEED_PAGES, len(pages)))
    return lid, pid


def timed(fn, items, repeats=SPEED_REPEATS, warmup=SPEED_WARMUP):
    for x in items[:warmup]:
        fn(x)
    reps = []
    for _ in range(repeats):
        ms = []
        for x in items:
            t0 = time.perf_counter()
            fn(x)
            ms.append(1000 * (time.perf_counter() - t0))
        reps.append(ms)
    flat = [v for r in reps for v in r]
    return {"items": len(items), "repeats": repeats, "warmup": warmup, **time_stats(flat),
            "repeat_medians_ms": [round(statistics.median(r), 3) for r in reps]}


def cmd_speed(a):
    work, into = Path(a.work), Path(a.into)
    frozen(work)
    rows, meta = CR.load_lines(work)
    by = {r["id"]: r for r in rows}
    lid, pid = speed_sample(rows)
    crops = [np.ascontiguousarray(CR.load_grey(meta["session"], by[i]["line"])) for i in lid]
    sample = {"lines": len(lid), "pages": len(pid), "seed": SPEED_SEED,
              "line_ids_sha256": hashlib.sha256("\n".join(lid).encode()).hexdigest(),
              "page_ids_sha256": hashlib.sha256("\n".join(pid).encode()).hexdigest(),
              "line_widths_px": {"median": int(statistics.median(c.shape[1] for c in crops)),
                                 "max": int(max(c.shape[1] for c in crops))}}
    out = {"engine": a.engine, "sample": sample, "threads": 1, "environment": environment(), "created": now()}
    if a.engine == "meiki":
        import clannad_longrun as LR
        out["network_interfaces"] = require_offline()
        t0 = time.perf_counter()
        m = load_meiki(1)
        out["model_load_ms"] = round(1000 * (time.perf_counter() - t0), 1)
        src = dict(page_sources(rows))
        frames = [frame_bgr(LR.verified_image(Path(meta["session"]), src[p])) for p in pid]
        out["rec_per_line"] = timed(lambda g: read_line(m, g), crops)
        out["det_per_screen"] = timed(lambda f: m.run_detection(f, DET_THRESHOLD), frames, warmup=2)
        out["ocr_per_screen"] = timed(lambda f: m.run_ocr(f, det_threshold=DET_THRESHOLD, rec_threshold=REC_THRESHOLD,
                                                          punct_conf_factor=PUNCT_FACTOR), frames, warmup=2)
    else:
        import vocr3_report as R
        if not a.blobs:
            raise SystemExit("--blobs DIR (the protocol arms' blobs) is required with --engine vocr")
        native = R.build_native(into / "native.speed", a.compiler)
        lib = R.load_library(native)
        out["native"] = R.public_native(native)
        arms = {x["name"]: x for x in CR.protocol()["arms"]}
        out["arms"] = {}
        for name in SPEED_VOCR:
            arm = R.Arm({k: arms[name][k] for k in ("name", "file", "sha256", "finder")}, a.blobs)
            t0 = time.perf_counter()
            reader = R.CReader(arm, lib)
            load = round(1000 * (time.perf_counter() - t0), 1)
            out["arms"][name] = {"sha256": arm.sha256, "blob_bytes": len(arm.data), "model_load_ms": load,
                                 "arena_w8i": reader.runtime.arena_bytes,
                                 "per_line": timed(lambda g, rd=reader: rd.text(g, False), crops)}
    CR.assert_public(out)
    CR.write_json(into / f"speed.{a.engine}.json", out)
    print(json.dumps({k: v for k, v in out.items() if k in ("model_load_ms", "rec_per_line", "det_per_screen",
                                                             "ocr_per_screen", "arms")}, default=str)[:2000])


def cmd_report(a):
    into = Path(a.into)

    def load(name):
        return json.loads((into / name).read_text(encoding="utf-8"))
    verify, res, sm, sv = (load(x) for x in ("verify.json", "resources.meiki.json", "speed.meiki.json",
                                              "speed.vocr.json"))
    deps = json.loads((ROOT / "ports" / "meiki" / "dependencies.json").read_text(encoding="utf-8"))
    out = {"schema": 1, "kind": "clannad-meiki-resources", "protocol_sha256": CR.sha256_file(PROTOCOL_PATH),
           "engine": verify["engine"], "verify_ok": verify["ok"], "models": res["models"],
           "arm_a_models": res["arm_a_models"], "arm_b_models": res["arm_b_models"],
           "cited": {"int8_mnn_recognizer": {k: deps["converted"][k] for k in ("filename", "bytes", "sha256")},
                     "int8_mnn_source": "ports/meiki/dependencies.json (converted; not measured here)",
                     "console": "notes/27-Meiki-hardware-and-project-lessons.md: two CLANNAD jobs on a PCH-1001 in "
                                "the game process, 7,904 / 8,043 ms end to end, recognition 3,483 / 3,478 ms, model "
                                "load 2,153 / 2,149 ms (int8 MNN recognizer, one 864x44 line; not measured here)"},
           "speed": {"meiki": {k: sm[k] for k in ("sample", "threads", "model_load_ms", "rec_per_line",
                                                   "det_per_screen", "ocr_per_screen")},
                     "vocr": {k: sv[k] for k in ("sample", "threads", "arms", "native")},
                     "same_sample": sm["sample"] == sv["sample"],
                     "environment": {"meiki": {k: sm["environment"][k] for k in ("cpu", "python", "packages")},
                                     "vocr": {k: sv["environment"][k] for k in ("cpu", "python")}},
                     "note": "host wall times on this box, one thread each, never a console measurement; the VOCR "
                             "library is the arms' scalar C build (-DVOCR_NO_NEON), onnxruntime uses its own "
                             "vectorised CPU kernels"},
           "created": now()}
    CR.assert_public(out)
    CR.write_json(Path(a.out), out, exclusive=True)
    print(json.dumps({"meiki_rec_ms": sm["rec_per_line"]["median_ms"],
                      "meiki_ocr_ms_per_screen": sm["ocr_per_screen"]["median_ms"],
                      "vocr_ms": {n: x["per_line"]["median_ms"] for n, x in sv["arms"].items()}}))


# ----------------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--into", type=Path, required=True)
    for name in ("lines", "screens"):
        p = sub.add_parser(name)
        p.add_argument("--work", type=Path, required=True)
        p.add_argument("--into", type=Path, required=True)
        p.add_argument("--procs", type=int, default=5, help="worker processes, one thread each (0: in process)")
    p = sub.add_parser("score-screens")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--into", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("resources")
    p.add_argument("--into", type=Path, required=True)
    p = sub.add_parser("speed")
    p.add_argument("--engine", choices=("meiki", "vocr"), required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--into", type=Path, required=True)
    p.add_argument("--blobs", type=Path)
    p.add_argument("--compiler", default="gcc")
    p = sub.add_parser("report")
    p.add_argument("--into", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    {"verify": cmd_verify, "lines": cmd_lines, "screens": cmd_screens, "score-screens": cmd_score_screens,
     "resources": cmd_resources, "speed": cmd_speed, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
