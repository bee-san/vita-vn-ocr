#!/usr/bin/env python3
"""VOCR3 locked report: any set of W8I blobs on vita-test and psp-test, scored like eval/run.py.

  vocr3_report.py shared  (--spec SPEC.json | --arm NAME=FILE[@finder][#sha256] ...) --blobs DIR
                          [--pair CAND:BASE] [--gate CAND:BASE] [--stored NAME:POLARITY:SET=RESULT.json]
                          [--local-set NAME=MANIFEST] [--data EVAL] --out DIR
  vocr3_report.py a9      (--spec | --arm ...) --blobs DIR --out DIR [--vitasdk V] [--arm-cc CC] [--qemu Q]
  vocr3_report.py overlay (--spec | --arm ...) --blobs DIR --overlay CLONE --data EVAL --out DIR
                          [--cmake CMAKE] [--ninja NINJA] [--extend-table]
  vocr3_report.py render  --out DIR [--fresh RESULT.json] [--clannad RESULT.json]

shared: every arm (a W8I blob and the finder it ships with) through the dependency-free C runtime,
built here from runtime/vocr.c as an explicit scalar library (never pyvocr's mtime cache):
  - fixed lines: eval/run.py's own recognizer_mode on every hand-checked line of each set, with
    console polarity (primary, the console's contract) and native polarity; integer errors and
    denominators per set, per style tag, per VN and pooled, full text and hiragana;
  - whole dialogue: each manifest region (regions holding an uncertain line are left out whole)
    at its base box and expanded by 4 and 8 native pixels, the arm's C finder, every returned crop
    recognized with the finder's polarity, joined in finder order and scored by the textbox guard's
    transcript_score (all text and body; full and hiragana) per title;
  - named pairs CAND:BASE: CAND - BASE in points with paired bootstrap 95% intervals over lines
    and over VN clusters (strata with at least --min-clusters VNs);
  - gates CAND:BASE: the VOCR2 fixed-line gate (no set's full-text or hiragana errors increase,
    pooled hiragana errors decrease; primary polarity) and the held-out whole-dialogue guard (no
    title/variant all/body full/hiragana regression; every variant improves console-balanced
    hiragana in both scopes and gains on at least two titles);
  - size and arena: blob bytes and the W8I/W8F arenas from runtime/pyvocr.py (vocr_arena_bytes).
a9: vita/a9_cycles.py on each blob (Vita compiler's machine code under plugin qemu-arm, a cycle
    model as a range). These are host estimates, never console measurements.
overlay: vita-jp-overlay's host vjo-ocr (its own finder, workspace and model checks) built from a
    scratch export of a read-only clone, through its tools/benchmark_local_ocr.py.
render: report.json and report.md from the parts present in DIR (plus fresh/CLANNAD adapter results).

Public JSON holds integer counts and hashes only; predictions go to DIR/private/ (keep it out of Git).
Never use this on a protected set for a model that has no committed protocol (AGENTS.md, docs/vocr3-harness.md).
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "eval"
for _p in (HERE, ROOT / "train", ROOT / "synth", ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import backends  # noqa: E402
import metrics  # noqa: E402
import run as eval_run  # noqa: E402
import sets  # noqa: E402
from fetch import data_dir  # noqa: E402

SHARED_SETS = ("vita-test", "psp-test")
POLARITIES = ("console", "native")
FINDERS = {"grad": 0, "textbox": 2, "textbox_recovery": 3}       # runtime/vocr_lines.h modes
VARIANTS = {"base": 0, "expanded-4": 4, "expanded-8": 8}         # native-pixel ROI expansion
SCOPES, SCRIPTS = ("all", "body"), ("full", "hira")
BUILD_ENV = {"PATH": os.defpath, "LC_ALL": "C", "TZ": "UTC", "SOURCE_DATE_EPOCH": "0"}
# The scalar host build of every frozen comparison since VOCR2 (general_vn_select.SCALAR_FLAGS).
SCALAR_FLAGS = ("-std=c99", "-O2", "-ffp-contract=off", "-DVOCR_NO_NEON", "-Wall", "-Wextra", "-Werror",
                "-Wpedantic", "-Wshadow", "-Wstrict-prototypes", "-Wmissing-prototypes", "-Wcast-align",
                "-Wpointer-arith", "-fPIC", "-shared")
FINDER_FLAGS = ("-std=c99", "-O2", "-ffp-contract=off", "-DVOCR_NO_NEON", "-Wall", "-Wextra", "-Werror")
RUNTIME_SOURCES = ("runtime/vocr.c", "runtime/vocr.h", "runtime/vocr_lines.c", "runtime/vocr_lines.h",
                   "runtime/vocr_lines_test.c")
CODE = ("eval/vocr3_report.py", "eval/run.py", "eval/sets.py", "eval/metrics.py", "eval/backends.py",
        "eval/group_metrics.py", "train/general_vn_textbox_guard.py", "train/export.py", "runtime/pyvocr.py",
        "synth/common.py", "vita/a9_cycles.py", *RUNTIME_SOURCES, "requirements-ci.lock")
SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}")
SHA = re.compile(r"[0-9a-f]{64}")
DEFAULT_BOOTSTRAP = {"reps": 10000, "seed": 0, "min_clusters": 5}
FORBIDDEN_PUBLIC_KEYS = {"text", "pred", "predictions", "reference_text"}


# ---------------------------------------------------------------- small helpers

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def write_new(path, value):
    """Exclusive creation: a result file is never overwritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        if isinstance(value, str):
            f.write(value)
        else:
            json.dump(value, f, ensure_ascii=False, indent=1, sort_keys=True)
            f.write("\n")
    return sha256(path)


def assert_public(value, where="report"):
    """Public parts carry counts and hashes only: no reference or predicted text."""
    if isinstance(value, dict):
        for k, v in value.items():
            if k in FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"{where}: public output would contain text field {k!r}")
            assert_public(v, f"{where}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            assert_public(v, f"{where}[{i}]")


def git_state():
    def git(*a):
        try:
            return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"head": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def environment():
    import importlib.metadata as md
    pk = {}
    for name in ("numpy", "Pillow", "opencv-python-headless", "torch"):
        try:
            pk[name] = md.version(name)
        except md.PackageNotFoundError:
            pk[name] = None
    return {"python": platform.python_version(), "python_executable_sha256": sha256(Path(sys.executable).resolve()),
            "host": platform.node(), "system": platform.system(), "machine": platform.machine(),
            "libc": list(platform.libc_ver()), "packages": pk, "git": git_state(),
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}


def code_identity(files=CODE):
    return {f: sha256(ROOT / f) for f in files}


# ---------------------------------------------------------------- spec: arms, pairs, gates

def parse_arm(text):
    """NAME=FILE[@finder][#sha256] -> arm dict."""
    name, sep, rest = text.partition("=")
    if not sep:
        raise ValueError(f"--arm needs NAME=FILE, got {text!r}")
    want = None
    if "#" in rest:
        rest, want = rest.rsplit("#", 1)
    finder = "grad"
    if "@" in rest:
        rest, finder = rest.rsplit("@", 1)
    arm = {"name": name, "file": rest, "finder": finder}
    if want:
        arm["sha256"] = want
    return arm


def parse_pair(text):
    cand, sep, base = text.partition(":")
    if not sep:
        raise ValueError(f"pair needs CANDIDATE:BASELINE, got {text!r}")
    return {"candidate": cand, "baseline": base}


def parse_stored(text):
    """NAME:POLARITY:SET=RESULT.json (an eval/run.py --out file, re-scored on today's manifest)."""
    head, sep, path = text.partition("=")
    parts = head.split(":")
    if not sep or len(parts) != 3:
        raise ValueError(f"--stored needs NAME:POLARITY:SET=RESULT.json, got {text!r}")
    return {"name": parts[0], "polarity": parts[1], "set": parts[2], "path": path}


def build_spec(spec_path=None, arms=(), pairs=(), gates=(), stored=(), local_sets=(), sets_=None,
               polarities=None, primary=None, bootstrap=None, dialogue=None):
    spec = {"schema": 1, "kind": "vocr3-report-spec", "arms": [], "pairs": [], "gates": [], "stored": [],
            "sets": list(SHARED_SETS), "local_sets": {}, "polarities": list(POLARITIES),
            "primary_polarity": "console", "bootstrap": dict(DEFAULT_BOOTSTRAP), "dialogue": True}
    if spec_path:
        loaded = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        unknown = set(loaded) - set(spec)
        if unknown:
            raise ValueError(f"unknown spec fields {sorted(unknown)}")
        spec.update(loaded)
        spec["bootstrap"] = {**DEFAULT_BOOTSTRAP, **loaded.get("bootstrap", {})}
    spec["arms"] = [dict(a) for a in spec["arms"]] + [parse_arm(a) if isinstance(a, str) else dict(a) for a in arms]
    spec["pairs"] = [dict(p) for p in spec["pairs"]] + [parse_pair(p) if isinstance(p, str) else p for p in pairs]
    spec["gates"] = [dict(g) for g in spec["gates"]] + [parse_pair(g) if isinstance(g, str) else g for g in gates]
    spec["stored"] = [dict(s) for s in spec["stored"]] + [parse_stored(s) if isinstance(s, str) else s
                                                          for s in stored]
    for item in local_sets:
        name, sep, path = item.partition("=")
        if not sep:
            raise ValueError("--local-set needs NAME=MANIFEST")
        spec["local_sets"][name] = path
    if sets_ is not None:
        spec["sets"] = list(sets_)
    if polarities is not None:
        spec["polarities"] = list(polarities)
    if primary is not None:
        spec["primary_polarity"] = primary
    if bootstrap:
        spec["bootstrap"].update({k: v for k, v in bootstrap.items() if v is not None})
    if dialogue is not None:
        spec["dialogue"] = bool(dialogue)
    for a in spec["arms"]:
        a.setdefault("finder", "grad")
    return validate_spec(spec)


def validate_spec(spec):
    if spec.get("schema") != 1 or spec.get("kind") != "vocr3-report-spec":
        raise ValueError("spec must be schema 1, kind vocr3-report-spec")
    names = []
    for a in spec["arms"]:
        extra = set(a) - {"name", "file", "finder", "sha256", "role"}
        if extra:
            raise ValueError(f"arm {a.get('name')}: unknown fields {sorted(extra)}")
        if not isinstance(a.get("name"), str) or not SLUG.fullmatch(a["name"]):
            raise ValueError(f"arm name must be a slug: {a.get('name')!r}")
        if not isinstance(a.get("file"), str) or not a["file"]:
            raise ValueError(f"arm {a['name']}: blob file required")
        if a["finder"] not in FINDERS:
            raise ValueError(f"arm {a['name']}: finder must be one of {sorted(FINDERS)}")
        if "sha256" in a and not (isinstance(a["sha256"], str) and SHA.fullmatch(a["sha256"])):
            raise ValueError(f"arm {a['name']}: sha256 must be 64 lowercase hex digits")
        names.append(a["name"])
    stored_names = sorted({s["name"] for s in spec["stored"]})
    for s in spec["stored"]:
        if not SLUG.fullmatch(s.get("name", "")) or s.get("polarity") not in POLARITIES or not s.get("path"):
            raise ValueError(f"stored arm needs a slug name, a polarity and a path: {s}")
    if len(set(names)) != len(names) or set(names) & set(stored_names):
        raise ValueError("arm names must be unique (blob and stored arms share one namespace)")
    if not names:
        raise ValueError("at least one blob arm is required")
    known = set(names) | set(stored_names)
    for key in ("pairs", "gates"):
        seen = set()
        for p in spec[key]:
            c, b = p.get("candidate"), p.get("baseline")
            if c not in known or b not in known or c == b:
                raise ValueError(f"{key[:-1]} {c}:{b} must name two different known arms")
            if key == "gates" and (c not in names or b not in names):
                raise ValueError(f"gate {c}:{b}: gates compare blob arms only")
            if (c, b) in seen:
                raise ValueError(f"duplicate {key[:-1]} {c}:{b}")
            seen.add((c, b))
    if not set(spec["sets"]) <= set(SHARED_SETS):
        raise ValueError(f"sets must be among {SHARED_SETS}; private captures go in local_sets")
    for name in spec["local_sets"]:
        if name in sets.SETS and name not in sets.LOCAL_MANIFESTS:
            raise ValueError(f"local set {name} would shadow a shared set")
    if not spec["sets"] and not spec["local_sets"]:
        raise ValueError("no set to score")
    if not spec["polarities"] or not set(spec["polarities"]) <= set(POLARITIES):
        raise ValueError(f"polarities must be among {POLARITIES}")
    if spec["primary_polarity"] not in spec["polarities"]:
        raise ValueError("primary polarity must be scored")
    bs = spec["bootstrap"]
    if set(bs) != set(DEFAULT_BOOTSTRAP) or any(type(bs[k]) is not int for k in bs) or bs["reps"] < 100 \
            or bs["min_clusters"] < 2 or bs["seed"] < 0:
        raise ValueError(f"bootstrap needs integer reps >= 100, seed >= 0, min_clusters >= 2: {bs}")
    return spec


def public_spec(spec):
    """The spec as recorded in public results: file names and hashes, no local paths."""
    out = json.loads(json.dumps(spec))
    for a in out["arms"]:
        a["file"] = Path(a["file"]).name
    out["stored"] = [{"name": s["name"], "polarity": s["polarity"], "set": s["set"], "file": Path(s["path"]).name,
                      "sha256": sha256(s["path"])} for s in spec["stored"]]
    out["local_sets"] = {n: {"file": Path(p).name, "sha256": sha256(p)} for n, p in spec["local_sets"].items()}
    return out


class Arm:
    """A blob arm: verified bytes, header, CRC, int8 (deployment W8I) and finder."""

    def __init__(self, spec_arm, blobs_dir):
        self.name, self.finder = spec_arm["name"], spec_arm["finder"]
        path = Path(spec_arm["file"])
        self.path = path if path.is_absolute() else Path(blobs_dir or ".") / path
        self.data = self.path.read_bytes()
        self.sha256 = hashlib.sha256(self.data).hexdigest()
        if spec_arm.get("sha256") and spec_arm["sha256"] != self.sha256:
            raise ValueError(f"arm {self.name}: {self.path} has sha256 {self.sha256}, the spec pins "
                             f"{spec_arm['sha256']}")
        import export
        self.blob = export.read_blob(self.data)          # header + CRC
        if not self.blob["int8"]:
            raise ValueError(f"arm {self.name}: deployment W8I needs an int8 blob")
        self.file_name = self.path.name

    def identity(self):
        return {"name": self.name, "file": self.file_name, "sha256": self.sha256, "bytes": len(self.data),
                "finder": self.finder, "finder_mode": FINDERS[self.finder]}


# ---------------------------------------------------------------- native builds (explicit, hashed)

def compiler_identity(compiler="gcc"):
    found = shutil.which(str(compiler), path=BUILD_ENV["PATH"]) or shutil.which(str(compiler))
    if found is None:
        raise ValueError(f"C compiler {compiler!r} not found")
    exe = Path(found).resolve()
    version = subprocess.run([str(exe), "--version"], check=True, capture_output=True, text=True,
                             env=BUILD_ENV).stdout.strip()
    return {"executable": str(exe), "sha256": sha256(exe), "version": version.splitlines()[0] if version else ""}


def build_native(directory, compiler="gcc"):
    """The scalar recognizer library and the finder's batch driver, compiled now with fixed flags."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cc = compiler_identity(compiler)
    sources = {f: sha256(ROOT / f) for f in RUNTIME_SOURCES}
    lib, finder = directory / "libvocr_scalar.so", directory / "vocr_lines_driver"
    lib_argv = [cc["executable"], *SCALAR_FLAGS, "-o", str(lib), str(ROOT / "runtime/vocr.c"), "-lm"]
    fin_argv = [cc["executable"], *FINDER_FLAGS, str(ROOT / "runtime/vocr_lines.c"),
                str(ROOT / "runtime/vocr_lines_test.c"), "-o", str(finder)]
    for argv in (lib_argv, fin_argv):
        subprocess.run(argv, check=True, capture_output=True, text=True, cwd=ROOT, env=BUILD_ENV)
    if sources != {f: sha256(ROOT / f) for f in RUNTIME_SOURCES}:
        raise ValueError("runtime sources changed during the build")

    def public(argv):     # the recorded command line names files relative to the repo / build dir
        return [str(Path(x).relative_to(ROOT)) if x.startswith(str(ROOT)) else
                (Path(x).name if x.startswith(str(directory)) else x) for x in argv]
    return {"compiler": cc, "sources": sources, "directory": str(directory),
            "library": {"path": str(lib), "name": lib.name, "sha256": sha256(lib), "argv": public(lib_argv)},
            "finder": {"path": str(finder), "name": finder.name, "sha256": sha256(finder), "argv": public(fin_argv)}}


def public_native(identity):
    out = json.loads(json.dumps(identity))
    out.pop("directory", None)
    for key in ("library", "finder"):
        out[key].pop("path", None)
    return out


def verify_native(identity):
    if {f: sha256(ROOT / f) for f in RUNTIME_SOURCES} != identity["sources"]:
        raise ValueError("runtime sources changed since the native build")
    for key in ("library", "finder"):
        if sha256(identity[key]["path"]) != identity[key]["sha256"]:
            raise ValueError(f"built {key} changed since the native build")
    if sha256(identity["compiler"]["executable"]) != identity["compiler"]["sha256"]:
        raise ValueError("compiler changed since the native build")


def load_library(identity):
    from pyvocr import load_lib
    verify_native(identity)
    return load_lib(identity["library"]["path"])


class CReader(backends.Recognizer):
    """backends.NativeVocr (vocr:PATH) with an explicit library: deployment W8I, max width 1024."""

    def __init__(self, arm, lib):
        from pyvocr import Vocr
        self.arm = arm
        self.runtime = Vocr(arm.data, mode="w8i", max_width=1024, lib=lib)
        self.name = arm.name + "/w8i"
        self.info = {"backend": "vocr", "mode": "w8i", "sha256": arm.sha256, "blob_bytes": self.runtime.blob_bytes,
                     "arena_bytes": self.runtime.arena_bytes, "vocab_set": set(arm.blob["vocab"][1:])}
        self.forwards = 0

    def read(self, crops):
        from common import to_gray
        out = []
        for c in crops:
            out.append(self.runtime.run(to_gray(c) if c.ndim == 3 else c))
            self.forwards += 1
        return out

    def text(self, image, invert):
        """One finder crop (grey or RGBA) with the finder's polarity flag."""
        self.forwards += 1
        return self.runtime.text(np.ascontiguousarray(image), invert=bool(invert))


def resources(arm, lib):
    """Blob bytes and arenas as the C runtime reports them, checked against the exporter's mirror."""
    from pyvocr import Vocr

    import export
    w8i = Vocr(arm.data, mode="w8i", max_width=1024, lib=lib).arena_bytes
    w8f = Vocr(arm.data, mode="w8f", max_width=1024, lib=lib).arena_bytes
    b = arm.blob
    if (export.plan_arena(b, 1024, 0, 0, export.MODE_W8I), export.plan_arena(b, 1024, 0, 0, export.MODE_W8F)) \
            != (w8i, w8f):
        raise ValueError(f"{arm.name}: C arena differs from train/export.py's plan_arena")
    macs, head = export.macs_per_frame(b)
    return {"blob_bytes": len(arm.data), "sha256": arm.sha256, "classes": b["n_classes"], "params": b["params"],
            "height": b["height"], "max_width": b["max_width"], "stride": b["stride"], "ops": len(b["ops"]),
            "arena_w8i": w8i, "arena_w8f": w8f, "macs_per_frame": macs, "head_macs_per_frame": head,
            "macs_640px_line": macs * (640 // b["stride"]), "finder": arm.finder}


# ---------------------------------------------------------------- finder driver (one batch per mode)

def run_finder(binary, images, mode):
    """images: HxW grey or HxWx4 RGBA uint8. Returns [(rc, boxes, dark, truncated)] in input order."""
    if not images:
        return []
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        for img in images:
            img = np.ascontiguousarray(img, dtype=np.uint8)
            h, w = img.shape[:2]
            bpp = 1 if img.ndim == 2 else img.shape[2]
            if bpp not in (1, 4):
                raise ValueError("finder input must be grey or RGBA")
            f.write(struct.pack("<5I", w, h, bpp, mode, w * bpp))
            f.write(img.tobytes())
        batch = f.name
    try:
        out = subprocess.run([str(binary), batch], check=True, capture_output=True, text=True).stdout
    finally:
        os.unlink(batch)
    rows = out.splitlines()
    if len(rows) != len(images):
        raise RuntimeError(f"finder driver returned {len(rows)} results for {len(images)} images")
    results = []
    for row in rows:
        v = [int(t) for t in row.split()]
        rc, n, truncated = v[0], v[1], bool(v[2])
        boxes = [tuple(v[3 + 5 * i:7 + 5 * i]) for i in range(n)]
        dark = [v[7 + 5 * i] for i in range(n)]
        results.append((rc, boxes, dark, truncated))
    return results


# ---------------------------------------------------------------- fixed-line scoring (eval/run.py)

def line_record(row):
    h = metrics.score_hiragana([row])
    return {"id": row["id"], "vn": row.get("vn") or "-", "styles": sorted(row.get("styles") or []),
            "n": row["n"], "errors": row["errors"], "hira_chars": h["chars"], "hira_errors": h["errors"],
            "inverted": bool(row.get("inverted"))}


def score_fixed(reader, set_name, data, polarity):
    """eval/run.py's recognizer_mode, unchanged, then integer per-line records."""
    args = argparse.Namespace(pad=None, limit=None, polarity=polarity)
    res = eval_run.recognizer_mode(reader, set_name, data, args)
    lines = [line_record(r) for r in res["lines"]]
    if (sum(x["errors"] for x in lines), sum(x["n"] for x in lines)) != (res["all"]["errors"], res["all"]["chars"]) \
            or sum(x["hira_errors"] for x in lines) != res["hiragana"]["errors"] \
            or sum(x["hira_chars"] for x in lines) != res["hiragana"]["chars"]:
        raise RuntimeError(f"{set_name}: per-line counts do not add up to recognizer_mode's totals")
    private = {"set": set_name, "polarity": polarity, "all": res["all"], "hiragana": res["hiragana"],
               "oov_chars": res["oov_chars"], "lines": res["lines"]}
    return lines, {"inverted_lines": res["inverted_lines"], "oov_chars": res["oov_chars"],
                   "host_ms_per_line": res["host_ms_per_line"]}, private


def score_stored(path, set_name):
    """An earlier eval/run.py --out result, re-scored on today's manifest (train/bar.py's rule)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if set_name not in d.get("sets", {}):
        raise ValueError(f"{path}: no {set_name} results")
    preds = {r["id"]: r["pred"] for r in d["sets"][set_name]["lines"]}
    rows = []
    for r in sets.read_manifest(set_name):
        if r.get("uncertain"):
            continue
        if r["id"] not in preds:
            raise ValueError(f"{path}: {set_name} line {r['id']} has no stored prediction")
        rows.append({"id": r["id"], "vn": r["vn"], "styles": r.get("styles") or [], "text": r["text"],
                     "pred": preds[r["id"]]})
    metrics.score_lines(rows)
    return [line_record(r) for r in rows], d.get("polarity")


def tally(lines):
    return {"lines": len(lines), "chars": sum(x["n"] for x in lines), "errors": sum(x["errors"] for x in lines),
            "exact": sum(x["errors"] == 0 for x in lines), "hira_chars": sum(x["hira_chars"] for x in lines),
            "hira_errors": sum(x["hira_errors"] for x in lines), "vns": len({x["vn"] for x in lines})}


def style_tags(lines):
    present = {t for x in lines for t in x["styles"]}
    return [t for t in sets.STYLES if t in present] + sorted(present - set(sets.STYLES))


def summarize(lines_by_set):
    out = {}
    for s, lines in lines_by_set.items():
        out[s] = {"all": tally(lines), "styles": {t: tally([x for x in lines if t in x["styles"]])
                                                   for t in style_tags(lines)},
                  "games": {vn: tally([x for x in lines if x["vn"] == vn]) for vn in sorted({x["vn"] for x in lines})}}
    pooled = [x for lines in lines_by_set.values() for x in lines]
    out["pooled"] = {"all": tally(pooled), "styles": {t: tally([x for x in pooled if t in x["styles"]])
                                                      for t in style_tags(pooled)}}
    return out


# ---------------------------------------------------------------- paired bootstraps

def paired_bootstrap(a, b, reps=10000, seed=0):
    """CER(a) - CER(b) in points over shared units, resampling units: a, b = {unit: (errors, chars)}.
    The train2 review's boot.py exactly: sorted ids, default_rng(seed).integers(0, N, (reps, N)),
    numpy's linear 2.5/97.5 percentiles. Resamples with no reference character are dropped."""
    ids = sorted(a)
    if set(ids) != set(b) or any(a[i][1] != b[i][1] for i in ids):
        raise ValueError("paired bootstrap needs identical units and denominators")
    ea = np.array([a[i][0] for i in ids], np.float64)
    eb = np.array([b[i][0] for i in ids], np.float64)
    n = np.array([a[i][1] for i in ids], np.float64)
    out = {"units": len(ids), "chars": int(n.sum()), "candidate_errors": int(ea.sum()),
           "baseline_errors": int(eb.sum()), "reps": reps, "seed": seed}
    if not ids or n.sum() == 0:
        return dict(out, delta=None, ci95=None, dropped_resamples=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), (reps, len(ids)))
    tot = n[idx].sum(1)
    ok = tot > 0
    ds = (ea[idx].sum(1)[ok] - eb[idx].sum(1)[ok]) / tot[ok]
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return dict(out, delta=round(100 * (ea.sum() - eb.sum()) / n.sum(), 6),
                ci95=[round(100 * float(lo), 6), round(100 * float(hi), 6)], dropped_resamples=int((~ok).sum()))


def cluster_bootstrap(a, b, cluster, reps=10000, seed=0, min_clusters=5):
    """The same difference with whole clusters (VNs) resampled: sorted cluster ids, cluster sums."""
    keys = sorted({cluster[i] for i in a})
    sums = {}
    for i in a:
        s = sums.setdefault(cluster[i], [0, 0, 0])
        s[0] += a[i][0]
        s[1] += b[i][0]
        s[2] += a[i][1]
    out = {"clusters": len(keys), "min_clusters": min_clusters, "reps": reps, "seed": seed}
    if len(keys) < min_clusters:
        return dict(out, ci95=None, reason=f"fewer than {min_clusters} VN clusters")
    ea = np.array([sums[k][0] for k in keys], np.float64)
    eb = np.array([sums[k][1] for k in keys], np.float64)
    n = np.array([sums[k][2] for k in keys], np.float64)
    if n.sum() == 0:
        return dict(out, ci95=None, reason="no reference characters")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), (reps, len(keys)))
    tot = n[idx].sum(1)
    ok = tot > 0
    ds = (ea[idx].sum(1)[ok] - eb[idx].sum(1)[ok]) / tot[ok]
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return dict(out, ci95=[round(100 * float(lo), 6), round(100 * float(hi), 6)], dropped_resamples=int((~ok).sum()))


def compare_lines(cand, base, bootstrap):
    """cand/base: {set: [line records]}; the sets both arms cover, each with identical lines, and their
    pooled union. Returns per-stratum CAND - BASE with line and VN-cluster intervals."""
    reps, seed, mc = bootstrap["reps"], bootstrap["seed"], bootstrap["min_clusters"]
    common = [s for s in cand if s in base]
    if not common:
        return {}
    out = {}
    for s in common + ["pooled"]:
        cl = [x for k in common for x in cand[k]] if s == "pooled" else cand[s]
        bl = [x for k in common for x in base[k]] if s == "pooled" else base[s]
        bmap = {x["id"]: x for x in bl}
        if {x["id"] for x in cl} != set(bmap) or any(bmap[x["id"]]["n"] != x["n"] for x in cl):
            raise ValueError(f"{s}: candidate and baseline were scored on different lines")
        strata = {"all": None, **{f"style:{t}": t for t in style_tags(cl)}}
        out[s] = {}
        for label, tag in strata.items():
            sel = [x for x in cl if tag is None or tag in x["styles"]]
            vn = {x["id"]: x["vn"] for x in sel}
            entry = {}
            for metric, (ek, nk) in {"full": ("errors", "n"), "hira": ("hira_errors", "hira_chars")}.items():
                a = {x["id"]: (x[ek], x[nk]) for x in sel}
                b = {x["id"]: (bmap[x["id"]][ek], bmap[x["id"]][nk]) for x in sel}
                entry[metric] = {"lines": paired_bootstrap(a, b, reps, seed),
                                 "vn_clusters": cluster_bootstrap(a, b, vn, reps, seed, mc)}
            entry["exact"] = {"candidate": sum(x["errors"] == 0 for x in sel),
                              "baseline": sum(bmap[x["id"]]["errors"] == 0 for x in sel), "lines": len(sel)}
            out[s][label] = entry
    return out


# ---------------------------------------------------------------- gates

def fixed_line_gate(cand, base):
    """VOCR2's fixed-line gate (docs/vocr2-qualification.md): no set's full-text or hiragana errors
    increase, and pooled hiragana errors decrease. cand/base: {set: tally}."""
    reasons, rows = [], {}
    for s in sorted(base):
        c, b = cand[s], base[s]
        if (c["lines"], c["chars"], c["hira_chars"]) != (b["lines"], b["chars"], b["hira_chars"]):
            reasons.append(f"{s}: candidate and baseline denominators differ")
        if c["errors"] > b["errors"]:
            reasons.append(f"{s}: full-text errors increase {b['errors']} -> {c['errors']}")
        if c["hira_errors"] > b["hira_errors"]:
            reasons.append(f"{s}: hiragana errors increase {b['hira_errors']} -> {c['hira_errors']}")
        rows[s] = {"chars": b["chars"], "hira_chars": b["hira_chars"], "baseline_errors": b["errors"],
                   "candidate_errors": c["errors"], "baseline_hira_errors": b["hira_errors"],
                   "candidate_hira_errors": c["hira_errors"]}
    pooled = {"chars": sum(base[s]["hira_chars"] for s in base),
              "baseline_errors": sum(base[s]["hira_errors"] for s in base),
              "candidate_errors": sum(cand[s]["hira_errors"] for s in base)}
    if not pooled["candidate_errors"] < pooled["baseline_errors"]:
        reasons.append(f"pooled hiragana errors do not decrease ({pooled['baseline_errors']} -> "
                       f"{pooled['candidate_errors']})")
    return {"passed": not reasons, "reasons": reasons, "sets": rows, "pooled_hiragana": pooled}


def balanced_rate(groups, scope):
    """Consoles weighted 1/2 each, titles equally within a console, exact fractions."""
    rates = []
    for console in ("psp", "vita"):
        scores = [g[scope]["hira"] for g in groups.values() if g["platform"] == console]
        if not scores:
            return None
        if any(s["chars"] <= 0 for s in scores):
            raise ValueError(f"{console}: a title has no {scope} hiragana reference characters")
        rates.append(sum((Fraction(s["errors"], s["chars"]) for s in scores), Fraction()) / len(scores))
    return sum(rates, Fraction()) / 2


def dialogue_guard(base, cand):
    """VOCR2's held-out whole-dialogue guard: no title/variant all/body full/hiragana error increase;
    every variant strictly improves the console-balanced hiragana rate in both scopes and gains (all
    and body hiragana both fewer) on at least two titles. base/cand: {variant: {title: group}}."""
    reasons, failed, gains, rates = [], [], {}, {}
    for variant in VARIANTS:
        b, c = base.get(variant), cand.get(variant)
        if b is None or c is None or set(b) != set(c):
            reasons.append(f"{variant}: title identities differ")
            continue
        local = set()
        for key in sorted(b):
            if c[key].get("reference_sha256") != b[key].get("reference_sha256"):
                reasons.append(f"{variant}/{key}: references differ")
            for scope in SCOPES:
                for script in SCRIPTS:
                    bb, cc = b[key][scope][script], c[key][scope][script]
                    if bb["chars"] != cc["chars"]:
                        reasons.append(f"{variant}/{key}/{scope}/{script}: denominators differ")
                    elif cc["errors"] > bb["errors"]:
                        failed.append({"variant": variant, "group": key, "scope": scope, "metric": script,
                                       "chars": bb["chars"], "baseline_errors": bb["errors"],
                                       "candidate_errors": cc["errors"]})
            if c[key]["all"]["hira"]["errors"] < b[key]["all"]["hira"]["errors"] and \
                    c[key]["body"]["hira"]["errors"] < b[key]["body"]["hira"]["errors"]:
                local.add(key)
        gains[variant] = sorted(local)
        if len(local) < 2:
            reasons.append(f"{variant}: fewer than two titles gain all and body hiragana")
        rates[variant] = {}
        for scope in SCOPES:
            try:
                rb, rc = balanced_rate(b, scope), balanced_rate(c, scope)
            except ValueError as e:
                reasons.append(f"{variant}/{scope}: {e}")
                continue
            if rb is None or rc is None:
                reasons.append(f"{variant}/{scope}: both consoles are needed for the balanced rate")
                continue
            rates[variant][scope] = {"baseline": f"{rb.numerator}/{rb.denominator}",
                                     "candidate": f"{rc.numerator}/{rc.denominator}",
                                     "baseline_float": float(rb), "candidate_float": float(rc)}
            if not rc < rb:
                reasons.append(f"{variant}/{scope}: no strict console-balanced hiragana gain")
    reasons += [f"{f['variant']}/{f['group']}/{f['scope']}/{f['metric']}: {f['baseline_errors']} -> "
                f"{f['candidate_errors']}" for f in failed]
    checks = sum(len(base.get(v, {})) for v in VARIANTS) * len(SCOPES) * len(SCRIPTS)
    return {"passed": not reasons, "reasons": reasons, "failed_cells": failed, "title_checks": checks,
            "gain_titles": gains, "balanced_hiragana": rates}


# ---------------------------------------------------------------- whole-dialogue mode

def expand(box, pad, dims):
    x, y, w, h = box
    x0, y0 = max(0, x - pad), max(0, y - pad)
    return [x0, y0, min(dims[0], x + w + pad) - x0, min(dims[1], y + h + pad) - y0]


def dialogue_regions(set_name, data):
    """Every manifest region of a set; regions holding an uncertain line are excluded whole."""
    platform_ = sets.SETS[set_name]
    out, excluded = [], {"regions": 0, "certain_lines": 0, "chars": 0}
    for path, region, rows in sets.regions(set_name, data):
        if any(r.get("uncertain") for r in rows):
            certain = [r for r in rows if not r.get("uncertain")]
            excluded["regions"] += 1
            excluded["certain_lines"] += len(certain)
            excluded["chars"] += sum(len(metrics.norm(r["text"])) for r in certain)
            continue
        vns = {r["vn"] for r in rows}
        if len(vns) != 1:
            raise ValueError(f"{set_name}: region {region} spans several VNs")
        lines = [{"id": r["id"], "text": r["text"], "styles": r.get("styles") or []}
                 for r in sorted(rows, key=lambda r: (r["box"][1], r["box"][0]))]
        out.append({"set": set_name, "platform": platform_, "vn": rows[0]["vn"], "path": path,
                    "region": list(region), "region_id": f"{Path(path).name}@{','.join(map(str, region))}",
                    "lines": lines})
    return out, excluded


def score_dialogue(arms, readers, finder_binary, set_names, data, record=None):
    """All arms on every region and variant. Returns ({arm: {variant: {title: group}}},
    {arm: [per-region integer records]}, accounting)."""
    from general_vn_textbox_guard import transcript_score
    groups = {a.name: {v: {} for v in VARIANTS} for a in arms}
    units = {a.name: [] for a in arms}
    account = {"sets": {}, "detections": {a.name: 0 for a in arms}, "blank_outputs": {a.name: 0 for a in arms},
               "zero_detection_cases": {a.name: 0 for a in arms}}
    modes = sorted({a.finder for a in arms})
    for set_name in set_names:
        regions, excluded = dialogue_regions(set_name, data)
        cache = sets.ImageCache()
        cases = []
        for reg in regions:
            img = cache.get(reg["path"])
            dims = [img.shape[1], img.shape[0]]
            for variant, pad in VARIANTS.items():
                x, y, w, h = expand(reg["region"], pad, dims)
                rgba = np.concatenate([img[y:y + h, x:x + w], np.full((h, w, 1), 255, np.uint8)], axis=2)
                cases.append((reg, variant, np.ascontiguousarray(rgba)))
        found = {m: run_finder(finder_binary, [c[2] for c in cases], FINDERS[m]) for m in modes}
        account["sets"][set_name] = {"regions": len(regions), "cases": len(cases), "excluded": excluded,
                                     "lines": sum(len(r["lines"]) for r in regions),
                                     "chars": sum(len(metrics.norm(x["text"])) for r in regions for x in r["lines"])}
        for i, (reg, variant, rgba) in enumerate(cases):
            for arm in arms:
                rc, boxes, dark, truncated = found[arm.finder][i]
                if rc != 0 or truncated or len(boxes) != len(dark) or any(d not in (0, 1) for d in dark):
                    raise RuntimeError(f"{arm.name} {reg['region_id']} {variant}: finder rc={rc} "
                                       f"truncated={truncated}; no blank coercion")
                preds = []
                for (bx, by, bw, bh), d in zip(boxes, dark):
                    if bw <= 0 or bh <= 0 or bx + bw > rgba.shape[1] or by + bh > rgba.shape[0]:
                        raise RuntimeError(f"{arm.name} {reg['region_id']}: finder box outside the ROI")
                    t = readers[arm.name].text(rgba[by:by + bh, bx:bx + bw], d)
                    if not isinstance(t, str):
                        raise RuntimeError("reader returned no text")
                    preds.append(t)
                score = transcript_score(reg["lines"], preds)
                key = f"{reg['platform']}:{reg['vn']}"
                g = groups[arm.name][variant].setdefault(key, {
                    "platform": reg["platform"], "vn": reg["vn"], "regions": 0, "detections": 0,
                    "blank_outputs": 0, "zero_detection_regions": 0, "region_ids": [],
                    **{s: {t: {"chars": 0, "errors": 0} for t in SCRIPTS} for s in SCOPES}})
                g["regions"] += 1
                g["detections"] += len(boxes)
                g["blank_outputs"] += sum(not metrics.norm(p) for p in preds)
                g["zero_detection_regions"] += not boxes
                g["region_ids"].append(reg["region_id"] + "|" + "\n".join(x["id"] + "\0" + x["text"]
                                                                         for x in reg["lines"]))
                for s in SCOPES:
                    for t in SCRIPTS:
                        for k in ("chars", "errors"):
                            g[s][t][k] += score[s][t][k]
                account["detections"][arm.name] += len(boxes)
                account["blank_outputs"][arm.name] += sum(not metrics.norm(p) for p in preds)
                account["zero_detection_cases"][arm.name] += not boxes
                units[arm.name].append({"unit": f"{set_name}|{reg['region_id']}", "set": set_name,
                                        "vn": reg["vn"], "platform": reg["platform"], "variant": variant,
                                        "detections": len(boxes), **{f"{s}_{t}_{k}": score[s][t][k]
                                                                      for s in SCOPES for t in SCRIPTS
                                                                      for k in ("chars", "errors")}})
                if record:
                    record({"arm": arm.name, "set": set_name, "region_id": reg["region_id"], "variant": variant,
                            "boxes": [list(b) for b in boxes], "dark": dark, "predictions": preds,
                            "score": score})
    for arm in groups.values():
        for variant in arm.values():
            for g in variant.values():
                g["reference_sha256"] = hashlib.sha256("\n\n".join(sorted(g.pop("region_ids"))).encode()).hexdigest()
    return groups, units, account


def dialogue_totals(groups):
    """{variant: {scope: {platform|pooled: {script: counts}}}} from per-title groups."""
    out = {}
    for variant, titles in groups.items():
        out[variant] = {}
        for plat in ("vita", "psp", "pooled"):
            sel = [g for g in titles.values() if plat == "pooled" or g["platform"] == plat]
            if not sel:
                continue
            out[variant][plat] = {s: {t: {k: sum(g[s][t][k] for g in sel) for k in ("chars", "errors")}
                                      for t in SCRIPTS} for s in SCOPES}
            out[variant][plat]["regions"] = sum(g.get("regions", g.get("frames", 0)) for g in sel)
            out[variant][plat]["detections"] = sum(g["detections"] for g in sel)
    return out


def compare_dialogue(cand_units, base_units, bootstrap):
    """Per variant and platform: CAND - BASE with region and VN-cluster bootstraps per scope/script."""
    reps, seed, mc = bootstrap["reps"], bootstrap["seed"], bootstrap["min_clusters"]
    out = {}
    bmap = {(u["unit"], u["variant"]): u for u in base_units}
    for variant in VARIANTS:
        out[variant] = {}
        for plat in ("vita", "psp", "pooled"):
            sel = [u for u in cand_units if u["variant"] == variant and (plat == "pooled" or u["platform"] == plat)]
            if not sel:
                continue
            out[variant][plat] = {}
            vn = {u["unit"]: u.get("cluster", u["vn"]) for u in sel}
            for s in SCOPES:
                for t in SCRIPTS:
                    ck, ek = f"{s}_{t}_chars", f"{s}_{t}_errors"
                    a = {u["unit"]: (u[ek], u[ck]) for u in sel}
                    b = {u["unit"]: (bmap[(u["unit"], variant)][ek], bmap[(u["unit"], variant)][ck]) for u in sel}
                    out[variant][plat][f"{s}/{t}"] = {"regions": paired_bootstrap(a, b, reps, seed),
                                                      "vn_clusters": cluster_bootstrap(a, b, vn, reps, seed, mc)}
    return out


# ---------------------------------------------------------------- shared command

def shared(spec, blobs_dir, data, out, compiler="gcc"):
    out = Path(out)
    if (out / "shared.json").exists():
        raise FileExistsError(f"{out}/shared.json exists: a report is never overwritten; use a new --out")
    out.mkdir(parents=True, exist_ok=True)
    for name, manifest in spec["local_sets"].items():
        if name not in sets.LOCAL_MANIFESTS:
            sets.register_local(name, manifest)
    set_names = list(spec["sets"]) + list(spec["local_sets"])
    arms = [Arm(a, blobs_dir) for a in spec["arms"]]
    native = build_native(out / "native", compiler)
    lib = load_library(native)
    readers = {a.name: CReader(a, lib) for a in arms}
    private = out / "private"
    pspec = public_spec(spec)
    report = {"schema": 1, "kind": "vocr3-shared-report", "spec": pspec, "spec_sha256": digest(pspec),
              "arms": {a.name: a.identity() for a in arms}, "native": public_native(native),
              "code_sha256": code_identity(), "environment": environment(), "estimates_are_not_measurements": True,
              "datasets": {}, "fixed_line": {}, "pairs": {}, "gates": {}}
    for s in set_names:
        manifest = sets.LOCAL_MANIFESTS.get(s, sets.MANIFESTS / f"{s}.jsonl")
        rows = sets.read_manifest(s)
        report["datasets"][s] = {"manifest_sha256": sha256(manifest), "platform": sets.SETS[s],
                                 "lines": sum(not r.get("uncertain") for r in rows),
                                 "uncertain_lines": sum(bool(r.get("uncertain")) for r in rows),
                                 "vns": len({r["vn"] for r in rows}),
                                 "sessions": sorted({r["session"] for r in rows if r.get("session")})}
    report["resources"] = {a.name: resources(a, lib) for a in arms}
    lines = {pol: {} for pol in spec["polarities"]}
    for pol in spec["polarities"]:
        report["fixed_line"][pol] = {}
        for a in arms:
            per_set, extra = {}, {}
            for s in set_names:
                per_set[s], extra[s], priv = score_fixed(readers[a.name], s, data, pol)
                write_new(private / f"fixed_{a.name}_{pol}_{s}.json", priv)
            lines[pol][a.name] = per_set
            report["fixed_line"][pol][a.name] = {"summary": summarize(per_set), "accounting": extra,
                                                 "lines": per_set}
    for st in spec["stored"]:
        if st["set"] not in set_names or st["polarity"] not in lines:
            raise ValueError(f"stored arm {st['name']}: {st['set']}/{st['polarity']} is not scored in this run")
        recs, declared = score_stored(st["path"], st["set"])
        if declared not in (None, st["polarity"]):
            raise ValueError(f"stored arm {st['name']}: {st['path']} was scored with polarity {declared}")
        lines[st["polarity"]].setdefault(st["name"], {})[st["set"]] = recs
        entry = report["fixed_line"][st["polarity"]].setdefault(st["name"], {"stored": [], "lines": {}})
        entry["stored"].append({"set": st["set"], "file": Path(st["path"]).name,
                                "sha256": sha256(st["path"]), "declared_polarity": declared})
        entry["lines"][st["set"]] = recs
        entry["summary"] = summarize(entry["lines"])
        if set(entry["lines"]) != set(set_names):
            entry["summary"].pop("pooled")       # a pooled figure over fewer sets would not compare
    for p in spec["pairs"]:
        key = f"{p['candidate']}:{p['baseline']}"
        report["pairs"][key] = {}
        for pol, by_arm in lines.items():
            c, b = by_arm.get(p["candidate"]), by_arm.get(p["baseline"])
            if c is None or b is None:
                continue
            compared = compare_lines(c, b, spec["bootstrap"])
            if compared:
                report["pairs"][key][pol] = compared
    dialogue = None
    if spec["dialogue"]:
        recs = []
        groups, units, account = score_dialogue(arms, readers, native["finder"]["path"], set_names, data,
                                                record=recs.append)
        write_new(private / "dialogue_predictions.json", recs)
        dialogue = {"groups": groups, "totals": {a: dialogue_totals(g) for a, g in groups.items()},
                    "units": units, "accounting": account, "variants": VARIANTS, "pairs": {}}
        for p in spec["pairs"]:
            if p["candidate"] in units and p["baseline"] in units:
                dialogue["pairs"][f"{p['candidate']}:{p['baseline']}"] = compare_dialogue(
                    units[p["candidate"]], units[p["baseline"]], spec["bootstrap"])
        report["dialogue"] = dialogue
    primary = spec["primary_polarity"]
    for g in spec["gates"]:
        key = f"{g['candidate']}:{g['baseline']}"
        fl = {pol: fixed_line_gate({s: report["fixed_line"][pol][g["candidate"]]["summary"][s]["all"]
                                    for s in set_names},
                                   {s: report["fixed_line"][pol][g["baseline"]]["summary"][s]["all"]
                                    for s in set_names}) for pol in spec["polarities"]}
        entry = {"fixed_line": fl[primary], "fixed_line_polarity": primary,
                 "fixed_line_other_polarities": {p: v for p, v in fl.items() if p != primary}}
        if dialogue is not None:
            entry["dialogue"] = dialogue_guard(dialogue["groups"][g["baseline"]], dialogue["groups"][g["candidate"]])
            entry["passed"] = entry["fixed_line"]["passed"] and entry["dialogue"]["passed"]
        else:
            entry["passed"] = None
            entry["note"] = "whole-dialogue guard not run (dialogue disabled): no qualification decision"
        report["gates"][key] = entry
    report["forwards"] = {a: r.forwards for a, r in readers.items()}
    assert_public(report)
    write_new(out / "shared.json", report)
    write_new(out / "shared.md", render_markdown({"shared": report}))
    return report


# ---------------------------------------------------------------- A9 estimate (vita/a9_cycles.py)

def a9(spec, blobs_dir, out, vitasdk=None, arm_cc=None, qemu=None, data=None, lines=40, modes="w8i,w8f",
       runner=subprocess.run):
    out = Path(out)
    if (out / "a9.json").exists():
        raise FileExistsError(f"{out}/a9.json exists")
    arms = [Arm(a, blobs_dir) for a in spec["arms"]]
    script = ROOT / "vita" / "a9_cycles.py"
    vitasdk = vitasdk or os.environ.get("VITASDK")
    if not vitasdk:
        raise SystemExit("set VITASDK (vita/scripts/fetch_vitasdk.sh) or pass --vitasdk")
    pin = Path(vitasdk) / ".vocr-pin"
    synth_val = Path(os.environ.get("VOCR_SYNTH_VAL", sets.SYNTH_VAL))
    result = {"schema": 1, "kind": "vocr3-a9-estimate", "estimate": True,
              "method": "vita/a9_cycles.py: arm-vita-eabi-gcc -O2 Thumb-2 machine code of runtime/vocr.c under "
                        "plugin qemu-arm (instruction mix + Vita-like caches), Cortex-A9 costs as a range; "
                        "not a console measurement",
              "vitasdk_pin": pin.read_text().strip() if pin.exists() else None, "lines": lines, "modes": modes,
              "synth_val": {"file": synth_val.name, "sha256": sha256(synth_val)},
              "code_sha256": code_identity(("vita/a9_cycles.py", "vita/qemu/insn_mix.c", "runtime/vocr.c",
                                            "runtime/vocr.h", "runtime/tools/vocr_cli.c")),
              "environment": environment(), "arms": {}}
    for arm in arms:
        d = out / "a9" / arm.name
        cmd = [sys.executable, str(script), "measure", "--out", str(d), "--blob", str(arm.path), "--lines", str(lines),
               "--modes", modes, "--vitasdk", str(vitasdk)]
        for flag, value in (("--arm-cc", arm_cc), ("--qemu", qemu), ("--eval-data", data)):
            if value:
                cmd += [flag, str(value)]
        runner(cmd, check=True, stdin=subprocess.DEVNULL)
        with (d / "report.txt").open("w") as txt:
            runner([sys.executable, str(script), "report", str(d), "--json", str(d / "report.json")], check=True,
                   stdin=subprocess.DEVNULL, stdout=txt)
        rep = json.loads((d / "report.json").read_text())
        meta = json.loads((d / "meta.json").read_text())
        if meta.get("blob_sha256") != arm.sha256:
            raise ValueError(f"{arm.name}: a9 measured another blob ({meta.get('blob_sha256')})")
        result["arms"][arm.name] = {
            "sha256": arm.sha256, "lines": rep["lines"], "pack_sha256": sha256(d / "lines.bin"),
            "vita_cli_sha256": sha256(d / "vocr_cli_vitacc"), "guest": meta.get("guest"),
            "modes": {m: {"frames": meta["modes"][m]["frames"],
                          "cycles_per_frame": {str(k): [round(float(x)) for x in v]
                                               for k, v in r["cycles_per_frame"].items()},
                          "ms_640px_line": {str(k): [round(float(x), 1) for x in v]
                                            for k, v in r["ms_640px"].items()},
                          "insns_per_frame": round(r["insns_per_frame"])}
                      for m, r in rep["modes"].items()}}
    write_new(out / "a9.json", result)
    return result


# ---------------------------------------------------------------- overlay host cross-check

OVERLAY_PINNED_RE = re.compile(r"set\(VJO_VOCR_COMMIT\s+([0-9a-f]{40})\)")
OVERLAY_TABLE_END = re.compile(r"(\n#endif\n)(\};\n#define N_MODELS)")
OVERLAY_ENTRY = re.compile(r'\{("[^"]+"|[A-Z0-9_]+),([A-Za-z0-9_]+),\{((?:0x[0-9a-fA-F]{2},?){32})\},(\d+),([A-Z_]+)\}')


def overlay_table(vocr_ocr_c, sizes_h=""):
    """{(file name, sha256): (arena, finder macro)} of the overlay's own model table (core/vocr_ocr.c;
    name macros resolved from include/vjo_vocr_sizes.h)."""
    macros = dict(re.findall(r'#define\s+(\w+)\s+"([^"]+)"', sizes_h + "\n" + vocr_ocr_c))
    out = {}
    for name, _size, hexes, arena, mode in OVERLAY_ENTRY.findall(re.sub(r"\s+", "", vocr_ocr_c)):
        name = name.strip('"') if name.startswith('"') else macros.get(name, name)
        sha = "".join(h[2:].lower() for h in hexes.rstrip(",").split(","))
        out[(name, sha)] = (int(arena), mode)
    return out


def overlay_table_accepts(vocr_ocr_c, name, sha, sizes_h=""):
    """True if the overlay's own model table holds this file name with exactly these bytes' hash."""
    return (name, sha) in overlay_table(vocr_ocr_c, sizes_h)


def overlay_table_patch(vocr_ocr_c, entries):
    """vocr_ocr.c with extra {name, bytes, sha256, W8I arena, finder} rows appended to its model table.
    Used only in a scratch export of the overlay, never in the clone; the patched source is hashed."""
    rows = []
    for e in entries:
        sha = ", ".join(f"0x{e['sha256'][i:i + 2]}" for i in range(0, 64, 2))
        mode = {"grad": "VOCR_LINES_GRAD", "textbox": "VOCR_LINES_TEXTBOX"}.get(e["finder"])
        if mode is None:
            raise ValueError(f"{e['name']}: the overlay's pinned runtime has no {e['finder']} finder")
        if not re.fullmatch(r"[A-Za-z0-9_.+-]+\.vocr", e["name"]):
            raise ValueError(f"{e['name']}: overlay model names are plain *.vocr file names")
        rows.append(f'    /* vocr3_report.py local cross-check entry (never shipped) */\n'
                    f'    {{"{e["name"]}", {int(e["bytes"])}u,\n     {{{sha}}},\n'
                    f'     {int(e["arena_w8i"])}, {mode}}},\n')
    patched, n = OVERLAY_TABLE_END.subn(lambda m: m.group(1) + "".join(rows) + m.group(2), vocr_ocr_c, count=1)
    if n != 1:
        raise ValueError("overlay core/vocr_ocr.c model table layout not recognised; refusing to patch")
    return patched


def overlay(spec, blobs_dir, clone, data, out, build_dir=None, cmake="cmake", ninja=None, extend_table=False,
            python=None):
    out = Path(out)
    if (out / "overlay.json").exists():
        raise FileExistsError(f"{out}/overlay.json exists")
    clone = Path(clone).resolve()
    build = Path(build_dir or out / "overlay-build").resolve()
    arms = [Arm(a, blobs_dir) for a in spec["arms"]]
    head = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(clone), "status", "--porcelain"], check=True, capture_output=True,
                           text=True).stdout.strip()
    if dirty:
        raise ValueError(f"{clone} has local changes; the cross-check builds a pristine export of HEAD")
    src = build / f"overlay-{head[:12]}"
    if src.exists():
        shutil.rmtree(src)
    src.mkdir(parents=True)
    archive = subprocess.run(["git", "-C", str(clone), "archive", "--format=tar", head], check=True,
                             capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(src)], input=archive, check=True)
    cmake_txt = (src / "cmake" / "VocrOCR.cmake").read_text()
    pinned = OVERLAY_PINNED_RE.search(cmake_txt).group(1)
    runtime_src = build / f"vita-vn-ocr-{pinned[:12]}"
    if runtime_src.exists():
        shutil.rmtree(runtime_src)
    runtime_src.mkdir(parents=True)
    tar = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", pinned, "runtime"], check=True,
                         capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(runtime_src)], input=tar, check=True)
    table_c = (src / "core" / "vocr_ocr.c").read_text()
    sizes_h = (src / "include" / "vjo_vocr_sizes.h").read_text() if (src / "include" / "vjo_vocr_sizes.h").exists() \
        else ""
    accepted = {a.name: overlay_table_accepts(table_c, a.file_name, a.sha256, sizes_h) for a in arms}
    patch = None
    if not all(accepted.values()):
        missing = [n for n, ok in accepted.items() if not ok]
        if not extend_table:
            raise ValueError(f"the overlay's model table does not accept {missing}; pass --extend-table to add them "
                             "to a local scratch build (recorded in the result)")
        native = build_native(out / "native-overlay")
        lib = load_library(native)
        entries = [dict(name=a.file_name, sha256=a.sha256, bytes=len(a.data), finder=a.finder,
                        arena_w8i=resources(a, lib)["arena_w8i"]) for a in arms if not accepted[a.name]]
        patched = overlay_table_patch(table_c, entries)
        (src / "core" / "vocr_ocr.c").write_text(patched)
        patch = {"entries": entries, "patched_vocr_ocr_c_sha256": hashlib.sha256(patched.encode()).hexdigest(),
                 "original_vocr_ocr_c_sha256": hashlib.sha256(table_c.encode()).hexdigest()}
    host = build / "host"
    gen = ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"] if ninja else []
    subprocess.run([cmake, "-S", str(src), "-B", str(host), *gen, "-DCMAKE_BUILD_TYPE=Release", "-DVJO_WITH_VOCR=ON",
                    "-DVJO_WITH_NCNN=OFF", f"-DVJO_VOCR_SOURCE_DIR={runtime_src}"], check=True,
                   stdin=subprocess.DEVNULL)
    subprocess.run([cmake, "--build", str(host), "--target", "vjo-ocr", "-j", "4"], check=True,
                   stdin=subprocess.DEVNULL)
    vjo = host / "vjo-ocr"
    models = build / "models"
    models.mkdir(exist_ok=True)
    runs = []
    for a in arms:
        dest = models / a.file_name
        if dest.exists() and sha256(dest) != a.sha256:
            raise ValueError(f"two arms share the overlay model name {a.file_name}")
        shutil.copyfile(a.path, dest)
        runs += ["--run", f"vocr={models}:{a.file_name}"]
    raw = out / "private" / "overlay_benchmark.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([python or sys.executable, str(src / "tools" / "benchmark_local_ocr.py"), "--vjo-ocr", str(vjo),
                    "--vocr-repo", str(ROOT), "--eval-data", str(data), "--set", ",".join(spec["sets"]), *runs,
                    "--out", str(raw)], check=True, stdin=subprocess.DEVNULL)
    bench = json.loads(raw.read_text(encoding="utf-8"))
    result = {"schema": 1, "kind": "vocr3-overlay-cross-check", "overlay_commit": head,
              "overlay_runtime_pin": pinned, "benchmark_sha256": sha256(src / "tools" / "benchmark_local_ocr.py"),
              "vocr_ocr_c_sha256": sha256(src / "core" / "vocr_ocr.c"), "table_patch": patch,
              "vjo_ocr_sha256": sha256(vjo), "environment": environment(),
              "host_only": ["rss_max", "rss_median", "ms_median"], "arms": {}}
    for a in arms:
        label = f"vocr:{a.file_name}"
        result["arms"][a.name] = {"model": a.file_name, "sha256": a.sha256, "table": "released" if accepted[a.name]
                                  else "extended-locally", "sets": {}}
        for s, by_label in bench["sets"].items():
            r = by_label[label]
            result["arms"][a.name]["sets"][s] = {k: r[k] for k in (
                "regions", "refused", "failed", "concat_errors", "concat_chars", "lines_read", "workspace_peak_max",
                "workspace_peak_median", "rss_max", "rss_median", "ms_median")}
    assert_public(result)
    write_new(out / "overlay.json", result)
    return result


# ---------------------------------------------------------------- one-shot private-set runs (adapters)

def stable_environment():
    """The parts of environment() that must not change between preparation and execution."""
    e = environment()
    return {k: e[k] for k in ("python", "python_executable_sha256", "system", "machine", "libc", "packages")}


def pinned_arms(spec, blobs_dir):
    """Arms for a private protected run: every arm must pin its sha256 in the spec."""
    missing = [a["name"] for a in spec["arms"] if not a.get("sha256")]
    if missing:
        raise ValueError(f"a protected run needs every arm pinned by sha256 in the spec: {missing}")
    return [Arm(a, blobs_dir) for a in spec["arms"]]


def expected_authorization(prepared_path, kind):
    """What the coordinating reviewer writes after checking a preparation; the CLIs never write it."""
    prepared = json.loads(Path(prepared_path).read_text(encoding="utf-8"))
    return {"schema": 1, "authorization": f"root-reviewed-one-shot-{kind}", "prepared_sha256": sha256(prepared_path),
            "protocol_sha256": prepared["protocol_sha256"], "input_fingerprint": prepared["input_fingerprint"],
            "arms": {n: a["sha256"] for n, a in sorted(prepared["arms"].items())}}


def run_once(prepared_path, authorization_path, kind, verify, execute):
    """Authorization before any verification or pixel access, one exclusive attempt, no retry.
    verify(images) -> state; execute(state, record) -> public result (written as result.json)."""
    import fcntl
    prepared_path = Path(prepared_path).resolve()
    directory = prepared_path.parent
    approved = sha256(prepared_path)
    if json.loads(Path(authorization_path).read_text(encoding="utf-8")) != expected_authorization(prepared_path, kind):
        raise ValueError("an exact root-reviewed authorization of this preparation is required before execution")
    with (directory / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "attempt.json").exists():
            raise FileExistsError("this preparation was already attempted; no retries")
        verify(False)
        write_new(directory / "attempt.json", {"schema": 1, "kind": kind, "prepared_sha256": approved,
                                               "authorization_sha256": sha256(authorization_path),
                                               "started": environment()["date"]})
        log = directory / "private-predictions.jsonl"
        try:
            state = verify(True)
            with log.open("x", encoding="utf-8") as stream:
                def record(row):
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    stream.flush()
                result = execute(state, record)
            verify(False)
            if sha256(prepared_path) != approved:
                raise ValueError("the authorized preparation changed during execution")
            result.update(prepared_sha256=approved, authorization_sha256=sha256(authorization_path),
                          raw_predictions_sha256=sha256(log), one_attempt=True, retry=False)
            assert_public(result)
            write_new(directory / "result.json", result)
            return result
        except BaseException as error:
            write_new(directory / "failure.json", {"schema": 1, "kind": kind, "status": "failed",
                                                   "error_type": type(error).__name__, "prepared_sha256": approved,
                                                   "retry_allowed": False,
                                                   "partial_predictions_retained": log.exists()})
            raise


# ---------------------------------------------------------------- markdown

def pct(e, n):
    return "-" if not n else f"{100 * e / n:.2f}%"


def ci(entry):
    if not entry or entry.get("ci95") is None:
        return "-"
    lo, hi = entry["ci95"]
    return f"[{lo:+.2f}, {hi:+.2f}]"


def render_markdown(parts):
    md = []
    sh = parts.get("shared")
    if sh:
        spec = sh["spec"]
        md.append("# VOCR3 locked report\n")
        md.append(f"Spec sha256 `{sh['spec_sha256']}`, git `{sh['environment']['git']['head']}`"
                  f"{' (dirty)' if sh['environment']['git']['dirty'] else ''}, C runtime library "
                  f"`{sh['native']['library']['sha256'][:16]}…` ({sh['native']['compiler']['version']}). "
                  "Every number is a host evaluation through the C W8I runtime; speeds are estimates.\n")
        md.append("## Arms\n")
        md.append("| arm | blob | sha256 | finder | blob bytes | arena W8I | arena W8F | classes | MACs/frame |")
        md.append("|---|---|---|---|---:|---:|---:|---:|---:|")
        for name, a in sh["arms"].items():
            r = sh["resources"][name]
            md.append(f"| {name} | {a['file']} | `{a['sha256'][:12]}…` | {a['finder']} | {r['blob_bytes']:,} | "
                      f"{r['arena_w8i']:,} | {r['arena_w8f']:,} | {r['classes']:,} | {r['macs_per_frame']:,} |")
        set_names = list(sh["datasets"])
        for pol in spec["polarities"]:
            tag = " (primary)" if pol == spec["primary_polarity"] else ""
            md.append(f"\n## Fixed lines, {pol} polarity{tag}\n")
            hdr = "| arm | " + " | ".join(f"{s} CER | {s} exact | {s} hiragana" for s in set_names + ["pooled"]) + " |"
            md.append(hdr)
            md.append("|---|" + "---|---|---|" * (len(set_names) + 1))
            for name, entry in sh["fixed_line"][pol].items():
                cells = []
                for s in set_names + ["pooled"]:
                    t = entry["summary"].get(s, {}).get("all")
                    if not t:
                        cells += ["-", "-", "-"]
                        continue
                    cells += [f"{pct(t['errors'], t['chars'])} ({t['errors']}/{t['chars']})",
                              f"{pct(t['exact'], t['lines'])} ({t['exact']}/{t['lines']})",
                              f"{pct(t['hira_errors'], t['hira_chars'])} ({t['hira_errors']}/{t['hira_chars']})"]
                md.append(f"| {name} | " + " | ".join(cells) + " |")
            for s in set_names:
                arms = [n for n, e in sh["fixed_line"][pol].items() if s in e["summary"]]
                tags = style_tags([x for n in arms for x in sh["fixed_line"][pol][n]["lines"][s]])
                if not tags:
                    continue
                md.append(f"\n{s}, {pol} polarity, per style (CER, errors/chars):\n")
                md.append("| style | lines | " + " | ".join(arms) + " |")
                md.append("|---|---:|" + "---|" * len(arms))
                for t in tags:
                    first = sh["fixed_line"][pol][arms[0]]["summary"][s]["styles"][t]
                    cells = []
                    for n in arms:
                        x = sh["fixed_line"][pol][n]["summary"][s]["styles"][t]
                        cells.append(f"{pct(x['errors'], x['chars'])} ({x['errors']}/{x['chars']})")
                    md.append(f"| {t} | {first['lines']} | " + " | ".join(cells) + " |")
        if sh["pairs"]:
            md.append("\n## Paired differences (candidate − baseline, points of CER; 95% bootstrap intervals)\n")
            bs = spec["bootstrap"]
            md.append(f"{bs['reps']:,} resamples, seed {bs['seed']}; VN-cluster intervals need at least "
                      f"{bs['min_clusters']} VNs in the stratum.\n")
            for key, by_pol in sh["pairs"].items():
                for pol, strata in by_pol.items():
                    md.append(f"\n{key}, {pol} polarity:\n")
                    md.append("| set | stratum | lines | full: Δ | lines CI | VN CI | hiragana: Δ | lines CI | VN CI |")
                    md.append("|---|---|---:|---:|---|---|---:|---|---|")
                    for s, rows in strata.items():
                        for label, e in rows.items():
                            f, h = e["full"], e["hira"]
                            fd = "-" if f["lines"]["delta"] is None else f"{f['lines']['delta']:+.2f}"
                            hd = "-" if h["lines"]["delta"] is None else f"{h['lines']['delta']:+.2f}"
                            md.append(f"| {s} | {label} | {f['lines']['units']} | {fd} | {ci(f['lines'])} | "
                                      f"{ci(f['vn_clusters'])} | {hd} | {ci(h['lines'])} | {ci(h['vn_clusters'])} |")
        dl = sh.get("dialogue")
        if dl:
            md.append("\n## Whole dialogue (manifest regions, each arm's C finder; errors/characters)\n")
            for s, acc in dl["accounting"]["sets"].items():
                ex = acc["excluded"]
                md.append(f"- {s}: {acc['regions']} regions, {acc['lines']} lines, {acc['chars']:,} characters; "
                          f"excluded whole: {ex['regions']} regions with an uncertain line "
                          f"({ex['certain_lines']} certain lines, {ex['chars']} characters).")
            md.append("\n| arm | variant | platform | all full | all hiragana | body full | body hiragana "
                      "| detections |")
            md.append("|---|---|---|---|---|---|---|---:|")
            for name, by_var in dl["totals"].items():
                for variant, by_plat in by_var.items():
                    for plat, t in by_plat.items():
                        cells = [f"{t[s][x]['errors']}/{t[s][x]['chars']}" for s in SCOPES for x in SCRIPTS]
                        md.append(f"| {name} | {variant} | {plat} | " + " | ".join(cells) + f" | {t['detections']} |")
        if sh["gates"]:
            md.append("\n## Gates (candidate against the model it would replace)\n")
            md.append("| gate | fixed-line gate | whole-dialogue guard | passed |")
            md.append("|---|---|---|---|")
            for key, g in sh["gates"].items():
                fl = g["fixed_line"]
                fl_cells = "; ".join(f"{s} {r['baseline_errors']}→{r['candidate_errors']} "
                                     f"(hiragana {r['baseline_hira_errors']}→{r['candidate_hira_errors']})"
                                     for s, r in fl["sets"].items())
                ph = fl["pooled_hiragana"]
                fl_txt = (f"{'pass' if fl['passed'] else 'FAIL'}: {fl_cells}; pooled hiragana "
                          f"{ph['baseline_errors']}→{ph['candidate_errors']}")
                d = g.get("dialogue")
                d_txt = "not run" if d is None else (
                    f"{'pass' if d['passed'] else 'FAIL'}: {len(d['failed_cells'])} of {d['title_checks']} title "
                    f"checks regress; gain titles " + ", ".join(f"{v} {len(t)}" for v, t in d["gain_titles"].items()))
                md.append(f"| {key} | {fl_txt} | {d_txt} | {g['passed']} |")
    a9r = parts.get("a9")
    if a9r:
        md.append("\n## Estimated Cortex-A9 time (host estimate, not a console measurement)\n")
        md.append(f"{a9r['method']}; {a9r['lines']} lines.\n")
        md.append("| arm | mode | cycles/frame | 640-px line at 444 MHz | at 333 MHz |")
        md.append("|---|---|---|---|---|")
        for name, a in a9r["arms"].items():
            for m, r in a["modes"].items():
                c, t4, t3 = r["cycles_per_frame"]["444"], r["ms_640px_line"]["444"], r["ms_640px_line"]["333"]
                md.append(f"| {name} | {m} | {c[0]:,}-{c[1]:,} | {t4[0]:.0f}-{t4[1]:.0f} ms | "
                          f"{t3[0]:.0f}-{t3[1]:.0f} ms |")
    ov = parts.get("overlay")
    if ov:
        md.append(f"\n## Overlay host cross-check (vita-jp-overlay `{ov['overlay_commit'][:12]}`, runtime pin "
                  f"`{ov['overlay_runtime_pin'][:12]}`)\n")
        md.append("| arm | table | set | regions | refused | failed | concat CER | lines read | workspace peak max |")
        md.append("|---|---|---|---:|---:|---:|---|---:|---:|")
        for name, a in ov["arms"].items():
            for s, r in a["sets"].items():
                md.append(f"| {name} | {a['table']} | {s} | {r['regions']} | {r['refused']} | {r['failed']} | "
                          f"{pct(r['concat_errors'], r['concat_chars'])} ({r['concat_errors']}/{r['concat_chars']}) | "
                          f"{r['lines_read']} | {r['workspace_peak_max']:,} |")
        cross = parts.get("overlay_vs_dialogue")
        if cross:
            md.append("\nOverlay concat errors against this harness's whole-dialogue base all-text errors: " +
                      "; ".join(f"{k} {v['overlay']} vs {v['harness']} ({'equal' if v['equal'] else 'DIFFER'})"
                                for k, v in cross.items()) + ".")
    for key, title in (("fresh", "Fresh native textbox test"), ("clannad", "CLANNAD long run (Google proxy)")):
        r = parts.get(key)
        if r:
            md.append(f"\n## {title}\n")
            md.append(r.get("markdown") or "(see JSON)")
    return "\n".join(md) + "\n"


def render(out, fresh=None, clannad=None):
    out = Path(out)
    parts = {}
    for name in ("shared", "a9", "overlay"):
        p = out / f"{name}.json"
        if p.exists():
            parts[name] = json.loads(p.read_text(encoding="utf-8"))
    for name, path in (("fresh", fresh), ("clannad", clannad)):
        if path:
            parts[name] = json.loads(Path(path).read_text(encoding="utf-8"))
    if "shared" in parts and "overlay" in parts and parts["shared"].get("dialogue"):
        cross = {}
        tot = parts["shared"]["dialogue"]["totals"]
        for name, a in parts["overlay"]["arms"].items():
            for s, r in a["sets"].items():
                plat = sets.SETS.get(s)
                h = tot.get(name, {}).get("base", {}).get(plat)
                if h is None:
                    continue
                cross[f"{name}/{s}"] = {"overlay": r["concat_errors"], "harness": h["all"]["full"]["errors"],
                                        "overlay_chars": r["concat_chars"], "harness_chars": h["all"]["full"]["chars"],
                                        "equal": (r["concat_errors"], r["concat_chars"]) ==
                                                 (h["all"]["full"]["errors"], h["all"]["full"]["chars"])}
        parts["overlay_vs_dialogue"] = cross
    combined = {"schema": 1, "kind": "vocr3-locked-report", "parts": sorted(parts),
                "part_sha256": {k: digest(v) for k, v in parts.items()}, **parts}
    assert_public(combined)
    write_new(out / "report.json", combined)
    write_new(out / "report_summary.json", compact(combined))
    write_new(out / "report.md", render_markdown(parts))
    return combined


def compact(value):
    """The report without per-line / per-region records (small enough to commit); counts stay exact."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            records = isinstance(v, list) or (isinstance(v, dict) and v and all(isinstance(x, list)
                                                                                  for x in v.values()))
            if k in ("lines", "units", "frames", "pages") and records:
                continue
            out[k] = compact(v)
        return out
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


# ---------------------------------------------------------------- CLI

def spec_from_args(args):
    return build_spec(args.spec, arms=args.arm, pairs=args.pair, gates=args.gate,
                      stored=getattr(args, "stored", []), local_sets=getattr(args, "local_set", []),
                      sets_=args.set.split(",") if getattr(args, "set", None) else None,
                      polarities=args.polarity.split(",") if getattr(args, "polarity", None) else None,
                      bootstrap={"reps": getattr(args, "reps", None), "seed": getattr(args, "seed", None),
                                 "min_clusters": getattr(args, "min_clusters", None)},
                      dialogue=False if getattr(args, "no_dialogue", False) else None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def arms_args(p):
        p.add_argument("--spec", type=Path, help="JSON spec (arms, pairs, gates, settings); docs/vocr3-harness.md")
        p.add_argument("--arm", action="append", default=[], help="NAME=FILE[@grad|textbox][#sha256]")
        p.add_argument("--pair", action="append", default=[], help="CANDIDATE:BASELINE")
        p.add_argument("--gate", action="append", default=[], help="CANDIDATE:BASELINE")
        p.add_argument("--blobs", type=Path, help="directory the arms' FILEs are relative to")
        p.add_argument("--out", type=Path, required=True)

    s = sub.add_parser("shared", help="fixed lines + whole dialogue on vita-test/psp-test")
    arms_args(s)
    s.add_argument("--stored", action="append", default=[], help="NAME:POLARITY:SET=RESULT.json (reference only)")
    s.add_argument("--local-set", action="append", default=[], help="NAME=MANIFEST (private hand-checked captures)")
    s.add_argument("--set", help="comma list of shared sets (default vita-test,psp-test)")
    s.add_argument("--polarity", help="comma list (default console,native; the spec's primary_polarity, default "
                                      "console, decides the gates)")
    s.add_argument("--reps", type=int)
    s.add_argument("--seed", type=int)
    s.add_argument("--min-clusters", type=int)
    s.add_argument("--no-dialogue", action="store_true")
    s.add_argument("--data", help="eval screenshots (default $VOCR_EVAL_DATA or data/eval)")
    s.add_argument("--compiler", default="gcc")
    a = sub.add_parser("a9", help="estimated Cortex-A9 time per blob (vita/a9_cycles.py)")
    arms_args(a)
    a.add_argument("--vitasdk")
    a.add_argument("--arm-cc")
    a.add_argument("--qemu")
    a.add_argument("--data")
    a.add_argument("--lines", type=int, default=40)
    a.add_argument("--modes", default="w8i,w8f")
    o = sub.add_parser("overlay", help="vita-jp-overlay host vjo-ocr through its benchmark_local_ocr.py")
    arms_args(o)
    o.add_argument("--overlay", type=Path, required=True, help="read-only vita-jp-overlay clone")
    o.add_argument("--data", required=True)
    o.add_argument("--build-dir", type=Path)
    o.add_argument("--cmake", default="cmake")
    o.add_argument("--ninja")
    o.add_argument("--extend-table", action="store_true",
                   help="add arms the overlay does not pin to a local scratch build's model table")
    r = sub.add_parser("render", help="report.json + report.md from the parts in --out")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--fresh", type=Path)
    r.add_argument("--clannad", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "render":
        render(args.out, args.fresh, args.clannad)
        print(f"wrote {args.out}/report.json and report.md")
        return
    spec = spec_from_args(args)
    if args.cmd == "shared":
        rep = shared(spec, args.blobs, data_dir(args.data), args.out, args.compiler)
        print((args.out / "shared.md").read_text(encoding="utf-8"))
        print(f"wrote {args.out}/shared.json; gates: " +
              ", ".join(f"{k} {g['passed']}" for k, g in rep["gates"].items()))
    elif args.cmd == "a9":
        a9(spec, args.blobs, args.out, args.vitasdk, args.arm_cc, args.qemu, args.data, args.lines, args.modes)
        print(f"wrote {args.out}/a9.json")
    else:
        overlay(spec, args.blobs, args.overlay, data_dir(args.data), args.out, args.build_dir, args.cmake,
                args.ninja, args.extend_table)
        print(f"wrote {args.out}/overlay.json")


if __name__ == "__main__":
    main()
