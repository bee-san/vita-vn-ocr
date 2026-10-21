#!/usr/bin/env python3
"""textspot3: the scoring wrapper of issue #108 (docs/textspot3.md, eval/textspot3_protocol.json). eval/textspot.py
(the harness), eval/textspot_metrics.py, eval/textspot2.py (textspot2's wrapper: its reading cache helpers, its
VTSP version-2 dispatch and C driver, its reference and baseline loaders) and eval/textspot3_rule.py (the rule) are
imported and used unchanged; nothing here re-implements a metric or the rule.

  textspot3.py cbuild --out DIR                      the C drivers from this checkout: textspot2's (textspot's seven
                                                     and runtime/vocr_textspot2_test) plus runtime/vocr_textspot3_test
                                                     once its sources exist
  textspot3.py seed-cache [--cache DIR]              copy textspot2's development reading cache once (never written)
  textspot3.py run --runs DIR [--names RUN ...] --out DIR --cbin DIR [--models N]
      every finished run (done.json; listed in finished.txt when the file exists) as the arms T3_<run>_det and
      T3_<run>_detref: eval/textspot.py's det / detref candidate specs through its evaluate path (the twins, C parity
      per frame, the reading by VOCR5-N for horizontal boxes and VOCR5-V for vertical ones, frame_unit, summarize),
      on the development set; an arm already scored for its blob is skipped. A VTSP version-2 blob (family T,
      half_above 480) goes through textspot2's path exactly (eval/textspot2.py v2_dispatch: synth/textspot2_twin.py
      and runtime/vocr_textspot2_test); a version-3 blob (family M) through synth/textspot3_twin.py and
      runtime/vocr_textspot3_test (dispatch).
  textspot3.py read-pick --names RUN [--dev JSON] [--models S,L]
      the pick's VOCR5-S and VOCR5-L development readings (report only): its stored boxes (never re-detected) read
      by the report recognizers, units_VOCR5-S.json and units_VOCR5-L.json beside its VOCR5-N units
  textspot3.py headline --units DIR [--a9 JSON ...] [--out MD]
      every seed's F1, CER all, dialogue, UI recall and est. ms, the representatives (median seed), stage 1's count
      and the stage-2 inputs read off them, as markdown for the conductor
  textspot3.py select --units DIR --a9 eval/textspot3_a9_<date>.json [--a9 ...] --out eval/textspot3_dev_<date>.json
      (1) every finished run's eligibility inputs through eval/textspot.py's eligibility (VTSP bytes, arena at
      960 x 544, est. ms with the refine term, C parity; the A9 rows of the conductor-pinned tables, each checked
      against the protocol's pin, a seed without a row refused; every finished run must be scored); (2) per arm
      (recipe, shape) and decode, every seed's summary and the representative by textspot3_rule.median_seed;
      (3) textspot3_rule.apply_rule3 once over every arm's representatives against ref (its units and boxes checked
      against the protocol's sha256), with textspot's eight baselines (their units sha256-checked as textspot's
      select does) for the report-only bar; (4) the report: every comparison, recall per class
      (textspot3_rule.recall_by_class) and whether UI recall rises with VTSP bytes, the Pareto front of the eligible
      representatives, what each arm bought, per part, the text-free frames with a box, the dialogue pick, the
      stage-2 inputs. Compact JSON under 2 MiB (counts, rates and hashes only) and an indented copy under $T3/eval.
  textspot3.py tables --dev JSON [--units DIR] [--out MD]
      markdown for docs/textspot3.md: the selection, the Pareto front, per-class recall, the bought table, every
      seed (and the pick's VOCR5-S / VOCR5-L readings when their units exist)

Paths: T3 = $HERD/shared/artifacts/textspot3 (units under T3/eval/units, the reading cache T3/eval/cache, the C
drivers T3/eval/cbuild, the frames' pixel hashes T3/eval/frames.json, the runs T3/runs). textspot's and textspot2's
artifacts and private files ($T, $P, $T2, $P2) are read, never written.

Runs and shapes: a run is <recipe>_<shape>_seed<n> (eval/textspot3_protocol.json "paths"), its shape one the
protocol registers ("shapes", added by amendments), whose "family" is T (VTSP version 2: textspot2's d2s format at
half_above 480; textspot2_export's layers and costs) or M (VTSP version 3: synth/textspot3_twin.py,
train/textspot3_model.py). The M size points S, M and L are the registry's own "point" field when it has one, else
the three M shapes in VTSP-byte order (the record says which).
"""
import argparse
import contextlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "runtime", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot as TS  # noqa: E402
import textspot2 as T2E  # noqa: E402
import textspot3_rule as R  # noqa: E402
import textspot_metrics as M  # noqa: E402

HERD = TS.HERD
T, P = TS.T, TS.P                  # textspot's artifacts and private files: read only
T2, P2 = T2E.T2, T2E.P2            # textspot2's: read only
T3 = TS.ART / "textspot3"
P3 = HERD / "shared" / "private" / "textspot3"
EVAL3 = T3 / "eval"
DEFAULT = {"units": EVAL3 / "units", "cache": EVAL3 / "cache", "cbuild": EVAL3 / "cbuild", "rbuild": EVAL3 / "rbuild",
           "frames": EVAL3 / "frames.json", "runs": T3 / "runs"}
PROTOCOL3 = ROOT / "eval" / "textspot3_protocol.json"
BASELINE = ROOT / "eval" / "textspot_baseline_20261019.json"
T2_CACHE = T2 / "eval" / "cache"
KINDS = ("det", "detref")
DECODE = {"det": "det", "detref": "det+refine"}
DECODES = ("det", "det+refine")
SEEDS = (1, 2, 3)
MHZ = 333 / 444
MAX_RECORD = 2 * 1024 * 1024       # the committed selection record stays under 2 MiB
FAMILY_VERSION = {"T": 2, "M": 3}
HALF_ABOVE_T = 480                 # family T: a frame wider than 480 px is halved
M_HALF = (480, 272)                # family M: halved when wider than 480 px or taller than 272 px
STAGE1_RECIPES = ("base", "size", "clean", "long")
POINTS = ("S", "M", "L")
POINT_KEYS = ("point", "size_point", "tier")

RUN_RE = re.compile(r"^(?P<recipe>[a-z][a-z0-9]*(?:\+[a-z0-9]+)*)_(?P<shape>[A-Za-z0-9][A-Za-z0-9_.-]*?)"
                    r"_seed(?P<seed>[0-9]+)$")


def protocol3(path=None):
    return json.loads(Path(path or PROTOCOL3).read_text(encoding="utf-8"))


def code_identity():
    """The harness's code identity, textspot2's wrapper files and this wrapper's own (those that exist)."""
    out = T2E.code_identity()
    out.update(TS.code_identity(("eval/textspot3.py", "eval/textspot3_protected.py", "eval/textspot3_rule.py",
                                 "eval/textspot3_protocol.json", "synth/textspot3_twin.py", "train/textspot3_model.py",
                                 "train/textspot3_shapes.py", "runtime/vocr_textspot3.c", "runtime/vocr_textspot3.h",
                                 "runtime/vocr_textspot3_test.c", "train/textspot2_export.py")))
    return out


def is_under(path, base):
    try:
        Path(path).resolve().relative_to(Path(base).resolve())
        return True
    except ValueError:
        return False


def check_writable(*paths):
    """Nothing is ever written under textspot's or textspot2's artifacts or private files."""
    for p in paths:
        if p and any(is_under(p, b) for b in (T, P, T2, P2)):
            raise SystemExit(f"{p}: textspot's and textspot2's files are read only (write under $T3 / $P3)")


# ----------------------------------------------------------------------------- the registered shapes

def family_of(shape):
    """A registered shape's family: its "family" field, else T for a d2s trunk (head kroa_x2), M for the
    multi-scale head (ms_x2)."""
    f = shape.get("family")
    if f in FAMILY_VERSION:
        return f
    return {"kroa_x2": "T", "ms_x2": "M"}.get(shape.get("head"))


def registry(proto3=None):
    """{name: shape dict with "name" and "family"} of every shape eval/textspot3_protocol.json registers: its
    "shapes" (which the conductor's amendments add to) and any amendment's own "shapes" object, later ones winning,
    as train/textspot3_shapes reads them; a shape of no known family is an error. A T shape's half_above defaults to
    480."""
    doc = proto3 if proto3 is not None else protocol3()
    out = {}
    blocks = [doc.get("shapes")] + [a.get("shapes") for a in doc.get("amendments", []) if isinstance(a, dict)]
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for name, s in block.items():
            if not isinstance(s, dict):
                continue
            fam = family_of(s)
            if fam is None:
                raise SystemExit(f"{name}: a registered shape of no known family (T or M)")
            out[name] = dict(s, name=name, family=fam)
            if fam == "T":
                out[name].setdefault("half_above", HALF_ABOVE_T)
    return out


def parse_run(name, reg=None):
    """'<recipe>_<shape>_seed<n>' -> {"run", "recipe", "shape", "seed", "arm_id", "family"}; the shape is one the
    protocol registers."""
    reg = registry() if reg is None else reg
    m = RUN_RE.match(name)
    if m is None or m.group("shape") not in reg:
        raise ValueError(f"{name!r} is not <recipe>_<shape>_seed<n> of a registered textspot3 shape")
    recipe, shape, seed = m.group("recipe"), m.group("shape"), int(m.group("seed"))
    return {"run": name, "recipe": recipe, "shape": shape, "seed": seed, "arm_id": f"{recipe}_{shape}",
            "family": reg[shape]["family"]}


def m_model():
    """train/textspot3_model.py (textspot3-runtime's M networks), or SystemExit while it does not exist."""
    if not (ROOT / "train" / "textspot3_model.py").exists():
        raise SystemExit("a family-M shape needs train/textspot3_model.py (textspot3-runtime), which this checkout "
                         "does not have")
    import textspot3_model
    return textspot3_model


def m_input(width=960, height=544):
    """Family M's network input for a width x height frame: textspot's exact 2 x 2 halving when the frame is wider
    than 480 px or taller than 272 px."""
    if width > M_HALF[0] or height > M_HALF[1]:
        return -(-width // 2), -(-height // 2)
    return width, height


def macs_of(shape, reg, width=960, height=544):
    """(exact MACs per width x height frame at the network's input, their source) of a registered shape, as
    train/textspot3_shapes.costs computes them (the A9 tables' macs_960): family T through
    train/textspot2_export.cost on the d2s trunk at the shape's half_above (480), family M through
    train/textspot3_model.costs (its cost at version 3's input when it has no costs)."""
    s = reg[shape]
    if s["family"] == "T":
        import textspot2_export as X
        w, h = TS.input_size(s.get("half_above", HALF_ABOVE_T), width, height)
        return int(X.cost(t_shape(s), w, h)["macs"]), "train/textspot2_export.cost"
    mod = m_model()
    if hasattr(mod, "costs"):
        return int(mod.costs(s, ((width, height),))["macs"][f"{width}x{height}"]), "train/textspot3_model.costs"
    return int(mod.cost(s, *m_input(width, height))["macs"]), "train/textspot3_model.cost"


def m_vtsp_bytes(s):
    """The exact VTSP bytes of a family-M shape (train/textspot3_model.costs, else its cost)."""
    mod = m_model()
    c = mod.costs(s, (M_HALF,)) if hasattr(mod, "costs") else mod.cost(s, *M_HALF)
    return int(c["vtsp_bytes"])


def t_shape(s):
    """A family-T registry entry as train/textspot2_export's d2s shape dict."""
    return {"name": s["name"], **{k: (list(s[k]) if isinstance(s[k], list) else s[k])
                                  for k in ("stride", "c0", "c1", "c2", "chans", "dil", "block", "head")}}


def size_points(reg, nbytes=None):
    """({"S": shape, "M": shape, "L": shape}, source) of the family-M size points: the registry's own "point" field
    (S, M or L) when every M shape has one, else the three M shapes ordered by VTSP bytes (nbytes {shape: bytes},
    e.g. their runs' blobs; else train/textspot3_model.cost's "vtsp_bytes"). ({}, reason) when neither works."""
    ms = sorted(n for n, s in reg.items() if s["family"] == "M")
    if not ms:
        return {}, "no family-M shape is registered"
    lab = {n: next((reg[n][k] for k in POINT_KEYS if reg[n].get(k) in POINTS), None) for n in ms}
    if all(lab.values()) and sorted(lab.values()) == sorted(POINTS) and len(ms) == 3:
        return {p: n for n, p in lab.items()}, "the registry's point field"
    nb = dict(nbytes or {})
    if not all(n in nb for n in ms):
        try:
            for n in ms:
                nb.setdefault(n, m_vtsp_bytes(reg[n]))
        except (SystemExit, KeyError, TypeError, ValueError, AttributeError):
            pass
    if len(ms) != 3 or not all(n in nb for n in ms):
        return {}, f"{len(ms)} family-M shapes, VTSP bytes known for {sum(1 for n in ms if n in nb)}"
    return dict(zip(POINTS, sorted(ms, key=lambda n: (nb[n], n)))), "VTSP-byte order"


def stage1_runs(reg, points):
    """Stage 1 as docs/textspot3.md "Arms and stages" lists it: base on every registered T shape; base on S, M and
    L; size, clean and long on M; seeds 1-3."""
    out = [f"base_{s}_seed{k}" for s in sorted(n for n, v in reg.items() if v["family"] == "T") for k in SEEDS]
    out += [f"base_{points[p]}_seed{k}" for p in POINTS if p in points for k in SEEDS]
    if "M" in points:
        out += [f"{r}_{points['M']}_seed{k}" for r in STAGE1_RECIPES[1:] for k in SEEDS]
    return out


# ----------------------------------------------------------------------------- VTSP versions

TWIN3 = ROOT / "synth" / "textspot3_twin.py"
DRIVER2, DRIVER3 = T2E.DRIVER2, "vocr_textspot3_test"
DRIVER3_SOURCE_SETS = (("vocr_lines.c", "vocr_lines_rows.c", "vocr_textspot3.c", "vocr_textspot3_test.c"),
                       ("vocr_lines.c", "vocr_textspot3.c", "vocr_textspot3_test.c"),
                       ("vocr_textspot3.c", "vocr_textspot3_test.c"))
blob_version = T2E.blob_version


def twin3():
    if not TWIN3.exists():
        raise SystemExit("a VTSP version-3 (family M) blob needs synth/textspot3_twin.py (textspot3-runtime), which "
                         "this checkout does not have")
    import textspot3_twin
    return textspot3_twin


def model_of(blob):
    """The twin's model of a blob: textspot's twin (version 1), textspot2_twin (2) or textspot3_twin (3)."""
    v = blob_version(blob)
    if v == 3:
        return twin3().Model(blob)
    return T2E.model_of(blob)


def shim_module3(real):
    """A stand-in for synth/textspot_twin with every name of the real module, whose Model and spot send a version-1
    blob (and its model) to the real twin, a version-2 blob to textspot2_twin and a version-3 blob to
    textspot3_twin, each unchanged."""
    t2, t3 = T2E.twin2(), twin3()
    mod = types.ModuleType("textspot_twin")
    mod.__dict__.update(real.__dict__)

    def Model(blob):  # noqa: N802 (the twin's class name)
        v = blob_version(blob)
        return real.Model(blob) if v == 1 else t2.Model(blob) if v == 2 else t3.Model(blob)

    def spot(model, img):
        if isinstance(model, t3.Model):
            return t3.spot(model, img)
        if isinstance(model, t2.Model):
            return t2.spot(model, img)
        return real.spot(model, img)

    mod.Model, mod.spot, mod.TEXTSPOT2_SHIM, mod.TEXTSPOT3_SHIM = Model, spot, True, True
    return mod


def _driver_spot(cd, driver, greys, blob):
    """One batch of grey frames through a spotter driver of textspot's batch protocol (records "<4I" width, height,
    1, 0 + pixels; one line per frame: rc, count, block count, pick, truncated, 8 integers per box, 7 per block)."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".vtsp", delete=False) as f:
        f.write(blob)
        bpath = f.name
    try:
        recs = [struct.pack("<4I", g.shape[1], g.shape[0], 1, 0) + g.tobytes() for g in greys]
        lines = cd._batch(driver, ["--batch", bpath], recs)
    finally:
        os.unlink(bpath)
    out = []
    for line in lines:
        v = [int(t) for t in line.split()]
        if v[0]:
            raise RuntimeError(f"{driver}: rc={v[0]}")
        n, nb, pick, tr = v[1:5]
        bx = [tuple(v[5 + 8 * i:13 + 8 * i]) for i in range(n)]
        o = 5 + 8 * n
        out.append((bx, [tuple(v[o + 7 * i:o + 7 + 7 * i]) for i in range(nb)], pick, tr))
    if len(out) != len(greys):
        raise RuntimeError(f"{driver}: one result per frame expected")
    return out


class CDrivers3(T2E.CDrivers2):
    """textspot2's C drivers (eval/textspot2.py CDrivers2: version 1 to textspot's driver, version 2 to
    runtime/vocr_textspot2_test, unchanged); a version-3 blob to runtime/vocr_textspot3_test, the same batch
    protocol."""

    def spot(self, greys, blob):
        if blob_version(blob) != 3:
            return super().spot(greys, blob)
        if not self.has(DRIVER3):
            return None
        return _driver_spot(self, DRIVER3, greys, blob)


@contextlib.contextmanager
def dispatch(versions):
    """The harness's `import textspot_twin` and its C parity for the VTSP versions of a batch: only version 1,
    nothing changes (textspot's path); versions 1 and 2, eval/textspot2.py's v2_dispatch exactly (textspot2's
    path); any version 3, shim_module3 and CDrivers3."""
    versions = set(versions)
    if not versions - {1}:
        yield
        return
    if versions <= {1, 2}:
        with T2E.v2_dispatch(True):
            yield
        return
    real = sys.modules.get("textspot_twin")
    if real is None:
        import textspot_twin as real
    if getattr(real, "TEXTSPOT3_SHIM", False):
        yield
        return
    if getattr(real, "TEXTSPOT2_SHIM", False):
        raise RuntimeError("textspot2's dispatch is active: a version-3 batch cannot nest in it")
    shim, saved = shim_module3(real), TS.CDrivers
    sys.modules["textspot_twin"], TS.CDrivers = shim, CDrivers3
    try:
        yield
    finally:
        sys.modules["textspot_twin"], TS.CDrivers = real, saved


def cbuild3(out, compiler="gcc"):
    """eval/textspot2.py's cbuild2 (textspot's drivers and runtime/vocr_textspot2_test), plus
    runtime/vocr_textspot3_test (textspot3-runtime's driver for version-3 blobs) when its sources exist: the first
    of DRIVER3_SOURCE_SETS that compiles and links with textspot's flags."""
    out = Path(out)
    check_writable(out)
    rec = T2E.cbuild2(out, compiler)
    rt = ROOT / "runtime"
    env = {"PATH": os.environ.get("PATH", os.defpath), "LC_ALL": "C", "SOURCE_DATE_EPOCH": "0"}
    rec["driver3"] = None
    for srcs in DRIVER3_SOURCE_SETS:
        if not all((rt / s).exists() for s in srcs):
            continue
        p = subprocess.run([compiler, *TS.C_FLAGS, "-o", str(out / DRIVER3), *srcs], cwd=rt, env=env,
                           capture_output=True, text=True)
        if p.returncode == 0:
            rec["binaries"][DRIVER3] = TS.sha256(out / DRIVER3)
            rec["driver3"] = {"sources": [f"runtime/{s}" for s in srcs]}
            for s in srcs:
                rec["sources_sha256"][f"runtime/{s}"] = TS.sha256(rt / s)
            break
    TS.write_json(out / "build.json", rec, indent=1)
    return rec


def driver_sha(cdir, version):
    """The sha256 of the C driver a blob version's parity runs on (None without one)."""
    name = {1: "vocr_textspot_test", 2: DRIVER2, 3: DRIVER3}[version]
    p = Path(cdir) / name if cdir is not None else None
    return TS.sha256(p) if p is not None and p.exists() else None


def twin_sha(version):
    """The sha256 of the twin a blob version runs through (textspot's twin identity covers version 1)."""
    f = {2: ROOT / "synth" / "textspot2_twin.py", 3: TWIN3}.get(version)
    return TS.sha256(f) if f is not None and f.exists() else None


# ----------------------------------------------------------------------------- the cache, frames, readers

def seed_cache(dst, src=T2_CACHE, kind="textspot3-cache-seed"):
    """The reading cache dst seeded once with a byte-for-byte copy (sha256-checked) of every shard of src
    (textspot2's development cache, only read), recorded in dst/SEEDED.json; later calls change nothing."""
    dst = Path(dst)
    check_writable(dst)
    marker = dst / "SEEDED.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    files = {}
    shards = sorted(Path(src).glob("*.jsonl"))
    if not shards:
        raise SystemExit(f"{src}: no cache shard to copy")
    dst.mkdir(parents=True, exist_ok=True)
    for p in shards:
        q, h = dst / p.name, TS.sha256(p)
        if not q.exists():
            shutil.copyfile(p, q)
        if TS.sha256(q) != h:
            raise SystemExit(f"{q}: not a copy of {p}")
        files[p.name] = {"sha256": h, "bytes": p.stat().st_size}
    rec = {"kind": kind, "seeded": TS.utc_now(), "from": TS.rel(src), "files": files}
    TS.write_json(marker, rec, indent=1)
    return rec


def record_frames(path, frames, manifest):
    """The frames' pixel sha256 (the reading cache's key) in a file, as eval/textspot2.py records them; a later
    write must agree with what is recorded."""
    if not path:
        return None
    p = Path(path)
    check_writable(p)
    doc = dict(T2E.frames_doc(frames, manifest), kind="textspot3-frames")
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        if old["manifest"]["sha256"] == manifest["sha256"]:
            if {k: old["frames"].get(k) for k in doc["frames"]} != doc["frames"]:
                raise SystemExit(f"{p}: these frames' pixels differ from the recorded ones")
            if set(old["frames"]) >= set(doc["frames"]):
                return old
    TS.write_json(p, doc, indent=1)
    return doc


def check_recognizers(proto, proto3):
    """textspot's and textspot3's protocols pin the same four recognizers."""
    for m, r in proto3["reused"]["recognizers"].items():
        if proto["recognizers"][m]["sha256"] != r["sha256"]:
            raise SystemExit(f"{m}: textspot's and textspot3's protocols pin different recognizers")


def check_dev_manifest(which, manifest, proto3):
    if which == "dev" and manifest["sha256"] != proto3["reused"]["dev_manifest"]["sha256"]:
        raise SystemExit("the development manifest is not the one textspot3's protocol reuses")


def resolve_cbin(text, out, compiler="gcc"):
    if not text:
        return None
    if text == "auto":
        cbuild3(Path(out) / "cbuild", compiler)
        return Path(out) / "cbuild"
    if not (Path(text) / "build.json").exists():
        raise SystemExit(f"{text}: no build.json (textspot3.py cbuild --out {text})")
    return Path(text)


# ----------------------------------------------------------------------------- the runs

def run_settings(d):
    """The trainer's settings of a run directory that the record carries: its recipe extras (run.json "extras",
    a list or a comma-separated string; none when absent)."""
    f = Path(d) / "run.json"
    if not f.exists():
        return {"extras": [], "run_family": None}
    doc = json.loads(f.read_text(encoding="utf-8"))
    ex = doc.get("extras", (doc.get("settings") or {}).get("extras"))
    if isinstance(ex, str):
        ex = [] if ex in ("", "none") else [x for x in ex.split(",") if x]
    return {"extras": sorted(ex or []), "run_family": doc.get("family")}


def finished_runs(runs_dir, names=None, reg=None):
    """({run: info}, [skipped]) of the finished runs: run/done.json and run/final.vtsp, the blob's sha256 and bytes
    as done.json records them and as finished.txt lists them (when it exists, a run it does not list is not finished
    yet), a name <recipe>_<shape>_seed<n> of a registered shape (others are skipped), and a blob of that shape's
    family: VTSP version 2 at half_above 480 (T) or version 3 (M), named as the shape, with conf_min 1."""
    reg = registry() if reg is None else reg
    runs_dir = Path(runs_dir)
    listed = T2E.finished_list(runs_dir)
    found = sorted(p.parent.name for p in runs_dir.glob("*/done.json"))
    if names:
        missing = sorted(set(names) - set(found))
        if missing:
            raise SystemExit(f"not finished (no done.json): {missing}")
        found = [n for n in found if n in set(names)]
    out, skipped = {}, []
    for name in found:
        if listed is not None and name not in listed:
            if names:
                raise SystemExit(f"{name}: not in finished.txt yet")
            continue
        try:
            info = parse_run(name, reg)
        except ValueError:
            if names:
                raise
            skipped.append(name)
            continue
        d = runs_dir / name
        done = json.loads((d / "done.json").read_text(encoding="utf-8"))
        path = d / "final.vtsp"
        blob = path.read_bytes()
        sha = TS.sha256_bytes(blob)
        if sha != done.get("vtsp_sha256") or len(blob) != done.get("vtsp_bytes"):
            raise SystemExit(f"{name}: final.vtsp is not the blob done.json records")
        if listed is not None and listed[name] != (sha, len(blob)):
            raise SystemExit(f"{name}: final.vtsp is not the blob finished.txt lists")
        v = blob_version(blob)
        if v != FAMILY_VERSION[info["family"]]:
            raise SystemExit(f"{name}: a family-{info['family']} run with a VTSP version-{v} blob")
        settings = run_settings(d)
        if settings.pop("run_family") not in (None, info["family"]):
            raise SystemExit(f"{name}: run.json names another family than the registered shape's")
        m = model_of(blob)
        want_half = reg[info["shape"]]["half_above"] if info["family"] == "T" else m.half_above
        if (m.name, m.half_above, m.conf_min) != (info["shape"], want_half, 1):
            raise SystemExit(f"{name}: the blob is {m.name} at half_above {m.half_above}, conf_min {m.conf_min}")
        out[name] = dict(info, path=path, blob=blob, sha256=sha, bytes=len(blob), version=v,
                         half_above=m.half_above, steps=done.get("steps"), **settings)
    return out, skipped


def run_specs(runs):
    """eval/textspot.py's det and detref candidate specs of each run: the arms T3_<run>_det and T3_<run>_detref."""
    out = []
    for name, r in sorted(runs.items()):
        out += TS.candidate_specs(f"T3_{name}", r["blob"], r["path"], "T3", name, r["half_above"], KINDS,
                                  text=f"T3:{name}")
    return out


def is_scored(out, spec, manifest, frame_ids, models, cdir):
    """The arm's boxes are reusable (eval/textspot.py reusable: the same spec, manifest, twins and C check) for every
    frame, every model's units score those boxes, and its sidecar names the twin and C driver of its blob's version
    as they are now."""
    d = Path(out) / spec.name
    doc = TS.reusable(d / "boxes.json", spec.meta(), manifest, cdir is not None)
    if doc is None or not set(doc["frames"]) >= set(frame_ids):
        return False
    side = d / "textspot3.json"
    if not side.exists():
        return False
    sd = json.loads(side.read_text(encoding="utf-8"))
    v = blob_version(spec.blob)
    if sd.get("twin_sha256") != twin_sha(v) or (cdir is not None and sd.get("driver_sha256") != driver_sha(cdir, v)):
        return False
    bsha = TS.sha256(d / "boxes.json")
    for m in models or [None]:
        f = d / f"units_{m or 'detection'}.json"
        if not f.exists():
            return False
        meta = json.loads(f.read_text(encoding="utf-8"))["meta"]
        if meta.get("boxes_sha256") != bsha or meta.get("spec_meta") != spec.meta():
            return False
    return True


def pending(out, runs, manifest, frame_ids, models, cdir, force=False):
    """The specs of the runs' arms that are not scored yet."""
    specs = run_specs(runs)
    with dispatch(r["version"] for r in runs.values()):
        return [s for s in specs if force or not is_scored(out, s, manifest, frame_ids, models, cdir)]


def score_runs(frames, manifest, runs, specs, out, models, readers=None, cache=None, cdir=None, workers=5,
               force=False, log=print, proto=None):
    """The specs through eval/textspot.py's evaluate (boxes, C parity, reading, units and summaries) under the
    versions' dispatch, then a sidecar per arm, OUT/<arm>/textspot3.json: its run, recipe, shape, family, seed, arm,
    extras, blob version and sha256, the twin and C driver of that version, and this wrapper's hashes."""
    with dispatch(blob_version(s.blob) for s in specs):
        TS.evaluate(frames, specs, manifest, out, models, readers, cache, cdir, workers, force, log, proto)
    for s in specs:
        r = runs[s.run]
        side = {"kind": "textspot3-arm", "arm": s.name, "run": s.run, "recipe": r["recipe"], "shape": r["shape"],
                "family": r["family"], "seed": r["seed"], "arm_id": r["arm_id"], "decode": s.decode,
                "extras": r.get("extras", []), "blob_version": r["version"], "blob_sha256": r["sha256"],
                "twin_sha256": twin_sha(r["version"]),
                "driver_sha256": driver_sha(cdir, r["version"]) if cdir is not None else None,
                "code_sha256": code_identity(), "computed": TS.utc_now()}
        TS.write_json(Path(out) / s.name / "textspot3.json", side, indent=1)
    return [s.name for s in specs]


def log_line(s):
    print(s, flush=True)


def cmd_run(args, readers=None):
    proto, proto3 = TS.protocol(), protocol3()
    check_recognizers(proto, proto3)
    out = Path(args.out)
    cache_dir = Path(args.cache) if args.cache else (DEFAULT["cache"] if args.set == "dev" else out / "cache")
    check_writable(out, cache_dir, args.rbuild, args.frames)
    rows, manifest = TS.load_set(args, proto)
    check_dev_manifest(args.set, manifest, proto3)
    if args.set == "dev" and not args.cbin:
        raise SystemExit("the development set needs --cbin (the rule's C parity per frame)")
    runs, skipped = finished_runs(args.runs, args.names)
    for s in skipped:
        print(f"skipped {s}: not a run name of a registered shape", flush=True)
    models = TS.parse_models(args.models)
    cdir = resolve_cbin(args.cbin, out, args.compiler)
    todo = pending(out, runs, manifest, [r["id"] for r in rows], models, cdir, args.force)
    if getattr(args, "read_only_boxes", False):
        with dispatch(r["version"] for r in runs.values()):
            stale = [s.name for s in run_specs(runs) if not is_scored(out, s, manifest, [r["id"] for r in rows],
                                                                      [TS.RULE_MODEL], cdir)]
        if stale:
            raise SystemExit(f"not scored by {TS.RULE_MODEL} for these boxes yet (textspot3.py run first): {stale}")
    print(f"{len(rows)} frames, {len(runs)} finished runs, {len(todo)} arms to score, models {models or 'none'}",
          flush=True)
    if not todo:
        return []
    frames = TS.load_frames(rows, [Path(d) for d in args.shots] + list(TS.SHOT_DIRS))
    record_frames(args.frames or (DEFAULT["frames"] if args.set == "dev" else None), frames, manifest)
    cache = build = None
    if models:
        if args.set == "dev":
            seed_cache(cache_dir)
        if readers is None:
            build = Path(args.rbuild) / str(os.getpid())
            readers = T2E.readers_for(models, proto, build, args.compiler)
        cache = TS.Cache(cache_dir)
    try:
        return score_runs(frames, manifest, runs, todo, out, models, readers, cache, cdir, args.workers, args.force,
                          log_line, proto)
    finally:
        if cache is not None:
            cache.close()
        if build is not None:
            shutil.rmtree(build, ignore_errors=True)


def cmd_read_pick(args, readers=None):
    """read-pick: the named runs' stored arms (already scored by VOCR5-N; never re-detected) read by the report
    recognizers. With --dev, every run must be the selection record's pick."""
    if args.dev:
        doc = json.loads(Path(args.dev).read_text(encoding="utf-8"))
        win = doc["result"]["winner"]
        if win is None:
            raise SystemExit(f"{args.dev}: no pick (nothing is read)")
        run = next(s["run"] for s in doc["seeds"] if s["name"] == win)
        if set(args.names) != {run}:
            raise SystemExit(f"--names {args.names}: the record's pick is {win} (run {run})")
    if set(TS.parse_models(args.models)) - {"VOCR5-S", "VOCR5-L"}:
        raise SystemExit("read-pick reads the report recognizers VOCR5-S / VOCR5-L only")
    args.read_only_boxes, args.force = True, False
    return cmd_run(args, readers)


# ----------------------------------------------------------------------------- the arms select reads

def seed_record(run, doc, units, d, side, reg):
    s = units["summary"]
    return dict(parse_run(run, reg), name=doc["arm"], decode=doc["spec_meta"]["decode"], source="textspot3",
                dir=Path(d), boxes=doc, units=units, extras=side.get("extras", []), blob_version=side["blob_version"],
                f1=s["detection"]["f1"], cer_all=s["cer_all"]["cer"], cer_dialogue=s["cer_dialogue"]["cer"],
                ui_recall=s["recall_by_class"]["ui"]["recall"])


def t3_seeds(units_dirs, runs_dir=None, reg=None):
    """Every T3_<run>_det / _detref arm of the unit directories (VOCR5-N units and the sidecar required); with
    runs_dir, each arm's blob must still be its run's final.vtsp (done.json)."""
    reg = registry() if reg is None else reg
    out = []
    for d in units_dirs:
        for b in sorted(Path(d).glob("T3_*/boxes.json")):
            doc, units = T2E.load_arm(b.parent)
            meta = doc["spec_meta"]
            if meta.get("kind") not in KINDS or meta.get("family") != "T3":
                continue
            if units is None:
                raise SystemExit(f"{b.parent}: no {TS.RULE_MODEL} units (textspot3.py run first)")
            side_f = b.parent / "textspot3.json"
            if not side_f.exists():
                raise SystemExit(f"{b.parent}: no textspot3.json sidecar")
            side = json.loads(side_f.read_text(encoding="utf-8"))
            if side["blob_sha256"] != meta["blob_sha256"] or side["run"] != meta["run"]:
                raise SystemExit(f"{b.parent}: the sidecar names another run or blob")
            done = Path(runs_dir) / meta["run"] / "done.json" if runs_dir else None
            if done is not None and done.exists():
                if json.loads(done.read_text(encoding="utf-8")).get("vtsp_sha256") != meta["blob_sha256"]:
                    raise SystemExit(f"{b.parent}: scored another blob than {meta['run']}'s final.vtsp")
            out.append(seed_record(meta["run"], doc, units, b.parent, side, reg))
    return out


def a9_docs(docs):
    """A list of A9 tables from one table, a list of them or None."""
    if docs is None:
        return []
    return [docs] if isinstance(docs, dict) else [d for d in docs if d]


def a9_rows3(docs):
    """The A9 rows eligibility reads: the conductor-pinned textspot3 tables (eval/textspot3_a9_*.json; family T's
    and, once it exists, the version-3 runtime's), normalised by eval/textspot.py's a9_rows, in the order given
    (first match wins)."""
    rows = []
    for d in a9_docs(docs):
        rows += TS.a9_rows(d)
    return rows


def refine_arena3(docs, cdir, proto3):
    """eval/textspot2.py refine_arena2 on the first table that records the refinement's arena (else the
    refinement driver's, else the protocol's 72,391 B; any other figure is refused)."""
    ds = a9_docs(docs)
    doc = next((d for d in ds if TS.refine_arena(d, None)[0] is not None), None)
    return T2E.refine_arena2(doc, cdir, proto3)


def check_pinned(path, proto3):
    """An A9 table of the record is a file the protocol pins (its sources or an amendment's), byte for byte."""
    rel = TS.rel(path)
    want = TS.pinned_sha(proto3, rel) or proto3["sources_sha256"].get(rel)
    if want is None:
        raise SystemExit(f"{path}: not pinned by textspot3's protocol (the A9 tables are the conductor-pinned ones)")
    if TS.sha256(path) != want:
        raise SystemExit(f"{path}: not the table textspot3's protocol pins ({want[:12]}...)")
    return want


def eligibility3(meta, boxes_doc, rows, ref_arena, reg):
    """eval/textspot.py eligibility, with the MACs it computes and checks against the A9 row taken from the shape's
    own cost function (macs_of: train/textspot2_export.cost for family T, train/textspot3_model.cost for M;
    textdet_model's cost cannot read either head); everything else unchanged."""
    macs, src = macs_of(meta["shape"], reg)
    saved = TS.macs_960
    TS.macs_960 = lambda shape, half_above: macs
    try:
        e = TS.eligibility(meta, boxes_doc, rows, ref_arena)
    finally:
        TS.macs_960 = saved
    return dict(e, macs_source=src, family=reg[meta["shape"]]["family"])


def report_costs(e, rows):
    """The report's cost columns of an eligibility record: est. ms at 444 MHz (x 333/444), and the spotter's est. ms
    and arena on the PSP's 480x272 frame (the A9 row's, without the refinement)."""
    ms = e.get("est_ms_333_960")
    r = TS.a9_lookup(rows, e["shape"], e["half_above"])
    row = (r or {}).get("row", {})
    psp = row.get("ms_333_480")
    return {"est_ms_444_960": round(ms * MHZ, 3) if ms is not None else None,
            "ms_333_480_spot": round(psp, 3) if psp is not None else None,
            "ms_444_480_spot": round(psp * MHZ, 3) if psp is not None else None,
            "arena_up_to_480": row.get("arena_up_to_480")}


def representatives(seeds):
    """{(arm_id, decode): {"order", "index", "rep", "seeds"}}: each arm's and decode's representative by
    textspot3_rule.median_seed over its finished seeds."""
    groups = {}
    for s in seeds:
        groups.setdefault((s["arm_id"], s["decode"]), []).append(s)
    out = {}
    for key, ss in sorted(groups.items()):
        if len({x["seed"] for x in ss}) != len(ss):
            raise SystemExit(f"{key}: a seed appears twice")
        rep, info = R.median_seed(ss)
        out[key] = dict(info, rep=rep, seeds=sorted(ss, key=lambda x: x["seed"]))
    return out


def candidate_record(seed):
    """textspot3_rule's candidate record of a representative."""
    e = seed["elig"]
    return {"name": seed["name"], "arm_id": seed["arm_id"], "seed": seed["seed"], "decode": seed["decode"],
            "blob_bytes": e["blob_bytes"], "arena_960": e["arena_960"], "est_ms_333_960": e["est_ms_333_960"],
            "c_equal": e["c_equal"], "arm": TS.rule_arm(seed["units"]["units"])}


def light(e):
    """The fields textspot3_rule.eligible reads, from an eligibility record."""
    return {"blob_bytes": e["blob_bytes"], "arena_960": e["arena_960"], "est_ms_333_960": e["est_ms_333_960"],
            "c_equal": e["c_equal"]}


def gather(units_dirs, proto3, a9_doc=None, cdir=None, runs_dir=None, reg=None, need_rows=False):
    """Every seed with its eligibility inputs -> (seeds, A9 rows, (refine arena, its source)). a9_doc: one A9
    table or a list of them. need_rows: a seed whose shape and scale have no row in the tables is an error (the
    tables are pinned before any score, so a missing row means another table)."""
    reg = registry(proto3) if reg is None else reg
    man = proto3["reused"]["dev_manifest"]["sha256"]
    rows_a9 = a9_rows3(a9_doc)
    ref_ar = refine_arena3(a9_doc, cdir, proto3)
    seeds = t3_seeds(units_dirs, runs_dir, reg)
    for s in seeds:
        if {s["boxes"]["manifest"]["sha256"], s["units"]["meta"]["manifest"]["sha256"]} != {man}:
            raise SystemExit(f"{s['name']}: scored on another manifest than the protocol's development set")
        s["elig"] = eligibility3(s["boxes"]["spec_meta"], s["boxes"], rows_a9, ref_ar[0], reg)
        s["elig"].update(report_costs(s["elig"], rows_a9))
    if need_rows:
        miss = sorted({(s["shape"], s["elig"]["half_above"]) for s in seeds if s["elig"]["a9_row"] is None})
        if miss:
            raise SystemExit(f"no A9 row at 960x544 in the tables for (shape, half_above) {miss}")
    return seeds, rows_a9, ref_ar


# ----------------------------------------------------------------------------- the report's pieces

PARETO = (("blob_bytes", 1), ("est_ms_333_960", 1), ("est_ms_444_960", 1), ("arena_960", 1), ("f1", -1),
          ("cer_all", 1), ("cer_dialogue", 1), ("ui_recall", -1))
NOT_EXTRAS_ON_L = ("base", "kd")       # on L every other recipe is base + the kept extras (size_, clean_,
                                       # size+clean_: train/textspot3_train.py's names)


def seed_row(s, rule):
    """One seed's row of the record: counts, rates, hashes, its eligibility inputs and its full summary."""
    su, e = s["units"]["summary"], s["elig"]
    return {"name": s["name"], "run": s["run"], "arm_id": s["arm_id"], "recipe": s["recipe"], "shape": s["shape"],
            "family": s["family"], "seed": s["seed"], "decode": s["decode"], "extras": s.get("extras", []),
            "blob_version": s.get("blob_version"), "units_sha256": s["units"]["_sha256"],
            "boxes_sha256": s["boxes"]["_sha256"], "blob_sha256": e["blob_sha256"], "blob_bytes": e["blob_bytes"],
            "macs_960": e["macs_960"], "macs_source": e.get("macs_source"), "a9_row": e["a9_row"],
            "arena_spot_960": e["arena_spot_960"], "arena_960": e["arena_960"], "est_ms_333_960": e["est_ms_333_960"],
            "est_ms_444_960": e.get("est_ms_444_960"), "ms_333_480_spot": e.get("ms_333_480_spot"),
            "ms_444_480_spot": e.get("ms_444_480_spot"), "arena_up_to_480": e.get("arena_up_to_480"),
            "median_h_boxes_vita": e.get("median_h_boxes_vita"), "c_equal": e["c_equal"],
            "c_unequal_frames": e["c_unequal_frames"], "c_unchecked_frames": e["c_unchecked_frames"],
            "eligible": R.eligible(light(e), rule), "f1": s["f1"], "cer_all": s["cer_all"],
            "cer_dialogue": s["cer_dialogue"], "ui_recall": s["ui_recall"], "summary": su}


def point_of(name, role, costs, summary):
    return {"role": role, "blob_bytes": costs["blob_bytes"], "est_ms_333_960": costs["est_ms_333_960"],
            "est_ms_444_960": round(costs["est_ms_333_960"] * MHZ, 3), "arena_960": costs["arena_960"],
            "f1": summary["detection"]["f1"], "cer_all": summary["cer_all"]["cer"],
            "cer_dialogue": summary["cer_dialogue"]["cer"], "ui_recall": summary["recall_by_class"]["ui"]["recall"]}


def pareto_front(points):
    """points: {name: {PARETO keys}} -> the names no other point dominates (no worse on every key, F1 and UI recall
    counted higher-better, and better on one), sorted by name."""
    vec = {n: tuple(sign * p[k] for k, sign in PARETO) for n, p in points.items()}
    return sorted(n for n, v in vec.items() if not any(w != v and all(a <= b for a, b in zip(w, v))
                                                       for w in vec.values()))


def ranks(xs):
    """Average ranks (1-based) of xs, ties sharing their mean rank."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(xs, ys):
    """Spearman's rank correlation of two equal-length lists (None for fewer than 3 points or a constant list)."""
    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return round(sxy / (sxx * syy) ** 0.5, 4) if sxx and syy else None


def ui_trend(reps, points):
    """Whether UI recall rises with VTSP bytes (report; docs/textspot3.md "Report only"), per decode: every
    representative's (VTSP bytes, UI recall), Spearman's rank correlation over them, and the base arms of family M
    in size order (S, M, L). "rises": the correlation is above 0 and base's UI recall does not fall from S to M to
    L."""
    out = {"definition": "per decode: Spearman's rank correlation of VTSP bytes and UI recall over every arm's "
                         "representative (development, VOCR5-N) is above 0, and the M base arms' UI recall is "
                         "non-decreasing from S to M to L", "by_decode": {}}
    for dec in DECODES:
        pts = sorted((g["rep"]["elig"]["blob_bytes"], g["rep"]["name"], g["rep"]["ui_recall"])
                     for (aid, d), g in reps.items() if d == dec and g["rep"]["ui_recall"] is not None)
        rho = spearman([p[0] for p in pts], [p[2] for p in pts])
        base = [(p, reps[(f"base_{points[p]}", dec)]["rep"]) for p in POINTS
                if p in points and (f"base_{points[p]}", dec) in reps]
        mono = None if len(base) < 2 else all(a[1]["ui_recall"] <= b[1]["ui_recall"] for a, b in zip(base, base[1:]))
        out["by_decode"][dec] = {
            "points": [{"name": n, "blob_bytes": b, "ui_recall": u} for b, n, u in pts], "spearman_rho": rho,
            "m_base": [{"point": p, "name": s["name"], "blob_bytes": s["elig"]["blob_bytes"],
                        "ui_recall": s["ui_recall"]} for p, s in base],
            "m_base_non_decreasing": mono, "rises": None if rho is None else bool(rho > 0 and mono is not False)}
    return out


def by_bytes(reps, shapes):
    """The shapes ordered by their runs' VTSP bytes (then name)."""
    nb = {g["rep"]["shape"]: g["rep"]["elig"]["blob_bytes"] for g in reps.values()}
    return sorted(shapes, key=lambda s: (nb.get(s, 1 << 62), s))


def bought(reps, ref, points, reg, winner=None):
    """docs/textspot3.md's "what each arm bought": paired development differences of median seeds, both decodes
    (textspot_metrics.compare): each M size's base against the next smaller one and against every T shape's base;
    size, clean and long minus base on M; kd minus its shape's base; base + the kept extras on L (any other recipe
    on L) minus base on L; every arm against ref, and the pick against ref."""
    ids = sorted({aid for aid, _ in reps})
    recipe = {aid: reps[next(k for k in reps if k[0] == aid)]["rep"]["recipe"] for aid in ids}
    shape = {aid: reps[next(k for k in reps if k[0] == aid)]["rep"]["shape"] for aid in ids}
    t_shapes = by_bytes(reps, [n for n, s in reg.items() if s["family"] == "T"])
    pairs = []
    for small, big in zip(POINTS, POINTS[1:]):
        if small in points and big in points:
            pairs.append((f"base {big} - base {small}", f"base_{points[big]}", f"base_{points[small]}"))
    for p in POINTS:
        if p in points:
            pairs += [(f"base {p} - base T {t}", f"base_{points[p]}", f"base_{t}") for t in t_shapes]
    if "M" in points:
        pairs += [(f"{r} M - base M", f"{r}_{points['M']}", f"base_{points['M']}") for r in STAGE1_RECIPES[1:]]
    pairs += [(f"kd {shape[a]} - base {shape[a]}", a, f"base_{shape[a]}") for a in ids if recipe[a] == "kd"]
    if "L" in points:
        pairs += [(f"{recipe[a]} L - base L (the kept extras)", a, f"base_{points['L']}") for a in ids
                  if shape[a] == points["L"] and recipe[a] not in NOT_EXTRAS_ON_L]
    out = []
    for what, a_id, b_id in pairs:
        for dec in DECODES:
            ga, gb = reps.get((a_id, dec)), reps.get((b_id, dec))
            if ga is None or gb is None:
                out.append({"what": what, "decode": dec, "minuend_arm": a_id, "subtrahend_arm": b_id, "missing": True})
                continue
            out.append({"what": what, "decode": dec, "minuend": ga["rep"]["name"], "subtrahend": gb["rep"]["name"],
                        "minuend_arm": a_id, "subtrahend_arm": b_id,
                        "comparison": M.compare(TS.rule_arm(ga["rep"]["units"]["units"]),
                                                TS.rule_arm(gb["rep"]["units"]["units"]))})
    by_name = {g["rep"]["name"]: g for g in reps.values()}
    for aid in ids:
        for dec in DECODES:
            g = reps.get((aid, dec))
            if g is not None:
                out.append({"what": f"{aid} - ref", "decode": dec, "minuend": g["rep"]["name"],
                            "subtrahend": ref["name"], "minuend_arm": aid, "subtrahend_arm": "ref",
                            "comparison": M.compare(TS.rule_arm(g["rep"]["units"]["units"]), ref["arm"])})
    if winner:
        g = by_name[winner]
        out.append({"what": "pick - ref", "decode": g["rep"]["decode"], "minuend": winner, "subtrahend": ref["name"],
                    "minuend_arm": g["rep"]["arm_id"], "subtrahend_arm": "ref",
                    "comparison": M.compare(TS.rule_arm(g["rep"]["units"]["units"]), ref["arm"])})
    return out


def brief(s):
    return {"name": s["name"], "run": s["run"], "shape": s["shape"], "recipe": s["recipe"], "decode": s["decode"],
            "blob_bytes": s["elig"]["blob_bytes"], "est_ms_333_960": s["elig"]["est_ms_333_960"], "f1": s["f1"],
            "cer_all": s["cer_all"], "cer_dialogue": s["cer_dialogue"]}


def stage2_inputs(reps, ref_arm, rule, points):
    """docs/textspot3.md's fixed stage-2 rules read off the stage-1 representatives (report; the conductor's
    amendment records the choices): the kept extras (det+refine on M), the kd teacher (the largest M shape with an
    eligible stage-1 candidate, its lowest all-text CER, any recipe or decode), the kd students (the two smallest
    shapes other than the teacher's whose stage-1 base candidate, either decode, qualifies against ref; the
    smallest eligible shapes make up two) and whether base + the kept extras runs on L."""
    def rep(aid, dec="det+refine"):
        g = reps.get((aid, dec))
        return g["rep"] if g else None

    out = {"points": dict(points)}
    m = points.get("M")
    b, sz, cl = (rep(f"base_{m}"), rep(f"size_{m}"), rep(f"clean_{m}")) if m else (None, None, None)
    out["size_kept"] = None if not (b and sz) else bool(sz["cer_all"] < b["cer_all"] and
                                                        sz["cer_dialogue"] <= b["cer_dialogue"] + 0.25)
    out["clean_kept"] = None if not (b and cl) else bool(cl["cer_dialogue"] < b["cer_dialogue"] and
                                                         cl["cer_all"] <= b["cer_all"] + 1.0)
    out["read"] = {k: brief(x) for k, x in (("base_M", b), ("size_M", sz), ("clean_M", cl)) if x}
    stage1 = [g["rep"] for g in reps.values() if g["rep"]["recipe"] in STAGE1_RECIPES]
    elig = [s for s in stage1 if R.eligible(light(s["elig"]), rule)]
    nb = {s["shape"]: s["elig"]["blob_bytes"] for s in stage1}
    teacher = None
    for sh in sorted({s["shape"] for s in stage1 if s["family"] == "M"}, key=lambda x: (-nb[x], x)):
        pool = [s for s in elig if s["shape"] == sh]
        if pool:
            teacher = min(pool, key=lambda s: (s["cer_all"], s["name"]))
            break
    out["kd_teacher"] = brief(teacher) if teacher else None
    base = sorted({s["shape"] for s in stage1 if s["recipe"] == "base"} - ({teacher["shape"]} if teacher else set()),
                  key=lambda x: (nb[x], x))
    qual, el_shapes = [], []
    for sh in base:
        cs = [s for s in elig if s["shape"] == sh and s["recipe"] == "base"]
        if cs:
            el_shapes.append(sh)
        if any(R.qualifies(M.compare(TS.rule_arm(s["units"]["units"]), ref_arm)) for s in cs):
            qual.append(sh)
    students = qual[:2] + [sh for sh in el_shapes if sh not in qual][:max(0, 2 - len(qual[:2]))]
    out["kd_students"] = [{"shape": sh, "blob_bytes": nb[sh], "base_qualifies": sh in qual} for sh in students]
    kept = [x for x, k in (("size", out["size_kept"]), ("clean", out["clean_kept"])) if k]
    out["extras_on_L"] = {"shape": points.get("L"), "extras": kept, "runs": bool(kept)}
    return out


# ----------------------------------------------------------------------------- select

def recall_of(units_doc):
    return R.recall_by_class(units_doc["units"])


def unscored_runs(runs_dir, seeds, reg):
    """The finished runs (done.json; listed in finished.txt when it exists) of a registered shape whose two arms
    (det and det+refine) are not both among the scored seeds: select reads every finished run or none."""
    runs_dir = Path(runs_dir)
    listed = T2E.finished_list(runs_dir)
    have = {}
    for s in seeds:
        have.setdefault(s["run"], set()).add(s["decode"])
    out = []
    for n in sorted(p.parent.name for p in runs_dir.glob("*/done.json")):
        if listed is not None and n not in listed:
            continue
        try:
            parse_run(n, reg)
        except ValueError:
            continue
        if have.get(n) != set(DECODES):
            out.append(n)
    return out


def part_report(summary):
    """The report's per-part, text-free and dialogue-pick figures of one summary (counts and rates)."""
    tf, pk = summary["text_free"], summary["pick"]
    return {"per_part": {p: {"frames": q["frames"], "f1": (q.get("detection") or {}).get("f1"),
                             "cer_all": (q.get("cer_all") or {}).get("cer"),
                             "cer_dialogue": (q.get("cer_dialogue") or {}).get("cer")}
                         for p, q in summary.get("per_part", {}).items()},
            "text_free": {"frames": tf["frames"], "with_any_box": tf["with_any_box"],
                          "boxes_per_frame": tf["boxes_per_frame"]},
            "dialogue_pick": {"frames_with_dialogue": pk["frames_with_dialogue"], "hits": pk["hits"],
                              "hit_share": pk["hit_share"], "cer": (pk.get("cer") or {}).get("cer")}}


def select(units_dirs, a9_doc, proto3, base_doc, cdir=None, runs_dir=None, baseline_units=None, ref=None,
           reg=None, log=print):
    """The selection record: every seed with its eligibility inputs and summary, the representatives,
    textspot3_rule.apply_rule3 once over every arm's representatives against ref (with textspot's eight baselines
    for the report-only bar), and the report (recall per class, UI recall against VTSP bytes, the Pareto front,
    what each arm bought, per part, the text-free frames, the dialogue pick, the pick, the stage-2 inputs). With
    runs_dir, every finished run must be scored."""
    rule = proto3["rule"]
    reg = registry(proto3) if reg is None else reg
    man = proto3["reused"]["dev_manifest"]["sha256"]
    if base_doc["manifest"]["sha256"] != man:
        raise SystemExit("the frozen baseline is of another manifest")
    seeds, rows_a9, (ref_ar, ref_src) = gather(units_dirs, proto3, a9_doc, cdir, runs_dir, reg, need_rows=True)
    if not seeds:
        raise SystemExit("no scored textspot3 arm under the unit directories")
    missing = unscored_runs(runs_dir, seeds, reg) if runs_dir else []
    if missing:
        raise SystemExit(f"finished runs not scored yet (textspot3.py run first): {missing}")
    bases = T2E.baselines_of(base_doc, baseline_units)
    ref = ref if ref is not None else T2E.reference(proto3, "ref")
    for x in [ref] + list(bases.values()):
        if {x["boxes"]["manifest"]["sha256"], x["units"]["meta"]["manifest"]["sha256"]} != {man}:
            raise SystemExit(f"{x['name']}: scored on another manifest than the protocol's development set")
    reps = representatives(seeds)
    cands = [candidate_record(g["rep"]) for g in reps.values()]
    log(f"{len(seeds)} seeds, {len(reps)} arms x decodes, {len(cands)} candidates")
    result = R.apply_rule3(cands, {"name": ref["name"], "arm": ref["arm"]},
                           {b: v["arm"] for b, v in bases.items()}, rule)
    extra = {}
    for c in cands:              # the ineligible representatives (read anyway): their comparisons, report only
        if c["name"] in result["comparisons"]:
            continue
        bars = {b: M.compare(c["arm"], v["arm"]) for b, v in sorted(bases.items())}
        vr = M.compare(c["arm"], ref["arm"])
        extra[c["name"]] = {"vs_ref": vr, "qualifies_if_eligible": R.qualifies(vr), "baselines": bars,
                            "bar": all(M.beats(x) for x in bars.values())}
    points, points_src = size_points(reg, {s["shape"]: s["elig"]["blob_bytes"] for s in seeds})
    arms, rep_out = {}, {}
    for (aid, dec), g in reps.items():
        s = g["rep"]
        arms[f"{aid}|{dec}"] = {"arm_id": aid, "decode": dec, "recipe": s["recipe"], "shape": s["shape"],
                                "family": s["family"], "seeds": [x["name"] for x in g["seeds"]], "order": g["order"],
                                "index": g["index"], "representative": s["name"], "finished": len(g["seeds"])}
        rep_out[s["name"]] = {"arm_id": aid, "decode": dec, "seed": s["seed"], "run": s["run"],
                              "eligibility": s["elig"], "eligible": R.eligible(light(s["elig"]), rule),
                              "recall_by_class": recall_of(s["units"]), **part_report(s["units"]["summary"])}
    pts = {g["rep"]["name"]: point_of(g["rep"]["name"], "candidate", g["rep"]["elig"], g["rep"]["units"]["summary"])
           for g in reps.values() if R.eligible(light(g["rep"]["elig"]), rule)}
    rp = proto3["references"]["ref"]
    ref_pt = point_of(ref["name"], "ref", {"blob_bytes": rp["blob_bytes"], "est_ms_333_960": rp["est_ms_333_960"],
                                           "arena_960": rp["arena_960"]}, ref["units"]["summary"])
    winner = result["winner"]
    by_name = {g["rep"]["name"]: g["rep"] for g in reps.values()}
    pick = None
    if winner:
        w = by_name[winner]
        pick = {"name": winner, "run": w["run"], "arm_id": w["arm_id"], "decode": w["decode"], "seed": w["seed"],
                "blob_sha256": w["elig"]["blob_sha256"], "blob_bytes": w["elig"]["blob_bytes"],
                "est_ms_333_960": w["elig"]["est_ms_333_960"], "arena_960": w["elig"]["arena_960"],
                "boxes_sha256": w["boxes"]["_sha256"], "units_sha256": w["units"]["_sha256"],
                "vs_ref": result["comparisons"][winner]["vs_ref"], "steps": len(result["steps"])}
    doc = {"kind": "textspot3-selection", "computed": TS.utc_now(), "rule": rule,
           "refine_arena": {"bytes_960x544": ref_ar, "source": ref_src}, "refine_ms_per_box": TS.REFINE_MS_PER_BOX,
           "a9_calibration": TS.A9_CAL, "manifest": base_doc["manifest"], "code_sha256": code_identity(),
           "shapes": {n: {"family": s["family"]} for n, s in sorted(reg.items())},
           "size_points": {"points": points, "source": points_src},
           "references": {ref["name"]: {"role": "ref", "units_sha256": ref["units"]["_sha256"],
                                        "boxes_sha256": ref["boxes"]["_sha256"], "blob_bytes": rp["blob_bytes"],
                                        "est_ms_333_960": rp["est_ms_333_960"], "arena_960": rp["arena_960"],
                                        "summary": ref["units"]["summary"],
                                        "recall_by_class": recall_of(ref["units"])}},
           "baselines": {b: {"units_sha256": v["units"]["_sha256"], "boxes_sha256": v["boxes"]["_sha256"],
                             "f1": v["units"]["summary"]["detection"]["f1"],
                             "cer_all": v["units"]["summary"]["cer_all"],
                             "cer_dialogue": v["units"]["summary"]["cer_dialogue"],
                             "recall_by_class": recall_of(v["units"])} for b, v in sorted(bases.items())},
           "seeds": [seed_row(s, rule) for s in sorted(seeds, key=lambda x: x["name"])],
           "arms": arms, "representatives": rep_out, "candidates": [c["name"] for c in cands], "result": result,
           "report_comparisons": extra, "pick": pick,
           "ui_recall_vs_bytes": ui_trend(reps, points),
           "pareto": {"keys": [k for k, _ in PARETO], "points": dict(pts, **{ref["name"]: ref_pt}),
                      "front": pareto_front(pts), "front_with_ref": pareto_front(dict(pts, **{ref["name"]: ref_pt}))},
           "bought": bought(reps, ref, points, reg, winner),
           "stage2_inputs": stage2_inputs(reps, ref["arm"], rule, points)}
    TS.assert_public(doc)
    return doc


write_compact = T2E.write_compact
load_json = T2E.load_json


def check_baseline_pin(path, proto3):
    """The frozen baseline is the file textspot3's protocol pins."""
    want = proto3["sources_sha256"].get("eval/textspot_baseline_20261019.json")
    if Path(path).resolve() == BASELINE.resolve() and want and TS.sha256(path) != want:
        raise SystemExit(f"{path}: not the baseline textspot3's protocol pins")


def cmd_select(args):
    proto3 = protocol3()
    check_baseline_pin(args.baseline, proto3)
    for a in args.a9:
        check_pinned(a, proto3)
    base_doc = load_json(args.baseline)
    doc = select(args.units, [load_json(a) for a in args.a9], proto3, base_doc,
                 Path(args.cbin) if args.cbin else None, args.runs, log=log_line)
    doc["inputs"] = {"protocol3_sha256": TS.sha256(PROTOCOL3), "baseline_sha256": TS.sha256(args.baseline),
                     "a9": [{"path": TS.rel(a), "sha256": TS.sha256(a)} for a in args.a9],
                     "units_dirs": [TS.rel(u) for u in args.units]}
    TS.assert_public(doc)
    n = write_compact(args.out, doc)
    copy = Path(args.indent_copy) if args.indent_copy else EVAL3 / f"{Path(args.out).stem}.indent1.json"
    check_writable(copy)
    TS.write_json(copy, doc, indent=1)
    res = doc["result"]
    print(f"{args.out}: {n:,} B (indented copy {copy}); {len(doc['seeds'])} seeds, {len(doc['candidates'])} "
          f"candidates, {len(res['eligible'])} eligible, {len(res['qualifying'])} qualify, {len(res['bar'])} clear "
          f"textspot's bar; pick {res['winner']}", flush=True)


# ----------------------------------------------------------------------------- headline

def _f(v, nd=2):
    return "–" if v is None else f"{v:.{nd}f}"


def _cer(c):
    return "–" if not c or c.get("cer") is None else f"{c['cer']:.2f}"


ci_txt = T2E.ci_txt


def headline_md(seeds, rule, reg, ref_arm=None, stage1=None):
    """Markdown for the conductor: every seed's F1, CER all, dialogue, UI recall and est. ms; the representatives
    (median seed) per arm and decode; stage 1's count; the stage-2 inputs read off the representatives."""
    reps = representatives(seeds)
    points, src = size_points(reg, {s["shape"]: s["elig"]["blob_bytes"] for s in seeds})
    md = ["| seed | decode | VTSP B | F1 | CER all | dialogue | UI recall | est. ms @333 / @444 | eligible |",
          "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for s in sorted(seeds, key=lambda x: (x["arm_id"], x["decode"], x["seed"])):
        e = s["elig"]
        md.append(f"| `{s['name']}` | {s['decode']} | {e['blob_bytes']:,} | {_f(s['f1'], 3)} | {_f(s['cer_all'])} | "
                  f"{_f(s['cer_dialogue'])} | {_f(s['ui_recall'], 3)} | {_f(e['est_ms_333_960'], 1)} / "
                  f"{_f(e.get('est_ms_444_960'), 1)} | {R.eligible(light(e), rule)} |")
    md += ["", "| arm | decode | seeds, best first | representative | F1 | CER all | dialogue | UI recall | "
               "est. ms @333 | eligible |", "|---|---|---|---|---:|---:|---:|---:|---:|---|"]
    for (aid, dec), g in reps.items():
        s = g["rep"]
        md.append(f"| `{aid}` | {dec} | {', '.join(map(str, g['order']))} | seed {s['seed']} | {_f(s['f1'], 3)} | "
                  f"{_f(s['cer_all'])} | {_f(s['cer_dialogue'])} | {_f(s['ui_recall'], 3)} | "
                  f"{_f(s['elig']['est_ms_333_960'], 1)} | {R.eligible(light(s['elig']), rule)} |")
    stage1 = stage1 if stage1 is not None else stage1_runs(reg, points)
    done = {s["run"] for s in seeds}
    missing = [n for n in stage1 if n not in done]
    md += ["", f"Stage 1: {len(stage1) - len(missing)} of {len(stage1)} runs scored"
               + (f"; not yet: {', '.join(missing)}" if missing else "")
               + f" (M size points {json.dumps(points)}, by {src}).", ""]
    if ref_arm is not None:
        md.append("Stage-2 inputs (the protocol's fixed rules read off these representatives; the conductor's "
                  "amendment decides): `" + json.dumps(stage2_inputs(reps, ref_arm, rule, points), ensure_ascii=False)
                  + "`")
    return "\n".join(md) + "\n"


def cmd_headline(args):
    proto3 = protocol3()
    reg = registry(proto3)
    seeds, _, _ = gather(args.units, proto3, [load_json(a) for a in args.a9], Path(args.cbin) if args.cbin else None,
                         args.runs, reg)
    ref = T2E.reference(proto3, "ref")
    stage1 = json.loads(Path(args.stage1).read_text(encoding="utf-8")) if args.stage1 else None
    text = headline_md(seeds, proto3["rule"], reg, ref["arm"], stage1)
    if args.out:
        check_writable(args.out)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


# ----------------------------------------------------------------------------- tables (docs/textspot3.md)

def _ms(ms):
    return "–" if ms is None else f"{ms:.1f} / {ms * MHZ:.1f}"


def _n(v):
    return "–" if v is None else f"{v:,}"


def comparison_of(doc, name):
    """(the record's comparison against ref, textspot's bar) of a representative: the rule's (eligible) or the
    report's (ineligible)."""
    c = doc["result"]["comparisons"].get(name) or doc["report_comparisons"].get(name) or {}
    return c.get("vs_ref"), c.get("bar")


def selection_md(doc):
    res = doc["result"]
    md = [f"- Pick: **{res['winner'] or 'none'}** (comparator `{res['comparator']}`; qualifying: "
          f"{', '.join(f'`{n}`' for n in res['qualifying']) or 'none'})."]
    for st in res["steps"]:
        c = st["comparison"]
        md.append(f"  - `{st['by']}` replaces `{st['replaces']}`: CER all {ci_txt(c['cer_all'], 2)}, dialogue "
                  f"{ci_txt(c['cer_dialogue'], 2)}, F1 {ci_txt(c['f1'], 4)}")
    md.append(f"- {len(doc['candidates'])} candidates (the median seed of every arm and decode), "
              f"{len(res['eligible'])} eligible ({len(res['ineligible'])} not), {len(res['qualifying'])} qualify "
              f"against `{res['comparator']}`, {len(res['bar'])} clear textspot's bar (report only).")
    if not res["winner"]:
        md.append("- No qualifier: the verdict is negative and textspot's winner stands.")
    return "\n".join(md)


def candidates_md(doc):
    res = doc["result"]
    rows = []
    for name in doc["candidates"]:
        r = doc["representatives"][name]
        e = r["eligibility"]
        rows.append(((e["blob_bytes"], e["est_ms_333_960"] if e["est_ms_333_960"] is not None else float("inf"),
                      e["arena_960"] if e["arena_960"] is not None else 1 << 62, r["decode"] != "det", name), name))
    md = ["| arm (median seed) | decode | VTSP B | est. ms @333 / @444 | arena B | C = twin | F1 | CER all | "
          "dialogue | UI recall | eligible | qualifies | dialogue − ref | CER all − ref | F1 − ref | bar |",
          "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|"]
    seeds = {s["name"]: s for s in doc["seeds"]}
    for _, name in sorted(rows):
        r = doc["representatives"][name]
        e, s = r["eligibility"], seeds[name]
        vr, bar = comparison_of(doc, name)
        md.append(f"| `{name}` | {r['decode']} | {e['blob_bytes']:,} | {_ms(e['est_ms_333_960'])} | "
                  f"{_n(e['arena_960'])} | {e['c_equal']} | {_f(s['f1'], 3)} | {_f(s['cer_all'])} | "
                  f"{_f(s['cer_dialogue'])} | {_f(s['ui_recall'], 3)} | {name in res['eligible']} | "
                  f"{name in res['qualifying']} | {ci_txt((vr or {}).get('cer_dialogue'), 2)} | "
                  f"{ci_txt((vr or {}).get('cer_all'), 2)} | {ci_txt((vr or {}).get('f1'), 4)} | {bar} |")
    return "\n".join(md)


def pareto_md(doc):
    p = doc["pareto"]
    md = ["| arm | role | VTSP B | est. ms @333 / @444 | arena B | F1 | CER all | dialogue | UI recall |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    names = sorted(set(p["front"]) | {n for n, x in p["points"].items() if x["role"] == "ref"},
                   key=lambda k: (p["points"][k]["blob_bytes"], p["points"][k]["est_ms_333_960"], k))
    for n in names:
        x = p["points"][n]
        md.append(f"| `{n}` | {x['role']}{' (front)' if n in p['front'] else ''} | {x['blob_bytes']:,} | "
                  f"{_ms(x['est_ms_333_960'])} | {x['arena_960']:,} | {_f(x['f1'], 3)} | {_f(x['cer_all'])} | "
                  f"{_f(x['cer_dialogue'])} | {_f(x['ui_recall'], 3)} |")
    n_c = sum(1 for x in p["points"].values() if x["role"] != "ref")
    md += ["", f"{len(p['front'])} of {n_c} eligible representatives are on the front of VTSP bytes, est. ms (333 "
               "and 444 MHz), arena, F1, CER all, dialogue and UI recall; with ref as a point the front is "
               f"{', '.join(f'`{n}`' for n in p['front_with_ref'])}. est. ms are A9 cycle-model estimates."]
    return "\n".join(md)


def recall_md(doc):
    cls = M.CLASSES
    ref = next(iter(doc["references"]))
    first = doc["references"][ref]["recall_by_class"]
    md = ["| recall by class | " + " | ".join(f"{c} ({first[c]['n']})" for c in cls) + " |",
          "|---|" + "---:|" * len(cls)]
    rows = [(f"`{n}`", doc["representatives"][n]["recall_by_class"]) for n in doc["candidates"]]
    rows += [(f"`{ref}` (ref)", first)] + [(f"`{b}` (baseline)", v["recall_by_class"])
                                           for b, v in doc["baselines"].items()]
    for label_, r in rows:
        md.append(f"| {label_} | " + " | ".join(_f(r[c]["recall"], 3) for c in cls) + " |")
    md.append("")
    for dec, t in doc["ui_recall_vs_bytes"]["by_decode"].items():
        verdict = {True: "rises", False: "does not rise", None: "cannot be said (too few arms)"}[t["rises"]]
        base = ", ".join(f"{b['point']} {_f(b['ui_recall'], 3)}" for b in t["m_base"]) or "–"
        md.append(f"- {dec}: UI recall {verdict} with VTSP bytes (Spearman ρ {_f(t['spearman_rho'], 3)} over "
                  f"{len(t['points'])} arms; M base {base}).")
    return "\n".join(md)


def bought_md(doc):
    md = ["| what | decode | minuend − subtrahend | dialogue Δ, points (95% CI) | CER all Δ, points | F1 Δ |",
          "|---|---|---|---:|---:|---:|"]
    for b in doc["bought"]:
        if b.get("missing"):
            md.append(f"| {b['what']} | {b['decode']} | not run | – | – | – |")
            continue
        c = b["comparison"]
        md.append(f"| {b['what']} | {b['decode']} | `{b['minuend']}` − `{b['subtrahend']}` | "
                  f"{ci_txt(c['cer_dialogue'], 2)} | {ci_txt(c['cer_all'], 2)} | {ci_txt(c['f1'], 4)} |")
    return "\n".join(md)


def parts_md(doc, name=None):
    """Per part (F1 / CER all / dialogue), the text-free frames with a box and the dialogue pick of the pick (else
    the eligible representative of the lowest dialogue CER) beside ref."""
    seeds = {s["name"]: s for s in doc["seeds"]}
    res = doc["result"]
    name = name or res["winner"]
    if name is None and res["eligible"]:
        name = min(res["eligible"], key=lambda n: (seeds[n]["cer_dialogue"], seeds[n]["cer_all"], n))
    if name is None:
        return "No eligible candidate: no per-part table."
    refn = next(iter(doc["references"]))
    arms = [(name, seeds[name]["summary"]), (refn, doc["references"][refn]["summary"])]
    md = ["| part | frames | " + " | ".join(f"`{a}` F1 / CER all / dialogue" for a, _ in arms) + " |",
          "|---|---:|" + "---|" * len(arms)]
    for p in arms[0][1]["per_part"]:
        cells = []
        for _, s in arms:
            q = s["per_part"].get(p, {})
            cells.append(f"{TS.fmt((q.get('detection') or {}).get('f1'), 3)} / {_cer(q.get('cer_all'))} / "
                         f"{_cer(q.get('cer_dialogue'))}")
        md.append(f"| {p} | {arms[0][1]['per_part'][p]['frames']} | " + " | ".join(cells) + " |")
    md += ["", "| arm | text-free frames with a box | boxes per text-free frame | dialogue pick hits | pick CER |",
           "|---|---:|---:|---:|---:|"]
    for a, s in arms:
        tf, pk = s["text_free"], s["pick"]
        md.append(f"| `{a}` | {tf['with_any_box']} / {tf['frames']} | {_f(tf['boxes_per_frame'], 2)} | "
                  f"{pk['hits']} / {pk['frames_with_dialogue']} | {_cer(pk.get('cer'))} |")
    return "\n".join(md)


def seeds_md(doc):
    md = ["| seed | decode | VTSP B | est. ms @333 | arena B | C = twin | F1 | CER all | dialogue | UI recall | "
          "eligible |", "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|"]
    for s in sorted(doc["seeds"], key=lambda x: (x["arm_id"], x["decode"], x["seed"])):
        md.append(f"| `{s['name']}` | {s['decode']} | {s['blob_bytes']:,} | {_f(s['est_ms_333_960'], 1)} | "
                  f"{_n(s['arena_960'])} | {s['c_equal']} | {_f(s['f1'], 3)} | {_f(s['cer_all'])} | "
                  f"{_f(s['cer_dialogue'])} | {_f(s['ui_recall'], 3)} | {s['eligible']} |")
    return "\n".join(md)


def report_sl(doc, units_dirs, names=None):
    """{arm: {model: summary}} of the report recognizers (VOCR5-S, VOCR5-L) of the pick (or names), from the unit
    directories (textspot3.py read-pick); each file must score the boxes the record scored."""
    names = names or ([doc["result"]["winner"]] if doc["result"]["winner"] else [])
    seeds = {s["name"]: s for s in doc["seeds"]}
    out = {}
    for n in dict.fromkeys(names):
        for d in units_dirs or []:
            for m in ("VOCR5-S", "VOCR5-L"):
                f = Path(d) / n / f"units_{m}.json"
                if f.exists():
                    u = json.loads(f.read_text(encoding="utf-8"))
                    if u["meta"]["boxes_sha256"] != seeds[n]["boxes_sha256"]:
                        raise SystemExit(f"{f}: the units of other boxes than the record's")
                    out.setdefault(n, {})[m] = u["summary"]
    return out


def sl_md(doc, sl):
    if not doc["result"]["winner"]:
        return "No pick: no VOCR5-S / VOCR5-L report reading."
    if not sl:
        return "No VOCR5-S / VOCR5-L units of the pick yet (textspot3.py read-pick --names RUN --models S,L)."
    seeds = {s["name"]: s for s in doc["seeds"]}
    md = ["| pick | recognizer | F1 | CER all | dialogue |", "|---|---|---:|---:|---:|"]
    for n, by in sl.items():
        md.append(f"| `{n}` | VOCR5-N | {_f(seeds[n]['f1'], 3)} | {_f(seeds[n]['cer_all'])} | "
                  f"{_f(seeds[n]['cer_dialogue'])} |")
        for m, s in sorted(by.items()):
            md.append(f"| `{n}` | {m} | {TS.fmt(s['detection']['f1'], 3)} | {_cer(s['cer_all'])} | "
                      f"{_cer(s['cer_dialogue'])} |")
    return "\n".join(md)


def render(doc, part_arm=None, sl=None):
    return "\n".join(["### The selection", "", selection_md(doc), "",
                      "### Candidates (median seeds, cost order)", "", candidates_md(doc), "",
                      "### Pareto front", "", pareto_md(doc), "",
                      "### Recall per class", "", recall_md(doc), "",
                      "### What each arm bought", "", bought_md(doc), "",
                      "### Per part, text-free frames and the dialogue pick", "", parts_md(doc, part_arm), "",
                      "### The pick read by VOCR5-S and VOCR5-L (report)", "", sl_md(doc, sl), "",
                      "### Every seed", "", seeds_md(doc), ""])


def cmd_tables(args):
    doc = load_json(args.dev)
    text = render(doc, args.part_arm or None, report_sl(doc, args.units))
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


# ----------------------------------------------------------------------------- main

def cmd_cbuild(args):
    rec = cbuild3(args.out, args.compiler)
    ref = T2 / "eval" / "cbuild" / "build.json"
    same = {}
    if ref.exists():
        old = json.loads(ref.read_text(encoding="utf-8"))["binaries"]
        same = {k: v == old.get(k) for k, v in rec["binaries"].items() if k in old}
    print(json.dumps({"binaries": rec["binaries"], "driver2": rec["driver2"], "driver3": rec["driver3"],
                      "same_as_textspot2": same}, indent=1))


def cmd_seed_cache(args):
    print(json.dumps(seed_cache(args.cache), indent=1))


def run_parser(sub, name):
    p = sub.add_parser(name)
    p.add_argument("--set", default="dev", choices=("dev", "pseudo"))
    p.add_argument("--manifest", default="")
    p.add_argument("--out", default=str(DEFAULT["units"]))
    p.add_argument("--cbin", default=str(DEFAULT["cbuild"]), help="a cbuild directory (textspot3.py cbuild), "
                                                                  "'auto', or '' for none")
    p.add_argument("--models", default="N" if name == "run" else "S,L",
                   help="horizontal recognizers (N,S,L; none: detection only)")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--shots", action="append", default=[], help="more source screenshot directories")
    p.add_argument("--cache", default="")
    p.add_argument("--rbuild", default=str(DEFAULT["rbuild"]))
    p.add_argument("--frames", default="", help="the frames' hashes file (dev default: T3/eval/frames.json)")
    p.add_argument("--compiler", default="gcc")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--runs", default=str(DEFAULT["runs"]))
    p.add_argument("--names", nargs="*" if name == "run" else "+", default=None, required=name != "run")
    if name == "run":
        p.add_argument("--force", action="store_true")
    else:
        p.add_argument("--dev", default="", help="the selection record: the runs must be its pick's")
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cbuild")
    p.add_argument("--out", default=str(DEFAULT["cbuild"]))
    p.add_argument("--compiler", default="gcc")
    p = sub.add_parser("seed-cache")
    p.add_argument("--cache", default=str(DEFAULT["cache"]))
    run_parser(sub, "run")
    run_parser(sub, "read-pick")
    for name in ("headline", "select"):
        p = sub.add_parser(name)
        p.add_argument("--units", action="append", default=None)
        p.add_argument("--a9", action="append", required=name == "select", default=[],
                       help="a pinned A9 table (eval/textspot3_a9_*.json); repeat for T's and M's, first match wins")
        p.add_argument("--cbin", default="", help="a cbuild directory (the refinement's arena from its driver)")
        p.add_argument("--runs", default=str(DEFAULT["runs"]))
        p.add_argument("--out", required=name == "select", default="")
        if name == "select":
            p.add_argument("--baseline", default=str(BASELINE))
            p.add_argument("--indent-copy", default="")
        else:
            p.add_argument("--stage1", default="", help="a JSON list of stage 1's run names (default: the "
                                                        "protocol's, from the registered shapes)")
    p = sub.add_parser("tables")
    p.add_argument("--dev", required=True)
    p.add_argument("--units", action="append", default=[], help="unit directories with the pick's S / L units")
    p.add_argument("--part-arm", default="")
    p.add_argument("--out", default="")
    args = ap.parse_args(argv)
    if getattr(args, "units", "x") is None:
        args.units = [str(DEFAULT["units"])]
    {"cbuild": cmd_cbuild, "seed-cache": cmd_seed_cache, "run": cmd_run, "read-pick": cmd_read_pick,
     "headline": cmd_headline, "select": cmd_select, "tables": cmd_tables}[args.cmd](args)


if __name__ == "__main__":
    main()
