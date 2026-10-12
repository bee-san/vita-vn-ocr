#!/usr/bin/env python3
"""Lock and score one horizontal-padding candidate without changing the model gate.

lock reads manifest identities but opens no pixels. run requires the locked clean
commit and scores both C pipelines once. dev is restricted to a local dev manifest.
Reports contain aggregates and hashes, never screenshot pixels or transcriptions.
"""
import argparse
import ctypes as C
import datetime
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "runtime"), str(ROOT / "synth"), str(ROOT / "eval")]
import finders  # noqa: E402
import metrics  # noqa: E402
import sets  # noqa: E402
from pyvocr import Vocr, load_lib  # noqa: E402

BASELINE = "d6ef77d"
BLOB_SHA = "500e0b842e5f0cb326f55e7d991a52ab66dd24977ccc89bd188228aedc825e84"
SOURCES = ("runtime/vocr.c", "runtime/vocr.h", "runtime/vocr_lines.c", "runtime/vocr_lines.h",
           "runtime/pyvocr.py", "synth/common.py", "eval/padding_protocol.py", "eval/finders.py",
           "eval/metrics.py", "eval/sets.py", "eval/fetch.py")
RULE = {"horizontal_extra_pixels_each_side": 4, "vertical_extra_pixels": 0,
        "crop_boundary": "user-selected dialogue ROI", "polarity": "C recomputed on emitted crop",
        "primary_matching": "original baseline boxes", "secondary_matching": "expanded candidate boxes",
        "require": "full errors <= baseline on all three sets and strict CLANNAD hiragana reduction, "
                   "under both matching methods; detection and vertical geometry unchanged",
        "candidate_count": 1, "model_gate": "hiragana_gate.py unchanged; recognizer reports retain old padding"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def source_at(revision, file):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", f"{revision}:{file}"])


def committed_sources(revision):
    hashes = {}
    for path in SOURCES:
        try:
            value = hashlib.sha256(source_at(revision, path)).hexdigest()
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"source is absent from locked commit: {path}") from exc
        if value != sha(ROOT / path):
            raise ValueError(f"source differs from locked commit: {path}")
        hashes[path] = value
    return hashes


def read_rows(path, role, local):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("manifest needs unique nonempty line ids")
    if local and any(row.get("split") != role or row.get("hand_checked") is not True
                     or not row.get("session") for row in rows):
        raise ValueError(f"local captures must be hand-checked {role} sessions, before pixels are opened")
    if any(not row.get("region") or not row.get("box") for row in rows):
        raise ValueError("every manifest line needs dialogue ROI and ink geometry")
    return rows


def inputs(clannad, data, role="test"):
    result = {}
    names = ("vita-test", "psp-test", "clannad-test") if role == "test" else ("clannad-dev",)
    for name in names:
        local = name.startswith("clannad-")
        path = Path(clannad if local else sets.MANIFESTS / f"{name}.jsonl").resolve()
        rows = read_rows(path, role, local)
        result[name] = {"manifest": str(path), "sha256": sha(path), "split": role, "local": local,
                        "data": str(Path(data).resolve()) if data else None,
                        "rows": len(rows), "scored_lines": sum(not r.get("uncertain") for r in rows),
                        "reference_sha256": digest([(r["id"], r["text"], bool(r.get("uncertain"))) for r in rows])}
    return result


def write_new(out, value):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def lock(args):
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("commit the candidate before locking it")
    if sha(args.blob) != BLOB_SHA:
        raise ValueError("padding experiment requires the unchanged released FL10 blob")
    candidate = git("rev-parse", "HEAD")
    candidate_sources = committed_sources(candidate)
    baseline = git("rev-parse", BASELINE)
    for path in ("runtime/vocr.c", "runtime/vocr.h"):
        if hashlib.sha256(source_at(baseline, path)).hexdigest() != sha(ROOT / path):
            raise ValueError("recognizer source changed; this protocol isolates crop padding")
    record = {"schema": 1, "purpose": "one_candidate_preprocessing_holdout_lock",
              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "baseline_revision": baseline, "candidate_revision": candidate,
              "artifact_dir": str(args.out.resolve().parent),
              "blob": str(args.blob.resolve()), "blob_sha256": sha(args.blob), "rule": RULE,
              "source_sha256": candidate_sources,
              "baseline_source_sha256": {path: hashlib.sha256(source_at(baseline, path)).hexdigest()
                                         for path in ("runtime/vocr_lines.c", "runtime/vocr_lines.h")},
              "datasets": inputs(args.clannad_manifest, args.data), "heldout_evaluated": False}
    write_new(args.out, record)
    print(f"Locked {record['candidate_revision']}; {args.out} SHA256 {sha(args.out)}")


class LinesImage(C.Structure):
    _fields_ = [("pixels", C.c_void_p), ("width", C.c_uint), ("height", C.c_uint),
                ("stride", C.c_size_t), ("bpp", C.c_uint)]


class Line(C.Structure):
    _fields_ = [(key, C.c_uint) for key in ("x", "y", "w", "h", "dark")]


class LineSet(C.Structure):
    _fields_ = [("lines", Line * 16), ("count", C.c_uint), ("truncated", C.c_uint)]


class Backend:
    def __init__(self, directory, blob, baseline=None):
        directory.mkdir(parents=True, exist_ok=True)
        if baseline:
            for file in ("vocr_lines.c", "vocr_lines.h"):
                (directory / file).write_bytes(source_at(baseline, f"runtime/{file}"))
            source = directory / "vocr_lines.c"
        else:
            source = ROOT / "runtime/vocr_lines.c"
        self.binary = directory / "libvocr.so"
        self.command = ["gcc", "-std=c99", "-ffp-contract=off", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-fPIC", "-shared", "-o", str(self.binary), str(ROOT / "runtime/vocr.c"),
                        str(source), "-lm"]
        subprocess.run(self.command, check=True)
        self.lib = load_lib(self.binary)
        self.lib.vocr_lines_arena_bytes.argtypes = [C.c_uint, C.c_uint]
        self.lib.vocr_lines_arena_bytes.restype = C.c_size_t
        self.lib.vocr_find_lines.argtypes = [C.POINTER(LinesImage), C.c_int, C.c_void_p,
                                           C.c_size_t, C.POINTER(LineSet)]
        self.lib.vocr_line_dark.argtypes = [C.POINTER(LinesImage), *([C.c_uint] * 4)]
        self.runtime = Vocr(blob, mode="w8i", max_width=1024, lib=self.lib)
        self.max_finder_arena_bytes = 0
        if self.runtime.blob_bytes != 657664 or self.runtime.arena_bytes != 124832:
            raise ValueError("FL10 memory budget changed")

    def find_read(self, rgb):
        h, w = rgb.shape[:2]
        rgba = np.concatenate([rgb, np.full((h, w, 1), 255, np.uint8)], axis=2)
        image = LinesImage(rgba.ctypes.data, w, h, rgba.strides[0], 4)
        needed = self.lib.vocr_lines_arena_bytes(w, h)
        self.max_finder_arena_bytes = max(self.max_finder_arena_bytes, needed)
        arena, found = C.create_string_buffer(needed), LineSet()
        rc = self.lib.vocr_find_lines(C.byref(image), 0, arena, needed, C.byref(found))
        if rc or found.truncated:
            raise ValueError(f"unexpected C finder rejection/truncation:{rc}/{found.truncated}")
        boxes, predictions, dark = [], [], []
        for line in found.lines[:found.count]:
            x, y, cw, ch = line.x, line.y, line.w, line.h
            if x + cw > w or y + ch > h or not cw or not ch:
                raise ValueError("finder crop escaped its selected ROI")
            polarity = self.lib.vocr_line_dark(C.byref(image), x, y, cw, ch)
            if polarity != line.dark or polarity not in (0, 1):
                raise ValueError("polarity was not recomputed on emitted pixels")
            boxes.append((x, y, cw, ch))
            dark.append(polarity)
            predictions.append(self.runtime.run(rgba[y:y + ch, x:x + cw], invert=bool(polarity)))
        return boxes, predictions, dark


def verify_expansion(previous, candidate, width):
    expected = []
    for x, y, w, h in previous:
        left, right = max(0, x - 4), min(width, x + w + 4)
        expected.append((left, y, right - left, h))
    if candidate != expected:
        raise ValueError("candidate changed detections or vertical geometry beyond the locked expansion")


def scored(boxes, gts, predictions):
    pairs, unmatched = finders.match(boxes, gts, 0.5)
    matched = {j for _, j in pairs}
    rows = [{"text": gts[j]["text"], "pred": predictions[i]} for i, j in pairs if not gts[j].get("uncertain")]
    rows.extend({"text": row["text"], "pred": ""} for j, row in enumerate(gts)
                if j not in matched and not row.get("uncertain"))
    rows.extend({"text": "", "pred": predictions[i]} for i in unmatched)
    found = sum(not gts[j].get("uncertain") for _, j in pairs)
    missed = sum(j not in matched and not row.get("uncertain") for j, row in enumerate(gts))
    return rows, {"found_lines": found, "missed_lines": missed, "false_positive_detections": len(unmatched)}, pairs


def score_dataset(identity, baseline, candidate):
    path = Path(identity["manifest"])
    rows = read_rows(path, identity["split"], identity["local"])
    if sha(path) != identity["sha256"]:
        raise ValueError("locked manifest changed before pixels were opened")
    groups, image_hashes = {}, {}
    for row in rows:
        if identity["local"]:
            image = path.parent / row["image"]
        else:
            image = Path(sets.image_path(path.stem, identity["data"], row))
        groups.setdefault((image, tuple(row["region"])), []).append(row)
    accum = {name: {"rows": [], "found_lines": 0, "missed_lines": 0, "false_positive_detections": 0}
             for name in ("baseline", "candidate_fixed_matching", "candidate_emitted_matching")}
    geometries, changes = [], []
    detections = clamped = polarity_changes = width_clamps = baseline_width_clamps = clipped_gt = 0
    for (path, region), group in groups.items():
        image_hashes.setdefault(path, sha(path))
        if any(row["sha256"] != image_hashes[path] for row in group):
            raise ValueError("screenshot differs from its locked manifest")
        image = np.asarray(Image.open(path).convert("RGB"))
        if any(row.get("dims", [image.shape[1], image.shape[0]]) != [image.shape[1], image.shape[0]]
               for row in group):
            raise ValueError("manifest frame dimensions differ from its pixels")
        rx, ry, rw, rh = region
        if rx < 0 or ry < 0 or rw <= 0 or rh <= 0 or rx + rw > image.shape[1] or ry + rh > image.shape[0]:
            raise ValueError("dialogue ROI is out of frame bounds")
        rgb = np.ascontiguousarray(image[ry:ry + rh, rx:rx + rw])
        gts = [{**row, "box_r": (row["box"][0] - rx, row["box"][1] - ry,
                                 row["box"][2], row["box"][3])} for row in group]
        for row in group:
            gx, gy, gw, gh = row["box"]
            if gx < 0 or gy < 0 or gw <= 0 or gh <= 0 or gx + gw > image.shape[1] or gy + gh > image.shape[0]:
                raise ValueError("annotated ink lies outside its frame")
            clipped_gt += not row.get("uncertain") and (gx < rx or gy < ry or gx + gw > rx + rw or gy + gh > ry + rh)
        old_boxes, old_predictions, old_dark = baseline.find_read(rgb)
        new_boxes, new_predictions, new_dark = candidate.find_read(rgb)
        verify_expansion(old_boxes, new_boxes, rw)
        if new_boxes != finders.console(rgb):
            raise ValueError("candidate C and Python console twin differ")
        detections += len(old_boxes)
        for (x, _, w, h), (nx, _, nw, nh), od, nd in zip(old_boxes, new_boxes, old_dark, new_dark):
            clamped += nx != x - 4 or nx + nw != x + w + 4
            polarity_changes += od != nd
            width_clamps += round(nw * 32 / nh) > 1024
            baseline_width_clamps += round(w * 32 / h) > 1024
        pairs = {}
        for name, boxes, predictions in [("baseline", old_boxes, old_predictions),
                                        ("candidate_fixed_matching", old_boxes, new_predictions),
                                        ("candidate_emitted_matching", new_boxes, new_predictions)]:
            scored_rows, counts, pairs[name] = scored(boxes, gts, predictions)
            accum[name]["rows"].extend(scored_rows)
            for key, value in counts.items():
                accum[name][key] += value
        changed = sorted(pairs["candidate_fixed_matching"]) != sorted(pairs["candidate_emitted_matching"])
        if changed:
            changes.append({"image_sha256": image_hashes[path], "region": list(region),
                            "original_pairs": pairs["candidate_fixed_matching"],
                            "expanded_pairs": pairs["candidate_emitted_matching"]})
        geometries.append([image_hashes[path], region, old_boxes, new_boxes, old_dark, new_dark])
    result = {}
    for name, aggregate in accum.items():
        scores = metrics.score_lines(aggregate.pop("rows"))
        result[name] = {"full": scores["all"], "hiragana": scores["hiragana"], **aggregate}
    return {"scores": result, "regions": len(groups), "detections": detections,
            "horizontal_padding_clamped_crops": clamped, "polarity_changes": polarity_changes,
            "resized_width_clamped_candidate_crops": width_clamps, "geometry_sha256": digest(geometries),
            "resized_width_clamped_baseline_crops": baseline_width_clamps,
            "scored_gt_lines_clipped_by_selected_roi": clipped_gt,
            "image_sha256": sorted(set(image_hashes.values())), "matching_changed_regions": len(changes),
            "matching_changes": changes}


def gate(results):
    if set(results) != {"vita-test", "psp-test", "clannad-test"}:
        raise ValueError("all three held-out sets are required")
    failures = []
    for name, result in results.items():
        controls = result["scores"]
        for mode in ("candidate_fixed_matching", "candidate_emitted_matching"):
            if controls[mode]["full"]["chars"] != controls["baseline"]["full"]["chars"]:
                raise ValueError("scored reference character counts differ")
            if controls[mode]["full"]["errors"] > controls["baseline"]["full"]["errors"]:
                failures.append(f"{name}: full error regression ({mode})")
            if name == "clannad-test":
                a, b = controls["baseline"]["hiragana"], controls[mode]["hiragana"]
                if not a["chars"] or a["chars"] != b["chars"]:
                    raise ValueError("CLANNAD hiragana reference counts differ")
                if b["errors"] >= a["errors"]:
                    failures.append(f"CLANNAD hiragana does not strictly improve ({mode})")
    return {"accepted": not failures, "failures": failures, "physical_console_validation": "pending"}


def evaluate(args):
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    if args.command == "run":
        lock_hash = sha(args.lock)
        if lock_hash != args.lock_sha256:
            raise ValueError("lock differs from the preregistered SHA256")
        record = json.loads(args.lock.read_text())
        if (record.get("schema") != 1 or record.get("rule") != RULE or record.get("heldout_evaluated") is not False
                or record.get("purpose") != "one_candidate_preprocessing_holdout_lock"):
            raise ValueError("invalid preprocessing lock")
        if git("rev-parse", "HEAD") != record["candidate_revision"] or git(
                "status", "--porcelain", "--untracked-files=no"):
            raise ValueError("candidate must be the locked clean commit")
        if record["source_sha256"] != committed_sources(record["candidate_revision"]):
            raise ValueError("candidate source hashes differ from lock")
        blob, baseline, datasets = Path(record["blob"]), record["baseline_revision"], record["datasets"]
        expected_baseline = git("rev-parse", BASELINE)
        expected_sources = {path: hashlib.sha256(source_at(expected_baseline, path)).hexdigest()
                            for path in ("runtime/vocr_lines.c", "runtime/vocr_lines.h")}
        if baseline != expected_baseline or record["baseline_source_sha256"] != expected_sources:
            raise ValueError("baseline source differs from the fixed control")
        if any(hashlib.sha256(source_at(baseline, path)).hexdigest() != sha(ROOT / path)
               for path in ("runtime/vocr.c", "runtime/vocr.h")):
            raise ValueError("recognizer source changed from the fixed control")
        if set(datasets) != {"vita-test", "psp-test", "clannad-test"}:
            raise ValueError("lock requires all three held-out dataset identities")
        expected_data = inputs(datasets["clannad-test"]["manifest"], datasets["vita-test"]["data"])
        if datasets != expected_data:
            raise ValueError("locked dataset roles, hashes or identities differ before pixels are opened")
        if record["artifact_dir"] != str(args.lock.resolve().parent) or not out.is_relative_to(record["artifact_dir"]):
            raise ValueError("lock and results must remain in the registered artifact directory")
        if sha(blob) != record["blob_sha256"] or sha(blob) != BLOB_SHA:
            raise ValueError("released blob differs from lock")
        marker = Path(record["artifact_dir"]) / f"{lock_hash}.used.json"
    else:
        blob, baseline = args.blob, git("rev-parse", BASELINE)
        if sha(blob) != BLOB_SHA:
            raise ValueError("unchanged released FL10 blob required")
        datasets, lock_hash = inputs(args.manifest, None, role="dev"), None
        marker = out.with_suffix(".started.json")
    # Persist the start before opening pixels; completed or interrupted runs cannot
    # silently be repeated against the same output path.
    write_new(marker, {"lock_sha256": lock_hash, "rule": RULE})
    native = out.parent / (out.stem + "-native")
    old, new = Backend(native / "baseline", blob, baseline), Backend(native / "candidate", blob)
    results = {name: score_dataset(identity, old, new) for name, identity in datasets.items()}
    result = {"schema": 1, "purpose": "development_only" if args.command == "dev" else "locked_holdout_once",
              "lock_sha256": lock_hash, "candidate_revision": git("rev-parse", "HEAD"),
              "candidate_source_sha256": {path: sha(ROOT / path) for path in SOURCES}, "rule": RULE,
              "baseline_revision": baseline, "blob_sha256": sha(blob), "datasets": datasets, "sets": results,
              "backend": "C GRAD finder and scalar C W8I recognizer", "blob_bytes": new.runtime.blob_bytes,
              "arena_bytes": new.runtime.arena_bytes, "max_width": 1024,
              "max_finder_arena_bytes": {"baseline": old.max_finder_arena_bytes,
                                         "candidate": new.max_finder_arena_bytes},
              "host": {"platform": platform.platform(), "python": platform.python_version(),
                       "compiler": subprocess.check_output(["gcc", "--version"], text=True).splitlines()[0]},
              "compile_commands": [old.command, new.command],
              "library_sha256": {"baseline": sha(old.binary), "candidate": sha(new.binary)},
              "heldout_evaluated": args.command == "run"}
    if args.command == "run":
        result["gate"] = gate(results)
    write_new(out, result)
    print(json.dumps({name: {"scores": data["scores"], "matching_changed_regions": data["matching_changed_regions"]}
                      for name, data in results.items()}, indent=2))
    if args.command == "run" and not result["gate"]["accepted"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("lock")
    prepare.add_argument("--blob", type=Path, required=True)
    prepare.add_argument("--clannad-manifest", type=Path, required=True)
    prepare.add_argument("--data", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--lock", type=Path, required=True)
    run.add_argument("--lock-sha256", required=True, help="exact digest emitted by lock, fixed before scoring")
    run.add_argument("--out", type=Path, required=True)
    dev = sub.add_parser("dev")
    dev.add_argument("--manifest", type=Path, required=True)
    dev.add_argument("--blob", type=Path, required=True)
    dev.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lock(args) if args.command == "lock" else evaluate(args)


if __name__ == "__main__":
    main()
