#!/usr/bin/env python3
"""realplus/native.py: more native PS Vita / PSP training lines (owner request, 2026-10-13).

Native console frames are scarce: most real training lines are PC screenshots scaled down. This
tool lists every VNDB release for PSP or PS Vita and its screenshots, finds the native frames the
training side may still use, and prepares them for realplus's labelling pipeline (devnative's
reads, Claude + Codex gpt-6-luna agreement as realplus v3 runs it), in a separate pool directory
$NATIVE_POOL (default $HERD/shared/artifacts/native-frames/pool; pool.py runs on it with
REALPLUS=$NATIVE_POOL).

  native.py enumerate [--dump DB]   every psp/psv release and screenshot of the dump, each dropped
                                    screenshot's reason -> $NATIVE_POOL/manifests/enumeration.json
  native.py select                  the frames to label: new native VNDB frames (none on 2026-10-13)
                                    and training-side native frames already on disk in realdata's /
                                    teacher's pools whose lines those pools left unlabelled
                                    -> $NATIVE_POOL/manifests/shots.jsonl
  native.py prep                    read.py's input (native kinds, read at native size)
                                    -> $NATIVE_POOL/reads/shots.jsonl
  native.py known                   the lines teacher / realdata already labelled on those frames
                                    -> $NATIVE_POOL/manifests/known.json
  native.py filter-plan             drop known lines from pool.py plan's review order

A frame is a native frame when it belongs to a psv release at 960x544 or a psp release at 480x272
(realdata's kinds). A VN is used only on the training side (realplus/shots.reserved_vns, docs/
reservations.md: not a vita-test / psp-test game, not the CLANNAD family, no release partner of those, not
in the development sample); a frame only when it is not a vita-test / psp-test screenshot and not from a
test capture session. Frames of another labelled pool (devnative, general-VN development) are left to
that pool's labels ("pooled-other"); eval's unchosen candidates and linefind's sets are training-side.
The order is console exclusives first: titles released only for PSP, then only for PS Vita, then for
both, then multi-platform titles.
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import shots as SH  # noqa: E402

ART = SH.ART
NP = Path(os.environ.get("NATIVE_POOL") or ART / "native-frames" / "pool")
DUMP = Path(os.environ.get("NATIVE_DUMP") or ART / "native-frames" / "vndb" / "dump" / "db")
NATIVE = {"psv": ("vita", (960, 544)), "psp": ("psp", (480, 272))}
CONSOLES = {"psp", "psv"}
# Screenshot sets a training pool must never take a frame from (reserved_screenshots' names): only the test sets'
# screenshots since 2026-10-14 (docs/reservations.md).
FORBIDDEN_SETS = {"eval-test"}
# Sets whose frames this pool may relabel: realdata's and teacher's training-side native frames.
POOLED_SETS = {"realdata", "teacher"}
GROUP_ORDER = ("psp-only", "vita-only", "psp+vita", "multi-platform")


def jl(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def wjl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(p)


def reserved_today():
    """reserved_vns of the dump, by today's rules. In an owner pool (NATIVE_OWNER_POOL=1: the owner's own
    console captures, shared/private/owner-games/_build/...) a VN the owner explicitly made training data
    (owner_frames.OWNER_DECISIONS, only for the reservation reasons listed there) is not held back, and neither is
    a development-sample VN: every owner screenshot trains unless its VN is a test VN (owner 2026-10-14 13:25Z,
    docs/reservations.md), and shots.py reserve leaves owner games out of the sample."""
    r = SH.reserved_vns(SH.load_dump(DUMP))
    if os.environ.get("NATIVE_OWNER_POOL") == "1":
        import owner_frames as OF
        r = {v: w for v, w in r.items() if not OF.owner_allowed(v, w) and not w.startswith("dev-sample")}
    return r


def native_kind(platforms, w, h):
    """'vita' or 'psp' for a native frame of a release on these platforms, else None."""
    for plat, (kind, dims) in NATIVE.items():
        if plat in platforms and (w, h) == dims:
            return kind
    return None


def title_groups(d):
    """{vn: group} over every VN with a psp or psv release: psp-only / vita-only / psp+vita (all its
    releases are for those consoles only) or multi-platform."""
    plats = collections.defaultdict(set)
    for rid, vs in d["rel_vns"].items():
        for v in vs:
            plats[v] |= d["plat"].get(rid, set())
    out = {}
    for v, ps in plats.items():
        if not ps & CONSOLES:
            continue
        if ps <= {"psp"}:
            out[v] = "psp-only"
        elif ps <= {"psv"}:
            out[v] = "vita-only"
        elif ps <= CONSOLES:
            out[v] = "psp+vita"
        else:
            out[v] = "multi-platform"
    return out


def classify(d, reserved_vn, reserved_img, sessions):
    """[row] for every screenshot of a psp/psv release, each with its first exclusion reason or
    'new' / 'pooled' (a training-side native frame already in realdata / teacher only)."""
    groups = title_groups(d)
    out = []
    for vn, image, rid in d["shots"]:
        plats = d["plat"].get(rid, set()) if rid else set()
        if not plats & CONSOLES:
            continue
        w, h, sexual, violence = d["img"].get(image, (0, 0, 9.0, 9.0))
        kind = native_kind(plats, w, h)
        sets = set(reserved_img.get(image, ()))
        r = {"image": image, "vn": vn, "release": rid, "platforms": sorted(plats), "dims": [w, h], "kind": kind,
             "group": groups.get(vn, "multi-platform"), "sets": sorted(sets)}
        if kind is None:
            why = "not-native-size"
        elif vn in reserved_vn:
            why = "vn:" + reserved_vn[vn]
        elif rid in sessions:
            why = "test-session"
        elif sets & FORBIDDEN_SETS:
            why = "set:" + "+".join(sorted(sets & FORBIDDEN_SETS))
        elif sexual >= SH.SEXUAL_MAX or violence >= SH.VIOLENCE_MAX:
            why = "rating"
        elif not SH.langs_ok(d["langs"].get(rid)):
            why = "language"
        elif d["olang"].get(vn) != "ja":
            why = "not-japanese-original"
        elif sets & POOLED_SETS:
            why = "pooled"
        elif sets:
            why = "pooled-other:" + "+".join(sorted(sets))
        else:
            why = "new"
        r["why"] = why
        r["url"] = SH.url_of(image)
        out.append(r)
    return out


def summary(rows, d):
    rel = {rid for rid, p in d["plat"].items() if p & CONSOLES}
    by_plat = collections.Counter("+".join(sorted(p & CONSOLES)) for rid, p in d["plat"].items() if p & CONSOLES)
    only = collections.Counter("console-only" if p <= CONSOLES else "with-other" for rid, p in d["plat"].items()
                               if p & CONSOLES)
    groups = title_groups(d)
    out = {
        "releases": {"psp_or_psv": len(rel), "by_console": dict(by_plat), "console_only_releases": dict(only)},
        "titles": {"with_psp_or_psv_release": len(groups), "by_group": dict(collections.Counter(groups.values()))},
        "screenshots": {"on_psp_or_psv_releases": len(rows),
                        "vns": len({r["vn"] for r in rows}),
                        "native": sum(r["kind"] is not None for r in rows),
                        "native_by_kind": dict(collections.Counter(r["kind"] for r in rows if r["kind"])),
                        "dims_not_native": dict(collections.Counter(
                            f"{'+'.join(sorted(set(r['platforms']) & CONSOLES))} {r['dims'][0]}x{r['dims'][1]}"
                            for r in rows if r["kind"] is None).most_common(20))},
        "why": dict(collections.Counter(r["why"] for r in rows).most_common()),
        "why_by_group": {g: dict(collections.Counter(r["why"] for r in rows if r["group"] == g).most_common())
                         for g in GROUP_ORDER},
        "new": {"screenshots": sum(r["why"] == "new" for r in rows),
                "vns": len({r["vn"] for r in rows if r["why"] == "new"})},
        "pooled": {"screenshots": sum(r["why"] == "pooled" for r in rows),
                   "vns": len({r["vn"] for r in rows if r["why"] == "pooled"})},
        "reserved_vn_reasons": dict(collections.Counter(r["why"][3:] for r in rows if r["why"].startswith("vn:"))),
    }
    return out


def cmd_enumerate(args):
    d = SH.load_dump(args.dump)
    tg = SH.test_games()
    rvn = SH.reserved_vns(d)
    rows = classify(d, rvn, SH.reserved_screenshots(), set(tg["sessions"]))
    stamp = args.dump.parent / "TIMESTAMP"
    doc = {"kind": "native-enumeration", "dump": stamp.read_text().strip() if stamp.exists() else None,
           **summary(rows, d)}
    (NP / "manifests").mkdir(parents=True, exist_ok=True)
    (NP / "manifests" / "enumeration.json").write_text(json.dumps(doc, indent=1) + "\n")
    wjl(NP / "manifests" / "enumeration_rows.jsonl", rows)
    print(json.dumps(doc, indent=1), flush=True)


# ----------------------------------------------------------------------------- select

def select_rows(rows, pooled_files, order_seed=20261013):
    """The frames to label: every 'new' native frame and every 'pooled' one with a local copy, in
    title-group order (PSP-only first), then round-robin over VNs."""
    keep = []
    for r in rows:
        key = f"{r['vn']}_{r['image']}"
        if r["why"] == "new":
            keep.append(dict(r, shot=key, source="vndb-new", path=None))
        elif (r["why"] == "pooled" or r["why"].startswith("pooled-other:")) and key in pooled_files:
            keep.append(dict(r, shot=key, source="+".join(r["sets"]), path=str(pooled_files[key])))
    rank = collections.defaultdict(int)
    for r in sorted(keep, key=lambda r: (r["vn"], SH.stable_rank(order_seed, r["image"]))):
        r["rank"] = rank[r["vn"]]
        rank[r["vn"]] += 1
    g = {k: i for i, k in enumerate(GROUP_ORDER)}
    keep.sort(key=lambda r: (g[r["group"]], r["rank"], SH.stable_rank(order_seed, r["vn"]), r["image"]))
    SH.assign_sessions(keep)
    for i, r in enumerate(keep):
        r["order"] = i
    return keep


def local_copies():
    """{shot key: path} of every local copy of a VNDB screenshot in realdata's shots and the scene
    directories (synth2's PSP shots, shared/data/vndb/shots), first found wins."""
    out = {}
    dirs = [ART / "realdata" / "shots" / k for k in ("vita", "psp")]
    dirs += [ART / "synth2" / "vndb_psp", SH.HERD / "shared" / "data" / "vndb" / "shots"]
    for d in dirs:
        if d.exists():
            for p in sorted(d.rglob("*")):
                m = SH.SHOT_FILE.match(p.name)
                if m and p.is_file():
                    out.setdefault(p.stem, p)
    return out


def cmd_select(args):
    rows = jl(NP / "manifests" / "enumeration_rows.jsonl")
    if not rows:
        sys.exit("run enumerate first")
    keep = select_rows(rows, local_copies())
    wjl(NP / "manifests" / "shots.jsonl", keep)
    c = collections.Counter((r["source"], r["kind"]) for r in keep)
    print(json.dumps({"frames": len(keep), "vns": len({r['vn'] for r in keep}),
                      "by_source_kind": {f"{a}/{b}": n for (a, b), n in sorted(c.items())},
                      "by_group": dict(collections.Counter(r["group"] for r in keep))}, indent=1), flush=True)


def cmd_prep(args):
    rows = jl(NP / "manifests" / "shots.jsonl")
    dups = json.loads((NP / "manifests" / "dups.json").read_text()) if (NP / "manifests" / "dups.json").exists() \
        else {}
    out = []
    for r in rows:
        if not r.get("path") or str(r["image"]) in dups:
            continue
        out.append({"shot": r["shot"], "vn": r["vn"], "kind": r["kind"], "url": r["url"], "path": r["path"],
                    "release": r["release"], "session": r["session"], "subset": "native", "flags": [],
                    "source_kind": r["kind"], "dims": r["dims"]})
    wjl(NP / "reads" / "shots.jsonl", out)
    print(f"{len(out)} frames to read (of {len(rows)} selected; {len(dups)} duplicates)", flush=True)


# ----------------------------------------------------------------------------- known lines

def known_boxes(shots):
    """{shot key: [[x0, y0, x1, y1], ...]} of every line teacher or realdata accepted (any split) on
    these frames, in native pixels (teacher's crop box, realdata's ink box)."""
    out = collections.defaultdict(list)
    for kind in ("vita", "psp"):
        for r in jl(ROOT / "teacher" / "manifests" / f"lines_{kind}.jsonl"):
            sk = r["id"].rsplit("_", 1)[0]
            if sk in shots:
                out[sk].append(list(r["crop"]))
        for r in jl(ROOT / "realdata" / "manifests" / f"lines_{kind}.jsonl"):
            sk = r["id"].rsplit("_", 1)[0]
            if sk in shots:
                out[sk].append(list(r["ink"]))
    return dict(out)


def overlaps(a, b, frac=0.5):
    """Same text line: vertical overlap >= frac of the smaller height, and horizontal overlap."""
    v = min(a[3], b[3]) - max(a[1], b[1])
    h = min(a[2], b[2]) - max(a[0], b[0])
    return h > 0 and v >= frac * max(1, min(a[3] - a[1], b[3] - b[1]))


def is_known(line, boxes):
    ink = line.get("ink")
    return bool(ink) and any(overlaps(ink, b) for b in boxes)


def cmd_known(args):
    import pool as PL
    shots = set(PL.load_reads())          # every frame read, also one a later rule took out of the selection
    kb = known_boxes(shots)
    (NP / "manifests" / "known.json").write_text(json.dumps(kb) + "\n")
    print(f"{sum(len(v) for v in kb.values())} labelled lines on {len(kb)} of {len(shots)} frames", flush=True)


def cmd_filter_plan(args):
    """Rewrite review/plan.jsonl without the lines teacher / realdata already labelled (kept as
    review/plan_all.jsonl), so review effort goes to new lines only."""
    import pool as PL
    kb = json.loads((NP / "manifests" / "known.json").read_text())
    plan_p = NP / "review" / "plan.jsonl"
    full = jl(NP / "review" / "plan_all.jsonl") or jl(plan_p)
    wjl(NP / "review" / "plan_all.jsonl", full)
    idx = PL.S.line_index(PL.load_reads())
    reserved = reserved_today()     # today's rules: a VN reserved after its frame was read
    known = [r for r in full if r["id"] in idx and is_known(idx[r["id"]][1], kb.get(idx[r["id"]][0]["shot"], []))]
    held = [r for r in full if r["id"] in idx and idx[r["id"]][0]["vn"] in reserved]
    drop = {r["id"] for r in known} | {r["id"] for r in held}
    keep = [r for r in full if r["id"] in idx and r["id"] not in drop]
    wjl(plan_p, keep)
    print(json.dumps({"review_lines": len(full), "known_dropped": len(known), "reserved_dropped": len(held),
                      "kept": len(keep), "by_tier": dict(collections.Counter(r["tier"] for r in keep))}, indent=1),
          flush=True)


def cmd_filter_labels(args):
    """After realplus v3's pool.py final: every label of a VN reserved by today's rules (reserved_vns of the
    dump; e.g. the owner's Memories Off series, reserved after this pool's frames were read) is set to
    status "dropped" (reason reserved-vn), so no confirm, edge or check batch shows it and the build skips it."""
    import pool as PL
    reserved = reserved_today()
    idx = PL.S.line_index(PL.load_reads())
    rows, n = jl(NP / "labels.jsonl"), collections.Counter()
    for r in rows:
        vn = r.get("vn") or (idx[r["id"]][0]["vn"] if r["id"] in idx else None)
        if vn in reserved and r.get("status") != "dropped":
            n[r.get("status")] += 1
            r["status"], r["why"] = "dropped", f"reserved-vn:{reserved[vn]}"
    wjl(NP / "labels.jsonl", rows)
    print(json.dumps({"labels": len(rows), "set_dropped": dict(n)}), flush=True)


# ----------------------------------------------------------------------------- build

def cmd_edge_boxes(args):
    """realplus v3's pool.py edge-boxes for native frames: the final crop of every kept or pending label
    and whether it needs an edge check, on the frame at its own console's size (pool.py's assumes a
    PC shot read at Vita scale). Writes $NATIVE_POOL/edge/boxes.jsonl in pool.py's format, which its
    edge-batches and confirm-batches then read unchanged."""
    import pool as PL

    from common import to_gray
    from devnative import paths as P
    labels = [r for r in jl(NP / "labels.jsonl") if r["status"] in ("kept", "pending")]
    shots = PL.load_reads()
    idx = PL.S.line_index(shots)
    kb = json.loads((NP / "manifests" / "known.json").read_text())
    reserved = reserved_today()
    # lines teacher / realdata already labelled, and reserved VNs, never reach the build: no check for them
    labels = [r for r in labels if r["vn"] not in reserved and not is_known(idx[r["id"]][1], kb.get(r["shot"], []))]
    out, cache = [], {}
    for r in sorted(labels, key=lambda r: (int(r["shot"].split("_")[0][1:]), r["id"])):
        s, ln = idx[r["id"]]
        if s["shot"] not in cache:
            cache.clear()
            rgb = P.load_screen(s)
            cache[s["shot"]] = (rgb, to_gray(rgb))
        rgb, gray = cache[s["shot"]]
        H, W = rgb.shape[:2]
        fc, rc, why = PL.edge_flags(gray, ln, s["kind"], W, H, r["label"])
        out.append({"id": r["id"], "shot": s["shot"], "label": r["label"], "crop": fc, "reviewed": rc,
                    "check": bool(why), "why": why, "status": r["status"]})
    wjl(NP / "edge" / "boxes.jsonl", out)
    print(json.dumps({"lines": len(out), "pending": sum(x["status"] == "pending" for x in out),
                      "check": sum(x["check"] for x in out if x["status"] == "kept"),
                      "why": dict(collections.Counter(w for x in out for w in x["why"]))}), flush=True)


def expand_box(box, m, W, H):
    """A crop box [x0, y0, x1, y1] grown by m px on every side, clamped to the frame (the fresh-native
    harness's +4 / +8 ROI variants, applied to one line)."""
    x0, y0, x1, y1 = box
    return [max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m)]


def clamp_margin(box, m, W, H, others):
    """The crop box [x0, y0, x1, y1] grown by m px on every side and clamped to the frame, then shrunk on the
    side of every other text box that enters it, so that none enters by more than 1 px (a short last line under
    a long one keeps its margin below, not the line above). None when another box already overlaps the box
    itself."""
    x0, y0, x1, y1 = box
    g = [max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m)]
    for ox0, oy0, ox1, oy1 in others:
        if min(g[2], ox1) - max(g[0], ox0) <= 1 or min(g[3], oy1) - max(g[1], oy0) <= 1:
            continue
        if min(x1, ox1) - max(x0, ox0) > 1 and min(y1, oy1) - max(y0, oy0) > 1:
            return None
        if oy1 <= y0 + 1:
            g[1] = max(g[1], oy1 - 1)
        elif oy0 >= y1 - 1:
            g[3] = min(g[3], oy0 + 1)
        elif ox1 <= x0 + 1:
            g[0] = max(g[0], ox1 - 1)
        else:
            g[2] = min(g[2], ox0 + 1)
    return g


def cut_native(s, recs, idx, edge=None, margins=()):
    """realplus/pool.cut_shot for a native frame: the frame was read at its own size, so the only
    target is its own console (a Vita frame gives Vita crops, a PSP frame PSP crops; neither is scaled).
    Each kept line gives eval's ink crop widened to the labelled characters (lines.final_crop) and, where
    the console's GRAD finder returns a clean box inside the reviewed frame that covers it, the GRAD
    crop. A line whose crop edgecheck flags needs an edge-check decision (`edge`), as in realplus."""
    import numpy as np
    import pool as PL

    from common import to_gray
    from devnative import build as B
    from devnative import gradcheck as GC
    from devnative import lines as L
    from devnative import paths as P
    kind = s["kind"]
    rgb = np.ascontiguousarray(P.load_screen(s))
    gray = to_gray(rgb)
    H, W = rgb.shape[:2]
    recs = sorted(recs, key=lambda r: r["id"])
    lines = [PL.read_line(idx[r["id"]][1]) for r in recs]
    dets = s.get("dets", [])
    grads = L.grad_assign(rgb, lines, dets, kind)
    drops, cut = collections.Counter(), {}
    for r, ln, g in zip(recs, lines, grads):
        raw = idx[r["id"]][1]
        e = (edge or {}).get(r["id"])
        if e is not None:
            fc, flags, why = e["crop"], e.get("flags") or [], e["edge_why"]
            if not e["edge_ok"]:
                drops[why] += 1
                continue
        else:
            fc, _, flags = PL.edge_flags(gray, raw, kind, W, H, r["label"])
            why = "unflagged"
            if flags:
                drops["edge-unchecked"] += 1
                continue
        frame = L.crop_box(raw, kind, W, H)
        out = []
        for crop in ("ink", "grad"):
            if crop == "ink":
                box = list(fc)
                if PL.ink_mixed(box, ln, dets):
                    drops["ink-mixed"] += 1
                    continue
            else:
                if g is None or g["status"] != "ok":
                    drops[f"grad-{g['status'] if g else 'none'}"] += 1
                    continue
                box = g["box"]
                if GC.needs_check(box, frame, L.ink_height(ln)):
                    drops["grad-unchecked"] += 1
                    continue
                if not B.covers(box, fc, kind):
                    drops["grad-cuts-label"] += 1
                    continue
            img, dark, inverted = L.cut(rgb, box)
            if img.shape[0] < 4 or img.shape[1] < 4:
                drops[f"{crop}-tiny"] += 1
                continue
            ch, ch_ok = PL.chars_in(ln, r["label"], box)
            meta = {"real": True, "pool": "owner" if os.environ.get("NATIVE_OWNER_POOL") == "1" else "native",
                    "id": r["id"], "vn": s["vn"], "session": s.get("session"),
                    "release": s.get("release"), "kind": kind, "target": kind, "crop": crop, "box": box,
                    "dark": dark, "inverted": inverted, "pol": "dark-on-light" if dark else "bright-on-dark",
                    "how": r["how"], "support": r["support"], "readers": r["readers"],
                    "decision": (r.get("pass1") or {}).get("decision"), "looks": r.get("looks"),
                    "reviewer_kind": "assistant", "reviewer_kinds": r.get("reviewer_kinds", ["assistant"]),
                    "luna": r.get("luna"), "confirm": r.get("confirm"), "name": r.get("name", False),
                    "chars": ch, "chars_match": ch_ok, "screen": [W, H], "edge": why}
            out.append((crop, img, meta))
            if crop == "ink":
                # multi-margin variants of the ink crop: grown by m px, then clamped so that no other read line's
                # ink box or detected text box enters by more than 1 px
                others = L.others_of(ln, dets) + [list(o["ink"]) for o in s.get("lines", [])
                                                  if o.get("ink") and list(o["ink"]) != list(raw["ink"])]
                for m in margins:
                    eb = clamp_margin(box, m, W, H, others)
                    if eb is None:
                        drops[f"ink_m{m}-overlap"] += 1
                        continue
                    eimg, edark, einv = L.cut(rgb, eb)
                    out.append((f"ink_m{m}", eimg, dict(meta, crop=f"ink_m{m}", box=eb, margin=m, dark=edark,
                                                        inverted=einv,
                                                        pol="dark-on-light" if edark else "bright-on-dark")))
        if out:
            cut[r["id"]] = out
        else:
            drops["no-crop"] += 1
    return cut, drops


def cmd_build(args):
    """labels.jsonl (realplus v3's pool.py final, run with REALPLUS=$NATIVE_POOL) -> native npz files
    $NATIVE_POOL/native_train_{ink,grad}_<tag>.npz and the line manifest; lines teacher or realdata
    already labelled, reserved VNs (recomputed from today's dump) and leak-dropped frames are left out."""
    import pool as PL

    from devnative import paths as P
    from teacher import vote
    margins = tuple(int(x) for x in args.margins.split(",") if x) if getattr(args, "margins", "") else ()
    out = {k: NP / f"native_train_{k}_{args.tag}.npz" for k in ("ink", "grad", *(f"ink_m{m}" for m in margins))}
    if any(p.exists() for p in out.values()):
        sys.exit(f"refusing to overwrite {out}: build with a new --tag")
    labels = jl(NP / "labels.jsonl")
    tiers = {r["id"]: r["tier"] for r in jl(NP / "review" / "plan_all.jsonl")}
    shots = PL.load_reads()
    idx = PL.S.line_index(shots)
    kb = json.loads((NP / "manifests" / "known.json").read_text())
    reserved = reserved_today() if args.dump == DUMP else SH.reserved_vns(SH.load_dump(args.dump))
    leak = set(json.loads((NP / "manifests" / "leak_drop.json").read_text()))
    # third looks: lines the precision check disagreed on are dropped (review/resolutions.jsonl, {id, why})
    resolved = {r["id"]: r.get("why", "resolution") for r in jl(NP / "review" / "resolutions.jsonl")}
    vocab = PL.load_vocab()
    edge_p = NP / "edge" / "edge.jsonl"
    edge = {r["id"]: r for r in jl(edge_p)} if edge_p.exists() else None
    drops = collections.Counter()
    cands = []
    for r in labels:
        if r["status"] != "kept":
            continue
        if r["vn"] in reserved or r["shot"] in leak:
            drops["excluded-late"] += 1
            continue
        if r["id"] in resolved:
            drops["resolution-drop"] += 1
            continue
        if r["id"] not in idx or is_known(idx[r["id"]][1], kb.get(r["shot"], [])):
            drops["known-line"] += 1
            continue
        if set(vote.key(r["label"])) - vocab or set(r["label"]) - vocab:
            drops["out-of-vocab"] += 1
            continue
        cands.append(r)
    by_shot = collections.defaultdict(list)
    for r in cands:
        by_shot[r["shot"]].append(r)
    cut = {}
    for sk in sorted(by_shot):
        part, d = cut_native(shots[sk], by_shot[sk], idx, edge, margins)
        cut.update(part)
        drops += d
    kept, capped = PL.cap_repeats([r for r in cands if r["id"] in cut])
    drops["repeat-cap"] += len(capped)
    files = collections.defaultdict(list)
    man = []
    for r in sorted(kept, key=lambda r: (int(r["vn"][1:]), r["id"])):
        for crop, img, meta in cut[r["id"]]:
            files[crop].append((img, r["label"], meta))
        man.append({"id": r["id"], "shot": r["shot"], "vn": r["vn"], "kind": shots[r["shot"]]["kind"],
                    "how": r["how"], "luna": r.get("luna"), "tier": tiers.get(r["id"], "auto"),
                    "crops": {c: meta["box"] for c, _, meta in cut[r["id"]]}})
    stats = {"lines": len(man), "vns": len({m["vn"] for m in man}), "dropped": dict(drops), "files": {}}
    for crop, items in files.items():
        PL.save_npz(out[crop], [i[0] for i in items], [i[1] for i in items], [i[2] for i in items])
        stats["files"][out[crop].name] = {"crops": len(items), "by_target": dict(collections.Counter(
            i[2]["target"] for i in items)), "sha256": P.file_sha256(out[crop])}
    wjl(NP / "manifests" / f"lines_{args.tag}.jsonl", man)
    wjl(NP / f"lines_full_{args.tag}.jsonl", [dict(m, label={r["id"]: r["label"] for r in kept}[m["id"]])
                                               for m in man])
    (NP / "manifests" / f"build_stats_{args.tag}.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1), flush=True)


OWNER_NOTE = ("\nThese are the owner's own live console captures: PRIVATE and copyrighted. Never send any sheet,\n"
              "crop, frame or transcription from this batch to Codex, gpt-6-luna or any other model or third\n"
              "party. View one image per tool call, at most 1280 px.\n")
CHECK_STRATA = {"auto": 40, "reviewed": 120}      # pre-registered here before any check sheet existed
CHECK_NOTE = ("\nThis is a label-precision check (native-frames): no candidate reading is shown.\n"
              "Type exactly the text inside the red frame as {\"text\": ...}, whatever it is (dialogue,\n"
              "a name, a menu), in the conventions above, every bracket and punctuation mark inside the\n"
              "frame included; do not use skip \"ui\". Mark unsure when a character cannot be settled;\n"
              "skip \"cut\" or \"bad\" only as defined above.\n")


def parse_strata(text):
    """"auto:60,reviewed:140" -> {"auto": 60, "reviewed": 140} (a pool's own pre-registered sample sizes)."""
    out = {}
    for part in text.split(","):
        k, n = part.split(":")
        if k not in CHECK_STRATA:
            raise ValueError(f"unknown stratum {k!r}")
        out[k] = int(n)
    return out


def check_sample(man, seed, strata=CHECK_STRATA):
    """[{id, crop, stratum}]: strata[s] built lines drawn at random from each stratum (auto: all four readings
    agreed; reviewed: Claude + Luna), each shown as one of the crops the npz holds for it, chosen at random."""
    import random
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for m in man:
        by["auto" if m["how"] == "auto" else "reviewed"].append(m)
    out = []
    for s, n in strata.items():
        rows = sorted(by.get(s, []), key=lambda m: m["id"])
        for m in rng.sample(rows, min(n, len(rows))):
            out.append({"id": m["id"], "crop": rng.choice(sorted(m["crops"])), "stratum": s})
    return out


def cmd_check(args):
    """The precision sample (review/check_spec_<tag>.json, drawn once) and its sheets (review/check/bNNNN:
    devnative's blind mode with no candidate, the crop the npz holds framed on its own screen)."""
    import random

    import pool as PL

    from devnative import sheets as S
    man = jl(NP / f"lines_full_{args.tag}.jsonl")
    if not man:
        sys.exit("build first")
    spec_p = NP / "review" / f"check_spec_{args.tag}.json"
    if spec_p.exists():
        sys.exit(f"{spec_p} exists: the sample is drawn once")
    strata = parse_strata(args.strata) if getattr(args, "strata", None) else CHECK_STRATA
    sample = check_sample(man, args.seed, strata)
    spec_p.write_text(json.dumps({"tag": args.tag, "seed": args.seed, "strata": strata, "from": len(man),
                                  "entries": sample}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    shots, rows = PL.load_reads(), {m["id"]: m for m in man}
    idx = PL.S.line_index(shots)
    order = list(range(len(sample)))
    random.Random(f"{args.seed}:order").shuffle(order)
    entries = []
    for i in order:
        e = sample[i]
        s, ln = idx[e["id"]]
        entries.append((f"{e['id']}@{e['crop']}", s, ln, [], rows[e["id"]]["crops"][e["crop"]]))
    for k in range(0, len(entries), args.size):
        out = NP / "review" / "check" / f"b{k // args.size + 1:04d}"
        n = S.make_batch(entries[k:k + args.size], out, "blind", name=f"native/check/{out.name}")
        with open(out / "README.md", "a", encoding="utf-8") as f:
            f.write(CHECK_NOTE + (OWNER_NOTE if os.environ.get("NATIVE_OWNER_POOL") == "1" else ""))
        print(f"{out}: {len(entries[k:k + args.size])} entries, {n} sheets", flush=True)


def cmd_check_score(args):
    """Agreement of the blind transcriptions with the pool's labels (eval's scoring key), per stratum and
    crop kind, with Wilson intervals; a skip or unsure counts as a disagreement."""
    from devnative import decide as D
    from teacher import vote
    spec = json.loads((NP / "review" / f"check_spec_{args.tag}.json").read_text(encoding="utf-8"))
    labels = {m["id"]: m["label"] for m in jl(NP / f"lines_full_{args.tag}.jsonl")}
    done = sorted(b for b in (NP / "review" / "check").glob("b*") if (b / "decisions.jsonl").exists())
    decided = D.collect(done)
    by, dis = collections.defaultdict(lambda: [0, 0]), []
    for e in spec["entries"]:
        d = decided.get(f"{e['id']}@{e['crop']}")
        if d is None:
            continue
        agree = d["decision"] in ("pick", "text") and vote.key(d.get("text") or "") == vote.key(labels[e["id"]])
        for k in ("all", f"stratum:{e['stratum']}", f"crop:{e['crop']}"):
            by[k][0] += agree
            by[k][1] += 1
        if not agree:
            dis.append({**e, "decision": d["decision"], "text": d.get("text"), "label": labels[e["id"]]})
    summ = {k: {"agree": a, "lines": n, "rate": round(a / n, 4), "wilson95": D.wilson95(a, n)}
            for k, (a, n) in sorted(by.items())}
    (NP / "review" / f"check_report_{args.tag}.json").write_text(json.dumps(
        {"summary": summ, "of": len(spec["entries"]), "disagreements": dis, "reviewer_kind": "assistant"},
        ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summ, indent=1), flush=True)


def cmd_publish(args):
    """The committed, text-free manifests (realplus/manifests/native/): the enumeration counts, the selected
    frames (URL, VNDB id, file and pixel sha256, kind, release, session, title group, why selected), the built
    lines (ids, crop boxes, review provenance; no text) and the counts and sha256 of every stage and file."""
    out = ROOT / "realplus" / "manifests" / "native"
    out.mkdir(parents=True, exist_ok=True)
    (out / "enumeration.json").write_text((NP / "manifests" / "enumeration.json").read_text())
    keys = ("shot", "vn", "image", "url", "kind", "dims", "release", "session", "group", "platforms", "source",
            "order")
    dups = json.loads((NP / "manifests" / "dups.json").read_text())
    rows = []
    for r in jl(NP / "manifests" / "shots.jsonl"):
        row = {k: r.get(k) for k in keys}
        # the local copy read (realdata's / synth2's file of the VNDB screenshot): file and decoded-pixel sha256
        row["sha256"] = SH.file_sha256(r["path"]) if r.get("path") else None
        row["pixel_sha256"] = SH.pixel_sha256(r["path"]) if r.get("path") else None
        row["dropped"] = dups.get(str(r["image"]))
        rows.append(row)
    wjl(out / "shots.jsonl", rows)
    wjl(out / f"lines_{args.tag}.jsonl", jl(NP / "manifests" / f"lines_{args.tag}.jsonl"))
    stats = {"tag": args.tag, "build": json.loads((NP / "manifests" / f"build_stats_{args.tag}.json").read_text())}
    # frames read before a later owner rule reserved their VN: none of their lines is in the pool
    reserved = reserved_today()
    selected = {r["shot"] for r in rows}
    late = collections.Counter((r["vn"], reserved.get(r["vn"], "not-reserved"))
                               for r in jl(NP / "reads" / "reads.jsonl") if r["shot"] not in selected)
    stats["read_not_selected"] = {f"{vn} {why}": n for (vn, why), n in sorted(late.items())}
    for name, p in (("luna_agreement", NP / "luna" / "agreement_all8.json"),
                    ("precision_check", NP / "review" / f"check_report_{args.check}.json")):
        if p.exists():
            d = json.loads(p.read_text())
            stats[name] = {k: v for k, v in d.items() if k not in ("disagreements", "batches")}
    (out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(f"published {len(rows)} frames, {stats['build']['lines']} lines -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("enumerate")
    p.add_argument("--dump", type=Path, default=DUMP)
    for c in ("select", "prep", "known", "filter-plan", "filter-labels", "edge-boxes"):
        sub.add_parser(c)
    p = sub.add_parser("build")
    p.add_argument("--tag", required=True)
    p.add_argument("--dump", type=Path, default=DUMP)
    p.add_argument("--margins", default="", help="extra ink-crop margins in px, e.g. 4,8 (files ink_m4, ink_m8)")
    p = sub.add_parser("check")
    p.add_argument("--tag", required=True)
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--size", type=int, default=160)
    p.add_argument("--strata", help="sample sizes per stratum, e.g. auto:60,reviewed:140 (default 40 / 120)")
    p = sub.add_parser("check-score")
    p.add_argument("--tag", required=True)
    p = sub.add_parser("publish")
    p.add_argument("--tag", required=True)
    p.add_argument("--check", default="pre", help="the build tag the precision sample was drawn from")
    a = ap.parse_args()
    cmds = {"enumerate": cmd_enumerate, "select": cmd_select, "prep": cmd_prep, "known": cmd_known,
            "filter-plan": cmd_filter_plan, "filter-labels": cmd_filter_labels, "edge-boxes": cmd_edge_boxes,
            "build": cmd_build, "check": cmd_check, "check-score": cmd_check_score, "publish": cmd_publish}
    cmds[a.cmd](a)


if __name__ == "__main__":
    main()
