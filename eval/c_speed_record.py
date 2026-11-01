#!/usr/bin/env python3
"""c-speed's record (#165, docs/c-speed.md): the runtime's speed change in numbers, from the measurement directories.

  c_speed_record.py --art DIR --out eval/c_speed_20261101.json

DIR is the work directory ($HERD/shared/artifacts/vocr-c-speed) with:
  a9/{base,final2}/{nopld,pld}_{S,N,L}/report.json, report444.json  vita/a9_layers.py on sub24 (Rubi's options)
  a9/final2_noattn/nopld_{S,N,L}/                                    the same, -DVOCR_NO_ATTN (Rubi's build)
  ts/{base,final}_{dark4,nodark}/                                    eval/textspot6_a9.py measure (t64n8c16_t4)
  parity/sp2_{S,N,L}_{logits,rubi}[.<build>.<mode>].json, sp2_rubipin_{S,N,L}.json   vita/speed_parity.py
  parity/ts_full_final2.json, parity/vita3k_*.json                   vita/ts_speed_parity.py, vita/vita3k_parity.py
  logs/twin_arm_final2.log, logs/check_arm_new2.log, logs/parity_arm_new2.log
  results_host_ms.json, size/{codesize,stack}_{base,final}.json      host timing, eval/textspot6_a9.py codesize / stack
  base_runtime/ (main's runtime/ and COMMIT), models/ (the v0.5.0 blobs)
and builds both versions' host drivers (arenas), every build without NEON (their .text: before == after) and the
Vita objects (their code sizes; vocr.c's stack chains).
Every time in it is a host estimate of the console (the Cortex-A9 cost model, x1.000676 for the recognizer and x1.0007
for the spotter: VOCR3-H15's console calibration) or a host measurement; nothing ran on a console. It holds hashes and
counts only (no text, no reading, no pixel).
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in ("eval", "vita", "train", "synth")]

TIERS = {"S": ("VOCR6.2_S_w8.vocr", "--topk 8"),
         "N": ("VOCR6.2_N_w8g2.vocr", "--topk 8 --head-gate on --two-level"),
         "L": ("VOCR6.2_L_w8_2l.vocr", "--topk 8 --two-level")}
BOX_FRAMES, BOX_FINDER_MS = 240, 100.0      # the nominal 3-line box of docs/vocr6.md (240 CTC frames + 0.1 s)
SPOT_TAG = "t64n8c16_t4_h480_{}_grey"
SPOT_SIZES = ("900x150", "900x200", "900x300", "960x544")
AFTER = "final2"            # the A9 runs of the final runtime (a9/final2, a9/final2_noattn)
SP = "sp2_"                 # its speed_parity runs


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def recognizer(art):
    out = {}
    for t in TIERS:
        row = {}
        for ver, d in (("before", "base"), ("after", AFTER), ("after_no_attn", AFTER + "_noattn")):
            for b in ("nopld", "pld") if ver != "after_no_attn" else ("nopld",):
                for mhz, rep in ((333, "report.json"), (444, "report444.json")):
                    r = json.loads((art / "a9" / d / f"{b}_{t}" / rep).read_text())
                    F, s = r["frames"], r["scale"]
                    ms = sum(v["high"] for v in r["phases"].values()) * s / F / (mhz * 1000)
                    row.setdefault(ver, {}).setdefault(b, {})[str(mhz)] = {
                        "ms_per_frame": round(ms, 9), "box_ms": round(BOX_FRAMES * ms + BOX_FINDER_MS, 1)}
                row[ver]["insns_per_frame"] = round(sum(v["insns"] for v in r["phases"].values()) / F)
                row[ver]["texts_sha256"] = r["meta"]["runs"]["prof"]["texts_sha256"]
                row[ver]["vocr_c_sha256"] = r["meta"]["vocr_c_sha256"]
                row[ver]["frames"] = F
                row[ver]["vocr_cflags"] = r["meta"].get("vocr_cflags", "-O2")   # runs before the option: -O2
        row["blob"], row["cli_opts"] = TIERS[t]
        row["pack_sha256"] = json.loads((art / "a9" / "base" / f"nopld_{t}" / "meta.json").read_text())["pack_sha256"]
        out[t] = row
    return out


def spotter(art):
    import a9_cycles as A
    out = {}
    for ver, pre in (("before", "base"), ("after", "final")):
        for mode in ("dark4", "nodark"):
            d = art / "ts" / f"{pre}_{mode}"
            rows = A.parse_dis((d / "driver.dis").read_text())
            funcs = set((d / "functions.txt").read_text().split())
            for size in SPOT_SIZES:
                tag = SPOT_TAG.format(size)
                cb, eb = A.read_counts(d / f"counts_{tag}_batch.txt")
                cd, ed = A.read_counts(d / f"counts_{tag}_dry.txt")
                delta = {a: v for a, v in ((a, cb.get(a, 0) - cd.get(a, 0)) for a in set(cb) | set(cd)) if v > 0}
                extra = {k: eb.get(k, 0) - ed.get(k, 0) for k in set(eb) | set(ed)}
                c1, c0 = A.read_cache(d / f"cache_{tag}_batch.txt"), A.read_cache(d / f"cache_{tag}_dry.txt")
                r = A.model_mode(rows, delta, extra, funcs, 1, {k: max(0, c1[k] - c0[k]) for k in c1})
                ms = 1.0007 * r["cycles_per_frame"][333][1] / 333e3
                out.setdefault(size, {}).setdefault("dark" if mode == "dark4" else "no_dark", {})[ver] = {
                    "ms_333": round(ms, 3), "ms_444": round(ms * 333 / 444, 3),
                    "insns": round(r["insns_per_frame"])}
            meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
            out.setdefault("sources", {})[f"{ver}_{mode}"] = meta.get("sources_sha256") or meta.get("sources")
    return out


BUILDS = ("host-gcc", "host-clang", "arm-neon", "arm-scalar", "vita-neon", "vita-neon-noattn")


def sp_run(p, stem):
    """A vita/speed_parity.py record: STEM.json, or the union of STEM.<build>.<mode>.json (one run split per build and
    mode to run in parallel), its cross-build check recomputed as speed_parity.py does it (every build's new hashes
    against the first build's base hashes, per blob and mode)."""
    one = p / f"{stem}.json"
    if one.exists():
        return json.loads(one.read_text())
    parts = [json.loads(f.read_text()) for f in sorted(p.glob(f"{stem}.*.json"))]
    if not parts:
        raise SystemExit(f"no {stem}.json or {stem}.*.json in {p}")
    for k in ("pack_sha256", "cli_opts", "logits"):
        if len({json.dumps(d[k]) for d in parts}) != 1:
            raise SystemExit(f"{stem}: the split runs differ in {k}")
    rows = sorted((r for d in parts for r in d["results"]),
                  key=lambda r: (r["blob"], r["mode"] != "w8i", BUILDS.index(r["build"])))
    keys = ("logits_sha256", "texts_sha256", "topk_sha256")
    ref = {}
    for r in rows:
        ref.setdefault((r["blob"], r["mode"]), tuple(r["base"][k] for k in keys))
        r["equals_other_builds"] = tuple(r["new"][k] for k in keys) == ref[r["blob"], r["mode"]]
        r["new_equals_base"] = all(r["new"][k] == r["base"][k] for k in keys)
    return {"ok": all(r["new_equals_base"] and r["equals_other_builds"] for r in rows),
            "pack_sha256": parts[0]["pack_sha256"], "cli_opts": parts[0]["cli_opts"], "logits": parts[0]["logits"],
            "results": rows, "split_runs": len(parts)}


def parity(art):
    p = art / "parity"
    res = {"recognizer": [], "recognizer_vs_rubi_pin": [], "spotter": None, "vita3k": None}
    for t in TIERS:
        for kind, key in (("logits", "recognizer"), ("rubi", "recognizer")):
            d = sp_run(p, f"{SP}{t}_{kind}")
            res[key].append({"tier": t, "run": kind, "cli_opts": d["cli_opts"], "logits": d.get("logits", True),
                             "pack_sha256": d["pack_sha256"], "ok": d["ok"], "split_runs": d.get("split_runs", 1),
                             "rows": [{k: r[k] for k in ("mode", "build", "new_equals_base", "equals_other_builds")}
                                      | {"logits_sha256": r["new"]["logits_sha256"], "logit_bytes":
                                         r["new"]["logit_bytes"], "texts_sha256": r["new"]["texts_sha256"],
                                         "topk_sha256": r["new"]["topk_sha256"]} for r in d["results"]]})
        d = sp_run(p, f"{SP}rubipin_{t}")
        res["recognizer_vs_rubi_pin"].append({"tier": t, "cli_opts": d["cli_opts"], "ok": d["ok"],
                                              "rows": [{"mode": r["mode"], "build": r["build"],
                                                        "new_equals_base": r["new_equals_base"],
                                                        "texts_sha256": r["new"]["texts_sha256"],
                                                        "topk_sha256": r["new"]["topk_sha256"]}
                                                       for r in d["results"]]})
    d = json.loads((p / "ts_full_final2.json").read_text())
    res["spotter"] = {"ok": d["ok"], "records": d["records"], "records_sha256": d["records_sha256"],
                      "frames": d["frames"],
                      "rows": [{"build": r["build"], "form": r["form"], "new_equals_base": r["new_equals_base"],
                                "equals_other_builds": r["equals_other_builds"],
                                "maps_sha256": r["new"]["maps"]["sha256"], "maps_lines": r["new"]["maps"]["lines"],
                                "boxes_sha256": r["new"]["batch"]["sha256"]} for r in d["results"]]}
    v3 = sorted(p.glob("vita3k_head*.json")) or sorted(p.glob("vita3k_final.json"))
    d = json.loads(v3[-1].read_text())
    res["vita3k"] = {"record": v3[-1].name, "ok": d["ok"], "jobs": d["jobs"],
                     "versions": {k: {kk: v[kk] for kk in ("vocr_c_sha256", "vocr_textspot6_c_sha256", "vpk_sha256")}
                                  for k, v in d["versions"].items()},
                     "outputs": [{k: o[k] for k in ("job", "file", "bytes", "sha256", "all_equal")}
                                 for o in d["outputs"]]}
    twin = (art / "logs" / "twin_arm_final2.log").read_text()
    m = re.search(r"TEXTSPOT6_PARITY_OK (\d+) runs \((\d+) frames, (\d+) blobs \((\d+) of version 4", twin)
    res["spotter_twin_neon"] = {"ok": bool(m), "runs": int(m.group(1)), "frames": int(m.group(2)),
                                "blobs": int(m.group(3)), "v4_blobs": int(m.group(4))} if m else {"ok": False}
    arm = (art / "logs" / "check_arm_new2.log").read_text()
    par = (art / "logs" / "parity_arm_new2.log").read_text()
    res["check_arm"] = arm.count("test_vocr: all checks passed") == 3   # NEON, scalar, NEON -DVOCR_NO_ATTN
    res["parity_arm"] = "PARITY_OK" in par
    return res


WARN = ["-std=c99", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror"]


def arenas(art, work):
    """vocr_arena_bytes at 1,024 px with Rubi's options and the spotter's arenas, before and after (host builds)."""
    import subprocess
    out = {}
    for ver, rt in (("before", art / "base_runtime"), ("after", ROOT / "runtime")):
        d = work / f"arena_{ver}"
        d.mkdir(parents=True, exist_ok=True)
        cli, ts6 = d / "cli", d / "ts6"
        subprocess.run(["gcc", *WARN, "-O2", "-I", str(rt), "-o", str(cli), *(str(rt / f) for f in
                        ("vocr.c", "vocr_lm.c", "vocr_post.c")), str(ROOT / "runtime/tools/vocr_cli.c"), "-lm"],
                       check=True)
        subprocess.run(["gcc", *WARN, "-O2", "-DVOCR_TEXTSPOT6_NO_MS", "-I", str(rt), "-o", str(ts6),
                        *(str(rt / f) for f in ("vocr_lines.c", "vocr_lines_rows.c", "vocr_textspot6.c")),
                        str(ROOT / "runtime/vocr_textspot6_test.c"), "-lm"], check=True)
        for t, (blob, opts) in TIERS.items():
            o = [x for x in opts.split() if x != "--topk" and not x.isdigit()]
            r = subprocess.run([str(cli), str(art / "models" / blob), "--mode", "w8i", *o, "--arena", "1024"],
                               capture_output=True, text=True, check=True)
            out.setdefault(t, {})[ver] = int(re.search(r"arena (\d+)", r.stdout + r.stderr).group(1))
        for size, flags in (("900x150", "4"), ("960x544", "0"), ("480x272", "0")):
            w, h = size.split("x")
            r = subprocess.run([str(ts6), "--arena", str(art / "models/textspot8_t64n8c16_w8.vtsp"), w, h, flags],
                               capture_output=True, text=True, check=True)
            out.setdefault(f"spot_{size}_flags{flags}", {})[ver] = int(r.stdout.strip())
    return out


def machine_code(art, work, vitasdk, arm_cc, pspdev):
    """The .text of both files' objects in every build without NEON: before == after."""
    import subprocess
    builds = {"host gcc -O2": (["gcc"], "objcopy"), "host clang -O2": (["clang"], "objcopy"),
              "vita gcc -O2 -DVOCR_NO_NEON": ([f"{vitasdk}/bin/arm-vita-eabi-gcc", "-mcpu=cortex-a9", "-mfpu=neon",
                                               "-mfloat-abi=hard", "-DVOCR_NO_NEON"],
                                              f"{vitasdk}/bin/arm-vita-eabi-objcopy"),
              "arm gnu 14.2 -O2 -DVOCR_NO_NEON": ([arm_cc, "-mcpu=cortex-a9", "-mfpu=neon", "-mfloat-abi=hard",
                                                   "-DVOCR_NO_NEON"], arm_cc[:-3] + "objcopy"),
              "psp-gcc -O2 -G0": ([f"{pspdev}/bin/psp-gcc", "-G0"], f"{pspdev}/bin/psp-objcopy"),
              "psp-gcc -O2 -G0 -DVOCR_PSP_VFPU": ([f"{pspdev}/bin/psp-gcc", "-G0", "-DVOCR_PSP_VFPU"],
                                                  f"{pspdev}/bin/psp-objcopy")}
    out = {}
    for name, (cc, objcopy) in builds.items():
        for defs in ([], ["-DVOCR_TEXTSPOT6_NO_MS"]):
            for f in ("vocr.c", "vocr_textspot6.c"):
                if defs and f == "vocr.c":
                    continue
                texts = []
                for ver, rt in (("before", art / "base_runtime"), ("after", ROOT / "runtime")):
                    o = work / f"mc_{len(out)}_{ver}.o"
                    subprocess.run([*cc, *WARN, "-O2", *defs, "-I", str(rt), "-c", "-o", str(o), str(rt / f)],
                                   check=True, capture_output=True)
                    subprocess.run([objcopy, "-O", "binary", "--only-section=.text", str(o), str(o) + ".bin"],
                                   check=True)
                    texts.append(Path(str(o) + ".bin").read_bytes())
                out[f"{name}{' ' + defs[0] if defs else ''}: {f}"] = {"identical": texts[0] == texts[1],
                                                                       "text_bytes": len(texts[1])}
    return out


VITA = ["-mcpu=cortex-a9", "-mfpu=neon", "-mfloat-abi=hard"]
RUBI = ["-ffunction-sections", "-fdata-sections"]   # Rubi's cmake/VocrOCR.cmake adds these to vjo_vocr_rt's -O2


def vita_objects(art, work, vitasdk):
    """The Vita objects' code (arm-vita-eabi-size's text) and vocr.c's deepest stack chains (-fcallgraph-info=su),
    before and after, the recognizer also with -DVOCR_NO_ATTN: at vita/CMakeLists.txt's -O2 and with Rubi's flags."""
    import subprocess

    import textspot3_a9r as A9R
    cc, size = f"{vitasdk}/bin/arm-vita-eabi-gcc", f"{vitasdk}/bin/arm-vita-eabi-size"
    code, stack = {}, {}
    for label, extra in (("vita -O2", []), ("rubi -O2 -ffunction-sections -fdata-sections", RUBI)):
        for f, defs in (("vocr.c", []), ("vocr.c", ["-DVOCR_NO_ATTN"]),
                        ("vocr_textspot6.c", ["-DVOCR_TEXTSPOT6_NO_MS"])):
            key = f"{label}: {f}{''.join(' ' + d for d in defs)}"
            for ver, rt in (("before", art / "base_runtime"), ("after", ROOT / "runtime")):
                d = work / f"vo_{len(code)}_{ver}"
                d.mkdir(parents=True, exist_ok=True)
                o = d / (Path(f).stem + ".o")
                subprocess.run([cc, *WARN, *VITA, "-O2", *extra, *defs, "-fcallgraph-info=su", "-I", str(rt), "-c",
                                "-o", str(o), str(rt / f)], check=True, capture_output=True, cwd=d)
                text = subprocess.run([size, str(o)], check=True, capture_output=True, text=True).stdout
                code.setdefault(key, {})[ver] = int(text.splitlines()[-1].split()[0])
                if f == "vocr.c":
                    ci = (d / "vocr.ci").read_text(encoding="utf-8")
                    pk, _ = A9R.peaks([ci], ["vocr_run"])
                    w8i, _ = A9R.peaks([ci], ["vocr_run"], exclude=("mm_f32",))   # W8I (Rubi's mode) has no f32 matmul
                    stack.setdefault(key, {})[ver] = {
                        "vocr_run": {"bytes": pk["vocr_run"]["bytes"], "chain": pk["vocr_run"]["chain"]},
                        "vocr_run without mm_f32": {"bytes": w8i["vocr_run"]["bytes"],
                                                    "chain": w8i["vocr_run"]["chain"]}}
    return code, stack


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--art", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--vitasdk", required=True)
    ap.add_argument("--arm-cc", required=True)
    ap.add_argument("--pspdev", required=True)
    a = ap.parse_args()
    art = a.art
    import tempfile
    work = Path(tempfile.mkdtemp(prefix="c-speed-record-"))
    base = {f: sha(art / "base_runtime" / f) for f in ("vocr.c", "vocr_textspot6.c")}
    new = {f: sha(ROOT / "runtime" / f) for f in ("vocr.c", "vocr_textspot6.c")}
    rec = {"kind": "c-speed", "issue": 165, "doc": "docs/c-speed.md",
           "note": "estimates: the Vita compiler's code under qemu-arm priced by vita/a9_cycles.py's Cortex-A9 model, "
                   "x1.000676 (recognizer) / x1.0007 (spotter), VOCR3-H15's console calibration; nothing ran on a "
                   "console. Development data only. Hashes and counts only.",
           "base_commit": (art / "base_runtime" / "COMMIT").read_text().strip(),
           "sources": {"before": base, "after": new},
           "box": {"frames": BOX_FRAMES, "finder_ms": BOX_FINDER_MS},
           "recognizer": recognizer(art), "spotter": spotter(art), "parity": parity(art),
           "host_ms": json.loads((art / "results_host_ms.json").read_text()),
           "memory": {k: json.loads((art / "size" / f"{k}.json").read_text()) for k in
                      ("codesize_base", "codesize_final")}}
    rec["memory"]["arena"] = arenas(art, work)
    rec["machine_code_without_neon"] = machine_code(art, work, a.vitasdk, a.arm_cc, a.pspdev)
    rec["memory"]["vita_code"], rec["memory"]["vocr_stack"] = vita_objects(art, work, a.vitasdk)
    rec["memory"]["stack"] = {}
    for v in ("base", "final"):
        s = json.loads((art / "size" / f"stack_{v}.json").read_text())
        rec["memory"]["stack"][v] = {b: {k: x["bytes"] for k, x in s["builds"][b]["peaks"].items()} |
                                     {"find_no_dark": s["builds"][b]["no_dark"]
                                      ["vocr_textspot6_find without the dark pass"]["bytes"]}
                                     for b in s["builds"]}
    a.out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
