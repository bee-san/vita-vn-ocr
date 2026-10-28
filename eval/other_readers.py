#!/usr/bin/env python3
"""Other OCR engines on the line crops VOCR6's final reads scored: the README's comparison (report only;
eval/other_readers_protocol.json).

  other_readers.py read  --set clannad|pcsg00807 --reader NAME --spec SPEC --into DIR     (heavy.sh; offline)
  other_readers.py score --into DIR --out eval/other_readers_<date>.json [--md DIR/report.md]

read   one engine (eval/backends.py make_recognizer SPEC: ppocr-v6-small, ncnn:DIR, manga-ocr, or
       cmd:.venv-meiki/bin/python eval/meiki_cmd.py) on one private set's stored crops, given as three identical
       channels: CLANNAD's 3,048 referenced lines (clannad_refs's stored GRAD crops, each checked against its capture
       hash) or the hooked test title's 1,152 clean dialogue lines (pools v1 test/line/PCSG00807.npz) ->
       DIR/<set>.<NAME>.jsonl (private readings) and .meta.json (the engine's identity, hashes, host time).
       vita-test and psp-test are read by eval/run.py itself (--polarity native --out DIR/run_<NAME>_<set>.json).
score  every set's dialogue CER for VOCR6 and VOCR5 (greedy: their final reads' private per-line records), Google
       Lens (its existing readings) and each engine, with the protocol's intervals and the pairs VOCR6 / VOCR5 tier -
       engine; it first recounts VOCR6's, VOCR5's, Lens's and MeikiOCR's committed numbers from those records and
       stops on any difference. The public JSON holds counts, rates, intervals and hashes only.

Everything under DIR and $HERD/shared/private is private (copyrighted game frames, readings, references): never
commit it, never send it to Codex or any third party.
"""
import argparse
import datetime
import hashlib
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (HERE, ROOT, ROOT / "data", ROOT / "runtime", ROOT / "train", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from metrics import lev, norm  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
PRIV = HERD / "shared" / "private"
PROTOCOL = HERE / "other_readers_protocol.json"
FOLLOWUP_RECORD = HERE / "vocr6_final_test_followup_20261027.json"
VOCR5_DIALOGUE = HERE / "vocr5_dialogue_20261016.json"
MEIKI_CLANNAD = HERE / "clannad_meiki_lines_20261014.json"
INPUTS = {
    "clannad_work": PRIV / "owner-drive-20261013" / "clannad-compare",
    "followup": PRIV / "vocr6-final-test" / "final-followup-20261027",
    "meiki_clannad_arms": PRIV / "meiki-clannad" / "arms.meiki.jsonl",
    "lens_runs": PRIV / "vocr5-train" / "lens",
    "pcsg_npz": HERD / "shared" / "artifacts" / "vocr6-ingest" / "pools" / "v1" / "test" / "line" / "PCSG00807.npz",
}
PCSG_NPZ_SHA256 = "918c487e4fac6215979141e6074323846fb0a55597a7125428290b855a9f8e84"
CLANNAD_REFS_SHA256 = "35411688749b2968b4b47306b65475503644c0bfa1ea8eba0b30885fe4a6212b"
CLANNAD_LINES_SHA256 = "a6be0300b651a2cdf3ae58f9705b6f100263b66b7030365b119e6a0a2323ffd1"
READERS = ("MeikiOCR", "PPOCRv6_small", "PPOCRv5_mobile_int8", "manga_ocr")
TIERS = ("S", "N", "L")
VOCR = [f"VOCR{g}_{t}" for g in (6, 5) for t in TIERS]
SETS = ("clannad", "vita-test", "psp-test", "pcsg00807")
REPS, SEED, MIN_CLUSTERS = 10000, 0, 5
# Each engine's VOCR6 / VOCR5 greedy readings: the follow-up's final read (the shipped VOCR6.2 blobs; v0.3.0's VOCR5)
VOCR6_SHA256 = {"S": "864bbd196d5bc5d3a7bd7918cf7274b3503dfe65d440017ef0781ea7e7ba37d2",
                "N": "107cf084cc0c1a3a61ce4c3f9b9cf6b31d6dff80f0a1bdb94afd3f7ee9ae3bc9",
                "L": "65310013e8f990886e4e8097b2fe55eb98cdf082d7193eea510601fc0ac26c82"}


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_exclusive(path, text):
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)


def code_identity():
    files = ("eval/other_readers.py", "eval/backends.py", "eval/meiki_cmd.py", "eval/metrics.py",
             "eval/other_readers_protocol.json")
    return {f: sha256_file(ROOT / f) for f in files}


def environment():
    from importlib import metadata
    pk = {}
    for p in ("numpy", "onnxruntime", "opencv-python-headless", "torch", "manga-ocr", "transformers", "pillow"):
        try:
            pk[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
            "packages": pk}


# ----------------------------------------------------------------------------- read

def clannad_crops():
    """[(id, grey)] of the frozen references' scored lines (each stored PNG checked against its capture hash)."""
    import clannad_refs as CR
    work = INPUTS["clannad_work"]
    freeze = json.loads((work / "refs.freeze.json").read_text(encoding="utf-8"))
    if freeze["refs_sha256"] != CLANNAD_REFS_SHA256 or sha256_file(work / "refs.jsonl") != CLANNAD_REFS_SHA256:
        raise SystemExit("CLANNAD's references are not the frozen ones the protocol names")
    rows, meta = CR.load_lines(work)
    if meta["lines_sha256"] != CLANNAD_LINES_SHA256:
        raise SystemExit("CLANNAD's lines.jsonl is not the protocol's")
    scored = {x["id"] for x in read_jsonl(work / "refs.jsonl") if x["status"] == "ref"}
    by = {r["id"]: r for r in rows}
    return [(i, CR.load_grey(meta["session"], by[i]["line"])) for i in sorted(scored)], \
        {"refs_sha256": CLANNAD_REFS_SHA256, "lines_sha256": CLANNAD_LINES_SHA256}


def pcsg_crops():
    """[(key, grey)] of the hooked test title's clean dialogue lines, keyed as eval/vocr6_baseline.py keys them."""
    import import_hooked as IH
    f = INPUTS["pcsg_npz"]
    if sha256_file(f) != PCSG_NPZ_SHA256:
        raise SystemExit(f"{f}: not the protocol's npz")
    return [(f"clean|line|{f.stem}|{i}", np.ascontiguousarray(g)) for i, (m, lab, g) in enumerate(IH.read_npz(f))], \
        {"npz_sha256": PCSG_NPZ_SHA256}


def public_info(info):
    return {k: v for k, v in (info or {}).items() if k != "vocab_set"}


def cmd_read(a):
    import backends
    if a.reader not in READERS:
        raise SystemExit(f"--reader is one of {READERS}")
    into = Path(a.into)
    into.mkdir(parents=True, exist_ok=True)
    out, meta_path = into / f"{a.set}.{a.reader}.jsonl", into / f"{a.set}.{a.reader}.meta.json"
    if out.exists() or meta_path.exists():
        raise SystemExit(f"{out} exists: a set is read once per engine")
    crops, ident = clannad_crops() if a.set == "clannad" else pcsg_crops()
    rec = backends.make_recognizer(a.spec)
    t0 = time.time()
    texts = []
    for i in range(0, len(crops), a.chunk):
        texts += rec.read([g for _, g in crops[i:i + a.chunk]])
    wall = time.time() - t0
    if len(texts) != len(crops):
        raise SystemExit(f"{len(texts)} readings for {len(crops)} crops")
    body = "".join(json.dumps({"id": k, "text": t}, ensure_ascii=False) + "\n" for (k, _), t in zip(crops, texts))
    write_exclusive(out, body)
    meta = {"kind": "other-readers-read", "set": a.set, "reader": a.reader, "spec": a.spec, "name": rec.name,
            "info": public_info(rec.info), "lines": len(texts), "jsonl_sha256": sha256_file(out),
            "input": ident, "input_kind": "stored grey crop as three identical channels",
            "host_ms_per_line": round(1000 * wall / max(1, len(texts)), 2), "wall_s": round(wall, 1),
            "threads": backends.threads(), "protocol_sha256": sha256_file(PROTOCOL),
            "code_sha256": code_identity(), "environment": environment(), "created": now()}
    write_exclusive(meta_path, json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"set": a.set, "reader": a.reader, "lines": len(texts), "wall_s": meta["wall_s"]}))


# ----------------------------------------------------------------------------- score

def load_reads(into, set_name, reader):
    meta = json.loads((into / f"{set_name}.{reader}.meta.json").read_text(encoding="utf-8"))
    path = into / f"{set_name}.{reader}.jsonl"
    if sha256_file(path) != meta["jsonl_sha256"]:
        raise SystemExit(f"{path} changed since it was read")
    return {r["id"]: r["text"] for r in read_jsonl(path)}, meta


def rate(errors, chars, clusters):
    import vocr6_baseline as VB
    r = VB.boot_ratio(errors, chars, clusters, seed=SEED, n=REPS)
    return {"cer": r["cer"], "errors": r["errors"], "chars": r["chars"], "clusters": r["clusters"], "ci95": r["ci95"]}


def pair(err_a, err_b, chars, clusters, ids):
    """A - B in points over the same units: unit and cluster bootstraps (vocr6_final_test.boot_pair's)."""
    import vocr3_report as R
    a = {i: (int(e), int(n)) for i, e, n in zip(ids, err_a, chars)}
    b = {i: (int(e), int(n)) for i, e, n in zip(ids, err_b, chars)}
    units = R.paired_bootstrap(a, b, REPS, SEED)
    clus = R.cluster_bootstrap(a, b, dict(zip(ids, clusters)), REPS, SEED, MIN_CLUSTERS)
    return {"delta": units["delta"], "ci95": clus.get("ci95"), "unit_ci95": units["ci95"], "units": units["units"],
            "clusters": clus["clusters"], "chars": units["chars"],
            "errors": [units["candidate_errors"], units["baseline_errors"]]}


def table(units, systems, cluster_kind):
    """units: [{"id", "cluster", "chars", "errors": {system: e}}] -> per-system rates and the pairs."""
    ids = [u["id"] for u in units]
    ch = [u["chars"] for u in units]
    cl = [u["cluster"] for u in units]
    out = {"units": len(units), "chars": int(sum(ch)), "clusters": len(set(cl)), "cluster": cluster_kind,
           "values": {}, "pairs": {}}
    for s in systems:
        e = [u["errors"][s] for u in units]
        out["values"][s] = dict(rate(e, ch, cl), lines=len(units), exact_lines=sum(1 for x in e if x == 0))
    for cand in VOCR:
        for base in systems:
            if base in VOCR or cand not in systems:
                continue
            out["pairs"][f"{cand}-{base}"] = pair([u["errors"][cand] for u in units],
                                                  [u["errors"][base] for u in units], ch, cl, ids)
    for t in TIERS:
        out["pairs"][f"VOCR6_{t}-VOCR5_{t}"] = pair([u["errors"][f"VOCR6_{t}"] for u in units],
                                                   [u["errors"][f"VOCR5_{t}"] for u in units], ch, cl, ids)
    return out


def tests_units(into, set_name):
    """vita-test / psp-test dialogue lines: VOCR (console polarity) from the follow-up's locked-report records, Lens
    and each engine (screen polarity) from their eval/run.py --out files, recounted against today's manifest."""
    import vocr5_dialogue as VD
    rep = json.loads((INPUTS["followup"] / "a" / "shared" / "report.json").read_text(encoding="utf-8"))
    fl = rep["shared"]["fixed_line"]["console"]
    recs = {s: {x["id"]: x for x in fl[s]["lines"][set_name]} for s in VOCR}
    recs["Lens"] = {x["id"]: x for x in VD.lens_lines(INPUTS["lens_runs"] / f"run_{set_name}.json", set_name)}
    files = {"Lens": INPUTS["lens_runs"] / f"run_{set_name}.json"}
    for r in READERS:
        p = into / f"run_{r}_{set_name}.json"
        recs[r] = {x["id"]: x for x in VD.lens_lines(p, set_name)}
        files[r] = p
    ids = sorted(recs["VOCR6_N"])
    for s, rr in recs.items():
        if sorted(rr) != ids:
            raise SystemExit(f"{set_name}: {s} read other lines than VOCR6's")
    units = []
    for i in ids:
        x = recs["VOCR6_N"][i]
        if "name" in x["styles"]:
            continue
        if any(recs[s][i]["n"] != x["n"] for s in recs):
            raise SystemExit(f"{set_name} {i}: reference lengths differ between systems")
        units.append({"id": i, "cluster": x["vn"], "chars": x["n"], "errors": {s: recs[s][i]["errors"] for s in recs}})
    return units, {k: sha256_file(v) for k, v in files.items()}


def clannad_units(into):
    import clannad_refs as CR
    work = INPUTS["clannad_work"]
    rows, meta = CR.load_lines(work)
    refs = read_jsonl(work / "refs.jsonl")
    if sha256_file(work / "refs.jsonl") != CLANNAD_REFS_SHA256 or meta["lines_sha256"] != CLANNAD_LINES_SHA256:
        raise SystemExit("CLANNAD's references or lines are not the protocol's")
    by = {r["id"]: r for r in rows}
    texts = {"Lens": {i: by[i]["lens"] for i in by}}
    am = json.loads((INPUTS["followup"] / "clannad" / "arms.v6ftfinal.meta.json").read_text(encoding="utf-8"))
    a6 = INPUTS["followup"] / "clannad" / "arms.v6ftfinal.jsonl"
    if sha256_file(a6) != am["arms_sha256"] or any(am["arms"][f"VOCR6_{t}"]["sha256"] != VOCR6_SHA256[t]
                                                   for t in TIERS):
        raise SystemExit("the follow-up's CLANNAD arms are not the shipped VOCR6 blobs' reads")
    v6 = {r["id"]: r for r in read_jsonl(a6)}          # the final read's VOCR6 and VOCR5 arms (both read there)
    mk = {r["id"]: r for r in read_jsonl(INPUTS["meiki_clannad_arms"])}
    for t in TIERS:
        for g in (6, 5):
            texts[f"VOCR{g}_{t}"] = {i: v6[i][f"VOCR{g}_{t}"] for i in v6}
    texts["MeikiOCR"] = {i: mk[i]["MeikiOCR"] for i in mk}
    files = {"vocr_arms": a6, "meiki_arms": INPUTS["meiki_clannad_arms"]}
    for r in READERS:
        if r == "MeikiOCR":
            continue
        texts[r], _ = load_reads(into, "clannad", r)
        files[r] = into / f"clannad.{r}.jsonl"
    units, names = [], []
    for x in refs:
        if x["status"] != "ref":
            continue
        u = {"id": x["id"], "cluster": x["page"], "errors": {}}
        for s, tx in texts.items():
            e, n, *_ = CR.line_unit(x["text"], tx[x["id"]])
            u["errors"][s], u["chars"] = e, n
        (names if x["name"] else units).append(u)
    return units, names, {k: sha256_file(v) for k, v in files.items()}


def pcsg_units(into):
    import import_hooked as IH
    f = INPUTS["pcsg_npz"]
    if sha256_file(f) != PCSG_NPZ_SHA256:
        raise SystemExit("the test title's npz is not the protocol's")
    refs = {f"clean|line|{f.stem}|{i}": (norm(lab), m["page"]) for i, (m, lab, g) in enumerate(IH.read_npz(f))}
    vrows = {}
    files = {}
    for fam in ("VOCR6", "VOCR5"):
        p = INPUTS["followup"] / "a" / "pcsg" / f"{fam}.rows.pkl"
        files[fam] = p
        with open(p, "rb") as fh:
            for r in pickle.load(fh)["rows"]["clean"]:
                vrows.setdefault(r["key"], {})[fam] = r
    if set(vrows) != set(refs):
        raise SystemExit("the follow-up's clean rows are not the npz's lines")
    texts = {}
    for r in READERS:
        texts[r], _ = load_reads(into, "pcsg00807", r)
        files[r] = into / f"pcsg00807.{r}.jsonl"
    units = []
    for k in sorted(refs):
        ref, page = refs[k]
        u = {"id": k, "cluster": page, "chars": len(ref), "errors": {}}
        for fam, r in vrows[k].items():
            if r["chars"] != len(ref) or r["page"] != page:
                raise SystemExit(f"{k}: the follow-up's row differs from the npz")
            for t in TIERS:
                u["errors"][f"{fam}_{t}"] = r["errors"][f"{t}:greedy"]
        for rd, tx in texts.items():
            u["errors"][rd] = lev(norm(tx[k]), ref)
        units.append(u)
    return units, {k: sha256_file(v) for k, v in files.items()}


def checks(tables):
    """The committed numbers each recount must reproduce: (where, want, got) -> differences."""
    fu = json.loads(FOLLOWUP_RECORD.read_text(encoding="utf-8"))["path_a"]
    vd = json.loads(VOCR5_DIALOGUE.read_text(encoding="utf-8"))
    mk = json.loads(MEIKI_CLANNAD.read_text(encoding="utf-8"))["scores"]["systems"]
    want = {}
    for t in TIERS:
        for g in (6, 5):
            k = f"VOCR{g}:{t}:greedy"
            want[("clannad", f"VOCR{g}_{t}")] = fu["clannad_lines"]["dialogue"]["rates"][k]
            want[("pcsg00807", f"VOCR{g}_{t}")] = fu["pcsg00807"]["clean"]["rates"][k]
            for s in ("vita-test", "psp-test"):
                want[(s, f"VOCR{g}_{t}")] = fu["vita_psp_fixed_lines"][f"{s}|dialogue"]["rates"][k]
    for s in ("vita-test", "psp-test"):
        want[(s, "Lens")] = vd["arms"]["native"]["Lens"]["dialogue"][s]
    for s in ("Lens", "MeikiOCR"):
        want[("clannad", s)] = mk[s]["not_name"]
    out, bad = [], []
    for (set_name, system), w in sorted(want.items()):
        g = tables[set_name]["values"][system]
        got = (g["errors"], g["chars"], g["lines"])
        exp = (w["errors"], w["chars"], w["lines"])
        out.append({"set": set_name, "system": system, "errors_chars_lines": list(got), "equal": got == exp})
        if got != exp:
            bad.append(f"{set_name} {system}: recount {got} != committed {exp}")
    if bad:
        raise SystemExit("recount differs from the committed records:\n" + "\n".join(bad))
    return {"compared": len(out), "equal": sum(x["equal"] for x in out), "rows": out}


def neutral(text):
    """A spec or model string with this checkout's and $HERD's absolute paths written relatively."""
    return None if text is None else str(text).replace(str(ROOT) + "/", "").replace(str(HERD), "$HERD")


def reader_identities(into):
    out = {}
    for r in READERS:
        metas = {}
        for s in ("clannad", "pcsg00807"):
            p = into / f"{s}.{r}.meta.json"
            if p.exists():
                m = json.loads(p.read_text(encoding="utf-8"))
                metas[s] = {k: m[k] for k in ("info", "lines", "jsonl_sha256", "host_ms_per_line", "threads",
                                              "created", "environment")}
                metas[s].update(spec=neutral(m["spec"]), name=neutral(m["name"]))
        for s in ("vita-test", "psp-test"):
            p = into / f"run_{r}_{s}.json"
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                metas[s] = {"model": neutral(d.get("model")), "model_name": neutral(d.get("model_name")),
                            "model_info": public_info(d.get("model_info")), "pad": d.get("pad"),
                            "host_ms_per_line": d["sets"][s].get("host_ms_per_line"), "run_sha256": sha256_file(p),
                            "env": {k: v for k, v in (d.get("env") or {}).items() if k != "host"}}
        out[r] = metas
    return out


def sizes(into):
    """Each engine's weights as it read here (files, not a memory measurement)."""
    import backends
    mk = json.loads((HERE / "clannad_meiki_resources_20261014.json").read_text(encoding="utf-8"))["models"]["rec"]
    v6 = HERD / "shared" / "artifacts" / "teacher" / "models" / backends.PPOCR["ppocr-v6-small"]
    backends.check_pinned(v6)
    run = {r: json.loads((into / f"run_{r}_vita-test.json").read_text(encoding="utf-8"))["model_info"]
           for r in ("PPOCRv5_mobile_int8", "manga_ocr")}
    return {
        "MeikiOCR": {"bytes": mk["bytes"], "params": mk["params"], "form": "fp32 ONNX", "file": mk["file"],
                     "source": "eval/clannad_meiki_resources_20261014.json models.rec (its vertical recognizer, "
                               "12,872,961 B, and its detector, 14,503,825 B, are separate files)"},
        "PPOCRv6_small": {"bytes": v6.stat().st_size, "form": "fp32 ONNX", "file": v6.name,
                          "source": "the pinned preset file (eval/backends.py PPOCR_FILES)"},
        "PPOCRv5_mobile_int8": {"bytes": run["PPOCRv5_mobile_int8"]["weights_file_bytes"], "form": "int8 ncnn",
                                "file": "model.bin (eval/ncnn/prepare_models.py ppocr-v5-mobile-jis-int8)",
                                "source": "eval/run.py's model_info"},
        "manga_ocr": {"bytes": run["manga_ocr"]["fp32_weight_bytes"], "params": run["manga_ocr"]["params"],
                      "form": "fp32 (4 B per parameter)", "file": run["manga_ocr"]["weights"],
                      "source": "eval/run.py's model_info"},
    }


def cmd_score(a):
    into = Path(a.into)
    proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    systems = VOCR + ["Lens"] + list(READERS)
    tables, inputs = {}, {}
    cu, names, inputs["clannad"] = clannad_units(into)
    tables["clannad"] = table(cu, systems, "page")
    tables["clannad"]["names"] = {s: rate([u["errors"][s] for u in names], [u["chars"] for u in names],
                                          [u["cluster"] for u in names]) for s in systems}
    for s in ("vita-test", "psp-test"):
        units, inputs[s] = tests_units(into, s)
        tables[s] = table(units, systems, "vn")
    pooled = []
    for s in ("vita-test", "psp-test"):
        pooled += [dict(u, id=f"{s}:{u['id']}") for u in tests_units(into, s)[0]]
    tables["vita+psp"] = table(pooled, systems, "vn")
    pu, inputs["pcsg00807"] = pcsg_units(into)
    pcsg_systems = [x for x in systems if x != "Lens"]
    tables["pcsg00807"] = table(pu, pcsg_systems, "page")
    doc = {"kind": "other-readers-report", "schema": 1, "protocol_sha256": sha256_file(PROTOCOL),
           "metric": proto["metric"], "readers": reader_identities(into),
           "systems": {"VOCR6_S": "VOCR6.2_S_w8.vocr, greedy", "VOCR6_N": "VOCR6.2_N_w8g2.vocr, greedy",
                       "VOCR6_L": "VOCR6.2_L_w8_2l.vocr, greedy", "VOCR5_S": "VOCR5_S_w8.vocr, greedy",
                       "VOCR5_N": "VOCR5_N_w8.vocr, greedy", "VOCR5_L": "VOCR5_L_w8.vocr, greedy",
                       "Lens": "Google Lens (owocr 1.26.7), existing readings", **{r: r for r in READERS}},
           "inputs_sha256": inputs, "sizes": sizes(into), "sets": tables, "checks": checks(tables),
           "notes": list(a.note),
           "lens_on_pcsg00807": "not read: private game data never goes to a third-party service",
           "training_or_selection_use": False, "created": now()}
    blob = json.dumps(doc, ensure_ascii=False)
    for word in ('"text"', '"pred"', '"ref"'):
        if word in blob:
            raise SystemExit(f"the public record would hold {word}")
    write_exclusive(Path(a.out), json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    md = markdown(doc)
    if a.md:
        Path(a.md).write_text(md + "\n", encoding="utf-8")
    print(md)


def markdown(doc):
    order = VOCR + ["Lens"] + list(READERS)
    lines = ["| system | " + " | ".join(SETS) + " |", "|---|" + "---|" * len(SETS)]
    for s in order:
        cells = []
        for k in SETS:
            v = doc["sets"][k]["values"].get(s)
            cells.append("-" if v is None else f"{v['cer']:.2f}% ({v['errors']}/{v['chars']}) "
                         f"[{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines += ["", "| pair | " + " | ".join(SETS) + " |", "|---|" + "---|" * len(SETS)]
    for p in doc["sets"]["clannad"]["pairs"]:
        cells = []
        for k in SETS:
            x = doc["sets"][k]["pairs"].get(p)
            ci = x and (x["ci95"] or x["unit_ci95"])
            cells.append("-" if not x else f"{x['delta']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]")
        lines.append(f"| {p} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("--set", required=True, choices=("clannad", "pcsg00807"))
    r.add_argument("--reader", required=True)
    r.add_argument("--spec", required=True)
    r.add_argument("--into", required=True)
    r.add_argument("--chunk", type=int, default=64)
    s = sub.add_parser("score")
    s.add_argument("--into", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--md")
    s.add_argument("--note", action="append", default=[], help="a line for the record's notes (e.g. a rerun)")
    a = ap.parse_args(argv)
    {"read": cmd_read, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
