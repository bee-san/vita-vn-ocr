#!/usr/bin/env python3
"""The README's chart: VOCR6 against VOCR5 and the other OCR engines, dialogue CER per test set, light and dark,
SVG + 2x PNG.

Every number comes from docs/charts/vocr-vs-lens.json, which `data` builds from committed records: the other
engines' read (eval/other_readers_*.json: every system's dialogue CER on the same line crops, greedy for VOCR, and the
pairs), VOCR6's follow-up records (Rubi's path for the table, its gate fits for the box times), VOCR5's protocol and
locked report (its box times), and VOCR5's dialogue tables (Lens's furigana counts). The JSON records where each
number came from. The words live in its "copy" object (kept when `data` runs again); the numbers in the row notes
and the footnote are filled into them from the data. Nothing here scores anything.

  python docs/charts/vocr_vs_lens.py fonts            # Inter 4.1 (OFL-1.1) into the cache, sha256-checked
  python docs/charts/vocr_vs_lens.py data --readers eval/other_readers_20261028.json \\
      --vocr6 eval/vocr6_results_20261027.json --followup eval/vocr6_final_test_followup_20261027.json \\
      --vocr5-report eval/vocr5_report_20261016.json --vocr5-protocol eval/vocr5_protocol.json \\
      --lens-dialogue eval/vocr5_dialogue_20261016.json
  python docs/charts/vocr_vs_lens.py check            # the charted numbers, and copy.claims tested on them
  python docs/charts/vocr_vs_lens.py render           # -> docs/charts/vocr-vs-lens-{light,dark}{.svg,-2x.png}

Environment: docs/charts/requirements.lock (matplotlib; not a repo or CI dependency). The font is never
committed: `fonts` downloads it to $VOCR_CACHE (default ~/.cache/vita-vn-ocr) and the SVGs hold the glyphs
as paths, so they render without it.
"""
import argparse
import hashlib
import json
import os
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "vocr-vs-lens.json"
INTER_URL = "https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip"
INTER_SHA256 = "9883fdd4a49d4fb66bd8177ba6625ef9a64aa45899767dde3d36aa425756b11e"
FACES = {  # role -> file inside the zip
    "display": "extras/ttf/InterDisplay-Bold.ttf",
    "semibold": "extras/ttf/Inter-SemiBold.ttf",
    "medium": "extras/ttf/Inter-Medium.ttf",
    "regular": "extras/ttf/Inter-Regular.ttf",
}

# key, label, family: the chart's rows, top to bottom
SYSTEMS = (("VOCR6_S", "VOCR6 Small", "vocr6"), ("VOCR6_N", "VOCR6 Normal", "vocr6"),
           ("VOCR6_L", "VOCR6 Large", "vocr6"), ("VOCR5_S", "VOCR5 Small", "vocr5"),
           ("VOCR5_N", "VOCR5 Normal", "vocr5"), ("VOCR5_L", "VOCR5 Large", "vocr5"),
           ("Lens", "Google Lens", "other"), ("MeikiOCR", "MeikiOCR", "other"),
           ("PPOCRv6_small", "PP-OCRv6 small", "other"), ("PPOCRv5_mobile_int8", "PP-OCRv5 mobile int8", "other"),
           ("manga_ocr", "manga-ocr", "other"))
# key, label, platform: the chart's columns, left to right
SETS = (("clannad", "CLANNAD", "PS Vita"), ("vita-test", "vita-test", "PS Vita"), ("psp-test", "psp-test", "PSP"),
        ("pcsg00807", "Dance with Devils", "PS Vita"))
TIERS = {"S": "Small", "N": "Normal", "L": "Large"}

THEMES = {
    "light": {"bg": ("#FFFFFF", "#F2F2F7"), "text": "#1D1D1F", "dim": "#6E6E73", "faint": "#6E6E73",
              "accent": "#0071E3", "rule": "#D2D2D7",
              "vocr6": [("#8EC5FF", "#5AA9FF"), ("#3D8BFF", "#0A6CFF"), ("#7B61FF", "#3A3AFF")],
              "vocr5": "#B7C9E2", "other": "#9E9EA3", "other_text": "#6E6E73"},
    "dark": {"bg": ("#000000", "#141416"), "text": "#F5F5F7", "dim": "#A1A1A6", "faint": "#A1A1A6",
             "accent": "#2997FF", "rule": "#333336",
             "vocr6": [("#6BB6FF", "#3A93F5"), ("#2F80FF", "#0060F0"), ("#8C7BFF", "#4B3BFF")],
             "vocr5": "#44546E", "other": "#5E5E63", "other_text": "#A1A1A6"},
}


def cache_dir():
    return Path(os.environ.get("VOCR_CACHE", Path.home() / ".cache" / "vita-vn-ocr")) / "fonts" / "inter-4.1"


def cmd_fonts(_):
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    z = d / "Inter-4.1.zip"
    if not z.exists():
        urllib.request.urlretrieve(INTER_URL, z)
    got = hashlib.sha256(z.read_bytes()).hexdigest()
    if got != INTER_SHA256:
        z.unlink()
        raise SystemExit(f"Inter-4.1.zip sha256 {got} != {INTER_SHA256}")
    with zipfile.ZipFile(z) as f:
        for name in [*FACES.values(), "LICENSE.txt"]:
            (d / Path(name).name).write_bytes(f.read(name))
    print("fonts in", d)


def faces():
    from matplotlib import font_manager
    d = cache_dir()
    missing = [n for n in FACES.values() if not (d / Path(n).name).exists()]
    if missing:
        raise SystemExit("run `vocr_vs_lens.py fonts` first")
    return {role: font_manager.FontProperties(fname=str(d / Path(n).name)) for role, n in FACES.items()}


# ---------------------------------------------------------------- data

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


VOCR5_SHAPES = {"VOCR5_S": "s_w100v4987", "VOCR5_N": "n_w150v5631", "VOCR5_L": "l_lw300v5631"}


def resources(vocr6_path, report_path, protocol_path, readers_path):
    """Per system: the weights' bytes and form, and for VOCR the W8I arena and the estimated seconds per nominal
    3-line box at 333 / 444 MHz (one A9 core, no prefetch credit). VOCR6: the follow-up's gate fits, as the blobs ship
    (their head sections on, as Rubi runs them); VOCR5: its protocol's table for v0.3.0's plain blobs (#85's runtime);
    the other engines: their weight files as they read (eval/other_readers_*.json "sizes")."""
    out = {}
    g = load(vocr6_path)["follow_up_gate"]["tiers"]
    for t in TIERS:
        e = g[t]
        out[f"VOCR6_{t}"] = {"bytes": e["blob"]["bytes"], "form": "int8", "arena_w8i": e["arena_w8i_1024"],
                             "box_s_333": round(e["est_box_ms"]["shipped_333"] / 1000, 4),
                             "box_s_444": round(e["est_box_ms"]["shipped_444"] / 1000, 4),
                             "path": f"follow_up_gate.tiers.{t}"}
    rep = load(report_path)["shared"]["resources"]
    est = load(protocol_path)["estimates"]["table"]
    for arm, shape in VOCR5_SHAPES.items():
        e = est[shape]["#85 no-PLD"]
        assert est[shape]["bytes"] == rep[arm]["blob_bytes"], arm
        out[arm] = {"bytes": rep[arm]["blob_bytes"], "form": "int8", "arena_w8i": rep[arm]["arena_w8i"],
                    "box_s_333": e["333"][1], "box_s_444": e["444"][1], "path": f"estimates.table.{shape}"}
    for k, v in load(readers_path)["sizes"].items():
        out[k] = {"bytes": v["bytes"], "form": v["form"], "path": f"sizes.{k}"}
    out["Lens"] = {"bytes": None, "form": "cloud service", "path": None}
    return out


def mb(n):
    return f"{n / 1e6:.1f} MB" if n < 1e8 else f"{n / 1e6:.0f} MB"


def row_note(data, key):
    """The line under a row's name: size and form, and VOCR's estimated box time, filled from `resources`."""
    r = data["resources"][key]
    tmpl = data["copy"]["row_notes"].get(key) or data["copy"]["row_notes"]["default"]
    vals = {"MB": mb(r["bytes"]) if r.get("bytes") else "", "FORM": r.get("form", ""),
            "S333": f"{r['box_s_333']:.1f}" if r.get("box_s_333") else "",
            "S444": f"{r['box_s_444']:.1f}" if r.get("box_s_444") else ""}
    return tmpl.format(**vals)


def footnote_text(data):
    """The footnote paragraphs with their numbers filled from `sets` and `facts`."""
    F = data["facts"]
    sets = {x["key"]: x for x in data["sets"]}
    vals = {f"{tag}_LINES": f"{sets[k]['lines']:,}" for k, tag in
            (("clannad", "CL"), ("vita-test", "VT"), ("psp-test", "PT"), ("pcsg00807", "DD"))}
    vals.update(VT_GAMES=F["games_vita-test"]["games"], PT_GAMES=F["games_psp-test"]["games"])
    for key, tag in (("psp-test", "PT"), ("vita-test", "VT")):
        f = F[f"lens_furigana_{key}"]
        vals[f"{tag}_FUR_ERR"], vals[f"{tag}_ALL_ERR"] = f["furigana_errors"], f["all_errors"]
        vals[f"{tag}_FUR_LINES"] = f["furigana_lines"]
    return [x.format(**vals) for x in data["copy"]["footnote"]]


def value_of(v, path):
    return {"errors": v["errors"], "chars": v["chars"], "lines": v["lines"], "exact_lines": v["exact_lines"],
            "cer": v["cer"], "ci95": v["ci95"], "clusters": v["clusters"], "path": path}


def pair_of(p, path):
    return {"delta": p["delta"], "ci95": p["ci95"], "unit_ci95": p["unit_ci95"], "path": path}


RUBI_KEYS = {"clannad": ("clannad_lines", "dialogue"), "vita-test": ("vita_psp_fixed_lines", "vita-test|dialogue"),
             "psp-test": ("vita_psp_fixed_lines", "psp-test|dialogue"), "pcsg00807": ("pcsg00807_clean", None)}


def rubi_path(followup):
    """VOCR6 / VOCR5 as Rubi runs them (path b, 'shipped': beam + the 1 MB LM, the cascade, textnorm, the line
    filter) on the same lines: the follow-up's final read."""
    b = load(followup)["path_b"]
    out = {}
    for s, (k1, k2) in RUBI_KEYS.items():
        node = b[k1] if k2 is None else b[k1][k2]
        out[s] = {}
        for g in (6, 5):
            for t in TIERS:
                r = node["rates"][f"VOCR{g}:{t}:shipped"]
                p = f"path_b.{k1}" + (f".{k2}" if k2 else "") + f".rates.VOCR{g}:{t}:shipped"
                out[s][f"VOCR{g}_{t}"] = {"errors": r["errors"], "chars": r["chars"], "cer": r["cer"], "path": p}
    return out


def cmd_data(a):
    old = load(DATA) if DATA.exists() else {}
    R = load(a.readers)
    out = {"kind": "vocr-vs-ocrs-chart-data", "schema": 3,
           "metric": "dialogue CER: character edit distance / reference characters over the dialogue lines "
                     "(eval/metrics.norm), in percent; lower is better",
           "systems": [{"key": k, "label": lab, "family": fam} for k, lab, fam in SYSTEMS],
           "sets": [], "values": {}, "rubi_path": rubi_path(a.followup), "sources": {}, "facts": {}}
    for name, path in (("readers", a.readers), ("vocr6", a.vocr6), ("followup", a.followup),
                       ("vocr5_report", a.vocr5_report), ("vocr5_protocol", a.vocr5_protocol),
                       ("lens_dialogue", a.lens_dialogue)):
        out["sources"][name] = {"file": path, "sha256": sha256(path)}
    for key, label, platform in SETS:
        t = R["sets"][key]
        vals = {}
        for k, _, _ in SYSTEMS:
            v = t["values"].get(k)
            vals[k] = None if v is None else value_of(v, f"sets.{key}.values.{k}")
        for k, _, fam in SYSTEMS:
            if fam == "other" or vals[k] is None:
                continue
            vals[k]["vs"] = {o: pair_of(t["pairs"][f"{k}-{o}"], f"sets.{key}.pairs.{k}-{o}")
                             for o, _, f2 in SYSTEMS if f2 == "other" and f"{k}-{o}" in t["pairs"]}
        first = vals["VOCR6_N"]
        out["sets"].append({"key": key, "label": label, "platform": platform, "lines": first["lines"],
                            "chars": first["chars"], "cluster": t["cluster"], "clusters": t["clusters"]})
        out["values"][key] = vals
    for s in ("vita-test", "psp-test"):
        out["facts"][f"games_{s}"] = {"games": R["sets"][s]["clusters"], "path": f"sets.{s}.clusters"}
    d = load(a.lens_dialogue)
    for s in ("vita-test", "psp-test"):
        fur = d["arms"]["native"]["Lens"]["styles"]["furigana"][s]
        allv = d["arms"]["native"]["Lens"]["all"][s]
        out["facts"][f"lens_furigana_{s}"] = {
            "furigana_lines": fur["lines"], "furigana_errors": fur["errors"], "all_errors": allv["errors"],
            "all_lines": allv["lines"], "path": f"arms.native.Lens.styles.furigana.{s} / arms.native.Lens.all.{s}"}
    out["resources"] = resources(a.vocr6, a.vocr5_report, a.vocr5_protocol, a.readers)
    out["copy"] = old.get("copy", {"eyebrow": "", "headline": "", "subhead": "", "row_notes": {"default": ""},
                                   "not_measured": {}, "footnote": [], "claims": []})
    DATA.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", DATA)


# ---------------------------------------------------------------- render

def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def vgrad(top, bottom, n=256):
    import numpy as np
    t, b = np.array(hex2rgb(top)), np.array(hex2rgb(bottom))
    k = np.linspace(0, 1, n)[:, None]
    return (t * (1 - k) + b * k)[:, None, :]


def hgrad(left, right, n=256):
    import numpy as np
    t, b = np.array(hex2rgb(left)), np.array(hex2rgb(right))
    k = np.linspace(0, 1, n)[None, :, None]
    return t * (1 - k) + b * k


FOOT_PT, FOOT_LH = 12.5, 19.0               # footnote size (pt) and line height (px)
W, H = 1600, 1000                           # logical px; the PNG is rendered at 2x
X0 = 80                                     # left margin
LABEL_W = 345                               # the row-label column
ROW_PITCH, GROUP_GAP, BAR_H = 41.0, 14.0, 17.0


def wrap(text, prop, size_pt, width_px):
    """Greedy word wrap by the rendered width (TextPath extents are in points; the canvas is 100 px per inch)."""
    from matplotlib.textpath import TextPath
    out, cur = [], ""
    for word in text.split(" "):
        trial = f"{cur} {word}".strip()
        if cur and TextPath((0, 0), trial, prop=prop, size=size_pt).get_extents().width * 100 / 72 > width_px:
            out.append(cur)
            cur = word
        else:
            cur = trial
    return out + ([cur] if cur else [])


def fmt_pct(x):
    return f"{x:.2f}%" if x < 10 else f"{x:.1f}%"


def row_ys(systems, top):
    """Each row's centre line, top to bottom, with a gap between families."""
    ys, y, last = [], top, None
    for s in systems:
        if last is not None and s["family"] != last:
            y -= GROUP_GAP
        ys.append(y)
        y -= ROW_PITCH
        last = s["family"]
    return ys


def render(data, theme, out_stem, F):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    plt.rcParams["svg.fonttype"] = "path"
    plt.rcParams["svg.hashsalt"] = "vocr-vs-lens"
    T = THEMES[theme]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    bg = fig.add_axes([0, 0, 1, 1])
    bg.imshow(vgrad(*T["bg"]), extent=[0, W, 0, H], aspect="auto", interpolation="bicubic", zorder=0)
    bg.set_xlim(0, W)
    bg.set_ylim(0, H)
    bg.axis("off")
    c = data["copy"]

    def txt(x, y, s, face, size, color, **kw):
        bg.text(x, y, s, fontproperties=F[face], fontsize=size, color=color, zorder=5, **kw)

    txt(X0, H - 92, c.get("eyebrow", ""), "semibold", 15, T["accent"], va="baseline")
    txt(X0, H - 152, c.get("headline", ""), "display", 42, T["text"], va="baseline")
    txt(X0, H - 192, c.get("subhead", ""), "regular", 17, T["dim"], va="baseline")

    systems, sets, vals = data["systems"], data["sets"], data["values"]
    head_y = H - 262                         # set names; their platform and line count under them
    ys = row_ys(systems, head_y - 62)
    px0 = X0 + LABEL_W
    pw = (W - X0 - px0) / len(sets)          # one panel per set
    bar_max = pw - 96                        # room right of the longest bar for its value

    for si, s in enumerate(sets):
        x = px0 + si * pw
        txt(x, head_y, s["label"], "semibold", 19, T["text"], va="baseline")
        txt(x, head_y - 24, f'{s["platform"]} · {s["lines"]:,} lines', "regular", 13.5, T["dim"], va="baseline")
        col = [v["cer"] for v in vals[s["key"]].values() if v]
        scale = bar_max / max(col)
        for ri, sysm in enumerate(systems):
            y = ys[ri]
            v = vals[s["key"]].get(sysm["key"])
            if v is None:
                txt(x + 10, y, c.get("not_measured", {}).get(s["key"], {}).get(sysm["key"], "not measured"),
                    "regular", 12.5, T["faint"], va="center")
                continue
            w = max(v["cer"] * scale, BAR_H)
            patch = FancyBboxPatch((x, y - BAR_H / 2), w, BAR_H, boxstyle=f"round,pad=0,rounding_size={BAR_H / 2}",
                                   linewidth=0, facecolor="none", zorder=3)
            bg.add_patch(patch)
            fam = sysm["family"]
            if fam == "vocr6":
                tier = "SNL".index(sysm["key"][-1])
                im = bg.imshow(hgrad(*T["vocr6"][tier]), extent=[x, x + w, y - BAR_H / 2, y + BAR_H / 2],
                               aspect="auto", interpolation="bicubic", zorder=3)
                im.set_clip_path(patch)
            else:
                patch.set_facecolor(T["vocr5"] if fam == "vocr5" else T["other"])
            mark = c.get("marks", {}).get(s["key"], {}).get(sysm["key"], "")
            txt(x + w + 8, y, fmt_pct(v["cer"]) + mark, "semibold" if fam == "vocr6" else "medium",
                14.5 if fam == "vocr6" else 13.5, T["text"] if fam != "other" else T["other_text"], va="center")
    for ri, sysm in enumerate(systems):
        y = ys[ri]
        fam = sysm["family"]
        txt(X0, y + 4, sysm["label"], "semibold" if fam == "vocr6" else "medium", 16 if fam == "vocr6" else 15,
            T["text"] if fam != "other" else T["other_text"], va="baseline")
        txt(X0, y - 13, row_note(data, sysm["key"]), "regular", 11.5, T["dim"], va="baseline")
    bg.plot([X0, W - X0], [ys[-1] - ROW_PITCH / 2 - 4] * 2, color=T["rule"], lw=1.0, zorder=2)

    lines = [ln for para in footnote_text(data) for ln in wrap(para, F["regular"], FOOT_PT, W - 2 * X0)]
    y = 30 + (len(lines) - 1) * FOOT_LH
    for line in lines:
        txt(X0, y, line, "regular", FOOT_PT, T["faint"], va="baseline")
        y -= FOOT_LH

    fig.savefig(f"{out_stem}.svg", format="svg", metadata={"Date": None})
    fig.savefig(f"{out_stem}-2x.png", format="png", dpi=200, metadata={"Software": None},
                pil_kwargs={"optimize": True})
    plt.close(fig)
    for ext in (".svg", "-2x.png"):
        f = Path(f"{out_stem}{ext}")
        print(f"{f}: {f.stat().st_size / 1024:.1f} KB")


def cmd_render(a):
    data = load(a.data)
    problems = check(data)
    if problems:
        raise SystemExit("claims not supported by the data:\n" + "\n".join(problems))
    F = faces()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for theme in a.themes.split(","):
        render(data, theme, str(out / f"vocr-vs-lens-{theme}"), F)
    if Path(a.data).resolve() == DATA and out.resolve() == HERE:
        data["rendered"] = {"inputs_sha256": inputs_sha256(data),
                            "files": {f.name: sha256(f) for f in sorted(out.glob("vocr-vs-lens-*"))
                                      if f.suffix in (".svg", ".png")}}
        DATA.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def inputs_sha256(data):
    """What the images are drawn from: the JSON without its own 'rendered' record."""
    d = {k: v for k, v in data.items() if k != "rendered"}
    return hashlib.sha256(json.dumps(d, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- claims

OPS = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}


def check(data):
    """Each entry of copy.claims is {"set", "system", "op", "than"} on dialogue CER point estimates (than: a system
    key or a number in percent), optionally "ci": true, which also needs the pair's 95% interval (values[set][system]
    ["vs"][than]["ci95"], in points: the cluster bootstrap's) entirely on the claimed side of 0. Returns the claims
    the data does not support."""
    bad = []
    for c in data.get("copy", {}).get("claims", []):
        vals = data["values"].get(c["set"]) or {}
        a = vals.get(c["system"])
        b = c["than"] if isinstance(c["than"], (int, float)) else (vals.get(c["than"]) or {}).get("cer")
        if a is None or b is None:
            bad.append(f"{c}: not measured")
            continue
        if not OPS[c["op"]](a["cer"], b):
            bad.append(f"{c}: {a['cer']} {c['op']} {b} is false")
            continue
        if c.get("ci"):
            ci = ((a.get("vs") or {}).get(c["than"]) or {}).get("ci95")
            side = (lambda lo, hi: hi < 0) if c["op"] in ("<", "<=") else (lambda lo, hi: lo > 0)
            if not ci or not side(*ci):
                bad.append(f"{c}: pair interval {ci} does not exclude 0 on the claimed side")
    return bad


def cmd_check(a):
    data = load(a.data)
    for s in data["sets"]:
        row = [f"{s['label']:>18} ({s['lines']:,} lines)"]
        for sysm in data["systems"]:
            v = data["values"][s["key"]].get(sysm["key"])
            row.append(f"{sysm['label']}: " + ("not measured" if v is None else f"{v['cer']:.2f}%"))
        print(" | ".join(row))
    bad = check(data)
    print("claims: " + ("all supported" if not bad else "NOT SUPPORTED:\n  " + "\n  ".join(bad)))
    raise SystemExit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fonts")
    d = sub.add_parser("data")
    d.add_argument("--readers", required=True, help="eval/other_readers_*.json (every system's dialogue CER)")
    d.add_argument("--vocr6", required=True, help="eval/vocr6_results_*.json (the follow-up's gate fits)")
    d.add_argument("--followup", required=True, help="eval/vocr6_final_test_followup_*.json (Rubi's path)")
    d.add_argument("--vocr5-report", required=True, help="VOCR5's locked report JSON (int8 bytes, W8I arenas)")
    d.add_argument("--vocr5-protocol", required=True, help="VOCR5's protocol JSON (the box-time estimate table)")
    d.add_argument("--lens-dialogue", required=True, help="VOCR5's dialogue tables (Lens's furigana counts)")
    r = sub.add_parser("render")
    r.add_argument("--data", default=str(DATA))
    r.add_argument("--out-dir", default=str(HERE))
    r.add_argument("--themes", default="light,dark")
    c = sub.add_parser("check", help="print the charted numbers and test copy.claims against them")
    c.add_argument("--data", default=str(DATA))
    a = ap.parse_args()
    {"fonts": cmd_fonts, "data": cmd_data, "render": cmd_render, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
