#!/usr/bin/env python3
"""The README's "VOCR vs Google Lens" chart: dialogue CER per protected set, light and dark, SVG + 2x PNG.

Every number comes from docs/charts/vocr-vs-lens.json, which `data` builds from the committed score files
(VOCR5's dialogue tables and its CLANNAD comparison) and which records where each number came from. The
headline and the other words live in that JSON's "copy" object (kept when `data` runs again). Nothing here
scores anything.

  python docs/charts/vocr_vs_lens.py fonts                 # Inter 4.1 (OFL-1.1) into the cache, sha256-checked
  python docs/charts/vocr_vs_lens.py data --dialogue eval/vocr5_dialogue_20261016.json \\
      --clannad eval/vocr5_clannad_20261016.json --arms 'VOCR5_S=VOCR5 Small,VOCR5_N=VOCR5 Normal,VOCR5_L=VOCR5 Large'
  python docs/charts/vocr_vs_lens.py check                 # the charted numbers, and copy.claims tested on them
  python docs/charts/vocr_vs_lens.py render                # -> docs/charts/vocr-vs-lens-{light,dark}{.svg,-2x.png}

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

THEMES = {
    "light": {"bg": ("#FFFFFF", "#F2F2F7"), "text": "#1D1D1F", "dim": "#6E6E73", "faint": "#86868B",
              "accent": "#0071E3", "lens": "#C7C7CC", "lens_text": "#6E6E73", "rule": "#D2D2D7",
              "tiers": [("#8EC5FF", "#5AA9FF"), ("#3D8BFF", "#0A6CFF"), ("#7B61FF", "#3A3AFF")]},
    "dark": {"bg": ("#000000", "#141416"), "text": "#F5F5F7", "dim": "#A1A1A6", "faint": "#86868B",
             "accent": "#2997FF", "lens": "#3A3A3C", "lens_text": "#A1A1A6", "rule": "#333336",
             "tiers": [("#6BB6FF", "#3A93F5"), ("#2F80FF", "#0060F0"), ("#8C7BFF", "#4B3BFF")]},
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


def rate(errors, chars):
    return round(100.0 * errors / chars, 4)


def dialogue_value(dlg, arm, set_name):
    """One arm's dialogue counts (lines without the 'name' style) on vita-test or psp-test from VOCR5's dialogue
    tables (eval/vocr5_dialogue_*.json): console polarity for our models, screen polarity for Lens, which reads
    the screen."""
    view = "native" if arm == "Lens" else "console"
    v = dlg["arms"][view][arm]["dialogue"][set_name]
    return {"errors": v["errors"], "chars": v["chars"], "lines": v["lines"], "exact_lines": v["exact"],
            "cer": rate(v["errors"], v["chars"]), "exact_pct": round(100.0 * v["exact"] / v["lines"], 2),
            "path": f"arms.{view}.{arm}.dialogue.{set_name}"}


def dialogue_pair(dlg, arm, set_name):
    """ARM - Lens on dialogue, CER points: paired bootstrap over lines and over games (VN clusters)."""
    p = dlg["pairs"][f"{arm}-Lens"]["console_vs_lens_native"]["dialogue"][set_name]
    return {"delta": round(p["delta"], 4), "ci95": [round(x, 4) for x in p["lines_ci95"]],
            "vn_ci95": [round(x, 4) for x in p["vn_ci95"]],
            "path": f"pairs.{arm}-Lens.console_vs_lens_native.dialogue.{set_name}"}


def cmd_data(a):
    arms = dict(x.split("=", 1) for x in a.arms.split(","))
    old = json.loads(DATA.read_text()) if DATA.exists() else {}
    out = {"kind": "vocr-vs-lens-chart-data", "schema": 2,
           "metric": "dialogue CER: character edit distance / reference characters over the lines without the "
                     "'name' style (eval/metrics.norm), in percent; lower is better",
           "systems": [{"key": k, "label": v} for k, v in arms.items()] + [{"key": "Lens", "label": "Google Lens"}],
           "sets": [], "values": {}, "sources": {}, "facts": {}}
    if a.clannad:
        c = json.loads(Path(a.clannad).read_text())
        out["sources"]["clannad"] = {"file": a.clannad, "sha256": sha256(a.clannad)}
        S = c["scores"]["systems"]
        vals = {}
        for k in [*arms, "Lens"]:
            v = S[k]["not_name"]
            vals[k] = {"errors": v["errors"], "chars": v["chars"], "lines": v["lines"], "cer": v["cer"],
                       "cer_ci95": v["cer_ci95"], "exact_pct": v["exact"], "path": f"scores.systems.{k}.not_name"}
        out["sets"].append({"key": "clannad", "label": "CLANNAD", "platform": "PS Vita",
                            "lines": vals["Lens"]["lines"], "chars": vals["Lens"]["chars"]})
        out["values"]["clannad"] = vals
    if a.dialogue:
        d = json.loads(Path(a.dialogue).read_text())
        out["sources"]["dialogue"] = {"file": a.dialogue, "sha256": sha256(a.dialogue)}
        for set_name, platform in (("vita-test", "PS Vita"), ("psp-test", "PSP")):
            vals = {k: dialogue_value(d, k, set_name) for k in [*arms, "Lens"]}
            for k in arms:
                vals[k]["vs"] = {"Lens": dialogue_pair(d, k, set_name)}
            out["sets"].append({"key": set_name, "label": set_name, "platform": platform,
                                "lines": vals["Lens"]["lines"], "chars": vals["Lens"]["chars"]})
            out["values"][set_name] = vals
            fur = d["arms"]["native"]["Lens"]["styles"]["furigana"][set_name]
            allv = d["arms"]["native"]["Lens"]["all"][set_name]
            out["facts"][f"lens_furigana_{set_name}"] = {
                "furigana_lines": fur["lines"], "furigana_errors": fur["errors"], "all_errors": allv["errors"],
                "all_lines": allv["lines"],
                "path": f"arms.native.Lens.styles.furigana.{set_name} / arms.native.Lens.all.{set_name}"}
    out["resources"] = json.loads(a.resources) if a.resources else old.get("resources", {})
    out["copy"] = old.get("copy", {"eyebrow": "", "headline": "", "subhead": "", "panel": {}, "footnote": [],
                                   "claims": []})
    DATA.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
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


FOOT_PT, FOOT_LH = 11.5, 18.0               # footnote size (pt) and line height (px)


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


def render(data, theme, out_stem, F):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    plt.rcParams["svg.fonttype"] = "path"
    plt.rcParams["svg.hashsalt"] = "vocr-vs-lens"
    T = THEMES[theme]
    W, H = 1600, 900                       # logical px; the PNG is rendered at 2x
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    bg = fig.add_axes([0, 0, 1, 1])
    bg.imshow(vgrad(*T["bg"]), extent=[0, W, 0, H], aspect="auto", interpolation="bicubic", zorder=0)
    bg.set_xlim(0, W)
    bg.set_ylim(0, H)
    bg.axis("off")
    c = data["copy"]

    def txt(x, y, s, face, size, color, **kw):
        bg.text(x, y, s, fontproperties=F[face], fontsize=size, color=color, zorder=5, **kw)

    X0 = 96
    txt(X0, 806, c.get("eyebrow", ""), "semibold", 15, T["accent"], va="baseline")
    txt(X0, 742, c.get("headline", ""), "display", 44, T["text"], va="baseline")
    txt(X0, 700, c.get("subhead", ""), "regular", 17, T["dim"], va="baseline")

    # --- the bars: one group per set, one bar per system (VOCR tiers, then Lens)
    systems = data["systems"]
    sets = data["sets"]
    vals = data["values"]
    cmax = max(v["cer"] for s in sets for v in vals[s["key"]].values() if v)
    top_px, base = 340.0, 262.0            # bar area: base line y and the tallest bar's height
    scale = top_px / (cmax * 1.08)
    gx0, gx1 = X0, 1066                    # bar groups span this x range; the panel sits right of it
    gw = (gx1 - gx0) / len(sets)
    bw, gap = 56.0, 24.0
    for gi, s in enumerate(sets):
        n = len(systems)
        span = n * bw + (n - 1) * gap
        x = gx0 + gi * gw + (gw - span) / 2
        for si, sysm in enumerate(systems):
            v = vals[s["key"]].get(sysm["key"])
            bx = x + si * (bw + gap)
            lens = sysm["key"] == "Lens"
            label = "Lens" if lens else sysm["label"].split()[-1]
            txt(bx + bw / 2, base - 28, label, "medium", 14, T["dim"], ha="center", va="baseline")
            if v is None:
                txt(bx + bw / 2, base + 14, "not\nmeasured", "regular", 12.5, T["faint"], ha="center",
                    va="bottom", linespacing=1.15)
                bg.plot([bx + 6, bx + bw - 6], [base, base], color=T["rule"], lw=1.2, zorder=3)
                continue
            h = max(v["cer"] * scale, 3.0)
            r = min(12.0, bw / 2, h / 2)
            patch = FancyBboxPatch((bx, base), bw, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                   linewidth=0, facecolor="none", zorder=3)
            bg.add_patch(patch)
            if lens:
                patch.set_facecolor(T["lens"])
            else:
                tier = min(si, len(T["tiers"]) - 1)
                im = bg.imshow(vgrad(*T["tiers"][tier]), extent=[bx, bx + bw, base, base + h], aspect="auto",
                               interpolation="bicubic", zorder=3)
                im.set_clip_path(patch)
            mark = c.get("marks", {}).get(s["key"], {}).get(sysm["key"], "")
            txt(bx + bw / 2, base + h + 10, fmt_pct(v["cer"]) + mark, "semibold", 16,
                T["lens_text"] if lens else T["text"], ha="center", va="bottom")
        cx = gx0 + gi * gw + gw / 2
        txt(cx, base - 66, s["label"], "semibold", 20, T["text"], ha="center", va="baseline")
        txt(cx, base - 92, f'{s["platform"]} · {s["lines"]:,} lines', "regular", 14, T["dim"], ha="center",
            va="baseline")
    bg.plot([gx0, gx1], [base, base], color=T["rule"], lw=1.0, zorder=2)

    # --- the right panel: where it runs
    p = c.get("panel", {})
    px = 1134
    bg.plot([px - 36, px - 36], [base - 90, base + top_px + 10], color=T["rule"], lw=1.0, zorder=2)
    txt(px, base + top_px - 18, p.get("kicker", ""), "semibold", 15, T["accent"], va="baseline")
    txt(px, base + top_px - 92, p.get("big", ""), "display", 58, T["text"], va="baseline")
    txt(px, base + top_px - 128, p.get("big_sub", ""), "regular", 15, T["dim"], va="baseline")
    yy = base + top_px - 180
    for line in p.get("lines", []):
        txt(px, yy, line, "medium", 14.5, T["text"], va="baseline")
        yy -= 27
    yy -= 14
    for line in p.get("lens_lines", []):
        txt(px, yy, line, "regular", 13.5, T["lens_text"], va="baseline")
        yy -= 23

    # --- footnote: each paragraph wrapped to the content width, the block set on the bottom margin
    lines = [ln for para in c.get("footnote", []) for ln in wrap(para, F["regular"], FOOT_PT, W - 2 * X0)]
    y = 34 + (len(lines) - 1) * FOOT_LH
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
    data = json.loads(Path(a.data).read_text())
    problems = check(data)
    if problems:
        raise SystemExit("claims not supported by the data:\n" + "\n".join(problems))
    F = faces()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for theme in a.themes.split(","):
        render(data, theme, str(out / f"vocr-vs-lens-{theme}"), F)


# ---------------------------------------------------------------- claims

OPS = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}


def check(data):
    """Each entry of copy.claims is {"set", "system", "op", "than"} on dialogue CER point estimates (than: a system
    key or a number in percent), optionally "ci": true, which also needs the pair's 95% interval (values[set][system]
    ["vs"][than]["ci95"], in points) entirely on the claimed side of 0. Returns the claims the data does not support."""
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
    data = json.loads(Path(a.data).read_text())
    for s in data["sets"]:
        row = [f"{s['label']:>10} ({s['lines']:,} lines)"]
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
    d.add_argument("--dialogue", help="VOCR5's dialogue tables JSON (vita-test, psp-test, Lens)")
    d.add_argument("--clannad", help="the CLANNAD comparison JSON (eval/clannad_refs.py score output)")
    d.add_argument("--arms", required=True, help="ARM=Label,... in bar order, e.g. VOCR5_S=VOCR5 Small,...")
    d.add_argument("--resources", help="JSON object: per-arm bytes, arena and time estimates for the panel")
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
