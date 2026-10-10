#!/usr/bin/env python3
"""Tables for psp/README.md from a psp/run_all.sh output directory.

  summarize.py OUT        markdown on stdout: parity, the console self-check and PPSSPP's
                          emulated time per build / mode / set / tile, memory, eval CER
"""
import json
import statistics
import sys
from pathlib import Path


def pct(v, n):
    return f"{v}/{n}"


def parity(out):
    rows = ["| build | set | mode | lines | bit-exact vs vocr_ref | logits within tol. | same text as PyTorch "
            "(+ near ties) | tiles | VOCR_INVERT | arena plan | pass |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(out.glob("parity_*_*.json")):
        build, s = p.stem.split("_", 2)[1:]
        d = json.loads(p.read_text())["summary"]
        n = d["lines"]
        for m in ("f32", "w8f", "w8i"):
            if m not in d:
                continue
            x = d[m]
            rows.append(f"| {build} | {s} | {m} | {n} | {pct(x['exact'], n)} | {pct(x['logits_ok'], n)} | "
                        f"{x['same_text']} (+{x['text_diff_tie_only']}) | {pct(x['tiles_ok'], x['tiles_checked'])} | "
                        f"{pct(x['invert_ok'], x['invert_checked'])} | {d['arena_ok']} | {d['pass']} |")
    return "\n".join(rows)


def per_line_us(results):
    us = []
    for ln in results.read_text(encoding="utf-8").splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.split("\t")
        us.append((int(f[1]), int(f[3])))       # frames, us
    return us


def bundles(out):
    rows = ["| build | mode | set | tile/head | arena B | same as host (text, logits) | emulated ms/line: mean | "
            "median | p90 | max | us/frame |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in sorted((out / "bundles").glob("*")):
        st, res = d / "stats.json", d / "results.txt"
        if not st.exists() or not res.exists():
            rows.append(f"| {d.name} | missing |")
            continue
        s = json.loads(st.read_text())
        us = [u for _, u in per_line_us(res)]
        q = sorted(us)
        build, mode, set_, _ = d.name.split("_")
        rows.append(f"| {build} | {mode} | {set_} | "
                    f"{s['tile']}/{s['head']} | {s['arena_bytes']} | {s['text_ok']}/{s['lines']}, "
                    f"{s['logits_ok']}/{s['lines']} | {s['us_total'] / 1e3 / s['lines']:.0f} | "
                    f"{statistics.median(q) / 1e3:.0f} | {q[int(0.9 * (len(q) - 1))] / 1e3:.0f} | "
                    f"{q[-1] / 1e3:.0f} | {s['us_total'] / max(s['frames'], 1):.0f} |")
    return "\n".join(rows)


def memory(out):
    rows = []
    b = out / "bundles"
    for d in sorted(b.glob("scalar_*_psp-test_t4")) + sorted(b.glob("scalar_*_psp-test_t16")):
        s = json.loads((d / "stats.json").read_text())
        rows.append(f"| {s['mode']} | {s['tile']}/{s['head']} | {s['blob_bytes']} | {s['arena_at']['256']} / "
                    f"{s['arena_at']['640']} / {s['arena_at']['1024']} | {s['sizeof_model']} + {s['sizeof_ctx']} | "
                    f"{s['stack_used']} of {s['stack_bytes']} | {s['user_free_start']} | {s['user_free_loaded']} | "
                    f"{s['crop_buffer_bytes']} + {s['logits_buffer_bytes']} |")
    head = ["| mode | tile/head | blob B | arena B at max width 256 / 640 / 1024 | vocr_model + vocr_ctx B | "
            "vocr thread stack used B | user memory free at start B | free with model, arena and test buffers B | "
            "test-only buffers (crop + logits) B |", "|---|---|---|---|---|---|---|---|---|"]
    return "\n".join(head + rows)


def evals(out):
    rows = ["| run | set | lines | CER | exact | identical to host |", "|---|---|---|---|---|---|"]
    for p in sorted(out.glob("eval_psp_*.json")):
        mode = p.stem.split("_")[-1]
        psp = json.loads(p.read_text())
        host_p = out / f"eval_host_{mode}.json"
        host = json.loads(host_p.read_text()) if host_p.exists() else {"sets": {}}
        for s, r in psp["sets"].items():
            a = r["all"]
            hl = host["sets"].get(s, {}).get("lines", [])
            same = sum(x["pred"] == y["pred"] for x, y in zip(r["lines"], hl)) if hl else None
            ha = host["sets"].get(s, {}).get("all", {})
            rows.append(f"| PSP build in PPSSPP, {mode} | {s} | {a.get('lines')} | {100 * a['cer']:.2f}% | "
                        f"{100 * a['exact']:.2f}% | {same}/{len(hl)} (host CER {100 * ha.get('cer', 0):.2f}%) |")
    return "\n".join(rows)


def main():
    out = Path(sys.argv[1])
    print("## Parity (runtime/tests/parity.py, C side = the PSP build in PPSSPPHeadless)\n")
    print(parity(out))
    print("\n## Console self-check and PPSSPP emulated time (333 MHz; not console time)\n")
    print(bundles(out))
    print("\n## Memory (PSP build, PPSSPP, PSP-1000 24 MB user partition)\n")
    print(memory(out))
    print("\n## eval/run.py through the PSP build\n")
    print(evals(out))
    size = out / "size.txt"
    if size.exists():
        print("\n## Code size and stack (psp/Makefile size)\n\n```\n" + size.read_text().strip() + "\n```")


if __name__ == "__main__":
    main()
