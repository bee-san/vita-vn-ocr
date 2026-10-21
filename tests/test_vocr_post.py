"""The counter post-processing (runtime/vocr_post.c, its twin eval/vocr_post.py; issue #109) on generated data only:
the text rule's corpus-derived contexts, the re-decode on hand-made top-K lines, and the C library against the
twin on a random-init model's real top-K output. No development, test or private data is read here.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "eval", ROOT / "runtime", ROOT / "train", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import vocr_post as P  # noqa: E402

# (in, out): the owner's console reading, every covered context, and the corpora's own ッ before a counter
# character (docs/counters.md), which stay
RULE = [
    ("ルッ月です」", "ルヶ月です」"), ("九ッ月", "九ヶ月"), ("ッ月", "ヶ月"), ("3ッ月と５ッ年", "3ヶ月と５ヶ年"),
    ("十ッ国、数ッ条、何ッ国", "十ヶ国、数ヶ条、何ヶ国"), ("三ッ所、幾ッ所、百ッ所", "三ヶ所、幾ヶ所、百ヶ所"),
    ("一ッ所に立つてちや", "一ッ所に立つてちや"),           # Aozora: ひとつところ
    ("「えッ国上山。", "「えッ国上山。"), ("『国の為ッ国の為ッて", "『国の為ッ国の為ッて"),
    ("生命に別ッ条ない", "生命に別ッ条ない"), ("三ッ日月。", "三ッ日月。"), ("イマドキッ日曜日", "イマドキッ日曜日"),
    ("じゃッ日曜日", "じゃッ日曜日"), ("アッ日の丸", "アッ日の丸"), ("えーッ町長さん", "えーッ町長さん"),
    ("ウラシペッ村", "ウラシペッ村"), ("四ッ谷", "四ッ谷"), ("三ッ村と二ッ町", "三ッ村と二ッ町"),
    ("九っ月、九ツ月、九ヵ月、九ケ月、九箇月", "九っ月、九ツ月、九ヵ月、九ケ月、九箇月"), ("ベッド", "ベッド"),
    ("三\nッ国", "三\nッ国"), ("九ッ\n月", "九ッ\n月"), ("九ッッ月", "九ッヶ月"), ("", ""),
]


@pytest.mark.parametrize("text,want", RULE)
def test_rule_twin(text, want):
    out, n = P.counters(text)
    assert out == want and n == sum(a != b for a, b in zip(text, want))
    assert len(out) == len(text)                              # the text never grows


def test_rule_counts_doc_matches_code():
    """docs/counters.md's count table has a row for every counter character, and the contexts it states are the
    ones the twin applies."""
    doc = (ROOT / "docs" / "counters.md").read_text(encoding="utf-8")
    for ch in "月年国条所日村町":
        assert f"\n| {ch} | " in doc
    assert "一ッ所" in doc and "ッ月 and ッ年 occur nowhere" in doc
    assert P.counters("ルッ月")[1] == P.counters("ルッ年")[1] == 1
    assert P.counters("二ッ所")[1] == 1 and P.counters("一ッ所")[1] == 0 and P.counters("ルッ国")[1] == 0


@pytest.fixture(scope="module")
def lib():
    from pyvocr import load_lib
    return load_lib()


def test_rule_c_equals_twin(lib):
    from pyvocr import post_counters
    rng = random.Random(1)
    alphabet = list("ッヶヵ月所年国条日一二三九十何数幾3５ルえ為別\n「」あ")
    for text, want in RULE:
        assert post_counters(text, lib) == P.counters(text)
    for _ in range(2000):
        t = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 10)))
        assert post_counters(t, lib) == P.counters(t)


# a tiny vocabulary: class 0 the blank, then these
TINY = ["", "「", "」", "九", "ル", "ッ", "ヶ", "月", "で", "す", "一", "ー", "所", "二", "ニ", "し", "ヵ", "国"]
C = {t: i for i, t in enumerate(TINY)}


def owner_line():
    """ルッ月です」 with 九 three nats under ル, and 「 five nats under the blank before it (the C self-test's)."""
    rows = [("", 10, "「", 2, "」", 1, "九", 0), ("", 10, "「", 5, "九", 4, "ル", 1),
            ("ル", 9, "九", 6, "", 5, "し", 1), ("ル", 9, "", 7, "九", 5.5, "し", 1),
            ("", 10, "ル", 3, "九", 2, "ッ", 1), ("ッ", 8, "ヶ", 6, "", 5, "ヵ", 1),
            ("", 10, "ッ", 2, "ヶ", 1, "で", 0), ("月", 9, "", 4, "す", 1, "で", 0),
            ("", 10, "月", 3, "で", 1, "す", 0), ("で", 9, "", 4, "す", 1, "」", 0),
            ("", 10, "で", 2, "す", 1, "」", 0), ("す", 9, "", 4, "で", 1, "」", 0),
            ("", 10, "す", 2, "」", 1, "で", 0), ("」", 9, "", 3, "す", 1, "で", 0),
            ("", 10, "」", 2, "す", 1, "で", 0)]
    tc = np.array([[C[r[k]] for k in (0, 2, 4, 6)] for r in rows], np.uint16)
    tv = np.array([[r[k] for k in (1, 3, 5, 7)] for r in rows], np.float32)
    return tv, tc


def test_redecode_twin_thresholds():
    tv, tc = owner_line()
    assert P.greedy(TINY, tc) == "ルッ月です」"
    same, st = P.redecode(TINY, tv, tc, 0, 0)
    assert (same == tc).all() and st["edits"] == 0 and st["contexts"] == 0       # off: not even read
    # 九 at its best frame (3 nats) and the blank at the other (2 nats): 5 nats = 5120 Q10
    t2, st = P.redecode(TINY, tv, tc, 4999, 0)
    assert st == {"edits": 0, "contexts": 1, "numerals": 0, "brackets": 0} and (t2 == tc).all()
    t2, st = P.redecode(TINY, tv, tc, 5000, 0)
    assert st["numerals"] == 1 and P.greedy(TINY, t2) == "九ッ月です」"
    assert P.counters(P.greedy(TINY, t2))[0] == "九ヶ月です」"
    t2, st = P.redecode(TINY, tv, tc, 5000, 5000)                 # and the 「 five nats under the blank
    assert st["brackets"] == 1 and P.greedy(TINY, t2) == "「九ッ月です」"
    t2, st = P.redecode(TINY, tv, tc, 5000, 4999)
    assert st["brackets"] == 0
    assert (np.sort(t2, axis=1) == np.sort(tc, axis=1)).all()     # an edit only reorders a frame's classes
    t2, st = P.redecode(TINY, tv[:13], tc[:13], 0, 64000)         # no closing bracket on the line: no 「
    assert st["contexts"] == 1 and st["edits"] == 0


def test_redecode_no_context_for_numeral_or_bracket_x():
    tv, tc = owner_line()
    tc = tc.copy()
    tc[2] = [C["九"], C["ル"], 0, C["し"]]
    tc[3] = [C["九"], 0, C["ル"], C["し"]]
    _, st = P.redecode(TINY, tv, tc, 64000, 64000)
    assert st["contexts"] == 0


def test_redecode_merge_guard():
    """A numeral right after the same numeral would merge into one (CTC): it takes a later frame."""
    rows = [("二", 9, "", 5, "ニ", 1, "「", 0), ("ニ", 9, "二", 8.5, "", 5, "「", 0),
            ("", 9, "ニ", 3, "二", 2, "「", 0), ("ッ", 8, "ヶ", 6, "", 5, "ヵ", 1), ("所", 9, "", 4, "す", 1, "で", 0)]
    tc = np.array([[C[r[k]] for k in (0, 2, 4, 6)] for r in rows], np.uint16)
    tv = np.array([[r[k] for k in (1, 3, 5, 7)] for r in rows], np.float32)
    t2, st = P.redecode(TINY, tv, tc, 64000, 0)
    assert st["numerals"] == 1 and P.greedy(TINY, t2) == "二二ッ所" and t2[1, 0] == 0 and t2[2, 0] == C["二"]
    assert P.counters(P.greedy(TINY, t2))[0] == "二二ヶ所"


def test_redecode_bad_input():
    tv, tc = owner_line()
    with pytest.raises(ValueError):
        P.redecode(TINY, tv, tc, -1, 0)
    with pytest.raises(ValueError):
        P.redecode(TINY, tv, tc, 0, P.MAX_TAU + 1)
    bad = tc.copy()
    bad[0, 1] = len(TINY)
    with pytest.raises(ValueError):
        P.redecode(TINY, tv, bad, 1000, 0)


VOCAB = list("「」九ルッヶヵ月所年国条一二三ーニしですケ")


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    import torch
    from pyvocr import Vocr
    from torch import nn

    import export
    from model import TinyVNOCR
    torch.manual_seed(3)
    m = TinyVNOCR(len(VOCAB) + 1, width=0.25).eval()
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_var.uniform_(0.5, 2.0)
                mod.running_mean.uniform_(-0.2, 0.2)
                mod.weight.uniform_(0.5, 1.5)
                mod.bias.uniform_(-0.5, 0.5)
            if isinstance(mod, nn.Conv2d):
                nn.init.normal_(mod.weight, 0, (2.0 / mod.weight[0].numel()) ** 0.5)
    blob, _ = export.build_blob(export.model_ops(m, 32), VOCAB, 32, True, 1024, b"")
    path = tmp_path_factory.mktemp("post") / "m.vocr"
    path.write_bytes(blob)
    return Vocr(path, mode="w8i", max_width=1024)


def test_c_equals_twin_on_model_output(model):
    """On a random-init model's top-K: the C greedy text of the extras is vocr_run()'s, and the C re-decode, its
    greedy text and the twin agree on every line and threshold (contexts forced in by swapping classes)."""
    vocab = [model.class_text(c) for c in range(model.n_classes)]
    rng = np.random.default_rng(0)
    seen = edited = 0
    for i in range(60):
        img = rng.integers(0, 256, (32, int(rng.integers(40, 400))), dtype=np.uint8)
        k = int(rng.choice([1, 4, 8, 16]))
        out = model.run_topk(img, k)
        assert P.greedy(vocab, out["topc"]) == out["text"]
        assert model.redecode(out, 0, 0)[0]["text"] == out["text"]
        if i % 2 and out["frames"] >= 6 and k >= 4:
            # force X + ッ + 月 into the top-1 at three emitting frames: a context on purpose
            tc = out["topc"].copy()
            f0 = int(rng.integers(0, out["frames"] - 5))
            for f, ch in zip((f0, f0 + 2, f0 + 4), ("ル", "ッ", "月")):
                c = vocab.index(ch)
                j = np.where(tc[f] == c)[0]
                if len(j):
                    tc[f, 0], tc[f, j[0]] = tc[f, j[0]], tc[f, 0]
                else:
                    tc[f, 0] = c
            out = dict(out, topc=tc)
        for tau, opn in ((0, 0), (1000, 0), (8000, 8000), (64000, 64000), (3000, 500)):
            res, st = model.redecode(out, tau, opn)
            t2, st2 = P.redecode(vocab, out["topv"], out["topc"], tau, opn)
            assert (res["topc"] == t2).all() and st == st2
            assert res["text"] == P.greedy(vocab, t2) or (st["edits"] == 0 and res["text"] == out["text"])
            seen += st["contexts"]
            edited += st["edits"]
    assert seen > 0                                            # the forced contexts were read


def test_c_bad_input(model):
    from pyvocr import VocrError
    out = model.run_topk(np.full((32, 80), 128, np.uint8), 4)
    with pytest.raises(VocrError):
        model.redecode(out, -1, 0)
    with pytest.raises(VocrError):
        model.redecode(out, 0, 64001)


# ---------------------------------------------------------------- the edge continuation (vocr_lines_widen.c)

def _strip(w=60, h=24, runs=(), bg=40, ink=230):
    s = np.full((h, w), bg, np.uint8)
    for a, b in runs:
        s[4:20, a:b] = ink
    return s


def test_widen_twin():
    import lines_widen as LW
    s = _strip(runs=((8, 16), (20, 30)))
    assert LW.widen(s, 12, 36, 0, 24) == (5, 31, LW.LEFT)            # a glyph the region's edge cuts
    assert LW.widen(s, 17, 23, 0, 20) == (17, 20, 0)                 # the edge in a gap: nothing to continue
    assert LW.widen(s, 12, 36, 1, 20) == (13, 20, 0)                 # not at the edge: only moved
    s = _strip(runs=((0, 16), (20, 30)))
    assert LW.widen(s, 12, 36, 0, 24) == (12, 24, 0)                 # ink up to the margin's end: no glyph
    s = _strip(runs=((30, 36), (40, 52)))
    assert LW.widen(s, 10, 34, 10, 24) == (20, 35, LW.RIGHT)
    rgba = np.dstack([_strip(runs=((8, 16), (20, 30)))] * 3 + [np.full((24, 60), 255, np.uint8)])
    assert LW.widen(rgba, 12, 36, 0, 24) == (5, 31, LW.LEFT)
    with pytest.raises(ValueError):
        LW.widen(s, 50, 20, 0, 10)


def test_widen_c_equals_twin(tmp_path):
    import subprocess
    exe = tmp_path / "widen"
    subprocess.run(["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-o", str(exe),
                    str(ROOT / "runtime" / "vocr_lines_widen.c"), str(ROOT / "runtime" / "vocr_lines_widen_test.c")],
                   check=True)
    assert subprocess.run([str(exe)], capture_output=True, text=True).returncode == 0
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "lines_widen_parity.py"), "--bin", str(exe),
                        "--cases", "400", "--seed", "3"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_dev_summary_is_consistent():
    """eval/counters_dev_20261021.json: the thresholds follow from its own grid by the pre-registered rule, the
    rule and the re-decode change no real development line, and the edge continuation is not adopted."""
    import json

    import kanum_dev as KD
    d = json.loads((ROOT / "eval" / "counters_dev_20261021.json").read_text(encoding="utf-8"))
    assert d["selection"] == {"tau": KD.select(d["grid"], "A"), "open": KD.select(d["grid"], "B")}
    assert d["selection"] == {"tau": 6000, "open": 4000}
    for m in "SNL":
        assert d["grid"]["models"][m]["rule_changes"] == {"G": 0, "L": 0}
        for dec in "GL":
            eff = d["final"][m][f"{dec}+A6000B4000+R"]["effect"]
            assert eff["tv"]["changed"] == eff["dn"]["changed"] == 0
            assert eff["wd"] == {"better": 0, "worse": 0} and eff["sy"]["worse"] == 0 and eff["sy"]["better"] > 0
    assert d["widen_adopted"] is KD.widen_adopted(d["widen"]) is False
    t = d["corpus"]["table"]
    assert t["月"]["tsu_any"] == 0 and t["年"]["tsu_any"] == 0 and t["国"]["tsu_numeral"] == 0
    assert t["条"]["tsu_numeral"] == 0 and sum(d["corpus"]["chars"].values()) == 700_680_047
    text = json.dumps(d, ensure_ascii=False)
    assert '"pred"' not in text and '"before"' not in text                # numbers only, no readings
