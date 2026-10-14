"""The character LM (lm/build.py, lm/vlm.py) and the CTC prefix beam decoder (runtime/vocr_lm.c), on generated
fixtures only: a random-init model, drawn images, a small generated corpus. No development or protected data is
read here, except by the last test, which runs on the development extracts when they exist (the fleet's shared
box) and is skipped elsewhere.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "lm", ROOT / "eval", ROOT / "runtime", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import vlm  # noqa: E402

import vocr3_fixtures as F  # noqa: E402

WORDS = ["あいう", "かきく", "さしす", "たちつ", "なにぬ", "はひふ", "漢字", "名", "そう", "こと", "ほへ", "せそ"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corpus(tmp, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    lines = []
    for _ in range(n):
        k = int(rng.integers(1, 6))
        w = "".join(WORDS[int(i)] for i in rng.choice(len(WORDS), k, p=np.linspace(2, 1, len(WORDS)) / 18))
        lines.append(w + ("！" if rng.random() < 0.5 else "？"))
    p = tmp / "corpus.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def sources(tmp):
    """A sources.json over generated files: the fixture model's vocabulary, a corpus, a Jiten-style list."""
    voc = tmp / "vocab.txt"
    voc.write_text("\n".join(F.VOCAB) + "\n", encoding="utf-8")
    cor = corpus(tmp)
    held = tmp / "held.txt"
    held.write_text("\n".join(corpus(tmp / "..", 200, seed=1).read_text(encoding="utf-8").split("\n")[:200]),
                    encoding="utf-8")
    jit = tmp / "jiten.csv"
    jit.write_text("Word,Form,Rank\nこと,こと,1\n漢字,かんじ,2\nそう,そう,3\nの,の,4\nほへと,ほへと,5\n",
                   encoding="utf-8")
    meta = {"licence": "test", "attribution": "generated"}
    src = {"vocab": {"path": str(voc), "sha256": sha(voc)},
           "corpora": {"gen": {"path": str(cor), "sha256": sha(cor), **meta}},
           "jiten": {"vn": {"path": str(jit), "url": "test", "sha256": sha(jit), **meta}}}
    p = tmp / "sources.json"
    p.write_text(json.dumps(src), encoding="utf-8")
    return p, held


def build(tmp, out, order=3, nbytes=60_000, jiten="vn=0.2"):
    src, held = sources(tmp)
    cmd = [sys.executable, str(ROOT / "lm/build.py"), "--out", str(out), "--manifest", str(out) + ".json",
           "--sources", str(src), "--corpora", "gen", "--jiten", jiten, "--order", str(order), "--bytes", str(nbytes),
           "--heldout", str(held), "--heldout-lines", "100"]
    subprocess.run(cmd, check=True, capture_output=True, cwd=tmp)
    return out.read_bytes(), json.loads(Path(str(out) + ".json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("lm")
    (d / "a").mkdir()
    blob, man = build(d / "a", d / "a" / "lm.vlm")
    return d, blob, man


BRACKET_VOCAB = F.VOCAB + list("「」『』（）【】")


def bracket_blob(path, seed=0, width=0.25):
    """vocr3_fixtures.fixture_blob with bracket classes in the vocabulary (for the bracket-pairing state)."""
    import torch
    from torch import nn

    import export
    from model import TinyVNOCR
    torch.manual_seed(seed)
    m = TinyVNOCR(len(BRACKET_VOCAB) + 1, width=width).eval()
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_var.uniform_(0.5, 2.0)
                mod.running_mean.uniform_(-0.2, 0.2)
                mod.weight.uniform_(0.5, 1.5)
                mod.bias.uniform_(-0.5, 0.5)
            if isinstance(mod, nn.Conv2d):
                nn.init.normal_(mod.weight, 0, (2.0 / mod.weight[0].numel()) ** 0.5)
    blob, _ = export.build_blob(export.model_ops(m, 32), BRACKET_VOCAB, 32, True, 1024, b"")
    Path(path).write_bytes(blob)
    return Path(path)


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    from pyvocr import Vocr
    return Vocr(bracket_blob(tmp_path_factory.mktemp("blob") / "m.vocr"), mode="w8i", max_width=1024)


def test_build_is_deterministic_and_fits(built, tmp_path):
    d, blob, man = built
    (tmp_path / "b").mkdir()
    again, man2 = build(tmp_path / "b", tmp_path / "b" / "lm.vlm")
    assert again == blob
    assert man2["sha256"] == man["sha256"] == hashlib.sha256(blob).hexdigest()
    assert len(blob) <= 60_000 and len(blob) % 16 == 0
    lm = vlm.LM(blob)
    assert lm.order == 3 and lm.V == len(F.VOCAB) + vlm.FIRST
    assert lm.sha16 == bytes.fromhex(man["manifest_sha256_prefix"])
    assert [c["name"] for c in man["corpora"]] == ["gen"] and man["jiten"][0]["name"] == "vn"
    hashed = json.dumps({k: v for k, v in man.items() if k not in ("paths", "heldout")}, ensure_ascii=False)
    assert "paths" in man and str(d) not in hashed and str(tmp_path) not in hashed


def test_distributions_are_normalised(built):
    _, blob, _ = built
    lm = vlm.LM(blob)
    for ctx in ("", "あ", "あい", "漢", "かき", "！"):
        st = lm.initial(vlm.FLAG_BOS)
        for ch in ctx:
            _, st = lm.score(st, lm.token_of(ch))
        p = sum(np.exp(lm.score(st, t)[0] / vlm.Q) for t in range(lm.V) if t != vlm.BOS)
        assert abs(p - 1) < 0.05, (ctx, p)          # quantisation to 256 levels moves the mass a little
    st = lm.initial(0)
    _, st = lm.score(st, lm.token_of("あ"))
    _, st = lm.score(st, lm.token_of("い"))
    assert lm.score(st, lm.token_of("う"))[0] > lm.score(st, lm.token_of("か"))[0]     # it learned the corpus


def test_c_scorer_equals_python(built, model):
    from pyvocr import VocrLM
    _, blob, _ = built
    lm = vlm.LM(blob)
    dec = VocrLM(model, blob, alpha=500)
    rng = np.random.default_rng(3)
    for flags in (0, vlm.FLAG_BOS):
        st = lm.initial(flags)
        for _ in range(400):
            t = int(rng.integers(0, lm.V))
            lp, nxt = lm.score(st, t)
            assert dec.score(st, t) == (lp, nxt)
            st = nxt if rng.random() < 0.9 else lm.initial(flags)


def random_line(rng, frames, k, n_classes):
    """Per frame k distinct classes with logits in the runtime's order (decreasing, ties: the lower class first),
    with exact ties, and a blank logit: class 0's when the blank is among them, else below the k-th logit."""
    topv = np.zeros((frames, k), np.float32)
    topc = np.zeros((frames, k), np.uint16)
    blank = np.zeros(frames, np.float32)
    for f in range(frames):
        cls = rng.choice(n_classes, k, replace=False)
        v = rng.normal(0, 3, k).astype(np.float32)
        if k > 1 and rng.random() < 0.3:
            v[1] = v[0]
        idx = sorted(range(k), key=lambda j: (-float(v[j]), int(cls[j])))
        topv[f], topc[f] = v[idx], cls[idx]
        hit = np.flatnonzero(topc[f] == 0)
        blank[f] = topv[f, hit[0]] if len(hit) else topv[f, -1] - np.float32(0.01 + 3 * rng.random())
    return topv, topc, blank


CFGS = [dict(beam=8, topk=8, alpha=0, beta=0, gate=5000), dict(beam=1, topk=1, alpha=700, beta=300, gate=3000),
        dict(beam=4, topk=4, alpha=400, beta=2000, gate=3500, flags=2),
        dict(beam=16, topk=8, alpha=1500, beta=-500, gate=12000, flags=3, bracket=3000),
        dict(beam=8, topk=6, alpha=8000, beta=8000, gate=64000, flags=1, bracket=64000),
        dict(beam=8, topk=8, alpha=0, beta=0, gate=8000, bracket=5000)]


@pytest.mark.parametrize("cfg", CFGS)
def test_c_decoder_equals_python_reference(built, model, cfg):
    from pyvocr import VocrLM
    _, blob, _ = built
    lm = vlm.LM(blob)
    vocab = [model.class_text(c) for c in range(model.n_classes)]
    cmap = vlm.class_map(lm, vocab)
    roles = [vlm.bracket_role(t) for t in vocab]
    assert sum(r != 0 for r in roles) == 8
    dec = VocrLM(model, blob, **cfg)
    rng = np.random.default_rng(cfg["beam"] * 31 + cfg["alpha"] + cfg.get("bracket", 0))
    changed = 0
    for n in range(150):
        frames = int(rng.integers(1, 60))
        v, c, b = random_line(rng, frames, 8, model.n_classes)
        blank = None if n % 5 == 0 else b
        py = "".join(vocab[x] for x in vlm.decode(lm, cmap, v, c, blank, roles=roles, **cfg))
        assert dec.decode(v, c, blank) == py
        changed += py != greedy(v, c, vocab)
    assert changed or not (cfg["alpha"] or cfg.get("bracket"))      # the options do something


def greedy(v, c, vocab):
    out, last = [], 0
    for x in c[:, 0]:
        x = int(x)
        if x and x != last:
            out.append(vocab[x])
        last = x
    return "".join(out)


def test_weight_zero_is_greedy_bit_for_bit(built, model):
    """alpha = beta = 0: the beam's text is the greedy text for every beam, top-K and gate, with or without an
    LM, on random top-K lines (ties included) and on the fixture model's own outputs."""
    from pyvocr import VocrLM
    _, blob, _ = built
    vocab = [model.class_text(c) for c in range(model.n_classes)]
    rng = np.random.default_rng(11)
    decs = [VocrLM(model, lm, beam=beam, topk=k, gate=g) for lm in (None, blob) for beam in (1, 3, 16)
            for k in (1, 4, 8) for g in (0, 2500, 64000)]
    for _ in range(60):
        v, c, b = random_line(rng, int(rng.integers(1, 80)), 8, model.n_classes)
        g = greedy(v, c, vocab)
        for d in decs:
            assert d.decode(v, c, b) == g
    for seed in range(12):
        rgb = F.draw_rows(np.full((40, 400, 3), 30, np.uint8), [(5, 8, 380, 22)], seed=seed)
        img = np.ascontiguousarray(rgb[..., 0])
        out = model.run_topk(img, 8)
        assert out["text"] == model.text(img)
        for d in decs:
            assert d.decode(out["topv"], out["topc"], out["blank"]) == out["text"]


def test_topk_extras_match_the_logits(model):
    import ctclm
    for seed in range(8):
        rgb = F.draw_rows(np.full((36, 300, 3), 20, np.uint8), [(4, 6, 290, 20)], seed=seed)
        img = np.ascontiguousarray(rgb[..., 0])
        text, logits = model.run(img, logits=True)
        for k in (1, 3, 16):
            out = model.run_topk(img, k)
            v, c = ctclm.topk_of(logits, k)
            assert out["text"] == text
            assert np.array_equal(out["topc"], c)
            assert np.array_equal(out["topv"].view(np.uint32), v.astype(np.float32).view(np.uint32))
            assert np.array_equal(out["blank"].view(np.uint32), logits[:, 0].astype(np.float32).view(np.uint32))
            if k == 1:
                assert out["conf_min"] == out["conf_mean"] == 1.0
            else:
                z = np.exp(v.astype(np.float64) - v[:, :1])
                p = 1 / z.sum(1)
                emit = [f for f in range(len(c)) if c[f, 0] and (f == 0 or c[f, 0] != c[f - 1, 0])]
                if emit:
                    assert abs(out["conf_min"] - p[emit].min()) < 1e-4
                    assert abs(out["conf_mean"] - p[emit].mean()) < 1e-4


def test_lm_blob_rejects_damage(built, model):
    from pyvocr import VocrError, VocrLM
    _, blob, _ = built
    bad = bytearray(blob)
    off_prob = int.from_bytes(blob[64 + 8:64 + 12], "little")       # order 1's probability codes: any value is valid
    bad[off_prob + 5] ^= 1
    with pytest.raises(VocrError):
        VocrLM(model, bytes(bad), alpha=300)                       # CRC
    VocrLM(model, bytes(bad), alpha=300, check_crc=False)          # structure still valid
    with pytest.raises(VocrError):
        VocrLM(model, blob[:-16], alpha=300)
    with pytest.raises(ValueError):
        VocrLM(model, None, alpha=300)                             # alpha > 0 needs an LM
    with pytest.raises(ValueError):
        VocrLM(model, blob, beam=17)


def test_bracket_state():
    r = {ch: vlm.bracket_role(ch) for ch in "「」『』（）()【】a"}
    assert r["a"] == 0 and r["("] == r["（"] == 3 and r[")"] == r["）"] == 0x83
    st, mis = 0, False
    for ch in "「『":
        st, mis = vlm.bracket_next(st, r[ch])
    assert st == (1 << 8 | 2) and not mis
    assert vlm.bracket_next(st, r["』"]) == (1, False)          # closes the inner
    assert vlm.bracket_next(st, r["」"]) == (0, True)           # closes the outer: the inner was left open
    assert vlm.bracket_next(st, r["】"]) == (st, True)          # closes nothing open
    assert vlm.bracket_next(0, r["」"]) == (0, False)           # nothing open on this line: free


def test_families_add_up():
    import ctclm
    from metrics import lev
    pairs = [("パンを食べた。", "バンを食べた、"), ("きゃー", "きやー"), ("中に", "申に"), ("ー", "一"), ("」", "』"),
             ("あいう", "あう"), ("あ", "あいい"), ("", "あ")]
    for ref, pred in pairs:
        f = ctclm.families(ref, pred)
        assert sum(f.values()) == lev(ref, pred)
    f = ctclm.families("パンを食べた。", "バンを食べた、")
    assert f["dakuten"] == 1 and f["punct_punct"] == 1
    assert ctclm.families("きゃー", "きやー")["small_kana"] == 1
    assert ctclm.families("中に", "申に")["kanji_kanji"] == 1
    assert ctclm.families("ー", "一")["dash_like"] == 1
    assert ctclm.fold_of("v1") == ctclm.fold_of("v1") and {ctclm.fold_of(f"v{i}") for i in range(20)} == {0, 1}


def test_sources_manifest_pins_every_input():
    src = json.loads((ROOT / "lm/sources.json").read_text(encoding="utf-8"))
    assert src["vocab"]["sha256"] == sha(ROOT / "data/vocab.txt")
    for group in ("corpora", "jiten"):
        for name, e in src[group].items():
            assert len(e["sha256"]) == 64 and e["licence"] and e["attribution"], name
    assert "CC BY-SA 4.0" in src["jiten"]["vn"]["licence"]


EXTRACT = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr")) / "shared/artifacts/ctc-lm-decode/extract"


@pytest.mark.skipif(not (EXTRACT / "v3FL10.npz").exists(), reason="development extracts are on the fleet box only")
def test_weight_zero_is_greedy_on_development_logits():
    """Every development line of every extracted model: the beam at alpha = beta = 0 gives the runtime's greedy
    text exactly (beam 8, top-K 8 and 16, gates 2.5 and 64 nats)."""
    import ctclm
    from pyvocr import Vocr, VocrLM
    for path in sorted(EXTRACT.glob("*.npz")):
        z = ctclm.load_extract(path)
        rt = Vocr(z["meta"]["blob"], mode="w8i", max_width=1024)
        decs = [VocrLM(rt, None, beam=8, topk=k, gate=g) for k in (8, 16) for g in (2500, 64000)]
        for i, line in enumerate(z["meta"]["lines"]):
            a, b = int(z["off"][i]), int(z["off"][i + 1])
            for d in decs:
                assert d.decode(z["topv"][a:b], z["topc"][a:b], z["blank"][a:b]) == line["greedy"], (path.name, i)
