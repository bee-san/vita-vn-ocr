"""train/vocab_expand.py and data/jiten_fetch.py: emittability under eval's norm, VN order and vocab
building, per-deck occurrence counting, the independence line model, Jiten rate limiting and record
parsing. No network, screenshots or shards are needed."""
import gzip
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "train", ROOT / "synth", ROOT / "eval", ROOT / "data"):
    sys.path.insert(0, str(p))
import jiten_fetch as JF  # noqa: E402
import vocab_expand as V  # noqa: E402

# --------------------------------------------------------------------------- emittability

def test_emittable_folds_fullwidth_and_dashes():
    es = V.emit_set(["P", "V", "―", "A", "あ"])
    # a full-width capital is emittable by its ASCII class (NFKC folds Ｐ -> P)
    assert V.emittable("Ｐ", es)
    assert V.emittable("Ｖ", es)
    # every folded dash maps to one class
    assert V.emittable("—", es) and V.emittable("‐", es)
    # a kanji not in the set is not emittable
    assert not V.emittable("膣", es)
    # whitespace folds away, so a space-only string is trivially emittable
    assert V.emittable(" ", es)


def test_unk_emits_nothing():
    es = V.emit_set(["あ", V.UNK])
    assert V.emittable("あ", es)
    assert not V.emittable("本", es)
    assert V.UNK not in es


def test_fl10_misses_fullwidth_capitals_only_as_glyphs_not_score():
    # FL10's label convention writes full-width capitals as ASCII, so an ASCII 'P' emits 'Ｐ'
    es = V.emit_set(["P"])
    assert V.emittable("Ｐ", es)




# --------------------------------------------------------------------------- aggregate lists and order

def _write_list(path, rows):
    path.write_text("Word,Form,Rank\n" + "".join(f"{w},{w},{r}\n" for w, r in rows), encoding="utf-8")


def _write_kanji(path, rows):
    path.write_text("Kanji,Rank\n" + "".join(f"{k},{r}\n" for k, r in rows), encoding="utf-8")


def test_load_list_and_char_weights(tmp_path):
    p = tmp_path / "l.csv"
    _write_list(p, [("本日", 1), ("日", 2), ("火", 4)])
    rows = V.load_list(p)
    assert rows == [("本日", 1), ("日", 2), ("火", 4)]
    w = V.char_weights(rows, 1.0)                    # rank^-1: 本 1, 日 1 + 1/2, 火 1/4
    assert abs(w["本"] - 1.0) < 1e-12 and abs(w["日"] - 1.5) < 1e-12 and abs(w["火"] - 0.25) < 1e-12
    k = tmp_path / "k.csv"
    _write_kanji(k, [("人", 1), ("見", 2), ("人", 5)])
    assert V.load_kanji_csv(k) == {"人": 1, "見": 2}  # a repeated kanji keeps its best rank


def test_missed_rate_counts_weight_outside_vocab():
    w = {"本": 3.0, "膣": 1.0, "あ": 6.0}
    es = V.emit_set(["本", "あ"])
    assert abs(V.missed_rate(w, es) - 100.0) < 1e-9          # 1 of 10, per 1,000
    assert abs(V.missed_rate(w, es, kanji_only=True) - 250.0) < 1e-9


def test_jis_x0213_keeps_japanese_drops_chinese_only():
    assert V.jis_x0213("剝") and V.jis_x0213("剥") and V.jis_x0213("々")
    assert not V.jis_x0213("组") and not V.jis_x0213("诸")


def test_build_order_vn_wins_then_japanese_kanji_csv():
    w_vn = {"日": 5.0, "本": 5.0, "月": 1.0, "あ": 9.0}
    kanji_rank = {"本": 1, "日": 2, "月": 3, "组": 4, "塡": 5, "鮸": 6}
    h15 = ["あ", "！", "日", "本", "月", "鮸", "仝"]
    nk, ko, counts = V.build_order(w_vn, kanji_rank, h15, japanese={"塡"})
    assert ko[:3] == ["本", "日", "月"]              # VN weight first; the 本/日 tie goes to the Kanji CSV rank
    assert ko[3:6] == ["塡", "鮸", "组"]             # Japanese context and JIS, JIS only, then non-JIS
    assert ko[6:] == ["仝"]                          # then data/vocab.txt's remaining kanji
    assert nk == ["あ", "！"] and counts["vn_list"] == 3 and counts["kanji_csv_only_not_jis"] == 1


def test_vocab_of_size_prefix_and_unk():
    order = {"non_kanji": ["あ", "！", "P"], "kanji": ["本", "日", "月", "火", "水"]}
    v = V.vocab_of_size(order, 6)                    # blank + 5 entries: 4 characters + <unk>
    assert v == ["あ", "！", "P", "本", V.UNK]
    big = V.vocab_of_size(order, 10)
    assert big[:-1] == ["あ", "！", "P", "本", "日", "月", "火", "水"] and big[-1] == V.UNK
    assert set(v) <= set(big)                        # sizes nest
    try:
        V.vocab_of_size(order, 100)
    except ValueError:
        return
    raise AssertionError("expected ValueError for too many classes")


def test_spearman_identity_and_reverse():
    assert abs(V.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-12
    assert abs(V.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12


def test_cmd_agg_end_to_end(tmp_path, monkeypatch):
    ld = tmp_path / "lists"
    ld.mkdir()
    for f in ("VisualNovel", "Novel", "Anime", "Drama", "VideoGame", "global"):
        _write_list(ld / f"jiten_freq_{f}.csv", [("本日", 1), ("月", 2), ("膣", 50)])
    _write_kanji(ld / "jiten_freq_kanji.csv", [("本", 1), ("日", 2), ("月", 3), ("膣", 9), ("塡", 10)])
    fl10 = tmp_path / "fl10.txt"
    fl10.write_text("あ\n本\n日\n〓\n", encoding="utf-8")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("あ\n本\n日\n月\n", encoding="utf-8")
    real = tmp_path / "real.jsonl"
    real.write_text("".join(json.dumps({"status": "kept", "label": t, "vn": f"v{i}"}, ensure_ascii=False) + "\n"
                            for i, t in enumerate(["本日", "月", "本"])), encoding="utf-8")
    monkeypatch.setattr(V, "FL10_VOCAB", fl10)
    monkeypatch.setattr(V, "PROOF_VOCAB", fl10)
    import synth4_gap
    monkeypatch.setattr(synth4_gap, "reserved_vns", lambda extra=(): set())

    class Args:
        pass
    a = Args()
    a.lists, a.real, a.vocab, a.vocab_file, a.vndb_chars = str(ld), [f"r={real}"], str(vocab), [], None
    a.exclude, a.out, a.order_out = [], str(tmp_path / "agg.json"), str(tmp_path / "order.json")
    V.cmd_agg(a)
    o = json.loads((tmp_path / "order.json").read_text(encoding="utf-8"))
    assert o["kanji"][:4] == ["本", "日", "月", "膣"]          # the VN list first, by weight
    assert o["kanji"][4] == "塡"                               # then the Kanji CSV
    d = json.loads((tmp_path / "agg.json").read_text(encoding="utf-8"))
    assert d["fixed"]["FL10"]["VN"]["chars"] > 0               # FL10 lacks 月 and 膣
    assert d["kanji_csv_check"]["vn_kanji"] == 4


# --------------------------------------------------------------------------- per-deck sample

def _write_deck_csv(path, rows):
    """rows: (word, occurrences, word_id). Writes a gzipped Jiten deck CSV export."""
    buf = io.StringIO()
    buf.write('"Word","ReadingFurigana","ReadingKana","Occurences","ReadingFrequency","PitchPositions",'
              '"Definitions","ExampleSentence","JmDictWordId"\n')
    for w, o, wid in rows:
        buf.write(f'"{w}","","","{o}","1","","","","{wid}"\n')
    path.write_bytes(gzip.compress(buf.getvalue().encode()))


def test_deck_counts_occurrences_and_names(tmp_path):
    p = tmp_path / "1.csv.gz"
    _write_deck_csv(p, [("本日", 10, 1469800), ("巴", 3, 5000001)])  # the second is a JMnedict (name) id
    cnt, names, tot = V.deck_counts(p)
    assert cnt["本"] == 10 and cnt["日"] == 10 and cnt["巴"] == 3
    assert names["巴"] == 3 and "本" not in names
    assert tot == 10 * 2 + 3 * 1


def test_sample_and_tail_row(tmp_path, monkeypatch):
    jd = tmp_path / "jiten"
    (jd / "decks").mkdir(parents=True)
    (jd / "words" / "VisualNovel").mkdir(parents=True)
    decks = [(1100, [("本", 990, 1), ("膣", 10, 2)]), (500, [("本", 500, 1)])]
    meta = [{"deckId": i + 1, "parentDeckId": None, "characterCount": cc, "originalTitle": f"t{i}",
             "links": [{"url": f"https://vndb.org/v{90000 + i}"}]} for i, (cc, _) in enumerate(decks)]
    (jd / "decks" / "decks_VisualNovel.json").write_text(json.dumps(meta), encoding="utf-8")
    for i, (_, rows) in enumerate(decks):
        _write_deck_csv(jd / "words" / "VisualNovel" / f"{i + 1}.csv.gz", rows)
    monkeypatch.setattr(V, "reserved_vns", lambda extra=(): set())
    monkeypatch.setattr(V, "is_reserved", lambda vns, reserved: False)

    class Args:
        pass
    a = Args()
    a.jiten, a.exclude, a.out = str(jd), [], str(tmp_path / "s.npz")
    V.cmd_sample(a)
    ch = V.Chars(tmp_path / "s.npz")
    assert len(ch.decks) == 2 and ch.decks[0]["wordform"] == 1000
    missed, names = V.deck_missed(ch, V.emit_set(["本"]))   # 膣 unemittable: 10 of deck 1's 1,000 form chars
    r = V.tail_row(ch, ch.select(reserved=False), missed, names)
    assert abs(r["corpus"] - 1000 * 10 / 1500) < 1e-3         # word-form basis
    assert r["titles_with_miss"] == 0.5 and abs(r["max"] - 10.0) < 1e-6
    assert abs(r["content_basis"]["corpus"] - round(1000 * 10 / 1600, 4)) < 1e-6


# --------------------------------------------------------------------------- lines

def test_predicted_line_share_monotone():
    lens = [2, 6, 23]
    assert V.predicted_line_share(0.0, lens) == 0.0
    a = V.predicted_line_share(0.001, lens)
    b = V.predicted_line_share(0.01, lens)
    assert 0 < a < b < 1


def test_content_len_counts_kana_kanji_alnum_only():
    assert V.content_len("本、日！A") == 3            # 、！ are punctuation, A is alnum
    assert V.content_len("あいう") == 3




# --------------------------------------------------------------------------- char recall (proof)

def test_char_recall_groups():
    group = V.char_groups(["本", "日"], ["本", "月"])   # added: 月, dropped: 日, shared: 本
    assert group("月") == "added" and group("日") == "dropped" and group("本") == "shared"
    # ref "本月" read exactly, ref "本" read as "本": 本 right twice, 月 (added) right once
    pg, pc = V.char_recall(["本月", "本"], ["本月", "本"], group)
    assert pc["本"] == [2, 2]
    assert pg["added"] == [1, 1]
    # a miss on an added class drops its recall
    pg2, _ = V.char_recall(["月"], ["日"], group)
    assert pg2["added"] == [0, 1]


# --------------------------------------------------------------------------- jiten fetcher

def test_vndb_ids_from_links():
    d = {"links": [{"url": "https://vndb.org/v55744"}, {"url": "https://example.com"}]}
    assert JF.vndb_ids(d) == ["v55744"]
    assert V.vndb_ids(d) == ["v55744"]


def test_request_waits_out_interval(monkeypatch):
    calls = []

    class FakeResp:
        headers = {"Content-Encoding": "identity", "Date": "now"}

        def read(self):
            return b"ok"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(JF.urllib.request, "urlopen", lambda *a, **k: FakeResp())
    monkeypatch.setattr(JF.time, "monotonic", lambda: calls[-1] if calls else 0.0)
    slept = []
    monkeypatch.setattr(JF.time, "sleep", lambda s: slept.append(s))
    JF._last[0] = 0.0
    calls.append(0.0)
    JF.request("http://x", interval=6.5)
    # with monotonic at 0 and _last 0, the limiter asks to sleep ~6.5 s
    assert slept and abs(slept[0] - 6.5) < 1e-6


def test_download_body_is_full_csv_no_examples():
    assert JF.DOWNLOAD_BODY["format"] == 2 and JF.DOWNLOAD_BODY["downloadType"] == 1
    assert JF.DOWNLOAD_BODY["excludeExampleSentences"] is True


def test_media_types_match_jiten_enum():
    assert JF.MEDIA_TYPES["VisualNovel"] == 7 and JF.MEDIA_TYPES["Novel"] == 4


def test_proof_vocab_committed_and_well_formed():
    v = V.read_vocab(ROOT / "data" / "vocab_4987_proof.txt")
    assert len(v) == 4986 and v[-1] == V.UNK
    assert len(set(v)) == len(v)
    # a subset of data/vocab.txt plus <unk>, so every class has synthetic and NVIDIA renders
    h15 = set(V.read_vocab(ROOT / "data" / "vocab.txt"))
    assert all(c in h15 for c in v if c != V.UNK)


def test_tier_vocabs_committed_nested():
    small = V.read_vocab(ROOT / "data" / "vocab_vn_4987.txt")
    normal = V.read_vocab(ROOT / "data" / "vocab_vn_5631.txt")
    large = V.read_vocab(ROOT / "data" / "vocab_vn_6459.txt")
    proof = V.read_vocab(ROOT / "data" / "vocab_4987_proof.txt")
    for v, n in ((small, 4987), (normal, 5631), (large, 6459)):
        assert len(v) + 1 == n and v[-1] == V.UNK and len(set(v)) == len(v)
    assert set(small) <= set(normal) <= set(large)          # one render for the largest serves every tier
    assert set(proof) <= set(normal)
    # data/vocab.txt's non-kanji (kana, punctuation, ASCII and full-width Latin and digits) are in every file
    nk = {c for c in V.read_vocab(ROOT / "data" / "vocab.txt") if not V.is_kanji(c)}
    assert nk <= set(small) and nk <= set(proof)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
