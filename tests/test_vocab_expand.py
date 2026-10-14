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


# --------------------------------------------------------------------------- order and build

def _order_fixture():
    # 486-ish non-kanji is irrelevant here; use a small non-kanji block and a kanji order
    return {"non_kanji": ["あ", "！", "P"], "non_kanji_extra": [], "kanji_outside_jiten": ["﨑"],
            "kanji": ["本", "日", "月", "火", "水"], "weights": {}}


def test_vocab_of_size_order_and_unk():
    order = _order_fixture()
    v = V.vocab_of_size(order, 6, unk=True, h15=["本", "日", "々"])
    # 6 classes = CTC blank + 5 entries: the non-kanji block, the compat kanji, then <unk>
    assert v[-1] == V.UNK
    assert len(v) == 5 and len(set(v)) == 5
    assert v[:3] == ["あ", "！", "P"]
    assert v[3] == "﨑"                                  # the compat kanji Jiten cannot see comes before VN kanji
    # a larger vocab reaches the VN kanji in order, then appends the H15 tail kanji the VN order lacks
    big = V.vocab_of_size(order, 12, unk=True, h15=["本", "々"])
    assert big[-1] == V.UNK and "々" in big and big.index("本") < big.index("々")


def test_vocab_of_size_rejects_too_many():
    order = _order_fixture()
    try:
        V.vocab_of_size(order, 100, unk=True, h15=[])
    except ValueError:
        return
    raise AssertionError("expected ValueError for too many classes")


# --------------------------------------------------------------------------- per-deck counting

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


def test_cmd_jiten_builds_sparse_counts(tmp_path, monkeypatch):
    jd = tmp_path / "jiten"
    (jd / "decks").mkdir(parents=True)
    (jd / "words" / "VisualNovel").mkdir(parents=True)
    deck = {"deckId": 1, "parentDeckId": None, "characterCount": 100, "uniqueKanjiCount": 2,
            "links": [{"url": "https://vndb.org/v99999"}]}
    (jd / "decks" / "decks_VisualNovel.json").write_text(json.dumps([deck]), encoding="utf-8")
    _write_deck_csv(jd / "words" / "VisualNovel" / "1.csv.gz", [("本日", 10, 1), ("巴", 3, 5000001)])
    monkeypatch.setattr(V, "reserved_vns", lambda extra=(): set())
    out = tmp_path / "chars.npz"

    class Args:
        pass
    a = Args()
    a.jiten, a.exclude, a.out = str(jd), [], str(out)
    V.cmd_jiten(a)
    ch = V.Chars(out)
    assert len(ch.decks) == 1 and ch.decks[0]["content"] == 100
    w = ch.per_char(ch.select(("VisualNovel",)), "corpus")
    assert w["本"] == 10 and w["巴"] == 3


# --------------------------------------------------------------------------- coverage and lines

def test_deck_missed_and_cover_row(tmp_path, monkeypatch):
    jd = tmp_path / "jiten"
    (jd / "decks").mkdir(parents=True)
    (jd / "words" / "VisualNovel").mkdir(parents=True)
    (jd / "decks" / "decks_VisualNovel.json").write_text(
        json.dumps([{"deckId": 1, "parentDeckId": None, "characterCount": 1000, "links": []}]), encoding="utf-8")
    _write_deck_csv(jd / "words" / "VisualNovel" / "1.csv.gz", [("本", 990, 1), ("膣", 10, 2)])
    monkeypatch.setattr(V, "reserved_vns", lambda extra=(): set())

    class Args:
        pass
    a = Args()
    a.jiten, a.exclude, a.out = str(jd), [], str(tmp_path / "c.npz")
    V.cmd_jiten(a)
    ch = V.Chars(tmp_path / "c.npz")
    missed, _ = V.deck_missed(ch, V.emit_set(["本"]))      # 膣 unemittable: 10 of 1000
    r = V.cover_row(ch, ch.select(("VisualNovel",)), missed)
    assert abs(r["corpus"] - 10.0) < 1e-6                   # 10 per 1,000 content chars


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


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
