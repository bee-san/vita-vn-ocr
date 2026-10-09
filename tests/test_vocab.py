"""data/vocab.txt: one recognizer class per line (class 0 is the CTC blank, class i + 1 is line i)."""
import unicodedata

from common import DATA, load_vocab

RAW = (DATA / "vocab.txt").read_text(encoding="utf-8")
VOCAB = load_vocab()
CHARS = set(VOCAB)


def test_one_code_point_per_line():
    lines = RAW.split("\n")
    assert lines[-1] == "", "vocab.txt should end with a newline"
    assert lines[:-1] == VOCAB, "load_vocab() must return every line, in file order"
    assert all(len(c) == 1 for c in VOCAB), [c for c in VOCAB if len(c) != 1]


def test_unique_and_sorted():
    assert len(CHARS) == len(VOCAB), "duplicate classes"
    assert VOCAB == sorted(VOCAB), "build_corpus.py writes the vocab in code point order"


def test_no_whitespace_control_or_unnormalised_entries():
    assert not [c for c in VOCAB if c.isspace() or unicodedata.category(c)[0] in "CZ"]
    assert all(unicodedata.normalize("NFC", c) == c for c in VOCAB)


def test_lookalikes_are_merged():
    # build_corpus.norm folds 〜 into ～, — into ―, and half-width katakana into full width,
    # so the recognizer never has to choose between identical-looking classes.
    assert not CHARS & {"〜", "—", "\u3000", " "}
    assert not [c for c in VOCAB if 0xFF61 <= ord(c) <= 0xFF9F]


def test_core_character_sets_present():
    sets = {
        "hiragana": {chr(c) for c in range(0x3041, 0x3097)},
        "katakana": {chr(c) for c in range(0x30A1, 0x30FB)} | set("ーヽヾゝゞ"),
        "full-width alphanumerics": {chr(c) for c in [*range(0xFF10, 0xFF1A), *range(0xFF21, 0xFF3B),
                                                      *range(0xFF41, 0xFF5B)]},
        "ASCII": set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!?.,:;'\"()-/&%#+=*~"),
        "VN punctuation": set("、。・「」『』（）【】！？…―～々"),
    }
    for name, s in sets.items():
        assert s <= CHARS, f"{name} missing: {''.join(sorted(s - CHARS))}"
