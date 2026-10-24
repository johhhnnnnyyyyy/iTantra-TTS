from pathlib import Path

import pytest

from itantra_tts.adapters.odia_tokenizer import OdiaMmsTokenizer


ROOT = Path(__file__).resolve().parents[2]
VOCAB = ROOT / "models/artifacts/mms-tts-ory-int8-research/vocab.json"
pytestmark = pytest.mark.skipif(not VOCAB.is_file(), reason="local Odia vocabulary artifact not provisioned")


def test_local_tokenizer_matches_mms_blank_and_character_ids() -> None:
    tokenizer = OdiaMmsTokenizer(VOCAB)

    assert tokenizer.encode("ନମସ୍କାର ଓଡ଼ିଶା").tolist() == [
        [0, 8, 0, 24, 0, 68, 0, 67, 0, 42, 0, 39, 0, 9, 0, 58,
         0, 28, 0, 38, 0, 50, 0, 31, 0, 34, 0, 39, 0]
    ]


def test_local_tokenizer_matches_mms_punctuation_filtering() -> None:
    tokenizer = OdiaMmsTokenizer(VOCAB)

    assert tokenizer.encode("ପାଣି, ୨୦୦!").tolist() == [[0, 7, 0, 39, 0, 35, 0, 31, 0]]


def test_local_tokenizer_rejects_text_without_vocab_symbols() -> None:
    tokenizer = OdiaMmsTokenizer(VOCAB)

    try:
        tokenizer.encode("!!!")
    except ValueError as error:
        assert "no" in str(error)
    else:
        raise AssertionError("empty tokenization must not synthesize")
