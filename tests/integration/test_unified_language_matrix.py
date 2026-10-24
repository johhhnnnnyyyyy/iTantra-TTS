"""Opt-in real audio checks for both precisions and every target language."""
import asyncio
import os
from pathlib import Path

import numpy as np
import pytest

from tools.unified_research_web import synthesize


@pytest.mark.parametrize(("language", "spoken"), [
    ("bn", "পাঁচ"), ("en", "five"), ("gu", "પાંચ"), ("hi", "पांच"),
    ("kn", "ಐದು"), ("ml", "അഞ്ച്"), ("mr", "पाच"), ("or", "ପାଞ୍ଚ"),
    ("ta", "ஐந்து"), ("te", "ఐదు"),
])
@pytest.mark.parametrize("precision", ["int8", "int4"])
def test_real_language_precision_matrix(language, spoken, precision):
    if os.environ.get("ITANTRA_MATRIX_REAL") != "1":
        pytest.skip("set ITANTRA_MATRIX_REAL=1 to run all 20 real model checks")
    library = os.environ.get("ITANTRA_W4_LIBRARY")
    if not library:
        pytest.fail("ITANTRA_W4_LIBRARY is required for this matrix")
    audio, status, actual, _, metrics = asyncio.run(
        synthesize("5", language, precision, Path(library), threads=4)
    )
    assert audio is not None, f"{language}/{precision}: {status}"
    assert audio[0] == (16000 if language == "or" else 22050)
    assert audio[1].size > 0 and np.any(audio[1])
    assert actual == spoken
    assert "total RTF" in metrics and "peak RSS" in metrics
    if precision == "int4" and language != "or":
        assert f"experimental-w4a8-per-channel-{language}" in metrics
