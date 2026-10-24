"""Research web boundary: validation, and optional real-model audio."""
import os
from pathlib import Path

import pytest

from tools.w4a8_research_web import synthesize


def test_research_web_rejects_invalid_language_without_starting_worker(tmp_path):
    audio, status, metrics = synthesize("hello", "xx", tmp_path / "missing.so")
    assert audio is None
    assert "Invalid" in status
    assert "Not measured" in metrics


def test_research_web_generates_actual_hindi_audio():
    library = os.environ.get("ITANTRA_W4_LIBRARY")
    if not library:
        pytest.skip("set ITANTRA_W4_LIBRARY for real W4A8 synthesis")
    audio, status, metrics = synthesize("हाँ.", "hi", Path(library))
    assert audio is not None and audio[0] == 22050 and len(audio[1]) > 1000
    assert "generated" in status.lower()
    assert "W4A8" in metrics and "desktop" in metrics
