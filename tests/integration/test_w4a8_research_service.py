"""Experimental W4A8 adapter integrated through the shared service."""
import asyncio
import os
from pathlib import Path

import pytest

from itantra_tts.adapters.w4a8_research import W4A8ResearchBackend
from itantra_tts.application.service import SynthesisService, SynthesisServiceError
from itantra_tts.domain import Precision, SynthesisRequest


def test_research_worker_thread_count_is_bounded(tmp_path):
    with pytest.raises(ValueError, match="threads"):
        W4A8ResearchBackend(tmp_path / "missing.so", threads=0)


def test_missing_library_fails_without_exposing_input(tmp_path):
    service = SynthesisService(W4A8ResearchBackend(tmp_path / "missing.so"),
                               allow_experimental_int4=True)
    with pytest.raises(SynthesisServiceError) as error:
        asyncio.run(service.synthesize(SynthesisRequest("receiver", 5, "private 123", "en", Precision.INT4)))
    assert "private" not in str(error.value)


def test_real_int4_service_normalizes_and_returns_audio():
    library = os.environ.get("ITANTRA_W4_LIBRARY")
    if not library:
        pytest.skip("set ITANTRA_W4_LIBRARY for real-model test")
    service = SynthesisService(W4A8ResearchBackend(Path(library)), allow_experimental_int4=True)
    output = asyncio.run(service.synthesize(SynthesisRequest("receiver", 5, "5.", "en", Precision.INT4)))
    assert output.spoken_text == "five."
    assert output.sequence_numbers == (5,)
    assert output.wav_bytes.startswith(b"RIFF") and len(output.wav_bytes) > 1000
