"""The unified research UI uses the shared service for both precision choices."""
import asyncio
import os
from pathlib import Path

import pytest

from tools.unified_research_web import synthesize


def test_int8_worker_receives_requested_thread_count(monkeypatch, tmp_path):
    from itantra_tts.adapters.async_local import AsyncLocalResearchSynthesisBackend
    from itantra_tts.domain import Precision, SynthesisRequest

    calls = []

    async def capture(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("captured")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    backend = AsyncLocalResearchSynthesisBackend(tmp_path, tmp_path, intra_op_threads=2)
    with pytest.raises(RuntimeError, match="captured"):
        asyncio.run(backend.synthesize(SynthesisRequest("test", 0, "hello", "en", Precision.INT8)))
    assert calls[0][-1] == "2"


def test_invalid_precision_is_rejected_without_audio(tmp_path):
    audio, status, spoken, normalization, metrics = asyncio.run(
        synthesize("hello", "en", "float32", tmp_path / "missing.so")
    )
    assert audio is None and "Invalid" in status
    assert not spoken and "Not measured" in metrics


def test_invalid_int4_thread_count_is_rejected(tmp_path):
    audio, status, _, _, _ = asyncio.run(
        synthesize("hello", "en", "int4", tmp_path / "missing.so", threads=0)
    )
    assert audio is None and "Invalid" in status


def test_int8_thread_count_is_visible_in_actual_worker_metrics():
    if not (Path(__file__).resolve().parents[2] / "models/artifacts/echobharat-int8/hi").is_dir():
        pytest.skip("local Hindi model pack is not available")
    audio, _, _, _, metrics = asyncio.run(
        synthesize("नमस्ते", "hi", "int8", Path("unused"), threads=1)
    )
    assert audio is not None
    assert "ORT intra-op threads per session: 1" in metrics


def test_real_int4_uses_shared_normalization():
    library = os.environ.get("ITANTRA_W4_LIBRARY")
    if not library:
        pytest.skip("set ITANTRA_W4_LIBRARY for real-model test")
    audio, status, spoken, normalization, metrics = asyncio.run(
        synthesize("5.", "en", "int4", Path(library), threads=4)
    )
    assert audio is not None and audio[0] == 22050
    assert spoken == "five." and "applied" in normalization
    assert "experimental" in status.lower() and "desktop" in metrics.lower()
    assert "peak RSS" in metrics and "CPU time" in metrics
    assert "per-stage timing: not measured" in metrics
    assert "summed across worker threads" in metrics
