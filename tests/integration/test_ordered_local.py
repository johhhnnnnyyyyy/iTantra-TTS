"""Offline integration tests for real local model execution through W3."""

import asyncio
from pathlib import Path

import pytest

from itantra_tts import Precision, SynthesisRequest
from itantra_tts.application import AsyncSynthesisPipeline, PipelineEventKind


@pytest.mark.parametrize("language,text", [
    ("hi", "नमस्ते।"), ("gu", "નમસ્તે."), ("mr", "नमस्कार."),
    ("kn", "ನಮಸ್ಕಾರ."), ("ml", "നമസ്കാരം."), ("ta", "வணக்கம்."),
    ("te", "నమస్కారం."), ("or", "ନମସ୍କାର।"), ("bn", "নমস্কার।"),
    ("en", "Hello there."),
])
def test_real_local_model_returns_ordered_pcm(language: str, text: str) -> None:
    from itantra_tts.adapters.async_local import AsyncLocalResearchSynthesisBackend

    root = Path(__file__).resolve().parents[2] / "models" / "artifacts"
    model = (root / "mms-tts-ory-int8-research" / "model.static-qdq.int8.onnx"
             if language == "or" else root / "echobharat-int8" / language)
    if not model.exists():
        pytest.skip("local model artifact not provisioned")

    async def run() -> None:
        backend = AsyncLocalResearchSynthesisBackend(
            root / "echobharat-int8", root / "mms-tts-ory-int8-research"
        )
        async with AsyncSynthesisPipeline(backend.synthesize, task_timeout=60) as pipeline:
            pipeline.register_stream("offline")
            await pipeline.submit(SynthesisRequest("offline", 0, text, language, Precision.INT8))
            event = await asyncio.wait_for(pipeline.receive("offline"), timeout=65)
            assert event.kind is PipelineEventKind.AUDIO
            assert event.result is not None
            assert len(event.result.audio_pcm_s16le) > 2
            assert event.result.metrics is not None

    asyncio.run(run())


@pytest.mark.parametrize("language,text", [
    ("hi", "नमस्ते।"), ("gu", "નમસ્તે."), ("mr", "नमस्कार."),
    ("kn", "ನಮಸ್ಕಾರ."), ("ml", "നമസ്കാരം."), ("ta", "வணக்கம்."),
    ("te", "నమస్కారం."), ("or", "ନମସ୍କାର।"), ("bn", "নমস্কার।"),
    ("en", "Hello there."),
])
def test_real_local_http_api_returns_playable_wav(language: str, text: str) -> None:
    """Exercise the W5 HTTP contract end to end against each offline W4 model."""
    import base64
    import io
    import wave

    from httpx import ASGITransport, AsyncClient

    from itantra_tts.api import create_app

    project = Path(__file__).resolve().parents[2]
    root = project / "models" / "artifacts"
    model = (root / "mms-tts-ory-int8-research" / "model.static-qdq.int8.onnx"
             if language == "or" else root / "echobharat-int8" / language)
    if not model.exists():
        pytest.skip("local model artifact not provisioned")

    async def run() -> None:
        app = create_app(project_root=project)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/v1/synthesize", json={
                "text": text, "language": language, "precision": "int8",
                "stream_id": f"api-{language}", "sequence_no": 23,
            })
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["stream_id"] == f"api-{language}"
        assert payload["sequence_no"] == 23
        assert payload["sentence_sequence_numbers"] == [23]
        assert payload["model_ids"]
        wav_bytes = base64.b64decode(payload["audio_base64"], validate=True)
        with wave.open(io.BytesIO(wav_bytes), "rb") as clip:
            assert clip.getnchannels() == 1
            assert clip.getframerate() == (16_000 if language == "or" else 22_050)
            assert clip.getnframes() > 100
        assert payload["metrics"]["elapsed_ms"] > 0

    asyncio.run(run())


def test_cancelling_real_inference_reaps_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    from itantra_tts.adapters.async_local import AsyncLocalResearchSynthesisBackend

    root = Path(__file__).resolve().parents[2] / "models" / "artifacts"
    started = asyncio.Event()
    processes = []
    create = asyncio.create_subprocess_exec

    async def capture(*args, **kwargs):
        process = await create(*args, **kwargs)
        processes.append(process)
        started.set()
        return process

    async def run() -> None:
        monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
        backend = AsyncLocalResearchSynthesisBackend(
            root / "echobharat-int8", root / "mms-tts-ory-int8-research"
        )
        request = SynthesisRequest("cancel", 0, "नमस्ते।", "hi", Precision.INT8)
        task = asyncio.create_task(backend.synthesize(request))
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert processes[0].returncode is not None

    asyncio.run(run())


def test_web_ordered_two_sentences_use_real_model_and_report_metrics() -> None:
    from itantra_tts.adapters import FakeSynthesisBackend
    from itantra_tts.adapters.async_local import AsyncLocalResearchSynthesisBackend
    from itantra_tts.domain import ModelCatalog
    from itantra_tts.web.demo import WebSynthesisController

    project = Path(__file__).resolve().parents[2]
    root = project / "models" / "artifacts"
    if not (root / "echobharat-int8" / "hi").is_dir():
        pytest.skip("local Hindi model artifact not provisioned")
    controller = WebSynthesisController(
        ModelCatalog.from_yaml(project / "models" / "registry.yaml"), FakeSynthesisBackend(),
        ordered_backend=AsyncLocalResearchSynthesisBackend(
            root / "echobharat-int8", root / "mms-tts-ory-int8-research"
        ),
    )
    view = asyncio.run(controller.submit_ordered("नमस्ते। धन्यवाद।", "hi", "int8"))
    assert view.audio is not None
    assert view.audio[0] == 22050
    assert len(view.audio[1]) > 661
    assert "2 sentence" in view.status
    assert "per-sentence" in view.metrics.lower()
    assert "CPUExecutionProvider" in view.metrics
    assert "isolated subprocess worker" in view.metrics
    assert "excludes process startup" in view.metrics
