import asyncio
import base64
import io
import wave
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from itantra_tts import Precision, SynthesisResult
from itantra_tts.api import create_app
from itantra_tts.application.service import SynthesisService
from itantra_tts.domain import SynthesisRequest


class _TinyAsyncBackend:
    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        return SynthesisResult(request.stream_id, request.sequence_no, request.language,
                              request.precision, "fixture-model", 16_000, b"\x01\x00" * 160)


def test_shared_service_returns_playable_wav_and_preserves_sequence() -> None:
    output = asyncio.run(SynthesisService(_TinyAsyncBackend()).synthesize(
        SynthesisRequest("stream-a", 7, "Hello.", "en", Precision.INT8)
    ))

    with wave.open(io.BytesIO(output.wav_bytes), "rb") as clip:
        assert clip.getnchannels() == 1
        assert clip.getframerate() == 16_000
        assert clip.readframes(clip.getnframes()) == b"\x01\x00" * 160
    assert output.sequence_numbers == (7,)
    assert output.model_ids == ("fixture-model",)
    assert output.real_time_factor > 0


def test_experimental_int4_service_normalizes_and_preserves_sequence() -> None:
    class RecordingBackend(_TinyAsyncBackend):
        def __init__(self):
            self.requests = []

        async def synthesize(self, request):
            self.requests.append(request)
            return await super().synthesize(request)

    backend = RecordingBackend()
    output = asyncio.run(SynthesisService(backend, allow_experimental_int4=True).synthesize(
        SynthesisRequest("stream-a", 7, "I have 5 apples.", "en", Precision.INT4)
    ))
    assert output.spoken_text == "I have five apples."
    assert backend.requests[0].text == "I have five apples."
    assert backend.requests[0].precision is Precision.INT4
    assert output.sequence_numbers == (7,)


def test_unreviewed_language_warns_when_number_is_passed_through() -> None:
    output = asyncio.run(SynthesisService(_TinyAsyncBackend()).synthesize(
        SynthesisRequest("stream-a", 0, "வாருங்கள் 123.", "ta", Precision.INT8)
    ))
    assert "123" in output.spoken_text
    assert "UNNORMALIZED_NUMERIC_FORM" in output.normalization


def test_http_contract_returns_wav_and_request_identifiers(monkeypatch) -> None:
    import itantra_tts.api as api_module

    monkeypatch.setattr(api_module, "AsyncLocalResearchSynthesisBackend",
                        lambda *args, **kwargs: _TinyAsyncBackend())
    async def request():
        async with AsyncClient(transport=ASGITransport(app=create_app(project_root=Path.cwd())),
                              base_url="http://test") as client:
            health_response = await client.get("/healthz")
            synthesis_response = await client.post("/v1/synthesize", json={
                "text": "नमस्ते।", "language": "hi", "precision": "int8",
                "stream_id": "receiver-1", "sequence_no": 12,
            })
            return health_response, synthesis_response

    health, response = asyncio.run(request())

    assert health.status_code == 200
    assert response.status_code == 200
    payload = response.json()
    assert payload["stream_id"] == "receiver-1"
    assert payload["sequence_no"] == 12
    assert payload["sentence_sequence_numbers"] == [12]
    assert payload["model_ids"] == ["fixture-model"]
    wav_bytes = base64.b64decode(payload["audio_base64"], validate=True)
    with wave.open(io.BytesIO(wav_bytes), "rb") as clip:
        assert clip.getnchannels() == 1
        assert clip.getframerate() == 16_000
        assert clip.getnframes() == 160


def test_http_contract_rejects_unavailable_int4(monkeypatch) -> None:
    import itantra_tts.api as api_module

    monkeypatch.setattr(api_module, "AsyncLocalResearchSynthesisBackend",
                        lambda *args, **kwargs: _TinyAsyncBackend())
    async def request():
        async with AsyncClient(transport=ASGITransport(app=create_app(project_root=Path.cwd())),
                              base_url="http://test") as client:
            return await client.post("/v1/synthesize", json={
                "text": "नमस्ते।", "language": "hi", "precision": "int4",
            })

    response = asyncio.run(request())

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "precision_unavailable"
