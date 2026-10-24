"""Optional loopback-first HTTP API for local offline synthesis."""

from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from itantra_tts.adapters.async_local import AsyncLocalResearchSynthesisBackend
from itantra_tts.application.service import SynthesisService, SynthesisServiceError
from itantra_tts.domain import Precision, SynthesisRequest


class SynthesizeBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=4_000)
    language: str
    precision: str = "int8"
    stream_id: str = Field(default="http-local", min_length=1, max_length=128)
    sequence_no: int = Field(default=0, ge=0)


def create_app(*, project_root: Path | None = None, max_concurrent_requests: int = 1) -> FastAPI:
    """Construct a local app. It never calls hosted/browser speech services."""
    root = project_root or Path(__file__).resolve().parents[2]
    backend = AsyncLocalResearchSynthesisBackend(
        root / "models" / "artifacts" / "echobharat-int8",
        root / "models" / "artifacts" / "mms-tts-ory-int8-research",
        os.environ.get("ITANTRA_TTS_ODIA_RESEARCH_MODEL_FILE", "model.static-qdq.int8.onnx"),
    )
    service = SynthesisService(backend)
    if max_concurrent_requests < 1:
        raise ValueError("max_concurrent_requests must be positive")
    slots = asyncio.Semaphore(max_concurrent_requests)
    app = FastAPI(title="iTantra TTS local API", version="0.1.0a0", docs_url=None, redoc_url=None)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "mode": "local-offline-research"}

    @app.post("/v1/synthesize")
    async def synthesize(body: SynthesizeBody) -> dict[str, object]:
        try:
            request = SynthesisRequest(body.stream_id, body.sequence_no, body.text,
                                       body.language, Precision(body.precision))
            if slots.locked():
                raise HTTPException(
                    429,
                    {"code": "busy", "message": "Local synthesis capacity is full."},
                )
            async with slots:
                result = await service.synthesize(request)
        except HTTPException:
            raise
        except (ValueError, TypeError) as exc:
            if isinstance(exc, SynthesisServiceError):
                raise HTTPException(422, {"code": exc.code, "message": str(exc)}) from exc
            raise HTTPException(422, {"code": "invalid_request", "message": str(exc)}) from exc
        return {
            "stream_id": request.stream_id, "sequence_no": request.sequence_no,
            "language": request.language, "precision": request.precision.value,
            "model_ids": list(result.model_ids), "sample_rate_hz": result.sample_rate_hz,
            "sentence_sequence_numbers": list(result.sequence_numbers),
            "audio_format": "wav",
            "audio_base64": base64.b64encode(result.wav_bytes).decode("ascii"),
            "spoken_text": result.spoken_text, "normalization": result.normalization,
            "metrics": {
                "elapsed_ms": result.elapsed_ms,
                "audio_duration_ms": result.audio_duration_ms,
                "real_time_factor": result.real_time_factor,
                "worker_requests": list(result.worker_metrics),
            },
        }

    return app
