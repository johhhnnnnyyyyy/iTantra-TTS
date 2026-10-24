"""Application service shared by the command line and local HTTP API."""

from __future__ import annotations

import io
import re
import wave
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Protocol

from itantra_tts.application.sentence_playback import synthesize_sentences
from itantra_tts.domain import (
    Precision,
    SynthesisFailure,
    SynthesisRequest,
    SynthesisResult,
)
from itantra_tts.normalization import SpeechNormalizer
from itantra_tts.normalization.indic_pilot import expand_cardinals


@dataclass(frozen=True, slots=True)
class SynthesisOutput:
    wav_bytes: bytes
    sample_rate_hz: int
    model_ids: tuple[str, ...]
    spoken_text: str
    normalization: str
    sequence_numbers: tuple[int, ...]
    elapsed_ms: float
    audio_duration_ms: float
    real_time_factor: float
    worker_metrics: tuple[dict[str, object] | None, ...]


class SynthesisServiceError(ValueError):
    """Safe, caller-displayable error without including submitted text."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _AsyncSynthesisBackend(Protocol):
    async def synthesize(
        self, request: SynthesisRequest
    ) -> SynthesisResult | SynthesisFailure: ...


class SynthesisService:
    """Normalize, synthesize in order, and package playable mono PCM WAV."""

    def __init__(self, backend: _AsyncSynthesisBackend, *, max_chars: int = 4_000,
                 max_audio_bytes: int = 20_000_000,
                 allow_experimental_int4: bool = False) -> None:
        self._backend = backend
        self._normalizer = SpeechNormalizer()
        self._max_chars = max_chars
        self._max_audio_bytes = max_audio_bytes
        self._allow_experimental_int4 = allow_experimental_int4

    async def synthesize(self, request: SynthesisRequest, *, gap_ms: int = 30) -> SynthesisOutput:
        if len(request.text) > self._max_chars:
            raise SynthesisServiceError(
                "text_too_long", f"Text exceeds {self._max_chars} characters."
            )
        if request.precision is Precision.INT4 and not self._allow_experimental_int4:
            raise SynthesisServiceError(
                "precision_unavailable", "INT4 is not enabled in this service."
            )
        if request.language in {"en", "hi"}:
            normalized = self._normalizer.normalize(request.text, request.language)
            sentences = normalized.sentences
            spoken_text = normalized.normalized_text
            warning_codes = ", ".join(item.code for item in normalized.warnings)
            normalization = f"W2 rule {normalized.rule_version} applied"
            if warning_codes:
                normalization += f"; unresolved: {warning_codes}"
        else:
            spoken_text, normalization = expand_cardinals(request.text.strip(), request.language)
            sentences = tuple(part for part in re.split(r"(?<=[.!?।])\s+", spoken_text) if part)
        if not sentences or len(sentences) > 32:
            raise SynthesisServiceError(
                "sentence_limit", "Provide text that splits into 1 to 32 sentences."
            )
        started = perf_counter()
        try:
            ordered = await synthesize_sentences(
                self._backend.synthesize, request.stream_id, sentences, request.language,
                request.precision, gap_ms=gap_ms, starting_sequence_no=request.sequence_no,
            )
        except Exception as exc:
            raise SynthesisServiceError(
                "synthesis_failed", "Local synthesis did not complete; no WAV was produced."
            ) from exc
        if len(ordered.audio_pcm_s16le) > self._max_audio_bytes:
            raise SynthesisServiceError(
                "audio_too_large", "Generated audio exceeded the configured output limit."
            )
        elapsed_ms = (perf_counter() - started) * 1000
        duration_ms = len(ordered.audio_pcm_s16le) / (2 * ordered.sample_rate_hz) * 1000
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(ordered.sample_rate_hz)
            wav.writeframes(ordered.audio_pcm_s16le)
        return SynthesisOutput(
            output.getvalue(), ordered.sample_rate_hz,
            tuple(sorted({item.model_id for item in ordered.results})), spoken_text,
            normalization, tuple(item.sequence_no for item in ordered.results),
            elapsed_ms, duration_ms,
            elapsed_ms / duration_ms if duration_ms else 0.0,
            tuple(asdict(item.metrics) if item.metrics is not None else
                  asdict(item.research_metrics) if item.research_metrics is not None else None
                  for item in ordered.results),
        )
