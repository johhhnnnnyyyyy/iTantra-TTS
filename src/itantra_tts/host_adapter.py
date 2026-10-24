"""iTantra-facing interface for already translated target-language chunks."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from itantra_tts.application.ordered_pipeline import (
    AsyncSynthesisPipeline,
    GapPolicy,
    PipelineEventKind,
)
from itantra_tts.domain import Precision, SynthesisFailure, SynthesisRequest, SynthesisResult
from itantra_tts.normalization import SpeechNormalizer


@dataclass(frozen=True, slots=True)
class HostTextChunk:
    """One host-provided, target-language chunk; transport and MT stay outside."""

    stream_id: str
    sequence_no: int
    message_id: str
    target_text: str = field(repr=False)
    language: str
    precision: Precision = Precision.INT8

    def __post_init__(self) -> None:
        if not isinstance(self.message_id, str) or not self.message_id.strip():
            raise ValueError("message_id must be non-empty")
        SynthesisRequest(self.stream_id, self.sequence_no, self.target_text,
                         self.language, self.precision)


@dataclass(frozen=True, slots=True)
class HostPlaybackEvent:
    """Ordered audio or a safe typed failure without the submitted text."""

    stream_id: str
    sequence_no: int
    message_id: str | None
    kind: PipelineEventKind
    result: SynthesisResult | SynthesisFailure | None = field(repr=False)
    normalization_warnings: tuple[str, ...] = ()


class ItantraTtsAdapter:
    """Map host message IDs onto the bounded TTS reorder pipeline."""

    def __init__(
        self,
        synthesize: Callable[[SynthesisRequest], Awaitable[SynthesisResult | SynthesisFailure]],
        *,
        workers: int = 1,
        gap_timeout: float = 5.0,
        gap_policy: GapPolicy = GapPolicy.WAIT,
    ) -> None:
        self._pipeline = AsyncSynthesisPipeline(
            synthesize, workers=workers, gap_timeout=gap_timeout, gap_policy=gap_policy
        )
        self._ids: dict[tuple[str, int], str] = {}
        self._used_ids: dict[str, set[str]] = {}
        self._warnings: dict[tuple[str, int], tuple[str, ...]] = {}
        self._normalizer = SpeechNormalizer()

    async def __aenter__(self) -> ItantraTtsAdapter:
        await self._pipeline.__aenter__()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self._pipeline.__aexit__(*args)
        self._ids.clear()
        self._used_ids.clear()
        self._warnings.clear()

    def open_stream(self, stream_id: str, *, start_sequence: int = 0) -> None:
        self._pipeline.register_stream(stream_id, start_sequence=start_sequence)
        self._used_ids[stream_id] = set()

    async def submit(self, chunk: HostTextChunk) -> None:
        if chunk.message_id in self._used_ids.get(chunk.stream_id, set()):
            raise ValueError("duplicate message_id in stream")
        text = chunk.target_text
        warnings: tuple[str, ...] = ()
        if chunk.language in {"en", "hi"}:
            normalized = self._normalizer.normalize(text, chunk.language)
            text = normalized.normalized_text
            warnings = tuple(item.code for item in normalized.warnings)
        request = SynthesisRequest(chunk.stream_id, chunk.sequence_no, text,
                                   chunk.language, chunk.precision)
        await self._pipeline.submit(request)
        key = (chunk.stream_id, chunk.sequence_no)
        self._ids[key] = chunk.message_id
        self._warnings[key] = warnings
        self._used_ids[chunk.stream_id].add(chunk.message_id)

    async def receive(self, stream_id: str) -> HostPlaybackEvent:
        event = await self._pipeline.receive(stream_id)
        key = (event.stream_id, event.sequence_no)
        message_id = self._ids.pop(key, None)
        result = event.result
        if isinstance(result, SynthesisFailure):
            result = SynthesisFailure(result.stream_id, result.sequence_no, result.code,
                                      "TTS synthesis failed; inspect the typed code.")
        return HostPlaybackEvent(event.stream_id, event.sequence_no, message_id,
                                 event.kind, result, self._warnings.pop(key, ()))

    async def cancel(self, stream_id: str, sequence_no: int) -> None:
        await self._pipeline.cancel(stream_id, sequence_no)

    def close_stream(self, stream_id: str) -> None:
        self._pipeline.close_stream(stream_id)
        self._used_ids.pop(stream_id, None)
