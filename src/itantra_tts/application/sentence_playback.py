"""Build one ordered playable PCM sequence from bounded sentence requests."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from itantra_tts.application.ordered_pipeline import AsyncSynthesisPipeline, PipelineEventKind
from itantra_tts.domain import Precision, SynthesisFailure, SynthesisRequest, SynthesisResult


class SentenceSynthesisError(RuntimeError):
    """One sentence failed; no partial audio should be presented as complete."""


@dataclass(frozen=True, slots=True)
class OrderedAudio:
    sample_rate_hz: int
    audio_pcm_s16le: bytes
    results: tuple[SynthesisResult, ...]


async def synthesize_sentences(
    synthesize: Callable[[SynthesisRequest], Awaitable[SynthesisResult | SynthesisFailure]],
    stream_id: str,
    sentences: Sequence[str],
    language: str,
    precision: Precision,
    *,
    workers: int = 1,
    gap_ms: int = 30,
    starting_sequence_no: int = 0,
) -> OrderedAudio:
    """Synthesize up to 32 ordered sentences, rejecting partial or mixed-rate output."""
    if not 1 <= len(sentences) <= 32:
        raise ValueError("expected between 1 and 32 sentences")
    if not 20 <= gap_ms <= 60:
        raise ValueError("gap must be between 20 and 60 ms")
    results: list[SynthesisResult] = []
    async with AsyncSynthesisPipeline(synthesize, workers=workers, max_pending=32) as pipeline:
        pipeline.register_stream(stream_id, start_sequence=starting_sequence_no)
        for index, sentence in enumerate(sentences):
            await pipeline.submit(SynthesisRequest(
                stream_id, starting_sequence_no + index, sentence, language, precision
            ))
        for _ in sentences:
            event = await pipeline.receive(stream_id)
            if event.kind is not PipelineEventKind.AUDIO or not isinstance(event.result, SynthesisResult):
                raise SentenceSynthesisError(f"sentence {event.sequence_no}: {event.kind.value}")
            results.append(event.result)
    rate = results[0].sample_rate_hz
    if any(item.sample_rate_hz != rate for item in results):
        raise SentenceSynthesisError("sentence sample rates differ")
    gap = bytes((rate * gap_ms // 1000) * 2)
    return OrderedAudio(rate, gap.join(item.audio_pcm_s16le for item in results), tuple(results))
