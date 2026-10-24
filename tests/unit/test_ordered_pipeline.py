import asyncio

import pytest

from itantra_tts import Precision, SynthesisRequest, SynthesisResult
from itantra_tts.application import AsyncSynthesisPipeline, GapPolicy, PipelineEventKind


def request(stream: str, sequence: int) -> SynthesisRequest:
    return SynthesisRequest(stream, sequence, "hello", "en", Precision.INT8)


def audio(item: SynthesisRequest) -> SynthesisResult:
    return SynthesisResult(item.stream_id, item.sequence_no, item.language, item.precision,
                           "fake", 16000, b"\x01\x00")


def test_out_of_order_completions_release_in_stream_order() -> None:
    async def run() -> None:
        gates = [asyncio.Event(), asyncio.Event()]

        async def synthesize(item: SynthesisRequest) -> SynthesisResult:
            await gates[item.sequence_no].wait()
            return audio(item)

        async with AsyncSynthesisPipeline(synthesize, workers=2) as pipeline:
            pipeline.register_stream("a")
            await pipeline.submit(request("a", 0))
            await pipeline.submit(request("a", 1))
            gates[1].set()
            await asyncio.sleep(0.01)
            assert pipeline.buffered_count("a") == 1
            gates[0].set()
            first = await pipeline.receive("a")
            second = await pipeline.receive("a")
            assert [first.sequence_no, second.sequence_no] == [0, 1]
            assert first.kind is PipelineEventKind.AUDIO

    asyncio.run(run())


def test_duplicate_and_stream_identity_are_checked() -> None:
    async def run() -> None:
        async def synthesize(item: SynthesisRequest) -> SynthesisResult:
            return audio(item)

        async with AsyncSynthesisPipeline(synthesize) as pipeline:
            pipeline.register_stream("a")
            pipeline.register_stream("b")
            await pipeline.submit(request("a", 0))
            with pytest.raises(ValueError, match="duplicate"):
                await pipeline.submit(request("a", 0))
            await pipeline.submit(request("b", 0))
            assert (await pipeline.receive("b")).stream_id == "b"
            assert (await pipeline.receive("a")).stream_id == "a"

    asyncio.run(run())


@pytest.mark.parametrize("policy", [GapPolicy.WAIT, GapPolicy.SKIP, GapPolicy.ABORT])
def test_gap_timeout_is_explicit_and_policy_controls_progress(policy: GapPolicy) -> None:
    async def run() -> None:
        async def synthesize(item: SynthesisRequest) -> SynthesisResult:
            return audio(item)

        async with AsyncSynthesisPipeline(synthesize, gap_timeout=0.02, gap_policy=policy) as p:
            p.register_stream("a")
            await p.submit(request("a", 1))
            event = await asyncio.wait_for(p.receive("a"), 1)
            assert event.kind is PipelineEventKind.MISSING_SEQUENCE_TIMEOUT
            assert event.sequence_no == 0
            if policy is GapPolicy.SKIP:
                assert (await p.receive("a")).sequence_no == 1
            elif policy is GapPolicy.WAIT:
                await p.submit(request("a", 0))
                assert (await p.receive("a")).sequence_no == 0
                assert (await p.receive("a")).sequence_no == 1
            else:
                with pytest.raises(RuntimeError, match="aborted"):
                    await p.receive("a")

    asyncio.run(run())


def test_cancelled_chunk_produces_ordered_event() -> None:
    async def run() -> None:
        async def synthesize(item: SynthesisRequest) -> SynthesisResult:
            await asyncio.sleep(10)
            return audio(item)

        async with AsyncSynthesisPipeline(synthesize) as pipeline:
            pipeline.register_stream("a")
            await pipeline.submit(request("a", 0))
            await pipeline.cancel("a", 0)
            event = await asyncio.wait_for(pipeline.receive("a"), 1)
            assert event.kind is PipelineEventKind.CANCELLED
            assert event.sequence_no == 0

    asyncio.run(run())


def test_bounded_capacity_preserves_room_for_missing_predecessor() -> None:
    async def run() -> None:
        gate = asyncio.Event()

        async def synthesize(item: SynthesisRequest) -> SynthesisResult:
            await gate.wait()
            return audio(item)

        async with AsyncSynthesisPipeline(synthesize, max_pending=3) as pipeline:
            pipeline.register_stream("a")
            await pipeline.submit(request("a", 1))
            await pipeline.submit(request("a", 2))
            pipeline.register_stream("b")
            with pytest.raises(BufferError):
                await pipeline.submit(request("b", 1))
            await pipeline.submit(request("a", 0))
            gate.set()
            assert (await pipeline.receive("a")).sequence_no == 0
            assert (await pipeline.receive("a")).sequence_no == 1
            assert (await pipeline.receive("a")).sequence_no == 2

    asyncio.run(run())


def test_all_capacity_is_usable_when_predecessor_already_submitted() -> None:
    async def run() -> None:
        gate = asyncio.Event()

        async def synthesize(item: SynthesisRequest) -> SynthesisResult:
            await gate.wait()
            return audio(item)

        async with AsyncSynthesisPipeline(synthesize, max_pending=3) as pipeline:
            pipeline.register_stream("a")
            for seq in range(3):
                await pipeline.submit(request("a", seq))
            gate.set()
            assert [(await pipeline.receive("a")).sequence_no for _ in range(3)] == [0, 1, 2]

    asyncio.run(run())


def test_backend_timeout_and_invalid_audio_are_ordered_failures_without_payload() -> None:
    async def run() -> None:
        async def synthesize(item: SynthesisRequest) -> SynthesisResult:
            if item.sequence_no == 0:
                await asyncio.sleep(1)
            result = audio(item)
            return SynthesisResult(result.stream_id, result.sequence_no, result.language,
                                   result.precision, result.model_id, result.sample_rate_hz, b"")

        async with AsyncSynthesisPipeline(synthesize, workers=2, task_timeout=0.02) as pipeline:
            pipeline.register_stream("a")
            await pipeline.submit(request("a", 0))
            await pipeline.submit(request("a", 1))
            first = await asyncio.wait_for(pipeline.receive("a"), 1)
            second = await pipeline.receive("a")
            assert first.kind is PipelineEventKind.SYNTHESIS_TIMEOUT
            assert second.kind is PipelineEventKind.INVALID_AUDIO
            assert "hello" not in repr(first)

    asyncio.run(run())


def test_completed_stream_can_be_closed_and_capacity_reused() -> None:
    async def run() -> None:
        async def synthesize(item: SynthesisRequest) -> SynthesisResult:
            return audio(item)

        async with AsyncSynthesisPipeline(synthesize, max_streams=1) as pipeline:
            pipeline.register_stream("a")
            await pipeline.submit(request("a", 0))
            with pytest.raises(RuntimeError, match="outstanding"):
                pipeline.close_stream("a")
            await pipeline.receive("a")
            pipeline.close_stream("a")
            pipeline.register_stream("b")
            await pipeline.submit(request("b", 0))
            assert (await pipeline.receive("b")).kind is PipelineEventKind.AUDIO

    asyncio.run(run())


def test_reused_stream_id_ignores_cancelled_queued_work_from_old_stream() -> None:
    async def run() -> None:
        gate = asyncio.Event()

        async def synthesize(item: SynthesisRequest) -> SynthesisResult:
            if item.stream_id == "busy":
                await gate.wait()
            result = audio(item)
            return SynthesisResult(result.stream_id, result.sequence_no, result.language,
                                   result.precision, result.model_id, result.sample_rate_hz,
                                   b"\x02\x00" if item.text == "new" else b"\x01\x00")

        async with AsyncSynthesisPipeline(synthesize, workers=1) as pipeline:
            pipeline.register_stream("busy")
            pipeline.register_stream("a")
            await pipeline.submit(request("busy", 0))
            await asyncio.sleep(0)
            await pipeline.submit(request("a", 0))
            await pipeline.cancel("a", 0)
            assert (await pipeline.receive("a")).kind is PipelineEventKind.CANCELLED
            pipeline.close_stream("a")
            pipeline.register_stream("a")
            await pipeline.submit(SynthesisRequest("a", 0, "new", "en", Precision.INT8))
            gate.set()
            event = await asyncio.wait_for(pipeline.receive("a"), 1)
            assert event.kind is PipelineEventKind.AUDIO
            assert isinstance(event.result, SynthesisResult)
            assert event.result.audio_pcm_s16le == b"\x02\x00"

    asyncio.run(run())


def test_oversized_text_and_audio_are_rejected_with_bounded_results() -> None:
    async def run() -> None:
        async def synthesize(item: SynthesisRequest) -> SynthesisResult:
            result = audio(item)
            return SynthesisResult(result.stream_id, result.sequence_no, result.language,
                                   result.precision, result.model_id, result.sample_rate_hz,
                                   b"\x01\x00" * 3)

        async with AsyncSynthesisPipeline(synthesize, max_text_chars=5, max_audio_bytes=4) as p:
            p.register_stream("a")
            with pytest.raises(ValueError, match="text"):
                await p.submit(SynthesisRequest("a", 0, "too long", "en", Precision.INT8))
            await p.submit(request("a", 0))
            assert (await p.receive("a")).kind is PipelineEventKind.INVALID_AUDIO

    asyncio.run(run())


def test_gap_age_starts_when_later_audio_completes_not_when_listener_arrives() -> None:
    async def run() -> None:
        async def synthesize(item: SynthesisRequest) -> SynthesisResult:
            return audio(item)

        async with AsyncSynthesisPipeline(synthesize, gap_timeout=0.03) as p:
            p.register_stream("a")
            await p.submit(request("a", 1))
            await asyncio.sleep(0.08)
            event = await asyncio.wait_for(p.receive("a"), 0.01)
            assert event.kind is PipelineEventKind.MISSING_SEQUENCE_TIMEOUT

    asyncio.run(run())
