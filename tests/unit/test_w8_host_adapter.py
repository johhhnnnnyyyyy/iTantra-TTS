import asyncio

import pytest

from itantra_tts import Precision, SynthesisRequest, SynthesisResult
from itantra_tts.application import PipelineEventKind
from itantra_tts.host_adapter import HostTextChunk, ItantraTtsAdapter
from itantra_tts.domain import SynthesisErrorCode, SynthesisFailure


def test_two_streams_preserve_message_ids_and_sequence_order() -> None:
    async def run() -> None:
        gate = asyncio.Event()

        async def synthesize(request: SynthesisRequest) -> SynthesisResult:
            if request.stream_id == "a" and request.sequence_no == 0:
                await gate.wait()
            return SynthesisResult(request.stream_id, request.sequence_no, request.language,
                                   request.precision, "fake", 16000, b"\x01\x00")

        async with ItantraTtsAdapter(synthesize, workers=2) as adapter:
            adapter.open_stream("a")
            adapter.open_stream("b")
            await adapter.submit(HostTextChunk("a", 0, "a0", "hello", "en"))
            await adapter.submit(HostTextChunk("a", 1, "a1", "world", "en"))
            await adapter.submit(HostTextChunk("b", 0, "b0", "नमस्ते", "hi"))
            b = await asyncio.wait_for(adapter.receive("b"), 1)
            assert (b.message_id, b.kind) == ("b0", PipelineEventKind.AUDIO)
            gate.set()
            assert [
                (await adapter.receive("a")).message_id for _ in range(2)
            ] == ["a0", "a1"]

    asyncio.run(run())


def test_failure_is_ordered_and_does_not_echo_private_text() -> None:
    async def run() -> None:
        async def synthesize(request: SynthesisRequest) -> SynthesisResult:
            raise RuntimeError("secret text leaked by upstream")

        async with ItantraTtsAdapter(synthesize) as adapter:
            adapter.open_stream("a", start_sequence=5)
            await adapter.submit(HostTextChunk("a", 5, "msg-5", "private payload", "en"))
            event = await asyncio.wait_for(adapter.receive("a"), 1)
            assert (event.stream_id, event.sequence_no, event.message_id) == ("a", 5, "msg-5")
            assert event.kind is PipelineEventKind.FAILURE
            assert "private payload" not in repr(event)
            assert "secret text" not in repr(event)

    asyncio.run(run())


def test_duplicate_message_id_is_rejected_without_overwriting_mapping() -> None:
    async def run() -> None:
        async def synthesize(request: SynthesisRequest) -> SynthesisResult:
            return SynthesisResult(request.stream_id, request.sequence_no, request.language,
                                   request.precision, "fake", 16000, b"\x01\x00")

        async with ItantraTtsAdapter(synthesize) as adapter:
            adapter.open_stream("a")
            await adapter.submit(HostTextChunk("a", 0, "same", "hello", "en"))
            with pytest.raises(ValueError, match="message_id"):
                await adapter.submit(HostTextChunk("a", 1, "same", "again", "en"))
            assert (await adapter.receive("a")).message_id == "same"

    asyncio.run(run())


def test_invalid_language_is_rejected_at_host_seam() -> None:
    with pytest.raises(ValueError, match="language"):
        HostTextChunk("a", 0, "id", "hello", "xx", Precision.INT8)


def test_host_seam_applies_w2_normalization_and_returns_warning_codes() -> None:
    async def run() -> None:
        seen: list[str] = []

        async def synthesize(request: SynthesisRequest) -> SynthesisResult:
            seen.append(request.text)
            return SynthesisResult(request.stream_id, request.sequence_no, request.language,
                                   request.precision, "fake", 16000, b"\x01\x00")

        async with ItantraTtsAdapter(synthesize) as adapter:
            adapter.open_stream("a")
            await adapter.submit(HostTextChunk("a", 0, "id", "Meet at 5:30.", "en"))
            event = await adapter.receive("a")
            assert seen == ["Meet at five thirty."]
            assert event.normalization_warnings == ()

    asyncio.run(run())


def test_host_boundary_redacts_text_and_upstream_failure_detail() -> None:
    async def run() -> None:
        async def synthesize(request: SynthesisRequest) -> SynthesisFailure:
            return SynthesisFailure(request.stream_id, request.sequence_no,
                                    SynthesisErrorCode.MODEL_UNAVAILABLE,
                                    "private payload from model")

        chunk = HostTextChunk("a", 0, "id", "private payload", "en")
        assert "private payload" not in repr(chunk)
        async with ItantraTtsAdapter(synthesize) as adapter:
            adapter.open_stream("a")
            await adapter.submit(chunk)
            event = await adapter.receive("a")
            assert event.kind is PipelineEventKind.FAILURE
            assert "private payload" not in repr(event)
            assert event.result is not None
            assert "private payload" not in repr(event.result)

    asyncio.run(run())
