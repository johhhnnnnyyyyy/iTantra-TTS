"""Bounded async synthesis with per-stream ordered event delivery.

The caller supplies already-normalized sentence requests and an asynchronous
backend. A backend must cooperate with asyncio cancellation; do not wrap a
blocking model call in a cancellable thread and assume it has stopped.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum

from itantra_tts.domain import SynthesisFailure, SynthesisRequest, SynthesisResult


class GapPolicy(StrEnum):
    WAIT = "wait"
    SKIP = "skip"
    ABORT = "abort"


class PipelineEventKind(StrEnum):
    AUDIO = "audio"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    MISSING_SEQUENCE_TIMEOUT = "missing_sequence_timeout"
    SYNTHESIS_TIMEOUT = "synthesis_timeout"
    INVALID_AUDIO = "invalid_audio"


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    stream_id: str
    sequence_no: int
    kind: PipelineEventKind
    result: SynthesisResult | SynthesisFailure | None = None


@dataclass(slots=True)
class _Stream:
    next_sequence: int
    outstanding: set[int] = field(default_factory=set)
    ready: dict[int, PipelineEvent] = field(default_factory=dict)
    jobs: dict[int, asyncio.Task[SynthesisResult | SynthesisFailure]] = field(default_factory=dict)
    gap_started: float | None = None
    gap_reported: bool = False
    aborted: bool = False


class AsyncSynthesisPipeline:
    """Bounded worker pool; receive() returns audio and failures in sequence order."""

    def __init__(
        self,
        synthesize: Callable[[SynthesisRequest], Awaitable[SynthesisResult | SynthesisFailure]],
        *,
        workers: int = 1,
        max_pending: int = 32,
        max_streams: int = 8,
        max_text_chars: int = 1000,
        max_audio_bytes: int = 4_000_000,
        gap_timeout: float = 5.0,
        task_timeout: float = 120.0,
        gap_policy: GapPolicy = GapPolicy.WAIT,
    ) -> None:
        if min(workers, max_pending, max_streams, max_text_chars, max_audio_bytes) < 1:
            raise ValueError("worker, pending, stream and payload limits must be positive")
        if gap_timeout <= 0 or task_timeout <= 0:
            raise ValueError("timeouts must be positive")
        self._synthesize = synthesize
        self._worker_count = workers
        self._max_pending = max_pending
        self._max_streams = max_streams
        self._max_text_chars = max_text_chars
        self._max_audio_bytes = max_audio_bytes
        self._gap_timeout = gap_timeout
        self._task_timeout = task_timeout
        self._policy = GapPolicy(gap_policy)
        self._streams: dict[str, _Stream] = {}
        self._queue: asyncio.Queue[tuple[SynthesisRequest, _Stream]] = asyncio.Queue(
            maxsize=max_pending
        )
        self._condition = asyncio.Condition()
        self._workers: list[asyncio.Task[None]] = []
        self._closed = False

    async def __aenter__(self) -> AsyncSynthesisPipeline:
        if self._workers or self._closed:
            raise RuntimeError("pipeline already started or closed")
        self._workers = [asyncio.create_task(self._worker()) for _ in range(self._worker_count)]
        return self

    async def __aexit__(self, *_: object) -> None:
        self._closed = True
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()
        async with self._condition:
            self._streams.clear()
            self._condition.notify_all()

    def register_stream(self, stream_id: str, *, start_sequence: int = 0) -> None:
        if not self._workers or self._closed:
            raise RuntimeError("pipeline is not running")
        if not isinstance(stream_id, str) or not stream_id.strip():
            raise ValueError("stream_id must be non-empty")
        if not isinstance(start_sequence, int) or isinstance(start_sequence, bool) or start_sequence < 0:
            raise ValueError("start_sequence must be a non-negative integer")
        if stream_id in self._streams:
            raise ValueError("duplicate stream")
        if len(self._streams) >= self._max_streams:
            raise BufferError("stream limit reached")
        self._streams[stream_id] = _Stream(start_sequence)

    async def submit(self, request: SynthesisRequest) -> None:
        async with self._condition:
            stream = self._stream(request.stream_id)
            if len(request.text) > self._max_text_chars:
                raise ValueError("text exceeds the configured limit")
            if stream.aborted:
                raise RuntimeError("stream aborted")
            seq = request.sequence_no
            if seq < stream.next_sequence or seq in stream.outstanding:
                raise ValueError("duplicate or expired sequence")
            # Reserve a slot for the missing predecessor so a full reorder
            # window cannot deadlock recovery. Bound the sequence horizon too.
            if seq - stream.next_sequence >= self._max_pending:
                raise BufferError("sequence outside reorder window")
            count = sum(len(s.outstanding) for s in self._streams.values())
            unresolved_gap = any(
                s.outstanding and s.next_sequence not in s.outstanding
                for s in self._streams.values() if s is not stream
            ) or (seq != stream.next_sequence and stream.next_sequence not in stream.outstanding)
            if count >= self._max_pending or (unresolved_gap and count >= self._max_pending - 1):
                raise BufferError("pipeline capacity reached")
            try:
                self._queue.put_nowait((request, stream))
            except asyncio.QueueFull as error:
                raise BufferError("worker queue capacity reached") from error
            stream.outstanding.add(seq)
            self._condition.notify_all()

    async def cancel(self, stream_id: str, sequence_no: int) -> None:
        async with self._condition:
            stream = self._stream(stream_id)
            if sequence_no not in stream.outstanding or sequence_no in stream.ready:
                raise ValueError("sequence is not cancellable")
            task = stream.jobs.get(sequence_no)
            if task is not None:
                task.cancel()
            stream.ready[sequence_no] = PipelineEvent(stream_id, sequence_no, PipelineEventKind.CANCELLED)
            self._condition.notify_all()

    def buffered_count(self, stream_id: str) -> int:
        return len(self._stream(stream_id).ready)

    def close_stream(self, stream_id: str) -> None:
        stream = self._stream(stream_id)
        if stream.outstanding or stream.jobs:
            raise RuntimeError("stream has outstanding work")
        del self._streams[stream_id]

    async def receive(self, stream_id: str) -> PipelineEvent:
        async with self._condition:
            stream = self._stream(stream_id)
            while True:
                if self._closed:
                    raise RuntimeError("pipeline closed")
                if stream.aborted:
                    raise RuntimeError("stream aborted")
                seq = stream.next_sequence
                event = stream.ready.pop(seq, None)
                if event is not None:
                    stream.outstanding.discard(seq)
                    stream.next_sequence += 1
                    stream.gap_started = None
                    stream.gap_reported = False
                    return event
                later_ready = any(number > seq for number in stream.ready)
                if not later_ready:
                    stream.gap_started = None
                    stream.gap_reported = False
                    await self._condition.wait()
                    continue
                loop = asyncio.get_running_loop()
                if stream.gap_started is None:
                    stream.gap_started = loop.time()
                remaining = stream.gap_started + self._gap_timeout - loop.time()
                if remaining <= 0 and not stream.gap_reported:
                    stream.gap_reported = True
                    if self._policy is GapPolicy.SKIP:
                        task = stream.jobs.get(seq)
                        if task:
                            task.cancel()
                        stream.outstanding.discard(seq)
                        stream.next_sequence += 1
                        stream.gap_started = None
                        stream.gap_reported = False
                    elif self._policy is GapPolicy.ABORT:
                        stream.aborted = True
                        for task in stream.jobs.values():
                            task.cancel()
                        stream.ready.clear()
                        stream.outstanding.clear()
                    return PipelineEvent(stream_id, seq, PipelineEventKind.MISSING_SEQUENCE_TIMEOUT)
                if stream.gap_reported:
                    await self._condition.wait()
                else:
                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout=max(remaining, 0))
                    except TimeoutError:
                        pass

    def _stream(self, stream_id: str) -> _Stream:
        if self._closed:
            raise RuntimeError("pipeline closed")
        if stream_id not in self._streams:
            raise ValueError("stream not registered")
        return self._streams[stream_id]

    async def _worker(self) -> None:
        while True:
            request, stream = await self._queue.get()
            try:
                seq = request.sequence_no
                if stream.aborted or seq not in stream.outstanding or seq in stream.ready:
                    continue
                task = asyncio.create_task(self._synthesize(request))
                stream.jobs[seq] = task
                try:
                    result = await asyncio.wait_for(task, self._task_timeout)
                    valid = (
                        isinstance(result, SynthesisResult)
                        and result.stream_id == request.stream_id
                        and result.sequence_no == seq
                        and result.language == request.language
                        and result.precision == request.precision
                        and bool(result.model_id)
                        and result.sample_rate_hz in {16000, 22050, 24000, 44100, 48000}
                        and isinstance(result.audio_pcm_s16le, bytes)
                        and len(result.audio_pcm_s16le) >= 2
                        and len(result.audio_pcm_s16le) <= self._max_audio_bytes
                        and len(result.audio_pcm_s16le) % 2 == 0
                    )
                    matching_failure = (
                        isinstance(result, SynthesisFailure)
                        and result.stream_id == request.stream_id and result.sequence_no == seq
                    )
                    if valid:
                        event = PipelineEvent(request.stream_id, seq, PipelineEventKind.AUDIO, result)
                    elif matching_failure:
                        event = PipelineEvent(request.stream_id, seq, PipelineEventKind.FAILURE, result)
                    else:
                        event = PipelineEvent(request.stream_id, seq, PipelineEventKind.INVALID_AUDIO)
                except asyncio.CancelledError:
                    if asyncio.current_task().cancelling():
                        raise
                    event = PipelineEvent(request.stream_id, seq, PipelineEventKind.CANCELLED)
                except TimeoutError:
                    event = PipelineEvent(request.stream_id, seq, PipelineEventKind.SYNTHESIS_TIMEOUT)
                except Exception:
                    # Do not expose exception messages, which may contain user text.
                    event = PipelineEvent(request.stream_id, seq, PipelineEventKind.FAILURE)
                finally:
                    stream.jobs.pop(seq, None)
                async with self._condition:
                    if not stream.aborted and seq in stream.outstanding and seq not in stream.ready:
                        stream.ready[seq] = event
                        if (
                            seq > stream.next_sequence
                            and stream.next_sequence not in stream.ready
                            and stream.gap_started is None
                        ):
                            stream.gap_started = asyncio.get_running_loop().time()
                    self._condition.notify_all()
            finally:
                self._queue.task_done()
