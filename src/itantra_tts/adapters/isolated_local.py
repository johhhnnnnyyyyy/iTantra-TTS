"""Low-retention desktop adapter: each local request runs in a disposable worker."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import replace
from pathlib import Path
from time import perf_counter_ns

from itantra_tts.domain import SynthesisErrorCode, SynthesisFailure, SynthesisRequest, SynthesisResult

from .local_research import LocalResearchSynthesisBackend
from .onnx_fastpitch_hifigan import OnnxFastPitchHifiGanBackend
from .onnx_odia_vits import OnnxOdiaVitsResearchBackend


class IsolatedLocalResearchSynthesisBackend:
    """Run one model in a child process, then terminate it to release desktop RSS."""

    memory_scope = "isolated synthesis worker process"

    def __init__(
        self,
        echobharat_root: Path,
        odia_root: Path,
        timeout_seconds: float = 120.0,
        odia_model_filename: str = "model.static-qdq.int8.onnx",
    ) -> None:
        self._echobharat_root = echobharat_root
        self._odia_root = odia_root
        self._timeout_seconds = timeout_seconds
        self._odia_model_filename = odia_model_filename

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult | SynthesisFailure:
        started_ns = perf_counter_ns()
        context = mp.get_context("fork")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_synthesize_once,
            args=(
                sender,
                request,
                self._echobharat_root,
                self._odia_root,
                self._odia_model_filename,
            ),
        )
        process.start()
        sender.close()
        if not receiver.poll(self._timeout_seconds):
            process.terminate()
            process.join()
            return SynthesisFailure(
                stream_id=request.stream_id,
                sequence_no=request.sequence_no,
                code=SynthesisErrorCode.MODEL_NOT_EXECUTABLE,
                detail="Local synthesis worker exceeded its timeout.",
            )
        result = receiver.recv()
        process.join()
        if isinstance(result, SynthesisResult) and result.metrics is not None:
            end_to_end_ms = (perf_counter_ns() - started_ns) / 1_000_000
            return replace(
                result,
                metrics=replace(
                    result.metrics,
                    wall_latency_ms=end_to_end_ms,
                    real_time_factor=(end_to_end_ms / result.metrics.audio_duration_ms),
                    process_cpu_utilization_percent=(
                        result.metrics.process_cpu_time_ms / end_to_end_ms * 100
                    ),
                    memory_scope=self.memory_scope,
                ),
            )
        return result

    def clear_cache(self) -> None:
        """Workers own model state, so the parent has no model cache to retain."""


def _synthesize_once(
    sender: object,
    request: SynthesisRequest,
    echobharat_root: Path,
    odia_root: Path,
    odia_model_filename: str,
) -> None:
    backend = LocalResearchSynthesisBackend(
        fastpitch_hifigan=OnnxFastPitchHifiGanBackend(
            echobharat_root, max_cached_languages=1, session_policy="sequential"
        ),
        odia_vits=OnnxOdiaVitsResearchBackend(odia_root, model_filename=odia_model_filename),
    )
    sender.send(backend.synthesize(request))
    sender.close()
