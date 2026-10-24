"""Executable ONNX Runtime adapter for the derived Odia dynamic-INT8 VITS graph."""

from __future__ import annotations

import gc
from pathlib import Path
from time import perf_counter_ns, process_time_ns

import numpy as np
import onnxruntime as ort
import psutil

from itantra_tts.domain import (
    Precision,
    SynthesisErrorCode,
    SynthesisFailure,
    SynthesisMetrics,
    SynthesisRequest,
    SynthesisResult,
)

from .onnx_fastpitch_hifigan import _peak_rss_bytes
from .odia_tokenizer import OdiaMmsTokenizer
from .ort_runtime import make_session_options


class OnnxOdiaVitsResearchBackend:
    """Run the local, derived dynamic-INT8 Odia VITS graph on CPU for research."""

    def __init__(
        self,
        artifact_root: Path,
        intra_op_threads: int = 4,
        model_filename: str = "model.static-qdq.int8.onnx",
    ) -> None:
        if model_filename not in {"model.int8.onnx", "model.static-qdq.int8.onnx"}:
            raise ValueError("unsupported Odia research model filename")
        self._artifact_root = artifact_root
        self._model_filename = model_filename
        self._session: ort.InferenceSession | None = None
        self._tokenizer: OdiaMmsTokenizer | None = None
        self._process = psutil.Process()
        self._intra_op_threads = intra_op_threads

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult | SynthesisFailure:
        if request.language != "or" or request.precision is not Precision.INT8:
            return self._unavailable(request, "This research adapter supports Odia INT8 selection only.")

        request_start = perf_counter_ns()
        cpu_start = process_time_ns()
        rss_before = self._process.memory_info().rss
        cold_start = self._session is None
        try:
            self._ensure_loaded()
            assert self._session is not None and self._tokenizer is not None
            input_ids = self._tokenizer.encode(request.text)
            vits_start = perf_counter_ns()
            waveform = self._session.run(["waveform"], {"input_ids": input_ids.astype(np.int64)})[0]
            vits_end = perf_counter_ns()
        except (OSError, ValueError, ort.OnnxRuntimeException) as error:
            return self._unavailable(request, f"Local Odia ONNX VITS execution failed: {error.__class__.__name__}.")

        mono = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if mono.size == 0 or not np.isfinite(mono).all():
            return self._unavailable(request, "Local Odia ONNX VITS model returned invalid audio.")
        rss_after = self._process.memory_info().rss
        wall_end = perf_counter_ns()
        wall_latency_ms = (wall_end - request_start) / 1_000_000
        cpu_time_ms = (process_time_ns() - cpu_start) / 1_000_000
        audio_duration_ms = mono.size / 16_000 * 1_000
        metrics = SynthesisMetrics(
            wall_latency_ms=wall_latency_ms,
            stage_one_label="VITS ONNX",
            fastpitch_latency_ms=(vits_end - vits_start) / 1_000_000,
            stage_two_label="n/a",
            hifigan_latency_ms=0.0,
            process_cpu_time_ms=cpu_time_ms,
            process_cpu_utilization_percent=(cpu_time_ms / wall_latency_ms * 100)
            if wall_latency_ms
            else 0.0,
            rss_before_bytes=rss_before,
            rss_after_bytes=rss_after,
            selected_model_rss_delta_bytes=rss_after - rss_before,
            peak_rss_bytes=_peak_rss_bytes(),
            audio_duration_ms=audio_duration_ms,
            real_time_factor=(wall_latency_ms / audio_duration_ms) if audio_duration_ms else 0.0,
            model_disk_bytes=sum(
                (self._artifact_root / name).stat().st_size
                for name in (self._model_filename, "vocab.json", "config.json")
            ),
            provider="CPUExecutionProvider",
            runtime_version=ort.__version__,
            cold_start=cold_start,
            resident_model_count=1,
            memory_scope="whole demo process",
        )
        pcm = np.rint(np.clip(mono, -1.0, 1.0) * 32_767.0).astype("<i2").tobytes()
        return SynthesisResult(
            stream_id=request.stream_id,
            sequence_no=request.sequence_no,
            language=request.language,
            precision=request.precision,
            model_id=(
                "mms-tts-ory-onnx-static-qdq-int8-research"
                if self._model_filename == "model.static-qdq.int8.onnx"
                else "mms-tts-ory-onnx-dynamic-int8-research"
            ),
            sample_rate_hz=16_000,
            audio_pcm_s16le=pcm,
            metrics=metrics,
        )

    def clear_cache(self) -> None:
        """Release the current ONNX session when another runtime is selected."""
        self._session = None
        self._tokenizer = None
        gc.collect()

    def _ensure_loaded(self) -> None:
        if self._session is not None and self._tokenizer is not None:
            return
        self._tokenizer = OdiaMmsTokenizer(self._artifact_root / "vocab.json")
        self._session = ort.InferenceSession(
            str(self._artifact_root / self._model_filename),
            sess_options=make_session_options(self._intra_op_threads),
            providers=["CPUExecutionProvider"],
        )

    @staticmethod
    def _unavailable(request: SynthesisRequest, detail: str) -> SynthesisFailure:
        return SynthesisFailure(
            stream_id=request.stream_id,
            sequence_no=request.sequence_no,
            code=SynthesisErrorCode.MODEL_NOT_EXECUTABLE,
            detail=detail,
        )
