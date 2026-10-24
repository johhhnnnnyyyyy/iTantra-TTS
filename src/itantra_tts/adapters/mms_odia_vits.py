"""Local, research-only Odia MMS VITS adapter kept separate from the public model bundle."""

from __future__ import annotations

import gc
from pathlib import Path
from time import perf_counter_ns, process_time_ns
from typing import Any

import numpy as np
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


class MmsOdiaVitsResearchBackend:
    """Run the downloaded Odia VITS checkpoint locally for research evaluation only."""

    def __init__(self, artifact_root: Path) -> None:
        self._artifact_root = artifact_root
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._process = psutil.Process()

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult | SynthesisFailure:
        if request.language != "or" or request.precision is not Precision.INT8:
            return self._unavailable(request, "This research VITS adapter supports Odia INT8 selection only.")

        request_start = perf_counter_ns()
        cpu_start = process_time_ns()
        rss_before = self._process.memory_info().rss
        cold_start = self._model is None
        try:
            self._ensure_loaded()
            assert self._model is not None and self._tokenizer is not None
            inputs = self._tokenizer(request.text, return_tensors="pt")
            vits_start = perf_counter_ns()
            import torch

            with torch.inference_mode():
                waveform = self._model(**inputs).waveform
            vits_end = perf_counter_ns()
        except (OSError, ValueError, RuntimeError) as error:
            return self._unavailable(request, f"Local Odia VITS execution failed: {error.__class__.__name__}.")

        mono = waveform.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        if mono.size == 0 or not np.isfinite(mono).all():
            return self._unavailable(request, "Local Odia VITS model returned invalid audio.")
        sample_rate_hz = int(self._model.config.sampling_rate)
        wall_end = perf_counter_ns()
        wall_latency_ms = (wall_end - request_start) / 1_000_000
        cpu_time_ms = (process_time_ns() - cpu_start) / 1_000_000
        audio_duration_ms = mono.size / sample_rate_hz * 1_000
        rss_after = self._process.memory_info().rss
        metrics = SynthesisMetrics(
            wall_latency_ms=wall_latency_ms,
            stage_one_label="VITS",
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
            model_disk_bytes=sum(path.stat().st_size for path in self._artifact_root.iterdir() if path.is_file()),
            provider="PyTorch CPU",
            runtime_version=_transformers_version(),
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
            model_id="mms-tts-ory-research-fp32",
            sample_rate_hz=sample_rate_hz,
            audio_pcm_s16le=pcm,
            metrics=metrics,
        )

    def clear_cache(self) -> None:
        """Release VITS weights when another model family is selected."""
        self._model = None
        self._tokenizer = None
        gc.collect()

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        from transformers import AutoTokenizer, VitsModel

        self._tokenizer = AutoTokenizer.from_pretrained(self._artifact_root, local_files_only=True)
        self._model = VitsModel.from_pretrained(self._artifact_root, local_files_only=True)
        self._model.eval()

    @staticmethod
    def _unavailable(request: SynthesisRequest, detail: str) -> SynthesisFailure:
        return SynthesisFailure(
            stream_id=request.stream_id,
            sequence_no=request.sequence_no,
            code=SynthesisErrorCode.MODEL_NOT_EXECUTABLE,
            detail=detail,
        )


def _transformers_version() -> str:
    import transformers

    return transformers.__version__
