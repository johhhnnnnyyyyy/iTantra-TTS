"""Local dynamic-INT8 EchoBharat FastPitch and HiFi-GAN synthesis adapter."""

from __future__ import annotations

import json
import gc
import resource
import sys
from dataclasses import dataclass
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
from .ort_runtime import make_session_options


_SAMPLE_RATE_HZ = 22_050
_PROVIDER = "CPUExecutionProvider"
_TOKEN_FALLBACKS = {"।": "."}


@dataclass(frozen=True, slots=True)
class _LoadedLanguageModel:
    fastpitch: ort.InferenceSession
    hifigan: ort.InferenceSession | None
    token_ids: dict[str, int]
    symbols_by_longest_match: tuple[str, ...]
    model_disk_bytes: int


class OnnxFastPitchHifiGanBackend:
    """Run downloaded EchoBharat INT8 graph pairs entirely on the local CPU."""

    def __init__(
        self,
        artifact_root: Path,
        max_cached_languages: int = 1,
        intra_op_threads: int = 4,
        session_policy: str = "resident",
    ) -> None:
        if max_cached_languages < 0:
            raise ValueError("max_cached_languages must be non-negative")
        if session_policy not in {"resident", "sequential"}:
            raise ValueError("session_policy must be resident or sequential")
        self._artifact_root = artifact_root
        self._models: dict[str, _LoadedLanguageModel] = {}
        self._process = psutil.Process()
        self._max_cached_languages = max_cached_languages
        self._intra_op_threads = intra_op_threads
        self._session_policy = session_policy

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult | SynthesisFailure:
        """Synthesize one request and report metrics from this exact execution only."""
        if request.precision is not Precision.INT8 or request.language == "or":
            return self._unavailable(request, "No local executable ONNX INT8 model exists for this choice.")

        request_start = perf_counter_ns()
        cpu_start = process_time_ns()
        rss_before = self._process.memory_info().rss
        cold_start = self._session_policy == "sequential" or request.language not in self._models
        try:
            if self._session_policy == "sequential":
                model = self._load_language_model(request.language, include_hifigan=False)
                input_tokens = np.asarray([self._tokenize(request.text, model)], dtype=np.int64)
                model_disk_bytes = model.model_disk_bytes
                fastpitch_start = perf_counter_ns()
                mel = model.fastpitch.run(None, {"text": input_tokens})[0]
                fastpitch_end = perf_counter_ns()
                del model
                gc.collect()
                hifigan_path = (
                    self._artifact_root / request.language / f"hifigan-{request.language}.int8.onnx"
                )
                hifigan = ort.InferenceSession(
                    str(hifigan_path),
                    sess_options=make_session_options(self._intra_op_threads),
                    providers=[_PROVIDER],
                )
                waveform = hifigan.run(None, {"mel": mel})[0]
                hifigan_end = perf_counter_ns()
                del hifigan
                gc.collect()
            else:
                model = self._models.get(request.language)
                if model is None:
                    model = self._load_language_model(request.language)
                    self._cache_model(request.language, model)
                input_tokens = np.asarray([self._tokenize(request.text, model)], dtype=np.int64)
                model_disk_bytes = model.model_disk_bytes
                fastpitch_start = perf_counter_ns()
                mel = model.fastpitch.run(None, {"text": input_tokens})[0]
                fastpitch_end = perf_counter_ns()
                assert model.hifigan is not None
                waveform = model.hifigan.run(None, {"mel": mel})[0]
                hifigan_end = perf_counter_ns()
        except (OSError, ValueError, json.JSONDecodeError, ort.OnnxRuntimeException) as error:
            return self._unavailable(request, f"Local ONNX execution failed: {error.__class__.__name__}.")

        mono = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if mono.size == 0 or not np.isfinite(mono).all():
            return self._unavailable(request, "Local ONNX model returned invalid audio.")
        pcm = np.rint(np.clip(mono, -1.0, 1.0) * 32_767.0).astype("<i2").tobytes()
        wall_end = perf_counter_ns()
        wall_latency_ms = (wall_end - request_start) / 1_000_000
        cpu_time_ms = (process_time_ns() - cpu_start) / 1_000_000
        audio_duration_ms = mono.size / _SAMPLE_RATE_HZ * 1_000
        metrics = SynthesisMetrics(
            wall_latency_ms=wall_latency_ms,
            stage_one_label="FastPitch",
            fastpitch_latency_ms=(fastpitch_end - fastpitch_start) / 1_000_000,
            stage_two_label="HiFi-GAN",
            hifigan_latency_ms=(hifigan_end - fastpitch_end) / 1_000_000,
            process_cpu_time_ms=cpu_time_ms,
            process_cpu_utilization_percent=(cpu_time_ms / wall_latency_ms * 100)
            if wall_latency_ms
            else 0.0,
            rss_before_bytes=rss_before,
            rss_after_bytes=self._process.memory_info().rss,
            selected_model_rss_delta_bytes=self._process.memory_info().rss - rss_before,
            peak_rss_bytes=_peak_rss_bytes(),
            audio_duration_ms=audio_duration_ms,
            real_time_factor=(wall_latency_ms / audio_duration_ms) if audio_duration_ms else 0.0,
            model_disk_bytes=model_disk_bytes,
            provider=_PROVIDER,
            runtime_version=ort.__version__,
            cold_start=cold_start,
            resident_model_count=len(self._models),
            memory_scope="whole demo process",
        )
        return SynthesisResult(
            stream_id=request.stream_id,
            sequence_no=request.sequence_no,
            language=request.language,
            precision=request.precision,
            model_id=f"echobharat-{request.language}-int8",
            sample_rate_hz=_SAMPLE_RATE_HZ,
            audio_pcm_s16le=pcm,
            metrics=metrics,
        )

    def clear_cache(self) -> None:
        """Release all model sessions before switching to another runtime family."""
        self._models.clear()
        gc.collect()

    def _cache_model(self, language: str, model: _LoadedLanguageModel) -> None:
        while len(self._models) >= self._max_cached_languages and self._models:
            oldest_language = next(iter(self._models))
            del self._models[oldest_language]
        if self._max_cached_languages:
            self._models[language] = model

    def _load_language_model(
        self, language: str, include_hifigan: bool = True
    ) -> _LoadedLanguageModel:
        language_root = self._artifact_root / language
        fastpitch_path = language_root / f"fastpitch-{language}.int8.onnx"
        hifigan_path = language_root / f"hifigan-{language}.int8.onnx"
        token_path = language_root / f"fastpitch-{language}.tokens.json"
        token_list = json.loads(token_path.read_text(encoding="utf-8"))
        if not isinstance(token_list, list) or not all(isinstance(token, str) for token in token_list):
            raise ValueError("invalid token map")
        token_ids: dict[str, int] = {}
        for token_id, symbol in enumerate(token_list):
            token_ids.setdefault(symbol, token_id)
        if "<BLNK>" not in token_ids:
            raise ValueError("token map has no blank token")
        return _LoadedLanguageModel(
            fastpitch=ort.InferenceSession(
                str(fastpitch_path), sess_options=make_session_options(self._intra_op_threads), providers=[_PROVIDER]
            ),
            hifigan=(
                ort.InferenceSession(
                    str(hifigan_path),
                    sess_options=make_session_options(self._intra_op_threads),
                    providers=[_PROVIDER],
                )
                if include_hifigan
                else None
            ),
            token_ids=token_ids,
            symbols_by_longest_match=tuple(
                sorted(
                    (symbol for symbol in token_ids if not symbol.startswith("<")),
                    key=len,
                    reverse=True,
                )
            ),
            model_disk_bytes=sum(path.stat().st_size for path in (fastpitch_path, hifigan_path, token_path)),
        )

    @staticmethod
    def _tokenize(text: str, model: _LoadedLanguageModel) -> list[int]:
        prepared = "".join(_TOKEN_FALLBACKS.get(character, character) for character in text)
        blank_id = model.token_ids["<BLNK>"]
        token_ids: list[int] = []
        position = 0
        while position < len(prepared):
            symbol = next(
                (candidate for candidate in model.symbols_by_longest_match if prepared.startswith(candidate, position)),
                None,
            )
            if symbol is None:
                token_ids.append(blank_id)
                position += 1
            else:
                token_ids.append(model.token_ids[symbol])
                position += len(symbol)
        if not token_ids:
            raise ValueError("text has no synthesizable tokens")
        return token_ids

    @staticmethod
    def _unavailable(request: SynthesisRequest, detail: str) -> SynthesisFailure:
        return SynthesisFailure(
            stream_id=request.stream_id,
            sequence_no=request.sequence_no,
            code=SynthesisErrorCode.MODEL_NOT_EXECUTABLE,
            detail=detail,
        )


def _peak_rss_bytes() -> int:
    """Return max RSS in bytes across Linux and macOS resource conventions."""
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum if sys.platform == "darwin" else maximum * 1024)
