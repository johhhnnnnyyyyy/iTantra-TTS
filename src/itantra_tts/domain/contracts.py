"""Public synthesis contracts independent of models, Android, and UI."""

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar


class Precision(StrEnum):
    """Requested model-weight precision exposed to callers."""

    INT8 = "int8"
    INT4 = "int4"


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    """A caller's request to synthesize one ordered text chunk."""

    stream_id: str
    sequence_no: int
    text: str
    language: str
    precision: Precision

    SUPPORTED_LANGUAGES: ClassVar[frozenset[str]] = frozenset(
        {"hi", "gu", "mr", "kn", "ml", "ta", "te", "or", "bn", "en"}
    )

    def __post_init__(self) -> None:
        if not isinstance(self.stream_id, str) or not self.stream_id.strip():
            raise ValueError("stream_id must be a non-empty string")
        if not isinstance(self.sequence_no, int) or isinstance(self.sequence_no, bool):
            raise ValueError("sequence_no must be an integer")
        if self.sequence_no < 0:
            raise ValueError("sequence_no must be non-negative")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must contain at least one non-whitespace character")
        if self.language not in self.SUPPORTED_LANGUAGES:
            raise ValueError("language is not in the SIH26173 language set")
        if not isinstance(self.precision, Precision):
            raise ValueError("precision must be a Precision value")


@dataclass(frozen=True, slots=True)
class SynthesisMetrics:
    """Measurements captured from one actual local synthesis request."""

    wall_latency_ms: float
    stage_one_label: str
    fastpitch_latency_ms: float
    stage_two_label: str
    hifigan_latency_ms: float
    process_cpu_time_ms: float
    process_cpu_utilization_percent: float
    rss_before_bytes: int
    rss_after_bytes: int
    selected_model_rss_delta_bytes: int
    peak_rss_bytes: int
    audio_duration_ms: float
    real_time_factor: float
    model_disk_bytes: int
    provider: str
    runtime_version: str
    cold_start: bool
    resident_model_count: int
    memory_scope: str


@dataclass(frozen=True, slots=True)
class ResearchWorkerMetrics:
    """Measured subset from one experimental isolated worker."""

    cpu_time_ms: float
    peak_rss_bytes: int
    wall_ms: float
    runtime_version: str
    memory_scope: str


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """Validated audio returned for one ordered synthesis request."""

    stream_id: str
    sequence_no: int
    language: str
    precision: Precision
    model_id: str
    sample_rate_hz: int
    audio_pcm_s16le: bytes
    metrics: SynthesisMetrics | None = None
    research_metrics: ResearchWorkerMetrics | None = None
