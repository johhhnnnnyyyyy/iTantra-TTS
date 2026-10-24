"""Typed failures safe to expose to a caller without leaking message payloads."""

from dataclasses import dataclass
from enum import StrEnum


class SynthesisErrorCode(StrEnum):
    """Stable public error codes for synthesis requests."""

    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_NOT_EXECUTABLE = "model_not_executable"


@dataclass(frozen=True, slots=True)
class SynthesisFailure:
    """A failure preserving stream identity without copying raw request text."""

    stream_id: str
    sequence_no: int
    code: SynthesisErrorCode
    detail: str
