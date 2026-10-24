"""Offline-first multilingual TTS core for ordered message playback."""

from .domain import (
    ModelAvailability,
    ModelCatalog,
    ModelProfile,
    Precision,
    SynthesisErrorCode,
    SynthesisFailure,
    SynthesisMetrics,
    SynthesisRequest,
    SynthesisResult,
)
from .adapters import FakeSynthesisBackend
from .normalization import NormalizationResult, NormalizationWarning, SpeechNormalizer

__version__ = "0.1.0a0"

__all__ = [
    "ModelAvailability",
    "ModelCatalog",
    "ModelProfile",
    "Precision",
    "SynthesisErrorCode",
    "SynthesisFailure",
    "SynthesisMetrics",
    "SynthesisRequest",
    "SynthesisResult",
    "FakeSynthesisBackend",
    "NormalizationResult",
    "NormalizationWarning",
    "SpeechNormalizer",
    "__version__",
]
