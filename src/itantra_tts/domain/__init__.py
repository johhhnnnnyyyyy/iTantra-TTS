"""Stable domain contracts for synthesis callers and adapters."""

from .contracts import Precision, SynthesisMetrics, SynthesisRequest, SynthesisResult
from .errors import SynthesisErrorCode, SynthesisFailure
from .model_catalog import ModelAvailability, ModelCatalog, ModelProfile

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
]
