"""Concrete adapters for model runtimes and development fakes."""

from .fake_synthesis import FakeSynthesisBackend
from .onnx_fastpitch_hifigan import OnnxFastPitchHifiGanBackend

__all__ = [
    "FakeSynthesisBackend",
    "OnnxFastPitchHifiGanBackend",
]
