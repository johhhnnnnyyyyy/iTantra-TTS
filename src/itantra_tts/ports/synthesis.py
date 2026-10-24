"""The narrow seam implemented by ONNX, VITS, and future C++ backends."""

from typing import Protocol

from itantra_tts.domain import SynthesisFailure, SynthesisRequest, SynthesisResult


class SynthesisBackend(Protocol):
    """Attempt to synthesize one already-normalized sentence chunk."""

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult | SynthesisFailure: ...
