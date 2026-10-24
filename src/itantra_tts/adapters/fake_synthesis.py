"""Deterministic unavailable-model adapter for contract tests and UI wiring."""

from itantra_tts.domain import SynthesisErrorCode, SynthesisFailure, SynthesisRequest


class FakeSynthesisBackend:
    """Returns a typed unavailable result until an executable model adapter exists."""

    def synthesize(self, request: SynthesisRequest) -> SynthesisFailure:
        return SynthesisFailure(
            stream_id=request.stream_id,
            sequence_no=request.sequence_no,
            code=SynthesisErrorCode.MODEL_NOT_EXECUTABLE,
            detail="No executable model adapter is configured for this selection.",
        )
