import itantra_tts
from itantra_tts import (
    FakeSynthesisBackend,
    Precision,
    SynthesisErrorCode,
    SynthesisRequest,
)


def test_package_exposes_a_typed_synthesis_failure() -> None:
    assert hasattr(itantra_tts, "SynthesisFailure")


def test_package_exposes_a_fake_synthesis_backend_for_contract_testing() -> None:
    assert hasattr(itantra_tts, "FakeSynthesisBackend")


def test_unavailable_precision_failure_preserves_ids_without_payload_text() -> None:
    request = SynthesisRequest(
        stream_id="stream-42",
        sequence_no=9,
        text="private message text",
        language="en",
        precision=Precision.INT4,
    )

    failure = FakeSynthesisBackend().synthesize(request)

    assert failure.stream_id == "stream-42"
    assert failure.sequence_no == 9
    assert failure.code is SynthesisErrorCode.MODEL_NOT_EXECUTABLE
    assert request.text not in failure.detail
