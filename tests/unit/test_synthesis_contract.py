from dataclasses import FrozenInstanceError

import pytest

import itantra_tts
from itantra_tts import Precision, SynthesisRequest


def test_package_exposes_a_synthesis_request_contract() -> None:
    assert hasattr(itantra_tts, "SynthesisRequest")


def test_package_exposes_a_synthesis_result_contract() -> None:
    assert hasattr(itantra_tts, "SynthesisResult")


def test_package_exposes_int4_and_int8_precision_options() -> None:
    assert hasattr(itantra_tts, "Precision")


def test_request_preserves_ordering_identity_and_selected_precision() -> None:
    request = SynthesisRequest(
        stream_id="stream-a",
        sequence_no=7,
        text="नमस्ते",
        language="hi",
        precision=Precision.INT8,
    )

    assert request.stream_id == "stream-a"
    assert request.sequence_no == 7
    assert request.precision is Precision.INT8
    with pytest.raises(FrozenInstanceError):
        request.text = "mutated"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("stream_id", "", "stream_id"),
        ("sequence_no", -1, "sequence_no"),
        ("text", "   ", "text"),
        ("language", "fr", "language"),
    ],
)
def test_request_rejects_invalid_public_contract_values(
    field: str, value: object, message: str
) -> None:
    values: dict[str, object] = {
        "stream_id": "stream-a",
        "sequence_no": 0,
        "text": "hello",
        "language": "en",
        "precision": Precision.INT8,
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        SynthesisRequest(**values)  # type: ignore[arg-type]
