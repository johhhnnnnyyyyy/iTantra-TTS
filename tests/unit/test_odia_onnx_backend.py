from pathlib import Path

import pytest

from itantra_tts.domain import Precision, SynthesisRequest, SynthesisResult


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "models" / "artifacts" / "mms-tts-ory-int8-research"


@pytest.mark.skipif(not ARTIFACT_ROOT.exists(), reason="local derived Odia INT8 research artifact absent")
def test_odia_onnx_dynamic_int8_backend_generates_valid_audio() -> None:
    from itantra_tts.adapters.onnx_odia_vits import OnnxOdiaVitsResearchBackend

    backend = OnnxOdiaVitsResearchBackend(ARTIFACT_ROOT, model_filename="model.int8.onnx")
    result = backend.synthesize(
        SynthesisRequest("test", 0, "ତୁମେ ଭଲ ଅଛ କି", "or", Precision.INT8)
    )

    assert isinstance(result, SynthesisResult)
    assert result.sample_rate_hz == 16_000
    assert len(result.audio_pcm_s16le) > 0
    assert result.model_id == "mms-tts-ory-onnx-dynamic-int8-research"
    assert result.metrics is not None
    assert result.metrics.model_disk_bytes < 50 * 1024 * 1024


@pytest.mark.skipif(
    not (ARTIFACT_ROOT / "model.static-qdq.int8.onnx").exists(),
    reason="local static-QDQ Odia research candidate absent",
)
def test_odia_research_backend_defaults_to_listening_approved_static_candidate() -> None:
    from itantra_tts.adapters.onnx_odia_vits import OnnxOdiaVitsResearchBackend

    backend = OnnxOdiaVitsResearchBackend(ARTIFACT_ROOT)
    result = backend.synthesize(
        SynthesisRequest("test", 0, "ତୁମେ ଭଲ ଅଛ କି", "or", Precision.INT8)
    )

    assert isinstance(result, SynthesisResult)
    assert result.model_id == "mms-tts-ory-onnx-static-qdq-int8-research"


@pytest.mark.skipif(
    not (ARTIFACT_ROOT / "model.static-qdq.int8.onnx").exists(),
    reason="local static-QDQ Odia research candidate absent",
)
def test_odia_static_qdq_candidate_is_selectable_and_generates_audio() -> None:
    from itantra_tts.adapters.onnx_odia_vits import OnnxOdiaVitsResearchBackend

    backend = OnnxOdiaVitsResearchBackend(
        ARTIFACT_ROOT, model_filename="model.static-qdq.int8.onnx", intra_op_threads=4
    )
    result = backend.synthesize(
        SynthesisRequest("test", 0, "ତୁମେ ଭଲ ଅଛ କି", "or", Precision.INT8)
    )

    assert isinstance(result, SynthesisResult)
    assert result.model_id == "mms-tts-ory-onnx-static-qdq-int8-research"
    assert result.sample_rate_hz == 16_000
    assert len(result.audio_pcm_s16le) > 0
