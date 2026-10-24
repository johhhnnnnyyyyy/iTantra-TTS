from pathlib import Path

import pytest

from itantra_tts.domain import Precision, SynthesisRequest, SynthesisResult


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "models/artifacts/echobharat-int8"


@pytest.mark.skipif(not (ARTIFACT_ROOT / "hi").exists(), reason="Hindi INT8 artifacts absent")
def test_sequential_sessions_release_both_models_after_valid_audio() -> None:
    from itantra_tts.adapters.onnx_fastpitch_hifigan import OnnxFastPitchHifiGanBackend

    backend = OnnxFastPitchHifiGanBackend(
        ARTIFACT_ROOT, intra_op_threads=4, session_policy="sequential"
    )
    result = backend.synthesize(
        SynthesisRequest("test", 0, "नमस्ते।", "hi", Precision.INT8)
    )

    assert isinstance(result, SynthesisResult)
    assert len(result.audio_pcm_s16le) > 0
    assert result.metrics is not None
    assert result.metrics.resident_model_count == 0


def test_session_policy_rejects_unrecognized_mode() -> None:
    from itantra_tts.adapters.onnx_fastpitch_hifigan import OnnxFastPitchHifiGanBackend

    with pytest.raises(ValueError, match="session_policy"):
        OnnxFastPitchHifiGanBackend(ARTIFACT_ROOT, session_policy="unknown")
