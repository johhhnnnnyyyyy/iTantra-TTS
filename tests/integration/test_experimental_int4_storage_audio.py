"""Research-only full-chain smoke; does not assert speech quality or Android speed."""

from pathlib import Path

import pytest

from itantra_tts.adapters.onnx_fastpitch_hifigan import OnnxFastPitchHifiGanBackend
from itantra_tts.adapters.onnx_odia_vits import OnnxOdiaVitsResearchBackend
from itantra_tts.domain import Precision, SynthesisRequest, SynthesisResult


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "models/artifacts/echobharat-int8"
PILOT = ROOT / "models/artifacts/indic-tts-fp32-source"
SAMPLES = {
    "bn": "নমস্কার।", "en": "Hello.", "gu": "નમસ્તે.", "hi": "नमस्ते।",
    "kn": "ನಮಸ್ಕಾರ.", "ml": "നമസ്കാരം.", "mr": "नमस्कार.",
    "ta": "வணக்கம்.", "te": "నమస్కారం.",
}


@pytest.mark.parametrize("language", sorted(SAMPLES))
def test_experimental_int4_storage_pack_synthesizes_audio(
    language: str, tmp_path: Path,
) -> None:
    fastpitch = PILOT / f"fastpitch-{language}.from-int8.int4-storage.experimental.onnx"
    fp32_vocoder = PILOT / f"hifigan-{language}.from-fp32.int4-qdq.compact.experimental.onnx"
    vocoder = fp32_vocoder if fp32_vocoder.is_file() else (
        PILOT / f"hifigan-{language}.from-int8.int4-qdq.compact.experimental.onnx"
    )
    if not (fastpitch.is_file() and vocoder.is_file()):
        pytest.skip("experimental pack not provisioned")
    folder = tmp_path / language
    folder.mkdir()
    (folder / f"fastpitch-{language}.int8.onnx").symlink_to(fastpitch)
    (folder / f"hifigan-{language}.int8.onnx").symlink_to(vocoder)
    (folder / f"fastpitch-{language}.tokens.json").symlink_to(
        BASE / language / f"fastpitch-{language}.tokens.json"
    )
    backend = OnnxFastPitchHifiGanBackend(tmp_path, intra_op_threads=4)
    result = backend.synthesize(SynthesisRequest(
        "int4-storage-research", 0, SAMPLES[language], language, Precision.INT8,
    ))
    assert isinstance(result, SynthesisResult), result
    assert len(result.audio_pcm_s16le) >= 1000
    assert result.metrics is not None
    assert result.metrics.real_time_factor > 0


def test_experimental_odia_int4_storage_pack_synthesizes_audio(tmp_path: Path) -> None:
    source = (
        ROOT / "models/artifacts/mms-tts-ory-int8-research/"
        "model.static-qdq.int4-storage.experimental.onnx"
    )
    original = ROOT / "models/artifacts/mms-tts-ory-int8-research"
    if not source.is_file():
        pytest.skip("experimental Odia model not provisioned")
    source_dir = tmp_path / "mms-tts-ory-int8-research"
    source_dir.mkdir()
    (source_dir / "model.static-qdq.int8.onnx").symlink_to(source)
    for name in ("vocab.json", "config.json"):
        (source_dir / name).symlink_to(original / name)
    backend = OnnxOdiaVitsResearchBackend(source_dir, intra_op_threads=4)
    result = backend.synthesize(SynthesisRequest(
        "int4-storage-research", 0, "ନମସ୍କାର।", "or", Precision.INT8,
    ))
    assert isinstance(result, SynthesisResult), result
    assert len(result.audio_pcm_s16le) >= 1000
    assert result.metrics is not None
