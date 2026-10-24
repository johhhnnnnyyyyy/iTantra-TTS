from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

import onnx

from tools.experimental_vocoder_int4_qdq import convert, quantize_per_output_channel


ROOT = Path(__file__).resolve().parents[2]
HI_VOCODER = ROOT / "models/artifacts/echobharat-int8/hi/hifigan-hi.int8.onnx"


def test_per_channel_int4_is_real_onnx_int4() -> None:
    weights = np.array([[[0.0, 0.5, 1.0]], [[0.0, -1.0, 0.0]]], dtype=np.float32)
    quantized, scales = quantize_per_output_channel(weights)
    tensor = onnx.numpy_helper.from_array(quantized)
    assert tensor.data_type == onnx.TensorProto.INT4
    assert scales.shape == (2,)
    assert np.max(onnx.numpy_helper.to_array(tensor)) <= 7
    assert np.min(onnx.numpy_helper.to_array(tensor)) >= -7


@pytest.mark.skipif(not HI_VOCODER.is_file(), reason="research model not provisioned")
def test_hindi_vocoder_conversion_produces_compact_executable_graph(tmp_path: Path) -> None:
    destination = tmp_path / "hi-int4-qdq.onnx"
    assert convert(HI_VOCODER, destination) == 74
    model = onnx.load(destination)
    counts = {op: sum(node.op_type == op for node in model.graph.node) for op in (
        "ConvInteger", "Conv", "DequantizeLinear",
    )}
    assert counts == {"ConvInteger": 0, "Conv": 74, "DequantizeLinear": 74}
    assert sum(item.data_type == onnx.TensorProto.INT4 for item in model.graph.initializer) == 74
    assert destination.stat().st_size < HI_VOCODER.stat().st_size
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(destination), sess_options=options, providers=["CPUExecutionProvider"])
    audio = session.run(None, {"mel": np.zeros((1, 80, 17), dtype=np.float32)})[0]
    assert audio.size > 0
    assert np.isfinite(audio).all()
    with pytest.raises(ValueError, match="new file"):
        convert(HI_VOCODER, destination)


def test_fp32_mode_requires_source_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a checkpoint"):
        convert(tmp_path / "missing.onnx", tmp_path / "candidate.onnx", precision="fp32")
