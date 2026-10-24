from pathlib import Path

import onnx
import onnxruntime as ort
import pytest

from tools.experimental_odia_qdq_repack import convert


ROOT = Path(__file__).resolve().parents[2]
ODIA_INT8 = ROOT / "models/artifacts/mms-tts-ory-int8-research/model.static-qdq.int8.onnx"
ODIA_FP32 = ROOT / "models/artifacts/mms-tts-ory-int8-research/model.fp32.onnx"


@pytest.mark.skipif(not (ODIA_INT8.is_file() and ODIA_FP32.is_file()),
                    reason="research models not provisioned")
def test_odia_static_qdq_weights_can_be_repacked(tmp_path: Path) -> None:
    output = tmp_path / "odia-int4-repacked.onnx"
    assert convert(ODIA_INT8, ODIA_FP32, output) >= 180
    model = onnx.load(output)
    assert sum(item.data_type == onnx.TensorProto.INT4 for item in model.graph.initializer) >= 180
    assert output.stat().st_size < ODIA_INT8.stat().st_size
    onnx.checker.check_model(model)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    ort.InferenceSession(str(output), opts, providers=["CPUExecutionProvider"])
    with pytest.raises(ValueError, match="new file"):
        convert(ODIA_INT8, ODIA_FP32, output)
