from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from tools.experimental_odia_int4_storage import convert


ROOT = Path(__file__).resolve().parents[2]
ODIA_FP32 = ROOT / "models/artifacts/mms-tts-ory-int8-research/model.fp32.onnx"


def test_conv_weights_use_real_int4_storage(tmp_path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 16])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 16])
    weight = numpy_helper.from_array(np.ones((2, 1, 17), dtype=np.float32), "weight")
    graph = helper.make_graph([helper.make_node("Conv", ["x", "weight"], ["y"],
                                                pads=[8, 8])], "test", [x], [y], [weight])
    source, output = tmp_path / "source.onnx", tmp_path / "int4.onnx"
    original = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    original.ir_version = 13
    onnx.save(original, source)
    assert convert(source, output) == 1
    model = onnx.load(output)
    assert any(item.data_type == TensorProto.INT4 for item in model.graph.initializer)
    assert output.stat().st_size > 0
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    result = session.run(None, {"x": np.ones((1, 1, 16), dtype=np.float32)})[0]
    assert np.isfinite(result).all()


@pytest.mark.skipif(not ODIA_FP32.is_file(), reason="research model not provisioned")
def test_odia_conv_model_is_compact_and_loadable(tmp_path: Path) -> None:
    output = tmp_path / "odia-int4-storage.onnx"
    assert convert(ODIA_FP32, output) >= 150
    assert output.stat().st_size < ODIA_FP32.stat().st_size / 2
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    ort.InferenceSession(str(output), opts, providers=["CPUExecutionProvider"])
