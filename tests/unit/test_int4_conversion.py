"""Ensure standard INT4 conversion never silently reports an unchanged graph."""

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tools.quantize_int4_onnx import quantize


def test_constant_matmul_becomes_standard_int4_qdq(tmp_path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 128])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 32])
    weight = numpy_helper.from_array(np.ones((128, 32), dtype=np.float32), "weight")
    graph = helper.make_graph([helper.make_node("MatMul", ["x", "weight"], ["y"])],
                              "test", [x], [y], [weight])
    source, output = tmp_path / "source.onnx", tmp_path / "out.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)]), source)
    assert quantize(source, output) >= 1
    model = onnx.load(output)
    assert any(t.data_type in (TensorProto.INT4, TensorProto.UINT4)
               for t in model.graph.initializer)
    assert any(n.op_type == "DequantizeLinear" and not n.domain for n in model.graph.node)


def test_no_eligible_weights_does_not_create_a_fake_int4_model(tmp_path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])],
                              "test", [x], [y])
    source, output = tmp_path / "source.onnx", tmp_path / "out.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)]), source)
    with pytest.raises(ValueError, match="no eligible"):
        quantize(source, output)
    assert not output.exists()
