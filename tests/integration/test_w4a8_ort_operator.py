"""Run the experimental W4A8 operator inside a real ONNX Runtime session."""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper


@pytest.fixture(scope="module", params=["automatic", "portable"])
def custom_library(tmp_path_factory, request):
    compiler = shutil.which("c++")
    header_dir = os.environ.get("ORT_HEADER_DIR")
    if not compiler or not header_dir:
        pytest.skip("C++ compiler and ORT_HEADER_DIR are required")
    source = Path(__file__).parents[2] / "tools" / "ort_w4a8_matmul.cpp"
    assert source.exists(), "W4A8 ONNX Runtime custom operator is missing"
    output = tmp_path_factory.mktemp("ort_w4") / "w4.so"
    command = [compiler, "-std=c++17", "-O2", "-shared", "-fPIC", "-Wall",
               "-Wextra", "-Werror", "-I", header_dir, str(source), "-o", str(output)]
    if request.param == "portable":
        command.append("-DITANTRA_DISABLE_AVX2=1")
    subprocess.run(command, check=True, capture_output=True)
    return output


def _session(library):
    inputs = [helper.make_tensor_value_info("a", TensorProto.UINT8, None),
              helper.make_tensor_value_info("packed", TensorProto.UINT8, [None]),
              helper.make_tensor_value_info("zero", TensorProto.UINT8, []),
              helper.make_tensor_value_info("shape", TensorProto.INT64, [2])]
    output = helper.make_tensor_value_info("y", TensorProto.INT32, None)
    node = helper.make_node("MatMulW4A8", [x.name for x in inputs], ["y"],
                            domain="org.itantra.experimental")
    model = helper.make_model(helper.make_graph([node], "w4", inputs, [output]),
                              opset_imports=[helper.make_opsetid("", 17),
                                             helper.make_opsetid("org.itantra.experimental", 1)],
                              ir_version=9)
    options = ort.SessionOptions()
    options.register_custom_ops_library(str(library))
    return ort.InferenceSession(model.SerializeToString(), options,
                                providers=["CPUExecutionProvider"])


def test_cmake_build_produces_loadable_integer_operator(tmp_path):
    header_dir = os.environ.get("ORT_HEADER_DIR")
    if not header_dir or not shutil.which("cmake") or not shutil.which("c++"):
        pytest.skip("CMake, C++ compiler and ORT_HEADER_DIR are required")
    source_dir = Path(__file__).parents[2] / "tools" / "w4a8_native"
    build_dir = tmp_path / "build"
    subprocess.run(["cmake", "-S", str(source_dir), "-B", str(build_dir),
                    f"-DORT_HEADER_DIR={header_dir}", "-DCMAKE_BUILD_TYPE=Release"],
                   check=True, capture_output=True)
    subprocess.run(["cmake", "--build", str(build_dir)], check=True, capture_output=True)
    library = build_dir / "libitantra_w4a8.so"
    result = _session(library).run(None, {
        "a": np.array([[128, 130, 127]], np.uint8),
        "packed": np.array([0xF2, 0x87, 0x35], np.uint8),
        "zero": np.array(128, np.uint8), "shape": np.array([3, 2], np.int64),
    })[0]
    np.testing.assert_array_equal(result, np.array([[9, -19]], np.int32))


def test_custom_operator_computes_packed_integer_matrix_product(custom_library):
    session = _session(custom_library)
    result = session.run(None, {"a": np.array([[128, 130, 127], [255, 0, 128]], np.uint8),
                                 "packed": np.array([0xF2, 0x87, 0x35], np.uint8),
                                 "zero": np.array(128, np.uint8),
                                 "shape": np.array([3, 2], np.int64)})[0]
    np.testing.assert_array_equal(result, np.array([[9, -19], [-642, 897]], np.int32))


def test_custom_operator_rejects_short_packed_tensor(custom_library):
    session = _session(custom_library)
    with pytest.raises(Exception, match="packed"):
        session.run(None, {"a": np.array([[1, 2, 3]], np.uint8),
                           "packed": np.array([0x21], np.uint8),
                           "zero": np.array(0, np.uint8),
                           "shape": np.array([3, 2], np.int64)})


def test_custom_operator_handles_batched_inputs_and_odd_weight_count(custom_library):
    session = _session(custom_library)
    result = session.run(None, {"a": np.array([[[2, 3, 4]], [[4, 3, 2]]], np.uint8),
                                 "packed": np.array([0xF1, 0x08], np.uint8),
                                 "zero": np.array(0, np.uint8),
                                 "shape": np.array([3, 1], np.int64)})[0]
    np.testing.assert_array_equal(result, np.array([[[-33]], [[-15]]], np.int32))


def test_custom_operator_rejects_incompatible_inner_dimension(custom_library):
    session = _session(custom_library)
    with pytest.raises(Exception, match="incompatible"):
        session.run(None, {"a": np.array([[1, 2]], np.uint8),
                           "packed": np.array([0x21, 0x43, 0x65], np.uint8),
                           "zero": np.array(0, np.uint8),
                           "shape": np.array([3, 2], np.int64)})


def test_custom_operator_handles_empty_batch(custom_library):
    result = _session(custom_library).run(None, {
        "a": np.empty((0, 3), dtype=np.uint8),
        "packed": np.array([0x21, 0x43, 0x65], np.uint8),
        "zero": np.array(0, np.uint8),
        "shape": np.array([3, 2], np.int64),
    })[0]
    assert result.shape == (0, 2)


def test_custom_operator_many_rows_and_odd_columns_match_integer_reference(custom_library):
    rng = np.random.default_rng(18)
    x = rng.integers(0, 256, size=(7, 13), dtype=np.uint8)
    weights = rng.integers(-8, 8, size=(13, 9), dtype=np.int8)
    nibbles = weights.reshape(-1).astype(np.int16) & 15
    padded = np.pad(nibbles, (0, len(nibbles) % 2))
    packed = (padded[0::2] | (padded[1::2] << 4)).astype(np.uint8)
    expected = (x.astype(np.int32) - 117) @ weights.astype(np.int32)
    result = _session(custom_library).run(None, {
        "a": x, "packed": packed, "zero": np.array(117, np.uint8),
        "shape": np.array([13, 9], np.int64),
    })[0]
    np.testing.assert_array_equal(result, expected)


def test_custom_convolution_executes_unsigned_activations_and_packed_weights(custom_library):
    inputs = [helper.make_tensor_value_info("a", TensorProto.UINT8, [1, 1, 5]),
              helper.make_tensor_value_info("w", TensorProto.UINT8, [2]),
              helper.make_tensor_value_info("z", TensorProto.UINT8, []),
              helper.make_tensor_value_info("shape", TensorProto.INT64, [3]),
              helper.make_tensor_value_info("geometry", TensorProto.INT64, [5])]
    output = helper.make_tensor_value_info("y", TensorProto.INT32, [1, 1, 3])
    node = helper.make_node("ConvW4A8", [i.name for i in inputs], ["y"],
                            domain="org.itantra.experimental")
    model = helper.make_model(helper.make_graph([node], "conv", inputs, [output]),
                              opset_imports=[helper.make_opsetid("", 17),
                                             helper.make_opsetid("org.itantra.experimental", 1)], ir_version=9)
    options = ort.SessionOptions()
    options.register_custom_ops_library(str(custom_library))
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    result = session.run(None, {"a": np.array([[[129, 120, 135, 127, 130]]], np.uint8),
                                "w": np.array([0x3E, 0x0F], np.uint8), "z": np.array(128, np.uint8),
                                "shape": np.array([1, 1, 3], np.int64),
                                "geometry": np.array([2, 1, 1, 1, 1], np.int64)})[0]
    np.testing.assert_array_equal(result, np.array([[[11, 38, 8]]], np.int32))


def test_custom_convolution_handles_groups_dilation_and_asymmetric_padding(custom_library):
    inputs = [helper.make_tensor_value_info("a", TensorProto.UINT8, [1, 2, 5]),
              helper.make_tensor_value_info("w", TensorProto.UINT8, [2]),
              helper.make_tensor_value_info("z", TensorProto.UINT8, []),
              helper.make_tensor_value_info("shape", TensorProto.INT64, [3]),
              helper.make_tensor_value_info("geometry", TensorProto.INT64, [5])]
    output = helper.make_tensor_value_info("y", TensorProto.INT32, [1, 2, 4])
    node = helper.make_node("ConvW4A8", [i.name for i in inputs], ["y"],
                            domain="org.itantra.experimental")
    model = helper.make_model(helper.make_graph([node], "grouped", inputs, [output]),
                              opset_imports=[helper.make_opsetid("", 17),
                                             helper.make_opsetid("org.itantra.experimental", 1)], ir_version=9)
    options = ort.SessionOptions()
    options.register_custom_ops_library(str(custom_library))
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    result = session.run(None, {"a": np.array([[[1, 2, 3, 4, 5], [1, 1, 1, 1, 1]]], np.uint8),
                                "w": np.array([0xF1, 0x32], np.uint8), "z": np.array(0, np.uint8),
                                "shape": np.array([2, 1, 2], np.int64),
                                "geometry": np.array([1, 1, 0, 2, 2], np.int64)})[0]
    np.testing.assert_array_equal(result, np.array([[[-2, -2, -2, -2], [3, 5, 5, 5]]], np.int32))


@pytest.mark.parametrize("threads", [1, 4])
def test_convolution_batches_and_channel_tasks_match_independent_reference(custom_library, threads):
    rng = np.random.default_rng(17)
    x = rng.integers(0, 256, size=(2, 4, 19), dtype=np.uint8)
    weights = rng.integers(-8, 8, size=(6, 2, 3), dtype=np.int8)
    flat = weights.reshape(-1).astype(np.int16) & 15
    packed = (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8)
    expected = np.zeros((2, 6, 10), np.int32)
    for batch in range(2):
        for oc in range(6):
            for pos in range(10):
                for ic in range(2):
                    for k in range(3):
                        ix = pos * 2 - 1 + k
                        if 0 <= ix < 19:
                            expected[batch, oc, pos] += (int(x[batch, (oc // 3) * 2 + ic, ix]) - 120) * int(weights[oc, ic, k])
    inputs = [helper.make_tensor_value_info("a", TensorProto.UINT8, None),
              helper.make_tensor_value_info("w", TensorProto.UINT8, None),
              helper.make_tensor_value_info("z", TensorProto.UINT8, []),
              helper.make_tensor_value_info("shape", TensorProto.INT64, [3]),
              helper.make_tensor_value_info("geometry", TensorProto.INT64, [5])]
    output = helper.make_tensor_value_info("y", TensorProto.INT32, None)
    node = helper.make_node("ConvW4A8", [i.name for i in inputs], ["y"], domain="org.itantra.experimental")
    model = helper.make_model(helper.make_graph([node], "multi", inputs, [output]),
                              opset_imports=[helper.make_opsetid("", 17),
                                             helper.make_opsetid("org.itantra.experimental", 1)], ir_version=9)
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.register_custom_ops_library(str(custom_library))
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    actual = session.run(None, {"a": x, "w": packed, "z": np.array(120, np.uint8),
                                "shape": np.array([6, 2, 3], np.int64),
                                "geometry": np.array([2, 1, 1, 1, 2], np.int64)})[0]
    np.testing.assert_array_equal(actual, expected)


def test_convolution_stride_one_interior_and_tail_match_scalar_reference(custom_library):
    rng = np.random.default_rng(90)
    x = rng.integers(0, 256, size=(1, 3, 43), dtype=np.uint8)
    weights = rng.integers(-8, 8, size=(2, 3, 3), dtype=np.int8)
    flat = weights.reshape(-1).astype(np.int16) & 15
    packed = (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8)
    expected = np.zeros((1, 2, 41), dtype=np.int32)
    for oc in range(2):
        for pos in range(41):
            for ic in range(3):
                for k in range(3):
                    expected[0, oc, pos] += (int(x[0, ic, pos + k]) - 111) * int(weights[oc, ic, k])
    inputs = [helper.make_tensor_value_info("a", TensorProto.UINT8, None),
              helper.make_tensor_value_info("w", TensorProto.UINT8, None),
              helper.make_tensor_value_info("z", TensorProto.UINT8, []),
              helper.make_tensor_value_info("shape", TensorProto.INT64, [3]),
              helper.make_tensor_value_info("geometry", TensorProto.INT64, [5])]
    output = helper.make_tensor_value_info("y", TensorProto.INT32, None)
    node = helper.make_node("ConvW4A8", [i.name for i in inputs], ["y"],
                            domain="org.itantra.experimental")
    model = helper.make_model(helper.make_graph([node], "interior", inputs, [output]),
                              opset_imports=[helper.make_opsetid("", 17),
                                             helper.make_opsetid("org.itantra.experimental", 1)],
                              ir_version=9)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.register_custom_ops_library(str(custom_library))
    session = ort.InferenceSession(model.SerializeToString(), options,
                                   providers=["CPUExecutionProvider"])
    actual = session.run(None, {"a": x, "w": packed, "z": np.array(111, np.uint8),
                                "shape": np.array([2, 3, 3], np.int64),
                                "geometry": np.array([1, 0, 0, 1, 1], np.int64)})[0]
    np.testing.assert_array_equal(actual, expected)
