"""Exercise graph rewriting and scaling through real ONNX Runtime inference."""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from tools.w4a8_graph_rewrite import rewrite_all, rewrite_one

ROOT = Path(__file__).parents[2]
SAMPLES = {"bn": "নমস্কার.", "en": "Hello.", "gu": "નમસ્તે.", "hi": "नमस्ते.",
           "kn": "ನಮಸ್ಕಾರ.", "ml": "നമസ്കാരം.", "mr": "नमस्कार.",
           "ta": "வணக்கம்.", "te": "నమస్కారం.", "or": "ନମସ୍କାର।"}

def _model():
    nodes = [helper.make_node("MatMulInteger", ["a", "w", "az", "wz"], ["integer"], name="target"),
             helper.make_node("Cast", ["integer"], ["converted"], to=TensorProto.FLOAT),
             helper.make_node("Mul", ["converted", "scale"], ["y"])]
    constants = [numpy_helper.from_array(np.array([[14, -8], [6, 0], [-6, 8]], np.int8), "w"),
                 numpy_helper.from_array(np.array(128, np.uint8), "az"),
                 numpy_helper.from_array(np.array(0, np.int8), "wz"),
                 numpy_helper.from_array(np.array(0.1, np.float32), "scale")]
    graph = helper.make_graph(nodes, "test", [helper.make_tensor_value_info("a", TensorProto.UINT8, [1, 3])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])], constants)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)


def test_rewrite_rejects_nonzero_weight_zero_point():
    model = _model()
    next(x for x in model.graph.initializer if x.name == "wz").CopyFrom(
        numpy_helper.from_array(np.array(1, np.int8), "wz"))
    with pytest.raises(ValueError, match="zero"):
        rewrite_one(model, "target")


def test_rewrite_all_converts_all_supported_nodes():
    model = rewrite_all(_model())
    assert [n.op_type for n in model.graph.node].count("MatMulInteger") == 0
    assert [n.op_type for n in model.graph.node].count("MatMulW4A8") == 1


def test_per_channel_rewrite_preserves_small_weight_columns():
    model = _model()
    weights = np.array([[127, 7], [0, 1], [0, -1]], np.int8)
    next(x for x in model.graph.initializer if x.name == "w").CopyFrom(
        numpy_helper.from_array(weights, "w"))
    rewritten = rewrite_one(model, "target", per_channel=True)
    initializers = {x.name: numpy_helper.to_array(x) for x in rewritten.graph.initializer}
    factor = initializers["target.itantra_w4.factor"]
    assert factor.shape == (2,)
    packed = initializers["target.itantra_w4.packed"]
    nibbles = np.column_stack((packed & 15, packed >> 4)).ravel()[:weights.size].astype(np.int8)
    nibbles[nibbles >= 8] -= 16
    restored = nibbles.reshape(weights.shape) * factor
    assert np.linalg.norm(restored - weights) < np.linalg.norm(
        np.clip(np.rint(weights.astype(np.float32) / (127 / 7)), -7, 7) * (127 / 7) - weights)


def test_per_channel_convolution_scales_broadcast_by_output_channel():
    model = _model()
    model.graph.input[0].CopyFrom(helper.make_tensor_value_info("a", TensorProto.UINT8, [1, 1, 3]))
    model.graph.output[0].CopyFrom(helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 1]))
    next(x for x in model.graph.initializer if x.name == "w").CopyFrom(
        numpy_helper.from_array(np.array([[[127, 0, 0]], [[7, 1, -1]]], np.int8), "w"))
    model.graph.node[0].CopyFrom(helper.make_node("ConvInteger", ["a", "w", "az", "wz"],
                                                  ["integer"], name="target"))
    rewritten = rewrite_one(model, "target", per_channel=True)
    factor = next(numpy_helper.to_array(x) for x in rewritten.graph.initializer
                  if x.name == "target.itantra_w4.factor")
    assert factor.shape == (1, 2, 1)
    np.testing.assert_allclose(factor.ravel(), [127 / 7, 1])
    library = os.environ.get("ITANTRA_W4_LIBRARY")
    if library:
        options = ort.SessionOptions()
        options.register_custom_ops_library(library)
        session = ort.InferenceSession(rewritten.SerializeToString(), options, providers=["CPUExecutionProvider"])
        result = session.run(None, {"a": np.array([[[129, 130, 127]]], np.uint8)})[0]
        np.testing.assert_allclose(result, [[[12.7], [1.0]]], atol=1e-5)


def test_rewrite_rejects_weight_exposed_as_overridable_model_input():
    model = _model()
    model.graph.input.append(helper.make_tensor_value_info("w", TensorProto.INT8, [3, 2]))
    with pytest.raises(ValueError, match="input"):
        rewrite_one(model, "target")


def test_rewrite_rejects_overridable_weight_zero_point():
    model = _model()
    model.graph.input.append(helper.make_tensor_value_info("wz", TensorProto.INT8, []))
    with pytest.raises(ValueError, match="input"):
        rewrite_one(model, "target")


def test_rewritten_node_executes_packed_weights_with_correct_scale(tmp_path):
    header = os.environ.get("ORT_HEADER_DIR")
    compiler = shutil.which("c++")
    if not header or not compiler:
        pytest.skip("C++ compiler and ORT_HEADER_DIR required")
    library = tmp_path / "custom.so"
    source = Path(__file__).parents[2] / "tools" / "ort_w4a8_matmul.cpp"
    subprocess.run([compiler, "-std=c++17", "-O2", "-shared", "-fPIC", "-I", header,
                    str(source), "-o", str(library)], check=True)
    model = rewrite_one(_model(), "target")
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.register_custom_ops_library(str(library))
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    result = session.run(None, {"a": np.array([[128, 130, 127]], np.uint8)})[0]
    np.testing.assert_allclose(result, np.array([[1.8, -0.8]], np.float32), atol=1e-6)


def test_rewritten_convolution_executes_with_correct_scale(tmp_path):
    header, compiler = os.environ.get("ORT_HEADER_DIR"), shutil.which("c++")
    if not header or not compiler:
        pytest.skip("C++ compiler and ORT_HEADER_DIR required")
    library = tmp_path / "custom.so"
    subprocess.run([compiler, "-std=c++17", "-O2", "-shared", "-fPIC", "-I", header,
                    str(ROOT / "tools/ort_w4a8_matmul.cpp"), "-o", str(library)], check=True)
    model = _model()
    model.graph.input[0].CopyFrom(helper.make_tensor_value_info("a", TensorProto.UINT8, [1, 1, 5]))
    model.graph.output[0].CopyFrom(helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3]))
    next(x for x in model.graph.initializer if x.name == "w").CopyFrom(
        numpy_helper.from_array(np.array([[[14, -8, 6]]], np.int8), "w"))
    model.graph.node[0].CopyFrom(helper.make_node("ConvInteger", ["a", "w", "az", "wz"],
                                                  ["integer"], name="target", pads=[1, 1], strides=[2]))
    model = rewrite_one(model, "target")
    options = ort.SessionOptions()
    options.register_custom_ops_library(str(library))
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    result = session.run(None, {"a": np.array([[[129, 120, 135, 127, 130]]], np.uint8)})[0]
    np.testing.assert_allclose(result, np.array([[[-5.6, -17.4, -3.0]]], np.float32), atol=1e-6)


@pytest.mark.parametrize("language", sorted(SAMPLES))
@pytest.mark.parametrize("mode", ["pilot", "full"])
def test_rewritten_real_model_synthesizes(language, mode, tmp_path):
    """The full case converts integer nodes, while retaining baseline float islands."""
    import json

    if os.environ.get("ITANTRA_W4_REAL_SMOKE") != "1":
        pytest.skip("set ITANTRA_W4_REAL_SMOKE=1 to exercise real model artifacts")
    header, compiler = os.environ.get("ORT_HEADER_DIR"), shutil.which("c++")
    if not header or not compiler:
        pytest.skip("C++ compiler and ORT_HEADER_DIR required")
    library = tmp_path / "custom.so"
    subprocess.run([compiler, "-std=c++17", "-O2", "-shared", "-fPIC", "-I", header,
                    str(ROOT / "tools/ort_w4a8_matmul.cpp"), "-o", str(library)], check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.register_custom_ops_library(str(library))
    if language == "or":
        from itantra_tts.adapters.odia_tokenizer import OdiaMmsTokenizer

        folder = ROOT / "models/artifacts/mms-tts-ory-int8-research"
        source = folder / "model.int8.onnx"
        node = "/text_encoder/encoder/layers.0/attention/q_proj/MatMul_quant"
        inputs = {"input_ids": OdiaMmsTokenizer(folder / "vocab.json").encode(SAMPLES[language]).astype(np.int64)}
    else:
        folder = ROOT / "models/artifacts/echobharat-int8" / language
        source = folder / f"fastpitch-{language}.int8.onnx"
        node = "node_MatMul_16_quant"
        tokens = json.loads((folder / f"fastpitch-{language}.tokens.json").read_text())
        ids = {s: i for i, s in enumerate(tokens)}
        inputs = {"text": np.array([[ids.get(c, ids["<BLNK>"]) for c in SAMPLES[language]]], np.int64)}
    if not source.exists():
        pytest.skip("local research model artifact not present")
    model = onnx.load(source)
    if mode == "full":
        model = rewrite_all(model)
        assert not any(n.op_type in {"ConvInteger", "MatMulInteger"} for n in model.graph.node)
    else:
        conv_node = next(n.name for n in model.graph.node if n.op_type == "ConvInteger")
        model = rewrite_one(rewrite_one(model, node), conv_node)
    path = tmp_path / "single-node-pilot.onnx"
    onnx.save(model, path)
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    result = session.run(None, inputs)[0]
    if language != "or":
        vocoder_source = folder / f"hifigan-{language}.int8.onnx"
        if mode == "full":
            vocoder_model = rewrite_all(onnx.load(vocoder_source))
            assert not any(n.op_type in {"ConvInteger", "MatMulInteger"} for n in vocoder_model.graph.node)
            vocoder_source = tmp_path / "hifigan-full.onnx"
            onnx.save(vocoder_model, vocoder_source)
        vocoder = ort.InferenceSession(str(vocoder_source), options, providers=["CPUExecutionProvider"])
        result = vocoder.run(None, {"mel": result})[0]
    assert result.size > 1000
    assert np.isfinite(result).all()
    assert np.max(np.abs(result)) > 0
