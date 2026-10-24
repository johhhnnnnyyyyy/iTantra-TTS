from itantra_tts.mobile.odia_readiness import analyze_required_operators


def test_dynamic_integer_operators_block_full_nnapi_delegation() -> None:
    config = """\
ai.onnx;10;ConvInteger
ai.onnx;11;ConvTranspose,DynamicQuantizeLinear
com.microsoft;1;DynamicQuantizeMatMul,MatMulIntegerToFloat
"""

    report = analyze_required_operators(config)

    assert report.mobile_ready is False
    assert report.nnapi_blockers == (
        "ai.onnx:ConvInteger",
        "ai.onnx:ConvTranspose",
        "ai.onnx:DynamicQuantizeLinear",
        "com.microsoft:DynamicQuantizeMatMul",
        "com.microsoft:MatMulIntegerToFloat",
    )


def test_known_nnapi_subset_does_not_block_mobile_readiness_gate() -> None:
    config = """\
ai.onnx;13;Cast,Concat,DequantizeLinear,MatMul,Reshape,Softmax
ai.onnx;14;Add,Mul,Relu
"""

    report = analyze_required_operators(config)

    assert report.mobile_ready is True
    assert report.nnapi_blockers == ()


def test_type_annotations_do_not_get_mistaken_for_operators() -> None:
    config = 'ai.onnx;13;Cast{"inputs": {"0": ["bool", "float"]}},MatMul\n'

    report = analyze_required_operators(config)

    assert report.required_operators == ("ai.onnx:Cast", "ai.onnx:MatMul")
