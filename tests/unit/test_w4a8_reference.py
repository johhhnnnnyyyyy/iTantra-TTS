"""Check W4A8 accumulation with packed INT4 weights and INT8 activations."""

import ctypes
import random
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def conv(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = Path(__file__).parents[2] / "tools" / "w4a4_reference.cpp"
    library = tmp_path_factory.mktemp("w4a8") / "reference.so"
    subprocess.run([compiler, "-std=c++17", "-O2", "-shared", "-fPIC", str(source),
                    "-o", str(library)], check=True, capture_output=True)
    function = ctypes.CDLL(str(library)).itantra_w4a8_conv1d
    function.argtypes = [ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_uint8),
                         ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                         ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                         ctypes.POINTER(ctypes.c_int32)]
    function.restype = ctypes.c_int
    return function


def test_w4a8_convolution_uses_int8_activations_and_packed_signed_int4_weights(conv):
    activations = (ctypes.c_int8 * 5)(1, -8, 7, -1, 2)
    weights = (ctypes.c_uint8 * 2)(0x3E, 0x0F)  # [-2, 3, -1]
    result = (ctypes.c_int32 * 3)()
    assert conv(activations, weights, 5, 1, 1, 3, 2, 1, result) == 0
    assert list(result) == [11, 38, 8]


def test_w4a8_convolution_rejects_zero_stride(conv):
    activations = (ctypes.c_int8 * 1)(1)
    weights = (ctypes.c_uint8 * 1)(0x01)
    result = (ctypes.c_int32 * 1)()
    assert conv(activations, weights, 1, 1, 1, 1, 0, 0, result) != 0


@pytest.fixture(scope="module")
def matmul(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = Path(__file__).parents[2] / "tools" / "w4a4_reference.cpp"
    library = tmp_path_factory.mktemp("w4a8_matmul") / "reference.so"
    subprocess.run([compiler, "-std=c++17", "-O2", "-shared", "-fPIC", str(source),
                    "-o", str(library)], check=True, capture_output=True)
    function = ctypes.CDLL(str(library)).itantra_w4u8_matmul
    function.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
                         ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                         ctypes.c_uint8, ctypes.POINTER(ctypes.c_int32)]
    function.restype = ctypes.c_int
    return function


def test_w4u8_matmul_uses_zero_point_and_packed_weights(matmul):
    # A is [2, 3], B is [3, 2], packed in row-major order.
    activations = (ctypes.c_uint8 * 6)(128, 130, 127, 255, 0, 128)
    weights = (ctypes.c_uint8 * 3)(0xF2, 0x87, 0x35)  # [[2,-1],[7,-8],[5,3]]
    result = (ctypes.c_int32 * 4)()
    assert matmul(activations, weights, 2, 3, 2, 128, result) == 0
    assert list(result) == [9, -19, -642, 897]


def test_w4u8_matmul_supports_an_odd_weight_count(matmul):
    activations = (ctypes.c_uint8 * 3)(2, 3, 4)
    weights = (ctypes.c_uint8 * 2)(0xF1, 0x08)
    result = (ctypes.c_int32 * 1)()
    assert matmul(activations, weights, 1, 3, 1, 0, result) == 0
    assert result[0] == -33


def test_w4u8_matmul_rejects_invalid_shape(matmul):
    activations = (ctypes.c_uint8 * 1)(1)
    weights = (ctypes.c_uint8 * 1)(1)
    result = (ctypes.c_int32 * 1)()
    assert matmul(activations, weights, 0, 1, 1, 0, result) != 0


def test_w4u8_matmul_matches_scalar_reference_for_varied_shapes(matmul):
    rng = random.Random(20261021)
    for rows, inner, columns in [(1, 1, 1), (2, 5, 3), (3, 7, 4), (4, 2, 5)]:
        values = [rng.randrange(256) for _ in range(rows * inner)]
        weights = [rng.randrange(-8, 8) for _ in range(inner * columns)]
        zero_point = rng.randrange(256)
        packed = bytearray((len(weights) + 1) // 2)
        for index, value in enumerate(weights):
            packed[index // 2] |= (value & 15) << (4 * (index % 2))
        activation_buffer = (ctypes.c_uint8 * len(values))(*values)
        weight_buffer = (ctypes.c_uint8 * len(packed))(*packed)
        result = (ctypes.c_int32 * (rows * columns))()
        assert matmul(activation_buffer, weight_buffer, rows, inner, columns,
                      zero_point, result) == 0
        expected = [sum((values[row * inner + k] - zero_point) *
                        weights[k * columns + col] for k in range(inner))
                    for row in range(rows) for col in range(columns)]
        assert list(result) == expected
