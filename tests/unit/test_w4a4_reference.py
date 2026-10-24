"""Exercise the packed integer convolution without widening whole tensors."""

import ctypes
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
    library = tmp_path_factory.mktemp("w4a4") / "reference.so"
    subprocess.run([compiler, "-std=c++17", "-O2", "-shared", "-fPIC", str(source),
                    "-o", str(library)], check=True, capture_output=True)
    fn = ctypes.CDLL(str(library)).itantra_w4a4_conv1d
    ptr8 = ctypes.POINTER(ctypes.c_uint8)
    ptr32 = ctypes.POINTER(ctypes.c_int32)
    fn.argtypes = [ptr8, ptr8, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                   ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ptr32]
    fn.restype = ctypes.c_int
    matmul = ctypes.CDLL(str(library)).itantra_w4a4_matmul
    matmul.argtypes = [ptr8, ptr8, ctypes.c_size_t, ctypes.c_size_t,
                       ctypes.c_size_t, ptr32]
    matmul.restype = ctypes.c_int
    transposed = ctypes.CDLL(str(library)).itantra_w4a4_conv_transpose1d
    transposed.argtypes = [ptr8, ptr8, ctypes.c_size_t, ctypes.c_size_t,
                           ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                           ctypes.c_size_t, ptr32]
    transposed.restype = ctypes.c_int
    return fn, matmul, transposed


def test_signed_int4_convolution_padding_stride_and_odd_tensor(conv):
    # Input [1, -8, 7, -1, 2], weight [-2, 3, -1].
    # Padding 1, stride 2 -> [11, 38, 8].
    inputs = (ctypes.c_uint8 * 3)(0x81, 0xF7, 0x02)
    weights = (ctypes.c_uint8 * 2)(0x3E, 0x0F)
    result = (ctypes.c_int32 * 3)()
    assert conv[0](inputs, weights, 5, 1, 1, 3, 2, 1, result) == 0
    assert list(result) == [11, 38, 8]


def test_convolution_rejects_invalid_stride(conv):
    values = (ctypes.c_uint8 * 1)(0x11)
    result = (ctypes.c_int32 * 2)()
    assert conv[0](values, values, 2, 1, 1, 1, 0, 0, result) != 0


def test_convolution_accumulates_input_channels_and_separates_outputs(conv):
    inputs = (ctypes.c_uint8 * 3)(0x21, 0x43, 0x65)
    weights = (ctypes.c_uint8 * 4)(0x11, 0x1F, 0x02, 0xE0)
    result = (ctypes.c_int32 * 4)()
    assert conv[0](inputs, weights, 3, 2, 2, 2, 1, 0, result) == 0
    assert list(result) == [4, 6, -8, -8]


def test_packed_matmul_with_signed_values_and_odd_dimension(conv):
    # [[1, -2, 3], [-8, 7, 1]] @ [[2, -1], [3, 4], [-2, 1]]
    left = (ctypes.c_uint8 * 3)(0xE1, 0x83, 0x17)
    right = (ctypes.c_uint8 * 3)(0xF2, 0x43, 0x1E)
    result = (ctypes.c_int32 * 4)()
    assert conv[1](left, right, 2, 3, 2, result) == 0
    assert list(result) == [-10, -6, 3, 37]


def test_transposed_convolution_stride_padding(conv):
    # Input [1, -2, 3], kernel [2, -1, 1], stride 2, padding 1.
    # Uncropped [2, -1, -3, 2, 4, -3, 3] -> [-1, -3, 2, 4, -3].
    inputs = (ctypes.c_uint8 * 2)(0xE1, 0x03)
    weights = (ctypes.c_uint8 * 2)(0xF2, 0x01)
    result = (ctypes.c_int32 * 5)()
    assert conv[2](inputs, weights, 3, 1, 1, 3, 2, 1, result) == 0
    assert list(result) == [-1, -3, 2, 4, -3]
