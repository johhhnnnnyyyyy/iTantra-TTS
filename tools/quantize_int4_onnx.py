"""Create an experimental ONNX-standard INT4 weight-only QDQ graph.

Only eligible constant floating-point MatMul weights are quantized. The output
is mixed precision and requires independent execution and listening tests.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantFormat, matmul_nbits_quantizer


def quantize(source: Path, destination: Path) -> int:
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError("destination must be a new file distinct from the input")
    model = onnx.load(source)
    config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
        block_size=128, is_symmetric=True, quant_format=QuantFormat.QDQ,
        op_types_to_quantize=("MatMul",), bits=4,
    )
    logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.WARNING)
    quantizer = matmul_nbits_quantizer.MatMulNBitsQuantizer(model, algo_config=config)
    quantizer.process()
    result = quantizer.model.model
    count = sum(t.data_type in (onnx.TensorProto.INT4, onnx.TensorProto.UINT4)
                for t in result.graph.initializer)
    if not count:
        raise ValueError("no eligible constant MatMul weights were quantized to INT4")
    onnx.checker.check_model(result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(result, destination)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(f"INT4 initializer count: {quantize(args.source, args.destination)}")


if __name__ == "__main__":
    main()
