"""Pack constant Odia VITS Conv/ConvTranspose/MatMul weights into ONNX INT4.

Research only. Each weight is dequantized to FP32 for the original operator;
this is not native INT4 CPU inference and must not be marketed as such.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx
from onnx import helper, numpy_helper, version_converter


def convert(source: Path, destination: Path) -> int:
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError("destination must be a new file distinct from the source")
    model = onnx.load(source)
    if model.opset_import[0].version < 21:
        model = version_converter.convert_version(model, 21)
    original = {item.name: item for item in model.graph.initializer}
    unpack_nodes = []
    packed_weights = []
    used_names = set()
    converted = 0
    for node in model.graph.node:
        if node.op_type not in {"Conv", "ConvTranspose", "MatMul"} or len(node.input) < 2:
            continue
        name = node.input[1]
        tensor = original.get(name)
        if tensor is None or tensor.data_type != onnx.TensorProto.FLOAT:
            continue
        weight = numpy_helper.to_array(tensor)
        if weight.ndim < 2 or weight.size < 32:
            continue
        axis = 1 if node.op_type in {"ConvTranspose", "MatMul"} else 0
        reduce_axes = tuple(i for i in range(weight.ndim) if i != axis)
        maxima = np.max(np.abs(weight), axis=reduce_axes)
        if not np.isfinite(maxima).all():
            raise ValueError(f"non-finite weight: {name}")
        scales = np.where(maxima == 0, 1.0, maxima / 7.0).astype(np.float32)
        shape = [1] * weight.ndim
        shape[axis] = weight.shape[axis]
        packed = np.clip(np.rint(weight / scales.reshape(shape)), -7, 7).astype(ml_dtypes.int4)
        tag = f"int4_storage_{converted}"
        packed_name = f"{tag}.weight"
        scale_name = f"{tag}.scale"
        restored_name = f"{tag}.dequantized"
        packed_weights.extend((
            numpy_helper.from_array(packed, packed_name),
            numpy_helper.from_array(scales, scale_name),
        ))
        unpack_nodes.append(helper.make_node(
            "DequantizeLinear", [packed_name, scale_name], [restored_name],
            axis=axis, name=f"{tag}.unpack",
        ))
        node.input[1] = restored_name
        used_names.add(name)
        converted += 1
    if not converted:
        raise ValueError("no eligible constant convolution or MatMul weights")
    # Keep any FP32 initializer still used by a different node.
    remaining_inputs = {name for node in model.graph.node for name in node.input}
    kept = [item for item in model.graph.initializer
            if item.name not in used_names or item.name in remaining_inputs]
    original_nodes = list(model.graph.node)
    del model.graph.initializer[:]
    model.graph.initializer.extend([*kept, *packed_weights])
    del model.graph.node[:]
    model.graph.node.extend([*unpack_nodes, *original_nodes])
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(f"packed constant weights: {convert(args.source, args.destination)}")


if __name__ == "__main__":
    main()
