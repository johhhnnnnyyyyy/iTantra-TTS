"""Pack FastPitch's INT8 constant weights as ONNX INT4 for research.

The graph dequantizes INT4 weights to FP32, then requantizes them to INT8 for
the original ConvInteger/MatMulInteger nodes. This is weight *storage* only:
it is not native INT4 compute, and it may increase load time or peak RSS.
Never replace a pinned INT8 model without listening and device benchmarks.
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
    initializers = {item.name: item for item in model.graph.initializer}
    arrays = {name: numpy_helper.to_array(item) for name, item in initializers.items()}
    new_nodes = []
    new_initializers = []
    replaced_names = set()
    for node in model.graph.node:
        if node.op_type not in {"ConvInteger", "MatMulInteger"}:
            continue
        weight_name = node.input[1]
        if weight_name in replaced_names:
            continue
        if weight_name not in arrays or arrays[weight_name].dtype != np.int8:
            raise ValueError(f"expected constant INT8 weight: {weight_name}")
        if len(node.input) < 4 or node.input[3] not in arrays:
            raise ValueError(f"missing weight zero point: {weight_name}")
        base = weight_name.removesuffix("_quantized")
        source_scale = f"{base}_scale"
        if source_scale not in arrays:
            raise ValueError(f"missing weight scale: {weight_name}")
        weight = arrays[weight_name]
        scale8 = arrays[source_scale]
        zero8 = arrays[node.input[3]]
        if scale8.ndim or zero8.ndim:
            raise ValueError(f"non-scalar INT8 quantization: {weight_name}")
        if not np.isfinite(scale8).all() or float(scale8) <= 0:
            raise ValueError(f"invalid INT8 weight scale: {weight_name}")
        source_fp32 = (weight.astype(np.int16) - int(zero8)).astype(np.float32) * float(scale8)
        axis = 0 if node.op_type == "ConvInteger" else 1
        reduce_axes = tuple(index for index in range(weight.ndim) if index != axis)
        maxima = np.max(np.abs(source_fp32), axis=reduce_axes)
        scale4 = np.where(maxima == 0, 1.0, maxima / 7.0).astype(np.float32)
        shape = [1] * weight.ndim
        shape[axis] = weight.shape[axis]
        weight4 = np.clip(np.rint(source_fp32 / scale4.reshape(shape)), -7, 7).astype(ml_dtypes.int4)
        packed_name = f"{weight_name}.int4_storage"
        scale_name = f"{weight_name}.int4_scale"
        float_name = f"{weight_name}.int4_dequantized"
        new_initializers.extend((
            numpy_helper.from_array(weight4, packed_name),
            numpy_helper.from_array(scale4, scale_name),
        ))
        new_nodes.extend((
            helper.make_node("DequantizeLinear", [packed_name, scale_name], [float_name],
                             axis=axis, name=f"{base}.unpack_int4"),
            helper.make_node("QuantizeLinear", [float_name, source_scale, node.input[3]],
                             [weight_name], name=f"{base}.restore_int8_compute"),
        ))
        replaced_names.add(weight_name)
    if not replaced_names:
        raise ValueError("no eligible constant INT8 weights")
    kept = [item for item in model.graph.initializer if item.name not in replaced_names]
    original_nodes = list(model.graph.node)
    del model.graph.initializer[:]
    model.graph.initializer.extend([*kept, *new_initializers])
    del model.graph.node[:]
    model.graph.node.extend([*new_nodes, *original_nodes])
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)
    return len(replaced_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(f"packed constant weights: {convert(args.source, args.destination)}")


if __name__ == "__main__":
    main()
