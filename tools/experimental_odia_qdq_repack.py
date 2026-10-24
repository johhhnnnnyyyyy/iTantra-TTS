"""Repack Odia static-QDQ weights as ONNX INT4 while retaining INT8 execution.

Each original FP32 weight is packed into INT4, then dequantized and quantized
back to the original INT8 value type before the original QDQ graph runs.
This is storage-only INT4, not native INT4 arithmetic. Research use only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx
from onnx import helper, numpy_helper, version_converter


def convert(source_int8: Path, source_fp32: Path, destination: Path) -> int:
    if destination.exists() or destination.resolve() in {source_int8.resolve(), source_fp32.resolve()}:
        raise ValueError("destination must be a new file distinct from the sources")
    model = onnx.load(source_int8)
    if model.opset_import[0].version < 21:
        model = version_converter.convert_version(model, 21)
    source = onnx.load(source_fp32)
    fp32 = {item.name: numpy_helper.to_array(item) for item in source.graph.initializer}
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    insertions = []
    packed = []
    replaced = set()
    for node in model.graph.node:
        if node.op_type != "DequantizeLinear" or not node.input:
            continue
        name = node.input[0]
        if name in replaced or name not in initializers or not name.endswith("_quantized"):
            continue
        weight8 = initializers[name]
        if weight8.dtype not in (np.int8, np.uint8) or weight8.size < 32:
            continue
        original_name = name.removesuffix("_quantized")
        if original_name not in fp32 or fp32[original_name].shape != weight8.shape:
            raise ValueError(f"missing matching FP32 weight: {original_name}")
        weight = fp32[original_name].astype(np.float32)
        if not np.isfinite(weight).all():
            raise ValueError(f"non-finite FP32 weight: {original_name}")
        axis = next((attribute.i for attribute in node.attribute if attribute.name == "axis"), 1)
        if axis < 0:
            axis += weight.ndim
        if axis >= weight.ndim:
            raise ValueError(f"invalid QDQ axis: {name}")
        reduce_axes = tuple(index for index in range(weight.ndim) if index != axis)
        maxima = np.max(np.abs(weight), axis=reduce_axes)
        scales = np.where(maxima == 0, 1.0, maxima / 7.0).astype(np.float32)
        shape = [1] * weight.ndim
        shape[axis] = weight.shape[axis]
        q4 = np.clip(np.rint(weight / scales.reshape(shape)), -7, 7).astype(ml_dtypes.int4)
        q4_name = f"{name}.int4_storage"
        scale4_name = f"{name}.int4_scale"
        restored_name = f"{name}.int4_dequantized"
        packed.extend((numpy_helper.from_array(q4, q4_name),
                       numpy_helper.from_array(scales, scale4_name)))
        insertions.extend((
            helper.make_node("DequantizeLinear", [q4_name, scale4_name], [restored_name],
                             axis=axis, name=f"{original_name}.unpack_int4"),
            helper.make_node("QuantizeLinear", [restored_name, node.input[1], node.input[2]],
                             [name], axis=axis, name=f"{original_name}.restore_int8_compute"),
        ))
        replaced.add(name)
    if not replaced:
        raise ValueError("no eligible static-QDQ weights")
    kept = [item for item in model.graph.initializer if item.name not in replaced]
    original_nodes = list(model.graph.node)
    del model.graph.initializer[:]
    model.graph.initializer.extend([*kept, *packed])
    del model.graph.node[:]
    model.graph.node.extend([*insertions, *original_nodes])
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)
    return len(replaced)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_int8", type=Path)
    parser.add_argument("source_fp32", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(f"packed static-QDQ weights: {convert(args.source_int8, args.source_fp32, args.destination)}")


if __name__ == "__main__":
    main()
