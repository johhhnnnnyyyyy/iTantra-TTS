"""Experimental, narrowly checked integer operator to packed W4A8 rewrite.

This rewrites one node only. It does not make a complete TTS model INT4.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def rewrite_one(source: onnx.ModelProto, node_name: str, *, per_channel: bool = False) -> onnx.ModelProto:
    model = copy.deepcopy(source)
    nodes = list(model.graph.node)
    matches = [n for n in nodes if n.name == node_name and n.op_type in {"MatMulInteger", "ConvInteger"}]
    if len(matches) != 1:
        raise ValueError("expected exactly one named supported integer node")
    node = matches[0]
    initializers = {x.name: x for x in model.graph.initializer}
    if len(node.input) != 4 or not node.input[2] or node.input[1] not in initializers or node.input[3] not in initializers:
        raise ValueError("constant weights and explicit zero points are required")
    if {node.input[1], node.input[3]} & {x.name for x in model.graph.input}:
        raise ValueError("weight or zero point exposed as overridable graph input")
    weights = numpy_helper.to_array(initializers[node.input[1]])
    zero = numpy_helper.to_array(initializers[node.input[3]])
    expected_rank = 2 if node.op_type == "MatMulInteger" else 3
    if weights.dtype != np.int8 or weights.ndim != expected_rank or not weights.size:
        raise ValueError("weights must be a nonempty INT8 tensor of the expected rank")
    if zero.dtype != np.int8 or np.any(zero != 0):
        raise ValueError("weight zero point must be symmetric INT8 zero")
    consumers = [n for n in nodes if node.output[0] in n.input]
    if len(consumers) != 1 or consumers[0].op_type != "Cast":
        raise ValueError("integer output must have exactly one Cast consumer")
    cast = consumers[0]
    cast_to = [helper.get_attribute_value(a) for a in cast.attribute if a.name == "to"]
    if cast_to != [TensorProto.FLOAT] or node.output[0] in {x.name for x in model.graph.output}:
        raise ValueError("integer output must feed a float Cast only")
    prefix = f"{node_name}.itantra_w4"
    existing = ({name for n in nodes for name in [*n.input, *n.output]} | set(initializers) |
                {x.name for x in [*model.graph.input, *model.graph.output, *model.graph.value_info]})
    names = [prefix + suffix for suffix in (".packed", ".shape", ".factor", ".cast", ".geometry")]
    if any(name in existing for name in names):
        raise ValueError("generated graph name already exists")
    if per_channel:
        axes = (0,) if node.op_type == "MatMulInteger" else (1, 2)
        maximum = np.max(np.abs(weights.astype(np.int16)), axis=axes)
        factors = np.where(maximum == 0, 1.0, maximum / 7.0).astype(np.float32)
        weight_factor = factors if node.op_type == "MatMulInteger" else factors[:, None, None]
        factor = factors if node.op_type == "MatMulInteger" else factors.reshape(1, -1, 1)
    else:
        maximum = int(np.max(np.abs(weights.astype(np.int16))))
        factor = np.float32(maximum / 7.0) if maximum else np.float32(1.0)
        weight_factor = factor
    quantized = np.clip(np.rint(weights.astype(np.float32) / weight_factor), -7, 7).astype(np.int8).ravel()
    nibble = quantized.astype(np.int16) & 15
    packed = np.zeros((nibble.size + 1) // 2, dtype=np.uint8)
    packed[:] = nibble[::2].astype(np.uint8)
    packed[:nibble[1::2].size] |= (nibble[1::2] << 4).astype(np.uint8)
    model.graph.initializer.extend([
        numpy_helper.from_array(packed, names[0]),
        numpy_helper.from_array(np.array(weights.shape, dtype=np.int64), names[1]),
        numpy_helper.from_array(np.array(factor, dtype=np.float32), names[2]),
    ])
    inputs = [node.input[0], names[0], node.input[2], names[1]]
    if node.op_type == "ConvInteger":
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if set(attrs) - {"auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"}:
            raise ValueError("unsupported convolution attribute")
        if attrs.get("auto_pad", b"NOTSET") not in (b"NOTSET", b""):
            raise ValueError("unsupported auto padding")
        strides, pads, dilations = attrs.get("strides", [1]), attrs.get("pads", [0, 0]), attrs.get("dilations", [1])
        if len(strides) != 1 or len(pads) != 2 or len(dilations) != 1:
            raise ValueError("only one-dimensional convolution is supported")
        if "kernel_shape" in attrs and list(attrs["kernel_shape"]) != [weights.shape[2]]:
            raise ValueError("kernel shape does not match weights")
        geometry = np.array([strides[0], *pads, dilations[0], attrs.get("group", 1)], dtype=np.int64)
        if geometry[0] <= 0 or geometry[3] <= 0 or geometry[4] <= 0 or np.any(geometry[1:3] < 0):
            raise ValueError("invalid convolution geometry")
        model.graph.initializer.append(numpy_helper.from_array(geometry, names[4]))
        inputs.append(names[4])
    replacement = helper.make_node("MatMulW4A8" if node.op_type == "MatMulInteger" else "ConvW4A8", inputs,
                                   list(node.output), name=node.name, domain="org.itantra.experimental")
    old_cast_output = cast.output[0]
    cast.output[0] = names[3]
    correction = helper.make_node("Mul", [names[3], names[2]], [old_cast_output], name=prefix + ".rescale")
    replaced_nodes = []
    for current in nodes:
        if current is node:
            replaced_nodes.append(replacement)
        elif current is cast:
            replaced_nodes.extend([cast, correction])
        else:
            replaced_nodes.append(current)
    del model.graph.node[:]
    model.graph.node.extend(replaced_nodes)
    weight_name = node.input[1]
    if not any(weight_name in n.input for n in model.graph.node) and weight_name not in {x.name for x in model.graph.output}:
        kept = [x for x in model.graph.initializer if x.name != weight_name]
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
    if not any(op.domain == "org.itantra.experimental" for op in model.opset_import):
        model.opset_import.append(helper.make_opsetid("org.itantra.experimental", 1))
    onnx.checker.check_model(model)
    return model


def rewrite_all(source: onnx.ModelProto, *, per_channel: bool = False) -> onnx.ModelProto:
    targets = [n.name for n in source.graph.node if n.op_type in {"MatMulInteger", "ConvInteger"}]
    if not targets or any(not name for name in targets) or len(set(targets)) != len(targets):
        raise ValueError("integer nodes must have unique, nonempty names")
    result = source
    for name in targets:
        result = rewrite_one(result, name, per_channel=per_channel)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("node", help="node name, or 'all' for every eligible node")
    parser.add_argument("--per-channel", action="store_true", help="Experimental per-output-channel W4 scaling")
    args = parser.parse_args()
    if args.destination.exists() or args.source.resolve() == args.destination.resolve():
        raise ValueError("destination must be a new path")
    source = onnx.load(args.source)
    result = (rewrite_all(source, per_channel=args.per_channel) if args.node == "all"
              else rewrite_one(source, args.node, per_channel=args.per_channel))
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(result, args.destination)


if __name__ == "__main__":
    main()
