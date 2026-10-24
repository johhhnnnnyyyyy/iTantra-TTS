"""Build an experimental standard-ONNX INT4-QDQ HiFi-GAN from an INT8 graph.

This replaces ConvInteger chains with INT4 DequantizeLinear -> FP32 Conv.
It is *not* a native INT4 convolution kernel and may be slower or use more RAM.
The optional checkpoint supplies original FP32 weights for a source comparison.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx
from onnx import helper, numpy_helper, version_converter
import torch

from tools.audit_indic_tts_quantization import fold_weight_norm


def quantize_per_output_channel(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if weights.ndim != 3:
        raise ValueError("expected Conv1d weights [out, in, kernel]")
    maxima = np.max(np.abs(weights), axis=(1, 2))
    scales = np.where(maxima == 0, 1.0, maxima / 7.0).astype(np.float32)
    quantized = np.clip(np.rint(weights / scales[:, None, None]), -7, 7)
    return quantized.astype(ml_dtypes.int4), scales


def remove_dead_graph_nodes(model: onnx.ModelProto) -> None:
    """Drop the unused INT8 quantization chains left by Conv replacement."""
    producers = {output: node for node in model.graph.node for output in node.output}
    live_names = set(output.name for output in model.graph.output)
    live_nodes = set()
    pending = list(live_names)
    while pending:
        name = pending.pop()
        node = producers.get(name)
        if node is None or id(node) in live_nodes:
            continue
        live_nodes.add(id(node))
        for input_name in node.input:
            if input_name and input_name not in live_names:
                live_names.add(input_name)
                pending.append(input_name)
    kept_nodes = [node for node in model.graph.node if id(node) in live_nodes]
    kept_initializers = [item for item in model.graph.initializer if item.name in live_names]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)


def convert(
    source: Path, destination: Path, checkpoint: Path | None = None, precision: str = "int4"
) -> int:
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError("destination must be a new file distinct from the source")
    if precision not in {"int4", "fp32"} or (precision == "fp32" and checkpoint is None):
        raise ValueError("FP32 output requires a checkpoint; precision must be int4 or fp32")
    model = onnx.load(source)
    if model.opset_import[0].version < 21:
        model = version_converter.convert_version(model, 21)
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    state = None
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)["model"]

    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    producers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers[name].append(node)
        for name in node.output:
            producers[name] = node

    removed_ids = set()
    insertions: dict[int, list[onnx.NodeProto]] = {}
    new_initializers = []
    converted = 0
    for node in model.graph.node:
        if node.op_type != "ConvInteger":
            continue
        quantized_name = node.input[1]
        if not quantized_name.startswith("generator.") or not quantized_name.endswith(".weight_quantized"):
            raise ValueError(f"unexpected quantized weight: {quantized_name}")
        prefix = quantized_name.removesuffix(".weight_quantized")
        activation_quantizer = producers.get(node.input[0])
        if activation_quantizer is None or activation_quantizer.op_type != "DynamicQuantizeLinear":
            raise ValueError(f"unexpected ConvInteger activation input: {prefix}")
        cast_nodes = consumers[node.output[0]]
        if len(cast_nodes) != 1 or cast_nodes[0].op_type != "Cast":
            raise ValueError(f"unexpected ConvInteger output chain: {prefix}")
        cast = cast_nodes[0]
        scaled_nodes = consumers[cast.output[0]]
        if len(scaled_nodes) != 1 or scaled_nodes[0].op_type != "Mul":
            raise ValueError(f"unexpected ConvInteger scale chain: {prefix}")
        scaled = scaled_nodes[0]
        bias_nodes = consumers[scaled.output[0]]
        if len(bias_nodes) != 1 or bias_nodes[0].op_type != "Add":
            raise ValueError(f"unexpected ConvInteger bias chain: {prefix}")
        biased = bias_nodes[0]
        bias_name = f"{prefix}.bias"
        if bias_name not in initializers:
            raise ValueError(f"missing Conv bias: {prefix}")

        if state is None:
            zero_point = initializers[node.input[3]].astype(np.int32)
            weights = (initializers[quantized_name].astype(np.int32) - zero_point).astype(np.float32)
            weights *= initializers[f"{prefix}.weight_scale"]
        else:
            state_prefix = "model_g." + prefix.removeprefix("generator.")
            weights = fold_weight_norm(
                state[f"{state_prefix}.weight_v"].numpy(),
                state[f"{state_prefix}.weight_g"].numpy(),
            )
        if precision == "int4":
            quantized, scales = quantize_per_output_channel(weights)
            weight_name = f"{prefix}.weight_int4"
            scale_name = f"{prefix}.weight_int4_scale"
            dequantized_name = f"{prefix}.weight_int4_dequantized"
            new_initializers.extend(
                [numpy_helper.from_array(quantized, weight_name), numpy_helper.from_array(scales, scale_name)]
            )
            weight_nodes = [helper.make_node(
                "DequantizeLinear", [weight_name, scale_name], [dequantized_name],
                axis=0, name=f"{prefix}.int4_dequantize",
            )]
            weight_input = dequantized_name
        else:
            weight_input = f"{prefix}.weight_fp32"
            new_initializers.append(numpy_helper.from_array(weights, weight_input))
            weight_nodes = []
        conv = helper.make_node(
            "Conv", [activation_quantizer.input[0], weight_input, bias_name],
            list(biased.output), name=f"{prefix}.{precision}_conv",
        )
        conv.attribute.extend(node.attribute)
        insertions[id(node)] = [*weight_nodes, conv]
        removed_ids.update((id(node), id(cast), id(scaled), id(biased)))
        converted += 1
    if not converted:
        raise ValueError("no ConvInteger weights converted")
    nodes = []
    for node in model.graph.node:
        nodes.extend(insertions.get(id(node), []))
        if id(node) not in removed_ids:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(new_initializers)
    remove_dead_graph_nodes(model)
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--precision", choices=("int4", "fp32"), default="int4")
    args = parser.parse_args()
    print(f"converted ConvInteger nodes: {convert(args.source, args.destination, args.checkpoint, args.precision)}")


if __name__ == "__main__":
    main()
