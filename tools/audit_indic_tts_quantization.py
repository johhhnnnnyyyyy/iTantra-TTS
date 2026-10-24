"""Compare EchoBharat INT8 vocoder weights with an Indic-TTS FP32 checkpoint.

This is a weight-level source audit, not an audio-quality or INT4 runtime test.
It uses only local files and never loads untrusted pickle objects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def fold_weight_norm(weight_v: np.ndarray, weight_g: np.ndarray) -> np.ndarray:
    """Fold PyTorch weight_norm(dim=0) as used by HiFi-GAN Conv1d."""
    axes = tuple(range(1, weight_v.ndim))
    norms = np.sqrt(np.sum(np.square(weight_v, dtype=np.float64), axis=axes))
    if np.any(norms == 0):
        raise ValueError("zero-norm checkpoint weight")
    shape = (weight_v.shape[0],) + (1,) * (weight_v.ndim - 1)
    return (weight_v * (weight_g.reshape(shape) / norms.reshape(shape))).astype(np.float32)


def simulated_int4_roundtrip(weights: np.ndarray, block_size: int = 128) -> np.ndarray:
    """Analytical symmetric blockwise W4 approximation; not an ONNX conversion."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    rows = weights.reshape(weights.shape[0], -1)
    result = np.empty_like(rows)
    for start in range(0, rows.shape[1], block_size):
        block = rows[:, start : start + block_size]
        scale = np.max(np.abs(block), axis=1, keepdims=True) / 7.0
        scale = np.where(scale == 0, 1.0, scale)
        result[:, start : start + block.shape[1]] = (
            np.clip(np.rint(block / scale), -7, 7) * scale
        )
    return result.reshape(weights.shape)


def relative_rmse(reference: np.ndarray, candidate: np.ndarray) -> float:
    if reference.shape != candidate.shape:
        raise ValueError("weight shapes differ")
    denominator = np.sqrt(np.mean(np.square(reference, dtype=np.float64)))
    if denominator == 0:
        raise ValueError("zero-energy reference weight")
    return float(np.sqrt(np.mean(np.square(reference - candidate, dtype=np.float64))) / denominator)


def audit(checkpoint: Path, int8_graph: Path) -> dict[str, object]:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The checkpoint audit requires the research-fp32 extra. "
            "Install with: pip install -e '.[research-fp32]'"
        ) from error
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = saved["model"]
    model = onnx.load(int8_graph)
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    quantized_names = sorted(name for name in initializers if name.endswith(".weight_quantized"))
    if not quantized_names:
        raise ValueError("no quantized vocoder weights in graph")

    layer_results = []
    for quantized_name in quantized_names:
        prefix = quantized_name.removeprefix("generator.").removesuffix(".weight_quantized")
        state_prefix = f"model_g.{prefix}"
        if f"{state_prefix}.weight_v" not in state or f"{state_prefix}.weight_g" not in state:
            raise ValueError(f"checkpoint has no matching weight-norm pair: {prefix}")
        reference = fold_weight_norm(
            state[f"{state_prefix}.weight_v"].numpy(),
            state[f"{state_prefix}.weight_g"].numpy(),
        )
        int8 = initializers[quantized_name].astype(np.float32)
        scale = initializers[f"generator.{prefix}.weight_scale"]
        int8 *= scale
        if reference.shape != int8.shape:
            raise ValueError(f"checkpoint and graph shapes differ: {prefix}")
        direct_int4 = simulated_int4_roundtrip(reference)
        requantized_int4 = simulated_int4_roundtrip(int8)
        layer_results.append(
            {
                "layer": prefix,
                "elements": int(reference.size),
                "int8_vs_fp32_relative_rmse": relative_rmse(reference, int8),
                "direct_int4_vs_fp32_relative_rmse": relative_rmse(reference, direct_int4),
                "int8_to_int4_vs_fp32_relative_rmse": relative_rmse(reference, requantized_int4),
            }
        )
    return {
        "checkpoint": str(checkpoint),
        "int8_graph": str(int8_graph),
        "scope": "HiFi-GAN ConvInteger weights only; no speech inference or ONNX INT4 graph",
        "int4_method": "simulated symmetric per-output-channel blocks of 128, values -7..7",
        "layer_count": len(layer_results),
        "layers": layer_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("int8_graph", type=Path)
    parser.add_argument("--summary", action="store_true", help="omit per-layer results")
    args = parser.parse_args()
    report = audit(args.checkpoint, args.int8_graph)
    if args.summary:
        layers = report.pop("layers")
        count = sum(layer["elements"] for layer in layers)
        for key in (
            "int8_vs_fp32_relative_rmse",
            "direct_int4_vs_fp32_relative_rmse",
            "int8_to_int4_vs_fp32_relative_rmse",
        ):
            report[f"element_weighted_{key}"] = sum(
                layer[key] * layer["elements"] for layer in layers
            ) / count
        report["weight_elements"] = count
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
