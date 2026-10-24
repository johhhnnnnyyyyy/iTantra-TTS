"""Measure one local TTS model and thread setting in a fresh process.

Run this tool separately for every setting so peak RSS is comparable. The
first request is cold; subsequent requests reuse the loaded model and are warm.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path

import onnxruntime as ort

from itantra_tts.adapters.onnx_fastpitch_hifigan import OnnxFastPitchHifiGanBackend
from itantra_tts.adapters.onnx_odia_vits import OnnxOdiaVitsResearchBackend
from itantra_tts.domain import Precision, SynthesisRequest, SynthesisResult


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language",
        choices=("bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "ta", "te"),
        required=True,
    )
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--warm-runs", type=int, default=2)
    parser.add_argument(
        "--echo-session-policy", choices=("resident", "sequential"), default="resident"
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("models/artifacts"))
    parser.add_argument(
        "--variant-label",
        help="Explicit identity for a substituted artifact root; required with --artifacts",
    )
    parser.add_argument(
        "--odia-model",
        choices=("model.int8.onnx", "model.static-qdq.int8.onnx"),
        default="model.int8.onnx",
    )
    args = parser.parse_args()
    if args.warm_runs < 1:
        parser.error("--warm-runs must be positive")
    if args.artifacts.resolve() != Path("models/artifacts").resolve() and not args.variant_label:
        parser.error("--variant-label is required for a substituted artifact root")

    if args.language == "or":
        backend = OnnxOdiaVitsResearchBackend(
            args.artifacts / "mms-tts-ory-int8-research",
            intra_op_threads=args.threads,
            model_filename=args.odia_model,
        )
    else:
        backend = OnnxFastPitchHifiGanBackend(
            args.artifacts / "echobharat-int8",
            intra_op_threads=args.threads,
            session_policy=args.echo_session_policy,
        )

    rows = []
    for index in range(args.warm_runs + 1):
        result = backend.synthesize(
            SynthesisRequest("benchmark", index, args.text, args.language, Precision.INT8)
        )
        if not isinstance(result, SynthesisResult) or result.metrics is None:
            raise RuntimeError(f"Synthesis failed at run {index}: {result}")
        metrics = result.metrics
        rows.append(
            {
                "cold": metrics.cold_start,
                "wall_ms": round(metrics.wall_latency_ms, 1),
                "rtf": round(metrics.real_time_factor, 3),
                "audio_ms": round(metrics.audio_duration_ms, 1),
                "peak_rss_mib": round(metrics.peak_rss_bytes / 1048576, 1),
                "stage_one_ms": round(metrics.fastpitch_latency_ms, 1),
                "stage_two_ms": round(metrics.hifigan_latency_ms, 1),
                "pcm_bytes": len(result.audio_pcm_s16le),
            }
        )
    repeated = rows[1:]
    warm = [row for row in repeated if not row["cold"]]
    print(
        json.dumps(
            {
                "language": args.language,
                "model": args.odia_model if args.language == "or" else "echobharat-int8",
                "variant_label": args.variant_label or "pinned-int8",
                "session_policy": args.echo_session_policy if args.language != "or" else "resident",
                "threads": args.threads,
                "machine": platform.machine(),
                "runtime": ort.__version__,
                "warm_runs": args.warm_runs,
                "cold": rows[0],
                "repeated": repeated,
                "repeat_median_rtf": round(
                    statistics.median(row["rtf"] for row in repeated), 3
                ),
                "warm_median_rtf": (
                    round(statistics.median(row["rtf"] for row in warm), 3) if warm else None
                ),
                "process_peak_rss_mib": max(row["peak_rss_mib"] for row in rows),
                "scope": "desktop local process; not Android device evidence",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
