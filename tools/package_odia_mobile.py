"""Create an ARM-oriented ORT package and a conservative NNAPI readiness report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from itantra_tts.mobile.odia_readiness import analyze_required_operators


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Derived Odia INT8 ONNX model")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "onnxruntime.tools.convert_onnx_models_to_ort",
            str(args.model),
            "--output_dir",
            str(args.output_dir),
            "--optimization_style",
            "Fixed",
            "Runtime",
            "--enable_type_reduction",
            "--target_platform",
            "arm",
        ],
        check=True,
    )
    config = args.output_dir / f"{args.model.stem}.required_operators_and_types.config"
    report = analyze_required_operators(config.read_text(encoding="utf-8"))
    payload = {
        "artifact": f"{args.model.stem}.ort",
        "classification": "candidate" if report.mobile_ready else "cpu_fallback_required",
        "mobile_ready": report.mobile_ready,
        "required_operators": report.required_operators,
        "nnapi_blockers": report.nnapi_blockers,
        "next_gate": "instrumented Android ARM64 run on a declared target device",
    }
    (args.output_dir / "mobile_readiness.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
