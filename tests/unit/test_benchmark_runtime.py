"""The benchmark entry point must cover every supported TTS language."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    not (ROOT / "models/artifacts/echobharat-int8/gu/hifigan-gu.int8.onnx").exists(),
    reason="Gujarati INT8 artifact absent",
)
def test_benchmark_accepts_non_hindi_echobharat_language() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "tools/benchmark_runtime.py",
            "--language",
            "gu",
            "--threads",
            "4",
            "--warm-runs",
            "1",
            "--echo-session-policy",
            "sequential",
            "--variant-label",
            "test-labelled-variant",
            "--text",
            "નમસ્તે.",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    measurement = json.loads(completed.stdout)
    assert measurement["language"] == "gu"
    assert measurement["variant_label"] == "test-labelled-variant"
    assert measurement["session_policy"] == "sequential"
    assert measurement["repeated"][0]["pcm_bytes"] > 0
