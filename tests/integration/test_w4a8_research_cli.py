"""Standalone research CLI checks; no UI or production support is implied."""

import json
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import pytest


def test_model_selection_requires_per_channel_for_echobharat(tmp_path):
    from tools.w4a8_research_demo import select_model_dir

    old = tmp_path / "experimental-w4a8-scalar-20261021" / "hi"
    old.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        select_model_dir(tmp_path, "hi")
    new = tmp_path / "experimental-w4a8-per-channel" / "hi"
    new.mkdir(parents=True)
    assert select_model_dir(tmp_path, "hi") == new
    odia = tmp_path / "experimental-w4a8-scalar-20261021" / "or"
    odia.mkdir(parents=True)
    assert select_model_dir(tmp_path, "or") == odia

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "tools/w4a8_research_demo.py"


def test_research_cli_exposes_required_arguments():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--language" in result.stdout and "--library" in result.stdout


def test_research_cli_refuses_overwriting_existing_audio(tmp_path):
    output = tmp_path / "existing.wav"
    output.write_bytes(b"preserve")
    result = subprocess.run([sys.executable, str(SCRIPT), "--language", "hi", "--text", "हाँ",
                             "--library", str(tmp_path / "unused.so"), "--output", str(output)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert output.read_bytes() == b"preserve"


def test_research_cli_accepts_text_from_stdin_without_command_line_text(tmp_path):
    output = tmp_path / "existing.wav"
    output.write_bytes(b"preserve")
    result = subprocess.run([sys.executable, str(SCRIPT), "--language", "hi", "--text-stdin",
                             "--library", str(tmp_path / "unused.so"), "--output", str(output)],
                            input="नमस्ते", capture_output=True, text=True)
    assert result.returncode != 0
    assert "output already exists" in result.stderr
    assert "नमस्ते" not in result.stderr


def test_research_cli_rejects_unbounded_thread_count(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), "--language", "hi", "--text", "हाँ",
                             "--threads", "0", "--library", str(tmp_path / "unused.so"),
                             "--output", str(tmp_path / "new.wav")], capture_output=True, text=True)
    assert result.returncode != 0
    assert "threads must be between 1 and 8" in result.stderr


def test_research_cli_produces_real_hindi_wav(tmp_path):
    if os.environ.get("ITANTRA_W4_REAL_SMOKE") != "1":
        pytest.skip("set ITANTRA_W4_REAL_SMOKE=1 for real model synthesis")
    compiler, headers = shutil.which("c++"), os.environ.get("ORT_HEADER_DIR")
    if not compiler or not headers:
        pytest.skip("C++ compiler and ORT_HEADER_DIR required")
    library = tmp_path / "w4.so"
    subprocess.run([compiler, "-std=c++17", "-O2", "-shared", "-fPIC", "-I", headers,
                    str(ROOT / "tools/ort_w4a8_matmul.cpp"), "-o", str(library)], check=True)
    output = tmp_path / "audio.wav"
    result = subprocess.run([sys.executable, str(SCRIPT), "--language", "hi", "--text", "हाँ.",
                             "--library", str(library), "--output", str(output)],
                            cwd=ROOT, capture_output=True, text=True, timeout=180, check=True)
    metrics = json.loads(result.stdout)
    with wave.open(str(output), "rb") as audio:
        assert audio.getframerate() == 22050 and audio.getnframes() > 1000
    assert metrics["execution_precision"] == "W4A8 mixed"
    assert metrics["wall_ms"] > 0 and metrics["audio_duration_ms"] > 0
