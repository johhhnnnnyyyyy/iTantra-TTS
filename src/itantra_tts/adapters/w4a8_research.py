"""Isolated, explicitly experimental W4A8 adapter for the shared service."""
from __future__ import annotations

import asyncio
import io
import json
import sys
import tempfile
import wave
from pathlib import Path

from itantra_tts.domain import (
    Precision, SynthesisErrorCode, SynthesisFailure, SynthesisRequest, SynthesisResult,
)
from itantra_tts.domain.contracts import ResearchWorkerMetrics

ROOT = Path(__file__).resolve().parents[3]


class W4A8ResearchBackend:
    """Use local research graphs and a matching custom operator; not a release backend."""

    def __init__(self, library: Path, *, timeout: float = 300, threads: int = 2) -> None:
        if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 8:
            raise ValueError("threads must be between 1 and 8")
        self._library = library
        self._timeout = timeout
        self._threads = threads

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult | SynthesisFailure:
        failure = SynthesisFailure(request.stream_id, request.sequence_no,
                                   SynthesisErrorCode.MODEL_NOT_EXECUTABLE,
                                   "Experimental W4A8 synthesis failed or is unavailable.")
        if request.precision is not Precision.INT4 or not self._library.is_file():
            return failure
        process = None
        try:
            with tempfile.TemporaryDirectory(prefix="itantra-w4-service-") as folder:
                output = Path(folder) / "audio.wav"
                process = await asyncio.create_subprocess_exec(
                    sys.executable, str(ROOT / "tools/w4a8_research_demo.py"),
                    "--language", request.language, "--text-stdin", "--library", str(self._library),
                    "--output", str(output), "--threads", str(self._threads), stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await asyncio.wait_for(process.communicate(request.text.encode("utf-8")), self._timeout)
                if process.returncode != 0 or not output.is_file() or output.stat().st_size > 20_000_000:
                    return failure
                data = json.loads(stdout)
                metrics = ResearchWorkerMetrics(float(data["cpu_time_ms"]), int(data["process_peak_rss_bytes"]),
                                                float(data["wall_ms"]), str(data["runtime"]),
                                                "isolated cold desktop worker")
                with wave.open(io.BytesIO(output.read_bytes()), "rb") as wav:
                    if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getnframes() == 0:
                        return failure
                    rate = wav.getframerate()
                    pcm = wav.readframes(wav.getnframes())
                variant = "scalar" if request.language == "or" else "per-channel"
                return SynthesisResult(request.stream_id, request.sequence_no, request.language,
                                       request.precision, f"experimental-w4a8-{variant}-{request.language}",
                                       rate, pcm, research_metrics=metrics)
        except (OSError, ValueError, KeyError, TypeError, wave.Error, asyncio.TimeoutError):
            return failure
        finally:
            if process is not None and process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
