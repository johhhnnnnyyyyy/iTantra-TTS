"""Cancellable async access to the existing offline research backends."""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from itantra_tts.domain import (
    Precision, SynthesisErrorCode, SynthesisFailure, SynthesisMetrics,
    SynthesisRequest, SynthesisResult,
)


class AsyncLocalResearchSynthesisBackend:
    """Run one local inference per subprocess, with cancellation killing that process.

    This desktop research adapter trades warm starts for isolation; it is not
    the mobile runtime. Content is sent through pipes, never command arguments.
    """

    def __init__(self, echobharat_root: Path, odia_root: Path,
                 odia_model_filename: str = "model.static-qdq.int8.onnx", *,
                 intra_op_threads: int = 4) -> None:
        if isinstance(intra_op_threads, bool) or not isinstance(intra_op_threads, int) or not 1 <= intra_op_threads <= 8:
            raise ValueError("intra_op_threads must be between 1 and 8")
        self._echo = echobharat_root
        self._odia = odia_root
        self._odia_model_filename = odia_model_filename
        self._intra_op_threads = intra_op_threads

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult | SynthesisFailure:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "itantra_tts.adapters.local_worker",
            str(self._echo), str(self._odia), self._odia_model_filename, str(self._intra_op_threads),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        assert process.stdin is not None and process.stdout is not None
        try:
            data = json.dumps(asdict(request), ensure_ascii=False).encode("utf-8")
            process.stdin.write(data)
            await process.stdin.drain()
            process.stdin.close()
            header = await process.stdout.readexactly(4)
            size = int.from_bytes(header, "big")
            if size > 6_000_000:
                raise ValueError("worker response exceeds limit")
            response = json.loads(await process.stdout.readexactly(size))
            if await process.wait() != 0:
                raise RuntimeError("local synthesis worker failed")
            if response["kind"] == "failure":
                return SynthesisFailure(response["stream_id"], response["sequence_no"],
                                        SynthesisErrorCode(response["code"]), response["detail"])
            if response["kind"] != "audio":
                raise ValueError("invalid worker response")
            metrics = SynthesisMetrics(**response["metrics"]) if response["metrics"] is not None else None
            if metrics is not None:
                metrics = replace(metrics, memory_scope="isolated subprocess worker")
            return SynthesisResult(
                response["stream_id"], response["sequence_no"], response["language"],
                Precision(response["precision"]), response["model_id"], response["sample_rate_hz"],
                base64.b64decode(response["audio"], validate=True),
                metrics,
            )
        finally:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
