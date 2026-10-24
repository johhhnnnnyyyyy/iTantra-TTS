"""Command line entry point for standalone local synthesis and HTTP serving."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from itantra_tts.adapters.async_local import AsyncLocalResearchSynthesisBackend
from itantra_tts.application.service import SynthesisService
from itantra_tts.domain import Precision, SynthesisRequest


def _backend(root: Path) -> AsyncLocalResearchSynthesisBackend:
    return AsyncLocalResearchSynthesisBackend(
        root / "models" / "artifacts" / "echobharat-int8",
        root / "models" / "artifacts" / "mms-tts-ory-int8-research",
        os.environ.get("ITANTRA_TTS_ODIA_RESEARCH_MODEL_FILE", "model.static-qdq.int8.onnx"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="itantra-tts", description="Local offline multilingual TTS research tools"
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    synth = subparsers.add_parser("synthesize", help="write a playable WAV file")
    synth.add_argument("--text", required=True)
    synth.add_argument("--language", required=True)
    synth.add_argument("--precision", choices=[p.value for p in Precision], default="int8")
    synth.add_argument("--stream-id", default="cli-local")
    synth.add_argument("--sequence-no", type=int, default=0)
    synth.add_argument("--output", type=Path, required=True)
    serve = subparsers.add_parser("serve", help="start the local HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.command == "synthesize":
        request = SynthesisRequest(args.stream_id, args.sequence_no, args.text,
                                   args.language, Precision(args.precision))
        try:
            result = asyncio.run(SynthesisService(_backend(root)).synthesize(request))
        except ValueError as exc:
            parser.error(str(exc))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(result.wav_bytes)
        print(f"Wrote playable WAV: {args.output} ({result.audio_duration_ms:.1f} ms)")
        print(f"Models: {', '.join(result.model_ids)}; normalization: {result.normalization}")
        return
    import uvicorn

    from itantra_tts.api import create_app

    uvicorn.run(
        create_app(project_root=root), host=args.host, port=args.port,
        access_log=False, log_level="warning",
    )


if __name__ == "__main__":
    main()
