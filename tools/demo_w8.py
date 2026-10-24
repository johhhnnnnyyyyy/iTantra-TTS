"""Offline two-stream host-adapter drill; emits metadata, never user text or audio."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from itantra_tts.adapters.async_local import AsyncLocalResearchSynthesisBackend
from itantra_tts.domain import Precision, SynthesisResult
from itantra_tts.host_adapter import HostTextChunk, ItantraTtsAdapter


async def run(project_root: Path) -> list[dict[str, object]]:
    backend = AsyncLocalResearchSynthesisBackend(
        project_root / "models/artifacts/echobharat-int8",
        project_root / "models/artifacts/mms-tts-ory-int8-research",
    )
    events: list[dict[str, object]] = []
    async with ItantraTtsAdapter(backend.synthesize, workers=2) as adapter:
        adapter.open_stream("hindi")
        adapter.open_stream("english")
        # Submit the later Hindi chunk first; the adapter must still release 0 then 1.
        await adapter.submit(HostTextChunk("hindi", 1, "hi-1", "नमस्ते।", "hi", Precision.INT4))
        await adapter.submit(HostTextChunk("english", 0, "en-0", "Hello.", "en"))
        await adapter.submit(HostTextChunk("hindi", 0, "hi-0", "नमस्ते।", "hi"))
        for stream, count in (("english", 1), ("hindi", 2)):
            for _ in range(count):
                event = await asyncio.wait_for(adapter.receive(stream), 120)
                audio = event.result if isinstance(event.result, SynthesisResult) else None
                events.append({"stream_id": event.stream_id, "sequence_no": event.sequence_no,
                               "message_id": event.message_id, "kind": event.kind.value,
                               "audio_bytes": len(audio.audio_pcm_s16le) if audio else 0})
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.project_root)), indent=2))


if __name__ == "__main__":
    main()
