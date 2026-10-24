"""One-request child process for the async local research adapter."""

import base64
import json
import sys
from dataclasses import asdict
from pathlib import Path

from itantra_tts.adapters.local_research import LocalResearchSynthesisBackend
from itantra_tts.adapters.onnx_fastpitch_hifigan import OnnxFastPitchHifiGanBackend
from itantra_tts.adapters.onnx_odia_vits import OnnxOdiaVitsResearchBackend
from itantra_tts.domain import Precision, SynthesisFailure, SynthesisRequest


def main() -> None:
    payload = json.loads(sys.stdin.buffer.read(64_000))
    payload["precision"] = Precision(payload["precision"])
    request = SynthesisRequest(**payload)
    backend = LocalResearchSynthesisBackend(
        OnnxFastPitchHifiGanBackend(Path(sys.argv[1]), max_cached_languages=1,
                                    session_policy="sequential", intra_op_threads=int(sys.argv[4])),
        OnnxOdiaVitsResearchBackend(Path(sys.argv[2]), model_filename=sys.argv[3],
                                    intra_op_threads=int(sys.argv[4])),
    )
    result = backend.synthesize(request)
    response = asdict(result)
    if isinstance(result, SynthesisFailure):
        response["kind"] = "failure"
    else:
        response["kind"] = "audio"
        response["audio"] = base64.b64encode(response.pop("audio_pcm_s16le")).decode("ascii")
    encoded = json.dumps(response).encode("utf-8")
    sys.stdout.buffer.write(len(encoded).to_bytes(4, "big") + encoded)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
