from itantra_tts.domain import Precision, SynthesisRequest
from itantra_tts.ports import SynthesisBackend
from pathlib import Path
from time import perf_counter

import pytest


class RecordingBackend:
    def __init__(self) -> None:
        self.requests: list[SynthesisRequest] = []
        self.clear_calls = 0

    def synthesize(self, request: SynthesisRequest) -> str:
        self.requests.append(request)
        return request.language

    def clear_cache(self) -> None:
        self.clear_calls += 1


def test_local_research_backend_routes_odia_and_releases_the_inactive_runtime() -> None:
    from itantra_tts.adapters.local_research import LocalResearchSynthesisBackend

    onnx = RecordingBackend()
    odia = RecordingBackend()
    backend: SynthesisBackend = LocalResearchSynthesisBackend(
        fastpitch_hifigan=onnx,
        odia_vits=odia,
    )

    hindi = backend.synthesize(SynthesisRequest("test", 0, "नमस्ते", "hi", Precision.INT8))
    assert hindi == "hi"
    assert [request.language for request in onnx.requests] == ["hi"]
    assert odia.clear_calls == 1

    odia_result = backend.synthesize(SynthesisRequest("test", 1, "ନମସ୍କାର", "or", Precision.INT8))
    assert odia_result == "or"
    assert [request.language for request in odia.requests] == ["or"]
    assert onnx.clear_calls == 1


def test_isolated_backend_marks_memory_as_worker_scoped() -> None:
    from itantra_tts.adapters.isolated_local import IsolatedLocalResearchSynthesisBackend

    backend = IsolatedLocalResearchSynthesisBackend.__new__(IsolatedLocalResearchSynthesisBackend)

    assert backend.memory_scope == "isolated synthesis worker process"


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "models/artifacts/mms-tts-ory-int8-research/model.static-qdq.int8.onnx").exists(),
    reason="static Odia research artifact absent",
)
def test_isolated_latency_includes_worker_start_and_ipc() -> None:
    from itantra_tts.adapters.isolated_local import IsolatedLocalResearchSynthesisBackend

    artifacts = Path(__file__).resolve().parents[2] / "models/artifacts"
    backend = IsolatedLocalResearchSynthesisBackend(
        artifacts / "echobharat-int8",
        artifacts / "mms-tts-ory-int8-research",
        odia_model_filename="model.static-qdq.int8.onnx",
    )
    started = perf_counter()
    result = backend.synthesize(SynthesisRequest("test", 0, "ନମସ୍କାର", "or", Precision.INT8))
    observed_ms = (perf_counter() - started) * 1000

    assert result.metrics is not None
    assert result.metrics.wall_latency_ms >= observed_ms - 5
    assert result.metrics.process_cpu_utilization_percent == pytest.approx(
        result.metrics.process_cpu_time_ms / result.metrics.wall_latency_ms * 100,
        rel=0.001,
    )
