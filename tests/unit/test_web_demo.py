import asyncio
from pathlib import Path

import pytest

from itantra_tts.adapters import FakeSynthesisBackend
from itantra_tts.domain import (
    ModelCatalog,
    Precision,
    SynthesisMetrics,
    SynthesisRequest,
    SynthesisResult,
)
from itantra_tts.web.demo import WebSynthesisController, build_demo


ROOT = Path(__file__).resolve().parents[2]


def make_controller() -> WebSynthesisController:
    return WebSynthesisController(
        catalog=ModelCatalog.from_yaml(ROOT / "models" / "registry.yaml"),
        backend=FakeSynthesisBackend(),
    )


def test_web_controller_exposes_truthful_unavailable_int4_state() -> None:
    view = make_controller().submit(
        text="Please meet at 5:30.", language="en", precision=Precision.INT4.value
    )

    assert view.audio is None
    assert "not executable" in view.status.lower()
    assert "INT4" in view.provenance
    assert "pending_int4_conversion" in view.provenance
    assert "Please meet at 5:30." not in view.status


def test_web_controller_shows_odia_research_only_provenance() -> None:
    view = make_controller().submit(text="ନମସ୍କାର", language="or", precision=Precision.INT8.value)

    assert view.audio is None
    assert "research_only" in view.provenance
    assert "CC-BY-NC-4.0" in view.provenance


def test_web_provenance_shows_source_without_internal_revision() -> None:
    view = make_controller().submit(text="hello", language="en", precision=Precision.INT8.value)

    assert "RaunakSaha/echobharat-models" in view.provenance
    assert "2ad314c3ecafeb252bac82785ac541325596ac9c" not in view.provenance


def test_web_demo_has_typed_text_and_all_language_choices() -> None:
    demo = build_demo(make_controller())
    config = demo.get_config_file()
    label_order = [component.get("props", {}).get("label") for component in config["components"]]
    labels = set(label_order)

    assert "Text to synthesize" in labels
    assert "Text sent to TTS" in labels
    assert "Normalization status" in labels
    assert label_order.index("Text sent to TTS") < label_order.index("Generated local audio")
    assert "Target language" in labels
    assert "Precision" in labels
    audio_player = next(
        component
        for component in config["components"]
        if component.get("props", {}).get("label") == "Generated local audio"
    )
    assert audio_player["props"]["buttons"] == []
    language_picker = next(
        component
        for component in config["components"]
        if component.get("props", {}).get("label") == "Target language"
    )
    assert {choice[0] for choice in language_picker["props"]["choices"]} == {
        "Bengali (bn)",
        "English (en)",
        "Gujarati (gu)",
        "Hindi (hi)",
        "Kannada (kn)",
        "Malayalam (ml)",
        "Marathi (mr)",
        "Odia (or)",
        "Tamil (ta)",
        "Telugu (te)",
    }


class SuccessfulBackend:
    """Minimal local-result stand-in proving the player receives NumPy-ready audio."""

    def __init__(self, model_id: str = "echobharat-hi-int8") -> None:
        self.model_id = model_id
        self.requests: list[SynthesisRequest] = []

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        self.requests.append(request)
        return SynthesisResult(
            stream_id=request.stream_id,
            sequence_no=request.sequence_no,
            language=request.language,
            precision=request.precision,
            model_id=self.model_id,
            sample_rate_hz=22_050,
            audio_pcm_s16le=b"\x00\x00\xff\x7f",
            metrics=SynthesisMetrics(
                wall_latency_ms=12.5,
                stage_one_label="FastPitch",
                fastpitch_latency_ms=4.0,
                stage_two_label="HiFi-GAN",
                hifigan_latency_ms=8.0,
                process_cpu_time_ms=10.0,
                process_cpu_utilization_percent=80.0,
                rss_before_bytes=1_000,
                rss_after_bytes=1_500,
                selected_model_rss_delta_bytes=500,
                peak_rss_bytes=2_000,
                audio_duration_ms=0.1,
                real_time_factor=125.0,
                model_disk_bytes=3_000,
                provider="CPUExecutionProvider",
                runtime_version="test",
                cold_start=True,
                resident_model_count=1,
                memory_scope="whole demo process",
            ),
        )


def test_web_controller_returns_browser_playable_numpy_audio_and_measured_metrics() -> None:
    controller = WebSynthesisController(
        catalog=ModelCatalog.from_yaml(ROOT / "models" / "registry.yaml"),
        backend=SuccessfulBackend(),
    )

    view = controller.submit(text="hello", language="en", precision=Precision.INT8.value)

    assert view.audio is not None
    sample_rate, samples = view.audio
    assert sample_rate == 22_050
    assert samples.dtype.name == "int16"
    assert samples.tolist() == [0, 32_767]
    assert "Wall latency" in view.metrics
    assert "Model disk footprint" in view.metrics


def test_web_controller_labels_static_odia_candidate_as_research() -> None:
    controller = WebSynthesisController(
        catalog=ModelCatalog.from_yaml(ROOT / "models" / "registry.yaml"),
        backend=SuccessfulBackend("mms-tts-ory-onnx-static-qdq-int8-research"),
    )

    view = controller.submit(text="ନମସ୍କାର", language="or", precision=Precision.INT8.value)

    assert "static QDQ" in view.status
    assert "mms-tts-ory-onnx-static-qdq-int8-research" in view.provenance


@pytest.mark.parametrize(
    ("language", "original", "spoken"),
    [
        ("en", "Meet me at 5:30.", "Meet me at five thirty."),
        ("hi", "बैठक 5:30 है।", "बैठक पांच बजकर तीस मिनट है।"),
        ("en", "Bring ₹200 and 10%!", "Bring two hundred rupees and ten percent!"),
        ("hi", "₹200 और 10% लाना।", "दो सौ रुपये और दस प्रतिशत लाना।"),
    ],
)
def test_web_synthesizes_the_reviewed_normalized_text(
    language: str, original: str, spoken: str
) -> None:
    backend = SuccessfulBackend()
    controller = WebSynthesisController(
        catalog=ModelCatalog.from_yaml(ROOT / "models" / "registry.yaml"), backend=backend
    )

    view = controller.submit(text=original, language=language, precision=Precision.INT8.value)

    assert backend.requests[0].text == spoken
    assert view.spoken_text == spoken
    assert "w2-v1" in view.normalization_status


def test_web_exposes_unsafe_form_warning_without_silent_expansion() -> None:
    backend = SuccessfulBackend()
    controller = WebSynthesisController(
        catalog=ModelCatalog.from_yaml(ROOT / "models" / "registry.yaml"), backend=backend
    )

    view = controller.submit(
        text="Price ₹200.50 on 05/06/2026.", language="en", precision=Precision.INT8.value
    )

    assert backend.requests[0].text == "Price ₹200.50 on 05/06/2026."
    assert view.spoken_text == backend.requests[0].text
    assert "UNSAFE_DECIMAL_CURRENCY_LEFT_UNCHANGED" in view.normalization_status
    assert "AMBIGUOUS_DATE_LEFT_UNCHANGED" in view.normalization_status


def test_web_labels_unsupported_normalization_language_as_passthrough() -> None:
    backend = SuccessfulBackend()
    controller = WebSynthesisController(
        catalog=ModelCatalog.from_yaml(ROOT / "models" / "registry.yaml"), backend=backend
    )

    view = controller.submit(text="નમસ્તે.", language="gu", precision=Precision.INT8.value)

    assert backend.requests[0].text == "નમસ્તે."
    assert "passed through unchanged" in view.normalization_status


def test_ordered_web_path_normalizes_sentences_and_returns_one_playable_clip() -> None:
    class AsyncBackend:
        async def synthesize(self, request):
            return SuccessfulBackend().synthesize(request)

    controller = WebSynthesisController(
        catalog=ModelCatalog.from_yaml(ROOT / "models" / "registry.yaml"),
        backend=SuccessfulBackend(), ordered_backend=AsyncBackend(),
    )
    view = asyncio.run(controller.submit_ordered("Meet at 5:30. Bring 10%.", "en", "int8"))
    assert view.audio is not None
    assert view.spoken_text == "Meet at five thirty. Bring ten percent."
    assert "2 sentence" in view.status
    assert len(view.audio[1]) == 4 + 661
    assert "per-sentence" in view.metrics.lower()
    assert "echobharat-hi-int8" in view.provenance
