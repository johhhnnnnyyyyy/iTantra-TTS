"""Truthful local Gradio demonstration for the portable synthesis contract."""

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import gradio as gr
import numpy as np

from itantra_tts.adapters.isolated_local import IsolatedLocalResearchSynthesisBackend
from itantra_tts.adapters.async_local import AsyncLocalResearchSynthesisBackend
from itantra_tts.application.sentence_playback import SentenceSynthesisError, synthesize_sentences
from itantra_tts.domain import (
    ModelCatalog,
    ModelProfile,
    Precision,
    SynthesisFailure,
    SynthesisRequest,
)
from itantra_tts.normalization import SpeechNormalizer
from itantra_tts.ports import SynthesisBackend


LANGUAGE_CHOICES = (
    ("Bengali (bn)", "bn"),
    ("English (en)", "en"),
    ("Gujarati (gu)", "gu"),
    ("Hindi (hi)", "hi"),
    ("Kannada (kn)", "kn"),
    ("Malayalam (ml)", "ml"),
    ("Marathi (mr)", "mr"),
    ("Odia (or)", "or"),
    ("Tamil (ta)", "ta"),
    ("Telugu (te)", "te"),
)


@dataclass(frozen=True, slots=True)
class WebSynthesisView:
    """Safe, display-ready output from one demo request."""

    audio: Any | None
    status: str
    provenance: str
    metrics: str
    spoken_text: str = ""
    normalization_status: str = "Normalization not applied."


class WebSynthesisController:
    """Turn browser controls into the canonical synthesis request without retaining text."""

    def __init__(
        self,
        catalog: ModelCatalog,
        backend: SynthesisBackend,
        normalizer: SpeechNormalizer | None = None,
        ordered_backend: Any | None = None,
    ) -> None:
        self._catalog = catalog
        self._backend = backend
        self._normalizer = normalizer or SpeechNormalizer()
        self._ordered_backend = ordered_backend

    async def submit_ordered(self, text: str, language: str, precision: str) -> WebSynthesisView:
        """Produce a complete ordered clip, retaining each worker's actual metrics."""
        try:
            selected = Precision(precision)
            request = SynthesisRequest("web-ordered", 0, text, language, selected)
            if len(text) > 1000:
                raise ValueError("text too long")
        except (TypeError, ValueError):
            return WebSynthesisView(None, "Invalid input.", "No model selected.", "Not measured.")
        profile = self._profile_for(language, selected)
        if profile is None or self._ordered_backend is None:
            return WebSynthesisView(None, "Ordered synthesis is unavailable.", "No model selected.", "Not measured.")
        if language in {"en", "hi"}:
            normalized = self._normalizer.normalize(request.text, language)
            sentences = normalized.sentences
            spoken = normalized.normalized_text
            warnings = ", ".join(w.code for w in normalized.warnings)
            normalization_status = f"Rule {normalized.rule_version} applied."
            if warnings:
                normalization_status += f" Unresolved forms: {warnings}."
        else:
            spoken = text.strip()
            sentences = tuple(part for part in re.split(r"(?<=[.!?।])\s+", spoken) if part)
            normalization_status = f"No W2 normalizer for {language}; text passed through unchanged."
        started = perf_counter()
        try:
            audio = await synthesize_sentences(self._ordered_backend.synthesize, "web-ordered",
                                               sentences, language, selected)
        except (ValueError, BufferError, SentenceSynthesisError):
            return WebSynthesisView(None, "Ordered synthesis failed; no partial audio was played.",
                                    self._provenance(profile), "Not measured: ordered synthesis did not complete.",
                                    spoken, normalization_status)
        elapsed_ms = (perf_counter() - started) * 1000
        duration_ms = len(audio.audio_pcm_s16le) / (2 * audio.sample_rate_hz) * 1000
        metrics = (f"### Ordered request metrics\n- Total elapsed: `{elapsed_ms:.1f} ms`"
                   f"\n- Stitched audio duration / RTF: `{duration_ms:.1f} ms` / `{elapsed_ms / duration_ms:.3f}x`"
                   "\n- Per-sentence measurements below are from their individual local workers; "
                   "wall latency excludes process startup and IPC.\n\n"
                   + "\n\n".join(self._metrics(item.metrics) for item in audio.results))
        samples = np.frombuffer(audio.audio_pcm_s16le, dtype="<i2").copy()
        model_ids = ", ".join(sorted({item.model_id for item in audio.results}))
        return WebSynthesisView((audio.sample_rate_hz, samples),
                                f"{len(audio.results)} sentence(s) generated locally in order; play below.",
                                self._provenance(profile) + f"\n- Executed model(s): `{model_ids}`",
                                metrics, spoken, normalization_status)

    def submit(self, text: str, language: str, precision: str) -> WebSynthesisView:
        """Return audio only when the selected backend actually produced it."""
        try:
            selected_precision = Precision(precision)
            request = SynthesisRequest(
                stream_id="web-demo",
                sequence_no=0,
                text=text,
                language=language,
                precision=selected_precision,
            )
        except (TypeError, ValueError):
            return WebSynthesisView(
                audio=None,
                status=(
                    "Input is not ready for synthesis. "
                    "Enter non-empty text and choose a supported option."
                ),
                provenance="No model was selected.",
                metrics="### Run metrics\nNot measured: input was invalid.",
            )

        profile = self._profile_for(request.language, request.precision)
        if profile is None:
            return WebSynthesisView(
                audio=None,
                status="The requested model profile is not registered.",
                provenance="No model was selected.",
                metrics="### Run metrics\nNot measured: no model profile was selected.",
            )

        if request.language in {"en", "hi"}:
            normalization = self._normalizer.normalize(request.text, request.language)
            request = replace(request, text=normalization.normalized_text)
            warning_codes = ", ".join(warning.code for warning in normalization.warnings)
            normalization_status = f"Rule {normalization.rule_version} applied."
            if warning_codes:
                normalization_status += f" Unresolved forms: {warning_codes}."
        else:
            normalization_status = (
                f"No W2 normalizer for {request.language}; text passed through unchanged."
            )

        result = self._backend.synthesize(request)
        if isinstance(result, SynthesisFailure):
            return WebSynthesisView(
                audio=None,
                status="Selected model is not executable on this demo. No audio was generated.",
                provenance=self._provenance(profile),
                metrics="### Run metrics\nNot measured: local synthesis did not complete.",
                spoken_text=request.text,
                normalization_status=normalization_status,
            )

        samples = np.frombuffer(result.audio_pcm_s16le, dtype="<i2").copy()
        status = "Audio generated locally. Use the player below to listen immediately; no download is required."
        if result.model_id == "mms-tts-ory-onnx-dynamic-int8-research":
            status += " Odia uses a research-only ONNX dynamic-INT8 VITS candidate."
        elif result.model_id == "mms-tts-ory-onnx-static-qdq-int8-research":
            status += " Odia uses a research-only static QDQ INT8 candidate; listening quality is pending review."
        return WebSynthesisView(
            audio=(result.sample_rate_hz, samples),
            status=status,
            provenance=self._provenance(profile) + f"\n- Executed model: `{result.model_id}`",
            metrics=self._metrics(result.metrics),
            spoken_text=request.text,
            normalization_status=normalization_status,
        )

    def _profile_for(self, language: str, precision: Precision) -> ModelProfile | None:
        return next(
            (
                profile
                for profile in self._catalog.profiles_for(language)
                if profile.precision == precision
            ),
            None,
        )

    @staticmethod
    def _provenance(profile: ModelProfile) -> str:
        lines = [
            "### Selected model profile",
            f"- Language: `{profile.language}`",
            f"- Precision: `{profile.precision.value.upper()}`",
            f"- Availability: `{profile.availability.value}`",
            f"- Executable: `{str(profile.executable).lower()}`",
            f"- Backend: `{profile.backend}`",
            f"- Source: `{profile.source}`",
            f"- Upstream licence: `{profile.upstream_license}`",
        ]
        if profile.availability.value == "research_only":
            lines.append(
                "- Research caveat: Odia uses a derived ONNX dynamic-INT8 graph; validate quality and Android behavior before release."
            )
        return "\n".join(lines)

    @staticmethod
    def _metrics(metrics: Any | None) -> str:
        if metrics is None:
            return "### Run metrics\nNot measured: the backend did not report a local run."
        return "\n".join(
            (
                "### Run metrics — this exact local request",
                f"- Wall latency: `{metrics.wall_latency_ms:.1f} ms`",
                f"- {metrics.stage_one_label} / {metrics.stage_two_label}: `{metrics.fastpitch_latency_ms:.1f} ms` / `{metrics.hifigan_latency_ms:.1f} ms`",
                f"- Process CPU time: `{metrics.process_cpu_time_ms:.1f} ms`",
                f"- Process CPU utilization: `{metrics.process_cpu_utilization_percent:.1f}%` (aggregate across cores)",
                f"- Measured process RSS before / after: `{_format_bytes(metrics.rss_before_bytes)}` / `{_format_bytes(metrics.rss_after_bytes)}`",
                f"- Selected-model RSS change this request: `{_format_bytes(metrics.selected_model_rss_delta_bytes)}`",
                f"- Measured process peak RSS: `{_format_bytes(metrics.peak_rss_bytes)}`",
                f"- Audio duration / real-time factor: `{metrics.audio_duration_ms:.1f} ms` / `{metrics.real_time_factor:.3f}x`",
                f"- Model disk footprint: `{_format_bytes(metrics.model_disk_bytes)}`",
                f"- Runtime: `{metrics.runtime_version}` via `{metrics.provider}`; cold start: `{str(metrics.cold_start).lower()}`",
                f"- Resident model packs: `{metrics.resident_model_count}` (cache cap: one language/runtime family)",
                f"- Memory scope: `{metrics.memory_scope}`",
                "- Scope: desktop CPU baseline, not an Android-device measurement.",
            )
        )


def _format_bytes(value: int) -> str:
    return f"{value / (1024 * 1024):.1f} MiB"


def build_demo(controller: WebSynthesisController, *, hosted: bool = False) -> gr.Blocks:
    """Build the local web adapter without invoking a hosted TTS provider."""
    labels = [label for label, _ in LANGUAGE_CHOICES]
    tag_for_label = dict(LANGUAGE_CHOICES)

    def synthesize(
        text: str, language_label: str, precision: str
    ) -> tuple[Any | None, str, str, str, str, str]:
        view = controller.submit(text, tag_for_label[language_label], precision)
        return (
            view.audio,
            view.status,
            view.provenance,
            view.metrics,
            view.spoken_text,
            view.normalization_status,
        )

    async def synthesize_ordered(
        text: str, language_label: str, precision: str
    ) -> tuple[Any | None, str, str, str, str, str]:
        view = await controller.submit_ordered(text, tag_for_label[language_label], precision)
        return (view.audio, view.status, view.provenance, view.metrics, view.spoken_text,
                view.normalization_status)

    with gr.Blocks(title="iTantra TTS — local evidence demo") as demo:
        if hosted:
            gr.Markdown(
                "# iTantra TTS supplemental research demo\n"
                "Inference runs on this cloud server using the repository's ONNX models, "
                "not browser TTS or a hosted TTS API. This is not the offline Android judging "
                "build. Gradio may retain input briefly in its queue and temporary playback "
                "files on the server. Do not enter sensitive or emergency messages. "
                "Odia is CC-BY-NC-4.0 research-only; INT4 is unavailable."
            )
        else:
            gr.Markdown(
                "# iTantra TTS local evidence demo\n"
                "This runs the repository's selected local backend only. "
                "It does not use browser or cloud TTS. "
                "The generated waveform is playable in this page; no download is required. "
                "Research demo only; do not use for emergencies."
            )
        text = gr.Textbox(
            label="Text to synthesize",
            lines=5,
            max_length=1000,
            placeholder="Type or paste text to test the selected local backend.",
        )
        with gr.Row():
            language = gr.Dropdown(labels, value="English (en)", label="Target language")
            precision = gr.Dropdown(
                [Precision.INT8.value] if hosted else [Precision.INT8.value, Precision.INT4.value],
                value=Precision.INT8.value,
                label="Precision",
            )
        submit = gr.Button("Synthesize locally", variant="primary")
        ordered_submit = gr.Button("Synthesize sentences in order (W3)")
        status = gr.Markdown(
            "Select options and synthesize. Gradio may create temporary playback files on this device."
        )
        spoken_text = gr.Textbox(label="Text sent to TTS", interactive=False)
        normalization_status = gr.Textbox(label="Normalization status", interactive=False)
        with gr.Row():
            audio = gr.Audio(
                label="Generated local audio",
                type="numpy",
                format="wav",
                interactive=False,
                buttons=[],
            )
            provenance = gr.Markdown("No model selected yet.")
        metrics = gr.Markdown("### Run metrics\nNot measured yet.")
        submit.click(
            synthesize,
            inputs=[text, language, precision],
            outputs=[audio, status, provenance, metrics, spoken_text, normalization_status],
        )
        ordered_submit.click(
            synthesize_ordered,
            inputs=[text, language, precision],
            outputs=[audio, status, provenance, metrics, spoken_text, normalization_status],
        )

    return demo.queue(default_concurrency_limit=1, max_size=8)


def create_default_demo(*, hosted: bool = False) -> gr.Blocks:
    """Create the runnable local demo using the repository model registry."""
    project_root = Path(__file__).resolve().parents[3]
    catalog = ModelCatalog.from_yaml(project_root / "models" / "registry.yaml")
    return build_demo(
        WebSynthesisController(
            catalog=catalog,
            backend=IsolatedLocalResearchSynthesisBackend(
                echobharat_root=project_root / "models" / "artifacts" / "echobharat-int8",
                odia_root=project_root / "models" / "artifacts" / "mms-tts-ory-int8-research",
                odia_model_filename=(
                    "model.int8.onnx"
                    if os.environ.get("ITANTRA_TTS_ODIA_RESEARCH_MODEL") == "dynamic"
                    else "model.static-qdq.int8.onnx"
                ),
            ),
            ordered_backend=AsyncLocalResearchSynthesisBackend(
                project_root / "models" / "artifacts" / "echobharat-int8",
                project_root / "models" / "artifacts" / "mms-tts-ory-int8-research",
                "model.int8.onnx" if os.environ.get("ITANTRA_TTS_ODIA_RESEARCH_MODEL") == "dynamic"
                else "model.static-qdq.int8.onnx",
            ),
        ), hosted=hosted,
    )


if __name__ == "__main__":
    port = int(os.environ.get("ITANTRA_TTS_DEMO_PORT", "7860"))
    create_default_demo().launch(server_name="127.0.0.1", server_port=port)
