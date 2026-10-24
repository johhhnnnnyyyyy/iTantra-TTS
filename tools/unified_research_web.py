"""Local interface for shared-service INT8 and W4A8 synthesis."""
from __future__ import annotations

import argparse
import asyncio
import io
import wave
from pathlib import Path
from time import perf_counter

import numpy as np

from itantra_tts.adapters.async_local import AsyncLocalResearchSynthesisBackend
from itantra_tts.adapters.w4a8_research import W4A8ResearchBackend
from itantra_tts.application.service import SynthesisService, SynthesisServiceError
from itantra_tts.domain import Precision, SynthesisRequest

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "ta", "te")
LANGUAGE_CHOICES = [("Bengali", "bn"), ("English", "en"), ("Gujarati", "gu"),
                    ("Hindi", "hi"), ("Kannada", "kn"), ("Malayalam", "ml"),
                    ("Marathi", "mr"), ("Odia", "or"), ("Tamil", "ta"), ("Telugu", "te")]

DEMO_CSS = """
.gradio-container { max-width: none !important; width: 100% !important; margin: 0 !important; padding: 0 clamp(16px, 2.5vw, 48px) !important; background: #f5f7fa !important; }
.gradio-container {
  --body-text-color: #2E4365; --body-text-color-subdued: #596b82;
  --background-fill-primary: #f5f7fa; --background-fill-secondary: #edf1f6;
  --block-background-fill: #ffffff; --block-border-color: #dce3eb;
  --block-label-background-fill: #ffffff; --block-label-text-color: #2E4365;
  --block-title-text-color: #2E4365; --block-info-text-color: #596b82;
  --input-background-fill: #ffffff; --input-background-fill-hover: #ffffff;
  --input-background-fill-focus: #ffffff; --input-border-color: #cad5e2;
  --input-border-color-hover: #8294aa; --input-border-color-focus: #E59D2C;
  --input-placeholder-color: #78899b; --accordion-text-color: #2E4365;
  --panel-background-fill: #ffffff; --panel-border-color: #dce3eb;
  --border-color-primary: #dce3eb; --border-color-accent: #E59D2C;
  --button-secondary-background-fill: #edf1f6; --button-secondary-text-color: #2E4365;
  --button-secondary-background-fill-hover: #e1e8f0; --button-secondary-text-color-hover: #1e304c;
}
.lab-header { padding: 24px 0 12px; border-bottom: 3px solid #E59D2C; }
.lab-header h1 { color: #2E4365; font-size: clamp(28px, 3vw, 42px) !important; letter-spacing: -.035em; margin: 5px 0 !important; }
.lab-header p { color: #596b82; }
.lab-header h1, .lab-header p { display: block; }
.eyebrow { color: #2E4365; font-size: 12px; letter-spacing: .16em; font-weight: 700; text-transform: uppercase; }
.lab-section { background: #fff !important; border: 1px solid #dce3eb !important; border-radius: 16px !important; padding: 22px !important; }
.lab-section h2 { font-size: 20px !important; margin: 0 !important; color: #2E4365; }
.section-hint { color: #596b82; font-size: 14px; }
.primary-action { background: #2E4365 !important; color: white !important; border: none !important; min-height: 48px; }
.primary-action:hover { background: #1e304c !important; }
.detail-panel { margin-top: 10px; }
.research-note { color: #596b82; font-size: 13px; border-top: 1px solid #dce3eb; padding-top: 18px; }
.research-note * { color: #596b82 !important; }
:focus-visible { outline: 3px solid #E59D2C !important; outline-offset: 2px; }
@media (max-width: 700px) { .lab-section { padding: 16px !important; } .lab-header { padding-top: 12px; } }
"""


async def synthesize(text: str, language: str, precision: str, library: Path, *, threads: int = 2):
    """Use the common service, exposing only same-request metrics."""
    try:
        if not isinstance(text, str) or not text.strip() or len(text) > 200:
            raise ValueError("invalid text")
        selected = Precision(precision)
        if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 8:
            raise ValueError("invalid thread count")
        request = SynthesisRequest("unified-research", 0, text, language, selected)
    except (TypeError, ValueError):
        return None, "Invalid input (1–200 characters and supported language required).", "", "", "Not measured."
    if selected is Precision.INT4:
        backend = W4A8ResearchBackend(library, threads=threads)
    else:
        backend = AsyncLocalResearchSynthesisBackend(
            ROOT / "models/artifacts/echobharat-int8",
            ROOT / "models/artifacts/mms-tts-ory-int8-research",
            intra_op_threads=threads,
        )
    service = SynthesisService(backend, max_chars=200, allow_experimental_int4=selected is Precision.INT4)
    started = perf_counter()
    try:
        output = await service.synthesize(request)
        elapsed_ms = (perf_counter() - started) * 1000
        with wave.open(io.BytesIO(output.wav_bytes), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise ValueError("invalid audio")
            audio = (wav.getframerate(), np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").copy())
    except (OSError, ValueError, SynthesisServiceError, wave.Error):
        return None, "Local synthesis failed or is unavailable.", "", "", "Not measured."
    worker_lines = []
    for index, metric in enumerate(output.worker_metrics):
        prefix = f"Sentence {index + 1}"
        if metric is None:
            worker_lines.append(f"{prefix}: worker metrics not measured.")
        elif "process_cpu_time_ms" in metric:
            worker_lines.append(
                f"{prefix}: worker wall {metric['wall_latency_ms']:.1f} ms; "
                f"CPU time {metric['process_cpu_time_ms']:.1f} ms "
                f"({metric['process_cpu_utilization_percent']:.1f}% of one core, aggregate); "
                f"peak RSS {metric['peak_rss_bytes'] / 1048576:.1f} MiB.\n"
                f"  {metric['stage_one_label']}: {metric['fastpitch_latency_ms']:.1f} ms; "
                f"{metric['stage_two_label']}: {metric['hifigan_latency_ms']:.1f} ms.\n"
                f"  Model files: {metric['model_disk_bytes'] / 1048576:.1f} MiB; "
                f"ORT {metric['runtime_version']} / {metric['provider']}; "
                f"cold start: {metric['cold_start']}; memory scope: {metric['memory_scope']}; "
                f"ORT intra-op threads per session: {threads}.")
        else:
            worker_wall = float(metric['wall_ms'])
            utilization = (100 * float(metric['cpu_time_ms']) / worker_wall) if worker_wall > 0 else None
            cpu_detail = f"{utilization:.1f}% of one core, aggregate" if utilization is not None else "not measured"
            worker_lines.append(f"{prefix}: worker wall {worker_wall:.1f} ms; "
                                f"CPU time {metric['cpu_time_ms']:.1f} ms ({cpu_detail}); "
                                f"peak RSS {metric['peak_rss_bytes'] / 1048576:.1f} MiB.\n"
                                f"  ORT {metric['runtime_version']}; memory scope {metric['memory_scope']}; "
                                f"INT4 threads {threads}; per-stage timing: not measured.")
    metrics = (f"Desktop total wall including worker startup: {elapsed_ms:.1f} ms\n"
               f"Audio: {output.audio_duration_ms:.1f} ms; sample rate: {audio[0]} Hz; "
               f"total RTF: {elapsed_ms / output.audio_duration_ms:.3f}\n"
               f"Executed model: {', '.join(output.model_ids)}\n"
               "CPU time is summed across worker threads; it is not elapsed time. "
               "Peak RSS is worker-process memory, not model weights alone.\n"
               + "\n".join(worker_lines))
    status = ("W4A8 audio generated locally." if selected is Precision.INT4
              else "INT8 audio generated locally.")
    normalization = output.normalization.replace(
        "Experimental cardinal rules; native-speaker review pending.",
        "Indic-language number normalization applied.",
    )
    return audio, status, output.spoken_text, normalization, metrics


def build_demo(library: Path, *, hosted: bool = False):
    """Build the shared INT8/W4A8 interface without loading a model session."""
    import gradio as gr

    with gr.Blocks(title="iTantra | Speech research lab") as demo:
        gr.Markdown("<div class='eyebrow'>ITANTRA · SPEECH RESEARCH</div>"
                    "<h1>A voice for every message</h1>"
                    f"<p>Explore multilingual text-to-speech with "
                    f"{'server' if hosted else 'local'} inference.</p>",
                    elem_classes="lab-header")
        if not library.is_file():
            gr.Markdown("**INT4 setup required:** provide the custom W4A8 kernel library "
                        "to enable the INT4 selection.",
                        elem_classes="research-note")
        with gr.Row(equal_height=False):
            with gr.Column(scale=6, min_width=300, elem_classes="lab-section"):
                gr.Markdown("## Create speech\n"
                            "<span class='section-hint'>Enter text in the language you want to hear.</span>")
                text = gr.Textbox(label="Text to speak", lines=5, max_lines=8,
                                  placeholder="Type or paste up to 200 characters…")
                with gr.Row():
                    language = gr.Dropdown(choices=LANGUAGE_CHOICES, value="hi", label="Language")
                    precision = gr.Dropdown(choices=["int8", "int4"], value="int8", label="Precision")
                with gr.Accordion("Advanced settings", open=False):
                    threads = gr.Dropdown(choices=[1, 2, 4, 8], value=4,
                                          label="Inference threads",
                                          info="Maximum ONNX Runtime intra-operation threads per model session, for either precision. More threads may reduce latency but increase CPU use.")
                button = gr.Button("Generate speech", variant="primary", elem_classes="primary-action")
            with gr.Column(scale=5, min_width=300, elem_classes="lab-section"):
                gr.Markdown("## Listen & inspect\n"
                            "<span class='section-hint'>Play the generated audio here; no download required.</span>")
                audio = gr.Audio(label="Generated audio", type="numpy")
                status = gr.Textbox(label="Status", value="Ready to synthesize.", interactive=False)
                spoken = gr.Textbox(label="Text sent to synthesis", interactive=False)
                normalization = gr.Textbox(label="Normalization and warnings", interactive=False)
        with gr.Accordion("Request metrics · measured on this host", open=True, elem_classes="detail-panel"):
            gr.Markdown("Values are measured for this request on the machine running the model. "
                        "W4A8 uses packed INT4 weights for eligible operators.")
            metrics = gr.Textbox(label="Measured metrics", value="Not measured yet.", lines=5, interactive=False)
        gr.Markdown("Offline multilingual speech research · local inference · "
                    "request-scoped metrics · MIT project code · Odia: CC-BY-NC-4.0 · "
                    "Use ordinary demonstration text.",
                    elem_classes="research-note")

        async def submit(t, lang, p, n):
            return await synthesize(t, lang, p, library, threads=n)

        button.click(submit, [text, language, precision, threads], [audio, status, spoken, normalization, metrics],
                     concurrency_limit=1)
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18082)
    args = parser.parse_args()
    build_demo(args.library).queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=args.port, show_error=False, share=False, css=DEMO_CSS
    )


if __name__ == "__main__":
    main()
