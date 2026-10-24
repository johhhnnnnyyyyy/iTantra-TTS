"""Local-only experimental W4A8 browser demo; not suitable for emergencies."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "ta", "te")


def synthesize(text: str, language: str, library: Path):
    """Run a bounded isolated worker; return its real WAV and same-request metrics."""
    if not isinstance(text, str) or not text.strip() or len(text) > 200 or language not in LANGUAGES:
        return None, "Invalid text or language (1–200 characters required).", "Not measured."
    if not library.is_file():
        return None, "Experimental operator library is unavailable.", "Not measured."
    try:
        with tempfile.TemporaryDirectory(prefix="itantra-w4-web-") as temp:
            output = Path(temp) / "speech.wav"
            run = subprocess.run(
                [sys.executable, str(ROOT / "tools/w4a8_research_demo.py"), "--language", language,
                 "--text-stdin", "--library", str(library), "--output", str(output)],
                input=text, text=True, capture_output=True, timeout=300, cwd=ROOT, check=True,
            )
            metrics = json.loads(run.stdout)
            with wave.open(str(output), "rb") as wav:
                if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                    raise ValueError("invalid audio format")
                rate = wav.getframerate()
                samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").copy()
            if not samples.size:
                raise ValueError("empty audio")
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError, wave.Error):
        return None, "Experimental synthesis failed; no audio was returned.", "Not measured."
    report = (f"W4A8 mixed; ORT {metrics['runtime']} / {metrics['provider']}\n"
              f"Cold wall: {metrics['wall_ms']} ms; CPU time: {metrics['cpu_time_ms']} ms\n"
              f"Audio: {metrics['audio_duration_ms']} ms; cold RTF: {metrics['cold_rtf']}\n"
              f"Whole worker peak RSS: {metrics['process_peak_rss_bytes'] / 1048576:.1f} MiB\n"
              "Scope: cold desktop worker, excludes worker startup and IPC; no Android measurement. "
              "Normalization not applied.")
    return (rate, samples), "Audio generated locally; play it below.", report


def main():
    import gradio as gr

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18081)
    args = parser.parse_args()
    if not args.library.is_file():
        parser.error("custom operator library not found")
    with gr.Blocks(title="iTantra experimental W4A8 research") as demo:
        gr.Markdown("# Experimental INT4 (W4A8) TTS research\n"
                    "Local desktop inference, not an Android result. Not for emergencies. "
                    "Speech quality and performance are unvalidated; synthesis may take minutes. "
                    "Text is not normalized. Odia model: CC-BY-NC-4.0. Gradio may cache audio "
                    "temporarily on this machine. Do not enter sensitive text.")
        text = gr.Textbox(label="Text in the selected language", max_lines=6)
        language = gr.Dropdown(choices=list(LANGUAGES), value="hi", label="Language")
        button = gr.Button("Synthesize experimental W4A8")
        status = gr.Textbox(label="Status", interactive=False)
        audio = gr.Audio(label="Generated audio", type="numpy")
        metrics = gr.Textbox(label="Measured metrics", lines=5, interactive=False)
        button.click(lambda t, lang: synthesize(t, lang, args.library), [text, language],
                     [audio, status, metrics], concurrency_limit=1)
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=args.port,
                                                  show_error=False, share=False)


if __name__ == "__main__":
    main()
