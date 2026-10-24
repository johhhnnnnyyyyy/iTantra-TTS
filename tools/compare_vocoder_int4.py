"""Desktop-only comparison of an experimental HiFi-GAN graph with INT8.

The same FastPitch mel tensor feeds both vocoders. This does not replace a
native-speaker listening review or an isolated Android device benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter
import wave

import numpy as np
import onnxruntime as ort

from itantra_tts.adapters.onnx_fastpitch_hifigan import OnnxFastPitchHifiGanBackend
from itantra_tts.adapters.ort_runtime import make_session_options


def compare(
    artifact_root: Path, candidate: Path, text: str, repetitions: int = 3,
    reference: Path | None = None, wav_dir: Path | None = None,
) -> dict:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    backend = OnnxFastPitchHifiGanBackend(artifact_root)
    loaded = backend._load_language_model("hi", include_hifigan=False)
    tokens = np.asarray([backend._tokenize(text, loaded)], dtype=np.int64)
    mel = loaded.fastpitch.run(None, {"text": tokens})[0]
    if mel.size == 0 or not np.isfinite(mel).all():
        raise ValueError("FastPitch returned invalid mel")

    paths = {
        "reference": reference or artifact_root / "hi" / "hifigan-hi.int8.onnx",
        "candidate": candidate,
    }
    results = {}
    waveforms = {}
    for label, path in paths.items():
        session = ort.InferenceSession(
            str(path), sess_options=make_session_options(4), providers=["CPUExecutionProvider"]
        )
        session.run(None, {"mel": mel})
        times = []
        for _ in range(repetitions):
            start = perf_counter()
            waveform = np.asarray(session.run(None, {"mel": mel})[0], dtype=np.float32).reshape(-1)
            times.append(perf_counter() - start)
        if waveform.size == 0 or not np.isfinite(waveform).all():
            raise ValueError(f"{label} returned invalid audio")
        waveforms[label] = waveform
        if wav_dir is not None:
            wav_dir.mkdir(parents=True, exist_ok=True)
            wav_path = wav_dir / f"{label}.wav"
            pcm = np.rint(np.clip(waveform, -1.0, 1.0) * 32767).astype("<i2")
            with wave.open(str(wav_path), "wb") as file:
                file.setnchannels(1)
                file.setsampwidth(2)
                file.setframerate(22050)
                file.writeframes(pcm.tobytes())
        results[label] = {
            "graph_bytes": path.stat().st_size,
            "vocoder_median_ms": median(times) * 1000,
            "audio_duration_ms": waveform.size / 22050 * 1000,
            "vocoder_rtf": median(times) / (waveform.size / 22050),
            "samples": int(waveform.size),
        }

    baseline_waveform, proposed = waveforms["reference"], waveforms["candidate"]
    if baseline_waveform.shape != proposed.shape:
        raise ValueError("candidate produced a different number of audio samples")
    rms = np.sqrt(np.mean(np.square(baseline_waveform, dtype=np.float64)))
    results["candidate_vs_reference"] = {
        "relative_waveform_rmse": float(
            np.sqrt(np.mean(np.square(baseline_waveform - proposed, dtype=np.float64))) / rms
        ) if rms else None,
        "waveform_correlation": float(np.corrcoef(baseline_waveform, proposed)[0, 1]),
    }
    results["scope"] = "desktop CPU, vocoder-only, same INT8 FastPitch mel, warm sessions"
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--artifact-root", type=Path, default=Path("models/artifacts/echobharat-int8"))
    parser.add_argument("--text", default="नमस्ते।")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--reference", type=Path, help="default: existing Hindi INT8 vocoder")
    parser.add_argument("--wav-dir", type=Path, help="optional local listening files")
    args = parser.parse_args()
    print(json.dumps(compare(
        args.artifact_root, args.candidate, args.text, args.repetitions,
        args.reference, args.wav_dir,
    ), indent=2))


if __name__ == "__main__":
    main()
