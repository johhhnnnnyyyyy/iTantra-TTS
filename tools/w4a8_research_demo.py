"""Generate a WAV with experimental scalar W4A8 models; not for emergencies."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import wave
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "ta", "te")


def select_model_dir(artifacts: Path, language: str) -> Path:
    """Require the channel-scaled EchoBharat pack; preserve the Odia VITS pack."""
    variant = "experimental-w4a8-scalar-20261021" if language == "or" else "experimental-w4a8-per-channel"
    selected = artifacts / variant / language
    if not selected.is_dir():
        raise FileNotFoundError("selected W4A8 model pack is unavailable")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=LANGUAGES, required=True)
    text_input = parser.add_mutually_exclusive_group(required=True)
    text_input.add_argument("--text", help="Already in the spoken language; not normalized.")
    text_input.add_argument("--text-stdin", action="store_true", help="Read text from standard input")
    parser.add_argument("--library", type=Path, required=True, help="Matching ONNX Runtime custom-op library")
    parser.add_argument("--output", type=Path, required=True, help="New WAV path; existing files are not replaced")
    parser.add_argument("--threads", type=int, default=2, help="ORT intra-op threads (1–8)")
    args = parser.parse_args()
    if args.text_stdin:
        args.text = sys.stdin.read(201)
    if not 1 <= args.threads <= 8:
        parser.error("threads must be between 1 and 8")
    if not args.text.strip() or len(args.text) > 200:
        parser.error("text must contain 1–200 characters")
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists")
    if not args.library.is_file():
        parser.error("custom operator library not found")

    import numpy as np
    import onnxruntime as ort

    from itantra_tts.adapters.odia_tokenizer import OdiaMmsTokenizer
    from itantra_tts.adapters.onnx_fastpitch_hifigan import OnnxFastPitchHifiGanBackend

    artifacts = ROOT / "models/artifacts"
    experimental = select_model_dir(artifacts, args.language)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.register_custom_ops_library(str(args.library.resolve()))
    start = time.perf_counter()
    cpu_start = time.process_time()
    if args.language == "or":
        vocab = artifacts / "mms-tts-ory-int8-research/vocab.json"
        input_ids = OdiaMmsTokenizer(vocab).encode(args.text).astype(np.int64)
        session = ort.InferenceSession(str(experimental / "model.w4a8.onnx"), options,
                                       providers=["CPUExecutionProvider"])
        waveform = session.run(["waveform"], {"input_ids": input_ids})[0]
        rate = 16000
    else:
        language = args.language
        token_file = artifacts / "echobharat-int8" / language / f"fastpitch-{language}.tokens.json"
        tokens = json.loads(token_file.read_text(encoding="utf-8"))
        token_ids = {}
        for index, token in enumerate(tokens):
            token_ids.setdefault(token, index)
        symbols = tuple(sorted((s for s in token_ids if not s.startswith("<")), key=len, reverse=True))
        metadata = SimpleNamespace(token_ids=token_ids, symbols_by_longest_match=symbols)
        ids = OnnxFastPitchHifiGanBackend._tokenize(args.text, metadata)
        fastpitch = ort.InferenceSession(str(experimental / f"fastpitch-{language}.w4a8.onnx"), options,
                                         providers=["CPUExecutionProvider"])
        mel = fastpitch.run(None, {"text": np.asarray([ids], dtype=np.int64)})[0]
        hifigan = ort.InferenceSession(str(experimental / f"hifigan-{language}.w4a8.onnx"), options,
                                       providers=["CPUExecutionProvider"])
        waveform = hifigan.run(None, {"mel": mel})[0]
        rate = 22050
    mono = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if not mono.size or not np.isfinite(mono).all() or not np.any(mono):
        raise ValueError("model produced invalid or silent audio")
    pcm = np.rint(np.clip(mono, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    with args.output.open("xb") as target:
        with wave.open(target, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(rate)
            output.writeframes(pcm)
    elapsed = time.perf_counter() - start
    cpu = time.process_time() - cpu_start
    duration = mono.size / rate
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
    print(json.dumps({"language": args.language, "execution_precision": "W4A8 mixed",
                      "runtime": ort.__version__, "provider": "CPUExecutionProvider",
                      "intra_op_threads": args.threads,
                      "wall_ms": round(elapsed * 1000, 1), "cpu_time_ms": round(cpu * 1000, 1),
                      "audio_duration_ms": round(duration * 1000, 1),
                      "cold_rtf": round(elapsed / duration, 3), "process_peak_rss_bytes": peak_bytes,
                      "metrics_scope": "whole cold desktop process, not Android; includes file output",
                      "normalization": "not applied", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
