"""Strict, payload-free summaries of measured TTS benchmark samples."""

from __future__ import annotations

import math
import statistics
from typing import Any

from itantra_tts.domain import SynthesisRequest


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive measurement")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be a positive measurement")
    return number


def _percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percent
    lower = math.floor(rank)
    upper = math.ceil(rank)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def evaluate(data: dict[str, Any]) -> dict[str, Any]:
    """Evaluate real samples; never infer Android acceptance from desktop data."""
    platform = data.get("platform")
    if platform not in {"desktop", "android"}:
        raise ValueError("platform must be desktop or android")
    corpus_id = data.get("corpus_id")
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise ValueError("corpus_id is required")
    corpus_license = data.get("corpus_license")
    if platform == "android" and (not isinstance(corpus_license, str) or not corpus_license.strip()):
        raise ValueError("Android corpus licence is required")
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("at least one measured sample is required")
    if platform == "android":
        device = data.get("device")
        required = ("make", "model", "chipset", "android_build", "ram_mib",
                    "free_storage_mib", "battery_percent", "charging", "thermal_state",
                    "airplane_mode", "runtime_version", "model_checksum", "power_state",
                    "lock_hash", "commit")
        if not isinstance(device, dict) or any(device.get(key) is None for key in required):
            raise ValueError("Android device metadata is incomplete")
        if device["airplane_mode"] is not True:
            raise ValueError("Android measurement requires airplane mode")
    parsed: list[dict[str, Any]] = []
    for row in samples:
        if not isinstance(row, dict) or row.get("language") not in SynthesisRequest.SUPPORTED_LANGUAGES:
            raise ValueError("sample language is not in the required matrix")
        if type(row.get("cold")) is not bool:
            raise ValueError("sample cold state is required")
        wall = _positive(row.get("wall_ms"), "wall_ms")
        audio = _positive(row.get("audio_ms"), "audio_ms")
        rss = _positive(row.get("peak_rss_mib"), "peak_rss_mib")
        first = row.get("first_playable_ms")
        if platform == "android":
            _positive(first, "first_playable_ms")
        parsed.append({"language": row["language"], "cold": row["cold"],
                       "wall_ms": wall, "rtf": wall / audio, "peak_rss_mib": rss,
                       "first_playable_ms": first})
    warm = [row for row in parsed if not row["cold"]]
    first_playable = [float(row["first_playable_ms"]) for row in parsed
                      if row["first_playable_ms"] is not None]
    return {
        "platform": platform, "corpus_id": corpus_id, "sample_count": len(parsed),
        "corpus_license": corpus_license,
        "languages_measured": sorted({row["language"] for row in parsed}),
        "warm_sample_count": len(warm),
        "warm_median_rtf": statistics.median(row["rtf"] for row in warm) if warm else None,
        "warm_p95_wall_ms": _percentile([row["wall_ms"] for row in warm], .95) if warm else None,
        "first_playable_median_ms": statistics.median(first_playable) if first_playable else None,
        "first_playable_p95_ms": _percentile(first_playable, .95) if first_playable else None,
        "peak_rss_mib": max(row["peak_rss_mib"] for row in parsed),
        # A single report cannot establish both required device tiers, sustained
        # thermal behavior, audio quality, or the host's end-to-end trace.
        "android_acceptance": False,
    }
