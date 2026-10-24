import pytest

from itantra_tts.evaluation import evaluate


def test_evaluation_reports_measured_distribution_and_scope() -> None:
    report = evaluate({
        "platform": "desktop", "corpus_id": "w6-research-v1",
        "samples": [
            {"language": "hi", "cold": True, "wall_ms": 1200, "audio_ms": 2000,
             "peak_rss_mib": 210},
            {"language": "hi", "cold": False, "wall_ms": 800, "audio_ms": 2000,
             "peak_rss_mib": 215},
            {"language": "hi", "cold": False, "wall_ms": 1000, "audio_ms": 2000,
             "peak_rss_mib": 220},
        ],
    })
    assert report["sample_count"] == 3
    assert report["warm_median_rtf"] == pytest.approx(0.45)
    assert report["warm_p95_wall_ms"] == pytest.approx(990)
    assert report["peak_rss_mib"] == 220
    assert report["android_acceptance"] is False


def test_android_claim_requires_device_metadata_and_first_playable_measurement() -> None:
    with pytest.raises(ValueError, match="device metadata"):
        evaluate({"platform": "android", "corpus_id": "w6-research-v1",
                  "corpus_license": "internal", "samples": [
            {"language": "hi", "cold": False, "wall_ms": 800, "audio_ms": 2000,
             "peak_rss_mib": 200}
        ]})


def test_invalid_or_empty_measurements_are_rejected() -> None:
    with pytest.raises(ValueError):
        evaluate({"platform": "desktop", "corpus_id": "x", "samples": []})
    with pytest.raises(ValueError):
        evaluate({"platform": "desktop", "corpus_id": "x", "samples": [
            {"language": "hi", "cold": False, "wall_ms": 1, "audio_ms": 0,
             "peak_rss_mib": 1}
        ]})


def test_android_row_must_include_first_playable_time_even_with_device_metadata() -> None:
    device = {
        "make": "Example", "model": "Test", "chipset": "ARM64", "android_build": "11",
        "ram_mib": 4096, "free_storage_mib": 2000, "battery_percent": 50,
        "charging": False, "thermal_state": "nominal", "airplane_mode": True,
        "runtime_version": "1.31.0", "model_checksum": "sha256:example",
        "power_state": "battery", "lock_hash": "sha256:example", "commit": "unknown",
    }
    with pytest.raises(ValueError, match="first_playable_ms"):
        evaluate({"platform": "android", "corpus_id": "w6-research-v1",
                  "corpus_license": "internal", "device": device,
                  "samples": [{"language": "hi", "cold": False, "wall_ms": 800,
                               "audio_ms": 2000, "peak_rss_mib": 200}]})
