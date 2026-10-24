# iTantra TTS — architecture and evidence map

## Purpose

iTantra TTS is the speech-output layer for multilingual incoming messages. It
is also designed as a standalone local TTS component, so the same core can be
demonstrated in a browser, embedded through an API or integrated into a mobile
host application.

## End-to-end flow

```text
Incoming target-language text + stream ID + sequence number
                         │
                         ▼
          Speech normalization and segmentation
                         │
                         ▼
      INT8 or W4A8 local model selection and synthesis
                         │
                         ▼
          Audio validation and per-request telemetry
                         │
                         ▼
          Reorder buffer keyed by stream + sequence
                         │
                         ▼
                 Ordered host playback
```

## Model families

| Languages | Model chain | Precision choices |
|---|---|---|
| Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Tamil, Telugu | EchoBharat FastPitch + HiFi-GAN | INT8 and W4A8 |
| Odia | MMS VITS | INT8 and W4A8 |

The W4A8 implementation retains INT4 weights in packed form while custom ONNX
Runtime operators perform eligible integer operations. The graph uses UINT8
activations and INT32 accumulation, with scale/output stages and baseline
floating-point layers retained where required by the original model topology.

## What a live run demonstrates

- Speech generation and in-page audio playback.
- Selected language, precision and concrete model variant.
- Normalized text that reaches synthesis.
- Request-scoped wall time, CPU time, worker peak RSS, model disk footprint,
  audio duration and RTF.
- Sequence-preserving output through the shared ordered pipeline.

## Reproducible evidence

```bash
sha256sum --check docs/benchmarks/w4a8-per-channel-sha256.txt

ITANTRA_MATRIX_REAL=1 ITANTRA_W4_LIBRARY=build/w4a8/libitantra_w4a8.so \
  .venv/bin/python -m pytest -o addopts='' \
  tests/integration/test_unified_language_matrix.py \
  tests/integration/test_w4a8_research_cli.py \
  tests/integration/test_unified_research_web.py -q
```

The verification record and model-operator details are available in the
[current W4A8 status note](../benchmarks/2026-10-23-w4a8-per-channel-status.md).

