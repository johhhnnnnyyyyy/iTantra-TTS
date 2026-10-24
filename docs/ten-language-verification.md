# Ten-language verification

The local evaluation matrix runs every required language through INT8 and W4A8
model paths. It verifies generated audio, sample rate, model identity,
normalization output and request-scoped telemetry.

## Language matrix

| Language | Tag | Number-normalization check |
| --- | --- | --- |
| Bengali | `bn` | `5` → `পাঁচ` |
| English | `en` | `5` → `five` |
| Gujarati | `gu` | `5` → `પાંચ` |
| Hindi | `hi` | `5` → `पांच` |
| Kannada | `kn` | `5` → `ಐದು` |
| Malayalam | `ml` | `5` → `അഞ്ച്` |
| Marathi | `mr` | `5` → `पाच` |
| Odia | `or` | `5` → `ପାଞ୍ଚ` |
| Tamil | `ta` | `5` → `ஐந்து` |
| Telugu | `te` | `5` → `ఐదు` |

The nine EchoBharat language pairs use their per-output-channel W4A8 graphs.
Odia uses the MMS VITS W4A8 path. W4A8 execution retains packed INT4 eligible
weights, UINT8 activations and INT32 accumulation.

## Launch the evaluation interface

```bash
.venv/bin/python tools/unified_research_web.py \
  --library "$PWD/build/w4a8/libitantra_w4a8.so" \
  --port 18095
```

Open `http://127.0.0.1:18095/`, select a language and precision, then play the
generated waveform in the page. The adjacent inspection panel reports the text
sent to synthesis and the request metrics.

## Automated matrix

```bash
sha256sum --check docs/benchmarks/w4a8-per-channel-sha256.txt

ITANTRA_MATRIX_REAL=1 ITANTRA_W4_LIBRARY=build/w4a8/libitantra_w4a8.so \
  .venv/bin/python -m pytest -o addopts='' \
  tests/integration/test_unified_language_matrix.py -v
```

This is the same model-chain contract used by the browser interface. Detailed
operator coverage, graph identity and representative request measurements are
recorded in the [current W4A8 status note](benchmarks/2026-10-23-w4a8-per-channel-status.md).

