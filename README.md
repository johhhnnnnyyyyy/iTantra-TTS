# iTantra TTS

An offline-first, multilingual text-to-speech component for ordered message
playback. It is designed both as a standalone TTS service and as a reusable
speech-output layer for iTantra-style applications.

## What it demonstrates

- Speech synthesis for Hindi, Gujarati, Marathi, Kannada, Malayalam, Tamil,
  Telugu, Odia, Bengali and English.
- Local ONNX Runtime inference with INT8 and experimental W4A8 research paths.
- Spoken-text normalization, sentence-level synthesis and sequence-aware audio
  release.
- In-page audio playback and request-scoped latency, CPU, memory, model-size
  and real-time-factor metrics.
- CLI, HTTP API, browser demo and host-adapter interfaces over the shared
  synthesis service.

The W4A8 research path stores eligible weights in packed signed INT4 form,
uses UINT8 activations and INT32 accumulation in supported custom operators,
and retains floating-point stages where the model graph requires them. It is a
mixed-precision path, not an all-INT4 graph.

## Free browser companion

The [Static Space companion](spaces/static/README.md) is a separate,
no-server deployment target. It fetches the visitor-selected pinned INT8
EchoBharat pack into the browser and attempts local browser synthesis. It is
not a replacement for this local engine: it does not claim the native W4A8
path, the separate Odia adapter, full normalization/order handling, or
comparable benchmark telemetry.

Stage it for a Hugging Face Static Space without copying local artifacts:

```bash
.venv/bin/python tools/stage_static_space.py /tmp/itantra-static-space
```

Upload the four files in that output directory to a new Hugging Face Space
using the **Static** SDK. The companion README contains its model-source and
local-engine instructions.

## Benchmark snapshot

Recorded local desktop results show that both paths execute and produce audio.
These observations use different inputs and run protocols, so they are not a
direct INT8-versus-INT4 ranking and do not predict low-end-phone performance.

| Path and example | Recorded result | Run context |
| --- | --- | --- |
| INT8, Hindi | RTF `0.488`; peak process RSS `205.7 MiB` | ORT 1.31 CPU, x86-64, 4 threads; two runs |
| INT8, Odia static-QDQ research graph | Warm RTF `0.042`; peak process RSS `204.8 MiB` | ORT 1.31 CPU, x86-64, 4 threads; two warm runs |
| W4A8, Hindi per-channel browser smoke | RTF `0.816`; worker peak RSS `224.6 MiB`; `3.576 s` audio | Single desktop request; thread count not captured |

Functional synthesis has been smoke-tested for all ten languages on INT8 and
W4A8. A complete, controlled performance matrix for every language at 1, 2,
and 4 threads is still being collected; missing cells are listed as such in
the [full benchmark matrix](docs/benchmarks/quantization-language-thread-matrix.md).
The storage-only INT4 experiments are excluded from this snapshot because they
do not execute INT4 arithmetic.

## Start here

1. Read the [faculty walkthrough](docs/faculty-demo.md).
2. Review the [architecture and evidence map](docs/showcase/README.md).
3. See the [ten-language verification guide](docs/ten-language-verification.md)
   and [current W4A8 execution record](docs/benchmarks/2026-10-23-w4a8-per-channel-status.md).
4. Explore the [ordered synthesis pipeline](docs/w3-ordered-pipeline.md) and
   [host adapter](docs/w8-host-adapter.md).

## Install the project

Requires Python 3.11 or newer. From the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[demo,api,test]'
```

Run the automated tests:

```bash
.venv/bin/python -m pytest
```

Run the local browser demo after placing the separately obtained model files in
the expected `models/artifacts/` layout:

```bash
.venv/bin/python tools/unified_research_web.py --port 18095
```

Open <http://127.0.0.1:18095/>. To enable the W4A8 custom-operator path, build
the Linux/x86-64 library against the matching ONNX Runtime headers and pass its
path with `--library`; see the [W4A8 execution record](docs/benchmarks/2026-10-23-w4a8-per-channel-status.md).

For the command-line and HTTP interfaces, see the examples in the
[ten-language verification guide](docs/ten-language-verification.md) and the
package entry points in `pyproject.toml`.

## Model artifacts and terms

Model weights are not included in this source repository. The EchoBharat
FastPitch/HiFi-GAN models and Facebook MMS Odia model are separate upstream
artifacts and retain their own repository and model terms; the project MIT
license applies to this repository's code only. Check the current upstream
terms and attribution before obtaining or redistributing any model files.

## Project documentation

The [documentation index](docs/README.md) contains only the project overview,
architecture, verification and integration material intended for this public
repository.

## License

Project source code: [MIT](LICENSE). Model artifact terms are independent and
are not changed by the project license.
