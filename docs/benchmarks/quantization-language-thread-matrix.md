# Quantization benchmark matrix by language and thread count

This report consolidates the benchmark evidence currently recorded for the
ten-language TTS demo. It is a coverage ledger as well as a results summary:
`Not measured` means there is no qualifying result in the retained benchmark
records. Functional audio smoke tests are not performance measurements.

## How to read this report

- **INT8** is the pinned executable baseline. The EchoBharat languages use
  FastPitch + HiFi-GAN; Odia uses MMS VITS.
- **INT4 W4A8** is the executable research path: packed INT4 weights with
  UINT8 activations and INT32 accumulation in eligible custom operators. It is
  a mixed-precision graph, not an all-INT4 graph. The current selection is
  per-channel for the nine EchoBharat languages and scalar W4A8 for Odia.
- **INT4 storage-only** is a separate historical ONNX QDQ/repacking experiment.
  It stores some weights in INT4 but restores INT8 or FP32 computation. Its
  timings must not be presented as executable INT4 compute performance.
- `RTF` is elapsed synthesis time divided by generated audio duration. A lower
  value is faster. Cold CLI RTF, resident warm RTF, and browser end-to-end RTF
  are distinct measurements and are not averaged together.
- Thread count means the recorded ONNX Runtime intra-op setting. `Not
  recorded` results cannot be assigned to a thread-count column.
- RSS refers to the process/worker scope named in the evidence, not a phone
  memory measurement.

## Coverage summary

The full requested matrix has 10 languages × 2 executable quantization paths
× 3 thread counts (1, 2, 4). The retained evidence does **not** yet provide a
matched performance result for every cell. Current functional coverage is
broader: the W4A8 integration smoke exercised INT8 and W4A8 for all ten
languages, but it did not record controlled per-language timing/RSS at each
thread count.

Cells below use `RTF / peak RSS`; each value includes its evidence ID. Values
from distinct inputs or benchmark protocols are intentionally not compared as
if they were one controlled sweep. `–` is missing data, not zero or a failed
run.

### Executable INT8 — recorded desktop observations

| Language | 1 thread | 2 threads | 4 threads |
| --- | --- | --- | --- |
| Bengali | – | – | – |
| English | – | – | `0.490 / 244.3 MiB` (I4-E, warm) |
| Gujarati | – | – | `0.457 / 184.9 MiB` (I4-G, warm) |
| Hindi | `1.519 / 259.4 MiB` (R131-H1, warm); `1.526 / 277.0 MiB` (I4-H1, warm) | – | `0.481 / 277.5 MiB` (I4-Ha, warm); `0.488 / 205.7 MiB` (R131-H4, warm) |
| Kannada | – | – | – |
| Malayalam | – | – | – |
| Marathi | – | – | – |
| Odia | `0.119 / 200.0 MiB` (R131-O1, warm) | `0.075 / 200.0 MiB` (R131-O2, warm) | `0.051 / 200.1 MiB` (R131-O4, warm); `0.047 / 180.4 MiB` (I4-O, warm) |
| Tamil | – | – | – |
| Telugu | – | – | – |

### Executable INT4 W4A8 — recorded desktop observations

| Language | 1 thread | 2 threads | 4 threads |
| --- | --- | --- | --- |
| Bengali | – | – | – |
| English | – | – | – |
| Gujarati | – | – | – |
| Hindi | – | `~27.4 / ~136–138 MiB` (W4-H2, cold scalar parallel; RTF derived from wall/audio) | `7.354 / 135.7 MiB` and `7.426 / 137.1 MiB` (W4-H4, cold scalar tile-8); current per-channel browser smoke has no thread count recorded |
| Kannada | – | – | – |
| Malayalam | – | – | – |
| Marathi | – | – | – |
| Odia | – | – | – |
| Tamil | – | – | `0.895 / not recorded` (W4-T4, cold AVX2 tile-32) |
| Telugu | – | – | – |

The W4A8 figures above are historical kernel/input experiments, not a
complete evaluation of the currently selected per-channel pack. The current
per-channel pack has one reported Hindi browser smoke at `RTF 0.816`, worker
peak RSS `224.6 MiB`, generated audio `3.576 s`, and desktop total wall
`2.918 s`; its thread count and full controlled-run metadata were not
recorded, so it is listed outside the thread matrix. The same report includes
another owner-reported run with `7,857.2 ms` worker wall and `273.8 MiB` peak
RSS, but language, input, audio duration, thread count, and cold/warm state
are absent; it is not a per-language benchmark cell.

### Historical INT4 storage-only probes — not INT4 compute

These same-input rows were measured using resident sessions, four intra-op
threads, one cold request followed by three warm requests. Reported RTF and
RSS below are warm median RTF / peak process RSS.

| Language | 1 thread | 2 threads | 4 threads |
| --- | --- | --- | --- |
| Bengali | – | – | – |
| English | – | – | `0.228 / 192.6 MiB` (I4-E, storage-only) |
| Gujarati | – | – | `0.271 / 151.5 MiB` (I4-G, storage-only) |
| Hindi | `0.598 / 232.5 MiB` (I4-H1, storage-only) | – | `0.190 / 233.5 MiB` (I4-Hb, storage-only); `0.206 / 214.3 MiB` (I4-Ha, storage-only) |
| Kannada | – | – | – |
| Malayalam | – | – | – |
| Marathi | – | – | – |
| Odia | – | – | `0.147 / 167.1 MiB` (I4-O, storage-only) |
| Tamil | – | – | – |
| Telugu | – | – | – |

For the Hindi storage-only comparison, the unchanged INT8 baseline measured
`0.481 / 277.5 MiB` at four threads and `1.526 / 277.0 MiB` at one thread.
These measurements apply to that particular INT4-QDQ vocoder experiment and
must not be attributed to current executable W4A8.

## Evidence records and conditions

| ID | Quantization / language | Input and protocol | Recorded result | Evidence source |
| --- | --- | --- | --- | --- |
| I4-Ha | INT8 Hindi baseline and storage-only INT4 candidate | Long Hindi sentence; ORT 1.31 CPU; x86-64; resident sessions; 4 threads; one cold + three warm | INT8 `0.481 / 277.5 MiB`; storage-only `0.206 / 214.3 MiB` | `int4-model-inventory.md`, “Measured desktop probes” |
| I4-Hb | INT8 Hindi baseline and storage-only INT4 candidate | Same Hindi sentence; ORT 1.31 CPU; resident sessions; 4 threads; one cold + three warm; separately recorded comparison | INT8 `0.481 / 277.5 MiB`; storage-only `0.190 / 233.5 MiB` | `int4-model-inventory.md`, “Same-input desktop runtime gate” |
| I4-H1 | INT8 Hindi baseline and storage-only INT4 candidate | Same input family; ORT 1.31 CPU; resident sessions; 1 thread; one cold + three warm | INT8 `1.526 / 277.0 MiB`; storage-only `0.598 / 232.5 MiB` | `int4-model-inventory.md`, “Same-input desktop runtime gate” |
| I4-E | INT8 English baseline and storage-only INT4 candidate | English first-aid sentence; ORT 1.31 CPU; resident sessions; 4 threads; one cold + three warm | INT8 `0.490 / 244.3 MiB`; storage-only `0.228 / 192.6 MiB` | `int4-model-inventory.md`, “Measured desktop probes” |
| I4-G | INT8 Gujarati baseline and storage-only INT4 candidate | Gujarati greeting; ORT 1.31 CPU; resident sessions; 4 threads; one cold + three warm | INT8 `0.457 / 184.9 MiB`; storage-only `0.271 / 151.5 MiB` | `int4-model-inventory.md`, “Measured desktop probes” |
| I4-O | INT8 Odia baseline and storage-only INT4 candidate | Odia greeting; ORT 1.31 CPU; resident sessions; 4 threads; one cold + three warm | INT8 `0.047 / 180.4 MiB`; storage-only `0.147 / 167.1 MiB` | `int4-model-inventory.md`, “Measured desktop probes” |
| R131-H1 | INT8 Hindi | Short Hindi sentence; ORT 1.31 CPU; resident; 1 thread; 2 warm runs | `RTF 1.519`, peak RSS `259.4 MiB` | `2026-10-20-runtime-131-evaluation.md` |
| R131-H4 | INT8 Hindi | Full Hindi sentence; ORT 1.31 CPU; sequential; 4 threads; 2 runs | `RTF 0.488`, peak RSS `205.7 MiB` | `2026-10-20-runtime-131-evaluation.md` |
| R131-O1/O2/O4 | INT8 Odia static-QDQ research graph | Odia question; ORT 1.30 CPU; resident; 1, 2, or 4 threads; 2 warm runs | `0.119 / 200.0`, `0.075 / 200.0`, `0.051 / 200.1 MiB` respectively | `2026-10-20-runtime-131-evaluation.md` |
| W4-H2 | W4A8 Hindi scalar parallel kernel | `हाँ.`; ORT 1.31; 2 threads; cold alternating runs; 290.2 ms audio | Before `12,909.5 / 12,957.0 ms`; after `7,972.6 / 7,938.1 ms`; RSS about `136–138 MiB`; after median cold RTF about `27.4` | `w4a8-scalar-pilot.md` |
| W4-H4 | W4A8 Hindi scalar tile-8 kernel | `हाँ.`; ORT 1.31; 4 threads; two cold runs; 290.2 ms audio | Paired wall/RSS/RTF: `2,134.6 ms / 135.7 MiB / 7.354`; `2,155.5 ms / 137.1 MiB / 7.426` | `2026-10-21-w4a8-tiled-conv.md` |
| W4-T4 | W4A8 Tamil AVX2 tile-32 kernel | `மூன்று புள்ளி ஐந்து`; ORT 1.31; x86-64; 4 threads; two cold CLI runs; 1,486.1 ms audio | `1,399.7 / 1,330.6 ms`; RTF `0.942 / 0.895`; RSS not recorded | `2026-10-21-w4a8-avx2.md` |

Additional INT8 Odia measurements in `2026-10-20-desktop-runtime-sweep.md`
use a different text, runtime/session setup and warm-run protocol. They are
retained there as individual observations and are not merged into the matrix.

## What is and is not established

- All ten languages have functional INT8 and W4A8 smoke coverage through the
  shared synthesis service; that does not fill the timing/RSS matrix.
- Current controlled per-channel W4A8 performance coverage is incomplete.
  Historical scalar and AVX2 measurements belong to those kernel versions and
  inputs; the AVX2 path is an x86 desktop result.
- Storage-only INT4 observations can show the effect of a smaller serialized
  artifact and its chosen dequantized execution graph. They do not establish
  native INT4 arithmetic speed or memory behavior.
- These are desktop measurements. They do not constitute low-end Android
  measurements or a device acceptance result.
- No result should be inferred for an unmeasured language/thread cell, and
  unlike cells should not be compared as a causal A/B result.

## Completing the matrix

For a comparable matrix, run the same versioned corpus item for every language
and both executable paths at 1, 2, and 4 intra-op threads. Record at least
three cold processes and five warm requests per cell, using the same model
policy (resident or sequential) across candidates. Store, for each run:

1. language, corpus ID/license, and normalized-text hash (not raw private text);
2. quantization path, model/checksum identity, custom-kernel build identity,
   runtime version, execution provider, CPU/OS, and thread count;
3. cold/warm status, audio duration, worker wall and end-to-end wall time,
   CPU time, worker peak RSS, model disk size, and RTF;
4. synthesis validity and quality-review status.

Report per-cell median and spread, retain raw machine-readable JSON/CSV, and
keep current per-channel W4A8 results separate from historical scalar/AVX2
kernel experiments. Do not substitute projected mobile values for measured
results.
