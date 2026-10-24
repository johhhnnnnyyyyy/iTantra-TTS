# Per-channel W4A8 status and performance interpretation

This is the current technical status for the local research demo. It supersedes
earlier statements that the per-channel candidate was only a temporary Hindi
experiment. Earlier benchmark reports remain historical evidence for their
own graph set and machine conditions.

## Executed model paths

The local research UI selects one complete model chain per request:

| Language family | INT8 path | INT4 research path |
| --- | --- | --- |
| Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Tamil, Telugu | Pinned EchoBharat FastPitch + HiFi-GAN INT8 graphs | `experimental-w4a8-per-channel/<language>` FastPitch + HiFi-GAN graphs |
| Odia | Pinned MMS VITS INT8 graph | Preserved scalar W4A8 Odia graph |

The nine EchoBharat W4A8 pairs are generated from the pinned executable INT8
graphs and are selected fail-closed: a missing per-channel artifact makes the
INT4 request fail rather than silently select the earlier scalar graph. The
checksums in `w4a8-per-channel-sha256.txt` are the local artifact identity.

Each EchoBharat pair has 106 `ConvW4A8` and 24 `MatMulW4A8` operators. Their
eligible weight operands remain packed signed INT4 throughout custom-operator
execution. Activations are UINT8, integer products accumulate to INT32, and
the surrounding graph applies floating-point scales. Four `ConvTranspose`
nodes per pair remain FP32, as they do in the audited INT8 baseline. This is
therefore a mixed-precision W4A8 graph, not an all-INT4 or W4A4 claim.

## Quality decision record

The scalar W4 rewrite changed a Hindi FastPitch output from 308 to 295 frames
for one fixed input. With the same INT8 mel input, its vocoder waveform
diagnostics relative to the reference were 1.367 relative RMSE and 0.580
correlation. The corresponding per-output-channel candidate measured 0.814
and 0.741; its FastPitch preserved 308 frames. These values are numerical
diagnostics, not MOS or intelligibility scores.

The project owner listened to the candidate clips and selected the
per-output-channel EchoBharat result for the local demo. This explains why the
research UI now defaults to that pack. It does not establish a perceptual
quality score across every text or language; future quality claims need a
defined listener/corpus/rubric record.

## Current verified execution evidence

The exact following checks were run against the current local artifact pack:

```sh
sha256sum --check docs/benchmarks/w4a8-per-channel-sha256.txt
ITANTRA_MATRIX_REAL=1 ITANTRA_W4_LIBRARY=build/w4a8/libitantra_w4a8.so \
  .venv/bin/python -m pytest -o addopts='' \
  tests/integration/test_unified_language_matrix.py \
  tests/integration/test_w4a8_research_cli.py \
  tests/integration/test_unified_research_web.py -q
```

The checksum command reported all 18 EchoBharat graphs valid. The focused
matrix reported 30 passed and 1 skipped, and generated finite, non-silent
audio for every required language/precision pair through the shared service.
The full regression afterwards reported 195 passed and 66 skipped, with one
existing multiprocessing-fork deprecation warning.

One browser smoke request for Hindi used
`experimental-w4a8-per-channel-hi`, produced 3.576 seconds of audio in 2.918
seconds desktop total wall time (RTF 0.816), and reported 224.6 MiB worker
peak RSS. It was a single cold request, not a representative benchmark.

## Interpreting the displayed metrics

`Desktop total wall` measures end-to-end elapsed time, including disposable
worker startup and IPC. `Worker wall` is time inside the worker. `CPU time` is
the sum of CPU consumed on all executing cores, so it can be higher than wall
time; CPU time divided by worker wall is the displayed aggregate CPU
utilization, not a whole-machine percentage. `Peak RSS` is the disposable
worker's resident memory peak, not the Gradio server or system memory.

After the per-channel selection, the owner reported a separate browser run
with 7,857.2 ms worker wall, 15,471.2 ms CPU time, and 273.8 MiB peak RSS. Its
language, normalized text, audio duration, thread count, cold/resident state,
and total wall time were not captured with that report. It is useful feedback,
but it is **not comparable** with the Hindi smoke above or a controlled
benchmark. The CPU/wall ratio of about 1.97 only indicates roughly two cores'
worth of work on average; it does not mean 15 seconds of user-visible delay.

Per-channel scaling adds per-output-channel scale vectors and broadcast
multiplies around custom operators. It can add cost, but the unpaired report
cannot attribute the larger wall time to scaling: model loading, generated
audio length, text segmentation, ORT scheduling, competing work, and thread
count can all change the result.

## Next controlled profiling experiment

Use the same normalized text, language, graph checksums, ORT version,
thread-count setting and machine state for each candidate. Capture three cold
and at least five resident/warm repetitions for thread counts 1, 2 and 4.
For every row record audio duration, desktop total wall, worker wall, CPU
time, worker peak RSS, RTF, and model IDs. Profile FastPitch and HiFi-GAN
separately before changing the kernel. This isolates whether time is spent in
model loading, FastPitch, custom W4 convolution, FP32 `ConvTranspose`,
HiFi-GAN output handling, or process setup.

Do not claim an RTF, RAM, quality, or low-end-device improvement until that
matched experiment is recorded. The current browser values demonstrate an
executable research path, not a general performance guarantee.
