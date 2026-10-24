# Runtime metrics contract

The interface reports metrics from the same local request that generated the
audio. This keeps playback, model selection and performance inspection tied to
one reproducible execution path.

| Field | Measurement |
| --- | --- |
| Total wall time | `perf_counter_ns` around the web-backend request, including isolated-worker startup and IPC. |
| Worker wall time | Elapsed time inside the isolated synthesis worker, including model loading and synthesis. |
| Stage timing | FastPitch and HiFi-GAN intervals when the backend exposes them; W4A8 reports a unified worker interval. |
| CPU time / utilization | Worker CPU-clock delta and CPU time divided by worker wall time. The utilization represents aggregate model-worker CPU use across cores. |
| RSS / peak RSS | Resident memory at request boundaries and peak in the displayed worker scope. |
| Audio duration / RTF | Generated samples divided by sample rate; total wall time divided by generated-audio duration. |
| Provenance | Executed model ID, precision, ONNX Runtime/provider, thread count and worker mode. |

Each browser request runs in an isolated synthesis worker. This provides a
repeatable worker-memory boundary while allowing evaluators to switch language
and precision choices in the same interface.

For a controlled precision, kernel or thread-count comparison, retain the
normalized text, language, graph checksum, audio duration, runtime version,
thread count and worker lifetime across repetitions. The benchmark protocol
uses both cold and resident runs to characterize the complete lifecycle.

