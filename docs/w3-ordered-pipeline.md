# W3 ordered synthesis pipeline

`itantra_tts.application.AsyncSynthesisPipeline` is a reusable asynchronous
worker pool for normalized, sentence-sized `SynthesisRequest` values. It
preserves the `(stream_id, sequence_no)` contract from message arrival through
audio release.

```python
from itantra_tts import Precision, SynthesisRequest
from itantra_tts.application import AsyncSynthesisPipeline, PipelineEventKind

async def play_example(synthesize_async):
    async with AsyncSynthesisPipeline(synthesize_async, workers=2) as pipeline:
        pipeline.register_stream("receiver-1", start_sequence=0)
        await pipeline.submit(SynthesisRequest("receiver-1", 0, "Hello.", "en", Precision.INT8))
        event = await pipeline.receive("receiver-1")
        if event.kind is PipelineEventKind.AUDIO:
            return event.result.audio_pcm_s16le
```

## Pipeline design

- Bounded streams, requests, queue entries, worker tasks, text length and PCM
  bytes keep synthesis resource-aware.
- The reorder buffer releases only the next expected item in a stream.
- Independent streams progress independently.
- Explicit event types preserve ordering across audio, cancellation, timeout,
  validation and retry decisions.
- `WAIT`, `SKIP` and `ABORT` policies allow the host to select its delivery
  behavior while retaining a visible sequence decision.
- The backend contract supports cooperative cancellation and validates canonical
  mono PCM before a clip enters the reorder buffer.

The web demonstration joins ordered sentence audio with a short silence gap.
The same pipeline is exposed to host integrations through the W8 adapter.
Translation remains a host concern: the pipeline receives text in its intended
spoken language.

