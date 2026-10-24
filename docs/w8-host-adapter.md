# W8 iTantra TTS host interface

`itantra_tts.host_adapter.ItantraTtsAdapter` provides the portable integration
boundary between an iTantra-style message host and ordered speech synthesis.

The host supplies:

```text
HostTextChunk(stream_id, sequence_no, message_id, target_text, language, precision)
```

`target_text` is already in the intended spoken language. The adapter applies
the configured normalization pipeline, invokes bounded synthesis and releases
`HostPlaybackEvent` values in stream sequence order.

## Integration responsibilities

| Host application | iTantra TTS adapter |
| --- | --- |
| Transport, message encryption, translation, alert policy and playback | Normalization, synthesis, validation, worker coordination and ordered release |
| Stream registration and retry policy | Typed playback events with stream/message/sequence identity |
| User-facing delivery policy | Redacted operational event details and model provenance |

Only `AUDIO` events contain canonical playable PCM. Typed event codes let the
host apply its own delivery policy while maintaining stream order and message
identity.

## Demonstration drill

```bash
.venv/bin/python tools/demo_w8.py
```

The drill submits independent English and Hindi streams with deliberately
reordered arrival. It demonstrates that each stream is released in intended
sequence while remaining independent of the other.

