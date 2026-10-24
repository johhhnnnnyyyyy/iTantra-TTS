# Faculty walkthrough

## The project in one sentence

iTantra TTS is an offline-first multilingual speech component that turns
already-translated incoming text into ordered, playable audio across ten Indian
languages and English.

## Five-minute demonstration

1. Start the local interface:

   ```bash
   .venv/bin/python tools/unified_research_web.py \
     --library "$PWD/build/w4a8/libitantra_w4a8.so" --port 18095
   ```

2. Open `http://127.0.0.1:18095/`. Enter a Hindi, Tamil or English sentence,
   select INT8, generate speech and play it in the page.

3. Repeat with W4A8. The same interface reports the selected model variant,
   normalized text, generated waveform, wall time, CPU time, worker peak RSS,
   model footprint, audio duration and RTF for that request.

4. Change languages to demonstrate the complete language matrix: Hindi,
   Gujarati, Marathi, Kannada, Malayalam, Tamil, Telugu, Odia, Bengali and
   English.

5. Run the ordered-message drill:

   ```bash
   .venv/bin/python tools/demo_w8.py
   ```

   It demonstrates sequence-aware audio release across independent streams.

## Technical contribution

- Local FastPitch + HiFi-GAN synthesis for nine languages and a VITS path for
  Odia.
- Executable mixed-precision W4A8 graphs: packed INT4 weights, UINT8
  activations and INT32 accumulation for eligible operators.
- Text normalization before synthesis and sentence-sized worker jobs.
- Bounded worker pools, audio validation and sequence-number reorder buffers.
- A host adapter that preserves stream, message and sequence identifiers.
- A web research interface that uses the same synthesis service as the CLI and
  API, so live playback and telemetry come from the model path under review.

## Evidence available during discussion

- [language and precision verification](ten-language-verification.md)
- [W4A8 architecture and execution record](benchmarks/2026-10-23-w4a8-per-channel-status.md)
- [metrics contract](metrics.md)
- [ordered pipeline design](w3-ordered-pipeline.md)
- [host adapter](w8-host-adapter.md)

## Scope

The TTS component is intentionally modular. A host application provides the
already-translated text and owns transport, translation, alert policy and final
playback integration. This keeps the speech layer independently reusable for
research prototypes, local applications and iTantra integration.
