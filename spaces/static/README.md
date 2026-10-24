---
title: iTantra TTS browser companion
emoji: 🔊
colorFrom: yellow
colorTo: blue
sdk: static
app_file: index.html
pinned: false
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# iTantra TTS browser companion

This is a **browser companion**, not the full iTantra TTS engine or the
Android delivery runtime. It attempts selected INT8 EchoBharat synthesis in
the visitor's browser with ONNX Runtime Web. Text stays in the browser; only
the selected public model files are fetched from the pinned source revision.
The page loads the full ONNX Runtime Web 1.30.0 browser bundle because these
INT8 graphs use integer operators such as `ConvInteger`.

## What this page can do

- Attempt browser-local INT8 synthesis for Bengali, English, Gujarati, Hindi,
  Kannada, Malayalam, Marathi, Tamil and Telugu.
- Download only the selected language's FastPitch, HiFi-GAN and token files on
  first use; the page retains them in the browser's Cache Storage for later
  visits on the same device.
- Play the generated audio in the page and show browser-scoped load/inference
  timing and real-time factor. Cross-origin isolation allows ONNX Runtime Web
  to use up to four WASM threads where the visitor's browser supports it.

## What remains in the full local engine

The cloned project is the authoritative research engine. It provides its
normalization pipeline, sentence ordering, API/CLI, request-scoped worker
metrics, the Odia research adapter, and the executable native W4A8 path.

INT4/W4A8 is deliberately unavailable here. It depends on the project's native
custom operator (`libitantra_w4a8.so`) and an ONNX Runtime build for the local
Linux/x86 research path; this Static Space does not silently substitute INT8.
Odia is also directed to the local research engine because it uses a separate
VITS model family and is governed by its own CC-BY-NC-4.0 research terms.

## Run the full engine locally

Clone the project, provision the separately obtained model artifacts according
to `models/artifacts.lock.yaml`, then run:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[demo,api,test]'
.venv/bin/python tools/unified_research_web.py \
  --library "$PWD/build/w4a8/libitantra_w4a8.so" --port 18095
```

Open `http://127.0.0.1:18095/`. The full local engine is offline after model
provisioning and is the correct route for its native INT8/W4A8 functionality.

## Model source and terms

The browser adapter requests the pinned public EchoBharat revision
`RaunakSaha/echobharat-models@2ad314c3ecafeb252bac82785ac541325596ac9c`.
No model files are included in this Static Space. Check the upstream model
terms and provenance before redistributing artifacts or using them beyond the
project's research scope.
