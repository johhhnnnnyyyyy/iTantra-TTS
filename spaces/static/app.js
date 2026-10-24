/* Browser-local companion: no browser speech or hosted TTS fallback. */
import { normalizeForSpeech } from "./normalizer.js";

(() => {
  "use strict";

  const MODEL_SOURCE = Object.freeze({
    repository: "RaunakSaha/echobharat-models",
    revision: "2ad314c3ecafeb252bac82785ac541325596ac9c",
  });
  const BROWSER_INT8_LANGUAGES = new Set(["bn", "en", "gu", "hi", "kn", "ml", "mr", "ta", "te"]);
  const SAMPLE_RATE_HZ = 22050;
  const MODEL_CACHE_NAME = `itantra-static-models-${MODEL_SOURCE.revision}`;
  const MAX_WASM_THREADS = 4;
  const sessions = new Map();

  const element = (id) => document.getElementById(id);
  const text = element("text");
  const language = element("language");
  const precision = element("precision");
  const synthesizeButton = element("synthesize");
  const characterCount = element("character-count");
  const status = element("status");
  const audio = element("audio");
  const download = element("download");
  const metrics = element("metrics");
  const normalization = element("normalization");
  const normalizedText = element("normalized-text");
  const normalizationStatus = element("normalization-status");

  const setStatus = (message) => { status.textContent = message; };
  const showNormalization = (result) => {
    normalizedText.textContent = result.normalizedText;
    const suffix = result.warnings.length ? ` Preserved: ${result.warnings.join(", ")}.` : "";
    normalizationStatus.textContent = `${result.rule} applied across ${result.sentences.length} sentence${result.sentences.length === 1 ? "" : "s"}.${suffix}`;
    normalization.hidden = false;
  };
  const hideNormalization = () => {
    normalization.hidden = true;
    normalizedText.textContent = "";
    normalizationStatus.textContent = "";
  };
  const setMetrics = ({ load = "Not measured", inference = "Not measured", audioRtf = "Not measured", runtime = runtimeLabel() } = {}) => {
    metrics.innerHTML = [
      ["Model fetch / session", load],
      ["Inference wall time", inference],
      ["Audio duration / RTF", audioRtf],
      ["WASM execution", runtime],
      ["CPU / RSS", "Not exposed by standard browser APIs"],
    ].map(([name, value]) => `<div><dt>${name}</dt><dd>${value}</dd></div>`).join("");
  };

  const runtimeProfile = () => {
    const crossOriginIsolated = globalThis.crossOriginIsolated === true;
    const available = Number(globalThis.navigator?.hardwareConcurrency || 1);
    return Object.freeze({
      crossOriginIsolated,
      threads: crossOriginIsolated ? Math.max(1, Math.min(MAX_WASM_THREADS, available)) : 1,
    });
  };

  const runtimeLabel = () => {
    const profile = runtimeProfile();
    return profile.crossOriginIsolated
      ? `WASM SIMD · ${profile.threads} threads`
      : "WASM SIMD · 1 thread (host lacks isolation)";
  };

  const configureRuntime = () => {
    if (!globalThis.ort) throw new Error("ONNX Runtime Web did not load; check your connection and reload the page");
    const profile = runtimeProfile();
    globalThis.ort.env.wasm.numThreads = profile.threads;
    globalThis.ort.env.wasm.proxy = true;
    return profile;
  };

  const modelUrl = (path) => (
    `https://huggingface.co/${MODEL_SOURCE.repository}/resolve/${MODEL_SOURCE.revision}/${path}?download=true`
  );

  const fetchRequired = async (path) => {
    const url = modelUrl(path);
    const request = new Request(url, { mode: "cors" });
    const cache = globalThis.caches ? await globalThis.caches.open(MODEL_CACHE_NAME) : null;
    const cached = cache ? await cache.match(request) : undefined;
    if (cached) return { response: cached, cacheHit: true };
    const response = await fetch(request);
    if (!response.ok) throw new Error(`model fetch failed (${response.status})`);
    if (cache) void cache.put(request, response.clone()).catch(() => {});
    return { response, cacheHit: false };
  };

  const tokenize = (input, tokens) => {
    const ids = new Map();
    tokens.forEach((symbol, index) => { if (!ids.has(symbol)) ids.set(symbol, index); });
    const blankId = ids.get("<BLNK>");
    if (blankId === undefined) throw new Error("selected token file has no blank token");
    const symbols = [...ids.keys()]
      .filter((symbol) => !symbol.startsWith("<"))
      .sort((left, right) => right.length - left.length);
    const prepared = [...input].map((character) => character === "।" ? "." : character).join("");
    const encoded = [];
    for (let index = 0; index < prepared.length;) {
      const symbol = symbols.find((candidate) => prepared.startsWith(candidate, index));
      if (symbol === undefined) { encoded.push(blankId); index += 1; }
      else { encoded.push(ids.get(symbol)); index += symbol.length; }
    }
    if (encoded.length === 0) throw new Error("text has no synthesizable tokens for the selected model");
    return encoded;
  };

  const loadLanguage = async (selectedLanguage) => {
    const cached = sessions.get(selectedLanguage);
    if (cached) return { value: cached, loadMs: 0, cached: true, modelCacheHit: true };
    configureRuntime();
    const started = performance.now();
    const [fastpitchAsset, hifiganAsset, tokenAsset] = await Promise.all([
      fetchRequired(`${selectedLanguage}/fastpitch-${selectedLanguage}.int8.onnx`),
      fetchRequired(`${selectedLanguage}/hifigan-${selectedLanguage}.int8.onnx`),
      fetchRequired(`${selectedLanguage}/fastpitch-${selectedLanguage}.tokens.json`),
    ]);
    const [fastpitchBytes, hifiganBytes, tokens] = await Promise.all([
      fastpitchAsset.response.arrayBuffer(), hifiganAsset.response.arrayBuffer(), tokenAsset.response.json(),
    ]);
    const [fastpitch, hifigan] = await Promise.all([
      globalThis.ort.InferenceSession.create(fastpitchBytes, {
        executionProviders: ["wasm"], graphOptimizationLevel: "all",
      }),
      globalThis.ort.InferenceSession.create(hifiganBytes, {
        executionProviders: ["wasm"], graphOptimizationLevel: "all",
      }),
    ]);
    const value = { fastpitch, hifigan, tokens };
    sessions.clear();
    sessions.set(selectedLanguage, value);
    return {
      value,
      loadMs: performance.now() - started,
      cached: false,
      modelCacheHit: [fastpitchAsset, hifiganAsset, tokenAsset].every((asset) => asset.cacheHit),
    };
  };

  const toWavBlob = (samples) => {
    const pcm = new Int16Array(samples.length);
    for (let index = 0; index < samples.length; index += 1) {
      pcm[index] = Math.max(-1, Math.min(1, samples[index])) * 32767;
    }
    const buffer = new ArrayBuffer(44 + pcm.byteLength);
    const view = new DataView(buffer);
    const write = (offset, value) => [...value].forEach((character, i) => view.setUint8(offset + i, character.charCodeAt(0)));
    write(0, "RIFF"); view.setUint32(4, 36 + pcm.byteLength, true); write(8, "WAVE"); write(12, "fmt ");
    view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, SAMPLE_RATE_HZ, true); view.setUint32(28, SAMPLE_RATE_HZ * 2, true);
    view.setUint16(32, 2, true); view.setUint16(34, 16, true); write(36, "data");
    view.setUint32(40, pcm.byteLength, true); new Int16Array(buffer, 44).set(pcm);
    return new Blob([buffer], { type: "audio/wav" });
  };

  const selectFeedName = (session, preferred) => session.inputNames.includes(preferred) ? preferred : session.inputNames[0];
  const synthesizeSentence = async (sentence, selected) => {
    const tokenIds = tokenize(sentence, selected.tokens);
    const tokens = new globalThis.ort.Tensor("int64", BigInt64Array.from(tokenIds, BigInt), [1, tokenIds.length]);
    const melResult = await selected.fastpitch.run({ [selectFeedName(selected.fastpitch, "text")]: tokens });
    const mel = melResult[selected.fastpitch.outputNames[0]];
    const waveformResult = await selected.hifigan.run({ [selectFeedName(selected.hifigan, "mel")]: mel });
    return Float32Array.from(waveformResult[selected.hifigan.outputNames[0]].data);
  };
  const stitchSentences = (clips) => {
    const gap = Math.round(SAMPLE_RATE_HZ * 0.03);
    const total = clips.reduce((length, clip) => length + clip.length, 0) + Math.max(0, clips.length - 1) * gap;
    const output = new Float32Array(total);
    let offset = 0;
    clips.forEach((clip, index) => {
      output.set(clip, offset);
      offset += clip.length + (index < clips.length - 1 ? gap : 0);
    });
    return output;
  };

  const synthesize = async () => {
    const selectedLanguage = language.value;
    const selectedPrecision = precision.value;
    const input = text.value.trim();
    audio.hidden = true;
    download.hidden = true;
    hideNormalization();
    if (!input || input.length > 200) {
      setStatus("Enter 1–200 characters in the selected spoken language.");
      return;
    }
    if (selectedPrecision === "int4") {
      setStatus("INT4/W4A8 is intentionally local-engine only: it requires iTantra’s native custom operator. This page will not substitute INT8.");
      setMetrics();
      return;
    }
    if (selectedLanguage === "or") {
      setStatus("Odia uses iTantra’s separate local VITS research adapter. Run the full local engine for this route.");
      setMetrics();
      return;
    }
    if (!BROWSER_INT8_LANGUAGES.has(selectedLanguage)) {
      setStatus("This selection has no browser-compatible model route. Use the full local engine.");
      return;
    }
    synthesizeButton.disabled = true;
    try {
      const normalized = normalizeForSpeech(input, selectedLanguage);
      showNormalization(normalized);
      setStatus("Fetching the selected pinned INT8 pack and creating browser sessions…");
      const loaded = await loadLanguage(selectedLanguage);
      setStatus("Synthesizing locally in this browser…");
      const inferenceStarted = performance.now();
      const clips = [];
      for (const sentence of normalized.sentences) clips.push(await synthesizeSentence(sentence, loaded.value));
      const inferenceMs = performance.now() - inferenceStarted;
      const samples = stitchSentences(clips);
      if (!samples.length || samples.some((sample) => !Number.isFinite(sample))) throw new Error("model returned invalid audio");
      const durationMs = samples.length / SAMPLE_RATE_HZ * 1000;
      const blob = toWavBlob(samples);
      const url = URL.createObjectURL(blob);
      if (audio.src) URL.revokeObjectURL(audio.src);
      audio.src = url; audio.hidden = false; download.href = url; download.hidden = false;
      setMetrics({
        load: loaded.cached
          ? "In-page session cache hit"
          : `${loaded.loadMs.toFixed(1)} ms (${loaded.modelCacheHit ? "persistent model cache" : "network + session creation"})`,
        inference: `${inferenceMs.toFixed(1)} ms`,
        audioRtf: `${durationMs.toFixed(1)} ms / ${(inferenceMs / durationMs).toFixed(3)}×`,
      });
      setStatus("Audio generated locally in this browser. These values are browser-scoped, not the full-engine benchmark.");
    } catch (error) {
      setMetrics();
      const message = String(error?.message || error);
      if (/Could not find an implementation for ConvInteger/i.test(message)) {
        setStatus("The loaded runtime is missing ConvInteger, required by this INT8 graph. Reload the latest Space build (ONNX Runtime Web 1.30.0) and retry; no audio was generated.");
      } else {
        setStatus(`Browser INT8 request could not run (${message}). Use the full local engine; no alternate TTS backend was used.`);
      }
    } finally {
      synthesizeButton.disabled = false;
    }
  };

  text.addEventListener("input", () => {
    characterCount.textContent = `${text.value.length} / 200`;
  });
  synthesizeButton.addEventListener("click", synthesize);
  window.iTantraStaticDemo = Object.freeze({ MODEL_SOURCE, BROWSER_INT8_LANGUAGES, normalizeForSpeech });
})();
