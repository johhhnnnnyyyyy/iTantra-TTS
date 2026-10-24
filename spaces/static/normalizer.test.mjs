import assert from "node:assert/strict";
import test from "node:test";
import { normalizeForSpeech } from "./normalizer.js";

test("English W2 rules feed a speech-ready sentence", () => {
  const result = normalizeForSpeech("Dr. Rao will bring ₹200 at 05:30. The score is 20%.", "en");
  assert.equal(result.normalizedText, "Doctor Rao will bring two hundred rupees at five thirty. The score is twenty percent.");
  assert.deepEqual(result.sentences, ["Doctor Rao will bring two hundred rupees at five thirty.", "The score is twenty percent."]);
  assert.equal(result.rule, "w2-browser-v1");
});

test("Hindi W2 rules preserve unsafe decimal currency while normalizing safe forms", () => {
  const result = normalizeForSpeech("बैठक 05:30 बजे है और ₹200 लाना। ₹12.50 भी रखना।", "hi");
  assert.match(result.normalizedText, /पांच बजकर तीस मिनट/);
  assert.match(result.normalizedText, /दो सौ रुपये/);
  assert.match(result.normalizedText, /₹12\.50/);
  assert.ok(result.warnings.includes("UNSAFE_DECIMAL_CURRENCY_LEFT_UNCHANGED"));
});

test("other browser languages receive safe cleanup, segmentation, and an explicit numeric preservation warning", () => {
  const result = normalizeForSpeech("  আমি 12 টাকায় আসব।  আবার দেখা হবে। ", "bn");
  assert.equal(result.normalizedText, "আমি 12 টাকায় আসব। আবার দেখা হবে।");
  assert.deepEqual(result.sentences, ["আমি 12 টাকায় আসব।", "আবার দেখা হবে।"]);
  assert.ok(result.warnings.includes("NUMERIC_FORM_LEFT_UNCHANGED"));
});
