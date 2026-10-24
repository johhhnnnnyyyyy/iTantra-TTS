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

test("Bengali expands safe numeric forms and segments sentences", () => {
  const result = normalizeForSpeech("  আমি 12 টাকায় আসব।  আবার দেখা হবে। ", "bn");
  assert.equal(result.normalizedText, "আমি বারো টাকায় আসব। আবার দেখা হবে।");
  assert.deepEqual(result.sentences, ["আমি বারো টাকায় আসব।", "আবার দেখা হবে।"]);
  assert.equal(result.rule, "indic-browser-v1");
});

test("every Indic route has native cardinal normalization and preserves unsafe forms", () => {
  for (const language of ["bn", "gu", "kn", "ml", "mr", "ta", "te", "or"]) {
    const normalized = normalizeForSpeech("₹25 at 05:30, 20%. 12.50 12/10/2026 1234", language);
    assert.notEqual(normalized.normalizedText, "₹25 at 05:30, 20%. 12.50 12/10/2026 1234");
    assert.doesNotMatch(normalized.normalizedText, /05:30/);
    assert.ok(normalized.warnings.includes("UNSAFE_DECIMAL_NUMBER_LEFT_UNCHANGED"));
    assert.ok(normalized.warnings.includes("AMBIGUOUS_DATE_LEFT_UNCHANGED"));
    assert.ok(normalized.warnings.includes("UNSAFE_NUMBER_LEFT_UNCHANGED"));
  }
});
