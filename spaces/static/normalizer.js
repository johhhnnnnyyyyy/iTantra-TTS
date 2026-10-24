const RULE_VERSION = "w2-browser-v1";

const HINDI_UNDER_HUNDRED = [
  "शून्य", "एक", "दो", "तीन", "चार", "पांच", "छह", "सात", "आठ", "नौ", "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस",
  "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस", "तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस",
  "चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास", "पचास", "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",
  "साठ", "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर", "सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उन्नासी",
  "अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सतासी", "अट्ठासी", "नवासी", "नब्बे", "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पंचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे",
];
const ENGLISH_UNDER_TWENTY = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
const ENGLISH_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

const englishNumber = (value) => {
  const number = Number(value);
  if (number < 20) return ENGLISH_UNDER_TWENTY[number];
  if (number < 100) return ENGLISH_TENS[Math.floor(number / 10)] + (number % 10 ? ` ${ENGLISH_UNDER_TWENTY[number % 10]}` : "");
  if (number < 1000) return `${ENGLISH_UNDER_TWENTY[Math.floor(number / 100)]} hundred${number % 100 ? ` ${englishNumber(number % 100)}` : ""}`;
  return `${ENGLISH_UNDER_TWENTY[Math.floor(number / 1000)]} thousand${number % 1000 ? ` ${englishNumber(number % 1000)}` : ""}`;
};
const hindiNumber = (value) => {
  const number = Number(value);
  if (number < 100) return HINDI_UNDER_HUNDRED[number];
  if (number < 1000) return `${HINDI_UNDER_HUNDRED[Math.floor(number / 100)]} सौ${number % 100 ? ` ${hindiNumber(number % 100)}` : ""}`;
  return `${HINDI_UNDER_HUNDRED[Math.floor(number / 1000)]} हजार${number % 1000 ? ` ${hindiNumber(number % 1000)}` : ""}`;
};
const isSafeNumber = (value) => !(value.length > 1 && value.startsWith("0")) && Number(value) <= 9999;
const replaceSafeNumbers = (text, words, warnings) => text.replace(/\b\d+\b/g, (value) => {
  if (!isSafeNumber(value)) { warnings.push("UNSAFE_NUMBER_LEFT_UNCHANGED"); return value; }
  return words(value);
});

const protectUnsafeForms = (input, warnings) => {
  const entries = [];
  let text = input;
  for (const [pattern, warning] of [
    [/\b\d{1,2}\/\d{1,2}\/\d{2,4}\b/g, "AMBIGUOUS_DATE_LEFT_UNCHANGED"],
    [/₹\d+\.\d+/g, "UNSAFE_DECIMAL_CURRENCY_LEFT_UNCHANGED"],
    [/\b\d+\.\d+\b/g, "UNSAFE_DECIMAL_NUMBER_LEFT_UNCHANGED"],
  ]) {
    text = text.replace(pattern, (match) => { const token = String.fromCharCode(0xE000 + entries.length); entries.push([token, match]); warnings.push(warning); return token; });
  }
  return { text, entries };
};
const restoreUnsafeForms = (text, entries) => entries.reduce((result, [token, value]) => result.replace(token, value), text);

const normalizeEnglish = (text, warnings) => {
  let result = text.replaceAll("Dr.", "Doctor").replaceAll("Mr.", "Mister").replaceAll("Mrs.", "Missus");
  result = result.replace(/₹(\d+)\b/g, (_, amount) => `${englishNumber(amount)} rupees`);
  result = result.replace(/\b([01]?\d|2[0-3]):([0-5]\d)\b/g, (_, hour, minute) => Number(minute) === 0 ? `${englishNumber(hour)} o'clock` : `${englishNumber(hour)} ${englishNumber(minute)}`);
  result = result.replace(/\b(\d+)%/g, (_, amount) => `${englishNumber(amount)} percent`);
  return replaceSafeNumbers(result, englishNumber, warnings).replace(/\b[A-Z]{2,}\b/g, (value) => [...value].join(" "));
};
const normalizeHindi = (text, warnings) => {
  let result = text.replace(/₹(\d+)\b/g, (_, amount) => `${hindiNumber(amount)} रुपये`);
  result = result.replace(/\b([01]?\d|2[0-3]):([0-5]\d)\b/g, (_, hour, minute) => Number(minute) === 0 ? `${hindiNumber(hour)} बजे` : `${hindiNumber(hour)} बजकर ${hindiNumber(minute)} मिनट`);
  result = result.replace(/\b(\d+)%/g, (_, amount) => `${hindiNumber(amount)} प्रतिशत`);
  return replaceSafeNumbers(result, hindiNumber, warnings);
};

export const normalizeForSpeech = (input, language) => {
  if (typeof input !== "string" || !input.trim()) throw new Error("text must contain at least one non-whitespace character");
  let text = input.normalize("NFKC").replace(/\s+/g, " ").trim();
  const warnings = [];
  let rule = "browser-cleanup-v1";
  if (language === "en" || language === "hi") {
    const protectedForms = protectUnsafeForms(text, warnings);
    text = language === "en" ? normalizeEnglish(protectedForms.text, warnings) : normalizeHindi(protectedForms.text, warnings);
    text = restoreUnsafeForms(text, protectedForms.entries);
    rule = RULE_VERSION;
  } else if (/\d/.test(text)) warnings.push("NUMERIC_FORM_LEFT_UNCHANGED");
  text = text.replace(/\s+([,.;:!?।])/g, "$1");
  const sentences = text.split(/(?<=[.!?।])\s+/).map((sentence) => sentence.trim()).filter(Boolean);
  return Object.freeze({ originalText: input, normalizedText: text, sentences: Object.freeze(sentences), rule, warnings: Object.freeze(warnings) });
};
