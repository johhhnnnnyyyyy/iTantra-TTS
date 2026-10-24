"""Conservative experimental cardinal expansion, not native-speaker approval."""
import asyncio

import pytest

from itantra_tts.application.service import SynthesisService
from itantra_tts.domain import Precision, SynthesisRequest, SynthesisResult

pytest.importorskip("indic_numtowords")


class Backend:
    async def synthesize(self, request):
        return SynthesisResult(request.stream_id, request.sequence_no, request.language,
                               request.precision, "fixture", 16000, b"\x01\x00" * 160)


@pytest.mark.parametrize(("language", "expected"), [
    ("bn", "পাঁচ"), ("gu", "પાંચ"), ("kn", "ಐದು"), ("ml", "അഞ്ച്"),
    ("mr", "पाच"), ("or", "ପାଞ୍ଚ"), ("ta", "ஐந்து"), ("te", "ఐదు"),
])
def test_plain_cardinal_is_expanded_in_requested_language(language, expected):
    result = asyncio.run(SynthesisService(Backend()).synthesize(
        SynthesisRequest("s", 0, "5", language, Precision.INT8)))
    assert result.spoken_text == expected
    assert "experimental" in result.normalization.lower()


@pytest.mark.parametrize("text", [
    "5:30", "1/2/2026", "5%", "005", "123", "1 2 3", "5 pm", "5 A.M.",
    "₹2.50", "1.2.3", "3.5%", "3.5°C", "₹200-300",
])
def test_ambiguous_numeric_forms_remain_unchanged_with_warning(text):
    result = asyncio.run(SynthesisService(Backend()).synthesize(
        SynthesisRequest("s", 0, text, "ta", Precision.INT8)))
    assert result.spoken_text == text
    assert "UNNORMALIZED_NUMERIC_FORM" in result.normalization


@pytest.mark.parametrize(("language", "decimal", "currency"), [
    ("bn", "তিন দশমিক পাঁচ", "দুশো টাকা"),
    ("gu", "ત્રણ દશાંશ પાંચ", "બસો રૂપિયા"),
    ("kn", "ಮೂರು ದಶಮಾಂಶ ಐದು", "ಇನ್ನೂರು ರೂಪಾಯಿ"),
    ("ml", "മൂന്ന് ദശാംശം അഞ്ച്", "ഇരുന്നൂറ് രൂപ"),
    ("mr", "तीन दशांश पाच", "दोनशे रुपये"),
    ("or", "ତିନି ଦଶମିକ ପାଞ୍ଚ", "ଦୁଇ ଶହ ଟଙ୍କା"),
    ("ta", "மூன்று புள்ளி ஐந்து", "இருநூறு ரூபாய்"),
    ("te", "మూడు దశాంశ ఐదు", "రెండు వందలు రూపాయలు"),
])
def test_standalone_decimal_and_rupee_expressions_are_expanded_experimentally(
    language, decimal, currency
):
    for value, expected in [("3.5", decimal), ("₹200", currency)]:
        result = asyncio.run(SynthesisService(Backend()).synthesize(
            SynthesisRequest("s", 0, value, language, Precision.INT8)))
        assert result.spoken_text == expected
        assert "native-speaker review pending" in result.normalization


@pytest.mark.parametrize(("text", "expected"), [
    ("விலை 3.5", "விலை மூன்று புள்ளி ஐந்து"),
    ("தொகை ₹200", "தொகை இருநூறு ரூபாய்"),
])
def test_supported_decimal_and_rupee_expressions_inside_text(text, expected):
    result = asyncio.run(SynthesisService(Backend()).synthesize(
        SynthesisRequest("s", 0, text, "ta", Precision.INT8)))
    assert result.spoken_text == expected
    assert "native-speaker review pending" in result.normalization


def test_existing_private_use_character_is_preserved_when_expanding_decimal():
    result = asyncio.run(SynthesisService(Backend()).synthesize(
        SynthesisRequest("s", 0, "\ue000 3.5", "ta", Precision.INT8)))
    assert result.spoken_text == "\ue000 மூன்று புள்ளி ஐந்து"
