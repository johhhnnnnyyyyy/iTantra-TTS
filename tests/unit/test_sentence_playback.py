import asyncio

import pytest

from itantra_tts import Precision, SynthesisResult


def test_sentence_audio_is_joined_in_order_with_silence() -> None:
    from itantra_tts.application.sentence_playback import synthesize_sentences

    async def synthesize(request):
        await asyncio.sleep(0.02 if request.sequence_no == 0 else 0)
        sample = b"\x01\x00" if request.sequence_no == 0 else b"\x02\x00"
        return SynthesisResult(request.stream_id, request.sequence_no, request.language,
                               request.precision, "test", 16000, sample)

    async def run():
        return await synthesize_sentences(synthesize, "a", ("First.", "Second."), "en", Precision.INT8, workers=2)

    output = asyncio.run(run())
    assert output.audio_pcm_s16le == b"\x01\x00" + bytes(960) + b"\x02\x00"
    assert [item.sequence_no for item in output.results] == [0, 1]


def test_sentence_failure_never_returns_partial_audio() -> None:
    from itantra_tts.application.sentence_playback import SentenceSynthesisError, synthesize_sentences

    async def synthesize(request):
        if request.sequence_no == 1:
            raise RuntimeError("secret input")
        return SynthesisResult(request.stream_id, request.sequence_no, request.language,
                               request.precision, "test", 16000, b"\x01\x00")

    async def run():
        with pytest.raises(SentenceSynthesisError, match="failure") as error:
            await synthesize_sentences(synthesize, "a", ("First.", "Second."), "en", Precision.INT8)
        assert "secret input" not in str(error.value)

    asyncio.run(run())
