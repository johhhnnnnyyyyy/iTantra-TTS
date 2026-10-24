"""One local synthesis seam which never retains both runtime families at once."""

from __future__ import annotations

from typing import Protocol

from itantra_tts.domain import SynthesisRequest, SynthesisResult, SynthesisFailure
from itantra_tts.ports import SynthesisBackend


class _ReleasableSynthesisBackend(SynthesisBackend, Protocol):
    def clear_cache(self) -> None: ...


class LocalResearchSynthesisBackend:
    """Route by language and evict the inactive model family to bound process RSS."""

    def __init__(
        self,
        fastpitch_hifigan: _ReleasableSynthesisBackend,
        odia_vits: _ReleasableSynthesisBackend,
    ) -> None:
        self._fastpitch_hifigan = fastpitch_hifigan
        self._odia_vits = odia_vits

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult | SynthesisFailure:
        if request.language == "or":
            self._fastpitch_hifigan.clear_cache()
            return self._odia_vits.synthesize(request)
        self._odia_vits.clear_cache()
        return self._fastpitch_hifigan.synthesize(request)
