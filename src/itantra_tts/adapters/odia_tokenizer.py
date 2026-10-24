"""Small tokenizer for the pinned Odia MMS VITS vocabulary.

The pinned model has character tokens, ``normalize=True``, ``phonemize=False``,
and ``add_blank=True``. Keeping this logic local avoids importing the large
Transformers and PyTorch packages into the ONNX inference worker.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class OdiaMmsTokenizer:
    """Encode Odia text exactly as the pinned VitsTokenizer configuration does."""

    def __init__(self, vocab_path: Path) -> None:
        data = json.loads(vocab_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not all(
            isinstance(key, str) and len(key) == 1 and isinstance(value, int)
            for key, value in data.items()
        ):
            raise ValueError("Odia MMS vocabulary must contain character-to-id entries")
        self._vocab: dict[str, int] = data
        if self._vocab.get("ଛ") != 0:
            raise ValueError("Odia MMS blank token must have id zero")

    def encode(self, text: str) -> np.ndarray:
        """Return a [1, tokens] int64 array with blanks between characters."""
        normalized = "".join(
            character if character in self._vocab else character.lower()
            for character in text
        )
        filtered = "".join(
            character for character in normalized if character in self._vocab
        ).strip()
        if not filtered:
            raise ValueError("Odia text has no synthesizable vocabulary symbols")
        ids = [0] * (len(filtered) * 2 + 1)
        ids[1::2] = (self._vocab[character] for character in filtered)
        return np.asarray([ids], dtype=np.int64)
