"""Create a research-only static-QDQ INT8 Odia candidate from FP32 ONNX.

The bundled five-line corpus reproduces the exploratory candidate only. A
release candidate requires a licensed, representative Odia calibration corpus
and native-speaker quality review.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

from itantra_tts.adapters.odia_tokenizer import OdiaMmsTokenizer


class OdiaCalibrationReader(CalibrationDataReader):
    """Feed the exact pinned MMS token IDs to ORT's calibration pass."""

    def __init__(self, corpus: Path, vocab: Path) -> None:
        self._tokenizer = OdiaMmsTokenizer(vocab)
        texts = [line.strip() for line in corpus.read_text(encoding="utf-8").splitlines()]
        self._texts = iter(text for text in texts if text and not text.startswith("#"))

    def get_next(self) -> dict[str, object] | None:
        text = next(self._texts, None)
        return None if text is None else {"input_ids": self._tokenizer.encode(text)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="FP32 MMS Odia ONNX graph")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--calibration-corpus", type=Path, required=True)
    args = parser.parse_args()
    reader = OdiaCalibrationReader(args.calibration_corpus, args.vocab)
    if reader.get_next() is None:
        parser.error("calibration corpus has no usable Odia sentences")
    reader = OdiaCalibrationReader(args.calibration_corpus, args.vocab)
    quantize_static(
        args.input,
        args.output,
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["Conv", "MatMul"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
