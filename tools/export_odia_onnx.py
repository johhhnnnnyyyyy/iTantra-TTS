"""Export the pinned FP32 MMS Odia checkpoint as a research ONNX graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import VitsModel

from itantra_tts.adapters.odia_tokenizer import OdiaMmsTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = OdiaMmsTokenizer(args.checkpoint / "vocab.json")
    input_ids = torch.from_numpy(tokenizer.encode("ତୁମେ ଭଲ ଅଛ କି"))
    model = VitsModel.from_pretrained(args.checkpoint, local_files_only=True)
    model.eval()
    torch.onnx.export(
        model,
        (input_ids,),
        str(args.output),
        input_names=["input_ids"],
        output_names=["waveform", "sequence_lengths", "spectrogram"],
        dynamic_axes={"input_ids": {1: "tokens"}, "waveform": {1: "samples"}},
        opset_version=17,
        dynamo=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
