"""Stage a Hugging Face Space with checksum-verified local model files."""

import argparse
from pathlib import Path

from itantra_tts.space_package import stage_space

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("destination", type=Path, help="new or empty staging directory")
parser.add_argument("--include-models", action="store_true", help="include pinned model files")
parser.add_argument("--include-w4a8", action="store_true",
                    help="include W4A8 models and the Linux x86-64 custom operator")
args = parser.parse_args()
stage_space(Path(__file__).resolve().parents[1], args.destination,
            include_models=args.include_models, include_w4a8=args.include_w4a8)
print(f"Space staged at {args.destination}")
