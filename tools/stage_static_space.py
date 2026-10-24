"""Stage the source-only browser companion for a Hugging Face Static Space."""

import argparse
from pathlib import Path

from itantra_tts.space_package import stage_static_space


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("destination", type=Path, help="new or empty staging directory")
args = parser.parse_args()

stage_static_space(Path(__file__).resolve().parents[1], args.destination)
print(f"Static Space staged at {args.destination}")
