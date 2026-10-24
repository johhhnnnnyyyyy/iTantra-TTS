"""Summarize a measured W6 sample JSON file without inventing missing evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from itantra_tts.evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement_file", type=Path)
    args = parser.parse_args()
    data = json.loads(args.measurement_file.read_text(encoding="utf-8"))
    print(json.dumps(evaluate(data), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
