#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="$project_root/.venv/bin/python"
if [[ ! -x "$python" ]]; then
    printf '%s\n' 'Project .venv not found. Follow the README setup instructions.' >&2
    exit 1
fi
cd "$project_root"
exec "$python" -m itantra_tts.web.demo
