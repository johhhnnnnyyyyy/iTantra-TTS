import shutil
import subprocess
from pathlib import Path


def test_launcher_uses_project_python_from_any_directory(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "tools").mkdir(parents=True)
    (root / ".venv" / "bin").mkdir(parents=True)
    launcher = root / "tools" / "run_demo.sh"
    source = Path(__file__).resolve().parents[2] / "tools" / "run_demo.sh"
    shutil.copy(source, launcher)
    python = root / ".venv" / "bin" / "python"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$@"\n')
    python.chmod(0o755)
    result = subprocess.run(
        ["bash", str(launcher)], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert result.stdout.splitlines() == [str(root), "-m", "itantra_tts.web.demo"]


def test_launcher_reports_missing_virtual_environment(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "tools").mkdir(parents=True)
    source = Path(__file__).resolve().parents[2] / "tools" / "run_demo.sh"
    launcher = root / "tools" / "run_demo.sh"
    shutil.copy(source, launcher)
    result = subprocess.run(["bash", str(launcher)], capture_output=True, text=True)
    assert result.returncode != 0
    assert ".venv" in result.stderr
