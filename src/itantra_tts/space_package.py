"""Prepare a reviewable Space tree with optional pinned research model files."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_static_space(project_root: Path, destination: Path) -> None:
    """Copy the browser-only companion template without local runtime artifacts."""
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("refusing to overwrite an existing Static Space directory")
    destination.mkdir(parents=True, exist_ok=True)
    template = project_root / "spaces" / "static"
    for filename in ("README.md", "index.html", "app.js", "styles.css"):
        source = template / filename
        if not source.is_file():
            raise FileNotFoundError(f"missing Static Space template asset: {filename}")
        shutil.copy2(source, destination / filename)


def stage_space(
    project_root: Path,
    destination: Path,
    *,
    include_models: bool = False,
    include_w4a8: bool = False,
) -> None:
    """Stage only code and provenance; deployment and model rights remain separate gates."""
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("refusing to overwrite an existing Space directory")
    if include_w4a8 and not include_models:
        raise ValueError("W4A8 staging requires the shared INT8 tokenizer assets")
    destination.mkdir(parents=True, exist_ok=True)
    template = project_root / "spaces" / "demo"
    for filename in ("app.py", "requirements.txt", "README.md"):
        shutil.copy2(template / filename, destination / filename)
    for filename in ("pyproject.toml", "LICENSE"):
        shutil.copy2(project_root / filename, destination / filename)
    shutil.copytree(project_root / "src" / "itantra_tts", destination / "src" / "itantra_tts",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (destination / "models").mkdir(exist_ok=True)
    for filename in ("registry.yaml", "artifacts.lock.yaml"):
        shutil.copy2(project_root / "models" / filename, destination / "models" / filename)
    if include_models:
        lock = yaml.safe_load((project_root / "models/artifacts.lock.yaml").read_text())
        for entry, directory in (("echobharat_int8", "echobharat-int8"),
                                 ("mms_tts_ory_research", "mms-tts-ory-int8-research")):
            for name, expected in lock["artifacts"][entry]["local_file_sha256"].items():
                source = project_root / "models/artifacts" / directory / name
                if _sha256(source) != expected:
                    raise ValueError(f"Model checksum mismatch: {directory}/{name}")
                target = destination / "models/artifacts" / directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    if include_w4a8:
        for script in ("unified_research_web.py", "w4a8_research_demo.py"):
            source = project_root / "tools" / script
            target = destination / "tools" / script
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        library = project_root / "build" / "w4a8" / "libitantra_w4a8.so"
        if not library.is_file():
            raise FileNotFoundError("missing Linux x86-64 W4A8 custom-op library")
        target_library = destination / "build" / "w4a8" / library.name
        target_library.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(library, target_library)
        staged_files: dict[str, str] = {}
        selected_models = [
            ("experimental-w4a8-per-channel", language, f"fastpitch-{language}.w4a8.onnx")
            for language in ("bn", "en", "gu", "hi", "kn", "ml", "mr", "ta", "te")
        ]
        selected_models += [
            ("experimental-w4a8-per-channel", language, f"hifigan-{language}.w4a8.onnx")
            for language in ("bn", "en", "gu", "hi", "kn", "ml", "mr", "ta", "te")
        ]
        selected_models.append(("experimental-w4a8-scalar-20261021", "or", "model.w4a8.onnx"))
        for pack, language, name in selected_models:
            source = project_root / "models" / "artifacts" / pack / language / name
            if not source.is_file():
                raise FileNotFoundError(f"missing selected W4A8 model: {pack}/{language}/{name}")
            target = destination / "models" / "artifacts" / pack / language / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            staged_files[str(target.relative_to(destination))] = _sha256(target)
        staged_files[str(target_library.relative_to(destination))] = _sha256(target_library)
        (destination / "models" / "w4a8-stage-manifest.json").write_text(
            json.dumps({"variant": "W4A8", "sha256": staged_files}, indent=2, sort_keys=True) + "\n"
        )
