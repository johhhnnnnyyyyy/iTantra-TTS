import importlib.util
import hashlib
import shutil
import sys
from pathlib import Path

from itantra_tts.space_package import stage_space
from itantra_tts.web.demo import build_demo, WebSynthesisController
from itantra_tts.domain import ModelCatalog
from itantra_tts.adapters.fake_synthesis import FakeSynthesisBackend


def _small_release_root(tmp_path: Path) -> Path:
    """Build the smallest complete release fixture without copying model gigabytes."""
    source_root = Path(__file__).resolve().parents[2]
    root = tmp_path / "source"
    (root / "spaces" / "demo").mkdir(parents=True)
    for filename in ("app.py", "requirements.txt", "README.md"):
        shutil.copy2(source_root / "spaces" / "demo" / filename, root / "spaces" / "demo" / filename)
    for filename in ("pyproject.toml", "LICENSE"):
        shutil.copy2(source_root / filename, root / filename)
    shutil.copytree(source_root / "src" / "itantra_tts", root / "src" / "itantra_tts")
    (root / "tools").mkdir()
    for filename in ("unified_research_web.py", "w4a8_research_demo.py"):
        shutil.copy2(source_root / "tools" / filename, root / "tools" / filename)

    int8 = root / "models" / "artifacts" / "echobharat-int8" / "en" / "tokens.json"
    odia = root / "models" / "artifacts" / "mms-tts-ory-int8-research" / "vocab.json"
    int8.parent.mkdir(parents=True)
    odia.parent.mkdir(parents=True)
    int8.write_bytes(b"int8-token")
    odia.write_bytes(b"odia-vocab")
    (root / "models" / "registry.yaml").write_text("version: 1\n")
    (root / "models" / "artifacts.lock.yaml").write_text(
        "artifacts:\n"
        "  echobharat_int8:\n"
        "    local_file_sha256:\n"
        f"      en/tokens.json: {hashlib.sha256(int8.read_bytes()).hexdigest()}\n"
        "  mms_tts_ory_research:\n"
        "    local_file_sha256:\n"
        f"      vocab.json: {hashlib.sha256(odia.read_bytes()).hexdigest()}\n"
    )
    for language in ("bn", "en", "gu", "hi", "kn", "ml", "mr", "ta", "te"):
        for name in (f"fastpitch-{language}.w4a8.onnx", f"hifigan-{language}.w4a8.onnx"):
            model = root / "models" / "artifacts" / "experimental-w4a8-per-channel" / language / name
            model.parent.mkdir(parents=True, exist_ok=True)
            model.write_bytes(b"w4a8-model")
    for relative in ("or/model.w4a8.onnx", "hi/fastpitch-hi.w4a8.onnx"):
        model = root / "models" / "artifacts" / "experimental-w4a8-scalar-20261021" / relative
        model.parent.mkdir(parents=True)
        model.write_bytes(b"w4a8-model")
    library = root / "build" / "w4a8" / "libitantra_w4a8.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"w4a8-library")
    return root


def test_space_stage_contains_core_but_never_copies_local_model_artifacts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    destination = tmp_path / "space"
    stage_space(root, destination)
    assert (destination / "app.py").exists()
    assert (destination / "src/itantra_tts/web/demo.py").exists()
    assert (destination / "models/registry.yaml").exists()
    assert "sdk: gradio" in (destination / "README.md").read_text()
    assert "-e .[demo]" in (destination / "requirements.txt").read_text()
    assert not (destination / "models/artifacts").exists()


def test_space_stage_refuses_nonempty_destination(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    destination = tmp_path / "space"
    destination.mkdir()
    (destination / "keep.txt").write_text("user data")
    try:
        stage_space(root, destination)
    except FileExistsError:
        pass
    else:
        raise AssertionError("staging must not overwrite a nonempty destination")
    assert (destination / "keep.txt").read_text() == "user data"


def test_full_research_space_stage_contains_on_demand_w4a8_runtime(tmp_path: Path) -> None:
    """Catch a release package that advertises W4A8 without its executable chain."""
    root = _small_release_root(tmp_path)
    destination = tmp_path / "space"

    stage_space(root, destination, include_models=True, include_w4a8=True)

    assert (destination / "tools" / "unified_research_web.py").is_file()
    assert (destination / "tools" / "w4a8_research_demo.py").is_file()
    assert (destination / "build" / "w4a8" / "libitantra_w4a8.so").is_file()
    assert (destination / "models" / "artifacts" / "experimental-w4a8-per-channel" / "hi"
            / "fastpitch-hi.w4a8.onnx").is_file()
    assert (destination / "models" / "artifacts" / "experimental-w4a8-scalar-20261021" / "or"
            / "model.w4a8.onnx").is_file()
    assert not (destination / "models" / "artifacts" / "experimental-w4a8-scalar-20261021" / "hi"
                / "fastpitch-hi.w4a8.onnx").exists()


def test_full_research_space_app_offers_int4_selection(tmp_path: Path) -> None:
    """Catch a staged public UI that silently leaves W4A8 unavailable."""
    root = _small_release_root(tmp_path)
    destination = tmp_path / "space"
    stage_space(root, destination, include_models=True, include_w4a8=True)
    spec = importlib.util.spec_from_file_location("staged_space_app", destination / "app.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(destination))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(destination))
    config = module.demo.get_config_file()
    precision = next(
        component for component in config["components"]
        if component.get("props", {}).get("label") == "Precision"
    )
    assert {choice[0] for choice in precision["props"]["choices"]} == {"int8", "int4"}


def test_hosted_demo_discloses_cloud_execution_and_retention() -> None:
    demo = build_demo(WebSynthesisController(ModelCatalog.default(), FakeSynthesisBackend()),
                      hosted=True)
    rendered = str(demo.config)
    assert "cloud server" in rendered
    assert "temporary playback files" in rendered
    assert "not the offline Android" in rendered
