from pathlib import Path

import itantra_tts
from itantra_tts import ModelAvailability, Precision


def test_package_exposes_model_catalog_for_precision_selection() -> None:
    assert hasattr(itantra_tts, "ModelCatalog")


def test_model_catalog_exposes_a_default_selection_matrix() -> None:
    assert hasattr(itantra_tts.ModelCatalog, "default")


def test_model_catalog_can_be_loaded_from_the_authoritative_registry() -> None:
    assert hasattr(itantra_tts.ModelCatalog, "from_yaml")


def test_model_catalog_exposes_profiles_for_a_language() -> None:
    assert hasattr(itantra_tts.ModelCatalog, "profiles_for")


def test_registry_offers_int8_and_int4_for_every_required_language() -> None:
    registry = Path(__file__).parents[2] / "models" / "registry.yaml"
    catalog = itantra_tts.ModelCatalog.from_yaml(registry)

    for language in ("hi", "gu", "mr", "kn", "ml", "ta", "te", "or", "bn", "en"):
        profiles = catalog.profiles_for(language)
        assert [profile.precision for profile in profiles] == [Precision.INT8, Precision.INT4]
        assert profiles[1].executable is False
        assert profiles[0].executable is True

    odia_int8 = catalog.profiles_for("or")[0]
    assert odia_int8.availability is ModelAvailability.RESEARCH_ONLY
    assert odia_int8.executable is True
    assert odia_int8.backend == "vits_onnx_static_qdq_int8_research"


def test_package_exposes_model_availability_for_a_precision_picker() -> None:
    assert hasattr(itantra_tts, "ModelAvailability")
