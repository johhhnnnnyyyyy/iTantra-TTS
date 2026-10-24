"""Truthful model/precision selection for callers and future UIs."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping

import yaml

from .contracts import Precision, SynthesisRequest


class ModelAvailability(StrEnum):
    """Whether a requested model can be selected for local execution."""

    PENDING_ARTIFACT_VALIDATION = "pending_artifact_validation"
    PENDING_INT4_CONVERSION = "pending_int4_conversion"
    RESEARCH_ONLY = "research_only"
    EXECUTABLE = "executable"


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """One language/precision model choice shown to callers and UIs."""

    language: str
    precision: Precision
    backend: str
    source: str
    source_revision: str
    upstream_license: str
    availability: ModelAvailability
    executable: bool


class ModelCatalog:
    """Selects only model artifacts whose executable status is known."""

    def __init__(self, profiles: Mapping[str, tuple[ModelProfile, ...]] | None = None) -> None:
        self._profiles = dict(profiles or {})

    @classmethod
    def default(cls) -> "ModelCatalog":
        """Return the planned SIH26173 language/precision matrix."""
        return cls()

    @classmethod
    def from_yaml(cls, registry_path: Path) -> "ModelCatalog":
        """Load the catalog from the one authoritative model registry."""
        with registry_path.open(encoding="utf-8") as registry_file:
            document = yaml.safe_load(registry_file)
        if not isinstance(document, dict):
            raise ValueError("model registry must be a mapping")

        languages = document.get("languages")
        precision_profiles = document.get("precision_profiles")
        if not isinstance(languages, dict) or not isinstance(precision_profiles, dict):
            raise ValueError("model registry must declare languages and precision_profiles")

        profiles: dict[str, tuple[ModelProfile, ...]] = {}
        for language in SynthesisRequest.SUPPORTED_LANGUAGES:
            language_entry = languages.get(language)
            if not isinstance(language_entry, dict):
                raise ValueError(f"model registry has no {language!r} entry")
            language_profiles: list[ModelProfile] = []
            for precision in (Precision.INT8, Precision.INT4):
                precision_entry = precision_profiles.get(precision.value)
                if not isinstance(precision_entry, dict):
                    raise ValueError(f"model registry has no {precision.value!r} entry")
                availability = ModelAvailability(precision_entry["availability"])
                executable = bool(precision_entry["executable"])
                if language == "or" and precision is Precision.INT8:
                    availability = ModelAvailability.RESEARCH_ONLY
                language_profiles.append(
                    ModelProfile(
                        language=language,
                        precision=precision,
                        backend=str(language_entry["backend"]),
                        source=str(language_entry["source"]),
                        source_revision=str(language_entry["source_revision"]),
                        upstream_license=str(language_entry["upstream_license"]),
                        availability=availability,
                        executable=executable,
                    )
                )
            profiles[language] = tuple(language_profiles)
        return cls(profiles)

    def profiles_for(self, language: str) -> tuple[ModelProfile, ...]:
        """Return profiles offered to a language/precision picker."""
        return self._profiles.get(language, ())
