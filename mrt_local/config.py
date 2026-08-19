from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .core import DEFAULT_MODEL_NAME, ModelConfig, SamplingConfig


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_model_root() -> Path:
    value = os.environ.get("MRT_MODEL_ROOT")
    return Path(value).expanduser().resolve() if value else project_root() / "models"


def default_model_config() -> ModelConfig:
    return ModelConfig(name=DEFAULT_MODEL_NAME, root=default_model_root())


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    model: ModelConfig = field(default_factory=default_model_config)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
