from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ModelName = Literal["mrt2_small", "mrt2_base"]
SUPPORTED_MODELS: tuple[ModelName, ...] = ("mrt2_small", "mrt2_base")
SAMPLE_RATE = 48_000
CHANNELS = 2


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_model_root() -> Path:
    value = os.environ.get("MRT_MODEL_ROOT")
    return Path(value).expanduser().resolve() if value else project_root() / "models"


@dataclass(frozen=True, slots=True)
class EngineConfig:
    model: ModelName = "mrt2_small"
    model_root: Path = default_model_root()

    @property
    def model_dir(self) -> Path:
        return self.model_root / "models" / self.model

    @property
    def model_path(self) -> Path:
        return self.model_dir / f"{self.model}.mlxfn"

    @property
    def state_path(self) -> Path:
        return self.model_dir / f"{self.model}_state.safetensors"

    @property
    def resources_path(self) -> Path:
        return self.model_root / "resources"
