from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import numpy as np

ModelName = Literal["mrt2_small", "mrt2_base"]
SUPPORTED_MODELS: tuple[ModelName, ...] = ("mrt2_small", "mrt2_base")
DEFAULT_MODEL_NAME: ModelName = "mrt2_small"
SAMPLE_RATE = 48_000
CHANNELS = 2
MAX_DURATION = 300.0
DEFAULT_DURATION = 10.0
DEFAULT_WARMUP_STEPS = 5


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: ModelName
    root: Path
    warmup_steps: int = DEFAULT_WARMUP_STEPS

    def validate(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps 必须大于等于 0")

    @property
    def model_dir(self) -> Path:
        return self.root / "models" / self.name

    @property
    def model_path(self) -> Path:
        return self.model_dir / f"{self.name}.mlxfn"

    @property
    def state_path(self) -> Path:
        return self.model_dir / f"{self.name}_state.safetensors"

    @property
    def resources_path(self) -> Path:
        return self.root / "resources"


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    temperature: float = 1.3
    top_k: int = 40
    cfg_musiccoca: float = 3.0
    cfg_notes: float = 1.0
    cfg_drums: float = 1.0
    seed: int = 0
    use_mapper: bool = True
    pool_across_time: bool = True

    def validate(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature 必须是大于 0 的有限数")
        if self.top_k < 1:
            raise ValueError("top_k 必须大于等于 1")
        if not all(
            math.isfinite(value)
            for value in (self.cfg_musiccoca, self.cfg_notes, self.cfg_drums)
        ):
            raise ValueError("CFG 参数必须是有限数")

    @property
    def cfg_scales(self) -> dict[str, float]:
        return {
            "musiccoca": self.cfg_musiccoca,
            "notes": self.cfg_notes,
            "drums": self.cfg_drums,
        }


@dataclass(frozen=True, slots=True)
class SamplingOverrides:
    temperature: float | None = None
    top_k: int | None = None
    cfg_musiccoca: float | None = None
    cfg_notes: float | None = None
    cfg_drums: float | None = None
    seed: int | None = None
    use_mapper: bool | None = None
    pool_across_time: bool | None = None

    def resolve(self, defaults: SamplingConfig) -> SamplingConfig:
        values = {
            field.name: (
                getattr(defaults, field.name)
                if getattr(self, field.name) is None
                else getattr(self, field.name)
            )
            for field in fields(self)
        }
        resolved = SamplingConfig(**values)
        resolved.validate()
        return resolved


@dataclass(frozen=True, slots=True)
class GenerateCommand:
    prompt: str
    duration: float = DEFAULT_DURATION
    sampling: SamplingOverrides = SamplingOverrides()

    def resolve(self, defaults: SamplingConfig) -> ResolvedGenerateCommand:
        prompt = self.prompt.strip()
        if not prompt:
            raise ValueError("prompt 不能为空")
        if (
            not math.isfinite(self.duration)
            or self.duration <= 0
            or self.duration > MAX_DURATION
        ):
            raise ValueError("duration 必须大于 0 且不超过 300 秒")
        return ResolvedGenerateCommand(
            prompt=prompt,
            duration=self.duration,
            sampling=self.sampling.resolve(defaults),
        )


@dataclass(frozen=True, slots=True)
class ResolvedGenerateCommand:
    prompt: str
    duration: float
    sampling: SamplingConfig

    @property
    def sample_count(self) -> int:
        return round(self.duration * SAMPLE_RATE)


@dataclass(frozen=True, slots=True)
class GenerateResult:
    sample_rate: int
    channels: int
    audio: np.ndarray
