from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf

from mrt_local.config import RuntimeConfig
from mrt_local.core import (
    GenerateCommand,
    ModelConfig,
    ResolvedGenerateCommand,
    SamplingConfig,
    SamplingOverrides,
)
from mrt_local.service import GenerationService
from mrt_local.encoding import encode_audio


class FakeBackend:
    def __init__(self) -> None:
        self.command: ResolvedGenerateCommand | None = None

    def generate(self, command: ResolvedGenerateCommand) -> np.ndarray:
        self.command = command
        return np.zeros((command.sample_count, 2), dtype=np.float32)


def prepared_config(tmp_path: Path) -> RuntimeConfig:
    model = ModelConfig(name="mrt2_small", root=tmp_path)
    model.model_dir.mkdir(parents=True)
    model.state_path.touch()
    model.model_path.touch()
    (model.resources_path / "musiccoca").mkdir(parents=True)
    return RuntimeConfig(model=model)


def test_load_once_and_generate_exact_duration(tmp_path: Path) -> None:
    backend = FakeBackend()
    factory_calls = 0

    def factory(config: RuntimeConfig) -> FakeBackend:
        nonlocal factory_calls
        factory_calls += 1
        return backend

    service = GenerationService(prepared_config(tmp_path), backend_factory=factory)
    service.load()
    service.load()
    result = service.generate(GenerateCommand(prompt="ambient pads", duration=0.05))

    assert factory_calls == 1
    assert backend.command == ResolvedGenerateCommand(
        prompt="ambient pads",
        duration=0.05,
        sampling=SamplingConfig(),
    )
    assert result.audio.shape == (2400, 2)
    assert result.audio.dtype == np.float32

    audio, sample_rate = sf.read(
        io.BytesIO(encode_audio(result).data),
        dtype="float32",
    )
    assert sample_rate == 48_000
    assert audio.shape == (2400, 2)


def test_missing_model_has_clear_error(tmp_path: Path) -> None:
    config = RuntimeConfig(model=ModelConfig(name="mrt2_base", root=tmp_path))
    service = GenerationService(config)
    try:
        service.load()
    except FileNotFoundError as exc:
        assert "mrt2_base.mlxfn" in str(exc)
        assert "musiccoca" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_generate_resolves_parameter_overrides(tmp_path: Path) -> None:
    backend = FakeBackend()
    service = GenerationService(
        prepared_config(tmp_path),
        backend_factory=lambda config: backend,
    )
    service.load()
    service.generate(
        GenerateCommand(
            prompt=" jazz trio ",
            duration=0.04,
            sampling=SamplingOverrides(
                temperature=0.9,
                top_k=12,
                cfg_musiccoca=2.5,
                cfg_notes=0.5,
                cfg_drums=0.25,
                seed=42,
                use_mapper=False,
                pool_across_time=False,
            ),
        )
    )

    assert backend.command == ResolvedGenerateCommand(
        prompt="jazz trio",
        duration=0.04,
        sampling=SamplingConfig(
            temperature=0.9,
            top_k=12,
            cfg_musiccoca=2.5,
            cfg_notes=0.5,
            cfg_drums=0.25,
            seed=42,
            use_mapper=False,
            pool_across_time=False,
        ),
    )
