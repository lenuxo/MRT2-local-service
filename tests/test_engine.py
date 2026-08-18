from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from mrt_local.config import EngineConfig
from mrt_local.engine import MrtEngine


class FakeBackend:
    def __init__(self) -> None:
        self.generate_calls = 0

    def embed_style(self, prompt: str, *, use_mapper: bool) -> str:
        assert use_mapper is True
        return f"embedding:{prompt}"

    def generate(self, *, conditioning, frames: int, state):
        self.generate_calls += 1
        assert state is None
        samples = np.zeros((frames * 1920, 2), dtype=np.float32)
        return SimpleNamespace(samples=samples), []


def prepared_config(tmp_path: Path) -> EngineConfig:
    config = EngineConfig(model="mrt2_small", model_root=tmp_path)
    config.model_path.mkdir(parents=True)
    config.state_path.touch()
    (config.resources_path / "musiccoca").mkdir(parents=True)
    return config


def test_load_once_and_generate_exact_duration(tmp_path: Path) -> None:
    backend = FakeBackend()
    factory_calls = 0

    def factory(config: EngineConfig):
        nonlocal factory_calls
        factory_calls += 1
        return backend, "musiccoca"

    engine = MrtEngine(prepared_config(tmp_path), backend_factory=factory)
    engine.load()
    engine.load()
    result = engine.generate("ambient pads", 0.05)

    assert factory_calls == 1
    assert backend.generate_calls == 1
    assert result.audio.shape == (2400, 2)
    assert result.audio.dtype == np.float32

    audio, sample_rate = sf.read(io.BytesIO(result.to_wav_bytes()), dtype="float32")
    assert sample_rate == 48_000
    assert audio.shape == (2400, 2)


def test_missing_model_has_clear_error(tmp_path: Path) -> None:
    engine = MrtEngine(EngineConfig(model="mrt2_base", model_root=tmp_path))
    try:
        engine.load()
    except FileNotFoundError as exc:
        assert "mrt2_base.mlxfn" in str(exc)
        assert "musiccoca" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")
