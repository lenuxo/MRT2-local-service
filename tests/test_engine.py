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
        self.embed_call = None
        self.generate_call = None

    def embed_style(
        self,
        prompt: str,
        *,
        pool_across_time: bool,
        use_mapper: bool,
        seed: int,
    ) -> str:
        self.embed_call = (prompt, pool_across_time, use_mapper, seed)
        return f"embedding:{prompt}"

    def generate(
        self,
        *,
        conditioning,
        cfg_scales,
        temperature: float,
        top_k: int,
        frames: int,
        state,
    ):
        self.generate_calls += 1
        assert state is None
        self.generate_call = (conditioning, cfg_scales, temperature, top_k, frames)
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
    assert backend.embed_call == ("ambient pads", True, True, 0)
    assert backend.generate_call == (
        {"musiccoca": "embedding:ambient pads"},
        {"musiccoca": 3.0, "notes": 1.0, "drums": 1.0},
        1.3,
        40,
        2,
    )
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


def test_generate_accepts_official_parameter_overrides(tmp_path: Path) -> None:
    backend = FakeBackend()
    engine = MrtEngine(
        prepared_config(tmp_path),
        backend_factory=lambda config: (backend, "musiccoca"),
    )
    engine.load()
    engine.generate(
        "jazz trio",
        0.04,
        temperature=0.9,
        top_k=12,
        cfg_musiccoca=2.5,
        cfg_notes=0.5,
        cfg_drums=0.25,
        seed=42,
        use_mapper=False,
        pool_across_time=False,
    )

    assert backend.embed_call == ("jazz trio", False, False, 42)
    assert backend.generate_call == (
        {"musiccoca": "embedding:jazz trio"},
        {"musiccoca": 2.5, "notes": 0.5, "drums": 0.25},
        0.9,
        12,
        1,
    )
