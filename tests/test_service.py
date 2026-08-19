from __future__ import annotations

import io
from pathlib import Path
import threading

import numpy as np
import soundfile as sf

from mrt_local.config import RuntimeConfig
from mrt_local.core import (
    AudioInput,
    GenerateCommand,
    ModelConfig,
    ResolvedGenerateCommand,
    SamplingConfig,
    SamplingOverrides,
    StreamGenerateCommand,
)
from mrt_local.service import GenerationService, ModelBusyError
from mrt_local.encoding import encode_audio


class FakeBackend:
    def __init__(self) -> None:
        self.command: ResolvedGenerateCommand | None = None

    def generate(self, command: ResolvedGenerateCommand) -> np.ndarray:
        self.command = command
        return np.zeros((command.sample_count, 2), dtype=np.float32)

    def open_stream(self, command):
        return FakeBackendStream()


class FakeBackendStream:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self.closed = False

    def generate_chunk(self, frames: int) -> np.ndarray:
        self.calls.append(frames)
        return np.zeros((frames * 1_920, 2), np.float32)

    def close(self) -> None:
        self.closed = True


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
        reference_audio=None,
        text_weight=1.0,
        audio_weight=0.0,
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
        reference_audio=None,
        text_weight=1.0,
        audio_weight=0.0,
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


def test_generate_accepts_and_normalizes_text_audio_mix(tmp_path: Path) -> None:
    backend = FakeBackend()
    service = GenerationService(
        prepared_config(tmp_path),
        backend_factory=lambda config: backend,
    )
    service.load()
    reference = AudioInput(np.zeros((480, 2), np.float32), 48_000)

    service.generate(GenerateCommand(reference_audio=reference, duration=0.01))
    assert backend.command is not None
    assert backend.command.prompt is None
    assert backend.command.reference_audio is reference
    assert (backend.command.text_weight, backend.command.audio_weight) == (0.0, 1.0)

    service.generate(GenerateCommand(
        prompt="ambient", reference_audio=reference,
        text_weight=1, audio_weight=3, duration=0.01,
    ))
    assert backend.command is not None
    assert (backend.command.text_weight, backend.command.audio_weight) == (0.25, 0.75)

    for command in (
        GenerateCommand(duration=0.01),
        GenerateCommand(prompt="x", text_weight=0, duration=0.01),
        GenerateCommand(
            prompt="x", reference_audio=reference, duration=0.01,
            sampling=SamplingOverrides(pool_across_time=False),
        ),
    ):
        try:
            service.generate(command)
        except ValueError as exc:
            assert str(exc)
        else:
            raise AssertionError("expected ValueError")


def test_stream_chunks_trim_duration_and_hold_exclusive_lease(tmp_path: Path) -> None:
    backend = FakeBackend()
    service = GenerationService(
        prepared_config(tmp_path), backend_factory=lambda config: backend
    )
    service.load()
    session = service.open_stream(StreamGenerateCommand(
        prompt="ambient", duration=0.21, chunk_frames=5
    ))

    first = session.next_chunk()
    assert first is not None
    assert first.sequence == 0
    assert first.start_sample == 0
    assert first.audio.shape == (9_600, 2)
    try:
        service.generate(GenerateCommand(prompt="busy", duration=0.01))
    except ModelBusyError:
        pass
    else:
        raise AssertionError("expected ModelBusyError")

    second = session.next_chunk()
    assert second is not None
    assert second.sequence == 1
    assert second.start_sample == 9_600
    assert second.audio.shape == (480, 2)
    assert session.next_chunk() is None
    session.close()

    service.generate(GenerateCommand(prompt="released", duration=0.01))


def test_all_backend_operations_use_one_dedicated_thread(tmp_path: Path) -> None:
    thread_ids: list[int] = []

    class AffineStream(FakeBackendStream):
        def generate_chunk(self, frames: int) -> np.ndarray:
            thread_ids.append(threading.get_ident())
            return super().generate_chunk(frames)

        def close(self) -> None:
            thread_ids.append(threading.get_ident())
            super().close()

    class AffineBackend(FakeBackend):
        def generate(self, command: ResolvedGenerateCommand) -> np.ndarray:
            thread_ids.append(threading.get_ident())
            return super().generate(command)

        def open_stream(self, command):
            thread_ids.append(threading.get_ident())
            return AffineStream()

    backend = AffineBackend()

    def factory(config: RuntimeConfig) -> AffineBackend:
        thread_ids.append(threading.get_ident())
        return backend

    caller_thread = threading.get_ident()
    service = GenerationService(prepared_config(tmp_path), backend_factory=factory)
    service.load()
    service.generate(GenerateCommand(prompt="ambient", duration=0.04))
    session = service.open_stream(StreamGenerateCommand(
        prompt="ambient", duration=0.04, chunk_frames=1
    ))
    session.next_chunk()
    session.close()
    service.close()

    assert len(set(thread_ids)) == 1
    assert thread_ids[0] != caller_thread
