from types import SimpleNamespace

import numpy as np

from mrt_local.backend import MagentaMlxBackend
from mrt_local.core import AudioInput, ResolvedGenerateCommand, SamplingConfig


class FakeNativeBackend:
    def __init__(self) -> None:
        self.embed_call = None
        self.generate_call = None

    def embed_style(self, prompt, **kwargs):
        self.embed_call = (prompt, kwargs)
        return "embedding"

    def generate(self, **kwargs):
        self.generate_call = kwargs
        return SimpleNamespace(samples=np.zeros((1920, 2))), []


def test_magenta_adapter_translates_core_command() -> None:
    native = FakeNativeBackend()
    backend = MagentaMlxBackend.__new__(MagentaMlxBackend)
    backend._backend = native
    backend._conditioning_key = "musiccoca"
    sampling = SamplingConfig(
        temperature=0.9,
        top_k=12,
        cfg_musiccoca=2.5,
        cfg_notes=0.5,
        cfg_drums=0.25,
        seed=42,
        use_mapper=False,
        pool_across_time=False,
    )

    audio = backend.generate(
        ResolvedGenerateCommand(
            prompt="jazz",
            reference_audio=None,
            duration=0.04,
            sampling=sampling,
        )
    )

    assert audio.shape == (1920, 2)
    assert native.embed_call == (
        "jazz",
        {"pool_across_time": False, "use_mapper": False, "seed": 42},
    )
    assert native.generate_call == {
        "conditioning": {"musiccoca": "embedding"},
        "cfg_scales": {"musiccoca": 2.5, "notes": 0.5, "drums": 0.25},
        "temperature": 0.9,
        "top_k": 12,
        "frames": 1,
        "state": None,
    }


def test_magenta_adapter_converts_reference_audio_to_waveform() -> None:
    native = FakeNativeBackend()
    backend = MagentaMlxBackend.__new__(MagentaMlxBackend)
    backend._backend = native
    backend._conditioning_key = "musiccoca"
    reference = AudioInput(np.zeros((480, 2), np.float32), 48_000)

    backend.generate(
        ResolvedGenerateCommand(
            prompt=None,
            reference_audio=reference,
            duration=0.01,
            sampling=SamplingConfig(),
        )
    )

    style_input = native.embed_call[0]
    assert style_input.sample_rate == 48_000
    assert style_input.samples.shape == (480, 2)
