from types import SimpleNamespace

import numpy as np

from mrt_local.backend import MagentaMlxBackend
from mrt_local.core import AudioInput, ResolvedGenerateCommand, SamplingConfig


class FakeNativeBackend:
    def __init__(self) -> None:
        self.embed_calls = []
        self.generate_call = None

    def embed_style(self, prompt, **kwargs):
        self.embed_calls.append((prompt, kwargs))
        return np.array([1.0, 2.0], dtype=np.float32) if isinstance(prompt, str) else np.array([3.0, 4.0], dtype=np.float32)

    def generate(self, **kwargs):
        self.generate_call = kwargs
        return SimpleNamespace(samples=np.zeros((1920, 2))), []


def test_magenta_adapter_translates_core_command() -> None:
    native = FakeNativeBackend()
    backend = MagentaMlxBackend.__new__(MagentaMlxBackend)
    backend._backend = native
    backend._conditioning_key = "musiccoca"
    backend._notes_conditioning_key = "notes"
    backend._drums_conditioning_key = "drums"
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
            text_weight=1.0,
            audio_weight=0.0,
            duration=0.04,
            sampling=sampling,
        )
    )

    assert audio.shape == (1920, 2)
    assert native.embed_calls == [(
        "jazz",
        {"pool_across_time": False, "use_mapper": False, "seed": 42},
    )]
    np.testing.assert_allclose(native.generate_call["conditioning"]["musiccoca"], [1, 2])
    assert native.generate_call | {"conditioning": None} == {
        "conditioning": None,
        "cfg_scales": {"musiccoca": 2.5, "notes": 0.5, "drums": 0.25},
        "temperature": 0.9, "top_k": 12, "frames": 1, "state": None,
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
            text_weight=0.0,
            audio_weight=1.0,
            duration=0.01,
            sampling=SamplingConfig(),
        )
    )

    style_input = native.embed_calls[0][0]
    assert style_input.sample_rate == 48_000
    assert style_input.samples.shape == (480, 2)


def test_magenta_adapter_blends_text_and_audio_embeddings() -> None:
    native = FakeNativeBackend()
    backend = MagentaMlxBackend.__new__(MagentaMlxBackend)
    backend._backend = native
    backend._conditioning_key = "musiccoca"
    reference = AudioInput(np.zeros((480, 2), np.float32), 48_000)

    backend.generate(ResolvedGenerateCommand(
        prompt="ambient", reference_audio=reference,
        text_weight=0.25, audio_weight=0.75,
        duration=0.01, sampling=SamplingConfig(),
    ))

    assert len(native.embed_calls) == 2
    np.testing.assert_allclose(native.generate_call["conditioning"]["musiccoca"], [2.5, 3.5])


def test_magenta_stream_adapter_reuses_native_state() -> None:
    class StatefulNative(FakeNativeBackend):
        def __init__(self):
            super().__init__()
            self.states = []

        def generate(self, **kwargs):
            self.states.append(kwargs["state"])
            return SimpleNamespace(samples=np.zeros((1_920, 2))), f"state-{len(self.states)}"

    from mrt_local.core import ResolvedStreamGenerateCommand

    native = StatefulNative()
    backend = MagentaMlxBackend.__new__(MagentaMlxBackend)
    backend._backend = native
    backend._conditioning_key = "musiccoca"
    backend._notes_conditioning_key = "notes"
    backend._drums_conditioning_key = "drums"
    session = backend.open_stream(ResolvedStreamGenerateCommand(
        prompt="ambient", reference_audio=None,
        text_weight=1, audio_weight=0, duration=0.08, chunk_frames=1,
        sampling=SamplingConfig(),
    ))

    session.generate_chunk(1)
    session.generate_chunk(1)
    assert native.states == [None, "state-1"]


def test_magenta_stream_adapter_applies_control_per_frame() -> None:
    from mrt_local.core import ControlTimeline, ResolvedStreamGenerateCommand

    class StatefulNative(FakeNativeBackend):
        def __init__(self):
            super().__init__()
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(samples=np.zeros((1_920, 2))), len(self.calls)

    native = StatefulNative()
    backend = MagentaMlxBackend.__new__(MagentaMlxBackend)
    backend._backend = native
    backend._conditioning_key = "musiccoca"
    backend._notes_conditioning_key = "notes"
    backend._drums_conditioning_key = "drums"
    notes = np.full((2, 128), -1, dtype=np.int8)
    notes[0, 60] = 2
    notes[1, 60] = 1
    timeline = ControlTimeline(notes=notes, drums=np.array([1, -1], dtype=np.int8))

    session = backend.open_stream(ResolvedStreamGenerateCommand(
        prompt=None, reference_audio=None,
        text_weight=0, audio_weight=0, duration=0.08, chunk_frames=2,
        sampling=SamplingConfig(), control_timeline=timeline,
    ))
    audio = session.generate_chunk(2)

    assert audio.shape == (3_840, 2)
    assert [call["frames"] for call in native.calls] == [1, 1]
    assert [call["state"] for call in native.calls] == [None, 1]
    np.testing.assert_array_equal(native.calls[0]["conditioning"]["notes"], notes[0])
    assert native.calls[0]["conditioning"]["drums"] == [1]
    assert native.calls[1]["conditioning"]["drums"] == [-1]
