from types import SimpleNamespace

import numpy as np

from mrt_local.backend import MagentaMlxBackend
from mrt_local.core import (
    AudioInput,
    DrumEvent,
    NoteEvent,
    ResolvedGenerateCommand,
    SamplingConfig,
    SamplingOverrides,
    StreamUpdateCommand,
)


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


def test_magenta_stream_extension_preserves_strict_control_mode() -> None:
    from mrt_local.core import ControlTimeline, ResolvedStreamGenerateCommand

    class RecordingNative(FakeNativeBackend):
        def __init__(self):
            super().__init__()
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(samples=np.zeros((1_920, 2))), len(self.calls)

    native = RecordingNative()
    backend = MagentaMlxBackend.__new__(MagentaMlxBackend)
    backend._backend = native
    backend._conditioning_key = "musiccoca"
    backend._notes_conditioning_key = "notes"
    backend._drums_conditioning_key = "drums"
    timeline = ControlTimeline(
        notes=np.zeros((1, 128), dtype=np.int8),
        drums=np.zeros(1, dtype=np.int8),
        notes_default=0,
        drums_default=0,
    )
    session = backend.open_stream(ResolvedStreamGenerateCommand(
        prompt=None, reference_audio=None,
        text_weight=0, audio_weight=0, duration=0.04, chunk_frames=1,
        sampling=SamplingConfig(), control_timeline=timeline,
    ))

    session.generate_chunk(1)
    session.extend_to(2)
    session.generate_chunk(1)

    np.testing.assert_array_equal(
        native.calls[1]["conditioning"]["notes"],
        np.zeros(128, dtype=np.int8),
    )
    assert native.calls[1]["conditioning"]["drums"] == [0]


def test_magenta_stream_adapter_updates_future_generation_without_resetting_state() -> None:
    from mrt_local.core import ResolvedStreamGenerateCommand

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
    session = backend.open_stream(ResolvedStreamGenerateCommand(
        prompt="ambient", reference_audio=None,
        text_weight=1, audio_weight=0, duration=0.16, chunk_frames=1,
        sampling=SamplingConfig(),
    ))

    session.generate_chunk(1)
    result = session.update(StreamUpdateCommand(
        revision=7,
        prompt_present=True,
        prompt="techno",
        sampling=SamplingOverrides(
            temperature=0.8,
            top_k=12,
            cfg_musiccoca=4.0,
            cfg_notes=1.4,
            cfg_drums=2.0,
        ),
        notes=(NoteEvent(64, 0, 0.08),),
        drums=(DrumEvent(0),),
        notes_mode="strict",
        drums_mode="strict",
    ))
    session.generate_chunk(1)

    assert result.revision == 7
    assert result.effective_frame == 1
    assert native.embed_calls[-1][0] == "techno"
    call = native.calls[1]
    assert call["state"] == 1
    assert call["temperature"] == 0.8
    assert call["top_k"] == 12
    assert call["cfg_scales"] == {
        "musiccoca": 4.0, "notes": 1.4, "drums": 2.0,
    }
    np.testing.assert_allclose(call["conditioning"]["musiccoca"], [1, 2])
    assert call["conditioning"]["notes"][64] == 2
    assert call["conditioning"]["notes"][63] == 0
    assert call["conditioning"]["drums"] == [1]


def test_magenta_stream_adapter_updates_and_clears_reference_audio() -> None:
    from mrt_local.core import AudioInput, ResolvedStreamGenerateCommand

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
    session = backend.open_stream(ResolvedStreamGenerateCommand(
        prompt="ambient", reference_audio=None,
        text_weight=1, audio_weight=0, duration=0.16, chunk_frames=1,
        sampling=SamplingConfig(),
    ))
    reference = AudioInput(np.zeros((480, 2), np.float32), 48_000)

    session.update(StreamUpdateCommand(
        revision=1,
        reference_audio_present=True,
        reference_audio=reference,
        text_weight=1,
        audio_weight=3,
    ))
    session.generate_chunk(1)
    np.testing.assert_allclose(
        native.calls[-1]["conditioning"]["musiccoca"], [2.5, 3.5]
    )

    session.update(StreamUpdateCommand(
        revision=2,
        prompt_present=True,
        prompt="techno",
    ))
    session.generate_chunk(1)
    np.testing.assert_allclose(
        native.calls[-1]["conditioning"]["musiccoca"], [2.5, 3.5]
    )

    session.update(StreamUpdateCommand(
        revision=3,
        reference_audio_present=True,
        reference_audio=None,
    ))
    session.generate_chunk(1)
    np.testing.assert_allclose(
        native.calls[-1]["conditioning"]["musiccoca"], [1, 2]
    )
    assert native.calls[-1]["state"] == 2


def test_magenta_stream_adapter_schedules_update_at_future_frame() -> None:
    from mrt_local.core import ResolvedStreamGenerateCommand

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
    session = backend.open_stream(ResolvedStreamGenerateCommand(
        prompt="ambient", reference_audio=None,
        text_weight=1, audio_weight=0, duration=0.2, chunk_frames=5,
        sampling=SamplingConfig(),
    ))
    session.update(StreamUpdateCommand(
        revision=1,
        effective_frame=3,
        sampling=SamplingOverrides(temperature=0.7),
    ))

    session.generate_chunk(5)

    assert [call["temperature"] for call in native.calls] == [1.3, 1.3, 1.3, 0.7, 0.7]
    assert [call["state"] for call in native.calls] == [None, 1, 2, 3, 4]
