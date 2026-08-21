from io import BytesIO

import mido
import numpy as np
import pytest

from mrt_local.core import (
    ControlInput,
    DrumEvent,
    NoteEvent,
    SamplingConfig,
    StreamGenerateCommand,
    StreamUpdateCommand,
    build_control_timeline,
)
from mrt_local.midi import decode_midi


def test_control_timeline_encodes_off_onset_and_sustain_states() -> None:
    timeline = build_control_timeline(
        ControlInput(
            notes=(NoteEvent(pitch=60, start=0, duration=0.08),),
            drums=(DrumEvent(time=0.04),),
            notes_mode="strict",
            drums_mode="guide",
        ),
        duration=0.12,
    )

    assert timeline.notes.shape == (3, 128)
    assert timeline.notes[:, 60].tolist() == [2, 1, 0]
    assert np.all(timeline.notes[:, 61] == 0)
    assert timeline.drums.tolist() == [-1, 1, -1]


def test_decode_midi_separates_melodic_notes_and_channel_10_drums() -> None:
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.Message("note_on", channel=0, note=60, velocity=100, time=0))
    track.append(mido.Message("note_on", channel=9, note=36, velocity=100, time=48))
    track.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=48))
    output = BytesIO()
    midi.save(file=output)

    control = decode_midi(output.getvalue(), notes_mode="strict")

    assert len(control.notes) == 1
    assert control.notes[0].pitch == 60
    assert control.notes[0].duration == 0.1
    assert len(control.drums) == 1
    assert control.drums[0].time == 0.05
    assert control.notes_mode == "strict"


def test_drumless_builds_all_off_timeline_and_rejects_events() -> None:
    timeline = build_control_timeline(ControlInput(drumless=True), duration=0.12)
    assert timeline.drums.tolist() == [0, 0, 0]
    assert timeline.drums_default == 0

    with pytest.raises(ValueError, match="不能同时提供"):
        build_control_timeline(
            ControlInput(drums=(DrumEvent(0),), drumless=True),
            duration=0.04,
        )
    with pytest.raises(ValueError, match="分开更新"):
        StreamUpdateCommand(
            revision=1, drumless=True, drums=()
        ).validate()


def test_live_midi_mode_rejects_plan_events_and_allows_empty_start() -> None:
    resolved = StreamGenerateCommand(
        midi_mode="live", duration=0.04, chunk_frames=1
    ).resolve(SamplingConfig())
    assert resolved.midi_mode == "live"
    assert resolved.prompt is None

    with pytest.raises(ValueError, match="计划式 notes"):
        StreamGenerateCommand(
            midi_mode="live",
            control=ControlInput(notes=(NoteEvent(60, 0, 0.04),)),
        ).resolve(SamplingConfig())
