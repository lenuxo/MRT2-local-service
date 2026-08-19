from __future__ import annotations

import io
from pathlib import Path

from .core import ControlInput, ControlMode, DrumEvent, NoteEvent


class MidiDecodingError(ValueError):
    """MIDI 文件无法解析或不包含可用控制事件。"""


def decode_midi(
    data: bytes,
    *,
    notes_mode: ControlMode = "guide",
    drums_mode: ControlMode = "guide",
    include_drums: bool = True,
) -> ControlInput:
    if not data:
        raise MidiDecodingError("MIDI 文件为空")
    try:
        import mido

        midi_file = mido.MidiFile(file=io.BytesIO(data))
    except Exception as exc:
        raise MidiDecodingError(f"无法解析 MIDI 文件：{exc}") from exc

    current_time = 0.0
    active: dict[tuple[int, int], list[float]] = {}
    notes: list[NoteEvent] = []
    drums: list[DrumEvent] = []
    for message in midi_file:
        current_time += float(message.time)
        if message.is_meta or message.type not in ("note_on", "note_off"):
            continue
        is_on = message.type == "note_on" and message.velocity > 0
        if message.channel == 9:
            if include_drums and is_on:
                drums.append(DrumEvent(time=current_time))
            continue
        key = (message.channel, message.note)
        if is_on:
            active.setdefault(key, []).append(current_time)
            continue
        starts = active.get(key)
        if not starts:
            continue
        start = starts.pop(0)
        notes.append(NoteEvent(
            pitch=message.note,
            start=start,
            duration=max(current_time - start, 0.04),
        ))

    for (_, pitch), starts in active.items():
        for start in starts:
            notes.append(NoteEvent(
                pitch=pitch,
                start=start,
                duration=max(current_time - start, 0.04),
            ))
    control = ControlInput(
        notes=tuple(notes),
        drums=tuple(drums),
        notes_mode=notes_mode,
        drums_mode=drums_mode,
    )
    control.validate()
    if not control.has_events:
        raise MidiDecodingError("MIDI 文件不包含可用的音符或鼓点事件")
    return control


def decode_midi_file(path: Path, **kwargs) -> ControlInput:
    if not path.is_file():
        raise MidiDecodingError(f"MIDI 文件不存在：{path}")
    return decode_midi(path.read_bytes(), **kwargs)
