from __future__ import annotations

import math
import os
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .config import RuntimeConfig
from .core import (
    ControlTimeline,
    LiveMidiCommand,
    LiveMidiEvent,
    LiveMidiQueueResult,
    MAX_PENDING_LIVE_MIDI_EVENTS,
    PromptComponent,
    ResolvedGenerateCommand,
    ResolvedStreamGenerateCommand,
    SamplingConfig,
    StreamUpdateCommand,
    StreamUpdateResult,
    build_drums_timeline,
    build_notes_timeline,
    resolve_prompt_components,
)

MAGENTA_FRAMES_PER_SECOND = 25


def _embed_text_components(
    backend: Any,
    components: tuple[PromptComponent, ...],
    sampling: SamplingConfig,
) -> np.ndarray | None:
    if not components:
        return None
    embeddings = [
        np.asarray(backend.embed_style(
            component.text,
            pool_across_time=sampling.pool_across_time,
            use_mapper=sampling.use_mapper,
            seed=sampling.seed,
        ), dtype=np.float32)
        for component in components
    ]
    return np.asarray(np.average(
        np.stack(embeddings),
        axis=0,
        weights=[component.weight for component in components],
    ), dtype=np.float32)


class GenerationBackend(Protocol):
    def generate(self, command: ResolvedGenerateCommand) -> np.ndarray: ...
    def open_stream(self, command: ResolvedStreamGenerateCommand) -> StreamingBackendSession: ...


class StreamingBackendSession(Protocol):
    def generate_chunk(self, frames: int) -> np.ndarray: ...
    def update(self, command: StreamUpdateCommand) -> StreamUpdateResult: ...
    def queue_live_midi(self, command: LiveMidiCommand) -> LiveMidiQueueResult: ...
    def extend_to(self, total_frames: int) -> None: ...
    def close(self) -> None: ...


class MagentaMlxBackend:
    """把 Magenta 的 MLX API 隔离在基础设施适配层。"""

    def __init__(self, config: RuntimeConfig) -> None:
        model = config.model
        defaults = config.sampling

        # Magenta 的 paths 模块在导入时读取路径，因此必须先设置环境变量。
        os.environ["MAGENTA_HOME"] = str(model.root.parent)
        from magenta_rt import MagentaRT2StdMlxfn, paths
        from magenta_rt.config import DRUM_PIANOROLL, MUSICCOCA, PIANOROLL_WITH_ONSETS

        paths.set_magenta_home(model.root)
        self._conditioning_key = MUSICCOCA.key
        self._notes_conditioning_key = PIANOROLL_WITH_ONSETS.key
        self._drums_conditioning_key = DRUM_PIANOROLL.key
        try:
            self._backend: Any = MagentaRT2StdMlxfn(
                size=model.name,
                temperature=defaults.temperature,
                top_k=defaults.top_k,
                cfg_scales=defaults.cfg_scales,
                warmup_steps=model.warmup_steps,
            )
        except RuntimeError as exc:
            if "[import_function] Invalid string size" not in str(exc):
                raise
            raise RuntimeError(
                "MLX 无法导入官方 .mlxfn 模型；请运行 `uv sync` 恢复锁定的 "
                "MLX 版本，若仍失败请重新下载模型"
            ) from exc

    def _embed_style_input(
        self,
        style_input: Any,
        sampling: SamplingConfig,
    ) -> np.ndarray:
        return np.asarray(self._backend.embed_style(
            style_input,
            pool_across_time=sampling.pool_across_time,
            use_mapper=sampling.use_mapper,
            seed=sampling.seed,
        ), dtype=np.float32)

    def _build_style_components(
        self,
        command: ResolvedGenerateCommand | ResolvedStreamGenerateCommand,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        sampling = command.sampling
        text_embedding = None
        audio_embedding = None
        components = command.prompt_components or resolve_prompt_components(
            command.prompt, ()
        )
        text_embedding = _embed_text_components(
            self._backend, components, sampling
        )
        if command.reference_audio is not None:
            from magenta_rt.audio import Waveform

            style_input: Any = Waveform(
                command.reference_audio.samples,
                command.reference_audio.sample_rate,
            )
            audio_embedding = self._embed_style_input(style_input, sampling)
        return text_embedding, audio_embedding

    @staticmethod
    def _mix_style_embeddings(
        text_embedding: np.ndarray | None,
        audio_embedding: np.ndarray | None,
        text_weight: float,
        audio_weight: float,
    ) -> np.ndarray | None:
        embeddings: list[np.ndarray] = []
        weights: list[float] = []
        if text_embedding is not None:
            embeddings.append(text_embedding)
            weights.append(text_weight)
        if audio_embedding is not None:
            embeddings.append(audio_embedding)
            weights.append(audio_weight)
        if not embeddings:
            return None
        if not any(weight > 0 for weight in weights):
            raise ValueError("当前文本和参考音频条件的权重不能同时为 0")
        return np.asarray(
            np.average(np.stack(embeddings), axis=0, weights=weights),
            dtype=np.float32,
        )

    def generate(self, command: ResolvedGenerateCommand) -> np.ndarray:
        sampling = command.sampling
        text_embedding, audio_embedding = self._build_style_components(command)
        embedding = self._mix_style_embeddings(
            text_embedding,
            audio_embedding,
            command.text_weight,
            command.audio_weight,
        )
        if command.control_timeline is not None:
            session = MagentaMlxStreamSession(
                backend=self._backend,
                conditioning_key=self._conditioning_key,
                notes_conditioning_key=self._notes_conditioning_key,
                drums_conditioning_key=self._drums_conditioning_key,
                text_embedding=text_embedding,
                audio_embedding=audio_embedding,
                text_weight=command.text_weight,
                audio_weight=command.audio_weight,
                sampling=sampling,
                control_timeline=command.control_timeline,
                total_frames=math.ceil(
                    command.duration * MAGENTA_FRAMES_PER_SECOND
                ),
                midi_mode="plan",
                live_notes_mode="guide",
                drumless=False,
            )
            chunks = [
                session.generate_chunk(1)
                for _ in range(math.ceil(command.duration * MAGENTA_FRAMES_PER_SECOND))
            ]
            session.close()
            return np.concatenate(chunks, axis=0)
        conditioning = (
            {self._conditioning_key: embedding} if embedding is not None else {}
        )
        waveform, _ = self._backend.generate(
            conditioning=conditioning,
            cfg_scales=sampling.cfg_scales,
            temperature=sampling.temperature,
            top_k=sampling.top_k,
            frames=math.ceil(command.duration * MAGENTA_FRAMES_PER_SECOND),
            state=None,
        )
        return np.asarray(waveform.samples)

    def open_stream(
        self,
        command: ResolvedStreamGenerateCommand,
    ) -> StreamingBackendSession:
        text_embedding, audio_embedding = self._build_style_components(command)
        return MagentaMlxStreamSession(
            backend=self._backend,
            conditioning_key=self._conditioning_key,
            notes_conditioning_key=self._notes_conditioning_key,
            drums_conditioning_key=self._drums_conditioning_key,
            text_embedding=text_embedding,
            audio_embedding=audio_embedding,
            text_weight=command.text_weight,
            audio_weight=command.audio_weight,
            sampling=command.sampling,
            control_timeline=command.control_timeline,
            total_frames=math.ceil(command.duration * MAGENTA_FRAMES_PER_SECOND),
            midi_mode=command.midi_mode,
            live_notes_mode=command.live_notes_mode,
            drumless=command.drumless,
        )


@dataclass(slots=True)
class _PendingUpdate:
    effective_frame: int
    sequence: int
    command: StreamUpdateCommand


class _LiveMidiState:
    """线程安全接收事件，并只在 MLX 线程上推进逐帧 MIDI 状态。"""

    _IDLE = 0
    _ONSET = 1
    _SUSTAIN = 2
    _ONSET_RELEASED = 3

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: deque[LiveMidiEvent] = deque()
        self._states = np.zeros(128, dtype=np.int8)
        self._held: dict[tuple[int, int], int] = {}
        self._pedal = [False] * 16
        self._deferred: set[tuple[int, int]] = set()
        self._drum_trigger = False

    def enqueue(
        self,
        command: LiveMidiCommand,
        *,
        earliest_frame: int,
        drumless: bool,
    ) -> LiveMidiQueueResult:
        command.validate()
        if drumless and any(
            event.kind == "note_on"
            and event.channel == 9
            and (event.velocity or 0) > 0
            for event in command.events
        ):
            raise ValueError("drumless=true 时不能发送 MIDI 通道 10 的鼓触发")
        with self._lock:
            if len(self._pending) + len(command.events) > MAX_PENDING_LIVE_MIDI_EVENTS:
                self._pending.clear()
                self._pending.append(LiveMidiEvent("panic"))
                raise ValueError("实时 MIDI 队列已满；已触发 panic 以防止音符卡住")
            self._pending.extend(command.events)
        return LiveMidiQueueResult(
            command.event_sequence, earliest_frame, len(command.events)
        )

    def snapshot(self, mode: str, *, drumless: bool) -> tuple[np.ndarray, int | None]:
        with self._lock:
            pending = tuple(self._pending)
            self._pending.clear()
        for event in pending:
            self._apply(event)

        baseline = -1 if mode == "guide" else 0
        notes = np.full(128, baseline, dtype=np.int8)
        onset = (self._states == self._ONSET) | (
            self._states == self._ONSET_RELEASED
        )
        notes[self._states == self._SUSTAIN] = 1
        notes[onset] = 2
        self._states[self._states == self._ONSET] = self._SUSTAIN
        self._states[self._states == self._ONSET_RELEASED] = self._IDLE

        drum = 0 if drumless else 1 if self._drum_trigger else None
        self._drum_trigger = False
        return notes, drum

    def panic(self) -> None:
        with self._lock:
            self._pending.clear()
        self._held.clear()
        self._deferred.clear()
        self._pedal = [False] * 16
        self._states.fill(self._IDLE)
        self._drum_trigger = False

    def _apply(self, event: LiveMidiEvent) -> None:
        if event.kind == "panic":
            self._release_channel(event.channel)
            return
        assert event.channel is not None
        if event.kind == "control_change":
            assert event.controller is not None and event.value is not None
            if event.controller == 64:
                enabled = event.value >= 64
                if self._pedal[event.channel] and not enabled:
                    self._release_pedal(event.channel)
                self._pedal[event.channel] = enabled
            elif event.controller in (120, 123):
                self._release_channel(event.channel)
            return

        assert event.pitch is not None and event.velocity is not None
        if event.channel == 9:
            if event.kind == "note_on" and event.velocity > 0:
                self._drum_trigger = True
            return
        if event.kind == "note_on" and event.velocity > 0:
            key = (event.channel, event.pitch)
            self._held[key] = self._held.get(key, 0) + 1
            self._deferred.discard(key)
            self._states[event.pitch] = self._ONSET
        else:
            self._note_off(event.channel, event.pitch)

    def _note_off(self, channel: int, pitch: int) -> None:
        key = (channel, pitch)
        count = self._held.get(key, 0)
        if count == 0:
            return
        if count > 1:
            self._held[key] = count - 1
            return
        self._held.pop(key, None)
        if self._pedal[channel]:
            self._deferred.add(key)
            return
        self._release_pitch_if_inactive(pitch)

    def _release_pedal(self, channel: int) -> None:
        pitches = {pitch for ch, pitch in self._deferred if ch == channel}
        self._deferred = {
            key for key in self._deferred if key[0] != channel
        }
        for pitch in pitches:
            self._release_pitch_if_inactive(pitch)

    def _release_channel(self, channel: int | None) -> None:
        if channel is None:
            self.panic()
            return
        pitches = {
            pitch for ch, pitch in (*self._held.keys(), *self._deferred)
            if ch == channel
        }
        self._held = {
            key: count for key, count in self._held.items() if key[0] != channel
        }
        self._deferred = {key for key in self._deferred if key[0] != channel}
        self._pedal[channel] = False
        for pitch in pitches:
            self._release_pitch_if_inactive(pitch)

    def _release_pitch_if_inactive(self, pitch: int) -> None:
        active = any(key[1] == pitch for key in self._held) or any(
            key[1] == pitch for key in self._deferred
        )
        if active:
            return
        self._states[pitch] = (
            self._ONSET_RELEASED
            if self._states[pitch] == self._ONSET
            else self._IDLE
        )


class MagentaMlxStreamSession:
    """持有官方生成 state，并按固定帧数连续生成 PCM。"""

    def __init__(
        self,
        *,
        backend: Any,
        conditioning_key: str,
        notes_conditioning_key: str,
        drums_conditioning_key: str,
        text_embedding: np.ndarray | None,
        audio_embedding: np.ndarray | None,
        text_weight: float,
        audio_weight: float,
        sampling: SamplingConfig,
        control_timeline: ControlTimeline | None,
        total_frames: int,
        midi_mode: str,
        live_notes_mode: str,
        drumless: bool,
    ) -> None:
        self._backend = backend
        self._conditioning_key = conditioning_key
        self._notes_conditioning_key = notes_conditioning_key
        self._drums_conditioning_key = drums_conditioning_key
        self._text_embedding = text_embedding
        self._audio_embedding = audio_embedding
        self._text_weight = text_weight
        self._audio_weight = audio_weight
        self._embedding = MagentaMlxBackend._mix_style_embeddings(
            text_embedding, audio_embedding, text_weight, audio_weight
        )
        self._sampling = sampling
        self._control_timeline = control_timeline
        self._midi_mode = midi_mode
        self._live_notes_mode = live_notes_mode
        self._live_drumless = drumless
        self._live_midi = _LiveMidiState() if midi_mode == "live" else None
        self._total_frames = total_frames
        self._frame_cursor = 0
        self._pending_updates: list[_PendingUpdate] = []
        self._update_sequence = 0
        self._state: Any = None
        self._closed = False

    def generate_chunk(self, frames: int) -> np.ndarray:
        if self._closed:
            raise RuntimeError("流式后端会话已经关闭")
        frames = min(frames, self._total_frames - self._frame_cursor)
        if frames <= 0:
            return np.empty((0, 2), dtype=np.float32)
        self._apply_due_updates()
        if (
            self._midi_mode == "plan"
            and self._control_timeline is None
            and not self._pending_updates
        ):
            conditioning = (
                {self._conditioning_key: self._embedding}
                if self._embedding is not None else {}
            )
            waveform, self._state = self._generate(conditioning, frames)
            self._frame_cursor += frames
            return np.asarray(waveform.samples, dtype=np.float32)

        chunks: list[np.ndarray] = []
        for _ in range(frames):
            self._apply_due_updates()
            conditioning = {}
            if self._embedding is not None:
                conditioning[self._conditioning_key] = self._embedding
            if self._live_midi is not None:
                live_notes, live_drum = self._live_midi.snapshot(
                    self._live_notes_mode, drumless=self._live_drumless
                )
                conditioning[self._notes_conditioning_key] = live_notes
                if live_drum is not None:
                    conditioning[self._drums_conditioning_key] = [live_drum]
            elif (
                self._control_timeline is not None
                and self._control_timeline.notes is not None
            ):
                conditioning[self._notes_conditioning_key] = self._control_timeline.notes[
                    self._frame_cursor
                ]
            if (
                self._control_timeline is not None
                and self._control_timeline.drums is not None
            ):
                conditioning[self._drums_conditioning_key] = [int(
                    self._control_timeline.drums[self._frame_cursor]
                )]
            waveform, self._state = self._generate(conditioning, 1)
            chunks.append(np.asarray(waveform.samples, dtype=np.float32))
            self._frame_cursor += 1
        return np.concatenate(chunks, axis=0)

    def update(self, command: StreamUpdateCommand) -> StreamUpdateResult:
        if self._closed:
            raise RuntimeError("流式后端会话已经关闭")
        command.validate()
        if self._midi_mode == "live" and (
            command.notes is not None or command.drums is not None
        ):
            raise ValueError(
                "实时 MIDI 会话不能使用 update 替换计划式 notes 或 drums"
            )
        effective_frame = max(
            self._frame_cursor,
            command.effective_frame
            if command.effective_frame is not None else self._frame_cursor,
        )
        if effective_frame >= self._total_frames:
            raise ValueError("更新生效帧必须早于流式会话结束帧")
        self._pending_updates.append(_PendingUpdate(
            effective_frame=effective_frame,
            sequence=self._update_sequence,
            command=command,
        ))
        self._update_sequence += 1
        self._pending_updates.sort(
            key=lambda item: (item.effective_frame, item.sequence)
        )
        self._apply_due_updates()
        return StreamUpdateResult(command.revision, effective_frame)

    def queue_live_midi(
        self, command: LiveMidiCommand
    ) -> LiveMidiQueueResult:
        if self._closed:
            raise RuntimeError("流式后端会话已经关闭")
        if self._live_midi is None:
            raise ValueError("当前会话未启用 midiMode=live")
        return self._live_midi.enqueue(
            command,
            earliest_frame=self._frame_cursor,
            drumless=self._live_drumless,
        )

    def extend_to(self, total_frames: int) -> None:
        if self._closed:
            raise RuntimeError("流式后端会话已经关闭")
        if total_frames <= self._total_frames:
            return
        additional = total_frames - self._total_frames
        if self._control_timeline is not None:
            notes = self._control_timeline.notes
            drums = self._control_timeline.drums
            if notes is not None:
                notes = np.pad(
                    notes,
                    ((0, additional), (0, 0)),
                    constant_values=self._control_timeline.notes_default,
                )
            if drums is not None:
                drums = np.pad(
                    drums,
                    (0, additional),
                    constant_values=self._control_timeline.drums_default,
                )
            self._control_timeline = ControlTimeline(
                notes,
                drums,
                self._control_timeline.notes_default,
                self._control_timeline.drums_default,
            )
        self._total_frames = total_frames

    def _apply_due_updates(self) -> None:
        while (
            self._pending_updates
            and self._pending_updates[0].effective_frame <= self._frame_cursor
        ):
            pending = self._pending_updates.pop(0)
            command = pending.command
            sampling = command.sampling.resolve(self._sampling)
            text_embedding = self._text_embedding
            audio_embedding = self._audio_embedding
            text_weight = (
                command.text_weight
                if command.text_weight is not None else self._text_weight
            )
            audio_weight = (
                command.audio_weight
                if command.audio_weight is not None else self._audio_weight
            )
            if command.prompt_present or command.prompt_components_present:
                components = resolve_prompt_components(
                    command.prompt if command.prompt_present else None,
                    command.prompt_components
                    if command.prompt_components_present else (),
                )
                text_embedding = _embed_text_components(
                    self._backend, components, sampling
                )
            if command.reference_audio_present:
                if command.reference_audio is None:
                    audio_embedding = None
                else:
                    from magenta_rt.audio import Waveform

                    audio_embedding = np.asarray(self._backend.embed_style(
                        Waveform(
                            command.reference_audio.samples,
                            command.reference_audio.sample_rate,
                        ),
                        pool_across_time=sampling.pool_across_time,
                        use_mapper=sampling.use_mapper,
                        seed=sampling.seed,
                    ), dtype=np.float32)
            if (
                text_embedding is not None
                and audio_embedding is not None
                and not sampling.pool_across_time
            ):
                raise ValueError(
                    "文本与音频混合时 poolAcrossTime 必须为 true"
                )
            embedding = MagentaMlxBackend._mix_style_embeddings(
                text_embedding,
                audio_embedding,
                text_weight,
                audio_weight,
            )
            self._text_embedding = text_embedding
            self._audio_embedding = audio_embedding
            self._text_weight = text_weight
            self._audio_weight = audio_weight
            self._embedding = embedding
            self._sampling = sampling
            remaining = self._total_frames - self._frame_cursor
            if command.notes is not None:
                notes = build_notes_timeline(
                    command.notes, command.notes_mode, remaining
                )
                if self._control_timeline is None:
                    self._control_timeline = ControlTimeline(None, None)
                if self._control_timeline.notes is None:
                    self._control_timeline = ControlTimeline(
                        np.full(
                            (self._total_frames, 128), -1, dtype=np.int8
                        ),
                        self._control_timeline.drums,
                        self._control_timeline.notes_default,
                        self._control_timeline.drums_default,
                    )
                self._control_timeline.notes[self._frame_cursor:] = notes
                self._control_timeline = ControlTimeline(
                    self._control_timeline.notes,
                    self._control_timeline.drums,
                    -1 if command.notes_mode == "guide" else 0,
                    self._control_timeline.drums_default,
                )
            drum_events = command.drums
            drum_mode = command.drums_mode
            if command.drumless is not None:
                drum_events = ()
                drum_mode = "strict" if command.drumless else "guide"
                self._live_drumless = command.drumless
            if drum_events is not None:
                drums = build_drums_timeline(
                    drum_events, drum_mode, remaining
                )
                if self._control_timeline is None:
                    self._control_timeline = ControlTimeline(None, None)
                if self._control_timeline.drums is None:
                    self._control_timeline = ControlTimeline(
                        self._control_timeline.notes,
                        np.full(self._total_frames, -1, dtype=np.int8),
                        self._control_timeline.notes_default,
                        self._control_timeline.drums_default,
                    )
                self._control_timeline.drums[self._frame_cursor:] = drums
                self._control_timeline = ControlTimeline(
                    self._control_timeline.notes,
                    self._control_timeline.drums,
                    self._control_timeline.notes_default,
                    -1 if drum_mode == "guide" else 0,
                )

    def _generate(self, conditioning: dict[str, Any], frames: int):
        return self._backend.generate(
            conditioning=conditioning,
            cfg_scales=self._sampling.cfg_scales,
            temperature=self._sampling.temperature,
            top_k=self._sampling.top_k,
            frames=frames,
            state=self._state,
        )

    def close(self) -> None:
        if self._live_midi is not None:
            self._live_midi.panic()
        self._closed = True


def create_magenta_backend(config: RuntimeConfig) -> GenerationBackend:
    return MagentaMlxBackend(config)
