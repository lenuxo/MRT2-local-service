from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import numpy as np

ModelName = Literal["mrt2_small", "mrt2_base"]
ControlMode = Literal["guide", "strict"]
SUPPORTED_MODELS: tuple[ModelName, ...] = ("mrt2_small", "mrt2_base")
DEFAULT_MODEL_NAME: ModelName = "mrt2_small"
SAMPLE_RATE = 48_000
CHANNELS = 2
MAX_DURATION = 300.0
MAX_REFERENCE_AUDIO_DURATION = 300.0
DEFAULT_DURATION = 10.0
DEFAULT_WARMUP_STEPS = 5
DEFAULT_STYLE_WEIGHT = 0.5
DEFAULT_STREAM_CHUNK_FRAMES = 5
MAX_STREAM_CHUNK_FRAMES = 25


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: ModelName
    root: Path
    warmup_steps: int = DEFAULT_WARMUP_STEPS

    def validate(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps 必须大于等于 0")

    @property
    def model_dir(self) -> Path:
        return self.root / "models" / self.name

    @property
    def model_path(self) -> Path:
        return self.model_dir / f"{self.name}.mlxfn"

    @property
    def state_path(self) -> Path:
        return self.model_dir / f"{self.name}_state.safetensors"

    @property
    def resources_path(self) -> Path:
        return self.root / "resources"


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    temperature: float = 1.3
    top_k: int = 40
    cfg_musiccoca: float = 3.0
    cfg_notes: float = 1.0
    cfg_drums: float = 1.0
    seed: int = 0
    use_mapper: bool = True
    pool_across_time: bool = True

    def validate(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature 必须是大于 0 的有限数")
        if self.top_k < 1:
            raise ValueError("top_k 必须大于等于 1")
        if not all(
            math.isfinite(value)
            for value in (self.cfg_musiccoca, self.cfg_notes, self.cfg_drums)
        ):
            raise ValueError("CFG 参数必须是有限数")

    @property
    def cfg_scales(self) -> dict[str, float]:
        return {
            "musiccoca": self.cfg_musiccoca,
            "notes": self.cfg_notes,
            "drums": self.cfg_drums,
        }


@dataclass(frozen=True, slots=True)
class SamplingOverrides:
    temperature: float | None = None
    top_k: int | None = None
    cfg_musiccoca: float | None = None
    cfg_notes: float | None = None
    cfg_drums: float | None = None
    seed: int | None = None
    use_mapper: bool | None = None
    pool_across_time: bool | None = None

    def resolve(self, defaults: SamplingConfig) -> SamplingConfig:
        values = {
            field.name: (
                getattr(defaults, field.name)
                if getattr(self, field.name) is None
                else getattr(self, field.name)
            )
            for field in fields(self)
        }
        resolved = SamplingConfig(**values)
        resolved.validate()
        return resolved


@dataclass(frozen=True, slots=True)
class AudioInput:
    samples: np.ndarray
    sample_rate: int

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("参考音频采样率必须大于 0")
        if self.samples.ndim not in (1, 2) or self.samples.size == 0:
            raise ValueError("参考音频必须包含至少一个声道和一个采样")
        if not np.issubdtype(self.samples.dtype, np.floating):
            raise ValueError("参考音频必须解码为浮点 PCM")
        if not np.isfinite(self.samples).all():
            raise ValueError("参考音频包含非有限采样值")
        if self.samples.shape[0] / self.sample_rate > MAX_REFERENCE_AUDIO_DURATION:
            raise ValueError("参考音频不能超过 300 秒")


@dataclass(frozen=True, slots=True)
class NoteEvent:
    pitch: int
    start: float
    duration: float

    def validate(self) -> None:
        if not 0 <= self.pitch <= 127:
            raise ValueError("音符 pitch 必须在 0 到 127 之间")
        if not math.isfinite(self.start) or self.start < 0:
            raise ValueError("音符 start 必须是非负有限秒数")
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("音符 duration 必须是大于 0 的有限秒数")


@dataclass(frozen=True, slots=True)
class DrumEvent:
    time: float

    def validate(self) -> None:
        if not math.isfinite(self.time) or self.time < 0:
            raise ValueError("鼓点 time 必须是非负有限秒数")


@dataclass(frozen=True, slots=True)
class ControlInput:
    notes: tuple[NoteEvent, ...] = ()
    drums: tuple[DrumEvent, ...] = ()
    notes_mode: ControlMode = "guide"
    drums_mode: ControlMode = "guide"

    def validate(self) -> None:
        if self.notes_mode not in ("guide", "strict"):
            raise ValueError("notes_mode 必须是 guide 或 strict")
        if self.drums_mode not in ("guide", "strict"):
            raise ValueError("drums_mode 必须是 guide 或 strict")
        for event in self.notes:
            event.validate()
        for event in self.drums:
            event.validate()

    @property
    def has_events(self) -> bool:
        return bool(self.notes or self.drums)


@dataclass(frozen=True, slots=True)
class ControlTimeline:
    notes: np.ndarray | None
    drums: np.ndarray | None
    notes_default: int = -1
    drums_default: int = -1

    @property
    def frame_count(self) -> int:
        source = self.notes if self.notes is not None else self.drums
        return 0 if source is None else len(source)


def build_control_timeline(control: ControlInput | None, duration: float) -> ControlTimeline | None:
    if control is None or not control.has_events:
        return None
    control.validate()
    frame_count = math.ceil(duration * 25)
    notes = None
    drums = None
    if control.notes:
        notes = build_notes_timeline(control.notes, control.notes_mode, frame_count)
    if control.drums:
        drums = build_drums_timeline(control.drums, control.drums_mode, frame_count)
    return ControlTimeline(
        notes=notes,
        drums=drums,
        notes_default=-1 if control.notes_mode == "guide" else 0,
        drums_default=-1 if control.drums_mode == "guide" else 0,
    )


def build_notes_timeline(
    events: tuple[NoteEvent, ...], mode: ControlMode, frame_count: int
) -> np.ndarray:
    baseline = -1 if mode == "guide" else 0
    notes = np.full((frame_count, 128), baseline, dtype=np.int8)
    for event in sorted(events, key=lambda item: (item.start, item.pitch)):
        event.validate()
        start = min(round(event.start * 25), frame_count)
        if start >= frame_count:
            continue
        end = min(
            max(start + 1, round((event.start + event.duration) * 25)),
            frame_count,
        )
        notes[start, event.pitch] = 2
        if end > start + 1:
            notes[start + 1:end, event.pitch] = 1
    return notes


def build_drums_timeline(
    events: tuple[DrumEvent, ...], mode: ControlMode, frame_count: int
) -> np.ndarray:
    baseline = -1 if mode == "guide" else 0
    drums = np.full(frame_count, baseline, dtype=np.int8)
    for event in events:
        event.validate()
        frame = round(event.time * 25)
        if 0 <= frame < frame_count:
            drums[frame] = 1
    return drums


@dataclass(frozen=True, slots=True)
class GenerateCommand:
    prompt: str | None = None
    reference_audio: AudioInput | None = None
    text_weight: float = DEFAULT_STYLE_WEIGHT
    audio_weight: float = DEFAULT_STYLE_WEIGHT
    duration: float = DEFAULT_DURATION
    sampling: SamplingOverrides = SamplingOverrides()
    control: ControlInput | None = None

    def resolve(self, defaults: SamplingConfig) -> ResolvedGenerateCommand:
        prompt = self.prompt.strip() if self.prompt is not None else None
        has_text = bool(prompt)
        has_audio = self.reference_audio is not None
        has_control = self.control is not None and self.control.has_events
        if not has_text and not has_audio and not has_control:
            raise ValueError("prompt、reference_audio 或音符/鼓点事件至少需要提供一个")
        if self.reference_audio is not None:
            self.reference_audio.validate()
        if not all(
            math.isfinite(value) and value >= 0
            for value in (self.text_weight, self.audio_weight)
        ):
            raise ValueError("text_weight 和 audio_weight 必须是非负有限数")
        active_text_weight = self.text_weight if has_text else 0.0
        active_audio_weight = self.audio_weight if has_audio else 0.0
        total_weight = active_text_weight + active_audio_weight
        if (has_text or has_audio) and total_weight <= 0:
            raise ValueError("至少一个已提供输入的权重必须大于 0")
        sampling = self.sampling.resolve(defaults)
        if has_text and has_audio and not sampling.pool_across_time:
            raise ValueError("文本与音频混合时 pool_across_time 必须为 true")
        if (
            not math.isfinite(self.duration)
            or self.duration <= 0
            or self.duration > MAX_DURATION
        ):
            raise ValueError("duration 必须大于 0 且不超过 300 秒")
        return ResolvedGenerateCommand(
            prompt=prompt,
            reference_audio=self.reference_audio,
            text_weight=active_text_weight / total_weight if total_weight else 0.0,
            audio_weight=active_audio_weight / total_weight if total_weight else 0.0,
            duration=self.duration,
            sampling=sampling,
            control_timeline=build_control_timeline(self.control, self.duration),
        )


@dataclass(frozen=True, slots=True)
class ResolvedGenerateCommand:
    prompt: str | None
    reference_audio: AudioInput | None
    text_weight: float
    audio_weight: float
    duration: float
    sampling: SamplingConfig
    control_timeline: ControlTimeline | None = None

    @property
    def sample_count(self) -> int:
        return round(self.duration * SAMPLE_RATE)


@dataclass(frozen=True, slots=True)
class StreamGenerateCommand:
    prompt: str | None = None
    reference_audio: AudioInput | None = None
    text_weight: float = DEFAULT_STYLE_WEIGHT
    audio_weight: float = DEFAULT_STYLE_WEIGHT
    duration: float = DEFAULT_DURATION
    chunk_frames: int = DEFAULT_STREAM_CHUNK_FRAMES
    sampling: SamplingOverrides = SamplingOverrides()
    control: ControlInput | None = None

    def resolve(self, defaults: SamplingConfig) -> ResolvedStreamGenerateCommand:
        if not 1 <= self.chunk_frames <= MAX_STREAM_CHUNK_FRAMES:
            raise ValueError("chunk_frames 必须在 1 到 25 之间")
        resolved = GenerateCommand(
            prompt=self.prompt,
            reference_audio=self.reference_audio,
            text_weight=self.text_weight,
            audio_weight=self.audio_weight,
            duration=self.duration,
            sampling=self.sampling,
            control=self.control,
        ).resolve(defaults)
        return ResolvedStreamGenerateCommand(
            prompt=resolved.prompt,
            reference_audio=resolved.reference_audio,
            text_weight=resolved.text_weight,
            audio_weight=resolved.audio_weight,
            duration=resolved.duration,
            chunk_frames=self.chunk_frames,
            sampling=resolved.sampling,
            control_timeline=resolved.control_timeline,
        )


@dataclass(frozen=True, slots=True)
class ResolvedStreamGenerateCommand:
    prompt: str | None
    reference_audio: AudioInput | None
    text_weight: float
    audio_weight: float
    duration: float
    chunk_frames: int
    sampling: SamplingConfig
    control_timeline: ControlTimeline | None = None

    @property
    def sample_count(self) -> int:
        return round(self.duration * SAMPLE_RATE)


@dataclass(frozen=True, slots=True)
class StreamUpdateCommand:
    revision: int
    effective_frame: int | None = None
    prompt_present: bool = False
    prompt: str | None = None
    sampling: SamplingOverrides = SamplingOverrides()
    notes: tuple[NoteEvent, ...] | None = None
    drums: tuple[DrumEvent, ...] | None = None
    notes_mode: ControlMode = "guide"
    drums_mode: ControlMode = "guide"

    def validate(self) -> None:
        if self.revision < 0:
            raise ValueError("revision 必须大于等于 0")
        if self.effective_frame is not None and self.effective_frame < 0:
            raise ValueError("effectiveFrame 必须大于等于 0")
        if self.prompt_present and self.prompt is not None and not self.prompt.strip():
            raise ValueError("prompt 必须为非空字符串或 null")
        if self.notes_mode not in ("guide", "strict"):
            raise ValueError("notesMode 必须是 guide 或 strict")
        if self.drums_mode not in ("guide", "strict"):
            raise ValueError("drumsMode 必须是 guide 或 strict")
        for event in self.notes or ():
            event.validate()
        for event in self.drums or ():
            event.validate()
        sampling_changed = any(
            getattr(self.sampling, field.name) is not None
            for field in fields(self.sampling)
        )
        if not (
            self.prompt_present
            or sampling_changed
            or self.notes is not None
            or self.drums is not None
        ):
            raise ValueError("update 消息至少需要包含一个可更新字段")


@dataclass(frozen=True, slots=True)
class StreamUpdateResult:
    revision: int
    effective_frame: int


@dataclass(frozen=True, slots=True)
class StreamExtendCommand:
    revision: int
    additional_duration: float

    def validate(self) -> None:
        if self.revision < 0:
            raise ValueError("revision 必须大于等于 0")
        if (
            not math.isfinite(self.additional_duration)
            or self.additional_duration <= 0
            or self.additional_duration > MAX_DURATION
        ):
            raise ValueError("additionalDuration 必须大于 0 且不超过 300 秒")


@dataclass(frozen=True, slots=True)
class StreamExtendResult:
    revision: int
    previous_sample_count: int
    sample_count: int

    @property
    def previous_duration_ms(self) -> int:
        return round(self.previous_sample_count * 1000 / SAMPLE_RATE)

    @property
    def duration_ms(self) -> int:
        return round(self.sample_count * 1000 / SAMPLE_RATE)


@dataclass(frozen=True, slots=True)
class AudioChunk:
    sequence: int
    start_sample: int
    sample_rate: int
    channels: int
    audio: np.ndarray

    @property
    def timestamp_ms(self) -> int:
        return round(self.start_sample * 1000 / self.sample_rate)


@dataclass(frozen=True, slots=True)
class GenerateResult:
    sample_rate: int
    channels: int
    audio: np.ndarray
