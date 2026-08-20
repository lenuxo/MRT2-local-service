from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .core import (
    AudioInput,
    ControlInput,
    DrumEvent,
    DEFAULT_DURATION,
    DEFAULT_STREAM_CHUNK_FRAMES,
    DEFAULT_STYLE_WEIGHT,
    GenerateCommand,
    MAX_PROMPT_COMPONENTS,
    MAX_PROMPT_COMPONENT_CHARS,
    MAX_PROMPT_COMPONENT_WEIGHT,
    NoteEvent,
    PromptComponent,
    StreamGenerateCommand,
    SamplingOverrides,
)
from .encoding import AudioEncodingOptions, AudioFormat
from . import parameter_docs as parameter_help


class NoteEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pitch: Annotated[int, Field(ge=0, le=127, description="MIDI 音高，0～127")]
    start: Annotated[float, Field(ge=0, description="起音时间（秒）")]
    duration: Annotated[float, Field(gt=0, description="持续时间（秒）")]

    def to_core(self) -> NoteEvent:
        return NoteEvent(self.pitch, self.start, self.duration)


class DrumEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    time: Annotated[float, Field(ge=0, description="鼓点触发时间（秒）")]

    def to_core(self) -> DrumEvent:
        return DrumEvent(self.time)


class PromptComponentRequest(BaseModel):
    """高级文本风格混合中的一个独立语义片段。"""

    model_config = ConfigDict(extra="forbid")
    text: Annotated[str, Field(
        min_length=1,
        max_length=MAX_PROMPT_COMPONENT_CHARS,
        description=parameter_help.PROMPT_COMPONENT_TEXT,
    )]
    weight: Annotated[float, Field(
        ge=0,
        le=MAX_PROMPT_COMPONENT_WEIGHT,
        description=parameter_help.PROMPT_COMPONENT_WEIGHT,
    )] = 1.0

    def to_core(self) -> PromptComponent:
        return PromptComponent(self.text, self.weight)


def _prompt_components(
    items: list[PromptComponentRequest],
) -> tuple[PromptComponent, ...]:
    return tuple(item.to_core() for item in items)


def _validate_prompt_modes(
    prompt: str | None, items: list[PromptComponentRequest]
) -> None:
    if prompt is not None and items:
        raise ValueError("prompt 与 prompt_components 不能同时提供")


class SamplingOptions(BaseModel):
    """所有传输外壳共用的生成参数。"""

    model_config = ConfigDict(extra="forbid")
    duration: Annotated[float, Field(gt=0, le=300, description=parameter_help.DURATION)] = DEFAULT_DURATION
    temperature: Annotated[float | None, Field(gt=0, description=parameter_help.TEMPERATURE)] = None
    top_k: Annotated[int | None, Field(ge=1, description=parameter_help.TOP_K)] = None
    cfg_musiccoca: Annotated[float | None, Field(description=parameter_help.CFG_MUSICCOCA)] = None
    cfg_notes: Annotated[float | None, Field(description=parameter_help.CFG_NOTES)] = None
    cfg_drums: Annotated[float | None, Field(description=parameter_help.CFG_DRUMS)] = None
    seed: Annotated[int | None, Field(description=parameter_help.SEED)] = None
    use_mapper: Annotated[bool | None, Field(description=parameter_help.USE_MAPPER)] = None
    pool_across_time: Annotated[bool | None, Field(description=parameter_help.POOL_ACROSS_TIME)] = None
    notes: list[NoteEventRequest] = Field(default_factory=list, description="按秒表示的 MIDI 音符事件")
    drums: list[DrumEventRequest] = Field(default_factory=list, description="按秒表示的通用鼓点触发事件")
    notes_mode: Literal["guide", "strict"] = Field("guide", description="guide 允许额外音高；strict 关闭未指定音高")
    drums_mode: Literal["guide", "strict"] = Field("guide", description="guide 允许额外鼓点；strict 关闭未指定鼓点")

    def sampling_overrides(self) -> SamplingOverrides:
        return SamplingOverrides(
            temperature=self.temperature,
            top_k=self.top_k,
            cfg_musiccoca=self.cfg_musiccoca,
            cfg_notes=self.cfg_notes,
            cfg_drums=self.cfg_drums,
            seed=self.seed,
            use_mapper=self.use_mapper,
            pool_across_time=self.pool_across_time,
        )

    def control_input(self) -> ControlInput | None:
        control = ControlInput(
            notes=tuple(item.to_core() for item in self.notes),
            drums=tuple(item.to_core() for item in self.drums),
            notes_mode=self.notes_mode,
            drums_mode=self.drums_mode,
        )
        return control if control.has_events else None


class GenerationOptions(SamplingOptions):
    """完整文件生成的参数和输出编码选项。"""

    format: Annotated[AudioFormat, Field(description="输出音频格式")] = "wav"
    bitrate: Annotated[int | None, Field(ge=32, le=320, description="MP3 比特率（kbps）")] = None

    def encoding_options(self) -> AudioEncodingOptions:
        options = AudioEncodingOptions(format=self.format, bitrate=self.bitrate)
        options.validate()
        return options


class GenerateRequest(GenerationOptions):
    prompt: Annotated[str | None, Field(min_length=1, description="可选文本提示词；没有事件控制时必填")] = None
    prompt_components: list[PromptComponentRequest] = Field(
        default_factory=list,
        max_length=MAX_PROMPT_COMPONENTS,
        description=parameter_help.PROMPT_COMPONENTS,
    )

    @model_validator(mode="after")
    def validate_prompt_modes(self) -> GenerateRequest:
        _validate_prompt_modes(self.prompt, self.prompt_components)
        return self

    def to_command(self) -> GenerateCommand:
        return GenerateCommand(
            prompt=self.prompt,
            prompt_components=_prompt_components(self.prompt_components),
            duration=self.duration,
            sampling=self.sampling_overrides(),
            control=self.control_input(),
        )


class AudioGenerateRequest(GenerationOptions):
    prompt: Annotated[str | None, Field(min_length=1, description="可选文本风格；与参考音频同时提供时进行加权混合")] = None
    prompt_components: list[PromptComponentRequest] = Field(
        default_factory=list,
        max_length=MAX_PROMPT_COMPONENTS,
        description=parameter_help.PROMPT_COMPONENTS,
    )
    text_weight: Annotated[float, Field(ge=0, description=parameter_help.TEXT_WEIGHT)] = DEFAULT_STYLE_WEIGHT
    audio_weight: Annotated[float, Field(ge=0, description=parameter_help.AUDIO_WEIGHT)] = DEFAULT_STYLE_WEIGHT

    @model_validator(mode="after")
    def validate_prompt_modes(self) -> AudioGenerateRequest:
        _validate_prompt_modes(self.prompt, self.prompt_components)
        return self

    def to_command(
        self,
        reference_audio: AudioInput | None,
        control: ControlInput | None = None,
    ) -> GenerateCommand:
        return GenerateCommand(
            prompt=self.prompt,
            prompt_components=_prompt_components(self.prompt_components),
            reference_audio=reference_audio,
            text_weight=self.text_weight,
            audio_weight=self.audio_weight,
            duration=self.duration,
            sampling=self.sampling_overrides(),
            control=control or self.control_input(),
        )


class StreamGenerateRequest(SamplingOptions):
    prompt: Annotated[str | None, Field(min_length=1, description="文本提示词")] = None
    prompt_components: list[PromptComponentRequest] = Field(
        default_factory=list,
        max_length=MAX_PROMPT_COMPONENTS,
        description=parameter_help.PROMPT_COMPONENTS,
    )
    text_weight: Annotated[float, Field(ge=0, description=parameter_help.TEXT_WEIGHT)] = DEFAULT_STYLE_WEIGHT
    audio_weight: Annotated[float, Field(ge=0, description=parameter_help.AUDIO_WEIGHT)] = DEFAULT_STYLE_WEIGHT
    chunk_frames: Annotated[int, Field(ge=1, le=25, description=parameter_help.CHUNK_FRAMES)] = DEFAULT_STREAM_CHUNK_FRAMES

    @model_validator(mode="after")
    def validate_prompt_modes(self) -> StreamGenerateRequest:
        _validate_prompt_modes(self.prompt, self.prompt_components)
        return self

    def to_command(
        self,
        reference_audio: AudioInput | None = None,
        control: ControlInput | None = None,
    ) -> StreamGenerateCommand:
        return StreamGenerateCommand(
            prompt=self.prompt,
            prompt_components=_prompt_components(self.prompt_components),
            reference_audio=reference_audio,
            text_weight=self.text_weight,
            audio_weight=self.audio_weight,
            duration=self.duration,
            chunk_frames=self.chunk_frames,
            sampling=self.sampling_overrides(),
            control=control or self.control_input(),
        )
