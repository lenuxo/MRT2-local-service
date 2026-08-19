from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from .core import (
    AudioInput,
    DEFAULT_DURATION,
    DEFAULT_STREAM_CHUNK_FRAMES,
    DEFAULT_STYLE_WEIGHT,
    GenerateCommand,
    StreamGenerateCommand,
    SamplingOverrides,
)
from .encoding import AudioEncodingOptions, AudioFormat


class SamplingOptions(BaseModel):
    """所有传输外壳共用的生成参数。"""

    model_config = ConfigDict(extra="forbid")
    duration: Annotated[float, Field(gt=0, le=300, description="生成时长（秒）")] = DEFAULT_DURATION
    temperature: Annotated[float | None, Field(gt=0, description="采样温度；空值使用服务默认值")] = None
    top_k: Annotated[int | None, Field(ge=1, description="Top-k 采样阈值；空值使用服务默认值")] = None
    cfg_musiccoca: Annotated[float | None, Field(description="文本/音频风格 CFG；空值使用服务默认值")] = None
    cfg_notes: Annotated[float | None, Field(description="音符条件 CFG；空值使用服务默认值")] = None
    cfg_drums: Annotated[float | None, Field(description="鼓条件 CFG；空值使用服务默认值")] = None
    seed: Annotated[int | None, Field(description="MusicCoCa embedding 随机种子；空值使用服务默认值")] = None
    use_mapper: Annotated[bool | None, Field(description="是否使用 MusicCoCa mapper；空值使用服务默认值")] = None
    pool_across_time: Annotated[bool | None, Field(description="是否在时间维聚合 embedding；空值使用服务默认值")] = None
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


class GenerationOptions(SamplingOptions):
    """完整文件生成的参数和输出编码选项。"""

    format: Annotated[AudioFormat, Field(description="输出音频格式")] = "wav"
    bitrate: Annotated[int | None, Field(ge=32, le=320, description="MP3 比特率（kbps）")] = None

    def encoding_options(self) -> AudioEncodingOptions:
        options = AudioEncodingOptions(format=self.format, bitrate=self.bitrate)
        options.validate()
        return options


class GenerateRequest(GenerationOptions):
    prompt: Annotated[str, Field(min_length=1, description="文本提示词")]

    def to_command(self) -> GenerateCommand:
        return GenerateCommand(
            prompt=self.prompt,
            duration=self.duration,
            sampling=self.sampling_overrides(),
        )


class AudioGenerateRequest(GenerationOptions):
    prompt: Annotated[str | None, Field(min_length=1)] = None
    text_weight: Annotated[float, Field(ge=0)] = DEFAULT_STYLE_WEIGHT
    audio_weight: Annotated[float, Field(ge=0)] = DEFAULT_STYLE_WEIGHT

    def to_command(self, reference_audio: AudioInput) -> GenerateCommand:
        return GenerateCommand(
            prompt=self.prompt,
            reference_audio=reference_audio,
            text_weight=self.text_weight,
            audio_weight=self.audio_weight,
            duration=self.duration,
            sampling=self.sampling_overrides(),
        )


class StreamGenerateRequest(SamplingOptions):
    prompt: Annotated[str | None, Field(min_length=1, description="文本提示词")] = None
    text_weight: Annotated[float, Field(ge=0)] = DEFAULT_STYLE_WEIGHT
    audio_weight: Annotated[float, Field(ge=0)] = DEFAULT_STYLE_WEIGHT
    chunk_frames: Annotated[int, Field(ge=1, le=25, description="每个 PCM 分片包含的 40ms 模型帧数")] = DEFAULT_STREAM_CHUNK_FRAMES

    def to_command(self, reference_audio: AudioInput | None = None) -> StreamGenerateCommand:
        return StreamGenerateCommand(
            prompt=self.prompt,
            reference_audio=reference_audio,
            text_weight=self.text_weight,
            audio_weight=self.audio_weight,
            duration=self.duration,
            chunk_frames=self.chunk_frames,
            sampling=self.sampling_overrides(),
        )
