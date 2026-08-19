from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from .core import DEFAULT_DURATION, GenerateCommand, SamplingOverrides


class GenerateRequest(BaseModel):
    """HTTP 与 WebSocket 共用的 JSON 生成请求。"""

    model_config = ConfigDict(extra="forbid")
    prompt: Annotated[str, Field(min_length=1, description="文本提示词")]
    duration: Annotated[
        float,
        Field(gt=0, le=300, description="生成时长（秒）"),
    ] = DEFAULT_DURATION
    temperature: Annotated[
        float | None,
        Field(gt=0, description="采样温度；空值使用服务默认值"),
    ] = None
    top_k: Annotated[
        int | None,
        Field(ge=1, description="Top-k 采样阈值；空值使用服务默认值"),
    ] = None
    cfg_musiccoca: Annotated[
        float | None,
        Field(description="文本/音频风格 CFG；空值使用服务默认值"),
    ] = None
    cfg_notes: Annotated[
        float | None,
        Field(description="音符条件 CFG；空值使用服务默认值"),
    ] = None
    cfg_drums: Annotated[
        float | None,
        Field(description="鼓条件 CFG；空值使用服务默认值"),
    ] = None
    seed: Annotated[
        int | None,
        Field(description="MusicCoCa embedding 随机种子；空值使用服务默认值"),
    ] = None
    use_mapper: Annotated[
        bool | None,
        Field(description="是否使用 MusicCoCa mapper；空值使用服务默认值"),
    ] = None
    pool_across_time: Annotated[
        bool | None,
        Field(description="是否在时间维聚合 embedding；空值使用服务默认值"),
    ] = None

    def to_command(self) -> GenerateCommand:
        return GenerateCommand(
            prompt=self.prompt,
            duration=self.duration,
            sampling=SamplingOverrides(
                temperature=self.temperature,
                top_k=self.top_k,
                cfg_musiccoca=self.cfg_musiccoca,
                cfg_notes=self.cfg_notes,
                cfg_drums=self.cfg_drums,
                seed=self.seed,
                use_mapper=self.use_mapper,
                pool_across_time=self.pool_across_time,
            ),
        )
