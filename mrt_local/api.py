from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .config import CHANNELS, SAMPLE_RATE, EngineConfig
from .engine import MrtEngine


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: Annotated[str, Field(min_length=1, description="文本提示词")]
    duration: Annotated[float, Field(gt=0, le=300, description="生成时长（秒）")] = 10
    temperature: Annotated[float | None, Field(gt=0, description="采样温度；空值使用服务默认值")] = None
    top_k: Annotated[int | None, Field(ge=1, description="Top-k 采样阈值；空值使用服务默认值")] = None
    cfg_musiccoca: Annotated[float | None, Field(description="文本/音频风格 CFG；空值使用服务默认值")] = None
    cfg_notes: Annotated[float | None, Field(description="音符条件 CFG；空值使用服务默认值")] = None
    cfg_drums: Annotated[float | None, Field(description="鼓条件 CFG；空值使用服务默认值")] = None
    seed: Annotated[int | None, Field(description="MusicCoCa embedding 随机种子；空值使用服务默认值")] = None
    use_mapper: Annotated[bool | None, Field(description="是否使用 MusicCoCa mapper；空值使用服务默认值")] = None
    pool_across_time: Annotated[bool | None, Field(description="是否在时间维聚合 embedding；空值使用服务默认值")] = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model: Literal["mrt2_small", "mrt2_base"]
    loaded: bool


class InfoResponse(BaseModel):
    model: Literal["mrt2_small", "mrt2_base"]
    backend: Literal["mlx"] = "mlx"
    sample_rate: int = Field(SAMPLE_RATE, serialization_alias="sampleRate")
    channels: int = CHANNELS
    platform: Literal["macos"] = "macos"
    architecture: Literal["arm64"] = "arm64"
    temperature: float
    top_k: int
    cfg_musiccoca: float
    cfg_notes: float
    cfg_drums: float
    warmup_steps: int
    seed: int
    use_mapper: bool
    pool_across_time: bool


EngineFactory = Callable[[EngineConfig], MrtEngine]


def create_app(
    config: EngineConfig,
    engine_factory: EngineFactory = MrtEngine,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = engine_factory(config)
        engine.load()
        app.state.engine = engine
        yield
        app.state.engine = None

    app = FastAPI(
        title="MRT2 本地服务 API",
        summary="本地 Magenta RealTime 2 音频生成服务",
        description="模型在服务启动时加载一次，所有生成请求串行执行。",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    def get_engine(request: Request) -> MrtEngine:
        return request.app.state.engine

    @app.get("/health", response_model=HealthResponse, tags=["服务"])
    def health(request: Request) -> HealthResponse:
        engine = get_engine(request)
        return HealthResponse(status="ok", model=config.model, loaded=engine.is_loaded)

    @app.get(
        "/info",
        response_model=InfoResponse,
        response_model_by_alias=True,
        tags=["服务"],
    )
    def info() -> InfoResponse:
        return InfoResponse(
            model=config.model,
            temperature=config.temperature,
            top_k=config.top_k,
            cfg_musiccoca=config.cfg_musiccoca,
            cfg_notes=config.cfg_notes,
            cfg_drums=config.cfg_drums,
            warmup_steps=config.warmup_steps,
            seed=config.seed,
            use_mapper=config.use_mapper,
            pool_across_time=config.pool_across_time,
        )

    @app.post(
        "/generate",
        response_class=Response,
        responses={
            200: {
                "description": "生成的 48 kHz 双声道浮点 WAV",
                "content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}},
            },
            500: {"description": "推理失败"},
        },
        tags=["生成"],
    )
    def generate(body: GenerateRequest, request: Request) -> Response:
        try:
            result = get_engine(request).generate(
                body.prompt,
                body.duration,
                temperature=body.temperature,
                top_k=body.top_k,
                cfg_musiccoca=body.cfg_musiccoca,
                cfg_notes=body.cfg_notes,
                cfg_drums=body.cfg_drums,
                seed=body.seed,
                use_mapper=body.use_mapper,
                pool_across_time=body.pool_across_time,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="音频生成失败") from exc
        return Response(
            content=result.to_wav_bytes(),
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="output.wav"'},
        )

    return app
