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
        return InfoResponse(model=config.model)

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
            result = get_engine(request).generate(body.prompt, body.duration)
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
