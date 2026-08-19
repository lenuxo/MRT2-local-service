from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, Field

from . import __version__
from .config import RuntimeConfig
from .core import CHANNELS, DEFAULT_DURATION, SAMPLE_RATE
from .encoding import AudioEncodingError, AudioFormat, decode_audio, encode_audio
from .schemas import AudioGenerateRequest, GenerateRequest
from .service import GenerationService
from .ws import router as websocket_router


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


ServiceFactory = Callable[[RuntimeConfig], GenerationService]


def audio_generation_options(
    duration: Annotated[float, Form(gt=0, le=300)] = DEFAULT_DURATION,
    temperature: Annotated[float | None, Form(gt=0)] = None,
    top_k: Annotated[int | None, Form(ge=1)] = None,
    cfg_musiccoca: Annotated[float | None, Form()] = None,
    cfg_notes: Annotated[float | None, Form()] = None,
    cfg_drums: Annotated[float | None, Form()] = None,
    seed: Annotated[int | None, Form()] = None,
    use_mapper: Annotated[bool | None, Form()] = None,
    pool_across_time: Annotated[bool | None, Form()] = None,
    format: Annotated[AudioFormat, Form()] = "wav",
    bitrate: Annotated[int | None, Form(ge=32, le=320)] = None,
) -> AudioGenerateRequest:
    return AudioGenerateRequest(
        duration=duration,
        temperature=temperature,
        top_k=top_k,
        cfg_musiccoca=cfg_musiccoca,
        cfg_notes=cfg_notes,
        cfg_drums=cfg_drums,
        seed=seed,
        use_mapper=use_mapper,
        pool_across_time=pool_across_time,
        format=format,
        bitrate=bitrate,
    )


def create_app(
    config: RuntimeConfig,
    service_factory: ServiceFactory = GenerationService,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        service = service_factory(config)
        service.load()
        app.state.service = service
        yield
        app.state.service = None

    app = FastAPI(
        title="MRT2 本地服务 API",
        summary="本地 Magenta RealTime 2 音频生成服务",
        description="模型在服务启动时加载一次，所有生成请求串行执行。",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.include_router(websocket_router)

    def get_service(request: Request) -> GenerationService:
        return request.app.state.service

    @app.get("/health", response_model=HealthResponse, tags=["服务"])
    def health(request: Request) -> HealthResponse:
        service = get_service(request)
        return HealthResponse(
            status="ok",
            model=config.model.name,
            loaded=service.is_loaded,
        )

    @app.get(
        "/info",
        response_model=InfoResponse,
        response_model_by_alias=True,
        tags=["服务"],
    )
    def info() -> InfoResponse:
        return InfoResponse(
            model=config.model.name,
            temperature=config.sampling.temperature,
            top_k=config.sampling.top_k,
            cfg_musiccoca=config.sampling.cfg_musiccoca,
            cfg_notes=config.sampling.cfg_notes,
            cfg_drums=config.sampling.cfg_drums,
            warmup_steps=config.model.warmup_steps,
            seed=config.sampling.seed,
            use_mapper=config.sampling.use_mapper,
            pool_across_time=config.sampling.pool_across_time,
        )

    @app.post(
        "/generate",
        response_class=Response,
        responses={
            200: {
                "description": "生成的 WAV 或 MP3 音频",
                "content": {
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}},
                    "audio/mpeg": {"schema": {"type": "string", "format": "binary"}},
                },
            },
            500: {"description": "推理失败"},
        },
        tags=["生成"],
    )
    def generate(body: GenerateRequest, request: Request) -> Response:
        try:
            encoding = body.encoding_options()
            result = get_service(request).generate(body.to_command())
            encoded = encode_audio(result, encoding)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except AudioEncodingError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="音频生成失败") from exc
        return Response(
            content=encoded.data,
            media_type=encoded.media_type,
            headers={
                "Content-Disposition":
                    f'attachment; filename="output{encoded.extension}"'
            },
        )

    @app.post(
        "/generate/audio",
        response_class=Response,
        responses={
            200: {
                "description": "根据参考音频风格生成的 WAV 或 MP3",
                "content": {
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}},
                    "audio/mpeg": {"schema": {"type": "string", "format": "binary"}},
                },
            },
            500: {"description": "推理或编码失败"},
        },
        tags=["生成"],
    )
    async def generate_with_audio(
        request: Request,
        options: Annotated[AudioGenerateRequest, Depends(audio_generation_options)],
        audio: Annotated[UploadFile, File(description="参考音频文件")],
    ) -> Response:
        try:
            reference_audio = decode_audio(await audio.read())
            encoding = options.encoding_options()
            result = await asyncio.to_thread(
                get_service(request).generate,
                options.to_command(reference_audio),
            )
            encoded = await asyncio.to_thread(encode_audio, result, encoding)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except AudioEncodingError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="音频生成失败") from exc
        return Response(
            content=encoded.data,
            media_type=encoded.media_type,
            headers={
                "Content-Disposition":
                    f'attachment; filename="output{encoded.extension}"'
            },
        )

    return app
