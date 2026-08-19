from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

from . import __version__
from .config import RuntimeConfig
from .core import CHANNELS, DEFAULT_DURATION, DEFAULT_STREAM_CHUNK_FRAMES, DEFAULT_STYLE_WEIGHT, SAMPLE_RATE
from .encoding import AudioEncodingError, AudioFormat, decode_audio, encode_audio
from .pcm import PCM_MEDIA_TYPE, PCM_SAMPLE_FORMAT, encode_pcm_chunk
from . import parameter_docs as parameter_help
from .schemas import AudioGenerateRequest, GenerateRequest, StreamGenerateRequest
from .service import GenerationService, ModelBusyError, StreamingSession
from .ws import router as websocket_router
from .streaming_ws import router as streaming_websocket_router


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
    temperature: float = Field(description=parameter_help.TEMPERATURE)
    top_k: int = Field(description=parameter_help.TOP_K)
    cfg_musiccoca: float = Field(description=parameter_help.CFG_MUSICCOCA)
    cfg_notes: float = Field(description=parameter_help.CFG_NOTES)
    cfg_drums: float = Field(description=parameter_help.CFG_DRUMS)
    warmup_steps: int = Field(description=parameter_help.WARMUP_STEPS)
    seed: int = Field(description=parameter_help.SEED)
    use_mapper: bool = Field(description=parameter_help.USE_MAPPER)
    pool_across_time: bool = Field(description=parameter_help.POOL_ACROSS_TIME)


ServiceFactory = Callable[[RuntimeConfig], GenerationService]


def audio_generation_options(
    prompt: Annotated[str | None, Form(min_length=1, description="可选文本风格；与参考音频同时提供时进行加权混合")] = None,
    text_weight: Annotated[float, Form(ge=0, description=parameter_help.TEXT_WEIGHT)] = DEFAULT_STYLE_WEIGHT,
    audio_weight: Annotated[float, Form(ge=0, description=parameter_help.AUDIO_WEIGHT)] = DEFAULT_STYLE_WEIGHT,
    duration: Annotated[float, Form(gt=0, le=300, description=parameter_help.DURATION)] = DEFAULT_DURATION,
    temperature: Annotated[float | None, Form(gt=0, description=parameter_help.TEMPERATURE)] = None,
    top_k: Annotated[int | None, Form(ge=1, description=parameter_help.TOP_K)] = None,
    cfg_musiccoca: Annotated[float | None, Form(description=parameter_help.CFG_MUSICCOCA)] = None,
    cfg_notes: Annotated[float | None, Form(description=parameter_help.CFG_NOTES)] = None,
    cfg_drums: Annotated[float | None, Form(description=parameter_help.CFG_DRUMS)] = None,
    seed: Annotated[int | None, Form(description=parameter_help.SEED)] = None,
    use_mapper: Annotated[bool | None, Form(description=parameter_help.USE_MAPPER)] = None,
    pool_across_time: Annotated[bool | None, Form(description=parameter_help.POOL_ACROSS_TIME)] = None,
    format: Annotated[AudioFormat, Form()] = "wav",
    bitrate: Annotated[int | None, Form(ge=32, le=320)] = None,
) -> AudioGenerateRequest:
    return AudioGenerateRequest(
        prompt=prompt,
        text_weight=text_weight,
        audio_weight=audio_weight,
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


def audio_stream_options(
    prompt: Annotated[str | None, Form(min_length=1, description="可选文本风格；与参考音频同时提供时进行加权混合")] = None,
    text_weight: Annotated[float, Form(ge=0, description=parameter_help.TEXT_WEIGHT)] = DEFAULT_STYLE_WEIGHT,
    audio_weight: Annotated[float, Form(ge=0, description=parameter_help.AUDIO_WEIGHT)] = DEFAULT_STYLE_WEIGHT,
    duration: Annotated[float, Form(gt=0, le=300, description=parameter_help.DURATION)] = DEFAULT_DURATION,
    chunk_frames: Annotated[int, Form(ge=1, le=25, description=parameter_help.CHUNK_FRAMES)] = DEFAULT_STREAM_CHUNK_FRAMES,
    temperature: Annotated[float | None, Form(gt=0, description=parameter_help.TEMPERATURE)] = None,
    top_k: Annotated[int | None, Form(ge=1, description=parameter_help.TOP_K)] = None,
    cfg_musiccoca: Annotated[float | None, Form(description=parameter_help.CFG_MUSICCOCA)] = None,
    cfg_notes: Annotated[float | None, Form(description=parameter_help.CFG_NOTES)] = None,
    cfg_drums: Annotated[float | None, Form(description=parameter_help.CFG_DRUMS)] = None,
    seed: Annotated[int | None, Form(description=parameter_help.SEED)] = None,
    use_mapper: Annotated[bool | None, Form(description=parameter_help.USE_MAPPER)] = None,
    pool_across_time: Annotated[bool | None, Form(description=parameter_help.POOL_ACROSS_TIME)] = None,
) -> StreamGenerateRequest:
    return StreamGenerateRequest(
        prompt=prompt, text_weight=text_weight, audio_weight=audio_weight,
        duration=duration, chunk_frames=chunk_frames,
        temperature=temperature, top_k=top_k,
        cfg_musiccoca=cfg_musiccoca, cfg_notes=cfg_notes, cfg_drums=cfg_drums,
        seed=seed, use_mapper=use_mapper, pool_across_time=pool_across_time,
    )


def _pcm_headers() -> dict[str, str]:
    return {
        "X-Audio-Sample-Rate": str(SAMPLE_RATE),
        "X-Audio-Channels": str(CHANNELS),
        "X-Audio-Sample-Format": PCM_SAMPLE_FORMAT,
        "Cache-Control": "no-store",
    }


def _stream_pcm(session: StreamingSession) -> Iterator[bytes]:
    with session:
        while (chunk := session.next_chunk()) is not None:
            yield encode_pcm_chunk(chunk)


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
        description="模型在服务启动时加载一次；同一时间只运行一个普通生成或流式会话。",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    app.include_router(websocket_router)
    app.include_router(streaming_websocket_router)

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
            400: {"description": "业务参数或编码选项无效"},
            409: {"description": "模型正被另一个生成会话占用"},
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
        except ModelBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
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
            400: {"description": "业务参数或编码选项无效"},
            409: {"description": "模型正被另一个生成会话占用"},
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
        except ModelBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
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
        "/stream",
        response_class=StreamingResponse,
        responses={
            200: {
                "description": "连续返回 48kHz 双声道 float32le PCM 分片",
                "content": {PCM_MEDIA_TYPE: {"schema": {"type": "string", "format": "binary"}}},
            },
            400: {"description": "流式生成参数无效"},
            409: {"description": "模型正被另一个生成会话占用"},
            500: {"description": "流式生成启动失败"},
        },
        tags=["流式生成"],
    )
    def stream(body: StreamGenerateRequest, request: Request) -> StreamingResponse:
        try:
            session = get_service(request).open_stream(body.to_command())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ModelBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="流式生成启动失败") from exc
        return StreamingResponse(
            _stream_pcm(session), media_type=PCM_MEDIA_TYPE, headers=_pcm_headers()
        )

    @app.post(
        "/stream/audio",
        response_class=StreamingResponse,
        responses={
            200: {
                "description": "根据参考音频连续返回 float32le PCM 分片",
                "content": {PCM_MEDIA_TYPE: {"schema": {"type": "string", "format": "binary"}}},
            },
            400: {"description": "流式生成参数或参考音频无效"},
            409: {"description": "模型正被另一个生成会话占用"},
            500: {"description": "流式生成启动失败"},
        },
        tags=["流式生成"],
    )
    async def stream_with_audio(
        request: Request,
        options: Annotated[StreamGenerateRequest, Depends(audio_stream_options)],
        audio: Annotated[UploadFile, File(description="参考音频文件")],
    ) -> StreamingResponse:
        try:
            reference_audio = decode_audio(await audio.read())
            session = await asyncio.to_thread(
                get_service(request).open_stream,
                options.to_command(reference_audio),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ModelBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="流式生成启动失败") from exc
        return StreamingResponse(
            _stream_pcm(session), media_type=PCM_MEDIA_TYPE, headers=_pcm_headers()
        )

    return app
