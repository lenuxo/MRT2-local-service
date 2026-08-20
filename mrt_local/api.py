from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from fastapi.responses import StreamingResponse

from . import __version__
from .config import RuntimeConfig
from .capabilities import stream_capabilities
from .core import CHANNELS, DEFAULT_DURATION, DEFAULT_STREAM_CHUNK_FRAMES, DEFAULT_STYLE_WEIGHT, SAMPLE_RATE
from .encoding import AudioEncodingError, AudioFormat, decode_audio, encode_audio
from .midi import decode_midi
from .pcm import PCM_MEDIA_TYPE, PCM_SAMPLE_FORMAT, encode_pcm_chunk
from . import parameter_docs as parameter_help
from .schemas import (
    AudioGenerateRequest,
    GenerateRequest,
    PromptComponentRequest,
    StreamGenerateRequest,
)
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


class StreamLimitsResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    control_message_bytes: int = Field(alias="controlMessageBytes")
    reference_audio_bytes: int = Field(alias="referenceAudioBytes")
    reference_audio_timeout_seconds: float = Field(
        alias="referenceAudioTimeoutSeconds"
    )
    prompt_components: int = Field(alias="promptComponents")
    prompt_component_chars: int = Field(alias="promptComponentChars")
    prompt_total_chars: int = Field(alias="promptTotalChars")


class StreamCapabilitiesResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    protocol_version: int = Field(alias="protocolVersion")
    update: list[str]
    effective_frame: bool = Field(alias="effectiveFrame")
    extend_duration: bool = Field(alias="extendDuration")
    chunk_frames: bool = Field(alias="chunkFrames")
    realtime: bool
    reference_audio: bool = Field(alias="referenceAudio")
    style_weights: bool = Field(alias="styleWeights")
    prompt_components: bool = Field(alias="promptComponents")
    metrics: bool
    revision_policy: Literal["strictly_increasing_idempotent_replay"] = Field(
        alias="revisionPolicy"
    )
    limits: StreamLimitsResponse


class CapabilitiesResponse(BaseModel):
    version: str
    models: list[Literal["mrt2_small", "mrt2_base"]]
    active_model: Literal["mrt2_small", "mrt2_base"] = Field(
        serialization_alias="activeModel"
    )
    sample_rate: int = Field(SAMPLE_RATE, serialization_alias="sampleRate")
    channels: int = CHANNELS
    output_formats: list[str] = Field(
        default_factory=lambda: ["wav", "mp3", "float32le"],
        serialization_alias="outputFormats",
    )
    transports: list[str] = Field(
        default_factory=lambda: ["http", "http_streaming", "websocket"]
    )
    stream: StreamCapabilitiesResponse


class StatusResponse(BaseModel):
    model: Literal["mrt2_small", "mrt2_base"]
    loaded: bool
    busy: bool
    operation: Literal["generate", "stream"] | None
    session_id: str | None = Field(serialization_alias="sessionId")
    generated_samples: int = Field(serialization_alias="generatedSamples")
    target_samples: int = Field(serialization_alias="targetSamples")
    elapsed_ms: int | None = Field(serialization_alias="elapsedMs")


ServiceFactory = Callable[[RuntimeConfig], GenerationService]
_PROMPT_COMPONENTS_ADAPTER = TypeAdapter(list[PromptComponentRequest])


def _parse_prompt_components_json(
    value: str | None,
) -> list[PromptComponentRequest]:
    if value is None:
        return []
    try:
        return _PROMPT_COMPONENTS_ADAPTER.validate_json(value)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail="prompt_components 必须是有效的 JSON 数组",
        ) from exc


def audio_generation_options(
    prompt: Annotated[str | None, Form(min_length=1, description="可选文本风格；与参考音频同时提供时进行加权混合")] = None,
    prompt_components: Annotated[str | None, Form(description=parameter_help.PROMPT_COMPONENTS + "；multipart 中传 JSON 数组字符串")] = None,
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
    notes_mode: Annotated[Literal["guide", "strict"], Form(description="音符控制模式")] = "guide",
    drums_mode: Annotated[Literal["guide", "strict"], Form(description="鼓点控制模式")] = "guide",
    format: Annotated[AudioFormat, Form()] = "wav",
    bitrate: Annotated[int | None, Form(ge=32, le=320)] = None,
) -> AudioGenerateRequest:
    return AudioGenerateRequest(
        prompt=prompt,
        prompt_components=_parse_prompt_components_json(prompt_components),
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
        notes_mode=notes_mode,
        drums_mode=drums_mode,
        format=format,
        bitrate=bitrate,
    )


def audio_stream_options(
    prompt: Annotated[str | None, Form(min_length=1, description="可选文本风格；与参考音频同时提供时进行加权混合")] = None,
    prompt_components: Annotated[str | None, Form(description=parameter_help.PROMPT_COMPONENTS + "；multipart 中传 JSON 数组字符串")] = None,
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
    notes_mode: Annotated[Literal["guide", "strict"], Form(description="音符控制模式")] = "guide",
    drums_mode: Annotated[Literal["guide", "strict"], Form(description="鼓点控制模式")] = "guide",
) -> StreamGenerateRequest:
    return StreamGenerateRequest(
        prompt=prompt,
        prompt_components=_parse_prompt_components_json(prompt_components),
        text_weight=text_weight, audio_weight=audio_weight,
        duration=duration, chunk_frames=chunk_frames,
        temperature=temperature, top_k=top_k,
        cfg_musiccoca=cfg_musiccoca, cfg_notes=cfg_notes, cfg_drums=cfg_drums,
        seed=seed, use_mapper=use_mapper, pool_across_time=pool_across_time,
        notes_mode=notes_mode, drums_mode=drums_mode,
    )


def _pcm_headers() -> dict[str, str]:
    return {
        "X-Audio-Sample-Rate": str(SAMPLE_RATE),
        "X-Audio-Channels": str(CHANNELS),
        "X-Audio-Sample-Format": PCM_SAMPLE_FORMAT,
        "Cache-Control": "no-store",
    }


async def _stream_pcm(session: StreamingSession) -> AsyncIterator[bytes]:
    try:
        while (chunk := await session.next_chunk_async()) is not None:
            yield encode_pcm_chunk(chunk)
    finally:
        await session.close_async()


def create_app(
    config: RuntimeConfig,
    service_factory: ServiceFactory = GenerationService,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        service = service_factory(config)
        await service.load_async()
        app.state.service = service
        try:
            yield
        finally:
            app.state.service = None
            await asyncio.to_thread(service.close)

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

    @app.get(
        "/v1/capabilities",
        response_model=CapabilitiesResponse,
        response_model_by_alias=True,
        tags=["服务"],
    )
    def capabilities() -> CapabilitiesResponse:
        return CapabilitiesResponse(
            version=__version__,
            models=["mrt2_small", "mrt2_base"],
            active_model=config.model.name,
            stream=StreamCapabilitiesResponse.model_validate(
                stream_capabilities()
            ),
        )

    @app.get(
        "/v1/status",
        response_model=StatusResponse,
        response_model_by_alias=True,
        tags=["服务"],
    )
    def status(request: Request) -> StatusResponse:
        current = get_service(request).status()
        return StatusResponse(model=config.model.name, **current)

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
    async def generate(body: GenerateRequest, request: Request) -> Response:
        try:
            encoding = body.encoding_options()
            result = await get_service(request).generate_async(body.to_command())
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
            result = await get_service(request).generate_async(
                options.to_command(reference_audio)
            )
            encoded = await asyncio.to_thread(
                encode_audio,
                result,
                encoding,
            )
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
    async def stream(body: StreamGenerateRequest, request: Request) -> StreamingResponse:
        try:
            session = await get_service(request).open_stream_async(body.to_command())
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
        "/generate/midi",
        response_class=Response,
        responses={
            200: {"description": "根据 MIDI 音符/鼓点控制生成 WAV 或 MP3"},
            400: {"description": "MIDI 或生成参数无效"},
            409: {"description": "模型正被另一个生成会话占用"},
            500: {"description": "推理或编码失败"},
        },
        tags=["生成"],
    )
    async def generate_with_midi(
        request: Request,
        options: Annotated[AudioGenerateRequest, Depends(audio_generation_options)],
        midi: Annotated[UploadFile, File(description="Standard MIDI File（.mid/.midi）")],
        reference_audio: Annotated[
            UploadFile | None, File(description="可选参考音频")
        ] = None,
    ) -> Response:
        try:
            control = decode_midi(
                await midi.read(),
                notes_mode=options.notes_mode,
                drums_mode=options.drums_mode,
            )
            audio_input = (
                decode_audio(await reference_audio.read())
                if reference_audio is not None else None
            )
            encoding = options.encoding_options()
            result = await get_service(request).generate_async(
                options.to_command(audio_input, control)
            )
            encoded = await asyncio.to_thread(encode_audio, result, encoding)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ModelBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="MIDI 条件生成失败") from exc
        return Response(
            content=encoded.data,
            media_type=encoded.media_type,
            headers={"Content-Disposition": f'attachment; filename="output{encoded.extension}"'},
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
            session = await get_service(request).open_stream_async(
                options.to_command(reference_audio)
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

    @app.post(
        "/stream/midi",
        response_class=StreamingResponse,
        responses={
            200: {"description": "根据 MIDI 控制连续返回 float32le PCM"},
            400: {"description": "MIDI 或流式参数无效"},
            409: {"description": "模型正被另一个生成会话占用"},
            500: {"description": "流式生成启动失败"},
        },
        tags=["流式生成"],
    )
    async def stream_with_midi(
        request: Request,
        options: Annotated[StreamGenerateRequest, Depends(audio_stream_options)],
        midi: Annotated[UploadFile, File(description="Standard MIDI File（.mid/.midi）")],
        reference_audio: Annotated[
            UploadFile | None, File(description="可选参考音频")
        ] = None,
    ) -> StreamingResponse:
        try:
            control = decode_midi(
                await midi.read(),
                notes_mode=options.notes_mode,
                drums_mode=options.drums_mode,
            )
            audio_input = (
                decode_audio(await reference_audio.read())
                if reference_audio is not None else None
            )
            command = options.to_command(audio_input, control)
            session = await get_service(request).open_stream_async(command)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ModelBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="MIDI 流式生成启动失败") from exc
        return StreamingResponse(
            _stream_pcm(session), media_type=PCM_MEDIA_TYPE, headers=_pcm_headers()
        )

    return app
