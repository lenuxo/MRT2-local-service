from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import math
from typing import Annotated, Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, model_validator

from .core import (
    DEFAULT_STREAM_CHUNK_FRAMES,
    DEFAULT_STYLE_WEIGHT,
    SamplingOverrides,
    StreamGenerateCommand,
    StreamUpdateCommand,
)
from .encoding import decode_audio
from .pcm import PCM_SAMPLE_FORMAT, encode_pcm_chunk
from .schemas import DrumEventRequest, NoteEventRequest, SamplingOptions
from .service import GenerationService, ModelBusyError, StreamingSession

router = APIRouter()


class WebSocketStreamRequest(SamplingOptions):
    type: Literal["start"] = "start"
    request_id: Annotated[str | None, Field(alias="requestId", min_length=1, max_length=128)] = None
    input_type: Literal["text", "audio"] = Field("text", alias="inputType")
    prompt: Annotated[str | None, Field(min_length=1)] = None
    text_weight: float = Field(DEFAULT_STYLE_WEIGHT, alias="textWeight", ge=0)
    audio_weight: float = Field(DEFAULT_STYLE_WEIGHT, alias="audioWeight", ge=0)
    chunk_frames: int = Field(DEFAULT_STREAM_CHUNK_FRAMES, alias="chunkFrames", ge=1, le=25)
    realtime: bool = True
    notes_mode: Literal["guide", "strict"] = Field(
        "guide",
        validation_alias=AliasChoices("notesMode", "notes_mode"),
        serialization_alias="notesMode",
    )
    drums_mode: Literal["guide", "strict"] = Field(
        "guide",
        validation_alias=AliasChoices("drumsMode", "drums_mode"),
        serialization_alias="drumsMode",
    )

    def to_command(self, reference_audio=None) -> StreamGenerateCommand:
        return StreamGenerateCommand(
            prompt=self.prompt,
            reference_audio=reference_audio,
            text_weight=self.text_weight,
            audio_weight=self.audio_weight,
            duration=self.duration,
            chunk_frames=self.chunk_frames,
            sampling=self.sampling_overrides(),
            control=self.control_input(),
        )


class WebSocketStreamUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["update"] = "update"
    request_id: str | None = Field(
        None,
        validation_alias=AliasChoices("requestId", "request_id"),
        min_length=1,
        max_length=128,
    )
    revision: int = Field(ge=0)
    effective_frame: int | None = Field(
        None,
        validation_alias=AliasChoices("effectiveFrame", "effective_frame"),
        ge=0,
    )
    prompt: str | None = None
    temperature: float | None = Field(None, gt=0)
    top_k: int | None = Field(
        None, validation_alias=AliasChoices("topK", "top_k"), ge=1
    )
    cfg_musiccoca: float | None = Field(
        None,
        validation_alias=AliasChoices("cfgMusiccoca", "cfg_musiccoca"),
    )
    cfg_notes: float | None = Field(
        None, validation_alias=AliasChoices("cfgNotes", "cfg_notes")
    )
    cfg_drums: float | None = Field(
        None, validation_alias=AliasChoices("cfgDrums", "cfg_drums")
    )
    seed: int | None = None
    use_mapper: bool | None = Field(
        None, validation_alias=AliasChoices("useMapper", "use_mapper")
    )
    pool_across_time: bool | None = Field(
        None,
        validation_alias=AliasChoices("poolAcrossTime", "pool_across_time"),
    )
    notes: list[NoteEventRequest] | None = None
    drums: list[DrumEventRequest] | None = None
    notes_mode: Literal["guide", "strict"] = Field(
        "guide", validation_alias=AliasChoices("notesMode", "notes_mode")
    )
    drums_mode: Literal["guide", "strict"] = Field(
        "guide", validation_alias=AliasChoices("drumsMode", "drums_mode")
    )

    @model_validator(mode="after")
    def validate_update_fields(self) -> WebSocketStreamUpdateRequest:
        update_fields = {
            "prompt", "temperature", "top_k", "cfg_musiccoca", "cfg_notes",
            "cfg_drums", "seed", "use_mapper", "pool_across_time", "notes",
            "drums",
        }
        if not self.model_fields_set.intersection(update_fields):
            raise ValueError("update 消息至少需要包含一个可更新字段")
        if self.prompt is not None and not self.prompt.strip():
            raise ValueError("prompt 必须为非空字符串或 null")
        return self

    def to_command(self) -> StreamUpdateCommand:
        return StreamUpdateCommand(
            revision=self.revision,
            effective_frame=self.effective_frame,
            prompt_present="prompt" in self.model_fields_set,
            prompt=self.prompt,
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
            notes=(
                tuple(item.to_core() for item in self.notes)
                if self.notes is not None else None
            ),
            drums=(
                tuple(item.to_core() for item in self.drums)
                if self.drums is not None else None
            ),
            notes_mode=self.notes_mode,
            drums_mode=self.drums_mode,
        )


async def _receive_controls(
    websocket: WebSocket,
    request_id: str | None,
    stop_event: asyncio.Event,
    updates: asyncio.Queue[object],
    message_event: asyncio.Event,
) -> str:
    try:
        while True:
            message = await websocket.receive_json()
            if (
                isinstance(message, dict)
                and message.get("type") == "stop"
                and message.get("requestId") in (None, request_id)
            ):
                stop_event.set()
                message_event.set()
                return "client_stop"
            await updates.put(message)
            message_event.set()
    except WebSocketDisconnect:
        stop_event.set()
        message_event.set()
        return "client_disconnected"
    except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
        stop_event.set()
        message_event.set()
        return "invalid_message"


async def _apply_queued_updates(
    websocket: WebSocket,
    session: StreamingSession,
    request_id: str | None,
    updates: asyncio.Queue[object],
) -> None:
    while not updates.empty():
        payload = updates.get_nowait()
        try:
            body = WebSocketStreamUpdateRequest.model_validate(payload)
            if body.request_id not in (None, request_id):
                raise ValueError("update 的 requestId 与当前会话不一致")
            result = await session.update_async(body.to_command())
            await websocket.send_json({
                "type": "updateAccepted",
                "requestId": request_id,
                "revision": result.revision,
                "effectiveFrame": result.effective_frame,
                "effectiveTimestampMs": result.effective_frame * 40,
            })
        except ValidationError as exc:
            await websocket.send_json({
                "type": "error",
                "requestId": request_id,
                "code": "update_validation_error",
                "message": "动态更新参数验证失败",
                "details": exc.errors(include_url=False, include_context=False),
            })
        except ValueError as exc:
            await websocket.send_json({
                "type": "error",
                "requestId": request_id,
                "code": "update_validation_error",
                "message": str(exc),
            })


@router.websocket("/ws/stream")
async def stream_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    service: GenerationService = websocket.app.state.service
    session: StreamingSession | None = None
    request_id: str | None = None
    stop_task: asyncio.Task[str] | None = None
    reason = "duration_reached"
    can_send = True
    try:
        payload = await websocket.receive_json()
        body = WebSocketStreamRequest.model_validate(payload)
        request_id = body.request_id
        reference_audio = None
        if body.input_type == "audio":
            reference_audio = decode_audio(await websocket.receive_bytes())
        session = await service.open_stream_async(body.to_command(reference_audio))
        await websocket.send_json({
            "type": "ready",
            "requestId": request_id,
            "sampleRate": 48_000,
            "channels": 2,
            "sampleFormat": PCM_SAMPLE_FORMAT,
            "chunkFrames": body.chunk_frames,
            "frameDurationMs": 40,
            "realtime": body.realtime,
        })

        stop_event = asyncio.Event()
        message_event = asyncio.Event()
        updates: asyncio.Queue[object] = asyncio.Queue()
        stop_task = asyncio.create_task(
            _receive_controls(
                websocket, request_id, stop_event, updates, message_event
            )
        )
        clock_origin: float | None = None
        loop = asyncio.get_running_loop()
        while not stop_event.is_set():
            if body.realtime and clock_origin is not None:
                lead_seconds = body.chunk_frames * 0.04
                target = (
                    clock_origin
                    + session.generated_samples / 48_000
                    - lead_seconds
                )
                while (delay := target - loop.time()) > 0:
                    if not updates.empty():
                        await _apply_queued_updates(
                            websocket, session, request_id, updates
                        )
                        if stop_event.is_set():
                            break
                        continue
                    message_event.clear()
                    if stop_event.is_set():
                        break
                    if not updates.empty():
                        continue
                    try:
                        await asyncio.wait_for(
                            message_event.wait(), timeout=delay
                        )
                    except TimeoutError:
                        break
                    await _apply_queued_updates(
                        websocket, session, request_id, updates
                    )
                    if stop_event.is_set():
                        break
            else:
                await asyncio.sleep(0)
            await _apply_queued_updates(
                websocket, session, request_id, updates
            )
            if stop_event.is_set():
                break
            chunk = await session.next_chunk_async()
            if chunk is None:
                break
            data = encode_pcm_chunk(chunk)
            await websocket.send_json({
                "type": "chunk",
                "requestId": request_id,
                "sequence": chunk.sequence,
                "frames": math.ceil(len(chunk.audio) / 1_920),
                "samplesPerChannel": len(chunk.audio),
                "byteLength": len(data),
                "timestampMs": chunk.timestamp_ms,
            })
            await websocket.send_bytes(data)
            if clock_origin is None:
                clock_origin = loop.time()
        if stop_event.is_set():
            reason = await stop_task
    except WebSocketDisconnect:
        can_send = False
        reason = "client_disconnected"
    except ValidationError as exc:
        await websocket.send_json({
            "type": "error", "requestId": request_id,
            "code": "validation_error", "message": "流式生成参数验证失败",
            "details": exc.errors(include_url=False, include_context=False),
        })
        return
    except ModelBusyError as exc:
        await websocket.send_json({
            "type": "error", "requestId": request_id,
            "code": "model_busy", "message": str(exc),
        })
        return
    except (ValueError, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        await websocket.send_json({
            "type": "error", "requestId": request_id,
            "code": "validation_error", "message": str(exc),
        })
        return
    except Exception:
        if can_send:
            await websocket.send_json({
                "type": "error", "requestId": request_id,
                "code": "generation_error", "message": "流式音频生成失败",
            })
        return
    finally:
        if stop_task is not None:
            stop_task.cancel()
            with suppress(asyncio.CancelledError):
                await stop_task
        if session is not None:
            await session.close_async()

    if can_send:
        await websocket.send_json({
            "type": "completed",
            "requestId": request_id,
            "reason": reason,
            "generatedSamples": session.generated_samples if session else 0,
        })
