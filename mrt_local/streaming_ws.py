from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import math
from typing import Annotated, Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import AliasChoices, Field, ValidationError

from .core import DEFAULT_STREAM_CHUNK_FRAMES, DEFAULT_STYLE_WEIGHT, StreamGenerateCommand
from .encoding import decode_audio
from .pcm import PCM_SAMPLE_FORMAT, encode_pcm_chunk
from .schemas import SamplingOptions
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


async def _receive_stop(
    websocket: WebSocket,
    request_id: str | None,
    stop_event: asyncio.Event,
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
                return "client_stop"
    except WebSocketDisconnect:
        stop_event.set()
        return "client_disconnected"
    except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
        stop_event.set()
        return "invalid_message"


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
        session = await asyncio.to_thread(
            service.open_stream, body.to_command(reference_audio)
        )
        await websocket.send_json({
            "type": "ready",
            "requestId": request_id,
            "sampleRate": 48_000,
            "channels": 2,
            "sampleFormat": PCM_SAMPLE_FORMAT,
            "chunkFrames": body.chunk_frames,
            "frameDurationMs": 40,
        })

        stop_event = asyncio.Event()
        stop_task = asyncio.create_task(
            _receive_stop(websocket, request_id, stop_event)
        )
        while not stop_event.is_set():
            chunk = await asyncio.to_thread(session.next_chunk)
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
            await asyncio.to_thread(session.close)

    if can_send:
        await websocket.send_json({
            "type": "completed",
            "requestId": request_id,
            "reason": reason,
            "generatedSamples": session.generated_samples if session else 0,
        })
