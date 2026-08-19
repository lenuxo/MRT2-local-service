from __future__ import annotations

import asyncio
import json
from typing import Annotated, Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .schemas import GenerateRequest
from .service import GenerationService

router = APIRouter()


class WebSocketGenerateRequest(GenerateRequest):
    request_id: Annotated[
        str | None,
        Field(alias="requestId", min_length=1, max_length=128),
    ] = None


class WebSocketResultMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    type: Literal["result"] = "result"
    request_id: str | None = Field(None, serialization_alias="requestId")
    content_type: Literal["audio/wav"] = Field(
        "audio/wav",
        serialization_alias="contentType",
    )
    byte_length: int = Field(serialization_alias="byteLength")


class WebSocketErrorMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    type: Literal["error"] = "error"
    request_id: str | None = Field(None, serialization_alias="requestId")
    code: Literal["invalid_message", "validation_error", "generation_error"]
    message: str
    details: list[dict] | None = None


async def _send_message(websocket: WebSocket, message: BaseModel) -> None:
    await websocket.send_json(
        message.model_dump(by_alias=True, exclude_none=True, mode="json")
    )


@router.websocket("/ws/generate")
async def generate_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    service: GenerationService = websocket.app.state.service

    while True:
        request_id: str | None = None
        try:
            payload = await websocket.receive_json()
            if isinstance(payload, dict):
                candidate = payload.get("requestId")
                request_id = candidate if isinstance(candidate, str) else None
            body = WebSocketGenerateRequest.model_validate(payload)
            request_id = body.request_id
        except WebSocketDisconnect:
            return
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
            await _send_message(
                websocket,
                WebSocketErrorMessage(
                    request_id=request_id,
                    code="invalid_message",
                    message="消息必须是 UTF-8 JSON 文本",
                ),
            )
            continue
        except ValidationError as exc:
            await _send_message(
                websocket,
                WebSocketErrorMessage(
                    request_id=request_id,
                    code="validation_error",
                    message="生成参数验证失败",
                    details=exc.errors(include_url=False),
                ),
            )
            continue

        try:
            result = await asyncio.to_thread(service.generate, body.to_command())
            wav = result.to_wav_bytes()
        except ValueError as exc:
            await _send_message(
                websocket,
                WebSocketErrorMessage(
                    request_id=request_id,
                    code="validation_error",
                    message=str(exc),
                ),
            )
            continue
        except Exception:
            await _send_message(
                websocket,
                WebSocketErrorMessage(
                    request_id=request_id,
                    code="generation_error",
                    message="音频生成失败",
                ),
            )
            continue

        await _send_message(
            websocket,
            WebSocketResultMessage(
                request_id=request_id,
                byte_length=len(wav),
            ),
        )
        await websocket.send_bytes(wav)
