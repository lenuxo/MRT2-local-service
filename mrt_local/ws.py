from __future__ import annotations

import asyncio
import json
from typing import Annotated, Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, model_validator

from .core import AudioInput, DEFAULT_STYLE_WEIGHT, GenerateCommand
from .schemas import GenerationOptions
from .service import GenerationService, ModelBusyError
from .encoding import AudioEncodingError, AudioFormat, decode_audio, encode_audio

router = APIRouter()


class WebSocketGenerateRequest(GenerationOptions):
    request_id: Annotated[
        str | None,
        Field(alias="requestId", min_length=1, max_length=128),
    ] = None
    input_type: Literal["text", "audio"] = Field("text", alias="inputType")
    prompt: str | None = None
    text_weight: float = Field(DEFAULT_STYLE_WEIGHT, alias="textWeight", ge=0)
    audio_weight: float = Field(DEFAULT_STYLE_WEIGHT, alias="audioWeight", ge=0)
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

    @model_validator(mode="after")
    def validate_style_input(self) -> WebSocketGenerateRequest:
        if (
            self.input_type == "text"
            and not (self.prompt or "").strip()
            and not self.control_input()
        ):
            raise ValueError("必须提供非空 prompt、notes 或 drums")
        return self

    def to_command(self, reference_audio: AudioInput | None = None) -> GenerateCommand:
        return GenerateCommand(
            prompt=self.prompt,
            reference_audio=reference_audio,
            text_weight=self.text_weight,
            audio_weight=self.audio_weight,
            duration=self.duration,
            sampling=self.sampling_overrides(),
            control=self.control_input(),
        )


class WebSocketResultMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    type: Literal["result"] = "result"
    request_id: str | None = Field(None, serialization_alias="requestId")
    format: AudioFormat
    content_type: Literal["audio/wav", "audio/mpeg"] = Field(
        serialization_alias="contentType",
    )
    byte_length: int = Field(serialization_alias="byteLength")


class WebSocketErrorMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    type: Literal["error"] = "error"
    request_id: str | None = Field(None, serialization_alias="requestId")
    code: Literal[
        "invalid_message",
        "validation_error",
        "encoding_error",
        "generation_error",
        "model_busy",
    ]
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
                    details=exc.errors(include_url=False, include_context=False),
                ),
            )
            continue

        try:
            encoding = body.encoding_options()
            reference_audio = None
            if body.input_type == "audio":
                try:
                    reference_audio = decode_audio(await websocket.receive_bytes())
                except WebSocketDisconnect:
                    return
                except (KeyError, UnicodeDecodeError, ValueError) as exc:
                    await _send_message(
                        websocket,
                        WebSocketErrorMessage(
                            request_id=request_id,
                            code="invalid_message",
                            message=f"音频输入需要紧随 JSON 的二进制音频消息：{exc}",
                        ),
                    )
                    continue
            result = await asyncio.to_thread(
                service.generate,
                body.to_command(reference_audio),
            )
            encoded = await asyncio.to_thread(
                encode_audio,
                result,
                encoding,
            )
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
        except ModelBusyError as exc:
            await _send_message(
                websocket,
                WebSocketErrorMessage(
                    request_id=request_id,
                    code="model_busy",
                    message=str(exc),
                ),
            )
            continue
        except AudioEncodingError as exc:
            await _send_message(
                websocket,
                WebSocketErrorMessage(
                    request_id=request_id,
                    code="encoding_error",
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
                format=encoded.format,
                content_type=encoded.media_type,
                byte_length=len(encoded.data),
            ),
        )
        await websocket.send_bytes(encoded.data)
