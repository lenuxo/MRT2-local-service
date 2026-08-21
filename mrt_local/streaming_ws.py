from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
import hashlib
import json
import math
import time
from typing import Annotated, Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, model_validator

from .core import (
    DEFAULT_STREAM_CHUNK_FRAMES,
    DEFAULT_STYLE_WEIGHT,
    AudioInput,
    LiveMidiCommand,
    LiveMidiEvent,
    MAX_LIVE_MIDI_BATCH_EVENTS,
    SamplingOverrides,
    StreamGenerateCommand,
    StreamExtendCommand,
    StreamUpdateCommand,
)
from .capabilities import (
    MAX_CONTROL_MESSAGE_BYTES,
    MAX_REFERENCE_AUDIO_BYTES,
    REFERENCE_AUDIO_TIMEOUT_SECONDS,
    stream_capabilities,
)
from .encoding import decode_audio
from .pcm import PCM_SAMPLE_FORMAT, encode_pcm_chunk
from .schemas import (
    DrumEventRequest,
    NoteEventRequest,
    PromptComponentRequest,
    SamplingOptions,
)
from .service import GenerationService, ModelBusyError, StreamingSession

router = APIRouter()


class WebSocketStreamRequest(SamplingOptions):
    type: Literal["start"] = "start"
    request_id: Annotated[str | None, Field(alias="requestId", min_length=1, max_length=128)] = None
    input_type: Literal["text", "audio"] = Field("text", alias="inputType")
    prompt: Annotated[str | None, Field(min_length=1)] = None
    prompt_components: list[PromptComponentRequest] = Field(
        default_factory=list,
        validation_alias=AliasChoices("promptComponents", "prompt_components"),
        max_length=8,
    )
    text_weight: float = Field(DEFAULT_STYLE_WEIGHT, alias="textWeight", ge=0)
    audio_weight: float = Field(DEFAULT_STYLE_WEIGHT, alias="audioWeight", ge=0)
    chunk_frames: int = Field(DEFAULT_STREAM_CHUNK_FRAMES, alias="chunkFrames", ge=1, le=25)
    realtime: bool = True
    midi_mode: Literal["plan", "live"] = Field(
        "plan",
        validation_alias=AliasChoices("midiMode", "midi_mode"),
        serialization_alias="midiMode",
    )
    live_notes_mode: Literal["guide", "strict"] = Field(
        "guide",
        validation_alias=AliasChoices("liveNotesMode", "live_notes_mode"),
        serialization_alias="liveNotesMode",
    )
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
    def validate_prompt_modes(self) -> WebSocketStreamRequest:
        if self.prompt is not None and self.prompt_components:
            raise ValueError("prompt 与 promptComponents 不能同时提供")
        if self.midi_mode == "live" and (self.notes or self.drums):
            raise ValueError(
                "midiMode=live 时不能同时提供计划式 notes 或 drums"
            )
        return self

    def to_command(self, reference_audio=None) -> StreamGenerateCommand:
        return StreamGenerateCommand(
            prompt=self.prompt,
            prompt_components=tuple(
                item.to_core() for item in self.prompt_components
            ),
            reference_audio=reference_audio,
            text_weight=self.text_weight,
            audio_weight=self.audio_weight,
            duration=self.duration,
            chunk_frames=self.chunk_frames,
            sampling=self.sampling_overrides(),
            control=self.control_input(),
            midi_mode=self.midi_mode,
            live_notes_mode=self.live_notes_mode,
        )


class WebSocketLiveMidiEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["noteOn", "noteOff", "controlChange", "panic"]
    channel: int | None = Field(None, ge=0, le=15)
    pitch: int | None = Field(None, ge=0, le=127)
    velocity: int | None = Field(None, ge=0, le=127)
    controller: int | None = Field(None, ge=0, le=127)
    value: int | None = Field(None, ge=0, le=127)

    @model_validator(mode="after")
    def validate_event(self) -> WebSocketLiveMidiEventRequest:
        self.to_core().validate()
        return self

    def to_core(self) -> LiveMidiEvent:
        kinds = {
            "noteOn": "note_on",
            "noteOff": "note_off",
            "controlChange": "control_change",
            "panic": "panic",
        }
        return LiveMidiEvent(
            kind=kinds[self.kind],
            channel=self.channel,
            pitch=self.pitch,
            velocity=self.velocity,
            controller=self.controller,
            value=self.value,
        )


class WebSocketLiveMidiRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["midi"] = "midi"
    request_id: str | None = Field(
        None,
        validation_alias=AliasChoices("requestId", "request_id"),
        min_length=1,
        max_length=128,
    )
    event_sequence: int = Field(
        validation_alias=AliasChoices("eventSequence", "event_sequence"),
        ge=0,
    )
    events: list[WebSocketLiveMidiEventRequest] = Field(
        min_length=1, max_length=MAX_LIVE_MIDI_BATCH_EVENTS
    )

    def to_command(self) -> LiveMidiCommand:
        return LiveMidiCommand(
            self.event_sequence, tuple(event.to_core() for event in self.events)
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
    prompt_components: list[PromptComponentRequest] | None = Field(
        None,
        validation_alias=AliasChoices("promptComponents", "prompt_components"),
        max_length=8,
    )
    reference_audio: Literal["replace", "clear"] | None = Field(
        None,
        validation_alias=AliasChoices("referenceAudio", "reference_audio"),
    )
    text_weight: float | None = Field(
        None,
        validation_alias=AliasChoices("textWeight", "text_weight"),
        ge=0,
    )
    audio_weight: float | None = Field(
        None,
        validation_alias=AliasChoices("audioWeight", "audio_weight"),
        ge=0,
    )
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
    drumless: bool | None = None
    notes_mode: Literal["guide", "strict"] = Field(
        "guide", validation_alias=AliasChoices("notesMode", "notes_mode")
    )
    drums_mode: Literal["guide", "strict"] = Field(
        "guide", validation_alias=AliasChoices("drumsMode", "drums_mode")
    )

    @model_validator(mode="after")
    def validate_update_fields(self) -> WebSocketStreamUpdateRequest:
        update_fields = {
            "prompt", "prompt_components", "temperature", "top_k", "cfg_musiccoca", "cfg_notes",
            "cfg_drums", "seed", "use_mapper", "pool_across_time", "notes",
            "drums", "drumless", "reference_audio", "text_weight", "audio_weight",
        }
        if not self.model_fields_set.intersection(update_fields):
            raise ValueError("update 消息至少需要包含一个可更新字段")
        if self.prompt is not None and not self.prompt.strip():
            raise ValueError("prompt 必须为非空字符串或 null")
        if "prompt" in self.model_fields_set and "prompt_components" in self.model_fields_set:
            raise ValueError("prompt 与 promptComponents 不能在同一次更新中提供")
        if "drumless" in self.model_fields_set and "drums" in self.model_fields_set:
            raise ValueError("drumless 与 drums 不能在同一次更新中提供")
        return self

    def to_command(
        self, reference_audio: AudioInput | None = None
    ) -> StreamUpdateCommand:
        if self.reference_audio == "replace" and reference_audio is None:
            raise ValueError("referenceAudio=replace 需要紧随二进制音频消息")
        return StreamUpdateCommand(
            revision=self.revision,
            effective_frame=self.effective_frame,
            prompt_present="prompt" in self.model_fields_set,
            prompt=self.prompt,
            prompt_components_present="prompt_components" in self.model_fields_set,
            prompt_components=tuple(
                item.to_core() for item in (self.prompt_components or [])
            ),
            reference_audio_present=self.reference_audio is not None,
            reference_audio=reference_audio,
            text_weight=self.text_weight,
            audio_weight=self.audio_weight,
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
            drumless=self.drumless,
            notes_mode=self.notes_mode,
            drums_mode=self.drums_mode,
        )


class WebSocketStreamExtendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["extend"] = "extend"
    request_id: str | None = Field(
        None,
        validation_alias=AliasChoices("requestId", "request_id"),
        min_length=1,
        max_length=128,
    )
    revision: int = Field(ge=0)
    additional_duration: float = Field(
        validation_alias=AliasChoices(
            "additionalDuration", "additional_duration"
        ),
        gt=0,
        le=300,
    )

    def to_command(self) -> StreamExtendCommand:
        return StreamExtendCommand(
            revision=self.revision,
            additional_duration=self.additional_duration,
        )


class WebSocketStreamConfigureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["configure"] = "configure"
    request_id: str | None = Field(
        None,
        validation_alias=AliasChoices("requestId", "request_id"),
        min_length=1,
        max_length=128,
    )
    revision: int = Field(ge=0)
    chunk_frames: int | None = Field(
        None,
        validation_alias=AliasChoices("chunkFrames", "chunk_frames"),
        ge=1,
        le=25,
    )
    realtime: bool | None = None

    @model_validator(mode="after")
    def validate_configuration(self) -> WebSocketStreamConfigureRequest:
        if self.chunk_frames is None and self.realtime is None:
            raise ValueError("configure 必须包含 chunkFrames 或 realtime")
        return self


@dataclass(frozen=True, slots=True)
class _QueuedControl:
    payload: object
    reference_audio_data: bytes | None = None


@dataclass(frozen=True, slots=True)
class _QueuedProtocolError:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class _QueuedResponse:
    payload: dict[str, object]


class _ControlProtocolError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(slots=True)
class _ControlState:
    last_revision: int = -1
    last_midi_sequence: int = -1
    control_sequence: int = 0
    accepted: dict[int, tuple[str, dict[str, object]]] = field(
        default_factory=dict
    )
    accepted_midi: dict[int, tuple[str, dict[str, object]]] = field(
        default_factory=dict
    )

    def next_sequence(self) -> int:
        value = self.control_sequence
        self.control_sequence += 1
        return value

    def check_revision(
        self, revision: int, fingerprint: str
    ) -> dict[str, object] | None:
        previous = self.accepted.get(revision)
        if previous is not None:
            if previous[0] != fingerprint:
                raise _ControlProtocolError(
                    "revision_conflict",
                    "相同 revision 已用于不同的控制消息",
                )
            return previous[1]
        if revision <= self.last_revision:
            raise _ControlProtocolError(
                "stale_revision",
                f"revision 必须大于已接受的 {self.last_revision}",
            )
        return None

    def accept(
        self, revision: int, fingerprint: str, response: dict[str, object]
    ) -> None:
        self.last_revision = revision
        self.accepted[revision] = (fingerprint, response)
        if len(self.accepted) > 256:
            self.accepted.pop(next(iter(self.accepted)))

    def check_midi_sequence(
        self, sequence: int, fingerprint: str
    ) -> dict[str, object] | None:
        previous = self.accepted_midi.get(sequence)
        if previous is not None:
            if previous[0] != fingerprint:
                raise _ControlProtocolError(
                    "midi_sequence_conflict",
                    "相同 eventSequence 已用于不同的 MIDI 消息",
                )
            return previous[1]
        if sequence <= self.last_midi_sequence:
            raise _ControlProtocolError(
                "stale_midi_sequence",
                f"eventSequence 必须大于已接受的 {self.last_midi_sequence}",
            )
        return None

    def accept_midi(
        self, sequence: int, fingerprint: str, response: dict[str, object]
    ) -> None:
        self.last_midi_sequence = sequence
        self.accepted_midi[sequence] = (fingerprint, response)
        if len(self.accepted_midi) > 256:
            self.accepted_midi.pop(next(iter(self.accepted_midi)))


def _control_fingerprint(payload: dict, audio_data: bytes | None) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8"))
    if audio_data is not None:
        digest.update(audio_data)
    return digest.hexdigest()


async def _receive_json_message(websocket: WebSocket) -> object:
    message = await websocket.receive()
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", 1000))
    text = message.get("text")
    if text is None:
        raise _ControlProtocolError(
            "unexpected_binary", "此处需要 JSON 文本消息"
        )
    if len(text.encode("utf-8")) > MAX_CONTROL_MESSAGE_BYTES:
        raise _ControlProtocolError(
            "message_too_large",
            f"控制消息不能超过 {MAX_CONTROL_MESSAGE_BYTES} 字节",
        )
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise _ControlProtocolError("invalid_json", "控制消息不是有效 JSON") from exc


async def _receive_reference_audio(websocket: WebSocket) -> bytes:
    try:
        message = await asyncio.wait_for(
            websocket.receive(), timeout=REFERENCE_AUDIO_TIMEOUT_SECONDS
        )
    except TimeoutError as exc:
        raise _ControlProtocolError(
            "reference_audio_timeout",
            f"等待参考音频超过 {REFERENCE_AUDIO_TIMEOUT_SECONDS:g} 秒",
        ) from exc
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", 1000))
    data = message.get("bytes")
    if data is None:
        raise _ControlProtocolError(
            "reference_audio_binary_required",
            "referenceAudio=replace 后必须发送二进制音频消息",
        )
    if len(data) > MAX_REFERENCE_AUDIO_BYTES:
        raise _ControlProtocolError(
            "reference_audio_too_large",
            f"参考音频消息不能超过 {MAX_REFERENCE_AUDIO_BYTES} 字节",
        )
    return data


async def _receive_controls(
    websocket: WebSocket,
    request_id: str | None,
    session: StreamingSession,
    session_id: str,
    midi_mode: str,
    control_state: _ControlState,
    stop_event: asyncio.Event,
    updates: asyncio.Queue[object],
    message_event: asyncio.Event,
) -> str:
    try:
        while True:
            try:
                message = await _receive_json_message(websocket)
            except _ControlProtocolError as exc:
                await updates.put(_QueuedProtocolError(exc.code, str(exc)))
                message_event.set()
                continue
            if (
                isinstance(message, dict)
                and message.get("type") == "stop"
                and message.get("requestId") in (None, request_id)
            ):
                stop_event.set()
                message_event.set()
                return "client_stop"
            if isinstance(message, dict) and message.get("type") == "midi":
                started_at = time.perf_counter()
                try:
                    if midi_mode != "live":
                        raise ValueError("当前会话未启用 midiMode=live")
                    body = WebSocketLiveMidiRequest.model_validate(message)
                    if body.request_id not in (None, request_id):
                        raise ValueError("midi 的 requestId 与当前会话不一致")
                    fingerprint = _control_fingerprint(message, None)
                    replay = control_state.check_midi_sequence(
                        body.event_sequence, fingerprint
                    )
                    if replay is not None:
                        response = {**replay, "duplicate": True}
                    else:
                        result = await session.queue_live_midi_async(
                            body.to_command()
                        )
                        response = {
                            "type": "midiQueued",
                            "requestId": request_id,
                            "sessionId": session_id,
                            "eventSequence": result.event_sequence,
                            "earliestEffectiveFrame": (
                                result.earliest_effective_frame
                            ),
                            "earliestEffectiveTimestampMs": (
                                result.earliest_effective_frame * 40
                            ),
                            "acceptedEvents": result.accepted_events,
                            "processingTimeMs": round(
                                (time.perf_counter() - started_at) * 1000, 3
                            ),
                        }
                        control_state.accept_midi(
                            body.event_sequence, fingerprint, response
                        )
                    await updates.put(_QueuedResponse(response))
                except ValidationError as exc:
                    await updates.put(_QueuedResponse({
                        "type": "error",
                        "requestId": request_id,
                        "sessionId": session_id,
                        "code": "control_validation_error",
                        "message": "实时 MIDI 参数验证失败",
                        "details": exc.errors(
                            include_url=False, include_context=False
                        ),
                    }))
                except _ControlProtocolError as exc:
                    await updates.put(_QueuedProtocolError(exc.code, str(exc)))
                except ValueError as exc:
                    await updates.put(_QueuedProtocolError(
                        "control_validation_error", str(exc)
                    ))
                message_event.set()
                continue
            audio_action = None
            if isinstance(message, dict):
                audio_action = message.get(
                    "referenceAudio", message.get("reference_audio")
                )
            audio_data = None
            if (
                isinstance(message, dict)
                and message.get("type") == "update"
                and audio_action == "replace"
            ):
                try:
                    audio_data = await _receive_reference_audio(websocket)
                except _ControlProtocolError as exc:
                    await updates.put(_QueuedProtocolError(exc.code, str(exc)))
                    message_event.set()
                    continue
            await updates.put(_QueuedControl(message, audio_data))
            message_event.set()
    except WebSocketDisconnect:
        stop_event.set()
        message_event.set()
        return "client_disconnected"
    except (KeyError, UnicodeDecodeError):
        stop_event.set()
        message_event.set()
        return "invalid_message"


async def _apply_queued_commands(
    websocket: WebSocket,
    session: StreamingSession,
    request_id: str | None,
    updates: asyncio.Queue[object],
    transport_config: dict[str, int | bool],
    session_id: str,
    control_state: _ControlState,
) -> None:
    while not updates.empty():
        queued = updates.get_nowait()
        if isinstance(queued, _QueuedResponse):
            await websocket.send_json({
                **queued.payload,
                "controlSequence": control_state.next_sequence(),
            })
            continue
        if isinstance(queued, _QueuedProtocolError):
            await websocket.send_json({
                "type": "error",
                "requestId": request_id,
                "sessionId": session_id,
                "controlSequence": control_state.next_sequence(),
                "code": queued.code,
                "message": queued.message,
            })
            continue
        if isinstance(queued, _QueuedControl):
            payload = queued.payload
            reference_audio_data = queued.reference_audio_data
        else:
            payload = queued
            reference_audio_data = None
        try:
            if not isinstance(payload, dict):
                raise ValueError("控制消息必须是 JSON object")
            started_at = time.perf_counter()
            revision = payload.get("revision")
            if payload.get("type") in {"update", "extend", "configure"}:
                if not isinstance(revision, int) or isinstance(revision, bool):
                    raise ValueError("控制消息必须包含非负整数 revision")
                if revision < 0:
                    raise ValueError("revision 必须大于等于 0")
                fingerprint = _control_fingerprint(payload, reference_audio_data)
                replay = control_state.check_revision(revision, fingerprint)
                if replay is not None:
                    await websocket.send_json({
                        **replay,
                        "controlSequence": control_state.next_sequence(),
                        "duplicate": True,
                    })
                    continue
            if payload.get("type") == "extend":
                body = WebSocketStreamExtendRequest.model_validate(payload)
                if body.request_id not in (None, request_id):
                    raise ValueError("extend 的 requestId 与当前会话不一致")
                result = await session.extend_async(body.to_command())
                response = {
                    "type": "extended",
                    "requestId": request_id,
                    "sessionId": session_id,
                    "revision": result.revision,
                    "previousDurationMs": result.previous_duration_ms,
                    "durationMs": result.duration_ms,
                    "processingTimeMs": round(
                        (time.perf_counter() - started_at) * 1000, 3
                    ),
                }
                control_state.accept(body.revision, fingerprint, response)
                await websocket.send_json({
                    **response,
                    "controlSequence": control_state.next_sequence(),
                })
                continue
            if payload.get("type") == "configure":
                body = WebSocketStreamConfigureRequest.model_validate(payload)
                if body.request_id not in (None, request_id):
                    raise ValueError("configure 的 requestId 与当前会话不一致")
                if body.chunk_frames is not None:
                    await session.configure_chunk_frames_async(body.chunk_frames)
                    transport_config["chunkFrames"] = body.chunk_frames
                if body.realtime is not None:
                    transport_config["realtime"] = body.realtime
                response = {
                    "type": "configured",
                    "requestId": request_id,
                    "sessionId": session_id,
                    "revision": body.revision,
                    "effectiveFrame": session.generated_samples // 1_920,
                    "chunkFrames": transport_config["chunkFrames"],
                    "realtime": transport_config["realtime"],
                    "processingTimeMs": round(
                        (time.perf_counter() - started_at) * 1000, 3
                    ),
                }
                control_state.accept(body.revision, fingerprint, response)
                await websocket.send_json({
                    **response,
                    "controlSequence": control_state.next_sequence(),
                })
                continue
            body = WebSocketStreamUpdateRequest.model_validate(payload)
            if body.request_id not in (None, request_id):
                raise ValueError("update 的 requestId 与当前会话不一致")
            audio_decode_started = time.perf_counter()
            reference_audio = None
            audio_decode_ms = None
            if reference_audio_data is not None:
                reference_audio = await asyncio.to_thread(
                    decode_audio, reference_audio_data
                )
                audio_decode_ms = round(
                    (time.perf_counter() - audio_decode_started) * 1000, 3
                )
            result = await session.update_async(body.to_command(reference_audio))
            response = {
                "type": "updateAccepted",
                "requestId": request_id,
                "sessionId": session_id,
                "revision": result.revision,
                "effectiveFrame": result.effective_frame,
                "effectiveTimestampMs": result.effective_frame * 40,
                "processingTimeMs": round(
                    (time.perf_counter() - started_at) * 1000, 3
                ),
            }
            if audio_decode_ms is not None:
                response["audioDecodeTimeMs"] = audio_decode_ms
            control_state.accept(body.revision, fingerprint, response)
            await websocket.send_json({
                **response,
                "controlSequence": control_state.next_sequence(),
            })
        except ValidationError as exc:
            await websocket.send_json({
                "type": "error",
                "requestId": request_id,
                "sessionId": session_id,
                "controlSequence": control_state.next_sequence(),
                "code": "control_validation_error",
                "message": "流式控制参数验证失败",
                "details": exc.errors(include_url=False, include_context=False),
            })
        except _ControlProtocolError as exc:
            await websocket.send_json({
                "type": "error",
                "requestId": request_id,
                "sessionId": session_id,
                "controlSequence": control_state.next_sequence(),
                "code": exc.code,
                "message": str(exc),
            })
        except ValueError as exc:
            await websocket.send_json({
                "type": "error",
                "requestId": request_id,
                "sessionId": session_id,
                "controlSequence": control_state.next_sequence(),
                "code": "control_validation_error",
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
        stream_started_at = time.perf_counter()
        payload = await _receive_json_message(websocket)
        body = WebSocketStreamRequest.model_validate(payload)
        request_id = body.request_id
        reference_audio = None
        if body.input_type == "audio":
            reference_audio = decode_audio(
                await _receive_reference_audio(websocket)
            )
        session = await service.open_stream_async(body.to_command(reference_audio))
        session_id = session.session_id
        await websocket.send_json({
            "type": "ready",
            "requestId": request_id,
            "sessionId": session_id,
            "sampleRate": 48_000,
            "channels": 2,
            "sampleFormat": PCM_SAMPLE_FORMAT,
            "chunkFrames": body.chunk_frames,
            "frameDurationMs": 40,
            "realtime": body.realtime,
            "midiMode": body.midi_mode,
            "liveNotesMode": body.live_notes_mode,
            "dynamicCapabilities": stream_capabilities(),
        })

        stop_event = asyncio.Event()
        message_event = asyncio.Event()
        updates: asyncio.Queue[object] = asyncio.Queue()
        transport_config: dict[str, int | bool] = {
            "chunkFrames": body.chunk_frames,
            "realtime": body.realtime,
        }
        control_state = _ControlState()
        stop_task = asyncio.create_task(
            _receive_controls(
                websocket, request_id, session, session_id, body.midi_mode,
                control_state,
                stop_event, updates, message_event
            )
        )
        clock_origin: float | None = None
        loop = asyncio.get_running_loop()
        while not stop_event.is_set():
            if transport_config["realtime"] and clock_origin is not None:
                lead_seconds = int(transport_config["chunkFrames"]) * 0.04
                target = (
                    clock_origin
                    + session.generated_samples / 48_000
                    - lead_seconds
                )
                while (delay := target - loop.time()) > 0:
                    if not updates.empty():
                        await _apply_queued_commands(
                            websocket, session, request_id, updates,
                            transport_config, session_id, control_state,
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
                    await _apply_queued_commands(
                        websocket, session, request_id, updates,
                        transport_config, session_id, control_state,
                    )
                    if stop_event.is_set():
                        break
            else:
                await asyncio.sleep(0)
            await _apply_queued_commands(
                websocket, session, request_id, updates, transport_config,
                session_id, control_state,
            )
            if stop_event.is_set():
                break
            generation_started_at = time.perf_counter()
            chunk = await session.next_chunk_async()
            generation_time_ms = (
                time.perf_counter() - generation_started_at
            ) * 1000
            if chunk is None:
                break
            data = encode_pcm_chunk(chunk)
            await websocket.send_json({
                "type": "chunk",
                "requestId": request_id,
                "sessionId": session_id,
                "sequence": chunk.sequence,
                "frames": math.ceil(len(chunk.audio) / 1_920),
                "samplesPerChannel": len(chunk.audio),
                "byteLength": len(data),
                "timestampMs": chunk.timestamp_ms,
            })
            await websocket.send_bytes(data)
            audio_duration_ms = len(chunk.audio) * 1000 / 48_000
            generated_audio_ms = session.generated_samples * 1000 / 48_000
            elapsed_ms = (time.perf_counter() - stream_started_at) * 1000
            await websocket.send_json({
                "type": "metrics",
                "requestId": request_id,
                "sessionId": session_id,
                "sequence": chunk.sequence,
                "generationTimeMs": round(generation_time_ms, 3),
                "generatedAudioMs": round(generated_audio_ms, 3),
                "realtimeFactor": round(
                    generation_time_ms / audio_duration_ms, 4
                ),
                "bufferLeadMs": round(generated_audio_ms - elapsed_ms, 3),
                "firstChunkLatencyMs": (
                    round(elapsed_ms, 3) if chunk.sequence == 0 else None
                ),
            })
            if clock_origin is None:
                clock_origin = loop.time()
        if stop_event.is_set():
            reason = await stop_task
            if reason == "client_disconnected":
                can_send = False
    except WebSocketDisconnect:
        can_send = False
        reason = "client_disconnected"
    except _ControlProtocolError as exc:
        await websocket.send_json({
            "type": "error", "requestId": request_id,
            "code": exc.code, "message": str(exc),
        })
        return
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
            "sessionId": session.session_id if session else None,
            "reason": reason,
            "generatedSamples": session.generated_samples if session else 0,
        })
