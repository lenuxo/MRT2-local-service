from __future__ import annotations

import asyncio
from pathlib import Path
import time

import numpy as np
from fastapi.testclient import TestClient

from mrt_local.api import create_app
from mrt_local.config import RuntimeConfig
from mrt_local.core import (
    AudioChunk,
    GenerateCommand,
    GenerateResult,
    ModelConfig,
    StreamUpdateResult,
)
from mrt_local.encoding import encode_audio


class FakeService:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.is_loaded = False
        self.commands: list[GenerateCommand] = []
        self.stream_session = None

    def load(self) -> None:
        self.is_loaded = True

    async def load_async(self) -> None:
        self.load()

    def close(self) -> None:
        self.is_loaded = False

    def generate(self, command: GenerateCommand) -> GenerateResult:
        self.commands.append(command)
        if command.prompt == "fail":
            raise RuntimeError("backend details must not leak")
        return GenerateResult(
            48_000,
            2,
            np.zeros((round(command.duration * 48_000), 2), np.float32),
        )

    async def generate_async(self, command: GenerateCommand) -> GenerateResult:
        return self.generate(command)

    def open_stream(self, command):
        self.commands.append(command)
        self.stream_session = FakeStreamingSession(
            round(command.duration * 48_000), slow=command.prompt == "slow"
        )
        return self.stream_session

    async def open_stream_async(self, command):
        return self.open_stream(command)


class FakeStreamingSession:
    def __init__(self, samples: int, slow: bool = False) -> None:
        self.remaining = samples
        self.generated_samples = 0
        self.slow = slow
        self.updates = []

    def next_chunk(self):
        if self.slow:
            time.sleep(0.03)
        if not self.remaining:
            return None
        count = min(self.remaining, 1_920)
        chunk = AudioChunk(0, self.generated_samples, 48_000, 2, np.zeros((count, 2), np.float32))
        self.remaining -= count
        self.generated_samples += count
        return chunk

    async def next_chunk_async(self):
        if self.slow:
            await asyncio.sleep(0.03)
        return self.next_chunk()

    async def update_async(self, command):
        self.updates.append(command)
        return StreamUpdateResult(
            command.revision, self.generated_samples // 1_920
        )

    def close(self):
        pass

    async def close_async(self):
        self.close()


def create_test_app(tmp_path: Path):
    config = RuntimeConfig(model=ModelConfig(name="mrt2_small", root=tmp_path))
    return create_app(config, service_factory=FakeService)


def test_websocket_returns_metadata_then_binary_wav(tmp_path: Path) -> None:
    app = create_test_app(tmp_path)

    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws/generate", headers={"Origin": "https://example.com"}
        ) as websocket:
            websocket.send_json(
                {
                    "requestId": "job-1",
                    "prompt": "ambient",
                    "duration": 0.04,
                    "temperature": 0.8,
                }
            )
            metadata = websocket.receive_json()
            wav = websocket.receive_bytes()
            commands = list(app.state.service.commands)

    assert metadata == {
        "type": "result",
        "requestId": "job-1",
        "format": "wav",
        "contentType": "audio/wav",
        "byteLength": len(wav),
    }
    assert wav[:4] == b"RIFF"
    assert commands[0].prompt == "ambient"
    assert commands[0].sampling.temperature == 0.8


def test_websocket_reports_errors_and_keeps_connection_open(tmp_path: Path) -> None:
    app = create_test_app(tmp_path)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/generate") as websocket:
            websocket.send_text("not-json")
            invalid_message = websocket.receive_json()

            websocket.send_json({"requestId": "bad-1", "prompt": ""})
            validation_error = websocket.receive_json()

            websocket.send_json({"requestId": "bad-2", "prompt": "fail"})
            generation_error = websocket.receive_json()

            websocket.send_json({"requestId": "ok-1", "prompt": "recovered", "duration": 0.01})
            result = websocket.receive_json()
            wav = websocket.receive_bytes()

    assert invalid_message == {
        "type": "error",
        "code": "invalid_message",
        "message": "消息必须是 UTF-8 JSON 文本",
    }
    assert validation_error["type"] == "error"
    assert validation_error["requestId"] == "bad-1"
    assert validation_error["code"] == "validation_error"
    assert generation_error == {
        "type": "error",
        "requestId": "bad-2",
        "code": "generation_error",
        "message": "音频生成失败",
    }
    assert result["requestId"] == "ok-1"
    assert wav[:4] == b"RIFF"


def test_websocket_accepts_binary_reference_audio(tmp_path: Path) -> None:
    app = create_test_app(tmp_path)
    reference_wav = encode_audio(
        GenerateResult(48_000, 2, np.zeros((480, 2), np.float32))
    ).data

    with TestClient(app) as client:
        with client.websocket_connect("/ws/generate") as websocket:
            websocket.send_json(
                {
                    "requestId": "audio-1",
                    "inputType": "audio",
                    "prompt": "ambient",
                    "textWeight": 1,
                    "audioWeight": 3,
                    "duration": 0.01,
                }
            )
            websocket.send_bytes(reference_wav)
            metadata = websocket.receive_json()
            output = websocket.receive_bytes()
            command = app.state.service.commands[0]

    assert metadata["requestId"] == "audio-1"
    assert output[:4] == b"RIFF"
    assert command.prompt == "ambient"
    assert command.text_weight == 1
    assert command.audio_weight == 3
    assert command.reference_audio is not None
    assert command.reference_audio.samples.shape == (480, 2)


def test_websocket_accepts_control_without_prompt(tmp_path: Path) -> None:
    app = create_test_app(tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/generate") as websocket:
            websocket.send_json({
                "requestId": "notes-1",
                "duration": 0.04,
                "notes": [{"pitch": 64, "start": 0, "duration": 0.04}],
                "notesMode": "strict",
            })
            websocket.receive_json()
            websocket.receive_bytes()
            command = app.state.service.commands[0]

    assert command.prompt is None
    assert command.control.notes[0].pitch == 64
    assert command.control.notes_mode == "strict"


def test_streaming_websocket_returns_pcm_chunks(tmp_path: Path) -> None:
    app = create_test_app(tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as websocket:
            websocket.send_json({
                "type": "start", "requestId": "stream-1",
                "prompt": "ambient", "duration": 0.04, "chunkFrames": 1,
            })
            ready = websocket.receive_json()
            metadata = websocket.receive_json()
            pcm = websocket.receive_bytes()
            completed = websocket.receive_json()

    assert ready["type"] == "ready"
    assert ready["sampleFormat"] == "float32le"
    assert metadata == {
        "type": "chunk", "requestId": "stream-1", "sequence": 0,
        "frames": 1, "samplesPerChannel": 1920,
        "byteLength": len(pcm), "timestampMs": 0,
    }
    assert len(pcm) == 1_920 * 2 * 4
    assert completed["type"] == "completed"
    assert completed["reason"] == "duration_reached"


def test_streaming_websocket_can_stop_early(tmp_path: Path) -> None:
    app = create_test_app(tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as websocket:
            websocket.send_json({
                "type": "start", "requestId": "stop-1",
                "prompt": "slow", "duration": 1, "chunkFrames": 1,
            })
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"type": "stop", "requestId": "stop-1"})
            messages = []
            while True:
                message = websocket.receive_json()
                messages.append(message)
                if message["type"] == "chunk":
                    websocket.receive_bytes()
                if message["type"] == "completed":
                    break

    assert messages[-1]["reason"] == "client_stop"
    assert messages[-1]["generatedSamples"] < 48_000


def test_streaming_websocket_accepts_dynamic_updates(tmp_path: Path) -> None:
    app = create_test_app(tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as websocket:
            websocket.send_json({
                "type": "start", "requestId": "dynamic-1",
                "prompt": "slow", "duration": 0.16, "chunkFrames": 1,
            })
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({
                "type": "update",
                "requestId": "dynamic-1",
                "revision": 4,
                "prompt": "driving techno",
                "temperature": 0.8,
                "topK": 12,
                "cfgMusiccoca": 4.0,
                "notes": [{"pitch": 64, "start": 0, "duration": 0.08}],
                "drums": [{"time": 0}],
                "notesMode": "strict",
            })
            messages = []
            while True:
                message = websocket.receive_json()
                messages.append(message)
                if message["type"] == "chunk":
                    websocket.receive_bytes()
                if message["type"] == "completed":
                    break
            updates = list(app.state.service.stream_session.updates)

    accepted = next(item for item in messages if item["type"] == "updateAccepted")
    assert accepted["revision"] == 4
    assert accepted["effectiveTimestampMs"] == accepted["effectiveFrame"] * 40
    assert updates[0].prompt == "driving techno"
    assert updates[0].sampling.temperature == 0.8
    assert updates[0].sampling.top_k == 12
    assert updates[0].notes[0].pitch == 64
    assert updates[0].notes_mode == "strict"
