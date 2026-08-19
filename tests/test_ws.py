from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from mrt_local.api import create_app
from mrt_local.config import RuntimeConfig
from mrt_local.core import GenerateCommand, GenerateResult, ModelConfig


class FakeService:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.is_loaded = False
        self.commands: list[GenerateCommand] = []

    def load(self) -> None:
        self.is_loaded = True

    def generate(self, command: GenerateCommand) -> GenerateResult:
        self.commands.append(command)
        if command.prompt == "fail":
            raise RuntimeError("backend details must not leak")
        return GenerateResult(
            48_000,
            2,
            np.zeros((round(command.duration * 48_000), 2), np.float32),
        )


def create_test_app(tmp_path: Path):
    config = RuntimeConfig(model=ModelConfig(name="mrt2_small", root=tmp_path))
    return create_app(config, service_factory=FakeService)


def test_websocket_returns_metadata_then_binary_wav(tmp_path: Path) -> None:
    app = create_test_app(tmp_path)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/generate") as websocket:
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
