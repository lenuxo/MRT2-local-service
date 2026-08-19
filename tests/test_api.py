from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from mrt_local.api import create_app
from mrt_local.config import RuntimeConfig
from mrt_local.core import AudioChunk, GenerateCommand, GenerateResult, ModelConfig, SamplingOverrides
from mrt_local.encoding import encode_audio


class FakeService:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.is_loaded = False
        self.command = None

    def load(self) -> None:
        self.is_loaded = True

    def generate(self, command: GenerateCommand) -> GenerateResult:
        self.command = command
        return GenerateResult(
            48_000,
            2,
            np.zeros((round(command.duration * 48_000), 2), np.float32),
        )

    def open_stream(self, command):
        self.command = command
        return FakeStreamingSession(round(command.duration * 48_000))


class FakeStreamingSession:
    def __init__(self, samples: int) -> None:
        self.remaining = samples
        self.generated_samples = 0
        self.closed = False

    def next_chunk(self):
        if not self.remaining:
            return None
        count = min(self.remaining, 9_600)
        chunk = AudioChunk(0, self.generated_samples, 48_000, 2, np.zeros((count, 2), np.float32))
        self.remaining -= count
        self.generated_samples += count
        return chunk

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_api_and_openapi(tmp_path: Path) -> None:
    config = RuntimeConfig(model=ModelConfig(name="mrt2_base", root=tmp_path))
    app = create_app(config, service_factory=FakeService)

    with TestClient(app) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "model": "mrt2_base",
            "loaded": True,
        }
        info = client.get("/info").json()
        assert info["sampleRate"] == 48_000
        assert info["temperature"] == 1.3
        response = client.post(
            "/generate",
            json={
                "prompt": "techno",
                "duration": 0.04,
                "temperature": 0.8,
                "top_k": 16,
                "cfg_musiccoca": 2.0,
                "cfg_notes": 0.5,
                "cfg_drums": 0.25,
                "seed": 7,
                "use_mapper": False,
                "pool_across_time": False,
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.content[:4] == b"RIFF"

        schema = client.get("/openapi.json").json()
        assert schema["info"]["title"] == "MRT2 本地服务 API"
        assert "audio/wav" in schema["paths"]["/generate"]["post"]["responses"]["200"]["content"]
        assert "audio/mpeg" in schema["paths"]["/generate"]["post"]["responses"]["200"]["content"]
        request_schema = schema["components"]["schemas"]["GenerateRequest"]["properties"]
        assert "temperature" in request_schema
        assert "pool_across_time" in request_schema
        assert request_schema["format"]["default"] == "wav"
        assert "bitrate" in request_schema
        assert app.state.service.command == GenerateCommand(
            prompt="techno",
            duration=0.04,
            sampling=SamplingOverrides(
                temperature=0.8,
                top_k=16,
                cfg_musiccoca=2.0,
                cfg_notes=0.5,
                cfg_drums=0.25,
                seed=7,
                use_mapper=False,
                pool_across_time=False,
            ),
        )


def test_generate_validation(tmp_path: Path) -> None:
    app = create_app(
        RuntimeConfig(model=ModelConfig(name="mrt2_small", root=tmp_path)),
        service_factory=FakeService,
    )
    with TestClient(app) as client:
        assert client.post("/generate", json={"prompt": ""}).status_code == 422
        assert client.post("/generate", json={"prompt": "x", "unknown": 1}).status_code == 422
        assert client.post("/generate", json={"prompt": "x", "temperature": 0}).status_code == 422
        assert client.post("/generate", json={"prompt": "x", "top_k": 0}).status_code == 422
        assert client.post("/generate", json={"prompt": "x", "format": "flac"}).status_code == 422
        assert client.post("/generate", json={"prompt": "x", "bitrate": 192}).status_code == 400


def test_generate_mp3_response(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("mrt_local.encoding._encode_mp3", lambda result, bitrate: b"ID3-mp3")
    app = create_app(
        RuntimeConfig(model=ModelConfig(name="mrt2_small", root=tmp_path)),
        service_factory=FakeService,
    )

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={"prompt": "ambient", "duration": 0.01, "format": "mp3"},
        )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.headers["content-disposition"] == 'attachment; filename="output.mp3"'
    assert response.content == b"ID3-mp3"


def test_generate_with_uploaded_reference_audio(tmp_path: Path) -> None:
    app = create_app(
        RuntimeConfig(model=ModelConfig(name="mrt2_small", root=tmp_path)),
        service_factory=FakeService,
    )
    wav = encode_audio(
        GenerateResult(48_000, 2, np.zeros((480, 2), np.float32))
    ).data

    with TestClient(app) as client:
        response = client.post(
            "/generate/audio",
            data={"prompt": "ambient", "text_weight": "1", "audio_weight": "3", "duration": "0.01", "format": "wav"},
            files={"audio": ("reference.wav", wav, "audio/wav")},
        )
        schema = client.get("/openapi.json").json()
        command = app.state.service.command

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/wav"
    assert command.prompt == "ambient"
    assert command.text_weight == 1
    assert command.audio_weight == 3
    assert command.reference_audio is not None
    assert command.reference_audio.samples.shape == (480, 2)
    assert "/generate/audio" in schema["paths"]


def test_http_pcm_stream_and_openapi(tmp_path: Path) -> None:
    app = create_app(
        RuntimeConfig(model=ModelConfig(name="mrt2_small", root=tmp_path)),
        service_factory=FakeService,
    )
    with TestClient(app) as client:
        response = client.post("/stream", json={
            "prompt": "ambient", "duration": 0.01, "chunk_frames": 1
        })
        schema = client.get("/openapi.json").json()

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["x-audio-sample-rate"] == "48000"
    assert response.headers["x-audio-channels"] == "2"
    assert response.headers["x-audio-sample-format"] == "float32le"
    assert len(response.content) == 480 * 2 * 4
    assert "/stream" in schema["paths"]
    assert "/stream/audio" in schema["paths"]


def test_http_pcm_stream_accepts_reference_audio_and_prompt(tmp_path: Path) -> None:
    app = create_app(
        RuntimeConfig(model=ModelConfig(name="mrt2_small", root=tmp_path)),
        service_factory=FakeService,
    )
    wav = encode_audio(
        GenerateResult(48_000, 2, np.zeros((480, 2), np.float32))
    ).data
    with TestClient(app) as client:
        response = client.post(
            "/stream/audio",
            data={"prompt": "ambient", "duration": "0.01", "chunk_frames": "1"},
            files={"audio": ("reference.wav", wav, "audio/wav")},
        )
        command = app.state.service.command

    assert response.status_code == 200
    assert len(response.content) == 480 * 2 * 4
    assert command.prompt == "ambient"
    assert command.reference_audio is not None
