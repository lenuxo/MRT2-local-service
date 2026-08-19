from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from mrt_local.api import create_app
from mrt_local.config import RuntimeConfig
from mrt_local.core import GenerateCommand, GenerateResult, ModelConfig, SamplingOverrides


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
        request_schema = schema["components"]["schemas"]["GenerateRequest"]["properties"]
        assert "temperature" in request_schema
        assert "pool_across_time" in request_schema
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
