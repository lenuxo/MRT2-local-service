from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from mrt_local.api import create_app
from mrt_local.config import EngineConfig
from mrt_local.engine import GenerateResult


class FakeEngine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.is_loaded = False

    def load(self) -> None:
        self.is_loaded = True

    def generate(self, prompt: str, duration: float) -> GenerateResult:
        return GenerateResult(48_000, 2, np.zeros((round(duration * 48_000), 2), np.float32))


def test_api_and_openapi(tmp_path: Path) -> None:
    config = EngineConfig(model="mrt2_base", model_root=tmp_path)
    app = create_app(config, engine_factory=FakeEngine)

    with TestClient(app) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "model": "mrt2_base",
            "loaded": True,
        }
        assert client.get("/info").json()["sampleRate"] == 48_000
        response = client.post("/generate", json={"prompt": "techno", "duration": 0.04})
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.content[:4] == b"RIFF"

        schema = client.get("/openapi.json").json()
        assert schema["info"]["title"] == "MRT2 本地服务 API"
        assert "audio/wav" in schema["paths"]["/generate"]["post"]["responses"]["200"]["content"]


def test_generate_validation(tmp_path: Path) -> None:
    app = create_app(
        EngineConfig(model="mrt2_small", model_root=tmp_path),
        engine_factory=FakeEngine,
    )
    with TestClient(app) as client:
        assert client.post("/generate", json={"prompt": ""}).status_code == 422
        assert client.post("/generate", json={"prompt": "x", "unknown": 1}).status_code == 422
