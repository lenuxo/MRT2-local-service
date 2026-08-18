from __future__ import annotations

import io
import math
import threading
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import soundfile as sf

from .config import CHANNELS, SAMPLE_RATE, EngineConfig


@dataclass(frozen=True, slots=True)
class GenerateResult:
    sample_rate: int
    channels: int
    audio: np.ndarray

    def to_wav_bytes(self) -> bytes:
        output = io.BytesIO()
        sf.write(
            output,
            self.audio,
            self.sample_rate,
            format="WAV",
            subtype="FLOAT",
        )
        return output.getvalue()


BackendFactory = Callable[[EngineConfig], tuple[Any, str]]


def _create_magenta_backend(config: EngineConfig) -> tuple[Any, str]:
    # Magenta 的 paths 模块在导入时读取路径，因此必须先设置环境变量。
    import os

    os.environ["MAGENTA_HOME"] = str(config.model_root.parent)
    from magenta_rt import MagentaRT2StdMlxfn, paths
    from magenta_rt.config import MUSICCOCA

    paths.set_magenta_home(config.model_root)
    backend = MagentaRT2StdMlxfn(size=config.model)
    return backend, MUSICCOCA.key


class MrtEngine:
    """CLI 与 HTTP API 共用的 MRT2 Python/MLX 推理生命周期。"""

    def __init__(
        self,
        config: EngineConfig,
        backend_factory: BackendFactory = _create_magenta_backend,
    ) -> None:
        self.config = config
        self._backend_factory = backend_factory
        self._backend: Any | None = None
        self._conditioning_key = "musiccoca"
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    def load(self) -> None:
        with self._lock:
            if self.is_loaded:
                return
            missing = [
                path
                for path in (
                    self.config.model_path,
                    self.config.state_path,
                    self.config.resources_path / "musiccoca",
                )
                if not path.exists()
            ]
            if missing:
                details = "\n".join(f"- {path}" for path in missing)
                raise FileNotFoundError(f"缺少 MRT2 模型或资源：\n{details}")
            self._backend, self._conditioning_key = self._backend_factory(self.config)

    def generate(self, prompt: str, duration: float) -> GenerateResult:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt 不能为空")
        if not math.isfinite(duration) or duration <= 0 or duration > 300:
            raise ValueError("duration 必须大于 0 且不超过 300 秒")

        with self._lock:
            if self._backend is None:
                raise RuntimeError("MRT2 模型尚未加载")
            embedding = self._backend.embed_style(prompt, use_mapper=True)
            frame_count = math.ceil(duration * 25)
            waveform, _ = self._backend.generate(
                conditioning={self._conditioning_key: embedding},
                frames=frame_count,
                state=None,
            )
            sample_count = round(duration * SAMPLE_RATE)
            audio = np.asarray(waveform.samples[:sample_count], dtype=np.float32)

        if audio.ndim != 2 or audio.shape[1] != CHANNELS:
            raise RuntimeError(f"MRT2 返回了非双声道音频：{audio.shape}")
        return GenerateResult(SAMPLE_RATE, CHANNELS, audio)
