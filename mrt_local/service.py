from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np

from .backend import GenerationBackend, create_magenta_backend
from .config import RuntimeConfig
from .core import CHANNELS, SAMPLE_RATE, GenerateCommand, GenerateResult

BackendFactory = Callable[[RuntimeConfig], GenerationBackend]


class GenerationService:
    """与 CLI、HTTP 或 Socket 无关的音乐生成用例。"""

    def __init__(
        self,
        config: RuntimeConfig,
        backend_factory: BackendFactory = create_magenta_backend,
    ) -> None:
        self.config = config
        self._backend_factory = backend_factory
        self._backend: GenerationBackend | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    def load(self) -> None:
        with self._lock:
            if self.is_loaded:
                return
            self.config.model.validate()
            self.config.sampling.validate()
            model = self.config.model
            missing = [
                path
                for path in (
                    model.model_path,
                    model.state_path,
                    model.resources_path / "musiccoca",
                )
                if not path.exists()
            ]
            if missing:
                details = "\n".join(f"- {path}" for path in missing)
                raise FileNotFoundError(f"缺少 MRT2 模型或资源：\n{details}")
            self._backend = self._backend_factory(self.config)

    def generate(self, command: GenerateCommand) -> GenerateResult:
        resolved = command.resolve(self.config.sampling)
        with self._lock:
            if self._backend is None:
                raise RuntimeError("MRT2 模型尚未加载")
            audio = np.asarray(
                self._backend.generate(resolved)[: resolved.sample_count],
                dtype=np.float32,
            )

        if audio.ndim != 2 or audio.shape[1] != CHANNELS:
            raise RuntimeError(f"MRT2 返回了非双声道音频：{audio.shape}")
        return GenerateResult(SAMPLE_RATE, CHANNELS, audio)
