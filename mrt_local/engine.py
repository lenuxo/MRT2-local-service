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
    backend = MagentaRT2StdMlxfn(
        size=config.model,
        temperature=config.temperature,
        top_k=config.top_k,
        cfg_scales={
            "musiccoca": config.cfg_musiccoca,
            "notes": config.cfg_notes,
            "drums": config.cfg_drums,
        },
        warmup_steps=config.warmup_steps,
    )
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
            if not math.isfinite(self.config.temperature) or self.config.temperature <= 0:
                raise ValueError("temperature 必须是大于 0 的有限数")
            if self.config.top_k < 1:
                raise ValueError("top_k 必须大于等于 1")
            if self.config.warmup_steps < 0:
                raise ValueError("warmup_steps 必须大于等于 0")
            if not all(
                math.isfinite(value)
                for value in (
                    self.config.cfg_musiccoca,
                    self.config.cfg_notes,
                    self.config.cfg_drums,
                )
            ):
                raise ValueError("CFG 参数必须是有限数")
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

    def generate(
        self,
        prompt: str,
        duration: float,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        cfg_musiccoca: float | None = None,
        cfg_notes: float | None = None,
        cfg_drums: float | None = None,
        seed: int | None = None,
        use_mapper: bool | None = None,
        pool_across_time: bool | None = None,
    ) -> GenerateResult:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt 不能为空")
        if not math.isfinite(duration) or duration <= 0 or duration > 300:
            raise ValueError("duration 必须大于 0 且不超过 300 秒")
        temperature = self.config.temperature if temperature is None else temperature
        top_k = self.config.top_k if top_k is None else top_k
        cfg_musiccoca = self.config.cfg_musiccoca if cfg_musiccoca is None else cfg_musiccoca
        cfg_notes = self.config.cfg_notes if cfg_notes is None else cfg_notes
        cfg_drums = self.config.cfg_drums if cfg_drums is None else cfg_drums
        seed = self.config.seed if seed is None else seed
        use_mapper = self.config.use_mapper if use_mapper is None else use_mapper
        pool_across_time = (
            self.config.pool_across_time
            if pool_across_time is None
            else pool_across_time
        )
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature 必须是大于 0 的有限数")
        if top_k < 1:
            raise ValueError("top_k 必须大于等于 1")
        cfg_values = (cfg_musiccoca, cfg_notes, cfg_drums)
        if not all(math.isfinite(value) for value in cfg_values):
            raise ValueError("CFG 参数必须是有限数")

        with self._lock:
            if self._backend is None:
                raise RuntimeError("MRT2 模型尚未加载")
            embedding = self._backend.embed_style(
                prompt,
                pool_across_time=pool_across_time,
                use_mapper=use_mapper,
                seed=seed,
            )
            frame_count = math.ceil(duration * 25)
            waveform, _ = self._backend.generate(
                conditioning={self._conditioning_key: embedding},
                cfg_scales={
                    "musiccoca": cfg_musiccoca,
                    "notes": cfg_notes,
                    "drums": cfg_drums,
                },
                temperature=temperature,
                top_k=top_k,
                frames=frame_count,
                state=None,
            )
            sample_count = round(duration * SAMPLE_RATE)
            audio = np.asarray(waveform.samples[:sample_count], dtype=np.float32)

        if audio.ndim != 2 or audio.shape[1] != CHANNELS:
            raise RuntimeError(f"MRT2 返回了非双声道音频：{audio.shape}")
        return GenerateResult(SAMPLE_RATE, CHANNELS, audio)
