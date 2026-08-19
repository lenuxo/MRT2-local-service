from __future__ import annotations

import math
import os
from typing import Any, Protocol

import numpy as np

from .config import RuntimeConfig
from .core import ResolvedGenerateCommand, ResolvedStreamGenerateCommand, SamplingConfig

MAGENTA_FRAMES_PER_SECOND = 25


class GenerationBackend(Protocol):
    def generate(self, command: ResolvedGenerateCommand) -> np.ndarray: ...
    def open_stream(self, command: ResolvedStreamGenerateCommand) -> StreamingBackendSession: ...


class StreamingBackendSession(Protocol):
    def generate_chunk(self, frames: int) -> np.ndarray: ...
    def close(self) -> None: ...


class MagentaMlxBackend:
    """把 Magenta 的 MLX API 隔离在基础设施适配层。"""

    def __init__(self, config: RuntimeConfig) -> None:
        model = config.model
        defaults = config.sampling

        # Magenta 的 paths 模块在导入时读取路径，因此必须先设置环境变量。
        os.environ["MAGENTA_HOME"] = str(model.root.parent)
        from magenta_rt import MagentaRT2StdMlxfn, paths
        from magenta_rt.config import MUSICCOCA

        paths.set_magenta_home(model.root)
        self._conditioning_key = MUSICCOCA.key
        try:
            self._backend: Any = MagentaRT2StdMlxfn(
                size=model.name,
                temperature=defaults.temperature,
                top_k=defaults.top_k,
                cfg_scales=defaults.cfg_scales,
                warmup_steps=model.warmup_steps,
            )
        except RuntimeError as exc:
            if "[import_function] Invalid string size" not in str(exc):
                raise
            raise RuntimeError(
                "MLX 无法导入官方 .mlxfn 模型；请运行 `uv sync` 恢复锁定的 "
                "MLX 版本，若仍失败请重新下载模型"
            ) from exc

    def _build_style_embedding(
        self,
        command: ResolvedGenerateCommand | ResolvedStreamGenerateCommand,
    ) -> np.ndarray:
        sampling = command.sampling
        embeddings: list[np.ndarray] = []
        weights: list[float] = []
        if command.prompt is not None:
            embeddings.append(np.asarray(self._backend.embed_style(
                command.prompt,
                pool_across_time=sampling.pool_across_time,
                use_mapper=sampling.use_mapper,
                seed=sampling.seed,
            )))
            weights.append(command.text_weight)
        if command.reference_audio is not None:
            from magenta_rt.audio import Waveform

            style_input: Any = Waveform(
                command.reference_audio.samples,
                command.reference_audio.sample_rate,
            )
            embeddings.append(np.asarray(self._backend.embed_style(
                style_input,
                pool_across_time=sampling.pool_across_time,
                use_mapper=sampling.use_mapper,
                seed=sampling.seed,
            )))
            weights.append(command.audio_weight)
        return np.asarray(
            np.average(np.stack(embeddings), axis=0, weights=weights),
            dtype=np.float32,
        )

    def generate(self, command: ResolvedGenerateCommand) -> np.ndarray:
        sampling = command.sampling
        embedding = self._build_style_embedding(command)
        waveform, _ = self._backend.generate(
            conditioning={self._conditioning_key: embedding},
            cfg_scales=sampling.cfg_scales,
            temperature=sampling.temperature,
            top_k=sampling.top_k,
            frames=math.ceil(command.duration * MAGENTA_FRAMES_PER_SECOND),
            state=None,
        )
        return np.asarray(waveform.samples)

    def open_stream(
        self,
        command: ResolvedStreamGenerateCommand,
    ) -> StreamingBackendSession:
        return MagentaMlxStreamSession(
            backend=self._backend,
            conditioning_key=self._conditioning_key,
            embedding=self._build_style_embedding(command),
            sampling=command.sampling,
        )


class MagentaMlxStreamSession:
    """持有官方生成 state，并按固定帧数连续生成 PCM。"""

    def __init__(
        self,
        *,
        backend: Any,
        conditioning_key: str,
        embedding: np.ndarray,
        sampling: SamplingConfig,
    ) -> None:
        self._backend = backend
        self._conditioning_key = conditioning_key
        self._embedding = embedding
        self._sampling = sampling
        self._state: Any = None
        self._closed = False

    def generate_chunk(self, frames: int) -> np.ndarray:
        if self._closed:
            raise RuntimeError("流式后端会话已经关闭")
        waveform, self._state = self._backend.generate(
            conditioning={self._conditioning_key: self._embedding},
            cfg_scales=self._sampling.cfg_scales,
            temperature=self._sampling.temperature,
            top_k=self._sampling.top_k,
            frames=frames,
            state=self._state,
        )
        return np.asarray(waveform.samples, dtype=np.float32)

    def close(self) -> None:
        self._closed = True


def create_magenta_backend(config: RuntimeConfig) -> GenerationBackend:
    return MagentaMlxBackend(config)
