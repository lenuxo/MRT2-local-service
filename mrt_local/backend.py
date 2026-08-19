from __future__ import annotations

import math
import os
from typing import Any, Protocol

import numpy as np

from .config import RuntimeConfig
from .core import ResolvedGenerateCommand

MAGENTA_FRAMES_PER_SECOND = 25


class GenerationBackend(Protocol):
    def generate(self, command: ResolvedGenerateCommand) -> np.ndarray: ...


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
        self._backend: Any = MagentaRT2StdMlxfn(
            size=model.name,
            temperature=defaults.temperature,
            top_k=defaults.top_k,
            cfg_scales=defaults.cfg_scales,
            warmup_steps=model.warmup_steps,
        )

    def generate(self, command: ResolvedGenerateCommand) -> np.ndarray:
        sampling = command.sampling
        embedding = self._backend.embed_style(
            command.prompt,
            pool_across_time=sampling.pool_across_time,
            use_mapper=sampling.use_mapper,
            seed=sampling.seed,
        )
        waveform, _ = self._backend.generate(
            conditioning={self._conditioning_key: embedding},
            cfg_scales=sampling.cfg_scales,
            temperature=sampling.temperature,
            top_k=sampling.top_k,
            frames=math.ceil(command.duration * MAGENTA_FRAMES_PER_SECOND),
            state=None,
        )
        return np.asarray(waveform.samples)


def create_magenta_backend(config: RuntimeConfig) -> GenerationBackend:
    return MagentaMlxBackend(config)
