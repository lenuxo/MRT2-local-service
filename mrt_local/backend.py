from __future__ import annotations

import math
import os
from typing import Any, Protocol

import numpy as np

from .config import RuntimeConfig
from .core import (
    ControlTimeline,
    ResolvedGenerateCommand,
    ResolvedStreamGenerateCommand,
    SamplingConfig,
)

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
        from magenta_rt.config import DRUM_PIANOROLL, MUSICCOCA, PIANOROLL_WITH_ONSETS

        paths.set_magenta_home(model.root)
        self._conditioning_key = MUSICCOCA.key
        self._notes_conditioning_key = PIANOROLL_WITH_ONSETS.key
        self._drums_conditioning_key = DRUM_PIANOROLL.key
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
    ) -> np.ndarray | None:
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
        if not embeddings:
            return None
        return np.asarray(
            np.average(np.stack(embeddings), axis=0, weights=weights),
            dtype=np.float32,
        )

    def generate(self, command: ResolvedGenerateCommand) -> np.ndarray:
        sampling = command.sampling
        embedding = self._build_style_embedding(command)
        if command.control_timeline is not None:
            session = MagentaMlxStreamSession(
                backend=self._backend,
                conditioning_key=self._conditioning_key,
                notes_conditioning_key=self._notes_conditioning_key,
                drums_conditioning_key=self._drums_conditioning_key,
                embedding=embedding,
                sampling=sampling,
                control_timeline=command.control_timeline,
            )
            chunks = [
                session.generate_chunk(1)
                for _ in range(math.ceil(command.duration * MAGENTA_FRAMES_PER_SECOND))
            ]
            session.close()
            return np.concatenate(chunks, axis=0)
        conditioning = (
            {self._conditioning_key: embedding} if embedding is not None else {}
        )
        waveform, _ = self._backend.generate(
            conditioning=conditioning,
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
            notes_conditioning_key=self._notes_conditioning_key,
            drums_conditioning_key=self._drums_conditioning_key,
            embedding=self._build_style_embedding(command),
            sampling=command.sampling,
            control_timeline=command.control_timeline,
        )


class MagentaMlxStreamSession:
    """持有官方生成 state，并按固定帧数连续生成 PCM。"""

    def __init__(
        self,
        *,
        backend: Any,
        conditioning_key: str,
        notes_conditioning_key: str,
        drums_conditioning_key: str,
        embedding: np.ndarray | None,
        sampling: SamplingConfig,
        control_timeline: ControlTimeline | None,
    ) -> None:
        self._backend = backend
        self._conditioning_key = conditioning_key
        self._notes_conditioning_key = notes_conditioning_key
        self._drums_conditioning_key = drums_conditioning_key
        self._embedding = embedding
        self._sampling = sampling
        self._control_timeline = control_timeline
        self._control_frame = 0
        self._state: Any = None
        self._closed = False

    def generate_chunk(self, frames: int) -> np.ndarray:
        if self._closed:
            raise RuntimeError("流式后端会话已经关闭")
        if self._control_timeline is None:
            conditioning = (
                {self._conditioning_key: self._embedding}
                if self._embedding is not None else {}
            )
            waveform, self._state = self._generate(conditioning, frames)
            return np.asarray(waveform.samples, dtype=np.float32)

        chunks: list[np.ndarray] = []
        for _ in range(frames):
            conditioning = {}
            if self._embedding is not None:
                conditioning[self._conditioning_key] = self._embedding
            if self._control_timeline.notes is not None:
                conditioning[self._notes_conditioning_key] = self._control_timeline.notes[
                    self._control_frame
                ]
            if self._control_timeline.drums is not None:
                conditioning[self._drums_conditioning_key] = [int(
                    self._control_timeline.drums[self._control_frame]
                )]
            waveform, self._state = self._generate(conditioning, 1)
            chunks.append(np.asarray(waveform.samples, dtype=np.float32))
            self._control_frame += 1
        return np.concatenate(chunks, axis=0)

    def _generate(self, conditioning: dict[str, Any], frames: int):
        return self._backend.generate(
            conditioning=conditioning,
            cfg_scales=self._sampling.cfg_scales,
            temperature=self._sampling.temperature,
            top_k=self._sampling.top_k,
            frames=frames,
            state=self._state,
        )

    def close(self) -> None:
        self._closed = True


def create_magenta_backend(config: RuntimeConfig) -> GenerationBackend:
    return MagentaMlxBackend(config)
