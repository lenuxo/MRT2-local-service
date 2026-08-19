from __future__ import annotations

import math
import threading
from collections.abc import Callable

import numpy as np

from .backend import MAGENTA_FRAMES_PER_SECOND, GenerationBackend, StreamingBackendSession, create_magenta_backend
from .config import RuntimeConfig
from .core import AudioChunk, CHANNELS, SAMPLE_RATE, GenerateCommand, GenerateResult, StreamGenerateCommand

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
        self._lifecycle_lock = threading.Lock()
        self._model_lease = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    def load(self) -> None:
        with self._lifecycle_lock:
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
        if not self._model_lease.acquire(blocking=False):
            raise ModelBusyError("模型正在处理另一个生成会话")
        try:
            if self._backend is None:
                raise RuntimeError("MRT2 模型尚未加载")
            audio = np.asarray(
                self._backend.generate(resolved)[: resolved.sample_count],
                dtype=np.float32,
            )
        finally:
            self._model_lease.release()

        if audio.ndim != 2 or audio.shape[1] != CHANNELS:
            raise RuntimeError(f"MRT2 返回了非双声道音频：{audio.shape}")
        if len(audio) == 0:
            raise RuntimeError("MRT2 返回了空音频")
        return GenerateResult(SAMPLE_RATE, CHANNELS, audio)

    def open_stream(self, command: StreamGenerateCommand) -> StreamingSession:
        resolved = command.resolve(self.config.sampling)
        if not self._model_lease.acquire(blocking=False):
            raise ModelBusyError("模型正在处理另一个生成会话")
        try:
            if self._backend is None:
                raise RuntimeError("MRT2 模型尚未加载")
            backend_session = self._backend.open_stream(resolved)
        except Exception:
            self._model_lease.release()
            raise
        return StreamingSession(
            backend_session=backend_session,
            sample_count=resolved.sample_count,
            chunk_frames=resolved.chunk_frames,
            release_lease=self._model_lease.release,
        )


class ModelBusyError(RuntimeError):
    """唯一模型实例已经被普通生成或流式会话占用。"""


class StreamingSession:
    """协议无关、独占模型的有限时长 PCM 流式会话。"""

    def __init__(
        self,
        *,
        backend_session: StreamingBackendSession,
        sample_count: int,
        chunk_frames: int,
        release_lease: Callable[[], None],
    ) -> None:
        self._backend_session = backend_session
        self._sample_count = sample_count
        self._chunk_frames = chunk_frames
        self._release_lease = release_lease
        self._generated_samples = 0
        self._sequence = 0
        self._closed = False

    @property
    def completed(self) -> bool:
        return self._generated_samples >= self._sample_count

    @property
    def generated_samples(self) -> int:
        return self._generated_samples

    def next_chunk(self) -> AudioChunk | None:
        if self._closed:
            raise RuntimeError("流式会话已经关闭")
        if self.completed:
            return None
        remaining = self._sample_count - self._generated_samples
        samples_per_frame = SAMPLE_RATE // MAGENTA_FRAMES_PER_SECOND
        frames = min(
            self._chunk_frames,
            math.ceil(remaining / samples_per_frame),
        )
        audio = np.asarray(self._backend_session.generate_chunk(frames), dtype=np.float32)
        if audio.ndim != 2 or audio.shape[1] != CHANNELS:
            raise RuntimeError(f"MRT2 返回了非双声道音频：{audio.shape}")
        audio = audio[:remaining]
        if len(audio) == 0:
            raise RuntimeError("MRT2 返回了空音频分片")
        chunk = AudioChunk(
            sequence=self._sequence,
            start_sample=self._generated_samples,
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            audio=audio,
        )
        self._generated_samples += len(audio)
        self._sequence += 1
        return chunk

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._backend_session.close()
        finally:
            self._release_lease()

    def __enter__(self) -> StreamingSession:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
