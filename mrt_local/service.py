from __future__ import annotations

import asyncio
import math
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

import numpy as np

from .backend import MAGENTA_FRAMES_PER_SECOND, GenerationBackend, StreamingBackendSession, create_magenta_backend
from .config import RuntimeConfig
from .core import (
    AudioChunk,
    CHANNELS,
    SAMPLE_RATE,
    GenerateCommand,
    GenerateResult,
    LiveMidiCommand,
    LiveMidiQueueResult,
    ResolvedGenerateCommand,
    ResolvedStreamGenerateCommand,
    StreamGenerateCommand,
    StreamExtendCommand,
    StreamExtendResult,
    StreamUpdateCommand,
    StreamUpdateResult,
)

BackendFactory = Callable[[RuntimeConfig], GenerationBackend]
T = TypeVar("T")


async def _await_executor_future(future: Future[T]) -> T:
    """等待专用线程任务；调用方取消时仍等任务安全落地。"""
    wrapped = asyncio.wrap_future(future)
    try:
        return await asyncio.shield(wrapped)
    except asyncio.CancelledError:
        await wrapped
        raise


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
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mrt2-mlx",
        )
        self._closed = False
        self._status_lock = threading.Lock()
        self._active_operation: str | None = None
        self._active_session_id: str | None = None
        self._active_generated_samples = 0
        self._active_target_samples = 0
        self._active_started_at: float | None = None

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    def load(self) -> None:
        with self._lifecycle_lock:
            if self.is_loaded:
                return
            self._ensure_open()
            self._executor.submit(self._load_on_model_thread).result()

    async def load_async(self) -> None:
        with self._lifecycle_lock:
            if self.is_loaded:
                return
            self._ensure_open()
            await _await_executor_future(
                self._executor.submit(self._load_on_model_thread)
            )

    def _load_on_model_thread(self) -> None:
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
        self._begin_operation("generate", resolved.sample_count)
        try:
            return self._executor.submit(
                self._generate_on_model_thread, resolved
            ).result()
        finally:
            self._end_operation()
            self._model_lease.release()

    async def generate_async(self, command: GenerateCommand) -> GenerateResult:
        resolved = command.resolve(self.config.sampling)
        if not self._model_lease.acquire(blocking=False):
            raise ModelBusyError("模型正在处理另一个生成会话")
        self._begin_operation("generate", resolved.sample_count)
        try:
            return await _await_executor_future(
                self._executor.submit(self._generate_on_model_thread, resolved)
            )
        finally:
            self._end_operation()
            self._model_lease.release()

    def _generate_on_model_thread(
        self, resolved: ResolvedGenerateCommand
    ) -> GenerateResult:
        if self._backend is None:
            raise RuntimeError("MRT2 模型尚未加载")
        audio = np.asarray(
            self._backend.generate(resolved)[: resolved.sample_count],
            dtype=np.float32,
        )
        if audio.ndim != 2 or audio.shape[1] != CHANNELS:
            raise RuntimeError(f"MRT2 返回了非双声道音频：{audio.shape}")
        if len(audio) == 0:
            raise RuntimeError("MRT2 返回了空音频")
        return GenerateResult(SAMPLE_RATE, CHANNELS, audio)

    def open_stream(self, command: StreamGenerateCommand) -> StreamingSession:
        resolved = command.resolve(self.config.sampling)
        if not self._model_lease.acquire(blocking=False):
            raise ModelBusyError("模型正在处理另一个生成会话")
        session_id = self._begin_operation("stream", resolved.sample_count)
        try:
            backend_session = self._executor.submit(
                self._open_stream_on_model_thread, resolved
            ).result()
        except Exception:
            self._end_operation()
            self._model_lease.release()
            raise
        return StreamingSession(
            backend_session=backend_session,
            sample_count=resolved.sample_count,
            chunk_frames=resolved.chunk_frames,
            release_lease=self._model_lease.release,
            executor=self._executor,
            session_id=session_id,
            report_progress=self._report_stream_progress,
            end_operation=self._end_operation,
        )

    async def open_stream_async(
        self, command: StreamGenerateCommand
    ) -> StreamingSession:
        resolved = command.resolve(self.config.sampling)
        if not self._model_lease.acquire(blocking=False):
            raise ModelBusyError("模型正在处理另一个生成会话")
        session_id = self._begin_operation("stream", resolved.sample_count)
        try:
            backend_session = await _await_executor_future(
                self._executor.submit(self._open_stream_on_model_thread, resolved)
            )
        except BaseException:
            self._end_operation()
            self._model_lease.release()
            raise
        return StreamingSession(
            backend_session=backend_session,
            sample_count=resolved.sample_count,
            chunk_frames=resolved.chunk_frames,
            release_lease=self._model_lease.release,
            executor=self._executor,
            session_id=session_id,
            report_progress=self._report_stream_progress,
            end_operation=self._end_operation,
        )

    def _open_stream_on_model_thread(
        self, resolved: ResolvedStreamGenerateCommand
    ) -> StreamingBackendSession:
        if self._backend is None:
            raise RuntimeError("MRT2 模型尚未加载")
        return self._backend.open_stream(resolved)

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            self._executor.submit(self._unload_on_model_thread).result()
            self._executor.shutdown(wait=True)

    def _unload_on_model_thread(self) -> None:
        self._backend = None

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MRT2 服务已经关闭")

    def _begin_operation(self, kind: str, target_samples: int = 0) -> str:
        session_id = str(uuid.uuid4())
        with self._status_lock:
            self._active_operation = kind
            self._active_session_id = session_id
            self._active_generated_samples = 0
            self._active_target_samples = target_samples
            self._active_started_at = time.monotonic()
        return session_id

    def _report_stream_progress(self, generated: int, target: int) -> None:
        with self._status_lock:
            self._active_generated_samples = generated
            self._active_target_samples = target

    def _end_operation(self) -> None:
        with self._status_lock:
            self._active_operation = None
            self._active_session_id = None
            self._active_generated_samples = 0
            self._active_target_samples = 0
            self._active_started_at = None

    def status(self) -> dict[str, object]:
        with self._status_lock:
            elapsed_ms = (
                round((time.monotonic() - self._active_started_at) * 1000)
                if self._active_started_at is not None else None
            )
            return {
                "loaded": self.is_loaded,
                "busy": self._model_lease.locked(),
                "operation": self._active_operation,
                "session_id": self._active_session_id,
                "generated_samples": self._active_generated_samples,
                "target_samples": self._active_target_samples,
                "elapsed_ms": elapsed_ms,
            }


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
        executor: ThreadPoolExecutor,
        session_id: str,
        report_progress: Callable[[int, int], None],
        end_operation: Callable[[], None],
    ) -> None:
        self._backend_session = backend_session
        self._sample_count = sample_count
        self._chunk_frames = chunk_frames
        self._release_lease = release_lease
        self._executor = executor
        self.session_id = session_id
        self._report_progress = report_progress
        self._end_operation = end_operation
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
        return self._executor.submit(self._next_chunk_on_model_thread).result()

    async def next_chunk_async(self) -> AudioChunk | None:
        return await _await_executor_future(
            self._executor.submit(self._next_chunk_on_model_thread)
        )

    def _next_chunk_on_model_thread(self) -> AudioChunk | None:
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
        self._report_progress(self._generated_samples, self._sample_count)
        self._sequence += 1
        return chunk

    def close(self) -> None:
        if self._closed:
            return
        self._executor.submit(self._close_on_model_thread).result()

    def update(self, command: StreamUpdateCommand) -> StreamUpdateResult:
        if self._closed:
            raise RuntimeError("流式会话已经关闭")
        return self._executor.submit(
            self._backend_session.update, command
        ).result()

    async def update_async(
        self, command: StreamUpdateCommand
    ) -> StreamUpdateResult:
        if self._closed:
            raise RuntimeError("流式会话已经关闭")
        return await _await_executor_future(
            self._executor.submit(self._backend_session.update, command)
        )

    def queue_live_midi(
        self, command: LiveMidiCommand
    ) -> LiveMidiQueueResult:
        """无阻塞入队；后端会在 MLX 线程生成下一帧前读取。"""
        if self._closed:
            raise RuntimeError("流式会话已经关闭")
        return self._backend_session.queue_live_midi(command)

    async def queue_live_midi_async(
        self, command: LiveMidiCommand
    ) -> LiveMidiQueueResult:
        return self.queue_live_midi(command)

    def extend(self, command: StreamExtendCommand) -> StreamExtendResult:
        if self._closed:
            raise RuntimeError("流式会话已经关闭")
        return self._executor.submit(
            self._extend_on_model_thread, command
        ).result()

    async def extend_async(
        self, command: StreamExtendCommand
    ) -> StreamExtendResult:
        if self._closed:
            raise RuntimeError("流式会话已经关闭")
        return await _await_executor_future(
            self._executor.submit(self._extend_on_model_thread, command)
        )

    def _extend_on_model_thread(
        self, command: StreamExtendCommand
    ) -> StreamExtendResult:
        command.validate()
        previous = self._sample_count
        self._sample_count += round(command.additional_duration * SAMPLE_RATE)
        self._report_progress(self._generated_samples, self._sample_count)
        frames = math.ceil(
            self._sample_count / (SAMPLE_RATE // MAGENTA_FRAMES_PER_SECOND)
        )
        self._backend_session.extend_to(frames)
        return StreamExtendResult(command.revision, previous, self._sample_count)

    def configure_chunk_frames(self, chunk_frames: int) -> None:
        if self._closed:
            raise RuntimeError("流式会话已经关闭")
        self._executor.submit(
            self._configure_chunk_frames_on_model_thread, chunk_frames
        ).result()

    async def configure_chunk_frames_async(self, chunk_frames: int) -> None:
        if self._closed:
            raise RuntimeError("流式会话已经关闭")
        await _await_executor_future(self._executor.submit(
            self._configure_chunk_frames_on_model_thread, chunk_frames
        ))

    def _configure_chunk_frames_on_model_thread(self, chunk_frames: int) -> None:
        if not 1 <= chunk_frames <= 25:
            raise ValueError("chunkFrames 必须在 1 到 25 之间")
        self._chunk_frames = chunk_frames

    async def close_async(self) -> None:
        if self._closed:
            return
        await _await_executor_future(
            self._executor.submit(self._close_on_model_thread)
        )

    def _close_on_model_thread(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._backend_session.close()
        finally:
            self._end_operation()
            self._release_lease()

    def __enter__(self) -> StreamingSession:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
