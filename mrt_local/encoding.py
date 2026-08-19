from __future__ import annotations

import io
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import soundfile as sf

from .core import GenerateResult

AudioFormat = Literal["wav", "mp3"]
SUPPORTED_AUDIO_FORMATS: tuple[AudioFormat, ...] = ("wav", "mp3")
DEFAULT_AUDIO_FORMAT: AudioFormat = "wav"
DEFAULT_MP3_BITRATE = 192
MIN_MP3_BITRATE = 32
MAX_MP3_BITRATE = 320


class AudioEncodingError(RuntimeError):
    """音频编码器不可用或编码失败。"""


@dataclass(frozen=True, slots=True)
class AudioEncodingOptions:
    format: AudioFormat = DEFAULT_AUDIO_FORMAT
    bitrate: int | None = None

    def validate(self) -> None:
        if self.format not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError(f"不支持的音频格式：{self.format}")
        if self.format == "wav" and self.bitrate is not None:
            raise ValueError("bitrate 只适用于 MP3 格式")
        if self.format == "mp3" and self.bitrate is not None:
            if not MIN_MP3_BITRATE <= self.bitrate <= MAX_MP3_BITRATE:
                raise ValueError("MP3 bitrate 必须在 32 到 320 kbps 之间")

    @property
    def effective_bitrate(self) -> int:
        return self.bitrate or DEFAULT_MP3_BITRATE


@dataclass(frozen=True, slots=True)
class EncodedAudio:
    data: bytes
    format: AudioFormat
    media_type: str
    extension: str


def encode_audio(
    result: GenerateResult,
    options: AudioEncodingOptions = AudioEncodingOptions(),
) -> EncodedAudio:
    options.validate()
    if options.format == "wav":
        output = io.BytesIO()
        sf.write(
            output,
            result.audio,
            result.sample_rate,
            format="WAV",
            subtype="FLOAT",
        )
        return EncodedAudio(output.getvalue(), "wav", "audio/wav", ".wav")
    return EncodedAudio(
        _encode_mp3(result, options.effective_bitrate),
        "mp3",
        "audio/mpeg",
        ".mp3",
    )


def infer_cli_encoding(
    output: Path,
    requested: str | None,
    bitrate: int | None = None,
) -> AudioEncodingOptions:
    suffix = output.suffix.lower()
    inferred = suffix.removeprefix(".") if suffix in {".wav", ".mp3"} else None
    if requested is None and inferred is None:
        raise ValueError("无法从输出路径推断格式；请使用 .wav/.mp3 扩展名或 --format")
    audio_format = requested or inferred
    if inferred is not None and requested is not None and inferred != requested:
        raise ValueError("--format 与输出文件扩展名不一致")
    options = AudioEncodingOptions(
        format=cast(AudioFormat, audio_format),
        bitrate=bitrate,
    )
    options.validate()
    return options


def _encode_mp3(result: GenerateResult, bitrate: int) -> bytes:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioEncodingError(
            "MP3 编码需要 FFmpeg；请先运行 `brew install ffmpeg`"
        )

    audio = np.asarray(result.audio, dtype="<f4", order="C")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(result.sample_rate),
        "-ac",
        str(result.channels),
        "-i",
        "pipe:0",
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        f"{bitrate}k",
        "-f",
        "mp3",
        "pipe:1",
    ]
    try:
        process = subprocess.run(
            command,
            input=audio.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise AudioEncodingError(f"无法启动 FFmpeg：{exc}") from exc
    if process.returncode != 0:
        details = process.stderr.decode("utf-8", errors="replace").strip()
        raise AudioEncodingError(f"FFmpeg MP3 编码失败：{details or '未知错误'}")
    if not process.stdout:
        raise AudioEncodingError("FFmpeg 没有返回 MP3 数据")
    return process.stdout
