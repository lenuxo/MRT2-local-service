from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mrt_local.core import GenerateResult
from mrt_local.encoding import (
    AudioEncodingError,
    AudioEncodingOptions,
    encode_audio,
    decode_audio,
    infer_cli_encoding,
)


def audio_result() -> GenerateResult:
    return GenerateResult(48_000, 2, np.zeros((480, 2), dtype=np.float32))


def test_encode_wav() -> None:
    encoded = encode_audio(audio_result())
    assert encoded.format == "wav"
    assert encoded.media_type == "audio/wav"
    assert encoded.data[:4] == b"RIFF"


def test_decode_reference_audio() -> None:
    wav = encode_audio(audio_result()).data
    decoded = decode_audio(wav)
    assert decoded.sample_rate == 48_000
    assert decoded.samples.shape == (480, 2)
    assert decoded.samples.dtype == np.float32


def test_encode_mp3_invokes_ffmpeg(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr("mrt_local.encoding.shutil.which", lambda name: "/opt/ffmpeg")

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout=b"ID3-data", stderr=b"")

    monkeypatch.setattr("mrt_local.encoding.subprocess.run", fake_run)
    encoded = encode_audio(
        audio_result(),
        AudioEncodingOptions(format="mp3", bitrate=256),
    )

    assert encoded.data == b"ID3-data"
    assert encoded.media_type == "audio/mpeg"
    assert captured["command"] == [
        "/opt/ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "f32le",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-i",
        "pipe:0",
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "256k",
        "-f",
        "mp3",
        "pipe:1",
    ]
    assert len(captured["kwargs"]["input"]) == 480 * 2 * 4


def test_mp3_requires_ffmpeg(monkeypatch) -> None:
    monkeypatch.setattr("mrt_local.encoding.shutil.which", lambda name: None)
    with pytest.raises(AudioEncodingError, match="brew install ffmpeg"):
        encode_audio(audio_result(), AudioEncodingOptions(format="mp3"))


def test_cli_format_inference_and_validation() -> None:
    assert infer_cli_encoding(Path("out.mp3"), None).format == "mp3"
    assert infer_cli_encoding(Path("out.wav"), "wav").format == "wav"
    with pytest.raises(ValueError, match="扩展名不一致"):
        infer_cli_encoding(Path("out.wav"), "mp3")
    with pytest.raises(ValueError, match="只适用于 MP3"):
        infer_cli_encoding(Path("out.wav"), None, 192)
