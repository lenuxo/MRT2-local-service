from __future__ import annotations

import numpy as np

from .core import AudioChunk


PCM_MEDIA_TYPE = "application/octet-stream"
PCM_SAMPLE_FORMAT = "float32le"


def encode_pcm_chunk(chunk: AudioChunk) -> bytes:
    """把交错双声道浮点样本编码为小端 float32 PCM。"""
    return np.asarray(chunk.audio, dtype="<f4").tobytes(order="C")
