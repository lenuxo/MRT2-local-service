from __future__ import annotations

STREAM_PROTOCOL_VERSION = 3
MAX_CONTROL_MESSAGE_BYTES = 64 * 1024
MAX_REFERENCE_AUDIO_BYTES = 64 * 1024 * 1024
REFERENCE_AUDIO_TIMEOUT_SECONDS = 10.0

DYNAMIC_UPDATE_FIELDS = (
    "prompt",
    "temperature",
    "topK",
    "cfgMusiccoca",
    "cfgNotes",
    "cfgDrums",
    "seed",
    "useMapper",
    "poolAcrossTime",
    "notes",
    "drums",
    "notesMode",
    "drumsMode",
    "referenceAudio",
    "textWeight",
    "audioWeight",
)


def stream_capabilities() -> dict[str, object]:
    return {
        "protocolVersion": STREAM_PROTOCOL_VERSION,
        "update": list(DYNAMIC_UPDATE_FIELDS),
        "effectiveFrame": True,
        "extendDuration": True,
        "chunkFrames": True,
        "realtime": True,
        "referenceAudio": True,
        "styleWeights": True,
        "metrics": True,
        "revisionPolicy": "strictly_increasing_idempotent_replay",
        "limits": {
            "controlMessageBytes": MAX_CONTROL_MESSAGE_BYTES,
            "referenceAudioBytes": MAX_REFERENCE_AUDIO_BYTES,
            "referenceAudioTimeoutSeconds": REFERENCE_AUDIO_TIMEOUT_SECONDS,
        },
    }
