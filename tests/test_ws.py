from __future__ import annotations

import asyncio
from pathlib import Path
import time

import numpy as np
from fastapi.testclient import TestClient

from mrt_local.api import create_app
from mrt_local.config import RuntimeConfig
from mrt_local.core import (
    AudioChunk,
    GenerateCommand,
    GenerateResult,
    LiveMidiQueueResult,
    ModelConfig,
    StreamExtendResult,
    StreamUpdateResult,
)
from mrt_local.encoding import encode_audio


class FakeService:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.is_loaded = False
        self.commands: list[GenerateCommand] = []
        self.stream_session = None

    def load(self) -> None:
        self.is_loaded = True

    async def load_async(self) -> None:
        self.load()

    def close(self) -> None:
        self.is_loaded = False

    def generate(self, command: GenerateCommand) -> GenerateResult:
        self.commands.append(command)
        if command.prompt == "fail":
            raise RuntimeError("backend details must not leak")
        return GenerateResult(
            48_000,
            2,
            np.zeros((round(command.duration * 48_000), 2), np.float32),
        )

    async def generate_async(self, command: GenerateCommand) -> GenerateResult:
        return self.generate(command)

    def open_stream(self, command):
        self.commands.append(command)
        self.stream_session = FakeStreamingSession(
            round(command.duration * 48_000),
            chunk_frames=command.chunk_frames,
            slow=command.prompt == "slow",
        )
        return self.stream_session

    async def open_stream_async(self, command):
        return self.open_stream(command)


class FakeStreamingSession:
    def __init__(
        self, samples: int, chunk_frames: int = 1, slow: bool = False
    ) -> None:
        self.remaining = samples
        self.generated_samples = 0
        self.slow = slow
        self.chunk_frames = chunk_frames
        self.updates = []
        self.midi_commands = []
        self.session_id = "test-stream-session"

    def next_chunk(self):
        if self.slow:
            time.sleep(0.03)
        if not self.remaining:
            return None
        count = min(self.remaining, self.chunk_frames * 1_920)
        chunk = AudioChunk(0, self.generated_samples, 48_000, 2, np.zeros((count, 2), np.float32))
        self.remaining -= count
        self.generated_samples += count
        return chunk

    async def next_chunk_async(self):
        if self.slow:
            await asyncio.sleep(0.03)
        return self.next_chunk()

    async def update_async(self, command):
        self.updates.append(command)
        return StreamUpdateResult(
            command.revision, self.generated_samples // 1_920
        )

    async def queue_live_midi_async(self, command):
        self.midi_commands.append(command)
        return LiveMidiQueueResult(
            command.event_sequence,
            self.generated_samples // 1_920,
            len(command.events),
        )

    async def extend_async(self, command):
        previous = self.generated_samples + self.remaining
        additional = round(command.additional_duration * 48_000)
        self.remaining += additional
        return StreamExtendResult(
            command.revision, previous, previous + additional
        )

    async def configure_chunk_frames_async(self, chunk_frames):
        self.chunk_frames = chunk_frames

    def close(self):
        pass

    async def close_async(self):
        self.close()


def create_test_app(tmp_path: Path):
    config = RuntimeConfig(model=ModelConfig(name="mrt2_small", root=tmp_path))
    return create_app(config, service_factory=FakeService)


def test_websocket_returns_metadata_then_binary_wav(tmp_path: Path) -> None:
    app = create_test_app(tmp_path)

    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws/generate", headers={"Origin": "https://example.com"}
        ) as websocket:
            websocket.send_json(
                {
                    "requestId": "job-1",
                    "prompt": "ambient",
                    "duration": 0.04,
                    "temperature": 0.8,
                }
            )
            metadata = websocket.receive_json()
            wav = websocket.receive_bytes()
            commands = list(app.state.service.commands)

    assert metadata == {
        "type": "result",
        "requestId": "job-1",
        "format": "wav",
        "contentType": "audio/wav",
        "byteLength": len(wav),
    }
    assert wav[:4] == b"RIFF"
    assert commands[0].prompt == "ambient"
    assert commands[0].sampling.temperature == 0.8


def test_websocket_reports_errors_and_keeps_connection_open(tmp_path: Path) -> None:
    app = create_test_app(tmp_path)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/generate") as websocket:
            websocket.send_text("not-json")
            invalid_message = websocket.receive_json()

            websocket.send_json({"requestId": "bad-1", "prompt": ""})
            validation_error = websocket.receive_json()

            websocket.send_json({"requestId": "bad-2", "prompt": "fail"})
            generation_error = websocket.receive_json()

            websocket.send_json({"requestId": "ok-1", "prompt": "recovered", "duration": 0.01})
            result = websocket.receive_json()
            wav = websocket.receive_bytes()

    assert invalid_message == {
        "type": "error",
        "code": "invalid_message",
        "message": "消息必须是 UTF-8 JSON 文本",
    }
    assert validation_error["type"] == "error"
    assert validation_error["requestId"] == "bad-1"
    assert validation_error["code"] == "validation_error"
    assert generation_error == {
        "type": "error",
        "requestId": "bad-2",
        "code": "generation_error",
        "message": "音频生成失败",
    }
    assert result["requestId"] == "ok-1"
    assert wav[:4] == b"RIFF"


def test_websocket_accepts_binary_reference_audio(tmp_path: Path) -> None:
    app = create_test_app(tmp_path)
    reference_wav = encode_audio(
        GenerateResult(48_000, 2, np.zeros((480, 2), np.float32))
    ).data

    with TestClient(app) as client:
        with client.websocket_connect("/ws/generate") as websocket:
            websocket.send_json(
                {
                    "requestId": "audio-1",
                    "inputType": "audio",
                    "prompt": "ambient",
                    "textWeight": 1,
                    "audioWeight": 3,
                    "duration": 0.01,
                }
            )
            websocket.send_bytes(reference_wav)
            metadata = websocket.receive_json()
            output = websocket.receive_bytes()
            command = app.state.service.commands[0]

    assert metadata["requestId"] == "audio-1"
    assert output[:4] == b"RIFF"
    assert command.prompt == "ambient"
    assert command.text_weight == 1
    assert command.audio_weight == 3
    assert command.reference_audio is not None
    assert command.reference_audio.samples.shape == (480, 2)


def test_websocket_accepts_control_without_prompt(tmp_path: Path) -> None:
    app = create_test_app(tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/generate") as websocket:
            websocket.send_json({
                "requestId": "notes-1",
                "duration": 0.04,
                "notes": [{"pitch": 64, "start": 0, "duration": 0.04}],
                "notesMode": "strict",
            })
            websocket.receive_json()
            websocket.receive_bytes()
            command = app.state.service.commands[0]

    assert command.prompt is None
    assert command.control.notes[0].pitch == 64
    assert command.control.notes_mode == "strict"


def test_streaming_websocket_returns_pcm_chunks(tmp_path: Path) -> None:
    app = create_test_app(tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as websocket:
            websocket.send_json({
                "type": "start", "requestId": "stream-1",
                "prompt": "ambient", "duration": 0.04, "chunkFrames": 1,
            })
            ready = websocket.receive_json()
            metadata = websocket.receive_json()
            pcm = websocket.receive_bytes()
            websocket_metrics = websocket.receive_json()
            completed = websocket.receive_json()

    assert ready["type"] == "ready"
    assert ready["sampleFormat"] == "float32le"
    assert ready["sessionId"] == "test-stream-session"
    assert ready["dynamicCapabilities"]["extendDuration"] is True
    assert ready["dynamicCapabilities"]["chunkFrames"] is True
    assert ready["dynamicCapabilities"]["realtime"] is True
    assert metadata == {
        "type": "chunk", "requestId": "stream-1", "sequence": 0,
        "sessionId": "test-stream-session",
        "frames": 1, "samplesPerChannel": 1920,
        "byteLength": len(pcm), "timestampMs": 0,
    }
    assert len(pcm) == 1_920 * 2 * 4
    assert websocket_metrics["type"] == "metrics"
    assert websocket_metrics["sessionId"] == "test-stream-session"
    assert websocket_metrics["generationTimeMs"] >= 0
    assert websocket_metrics["generatedAudioMs"] == 40
    assert websocket_metrics["realtimeFactor"] >= 0
    assert websocket_metrics["firstChunkLatencyMs"] >= 0
    assert completed["type"] == "completed"
    assert completed["reason"] == "duration_reached"


def test_streaming_websocket_can_stop_early(tmp_path: Path) -> None:
    app = create_test_app(tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as websocket:
            websocket.send_json({
                "type": "start", "requestId": "stop-1",
                "prompt": "slow", "duration": 1, "chunkFrames": 1,
            })
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"type": "stop", "requestId": "stop-1"})
            messages = []
            while True:
                message = websocket.receive_json()
                messages.append(message)
                if message["type"] == "chunk":
                    websocket.receive_bytes()
                if message["type"] == "completed":
                    break

    assert messages[-1]["reason"] == "client_stop"
    assert messages[-1]["generatedSamples"] < 48_000


def test_streaming_websocket_accepts_dynamic_updates(tmp_path: Path) -> None:
    app = create_test_app(tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as websocket:
            websocket.send_json({
                "type": "start", "requestId": "dynamic-1",
                "prompt": "slow", "duration": 0.16, "chunkFrames": 1,
            })
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({
                "type": "update",
                "requestId": "dynamic-1",
                "revision": 4,
                "prompt": "driving techno",
                "temperature": 0.8,
                "topK": 12,
                "cfgMusiccoca": 4.0,
                "notes": [{"pitch": 64, "start": 0, "duration": 0.08}],
                "drums": [{"time": 0}],
                "notesMode": "strict",
            })
            messages = []
            while True:
                message = websocket.receive_json()
                messages.append(message)
                if message["type"] == "chunk":
                    websocket.receive_bytes()
                if message["type"] == "completed":
                    break
            updates = list(app.state.service.stream_session.updates)

    accepted = next(item for item in messages if item["type"] == "updateAccepted")
    assert accepted["revision"] == 4
    assert accepted["effectiveTimestampMs"] == accepted["effectiveFrame"] * 40
    assert updates[0].prompt == "driving techno"
    assert updates[0].sampling.temperature == 0.8
    assert updates[0].sampling.top_k == 12
    assert updates[0].notes[0].pitch == 64
    assert updates[0].notes_mode == "strict"


def test_streaming_websocket_replaces_weighted_prompts(tmp_path: Path) -> None:
    app = create_test_app(tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as websocket:
            websocket.send_json({
                "type": "start", "requestId": "weighted-1",
                "promptComponents": [
                    {"text": "ambient pads", "weight": 1},
                    {"text": "soft drums", "weight": 1},
                ],
                "duration": 0.16, "chunkFrames": 1,
            })
            ready = websocket.receive_json()
            websocket.send_json({
                "type": "update", "requestId": "weighted-1", "revision": 1,
                "promptComponents": [
                    {"text": "ambient pads", "weight": 1},
                    {"text": "powerful drums", "weight": 3},
                ],
            })
            while True:
                message = websocket.receive_json()
                if message["type"] == "chunk":
                    websocket.receive_bytes()
                if message["type"] == "completed":
                    break
            command = app.state.service.commands[0]
            update = app.state.service.stream_session.updates[0]

    assert ready["dynamicCapabilities"]["promptComponents"] is True
    assert command.prompt_components[1].text == "soft drums"
    assert update.prompt_components_present is True
    assert update.prompt_components[1].weight == 3


def test_streaming_websocket_toggles_drumless(tmp_path: Path) -> None:
    app = create_test_app(tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as websocket:
            websocket.send_json({
                "type": "start", "requestId": "drumless-1",
                "prompt": "ambient", "drumless": True,
                "duration": 0.16, "chunkFrames": 1,
            })
            ready = websocket.receive_json()
            websocket.send_json({
                "type": "update", "requestId": "drumless-1",
                "revision": 1, "drumless": False,
            })
            while True:
                message = websocket.receive_json()
                if message["type"] == "chunk":
                    websocket.receive_bytes()
                if message["type"] == "completed":
                    break
            command = app.state.service.commands[0]
            update = app.state.service.stream_session.updates[0]

    assert ready["dynamicCapabilities"]["drumless"] is True
    assert command.control.drumless is True
    assert update.drumless is False


def test_streaming_websocket_replaces_reference_audio_and_weights(
    tmp_path: Path,
) -> None:
    app = create_test_app(tmp_path)
    reference_wav = encode_audio(
        GenerateResult(48_000, 2, np.zeros((480, 2), np.float32))
    ).data
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as websocket:
            websocket.send_json({
                "type": "start", "requestId": "audio-update-1",
                "prompt": "slow", "duration": 0.16, "chunkFrames": 1,
            })
            ready = websocket.receive_json()
            websocket.send_json({
                "type": "update", "requestId": "audio-update-1",
                "revision": 8, "referenceAudio": "replace",
                "textWeight": 1, "audioWeight": 3,
            })
            websocket.send_bytes(reference_wav)
            messages = []
            while True:
                message = websocket.receive_json()
                messages.append(message)
                if message["type"] == "chunk":
                    websocket.receive_bytes()
                if message["type"] == "completed":
                    break
            updates = app.state.service.stream_session.updates

    assert ready["dynamicCapabilities"]["referenceAudio"] is True
    assert ready["dynamicCapabilities"]["styleWeights"] is True
    assert any(item["type"] == "updateAccepted" for item in messages)
    assert updates[0].reference_audio_present is True
    assert updates[0].reference_audio.samples.shape == (480, 2)
    assert updates[0].text_weight == 1
    assert updates[0].audio_weight == 3


def test_streaming_websocket_can_extend_running_session(tmp_path: Path) -> None:
    app = create_test_app(tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as websocket:
            websocket.send_json({
                "type": "start", "requestId": "extend-1",
                "prompt": "slow", "duration": 0.08, "chunkFrames": 1,
            })
            ready = websocket.receive_json()
            websocket.send_json({
                "type": "extend", "requestId": "extend-1",
                "revision": 5, "additionalDuration": 0.08,
            })
            messages = []
            while True:
                message = websocket.receive_json()
                messages.append(message)
                if message["type"] == "chunk":
                    websocket.receive_bytes()
                if message["type"] == "completed":
                    break

    extended = next(item for item in messages if item["type"] == "extended")
    assert ready["dynamicCapabilities"]["protocolVersion"] == 4
    assert extended["type"] == "extended"
    assert extended["requestId"] == "extend-1"
    assert extended["revision"] == 5
    assert extended["previousDurationMs"] == 80
    assert extended["durationMs"] == 160
    assert extended["controlSequence"] == 0
    assert messages[-1]["generatedSamples"] == 7_680


def test_streaming_websocket_can_reconfigure_transport(tmp_path: Path) -> None:
    app = create_test_app(tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as websocket:
            websocket.send_json({
                "type": "start", "requestId": "config-1",
                "prompt": "slow", "duration": 0.24, "chunkFrames": 1,
            })
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({
                "type": "configure", "requestId": "config-1",
                "revision": 6, "chunkFrames": 2, "realtime": False,
            })
            messages = []
            while True:
                message = websocket.receive_json()
                messages.append(message)
                if message["type"] == "chunk":
                    websocket.receive_bytes()
                if message["type"] == "completed":
                    break

    configured = next(item for item in messages if item["type"] == "configured")
    assert configured["revision"] == 6
    assert configured["chunkFrames"] == 2
    assert configured["realtime"] is False
    assert any(
        item.get("type") == "chunk" and item["frames"] == 2
        for item in messages
    )


def test_streaming_websocket_revisions_are_idempotent_and_ordered(
    tmp_path: Path,
) -> None:
    app = create_test_app(tmp_path)
    update = {
        "type": "update", "requestId": "revision-1",
        "revision": 2, "temperature": 0.8,
    }
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as websocket:
            websocket.send_json({
                "type": "start", "requestId": "revision-1",
                "prompt": "slow", "duration": 0.4, "chunkFrames": 1,
            })
            websocket.receive_json()
            websocket.send_json(update)
            websocket.send_json(update)
            websocket.send_json({**update, "temperature": 0.9})
            websocket.send_json({**update, "revision": 1})
            messages = []
            while True:
                message = websocket.receive_json()
                messages.append(message)
                if message["type"] == "chunk":
                    websocket.receive_bytes()
                if message["type"] == "completed":
                    break
            applied_updates = list(
                app.state.service.stream_session.updates
            )

    accepted = [item for item in messages if item["type"] == "updateAccepted"]
    errors = [item for item in messages if item["type"] == "error"]
    assert len(applied_updates) == 1
    assert len(accepted) == 2
    assert accepted[0]["controlSequence"] == 0
    assert accepted[1]["controlSequence"] == 1
    assert accepted[1]["duplicate"] is True
    assert {item["code"] for item in errors} == {
        "revision_conflict", "stale_revision"
    }


def test_streaming_websocket_accepts_live_midi_events(tmp_path: Path) -> None:
    app = create_test_app(tmp_path)
    message = {
        "type": "midi",
        "requestId": "midi-1",
        "eventSequence": 7,
        "events": [
            {"kind": "noteOn", "channel": 0, "pitch": 60, "velocity": 96},
            {"kind": "noteOff", "channel": 0, "pitch": 60, "velocity": 0},
            {"kind": "controlChange", "channel": 0, "controller": 64, "value": 0},
        ],
    }
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as websocket:
            websocket.send_json({
                "type": "start", "requestId": "midi-1",
                "midiMode": "live", "liveNotesMode": "strict",
                "prompt": "slow", "duration": 0.4, "chunkFrames": 1,
            })
            ready = websocket.receive_json()
            websocket.send_json(message)
            websocket.send_json(message)
            messages = []
            while True:
                received = websocket.receive_json()
                messages.append(received)
                if received["type"] == "chunk":
                    websocket.receive_bytes()
                if received["type"] == "completed":
                    break
            commands = app.state.service.stream_session.midi_commands

    queued = [item for item in messages if item["type"] == "midiQueued"]
    assert ready["midiMode"] == "live"
    assert ready["liveNotesMode"] == "strict"
    assert ready["dynamicCapabilities"]["liveMidi"] is True
    assert len(commands) == 1
    assert commands[0].events[0].kind == "note_on"
    assert len(queued) == 2
    assert queued[0]["eventSequence"] == 7
    assert queued[0]["acceptedEvents"] == 3
    assert queued[1]["duplicate"] is True


def test_streaming_websocket_midi_sequence_is_independent_from_revision(
    tmp_path: Path,
) -> None:
    app = create_test_app(tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as websocket:
            websocket.send_json({
                "type": "start", "requestId": "midi-seq",
                "midiMode": "live", "prompt": "slow",
                "duration": 0.4, "chunkFrames": 1,
            })
            websocket.receive_json()
            websocket.send_json({
                "type": "update", "requestId": "midi-seq",
                "revision": 0, "temperature": 0.8,
            })
            websocket.send_json({
                "type": "midi", "requestId": "midi-seq",
                "eventSequence": 0,
                "events": [{"kind": "panic"}],
            })
            messages = []
            while True:
                received = websocket.receive_json()
                messages.append(received)
                if received["type"] == "chunk":
                    websocket.receive_bytes()
                if received["type"] == "completed":
                    break

    assert any(item["type"] == "updateAccepted" for item in messages)
    assert any(item["type"] == "midiQueued" for item in messages)


def test_streaming_websocket_rejects_midi_events_in_plan_mode(
    tmp_path: Path,
) -> None:
    app = create_test_app(tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as websocket:
            websocket.send_json({
                "type": "start", "requestId": "midi-plan",
                "prompt": "slow", "duration": 0.16, "chunkFrames": 1,
            })
            websocket.receive_json()
            websocket.send_json({
                "type": "midi", "requestId": "midi-plan",
                "eventSequence": 0,
                "events": [{"kind": "panic"}],
            })
            messages = []
            while True:
                received = websocket.receive_json()
                messages.append(received)
                if received["type"] == "chunk":
                    websocket.receive_bytes()
                if received["type"] == "completed":
                    break

    error = next(item for item in messages if item["type"] == "error")
    assert error["code"] == "control_validation_error"
    assert "midiMode=live" in error["message"]


def test_streaming_websocket_rejects_oversized_reference_audio(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        "mrt_local.streaming_ws.MAX_REFERENCE_AUDIO_BYTES", 8
    )
    app = create_test_app(tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as websocket:
            websocket.send_json({
                "type": "start", "requestId": "limit-1",
                "prompt": "slow", "duration": 0.12, "chunkFrames": 1,
            })
            websocket.receive_json()
            websocket.send_json({
                "type": "update", "requestId": "limit-1", "revision": 1,
                "referenceAudio": "replace",
            })
            websocket.send_bytes(b"too-large")
            messages = []
            while True:
                message = websocket.receive_json()
                messages.append(message)
                if message["type"] == "chunk":
                    websocket.receive_bytes()
                if message["type"] == "completed":
                    break

    error = next(item for item in messages if item["type"] == "error")
    assert error["code"] == "reference_audio_too_large"
    assert messages[-1]["type"] == "completed"
