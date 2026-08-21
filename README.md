# MRT2 Local Service (Python)

English | [简体中文](README.zh-CN.md)

A streaming-first Magenta RealTime 2 service for macOS on Apple Silicon, implemented entirely in Python. Stateful real-time music generation is the primary feature; complete WAV/MP3 generation is a convenience layer around the same core:

- Runs inference with Magenta's official `magenta-rt[mlx]` package
- Provides complete-file and stateful PCM streaming APIs over HTTP and WebSocket
- Supports live text/reference-audio blending, sampling, CFG, note, and drum updates during WebSocket streaming
- Supports advanced weighted multi-prompt style blending, including atomic live replacement over WebSocket
- Reports per-chunk latency, real-time factor, buffer lead, capabilities, and active-session status
- Provides direct CLI generation and a persistent local service
- Accepts MIDI files or JSON note/drum events as time-varying model controls
- Accepts incremental Note On/Off, sustain-pedal, panic, and channel-10 drum events for live Web MIDI performance
- Exposes MRT2's official `drumless` condition in CLI, HTTP, and live WebSocket updates
- Shares one `GenerationService` and protocol-independent command models across all transports
- Supports `mrt2_small` and `mrt2_base`
- Stores models and shared resources in the project-local `models/` directory
- Allows browser HTTP and WebSocket access from any origin by default

## Requirements

- Apple Silicon Mac
- macOS 14 or later
- Python 3.11 or 3.12
- `uv` recommended
- FFmpeg only for MP3 output; WAV requires no FFmpeg installation

CMake, a C++ compiler, and a separate MLX C++ build are not required.

### Model hardware guidance

The hardware distinction mainly determines whether generation keeps up with playback in real time. Both models can run offline on Apple Silicon; a ❌ below means the official benchmark does not classify that model/device pair as real-time, not that the model cannot run.

| Apple Silicon device | `mrt2_small` (230M) | `mrt2_base` (2.4B) |
|---|---:|---:|
| M5 Max | Real-time | Real-time |
| M3 Max | Real-time | Real-time |
| M2 Max | Real-time | Real-time |
| M4 Pro | Real-time | Real-time |
| M2 Pro / M1 Pro | Real-time | Offline only |
| M4 Air / M3 Air / M1 Air | Real-time | Offline only |

Use `mrt2_small` as the default on Air and most Pro machines. Choose `mrt2_base` for higher quality when slower-than-playback generation is acceptable, or for official real-time support on the listed M4 Pro and Max devices. Devices absent from the official table have no documented real-time guarantee. See [Models and inference parameters](docs/MODELS.md) for details.

Install FFmpeg if you need MP3 output:

```bash
brew install ffmpeg
```

## UV environment management

UV is the recommended tool for Python, dependency, and command management. The Python version is recorded in `.python-version`, while exact dependency versions are recorded in `uv.lock`.

```bash
uv sync --extra dev
```

UV creates `.venv/` in the project root. Run subsequent commands through `uv run`; manual activation is unnecessary, and UV verifies that the environment matches the lockfile.

The project pins MLX `0.31.2` to match the current Magenta RealTime lockfile. Do not upgrade MLX independently: incompatible newer versions cause `[import_function] Invalid string size` while loading the official `.mlxfn` models. Restore a modified environment with:

```bash
uv sync --extra dev
```

View CLI help:

```bash
uv run mrt-local -h
uv run mrt-local generate -h
uv run mrt-download -h
uv run mrt-serve -h
```

## Download models

Download Small by default:

```bash
uv run mrt-download
```

Select one or both models:

```bash
uv run mrt-download mrt2_small
uv run mrt-download mrt2_base
uv run mrt-download mrt2_small mrt2_base
```

All downloaded files remain inside the project:

```text
models/
├── models/
│   ├── mrt2_small/
│   │   ├── mrt2_small.mlxfn
│   │   └── mrt2_small_state.safetensors
│   └── mrt2_base/
│       ├── mrt2_base.mlxfn
│       └── mrt2_base_state.safetensors
└── resources/
    ├── musiccoca/
    └── spectrostream/
```

The default download root is the project's `models/` directory, not Documents, Home, or a global cache. Downloaded content is excluded by `.gitignore`.

Use another location only when explicitly requested:

```bash
uv run mrt-download mrt2_small --model-root /absolute/custom/path
```

## CLI generation

Generate WAV with Small:

```bash
uv run mrt-local generate \
  --model mrt2_small \
  --prompt "minimal techno" \
  --duration 5 \
  --output output.wav
```

Generate with Base:

```bash
uv run mrt-local generate --model mrt2_base --prompt "ambient pads"
```

Advanced users can blend several complete style descriptions with relative weights:

```bash
uv run mrt-local generate \
  --weighted-prompt 1 "spacious ambient pads" \
  --weighted-prompt 2 "powerful acoustic drums" \
  --weighted-prompt 0.5 "subtle analog bass" \
  --output weighted.wav
```

This blends whole MusicCoCa embeddings; it is not exact word/token weighting. See
[Prompts and advanced weighted style blending](docs/PROMPTS.md) for limits and API examples.

Ask the model to avoid drums with the official drum condition:

```bash
uv run mrt-local generate --prompt "slow ambient strings" --drumless
```

`drumless` is a generation condition rather than post-processing, and is mutually exclusive with explicit drum events. See [MIDI and event controls](docs/CONTROL.md) for the complete precedence rules.

Use a reference audio file as the MusicCoCa style condition:

```bash
uv run mrt-local generate \
  --model mrt2_small \
  --reference-audio reference.wav \
  --duration 10 \
  --output styled.wav
```

Text and reference audio can be used separately or together. When both are present,
their MusicCoCa embeddings are blended with normalized weights (default `0.5/0.5`):

```bash
uv run mrt-local generate \
  --prompt "ambient pads" \
  --reference-audio reference.wav \
  --text-weight 1 --audio-weight 3 \
  --output mixed.wav
```

Reference audio controls style; it is not treated as audio continuation or editing input.

Control melody and rhythm with a Standard MIDI File. MIDI channel 10 becomes drum
triggers; note events on other channels become pitched-note controls:

```bash
uv run mrt-local generate \
  --midi arrangement.mid \
  --prompt "warm analog synths" \
  --duration 12 \
  --output controlled.wav
```

`--notes-mode guide` and `--drums-mode guide` let the model add unspecified
material. Use `strict` to mark unspecified pitches or drum frames as off. MIDI can
also be the only input; text and reference audio remain optional style controls.

Generate MP3. The format is inferred from the output extension by default, or it can be specified explicitly:

```bash
uv run mrt-local generate \
  --prompt "ambient pads" \
  --output output.mp3 \
  --format mp3 \
  --bitrate 192
```

Override official MLX inference parameters; less common parameters have defaults:

```bash
uv run mrt-local generate \
  --prompt "ambient techno" \
  --temperature 1.1 \
  --top-k 40 \
  --cfg-musiccoca 3.0
```

Inspect the resolved configuration without loading a model:

```bash
uv run mrt-local info --model mrt2_base
```

## Start the service

```bash
uv run mrt-serve --model mrt2_small
```

Choose another port when needed; omitting `--port` keeps the default `8765`:

```bash
uv run mrt-serve --model mrt2_small --port 9000
# Equivalent unified CLI command:
uv run mrt-local serve --model mrt2_small --port 9000
```

The service listens on `127.0.0.1:8765` by default. FastAPI loads and warms the model once during startup. All requests share that instance. One complete generation or streaming session owns the model at a time; overlapping HTTP requests receive `409 Conflict`, and WebSocket requests receive `model_busy`.

Browser cross-origin access is unrestricted by default. HTTP accepts any origin, method, and request header without credentialed Cookie mode; WebSocket origins are not filtered. This does not change network reachability: the default `127.0.0.1` listener remains local-only.

Endpoints:

- Swagger UI: <http://127.0.0.1:8765/docs>
- OpenAPI JSON: <http://127.0.0.1:8765/openapi.json>
- Health check: <http://127.0.0.1:8765/health>
- Runtime information: <http://127.0.0.1:8765/info>
- Capabilities: <http://127.0.0.1:8765/v1/capabilities>
- Active-session status: <http://127.0.0.1:8765/v1/status>
- WebSocket: `ws://127.0.0.1:8765/ws/generate`
- Streaming WebSocket: `ws://127.0.0.1:8765/ws/stream`

Generate WAV over HTTP:

```bash
curl -X POST http://127.0.0.1:8765/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"minimal techno","duration":5}' \
  --output output.wav
```

Generate MP3 over HTTP:

```bash
curl -X POST http://127.0.0.1:8765/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"minimal techno","duration":5,"format":"mp3","bitrate":192}' \
  --output output.mp3
```

Generate from an uploaded reference audio file:

```bash
curl -X POST http://127.0.0.1:8765/generate/audio \
  -F 'audio=@reference.wav' \
  -F 'duration=10' \
  -F 'format=wav' \
  --output styled.wav
```

Stream raw 48 kHz stereo float32le PCM over HTTP:

```bash
curl --no-buffer -X POST http://127.0.0.1:8765/stream \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"ambient pads","duration":10,"chunk_frames":5}' \
  --output output.f32le
```

For interactive generation, use `ws://127.0.0.1:8765/ws/stream`. While audio is
being generated, send `update` messages to change the prompt, temperature,
reference audio, blend weights, top-k, CFG values, notes, or drums without resetting the model state. See
[HTTP and WebSocket streaming](docs/STREAMING.md) for the live-control protocol.
For keyboard performance, start the stream with `midiMode: "live"` and send
incremental `midi` messages; see [real-time MIDI performance](docs/LIVE_MIDI.md).
WebSocket streams are paced in real time by default so the model stays close to
the playback position. A running client can also extend the session or change
`chunkFrames` and `realtime` without restarting it.

Detailed reference material is currently available in Chinese:

- [HTTP API](docs/API.md)
- [WebSocket protocol](docs/WEBSOCKET.md)
- [HTTP and WebSocket streaming](docs/STREAMING.md)
- [Models and inference parameters](docs/MODELS.md)
- [Prompts and advanced weighted style blending](docs/PROMPTS.md)
- [MIDI and event controls](docs/CONTROL.md)
- [Real-time MIDI performance](docs/LIVE_MIDI.md)
- [Architecture](docs/ARCHITECTURE.md)

## Tests

Unit tests use fake backends to verify lifecycle behavior, exact-duration audio, CLI, HTTP, WebSocket, stateful streaming, cancellation, encoding, and OpenAPI without requiring downloaded models:

```bash
uv run pytest
```

Real end-to-end tests require downloading a model before running the CLI or service.

## Project structure

```text
.
├── mrt_local/
│   ├── api.py                # HTTP transport and OpenAPI
│   ├── capabilities.py       # Streaming protocol capabilities and limits
│   ├── ws.py                 # WebSocket transport
│   ├── streaming_ws.py       # Stateful PCM streaming WebSocket
│   ├── pcm.py                # Raw PCM serialization
│   ├── schemas.py            # Shared HTTP/WebSocket request models
│   ├── cli.py                # CLI transport
│   ├── core.py               # Core commands, configuration, validation, and results
│   ├── config.py             # Runtime configuration and default paths
│   ├── backend.py            # Backend port and Magenta/MLX adapter
│   ├── encoding.py           # Shared WAV/MP3 encoding
│   ├── midi.py               # Standard MIDI File decoder
│   ├── service.py            # Transport-independent generation use case
│   └── download.py           # Model download command
├── tests/
├── docs/
│   ├── API.md                # API usage
│   ├── ARCHITECTURE.md       # Layering and extension guide
│   ├── MODELS.md             # Models, hardware, and inference parameters
│   ├── PROMPTS.md            # Prompt modes and weighted style blending
│   ├── CONTROL.md            # MIDI and JSON control events
│   ├── LIVE_MIDI.md          # Incremental Web MIDI performance protocol
│   ├── WEBSOCKET.md          # WebSocket message protocol
│   └── STREAMING.md          # HTTP/WebSocket PCM streaming protocol
├── pyproject.toml            # UV project configuration and commands
└── uv.lock                   # Fully resolved dependency lockfile
```

## Current limitations

- macOS on Apple Silicon with MLX only
- One fixed model per service process; restart the service to switch models
- One active generation or streaming session at a time; no multi-model concurrency
- WebSocket streaming supports live MIDI performance, text/reference-audio blending, condition updates, duration extension, and transport reconfiguration; OSC, a bundled player, and GUI are not yet included

The locked environment currently uses `magenta-rt 2.0.3` and its `MagentaRT2StdMlxfn`, `embed_style()`, and stateful `generate()` APIs.
