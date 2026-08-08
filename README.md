# MRT2 Local Service

MRT2 Local Service is a macOS Apple Silicon service that exposes Magenta RealTime 2 Small or Base through one command-line binary and a loopback HTTP API. Both entry points use the same `MrtEngine` wrapper around Google's official `magentart::core::MLXEngine`; the server loads the selected model once and serializes generation on the shared stateful engine.

## Requirements

- Apple Silicon Mac running macOS 14 or newer
- Xcode Command Line Tools
- Xcode's Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain` if it is missing)
- CMake 3.27 or newer (the upstream project recommends CMake `<3.28` if newer versions cause dependency issues)
- Git and an internet connection for the first CMake configure
- MRT2 Small or Base MLX model plus the shared MusicCoCa resources

Only the MLX backend, MRT2 Small, text prompting, finite-duration generation, and 48 kHz stereo float WAV output are in scope.

## Prepare the model

The supported preparation path is the official `magenta-rt` package:

Install the official downloader and use the project script. Downloads stay inside this repository:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install "magenta-rt[mlx]"

./scripts/download_models.sh mrt2_small
./scripts/download_models.sh mrt2_base
# Download both in one invocation:
./scripts/download_models.sh mrt2_small mrt2_base
```

The resulting layout is:

```text
./models/
├── models/
│   ├── mrt2_small/mrt2_small.mlxfn
│   └── mrt2_base/mrt2_base.mlxfn
└── resources/musiccoca/...
```

Downloaded weights are excluded from Git. Override paths when needed:

```bash
export MRT_MODEL_PATH=/absolute/path/to/mrt2_small.mlxfn
export MRT_RESOURCES_PATH=/absolute/path/to/resources
export MRT_MODEL_ROOT=/absolute/path/to/download-root
```

Path precedence is command option, environment variable, then the default above.

## Build

The first configure fetches pinned versions of the official Magenta source and its MLX, TensorFlow Lite, and SentencePiece dependencies, so it can take a while:

```bash
cmake -B build
cmake --build build --target mrt -j
./build/mrt --help
```

To reuse an existing official checkout and avoid fetching Magenta itself:

```bash
git clone --recurse-submodules https://github.com/magenta/magenta-realtime.git
cmake -B build -DMAGENTART_SOURCE_DIR=/absolute/path/to/magenta-realtime
cmake --build build --target mrt -j
```

Install wherever appropriate for your machine (installation may require elevated permissions):

```bash
cmake --install build --prefix /usr/local
```

## CLI

Generate a WAV file:

```bash
./build/mrt generate \
  --prompt "minimal techno" \
  --model mrt2_small \
  --duration 5 \
  --output test.wav
```

`--model` accepts `mrt2_small` or `mrt2_base`. Optional overrides are `--model-path` and `--resources-path`; generation defaults are 10 seconds and `./output.wav`.

Inspect resolved configuration without loading the model:

```bash
./build/mrt info
```

## Local HTTP API

Start the resident service (the model is loaded before the socket starts listening):

```bash
./build/mrt serve --model mrt2_small
```

It binds to `127.0.0.1:8765` by default. A different address or port must be explicit:

```bash
./build/mrt serve --host 127.0.0.1 --port 9000
```

Health and model information:

```bash
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/info
```

Interactive documentation and the OpenAPI 3.1 description are available while the server runs:

```text
http://127.0.0.1:8765/docs
http://127.0.0.1:8765/openapi.json
```

See [docs/API.md](docs/API.md) for request fields, responses, JavaScript usage, and errors.

Generate audio:

```bash
curl -X POST http://127.0.0.1:8765/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"minimal techno","duration":5}' \
  --output output.wav
```

`POST /generate` returns `audio/wav`. Duration defaults to 10 seconds, must be positive, and is capped at 300 seconds. Requests are processed through an engine mutex because `MLXEngine` generation state is not safe for concurrent lifecycle calls.

## Tests

After configuring the complete project:

```bash
cmake --build build --target wav_test
ctest --test-dir build --output-on-failure
```

The WAV unit test does not require a model. End-to-end generation requires a real MRT2 Small or Base model and resources; a synthetic fallback is intentionally not provided.

## Current limitations

- macOS Apple Silicon only
- MRT2 Small/Base and MLX only
- One in-process model and serialized generation
- No streaming, WebSocket, MIDI, OSC, playback, GUI, plug-in, or authentication
- The server returns a complete WAV after generation rather than progressive audio

The inference lifecycle is based on the official `examples/hello_mrt2` implementation at pinned Magenta commit `694a545e4ba0b88bf1150137b129582166d3e07f`.
