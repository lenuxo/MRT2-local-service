# MRT2 Local Service API

The service implements an OpenAPI 3.1-described loopback HTTP API. Start it with one resident model:

```bash
./build/mrt serve --model mrt2_small
# or
./build/mrt serve --model mrt2_base
```

The default endpoint is `http://127.0.0.1:8765`. While running:

- Human-readable documentation: `GET /docs`
- Machine-readable OpenAPI document: `GET /openapi.json`

## Generate audio

`POST /generate` accepts JSON and returns a complete 48 kHz, stereo, IEEE float WAV file.

```bash
curl -X POST http://127.0.0.1:8765/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"ambient synth pads","duration":8}' \
  --output ambient.wav
```

Request body:

| Field | Type | Required | Constraints |
|---|---|---:|---|
| `prompt` | string | yes | Non-empty text prompt |
| `duration` | number | no | `> 0` and `<= 300`; default `10` |

The server rejects unknown request fields. Only one inference runs at a time because the selected model has stateful generation. To change model, restart the service with a different `--model` value.

Example JavaScript:

```js
const response = await fetch("http://127.0.0.1:8765/generate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ prompt: "ambient techno", duration: 10 }),
});

if (!response.ok) {
  throw new Error(await response.text());
}

const wav = await response.arrayBuffer();
```

## Health

```http
GET /health
```

```json
{
  "status": "ok",
  "model": "mrt2_small",
  "loaded": true
}
```

## Runtime information

```http
GET /info
```

```json
{
  "model": "mrt2_small",
  "backend": "mlx",
  "sampleRate": 48000,
  "platform": "macos",
  "architecture": "arm64"
}
```

## Errors

Validation errors use HTTP `400`, an incorrect content type uses `415`, and inference failures use `500`. Error responses are JSON:

```json
{"error":"duration must be a number"}
```
