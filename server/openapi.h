#pragma once

namespace mrt_local {

inline constexpr const char* kOpenApiJson = R"OPENAPI({
  "openapi": "3.1.0",
  "info": {
    "title": "MRT2 Local Service API",
    "version": "0.1.0",
    "description": "Local, serialized Magenta RealTime 2 audio generation API."
  },
  "servers": [{"url": "http://127.0.0.1:8765"}],
  "paths": {
    "/health": {
      "get": {
        "operationId": "getHealth",
        "responses": {"200": {"description": "Service health", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Health"}}}}}
      }
    },
    "/info": {
      "get": {
        "operationId": "getInfo",
        "responses": {"200": {"description": "Runtime information", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Info"}}}}}
      }
    },
    "/generate": {
      "post": {
        "operationId": "generateAudio",
        "requestBody": {"required": true, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/GenerateRequest"}}}},
        "responses": {
          "200": {"description": "Generated 48 kHz stereo float WAV", "content": {"audio/wav": {"schema": {"type": "string", "contentEncoding": "binary"}}}},
          "400": {"description": "Invalid input", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
          "415": {"description": "Unsupported media type", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
          "500": {"description": "Generation failed", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}}
        }
      }
    },
    "/openapi.json": {"get": {"operationId": "getOpenApi", "responses": {"200": {"description": "OpenAPI 3.1 document", "content": {"application/json": {"schema": {"type": "object"}}}}}}},
    "/docs": {"get": {"operationId": "getApiDocs", "responses": {"200": {"description": "Human-readable API documentation", "content": {"text/html": {"schema": {"type": "string"}}}}}}}
  },
  "components": {
    "schemas": {
      "GenerateRequest": {"type": "object", "additionalProperties": false, "required": ["prompt"], "properties": {"prompt": {"type": "string", "minLength": 1}, "duration": {"type": "number", "exclusiveMinimum": 0, "maximum": 300, "default": 10}}},
      "Health": {"type": "object", "required": ["status", "model", "loaded"], "properties": {"status": {"const": "ok"}, "model": {"enum": ["mrt2_small", "mrt2_base"]}, "loaded": {"type": "boolean"}}},
      "Info": {"type": "object", "required": ["model", "backend", "sampleRate", "platform", "architecture"], "properties": {"model": {"enum": ["mrt2_small", "mrt2_base"]}, "backend": {"const": "mlx"}, "sampleRate": {"const": 48000}, "platform": {"const": "macos"}, "architecture": {"const": "arm64"}}},
      "Error": {"type": "object", "required": ["error"], "properties": {"error": {"type": "string"}}}
    }
  }
})OPENAPI";

inline constexpr const char* kApiDocsHtml = R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>MRT2 Local API</title><style>body{font:16px system-ui;max-width:760px;margin:48px auto;padding:0 20px;line-height:1.55}code,pre{background:#f3f3f3;border-radius:6px}code{padding:2px 5px}pre{padding:16px;overflow:auto}a{color:#065fd4}</style></head>
<body><h1>MRT2 Local API</h1><p>OpenAPI 3.1: <a href="/openapi.json">/openapi.json</a></p>
<h2>Generate audio</h2><pre>curl -X POST http://127.0.0.1:8765/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"ambient techno","duration":10}' \
  --output output.wav</pre>
<h2>Service metadata</h2><p><code>GET /health</code> · <code>GET /info</code></p>
<p>The selected model is fixed when the service starts. Start another process with <code>--model mrt2_base</code> to use Base.</p></body></html>)HTML";

}  // namespace mrt_local
