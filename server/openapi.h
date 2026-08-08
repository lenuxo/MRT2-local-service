#pragma once

namespace mrt_local {

inline constexpr const char* kOpenApiJson = R"OPENAPI({
  "openapi": "3.1.0",
  "info": {
    "title": "MRT2 本地服务 API",
    "version": "0.1.0",
    "description": "本地串行执行的 Magenta RealTime 2 音频生成 API。"
  },
  "servers": [{"url": "http://127.0.0.1:8765"}],
  "paths": {
    "/health": {
      "get": {
        "operationId": "getHealth",
        "responses": {"200": {"description": "服务健康状态", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Health"}}}}}
      }
    },
    "/info": {
      "get": {
        "operationId": "getInfo",
        "responses": {"200": {"description": "运行时信息", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Info"}}}}}
      }
    },
    "/generate": {
      "post": {
        "operationId": "generateAudio",
        "requestBody": {"required": true, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/GenerateRequest"}}}},
        "responses": {
          "200": {"description": "生成的 48 kHz 双声道浮点 WAV", "content": {"audio/wav": {"schema": {"type": "string", "contentEncoding": "binary"}}}},
          "400": {"description": "输入参数无效", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
          "415": {"description": "不支持的媒体类型", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
          "500": {"description": "音频生成失败", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}}
        }
      }
    },
    "/openapi.json": {"get": {"operationId": "getOpenApi", "responses": {"200": {"description": "OpenAPI 3.1 文档", "content": {"application/json": {"schema": {"type": "object"}}}}}}},
    "/docs": {"get": {"operationId": "getApiDocs", "responses": {"200": {"description": "适合用户阅读的 API 文档", "content": {"text/html": {"schema": {"type": "string"}}}}}}}
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
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>MRT2 本地服务 API</title><style>body{font:16px system-ui;max-width:760px;margin:48px auto;padding:0 20px;line-height:1.55}code,pre{background:#f3f3f3;border-radius:6px}code{padding:2px 5px}pre{padding:16px;overflow:auto}a{color:#065fd4}</style></head>
<body><h1>MRT2 本地服务 API</h1><p>OpenAPI 3.1 规范：<a href="/openapi.json">/openapi.json</a></p>
<h2>生成音频</h2><pre>curl -X POST http://127.0.0.1:8765/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"ambient techno","duration":10}' \
  --output output.wav</pre>
<h2>服务信息</h2><p><code>GET /health</code> · <code>GET /info</code></p>
<p>服务启动后会固定使用所选模型。如需使用 Base，请通过 <code>--model mrt2_base</code> 启动新的服务进程。</p></body></html>)HTML";

}  // namespace mrt_local
