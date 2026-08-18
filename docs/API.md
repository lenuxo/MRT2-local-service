# MRT2 本地服务 API

API 基于 FastAPI 实现，并自动生成 OpenAPI 3.1 规范。

```bash
uv run mrt-serve --model mrt2_small
# 或
uv run mrt-serve --model mrt2_base
```

默认地址为 `http://127.0.0.1:8765`：

- `GET /docs`：Swagger UI
- `GET /openapi.json`：OpenAPI JSON
- `GET /health`：健康状态
- `GET /info`：当前模型与运行环境
- `POST /generate`：生成 WAV

## 生成音频

```bash
curl -X POST http://127.0.0.1:8765/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"ambient synth pads","duration":8}' \
  --output ambient.wav
```

请求体：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `prompt` | string | 是 | 非空文本提示词 |
| `duration` | number | 否 | 秒数，必须 `> 0` 且 `<= 300`，默认 `10` |

成功响应为 `audio/wav`，内容是 48 kHz 双声道 IEEE float WAV。未知字段、空 prompt 和非法 duration 由 FastAPI/Pydantic 返回 HTTP `422`。

JavaScript 示例：

```js
const response = await fetch("http://127.0.0.1:8765/generate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ prompt: "ambient techno", duration: 10 }),
});

if (!response.ok) throw new Error(await response.text());
const wav = await response.arrayBuffer();
```

## 健康检查

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

## 运行信息

```http
GET /info
```

```json
{
  "model": "mrt2_small",
  "backend": "mlx",
  "sampleRate": 48000,
  "channels": 2,
  "platform": "macos",
  "architecture": "arm64"
}
```

## 错误

- `422`：请求格式或字段验证失败
- `400`：Engine 业务参数验证失败
- `500`：模型推理失败

模型在服务启动时加载一次。`POST /generate` 不能临时切换模型；如需使用另一个模型，请通过 `--model` 重启服务。
