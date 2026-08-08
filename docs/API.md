# MRT2 本地服务 API

本服务提供由 OpenAPI 3.1 描述的本地回环 HTTP API。启动时需要选择一个常驻模型：

```bash
./build/mrt serve --model mrt2_small
# 或
./build/mrt serve --model mrt2_base
```

默认服务地址为 `http://127.0.0.1:8765`。服务运行期间可访问：

- 中文接口说明：`GET /docs`
- 机器可读的 OpenAPI 文档：`GET /openapi.json`

## 生成音频

`POST /generate` 接收 JSON，并返回完整的 48 kHz、双声道、IEEE 浮点 WAV 文件。

```bash
curl -X POST http://127.0.0.1:8765/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"ambient synth pads","duration":8}' \
  --output ambient.wav
```

请求体字段：

| 字段 | 类型 | 必填 | 限制 |
|---|---|---:|---|
| `prompt` | string | 是 | 非空文本提示词 |
| `duration` | number | 否 | 必须 `> 0` 且 `<= 300`，默认为 `10` |

服务会拒绝未知字段。由于所选模型包含有状态的生成过程，同一时间只会执行一个推理任务。如需更换模型，请使用不同的 `--model` 参数重启服务。

JavaScript 示例：

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

## 健康检查

```http
GET /health
```

响应示例：

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

响应示例：

```json
{
  "model": "mrt2_small",
  "backend": "mlx",
  "sampleRate": 48000,
  "platform": "macos",
  "architecture": "arm64"
}
```

## 错误响应

请求参数错误返回 HTTP `400`，Content-Type 错误返回 `415`，推理失败返回 `500`。错误响应均为 JSON：

```json
{"error":"duration must be a number"}
```

## OpenAPI 文档

可以保存服务公开的规范文件，用于生成客户端或导入 API 调试工具：

```bash
curl http://127.0.0.1:8765/openapi.json --output openapi.json
```

OpenAPI 中的字段名、operation ID 和协议错误信息保留英文，以保证程序接口稳定；说明性内容使用中文。
