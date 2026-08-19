# WebSocket API

WebSocket 接口适合在一个长连接上连续提交多个生成任务。它与 HTTP `POST /generate` 共用同一个 `GenerationService`、参数模型和推理实例。

## 连接地址

启动服务：

```bash
uv run mrt-serve --model mrt2_small
```

连接：

```text
ws://127.0.0.1:8765/ws/generate
```

WebSocket 不是 OpenAPI 规范的一部分，因此不会出现在 `/openapi.json` 或 Swagger UI 的路径列表中。本页是该接口的协议文档。

## 请求

客户端每次发送一个 UTF-8 JSON 文本消息。字段与 HTTP `POST /generate` 相同，另外支持可选的 `requestId`：

```json
{
  "requestId": "job-001",
  "prompt": "minimal techno, deep bass",
  "duration": 8,
  "temperature": 1.1,
  "top_k": 40,
  "cfg_musiccoca": 3.0,
  "cfg_notes": 1.0,
  "cfg_drums": 1.0,
  "seed": 0,
  "use_mapper": true,
  "pool_across_time": true,
  "format": "mp3",
  "bitrate": 192
}
```

### 参考音频输入

参考音频任务先发送一条 JSON 文本消息，并把 `inputType` 设为 `audio`；随后立即发送一个包含完整参考音频文件的二进制消息：

```json
{
  "requestId": "audio-job-001",
  "inputType": "audio",
  "duration": 10,
  "format": "wav"
}
```

下一条 WebSocket 消息：

```text
Binary(reference.wav 的完整文件内容)
```

文本任务可省略 `inputType`（默认 `text`），并必须提供 `prompt`；音频任务不能提供 `prompt`。服务解码 WAV、FLAC、OGG、MP3 等常见格式，参考音频最长 300 秒。

除 `prompt` 外，其余生成参数都有默认值。完整含义见 [模型与推理参数](MODELS.md)。

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `requestId` | string | 否 | 1～128 个字符；原样出现在对应的结果或错误消息中 |
| `inputType` | `text`/`audio` | 否 | 输入类型，默认 `text` |
| `prompt` | string | 是 | 非空文本提示词 |
| `duration` | number | 否 | 生成秒数，默认 `10`，范围 `(0, 300]` |
| `format` | `wav`/`mp3` | 否 | 输出格式，默认 `wav` |
| `bitrate` | integer | 否 | MP3 比特率，32～320 kbps，默认 `192`；不适用于 WAV |
| 其他生成字段 | 见 HTTP API | 否 | 与 `POST /generate` 完全相同 |

当 `inputType=audio` 时，`prompt` 不适用，紧随其后的二进制消息是必需的。

一个连接可以连续发送多次请求。当前版本按收到顺序逐个处理同一连接中的任务，不支持在同一连接内并行或取消任务。

## 成功响应

服务依次发送两个 WebSocket 消息。

第一条是结果元数据 JSON 文本：

```json
{
  "type": "result",
  "requestId": "job-001",
  "format": "mp3",
  "contentType": "audio/mpeg",
  "byteLength": 3072044
}
```

第二条是一个二进制消息，其内容为完整 WAV 或 MP3。`byteLength` 是这个二进制消息的字节数，`contentType` 是对应的 `audio/wav` 或 `audio/mpeg`。客户端应读取完整二进制消息，而不是把它当作分块流式 PCM。

## 错误响应

请求错误以 JSON 文本返回，不会主动关闭连接：

```json
{
  "type": "error",
  "requestId": "job-001",
  "code": "validation_error",
  "message": "生成参数验证失败",
  "details": []
}
```

错误码：

| `code` | 含义 |
|---|---|
| `invalid_message` | 消息不是合法的 UTF-8 JSON 文本 |
| `validation_error` | JSON 结构或生成参数不合法 |
| `encoding_error` | FFmpeg 缺失或 MP3 编码失败 |
| `generation_error` | 后端生成失败；内部异常细节不会发送给客户端 |

## 浏览器示例

```js
const socket = new WebSocket("ws://127.0.0.1:8765/ws/generate");
socket.binaryType = "arraybuffer";
let pendingMetadata = null;

socket.addEventListener("open", () => {
  socket.send(JSON.stringify({
    requestId: "job-001",
    prompt: "ambient techno",
    duration: 10,
    format: "mp3",
    bitrate: 192,
  }));
});

socket.addEventListener("message", (event) => {
  if (typeof event.data === "string") {
    const message = JSON.parse(event.data);
    if (message.type === "error") {
      console.error(message);
    } else {
      pendingMetadata = message;
      console.log("即将接收音频：", message.byteLength);
    }
    return;
  }

  const blob = new Blob([event.data], { type: pendingMetadata.contentType });
  const audio = new Audio(URL.createObjectURL(blob));
  audio.play();
});
```

## 并发和流式说明

- WebSocket 处理协程会把同步模型生成移到工作线程，不阻塞服务事件循环。
- HTTP 和不同 WebSocket 连接共用同一个应用服务；底层锁会串行执行模型推理。
- 当前发送的是完成后的整段 WAV/MP3，不是模型生成过程中的实时音频流。
- MP3 编码需要系统安装 FFmpeg；WAV 不需要。

FastAPI WebSocket 行为参考[官方 WebSocket 文档](https://fastapi.tiangolo.com/advanced/websockets/)和[接口参考](https://fastapi.tiangolo.com/reference/websockets/)。
