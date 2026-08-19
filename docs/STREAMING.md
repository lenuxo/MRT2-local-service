# 流式生成

服务提供 HTTP Streaming 和 WebSocket 两种流式外壳。它们共享同一个有状态生成核心：风格 embedding 只计算一次，每个音频分片都会把官方 MRT2 返回的 `state` 传给下一次生成调用。

模型加载、会话创建、分片生成、state 更新和关闭始终在同一个专用 MLX 线程执行；异步传输层只等待生成结果，避免 MLX GPU stream 跨线程失效。

## PCM 格式

两种接口统一输出：

- 48,000 Hz；
- 双声道交错采样；
- little-endian float32 PCM；
- 每个模型 frame 为 40 ms，即每声道 1,920 个采样；
- `chunk_frames` / `chunkFrames` 默认 `5`，即约 200 ms。

PCM 没有 WAV 文件头，不能直接保存为 `.wav`。需要完整 WAV 或 MP3 时请继续使用非流式生成接口。

流式接口使用与完整文件生成相同的 prompt、参考音频、MIDI/事件控制、混合权重和采样参数。`duration` 默认 `10` 秒，范围 `(0, 300]`；当前所有流最终都受这个时长限制，WebSocket 可以用 `stop` 提前结束，但不提供无限时长模式。

| HTTP JSON/表单 | WebSocket | 默认值 | 说明 |
|---|---|---:|---|
| `prompt` | `prompt` | 无 | 文本条件；没有参考音频时必填 |
| multipart `audio` | `inputType=audio` 后的二进制消息 | 无 | 参考音频条件 |
| JSON `notes` / `drums` 或 multipart `midi` | `notes` / `drums` | 无 | 音符与鼓点控制；会话开始时固定 |
| `text_weight` | `textWeight` | `0.5` | 文本 embedding 权重 |
| `audio_weight` | `audioWeight` | `0.5` | 音频 embedding 权重 |
| `duration` | `duration` | `10` | 生成秒数，范围 `(0, 300]` |
| `chunk_frames` | `chunkFrames` | `5` | 每个应用分片的模型帧数，范围 `1～25` |

其余 temperature、top-k、CFG、seed、mapper 和时间聚合参数见[模型与推理参数](MODELS.md)。

## HTTP Streaming

文本输入：

```bash
curl --no-buffer -X POST http://127.0.0.1:8765/stream \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"ambient pads","duration":10,"chunk_frames":5}' \
  --output output.f32le
```

参考音频或文本/音频混合输入：

```bash
curl --no-buffer -X POST http://127.0.0.1:8765/stream/audio \
  -F 'audio=@reference.wav' \
  -F 'prompt=ambient pads' \
  -F 'text_weight=1' \
  -F 'audio_weight=3' \
  -F 'duration=10' \
  --output output.f32le
```

MIDI 流式输入：

```bash
curl --no-buffer -X POST http://127.0.0.1:8765/stream/midi \
  -F 'midi=@arrangement.mid' \
  -F 'prompt=ambient pads' \
  -F 'duration=10' \
  --output output.f32le
```

响应类型为 `application/octet-stream`，并通过以下响应头描述音频：

```text
X-Audio-Sample-Rate: 48000
X-Audio-Channels: 2
X-Audio-Sample-Format: float32le
```

HTTP 传输层可能合并或拆分应用生成的 chunk；客户端应把响应视为连续 PCM 字节流，不能假设一次 `read()` 恰好对应一个 `chunk_frames` 分片。WebSocket 接口则通过 `chunk` 元数据明确保留每个应用分片的边界。

客户端关闭响应或取消 Fetch 请求后，服务会在当前模型分片结束时关闭会话并释放模型。

## WebSocket

连接 `ws://127.0.0.1:8765/ws/stream`，首先发送：

```json
{
  "type": "start",
  "requestId": "stream-001",
  "prompt": "ambient pads",
  "duration": 10,
  "chunkFrames": 5
}
```

参考音频输入时，在启动消息中设置 `"inputType":"audio"`，然后立即发送一条包含完整参考音频文件的二进制消息。启动消息仍可包含 `prompt`、`textWeight` 和 `audioWeight`，用于文本/音频条件混合。

服务返回 `ready`，随后为每个分片依次返回一条 `chunk` JSON 和一条二进制 PCM 消息：

```json
{
  "type": "chunk",
  "requestId": "stream-001",
  "sequence": 0,
  "frames": 5,
  "samplesPerChannel": 9600,
  "byteLength": 76800,
  "timestampMs": 0
}
```

提前停止：

```json
{"type":"stop","requestId":"stream-001"}
```

正常结束或停止后返回：

```json
{
  "type": "completed",
  "requestId": "stream-001",
  "reason": "duration_reached",
  "generatedSamples": 480000
}
```

`reason` 可能为 `duration_reached` 或 `client_stop`。客户端断开时服务直接清理会话，无法再发送完成消息。

启动或生成错误使用 JSON `error` 消息。主要错误码为 `validation_error`、`model_busy` 和 `generation_error`。一个 `/ws/stream` 连接只承载一次流式会话；完成后如需再次生成，应新建连接。

## 并发规则

一个服务进程只加载一个模型实例。普通生成或流式会话会独占该实例；被占用时，HTTP 返回 `409 Conflict`，WebSocket 返回 `model_busy`。流式会话结束、取消、断开或失败时都会释放模型。

当前版本在会话开始后固定风格、音符/鼓点时间线与采样参数，不支持流中动态更新提示词、控制事件、CFG 或 temperature。
