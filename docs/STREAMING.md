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
| JSON `notes` / `drums` 或 multipart `midi` | `notes` / `drums` | 无 | 音符与鼓点控制；WebSocket 可动态替换后续计划 |
| `text_weight` | `textWeight` | `0.5` | 文本 embedding 权重 |
| `audio_weight` | `audioWeight` | `0.5` | 音频 embedding 权重 |
| `duration` | `duration` | `10` | 生成秒数，范围 `(0, 300]` |
| `chunk_frames` | `chunkFrames` | `5` | 每个应用分片的模型帧数，范围 `1～25` |
| 无 | `realtime` | `true` | 按播放节奏发送并只保留约一个 chunk 的预生成缓冲；设为 `false` 时尽快生成 |

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
  "chunkFrames": 5,
  "realtime": true
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

## WebSocket 动态更新

收到 `ready` 后，客户端可在音频持续生成期间发送 `update`。服务复用当前 MRT2 state，更新只影响生效帧及其后的音乐：

```json
{
  "type": "update",
  "requestId": "stream-001",
  "revision": 3,
  "prompt": "driving acid techno",
  "temperature": 0.9,
  "topK": 32,
  "cfgMusiccoca": 4.0,
  "cfgNotes": 1.4,
  "cfgDrums": 2.0,
  "notes": [
    {"pitch": 60, "start": 0, "duration": 0.5},
    {"pitch": 64, "start": 0.5, "duration": 0.5}
  ],
  "drums": [{"time": 0}, {"time": 0.5}],
  "notesMode": "guide",
  "drumsMode": "strict"
}
```

服务接受更新后返回：

```json
{
  "type": "updateAccepted",
  "requestId": "stream-001",
  "revision": 3,
  "effectiveFrame": 126,
  "effectiveTimestampMs": 5040
}
```

`revision` 由客户端定义，服务原样返回，便于把确认消息与本地 UI 状态对应。更新字段均可省略，但一条消息至少要包含一个实际更新字段。

| 更新字段 | 作用 |
|---|---|
| `prompt` | 重新计算文本风格 embedding；传 `null` 清除风格条件 |
| `temperature` / `topK` | 修改后续 token 的采样随机性和候选范围 |
| `cfgMusiccoca` / `cfgNotes` / `cfgDrums` | 修改后续帧对风格、音符和鼓点条件的遵循强度 |
| `seed` / `useMapper` / `poolAcrossTime` | 更新 embedding 配置；与同条或后续 prompt 更新配合使用 |
| `notes` | 替换生效帧之后的音符计划；空数组清除原计划 |
| `drums` | 替换生效帧之后的鼓点计划；空数组清除原计划 |
| `notesMode` / `drumsMode` | 对本次新计划选择 `guide` 或 `strict` |
| `effectiveFrame` | 可选的会话绝对生效帧；每帧 40 ms |

字段同时接受 camelCase 和 snake_case，例如 `topK` / `top_k`、`cfgNotes` / `cfg_notes`。

时序规则：

- 省略 `effectiveFrame` 时，更新在服务处理消息时的下一个未生成帧生效。
- 指定帧已经生成时，服务自动顺延，并在 `updateAccepted.effectiveFrame` 中返回实际帧。
- 指定未来帧时，更新会被预约到该帧；同一帧按服务接收顺序应用。
- `notes[].start`、`notes[].duration` 和 `drums[].time` 单位为秒，且相对于实际生效帧，不是相对于整个会话起点。
- 更新不会追回已经生成、发送或进入客户端播放缓冲区的音频。`chunkFrames` 越小，交互响应通常越快；`1` 对应 40 ms 模型粒度。

`realtime=true` 时，服务按音频播放时钟控制生成节奏，并仅提前生成约一个 chunk，避免模型在用户操作前已经跑完大段音频。默认 `chunkFrames=5` 时控制缓冲约为 200 ms；需要更敏捷的实时演奏可使用 `1`（约 40 ms），代价是增加调用与传输开销。离线测速或希望尽快拿到完整 PCM 时可设为 `false`。

动态 `prompt` 会把当前风格条件替换为新的纯文本 embedding。即使会话最初使用了参考音频，也不会继续混合旧音频 embedding；当前版本尚不支持在流中上传新的参考音频或动态调整文本/音频混合权重。

非法更新返回 `code=update_validation_error`，但不会关闭流式会话。更新 embedding 可能比普通数值参数耗时更长，因此 UI 应以 `updateAccepted` 为准，而不是假定发送瞬间已经生效。

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

启动或生成错误使用 JSON `error` 消息。主要错误码为 `validation_error`、`update_validation_error`、`model_busy` 和 `generation_error`。一个 `/ws/stream` 连接只承载一次流式会话；完成后如需再次生成，应新建连接。

## 并发规则

一个服务进程只加载一个模型实例。普通生成或流式会话会独占该实例；被占用时，HTTP 返回 `409 Conflict`，WebSocket 返回 `model_busy`。流式会话结束、取消、断开或失败时都会释放模型。

HTTP Streaming 的请求体在响应开始前已经结束，因此当前动态更新仅由双向 WebSocket 提供。HTTP Streaming 仍使用会话开始时固定的条件。WebSocket 暂不支持动态参考音频、模型切换和无限时长会话。
