# 流式生成

服务提供 HTTP Streaming 和 WebSocket 两种流式外壳。它们共享同一个有状态生成核心：会话启动时计算初始风格 embedding；WebSocket 动态替换文本、参考音频或混合权重时会原子更新它。每个音频分片都会把官方 MRT2 返回的 `state` 传给下一次生成调用。

模型加载、会话创建、分片生成、state 更新和关闭始终在同一个专用 MLX 线程执行；异步传输层只等待生成结果，避免 MLX GPU stream 跨线程失效。

## PCM 格式

两种接口统一输出：

- 48,000 Hz；
- 双声道交错采样；
- little-endian float32 PCM；
- 每个模型 frame 为 40 ms，即每声道 1,920 个采样；
- `chunk_frames` / `chunkFrames` 默认 `5`，即约 200 ms。

PCM 没有 WAV 文件头，不能直接保存为 `.wav`。需要完整 WAV 或 MP3 时请继续使用非流式生成接口。

流式接口使用与完整文件生成相同的 prompt、参考音频、MIDI/事件控制、混合权重和采样参数。`duration` 默认 `10` 秒，单次范围 `(0, 300]`；WebSocket 可在会话结束前反复发送 `extend` 延长时长，也可用 `stop` 提前结束。

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

服务返回 `ready`。随后每个分片严格按 `chunk` JSON、二进制 PCM、`metrics` JSON 的顺序发送：

```json
{
  "type": "chunk",
  "requestId": "stream-001",
  "sessionId": "934f6c3b-...",
  "sequence": 0,
  "frames": 5,
  "samplesPerChannel": 9600,
  "byteLength": 76800,
  "timestampMs": 0
}
```

## WebSocket 实时控制

`ready.dynamicCapabilities` 声明当前服务实际支持的动态能力。客户端应根据它启用控件，而不要只依据版本号猜测：

```json
{
  "protocolVersion": 3,
  "update": ["prompt", "temperature", "topK", "cfgMusiccoca", "cfgNotes", "cfgDrums", "seed", "useMapper", "poolAcrossTime", "notes", "drums", "notesMode", "drumsMode", "referenceAudio", "textWeight", "audioWeight"],
  "effectiveFrame": true,
  "extendDuration": true,
  "chunkFrames": true,
  "realtime": true,
  "referenceAudio": true,
  "styleWeights": true,
  "metrics": true,
  "revisionPolicy": "strictly_increasing_idempotent_replay",
  "limits": {
    "controlMessageBytes": 65536,
    "referenceAudioBytes": 67108864,
    "referenceAudioTimeoutSeconds": 10
  }
}
```

控制分为三类：`update` 修改模型后续生成条件，`extend` 延长会话时间线，`configure` 修改流的分片与发送行为。它们不需要启动所谓的“动态模式”；`/ws/stream` 会话天然支持这些消息。

### 更新生成条件

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
  "sessionId": "934f6c3b-...",
  "revision": 3,
  "controlSequence": 0,
  "effectiveFrame": 126,
  "effectiveTimestampMs": 5040,
  "processingTimeMs": 18.4
}
```

`revision` 由客户端定义，服务原样返回，便于把确认消息与本地 UI 状态对应。`update`、`extend` 和 `configure` 共用同一条严格递增的 revision 序列。完全相同的消息使用相同 revision 重发时不会重复执行，响应包含 `duplicate:true`；同 revision 内容不同返回 `revision_conflict`，低于最近已接受值返回 `stale_revision`。服务最多保留最近 256 条幂等记录。每个控制响应还包含从 0 开始单调递增的 `controlSequence`。更新字段均可省略，但一条消息至少要包含一个实际更新字段。

| 更新字段 | 作用 |
|---|---|
| `prompt` | 重新计算文本风格 embedding；传 `null` 清除文本条件，当前参考音频仍保留 |
| `referenceAudio` | `replace` 表示下一条消息是新参考音频；`clear` 清除当前参考音频 |
| `textWeight` / `audioWeight` | 修改当前文本与参考音频 embedding 的相对混合权重 |
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

动态更新文本时会保留当前参考音频，并按当前权重重新混合；更新参考音频时同样会保留当前文本。权重只对实际存在的条件生效，并会自动归一化。

替换参考音频需要连续发送两条消息。先发送 JSON：

```json
{
  "type": "update",
  "requestId": "stream-001",
  "revision": 4,
  "referenceAudio": "replace",
  "textWeight": 1,
  "audioWeight": 3
}
```

随后立即发送一条二进制 WebSocket 消息，内容为完整的 WAV、MP3 或其他服务可解码的音频文件。服务解码音频并在专用 MLX 线程计算新 embedding；成功后返回 `updateAccepted`。在二进制消息到达前，不应发送其他控制消息。

控制 JSON 最大 64 KiB；参考音频二进制消息最大 64 MiB，且必须在 JSON 后 10 秒内到达。超限、超时或消息类型错误会返回明确的协议错误，当前流保持运行。

清除参考音频不需要二进制消息：

```json
{"type":"update","requestId":"stream-001","revision":5,"referenceAudio":"clear"}
```

只调整混合比例时，可省略 `referenceAudio`，直接发送 `textWeight` 和/或 `audioWeight`。如果当前同时存在文本和音频，`poolAcrossTime` 必须为 `true`；所有实际存在条件的有效权重不能同时为 `0`。

### 延长会话

在收到最终 `completed` 之前发送：

```json
{"type":"extend","requestId":"stream-001","revision":6,"additionalDuration":30}
```

服务在保留 MRT2 state、当前采样位置和控制时间线的前提下增加时长，并返回：

```json
{"type":"extended","requestId":"stream-001","sessionId":"934f6c3b-...","revision":6,"controlSequence":3,"previousDurationMs":10000,"durationMs":40000,"processingTimeMs":0.2}
```

`additionalDuration` 单次范围为 `(0, 300]` 秒。它不是回放或重新生成；新增音频从当前会话 state 继续生成。客户端若要维持长期实时会话，应在剩余缓冲耗尽前续期。

### 修改传输行为

```json
{"type":"configure","requestId":"stream-001","revision":7,"chunkFrames":1,"realtime":true}
```

`chunkFrames` 范围 `1～25`，影响下一次尚未开始的模型调用；`realtime` 控制服务按播放时钟推进还是尽快生成。服务返回 `configured`，其中包含实际生效帧和当前配置。二者至少提供一个：

```json
{"type":"configured","requestId":"stream-001","sessionId":"934f6c3b-...","revision":7,"controlSequence":4,"effectiveFrame":130,"chunkFrames":1,"realtime":true,"processingTimeMs":0.1}
```

非法的 `update`、`extend` 或 `configure` 返回 `code=control_validation_error`，但不会关闭流式会话。更新 embedding 可能比普通数值参数耗时更长，因此 UI 应等待确认消息，而不是假定发送瞬间已经生效。

提前停止：

```json
{"type":"stop","requestId":"stream-001"}
```

正常结束或停止后返回：

```json
{
  "type": "completed",
  "requestId": "stream-001",
  "sessionId": "934f6c3b-...",
  "reason": "duration_reached",
  "generatedSamples": 480000
}
```

`reason` 可能为 `duration_reached` 或 `client_stop`。客户端断开时服务直接清理会话，无法再发送完成消息。

## 会话标识与性能指标

`ready` 会返回服务生成的 UUID `sessionId`；同一会话的 `chunk`、`metrics`、控制确认、错误和 `completed` 都携带它。客户端应以 `sessionId` 区分重连或并发建立的连接，不要把 `requestId` 当作服务端唯一标识。

每条 PCM 二进制分片之后会发送一条 `metrics`：

```json
{
  "type": "metrics",
  "requestId": "stream-001",
  "sessionId": "934f6c3b-...",
  "sequence": 12,
  "generationTimeMs": 31.4,
  "generatedAudioMs": 2600,
  "realtimeFactor": 0.157,
  "bufferLeadMs": 184.2,
  "firstChunkLatencyMs": null
}
```

- `generationTimeMs`：本分片等待模型生成完成的墙钟时间。
- `generatedAudioMs`：会话累计生成的音频时长。
- `realtimeFactor`：本分片生成耗时除以本分片音频时长；小于 `1` 表示生成速度快于播放速度。
- `bufferLeadMs`：累计音频时长减去会话已用墙钟时间；负值表示生成落后于实时播放。
- `firstChunkLatencyMs`：仅首个分片提供从收到启动消息到生成完成的延迟，其余为 `null`。

控制确认包含 `processingTimeMs`；动态参考音频更新另外包含 `audioDecodeTimeMs`。这些值用于观测服务端处理时间，不包含客户端播放缓冲或网络往返延迟。

启动或生成错误使用 JSON `error` 消息。主要错误码为 `validation_error`、`control_validation_error`、`revision_conflict`、`stale_revision`、`message_too_large`、`reference_audio_timeout`、`reference_audio_too_large`、`model_busy` 和 `generation_error`。连接状态依次为“等待 start → active → completed/断开”；只有 active 状态接受控制消息。一个 `/ws/stream` 连接只承载一次流式会话，完成后如需再次生成，应新建连接。

## 并发规则

一个服务进程只加载一个模型实例。普通生成或流式会话会独占该实例；被占用时，HTTP 返回 `409 Conflict`，WebSocket 返回 `model_busy`。流式会话结束、取消、断开或失败时都会释放模型。

HTTP Streaming 的请求体在响应开始前已经结束，因此实时控制仅由双向 WebSocket 提供。HTTP Streaming 仍使用会话开始时固定的条件。WebSocket 可通过续期保持长期生成并动态替换风格条件，但仍不支持在会话中切换模型。
