# MRT2 本地服务 API

API 基于 FastAPI 实现，并自动生成 OpenAPI 3.1 规范。

```bash
uv run mrt-serve --model mrt2_small
# 或
uv run mrt-serve --model mrt2_base
# 自定义端口（默认 8765）
uv run mrt-serve --model mrt2_small --port 9000
```

默认地址为 `http://127.0.0.1:8765`：

- `GET /docs`：Swagger UI
- `GET /openapi.json`：OpenAPI JSON
- `GET /health`：健康状态
- `GET /info`：当前模型与运行环境
- `POST /generate`：生成 WAV 或 MP3
- `POST /generate/audio`：上传参考音频并生成 WAV 或 MP3
- `POST /generate/midi`：上传 MIDI，可选叠加文本与参考音频
- `POST /stream`：根据 JSON 条件连续返回 float32le PCM
- `POST /stream/audio`：上传参考音频并连续返回 float32le PCM
- `POST /stream/midi`：上传 MIDI 并连续返回 float32le PCM
- `WS /ws/generate`：通过长连接连续生成 WAV 或 MP3
- `WS /ws/stream`：有状态地连续生成 float32le PCM

WebSocket 不属于 OpenAPI 规范，因此不会显示在 Swagger UI 中；消息协议见 [WebSocket API](WEBSOCKET.md)。

## 跨域访问

服务默认允许任意来源、HTTP 方法和请求头访问，并通过 `Access-Control-Expose-Headers: *` 暴露包括流式音频元数据在内的响应头。CORS 凭据模式未启用，因此该通配配置不用于跨域 Cookie 会话。WebSocket 接口也不校验 `Origin`。

默认监听地址仍是 `127.0.0.1`；CORS 只影响浏览器是否允许跨来源调用，不会把服务自动暴露到局域网。只有显式使用 `--host 0.0.0.0` 等地址时才会改变网络可达范围。

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
| `prompt` | string/null | 条件必填 | 可选风格提示词；没有 `notes` 或 `drums` 时必填 |
| `duration` | number | 否 | 生成秒数，范围 `(0, 300]`，默认 `10` |
| `temperature` | number/null | 否 | 采样随机度；越高变化越大，越低越保守；必须 `> 0`，默认 `1.3` |
| `top_k` | integer/null | 否 | 每一步只从概率最高的 K 个候选中采样；越小选择越集中；必须 `>= 1`，默认 `40` |
| `cfg_musiccoca` | number/null | 否 | 文本/参考音频风格的遵循强度；通常保留默认 `3.0` |
| `cfg_notes` | number/null | 否 | 模型遵循 `notes` 音高与起止时间的强度，默认 `1.0` |
| `cfg_drums` | number/null | 否 | 模型遵循 `drums` 鼓点触发时间的强度，默认 `1.0` |
| `notes` | array | 否 | 音符事件；每项为 `{pitch,start,duration}`，时间单位为秒 |
| `drums` | array | 否 | 鼓点事件；每项为 `{time}`，时间单位为秒 |
| `notes_mode` | `guide`/`strict` | 否 | 未指定音高允许自由生成或强制关闭，默认 `guide` |
| `drums_mode` | `guide`/`strict` | 否 | 未指定帧允许自由生成鼓点或强制关闭，默认 `guide` |
| `seed` | integer/null | 否 | 文本 mapper 的随机种子；仅在 `use_mapper=true` 且有文本时生效，不保证音频完全可复现，默认 `0` |
| `use_mapper` | boolean/null | 否 | 是否把文本 embedding 映射到音频风格空间；仅影响文本条件，默认 `true` |
| `pool_across_time` | boolean/null | 否 | 是否将参考音频各时间片平均为一个整体风格；文本/音频混合时必须为 `true`，默认 `true` |
| `format` | `wav`/`mp3` | 否 | 输出格式，默认 `wav` |
| `bitrate` | integer/null | 否 | MP3 比特率，范围 32～320 kbps，MP3 默认 `192`；WAV 不接受该字段 |

WAV 成功响应为 `audio/wav`，内容是 48 kHz 双声道 IEEE float WAV；MP3 成功响应为 `audio/mpeg`。未知字段、空 prompt、非法 duration/format/bitrate 由 FastAPI/Pydantic 返回 HTTP `422`；给 WAV 设置 bitrate 等跨字段错误返回 HTTP `400`。

MP3 示例：

```bash
curl -X POST http://127.0.0.1:8765/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"ambient synth pads","duration":8,"format":"mp3","bitrate":192}' \
  --output ambient.mp3
```

MP3 编码需要系统安装 FFmpeg；macOS 可以运行 `brew install ffmpeg`。缺少 FFmpeg 或 MP3 编码器时返回 HTTP `500` 和明确错误说明，WAV 不受影响。

## MIDI 与事件控制

JSON 接口可直接发送事件，且允许不传提示词：

```bash
curl -X POST http://127.0.0.1:8765/generate \
  -H 'Content-Type: application/json' \
  -d '{"duration":4,"notes":[{"pitch":60,"start":0,"duration":1}],"drums":[{"time":0},{"time":0.5}]}' \
  --output controlled.wav
```

上传 MIDI 使用 `multipart/form-data`；`midi` 必填，`reference_audio` 和 `prompt` 可选：

```bash
curl -X POST http://127.0.0.1:8765/generate/midi \
  -F 'midi=@arrangement.mid' \
  -F 'prompt=warm analog synths' \
  -F 'duration=8' \
  -F 'notes_mode=guide' \
  -F 'drums_mode=strict' \
  --output controlled.wav
```

MIDI 第 10 通道转换为无音高类别的鼓点触发，其他通道转换为音符事件。详细语义见[音符与鼓点控制](CONTROL.md)。

## 参考音频生成

`POST /generate/audio` 使用 `multipart/form-data`。`audio` 是必填文件字段；`prompt` 是可选文本条件，其余生成与编码字段和 JSON 接口含义一致：

```bash
curl -X POST http://127.0.0.1:8765/generate/audio \
  -F 'audio=@reference.wav' \
  -F 'prompt=ambient pads' \
  -F 'text_weight=1' \
  -F 'audio_weight=3' \
  -F 'duration=10' \
  -F 'temperature=1.3' \
  -F 'cfg_musiccoca=3.0' \
  -F 'format=mp3' \
  -F 'bitrate=192' \
  --output styled.mp3
```

`prompt` 可选。提供后，服务按 `text_weight` 和 `audio_weight` 混合文本与音频 embedding；两项默认均为 `0.5`，有效权重会自动归一化，且不能同时为零。

`text_weight` 和 `audio_weight` 是相对比例而不是百分比，例如 `1/3` 与 `0.25/0.75` 等价。只有一种输入时，该输入会自动归一化为 `1.0`，另一个权重不会产生作用；权重为 `0` 表示对应输入不参与最终风格 embedding。

服务优先使用 SoundFile 解码 WAV、FLAC、OGG 等格式；无法解码时回退到 FFmpeg，因此也可接收 MP3 和 FFmpeg 支持的常见音频格式。参考音频最长 300 秒。MusicCoCa 会转为单声道、重采样，并按 10 秒片段提取风格；默认对所有片段求平均。

参考音频仅作为风格条件，不是音频续写、翻唱或编辑输入。

## 流式生成

`POST /stream`、`POST /stream/audio` 和 `POST /stream/midi` 使用与完整文件接口相同的风格、控制与采样参数，并增加 `chunk_frames`（范围 `1～25`，默认 `5`）。响应为 48 kHz 双声道 `float32le` 裸 PCM，不能直接当作 WAV/MP3 文件读取。

完整协议、响应头和示例见[流式生成文档](STREAMING.md)。两个 HTTP 流式端点均会出现在 OpenAPI 文档中。

## JavaScript 完整文件示例

```js
const response = await fetch("http://127.0.0.1:8765/generate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ prompt: "ambient techno", duration: 10 }),
});

if (!response.ok) throw new Error(await response.text());
const wav = await response.arrayBuffer();
```

完整参数含义、官方默认值和模型差异见 [模型与推理参数](MODELS.md)。

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
  "architecture": "arm64",
  "temperature": 1.3,
  "top_k": 40,
  "cfg_musiccoca": 3.0,
  "cfg_notes": 1.0,
  "cfg_drums": 1.0,
  "warmup_steps": 5,
  "seed": 0,
  "use_mapper": true,
  "pool_across_time": true
}
```

## 错误

- `422`：请求格式或字段验证失败
- `400`：核心业务参数或编码选项验证失败
- `409`：模型正被另一个完整生成或流式会话占用
- `500`：模型推理失败

模型在服务启动时加载一次。`POST /generate` 不能临时切换模型；如需使用另一个模型，请通过 `--model` 重启服务。
