# 音符与鼓点控制

MRT2 除文本/参考音频风格外，还支持随时间变化的音符和鼓点条件。本项目提供两种输入：直接上传 Standard MIDI File，或传入便于程序生成的 JSON 事件。两者最终都会按官方 25 Hz 帧率转换，即每帧 40 ms。

## 数据格式

音符事件：

```json
{"pitch": 60, "start": 0.0, "duration": 0.5}
```

- `pitch` 是 MIDI 音高 `0～127`，中央 C 为 `60`；
- `start` 和 `duration` 的单位均为秒；
- 起音帧编码为官方 onset 状态，后续持续帧编码为 continuation 状态。

鼓点事件：

```json
{"time": 0.5}
```

`time` 单位为秒。官方鼓条件每帧只有“触发鼓点”这一维，不携带底鼓、军鼓或镲片类别；因此 MIDI 第 10 通道上的所有 note-on 都会合并为通用鼓点触发。

事件时间会对齐到最近的 40 ms 模型帧。超出请求 `duration` 的事件会被忽略。

## guide 与 strict

- `guide`：未指定的位置使用官方 masked 状态 `-1`，表示交给模型自由生成；适合把输入当作提示或骨架。
- `strict`：未指定的位置使用 off 状态 `0`，表示明确关闭；适合更强地约束旋律音高或鼓点密度。

`notes_mode` 与 `drums_mode` 可以分别设置。WebSocket 同时接受 camelCase 形式 `notesMode`、`drumsMode`。

## CLI

```bash
uv run mrt-local generate \
  --midi arrangement.mid \
  --prompt "minimal techno" \
  --notes-mode guide \
  --drums-mode strict \
  --cfg-notes 1.0 \
  --cfg-drums 1.0 \
  --duration 8 \
  --output controlled.wav
```

非第 10 通道的 note-on/note-off 会配对为音符；第 10 通道 note-on 会变为鼓点。`--no-midi-drums` 可忽略第 10 通道。MIDI、文本提示词和参考音频可以组合；只传 MIDI 也合法。

## HTTP JSON

`POST /generate` 与 `POST /stream` 接受：

```json
{
  "prompt": "warm synth ensemble",
  "duration": 4,
  "notes": [
    {"pitch": 60, "start": 0, "duration": 1},
    {"pitch": 64, "start": 1, "duration": 1}
  ],
  "drums": [{"time": 0}, {"time": 0.5}],
  "notes_mode": "guide",
  "drums_mode": "strict",
  "cfg_notes": 1.0,
  "cfg_drums": 1.0
}
```

文件上传使用 `POST /generate/midi` 或 `POST /stream/midi`。multipart 字段 `midi` 必填，`reference_audio`、`prompt` 及其他生成参数可选。所有 HTTP 路径都出现在 `/openapi.json` 和 Swagger UI `/docs` 中。

## WebSocket

`/ws/generate` 和 `/ws/stream` 的首条 JSON 可直接包含相同的 `notes` / `drums` 数组。WebSocket 不直接上传 MIDI 文件；客户端应先解析为事件，或者使用 HTTP multipart MIDI 端点。

流式生成不会改变条件含义：服务会逐个 40 ms 控制帧调用官方有状态生成，并把返回 state 传到下一帧。`/ws/stream` 可在生成途中通过 `update` 消息替换后续音符或鼓点计划；更新事件中的时间相对于实际 `effectiveFrame`，详见[流式生成](STREAMING.md)。

## CFG 怎么调

`cfg_notes` 和 `cfg_drums` 分别控制模型遵循音符和鼓点条件的强度，不是音符数量或鼓声音量。建议先保留 `1.0`：如果输出经常偏离输入，可逐档提高；如果音乐过于僵硬或出现质量下降，则降低。两项参数只有在对应事件存在时才有明确用途。
