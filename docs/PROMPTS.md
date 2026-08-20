# 提示词与高级多文本风格混合

MRT2 支持两种互斥的文本输入模式：

- `prompt`：简单模式。整段文字作为一个音乐风格描述，适合大多数请求。
- `prompt_components` / `promptComponents`：高级模式。分别编码多个风格描述，再按相对权重混合。

高级模式不是单词或 token 级加权。官方 MusicCoCa 文本编码器把每段文字编码为一个 768 维整体风格 embedding，服务只能在这些整体 embedding 之间混合。最终 embedding 还会经过 RVQ 量化，因此权重变化不保证带来严格线性的听感变化。

## 数据结构

JSON HTTP 接口使用 snake_case：

```json
{
  "prompt_components": [
    {"text": "spacious ambient pads", "weight": 1},
    {"text": "powerful acoustic drums", "weight": 2},
    {"text": "subtle analog bass", "weight": 0.5}
  ]
}
```

WebSocket 同时接受 camelCase 和 snake_case，推荐 camelCase：

```json
{
  "promptComponents": [
    {"text": "spacious ambient pads", "weight": 1},
    {"text": "powerful acoustic drums", "weight": 2}
  ]
}
```

服务自动把有效权重归一化。`1:2`、`0.5:1` 和 `10:20` 的含义相同。权重为 `0` 的片段保留在请求中但不影响结果，便于客户端使用统一模板临时关闭某个概念。

## 限制与错误

| 限制 | 数值 | 原因 |
|---|---:|---|
| 片段数量 | 最多 8 个 | 限制启动和动态更新时的编码延迟 |
| 单片段长度 | 最多 1,000 个 Unicode 字符 | 防止异常大的控制消息；官方编码器仍只使用前 127 个文本 token |
| 文本总长度 | 最多 4,000 个 Unicode 字符 | 限制一次请求的总编码成本 |
| 单项权重 | `0`～`1,000,000` 的有限数 | 拒绝负数、NaN、Infinity 和无意义的极端数值 |
| 有效权重 | 至少一项大于 `0` | 确保文本混合可以计算 |
| 重复文本 | 不允许 | 防止无意重复编码；应合并为一项并调整权重 |

文本会先去除首尾空白，再检查空文本与重复项。`prompt` 和 `prompt_components` 同时出现会被拒绝，因为两种输入的组合规则容易产生歧义。

## CLI

`--weighted-prompt WEIGHT TEXT` 可以重复使用：

```bash
uv run mrt-local generate \
  --weighted-prompt 1 "spacious ambient pads" \
  --weighted-prompt 2 "powerful acoustic drums" \
  --weighted-prompt 0.5 "subtle analog bass" \
  --duration 10 \
  --output weighted.wav
```

不能同时使用 `--prompt` 和 `--weighted-prompt`。

## HTTP 与参考音频

JSON 请求直接传数组：

```bash
curl -X POST http://127.0.0.1:8765/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt_components": [
      {"text":"ambient pads","weight":1},
      {"text":"powerful drums","weight":2}
    ],
    "duration": 10
  }' \
  --output output.wav
```

multipart 端点把 `prompt_components` 作为 JSON 数组字符串发送：

```bash
curl -X POST http://127.0.0.1:8765/generate/audio \
  -F 'audio=@reference.wav' \
  -F 'prompt_components=[{"text":"ambient pads","weight":1},{"text":"powerful drums","weight":2}]' \
  -F 'text_weight=1' \
  -F 'audio_weight=1' \
  --output output.wav
```

服务先用各组件的 `weight` 得到一个整体文本 embedding，再通过 `text_weight` / `audio_weight` 混合文本与参考音频。两级权重用途不同。

## WebSocket 动态更新

`/ws/stream` 的 `start` 消息可直接包含 `promptComponents`。生成过程中用 `update` 原子替换整组文本：

```json
{
  "type": "update",
  "requestId": "session-1",
  "revision": 2,
  "effectiveFrame": 100,
  "promptComponents": [
    {"text": "dark ambient pads", "weight": 1},
    {"text": "aggressive electronic drums", "weight": 3}
  ]
}
```

传空数组会清除文本条件；此时必须仍有参考音频或音符/鼓点控制支持后续生成。更新中的所有文本编码和 embedding 切换都在专用 MLX 线程中完成，并在同一个生效帧原子应用。HTTP Streaming 只在启动请求中支持高级文本，连接建立后不能发送反向控制消息。

## 使用建议

- 每个组件描述一个完整、可独立理解的音乐概念，例如配器、节奏或整体风格。
- 先用 `1:1` 建立基线，再小幅调整到 `1:2` 或 `1:3`。
- 不要把每个单词拆成组件；短而完整的风格短语通常更稳定。
- `cfg_musiccoca` 控制整个最终风格条件的遵循强度，不会单独放大某个组件。

官方编码与混合依据见 [Magenta RealTime MusicCoCa 实现](https://github.com/magenta/magenta-realtime/blob/main/magenta_rt/musiccoca.py)。
