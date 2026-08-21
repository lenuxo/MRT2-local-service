# 实时 MIDI 演奏

`/ws/stream` 支持浏览器或其他客户端在音乐生成期间持续发送 MIDI Note On、Note Off 和延音踏板事件。服务不会把每次按键改写成一份新的完整时间线，而是维护演奏状态，并在每个 MRT2 模型帧（40 ms）开始前生成新的 128 音高条件。

## 与计划式 MIDI 的区别

| 模式 | 启动字段 | 适用场景 | 后续输入 |
|---|---|---|---|
| 计划式 | `midiMode: "plan"`，默认 | MIDI 文件、已经知道起止时间的编曲 | 用 `update.notes` / `update.drums` 替换未来计划 |
| 实时 | `midiMode: "live"` | 键盘演奏、打击垫、实时控制器 | 用独立的 `midi` 消息发送增量事件 |

两种模式在一个会话中互斥。实时模式启动时不能同时提供计划式 `notes` 或 `drums`，运行中也不能用 `update` 替换它们；文本、参考音频、权重、CFG、采样参数和 `drumless` 仍可动态更新。

HTTP Streaming 能流式返回音频，但 HTTP 响应连接是单向的，无法在同一请求中继续接收演奏事件。因此实时 MIDI 只由双向 WebSocket `/ws/stream` 提供。

## 启动会话

实时演奏建议使用 `chunkFrames: 1`，把服务端预生成量压到约 40 ms：

```json
{
  "type": "start",
  "requestId": "live-midi-001",
  "midiMode": "live",
  "liveNotesMode": "guide",
  "prompt": "warm analog synth ensemble",
  "duration": 60,
  "chunkFrames": 1,
  "realtime": true
}
```

`prompt`、参考音频与实时 MIDI 可以组合。也可以省略风格输入，仅使用实时 MIDI 条件启动会话。

`liveNotesMode` 的含义：

- `guide`（默认）：未按下的音高值为 `-1`，模型可以自行添加其他音符。
- `strict`：未按下的音高值为 `0`，更强地要求模型只围绕当前演奏音高生成。

这只决定音符条件基线；实际遵循强度仍由 `cfgNotes` 控制。

## 发送事件

收到 `ready` 后发送：

```json
{
  "type": "midi",
  "requestId": "live-midi-001",
  "eventSequence": 42,
  "events": [
    {"kind": "noteOn", "channel": 0, "pitch": 60, "velocity": 96},
    {"kind": "noteOn", "channel": 0, "pitch": 64, "velocity": 88}
  ]
}
```

释放按键：

```json
{
  "type": "midi",
  "requestId": "live-midi-001",
  "eventSequence": 43,
  "events": [
    {"kind": "noteOff", "channel": 0, "pitch": 60, "velocity": 0},
    {"kind": "noteOff", "channel": 0, "pitch": 64, "velocity": 0}
  ]
}
```

支持的事件如下：

| `kind` | 字段 | 语义 |
|---|---|---|
| `noteOn` | `channel`、`pitch`、`velocity` | 按下音符；`velocity=0` 按 MIDI 约定作为 Note Off |
| `noteOff` | `channel`、`pitch`、`velocity` | 释放音符 |
| `controlChange` | `channel`、`controller`、`value` | 当前支持 CC64 延音踏板、CC120 All Sound Off、CC123 All Notes Off |
| `panic` | 可选 `channel` | 立即清除指定通道或所有通道的按键、踏板和延迟释放状态 |

`channel` 使用程序员常见的 `0..15`；MIDI 第 10 通道因此是 `channel: 9`。该通道的 Note On 会产生一个通用鼓点触发。MRT2 的鼓条件不区分底鼓、军鼓或镲片，所以 `pitch` 只用于接收合法 MIDI 消息，不会选择具体鼓件。`drumless=true` 与这些鼓触发互斥。

MRT2 音符条件只包含音高的起音、持续和关闭状态，不包含力度或通道维度。服务仍验证并维护 `velocity` 与 `channel`，用于正确处理 velocity-zero Note On、重复按键、跨通道同音高和踏板释放，但它们不会直接控制生成音量或音色。

快速按下并在同一个 40 ms 帧前释放的音符会保留一个起音帧，不会完全丢失。

## 确认、顺序与重试

服务成功入队后返回：

```json
{
  "type": "midiQueued",
  "requestId": "live-midi-001",
  "sessionId": "934f6c3b-...",
  "eventSequence": 42,
  "controlSequence": 3,
  "earliestEffectiveFrame": 126,
  "earliestEffectiveTimestampMs": 5040,
  "acceptedEvents": 2,
  "processingTimeMs": 0.12
}
```

`earliestEffectiveFrame` 是事件最早可能进入的尚未生成帧；事件与正在执行的单帧生成可能竞争，因此它不是播放端已经听到变化的保证。已经生成、发送或进入浏览器播放缓冲的音频无法追回。

`eventSequence` 是实时 MIDI 专用的严格递增序列，与低频 `update` / `extend` / `configure` 使用的 `revision` 相互独立。完全相同的消息可以用同一序号重发并得到 `duplicate: true`；相同序号但内容不同返回 `midi_sequence_conflict`，旧序号返回 `stale_midi_sequence`。

每条消息最多 128 个事件，待处理队列最多 2048 个事件。队列溢出时服务不会静默丢弃 Note Off，而是清空队列、安排全局 `panic` 并返回错误，以降低卡音风险。连接关闭时会清理整个实时 MIDI 状态。

## 浏览器 Web MIDI 示例

Web MIDI API 需要安全上下文和用户授权；本机 `localhost` 通常可作为可信来源。以下示例把浏览器收到的原始消息转换成服务协议，忽略当前不支持的 CC：

```js
const socket = new WebSocket("ws://127.0.0.1:8765/ws/stream");
let eventSequence = 0;

socket.addEventListener("open", () => {
  socket.send(JSON.stringify({
    type: "start",
    requestId: "live-midi-001",
    midiMode: "live",
    liveNotesMode: "guide",
    prompt: "responsive electronic ensemble",
    duration: 60,
    chunkFrames: 1,
    realtime: true,
  }));
});

const access = await navigator.requestMIDIAccess();
for (const input of access.inputs.values()) {
  input.addEventListener("midimessage", ({ data }) => {
    const [status, data1, data2] = data;
    const command = status & 0xf0;
    const channel = status & 0x0f;
    let event;
    if (command === 0x90) {
      event = data2 === 0
        ? { kind: "noteOff", channel, pitch: data1, velocity: 0 }
        : { kind: "noteOn", channel, pitch: data1, velocity: data2 };
    } else if (command === 0x80) {
      event = { kind: "noteOff", channel, pitch: data1, velocity: data2 };
    } else if (command === 0xb0 && [64, 120, 123].includes(data1)) {
      event = { kind: "controlChange", channel, controller: data1, value: data2 };
    } else {
      return;
    }
    socket.send(JSON.stringify({
      type: "midi",
      requestId: "live-midi-001",
      eventSequence: eventSequence++,
      events: [event],
    }));
  });
}
```

生产前端还应在 MIDI 设备断开、页面失焦或演奏状态重置时发送 `panic`，并使用 AudioWorklet 或等效播放队列控制客户端缓冲。Web MIDI 数据格式和权限模型见 [W3C Web MIDI API](https://www.w3.org/TR/webmidi/)。
