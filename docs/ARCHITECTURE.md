# 项目架构

项目采用“核心用例 + 后端端口 + 传输适配器”的分层结构。现有 CLI、HTTP、WebSocket，以及未来可能增加的其他 Socket 或 Python SDK，都是共享生成核心的不同外壳。

```text
CLI ───────────┐
HTTP API ──────┼──> GenerationService ──> GenerationBackend
WebSocket API ─┘            │                      │
                            │                      └─ MagentaMlxBackend
                            ├─ GenerateCommand / SamplingConfig
                            └─ AudioEncoder（WAV / MP3）
```

流式路径在同一模型实例上创建短生命周期的 `StreamingSession`：

```text
HTTP Streaming ─┐
WebSocket Stream ┴─> GenerationService.open_stream()
                         └─ StreamingSession
                              └─ MagentaMlxStreamSession（持有官方 state）
```

## 各层职责

### 核心数据模型：`mrt_local/core.py`

这里定义与传输协议和 Magenta 实现无关的数据：

- `ModelConfig`：模型名称、路径和加载参数。
- `SamplingConfig`：一组完整且已经解析的采样参数。
- `SamplingOverrides`：外部请求提供的可选覆盖值。
- `GenerateCommand`：完整文件生成命令。
- `StreamGenerateCommand`、`AudioChunk`：流式生成命令和协议无关 PCM 分片。
- `AudioInput`：协议无关的浮点 PCM 参考音频。
- `ControlInput`、`NoteEvent`、`DrumEvent`：协议无关的音符与鼓点事件。
- `ControlTimeline`：按官方 25 Hz 帧率转换后的动态模型条件。
- `GenerateResult`：统一的完整 PCM 音频结果；容器编码位于独立编码层。

默认值合并和最终业务校验发生在这一层。即使绕过 FastAPI、直接调用 Python 服务，也会得到一致的验证行为。

### 应用服务：`mrt_local/service.py`

`GenerationService` 负责模型生命周期、独占租约和生成用例编排。它接受核心命令，不认识 HTTP 请求、argparse Namespace 或 Socket 消息。模型被占用时不会让重叠请求无限排队，而是抛出 `ModelBusyError`，由 HTTP 映射为 `409`、WebSocket 映射为 `model_busy`。

流式生成使用 `StreamGenerateCommand`。一个流式会话在结束前独占模型，避免其他请求破坏实时生成时序；所有关闭和异常路径都会释放租约。

MLX 的 GPU stream 与创建它的线程绑定。因此服务持有一个 `max_workers=1` 的专用执行器，模型加载、完整生成、`open_stream`、所有 `next_chunk` 和后端关闭都只能在该线程执行。HTTP 与 WebSocket 的异步接口直接等待专用执行器的 Future；Starlette/AnyIO 工作线程只处理传输或编码，不执行 MLX。这样不会因请求或流式迭代被调度到不同线程而出现 `There is no Stream(gpu, ...) in current thread`。

### 后端端口与适配器：`mrt_local/backend.py`

`GenerationBackend` Protocol 定义应用服务所需的最小能力。`MagentaMlxBackend` 是当前实现，封装以下 Magenta 专用细节：

- MusicCoCa embedding；
- conditioning key 和 CFG 字典；
- 逐帧音符/鼓点 conditioning 与控制时间线游标；
- 秒数到 25 Hz frame 的转换；
- `.mlxfn` 原生生成调用；
- 流式会话中的官方 state 传递与复用。

应用服务测试可以注入假后端，因此不需要模型文件或 MLX 设备。

### 输出编码：`mrt_local/encoding.py`

生成服务只返回原始 PCM `GenerateResult`。共享编码层根据 `AudioEncodingOptions` 生成 WAV 或 MP3，并提供媒体类型和文件扩展名；CLI、HTTP 与 WebSocket 不各自实现编码。WAV 使用 SoundFile，MP3 使用 FFmpeg/libmp3lame。

### 传输外壳：`mrt_local/cli.py`、`mrt_local/api.py`、`mrt_local/ws.py` 和 `mrt_local/streaming_ws.py`

- CLI 将命令行参数转换成 `RuntimeConfig` 和 `GenerateCommand`，再把结果写入文件。
- HTTP API 将 JSON 事件、multipart MIDI/音频上传转换成同一个 `GenerateCommand`，再把结果包装成 HTTP WAV/MP3 响应。
- WebSocket API 在长连接上接收 JSON 元数据以及可选的参考音频二进制消息，返回结果元数据和二进制 WAV/MP3。
- HTTP Streaming 返回连续的裸 PCM 字节流；WebSocket Streaming 返回 `chunk` 元数据和保持消息边界的 PCM 二进制分片。
- Pydantic 仍保留传输层格式约束，以生成准确的 OpenAPI；核心层会执行最终业务校验。

文本、参考音频和控制事件最终进入完整或流式核心命令。文本/音频可按归一化权重混合；音符/鼓点由核心层构造统一时间线。音频文件解码位于共享媒体层，MIDI 文件解码位于 `mrt_local/midi.py`，Magenta 适配器只处理官方 conditioning；CLI、multipart HTTP 和 WebSocket 不直接依赖 Magenta 类型。

## 增加新外壳

新增其他 Socket 外壳时，不应直接调用 Magenta，也不应复制采样默认值或校验规则。它只需要：

1. 把消息反序列化为 `GenerateCommand`；
2. 调用共享的 `GenerationService.generate()`；
3. 把 `GenerateResult` 序列化回协议需要的格式。

HTTP Streaming 和 WebSocket 流式接口共用核心命令、服务会话和后端 state 管理；传输层只负责把 `AudioChunk` 编码为 float32le PCM。
