# 项目架构

项目采用“核心用例 + 后端端口 + 传输适配器”的分层结构。CLI、HTTP API，以及未来可能增加的 Socket、WebSocket 或 Python SDK，都是同一个生成用例的不同外壳。

```text
CLI ───────────┐
HTTP API ──────┼──> GenerationService ──> GenerationBackend
WebSocket API ─┘            │                      │
                            │                      └─ MagentaMlxBackend
                            ├─ GenerateCommand / SamplingConfig
                            └─ AudioEncoder（WAV / MP3）
```

## 各层职责

### 核心数据模型：`mrt_local/core.py`

这里定义与传输协议和 Magenta 实现无关的数据：

- `ModelConfig`：模型名称、路径和加载参数。
- `SamplingConfig`：一组完整且已经解析的采样参数。
- `SamplingOverrides`：外部请求提供的可选覆盖值。
- `GenerateCommand`：所有外壳共用的生成命令。
- `AudioInput`：协议无关的浮点 PCM 参考音频。
- `GenerateResult`：统一的音频结果及 WAV 编码。

默认值合并和最终业务校验发生在这一层。即使绕过 FastAPI、直接调用 Python 服务，也会得到一致的验证行为。

### 应用服务：`mrt_local/service.py`

`GenerationService` 负责模型生命周期、请求串行化和生成用例编排。它只接受 `GenerateCommand`，不认识 HTTP 请求、argparse Namespace 或 Socket 消息。

### 后端端口与适配器：`mrt_local/backend.py`

`GenerationBackend` Protocol 定义应用服务所需的最小能力。`MagentaMlxBackend` 是当前实现，封装以下 Magenta 专用细节：

- MusicCoCa embedding；
- conditioning key 和 CFG 字典；
- 秒数到 25 Hz frame 的转换；
- `.mlxfn` 原生生成调用。

应用服务测试可以注入假后端，因此不需要模型文件或 MLX 设备。

### 输出编码：`mrt_local/encoding.py`

生成服务只返回原始 PCM `GenerateResult`。共享编码层根据 `AudioEncodingOptions` 生成 WAV 或 MP3，并提供媒体类型和文件扩展名；CLI、HTTP 与 WebSocket 不各自实现编码。WAV 使用 SoundFile，MP3 使用 FFmpeg/libmp3lame。

### 传输外壳：`mrt_local/cli.py`、`mrt_local/api.py` 和 `mrt_local/ws.py`

- CLI 将命令行参数转换成 `RuntimeConfig` 和 `GenerateCommand`，再把结果写入文件。
- HTTP API 将 JSON 文本请求或 multipart 音频上传转换成同一个 `GenerateCommand`，再把结果包装成 HTTP WAV/MP3 响应。
- WebSocket API 在长连接上接收 JSON 元数据以及可选的参考音频二进制消息，返回结果元数据和二进制 WAV/MP3。
- Pydantic 仍保留传输层格式约束，以生成准确的 OpenAPI；核心层会执行最终业务校验。

文本和参考音频最终都进入同一个 `GenerateCommand`，可以单独使用或按归一化权重混合。音频文件解码位于共享媒体层，Magenta 适配器只负责把核心 `AudioInput` 转为官方 `Waveform` 并融合两个 embedding；CLI、multipart HTTP 和 WebSocket 不直接依赖 Magenta 类型。

## 增加新外壳

新增其他 Socket 外壳时，不应直接调用 Magenta，也不应复制采样默认值或校验规则。它只需要：

1. 把消息反序列化为 `GenerateCommand`；
2. 调用共享的 `GenerationService.generate()`；
3. 把 `GenerateResult` 序列化回协议需要的格式。

如果需要真正的流式生成，应先扩展核心端口和应用用例，再由 HTTP、Socket 等外壳分别适配，避免在单一传输层中形成第二套推理逻辑。
