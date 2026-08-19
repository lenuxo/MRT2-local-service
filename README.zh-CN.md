# MRT2 本地服务（Python）

[English](README.md) | 简体中文

这是一个面向 macOS Apple Silicon 的本地 Magenta RealTime 2 服务。项目现已完全迁移到 Python 技术栈：

- 使用 Magenta 官方 `magenta-rt[mlx]` Python 包执行推理
- 使用 FastAPI 通过 HTTP 和 WebSocket 提供完整文件与有状态 PCM 流式 API
- 使用标准 Python CLI 同时提供命令行生成和常驻服务
- CLI 与各 API 共用 `GenerationService` 和协议无关的核心命令，没有重复推理逻辑
- 支持 `mrt2_small` 和 `mrt2_base`
- 模型和共享资源保存在项目的 `models/` 目录中
- 默认允许浏览器从任意来源访问 HTTP 和 WebSocket 接口

## 环境要求

- Apple Silicon Mac
- macOS 14 或更高版本
- Python 3.11 或 3.12
- 推荐使用 `uv`
- 仅生成 MP3 时需要 FFmpeg（WAV 不需要）

不再需要 CMake、C++ 编译器或单独构建 MLX C++ 依赖。

如果需要 MP3：

```bash
brew install ffmpeg
```

## UV 环境管理

项目使用 UV 作为唯一推荐的 Python 环境、依赖和命令管理工具。Python 版本记录在 `.python-version`，完整依赖版本记录在 `uv.lock`。

```bash
uv sync --extra dev
```

UV 会在项目根目录创建 `.venv/`。后续命令统一通过 `uv run` 执行，不需要手动激活虚拟环境；`uv run` 也会检查环境是否与锁文件一致。

项目显式锁定 MLX `0.31.2`，与当前 Magenta RealTime 官方锁文件保持一致。不要单独升级 MLX；较新的不兼容版本会导致官方 `.mlxfn` 加载时报 `[import_function] Invalid string size`。如果环境曾被改动，运行：

```bash
uv sync --extra dev
```

查看完整 CLI 帮助：

```bash
uv run mrt-local -h
uv run mrt-local generate -h
uv run mrt-download -h
uv run mrt-serve -h
```

## 独立模型下载命令

默认下载 Small：

```bash
uv run mrt-download
```

选择模型：

```bash
uv run mrt-download mrt2_small
uv run mrt-download mrt2_base
uv run mrt-download mrt2_small mrt2_base
```

所有内容都保存在项目目录下：

```text
models/
├── models/
│   ├── mrt2_small/
│   │   ├── mrt2_small.mlxfn
│   │   └── mrt2_small_state.safetensors
│   └── mrt2_base/
│       ├── mrt2_base.mlxfn
│       └── mrt2_base_state.safetensors
└── resources/
    ├── musiccoca/
    └── spectrostream/
```

默认下载根目录是当前项目的 `models/`，不是用户 Documents、Home 或全局缓存目录。下载内容已被 `.gitignore` 排除。

只有显式指定时才会使用其他位置：

```bash
uv run mrt-download mrt2_small --model-root /absolute/custom/path
```

## CLI 生成

使用 UV 运行统一 CLI：

```bash
uv run mrt-local generate \
  --model mrt2_small \
  --prompt "minimal techno" \
  --duration 5 \
  --output output.wav
```

生成 Base 模型音频：

```bash
uv run mrt-local generate --model mrt2_base --prompt "ambient pads"
```

使用参考音频作为 MusicCoCa 风格条件：

```bash
uv run mrt-local generate \
  --model mrt2_small \
  --reference-audio reference.wav \
  --duration 10 \
  --output styled.wav
```

文本和参考音频既可单独使用，也可同时使用。同时提供时，服务会按归一化后的权重混合两者的 MusicCoCa embedding（默认 `0.5/0.5`）：

```bash
uv run mrt-local generate \
  --prompt "ambient pads" \
  --reference-audio reference.wav \
  --text-weight 1 --audio-weight 3 \
  --output mixed.wav
```

参考音频用于控制风格，不代表音频续写或编辑。

生成 MP3。格式默认根据输出扩展名推断，也可以显式指定：

```bash
uv run mrt-local generate \
  --prompt "ambient pads" \
  --output output.mp3 \
  --format mp3 \
  --bitrate 192
```

可以覆盖官方 MLX 推理参数；不常用参数均有默认值：

```bash
uv run mrt-local generate \
  --prompt "ambient techno" \
  --temperature 1.1 \
  --top-k 40 \
  --cfg-musiccoca 3.0
```

查看解析后的配置，不加载模型：

```bash
uv run mrt-local info --model mrt2_base
```

## 独立服务启动命令

```bash
uv run mrt-serve --model mrt2_small
```

需要时可指定其他端口；省略 `--port` 时仍使用默认端口 `8765`：

```bash
uv run mrt-serve --model mrt2_small --port 9000
# 等价的统一 CLI 命令：
uv run mrt-local serve --model mrt2_small --port 9000
```

默认只监听 `127.0.0.1:8765`。模型在 FastAPI lifespan 启动阶段加载并预热一次；所有请求共用该实例。同一时间由一个完整生成或流式会话独占模型；重叠的 HTTP 请求返回 `409 Conflict`，WebSocket 请求返回 `model_busy`。

浏览器跨域访问默认不受来源限制：HTTP 允许任意来源、方法和请求头，但不启用跨域 Cookie 凭据模式；WebSocket 不过滤 `Origin`。这不会改变网络可达范围，默认 `127.0.0.1` 仍然只允许本机连接。

服务入口：

- Swagger UI：<http://127.0.0.1:8765/docs>
- OpenAPI JSON：<http://127.0.0.1:8765/openapi.json>
- 健康检查：<http://127.0.0.1:8765/health>
- 运行信息：<http://127.0.0.1:8765/info>
- WebSocket：`ws://127.0.0.1:8765/ws/generate`
- 流式 WebSocket：`ws://127.0.0.1:8765/ws/stream`

生成音频：

```bash
curl -X POST http://127.0.0.1:8765/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"minimal techno","duration":5}' \
  --output output.wav
```

生成 MP3：

```bash
curl -X POST http://127.0.0.1:8765/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"minimal techno","duration":5,"format":"mp3","bitrate":192}' \
  --output output.mp3
```

上传参考音频生成：

```bash
curl -X POST http://127.0.0.1:8765/generate/audio \
  -F 'audio=@reference.wav' \
  -F 'duration=10' \
  -F 'format=wav' \
  --output styled.wav
```

通过 HTTP Streaming 接收 48 kHz 双声道 float32le 裸 PCM：

```bash
curl --no-buffer -X POST http://127.0.0.1:8765/stream \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"ambient pads","duration":10,"chunk_frames":5}' \
  --output output.f32le
```

完整字段和响应说明见 [API 文档](docs/API.md)，流式协议见 [流式生成](docs/STREAMING.md)，WebSocket 完整文件协议见 [WebSocket API](docs/WEBSOCKET.md)，模型差异、硬件要求和参数解释见 [模型与推理参数](docs/MODELS.md)。

## 测试

单元测试通过假后端验证生命周期、精确时长音频、CLI、HTTP、WebSocket、有状态流式生成、取消、编码和 OpenAPI，不需要下载真实模型：

```bash
uv run pytest
```

真实端到端测试需要先下载模型，然后运行 CLI 或启动服务。

## 项目结构

```text
.
├── mrt_local/
│   ├── api.py                # HTTP 传输适配器与 OpenAPI
│   ├── ws.py                 # WebSocket 传输适配器
│   ├── streaming_ws.py       # 有状态 PCM 流式 WebSocket
│   ├── pcm.py                # 裸 PCM 序列化
│   ├── schemas.py            # HTTP/WebSocket 共用 JSON 请求模型
│   ├── cli.py                # CLI 传输适配器
│   ├── core.py               # 核心命令、配置、校验与结果
│   ├── config.py             # 运行时配置与默认路径
│   ├── backend.py            # 后端端口与 Magenta/MLX 适配器
│   ├── encoding.py           # WAV/MP3 共享编码层
│   ├── service.py            # 与传输协议无关的生成用例
│   └── download.py           # 模型下载命令
├── tests/
├── docs/
│   ├── API.md               # API 使用说明
│   ├── ARCHITECTURE.md       # 分层设计与扩展方式
│   ├── MODELS.md            # 模型、硬件与推理参数
│   ├── WEBSOCKET.md         # WebSocket 消息协议
│   └── STREAMING.md         # HTTP/WebSocket PCM 流式协议
├── pyproject.toml            # UV 项目配置与独立命令
└── uv.lock                   # 完整依赖锁文件
```

## 当前限制

- 仅支持 macOS Apple Silicon 和 MLX
- 服务进程启动后固定使用一个模型；切换模型需要重启服务
- 同一时间只允许一个普通生成或流式会话，不提供多模型并发
- 流式接口支持固定参数和提前停止；暂不支持流中修改提示词/参数、MIDI、OSC、内置播放器和 GUI

当前锁定环境使用 `magenta-rt 2.0.3`，推理封装基于其 `MagentaRT2StdMlxfn`、`embed_style()` 和有状态 `generate()` API。

分层边界和新增 Socket 等传输外壳的方法见 [项目架构](docs/ARCHITECTURE.md)。
