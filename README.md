# MRT2 本地服务（Python）

这是一个面向 macOS Apple Silicon 的本地 Magenta RealTime 2 服务。项目现已完全迁移到 Python 技术栈：

- 使用 Magenta 官方 `magenta-rt[mlx]` Python 包执行推理
- 使用 FastAPI 提供 HTTP API、OpenAPI 规范和 Swagger UI
- 使用标准 Python CLI 同时提供命令行生成和常驻服务
- CLI 与 API 共用同一个 `MrtEngine`，没有两套推理逻辑
- 支持 `mrt2_small` 和 `mrt2_base`
- 模型和共享资源保存在项目的 `models/` 目录中

## 环境要求

- Apple Silicon Mac
- macOS 14 或更高版本
- Python 3.11 或 3.12
- 推荐使用 `uv`

不再需要 CMake、C++ 编译器或单独构建 MLX C++ 依赖。

## 安装

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

也可以使用普通 `pip`：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## 下载模型

```bash
# 下载单个模型
python scripts/download_models.py mrt2_small
python scripts/download_models.py mrt2_base

# 一次下载两个模型
python scripts/download_models.py mrt2_small mrt2_base
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
    └── musiccoca/
```

下载内容已被 `.gitignore` 排除。可通过 `MRT_MODEL_ROOT` 或 `--model-root` 使用其他根目录。

## CLI 生成

项目根目录提供了 `./mrt` 启动器：

```bash
./mrt generate \
  --model mrt2_small \
  --prompt "minimal techno" \
  --duration 5 \
  --output output.wav
```

如果已经执行可编辑安装，也可以使用：

```bash
mrt-local generate --model mrt2_base --prompt "ambient pads"
python -m mrt_local generate --model mrt2_small --prompt "disco funk"
```

查看解析后的配置，不加载模型：

```bash
./mrt info --model mrt2_base
```

## HTTP 服务

```bash
./mrt serve --model mrt2_small
```

默认只监听 `127.0.0.1:8765`。模型在 FastAPI lifespan 启动阶段加载并预热一次；所有请求共用该实例，推理通过锁串行执行。

服务入口：

- Swagger UI：<http://127.0.0.1:8765/docs>
- OpenAPI JSON：<http://127.0.0.1:8765/openapi.json>
- 健康检查：<http://127.0.0.1:8765/health>
- 运行信息：<http://127.0.0.1:8765/info>

生成音频：

```bash
curl -X POST http://127.0.0.1:8765/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"minimal techno","duration":5}' \
  --output output.wav
```

完整字段和响应说明见 [API 文档](docs/API.md)。

## 测试

单元测试通过假后端验证生命周期、精确时长 WAV、CLI、API 和 OpenAPI，不需要下载真实模型：

```bash
pytest
```

真实端到端测试需要先下载模型，然后运行 CLI 或启动服务。

## 项目结构

```text
.
├── mrt                       # 项目内 CLI 启动器
├── mrt_local/
│   ├── api.py                # FastAPI 与 OpenAPI
│   ├── cli.py                # CLI
│   ├── config.py             # 模型与路径配置
│   └── engine.py             # 共享 Magenta/MLX 推理封装
├── scripts/
│   └── download_models.py    # 项目内模型下载器
├── tests/
├── docs/API.md
└── pyproject.toml
```

## 当前限制

- 仅支持 macOS Apple Silicon 和 MLX
- 服务进程启动后固定使用一个模型；切换模型需要重启服务
- 推理串行执行，不提供多模型并发
- 暂不支持 WebSocket、流式 PCM、MIDI、OSC、实时播放和 GUI

当前推理封装依据 Magenta 官方提交 `694a545e4ba0b88bf1150137b129582166d3e07f` 的 `MagentaRT2StdMlxfn`、`embed_style()` 和 `generate()` API。
