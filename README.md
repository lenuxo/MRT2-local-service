# MRT2 本地服务

MRT2 本地服务是一个运行在 macOS Apple Silicon 上的 Magenta RealTime 2 服务，支持 Small 和 Base 两种模型，并通过同一个可执行文件提供命令行和本地 HTTP API。两种调用方式共享基于 Google 官方 `magentart::core::MLXEngine` 的 `MrtEngine` 封装；服务启动时只加载一次所选模型，后续生成请求按顺序执行。

## 环境要求

- 运行 macOS 14 或更高版本的 Apple Silicon Mac
- Xcode Command Line Tools
- Xcode Metal Toolchain；如果尚未安装，可运行 `xcodebuild -downloadComponent MetalToolchain`
- CMake 3.27 或更高版本；如果新版出现上游依赖兼容问题，可按照官方建议使用 CMake `<3.28`
- Git；首次运行 CMake 配置时需要联网下载依赖
- MRT2 Small 或 Base 的 MLX 模型，以及共享的 MusicCoCa 资源

当前仅支持 MLX 后端、文本提示词、指定时长生成和 48 kHz 双声道浮点 WAV 输出。

## 准备模型

先安装 Magenta 官方下载工具，再使用项目脚本下载模型。所有文件都会保存在当前项目的 `models/` 目录中，不会写入用户的 Documents 文件夹：

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install "magenta-rt[mlx]"

# 下载单个模型
./scripts/download_models.sh mrt2_small
./scripts/download_models.sh mrt2_base

# 一次下载两个模型
./scripts/download_models.sh mrt2_small mrt2_base
```

下载后的目录结构如下：

```text
./models/
├── models/
│   ├── mrt2_small/mrt2_small.mlxfn
│   └── mrt2_base/mrt2_base.mlxfn
└── resources/musiccoca/...
```

模型权重已经通过 `.gitignore` 排除，不会被提交到 Git。必要时可通过环境变量覆盖路径：

```bash
export MRT_MODEL_PATH=/absolute/path/to/mrt2_small.mlxfn
export MRT_RESOURCES_PATH=/absolute/path/to/resources
export MRT_MODEL_ROOT=/absolute/path/to/download-root
```

路径优先级为：命令行参数、环境变量、项目内默认路径。

## 构建

第一次配置会下载固定版本的 Magenta 官方源码及 MLX、TensorFlow Lite、SentencePiece 等依赖，因此可能需要较长时间：

```bash
cmake -B build
cmake --build build --target mrt -j
./build/mrt --help
```

如果本地已有 Magenta 官方仓库，可复用该目录，避免重新下载 Magenta 源码：

```bash
git clone --recurse-submodules https://github.com/magenta/magenta-realtime.git
cmake -B build -DMAGENTART_SOURCE_DIR=/absolute/path/to/magenta-realtime
cmake --build build --target mrt -j
```

安装到系统目录：

```bash
cmake --install build --prefix /usr/local
```

根据目标目录权限，这一步可能需要管理员权限。

## 命令行使用

生成 WAV 文件：

```bash
./build/mrt generate \
  --prompt "minimal techno" \
  --model mrt2_small \
  --duration 5 \
  --output test.wav
```

`--model` 支持 `mrt2_small` 和 `mrt2_base`。还可使用 `--model-path` 和 `--resources-path` 覆盖默认路径。默认生成时长为 10 秒，默认输出文件为 `./output.wav`。

查看最终解析出的配置，但不加载模型：

```bash
./build/mrt info --model mrt2_base
```

## 本地 HTTP API

启动常驻服务。服务会先完成模型加载，再开始监听端口：

```bash
./build/mrt serve --model mrt2_small
```

默认只监听 `127.0.0.1:8765`。如需修改地址或端口，必须显式指定：

```bash
./build/mrt serve --host 127.0.0.1 --port 9000
```

查看健康状态和运行信息：

```bash
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/info
```

服务运行期间可访问中文使用说明和 OpenAPI 3.1 规范：

```text
http://127.0.0.1:8765/docs
http://127.0.0.1:8765/openapi.json
```

完整接口说明请参阅 [API 文档](docs/API.md)。

生成音频：

```bash
curl -X POST http://127.0.0.1:8765/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"minimal techno","duration":5}' \
  --output output.wav
```

`POST /generate` 直接返回 `audio/wav`。`duration` 默认为 10 秒，必须大于 0，最大为 300 秒。由于 `MLXEngine` 的生成状态不支持并发生命周期调用，服务会通过互斥锁依次处理生成请求。

## 测试

完成项目配置后运行：

```bash
cmake --build build --target wav_test
ctest --test-dir build --output-on-failure
```

WAV 单元测试不需要模型。端到端生成测试必须准备真实的 MRT2 Small 或 Base 模型和共享资源；项目不会提供伪造的推理回退实现。

## 当前限制

- 仅支持 macOS Apple Silicon
- 仅支持 MRT2 Small/Base 和 MLX 后端
- 单进程只加载一个模型，生成请求串行执行
- 暂不支持流式输出、WebSocket、MIDI、OSC、实时播放、GUI、插件或身份认证
- 服务会在完整音频生成后一次性返回 WAV，不提供渐进式音频数据

推理生命周期依据 Magenta 官方 `examples/hello_mrt2` 实现，当前固定使用官方提交 `694a545e4ba0b88bf1150137b129582166d3e07f`。
