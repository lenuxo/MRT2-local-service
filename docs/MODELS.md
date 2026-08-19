# MRT2 模型与推理参数

本文说明本项目支持的 Magenta RealTime 2 模型、运行要求和推理参数。内容已于 2026-08-19 对照 Magenta 官方模型卡、文档和 Python/MLX 源码校验；对应官方仓库提交 `694a545e4ba0b88bf1150137b129582166d3e07f`，本项目使用 `magenta-rt[mlx] 2.0.x`。

当前官方仓库的 UV 锁文件使用 MLX `0.31.2`，本项目也显式锁定这个版本。MLX `0.32.1` 无法导入当前官方 Small/Base `.mlxfn`，会报 `[import_function] Invalid string size`。遇到该错误先运行 `uv sync --extra dev`；如果版本恢复后仍失败，再用 `uv run mrt-download <模型名>` 重新下载模型。

## 模型概览

Magenta RealTime 2 是实时音乐生成模型。官方系统由三部分组成：

- SpectroStream：48 kHz 双声道音频编解码器，以 25 Hz 帧率工作。
- MusicCoCa：把文本或参考音频编码为风格条件。
- MRT2：decoder-only Transformer/Depthformer，根据条件自回归生成音频 token。

官方模型能使用文本、音频和 MIDI 条件。本项目已经支持文本或参考音频风格条件，输出 48 kHz 双声道 WAV 或 MP3；MIDI 和实时流式交互尚未纳入本地服务接口。

参考音频由 MusicCoCa 转为风格 embedding：自动转单声道、重采样、按 10 秒片段处理，并在默认 `pool_across_time=true` 时对长音频的片段 embedding 求平均。它用于引导新生成音乐的风格，不保证旋律延续，也不是音频编辑功能。

| 模型 | 参数量 | 有效感受野 | 官方定位 |
|---|---:|---:|---|
| `mrt2_small` | 约 230M | 20 秒 | 速度更快，适合多数 Apple Silicon Mac |
| `mrt2_base` | 约 2.4B | 20 秒 | 质量更高，需要性能更强的设备 |

官方给出的实时运行表显示，Small 覆盖列出的 Air、Pro 和 Max 芯片；Base 只在 M5 Max、M3 Max、M2 Max 和 M4 Pro 上标为实时。未标为实时不代表完全不能运行，但可能无法达到实时速度。

模型文件默认下载到项目根目录的 `models/`，不会写入用户 Documents 目录：

```bash
uv run mrt-download mrt2_small
uv run mrt-download mrt2_base
```

## 支持的参数

常用参数是 `model`、`prompt`、`duration`、`temperature`、`top_k` 和 `cfg_musiccoca`。其余参数均提供经过官方实现校验的默认值，可以省略。

| 含义 | CLI 参数 | API 字段 | 默认值 | 作用域 |
|---|---|---|---:|---|
| 模型 | `--model` | 不可按请求切换 | `mrt2_small` | 进程启动 |
| 文本提示词 | `--prompt` | `prompt` | 无，必填 | 每次生成 |
| 时长（秒） | `--duration` | `duration` | `10` | 每次生成 |
| 采样温度 | `--temperature` | `temperature` | `1.3` | 启动默认值，可按请求覆盖 |
| Top-k | `--top-k` | `top_k` | `40` | 启动默认值，可按请求覆盖 |
| 风格条件 CFG | `--cfg-musiccoca` | `cfg_musiccoca` | `3.0` | 启动默认值，可按请求覆盖 |
| 音符条件 CFG | `--cfg-notes` | `cfg_notes` | `1.0` | 启动默认值，可按请求覆盖 |
| 鼓条件 CFG | `--cfg-drums` | `cfg_drums` | `1.0` | 启动默认值，可按请求覆盖 |
| 预热步数 | `--warmup-steps` | 无 | `5` | 进程启动 |
| embedding 随机种子 | `--seed` | `seed` | `0` | 启动默认值，可按请求覆盖 |
| MusicCoCa mapper | `--use-mapper` / `--no-use-mapper` | `use_mapper` | `true` | 启动默认值，可按请求覆盖 |
| 时间维聚合 | `--pool-across-time` / `--no-pool-across-time` | `pool_across_time` | `true` | 启动默认值，可按请求覆盖 |

服务启动参数决定 API 的默认值。`POST /generate` 省略可覆盖字段或传 `null` 时，会继承这些默认值。当前有效性约束为：`duration` 大于 0 且不超过 300，`temperature` 大于 0，`top_k` 大于等于 1，三个 CFG 值必须是有限数。

### 参数含义

- `temperature` 控制采样分布的随机程度。它是生成行为参数，不是音量或速度参数。
- `top_k` 将每步采样限制在概率最高的若干候选中。
- `cfg_musiccoca` 控制输出对文本风格 embedding 的引导强度。
- `cfg_notes` 和 `cfg_drums` 是官方生成器接受的音符与鼓条件 CFG。当前服务没有提供 MIDI/鼓条件输入，因此保留它们主要是为了与官方接口完整对齐。
- `warmup_steps` 是模型加载时的预热步数，只能在 CLI 启动时设置，不能由单个 HTTP 请求修改。
- `seed` 是 `embed_style()` 的 MusicCoCa embedding 种子，不是完整音频采样过程的全局随机种子，因此不应把它理解为完全确定性生成开关。
- `pool_across_time` 控制是否把 MusicCoCa embedding 沿时间维聚合。
- `use_mapper` 控制是否应用 MusicCoCa mapper。官方底层方法默认值是 `false`，但官方 MLX 生成命令显式启用它；本项目采用与官方生成命令一致的 `true`。

CLI 示例：

```bash
uv run mrt-local generate \
  --model mrt2_small \
  --prompt "minimal techno, deep bass, sparse percussion" \
  --duration 12 \
  --temperature 1.1 \
  --top-k 40 \
  --cfg-musiccoca 3.0 \
  --output output.wav
```

API 示例：

```json
{
  "prompt": "minimal techno, deep bass, sparse percussion",
  "duration": 12,
  "temperature": 1.1,
  "top_k": 40,
  "cfg_musiccoca": 3.0,
  "cfg_notes": 1.0,
  "cfg_drums": 1.0,
  "seed": 0,
  "use_mapper": true,
  "pool_across_time": true
}
```

## 官方接口中未作为服务参数暴露的项目

以下内容不是当前 `.mlxfn` 服务的逐次推理参数：

- `state` 是生成器在分块生成期间传递的内部状态。本项目一次请求完成整段生成，因此由 Engine 内部管理。
- `checkpoint`、`bits` 和 `use_mlxfn` 用于官方脚本选择原始 checkpoint、量化方式或导出执行路径。本项目固定使用已下载的官方 `.mlxfn` 导出模型，因此这些选项不适用。
- `frames` 是官方底层生成长度。本项目对外使用更直观的 `duration`，按官方 25 Hz 帧率换算，并将最终 WAV 精确裁剪到请求时长。

## 已知限制与许可

官方模型卡说明训练数据约 7.1 万小时、以器乐为主，因此人声能力有限。模型权重采用 CC BY 4.0，官方代码采用 Apache 2.0；分发或再利用模型时应分别遵守相应许可。

## 官方资料

- [Hugging Face 官方模型卡](https://huggingface.co/google/magenta-realtime-2)
- [官方模型与硬件说明](https://github.com/magenta/magenta-realtime/blob/main/docs/models.md)
- [官方推理说明](https://github.com/magenta/magenta-realtime/blob/main/docs/inference.md)
- [官方 MLX 生成命令源码](https://github.com/magenta/magenta-realtime/blob/main/magenta_rt/mlx/generate.py)
- [官方 MLX 系统接口源码](https://github.com/magenta/magenta-realtime/blob/main/magenta_rt/mlx/system.py)
