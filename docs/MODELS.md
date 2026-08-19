# MRT2 模型与推理参数

本文说明本项目支持的 Magenta RealTime 2 模型、运行要求和推理参数。内容已于 2026-08-19 对照 Magenta 官方模型卡、文档、Python/MLX 源码以及本项目锁文件校验；当前锁定版本为 `magenta-rt[mlx] 2.0.3`。

当前官方仓库的 UV 锁文件使用 MLX `0.31.2`，本项目也显式锁定这个版本。MLX `0.32.1` 无法导入当前官方 Small/Base `.mlxfn`，会报 `[import_function] Invalid string size`。遇到该错误先运行 `uv sync --extra dev`；如果版本恢复后仍失败，再用 `uv run mrt-download <模型名>` 重新下载模型。

## 模型概览

Magenta RealTime 2 是实时音乐生成模型。官方系统由三部分组成：

- SpectroStream：48 kHz 双声道音频编解码器，以 25 Hz 帧率工作。
- MusicCoCa：把文本或参考音频编码为风格条件。
- MRT2：decoder-only Transformer/Depthformer，根据条件自回归生成音频 token。

官方模型能使用文本、音频和 MIDI 条件。本项目完整接入了文本、参考音频、MIDI 文件与结构化音符/鼓点事件；这些条件可单独使用或组合使用，并可输出 48 kHz 双声道 WAV、MP3，或通过 HTTP Streaming/WebSocket 输出有状态 `float32le` PCM。

参考音频由 MusicCoCa 转为风格 embedding：自动转单声道、重采样、按 10 秒片段处理，并在默认 `pool_across_time=true` 时对长音频的片段 embedding 求平均。它用于引导新生成音乐的风格，不保证旋律延续，也不是音频编辑功能。

文本与音频同时输入时，本项目分别调用官方 `embed_style`，再对处于同一 MusicCoCa 空间的 embedding 做加权平均。默认文本/音频权重为 `0.5/0.5`，并自动归一化。官方示例展示了 embedding 平均，但没有规定文本/音频融合权重；因此这是本项目提供的组合策略，而不是官方预设参数。混合输入要求 `pool_across_time=true`，以保证 embedding 形状一致。

## 硬件要求与模型选择

| 模型 | 参数量 | 有效感受野 | 官方定位 |
|---|---:|---:|---|
| `mrt2_small` | 约 230M | 20 秒 | 速度更快，适合多数 Apple Silicon Mac |
| `mrt2_base` | 约 2.4B | 20 秒 | 质量更高，需要性能更强的设备 |

官方把“实时”定义为生成速度快于播放速度。当前官方设备表如下：

| Apple Silicon 设备 | `mrt2_small`（230M） | `mrt2_base`（2.4B） |
|---|---:|---:|
| M5 Max | ✅ 实时 | ✅ 实时 |
| M3 Max | ✅ 实时 | ✅ 实时 |
| M2 Max | ✅ 实时 | ✅ 实时 |
| M4 Pro | ✅ 实时 | ✅ 实时 |
| M2 Pro | ✅ 实时 | ❌ 仅离线 |
| M1 Pro | ✅ 实时 | ❌ 仅离线 |
| M4 Air | ✅ 实时 | ❌ 仅离线 |
| M3 Air | ✅ 实时 | ❌ 仅离线 |
| M1 Air | ✅ 实时 | ❌ 仅离线 |

这里的“仅离线”表示官方基准没有达到实时速度，不代表模型无法加载或生成。官方说明两种模型都能通过 Python 库在任意 Apple Silicon Mac 上执行离线推理；本项目只支持其中的 macOS MLX 路径，不支持官方另外提供的 NVIDIA GPU 路径。

选择建议：

- Air、M1 Pro、M2 Pro 或未明确列入 Base 实时支持表的设备，默认选择 `mrt2_small`。
- M4 Pro、M2 Max、M3 Max、M5 Max 可以在官方实时支持范围内选择 `mrt2_base`，以换取更高质量。
- 其他 Apple Silicon 设备仍可尝试 `mrt2_base` 离线生成，但实时性能没有官方保证。
- 官方没有给出统一内存下限，因此本文不推测 RAM 数字；能否实时应以设备实测和官方表格为准。

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
| 文本提示词 | `--prompt` | `prompt` | 无；存在参考音频或控制事件时可省略 | 每次生成 |
| 参考音频 | `--reference-audio` | multipart `audio` | 无 | 每次生成 |
| 文本/音频权重 | `--text-weight` / `--audio-weight` | `text_weight` / `audio_weight` | `0.5 / 0.5` | 两种条件混合时 |
| 时长（秒） | `--duration` | `duration` | `10` | 每次生成 |
| 采样温度 | `--temperature` | `temperature` | `1.3` | 启动默认值，可按请求覆盖 |
| Top-k | `--top-k` | `top_k` | `40` | 启动默认值，可按请求覆盖 |
| 风格遵循强度 | `--cfg-musiccoca` | `cfg_musiccoca` | `3.0` | 启动默认值，可按请求覆盖 |
| MIDI 音符引导强度（高级） | `--cfg-notes` | `cfg_notes` | `1.0` | 有音符控制时生效 |
| 鼓点序列引导强度（高级） | `--cfg-drums` | `cfg_drums` | `1.0` | 有鼓点控制时生效 |
| 预热步数 | `--warmup-steps` | 无 | `5` | 进程启动 |
| embedding 随机种子 | `--seed` | `seed` | `0` | 启动默认值，可按请求覆盖 |
| MusicCoCa mapper | `--use-mapper` / `--no-use-mapper` | `use_mapper` | `true` | 启动默认值，可按请求覆盖 |
| 时间维聚合 | `--pool-across-time` / `--no-pool-across-time` | `pool_across_time` | `true` | 启动默认值，可按请求覆盖 |

服务启动参数决定 API 的默认值。`POST /generate` 省略可覆盖字段或传 `null` 时，会继承这些默认值。当前有效性约束为：`duration` 大于 0 且不超过 300，`temperature` 大于 0，`top_k` 大于等于 1，三个 CFG 值必须是有限数。

### 参数含义

- `temperature` 控制每一步采样的随机度。较高通常带来更多变化和意外，较低通常更保守、集中。它不控制音量、速度或曲长，必须大于 `0`。
- `top_k` 表示每一步只保留概率最高的 K 个候选，再从中采样。较小会限制选择范围，较大允许更多低概率候选；它和 `temperature` 一起影响随机性。
- `cfg_musiccoca` 表示模型对文本或参考音频风格条件的遵循强度。较高通常更贴近条件，但不保证质量单调提高，普通使用建议从默认 `3.0` 开始。
- `cfg_notes` 调节模型遵循外部音符序列的强度。事件会转换为 128 个 MIDI 音高、每帧 40 ms 的 piano-roll；较高值通常更贴合指定音高与起止时间，建议从 `1.0` 开始。
- `cfg_drums` 调节模型遵循外部鼓点触发序列的强度。每个 40 ms 帧表示“触发鼓点”或“未触发/未约束”；较高值通常更贴合指定节奏，建议从 `1.0` 开始。当前官方输入不区分底鼓、军鼓等鼓件类别。
- 三项 `cfg_*` 在 MRT2 中作为离散的引导控制 token 输入，而不是额外扩展一批 CFG 推理。官方实现先把值截断到 `[-1, 7]`：MusicCoCa/notes 以 `0.2` 为步长，drums 以 `1.0` 为步长量化。因此超出范围或落在同一量化档位的数值不会提供更多精细控制。
- `warmup_steps` 是模型启动时执行的空推理步数，用于预热 MLX kernel。增加它可能延长启动、降低第一次正式生成的冷启动抖动，但不会改变生成内容。
- `seed` 只为文本 mapper 使用的随机噪声设种子，而且仅在 `use_mapper=true` 且存在文本条件时生效。音频 token 的自回归采样没有通过这个字段设置全局种子，因此相同参数不保证生成完全一致。
- `use_mapper` 把文本 embedding 映射到音频 embedding 空间，只影响文本条件，不影响纯参考音频。官方底层方法默认关闭，但官方 MLX 生成命令启用；本项目因此默认 `true`，混合文本和音频时尤其建议启用。
- `pool_across_time` 仅影响参考音频 embedding。启用时，把参考音频各时间片求平均，得到单一整体风格；关闭时保留时间维。文本和参考音频混合需要相同形状，因此混合输入强制为 `true`。

### 项目层参数

- `text_weight` / `audio_weight` 控制文本和参考音频 embedding 的相对混合比例。服务只对实际提供的输入取权重并自动归一化，所以 `1/3` 与 `0.25/0.75` 等价；权重 `0` 表示该输入不参与最终 embedding，两项有效权重不能同时为 `0`。
- `duration` 是服务层的目标输出时长，范围 `(0, 300]` 秒。底层按 25 Hz（每帧 40 ms）向上取整生成，再将结果精确裁剪到目标采样数。
- `chunk_frames` / `chunkFrames` 只用于流式接口，表示一次底层调用生成多少个 40 ms 帧。较小可以降低停止和首片延迟，但调用开销更高；默认 `5`，约 200 ms。
- `format` 和 `bitrate` 是输出编码参数，不参与模型推理。WAV 使用 48 kHz 双声道 float，MP3 默认 `192 kbps`。
- `notes`、`drums`、`notes_mode` 和 `drums_mode` 是项目层的易用输入，服务会将其转换为官方逐帧条件。完整格式见[音符与鼓点控制](CONTROL.md)。

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

- `state` 是生成器在分块生成期间传递的内部状态。完整文件路径以 `state=None` 开始并在单次调用结束后丢弃返回状态；HTTP Streaming 和 WebSocket 流式会话会保存返回状态并传给下一个分片，从而保持连续生成。
- `checkpoint`、`bits` 和 `use_mlxfn` 用于官方脚本选择原始 checkpoint、量化方式或导出执行路径。本项目固定使用已下载的官方 `.mlxfn` 导出模型，因此这些选项不适用。
- `frames` 是官方底层生成长度。本项目对外使用更直观的 `duration`，按官方 25 Hz 帧率换算，并将完整文件或最后一个 PCM 分片精确裁剪到请求时长；流式接口另外暴露 `chunk_frames` / `chunkFrames` 控制每次底层调用的分片大小。

## 已知限制与许可

官方模型卡说明训练数据约 7.1 万小时、以器乐为主，因此人声能力有限。模型权重采用 CC BY 4.0，官方代码采用 Apache 2.0；分发或再利用模型时应分别遵守相应许可。

## 官方资料

- [Hugging Face 官方模型卡](https://huggingface.co/google/magenta-realtime-2)
- [官方模型与硬件说明](https://github.com/magenta/magenta-realtime/blob/main/docs/models.md)
- [官方推理说明](https://github.com/magenta/magenta-realtime/blob/main/docs/inference.md)
- [官方 MLX 生成命令源码](https://github.com/magenta/magenta-realtime/blob/main/magenta_rt/mlx/generate.py)
- [官方 MLX 系统接口源码](https://github.com/magenta/magenta-realtime/blob/main/magenta_rt/mlx/system.py)
