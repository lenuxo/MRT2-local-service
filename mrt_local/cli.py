from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

from . import __version__
from .config import RuntimeConfig, default_model_root
from .core import (
    DEFAULT_DURATION,
    DEFAULT_MODEL_NAME,
    DEFAULT_WARMUP_STEPS,
    SUPPORTED_MODELS,
    GenerateCommand,
    ModelConfig,
    ModelName,
    PromptComponent,
    SamplingConfig,
)
from .encoding import (
    SUPPORTED_AUDIO_FORMATS,
    decode_audio_file,
    encode_audio,
    infer_cli_encoding,
)
from . import parameter_docs as parameter_help

DEFAULT_SAMPLING = SamplingConfig()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mrt-local",
        description="使用本地 MLX 模型生成音乐，或启动 MRT2 HTTP API 服务。",
        epilog=(
            "示例：\n"
            "  uv run mrt-local generate --prompt \"ambient techno\" --duration 5\n"
            "  uv run mrt-local serve --model mrt2_small\n"
            "  uv run mrt-local info --model mrt2_base\n\n"
            "独立命令：\n"
            "  uv run mrt-download -h    查看模型下载帮助\n"
            "  uv run mrt-serve -h       查看服务启动帮助"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        title="可用命令",
        metavar="COMMAND",
    )

    generate = subparsers.add_parser(
        "generate",
        help="直接生成 WAV 或 MP3 音频",
        description="加载指定的 MRT2 模型，根据文本、参考音频或二者加权混合生成音频文件。",
        epilog=(
            "示例：\n"
            "  uv run mrt-local generate \\\n"
            "    --prompt \"minimal techno\" \\\n"
            "    --duration 5 \\\n"
            "    --model mrt2_small \\\n"
            "    --output output.wav"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    generate.add_argument("--prompt", metavar="TEXT", help="用于控制音乐风格的文本提示词")
    generate.add_argument(
        "--weighted-prompt",
        action="append",
        nargs=2,
        default=[],
        metavar=("WEIGHT", "TEXT"),
        help="高级多文本混合，可重复提供，例如 --weighted-prompt 2 'powerful drums'；不能与 --prompt 同时使用",
    )
    generate.add_argument("--reference-audio", type=Path, metavar="PATH", help="用于提取音乐风格的参考音频文件")
    generate.add_argument("--midi", type=Path, metavar="PATH", help="MIDI 控制文件；非鼓通道转为音符，第 10 通道转为鼓点")
    generate.add_argument("--notes-mode", choices=("guide", "strict"), default="guide", help="音符控制模式：guide 允许额外音高，strict 关闭未指定音高（默认：guide）")
    generate.add_argument("--drums-mode", choices=("guide", "strict"), default="guide", help="鼓点控制模式：guide 允许额外鼓点，strict 关闭未指定鼓点（默认：guide）")
    generate.add_argument("--midi-drums", action=argparse.BooleanOptionalAction, default=True, help="是否提取 MIDI 第 10 通道为鼓点控制（默认：启用）")
    generate.add_argument("--text-weight", type=float, default=0.5, metavar="FLOAT", help="混合时的文本权重（默认：0.5）")
    generate.add_argument("--audio-weight", type=float, default=0.5, metavar="FLOAT", help="混合时的参考音频权重（默认：0.5）")
    generate.add_argument("--duration", type=float, default=DEFAULT_DURATION, metavar="SECONDS", help="生成时长，必须大于 0 且不超过 300 秒（默认：10）")
    generate.add_argument("--output", type=Path, default=Path("output.wav"), metavar="PATH", help="音频输出路径（默认：output.wav）")
    generate.add_argument("--format", choices=SUPPORTED_AUDIO_FORMATS, default=None, help="输出格式；默认根据文件扩展名推断")
    generate.add_argument("--bitrate", type=int, default=None, metavar="KBPS", help="MP3 比特率，范围 32～320（默认：192）")
    generate.add_argument("--model", choices=SUPPORTED_MODELS, default=DEFAULT_MODEL_NAME, help="使用的 MRT2 模型（默认：mrt2_small）")
    generate.add_argument("--model-root", type=Path, default=default_model_root(), metavar="PATH", help="模型与资源根目录（默认：项目根目录/models）")
    _add_inference_arguments(generate)

    serve = subparsers.add_parser(
        "serve",
        help="启动本地 HTTP API",
        description="加载一次指定模型，然后启动常驻的 FastAPI 服务。",
        epilog="示例：\n  uv run mrt-local serve --model mrt2_small --port 8765",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_serve_arguments(serve)

    info = subparsers.add_parser(
        "info",
        help="显示最终解析的配置",
        description="显示模型、资源和后端配置，不加载模型。",
    )
    info.add_argument("--model", choices=SUPPORTED_MODELS, default=DEFAULT_MODEL_NAME, help="要检查的 MRT2 模型（默认：mrt2_small）")
    info.add_argument("--model-root", type=Path, default=default_model_root(), metavar="PATH", help="模型与资源根目录（默认：项目根目录/models）")
    _add_inference_arguments(info)
    return parser


def _add_serve_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default=DEFAULT_MODEL_NAME, help="服务使用的 MRT2 模型（默认：mrt2_small）")
    parser.add_argument("--model-root", type=Path, default=default_model_root(), metavar="PATH", help="模型与资源根目录（默认：项目根目录/models）")
    parser.add_argument("--host", default="127.0.0.1", metavar="HOST", help="监听地址（默认：127.0.0.1，仅本机访问）")
    parser.add_argument("--port", type=int, default=8765, metavar="PORT", help="监听端口，范围为 1～65535（默认：8765）")
    _add_inference_arguments(parser)


def _add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("官方 MLX 推理参数")
    group.add_argument("--temperature", type=float, default=DEFAULT_SAMPLING.temperature, metavar="FLOAT", help=f"采样随机度；越高变化越大，越低越保守（默认：{DEFAULT_SAMPLING.temperature}）")
    group.add_argument("--top-k", type=int, default=DEFAULT_SAMPLING.top_k, metavar="INT", help=f"每步只从概率最高的 K 个候选中采样；越小越集中（默认：{DEFAULT_SAMPLING.top_k}）")
    group.add_argument("--cfg-musiccoca", type=float, default=DEFAULT_SAMPLING.cfg_musiccoca, metavar="FLOAT", help="文本/参考音频风格遵循强度；通常保留默认值（默认：3.0）")
    group.add_argument("--cfg-notes", type=float, default=DEFAULT_SAMPLING.cfg_notes, metavar="FLOAT", help="遵循 MIDI 音符控制的强度（默认：1.0）")
    group.add_argument("--cfg-drums", type=float, default=DEFAULT_SAMPLING.cfg_drums, metavar="FLOAT", help="遵循 MIDI 鼓点控制的强度（默认：1.0）")
    group.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS, metavar="INT", help=parameter_help.WARMUP_STEPS + "（默认：5）")
    group.add_argument("--seed", type=int, default=DEFAULT_SAMPLING.seed, metavar="INT", help="文本 mapper 随机种子；不保证整段音频可复现（默认：0）")
    group.add_argument("--use-mapper", action=argparse.BooleanOptionalAction, default=DEFAULT_SAMPLING.use_mapper, help="把文本 embedding 映射到音频风格空间；仅影响文本（默认：启用）")
    group.add_argument("--pool-across-time", action=argparse.BooleanOptionalAction, default=DEFAULT_SAMPLING.pool_across_time, help="将参考音频各时间片平均为整体风格；混合输入必须启用（默认：启用）")


def _runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        model=ModelConfig(
            name=cast(ModelName, args.model),
            root=args.model_root.expanduser().resolve(),
            warmup_steps=args.warmup_steps,
        ),
        sampling=SamplingConfig(
            temperature=args.temperature,
            top_k=args.top_k,
            cfg_musiccoca=args.cfg_musiccoca,
            cfg_notes=args.cfg_notes,
            cfg_drums=args.cfg_drums,
            seed=args.seed,
            use_mapper=args.use_mapper,
            pool_across_time=args.pool_across_time,
        ),
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = _runtime_config(args)

    if args.command == "info":
        print("MRT2 本地服务")
        print(f"模型：{config.model.name}")
        print(f"模型目录：{config.model.model_dir}")
        print(f"资源目录：{config.model.resources_path}")
        print("后端：MLX")
        print(f"Temperature：{config.sampling.temperature}")
        print(f"Top-k：{config.sampling.top_k}")
        print(
            "CFG MusicCoCa/Notes/Drums："
            f"{config.sampling.cfg_musiccoca}/"
            f"{config.sampling.cfg_notes}/"
            f"{config.sampling.cfg_drums}"
        )
        print(f"预热步数：{config.model.warmup_steps}")
        print(f"Embedding seed：{config.sampling.seed}")
        print(f"Use mapper：{config.sampling.use_mapper}")
        print(f"Pool across time：{config.sampling.pool_across_time}")
        return

    if args.command == "generate":
        from .service import GenerationService

        service = GenerationService(config)
        try:
            encoding = infer_cli_encoding(args.output, args.format, args.bitrate)
            reference_audio = (
                decode_audio_file(args.reference_audio)
                if args.reference_audio is not None
                else None
            )
            if args.midi is not None:
                from .midi import decode_midi_file

                control = decode_midi_file(
                    args.midi,
                    notes_mode=args.notes_mode,
                    drums_mode=args.drums_mode,
                    include_drums=args.midi_drums,
                )
            else:
                control = None
            command = GenerateCommand(
                prompt=args.prompt,
                prompt_components=tuple(
                    PromptComponent(text, float(weight))
                    for weight, text in args.weighted_prompt
                ),
                reference_audio=reference_audio,
                text_weight=args.text_weight,
                audio_weight=args.audio_weight,
                duration=args.duration,
                control=control,
            )
            command.resolve(config.sampling)
            service.load()
            result = service.generate(command)
            encoded = encode_audio(result, encoding)
        except Exception as exc:
            print(f"错误：{exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encoded.data)
        print(f"模型：{config.model.name}")
        if args.prompt is not None:
            print(f"提示词：{args.prompt}")
        if args.weighted_prompt:
            print("高级文本风格：")
            for weight, text in args.weighted_prompt:
                print(f"  {weight} × {text}")
        if args.reference_audio is not None:
            print(f"参考音频：{args.reference_audio}")
        if args.midi is not None:
            print(f"MIDI 控制：{args.midi}")
        if args.prompt is not None and args.reference_audio is not None:
            print(f"文本/音频权重：{args.text_weight:g}/{args.audio_weight:g}")
        print(f"时长：{args.duration:g} 秒")
        print(f"格式：{encoded.format}")
        print(f"已生成：{args.output}")
        return

    if not 1 <= args.port <= 65535:
        raise SystemExit("错误：端口必须在 1 到 65535 之间")
    import uvicorn
    from .api import create_app

    uvicorn.run(create_app(config), host=args.host, port=args.port)


def serve_main(argv: list[str] | None = None) -> None:
    """`uv run mrt-serve` 的独立服务启动入口。"""
    parser = argparse.ArgumentParser(
        prog="mrt-serve",
        description="加载本地 MRT2 模型并启动常驻 HTTP API 服务。",
        epilog=(
            "示例：\n"
            "  uv run mrt-serve --model mrt2_small\n"
            "  uv run mrt-serve --model mrt2_base --port 9000\n\n"
            "启动后访问：\n"
            "  API 文档：http://127.0.0.1:8765/docs\n"
            "  健康检查：http://127.0.0.1:8765/health"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_serve_arguments(parser)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("错误：端口必须在 1 到 65535 之间")

    import uvicorn
    from .api import create_app

    config = _runtime_config(args)
    uvicorn.run(create_app(config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
