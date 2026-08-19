from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

from . import __version__
from .config import SUPPORTED_MODELS, EngineConfig, ModelName, default_model_root


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
        help="直接生成 WAV 音频",
        description="加载指定的 MRT2 模型，根据文本提示词直接生成 WAV 文件。",
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
    generate.add_argument("--prompt", required=True, metavar="TEXT", help="用于控制音乐风格的文本提示词（必填）")
    generate.add_argument("--duration", type=float, default=10, metavar="SECONDS", help="生成时长，必须大于 0 且不超过 300 秒（默认：10）")
    generate.add_argument("--output", type=Path, default=Path("output.wav"), metavar="PATH", help="WAV 输出路径（默认：output.wav）")
    generate.add_argument("--model", choices=SUPPORTED_MODELS, default="mrt2_small", help="使用的 MRT2 模型（默认：mrt2_small）")
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
    info.add_argument("--model", choices=SUPPORTED_MODELS, default="mrt2_small", help="要检查的 MRT2 模型（默认：mrt2_small）")
    info.add_argument("--model-root", type=Path, default=default_model_root(), metavar="PATH", help="模型与资源根目录（默认：项目根目录/models）")
    _add_inference_arguments(info)
    return parser


def _add_serve_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="mrt2_small", help="服务使用的 MRT2 模型（默认：mrt2_small）")
    parser.add_argument("--model-root", type=Path, default=default_model_root(), metavar="PATH", help="模型与资源根目录（默认：项目根目录/models）")
    parser.add_argument("--host", default="127.0.0.1", metavar="HOST", help="监听地址（默认：127.0.0.1，仅本机访问）")
    parser.add_argument("--port", type=int, default=8765, metavar="PORT", help="监听端口，范围为 1～65535（默认：8765）")
    _add_inference_arguments(parser)


def _add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("官方 MLX 推理参数")
    group.add_argument("--temperature", type=float, default=1.3, metavar="FLOAT", help="采样温度，必须大于 0（官方默认：1.3）")
    group.add_argument("--top-k", type=int, default=40, metavar="INT", help="Top-k 采样阈值，必须大于等于 1（官方默认：40）")
    group.add_argument("--cfg-musiccoca", type=float, default=3.0, metavar="FLOAT", help="MusicCoCa 风格条件 CFG（官方默认：3.0）")
    group.add_argument("--cfg-notes", type=float, default=1.0, metavar="FLOAT", help="音符条件 CFG（官方默认：1.0）")
    group.add_argument("--cfg-drums", type=float, default=1.0, metavar="FLOAT", help="鼓条件 CFG（官方默认：1.0）")
    group.add_argument("--warmup-steps", type=int, default=5, metavar="INT", help="模型加载后的预热步数（官方默认：5）")
    group.add_argument("--seed", type=int, default=0, metavar="INT", help="MusicCoCa embedding 随机种子（官方默认：0）")
    group.add_argument("--use-mapper", action=argparse.BooleanOptionalAction, default=True, help="是否使用 MusicCoCa mapper（官方 CLI 默认：启用）")
    group.add_argument("--pool-across-time", action=argparse.BooleanOptionalAction, default=True, help="是否在时间维聚合 embedding（官方默认：启用）")


def _engine_config(args: argparse.Namespace) -> EngineConfig:
    return EngineConfig(
        model=cast(ModelName, args.model),
        model_root=args.model_root.expanduser().resolve(),
        temperature=args.temperature,
        top_k=args.top_k,
        cfg_musiccoca=args.cfg_musiccoca,
        cfg_notes=args.cfg_notes,
        cfg_drums=args.cfg_drums,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        use_mapper=args.use_mapper,
        pool_across_time=args.pool_across_time,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = _engine_config(args)

    if args.command == "info":
        print("MRT2 本地服务")
        print(f"模型：{config.model}")
        print(f"模型目录：{config.model_dir}")
        print(f"资源目录：{config.resources_path}")
        print("后端：MLX")
        print(f"Temperature：{config.temperature}")
        print(f"Top-k：{config.top_k}")
        print(f"CFG MusicCoCa/Notes/Drums：{config.cfg_musiccoca}/{config.cfg_notes}/{config.cfg_drums}")
        print(f"预热步数：{config.warmup_steps}")
        print(f"Embedding seed：{config.seed}")
        print(f"Use mapper：{config.use_mapper}")
        print(f"Pool across time：{config.pool_across_time}")
        return

    if args.command == "generate":
        from .engine import MrtEngine

        engine = MrtEngine(config)
        try:
            engine.load()
            result = engine.generate(args.prompt, args.duration)
        except Exception as exc:
            print(f"错误：{exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(result.to_wav_bytes())
        print(f"模型：{config.model}")
        print(f"提示词：{args.prompt}")
        print(f"时长：{args.duration:g} 秒")
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

    config = _engine_config(args)
    uvicorn.run(create_app(config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
