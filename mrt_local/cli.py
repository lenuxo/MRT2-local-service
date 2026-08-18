from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import SUPPORTED_MODELS, EngineConfig, ModelName, default_model_root


def _model(value: str) -> ModelName:
    if value not in SUPPORTED_MODELS:
        raise argparse.ArgumentTypeError("模型必须是 mrt2_small 或 mrt2_base")
    return value  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mrt", description="MRT2 本地服务")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="生成 WAV 音频")
    generate.add_argument("--prompt", required=True, help="文本提示词")
    generate.add_argument("--duration", type=float, default=10, help="生成时长（秒）")
    generate.add_argument("--output", type=Path, default=Path("output.wav"))
    generate.add_argument("--model", type=_model, default="mrt2_small")
    generate.add_argument("--model-root", type=Path, default=default_model_root())

    serve = subparsers.add_parser("serve", help="启动本地 HTTP API")
    serve.add_argument("--model", type=_model, default="mrt2_small")
    serve.add_argument("--model-root", type=Path, default=default_model_root())
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    info = subparsers.add_parser("info", help="显示配置")
    info.add_argument("--model", type=_model, default="mrt2_small")
    info.add_argument("--model-root", type=Path, default=default_model_root())
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = EngineConfig(model=args.model, model_root=args.model_root.expanduser().resolve())

    if args.command == "info":
        print("MRT2 本地服务")
        print(f"模型：{config.model}")
        print(f"模型目录：{config.model_dir}")
        print(f"资源目录：{config.resources_path}")
        print("后端：MLX")
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
    parser = argparse.ArgumentParser(prog="mrt-serve", description="启动 MRT2 本地服务")
    parser.add_argument("--model", type=_model, default="mrt2_small")
    parser.add_argument("--model-root", type=Path, default=default_model_root())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("错误：端口必须在 1 到 65535 之间")

    import uvicorn
    from .api import create_app

    config = EngineConfig(
        model=args.model,
        model_root=args.model_root.expanduser().resolve(),
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
