from __future__ import annotations

import argparse
from pathlib import Path

from .config import SUPPORTED_MODELS, default_model_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mrt-download",
        description="使用 Magenta 官方下载器把 MRT2 模型保存到项目目录",
    )
    parser.add_argument(
        "models",
        nargs="*",
        choices=SUPPORTED_MODELS,
        help="要下载的模型；默认下载 mrt2_small",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=default_model_root(),
        help="下载根目录；默认为项目根目录下的 models/",
    )
    return parser


def download(models: list[str] | tuple[str, ...], model_root: Path) -> None:
    model_root = model_root.expanduser().resolve()
    model_root.mkdir(parents=True, exist_ok=True)

    from magenta_rt.cli import main as magenta_cli

    magenta_cli(
        ["models", "init", "--download-path", str(model_root)],
        standalone_mode=False,
    )
    for model in models:
        magenta_cli(
            ["models", "download", model, "--download-path", str(model_root)],
            standalone_mode=False,
        )
    print(f"模型已保存到：{model_root}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    download(args.models or ["mrt2_small"], args.model_root)


if __name__ == "__main__":
    main()
