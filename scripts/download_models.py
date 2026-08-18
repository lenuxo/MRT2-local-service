#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 MRT2 模型到项目目录")
    parser.add_argument(
        "models",
        nargs="*",
        choices=("mrt2_small", "mrt2_base"),
        default=("mrt2_small",),
    )
    args = parser.parse_args()
    download_path = Path(__file__).resolve().parent.parent / "models"

    from magenta_rt.cli import main as magenta_cli

    magenta_cli(
        ["models", "init", "--download-path", str(download_path)],
        standalone_mode=False,
    )
    for model in args.models:
        magenta_cli(
            ["models", "download", model, "--download-path", str(download_path)],
            standalone_mode=False,
        )
    print(f"模型已保存到：{download_path}")


if __name__ == "__main__":
    main()
