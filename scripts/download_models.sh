#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(dirname -- "$script_dir")
download_dir="$project_dir/models"
magenta_cli=${MAGENTA_CLI:-mrt}

if [ "$#" -eq 0 ]; then
  set -- mrt2_small
fi

for model in "$@"; do
  case "$model" in
    mrt2_small|mrt2_base) ;;
    *)
      echo "Unsupported model: $model" >&2
      echo "Usage: $0 [mrt2_small] [mrt2_base]" >&2
      exit 2
      ;;
  esac
done

if ! command -v "$magenta_cli" >/dev/null 2>&1; then
  echo "Official Magenta CLI not found: $magenta_cli" >&2
  echo 'Install it with: uv pip install "magenta-rt[mlx]"' >&2
  exit 1
fi

"$magenta_cli" models init --download-path "$download_dir"
for model in "$@"; do
  "$magenta_cli" models download "$model" --download-path "$download_dir"
done

echo "Models ready under $download_dir"
