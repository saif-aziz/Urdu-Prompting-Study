#!/usr/bin/env bash
# Full experiment pipeline. Run from project root.
set -euo pipefail

CONFIG=${1:-configs/default.yaml}

echo "=== 1/3 Main prompt-x-model grid ==="
python -m src.main --config "$CONFIG"

echo "=== 2/3 Shot-count ablation on qwen15 ==="
python -m src.ablation --config "$CONFIG" --model qwen15

echo "=== 3/3 Generating figures and LaTeX table ==="
python -m scripts.make_figures

echo "Done. See results/ and paper/figures/"
