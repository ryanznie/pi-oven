#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_BIN="${LLAMA_BIN:-${ROOT_DIR}/vendor/llama.cpp/build/bin}"
MODEL="${MODEL:-${ROOT_DIR}/models/qwen2.5-1.5b-instruct-q4_k_m.gguf}"
THREADS="${THREADS:-4}"

"${LLAMA_BIN}/llama-cli" \
  -m "${MODEL}" \
  -p "Explain KV cache optimization in two sentences." \
  -n 96 \
  -c 1024 \
  -t "${THREADS}" \
  --no-conversation \
  --single-turn \
  --show-timings
