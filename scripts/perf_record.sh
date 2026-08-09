#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_BIN="${LLAMA_BIN:-${ROOT_DIR}/vendor/llama.cpp/build/bin}"
MODEL="${MODEL:-${ROOT_DIR}/models/qwen2.5-1.5b-instruct-q4_k_m.gguf}"
THREADS="${THREADS:-4}"

perf record -F 99 -g --call-graph dwarf -- \
  "${LLAMA_BIN}/llama-cli" \
  -m "${MODEL}" \
  -p "Explain why memory bandwidth matters for quantized LLM inference." \
  -n 192 \
  -c 2048 \
  -t "${THREADS}" \
  --no-conversation \
  --single-turn \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --show-timings
