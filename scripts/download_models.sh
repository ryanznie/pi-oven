#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${ROOT_DIR}/models"

TARGET_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
DRAFT_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"

mkdir -p "${MODEL_DIR}"

download_if_missing() {
  local url="$1"
  local output="$2"

  if [[ -s "${output}" ]]; then
    echo "Already present: ${output}"
    return
  fi

  echo "Downloading ${output}"
  wget --continue --show-progress -O "${output}" "${url}"
}

download_if_missing "${TARGET_URL}" "${MODEL_DIR}/qwen2.5-1.5b-instruct-q4_k_m.gguf"
download_if_missing "${DRAFT_URL}" "${MODEL_DIR}/qwen2.5-0.5b-instruct-q4_k_m.gguf"

ls -lh "${MODEL_DIR}"
