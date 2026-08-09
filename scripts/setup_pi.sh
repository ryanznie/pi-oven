#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="${ROOT_DIR}/vendor/llama.cpp"

sudo apt update
sudo apt install -y \
  git cmake build-essential python3 python3-venv python3-pip \
  linux-perf htop wget curl ca-certificates pkg-config

mkdir -p "${ROOT_DIR}/vendor" "${ROOT_DIR}/models" "${ROOT_DIR}/results"

if [[ ! -d "${LLAMA_DIR}/.git" ]]; then
  git clone https://github.com/ggml-org/llama.cpp "${LLAMA_DIR}"
fi

cd "${LLAMA_DIR}"
git pull --ff-only

if cmake -B build -DGGML_CPU_KLEIDIAI=ON -DCMAKE_BUILD_TYPE=Release; then
  cmake --build build --config Release -j"$(nproc)"
else
  echo "KleidiAI configure failed; falling back to plain CPU build."
  cmake -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build --config Release -j"$(nproc)"
fi

"${LLAMA_DIR}/build/bin/llama-cli" --version
"${LLAMA_DIR}/build/bin/llama-bench" --help >/dev/null

echo
echo "Setup complete."
echo "Next: ./scripts/download_models.sh"
