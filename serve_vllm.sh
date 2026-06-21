#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# serve_vllm.sh — launch a local vLLM OpenAI-compatible server on the GPU.
#
# Tuned for an RTX 5070 *Laptop* GPU (8 GB, Blackwell / sm_120) + Ryzen AI 9.
# The agent talks to this server when LLM_BACKEND=vllm (or auto + reachable).
#
# Install (match the driver's CUDA — let uv auto-detect; do NOT force cu128):
#   uv venv .venv-vllm --python 3.12
#   uv pip install --python .venv-vllm vllm --torch-backend=auto
#   source .venv-vllm/bin/activate
#
# Usage:
#   ./serve_vllm.sh                                   # 3B AWQ — safe on 8 GB
#   VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ VLLM_MAX_LEN=2048 ./serve_vllm.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# FlashInfer JIT-compiles its sampling kernel at runtime, which needs the full
# CUDA toolkit (nvcc). On a driver-only host that fails with
# "Could not find nvcc". Force vLLM's native Torch sampler (no compilation).
# Override by exporting VLLM_USE_FLASHINFER_SAMPLER=1 if you install the toolkit.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# A 3B AWQ model fits comfortably in 8 GB with room for the KV cache. Bump to a
# 7B AWQ model only if you also lower VLLM_MAX_LEN (e.g. 2048).
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct-AWQ}"
PORT="${VLLM_PORT:-8001}"
MAX_LEN="${VLLM_MAX_LEN:-8192}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"

# Pick the quantization kernel from the model name unless overridden.
QUANT="${VLLM_QUANTIZATION:-auto}"
if [ "${QUANT}" = "auto" ]; then
  case "${MODEL,,}" in
    *awq*)  QUANT="awq_marlin" ;;
    *gptq*) QUANT="gptq_marlin" ;;
    *)      QUANT="" ;;
  esac
fi

echo "▶ Serving '${MODEL}' with vLLM on port ${PORT} (RTX 5070 8 GB profile)…"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true

ARGS=(
  --port "${PORT}"
  --dtype float16
  --max-model-len "${MAX_LEN}"
  --gpu-memory-utilization "${GPU_UTIL}"
  --enable-prefix-caching
  --served-model-name "${MODEL}"
)
[ -n "${QUANT}" ] && ARGS+=(--quantization "${QUANT}")

echo "  args: ${ARGS[*]}"
exec vllm serve "${MODEL}" "${ARGS[@]}"
