#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# serve_vllm.sh — launch a local vLLM OpenAI-compatible server on the GPU.
#
# Tuned for an RTX 5070 (12 GB, Blackwell / sm_120) + Ryzen AI 9 host.
# The agent talks to this server when LLM_BACKEND=vllm (or auto + reachable).
#
# Usage:
#   ./serve_vllm.sh                       # default model below
#   VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct ./serve_vllm.sh
#
# Then point the agent at it (already the default base URL):
#   export LLM_BACKEND=vllm
#   export VLLM_BASE_URL=http://127.0.0.1:8001/v1
#   export VLLM_MODEL="$VLLM_MODEL"
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# AWQ 4-bit keeps a 7B model comfortably inside 12 GB of VRAM, leaving room for
# the KV cache. Swap to a 3B model if you also run the embedder/reranker on GPU.
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
PORT="${VLLM_PORT:-8001}"
MAX_LEN="${VLLM_MAX_LEN:-8192}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"

echo "▶ Serving '${MODEL}' with vLLM on port ${PORT} (RTX 5070 profile)…"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true

exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --quantization awq_marlin \
  --dtype float16 \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --enable-prefix-caching \
  --served-model-name "${MODEL}"
