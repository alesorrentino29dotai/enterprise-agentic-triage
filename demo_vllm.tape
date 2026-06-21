# VHS tape — records demo_vllm.gif on a GPU host (e.g. RTX 5070 + Ryzen AI 9).
#
# PREREQUISITE: start the local model server FIRST and wait for it to finish
# loading the weights (this can take 30–90s), then run this tape:
#
#     ./serve_vllm.sh          # terminal 1 — keep running
#     vhs demo_vllm.tape       # terminal 2
#
# Sleep values are generous to absorb local inference latency; tune for your GPU.

Output demo_vllm.gif

Set Shell bash
Set FontSize 15
Set Width 1500
Set Height 950
Set Theme "Dracula"
Set TypingSpeed 30ms
Set Padding 18

# ── Hidden setup: venv + local-GPU backend + mock REST API ───────────────────
Hide
Type "source .venv/bin/activate" Enter
Type "export LLM_BACKEND=vllm VLLM_BASE_URL=http://127.0.0.1:8001/v1" Enter
Type "export HF_HUB_DISABLE_PROGRESS_BARS=1" Enter
Type "uvicorn main:app --port 8000 >/tmp/triage_api.log 2>&1 &" Enter
Sleep 5s
Type "clear" Enter
Show

Type "# LLM backend: local vLLM on the GPU (RTX 5070) — fully on-prem, no cloud calls"
Sleep 800ms
Enter
Type "nvidia-smi --query-gpu=name,memory.used --format=csv,noheader"
Enter
Sleep 2s

# ── Knowledge ticket → Advanced RAG resolution (answer generated locally) ────
Type "# 1) KNOWLEDGE ticket → triage → Qdrant retrieve → CrossEncoder re-rank → local LLM answer"
Sleep 800ms
Enter
Type "python demo.py 1"
Enter
Sleep 30s
Type "clear"
Enter
Sleep 500ms

# ── Action ticket → HITL approval → mock REST ────────────────────────────────
Type "# 2) ACTION ticket → triage (local LLM) → mandatory Human-in-the-Loop approval"
Sleep 800ms
Enter
Type "python demo.py 2"
Enter
Sleep 22s
Type "y"
Sleep 500ms
Enter
Sleep 6s

Type "# ✅ End-to-end triage running entirely on local hardware"
Enter
Sleep 3s
