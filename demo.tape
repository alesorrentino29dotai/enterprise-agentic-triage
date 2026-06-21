# VHS tape — records demo.gif
# Render with:  vhs demo.tape
# Docs: https://github.com/charmbracelet/vhs

Output demo.gif

Set Shell bash
Set FontSize 15
Set Width 1500
Set Height 950
Set Theme "Dracula"
Set TypingSpeed 30ms
Set Padding 18

# ── Hidden setup: venv + mock REST API in the background ─────────────────────
Hide
Type "source .venv/bin/activate" Enter
Type "export LLM_BACKEND=heuristic HF_HUB_DISABLE_PROGRESS_BARS=1" Enter
Type "uvicorn main:app --port 8000 >/tmp/triage_api.log 2>&1 &" Enter
Sleep 5s
Type "clear" Enter
Show

# ── Knowledge ticket → Advanced RAG resolution ───────────────────────────────
Type "# 1) KNOWLEDGE ticket → triage → Qdrant retrieve → CrossEncoder re-rank → grounded answer"
Sleep 800ms
Enter
Type "python demo.py 1"
Enter
Sleep 16s
Type "clear"
Enter
Sleep 500ms

# ── Action ticket → mandatory Human-in-the-Loop approval → mock REST ─────────
Type "# 2) ACTION ticket → triage → mandatory Human-in-the-Loop approval gate"
Sleep 800ms
Enter
Type "python demo.py 2"
Enter
Sleep 13s
# Approve the state-changing action at the compliance gate:
Type "y"
Sleep 500ms
Enter
Sleep 6s

Type "# ✅ Action approved & executed against the mock REST API (with audit id)"
Enter
Sleep 3s
