"""
agent.py — Enterprise Triage & Resolution Agent
================================================

Stateful, cyclic LangGraph workflow that triages an incoming support ticket and
either:

  • resolves it via an Advanced RAG pipeline
        (Qdrant vector store → semantic chunking → CrossEncoder re-ranking → Claude),
  • or executes a state-changing "system action" via a mock REST API — but only
    after a mandatory Human-in-the-Loop (HITL) approval at the terminal.

All observability is rendered with `rich` for a high-visibility event flow.

The LLM layer uses the official Anthropic SDK (Claude Opus 4.8). If no
ANTHROPIC_API_KEY is present, the agent degrades to deterministic heuristics so
the workflow remains demonstrable end-to-end without credentials.
"""

from __future__ import annotations

import os
import re
import sys
import textwrap
import warnings

# Keep the high-visibility `rich` event flow clean of third-party deprecation noise.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", message=r".*allowed_objects.*")
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

import contextlib
import io

import httpx
from dotenv import load_dotenv

# LangGraph/LangChain emit a PendingDeprecation warning straight to stderr at
# import time and re-register their own warning filter, so `-W ignore` doesn't
# silence it. Capture stderr just for this import to keep the demo output clean.
with contextlib.redirect_stderr(io.StringIO()):
    from langgraph.graph import END, START, StateGraph

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.rule import Rule

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")

# ── LLM backend selection ────────────────────────────────────────────────────
# "auto" (default): use local vLLM if reachable, else Claude if a key is set,
#                   else deterministic heuristics.
# "vllm"      : force the local GPU server (OpenAI-compatible, e.g. RTX 5070).
# "anthropic" : force Claude Opus 4.8 (cloud).
# "heuristic" : force the no-LLM fallback.
LLM_BACKEND = os.getenv("LLM_BACKEND", "auto").lower()
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-3B-Instruct-AWQ")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", "knowledge_base")
MOCK_API_BASE = os.getenv("MOCK_API_BASE", "http://127.0.0.1:8000")
COLLECTION = "enterprise_kb"

# Retrieval knobs
TOP_K_RETRIEVE = 8       # candidates pulled from the vector store
TOP_N_RERANK = 3         # high-fidelity passages kept after re-ranking
SEMANTIC_BREAKPOINT = 0.55  # cosine similarity floor for staying in a chunk

console = Console()

# `auto` mode (set by main.py for webhook-driven runs) replaces the interactive
# HITL prompt with a deterministic policy so the server never blocks on stdin.
AUTO_APPROVE_ENV = "TRIAGE_AUTO_APPROVE"


# ─────────────────────────────────────────────────────────────────────────────
# Rich logging helpers
# ─────────────────────────────────────────────────────────────────────────────

def _node_banner(title: str, subtitle: str, color: str) -> None:
    console.print(
        Panel.fit(
            f"[bold]{title}[/bold]\n[dim]{subtitle}[/dim]",
            border_style=color,
            title="● NODE",
            title_align="left",
        )
    )


def log_info(msg: str) -> None:
    console.print(f"[cyan]ℹ[/cyan]  {msg}")


def log_ok(msg: str) -> None:
    console.print(f"[green]✔[/green]  {msg}")


def log_warn(msg: str) -> None:
    console.print(f"[yellow]⚠[/yellow]  {msg}")


def log_step(msg: str) -> None:
    console.print(f"[magenta]→[/magenta]  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Advanced RAG: Qdrant + semantic chunking + CrossEncoder re-ranking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RAGEngine:
    """Lazily-initialised Advanced RAG engine.

    Heavyweight models (sentence-transformers, CrossEncoder) and the Qdrant
    client are constructed once via :meth:`build` and reused for the process
    lifetime.
    """

    embedder: Any = None
    reranker: Any = None
    client: Any = None
    _ready: bool = False
    _chunks: list[str] = field(default_factory=list)

    def build(self) -> "RAGEngine":
        if self._ready:
            return self

        from sentence_transformers import CrossEncoder, SentenceTransformer
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct

        console.print(Rule("[bold]Advanced RAG · ingestion pipeline[/bold]", style="blue"))
        log_step(f"Loading embedding model [bold]{EMBED_MODEL}[/bold] …")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        log_step(f"Loading re-ranking CrossEncoder [bold]{RERANK_MODEL}[/bold] …")
        self.reranker = CrossEncoder(RERANK_MODEL)

        log_step("Starting in-memory Qdrant vector store …")
        self.client = QdrantClient(location=":memory:")
        dim = self.embedder.get_sentence_embedding_dimension()
        self.client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        # ── semantic chunking ────────────────────────────────────────────────
        docs = self._load_documents()
        points: list[PointStruct] = []
        pid = 0
        for source, text in docs:
            for chunk in self._semantic_chunks(text):
                vec = self.embedder.encode(chunk, normalize_embeddings=True)
                points.append(
                    PointStruct(
                        id=pid,
                        vector=vec.tolist(),
                        payload={"text": chunk, "source": source},
                    )
                )
                self._chunks.append(chunk)
                pid += 1

        self.client.upsert(collection_name=COLLECTION, points=points)
        log_ok(
            f"Indexed [bold]{len(points)}[/bold] semantic chunks "
            f"from [bold]{len(docs)}[/bold] documents into Qdrant."
        )
        self._ready = True
        return self

    # -- ingestion helpers ----------------------------------------------------

    def _load_documents(self) -> list[tuple[str, str]]:
        docs: list[tuple[str, str]] = []
        if not os.path.isdir(KNOWLEDGE_DIR):
            log_warn(f"Knowledge directory '{KNOWLEDGE_DIR}' not found.")
            return docs
        for name in sorted(os.listdir(KNOWLEDGE_DIR)):
            if name.endswith((".md", ".txt")):
                with open(os.path.join(KNOWLEDGE_DIR, name), encoding="utf-8") as fh:
                    docs.append((name, fh.read()))
        return docs

    def _semantic_chunks(self, text: str, max_sentences: int = 6) -> list[str]:
        """Group adjacent sentences into semantically coherent chunks.

        A new chunk is started when the cosine similarity between consecutive
        sentence embeddings drops below ``SEMANTIC_BREAKPOINT`` or the chunk
        reaches ``max_sentences`` — a lightweight semantic-chunking strategy
        that keeps related policy statements together.
        """
        import numpy as np

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        embs = self.embedder.encode(sentences, normalize_embeddings=True)
        chunks: list[str] = []
        current = [sentences[0]]
        for i in range(1, len(sentences)):
            sim = float(np.dot(embs[i], embs[i - 1]))
            if sim < SEMANTIC_BREAKPOINT or len(current) >= max_sentences:
                chunks.append(" ".join(current))
                current = [sentences[i]]
            else:
                current.append(sentences[i])
        if current:
            chunks.append(" ".join(current))
        return chunks

    # -- query path -----------------------------------------------------------

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
        qvec = self.embedder.encode(query, normalize_embeddings=True)
        hits = self.client.search(
            collection_name=COLLECTION,
            query_vector=qvec.tolist(),
            limit=top_k,
        )
        return [
            {"text": h.payload["text"], "source": h.payload["source"], "score": float(h.score)}
            for h in hits
        ]

    def rerank(self, query: str, candidates: list[dict], top_n: int = TOP_N_RERANK) -> list[dict]:
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_n]


# Singleton — built once at import-time use, reused across tickets.
RAG = RAGEngine()


# ─────────────────────────────────────────────────────────────────────────────
# LLM layer (Anthropic Claude Opus 4.8) with graceful degradation
# ─────────────────────────────────────────────────────────────────────────────

def _anthropic_client():
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic

        return anthropic.Anthropic()
    except Exception as exc:  # pragma: no cover - defensive
        log_warn(f"Could not initialise Anthropic client: {exc}")
        return None


_BACKEND_CACHE: dict[str, str] = {}


def _vllm_reachable() -> bool:
    """Probe the local vLLM server's OpenAI-compatible /models endpoint."""
    try:
        r = httpx.get(f"{VLLM_BASE_URL}/models",
                      headers={"Authorization": f"Bearer {VLLM_API_KEY}"}, timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def resolve_backend() -> str:
    """Decide which LLM backend to use, once, and cache it."""
    if "backend" in _BACKEND_CACHE:
        return _BACKEND_CACHE["backend"]

    if LLM_BACKEND in ("vllm", "anthropic", "heuristic"):
        backend = LLM_BACKEND
    else:  # auto
        if _vllm_reachable():
            backend = "vllm"
        elif os.getenv("ANTHROPIC_API_KEY"):
            backend = "anthropic"
        else:
            backend = "heuristic"

    label = {
        "vllm": f"local vLLM (GPU) · {VLLM_MODEL} @ {VLLM_BASE_URL}",
        "anthropic": f"Anthropic Claude · {MODEL_ID}",
        "heuristic": "deterministic heuristics (no LLM)",
    }[backend]
    log_info(f"LLM backend → [bold]{label}[/bold]")
    _BACKEND_CACHE["backend"] = backend
    return backend


def _vllm_chat(messages: list[dict], max_tokens: int = 1024,
               temperature: float = 0.2) -> str:
    """Call the local vLLM server via its OpenAI-compatible chat endpoint."""
    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = httpx.post(
        f"{VLLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {VLLM_API_KEY}",
                 "Content-Type": "application/json"},
        json=payload,
        timeout=120.0,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


CLASSIFY_SYSTEM = textwrap.dedent(
    """
    You are an enterprise support triage classifier. Read the ticket and decide
    how it should be handled.

    Return one of two categories:
      • "knowledge" — the user is asking a question answerable from internal
        policy / knowledge (how-tos, policy lookups, "what is", "am I allowed").
      • "action"    — the user is requesting a state-changing system operation
        (grant/revoke access, reset a password, provision a resource, create an
        account). These mutate systems and require governance.

    Be conservative: if the ticket asks the system to *do* something that changes
    state, classify it as "action".
    """
).strip()


def _heuristic_category(subject: str, body: str) -> dict:
    action_kw = (
        "grant", "revoke", "reset password", "provision", "create account",
        "give access", "add to group", "enable access", "deactivate", "remove access",
        "unlock", "elevate", "promote to admin",
    )
    blob = f"{subject} {body}".lower()
    if any(k in blob for k in action_kw):
        return {"category": "action",
                "reasoning": "Matched a state-changing action keyword.",
                "engine": "heuristic"}
    return {"category": "knowledge",
            "reasoning": "No state-changing keywords detected; treated as a question.",
            "engine": "heuristic"}


def _parse_triage_json(text: str) -> dict | None:
    """Extract {category, reasoning} from an LLM's (possibly chatty) JSON output."""
    import json

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except Exception:
        return None
    cat = str(data.get("category", "")).lower().strip()
    if cat not in ("knowledge", "action"):
        return None
    return {"category": cat, "reasoning": data.get("reasoning", "")}


def classify_ticket(ticket: dict) -> dict:
    """Classify a ticket as 'knowledge' or 'action' via the selected LLM backend,
    falling back to deterministic heuristics on any failure."""
    backend = resolve_backend()
    subject = ticket.get("subject", "")
    body = ticket.get("body", "")
    user_msg = f"Subject: {subject}\n\nBody: {body}"

    if backend == "anthropic":
        client = _anthropic_client()
        if client is not None:
            from pydantic import BaseModel

            class Triage(BaseModel):
                category: Literal["knowledge", "action"]
                reasoning: str

            try:
                resp = client.messages.parse(
                    model=MODEL_ID,
                    max_tokens=1024,
                    system=CLASSIFY_SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                    output_format=Triage,
                )
                parsed = resp.parsed_output
                return {"category": parsed.category, "reasoning": parsed.reasoning,
                        "engine": MODEL_ID}
            except Exception as exc:  # pragma: no cover - network/SDK issues
                log_warn(f"Claude classification failed ({exc}); using heuristic fallback.")

    elif backend == "vllm":
        try:
            prompt = (user_msg + "\n\nRespond ONLY with a JSON object of the form "
                      '{"category": "knowledge"|"action", "reasoning": "<one sentence>"}.')
            text = _vllm_chat(
                [{"role": "system", "content": CLASSIFY_SYSTEM},
                 {"role": "user", "content": prompt}],
                max_tokens=256, temperature=0.0,
            )
            parsed = _parse_triage_json(text)
            if parsed:
                parsed["engine"] = f"vllm:{VLLM_MODEL}"
                return parsed
            log_warn("vLLM returned unparseable triage JSON; using heuristic fallback.")
        except Exception as exc:
            log_warn(f"vLLM classification failed ({exc}); using heuristic fallback.")

    return _heuristic_category(subject, body)


RAG_SYSTEM = textwrap.dedent(
    """
    You are an enterprise support resolution agent. Answer the user's ticket using
    ONLY the provided policy context. Cite the source document(s) by filename.
    If the context does not contain the answer, say so plainly and recommend
    escalation to a human agent. Be concise and direct — lead with the answer.
    """
).strip()


def generate_answer(ticket: dict, passages: list[dict]) -> str:
    """Generate a grounded answer from the re-ranked passages."""
    context = "\n\n".join(
        f"[source: {p['source']}]\n{p['text']}" for p in passages
    )
    question = f"Subject: {ticket.get('subject','')}\n\nBody: {ticket.get('body','')}"
    user_msg = f"POLICY CONTEXT:\n{context}\n\n---\n\nTICKET:\n{question}"

    backend = resolve_backend()

    if backend == "anthropic":
        client = _anthropic_client()
        if client is not None:
            try:
                with client.messages.stream(
                    model=MODEL_ID,
                    max_tokens=1024,
                    thinking={"type": "adaptive"},
                    system=RAG_SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                ) as stream:
                    msg = stream.get_final_message()
                return "".join(b.text for b in msg.content if b.type == "text").strip()
            except Exception as exc:  # pragma: no cover
                log_warn(f"Claude answer generation failed ({exc}); extractive fallback.")

    elif backend == "vllm":
        try:
            return _vllm_chat(
                [{"role": "system", "content": RAG_SYSTEM},
                 {"role": "user", "content": user_msg}],
                max_tokens=1024, temperature=0.2,
            ).strip()
        except Exception as exc:
            log_warn(f"vLLM answer generation failed ({exc}); extractive fallback.")

    # Extractive fallback -----------------------------------------------------
    top = passages[0] if passages else None
    if not top:
        return ("I could not find a relevant policy passage. Escalating to a human "
                "support agent.")
    return (
        "Based on internal policy, here is the most relevant guidance:\n\n"
        f"{top['text']}\n\n(Source: {top['source']})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph state + nodes
# ─────────────────────────────────────────────────────────────────────────────

class TriageState(TypedDict, total=False):
    ticket: dict
    category: str
    reasoning: str
    engine: str
    retrieved: list[dict]
    reranked: list[dict]
    answer: str
    action: dict
    approved: bool
    result: dict
    status: str


def triage_node(state: TriageState) -> TriageState:
    _node_banner("TRIAGE", "Classifying ticket → knowledge vs. system action", "yellow")
    ticket = state["ticket"]
    log_info(f"Ticket #{ticket.get('id','?')} — [italic]{ticket.get('subject','')}[/italic]")
    result = classify_ticket(ticket)
    color = "blue" if result["category"] == "knowledge" else "red"
    console.print(
        f"   classified as [bold {color}]{result['category'].upper()}[/bold {color}] "
        f"[dim](via {result['engine']})[/dim]"
    )
    console.print(f"   [dim]reasoning:[/dim] {result['reasoning']}")
    return {**state, **result}


def route(state: TriageState) -> Literal["rag", "act"]:
    return "rag" if state["category"] == "knowledge" else "act"


def rag_node(state: TriageState) -> TriageState:
    _node_banner("KNOWLEDGE / RAG", "Retrieve → re-rank → ground answer in policy", "blue")
    RAG.build()
    ticket = state["ticket"]
    query = f"{ticket.get('subject','')} {ticket.get('body','')}".strip()

    log_step(f"Vector search → top {TOP_K_RETRIEVE} candidates from Qdrant …")
    candidates = RAG.retrieve(query)
    for c in candidates[:TOP_K_RETRIEVE]:
        console.print(f"     [dim]{c['score']:.3f}  {c['source']}[/dim]")

    log_step(f"CrossEncoder re-ranking → keeping top {TOP_N_RERANK} high-fidelity passages …")
    reranked = RAG.rerank(query, candidates)
    tbl = Table(show_header=True, header_style="bold blue", box=None)
    tbl.add_column("rerank", justify="right")
    tbl.add_column("source")
    tbl.add_column("passage", overflow="fold")
    for p in reranked:
        tbl.add_row(f"{p['rerank_score']:.2f}", p["source"], textwrap.shorten(p["text"], 90))
    console.print(tbl)

    log_step(f"Generating grounded answer via {resolve_backend()} backend …")
    answer = generate_answer(ticket, reranked)
    console.print(Panel(answer, title="📩 Resolution", border_style="green"))
    return {**state, "retrieved": candidates, "reranked": reranked,
            "answer": answer, "status": "resolved"}


# Maps the natural-language intent to a concrete mock REST operation.
ACTION_ROUTES: dict[str, dict] = {
    "grant_access": {"method": "POST", "path": "/mock/access/grant"},
    "revoke_access": {"method": "POST", "path": "/mock/access/revoke"},
    "reset_password": {"method": "POST", "path": "/mock/password/reset"},
}


def _plan_action(ticket: dict) -> dict:
    """Derive a concrete (mock) REST action from the ticket text."""
    blob = f"{ticket.get('subject','')} {ticket.get('body','')}".lower()
    if "reset" in blob and "password" in blob:
        op = "reset_password"
    elif any(k in blob for k in ("revoke", "remove access", "deactivate")):
        op = "revoke_access"
    else:
        op = "grant_access"

    # naive entity extraction for the demo payload
    user_match = re.search(r"\b([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})\b", blob)
    system_match = re.search(r"(production|prod|admin console|data warehouse|repo[a-z ]*|github|salesforce|s3)", blob)
    return {
        "operation": op,
        "route": ACTION_ROUTES[op],
        "payload": {
            "ticket_id": ticket.get("id"),
            "target_user": user_match.group(1) if user_match else ticket.get("requester", "unknown@company.com"),
            "system": system_match.group(1) if system_match else "internal-admin-console",
        },
    }


def _approval(action: dict) -> bool:
    """Mandatory Human-in-the-Loop gate before any state-changing call."""
    auto = os.getenv(AUTO_APPROVE_ENV)
    tbl = Table(show_header=False, box=None)
    tbl.add_row("[bold]operation[/bold]", action["operation"])
    tbl.add_row("[bold]method[/bold]", action["route"]["method"])
    tbl.add_row("[bold]endpoint[/bold]", action["route"]["path"])
    for k, v in action["payload"].items():
        tbl.add_row(f"[bold]{k}[/bold]", str(v))
    console.print(
        Panel(tbl, title="🛡  HUMAN-IN-THE-LOOP APPROVAL REQUIRED",
              border_style="red", subtitle="state-changing action — compliance gate")
    )

    if auto in ("approve", "deny"):
        decision = auto == "approve"
        log_warn(f"Non-interactive mode ({AUTO_APPROVE_ENV}={auto}): auto-{auto}.")
        return decision

    if not sys.stdin.isatty():
        log_warn("No interactive terminal available — denying by default (fail-safe). "
                 f"Set {AUTO_APPROVE_ENV}=approve to allow automated execution.")
        return False

    answer = Prompt.ask(
        "[bold red]Approve this action?[/bold red]", choices=["y", "n"], default="n"
    )
    return answer.lower() == "y"


def action_node(state: TriageState) -> TriageState:
    _node_banner("SYSTEM ACTION", "Plan action → HITL approval → execute mock REST call", "red")
    ticket = state["ticket"]
    action = _plan_action(ticket)
    log_step(f"Planned operation: [bold]{action['operation']}[/bold]")

    approved = _approval(action)
    if not approved:
        log_warn("Action DENIED by human reviewer. No system change performed.")
        console.print(Panel("Action was not approved. Ticket escalated for manual handling.",
                            title="⛔ Denied", border_style="yellow"))
        return {**state, "action": action, "approved": False, "status": "denied"}

    log_ok("Approved. Executing mock REST call …")
    route_cfg = action["route"]
    url = f"{MOCK_API_BASE}{route_cfg['path']}"
    try:
        resp = httpx.request(route_cfg["method"], url, json=action["payload"], timeout=10.0)
        result = {"status_code": resp.status_code, "body": resp.json()}
    except Exception as exc:
        log_warn(f"Mock REST call failed: {exc}")
        result = {"status_code": 0, "body": {"error": str(exc)}}

    console.print(Panel(str(result["body"]),
                        title=f"⚙ REST {route_cfg['method']} {route_cfg['path']} → {result['status_code']}",
                        border_style="green"))
    return {**state, "action": action, "approved": True,
            "result": result, "status": "executed"}


# ─────────────────────────────────────────────────────────────────────────────
# Graph assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(TriageState)
    g.add_node("triage", triage_node)
    g.add_node("rag", rag_node)
    g.add_node("act", action_node)

    g.add_edge(START, "triage")
    g.add_conditional_edges("triage", route, {"rag": "rag", "act": "act"})
    g.add_edge("rag", END)
    g.add_edge("act", END)
    return g.compile()


GRAPH = build_graph()


def run_ticket(ticket: dict) -> TriageState:
    """Execute the full triage workflow for one ticket and return final state."""
    console.print()
    console.print(Rule(f"[bold]TICKET #{ticket.get('id','?')}[/bold]", style="white"))
    final = GRAPH.invoke({"ticket": ticket})
    status = final.get("status", "unknown")
    log_ok(f"Workflow complete — final status: [bold]{status}[/bold]")
    return final
