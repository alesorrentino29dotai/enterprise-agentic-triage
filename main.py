"""
main.py — FastAPI ingestion & mock REST surface
===============================================

Two responsibilities:

  1. **Webhook ingestion** — `POST /webhook/ticket` receives a support ticket
     (the kind a Zendesk/Jira/ServiceNow connector would send) and drives it
     through the LangGraph triage workflow defined in `agent.py`.

  2. **Mock enterprise REST API** — the state-changing endpoints the Action Node
     calls (`/mock/access/grant`, `/mock/access/revoke`, `/mock/password/reset`).
     These stand in for real IAM / IdP systems.

Run the API:
    uvicorn main:app --reload

Note on HITL inside the webhook: a webhook worker has no interactive terminal,
so the Action Node's approval gate is fail-safe (it denies) unless
`TRIAGE_AUTO_APPROVE=approve|deny` is set. The interactive Y/N compliance gate
is demonstrated by `demo.py`, which runs the same graph in a real terminal.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field
from rich.console import Console

from agent import run_ticket

console = Console()

app = FastAPI(
    title="Enterprise Triage & Resolution Agent",
    description="Webhook ingestion + mock enterprise REST API for the triage agent.",
    version="1.0.0",
)


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class Ticket(BaseModel):
    id: str = Field(..., examples=["TCK-1001"])
    subject: str = Field(..., examples=["What is our MFA policy?"])
    body: str = Field(..., examples=["Can I still use SMS codes for login?"])
    requester: Optional[str] = Field(default=None, examples=["alice@company.com"])
    priority: Optional[str] = Field(default="normal")


class ActionRequest(BaseModel):
    ticket_id: Optional[str] = None
    target_user: str
    system: str


# ─────────────────────────────────────────────────────────────────────────────
# Webhook ingestion
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/webhook/ticket")
def ingest_ticket(ticket: Ticket):
    """Receive a ticket from an upstream connector and triage it."""
    console.print(f"[bold green]▶ Webhook received[/bold green] ticket {ticket.id}")
    final = run_ticket(ticket.model_dump())
    return {
        "ticket_id": ticket.id,
        "status": final.get("status"),
        "category": final.get("category"),
        "answer": final.get("answer"),
        "action": final.get("action"),
        "result": final.get("result"),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")}


# ─────────────────────────────────────────────────────────────────────────────
# Mock enterprise REST API (state-changing operations)
# ─────────────────────────────────────────────────────────────────────────────

def _audit(operation: str, req: ActionRequest) -> dict:
    record = {
        "operation": operation,
        "ticket_id": req.ticket_id,
        "target_user": req.target_user,
        "system": req.system,
        "status": "completed",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "audit_id": f"AUD-{abs(hash((operation, req.target_user, req.system))) % 100000:05d}",
    }
    console.print(f"[bold]⚙ mock-api[/bold] executed [cyan]{operation}[/cyan] → {record['audit_id']}")
    return record


@app.post("/mock/access/grant")
def grant_access(req: ActionRequest):
    return _audit("grant_access", req)


@app.post("/mock/access/revoke")
def revoke_access(req: ActionRequest):
    return _audit("revoke_access", req)


@app.post("/mock/password/reset")
def reset_password(req: ActionRequest):
    return _audit("reset_password", req)
