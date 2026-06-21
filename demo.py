"""
demo.py — interactive end-to-end demonstration
==============================================

Runs a curated set of sample tickets through the live LangGraph workflow in a
real terminal so the Human-in-the-Loop (Y/N) compliance gate is exercised
interactively. This is the script recorded by the VHS tape.

Prerequisite: the mock REST API must be running so the Action Node can execute
its calls:

    uvicorn main:app --port 8000     # in another terminal / background

Then:

    python demo.py
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel

from agent import RAG, run_ticket

console = Console()

SAMPLE_TICKETS = [
    {
        "id": "TCK-1001",
        "subject": "Do I still need to rotate my password every 90 days?",
        "body": "I keep getting reminders. What's our actual password rotation policy "
                "and is MFA required?",
        "requester": "alice@company.com",
        "priority": "low",
    },
    {
        "id": "TCK-1002",
        "subject": "Please grant me production access",
        "body": "I need write access to the production data warehouse to debug an "
                "incident. My account is bob@company.com.",
        "requester": "bob@company.com",
        "priority": "high",
    },
    {
        "id": "TCK-1003",
        "subject": "Can I share a customer report with an external partner?",
        "body": "A partner asked for a spreadsheet that includes customer names and "
                "emails. Am I allowed to send it?",
        "requester": "carol@company.com",
        "priority": "normal",
    },
]


def main() -> None:
    console.print(
        Panel.fit(
            "[bold]Enterprise Triage & Resolution Agent[/bold]\n"
            "[dim]Advanced RAG · LangGraph orchestration · HITL governance[/dim]",
            border_style="cyan",
            title="🚀 DEMO",
        )
    )

    # Warm the RAG engine once up-front so the ingestion logs appear before the
    # first ticket rather than mid-flow.
    RAG.build()

    tickets = SAMPLE_TICKETS
    # Allow `python demo.py 2` to run a single ticket by index (1-based).
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        idx = int(sys.argv[1]) - 1
        tickets = [SAMPLE_TICKETS[idx]]

    for ticket in tickets:
        run_ticket(ticket)

    console.print()
    console.print(
        Panel.fit("All sample tickets processed.", border_style="green", title="✅ DONE")
    )


if __name__ == "__main__":
    main()
