"""CLI: `research-team run "topic"`, `research-team chat`, `research-team graph`."""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

import typer
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from research_team.agents import default_deps
from research_team.graph import build_graph, make_checkpointer
from research_team.reporting import content_to_text

app = typer.Typer(add_completion=False, help="Multi-agent research team (LangChain + LangGraph).")
console = Console()


def _slug(text: str) -> str:
    return re.sub(r"[^\w]+", "-", text.lower(), flags=re.UNICODE).strip("-")[:60] or "report"


def _render_update(node: str, update: dict | None) -> None:
    if not update:
        return
    if node == "planner":
        qs = "\n".join(f"{q.id}. {q.question}" for q in update["plan"])
        console.print(Panel(qs, title="🧭 plan"))
    elif node == "researcher":
        f = update["findings"][0]
        console.print(
            f"🔎 researcher #{f.question_id}: {len(f.sources)} sources, "
            f"confidence {f.confidence:.2f}"
        )
    elif node == "writer":
        console.print(f"✍️  draft ready ({len(update['draft'])} chars)")
    elif node == "reviewer":
        r = update["review"]
        mark = "✅" if r.approved else "🔁"
        console.print(f"{mark} review: score {r.score}/10, approved={r.approved}")
        for issue in r.issues:
            console.print(f"   - {issue}", style="yellow")
    elif node == "finalize":
        console.print("📦 report finalized")


@app.command()
def run(
    topic: str = typer.Argument(..., help="What to research"),
    approval: bool = typer.Option(True, "--approval/--no-approval", help="Human-in-the-loop"),
    out_dir: Path = typer.Option(Path("reports"), help="Where to save the Markdown report"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the plan → parallel research → write ↔ review → approve workflow."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    graph = build_graph(default_deps(), require_approval=approval, checkpointer=make_checkpointer())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    payload: dict | Command = {"topic": topic}
    while True:
        pending = None
        for chunk in graph.stream(payload, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    pending = update[0].value
                else:
                    _render_update(node, update)
        if pending is None:
            break
        console.print(Panel(Markdown(pending["draft"]), title="📝 draft for approval"))
        payload = Command(resume=Prompt.ask(pending["hint"], default="approve"))

    report = graph.get_state(config).values["final_report"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_slug(topic)}.md"
    path.write_text(report, encoding="utf-8")
    console.print(Markdown(report))
    console.print(f"\nSaved to [bold]{path}[/bold]")


@app.command()
def chat() -> None:
    """Chat with the supervisor agent that delegates to subagents-as-tools."""
    from research_team.supervisor import build_supervisor

    supervisor = build_supervisor()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    console.print("Supervisor ready. Empty line or Ctrl+C to exit.\n")
    while True:
        try:
            text = Prompt.ask("[bold cyan]you[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            break
        if not text.strip():
            break
        answer = ""
        for chunk in supervisor.stream(
            {"messages": [{"role": "user", "content": text}]}, config, stream_mode="updates"
        ):
            for update in chunk.values():
                for msg in (update or {}).get("messages", []):
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            console.print(f"  → {tc['name']}({tc['args']})", style="dim")
                    elif isinstance(msg, ToolMessage):
                        console.print(f"  ← {msg.name}: {len(str(msg.content))} chars", style="dim")
                    elif isinstance(msg, AIMessage):
                        answer = content_to_text(msg.content)
        console.print(Markdown(answer))


@app.command()
def graph() -> None:
    """Print the workflow as a Mermaid diagram (paste into https://mermaid.live)."""
    from research_team.agents import TeamDeps

    def _noop(*_a, **_k):  # structure only, no LLMs needed
        raise NotImplementedError

    g = build_graph(TeamDeps(plan=_noop, research=_noop, write=_noop, review=_noop))
    console.print(g.get_graph().draw_mermaid())


def main() -> None:
    load_dotenv()
    app()


if __name__ == "__main__":
    main()
