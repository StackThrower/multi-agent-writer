"""The research workflow as an explicit LangGraph StateGraph.

    START → planner ──Send×N──► researcher (parallel) → writer → reviewer
                                                        ▲          │
                                       issues & budget  └──────────┤
                                                                   ▼
                                     human_approval (interrupt) → finalize → END
                                             │ feedback
                                             └──────► writer

Why a custom StateGraph and not just a supervisor agent? The LangChain docs
recommend it when deterministic steps (fan-out, budgets, rule-based checks,
human approval) are mixed with agentic ones:
https://docs.langchain.com/oss/python/migrate/langgraph-supervisor#when-to-use-a-custom-stategraph-instead
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, Send, interrupt

from research_team.agents import TeamDeps, default_deps
from research_team.config import Settings, get_settings
from research_team.reporting import (
    WriterInput,
    build_source_catalog,
    check_citations,
    render_final_report,
)
from research_team.schemas import Finding, Review, Source, SubQuestion

APPROVE_WORDS = {"", "y", "yes", "ok", "approve", "approved", "lgtm"}

# Checkpoints are deserialized with an allow-list (LangGraph blocks unknown types from
# untrusted checkpoints). Register every custom type that lives in the state.
STATE_TYPES = [("research_team.schemas", n) for n in ("SubQuestion", "Finding", "Source", "Review")]


def make_checkpointer() -> InMemorySaver:
    """In-memory persistence for local runs. In prod: PostgresSaver / SqliteSaver."""
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=STATE_TYPES))


class TeamState(TypedDict, total=False):
    topic: str
    plan: list[SubQuestion]
    # Reducer: parallel researchers append to the same list without overwriting each other.
    findings: Annotated[list[Finding], operator.add]
    sources: list[Source]
    draft: str
    review: Review
    revisions: int
    human_feedback: str | None
    final_report: str


class ResearchTask(TypedDict):
    """Private input of a single researcher branch (sent via `Send`)."""

    question: SubQuestion


def build_graph(
    deps: TeamDeps,
    settings: Settings | None = None,
    *,
    require_approval: bool = True,
    checkpointer: Any = None,
):
    s = settings or get_settings()

    # ---------------------------------------------------------------- nodes
    def planner(state: TeamState) -> dict:
        questions = deps.plan(state["topic"], s.max_subquestions).sub_questions
        if not questions:  # degenerate plan → research the topic as a whole
            questions = [SubQuestion(id=1, question=state["topic"], rationale="fallback")]
        return {"plan": questions, "revisions": 0}

    def dispatch(state: TeamState) -> list[Send]:
        # Map step: one parallel branch per sub-question.
        return [Send("researcher", {"question": q}) for q in state["plan"]]

    def researcher(task: ResearchTask) -> dict:
        return {"findings": [deps.research(task["question"])]}

    def writer(state: TeamState) -> dict:
        findings = state["findings"]
        sources = build_source_catalog(findings)
        review = state.get("review")
        draft = deps.write(
            WriterInput(
                topic=state["topic"],
                findings=findings,
                sources=sources,
                previous_draft=state.get("draft"),
                review_issues=review.issues if review and not review.approved else [],
                human_feedback=state.get("human_feedback"),
            )
        )
        return {"draft": draft, "sources": sources, "human_feedback": None}

    def reviewer(state: TeamState) -> dict:
        verdict = deps.review(state["topic"], state["draft"], state["sources"])
        # LLM judgement + deterministic rules: the rules always win.
        rule_issues = check_citations(state["draft"], len(state["sources"]))
        if rule_issues:
            verdict = verdict.model_copy(
                update={"approved": False, "issues": [*verdict.issues, *rule_issues]}
            )
        return {"review": verdict, "revisions": state.get("revisions", 0) + 1}

    def route_after_review(state: TeamState) -> Literal["writer", "human_approval", "finalize"]:
        done = state["review"].approved or state["revisions"] > s.max_revisions
        if not done:
            return "writer"
        return "human_approval" if require_approval else "finalize"

    def human_approval(state: TeamState) -> Command:
        # Execution pauses here; the CLI/API resumes it with Command(resume=answer).
        # Code above interrupt() re-runs on resume, so it must be side-effect free.
        answer = interrupt(
            {
                "draft": state["draft"],
                "review": state["review"].model_dump(mode="json"),
                "hint": "Press Enter / type 'approve' to publish, or type feedback to revise.",
            }
        )
        text = str(answer or "").strip()
        if text.lower() in APPROVE_WORDS:
            return Command(goto="finalize")
        return Command(goto="writer", update={"human_feedback": text, "revisions": 0})

    def finalize(state: TeamState) -> dict:
        return {"final_report": render_final_report(state["draft"], state["sources"])}

    # ---------------------------------------------------------------- wiring
    # Retries for transient provider errors (429/5xx/timeouts) are a graph concern,
    # not something every agent should re-implement.
    retry = RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2.0)

    builder = StateGraph(TeamState)
    builder.add_node("planner", planner, retry_policy=retry)
    builder.add_node("researcher", researcher)  # failures are handled inside (degradation)
    builder.add_node("writer", writer, retry_policy=retry)
    builder.add_node("reviewer", reviewer, retry_policy=retry)
    builder.add_node("human_approval", human_approval, destinations=("writer", "finalize"))
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", dispatch, ["researcher"])
    builder.add_edge("researcher", "writer")  # waits for ALL parallel branches
    builder.add_edge("writer", "reviewer")
    builder.add_conditional_edges(
        "reviewer", route_after_review, ["writer", "human_approval", "finalize"]
    )
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer, name="research_team")


def make_graph():
    """Entry point for `langgraph dev` / LangGraph Server (it injects its own checkpointer)."""
    return build_graph(default_deps(), get_settings())
