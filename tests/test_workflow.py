"""Workflow tests with stubbed agents: no network, no API keys, deterministic.

The graph depends only on `TeamDeps` callables, so we can script every scenario:
parallel fan-out, reviewer rejection loop, revision budget, human feedback.
"""

from __future__ import annotations

import uuid

import pytest
from langgraph.types import Command

from research_team.agents import TeamDeps
from research_team.config import Settings
from research_team.graph import build_graph, make_checkpointer
from research_team.schemas import Finding, ResearchPlan, Review, Source, SubQuestion


def _plan(topic: str, max_q: int) -> ResearchPlan:
    return ResearchPlan(
        sub_questions=[
            SubQuestion(id=i, question=f"{topic} — aspect {i}", rationale="test")
            for i in range(1, max_q + 1)
        ]
    )


def _research(q: SubQuestion) -> Finding:
    return Finding(
        question_id=q.id,
        summary=f"Answer {q.id}",
        key_facts=[f"fact {q.id}"],
        sources=[Source(title=f"S{q.id}", url=f"https://example.com/{q.id}")],
        confidence=0.9,
    )


class ScriptedWriter:
    def __init__(self) -> None:
        self.calls: list = []

    def __call__(self, inp) -> str:
        self.calls.append(inp)
        return f"# Report v{len(self.calls)}\n\nClaim [1]. Another claim [2]."


class ScriptedReviewer:
    """Returns verdicts from a script, then approves."""

    def __init__(self, verdicts: list[bool]) -> None:
        self.verdicts = list(verdicts)

    def __call__(self, topic: str, draft: str, sources) -> Review:
        ok = self.verdicts.pop(0) if self.verdicts else True
        return Review(approved=ok, score=9 if ok else 4, issues=[] if ok else ["Too shallow"])


def _settings(**kw) -> Settings:
    return Settings(**{"max_subquestions": 3, "max_revisions": 2, **kw})


def _config() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def test_happy_path_parallel_fanout_and_sources() -> None:
    writer = ScriptedWriter()
    deps = TeamDeps(plan=_plan, research=_research, write=writer, review=ScriptedReviewer([True]))
    graph = build_graph(deps, _settings(), require_approval=False)

    result = graph.invoke({"topic": "Vector DBs"})

    assert len(result["findings"]) == 3  # reducer merged all parallel branches
    assert [s.title for s in result["sources"]] == ["S1", "S2", "S3"]
    assert "## Sources" in result["final_report"]
    assert "https://example.com/3" in result["final_report"]
    assert len(writer.calls) == 1


def test_reviewer_rejection_triggers_rewrite_with_issues() -> None:
    writer = ScriptedWriter()
    deps = TeamDeps(
        plan=_plan, research=_research, write=writer, review=ScriptedReviewer([False, True])
    )
    result = build_graph(deps, _settings(), require_approval=False).invoke({"topic": "t"})

    assert len(writer.calls) == 2
    assert writer.calls[1].review_issues == ["Too shallow"]
    assert writer.calls[1].previous_draft.startswith("# Report v1")
    assert result["review"].approved


def test_revision_budget_is_enforced() -> None:
    writer = ScriptedWriter()
    deps = TeamDeps(
        plan=_plan, research=_research, write=writer, review=ScriptedReviewer([False] * 10)
    )
    result = build_graph(deps, _settings(), require_approval=False).invoke({"topic": "t"})

    # 1 initial draft + max_revisions rewrites, then publish best effort
    assert len(writer.calls) == 3
    assert result["review"].approved is False
    assert "final_report" in result


def test_rule_based_citation_check_overrides_llm_approval() -> None:
    def bad_writer(inp) -> str:
        return "# Report\n\nNo citations here."

    deps = TeamDeps(
        plan=_plan, research=_research, write=bad_writer, review=ScriptedReviewer([True] * 10)
    )
    result = build_graph(deps, _settings(max_revisions=0), require_approval=False).invoke(
        {"topic": "t"}
    )
    assert result["review"].approved is False
    assert any("no [n] citations" in i for i in result["review"].issues)


@pytest.mark.parametrize("answer", ["approve", ""])
def test_human_approval_interrupt_and_resume(answer: str) -> None:
    deps = TeamDeps(
        plan=_plan, research=_research, write=ScriptedWriter(), review=ScriptedReviewer([True])
    )
    graph = build_graph(deps, _settings(), checkpointer=make_checkpointer())
    cfg = _config()

    first = graph.invoke({"topic": "t"}, cfg)
    assert "__interrupt__" in first
    assert first["__interrupt__"][0].value["draft"].startswith("# Report v1")

    final = graph.invoke(Command(resume=answer), cfg)
    assert "final_report" in final


def test_human_feedback_sends_draft_back_to_writer() -> None:
    writer = ScriptedWriter()
    deps = TeamDeps(plan=_plan, research=_research, write=writer, review=ScriptedReviewer([]))
    graph = build_graph(deps, _settings(), checkpointer=make_checkpointer())
    cfg = _config()

    graph.invoke({"topic": "t"}, cfg)
    second = graph.invoke(Command(resume="Add a section about pricing"), cfg)
    assert "__interrupt__" in second  # asks again after the rewrite
    assert writer.calls[-1].human_feedback == "Add a section about pricing"

    final = graph.invoke(Command(resume="ok"), cfg)
    assert final["final_report"].startswith("# Report v2")


def test_empty_plan_falls_back_to_whole_topic() -> None:
    deps = TeamDeps(
        plan=lambda t, n: ResearchPlan(sub_questions=[]),
        research=_research,
        write=ScriptedWriter(),
        review=ScriptedReviewer([True]),
    )
    result = build_graph(deps, _settings(), require_approval=False).invoke({"topic": "t"})
    assert len(result["findings"]) == 1
