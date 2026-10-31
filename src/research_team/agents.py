"""Production implementations of the team members.

Each factory returns a *plain callable* with a narrow signature. The graph only
depends on these signatures (see `TeamDeps`), so tests can inject stubs and run
the whole workflow without network or API keys.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from research_team import prompts
from research_team.config import Settings, get_settings
from research_team.llm import fast_llm, main_llm, structured
from research_team.reporting import WriterInput, content_to_text, render_writer_context
from research_team.schemas import Finding, ResearchPlan, Review, Source, SubQuestion
from research_team.tools import fetch_page, web_search

log = logging.getLogger(__name__)

PlanFn = Callable[[str, int], ResearchPlan]
ResearchFn = Callable[[SubQuestion], Finding]
WriteFn = Callable[[WriterInput], str]
ReviewFn = Callable[[str, str, list[Source]], Review]


@dataclass(frozen=True)
class TeamDeps:
    """Everything the workflow needs from the outside world."""

    plan: PlanFn
    research: ResearchFn
    write: WriteFn
    review: ReviewFn


# --------------------------------------------------------------------------- planner
def make_planner(llm: BaseChatModel) -> PlanFn:
    planner_llm = structured(llm, ResearchPlan)

    def plan(topic: str, max_questions: int) -> ResearchPlan:
        result = planner_llm.invoke(
            [
                SystemMessage(prompts.PLANNER.format(max_q=max_questions)),
                HumanMessage(topic),
            ]
        )
        # Never trust the model with hard limits — enforce them in code.
        questions = result.sub_questions[:max_questions]
        return ResearchPlan(
            sub_questions=[q.model_copy(update={"id": i}) for i, q in enumerate(questions, 1)]
        )

    return plan


# ------------------------------------------------------------------------ researcher
def build_researcher_agent(llm: BaseChatModel, settings: Settings):
    """A ReAct-style agent (LangChain v1 `create_agent`) with web tools and typed output."""
    return create_agent(
        model=llm,
        tools=[web_search, fetch_page],
        system_prompt=prompts.RESEARCHER,
        # Passing the schema type lets LangChain pick ProviderStrategy (native structured
        # output) when the model profile supports it, otherwise ToolStrategy.
        response_format=Finding,
        middleware=[
            # Guardrails against runaway loops / costs.
            ToolCallLimitMiddleware(
                run_limit=settings.researcher_max_tool_calls, exit_behavior="continue"
            ),
            ModelCallLimitMiddleware(
                run_limit=settings.researcher_max_tool_calls + 4, exit_behavior="end"
            ),
        ],
        name="researcher",
    )


def make_researcher(llm: BaseChatModel, settings: Settings) -> ResearchFn:
    agent = build_researcher_agent(llm, settings)

    def research(question: SubQuestion) -> Finding:
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": question.question}]})
            finding: Finding | None = result.get("structured_response")
            if finding is None:
                raise RuntimeError("agent stopped without a structured answer")
            return finding.model_copy(update={"question_id": question.id})
        except Exception as exc:
            # Graceful degradation: one failed researcher must not kill the whole report.
            log.warning("researcher %s failed: %r", question.id, exc)
            return Finding(
                question_id=question.id,
                summary=f"Research failed for this sub-question ({type(exc).__name__}).",
                confidence=0.0,
            )

    return research


# ---------------------------------------------------------------------------- writer
def make_writer(llm: BaseChatModel) -> WriteFn:
    def write(inp: WriterInput) -> str:
        msg = llm.invoke([SystemMessage(prompts.WRITER), HumanMessage(render_writer_context(inp))])
        return content_to_text(msg.content).strip()

    return write


# -------------------------------------------------------------------------- reviewer
def make_reviewer(llm: BaseChatModel) -> ReviewFn:
    reviewer_llm = structured(llm, Review)

    def review(topic: str, draft: str, sources: list[Source]) -> Review:
        src = "\n".join(f"[{i}] {s.title} — {s.url}" for i, s in enumerate(sources, 1))
        return reviewer_llm.invoke(
            [
                SystemMessage(prompts.REVIEWER),
                HumanMessage(f"TOPIC: {topic}\n\nSOURCES:\n{src}\n\nDRAFT:\n{draft}"),
            ]
        )

    return review


def default_deps(settings: Settings | None = None) -> TeamDeps:
    s = settings or get_settings()
    smart, fast = main_llm(s), fast_llm(s)
    return TeamDeps(
        plan=make_planner(smart),
        research=make_researcher(fast, s),
        write=make_writer(smart),
        review=make_reviewer(smart),
    )
