"""Domain objects exchanged between agents.

Every hand-off between agents is a validated Pydantic object, never free text.
That is what makes a multi-agent system debuggable and testable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    """One independent line of research produced by the planner."""

    id: int = Field(description="1-based sequence number")
    question: str = Field(description="A focused, self-contained research question")
    rationale: str = Field(description="Why this question matters for the overall topic")


class ResearchPlan(BaseModel):
    sub_questions: list[SubQuestion] = Field(
        description="Independent sub-questions that together cover the topic"
    )


class Source(BaseModel):
    title: str
    # Plain `str` + pattern instead of `HttpUrl`: state is checkpointed with msgpack,
    # and keeping primitives in state keeps checkpoints portable and safe.
    url: str = Field(pattern=r"^https?://\S+$")


class Finding(BaseModel):
    """Structured output of a researcher agent for one sub-question."""

    question_id: int
    summary: str = Field(description="3-6 sentence answer to the sub-question, facts only")
    key_facts: list[str] = Field(default_factory=list, description="Atomic, checkable facts")
    sources: list[Source] = Field(
        default_factory=list, description="Pages the facts were taken from"
    )
    confidence: float = Field(ge=0, le=1, description="Self-assessed confidence 0..1")


class Review(BaseModel):
    """Reviewer verdict. `approved=False` sends the draft back to the writer."""

    approved: bool
    score: int = Field(ge=1, le=10, description="Overall quality 1..10")
    issues: list[str] = Field(
        default_factory=list, description="Concrete, actionable problems to fix"
    )
