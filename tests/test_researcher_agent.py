"""Integration test of the real `create_agent` researcher with a scripted chat model.

Checks our wiring of LangChain v1 pieces — tools, ToolStrategy structured output,
middleware — without calling any provider.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from typer.testing import CliRunner

from research_team.agents import make_researcher
from research_team.config import Settings
from research_team.schemas import SubQuestion


class ScriptedChatModel(BaseChatModel):
    """Replays pre-defined AIMessages; records what the agent sent to the model."""

    script: list[AIMessage]
    seen: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.script[len(self.seen) - 1])])


def test_researcher_agent_uses_tools_and_returns_typed_finding(monkeypatch) -> None:
    monkeypatch.setattr(
        "research_team.tools.search.search",
        lambda q, k=None: [
            {"title": "Qdrant docs", "url": "https://qdrant.tech/documentation/", "snippet": "…"}
        ],
    )
    model = ScriptedChatModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[{"name": "web_search", "args": {"query": "qdrant hybrid"}, "id": "c1"}],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "Finding",
                        "id": "c2",
                        "args": {
                            "question_id": 99,
                            "summary": "Qdrant supports hybrid dense+sparse search.",
                            "key_facts": ["RRF fusion is supported"],
                            "sources": [
                                {
                                    "title": "Qdrant docs",
                                    "url": "https://qdrant.tech/documentation/",
                                }
                            ],
                            "confidence": 0.8,
                        },
                    }
                ],
            ),
        ],
        seen=[],
    )
    research = make_researcher(model, Settings())

    finding = research(SubQuestion(id=3, question="Does Qdrant do hybrid search?", rationale=""))

    assert finding.question_id == 3  # overridden by our code, not trusted from the model
    assert finding.sources[0].url == "https://qdrant.tech/documentation/"
    # the second model call saw the tool result
    assert any(isinstance(m, ToolMessage) and "Qdrant docs" in m.content for m in model.seen[1])


def test_researcher_degrades_gracefully_on_failure() -> None:
    class Broken(ScriptedChatModel):
        def _generate(self, *a, **k):
            raise TimeoutError("provider down")

    finding = make_researcher(Broken(script=[], seen=[]), Settings())(
        SubQuestion(id=1, question="q", rationale="")
    )
    assert finding.confidence == 0.0
    assert "failed" in finding.summary


def test_cli_graph_prints_mermaid() -> None:
    from research_team.cli import app

    result = CliRunner().invoke(app, ["graph"])
    assert result.exit_code == 0
    assert "researcher" in result.output and "human_approval" in result.output
