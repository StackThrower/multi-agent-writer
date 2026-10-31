"""Pattern #2: conversational supervisor with subagents wrapped as tools.

This is the pattern LangChain currently recommends instead of the unmaintained
`langgraph-supervisor` package:
https://docs.langchain.com/oss/python/langchain/multi-agent/subagents

    user ⇄ supervisor (create_agent, checkpointer = conversation memory)
              ├── research_specialist  → researcher agent (web tools, typed Finding)
              ├── data_analyst         → analyst agent (calculator)
              └── write_report         → the whole StateGraph workflow from graph.py

Subagents are stateless and get only the task text (context isolation); the
supervisor keeps the conversation. Only the outermost agent owns a checkpointer.
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from research_team import prompts
from research_team.agents import build_researcher_agent, default_deps
from research_team.config import Settings, get_settings
from research_team.graph import build_graph
from research_team.llm import fast_llm, main_llm
from research_team.reporting import content_to_text
from research_team.schemas import Finding
from research_team.tools import calculator


def format_finding(f: Finding) -> str:
    """Subagent output shaping: the supervisor sees a compact, source-rich summary."""
    facts = "\n".join(f"- {x}" for x in f.key_facts)
    sources = "\n".join(f"- {s.title}: {s.url}" for s in f.sources)
    return f"{f.summary}\n\nKey facts:\n{facts}\n\nSources:\n{sources}\n(confidence {f.confidence})"


def build_supervisor(settings: Settings | None = None, checkpointer: Any = None):
    s = settings or get_settings()
    researcher = build_researcher_agent(fast_llm(s), s)
    analyst = create_agent(
        model=fast_llm(s), tools=[calculator], system_prompt=prompts.ANALYST, name="analyst"
    )
    report_workflow = build_graph(default_deps(s), s, require_approval=False)

    @tool(
        "research_specialist",
        description="Find up-to-date facts on the web for ONE focused question. "
        "Returns a summary, key facts and source URLs.",
    )
    def call_researcher(question: str) -> str:
        result = researcher.invoke({"messages": [{"role": "user", "content": question}]})
        finding = result.get("structured_response")
        if finding is None:
            return content_to_text(result["messages"][-1].content)
        return format_finding(finding)

    @tool(
        "data_analyst",
        description="Do calculations/comparisons. Pass all numbers and the exact question.",
    )
    def call_analyst(task: str) -> str:
        result = analyst.invoke({"messages": [{"role": "user", "content": task}]})
        return content_to_text(result["messages"][-1].content)

    @tool(
        "write_report",
        description="Produce a full cited research brief on a topic (slow: several minutes).",
    )
    def call_report_workflow(topic: str) -> str:
        return report_workflow.invoke({"topic": topic})["final_report"]

    return create_agent(
        model=main_llm(s),
        tools=[call_researcher, call_analyst, call_report_workflow],
        system_prompt=prompts.SUPERVISOR,
        checkpointer=checkpointer or InMemorySaver(),
        name="supervisor",
    )
