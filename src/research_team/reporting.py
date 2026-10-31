"""Deterministic helpers around the LLM steps.

Rule of thumb: everything that *can* be done with plain code (numbering sources,
checking citations, assembling the final document) is done with plain code.
LLMs are reserved for the parts that need language understanding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from research_team.schemas import Finding, Source

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def build_source_catalog(findings: list[Finding]) -> list[Source]:
    """Deduplicate sources across findings, keeping first-seen order."""
    seen: dict[str, Source] = {}
    for f in sorted(findings, key=lambda f: f.question_id):
        for s in f.sources:
            seen.setdefault(str(s.url), s)
    return list(seen.values())


@dataclass
class WriterInput:
    topic: str
    findings: list[Finding]
    sources: list[Source]
    previous_draft: str | None = None
    review_issues: list[str] = field(default_factory=list)
    human_feedback: str | None = None


def render_writer_context(inp: WriterInput) -> str:
    index = {str(s.url): i for i, s in enumerate(inp.sources, 1)}
    parts = [f"TOPIC: {inp.topic}", "", "SOURCES:"]
    parts += [f"[{i}] {s.title} — {s.url}" for i, s in enumerate(inp.sources, 1)]
    parts += ["", "FINDINGS:"]
    for f in sorted(inp.findings, key=lambda f: f.question_id):
        refs = ", ".join(str(index[str(s.url)]) for s in f.sources if str(s.url) in index)
        header = f"## Finding {f.question_id} (confidence {f.confidence:.2f}; sources: {refs})"
        parts.append("\n" + header)
        parts.append(f.summary)
        parts += [f"- {fact}" for fact in f.key_facts]
    if inp.previous_draft:
        parts += ["", "PREVIOUS DRAFT:", inp.previous_draft]
    if inp.review_issues:
        parts += ["", "REVIEWER ISSUES TO FIX:"] + [f"- {i}" for i in inp.review_issues]
    if inp.human_feedback:
        parts += ["", "HUMAN FEEDBACK (highest priority):", inp.human_feedback]
    return "\n".join(parts)


def check_citations(draft: str, n_sources: int) -> list[str]:
    """Rule-based review: citations exist and point to real sources."""
    issues: list[str] = []
    cited = {int(n) for grp in _CITATION_RE.findall(draft) for n in re.split(r"\s*,\s*", grp)}
    if n_sources and not cited:
        issues.append("The draft contains no [n] citations at all.")
    bad = sorted(n for n in cited if n < 1 or n > n_sources)
    if bad:
        issues.append(f"Citations reference non-existent sources: {bad} (valid: 1..{n_sources}).")
    return issues


def render_final_report(draft: str, sources: list[Source]) -> str:
    lines = [draft.rstrip(), "", "## Sources", ""]
    lines += [f"{i}. [{s.title or s.url}]({s.url})" for i, s in enumerate(sources, 1)]
    return "\n".join(lines) + "\n"


def content_to_text(content: object) -> str:
    """AIMessage.content may be a str or a list of content blocks (v1 format)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in content
            if not isinstance(b, dict) or b.get("type") in (None, "text")
        )
    return str(content)
