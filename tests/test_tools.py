from __future__ import annotations

import pytest

from research_team.reporting import build_source_catalog, check_citations
from research_team.schemas import Finding, Source
from research_team.tools.calculator import calculator, safe_eval
from research_team.tools.web import html_to_text, is_public_http_url


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("2 + 2 * 3", 8),
        ("(1250 - 980) / 980 * 100", pytest.approx(27.551, rel=1e-3)),
        ("sqrt(16) + abs(-2)", 6),
        ("round(pi, 2)", 3.14),
    ],
)
def test_safe_eval(expr: str, expected: float) -> None:
    assert safe_eval(expr) == expected


@pytest.mark.parametrize(
    "malicious", ["__import__('os').system('id')", "open('/etc/passwd')", "9 ** 99999", "x.y"]
)
def test_calculator_rejects_code(malicious: str) -> None:
    assert calculator.invoke({"expression": malicious}).startswith("Error")


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000",
        "http://127.0.0.1/admin",
        "http://10.0.0.5/",
        "file:///etc/passwd",
        "ftp://example.com",
        "http://169.254.169.254/latest/meta-data/",
    ],
)
def test_ssrf_guard_blocks_internal_urls(url: str) -> None:
    assert not is_public_http_url(url)


def test_html_to_text_strips_noise() -> None:
    html = "<html><nav>menu</nav><script>x()</script><p>Hello</p><footer>f</footer></html>"
    assert html_to_text(html) == "Hello"


def test_source_catalog_dedupes_in_question_order() -> None:
    a = Source(title="A", url="https://a.com")
    b = Source(title="B", url="https://b.com")
    findings = [
        Finding(question_id=2, summary="", sources=[b, a], confidence=1),
        Finding(question_id=1, summary="", sources=[a], confidence=1),
    ]
    assert [s.title for s in build_source_catalog(findings)] == ["A", "B"]


def test_check_citations() -> None:
    assert check_citations("Text [1], more [2, 3].", 3) == []
    assert check_citations("No refs.", 2) == ["The draft contains no [n] citations at all."]
    assert "non-existent" in check_citations("Bad [7].", 2)[0]
