"""`fetch_page` tool: downloads a public web page and returns readable text.

Includes a basic SSRF guard: an LLM decides which URL to open, so the tool must
refuse localhost / private networks / non-HTTP schemes. Treat every URL coming
from a model as untrusted input.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from langchain.tools import tool

from research_team.config import get_settings

_USER_AGENT = "Mozilla/5.0 (compatible; research-team/0.1; +https://github.com/)"


def is_public_http_url(url: str) -> bool:
    """True only for http(s) URLs whose host resolves exclusively to public IPs."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()
    lines = (line.strip() for line in soup.get_text("\n").splitlines())
    return "\n".join(line for line in lines if line)


def fetch(url: str) -> str:
    s = get_settings()
    if not is_public_http_url(url):
        raise ValueError(f"URL is not a public http(s) address: {url}")
    with httpx.Client(
        timeout=s.http_timeout_s, follow_redirects=True, headers={"User-Agent": _USER_AGENT}
    ) as client:
        resp = client.get(url)
        resp.raise_for_status()
        # re-check after redirects: a public URL may redirect to an internal one
        if not is_public_http_url(str(resp.url)):
            raise ValueError(f"Redirected to a non-public address: {resp.url}")
        ctype = resp.headers.get("content-type", "")
        text = html_to_text(resp.text) if "html" in ctype else resp.text
    return text[: s.fetch_max_chars]


@tool
def fetch_page(url: str) -> str:
    """Open a web page from search results and return its main text (truncated).

    Use it when a snippet is not enough to answer with confidence.
    """
    try:
        return fetch(url)
    except Exception as exc:
        return f"Could not fetch {url}: {exc!r}"
