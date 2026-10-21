"""Quoted-term anchors and answer-phrase splitting shared by the coverage proxies."""
from __future__ import annotations

import re

# Quoted UI terms in a question: "...", `...`, “...”, '...'. Frozen in E1 (commit 5894287).
QUOTED = re.compile(r'"([^"]{2,80})"|`([^`]{2,80})`|“([^”]{2,80})”|\'([^\']{3,80})\'')


def anchors(question: str) -> list[str]:
    """Ordered, de-duplicated, lower-cased quoted terms."""
    out: list[str] = []
    for m in QUOTED.finditer(question):
        t = next(g for g in m.groups() if g).strip().lower()
        if t and t not in out:
            out.append(t)
    return out


def quoted_terms(question: str) -> set[str]:
    return {next(g for g in m.groups() if g).strip().lower() for m in QUOTED.finditer(question)}


# One "Raw state result" section of an AgentRunbook-R memory context.
BLOCK_SPLIT = re.compile(r"(?=Raw state result \d+)")


def raw_blocks(memory_context: list[dict]) -> list[str]:
    """Lower-cased text of every raw-state block in a rendered memory context."""
    text = "\n".join(x["value"] for x in memory_context if x.get("type") == "text")
    blocks = [b for b in BLOCK_SPLIT.split(text) if b.startswith("Raw state result")]
    return [b.lower() for b in blocks]


def coverage_label_blocks(a: list[str], blocks: list[str]) -> str:
    """full / partial / none / no-anchor for anchors `a` over lower-cased blocks."""
    if not a:
        return "no-anchor"
    if any(all(x in b for x in a) for b in blocks):
        return "full"
    if any(any(x in b for x in a) for b in blocks):
        return "partial"
    return "none"


def coverage_label(q: str, ctx: list[dict]) -> str:
    """Quoted-anchor page-coverage label (E1 §3.8, page_coverage.py)."""
    a = anchors(q)
    if not a:
        return "no-anchor"
    return coverage_label_blocks(a, raw_blocks(ctx))
