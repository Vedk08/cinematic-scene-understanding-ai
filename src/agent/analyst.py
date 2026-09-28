"""
The grounded cinematography analyst — orchestration.

Flow:
    features --> route to knowledge domains --> retrieve grounding notes
            --> build a grounded prompt --> local LLM --> cited interpretation

Supports both a default whole-scene interpretation (question=None) and follow-up
questions with conversation history, so the same code powers a one-shot report
and a back-and-forth chat about the scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.knowledge import Retrieved, retrieve
from src.vision.schema import ClipFeatures

from .llm import CHAT_MODEL, chat
from .prompt import SYSTEM, build_user_message
from .router import select_queries


@dataclass
class AgentResult:
    interpretation: str
    sources: list[dict] = field(default_factory=list)


def _gather_notes(
    features: ClipFeatures, question: str | None, per_query_k: int = 2
) -> list[Retrieved]:
    queries = select_queries(features)
    # A user question gets an extra, unfiltered semantic search so we don't miss
    # relevant knowledge outside the feature-routed domains.
    if question:
        queries = [(None, question)] + queries

    notes: list[Retrieved] = []
    seen: set[str] = set()
    for domain, q in queries:
        domains = [domain] if domain else None
        for r in retrieve(q, k=per_query_k, domains=domains):
            key = f"{r.title}::{r.text[:40]}"
            if key not in seen:
                seen.add(key)
                notes.append(r)
    return notes


def analyze(
    features: ClipFeatures,
    question: str | None = None,
    history: list[dict] | None = None,
    model: str = CHAT_MODEL,
) -> AgentResult:
    notes = _gather_notes(features, question)

    messages = [{"role": "system", "content": SYSTEM}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": build_user_message(features, notes, question)})

    text = chat(messages, model=model)
    sources = [
        {"title": n.title, "source": n.source, "license": n.license, "domain": n.domain}
        for n in notes
    ]
    return AgentResult(interpretation=text, sources=sources)
