"""
Local chat LLM via Ollama.

Uses the ollama library directly (same as the embeddings) for reliability — the
LangChain/LangGraph stack is installed and you can wrap this orchestration in it
later for the resume keyword, but correctness of the local call comes first.

Pull the reasoning model once:
    ollama pull llama3.2:3b
Swap to a bigger model by changing CHAT_MODEL (e.g. "llama3.1" for 8B).
"""

from __future__ import annotations

import ollama

CHAT_MODEL = "llama3.2:3b"


def chat(messages: list[dict], model: str = CHAT_MODEL, temperature: float = 0.3) -> str:
    """messages = [{"role": "system"|"user"|"assistant", "content": "..."}]."""
    resp = ollama.chat(model=model, messages=messages, options={"temperature": temperature})
    return resp["message"]["content"]
