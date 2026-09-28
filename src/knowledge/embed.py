"""
Local embeddings via Ollama (nomic-embed-text).

No API keys, no network egress beyond localhost — the Ollama server runs on your
machine at http://localhost:11434. Pull the model once with:

    ollama pull nomic-embed-text
"""

from __future__ import annotations

import ollama

EMBED_MODEL = "nomic-embed-text"


def embed_text(text: str, model: str = EMBED_MODEL) -> list[float]:
    resp = ollama.embeddings(model=model, prompt=text)
    return resp["embedding"]


def embed_batch(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    # Ollama's embeddings endpoint is one-prompt-at-a-time; loop explicitly so the
    # behaviour is obvious. For a corpus this size that's perfectly fast.
    return [embed_text(t, model=model) for t in texts]
