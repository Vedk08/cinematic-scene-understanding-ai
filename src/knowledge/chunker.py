"""
Chunk knowledge documents into retrievable passages.

The seed notes are short (one passage each), but external sources (Wikibooks /
Wikipedia articles you add later) can be long, so we split on paragraph
boundaries with a soft size target and small overlap to preserve context across
a split. Each chunk inherits its parent document's metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

from .loader import KnowledgeDoc


@dataclass
class Chunk:
    chunk_id: str
    text: str
    title: str
    domain: str
    source: str
    license: str
    url: str


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_document(doc: KnowledgeDoc, target_chars: int = 900, overlap: int = 150) -> list[Chunk]:
    paragraphs = _split_paragraphs(doc.body) or [doc.body]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if not buf:
            buf = para
        elif len(buf) + len(para) + 2 <= target_chars:
            buf += "\n\n" + para
        else:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap and len(buf) > overlap else ""
            buf = (tail + "\n\n" + para).strip() if tail else para
    if buf:
        chunks.append(buf)

    return [
        Chunk(
            chunk_id=f"{doc.doc_id}::{i}",
            text=text,
            title=doc.title,
            domain=doc.domain,
            source=doc.source,
            license=doc.license,
            url=doc.url,
        )
        for i, text in enumerate(chunks)
    ]


def chunk_corpus(docs: list[KnowledgeDoc], **kwargs) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, **kwargs))
    return out
