"""Local RAG knowledge layer: load -> chunk -> embed -> store -> retrieve."""

from .loader import KnowledgeDoc, load_corpus, VALID_DOMAINS
from .chunker import Chunk, chunk_corpus, chunk_document
from .store import Retrieved, build_index, retrieve

__all__ = [
    "KnowledgeDoc",
    "load_corpus",
    "VALID_DOMAINS",
    "Chunk",
    "chunk_corpus",
    "chunk_document",
    "Retrieved",
    "build_index",
    "retrieve",
]
