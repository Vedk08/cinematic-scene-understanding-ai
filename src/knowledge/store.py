"""
Persistent Chroma vector store for the cinematography corpus.

Stores chunk embeddings on disk so the index is built once and reused. Retrieval
supports optional domain filtering, which is what lets us pull, say, only
lighting/color knowledge when the frame's standout feature is its lighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb

from .chunker import Chunk
from .embed import embed_batch, embed_text

COLLECTION = "cinematography"
DEFAULT_DB_PATH = ".chroma"


@dataclass
class Retrieved:
    text: str
    title: str
    domain: str
    source: str
    license: str
    url: str
    distance: float


def _client(db_path: str | Path = DEFAULT_DB_PATH):
    return chromadb.PersistentClient(path=str(db_path))


def build_index(chunks: list[Chunk], db_path: str | Path = DEFAULT_DB_PATH) -> int:
    client = _client(db_path)
    # Rebuild cleanly so re-running the script is idempotent.
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    embeddings = embed_batch([c.text for c in chunks])
    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "title": c.title,
                "domain": c.domain,
                "source": c.source,
                "license": c.license,
                "url": c.url,
            }
            for c in chunks
        ],
    )
    return len(chunks)


def retrieve(
    query: str,
    k: int = 4,
    domains: list[str] | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[Retrieved]:
    client = _client(db_path)
    collection = client.get_collection(COLLECTION)

    where = None
    if domains:
        where = {"domain": {"$in": domains}} if len(domains) > 1 else {"domain": domains[0]}

    res = collection.query(
        query_embeddings=[embed_text(query)],
        n_results=k,
        where=where,
    )

    out: list[Retrieved] = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        out.append(
            Retrieved(
                text=doc,
                title=meta.get("title", ""),
                domain=meta.get("domain", ""),
                source=meta.get("source", ""),
                license=meta.get("license", ""),
                url=meta.get("url", ""),
                distance=float(dist),
            )
        )
    return out
