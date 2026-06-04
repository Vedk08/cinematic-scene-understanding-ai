"""
Build the cinematography vector index from the knowledge/ corpus.

Usage (from the repo root):
    python -m scripts.build_index

Prereqs:
    ollama pull nomic-embed-text   # embedding model must be available
"""

from src.knowledge import load_corpus, chunk_corpus, build_index


def main() -> None:
    docs = load_corpus("knowledge")
    print(f"Loaded {len(docs)} documents from knowledge/")
    chunks = chunk_corpus(docs)
    print(f"Split into {len(chunks)} chunks. Embedding with Ollama...")
    n = build_index(chunks)
    print(f"Indexed {n} chunks into Chroma at .chroma/")
    by_domain: dict[str, int] = {}
    for c in chunks:
        by_domain[c.domain] = by_domain.get(c.domain, 0) + 1
    for domain, count in sorted(by_domain.items()):
        print(f"  {domain}: {count} chunks")


if __name__ == "__main__":
    main()
