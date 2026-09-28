"""
Quick retrieval sanity check.

Usage:
    python -m scripts.query_demo "low-key lighting with deep shadows and cool tone"
"""

import sys

from src.knowledge import retrieve


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "low-key dramatic lighting, cool palette, confined framing"
    print(f"Query: {query}\n")
    hits = retrieve(query, k=4)
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h.title}  ({h.domain}, dist={h.distance:.3f})")
        print(f"    source: {h.source} | license: {h.license}")
        print(f"    {h.text[:160].strip()}...\n")


if __name__ == "__main__":
    main()
