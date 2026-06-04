"""
Feature-driven retrieval routing.

Rather than ask a small local model to choose which knowledge to fetch (unreliable
at 3B scale), we map the frame's measured features to knowledge domains and build
targeted retrieval queries from them. Deterministic, debuggable, and it guarantees
the model always sees the relevant theory.
"""

from __future__ import annotations

from src.vision.schema import ClipFeatures


def select_queries(features: ClipFeatures) -> list[tuple[str | None, str]]:
    """Return (domain, query) pairs to retrieve. domain=None means search all domains."""
    queries: list[tuple[str | None, str]] = [
        (
            "lighting_color",
            f"{features.dominant_lighting}, {features.dominant_tone} color tone, "
            f"key light from {features.dominant_key_light}",
        ),
        (
            "shot_composition",
            f"{features.dominant_shot}, {features.dominant_composition}, {features.dominant_format}",
        ),
        (
            "blocking_mise_en_scene",
            f"{features.dominant_blocking}, {features.dominant_mise_en_scene}",
        ),
    ]

    lighting = features.dominant_lighting.lower()
    tone = features.dominant_tone.lower()
    if "low-key" in lighting or "high-key" in lighting or tone in ("warm", "cool"):
        queries.append(
            (
                "genre_director",
                f"{features.dominant_lighting}, {features.dominant_tone} tone genre conventions",
            )
        )

    return queries
