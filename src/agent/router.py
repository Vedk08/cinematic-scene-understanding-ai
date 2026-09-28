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
    tone = features.dominant_tone.lower()
    return [
        (
            "lighting_color",
            f"{features.dominant_lighting}, key light from {features.dominant_key_light}",
        ),
        (
            "lighting_color",
            f"{tone} color tone, {features.palette_summary()}, color meaning and mood",
        ),
        (
            "shot_composition",
            f"{features.dominant_shot}, {features.dominant_composition}, {features.dominant_format}",
        ),
        (
            "blocking_mise_en_scene",
            f"{features.dominant_blocking}, {features.dominant_mise_en_scene}",
        ),
        (
            "genre_director",
            f"{features.dominant_lighting}, {tone} tone, {features.dominant_composition}, "
            f"{features.dominant_format} genre conventions",
        ),
    ]
