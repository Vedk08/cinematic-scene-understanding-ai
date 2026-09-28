"""
Labelled evaluation set for the cinematography RAG system.

Each case is a representative scene (as a ClipFeatures profile) plus the set of
knowledge-note titles a correct system SHOULD surface for it. These labels are
the ground truth retrieval precision/recall is scored against.

Labels are deliberately conservative: only notes whose relevance is
uncontroversial for that scene are listed, so the metrics aren't inflated.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.vision.schema import ClipFeatures


@dataclass
class EvalCase:
    name: str
    features: ClipFeatures
    expected_titles: set[str]
    question: str | None = None


def _clip(**kw) -> ClipFeatures:
    base = dict(
        source_type="video", frame_count=5, usable_frame_count=5,
        dominant_shot="medium shot", dominant_lighting="neutral lighting",
        dominant_tone="neutral", dominant_composition="centered composition",
        dominant_blocking="single-subject blocking",
        dominant_mise_en_scene="moderately detailed mise-en-scène",
        dominant_format="standard widescreen frame", dominant_key_light="frontal",
        palette_hex=["#888888"], palette_proportions=[1.0],
        detected_objects=[], frames=[],
    )
    base.update(kw)
    return ClipFeatures(**base)


CASES: list[EvalCase] = [
    EvalCase(
        name="noir_interrogation",
        features=_clip(
            dominant_shot="close-up shot", dominant_lighting="low-key dramatic lighting",
            dominant_tone="cool", dominant_blocking="single-subject blocking",
            dominant_mise_en_scene="minimal mise-en-scène",
            palette_hex=["#12202f", "#1b2a3a", "#0d1117"], palette_proportions=[0.5, 0.3, 0.2],
        ),
        expected_titles={
            "Low-Key and High-Key Lighting",
            "Film Noir Visual Conventions",
            "Three-Point Lighting",
            "Contrast Ratio and Mood",
            "Shot Sizes and Their Function",
            "Color Temperature and Palette Meaning",
        },
    ),
    EvalCase(
        name="romance_warm_twoshot",
        features=_clip(
            dominant_shot="medium shot", dominant_lighting="high-key lighting",
            dominant_tone="warm", dominant_blocking="intimate / conversational blocking",
            palette_hex=["#caa56a", "#d9b98a", "#8a5a3a"], palette_proportions=[0.5, 0.3, 0.2],
        ),
        expected_titles={
            "Romance and Comedy Visual Conventions",
            "Color Temperature and Palette Meaning",
            "Low-Key and High-Key Lighting",
            "Blocking and Staging",
        },
    ),
    EvalCase(
        name="western_landscape",
        features=_clip(
            dominant_shot="wide shot", dominant_lighting="neutral lighting",
            dominant_tone="warm", dominant_format="cinematic ultra-wide frame",
            dominant_composition="rule-of-thirds composition",
            palette_hex=["#c8a06a", "#9ab0c0", "#7a6a4a"], palette_proportions=[0.5, 0.3, 0.2],
        ),
        expected_titles={
            "Western and Epic Visual Conventions",
            "Shot Sizes and Their Function",
            "Color Temperature and Palette Meaning",
            "Rule of Thirds, Balance, and Negative Space",
        },
    ),
    EvalCase(
        name="horror_dark_confined",
        features=_clip(
            dominant_shot="medium shot", dominant_lighting="low-key dramatic lighting",
            dominant_tone="cool", dominant_composition="rule-of-thirds composition",
            dominant_mise_en_scene="minimal mise-en-scène",
            palette_hex=["#0e1512", "#16221c", "#0a0f0c"], palette_proportions=[0.5, 0.3, 0.2],
        ),
        expected_titles={
            "Horror Visual Conventions",
            "Low-Key and High-Key Lighting",
            "Contrast Ratio and Mood",
            "Color Temperature and Palette Meaning",
        },
    ),
    EvalCase(
        name="scifi_cool_geometric",
        features=_clip(
            dominant_shot="wide shot", dominant_lighting="soft lighting",
            dominant_tone="cool", dominant_composition="centered composition",
            dominant_format="standard widescreen frame",
            palette_hex=["#2a4a6a", "#3a6a8a", "#1a2230"], palette_proportions=[0.5, 0.3, 0.2],
        ),
        expected_titles={
            "Science Fiction Visual Conventions",
            "Color Temperature and Palette Meaning",
            "Shot Sizes and Their Function",
            "Rule of Thirds, Balance, and Negative Space",
        },
    ),
    EvalCase(
        name="twoperson_depth",
        features=_clip(
            dominant_shot="wide shot", dominant_lighting="neutral lighting",
            dominant_tone="neutral", dominant_blocking="separated / emotionally distant blocking",
            dominant_mise_en_scene="dense / cluttered mise-en-scène",
            palette_hex=["#777777", "#555555", "#999999"], palette_proportions=[0.5, 0.3, 0.2],
        ),
        expected_titles={
            "Blocking and Staging",
            "Mise-en-Scene",
            "Staging in Depth",
            "Shot Sizes and Their Function",
        },
    ),
]
