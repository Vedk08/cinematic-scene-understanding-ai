"""
End-to-end demo of the grounded analyst on a sample scene (no video needed).

Builds a noir-like ClipFeatures by hand so you can test the agent independently
of the heavy CV models, then prints the grounded, cited interpretation.

Usage (from repo root, with Ollama running and the index built):
    python -m scripts.agent_demo
    python -m scripts.agent_demo "why does this scene feel tense?"

Prereqs:
    ollama pull llama3.2:3b
    python -m scripts.build_index
"""

import sys

from src.agent import analyze
from src.vision.schema import ClipFeatures


def sample_scene() -> ClipFeatures:
    return ClipFeatures(
        source_type="video",
        frame_count=5,
        usable_frame_count=5,
        dominant_shot="close-up shot",
        dominant_lighting="low-key dramatic lighting",
        dominant_tone="cool",
        dominant_composition="rule-of-thirds composition",
        dominant_blocking="single-subject blocking",
        dominant_mise_en_scene="minimal mise-en-scène",
        dominant_format="cinematic ultra-wide frame",
        dominant_key_light="camera-left",
        palette_hex=["#1b2a3a", "#33424f", "#0d1117"],
        palette_proportions=[0.5, 0.3, 0.2],
        detected_objects=["chair", "bottle"],
        frames=[],
    )


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else None
    result = analyze(sample_scene(), question=question)

    print("=== GROUNDED INTERPRETATION ===\n")
    print(result.interpretation)
    print("\n=== SOURCES RETRIEVED ===")
    for s in result.sources:
        print(f"  - {s['title']}  ({s['domain']}, {s['license']})")


if __name__ == "__main__":
    main()
