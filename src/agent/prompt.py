"""
Prompt assembly for the grounded cinematography analyst.

The system prompt is the guardrail that turns a generic model into a *grounded*
one: it must interpret the measurements, cite the supplied notes by title, and
refuse to invent theory the notes don't support. This is what makes the output
defensible rather than plausible-sounding filler.
"""

from __future__ import annotations

from src.knowledge import Retrieved
from src.vision.schema import ClipFeatures

SYSTEM = (
    "You are a cinematography analyst. You receive (1) objective visual measurements "
    "extracted from a film scene (sometimes frame by frame) and (2) reference notes on "
    "film theory. Your job: explain what the measurements mean for mood, story, and craft.\n\n"
    "Format:\n"
    "- For an overall interpretation: open with ONE short summary sentence, then give "
    "concise bullet points grouped by element — Lighting, Color, Composition, "
    "Blocking/Staging, and Overall mood. Use a bold label at the start of each bullet. "
    "Keep each bullet to one or two sentences.\n"
    "- For a specific question: answer directly and concisely (use bullets only if they help).\n\n"
    "Rules:\n"
    "- Ground every theory claim in the reference notes and cite them inline by title in "
    "square brackets, e.g. [Film Noir Visual Conventions].\n"
    "- Refer to the actual measured values; if the frames change across the clip, note the progression.\n"
    "- Do NOT invent techniques, films, or facts not supported by the notes or measurements.\n"
    "- If the notes don't cover something the measurements suggest, say so in one short clause.\n"
    "- Be concrete. No filler, no restating these rules."
)


def build_user_message(
    features: ClipFeatures, notes: list[Retrieved], question: str | None
) -> str:
    note_block = "\n\n".join(f"[{n.title}] (domain: {n.domain})\n{n.text}" for n in notes)
    task = (
        question.strip()
        if question
        else "Give an overall cinematographic interpretation of this scene."
    )
    progression = features.frame_progression()
    progression_block = (
        f"\nFRAME-BY-FRAME PROGRESSION (note any changes across the clip):\n{progression}\n"
        if progression
        else ""
    )
    return (
        "VISUAL MEASUREMENTS:\n"
        f"{features.to_prompt_context()}\n"
        f"{progression_block}\n"
        "REFERENCE NOTES (use these for any theory claim; cite by title):\n"
        f"{note_block}\n\n"
        f"TASK: {task}"
    )
