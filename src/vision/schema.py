"""
Structured feature schema for cinematic scene analysis.

This module defines the *contract* between the computer-vision pipeline and the
reasoning layer (agent + RAG). Every extractor produces plain, typed data here;
no prose, no interpretation. Turning these features into film-school language is
the job of the agent layer, grounded in retrieved cinematography knowledge.

Keeping interpretation out of this layer is the whole point of the refactor:
the old app.py mixed pixel measurements with hand-written template sentences.
Here, measurements stay measurements.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field, asdict
from typing import Any


def _color_family(hex_str: str) -> tuple[str, str]:
    """Classify a hex color into a (qualifier, family) pair, e.g. ('deep', 'blues')."""
    h = hex_str.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    hue, sat, val = colorsys.rgb_to_hsv(r, g, b)
    hue *= 360

    if val < 0.18:
        return "", "blacks"
    if sat < 0.15:
        if val > 0.80:
            return "light", "neutrals"
        if val < 0.35:
            return "dark", "neutrals"
        return "", "grays"

    qualifier = "deep" if val < 0.38 else ("bright" if val > 0.82 else "")
    if hue < 15 or hue >= 345:
        family = "reds"
    elif hue < 45:
        family = "ambers"
    elif hue < 65:
        family = "yellows"
    elif hue < 160:
        family = "greens"
    elif hue < 195:
        family = "teals"
    elif hue < 255:
        family = "blues"
    elif hue < 290:
        family = "purples"
    else:
        family = "pinks"
    return qualifier, family


@dataclass
class LightingSetup:
    key_direction: str
    fill_strength: str
    shadow_style: str
    vertical_light: str
    backlight_guess: str
    practical_guess: str


@dataclass
class AspectRatio:
    width: int
    height: int
    ratio: float
    closest_format: str
    format_type: str


@dataclass
class Composition:
    person_count: int
    subject_position: str
    composition_type: str
    framing_note: str
    subject_area_ratio: float
    object_labels: list[str] = field(default_factory=list)


@dataclass
class Blocking:
    blocking_type: str
    relationship: str
    dominance: str
    depth: str


@dataclass
class MiseEnScene:
    setting_type: str
    visual_density: str
    props_detected: list[str]
    subject_environment_relationship: str


@dataclass
class FrameFeatures:
    """Everything the CV pipeline measures about a single frame."""
    quality_status: str
    shot: str
    shot_confidence: float
    lighting: str
    mean_brightness: float
    contrast: float
    dark_ratio: float
    lighting_setup: LightingSetup
    aspect_ratio: AspectRatio
    palette_hex: list[str]
    palette_proportions: list[float]
    tone: str
    composition: Composition
    blocking: Blocking
    mise_en_scene: MiseEnScene
    symmetry_label: str
    symmetry_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClipFeatures:
    """Clip-level rollup across sampled frames. This is what the agent reads."""
    source_type: str  # "video" or "image"
    frame_count: int
    usable_frame_count: int
    dominant_shot: str
    dominant_lighting: str
    dominant_tone: str
    dominant_composition: str
    dominant_blocking: str
    dominant_mise_en_scene: str
    dominant_format: str
    dominant_key_light: str
    palette_hex: list[str]
    palette_proportions: list[float]
    detected_objects: list[str]
    frames: list[FrameFeatures] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def palette_groups(self) -> list[dict]:
        """Group the palette into color families with a share of the frame.

        Aggregates by family (so 'blues' and 'deep blues' merge into one blue
        group), labelling each with its dominant qualifier. Returns
        [{'label': 'deep blues', 'share': 0.58, 'hexes': [...]}, ...] by share.
        """
        fam: dict[str, dict] = {}
        props = self.palette_proportions or [1.0] * len(self.palette_hex)
        for hex_c, share in zip(self.palette_hex, props):
            qual, family = _color_family(hex_c)
            e = fam.setdefault(family, {"family": family, "share": 0.0, "hexes": [], "quals": {}})
            e["share"] += float(share)
            e["hexes"].append(hex_c)
            e["quals"][qual] = e["quals"].get(qual, 0.0) + float(share)
        groups = []
        for e in fam.values():
            best_qual = max(e["quals"], key=e["quals"].get)
            groups.append({
                "label": f"{best_qual} {e['family']}".strip(),
                "family": e["family"],
                "share": e["share"],
                "hexes": e["hexes"],
            })
        return sorted(groups, key=lambda g: g["share"], reverse=True)

    def palette_summary(self) -> str:
        """Human phrase, e.g. 'mostly deep blues with warm amber accents'."""
        groups = self.palette_groups()
        if not groups:
            return "no dominant palette"
        lead = groups[0]["label"]
        accents = [g["label"] for g in groups[1:] if g["share"] >= 0.12]
        if not accents:
            return f"mostly {lead}"
        return f"mostly {lead} with {', '.join(accents)} accents"

    def to_prompt_context(self) -> str:
        """A compact, readable summary the LLM can reason over.

        Note this is deliberately *neutral* description of measurements — it does
        NOT tell the model what they mean. The model interprets, grounded in RAG.
        """
        objects = ", ".join(self.detected_objects) if self.detected_objects else "none detected"
        return (
            f"Source: {self.source_type} ({self.usable_frame_count}/{self.frame_count} usable frames)\n"
            f"Dominant shot type: {self.dominant_shot}\n"
            f"Dominant lighting: {self.dominant_lighting}\n"
            f"Inferred key-light direction: {self.dominant_key_light}\n"
            f"Color tone: {self.dominant_tone}\n"
            f"Color palette: {self.palette_summary()}\n"
            f"Frame format: {self.dominant_format}\n"
            f"Composition: {self.dominant_composition}\n"
            f"Blocking: {self.dominant_blocking}\n"
            f"Mise-en-scene density: {self.dominant_mise_en_scene}\n"
            f"Detected objects/props: {objects}"
        )

    def frame_progression(self) -> str:
        """Per-frame arc across the clip — the whole reason for sampling 5 frames.

        Lighting and shot size shift within a scene; collapsing to a single
        dominant value hides that. This exposes the progression so the model can
        reason about change ("opens wide and bright, tightens to a dark close-up").
        Returns an empty string for stills.
        """
        if self.source_type != "video" or not self.frames:
            return ""
        lines = []
        for i, f in enumerate(self.frames, 1):
            if f.quality_status != "usable":
                lines.append(f"Frame {i}: too dark / unusable")
            else:
                lines.append(
                    f"Frame {i}: {f.shot}, {f.lighting}, {f.tone} tone, "
                    f"{f.composition.composition_type}"
                )
        return "\n".join(lines)
