"""
Orchestration layer: run the extractors and assemble the structured report.

This is where the old app.py produced `visual_interpretation` and `generate_summary`
strings from hand-written templates. Those are intentionally GONE. This layer now
stops at structured measurement. Interpretation moves to src/agent/, grounded in
retrieved cinematography knowledge — which is the entire reason for the rebuild.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from . import extractors as ex
from .schema import ClipFeatures, FrameFeatures


def analyze_frame(frame: np.ndarray, classifier, yolo_model) -> FrameFeatures:
    quality, _, _ = ex.get_frame_quality(frame)

    if quality == "unusable_black_frame":
        shot, shot_conf = "unavailable", 0.0
    else:
        shot, shot_conf = ex.classify_shot(frame, classifier)

    lighting, mean, contrast, dark = ex.analyze_lighting(frame)
    lighting_setup = ex.infer_lighting_setup(frame, lighting, mean, contrast, dark)
    aspect = ex.analyze_aspect_ratio(frame)
    colors, proportions = ex.extract_colors(frame, k=6)
    tone = ex.analyze_color_tone(colors, proportions)
    detections = ex.detect_objects(frame, yolo_model) if quality == "usable" else []
    composition = ex.analyze_composition(frame, detections, quality)
    blocking = ex.analyze_blocking(frame, detections, quality)
    mise = ex.analyze_mise_en_scene(frame, detections, composition, quality)
    sym_label, sym_score = ex.analyze_symmetry(frame, quality)

    return FrameFeatures(
        quality_status=quality,
        shot=shot,
        shot_confidence=shot_conf,
        lighting=lighting,
        mean_brightness=mean,
        contrast=contrast,
        dark_ratio=dark,
        lighting_setup=lighting_setup,
        aspect_ratio=aspect,
        palette_hex=[ex.rgb_to_hex(c) for c in colors],
        palette_proportions=[float(p) for p in proportions],
        tone=tone,
        composition=composition,
        blocking=blocking,
        mise_en_scene=mise,
        symmetry_label=sym_label,
        symmetry_score=sym_score,
    )


def _mode(values: list[str], default: str = "unavailable") -> str:
    values = [v for v in values if v]
    return Counter(values).most_common(1)[0][0] if values else default


def aggregate(frames: list[FrameFeatures], source_type: str) -> ClipFeatures:
    usable = [f for f in frames if f.quality_status == "usable"]
    base = usable or frames

    detected: list[str] = []
    for f in usable:
        detected.extend(f.mise_en_scene.props_detected)
    detected = sorted(set(detected))

    # Use the palette of the most-usable representative frame as the clip palette.
    rep = base[0] if base else None

    return ClipFeatures(
        source_type=source_type,
        frame_count=len(frames),
        usable_frame_count=len(usable),
        dominant_shot=_mode([f.shot for f in base]),
        dominant_lighting=_mode([f.lighting for f in base]),
        dominant_tone=_mode([f.tone for f in base]),
        dominant_composition=_mode([f.composition.composition_type for f in base]),
        dominant_blocking=_mode([f.blocking.blocking_type for f in base]),
        dominant_mise_en_scene=_mode([f.mise_en_scene.visual_density for f in base]),
        dominant_format=_mode([f.aspect_ratio.format_type for f in base]),
        dominant_key_light=_mode([f.lighting_setup.key_direction for f in base]),
        palette_hex=rep.palette_hex if rep else [],
        palette_proportions=rep.palette_proportions if rep else [],
        detected_objects=detected,
        frames=frames,
    )
