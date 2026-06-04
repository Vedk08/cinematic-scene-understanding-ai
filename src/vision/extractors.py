"""
Pure feature extractors — refactored from the original app.py.

These are the genuine computer-vision functions: frame sampling, CLIP shot
classification, lighting statistics, KMeans palette, YOLO detection, composition,
blocking, and mise-en-scene logic. They are unchanged in behaviour from the
original app — just lifted out of the 1,300-line UI file into testable functions
with no Streamlit imports.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from .schema import (
    AspectRatio,
    Blocking,
    Composition,
    LightingSetup,
    MiseEnScene,
)

# ----------------------------------------------------------------------------
# Frames
# ----------------------------------------------------------------------------

def extract_frames(video_path: str, num_frames: int = 5) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return frames
    indices = [int(i * (total - 1) / max(num_frames - 1, 1)) for i in range(num_frames)]
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def get_frame_quality(frame: np.ndarray) -> tuple[str, float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mean = float(np.mean(gray))
    contrast = float(np.std(gray))
    if mean < 10 and contrast < 5:
        return "unusable_black_frame", mean, contrast
    if mean < 25 and contrast < 10:
        return "too_dark_to_analyze_reliably", mean, contrast
    return "usable", mean, contrast


# ----------------------------------------------------------------------------
# Shot type (CLIP)
# ----------------------------------------------------------------------------

def classify_shot(frame: np.ndarray, classifier) -> tuple[str, float]:
    image = Image.fromarray(frame)
    results = classifier(
        image,
        candidate_labels=["close-up shot", "medium shot", "wide shot"],
        hypothesis_template="This image shows a {}",
    )
    return results[0]["label"], float(results[0]["score"])


# ----------------------------------------------------------------------------
# Lighting
# ----------------------------------------------------------------------------

def analyze_lighting(frame: np.ndarray) -> tuple[str, float, float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mean = float(np.mean(gray))
    contrast = float(np.std(gray))
    dark_ratio = float(np.sum(gray < 50) / gray.size)
    if mean < 10 and contrast < 5:
        label = "unusable black frame"
    elif dark_ratio > 0.5:
        label = "low-key dramatic lighting"
    elif mean > 170:
        label = "high-key lighting"
    elif contrast < 35:
        label = "soft lighting"
    else:
        label = "neutral lighting"
    return label, mean, contrast, dark_ratio


def infer_lighting_setup(frame, lighting_label, mean_brightness, contrast, dark_ratio) -> LightingSetup:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    left_mean = float(np.mean(gray[:, : w // 2]))
    right_mean = float(np.mean(gray[:, w // 2 :]))
    top_mean = float(np.mean(gray[: h // 2, :]))
    bottom_mean = float(np.mean(gray[h // 2 :, :]))
    side_diff = abs(left_mean - right_mean)
    vert_diff = abs(top_mean - bottom_mean)

    if side_diff < 8:
        key_direction = "fairly frontal or evenly distributed"
    elif left_mean > right_mean:
        key_direction = "camera-left"
    else:
        key_direction = "camera-right"

    if vert_diff > 12 and top_mean > bottom_mean:
        vertical_light = "top-weighted light"
    elif vert_diff > 12 and bottom_mean > top_mean:
        vertical_light = "low or under-light influence"
    else:
        vertical_light = "even vertical spread"

    if lighting_label == "low-key dramatic lighting":
        fill_strength, shadow_style = "minimal fill", "strong shadow contrast"
    elif lighting_label == "high-key lighting":
        fill_strength, shadow_style = "strong fill / even exposure", "soft or reduced shadows"
    elif lighting_label == "soft lighting":
        fill_strength, shadow_style = "gentle fill", "soft shadow transitions"
    else:
        fill_strength, shadow_style = "moderate fill", "balanced shadow structure"

    if contrast > 55 and dark_ratio > 0.35:
        backlight_guess = "possible rim/backlight separation"
    elif contrast < 30:
        backlight_guess = "little obvious backlight separation"
    else:
        backlight_guess = "subtle or unclear backlight separation"

    if mean_brightness > 150:
        practical_guess = "possible large soft source, window, or bright practical"
    elif dark_ratio > 0.45:
        practical_guess = "possible motivated practical light or narrow key source"
    else:
        practical_guess = "no strong practical light source inferred"

    return LightingSetup(
        key_direction=key_direction,
        fill_strength=fill_strength,
        shadow_style=shadow_style,
        vertical_light=vertical_light,
        backlight_guess=backlight_guess,
        practical_guess=practical_guess,
    )


# ----------------------------------------------------------------------------
# Geometry: aspect ratio + symmetry
# ----------------------------------------------------------------------------

def analyze_aspect_ratio(frame: np.ndarray) -> AspectRatio:
    h, w, _ = frame.shape
    ratio = w / h
    common = {
        "1.33:1 / 4:3 classic academy ratio": 1.33,
        "1.66:1 European widescreen": 1.66,
        "1.78:1 / 16:9 standard widescreen": 1.78,
        "1.85:1 theatrical widescreen": 1.85,
        "2.35:1 / 2.39:1 cinematic anamorphic widescreen": 2.39,
        "9:16 vertical / social media format": 0.56,
        "1:1 square format": 1.00,
    }
    closest = min(common, key=lambda k: abs(common[k] - ratio))
    if ratio > 2.1:
        fmt = "cinematic ultra-wide frame"
    elif ratio > 1.7:
        fmt = "standard widescreen frame"
    elif 1.2 <= ratio <= 1.5:
        fmt = "classic / boxier frame"
    elif 0.8 <= ratio <= 1.1:
        fmt = "square-like frame"
    elif ratio < 0.8:
        fmt = "vertical frame"
    else:
        fmt = "unusual frame shape"
    return AspectRatio(width=w, height=h, ratio=float(ratio), closest_format=closest, format_type=fmt)


def analyze_symmetry(frame: np.ndarray, quality_status: str) -> tuple[str, float]:
    if quality_status != "usable":
        return "too dark / invalid for symmetry analysis", 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    left = gray[:, : w // 2]
    right = gray[:, w - w // 2 :]
    right_flipped = cv2.flip(right, 1)
    min_w = min(left.shape[1], right_flipped.shape[1])
    diff = np.mean(np.abs(left[:, :min_w].astype("float") - right_flipped[:, :min_w].astype("float")))
    score = max(0.0, 100.0 - float(diff))
    if score > 75:
        label = "strong symmetrical composition"
    elif score > 55:
        label = "moderately balanced composition"
    else:
        label = "asymmetrical composition"
    return label, score


# ----------------------------------------------------------------------------
# Color
# ----------------------------------------------------------------------------

def extract_colors(frame: np.ndarray, k: int = 6):
    small = cv2.resize(frame, (100, 100), interpolation=cv2.INTER_AREA)
    pixels = small.reshape((-1, 3))
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(pixels)
    colors = km.cluster_centers_.astype(int)
    counts = np.bincount(km.labels_)
    pct = counts / counts.sum()
    order = np.argsort(pct)[::-1]
    return colors[order], pct[order]


def rgb_to_hex(c) -> str:
    return "#%02x%02x%02x" % tuple(int(x) for x in c)


def analyze_color_tone(colors, proportions) -> str:
    r = g = b = 0.0
    for (cr, cg, cb), p in zip(colors, proportions):
        r += cr * p
        g += cg * p
        b += cb * p
    if r > b + 20:
        return "warm"
    if b > r + 20:
        return "cool"
    return "neutral"


# ----------------------------------------------------------------------------
# Detection (YOLO) + subjects
# ----------------------------------------------------------------------------

def detect_objects(frame: np.ndarray, yolo_model, conf: float = 0.25) -> list[dict]:
    results = yolo_model.predict(frame, conf=conf, verbose=False)
    if not results:
        return []
    res = results[0]
    names = res.names
    out = []
    for box in res.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        out.append({
            "label": names[cls],
            "confidence": float(box.conf[0]),
            "box": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
        })
    return out


def _people(detections: list[dict]) -> list[dict]:
    return [d for d in detections if d["label"] == "person"]


def _primary_subject(people: list[dict]):
    if not people:
        return None
    return max(people, key=lambda d: d["box"][2] * d["box"][3])


def analyze_composition(frame, detections, quality_status) -> Composition:
    h, w, _ = frame.shape
    area = w * h
    if quality_status != "usable":
        return Composition(0, "unavailable", "composition unavailable due to poor frame quality",
                           "unavailable", 0.0, [])
    people = _people(detections)
    primary = _primary_subject(people)
    objects = [d["label"] for d in detections if d["label"] != "person"]
    if primary is None:
        return Composition(0, "no reliable person detected", "environment / object-focused composition",
                           "subject placement unavailable", 0.0, objects)
    x, _, bw, bh = primary["box"]
    cx = x + bw / 2
    ratio = (bw * bh) / area
    if cx < w / 3 or cx > 2 * w / 3:
        position = "left third" if cx < w / 3 else "right third"
        comp_type = "rule-of-thirds composition"
    else:
        position, comp_type = "center", "centered composition"
    if ratio < 0.12:
        framing = "heavy negative space"
    elif ratio < 0.28:
        framing = "moderate negative space"
    else:
        framing = "tight subject framing"
    return Composition(len(people), position, comp_type, framing, float(ratio), objects)


def analyze_blocking(frame, detections, quality_status) -> Blocking:
    h, w, _ = frame.shape
    diag = float(np.sqrt(w ** 2 + h ** 2))
    if quality_status != "usable":
        return Blocking("blocking unavailable due to poor frame quality", "unavailable", "unavailable", "unavailable")
    people = _people(detections)
    if not people:
        return Blocking("no human blocking detected", "no people detected", "unavailable", "environment-focused frame")
    if len(people) == 1:
        _, _, bw, bh = people[0]["box"]
        ar = (bw * bh) / (w * h)
        if ar > 0.30:
            dominance, depth = "single dominant subject", "foreground-heavy subject presence"
        elif ar > 0.12:
            dominance, depth = "clear primary subject", "moderate subject presence"
        else:
            dominance, depth = "small subject within environment", "environment-dominant staging"
        return Blocking("single-subject blocking", "one person staged alone", dominance, depth)
    people = sorted(people, key=lambda d: d["box"][2] * d["box"][3], reverse=True)
    (x1, y1, w1, h1), (x2, y2, w2, h2) = people[0]["box"], people[1]["box"]
    c1 = np.array([x1 + w1 / 2, y1 + h1 / 2])
    c2 = np.array([x2 + w2 / 2, y2 + h2 / 2])
    dist = float(np.linalg.norm(c1 - c2) / diag)
    area_ratio = max(w1 * h1, w2 * h2) / max(min(w1 * h1, w2 * h2), 1)
    if dist < 0.18:
        rel, btype = "close spatial relationship", "intimate / conversational blocking"
    elif dist < 0.38:
        rel, btype = "moderate spacing between characters", "balanced two-person staging"
    else:
        rel, btype = "strong physical separation", "separated / emotionally distant blocking"
    dominance = "one subject visually dominates the frame" if area_ratio > 2.0 else "subjects have relatively balanced visual weight"
    depth = "foreground-background separation" if area_ratio > 1.8 else "similar depth plane"
    return Blocking(btype, rel, dominance, depth)


_INDOOR = {"chair", "couch", "bed", "dining table", "tv", "laptop", "book", "clock", "vase",
           "refrigerator", "microwave", "oven"}
_OUTDOOR = {"car", "truck", "bus", "traffic light", "stop sign", "bicycle", "motorcycle", "bench", "boat"}


def analyze_mise_en_scene(frame, detections, composition: Composition, quality_status) -> MiseEnScene:
    if quality_status != "usable":
        return MiseEnScene("unavailable", "unavailable", [], "unavailable")
    h, w, _ = frame.shape
    area = w * h
    objs = [d for d in detections if d["label"] != "person"]
    labels = [d["label"] for d in objs]
    unique = sorted(set(labels))
    obj_area_ratio = (sum(d["box"][2] * d["box"][3] for d in objs) / area) if area else 0
    if any(o in _INDOOR for o in labels):
        setting = "interior / domestic or controlled space"
    elif any(o in _OUTDOOR for o in labels):
        setting = "exterior / public or street-like space"
    elif composition.person_count > 0 and not labels:
        setting = "minimal character-focused space"
    else:
        setting = "ambiguous or abstract environment"
    if len(labels) <= 1 and obj_area_ratio < 0.08:
        density = "minimal mise-en-scène"
    elif len(labels) <= 4 and obj_area_ratio < 0.22:
        density = "moderately detailed mise-en-scène"
    else:
        density = "dense / cluttered mise-en-scène"
    if composition.person_count == 0:
        rel = "environment carries the visual emphasis"
    elif composition.framing_note == "tight subject framing":
        rel = "subject dominates over the environment"
    elif composition.framing_note in ("heavy negative space", "moderate negative space"):
        rel = "environment strongly shapes the subject's presence"
    else:
        rel = "subject and environment feel visually balanced"
    return MiseEnScene(setting, density, unique, rel)
