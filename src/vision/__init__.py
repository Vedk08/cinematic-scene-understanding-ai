"""Cinematic scene understanding — computer-vision feature extraction package."""

from .models import load_shot_classifier, load_yolo
from .report import analyze_frame, aggregate
from .extractors import extract_frames, get_frame_quality
from .schema import ClipFeatures, FrameFeatures

__all__ = [
    "load_shot_classifier",
    "load_yolo",
    "analyze_frame",
    "aggregate",
    "extract_frames",
    "get_frame_quality",
    "ClipFeatures",
    "FrameFeatures",
]
