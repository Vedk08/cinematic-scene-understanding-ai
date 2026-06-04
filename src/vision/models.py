"""
Model loaders, decoupled from the UI.

The original app.py loaded models with @st.cache_resource, which tied the vision
code to Streamlit. Here we use a plain lru_cache so the pipeline can run from a
FastAPI worker, a script, a test, or a notebook — anywhere — and only loads each
model once per process.
"""

from __future__ import annotations

from functools import lru_cache

from transformers import pipeline
from ultralytics import YOLO

CLIP_MODEL = "openai/clip-vit-base-patch32"
YOLO_WEIGHTS = "yolov8n.pt"


@lru_cache(maxsize=1)
def load_shot_classifier():
    """Zero-shot image classifier (CLIP) for shot-type detection."""
    return pipeline("zero-shot-image-classification", model=CLIP_MODEL)


@lru_cache(maxsize=1)
def load_yolo():
    """YOLOv8 detector for people/objects."""
    return YOLO(YOLO_WEIGHTS)
