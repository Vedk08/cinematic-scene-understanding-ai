# 🎬 Cinematic Scene Understanding AI

An AI-powered film analysis tool that helps users understand the visual language of cinema.

The app analyzes video clips and still images through a film-school lens, identifying shot type, lighting, color palette, aspect ratio, composition, blocking, mise-en-scène, visual mood, and film context.

---

## What It Does

Users can:

- Upload a video clip
- Extract representative frames
- Upload a still image or movie frame
- Search film knowledge using OMDb
- Analyze visual language
- Download a cinematic report

---

## Core Features

### Video Clip Analysis
- Extracts representative frames from uploaded videos using OpenCV
- Analyzes each frame individually
- Produces a clip-level cinematic summary

### Still Image Analysis
- Supports movie stills, screenshots, and photographs
- Runs the same visual analysis pipeline on a single image

### Shot Type Classification
Uses CLIP zero-shot image classification to identify:

- Close-up shot
- Medium shot
- Wide shot

### Lighting Analysis
Uses image statistics to classify:

- Low-key dramatic lighting
- High-key lighting
- Soft lighting
- Neutral lighting

### Lighting Setup Inference
Infers possible lighting qualities such as:

- Key light direction
- Fill strength
- Shadow style
- Practical light source possibility

### Color Palette Analysis
Uses KMeans clustering to extract:

- Dominant color palette
- Warm / cool / neutral tone
- Palette visualization

### Aspect Ratio Analysis
Detects frame geometry and common formats:

- 4:3
- 16:9
- 1.85:1
- 2.39:1
- Square
- Vertical/social format

### Composition Analysis
Analyzes:

- Rule of thirds
- Centered composition
- Subject placement
- Framing
- Negative space

### Object and Subject Detection
Uses YOLOv8 to detect people and objects, supporting:

- Subject placement
- Blocking analysis
- Mise-en-scène analysis

### Blocking Analysis
Interprets how subjects are staged:

- Single-subject blocking
- Two-person staging
- Character spacing
- Visual dominance
- Foreground/background relationship

### Mise-en-scène Analysis
Analyzes:

- Setting type
- Visual density
- Props / visible objects
- Subject-environment relationship

### Film Knowledge
Uses OMDb API to retrieve:

- Poster
- Year
- Director
- Genre
- Runtime
- IMDb rating
- Cast
- Plot context

### Downloadable Report
Generates a clean text report containing:

- Film context
- Scene summary
- Visual interpretation
- AI-assisted analysis note

---

## Tech Stack

- Python
- Streamlit
- OpenCV
- NumPy
- Pillow
- Scikit-learn
- Hugging Face Transformers
- CLIP
- YOLOv8
- OMDb API
- Requests

---

## Architecture

```text
Video / Image Input
        ↓
Frame Extraction or Image Loading
        ↓
CLIP Shot Classification
        ↓
OpenCV Lighting + Frame Geometry Analysis
        ↓
KMeans Color Palette Extraction
        ↓
YOLO Subject / Object Detection
        ↓
Composition + Blocking + Mise-en-scène Logic
        ↓
Visual Interpretation Layer
        ↓
Film Knowledge Context
        ↓
Downloadable Cinematic Report