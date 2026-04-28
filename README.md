# 🎬 Cinematic Scene Understanding AI

An AI-powered cinematic analysis tool that studies video clips, movie stills, and photographs through the lens of film language.

The system analyzes not only what appears in a frame, but how the frame is constructed — including shot type, lighting, color palette, composition, subject placement, blocking, and visual mood.

---

## 🚀 Project Goal

The long-term goal is to build a lightweight “cinema school” AI that can help users understand visual storytelling.

It aims to answer questions like:

- What kind of shot is this?
- How is the lighting affecting the mood?
- What color palette dominates the scene?
- Where is the subject placed in the frame?
- Does the composition follow rule-of-thirds or centered framing?
- How are people blocked in relation to each other?
- What visual mood does the scene create?

---

## 🧠 Current System

The project currently supports:

### 🎞️ Video Clip Analysis
Upload a video and the app will:

- extract representative frames
- analyze each frame
- aggregate results into a clip-level summary

### 🖼️ Still / Photo Analysis
Upload a single movie still, screenshot, or photograph and the app will generate a cinematic breakdown.

---

## ✅ Features Built So Far

### Phase 1 — Video Upload & Frame Extraction
- Built a Streamlit app
- Uploaded video files
- Extracted 5 representative frames using OpenCV
- Displayed frames inside the app

---

### Phase 2 — Shot Type Classification
- Added CLIP zero-shot image classification
- Classified frames into:
  - close-up shot
  - medium shot
  - wide shot
- Displayed confidence scores

---

### Phase 3A — Lighting Analysis
Initially tested CLIP for lighting classification, but it produced unreliable results.

The system was redesigned to use image statistics instead:

- mean brightness
- contrast
- dark pixel ratio

Lighting labels include:

- low-key dramatic lighting
- high-key lighting
- soft lighting
- neutral lighting

---

### Phase 3B — Color Palette & Tone Analysis
- Extracted dominant colors using KMeans clustering
- Displayed visual color palettes with separator lines
- Classified overall tone as:
  - warm
  - cool
  - neutral

---

### Phase 4 — Clip-Level Cinematic Summary
- Aggregated frame-level outputs
- Generated dominant clip-level attributes:
  - dominant shot type
  - dominant lighting
  - dominant color tone
  - overall mood
  - clip palette
- Produced a written scene summary

---

### Phase 5 — Single Still / Photo Mode
- Added support for individual images
- Reused the same cinematic analysis pipeline for stills and photographs
- Enabled analysis of movie frames, screenshots, and visual references

---

### Phase 6 — Composition & Subject Placement
- Added composition analysis
- Detected subject placement:
  - left third
  - center
  - right third
- Added basic framing notes:
  - tight subject framing
  - moderate negative space
  - heavy negative space
- Added rule-of-thirds composition logic

---

### Phase 6.3 — YOLO-Based Subject/Object Detection
Earlier OpenCV HOG and Haar detectors were tested but were unreliable for cinematic frames.

The system was upgraded to YOLO for:

- person detection
- object detection
- subject placement
- composition analysis
- future mise-en-scène analysis

---

### Phase 7 — Blocking & Scene Dynamics
- Added single-subject and multi-subject blocking logic
- Analyzed:
  - number of people
  - subject dominance
  - spacing between people
  - foreground/background separation
  - emotional distance / intimacy cues

Example output:

> The frame uses single-subject blocking with a dominant subject, suggesting focus on an individual presence within the scene.

---

### Phase 7.1 — UI Improvements
- Moved technical metrics into a separate tab
- Added cleaner cinematic report-style cards
- Added optional rule-of-thirds grid overlay
- Kept YOLO detection visualization available in expanders

---

## 🛠️ Tech Stack

- Python
- Streamlit
- OpenCV
- NumPy
- Pillow
- Scikit-learn
- Hugging Face Transformers
- CLIP
- Ultralytics YOLO

---

## 🧩 Architecture

The project uses a hybrid AI + computer vision approach:

```text
Video / Image Input
        ↓
Frame Extraction / Image Loading
        ↓
CLIP Shot Classification
        ↓
OpenCV Lighting Analysis
        ↓
KMeans Color Palette Extraction
        ↓
YOLO Subject & Object Detection
        ↓
Composition + Blocking Logic
        ↓
Cinematic Summary