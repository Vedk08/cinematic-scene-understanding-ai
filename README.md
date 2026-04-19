# 🎬 Cinematic Scene Understanding AI

An AI-powered system that analyzes video scenes through a **cinematic lens** — breaking down shot types, lighting, and color palettes to understand visual storytelling.

---

## 🚀 Project Overview

This project aims to answer:

> *Can we teach AI to “see” like a filmmaker?*

Instead of just detecting objects, this system focuses on:

- 🎥 Shot composition (close-up, medium, wide)
- 💡 Lighting style (low-key, high-key, soft)
- 🎨 Color palette & tone (warm, cool, stylized)
- 📊 Frame-level → Scene-level understanding (coming next)

---

## 🧠 System Architecture

The project combines:

- **CLIP (Zero-shot learning)** → semantic understanding (shot type)
- **Classical Computer Vision** → lighting analysis
- **KMeans clustering** → color palette extraction

👉 This hybrid approach avoids over-relying on deep models where simple vision techniques work better.

---

## 📦 Features (Current)

### ✅ Phase 1 — Video Processing
- Upload video via Streamlit
- Extract frames using OpenCV

---

### ✅ Phase 2 — Shot Classification
- Uses **CLIP (openai/clip-vit-base-patch32)**
- Zero-shot classification:
  - close-up
  - medium shot
  - wide shot

---

### ✅ Phase 3A — Lighting Analysis
- Uses **image statistics instead of ML**
- Extracts:
  - mean brightness
  - contrast
  - dark pixel ratio

- Classifies:
  - low-key lighting
  - high-key lighting
  - soft lighting
  - neutral lighting

👉 Key insight:  
Lighting is a **low-level visual property**, not a semantic concept — better handled with CV than CLIP.

---

### ✅ Phase 3B — Color Palette & Tone
- Extracts dominant colors using **KMeans clustering**
- Displays palette visually (with separator UI)
- Infers tone:
  - warm
  - cool
  - neutral

👉 This aligns the system closer to real cinematic analysis.

---

## 🖥️ Demo UI

For each frame, the system outputs:

- Shot type
- Lighting style
- Color tone
- Dominant color palette

Technical metrics are hidden under:
> “See technical details”

---

## 🔧 Tech Stack

- Python
- Streamlit
- OpenCV
- NumPy
- Scikit-learn (KMeans)
- Hugging Face Transformers (CLIP)

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/cinematic-scene-understanding-ai.git
cd cinematic-scene-understanding-ai
