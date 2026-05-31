# 🎬 Cinematic Scene Understanding AI

A computer vision and AI-powered film analysis tool that helps filmmakers, cinematography enthusiasts, and film students analyze the visual language of movies.

The system extracts frames from videos or analyzes individual stills and automatically identifies cinematic characteristics such as framing, lighting, composition, blocking, color palette, aspect ratio, mise-en-scène, and visual storytelling techniques.

---

## Project Goal

Traditional AI tools can describe an image.

This project attempts to analyze a frame the way a cinematographer, director, or film student would.

The long-term goal is to build an AI-powered cinema school capable of understanding:

* Shot composition
* Lighting
* Blocking
* Mise-en-scène
* Color design
* Aspect ratios
* Visual storytelling
* Cinematography techniques
* Film metadata and production context

---

# Current Features

## Video Analysis

Upload a video and automatically:

* Extract representative frames
* Analyze each frame individually
* Generate a clip-level visual summary

---

## Still Image Analysis

Upload a single frame, screenshot, photograph, or movie still and receive a detailed cinematic breakdown.

---

## Shot Type Classification

Uses CLIP Zero-Shot Classification to identify:

* Close-Up Shot
* Medium Shot
* Wide Shot

---

## Lighting Analysis

Detects:

* Low-Key Dramatic Lighting
* High-Key Lighting
* Soft Lighting
* Neutral Lighting

Also calculates:

* Brightness
* Contrast
* Dark Pixel Ratio

---

## Lighting Setup Inference (Phase 11)

Attempts to infer:

* Key Light Direction
* Fill Strength
* Shadow Style
* Vertical Light Distribution
* Backlight / Rim Light Possibilities
* Practical Light Source Guesses

Generates a lighting interpretation based on image brightness distribution.

---

## Color Analysis

Extracts dominant colors using K-Means clustering.

Provides:

* Dominant Color Palette
* Color Tone Classification

  * Warm
  * Cool
  * Neutral
* Palette HEX Values

---

## Aspect Ratio Analysis

Identifies likely frame format:

* 4:3
* 16:9
* 1.85:1
* 2.39:1
* Vertical Formats
* Square Formats

Provides visual interpretation of how frame geometry affects storytelling.

---

## Composition Analysis

Analyzes:

* Rule of Thirds
* Subject Placement
* Negative Space
* Framing Tightness

Can display:

* Rule-of-Thirds Overlay

---

## Object Detection (YOLOv8)

Detects:

* People
* Props
* Objects

Used to improve:

* Composition Analysis
* Blocking Analysis
* Mise-en-scène Analysis

---

## Blocking Analysis

Identifies:

* Single Subject Blocking
* Two-Person Staging
* Character Separation
* Visual Dominance
* Spatial Relationships

Provides natural language interpretation of character arrangement.

---

## Symmetry Analysis

Evaluates:

* Balanced Composition
* Moderate Symmetry
* Asymmetrical Frames

---

## Mise-en-scène Analysis

Attempts to understand:

* Visual Density
* Environment Type
* Props
* Subject-Environment Relationship

Provides cinematic interpretation rather than simple object detection.

---

## Visual Language Interpretation

Combines:

* Shot Type
* Lighting
* Color
* Composition
* Blocking
* Mise-en-scène

Into a film-school style explanation of how the image communicates meaning.

---

## Film Knowledge Search (Phase 13)

Integrated OMDb API support.

Search any film and retrieve:

* Poster
* Director
* Cast
* Genre
* Runtime
* IMDb Rating
* Plot Summary

Includes research prompts for deeper cinematography analysis.

---

# Technical Stack

### Frontend

* Streamlit

### Computer Vision

* OpenCV

### Object Detection

* YOLOv8

### Machine Learning

* CLIP
* Hugging Face Transformers

### Data Processing

* NumPy
* Scikit-Learn

### Film Metadata

* OMDb API

---

# Project Roadmap

## Completed

### Phase 1

Video Upload & Frame Extraction

### Phase 2

Shot Classification using CLIP

### Phase 3

Lighting & Color Analysis

### Phase 4

Clip-Level Summaries

### Phase 5

Still Image Analysis

### Phase 6

Object Detection & Composition Analysis

### Phase 7

Blocking & Mise-en-scène Analysis

### Phase 8

Visual Interpretation Layer

### Phase 9

Technical Details Separation

### Phase 10

Advanced Cinematic UI & Analysis Cards

### Phase 11

Lighting Setup Inference

### Phase 12

Film Research Assistant

### Phase 13

OMDb Film Metadata Integration

---

## Upcoming

### Phase 14

Exportable Cinematic Reports (PDF)

### Phase 15

Movie Knowledge Database

### Phase 16

Advanced Cinematography Style Detection

### Phase 17

Director / Cinematographer Signature Analysis

### Phase 18

AI-Powered Film School Assistant

---

# Future Vision

Eventually the system should be capable of:

* Understanding cinematography at a film-school level
* Inferring lighting setups
* Identifying visual motifs
* Comparing directors' visual styles
* Explaining scenes visually
* Teaching filmmaking concepts interactively

The goal is not simply image recognition.

The goal is cinematic understanding.

---

Created by Vedansh Kumar
MSc Computer Science — TU Dresden
