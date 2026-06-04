"""
Cinematic Scene Understanding — RAG edition (rag-version branch).

Measures a scene with the src/vision pipeline, then grounds its interpretation
in a cinematography knowledge base via src/agent. Two modes: an automatic
one-shot grounded report, and a follow-up chat box.

Run:  streamlit run app.py
Prereqs: Ollama running; models pulled (llama3.2:3b, nomic-embed-text);
         index built (python -m scripts.build_index)
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import streamlit as st
from PIL import Image

from src.agent import analyze
from src.agent.llm import CHAT_MODEL
from src.vision import aggregate, analyze_frame, extract_frames
from src.vision.models import load_shot_classifier, load_yolo

st.set_page_config(page_title="Cinematic Scene Understanding — RAG", layout="wide")

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


st.markdown(
    """
    <style>
      .scene-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 14px 16px; height: 100%;
      }
      .scene-card .label {
        font-size: 0.72rem; letter-spacing: .08em; text-transform: uppercase;
        opacity: .55; margin-bottom: 4px;
      }
      .scene-card .value { font-size: 1.05rem; font-weight: 600; line-height: 1.3; }
      .mini-label {
        font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; opacity:.55;
      }
      .swatch-row { display: flex; gap: 8px; margin-top: 6px; }
      .swatch-wrap { flex: 1; text-align: center; }
      .swatch { height: 48px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.12); }
      .swatch-name { font-size:.74rem; opacity:.7; margin-top:4px; text-transform:capitalize; }
      .chat-zone {
        border: 1px solid rgba(120,160,255,0.35);
        background: rgba(90,130,255,0.06);
        border-radius: 14px; padding: 18px 20px; margin-top: 6px;
      }
      .chat-zone h3 { margin-top: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _models():
    return load_shot_classifier(), load_yolo()


def run_vision_pipeline(frames: list[np.ndarray], source_type: str):
    classifier, yolo = _models()
    frame_features = [analyze_frame(f, classifier, yolo) for f in frames]
    return aggregate(frame_features, source_type)


def load_frames(upload) -> tuple[list[np.ndarray], str]:
    name = upload.name.lower()
    ext = name[name.rfind("."):]
    if ext in VIDEO_EXTS:
        data = upload.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        try:
            tmp.write(data)
            tmp.flush()
            tmp.close()
            frames = extract_frames(tmp.name, num_frames=5)
        finally:
            os.unlink(tmp.name)
        return frames, "video"
    image = np.array(Image.open(upload).convert("RGB"))
    return [image], "image"


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
for key, default in [
    ("features", None), ("frames", None), ("report", None),
    ("sources", None), ("chat", []), ("pending_q", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def reset_analysis():
    st.session_state.features = None
    st.session_state.frames = None
    st.session_state.report = None
    st.session_state.sources = None
    st.session_state.chat = []
    st.session_state.pending_q = None


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Setup")
    model = st.text_input("Ollama reasoning model", value=CHAT_MODEL)
    st.caption("Ollama must be running and the index must be built.")
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. The clip is sampled into 5 frames\n"
        "2. Each frame is measured (shot, light, color, staging)\n"
        "3. Features route to relevant film-theory notes\n"
        "4. A local LLM writes a grounded, cited reading"
    )


# --------------------------------------------------------------------------
# Header + upload
# --------------------------------------------------------------------------
st.title("Cinematic Scene Understanding — RAG")
st.write(
    "Upload a film still or short clip. The system samples and measures it, "
    "then grounds its interpretation in a cinematography knowledge base."
)

upload = st.file_uploader(
    "Scene (image or video)",
    type=["png", "jpg", "jpeg", "mp4", "mov", "avi", "mkv", "m4v"],
)

if upload is not None:
    media_col, run_col = st.columns([3, 1])
    with media_col:
        if upload.type.startswith("image"):
            st.image(upload, use_container_width=True)
        else:
            st.video(upload)
    with run_col:
        if st.button("Analyze scene", type="primary", use_container_width=True):
            reset_analysis()
            with st.spinner("Sampling and measuring the scene..."):
                upload.seek(0)
                frames, source_type = load_frames(upload)
                if not frames:
                    st.error("Could not read any frames from that file.")
                else:
                    st.session_state.frames = frames
                    st.session_state.features = run_vision_pipeline(frames, source_type)
            if st.session_state.features is not None:
                with st.spinner("Grounding interpretation in film theory..."):
                    result = analyze(st.session_state.features, model=model)
                    st.session_state.report = result.interpretation
                    st.session_state.sources = result.sources


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
features = st.session_state.features


def card(label: str, value: str) -> str:
    return f"<div class='scene-card'><div class='label'>{label}</div><div class='value'>{value}</div></div>"


def run_question(question: str, model: str):
    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = analyze(
                features, question=question,
                history=st.session_state.chat[:-1], model=model,
            )
        st.write(result.interpretation)
    st.session_state.chat.append({"role": "assistant", "content": result.interpretation})


if features is not None:
    st.divider()

    # Frames analysed — proves the 5-frame sampling and shows the arc
    if features.source_type == "video" and st.session_state.frames:
        st.subheader(f"Frames analysed ({features.usable_frame_count}/{features.frame_count} usable)")
        cols = st.columns(len(st.session_state.frames))
        for col, frame, ff in zip(cols, st.session_state.frames, features.frames):
            with col:
                st.image(frame, use_container_width=True)
                st.caption(f"{ff.shot} · {ff.lighting}".replace(" lighting", ""))

    # Scene profile
    st.subheader("Scene profile")
    r1 = st.columns(3)
    r1[0].markdown(card("Shot", features.dominant_shot), unsafe_allow_html=True)
    r1[1].markdown(card("Lighting", features.dominant_lighting), unsafe_allow_html=True)
    r1[2].markdown(card("Color tone", features.dominant_tone.title()), unsafe_allow_html=True)
    st.write("")
    r2 = st.columns(3)
    r2[0].markdown(card("Composition", features.dominant_composition), unsafe_allow_html=True)
    r2[1].markdown(card("Blocking", features.dominant_blocking), unsafe_allow_html=True)
    r2[2].markdown(card("Format", features.dominant_format), unsafe_allow_html=True)

    if features.palette_hex:
        st.write("")
        st.markdown("<div class='mini-label'>Palette</div>", unsafe_allow_html=True)
        swatches = "".join(
            f"<div class='swatch' style='flex:1;background:{h}' title='{h}'></div>"
            for h in features.palette_hex
        )
        st.markdown(f"<div class='swatch-row'>{swatches}</div>", unsafe_allow_html=True)
        groups = features.palette_groups()
        pills = "".join(
            f"<span style='display:inline-block;padding:4px 12px;margin:6px 6px 0 0;"
            f"border-radius:999px;background:rgba(255,255,255,0.06);"
            f"border:1px solid rgba(255,255,255,0.12);font-size:.82rem;"
            f"text-transform:capitalize'>{g['label']} · {round(g['share']*100)}%</span>"
            for g in groups
        )
        st.markdown(f"<div style='margin-top:8px'>{pills}</div>", unsafe_allow_html=True)
        st.caption(f"Palette reads as {features.palette_summary()}.")

    if features.detected_objects:
        st.caption("Detected in frame: " + ", ".join(features.detected_objects))

    # Grounded interpretation — centerpiece
    if st.session_state.report:
        st.subheader("Grounded interpretation")
        st.write(st.session_state.report)
        with st.expander("Sources this reading draws on"):
            for s in st.session_state.sources or []:
                st.markdown(f"- **{s['title']}** — {s['domain']} ({s['license']})")

    # ---- Chat: the showpiece ----
    st.markdown("<div class='chat-zone'>", unsafe_allow_html=True)
    st.markdown("### Ask the scene")
    st.caption("This is the interactive part — ask anything about the cinematography and it answers, grounded in film theory.")

    suggestions = [
        "How does the lighting change across the clip?",
        "Why does this scene feel the way it does?",
        "What would warmer lighting change?",
    ]
    sug_cols = st.columns(len(suggestions))
    for c, s in zip(sug_cols, suggestions):
        if c.button(s, use_container_width=True):
            st.session_state.pending_q = s

    for turn in st.session_state.chat:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])

    st.markdown("</div>", unsafe_allow_html=True)

    typed = st.chat_input("Ask about this scene...")
    question = typed or st.session_state.pop("pending_q", None)
    if question:
        run_question(question, model)
