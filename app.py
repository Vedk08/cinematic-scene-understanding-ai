import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from sklearn.cluster import KMeans
from transformers import pipeline


st.title("🎬 Cinematic AI - Phase 3B")
st.write("Upload a video, extract 5 frames, classify shot type, analyze lighting, and extract dominant color palettes.")


@st.cache_resource
def load_classifier():
    return pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32"
    )


def extract_frames(video_path, num_frames=5):
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return frames

    if num_frames == 1:
        frame_indices = [total_frames // 2]
    else:
        frame_indices = [
            int(i * (total_frames - 1) / (num_frames - 1))
            for i in range(num_frames)
        ]

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()
    return frames


def classify_shot(image, classifier):
    labels = [
        "close-up shot",
        "medium shot",
        "wide shot"
    ]
    return classifier(
        image,
        candidate_labels=labels,
        hypothesis_template="This image shows a {}"
    )


def analyze_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    mean = float(np.mean(gray))
    contrast = float(np.std(gray))
    dark_ratio = float(np.sum(gray < 50) / gray.size)

    if dark_ratio > 0.5:
        label = "low-key dramatic lighting"
    elif mean > 170:
        label = "high-key lighting"
    elif contrast < 35:
        label = "soft lighting"
    else:
        label = "neutral lighting"

    return label, mean, contrast, dark_ratio


def extract_colors(frame, k=6):
    small = cv2.resize(frame, (100, 100), interpolation=cv2.INTER_AREA)
    pixels = small.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_

    counts = np.bincount(labels)
    percentages = counts / counts.sum()

    idx = np.argsort(percentages)[::-1]
    return colors[idx], percentages[idx]


def rgb_to_hex(c):
    return '#%02x%02x%02x' % tuple(c)


def analyze_color_tone(colors, percentages):
    r = g = b = 0.0

    for (cr, cg, cb), p in zip(colors, percentages):
        r += cr * p
        g += cg * p
        b += cb * p

    if r > b + 20:
        return "warm"
    elif b > r + 20:
        return "cool"
    else:
        return "neutral"


def show_palette(colors, percentages):
    html = """
    <div style="
        display:flex;
        width:100%;
        height:70px;
        border:3px solid white;
        margin:10px 0 10px 0;
        box-sizing:border-box;
        overflow:hidden;
    ">
    """

    for i, (color, p) in enumerate(zip(colors, percentages)):
        hex_color = rgb_to_hex(color)
        border = "border-right:3px solid white;" if i < len(colors) - 1 else ""
        width_percent = max(p * 100, 8)

        html += f"""
        <div style="
            background:{hex_color};
            width:{width_percent}%;
            height:70px;
            {border}
            box-sizing:border-box;
        "></div>
        """

    html += "</div>"
    components.html(html, height=85)


uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    classifier = load_classifier()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    st.video(uploaded_file)
    st.success("Video uploaded successfully!")

    frames = extract_frames(path, num_frames=5)

    if frames:
        st.subheader("Frames + Cinematic Predictions")

        for i, frame in enumerate(frames):
            st.image(frame, caption=f"Frame {i+1}", use_container_width=True)

            image = Image.fromarray(frame)

            shot_results = classify_shot(image, classifier)
            shot = shot_results[0]

            lighting, mean, contrast, dark = analyze_lighting(frame)
            colors, perc = extract_colors(frame, k=6)
            tone = analyze_color_tone(colors, perc)

            st.write(f"**Shot:** {shot['label']}")
            st.write(f"**Lighting:** {lighting}")
            st.write(f"**Color tone:** {tone}")
            st.write("**Dominant color palette:**")
            show_palette(colors, perc)

            with st.expander("See technical details"):
                st.write(f"**Shot confidence:** {shot['score']:.2f}")
                st.write(f"**Mean brightness:** {mean:.1f}")
                st.write(f"**Contrast:** {contrast:.1f}")
                st.write(f"**Dark pixel ratio:** {dark:.2f}")

                hex_codes = [rgb_to_hex(c) for c in colors]
                st.write("**Palette HEX:**", " | ".join(hex_codes))

                st.write("**All shot scores:**")
                for result in shot_results:
                    st.write(f"- {result['label']}: {result['score']:.3f}")

            st.markdown("---")
    else:
        st.error("Could not extract frames from this video.")

    os.remove(path)