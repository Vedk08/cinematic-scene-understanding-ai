import os
import tempfile
from collections import Counter

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from sklearn.cluster import KMeans
from transformers import pipeline


st.title("🎬 Cinematic AI - Phase 4")
st.write(
    "Upload a video, extract 5 frames, analyze shot type, lighting, color palette, and generate a clip-level cinematic summary."
)


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


def show_palette(colors, percentages, height=70):
    html = f"""
    <div style="
        display:flex;
        width:100%;
        height:{height}px;
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
            height:{height}px;
            {border}
            box-sizing:border-box;
        "></div>
        """

    html += "</div>"
    components.html(html, height=height + 20)


def simplify_hex_names(colors):
    """
    Very lightweight descriptive naming based on dominant RGB channel.
    """
    names = []
    for r, g, b in colors:
        if r < 35 and g < 35 and b < 35:
            names.append("deep black")
        elif b > r + 30 and b > g:
            if g > 100:
                names.append("teal-blue")
            else:
                names.append("blue")
        elif r > b + 30 and r > g:
            if g > 120:
                names.append("amber")
            else:
                names.append("red-orange")
        elif g > r and g > b:
            names.append("green")
        elif abs(r - b) < 20 and r > 100 and b > 100:
            names.append("magenta-violet")
        elif r > 140 and g > 140 and b > 140:
            names.append("light gray")
        else:
            names.append("muted neutral")
    return names


def aggregate_clip_palette(frame_results, num_colors=6):
    """
    Combine all frame palettes into one clip-level palette.
    """
    all_colors = []
    for result in frame_results:
        for color, proportion in zip(result["colors"], result["proportions"]):
            repeat_count = max(1, int(proportion * 100))
            all_colors.extend([color] * repeat_count)

    all_colors = np.array(all_colors, dtype=np.uint8)

    if len(all_colors) < num_colors:
        unique_colors = all_colors.tolist()
        while len(unique_colors) < num_colors:
            unique_colors.append((0, 0, 0))
        proportions = [1 / num_colors] * num_colors
        return unique_colors[:num_colors], proportions

    kmeans = KMeans(n_clusters=num_colors, n_init=10, random_state=42)
    labels = kmeans.fit_predict(all_colors)
    centers = kmeans.cluster_centers_.astype(int)

    counts = np.bincount(labels)
    proportions = counts / counts.sum()

    idx = np.argsort(proportions)[::-1]
    colors = [tuple(centers[i]) for i in idx]
    proportions = [float(proportions[i]) for i in idx]

    return colors, proportions


def infer_mood(dominant_shot, dominant_lighting, dominant_tone):
    """
    Lightweight mood inference from dominant attributes.
    """
    if dominant_lighting == "low-key dramatic lighting" and dominant_tone == "cool":
        return "moody, tense, and nocturnal"
    if dominant_lighting == "high-key lighting" and dominant_tone == "warm":
        return "bright, inviting, and energetic"
    if dominant_lighting == "soft lighting" and dominant_tone == "warm":
        return "gentle, intimate, and calm"
    if dominant_shot == "close-up shot" and dominant_lighting == "low-key dramatic lighting":
        return "intense and emotionally focused"
    if dominant_shot == "wide shot" and dominant_tone == "cool":
        return "atmospheric and spatially distant"
    return "cinematic and visually controlled"


def generate_clip_summary(dominant_shot, dominant_lighting, dominant_tone, palette_names, mood):
    palette_text = ", ".join(palette_names[:4])
    return (
        f"This clip predominantly uses {dominant_shot}s, {dominant_lighting}, "
        f"and a {dominant_tone}-toned palette built around {palette_text}. "
        f"Overall, the scene feels {mood}."
    )


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
        st.subheader("Clip-Level Summary")

        frame_results = []

        for frame in frames:
            image = Image.fromarray(frame)

            shot_results = classify_shot(image, classifier)
            shot = shot_results[0]["label"]

            lighting, mean, contrast, dark = analyze_lighting(frame)
            colors, perc = extract_colors(frame, k=6)
            tone = analyze_color_tone(colors, perc)

            frame_results.append({
                "frame": frame,
                "shot_results": shot_results,
                "shot": shot,
                "lighting": lighting,
                "mean_brightness": mean,
                "contrast": contrast,
                "dark_ratio": dark,
                "colors": [tuple(map(int, c)) for c in colors],
                "proportions": [float(p) for p in perc],
                "tone": tone,
            })

        dominant_shot = Counter([r["shot"] for r in frame_results]).most_common(1)[0][0]
        dominant_lighting = Counter([r["lighting"] for r in frame_results]).most_common(1)[0][0]
        dominant_tone = Counter([r["tone"] for r in frame_results]).most_common(1)[0][0]

        clip_colors, clip_proportions = aggregate_clip_palette(frame_results, num_colors=6)
        palette_names = simplify_hex_names(clip_colors)
        mood = infer_mood(dominant_shot, dominant_lighting, dominant_tone)
        clip_summary = generate_clip_summary(
            dominant_shot, dominant_lighting, dominant_tone, palette_names, mood
        )

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.write(f"**Dominant shot type:** {dominant_shot}")
            st.write(f"**Dominant lighting:** {dominant_lighting}")
            st.write(f"**Dominant color tone:** {dominant_tone}")
            st.write(f"**Overall mood:** {mood}")

        with col2:
            st.write("**Clip palette:**")
            show_palette(clip_colors, clip_proportions, height=50)
            st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

            st.write("**Palette colors:**")
            st.code(" | ".join([rgb_to_hex(c) for c in clip_colors]))

        st.write("**Scene summary:**")
        st.info(clip_summary)

        st.markdown("---")
        st.subheader("Frame-by-Frame Analysis")

        for i, result in enumerate(frame_results):
            frame = result["frame"]
            st.image(frame, caption=f"Frame {i+1}", use_container_width=True)

            st.write(f"**Shot:** {result['shot']}")
            st.write(f"**Lighting:** {result['lighting']}")
            st.write(f"**Color tone:** {result['tone']}")
            st.write("**Dominant color palette:**")
            show_palette(result["colors"], result["proportions"])

            with st.expander("See technical details"):
                top_shot_score = result["shot_results"][0]["score"]
                st.write(f"**Shot confidence:** {top_shot_score:.2f}")
                st.write(f"**Mean brightness:** {result['mean_brightness']:.1f}")
                st.write(f"**Contrast:** {result['contrast']:.1f}")
                st.write(f"**Dark pixel ratio:** {result['dark_ratio']:.2f}")

                hex_codes = [rgb_to_hex(c) for c in result["colors"]]
                st.write("**Palette HEX:**", " | ".join(hex_codes))

                st.write("**All shot scores:**")
                for shot_result in result["shot_results"]:
                    st.write(f"- {shot_result['label']}: {shot_result['score']:.3f}")

            st.markdown("---")

    else:
        st.error("Could not extract frames from this video.")

    os.remove(path)