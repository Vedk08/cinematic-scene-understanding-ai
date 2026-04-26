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
from ultralytics import YOLO


st.title("🎬 Cinematic Scene Understanding AI - Phase 6.3")
st.write(
    "Analyze video clips or stills for shot type, lighting, color, subject placement, objects, and composition using YOLO."
)


@st.cache_resource
def load_classifier():
    return pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32"
    )


@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")


def extract_frames(video_path, num_frames=5):
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return frames

    frame_indices = [
        int(i * (total_frames - 1) / (num_frames - 1))
        for i in range(num_frames)
    ]

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()

        if success:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def classify_shot(image, classifier):
    labels = ["close-up shot", "medium shot", "wide shot"]

    return classifier(
        image,
        candidate_labels=labels,
        hypothesis_template="This image shows a {}"
    )


def get_frame_quality(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    mean = float(np.mean(gray))
    contrast = float(np.std(gray))

    if mean < 10 and contrast < 5:
        return "unusable_black_frame", mean, contrast

    if mean < 25 and contrast < 10:
        return "too_dark_to_analyze_reliably", mean, contrast

    return "usable", mean, contrast


def analyze_lighting(frame):
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
    return "#%02x%02x%02x" % tuple(c)


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


def show_palette(colors, percentages, height=65):
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
    names = []

    for r, g, b in colors:
        if r < 35 and g < 35 and b < 35:
            names.append("deep black")
        elif b > r + 30 and b > g:
            names.append("blue" if g < 100 else "teal-blue")
        elif r > b + 30 and r > g:
            names.append("amber" if g > 120 else "red-orange")
        elif g > r and g > b:
            names.append("green")
        elif abs(r - b) < 20 and r > 100 and b > 100:
            names.append("magenta-violet")
        elif r > 140 and g > 140 and b > 140:
            names.append("light gray")
        else:
            names.append("muted neutral")

    return names


def detect_objects_yolo(frame, yolo_model, confidence_threshold=0.25):
    """
    Detect objects using YOLO.
    Returns detections as dictionaries.
    """
    results = yolo_model.predict(frame, conf=confidence_threshold, verbose=False)
    detections = []

    if len(results) == 0:
        return detections

    result = results[0]
    names = result.names

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "label": names[class_id],
            "confidence": confidence,
            "box": (
                int(x1),
                int(y1),
                int(x2 - x1),
                int(y2 - y1)
            )
        })

    return detections


def get_person_detections(detections):
    return [
        detection for detection in detections
        if detection["label"] == "person"
    ]


def get_primary_subject_box(person_detections):
    if not person_detections:
        return None

    return max(
        person_detections,
        key=lambda detection: detection["box"][2] * detection["box"][3]
    )


def analyze_symmetry(frame, quality_status):
    if quality_status != "usable":
        return "too dark / invalid for symmetry analysis", 0.0

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    left = gray[:, :w // 2]
    right = gray[:, w - w // 2:]
    right_flipped = cv2.flip(right, 1)

    min_width = min(left.shape[1], right_flipped.shape[1])
    left = left[:, :min_width]
    right_flipped = right_flipped[:, :min_width]

    difference = np.mean(np.abs(left.astype("float") - right_flipped.astype("float")))
    symmetry_score = max(0, 100 - difference)

    if symmetry_score > 75:
        label = "strong symmetrical composition"
    elif symmetry_score > 55:
        label = "moderately balanced composition"
    else:
        label = "asymmetrical composition"

    return label, float(symmetry_score)


def analyze_composition(frame, detections, quality_status):
    h, w, _ = frame.shape
    frame_area = w * h

    if quality_status != "usable":
        return {
            "person_count": 0,
            "subject_position": "unavailable",
            "composition_type": "composition unavailable due to poor frame quality",
            "framing_note": "unavailable",
            "subject_area_ratio": 0.0,
            "primary_box": None,
            "detections": [],
            "object_labels": []
        }

    person_detections = get_person_detections(detections)
    primary_subject = get_primary_subject_box(person_detections)

    object_labels = [
        detection["label"] for detection in detections
        if detection["label"] != "person"
    ]

    if primary_subject is None:
        return {
            "person_count": 0,
            "subject_position": "no reliable person detected",
            "composition_type": "environment / object-focused composition",
            "framing_note": "subject placement unavailable",
            "subject_area_ratio": 0.0,
            "primary_box": None,
            "detections": detections,
            "object_labels": object_labels
        }

    x, y, bw, bh = primary_subject["box"]
    subject_center_x = x + bw / 2
    subject_area_ratio = (bw * bh) / frame_area

    if subject_center_x < w / 3:
        subject_position = "left third"
        composition_type = "rule-of-thirds composition"
    elif subject_center_x > 2 * w / 3:
        subject_position = "right third"
        composition_type = "rule-of-thirds composition"
    else:
        subject_position = "center"
        composition_type = "centered composition"

    if subject_area_ratio < 0.12:
        framing_note = "heavy negative space"
    elif subject_area_ratio < 0.28:
        framing_note = "moderate negative space"
    else:
        framing_note = "tight subject framing"

    return {
        "person_count": len(person_detections),
        "subject_position": subject_position,
        "composition_type": composition_type,
        "framing_note": framing_note,
        "subject_area_ratio": float(subject_area_ratio),
        "primary_box": primary_subject["box"],
        "detections": detections,
        "object_labels": object_labels
    }


def draw_yolo_boxes(frame, detections):
    annotated = frame.copy()

    for detection in detections:
        x, y, w, h = detection["box"]
        label = detection["label"]
        confidence = detection["confidence"]

        color = (0, 255, 0) if label == "person" else (255, 0, 0)

        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            color,
            3
        )

        cv2.putText(
            annotated,
            f"{label} {confidence:.2f}",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    return annotated


def analyze_single_frame(frame, classifier, yolo_model):
    quality_status, _, _ = get_frame_quality(frame)

    image = Image.fromarray(frame)

    if quality_status == "unusable_black_frame":
        shot_results = [{"label": "unavailable", "score": 0.0}]
        shot = "unavailable"
        shot_score = 0.0
    else:
        shot_results = classify_shot(image, classifier)
        shot = shot_results[0]["label"]
        shot_score = shot_results[0]["score"]

    lighting, mean, contrast, dark = analyze_lighting(frame)
    colors, percentages = extract_colors(frame, k=6)
    tone = analyze_color_tone(colors, percentages)

    if quality_status == "usable":
        detections = detect_objects_yolo(frame, yolo_model)
    else:
        detections = []

    composition = analyze_composition(
        frame,
        detections,
        quality_status
    )

    symmetry_label, symmetry_score = analyze_symmetry(frame, quality_status)

    return {
        "frame": frame,
        "quality_status": quality_status,
        "shot_results": shot_results,
        "shot": shot,
        "shot_score": shot_score,
        "lighting": lighting,
        "mean_brightness": mean,
        "contrast": contrast,
        "dark_ratio": dark,
        "colors": [tuple(map(int, c)) for c in colors],
        "proportions": [float(p) for p in percentages],
        "tone": tone,
        "composition": composition,
        "symmetry_label": symmetry_label,
        "symmetry_score": symmetry_score,
    }


def aggregate_clip_palette(frame_results, num_colors=6):
    all_colors = []

    usable_results = [
        result for result in frame_results
        if result["quality_status"] == "usable"
    ]

    if not usable_results:
        return [(0, 0, 0)] * num_colors, [1 / num_colors] * num_colors

    for result in usable_results:
        for color, proportion in zip(result["colors"], result["proportions"]):
            repeat_count = max(1, int(proportion * 100))
            all_colors.extend([color] * repeat_count)

    all_colors = np.array(all_colors, dtype=np.uint8)

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


def generate_summary(
    dominant_shot,
    dominant_lighting,
    dominant_tone,
    palette_names,
    mood,
    dominant_composition=None
):
    palette_text = ", ".join(palette_names[:4])

    composition_text = ""
    if dominant_composition:
        composition_text = f" The framing often uses {dominant_composition}."

    return (
        f"This visual predominantly uses {dominant_shot}, {dominant_lighting}, "
        f"and a {dominant_tone}-toned palette built around {palette_text}. "
        f"Overall, it feels {mood}.{composition_text}"
    )


def display_frame_analysis(result):
    st.image(result["frame"], caption="Analyzed frame", use_container_width=True)

    if result["quality_status"] == "unusable_black_frame":
        st.warning("This frame appears to be almost completely black, so cinematic composition analysis is unavailable.")
    elif result["quality_status"] == "too_dark_to_analyze_reliably":
        st.warning("This frame is very dark, so composition and subject detection may be unreliable.")

    composition = result["composition"]

    st.write(f"**Shot:** {result['shot']}")
    st.write(f"**Lighting:** {result['lighting']}")
    st.write(f"**Color tone:** {result['tone']}")
    st.write(f"**Composition:** {composition['composition_type']}")
    st.write(f"**Subject placement:** {composition['subject_position']}")
    st.write(f"**People detected:** {composition['person_count']}")
    st.write(f"**Framing note:** {composition['framing_note']}")
    st.write(f"**Balance:** {result['symmetry_label']}")

    if composition["object_labels"]:
        st.write("**Other detected objects:**", ", ".join(sorted(set(composition["object_labels"]))))

    st.write("**Dominant color palette:**")
    show_palette(result["colors"], result["proportions"])

    if composition["detections"]:
        with st.expander("Show YOLO detections"):
            annotated = draw_yolo_boxes(result["frame"], composition["detections"])
            st.image(
                annotated,
                caption="YOLO detections: green = person, blue = objects",
                use_container_width=True
            )

    with st.expander("See technical details"):
        st.write(f"**Frame quality:** {result['quality_status']}")
        st.write(f"**Shot confidence:** {result['shot_score']:.2f}")
        st.write(f"**Mean brightness:** {result['mean_brightness']:.1f}")
        st.write(f"**Contrast:** {result['contrast']:.1f}")
        st.write(f"**Dark pixel ratio:** {result['dark_ratio']:.2f}")
        st.write(f"**Subject area ratio:** {composition['subject_area_ratio']:.3f}")
        st.write(f"**Symmetry score:** {result['symmetry_score']:.1f}")

        hex_codes = [rgb_to_hex(c) for c in result["colors"]]
        st.write("**Palette HEX:**", " | ".join(hex_codes))

        st.write("**All shot scores:**")
        for shot_result in result["shot_results"]:
            st.write(f"- {shot_result['label']}: {shot_result['score']:.3f}")


classifier = load_classifier()
yolo_model = load_yolo_model()

mode = st.radio(
    "Choose analysis mode:",
    ["Analyze Video Clip", "Analyze Single Still / Photo"]
)


if mode == "Analyze Video Clip":
    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            path = tmp.name

        st.video(uploaded_video)
        st.success("Video uploaded successfully!")

        frames = extract_frames(path, num_frames=5)

        if frames:
            frame_results = [
                analyze_single_frame(frame, classifier, yolo_model)
                for frame in frames
            ]

            usable_results = [
                r for r in frame_results
                if r["quality_status"] == "usable"
            ]

            st.subheader("Clip-Level Summary")

            summary_source = usable_results if usable_results else frame_results

            dominant_shot = Counter(
                [r["shot"] for r in summary_source]
            ).most_common(1)[0][0]

            dominant_lighting = Counter(
                [r["lighting"] for r in summary_source]
            ).most_common(1)[0][0]

            dominant_tone = Counter(
                [r["tone"] for r in summary_source]
            ).most_common(1)[0][0]

            dominant_composition = Counter(
                [r["composition"]["composition_type"] for r in summary_source]
            ).most_common(1)[0][0]

            clip_colors, clip_proportions = aggregate_clip_palette(frame_results)
            palette_names = simplify_hex_names(clip_colors)
            mood = infer_mood(dominant_shot, dominant_lighting, dominant_tone)

            summary = generate_summary(
                dominant_shot,
                dominant_lighting,
                dominant_tone,
                palette_names,
                mood,
                dominant_composition
            )

            col1, col2 = st.columns([1.2, 1])

            with col1:
                st.write(f"**Dominant shot type:** {dominant_shot}")
                st.write(f"**Dominant lighting:** {dominant_lighting}")
                st.write(f"**Dominant color tone:** {dominant_tone}")
                st.write(f"**Dominant composition:** {dominant_composition}")
                st.write(f"**Overall mood:** {mood}")

            with col2:
                st.write("**Clip palette:**")
                show_palette(clip_colors, clip_proportions, height=55)

            st.write("**Scene summary:**")
            st.info(summary)

            st.markdown("---")
            st.subheader("Frame-by-Frame Analysis")

            for i, result in enumerate(frame_results):
                st.write(f"### Frame {i + 1}")
                display_frame_analysis(result)
                st.markdown("---")

        else:
            st.error("Could not extract frames from this video.")

        os.remove(path)


if mode == "Analyze Single Still / Photo":
    uploaded_image = st.file_uploader(
        "Upload still / photo",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        frame = np.array(image)

        result = analyze_single_frame(frame, classifier, yolo_model)

        palette_names = simplify_hex_names(result["colors"])
        mood = infer_mood(result["shot"], result["lighting"], result["tone"])

        summary = generate_summary(
            result["shot"],
            result["lighting"],
            result["tone"],
            palette_names,
            mood,
            result["composition"]["composition_type"]
        )

        st.subheader("Still / Photo Analysis")
        display_frame_analysis(result)

        st.write("**Visual summary:**")
        st.info(summary)