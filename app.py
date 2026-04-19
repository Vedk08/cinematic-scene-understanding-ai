import os
import tempfile

import cv2
import streamlit as st
from PIL import Image
from transformers import pipeline


st.title("🎬 Cinematic AI - Phase 2")
st.write("Upload a video, extract 5 frames, and classify each frame as close-up, medium, or wide.")


@st.cache_resource
def load_classifier():
    return pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32"
    )


def extract_frames(video_path: str, num_frames: int = 5):
    """
    Extract evenly spaced frames from the video.
    """
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

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()

        if success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()
    return frames


uploaded_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "mov", "avi", "mkv"]
)

if uploaded_file is not None:
    classifier = load_classifier()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_video_path = temp_file.name

    st.video(uploaded_file)
    st.success("Video uploaded successfully!")

    frames = extract_frames(temp_video_path, num_frames=5)

    if frames:
        st.subheader("Frames + Shot Type Predictions")

        candidate_labels = ["close-up shot", "medium shot", "wide shot"]

        for i, frame in enumerate(frames):
            st.image(frame, caption=f"Frame {i + 1}", width="stretch")

            image = Image.fromarray(frame)
            results = classifier(image, candidate_labels=candidate_labels)

            for result in results:
                st.write(f"**{result['label']}**: {result['score']:.3f}")

            st.markdown("---")
    else:
        st.error("Could not extract frames from this video.")

    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)