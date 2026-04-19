import streamlit as st
import cv2
import tempfile
import os


def extract_frames(video_path: str, num_frames: int = 5):
    """
    Extract evenly spaced frames from a video file.

    Args:
        video_path: Path to the saved video file.
        num_frames: Number of frames to extract.

    Returns:
        List of frames in RGB format.
    """
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
    ] if num_frames > 1 else [total_frames // 2]

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()

        if success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()
    return frames


st.title("Cinematic Scene Understanding AI - Phase 1")
st.write("Upload a video, extract 5 frames, and display them.")

uploaded_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "mov", "avi", "mkv"]
)

if uploaded_file is not None:
    st.video(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_video_path = temp_file.name

    st.write("Extracting 5 frames...")

    frames = extract_frames(temp_video_path, num_frames=5)

    if frames:
        st.subheader("Extracted Frames")
        for i, frame in enumerate(frames):
            st.image(frame, caption=f"Frame {i + 1}", use_container_width=True)
    else:
        st.error("Could not extract frames from this video.")

    os.remove(temp_video_path)