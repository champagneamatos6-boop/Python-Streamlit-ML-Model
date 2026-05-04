import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
from collections import Counter
from pathlib import Path
import time

@st.cache_resource    
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

target_object = st.text_input("Alert object (e.g., person, bottle):", "person")


SAVE_DIR = Path("saved_frames")
SAVE_DIR.mkdir(exist_ok=True)


last_saved_time = {"time": time.time()}


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    
    results = model.track(
        img,
        persist=True,
        conf=0.5,
        verbose=False
    )

   
    annotated_frame = results[0].plot()

   
    names = model.names
    boxes = results[0].boxes

    detected_objects = []

    if boxes is not None:
        for cls in boxes.cls:
            detected_objects.append(names[int(cls)])


    counts = Counter(detected_objects)

   
    y = 30
    for obj, cnt in counts.items():
        text = f"{obj}: {cnt}"
        cv2.putText(annotated_frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 30

   
    if target_object in counts:
        cv2.putText(annotated_frame,
                    f"ALERT: {target_object} detected!",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2)

    if time.time() - last_saved_time["time"] > 5:
        filename = SAVE_DIR / f"frame_{int(time.time())}.jpg"
        cv2.imwrite(str(filename), annotated_frame)
        last_saved_time["time"] = time.time()

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)
