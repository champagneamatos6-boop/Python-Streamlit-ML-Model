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

SAVE_DIR = Path("saved_frames")
SAVE_DIR.mkdir(exist_ok=True)

if "last_saved_at" not in st.session_state:
    st.session_state.last_saved_at = 0.0

st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

with st.sidebar:
    st.header("Detection Options")
    confidence = st.slider("Confidence threshold", 0.1, 0.9, 0.5, 0.05)
    alert_targets_raw = st.text_input(
        "Alert objects",
        value="person",
        help="Comma-separated objects that should trigger an alert, e.g. person, car, dog",
    )
    save_frames = st.checkbox("Save detected frames as images", value=False)
    save_alert_only = st.checkbox("Save only when alert objects appear", value=True)
    save_cooldown = st.slider(
        "Save cooldown (seconds)",
        min_value=1,
        max_value=30,
        value=5,
        help="Minimum time between saved snapshots.",
    )

alert_targets = {
    item.strip().lower()
    for item in alert_targets_raw.split(",")
    if item.strip()
}


def process_frame(img):
    """Run YOLOv8 tracking, count objects, raise alerts, and optionally save frames."""
    results = model.track(
        img,
        persist=True,
        conf=confidence,
        verbose=False,
        imgsz=320, 
    )
    result = results[0]
    annotated = result.plot()

    names = result.names
    detected_names = []
    if result.boxes is not None and result.boxes.cls is not None:
        for cls_id in result.boxes.cls.tolist():
            detected_names.append(names[int(cls_id)])

    counts = Counter(detected_names)
    matching_alerts = sorted(
        name for name in counts.keys() if name.lower() in alert_targets
    )

    y = 30
    if counts:
        counts_text = ", ".join(
            f"{name}: {count}" for name, count in sorted(counts.items())
        )
        cv2.putText(
            annotated,
            f"Counts: {counts_text}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (50, 220, 50),
            2,
            cv2.LINE_AA,
        )
        y += 30

    if matching_alerts:
        alert_text = ", ".join(f"{name} ({counts[name]})" for name in matching_alerts)
        cv2.putText(
            annotated,
            f"ALERT: {alert_text}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    saved_path = None
    should_save = save_frames and bool(counts)
    if save_alert_only:
        should_save = should_save and bool(matching_alerts)

    now = time.time()
    if should_save and now - st.session_state.last_saved_at >= save_cooldown:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        saved_path = SAVE_DIR / f"detection_{timestamp}.jpg"
        cv2.imwrite(str(saved_path), annotated)
        st.session_state.last_saved_at = now

    return annotated, counts, matching_alerts, saved_path


mode = st.radio(
    "Select Camera Mode:",
    ["OpenCV (Reliable)"],
    index=0,
    help="OpenCV works almost everywhere.",
)


if mode == "OpenCV (Reliable)":
    st.write("📷 Using OpenCV fallback.")
    run = st.checkbox("▶️ Start Camera", value=False)

    if run:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            st.error("❌ Cannot open webcam. Make sure it's connected and not in use by another app.")
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            st.success("✅ Camera started! Uncheck the box above to stop.")
            frame_placeholder = st.empty()
            counts_placeholder = st.empty()
            alert_placeholder = st.empty()
            save_placeholder = st.empty()

            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to grab frame.")
                    break

               
                annotated, counts, matching_alerts, saved_path = process_frame(frame)

                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

              
                frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

                if counts:
                    counts_placeholder.info(
                        "Object counts: "
                        + ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))
                    )
                else:
                    counts_placeholder.info("Object counts: no objects detected.")

                if matching_alerts:
                    alert_placeholder.error(
                        "Alert triggered for: "
                        + ", ".join(f"{name} ({counts[name]})" for name in matching_alerts)
                    )
                else:
                    alert_placeholder.success("No alert objects detected.")

                if saved_path is not None:
                    save_placeholder.success(f"Saved snapshot: {saved_path}")

            cap.release()
            st.info("🛑 Camera stopped.")
    else:
        st.info("Check the '▶️ Start Camera' box above to begin.")

st.divider()
st.caption("Run with: `streamlit run main.py`")

