from ultralytics import YOLO
import cv2
import subprocess
import shutil
import os
import tempfile

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

air_model = YOLO("models/air_model.pt")
ground_model = YOLO("models/ground_model_v2.pt")


def get_model(mode):
    if mode == "Air Surveillance":
        return air_model
    else:
        return ground_model


# --------------------------------------------------
# GLOBAL ZONE
# --------------------------------------------------

ZONE = None


def set_zone(x1, y1, x2, y2):
    global ZONE
    ZONE = (x1, y1, x2, y2)


def get_zone(frame):
    global ZONE
    h, w = frame.shape[:2]
    if ZONE is None:
        return int(w * 0.3), int(h * 0.3), int(w * 0.7), int(h * 0.7)
    return ZONE


# --------------------------------------------------
# PROCESS IMAGE
# --------------------------------------------------

def process_image(frame, mode="GROUND", conf=0.25):

    model = get_model(mode)
    results = model.predict(frame, conf=conf)
    boxes = results[0].boxes

    x1, y1, x2, y2 = get_zone(frame)
    detections = []

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(frame, "RESTRICTED ZONE", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for box in boxes:
        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
        cls        = int(box.cls[0])
        conf_score = float(box.conf[0])
        label      = model.names[cls]

        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2
        inside = (x1 <= cx <= x2 and y1 <= cy <= y2)

        if inside and conf_score > 0.6:
            threat, color = "HIGH RISK", (0, 0, 255)
        elif inside:
            threat, color = "MEDIUM", (0, 165, 255)
        else:
            threat, color = "LOW", (0, 255, 0)

        detections.append({"label": label, "conf": conf_score, "threat": threat})

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 3)
        text = f"{label} {conf_score:.2f} | {threat}"
        (lw, lh), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (bx1, by1 - lh - 10), (bx1 + lw + 5, by1), (0, 0, 0), -1)
        cv2.putText(frame, text, (bx1, by1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame, detections


# --------------------------------------------------
# ENCODE WITH FFMPEG (subprocess)
# --------------------------------------------------

def _ffmpeg_encode(raw_path, final_path, fps):
    """Re-encode raw mp4v file to browser-compatible H.264."""
    ffmpeg_bin = shutil.which("ffmpeg") or shutil.which("ffmpeg3")

    # Also check common install locations on Linux (Streamlit Cloud)
    if not ffmpeg_bin:
        for p in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/opt/conda/bin/ffmpeg"]:
            if os.path.isfile(p):
                ffmpeg_bin = p
                break

    if not ffmpeg_bin:
        print("ffmpeg binary not found anywhere")
        return False

    print(f"Using ffmpeg at: {ffmpeg_bin}")
    cmd = [
        ffmpeg_bin, "-y",
        "-i", raw_path,
        "-vcodec", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        final_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("ffmpeg stdout:", result.stdout[-500:] if result.stdout else "")
    print("ffmpeg stderr:", result.stderr[-500:] if result.stderr else "")

    return (result.returncode == 0
            and os.path.exists(final_path)
            and os.path.getsize(final_path) > 0)


# --------------------------------------------------
# ENCODE WITH imageio (pure Python fallback)
# --------------------------------------------------

def _imageio_encode(raw_path, final_path, fps):
    """Pure-Python H.264 encode via imageio-ffmpeg (no system ffmpeg needed)."""
    try:
        import imageio
        import imageio.v3 as iio

        cap = cv2.VideoCapture(raw_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if not frames:
            return False

        iio.imwrite(
            final_path,
            frames,
            fps=fps,
            codec="libx264",
            output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        )
        return os.path.exists(final_path) and os.path.getsize(final_path) > 0

    except Exception as e:
        print(f"imageio encode failed: {e}")
        return False


# --------------------------------------------------
# PROCESS VIDEO
# --------------------------------------------------

def process_video(path, mode="GROUND", conf=0.25):

    cap = cv2.VideoCapture(path)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_dir    = tempfile.gettempdir()
    raw_path   = os.path.join(tmp_dir, "raw_processed.mp4")
    final_path = os.path.join(tmp_dir, "final_h264.mp4")

    # Clean up old files
    for p in [raw_path, final_path]:
        if os.path.exists(p):
            os.remove(p)

    # --- Step 1: Write annotated frames (mp4v) ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))

    frame_count   = 0
    total_objects = 0
    total_threats = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame, detections = process_image(frame, mode=mode, conf=conf)
        total_objects += len(detections)
        total_threats += sum(1 for d in detections if d["threat"] == "HIGH RISK")
        out.write(frame)

    cap.release()
    out.release()

    print(f"Frames processed = {frame_count}")
    print(f"Raw file size    = {os.path.getsize(raw_path)} bytes")

    # --- Step 2: Try ffmpeg system binary first ---
    if _ffmpeg_encode(raw_path, final_path, fps):
        print(f"ffmpeg encode success. Final size = {os.path.getsize(final_path)} bytes")
        os.remove(raw_path)
        return final_path, total_objects, total_threats

    # --- Step 3: Try imageio-ffmpeg (bundled ffmpeg) ---
    print("Trying imageio-ffmpeg fallback...")
    if _imageio_encode(raw_path, final_path, fps):
        print(f"imageio encode success. Final size = {os.path.getsize(final_path)} bytes")
        os.remove(raw_path)
        return final_path, total_objects, total_threats

    # --- Step 4: Return raw file (won't play in browser but better than nothing) ---
    print("All encode attempts failed — returning raw mp4v")
    return raw_path, total_objects, total_threats