# ============================================================
# modules/phone_detection.py
# ============================================================
#
# PURPOSE:
#   This module detects mobile phones in webcam frames using
#   YOLOv8n (YOLO version 8, nano variant) — a small, fast
#   pretrained object detection model.
#
# WHAT IS YOLO?
#   YOLO stands for "You Only Look Once". It's a family of
#   deep learning models that can detect and locate many types
#   of objects in a single forward pass through the network.
#   Unlike older methods that scanned regions one by one,
#   YOLO processes the ENTIRE image at once — which is why it's
#   fast enough for real-time video.
#
# WHY YOLOv8 NANO (yolov8n)?
#   The "nano" variant is the smallest and fastest member of
#   the YOLOv8 family. It trades some accuracy for speed,
#   which is exactly what we need for a CPU-only application.
#   Model size: ~6.3 MB.
#   Typical speed on a modern CPU: 15–30 ms per frame (~30–60 FPS).
#
# WHAT IS COCO?
#   The model was trained on the COCO dataset (Common Objects in
#   Context) — 80 categories of everyday objects. "Cell phone"
#   is one of those 80 categories, with class index 67.
#   We only care about class 67; all other detections are ignored.
#
# IMPORTANT DISCLAIMER:
#   Detecting a phone in the camera frame does NOT prove the
#   student is actively using or holding it. The phone might be
#   on the desk, in the background, or briefly visible. We label
#   this signal "potential phone distraction" — not confirmed use.
#
# MODEL DOWNLOAD:
#   YOLOv8n downloads automatically from GitHub the FIRST time
#   the model is loaded (requires internet). The ~6.3MB file is
#   cached locally (usually in ~/.config/Ultralytics/ or the
#   current directory). Subsequent runs use the cached file.
#
# PUBLIC FUNCTIONS:
#   load_phone_model(model_path)                  → model or None
#   detect_phone(bgr_frame, model, threshold)      → dict
#   draw_phone_boxes(frame, boxes)                 → annotated frame
# ============================================================

import cv2
import numpy as np
import os

# ---- Ultralytics import with graceful fallback ----
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


# ============================================================
# CONFIGURATION CONSTANTS
# ============================================================

# The COCO dataset class index for "cell phone".
# YOLO was trained on 80 classes (0–79). Class 67 = cell phone.
# Full class list: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
PHONE_CLASS_ID = 67
PHONE_CLASS_NAME = "cell phone"

# Default confidence threshold for phone detection.
# A detection is only reported if the model is at least this
# confident that the object is a phone.
# 0.4 = 40% minimum confidence. Lower = more detections but
# more false positives. Higher = fewer but more reliable detections.
DEFAULT_CONFIDENCE_THRESHOLD = 0.40

# Path to the YOLOv8n model weights file.
# If this file doesn't exist locally, Ultralytics will download it.
DEFAULT_MODEL_PATH = "yolov8n.pt"

# Color for phone detection bounding boxes and text (BGR format).
# We use a coral/red color to make phone detections visually distinct
# from the green face detection overlays.
PHONE_BOX_COLOR  = (68, 100, 215)   # Coral-red in BGR
PHONE_TEXT_COLOR = (255, 255, 255)  # White text

# Module-level cached model — loaded once, reused for every frame.
_model = None


# ============================================================
# PUBLIC FUNCTION: load_phone_model
# ============================================================
def load_phone_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Load the YOLOv8n object detection model.

    If the model file (.pt file) is not found locally, Ultralytics
    will automatically download it from GitHub on the first call.
    This requires an internet connection on the first run only.

    HOW YOLO MODELS WORK:
        A .pt file is a PyTorch model checkpoint — it contains the
        neural network architecture AND the trained weights (parameters)
        baked into one file. The YOLO class from Ultralytics knows
        how to load it, run inference, and interpret the outputs.

    WHY MODULE-LEVEL CACHING (_model)?
        Loading and parsing a 6.3MB PyTorch model takes ~0.5 seconds.
        We store the loaded model in _model (module-level variable)
        so it's loaded once and reused on every frame. The camera
        loop calls this function at startup and stores the result.

    Args:
        model_path: Path to the .pt model weights file.
                    Default: "yolov8n.pt" (auto-downloaded if missing).

    Returns:
        A loaded YOLO model object, or None if loading failed.
    """
    global _model

    # Return cached model immediately if already loaded
    if _model is not None:
        return _model

    # Check if Ultralytics was imported successfully
    if not ULTRALYTICS_AVAILABLE:
        print("[phone_detection] ERROR: ultralytics is not installed.")
        print("  Run: pip install ultralytics>=8.2.0")
        return None

    try:
        print(f"[phone_detection] Loading YOLOv8n model from '{model_path}'...")
        print("[phone_detection] If the file is missing, it will be downloaded (~6.3 MB).")

        # YOLO("yolov8n.pt") either:
        #   - Loads from disk if the file exists locally
        #   - Downloads from GitHub if it doesn't exist
        # The 'verbose=False' suppresses Ultralytics' startup banner.
        _model = YOLO(model_path)

        print("[phone_detection] YOLOv8n model loaded successfully.")
        print(f"[phone_detection] Watching for COCO class {PHONE_CLASS_ID}: '{PHONE_CLASS_NAME}'")
        return _model

    except Exception as e:
        print(f"[phone_detection] ERROR loading model: {e}")
        return None


# ============================================================
# PUBLIC FUNCTION: detect_phone
# ============================================================
def detect_phone(
    bgr_frame: np.ndarray,
    model,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> dict:
    """
    Run YOLOv8n inference on one frame and find any cell phones.

    HOW YOLO INFERENCE WORKS (simplified):
        1. The input image is resized to the model's expected size (640x640)
        2. The image is passed through ~100 convolutional layers
        3. The output is a grid of predictions — each grid cell predicts
           whether an object is present there, what class it is, and
           the bounding box coordinates
        4. Predictions below the confidence threshold are discarded
        5. Non-Maximum Suppression (NMS) removes overlapping boxes for
           the same object, keeping only the most confident one

    We then filter the final predictions to keep only class 67 (cell phone).

    WHY NOT RUN ON EVERY FRAME?
        YOLO is accurate but heavier than MediaPipe. On a mid-range CPU
        it takes 30–80ms per frame. Rather than running it on every frame,
        camera.py calls this function every 2nd frame and caches the result.

    Args:
        bgr_frame:            OpenCV BGR numpy array (the webcam frame).
        model:                Loaded YOLO model from load_phone_model().
        confidence_threshold: Minimum confidence to report a detection.

    Returns:
        dict with keys:
            "phone_detected" (bool)   — Was a phone found with enough confidence?
            "confidence"     (float)  — Highest confidence among all phone detections
            "boxes"          (list)   — List of dicts, each with:
                                          x1, y1, x2, y2 (pixel coords of box corners)
                                          confidence (float, 0.0–1.0)
            "error"          (str|None) — Error message, or None if successful
    """
    # Default result for when detection fails or model is unavailable
    default_result = {
        "phone_detected": False,
        "confidence":     0.0,
        "boxes":          [],
        "error":          None,
    }

    # Guard: if model failed to load, return default with error
    if model is None:
        default_result["error"] = "Phone model not loaded. Install ultralytics."
        return default_result

    try:
        # ── Run YOLO inference ─────────────────────────────────────
        # model(frame, ...) runs the full detection pipeline.
        # verbose=False suppresses per-frame console logging from Ultralytics.
        # conf sets the minimum confidence threshold.
        # classes=[PHONE_CLASS_ID] tells YOLO to only look for class 67.
        #   This is optional but speeds up post-processing slightly.
        results = model(
            bgr_frame,
            conf=confidence_threshold,
            classes=[PHONE_CLASS_ID],   # Only detect cell phones
            verbose=False,              # Suppress per-frame console output
        )

        # results is a list with one element per input image.
        # Since we process one frame at a time, results[0] has our data.
        result = results[0]

        # ── Extract phone bounding boxes ───────────────────────────
        # result.boxes contains all detected objects above the threshold.
        # Each box has:
        #   .cls  — class index (67 for cell phone)
        #   .conf — confidence score (0.0–1.0)
        #   .xyxy — bounding box in [x1, y1, x2, y2] pixel coordinates
        #           where (x1, y1) is top-left and (x2, y2) is bottom-right
        phone_boxes  = []
        max_conf     = 0.0

        for box in result.boxes:
            class_id   = int(box.cls[0])      # Class index (should be 67)
            confidence = float(box.conf[0])   # Confidence score

            # Double-check it's a phone (redundant since we set classes=[67],
            # but good defensive coding practice)
            if class_id != PHONE_CLASS_ID:
                continue

            # Extract bounding box pixel coordinates
            # box.xyxy[0] is a tensor [x1, y1, x2, y2]; we convert to ints
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            phone_boxes.append({
                "x1":         x1,
                "y1":         y1,
                "x2":         x2,
                "y2":         y2,
                "confidence": confidence,
            })

            # Track the highest confidence detection
            if confidence > max_conf:
                max_conf = confidence

        phone_detected = len(phone_boxes) > 0

        return {
            "phone_detected": phone_detected,
            "confidence":     round(max_conf, 3),
            "boxes":          phone_boxes,
            "error":          None,
        }

    except Exception as e:
        default_result["error"] = f"Phone detection error: {str(e)}"
        return default_result


# ============================================================
# PUBLIC FUNCTION: draw_phone_boxes
# ============================================================
def draw_phone_boxes(frame: np.ndarray, boxes: list) -> np.ndarray:
    """
    Draw bounding boxes around detected phones on the frame.

    This function is called by camera.py to draw phone detections
    ON TOP of the already face-annotated frame. The result is a
    single frame showing BOTH face landmarks AND phone boxes.

    WHAT WE DRAW FOR EACH PHONE:
        1. A coral-red rectangle outlining the phone in the frame
        2. A filled label background behind the text
        3. White text showing "PHONE" and the confidence percentage

    WHY DRAW HERE AND NOT IN detect_phone()?
        Keeping detection (data) and visualization (drawing) separate
        means you can:
        - Run detect_phone() without needing OpenCV drawing
        - Cache detection results and re-draw them on fresh frames
        - Test detection independently of drawing

    Args:
        frame: BGR numpy array to draw on (should be a copy).
        boxes: List of box dicts from detect_phone(), each with
               keys: x1, y1, x2, y2, confidence.

    Returns:
        The frame with bounding boxes drawn on it.
    """
    for box in boxes:
        x1   = box["x1"]
        y1   = box["y1"]
        x2   = box["x2"]
        y2   = box["y2"]
        conf = box["confidence"]

        # ── Draw bounding box rectangle ────────────────────────────
        # The rectangle outlines the phone's location.
        # Thickness=2 makes it visible but not too thick.
        cv2.rectangle(
            frame,
            (x1, y1),          # Top-left corner
            (x2, y2),          # Bottom-right corner
            PHONE_BOX_COLOR,   # Coral-red color
            thickness=2,
            lineType=cv2.LINE_AA
        )

        # ── Draw label: "PHONE 87%" ────────────────────────────────
        label_text = f"PHONE  {conf:.0%}"   # e.g. "PHONE  87%"

        # Calculate label text size to draw a matching background box
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,   # Font scale
            1       # Thickness
        )

        # Label position: just above the top-left corner of the box
        label_x = x1
        label_y = max(y1 - 10, text_h + 10)   # Ensure label stays on screen

        # Draw filled rectangle behind the text for readability
        cv2.rectangle(
            frame,
            (label_x, label_y - text_h - 4),
            (label_x + text_w + 6, label_y + 2),
            PHONE_BOX_COLOR,
            thickness=-1   # -1 = filled rectangle
        )

        # Draw white text on top of the filled label background
        cv2.putText(
            frame,
            label_text,
            (label_x + 3, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            PHONE_TEXT_COLOR,
            1,
            cv2.LINE_AA
        )

        # ── Draw corner accents ────────────────────────────────────
        # Small L-shaped lines at each corner add a "targeting" look
        # that makes the detection visually clear.
        corner_len = 10   # Length of each corner accent line
        thickness  = 2

        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), PHONE_BOX_COLOR, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), PHONE_BOX_COLOR, thickness)
        # Top-right corner
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), PHONE_BOX_COLOR, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), PHONE_BOX_COLOR, thickness)
        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), PHONE_BOX_COLOR, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), PHONE_BOX_COLOR, thickness)
        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), PHONE_BOX_COLOR, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), PHONE_BOX_COLOR, thickness)

    # ── Draw phone alert banner if any phone was detected ─────────
    if boxes:
        frame_height, frame_width = frame.shape[:2]

        # Semi-transparent alert bar at the TOP of the frame
        alert_text = f"PHONE DETECTED ({len(boxes)} object{'s' if len(boxes) > 1 else ''})"

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame_width, 32), PHONE_BOX_COLOR, -1)
        alpha = 0.75
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(
            frame,
            alert_text,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            PHONE_TEXT_COLOR,
            1,
            cv2.LINE_AA
        )

    return frame
