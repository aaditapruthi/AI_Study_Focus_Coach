# ============================================================
# modules/camera.py
# ============================================================
#
# PURPOSE:
#   Manages the webcam capture pipeline in a BACKGROUND THREAD.
#   Reads frames, runs face detection + phone detection + the
#   behavioral classifier, and stores everything in a shared
#   thread-safe state dictionary.
#
# PHASE 4 CHANGES:
#   - attention_classifier is now imported and used
#   - Each frame's CV results are fed into classify_frame()
#   - _state now includes "classified_state" (the official 4-state label)
#   - The classifier uses temporal smoothing (5-frame confirmation)
#
# PUBLIC FUNCTIONS (called from app.py):
#   start_camera(camera_index) → bool
#   stop_camera()
#   is_running() → bool
#   get_camera_state() → dict
# ============================================================

import cv2
import threading
import time
import numpy as np


# ============================================================
# CONFIGURATION CONSTANTS
# ============================================================

CAPTURE_WIDTH  = 640
CAPTURE_HEIGHT = 480

# MediaPipe runs every frame (fast enough on most CPUs)
FACE_DETECTION_EVERY_N_FRAMES = 1

# YOLO runs every 2nd frame (heavier model, needs throttling)
PHONE_DETECTION_EVERY_N_FRAMES = 2


# ============================================================
# MODULE-LEVEL SHARED STATE
# ============================================================
_state = {
    "running":          False,
    "annotated_frame":  None,   # BGR numpy array with all overlays
    "raw_frame":        None,   # BGR numpy array without overlays
    "face_result": {
        "face_detected": False,
        "gaze":          "unknown",
        "gaze_ratio":    0.5,
        "error":         None,
    },
    "phone_result": {
        "phone_detected": False,
        "confidence":     0.0,
        "boxes":          [],
        "error":          None,
    },
    "classified_state": "FOCUSED",   # Official behavioral state from classifier
    "fps":        0.0,
    "frame_count": 0,
    "error":      None,
}

_lock       = threading.Lock()
_stop_event = threading.Event()
_thread     = None


# ============================================================
# PUBLIC FUNCTION: start_camera
# ============================================================
def start_camera(camera_index: int = 0) -> bool:
    """
    Start the webcam capture and analysis pipeline.
    Returns True if the camera opened successfully.
    """
    global _thread

    if _state["running"]:
        return True

    _stop_event.clear()

    with _lock:
        _state["error"] = None

    _thread = threading.Thread(
        target=_camera_loop,
        args=(camera_index,),
        daemon=True,
        name="CameraThread",
    )
    _thread.start()

    for _ in range(20):
        time.sleep(0.1)
        if _state["running"] or _state["error"]:
            break

    return _state["running"] and _state["error"] is None


# ============================================================
# PUBLIC FUNCTION: stop_camera
# ============================================================
def stop_camera():
    """Stop the webcam capture and release all resources."""
    _stop_event.set()

    if _thread is not None and _thread.is_alive():
        _thread.join(timeout=3.0)

    with _lock:
        _state["running"]          = False
        _state["annotated_frame"]  = None
        _state["raw_frame"]        = None
        _state["fps"]              = 0.0
        _state["frame_count"]      = 0
        _state["classified_state"] = "FOCUSED"
        _state["face_result"]      = {
            "face_detected": False,
            "gaze":          "unknown",
            "gaze_ratio":    0.5,
            "error":         None,
        }
        _state["phone_result"]     = {
            "phone_detected": False,
            "confidence":     0.0,
            "boxes":          [],
            "error":          None,
        }


# ============================================================
# PUBLIC FUNCTION: is_running
# ============================================================
def is_running() -> bool:
    """Check whether the camera loop is currently active."""
    with _lock:
        return _state["running"]


# ============================================================
# PUBLIC FUNCTION: get_camera_state
# ============================================================
def get_camera_state() -> dict:
    """
    Get a thread-safe snapshot of all camera and analysis state.
    Returns a copy so the caller can read without race conditions.
    """
    with _lock:
        snapshot = dict(_state)
        snapshot["face_result"]  = dict(_state["face_result"])
        snapshot["phone_result"] = dict(_state["phone_result"])
        return snapshot


# ============================================================
# PRIVATE FUNCTION: _camera_loop
# ============================================================
def _camera_loop(camera_index: int):
    """
    Main webcam capture and processing loop (background thread).

    PIPELINE PER FRAME:
      Raw frame → flip → resize
        → MediaPipe face detection (every frame)
        → YOLOv8 phone detection (every 2nd frame)
        → Behavioral classifier (every frame)
        → Compose annotated frame
        → Update shared _state under lock
    """
    # ── Deferred imports (graceful if modules missing) ─────────
    try:
        from modules import face_detection as fd
        face_detection_available = True
    except ImportError as e:
        face_detection_available = False
        with _lock:
            _state["error"] = f"face_detection import failed: {e}"

    try:
        from modules import phone_detection as pd_mod
        phone_detection_available = True
    except ImportError as e:
        phone_detection_available = False

    # ── Import the classifier ──────────────────────────────────
    try:
        from modules import attention_classifier as ac
        classifier_available = True
    except ImportError as e:
        classifier_available = False

    # ── Open the webcam ────────────────────────────────────────
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        with _lock:
            _state["error"] = (
                f"Could not open camera at index {camera_index}. "
                "On macOS: System Settings → Privacy & Security → Camera → "
                "enable access for Terminal (or your Python app)."
            )
            _state["running"] = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # ── Load detection models ──────────────────────────────────
    face_model  = None
    phone_model = None

    if face_detection_available:
        face_model = fd.load_face_model()

    if phone_detection_available:
        phone_model = pd_mod.load_phone_model()

    # ── Create the behavioral classifier ───────────────────────
    # One classifier instance per session. It tracks temporal
    # smoothing state (candidate, confirmation counter, etc.)
    classifier = None
    if classifier_available:
        classifier = ac.create_classifier()

    # Mark camera as running
    with _lock:
        _state["running"] = True
        if face_model is None and face_detection_available:
            _state["error"] = (
                "Face model not loaded — check assets/face_landmarker.task"
            )

    # ── Tracking variables ─────────────────────────────────────
    frame_count      = 0
    fps_timer_start  = time.time()
    current_fps      = 0.0

    last_face_result    = {
        "face_detected": False,
        "gaze":          "unknown",
        "gaze_ratio":    0.5,
        "error":         None,
    }
    last_face_annotated = None

    last_phone_result   = {
        "phone_detected": False,
        "confidence":     0.0,
        "boxes":          [],
        "error":          None,
    }

    # ── Main capture loop ──────────────────────────────────────
    while not _stop_event.is_set():

        ret, frame = cap.read()
        if not ret:
            with _lock:
                _state["error"] = (
                    "Camera disconnected or failed to read frame. "
                    "Please end the session and try again."
                )
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (CAPTURE_WIDTH, CAPTURE_HEIGHT))
        frame_count += 1

        # ── FPS calculation ────────────────────────────────────
        if frame_count % 30 == 0:
            elapsed     = time.time() - fps_timer_start
            current_fps = 30.0 / elapsed if elapsed > 0 else 0.0
            fps_timer_start = time.time()

        # ── STEP A: Face detection ─────────────────────────────
        if frame_count % FACE_DETECTION_EVERY_N_FRAMES == 0:
            if face_detection_available and face_model is not None:
                raw_result = fd.analyze_frame(frame, face_model)
                face_frame          = raw_result.pop("annotated_frame", frame.copy())
                last_face_result    = raw_result
                last_face_annotated = face_frame
            else:
                last_face_annotated = frame.copy()

        if last_face_annotated is not None:
            composed_frame = last_face_annotated.copy()
        else:
            composed_frame = frame.copy()

        # ── STEP B: Phone detection ────────────────────────────
        if frame_count % PHONE_DETECTION_EVERY_N_FRAMES == 0:
            if phone_detection_available and phone_model is not None:
                phone_result = pd_mod.detect_phone(frame, phone_model)
                last_phone_result = phone_result

        # ── STEP C: Draw phone boxes ───────────────────────────
        if (phone_detection_available
                and last_phone_result.get("phone_detected", False)
                and last_phone_result.get("boxes")):
            composed_frame = pd_mod.draw_phone_boxes(
                composed_frame,
                last_phone_result["boxes"]
            )

        # ── STEP D: Behavioral classification ──────────────────
        # Feed the latest face + phone results into the classifier.
        # The classifier applies temporal smoothing and returns the
        # official state (which may lag behind the raw signal).
        classified_state = "FOCUSED"  # Default if classifier unavailable
        if classifier_available and classifier is not None:
            classified_state = ac.classify_frame(
                classifier,
                last_face_result,
                last_phone_result,
            )

        # ── STEP E: Draw FPS counter ───────────────────────────
        cv2.putText(
            composed_frame,
            f"FPS: {current_fps:.0f}",
            (CAPTURE_WIDTH - 72, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (100, 255, 100), 1, cv2.LINE_AA
        )

        # ── STEP F: Update shared state ────────────────────────
        with _lock:
            _state["annotated_frame"]  = composed_frame
            _state["raw_frame"]        = frame.copy()
            _state["face_result"]      = dict(last_face_result)
            _state["phone_result"]     = dict(last_phone_result)
            _state["classified_state"] = classified_state
            _state["fps"]              = current_fps
            _state["frame_count"]      = frame_count

        time.sleep(0.01)

    # ── Cleanup ────────────────────────────────────────────────
    cap.release()

    with _lock:
        _state["running"]         = False
        _state["annotated_frame"] = None
        _state["raw_frame"]       = None
