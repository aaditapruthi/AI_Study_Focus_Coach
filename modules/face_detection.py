# ============================================================
# modules/face_detection.py
# ============================================================
#
# PURPOSE:
#   This module handles everything related to detecting a human
#   face and estimating where the person is looking.
#
#   It uses Google's MediaPipe Face Landmarker — a pretrained
#   deep learning model that finds 478 specific points on the face
#   (called landmarks) in a single image in milliseconds.
#
#   We use ~6 of those 478 points for our gaze estimation:
#   the corners of each eye, and the center of each iris.
#
# PUBLIC FUNCTIONS (called from camera.py and app.py):
#   load_face_model(model_path)   → loads and returns the detector
#   analyze_frame(bgr_frame, detector) → dict with gaze result
#
# PRIVATE FUNCTIONS (only called inside this file):
#   _estimate_gaze(landmarks)      → (direction_str, ratio_float)
#   _draw_face_overlays(frame, ...) → annotated frame
#
# HOW TO GET THE MODEL FILE:
#   Run this in your terminal from the project root:
#   curl -o assets/face_landmarker.task \
#     https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
# ============================================================

import cv2
import numpy as np
import os

# ---- MediaPipe import with graceful fallback ----
# If MediaPipe is not installed, we set a flag and return
# clear error messages instead of crashing the entire app.
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


# ============================================================
# LANDMARK INDEX CONSTANTS
# ============================================================
# MediaPipe Face Landmarker produces 478 landmarks.
# - Landmarks 0–467:   The standard face mesh (same as old Face Mesh API)
# - Landmarks 468–477: The iris landmarks (10 extra points, 5 per eye)
#
# We only use 6 of these 478 points for gaze estimation.
# Using named constants (instead of raw numbers like 468) makes
# the code much easier to read and understand.
#
# VISUALIZE THESE INDICES:
#   https://mediapipe-studio.webapps.google.com/studio/demo/face-landmarker
#
# LEFT EYE (from the PERSON's perspective, appears on RIGHT side of screen
#           when the webcam image is shown as a mirror/selfie view):
LEFT_EYE_OUTER_CORNER = 33    # The corner of the eye toward the left ear
LEFT_EYE_INNER_CORNER = 133   # The corner of the eye toward the nose
LEFT_IRIS_CENTER      = 468   # The center of the left iris (pupil area)

# RIGHT EYE (from the PERSON's perspective, appears on LEFT side of screen):
RIGHT_EYE_INNER_CORNER = 362  # The corner of the right eye toward the nose
RIGHT_EYE_OUTER_CORNER = 263  # The corner of the right eye toward the right ear
RIGHT_IRIS_CENTER      = 473  # The center of the right iris

# ============================================================
# GAZE ESTIMATION THRESHOLDS
# ============================================================
# These values control when we classify gaze as "looking away".
#
# HOW THE RATIO WORKS:
#   For each eye, we compute:
#     ratio = (iris_x - eye_left_edge_x) / eye_total_width
#
#   This gives a number from 0.0 to 1.0:
#     0.0  = iris is at the far left of the eye  → looking hard left
#     0.5  = iris is perfectly centered           → looking straight at screen
#     1.0  = iris is at the far right of the eye → looking hard right
#
#   We then average the left and right eye ratios.
#
# THRESHOLD INTERPRETATION:
#   If average ratio < GAZE_MIN or > GAZE_MAX → "looking_away"
#   If GAZE_MIN ≤ ratio ≤ GAZE_MAX           → "centered"
#
# These thresholds (0.30 and 0.70) work well for typical webcam
# use. If gaze flickers too much, widen the range slightly (e.g. 0.25–0.75).
GAZE_MIN_THRESHOLD = 0.30
GAZE_MAX_THRESHOLD = 0.70

# Default path where the model file should be placed
DEFAULT_MODEL_PATH = "assets/face_landmarker.task"

# ---- Module-level cached detector ----
# Loaded once on the first call to load_face_model().
# Reused on every subsequent call without reloading from disk.
_detector = None


# ============================================================
# PUBLIC FUNCTION: load_face_model
# ============================================================
def load_face_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Load the MediaPipe Face Landmarker model from disk.

    This function uses the MediaPipe Tasks API, which is the CURRENT
    (2024+) way to use MediaPipe in Python. The older approach was
    mp.solutions.face_mesh — that still works but is considered legacy.

    The Tasks API requires a .task model file downloaded separately.
    It is NOT bundled with the mediapipe Python package because it
    would make the package too large.

    WHY WE CACHE THE DETECTOR (_detector):
        Loading a deep learning model from disk is slow (0.5–2 seconds).
        We store the loaded detector in the module-level variable _detector
        and return it on every call after the first. Python modules are
        only loaded once per process, so this variable persists.

    Args:
        model_path: Path to the face_landmarker.task file.
                    Default: "assets/face_landmarker.task"

    Returns:
        The loaded FaceLandmarker detector object, or None if loading failed.
    """
    global _detector

    # Return cached detector immediately if already loaded
    if _detector is not None:
        return _detector

    # Check if MediaPipe was imported successfully
    if not MEDIAPIPE_AVAILABLE:
        print("[face_detection] ERROR: MediaPipe is not installed.")
        print("  Run: pip install mediapipe>=0.10.14")
        return None

    # Check that the model file exists on disk, or download it automatically
    if not os.path.exists(model_path):
        print(f"[face_detection] Model file not found at '{model_path}'. Attempting automatic download...")
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else ".", exist_ok=True)
        try:
            import urllib.request
            model_url = (
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task"
            )
            urllib.request.urlretrieve(model_url, model_path)
            print(f"[face_detection] Successfully downloaded face model to '{model_path}'")
        except Exception as dl_err:
            print(f"[face_detection] ERROR: Could not auto-download model: {dl_err}")
            print(f"  Please download manually: curl -o {model_path} https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
            return None

    try:
        # ---- Step 1: Create BaseOptions pointing to the model file ----
        # BaseOptions tells MediaPipe where to find the model.
        base_options = mp_python.BaseOptions(model_asset_path=model_path)

        # ---- Step 2: Configure the FaceLandmarker ----
        # FaceLandmarkerOptions controls what the model outputs and
        # how it makes decisions.
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,

            # RunningMode.IMAGE = process one frame at a time (synchronous).
            # This is the simplest mode — we call detect() and immediately
            # get back a result. No callbacks, no async needed.
            running_mode=mp_vision.RunningMode.IMAGE,

            # We only expect one student in front of the camera.
            num_faces=1,

            # We don't need face expression data (smile, blink, etc.)
            output_face_blendshapes=False,

            # We don't need the head-rotation transformation matrix.
            output_facial_transformation_matrixes=False,

            # Confidence thresholds — how certain MediaPipe must be before
            # reporting a detection. 0.5 = 50% minimum confidence.
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # ---- Step 3: Create the detector ----
        _detector = mp_vision.FaceLandmarker.create_from_options(options)
        print(f"[face_detection] Model loaded successfully from '{model_path}'")
        return _detector

    except Exception as e:
        print(f"[face_detection] ERROR during model loading: {e}")
        return None


# ============================================================
# PUBLIC FUNCTION: analyze_frame
# ============================================================
def analyze_frame(bgr_frame: np.ndarray, detector) -> dict:
    """
    Run face landmark detection on one video frame and estimate gaze.

    This is the main function called on every frame from camera.py.
    It orchestrates the full pipeline:
        BGR frame → RGB conversion → MediaPipe detection →
        landmark extraction → gaze estimation → overlay drawing

    WHY BGR→RGB CONVERSION?
        OpenCV (which captures the webcam frame) stores images in BGR order
        (Blue, Green, Red). This is a historical quirk of OpenCV.
        MediaPipe, like most other image libraries, expects RGB order
        (Red, Green, Blue). If we skip conversion, colors look wrong
        and detection accuracy drops.

    Args:
        bgr_frame: A numpy array of shape (height, width, 3) in BGR color
                   order, as captured by OpenCV.
        detector:  The FaceLandmarker object returned by load_face_model().

    Returns:
        A dictionary with these keys:
            "face_detected"   (bool)        — True if a face was found
            "gaze"            (str)         — "centered", "looking_away", or "unknown"
            "gaze_ratio"      (float)       — iris position ratio, 0.0–1.0
            "annotated_frame" (np.ndarray)  — BGR frame with drawn overlays
            "error"           (str or None) — error message if something failed
    """
    # Build a default result to return when detection fails or finds nothing
    default_result = {
        "face_detected": False,
        "gaze": "unknown",
        "gaze_ratio": 0.5,
        "annotated_frame": bgr_frame.copy(),
        "error": None,
    }

    # Guard: if model failed to load, return default with an error
    if detector is None:
        default_result["error"] = (
            "Face model not loaded. "
            "Make sure assets/face_landmarker.task exists."
        )
        return default_result

    try:
        # ── Step 1: Convert BGR → RGB ──────────────────────────────
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # Get image dimensions (used later for drawing pixel coordinates)
        frame_height, frame_width = bgr_frame.shape[:2]

        # ── Step 2: Wrap in a MediaPipe Image object ───────────────
        # MediaPipe needs its own Image type, not a raw numpy array.
        # mp.Image accepts an RGB numpy array and stores it internally.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # ── Step 3: Run the detector ───────────────────────────────
        # This calls the neural network and returns structured results.
        # In IMAGE running mode, this call is SYNCHRONOUS — it blocks
        # until detection is complete (typically 10–40ms on a modern CPU).
        detection_result = detector.detect(mp_image)

        # ── Step 4: Check if any faces were found ──────────────────
        # detection_result.face_landmarks is a list of faces.
        # Each face is a list of 478 NormalizedLandmark objects.
        # If the list is empty, no face was detected in this frame.
        if not detection_result.face_landmarks:
            return default_result  # face_detected stays False

        # ── Step 5: Extract the first face's landmarks ─────────────
        # We only track one face (num_faces=1 in options), so index [0]
        # is always the only detected face.
        landmarks = detection_result.face_landmarks[0]
        # landmarks is now a list of 478 NormalizedLandmark objects.
        # Each has .x and .y in range [0.0, 1.0] relative to image size.
        # Example: landmarks[33].x = 0.42 means the outer left eye corner
        #          is at 42% of the way across the image horizontally.

        # ── Step 6: Estimate gaze direction ───────────────────────
        gaze, gaze_ratio = _estimate_gaze(landmarks)

        # ── Step 7: Draw visual overlays on the frame ─────────────
        # We work on a COPY of the frame so the original is unchanged.
        annotated_frame = _draw_face_overlays(
            bgr_frame.copy(), landmarks, gaze, frame_width, frame_height
        )

        return {
            "face_detected": True,
            "gaze": gaze,
            "gaze_ratio": round(gaze_ratio, 3),
            "annotated_frame": annotated_frame,
            "error": None,
        }

    except Exception as e:
        # Catch any unexpected error (e.g. landmark index out of range)
        # so the camera loop doesn't crash
        default_result["error"] = f"Frame analysis error: {str(e)}"
        return default_result


# ============================================================
# PRIVATE FUNCTION: _estimate_gaze
# ============================================================
def _estimate_gaze(landmarks: list) -> tuple:
    """
    Estimate where the person is looking by measuring iris position
    relative to eye corners.

    ALGORITHM EXPLANATION (simple version):
        Imagine the eye as a horizontal slot.
        The iris (the colored part of the eye) can slide left or right.
        When looking straight ahead, the iris is in the MIDDLE of the slot.
        When looking sideways, the iris shifts toward one edge.

        We measure this as a ratio from 0 (far left) to 1 (far right).
        A ratio near 0.5 means the iris is centered → person looking at screen.
        A ratio far from 0.5 means the iris is off-center → looking away.

    WHY BOTH EYES?
        Using both eyes and averaging them makes the estimate more stable.
        If one eye measurement is unreliable (e.g. partially out of frame),
        the other eye still contributes a good reading.

    LIMITATIONS:
        - Only estimates horizontal gaze, not vertical
        - Assumes the face is roughly front-facing
        - Poor lighting or partial face coverage reduces accuracy
        - Blinking may cause brief "looking away" reads
          (addressed in Phase 4 with temporal smoothing)

    Args:
        landmarks: List of 478 NormalizedLandmark objects from MediaPipe.

    Returns:
        (gaze_direction, ratio)
        gaze_direction: "centered", "looking_away", or "unknown"
        ratio:          Float from 0.0 to 1.0 (iris position)
    """
    try:
        # ── LEFT EYE MEASUREMENT ───────────────────────────────────
        # Get the x coordinates (horizontal position) of:
        #   - Outer corner (toward the left ear)
        #   - Inner corner (toward the nose)
        #   - Iris center
        left_outer_x = landmarks[LEFT_EYE_OUTER_CORNER].x
        left_inner_x = landmarks[LEFT_EYE_INNER_CORNER].x
        left_iris_x  = landmarks[LEFT_IRIS_CENTER].x

        # Find the leftmost and rightmost x of the left eye
        # (We use min/max because the "outer" corner might have a smaller
        # or larger x value depending on face orientation.)
        left_eye_x_min = min(left_outer_x, left_inner_x)
        left_eye_x_max = max(left_outer_x, left_inner_x)
        left_eye_width = left_eye_x_max - left_eye_x_min

        if left_eye_width < 0.01:
            # Eye too narrow to measure reliably (extreme profile view)
            left_ratio = 0.5
        else:
            # How far is the iris from the left edge, as a fraction of eye width?
            left_ratio = (left_iris_x - left_eye_x_min) / left_eye_width

        # ── RIGHT EYE MEASUREMENT ──────────────────────────────────
        right_inner_x = landmarks[RIGHT_EYE_INNER_CORNER].x
        right_outer_x = landmarks[RIGHT_EYE_OUTER_CORNER].x
        right_iris_x  = landmarks[RIGHT_IRIS_CENTER].x

        right_eye_x_min = min(right_inner_x, right_outer_x)
        right_eye_x_max = max(right_inner_x, right_outer_x)
        right_eye_width = right_eye_x_max - right_eye_x_min

        if right_eye_width < 0.01:
            right_ratio = 0.5
        else:
            right_ratio = (right_iris_x - right_eye_x_min) / right_eye_width

        # ── AVERAGE BOTH EYES ──────────────────────────────────────
        avg_ratio = (left_ratio + right_ratio) / 2.0

        # Clamp to [0, 1] in case of floating point rounding issues
        avg_ratio = max(0.0, min(1.0, avg_ratio))

        # ── CLASSIFY ───────────────────────────────────────────────
        if GAZE_MIN_THRESHOLD <= avg_ratio <= GAZE_MAX_THRESHOLD:
            return "centered", avg_ratio
        else:
            return "looking_away", avg_ratio

    except (IndexError, AttributeError):
        # IndexError:    landmark index out of range (model gave fewer than 478)
        # AttributeError: landmark object doesn't have .x attribute
        return "unknown", 0.5


# ============================================================
# PRIVATE FUNCTION: _draw_face_overlays
# ============================================================
def _draw_face_overlays(
    frame: np.ndarray,
    landmarks: list,
    gaze: str,
    frame_width: int,
    frame_height: int
) -> np.ndarray:
    """
    Draw visual landmark overlays directly onto the video frame.

    This function uses OpenCV drawing functions (cv2.circle, cv2.line,
    cv2.putText, cv2.rectangle) to annotate the frame.

    WHAT WE DRAW:
        1. Face oval — a rough outline of the face boundary
        2. Left and right eye corner dots (the reference points we use)
        3. Left and right iris center dots (the moving point we track)
        4. Lines connecting eye corners (shows the measurement axis)
        5. Status banner at the bottom of the frame

    WHY DRAW ON A COPY?
        We always call this with bgr_frame.copy() so the original
        frame in memory is never modified. This avoids bugs where
        overlays from one frame accidentally appear in the next.

    Args:
        frame:        BGR numpy array to draw on.
        landmarks:    List of 478 NormalizedLandmark objects.
        gaze:         "centered", "looking_away", or "unknown"
        frame_width:  Width of the frame in pixels.
        frame_height: Height of the frame in pixels.

    Returns:
        The same frame array with drawings added.
    """
    def to_px(landmark):
        """
        Convert a normalized landmark (x, y in [0,1]) to pixel coordinates.
        Multiplying by frame dimensions gives us actual pixel positions.
        """
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        return (x, y)

    # ── 1. Draw face oval contour ──────────────────────────────────
    # These 36 landmark indices form the outer boundary of the face.
    # Drawing lines between consecutive points creates the face outline.
    face_oval_indices = [
        10, 338, 297, 332, 284, 251, 389, 356, 454,
        323, 361, 288, 397, 365, 379, 378, 400, 377,
        152, 148, 176, 149, 150, 136, 172,  58, 132,
         93, 234, 127, 162,  21,  54, 103,  67, 109, 10
    ]
    oval_color = (160, 160, 160)   # Light grey in BGR
    for i in range(len(face_oval_indices) - 1):
        try:
            pt1 = to_px(landmarks[face_oval_indices[i]])
            pt2 = to_px(landmarks[face_oval_indices[i + 1]])
            cv2.line(frame, pt1, pt2, oval_color, 1, cv2.LINE_AA)
        except IndexError:
            break  # Stop if we run out of landmarks

    # ── 2. Draw eye corner reference points ───────────────────────
    # These are the 4 anchor points we use to define the eye span.
    # Drawn as filled circles (cv2.circle with thickness=-1 means filled).
    eye_corner_color = (50, 220, 80)   # Green in BGR
    eye_corner_indices = [
        LEFT_EYE_OUTER_CORNER,
        LEFT_EYE_INNER_CORNER,
        RIGHT_EYE_OUTER_CORNER,
        RIGHT_EYE_INNER_CORNER,
    ]
    for idx in eye_corner_indices:
        try:
            cv2.circle(frame, to_px(landmarks[idx]), 4, eye_corner_color, -1, cv2.LINE_AA)
        except IndexError:
            pass

    # ── 3. Draw lines connecting eye corners ──────────────────────
    # These horizontal lines show the "measurement axis" for each eye.
    try:
        cv2.line(
            frame,
            to_px(landmarks[LEFT_EYE_OUTER_CORNER]),
            to_px(landmarks[LEFT_EYE_INNER_CORNER]),
            eye_corner_color, 1, cv2.LINE_AA
        )
        cv2.line(
            frame,
            to_px(landmarks[RIGHT_EYE_INNER_CORNER]),
            to_px(landmarks[RIGHT_EYE_OUTER_CORNER]),
            eye_corner_color, 1, cv2.LINE_AA
        )
    except IndexError:
        pass

    # ── 4. Draw iris centers ───────────────────────────────────────
    # Iris centers are drawn as bright blue dots.
    # These are the MOVING points — they shift as the eyes move.
    iris_color = (220, 100, 30)   # Blue-ish in BGR
    for idx in [LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER]:
        try:
            cv2.circle(frame, to_px(landmarks[idx]), 5, iris_color, -1, cv2.LINE_AA)
            # Draw a small ring around the iris dot for visibility
            cv2.circle(frame, to_px(landmarks[idx]), 7, iris_color, 1, cv2.LINE_AA)
        except IndexError:
            pass

    # ── 5. Draw status banner ──────────────────────────────────────
    # A semi-transparent colored rectangle at the bottom shows the state.
    if gaze == "centered":
        banner_bgr = (60, 160, 80)         # Green
        label_text = "GAZE: Centered — Focused"
    elif gaze == "looking_away":
        banner_bgr = (40, 150, 200)        # Amber-orange
        label_text = "GAZE: Looking Away"
    else:
        banner_bgr = (100, 100, 100)       # Grey
        label_text = "GAZE: Analyzing..."

    # Create a semi-transparent overlay by blending with a solid rectangle
    banner_top = frame_height - 36
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, banner_top), (frame_width, frame_height),
                  banner_bgr, -1)
    # alpha controls transparency: 0.0 = invisible, 1.0 = fully opaque
    alpha = 0.65
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw white text on top of the banner
    cv2.putText(
        frame, label_text,
        (10, frame_height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56, (255, 255, 255), 1, cv2.LINE_AA
    )

    return frame
