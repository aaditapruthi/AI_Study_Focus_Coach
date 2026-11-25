# ============================================================
# modules/attention_classifier.py
# ============================================================
#
# PURPOSE:
#   This module takes the RAW computer vision signals produced
#   by face_detection.py and phone_detection.py and converts
#   them into one of four human-readable behavioral states:
#
#       1. FOCUSED               — Face present, gaze centered, no phone
#       2. DISTRACTED_PHONE      — Phone detected in frame
#       3. DISTRACTED_LOOK_AWAY  — Face present but gaze not centered
#       4. AWAY_FROM_DESK        — No face detected for several seconds
#
# WHY A SEPARATE CLASSIFIER?
#   The raw CV signals are noisy. MediaPipe might report
#   "looking_away" for a single frame due to a blink, shadow,
#   or head tilt. If we changed the displayed state on every
#   single frame, the UI would flicker constantly.
#
#   This module adds TEMPORAL SMOOTHING — it requires a signal
#   to be consistent for a configurable number of consecutive
#   frames before it commits to a state change. This prevents
#   brief glitches from being counted as real distractions.
#
# HOW IT WORKS (simple explanation):
#   Think of it as a vote counter. Each new frame casts a "vote"
#   for a candidate state. Only after the same candidate gets
#   enough consecutive votes (the confirmation threshold) does
#   the classifier change the official state. If a different
#   candidate appears before the threshold is reached, the
#   counter resets.
#
# PRIORITY ORDER:
#   When multiple signals fire simultaneously (e.g. phone visible
#   AND gaze away), the classifier uses this priority:
#       1. Phone detected    (highest priority — most disruptive)
#       2. No face detected  (second — implies physically absent)
#       3. Gaze away         (third — still at desk but distracted)
#       4. Focused           (default — everything looks good)
#
# PUBLIC FUNCTIONS:
#   create_classifier()         → new classifier dict (call once per session)
#   classify_frame(classifier, face_result, phone_result) → state string
#   get_current_state(classifier) → current state string
#
# This module uses NO classes. The classifier state is stored in
# a plain dictionary, which makes it simple to understand and
# easy to serialize if needed.
# ============================================================

import time


# ============================================================
# BEHAVIORAL STATE CONSTANTS
# ============================================================
# Using named constants avoids typos and lets the IDE help with
# autocomplete. Every module that needs to check a state string
# should import these constants rather than typing the string.

STATE_FOCUSED           = "FOCUSED"
STATE_DISTRACTED_PHONE  = "DISTRACTED - PHONE"
STATE_DISTRACTED_LOOK   = "DISTRACTED - LOOKING AWAY"
STATE_AWAY              = "AWAY FROM DESK"


# ============================================================
# TEMPORAL SMOOTHING CONFIGURATION
# ============================================================
# These values control how many consecutive frames must agree
# before the classifier commits to a state change.
#
# EXAMPLE with CONFIRMATION_FRAMES = 5:
#   Frame 1: raw signal = "looking_away"  → counter = 1 (keep FOCUSED)
#   Frame 2: raw signal = "looking_away"  → counter = 2 (keep FOCUSED)
#   Frame 3: raw signal = "centered"      → counter RESETS to 0
#   Frame 4: raw signal = "looking_away"  → counter = 1 (keep FOCUSED)
#   Frame 5: raw signal = "looking_away"  → counter = 2 (keep FOCUSED)
#   Frame 6: raw signal = "looking_away"  → counter = 3 (keep FOCUSED)
#   Frame 7: raw signal = "looking_away"  → counter = 4 (keep FOCUSED)
#   Frame 8: raw signal = "looking_away"  → counter = 5 → CHANGE STATE
#
# At ~15–30 FPS, 5 frames ≈ 0.2–0.3 seconds of consistent signal.

CONFIRMATION_FRAMES = 5

# How many seconds of continuous "no face" before we declare AWAY.
# This is measured in wall-clock time, not frames, because the
# user might genuinely glance down at a book for a moment.
# 5 seconds of no face = likely physically left the desk.
NO_FACE_TIMEOUT_SECONDS = 5.0


# ============================================================
# PUBLIC FUNCTION: create_classifier
# ============================================================
def create_classifier() -> dict:
    """
    Create a new classifier state dictionary.

    Call this ONCE when a new study session starts. It initializes
    all the tracking variables the classifier needs.

    WHY A DICTIONARY (not a class)?
        A dictionary is the simplest Python data structure.
        It's easy to inspect during debugging (just print it),
        easy to pass between functions, and doesn't require
        understanding __init__, self, or inheritance.

    Returns:
        A dict containing:
            current_state     — The official behavioral state
            candidate_state   — The state the raw signal is pushing toward
            candidate_count   — Consecutive frames the candidate has held
            no_face_since     — Timestamp when face was last lost (or None)
            last_update_time  — When the state was last changed
    """
    return {
        # The current OFFICIAL state displayed to the user.
        # Starts as FOCUSED because we assume the student is
        # ready when they press "Start Session".
        "current_state": STATE_FOCUSED,

        # The state that the raw CV signal is currently suggesting.
        # This may differ from current_state during the confirmation
        # period (before enough consecutive frames confirm it).
        "candidate_state": STATE_FOCUSED,

        # How many consecutive frames have voted for candidate_state.
        # Once this reaches CONFIRMATION_FRAMES, the state changes.
        "candidate_count": 0,

        # Timestamp (from time.time()) when face detection last
        # reported "no face". Set to None when a face IS detected.
        # Used to enforce the NO_FACE_TIMEOUT_SECONDS rule:
        # only classify as AWAY_FROM_DESK after the face has been
        # missing for 5+ continuous seconds.
        "no_face_since": None,

        # Timestamp of the most recent state change. Useful for
        # calculating how long the current state has lasted.
        "last_update_time": time.time(),
    }


# ============================================================
# PUBLIC FUNCTION: classify_frame
# ============================================================
def classify_frame(
    classifier: dict,
    face_result: dict,
    phone_result: dict,
) -> str:
    """
    Process one frame's CV results and return the behavioral state.

    This function is called by camera.py on every processed frame.
    It applies the priority rules and temporal smoothing, then
    returns the OFFICIAL state (which may lag behind the raw signal
    due to the confirmation requirement).

    PRIORITY RULES (applied in order):
        1. If phone_detected → candidate = DISTRACTED - PHONE
        2. Else if no face for ≥ 5 seconds → candidate = AWAY FROM DESK
        3. Else if face present but gaze = "looking_away" → candidate = DISTRACTED - LOOKING AWAY
        4. Else → candidate = FOCUSED

    Args:
        classifier:   The classifier dict from create_classifier().
        face_result:  Dict from face_detection.analyze_frame() with keys:
                        face_detected (bool), gaze (str), gaze_ratio (float)
        phone_result: Dict from phone_detection.detect_phone() with keys:
                        phone_detected (bool), confidence (float)

    Returns:
        str: One of the four STATE_* constants.

    Side effect:
        Modifies the classifier dict in-place (updates counters,
        timestamps, and state if the confirmation threshold is met).
    """
    now = time.time()

    # ── Extract raw signals from CV results ────────────────────────
    face_detected  = face_result.get("face_detected", False)
    gaze           = face_result.get("gaze", "unknown")
    phone_detected = phone_result.get("phone_detected", False)

    # ── Track "no face" duration ───────────────────────────────────
    # Update the no_face_since timestamp:
    #   - If face IS detected → reset to None (face is back)
    #   - If face is NOT detected and timer hasn't started → start it
    if face_detected:
        classifier["no_face_since"] = None
    else:
        if classifier["no_face_since"] is None:
            # Face just disappeared — start timing
            classifier["no_face_since"] = now

    # Calculate how long the face has been missing (0 if face is present)
    if classifier["no_face_since"] is not None:
        no_face_duration = now - classifier["no_face_since"]
    else:
        no_face_duration = 0.0

    # ── Apply priority rules to determine the RAW candidate ───────
    # This is what the frame is "voting" for, before smoothing.

    if phone_detected:
        # Priority 1: Phone detected → highest distraction
        raw_state = STATE_DISTRACTED_PHONE

    elif no_face_duration >= NO_FACE_TIMEOUT_SECONDS:
        # Priority 2: No face for 5+ seconds → away from desk
        # Note: we do NOT trigger immediately. Brief moments where
        # MediaPipe misses the face (shadows, glare) are ignored
        # because they don't last 5 seconds.
        raw_state = STATE_AWAY

    elif face_detected and gaze == "looking_away":
        # Priority 3: Face present but gaze is off-center
        raw_state = STATE_DISTRACTED_LOOK

    elif face_detected and gaze == "centered":
        # Priority 4: Everything looks good
        raw_state = STATE_FOCUSED

    else:
        # Fallback: face not detected but not long enough for AWAY,
        # or gaze is "unknown". Keep the current state to avoid
        # flickering during brief uncertain moments.
        raw_state = classifier["current_state"]

    # ── Temporal smoothing (vote counting) ─────────────────────────
    # Compare the raw vote to the current candidate:
    #   - Same as candidate → increment counter
    #   - Different → reset candidate and counter
    if raw_state == classifier["candidate_state"]:
        classifier["candidate_count"] += 1
    else:
        # New candidate — start counting from 1
        classifier["candidate_state"] = raw_state
        classifier["candidate_count"] = 1

    # ── State transition check ─────────────────────────────────────
    # Only change the official state if:
    #   1. The candidate differs from the current state, AND
    #   2. The candidate has been consistent for enough frames
    if (classifier["candidate_state"] != classifier["current_state"]
            and classifier["candidate_count"] >= CONFIRMATION_FRAMES):

        classifier["current_state"]   = classifier["candidate_state"]
        classifier["last_update_time"] = now
        # Reset the counter since the transition is complete
        classifier["candidate_count"] = 0

    return classifier["current_state"]


# ============================================================
# PUBLIC FUNCTION: get_current_state
# ============================================================
def get_current_state(classifier: dict) -> str:
    """
    Return the current official behavioral state.

    This is a convenience function for reading the state without
    calling classify_frame(). Useful when you need the state but
    don't have new CV results (e.g. between frames).

    Args:
        classifier: The classifier dict.

    Returns:
        str: One of the four STATE_* constants.
    """
    return classifier.get("current_state", STATE_FOCUSED)
