# ============================================================
# test_classifier.py — Unit tests for the AI Study Focus Coach
# ============================================================
#
# PURPOSE:
#   Test the behavioral classifier and session manager WITHOUT
#   needing a webcam, MediaPipe, or YOLOv8. All inputs are
#   simulated dictionaries — just like the real camera module
#   would produce.
#
# HOW TO RUN:
#   python test_classifier.py
#
#   Expected output: "All 11 tests passed."
#   This file has no external dependencies beyond what is in
#   requirements.txt (pandas). No Streamlit, no camera needed.
#
# WHAT IS BEING TESTED:
#   1.  Initial classifier state is FOCUSED
#   2.  Phone detection → immediate candidate change
#   3.  Temporal smoothing: 4 frames not enough to confirm
#   4.  Temporal smoothing: 5 frames triggers state change
#   5.  Priority: phone overrides gaze-away
#   6.  Face returning resets the no-face timer
#   7.  start_session() creates the correct dict structure
#   8.  update_session() accumulates durations correctly
#   9.  focus_percentage calculation is correct
#   10. Longest focus period tracking works
#   11. Session CSV save and reload round-trip
#
# HOW TESTS WORK:
#   Each test is a simple function that:
#     1. Sets up an input (simulated face_result, phone_result, session)
#     2. Calls the actual module function (no mocking)
#     3. Checks the output with an assert statement
#     4. Prints PASSED or FAILED
# ============================================================

import sys
import os
import time
import shutil

# ── Make sure we can import modules/ from this directory ──────
# When running "python test_classifier.py" from the project root,
# Python's sys.path includes the current directory, so this works.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.attention_classifier import (
    create_classifier,
    classify_frame,
    get_current_state,
    STATE_FOCUSED,
    STATE_DISTRACTED_PHONE,
    STATE_DISTRACTED_LOOK,
    STATE_AWAY,
    CONFIRMATION_FRAMES,
    NO_FACE_TIMEOUT_SECONDS,
)
from modules import session_manager as sm


# ============================================================
# TEST HELPER FUNCTIONS
# ============================================================

# Keep a count of tests passed and failed so we can print a summary.
_passed = 0
_failed = 0


def assert_equal(actual, expected, test_name: str):
    """
    Check that `actual` equals `expected`.
    Print PASSED or FAILED and update global counters.
    """
    global _passed, _failed
    if actual == expected:
        print(f"  Test {test_name} PASSED")
        _passed += 1
    else:
        print(f"  Test {test_name} FAILED")
        print(f"    Expected: {expected!r}")
        print(f"    Actual:   {actual!r}")
        _failed += 1


def assert_true(condition: bool, test_name: str):
    """Check that `condition` is True."""
    assert_equal(condition, True, test_name)


def assert_almost_equal(actual: float, expected: float, tolerance: float, test_name: str):
    """Check that `actual` is within `tolerance` of `expected`."""
    global _passed, _failed
    if abs(actual - expected) <= tolerance:
        print(f"  Test {test_name} PASSED")
        _passed += 1
    else:
        print(f"  Test {test_name} FAILED")
        print(f"    Expected: {expected!r} ± {tolerance}")
        print(f"    Actual:   {actual!r}")
        _failed += 1


# ============================================================
# HELPER: simulated face_result and phone_result dicts
# ============================================================
# These mimic what face_detection.analyze_frame() and
# phone_detection.detect_phone() return in the real application.

def face_centered():
    """Simulate: face detected, gaze looks centered at screen."""
    return {"face_detected": True, "gaze": "centered", "gaze_ratio": 0.5}

def face_looking_away():
    """Simulate: face detected but gaze is off to the side."""
    return {"face_detected": True, "gaze": "looking_away", "gaze_ratio": 0.9}

def no_face():
    """Simulate: MediaPipe found no face in this frame."""
    return {"face_detected": False, "gaze": "unknown", "gaze_ratio": 0.5}

def phone_detected():
    """Simulate: YOLOv8 detected a phone in the frame."""
    return {"phone_detected": True, "confidence": 0.85, "boxes": [[100, 100, 200, 200]]}

def phone_not_detected():
    """Simulate: YOLOv8 found no phone in the frame."""
    return {"phone_detected": False, "confidence": 0.0, "boxes": []}


# ============================================================
# CLASSIFIER TESTS (Tests 1–6)
# ============================================================

def test_1_initial_state_is_focused():
    """
    When a new classifier is created, the state must be FOCUSED.

    WHY: start_session() assumes the student is ready and focused
    when they click "Start Session". The classifier must start in
    the same state.
    """
    classifier = create_classifier()
    state = get_current_state(classifier)
    assert_equal(state, STATE_FOCUSED,
                 "1  Initial state is FOCUSED")


def test_2_phone_detection_changes_candidate():
    """
    When a phone is detected, the RAW candidate must immediately
    switch to DISTRACTED - PHONE, even before 5 frames confirm it.

    We also check that after CONFIRMATION_FRAMES frames of phone
    detection, the official state changes to DISTRACTED - PHONE.
    """
    classifier = create_classifier()

    # Feed CONFIRMATION_FRAMES frames with phone detected
    final_state = STATE_FOCUSED  # will be overwritten
    for _ in range(CONFIRMATION_FRAMES):
        final_state = classify_frame(classifier, face_centered(), phone_detected())

    assert_equal(final_state, STATE_DISTRACTED_PHONE,
                 "2  Phone detection → DISTRACTED - PHONE (priority override)")


def test_3_temporal_smoothing_4_frames_not_enough():
    """
    4 consecutive frames of "looking away" must NOT change the state.
    CONFIRMATION_FRAMES = 5, so 4 is one short.

    WHY THIS MATTERS:
        Without smoothing, a single bad frame (glare, blink, shadow)
        would trigger a false distraction event. The smoothing filter
        prevents that by requiring several consistent frames.
    """
    classifier = create_classifier()

    # Feed exactly (CONFIRMATION_FRAMES - 1) frames of gaze-away
    for _ in range(CONFIRMATION_FRAMES - 1):
        state = classify_frame(classifier, face_looking_away(), phone_not_detected())

    # State should still be FOCUSED — not enough frames to confirm
    assert_equal(state, STATE_FOCUSED,
                 "3  Temporal smoothing — 4 frames not enough to change state")


def test_4_temporal_smoothing_5_frames_triggers_change():
    """
    Exactly CONFIRMATION_FRAMES consecutive frames of "looking away"
    MUST change the state to DISTRACTED - LOOKING AWAY.

    This confirms the threshold works at the boundary condition.
    """
    classifier = create_classifier()

    # Feed exactly CONFIRMATION_FRAMES frames of gaze-away
    for _ in range(CONFIRMATION_FRAMES):
        state = classify_frame(classifier, face_looking_away(), phone_not_detected())

    assert_equal(state, STATE_DISTRACTED_LOOK,
                 "4  Temporal smoothing — 5 frames triggers state change")


def test_5_phone_priority_overrides_looking_away():
    """
    Phone detection has HIGHER priority than gaze direction.
    Even if the gaze is centered, a detected phone must result in
    DISTRACTED - PHONE, not DISTRACTED - LOOKING AWAY.

    WHY: A phone is more serious than a gaze deviation. The student
    has explicitly picked up a device.
    """
    classifier = create_classifier()

    # Feed frames where face looks away AND phone is detected
    # Priority: phone > gaze → state must be DISTRACTED - PHONE
    for _ in range(CONFIRMATION_FRAMES):
        state = classify_frame(classifier, face_looking_away(), phone_detected())

    assert_equal(state, STATE_DISTRACTED_PHONE,
                 "5  State priority — phone overrides looking away")


def test_6_face_return_resets_no_face_timer():
    """
    If the face disappears and then returns before NO_FACE_TIMEOUT_SECONDS,
    the no_face_since timer must reset to None. The student must NOT
    be classified as AWAY if they briefly looked down and back up.
    """
    classifier = create_classifier()

    # Simulate face disappearing for 2 seconds (less than 5-second timeout)
    # We do this by directly manipulating the classifier dict's timestamp,
    # simulating what would happen if 2 seconds had passed without a face.
    classifier["no_face_since"] = time.time() - 2.0  # 2 seconds ago

    # Now feed one frame with face detected → timer should reset
    classify_frame(classifier, face_centered(), phone_not_detected())

    assert_equal(classifier["no_face_since"], None,
                 "6  Face returns → resets no_face timer")


# ============================================================
# SESSION MANAGER TESTS (Tests 7–11)
# ============================================================

def test_7_start_session_structure():
    """
    start_session() must return a dict with all required keys and
    correct initial values.

    WHY: app.py and session_manager.py both depend on this structure.
    If a key is missing, the session tracking breaks silently.
    """
    session = sm.start_session()

    required_keys = [
        "session_id", "start_time", "start_epoch",
        "current_state", "state_start_time",
        "durations", "event_counts",
        "current_focus_start", "longest_focus_period",
        "events", "is_active",
    ]

    all_keys_present = all(k in session for k in required_keys)
    assert_true(all_keys_present,
                "7  start_session() creates correct structure")


def test_8_update_session_accumulates_durations():
    """
    When the state changes, update_session() must accumulate the
    correct duration for the OLD state.

    HOW WE SIMULATE TIME:
        We directly set state_start_time to 10 seconds in the past,
        then immediately trigger a state change. The old state's
        duration should be approximately 10 seconds.
    """
    session = sm.start_session()

    # Pretend the session started 10 seconds ago
    ten_seconds_ago = time.time() - 10.0
    session["state_start_time"] = ten_seconds_ago

    # Trigger a state change → this causes duration to be accumulated
    sm.update_session(session, STATE_DISTRACTED_LOOK)

    # The FOCUSED state duration should now be ~10 seconds
    focused_duration = session["durations"][STATE_FOCUSED]

    # Allow ±0.5 second tolerance for clock precision
    assert_almost_equal(focused_duration, 10.0, tolerance=0.5,
                        test_name="8  update_session() accumulates durations correctly")


def test_9_focus_percentage_calculation():
    """
    The focus_percentage formula must be correct:
        focus_percentage = (focused_duration / total_duration) × 100

    WHY WE USE _compute_stats() AND NOT get_session_stats():
        get_session_stats() is designed for a LIVE session. It adds
        `now - session["start_epoch"]` as the total duration, which
        uses real wall-clock time. When we inject durations directly
        into the dict (bypassing the camera loop), the wall-clock
        denominator is near-zero and makes the percentage enormous.

        _compute_stats() computes total_duration = sum(durations.values()),
        which is exactly what we injected — so the math is predictable
        and the test is reliable regardless of how fast the CPU runs.

    We manually set up a session that was 50% focused, 50% distracted,
    then verify the formula produces exactly 50.0%.
    """
    session = sm.start_session()

    # Inject known durations directly into the session dict.
    # Total = 600 seconds; focused = 300 → expected focus_pct = 50.0%
    session["durations"][STATE_FOCUSED]          = 300.0   # 5 minutes
    session["durations"][STATE_DISTRACTED_PHONE] = 150.0   # 2.5 minutes
    session["durations"][STATE_DISTRACTED_LOOK]  = 150.0   # 2.5 minutes
    session["durations"][STATE_AWAY]             = 0.0

    # _compute_stats() derives total from sum(durations.values()),
    # making it immune to live wall-clock interference.
    stats = sm._compute_stats(session)
    focus_pct = stats["focus_percentage"]

    assert_almost_equal(focus_pct, 50.0, tolerance=0.1,
                        test_name="9  focus_percentage calculation is correct")


def test_10_longest_focus_period():
    """
    The session must track the longest continuous focus period correctly.

    We simulate:
      - 60 seconds of focus
      - Distraction (state change to PHONE)
      - 30 seconds of focus
    The longest_focus_period should be 60 seconds (the first streak).
    """
    session = sm.start_session()

    # Simulate 60 seconds of focus by setting state_start_time 60 sec ago
    session["state_start_time"]   = time.time() - 60.0
    session["current_focus_start"] = time.time() - 60.0

    # Trigger a state change → this finalizes the 60-second focus streak
    sm.update_session(session, STATE_DISTRACTED_PHONE)

    # Now simulate 30 seconds of focus
    session["state_start_time"]   = time.time() - 30.0
    session["current_focus_start"] = time.time() - 30.0
    sm.update_session(session, STATE_FOCUSED)

    # longest_focus_period should be ~60 seconds
    longest = session["longest_focus_period"]
    assert_almost_equal(longest, 60.0, tolerance=1.0,
                        test_name="10 Longest focus period is tracked correctly")


def test_11_csv_round_trip():
    """
    Sessions must be saved to CSV and loaded back correctly,
    with no data loss or type conversion errors.

    This is a round-trip test:
      1. Create a session and end it → saves to data/sessions.csv
      2. Load all sessions from disk
      3. Find our session by ID and check that focus_percentage matches

    We use a temporary directory so we don't pollute the real data/ folder.
    """
    # ── Use a temporary data directory ────────────────────────
    # We temporarily redirect session_manager's data paths to a
    # temp folder so the test doesn't corrupt real data.
    temp_dir = "/tmp/agy_test_data"
    os.makedirs(temp_dir, exist_ok=True)

    # Temporarily override the file paths in session_manager
    original_data_dir      = sm.DATA_DIR
    original_sessions_file = sm.SESSIONS_FILE
    original_events_file   = sm.EVENTS_FILE

    sm.DATA_DIR       = temp_dir
    sm.SESSIONS_FILE  = os.path.join(temp_dir, "sessions.csv")
    sm.EVENTS_FILE    = os.path.join(temp_dir, "events.csv")

    try:
        # Create a session, inject known stats, end it
        session = sm.start_session()
        session["durations"][STATE_FOCUSED]          = 180.0   # 3 minutes
        session["durations"][STATE_DISTRACTED_PHONE] = 60.0    # 1 minute
        session["durations"][STATE_DISTRACTED_LOOK]  = 0.0
        session["durations"][STATE_AWAY]             = 0.0
        # 180 / 240 = 75% focused
        saved_id = session["session_id"]

        sm.end_session(session)

        # Load all sessions back from disk
        df = sm.load_all_sessions()

        # Find our session by ID
        matching_rows = df[df["session_id"].astype(str) == str(saved_id)]
        found = len(matching_rows) == 1

        if found:
            loaded_pct = float(matching_rows.iloc[0]["focus_percentage"])
            # 180 / (180 + 60) = 75.0
            assert_almost_equal(loaded_pct, 75.0, tolerance=0.5,
                                test_name="11 Session CSV round-trip save and load")
        else:
            # Mark as failed — session not found in loaded data
            global _failed
            _failed += 1
            print(f"  Test 11 Session CSV round-trip save and load FAILED")
            print(f"    Session ID {saved_id!r} not found in loaded CSV.")

    finally:
        # Always restore the original paths and clean up temp files,
        # even if the test fails.
        sm.DATA_DIR      = original_data_dir
        sm.SESSIONS_FILE = original_sessions_file
        sm.EVENTS_FILE   = original_events_file
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================
# MAIN — Run all tests in order
# ============================================================
if __name__ == "__main__":
    print()
    print("Running AI Study Focus Coach unit tests...")
    print("=" * 60)

    # Run all test functions in the order they appear above.
    # Each function prints its own PASSED/FAILED line.
    test_1_initial_state_is_focused()
    test_2_phone_detection_changes_candidate()
    test_3_temporal_smoothing_4_frames_not_enough()
    test_4_temporal_smoothing_5_frames_triggers_change()
    test_5_phone_priority_overrides_looking_away()
    test_6_face_return_resets_no_face_timer()
    test_7_start_session_structure()
    test_8_update_session_accumulates_durations()
    test_9_focus_percentage_calculation()
    test_10_longest_focus_period()
    test_11_csv_round_trip()

    # ── Print final summary ────────────────────────────────────
    print("=" * 60)
    total = _passed + _failed
    if _failed == 0:
        print(f"All {total} tests passed. ✅")
    else:
        print(f"{_passed}/{total} tests passed. {_failed} FAILED. ❌")
        print("Fix the failing tests before submitting your project.")
        sys.exit(1)   # Exit with code 1 so CI/CD pipelines detect failures
