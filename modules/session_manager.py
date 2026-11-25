# ============================================================
# modules/session_manager.py
# ============================================================
#
# PURPOSE:
#   This module manages the lifecycle of a study session:
#   starting it, tracking time in each behavioral state,
#   logging state transitions, computing final statistics,
#   and saving everything to local CSV/JSON files.
#
# SESSION LIFECYCLE:
#   1. User clicks "Start Session"
#      → start_session() creates a new session dict
#   2. Camera loop calls update_session() on every frame
#      → tracks time in each state, logs transitions
#   3. User clicks "End Session"
#      → end_session() computes final stats and saves to disk
#
# WHAT GETS SAVED:
#   data/sessions.csv  — One row per completed session (summary stats)
#   data/events.csv    — One row per state-change event (timeline)
#   data/reports.json  — AI-generated reports keyed by session_id (Phase 5)
#
# DATA INTEGRITY:
#   - State durations are computed from actual timestamps, not
#     frame counts. This means they are accurate even if the
#     frame rate varies during the session.
#   - The sum of all state durations equals total_duration
#     (guaranteed by computing the last state's duration at
#     session end using the end timestamp).
#
# PUBLIC FUNCTIONS:
#   start_session()                        → session dict
#   update_session(session, new_state)     → event dict or None
#   end_session(session)                   → stats dict
#   get_session_stats(session)             → live stats dict
#   load_all_sessions()                    → pd.DataFrame
#   load_session_events(session_id)        → pd.DataFrame
# ============================================================

import os
import time
import uuid
import json
from datetime import datetime

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# Directory where all session data files are stored.
# Created automatically on first save if it doesn't exist.
DATA_DIR        = "data"
SESSIONS_FILE   = os.path.join(DATA_DIR, "sessions.csv")
EVENTS_FILE     = os.path.join(DATA_DIR, "events.csv")
REPORTS_FILE    = os.path.join(DATA_DIR, "reports.json")

# Column names for sessions.csv — must match what app.py expects
SESSION_COLUMNS = [
    "session_id", "start_time", "end_time",
    "total_duration", "focused_duration", "distracted_duration",
    "away_duration", "focus_percentage",
    "phone_events", "look_away_events", "away_events",
    "longest_focus_period",
]

# Column names for events.csv
EVENT_COLUMNS = [
    "session_id", "timestamp", "previous_state", "new_state", "event_type",
]

# Import state constants so we use the same strings everywhere
from modules.attention_classifier import (
    STATE_FOCUSED,
    STATE_DISTRACTED_PHONE,
    STATE_DISTRACTED_LOOK,
    STATE_AWAY,
)


# ============================================================
# PUBLIC FUNCTION: start_session
# ============================================================
def start_session() -> dict:
    """
    Initialize a new study session.

    Call this when the user clicks "Start Session". It creates
    a session dictionary that tracks all the information needed
    to compute statistics when the session ends.

    HOW SESSION IDs WORK:
        We use uuid.uuid4() to generate a universally unique
        identifier. This is a 128-bit random number formatted
        as a string like "a3b2c1d4-e5f6-7890-abcd-ef1234567890".
        The chance of two UUIDs colliding is astronomically small
        (1 in 2^122), so we never need to check for duplicates.

    Returns:
        A dict containing all session tracking state.
    """
    now = time.time()
    session_id = str(uuid.uuid4())[:8]  # First 8 chars = short enough to display

    return {
        # ── Identity ──────────────────────────────────────────
        "session_id":   session_id,
        "start_time":   datetime.now().isoformat(),   # ISO 8601 string
        "start_epoch":  now,                          # Unix timestamp (float)

        # ── Current state tracking ────────────────────────────
        # The last known behavioral state. Starts as FOCUSED.
        "current_state":      STATE_FOCUSED,
        # When the current state began (epoch timestamp).
        "state_start_time":   now,

        # ── Accumulated durations (seconds) ───────────────────
        # Each key tracks total seconds spent in that state.
        # Updated every time the state changes.
        "durations": {
            STATE_FOCUSED:          0.0,
            STATE_DISTRACTED_PHONE: 0.0,
            STATE_DISTRACTED_LOOK:  0.0,
            STATE_AWAY:             0.0,
        },

        # ── Event counters ────────────────────────────────────
        # Count how many SEPARATE distraction events occurred.
        # A "separate event" means: the state changed TO this
        # distraction from a different state. Staying in the
        # same distraction state continuously counts as ONE event.
        "event_counts": {
            STATE_DISTRACTED_PHONE: 0,
            STATE_DISTRACTED_LOOK:  0,
            STATE_AWAY:             0,
        },

        # ── Focus period tracking ─────────────────────────────
        # Used to find the longest continuous focus period.
        "current_focus_start": now,  # When the current focus streak started
        "longest_focus_period": 0.0, # Longest focus streak so far (seconds)

        # ── Event log ─────────────────────────────────────────
        # List of all state-change events, stored as dicts.
        # Each event: {timestamp, previous_state, new_state, event_type}
        "events": [],

        # ── Session status ────────────────────────────────────
        "is_active": True,
    }


# ============================================================
# PUBLIC FUNCTION: update_session
# ============================================================
def update_session(session: dict, new_state: str) -> dict | None:
    """
    Update the session with the latest classified behavioral state.

    Called by camera.py on every processed frame. This function:
      1. Checks if the state has changed since the last call.
      2. If YES: accumulates the duration of the old state,
         logs the transition event, updates counters.
      3. If NO: does nothing (staying in the same state is not a new event).

    WHY TIMESTAMP-BASED DURATION?
        We compute duration = now - state_start_time, which uses
        wall-clock time. This is more accurate than counting frames
        because the frame rate may vary (e.g. if the CPU is busy,
        YOLO might slow down temporarily). Wall-clock time is
        always accurate regardless of processing speed.

    Args:
        session:   The session dict from start_session().
        new_state: The latest state string from classify_frame().

    Returns:
        An event dict if the state changed, or None if it didn't.
        The event dict contains: timestamp, previous_state, new_state, event_type.
    """
    if not session.get("is_active", False):
        return None

    now = time.time()
    old_state = session["current_state"]

    # ── No change — same state as before ──────────────────────
    if new_state == old_state:
        return None

    # ── State HAS changed — process the transition ────────────

    # Step 1: Calculate how long we spent in the OLD state
    state_duration = now - session["state_start_time"]

    # Step 2: Add that duration to the OLD state's accumulator
    session["durations"][old_state] += state_duration

    # Step 3: Track focus streaks
    # If we were FOCUSED and now we're leaving that state,
    # check if this was the longest focus period so far.
    if old_state == STATE_FOCUSED:
        focus_duration = now - session["current_focus_start"]
        if focus_duration > session["longest_focus_period"]:
            session["longest_focus_period"] = focus_duration

    # If we're ENTERING the focused state, start a new streak timer
    if new_state == STATE_FOCUSED:
        session["current_focus_start"] = now

    # Step 4: Count distraction EVENTS (not frames)
    # Only increment when we ENTER a distraction state.
    if new_state in session["event_counts"]:
        session["event_counts"][new_state] += 1

    # Step 5: Log the event
    # Determine a human-readable event type name
    event_type = _get_event_type(old_state, new_state)

    event = {
        "timestamp":      datetime.now().isoformat(),
        "previous_state": old_state,
        "new_state":      new_state,
        "event_type":     event_type,
    }
    session["events"].append(event)

    # Step 6: Update the session's current state
    session["current_state"]    = new_state
    session["state_start_time"] = now

    return event


# ============================================================
# PUBLIC FUNCTION: end_session
# ============================================================
def end_session(session: dict) -> dict:
    """
    Finalize the session, compute statistics, and save to disk.

    Call this when the user clicks "End Session". It:
      1. Accumulates the duration of the final state.
      2. Computes all summary statistics.
      3. Saves the session row to data/sessions.csv.
      4. Saves all events to data/events.csv.
      5. Returns the computed statistics for immediate display.

    Args:
        session: The session dict from start_session().

    Returns:
        A dict of computed statistics (same keys as sessions.csv columns).
    """
    if not session.get("is_active", False):
        # Already ended — return whatever stats we have
        return _compute_stats(session)

    now = time.time()

    # ── Accumulate the FINAL state's duration ──────────────────
    # The last state hasn't been accumulated yet because
    # update_session() only accumulates when the state CHANGES.
    # Since the session is ending, we need to close out the
    # current state's timer.
    final_duration = now - session["state_start_time"]
    session["durations"][session["current_state"]] += final_duration

    # ── Check if the final state was a focus streak ────────────
    if session["current_state"] == STATE_FOCUSED:
        focus_duration = now - session["current_focus_start"]
        if focus_duration > session["longest_focus_period"]:
            session["longest_focus_period"] = focus_duration

    # ── Mark session as inactive ───────────────────────────────
    session["is_active"] = False
    session["end_time"]  = datetime.now().isoformat()
    session["end_epoch"] = now

    # ── Compute final statistics ───────────────────────────────
    stats = _compute_stats(session)

    # ── Save to disk ───────────────────────────────────────────
    _save_session(stats)
    _save_events(session["session_id"], session["events"])

    return stats


# ============================================================
# PUBLIC FUNCTION: get_session_stats
# ============================================================
def get_session_stats(session: dict) -> dict:
    """
    Get LIVE statistics for a currently running session.

    Unlike end_session(), this does NOT finalize anything.
    It computes stats including the time spent in the current
    (still-active) state, so the numbers are always up-to-date.

    Used by app.py to display live focus percentage, timer, etc.

    Args:
        session: The session dict.

    Returns:
        A dict with keys: total_duration, focused_duration,
        distracted_duration, away_duration, focus_percentage,
        phone_events, look_away_events, away_events,
        longest_focus_period, current_state.
    """
    now = time.time()

    # Start with the accumulated durations
    durations = dict(session["durations"])

    # Add the time spent in the CURRENT state (not yet accumulated)
    current_state_time = now - session["state_start_time"]
    durations[session["current_state"]] += current_state_time

    # Total elapsed time
    total_duration = now - session["start_epoch"]

    # Focused duration
    focused_dur = durations.get(STATE_FOCUSED, 0.0)

    # Distracted duration = phone + looking away
    distracted_dur = (
        durations.get(STATE_DISTRACTED_PHONE, 0.0) +
        durations.get(STATE_DISTRACTED_LOOK, 0.0)
    )

    # Away duration
    away_dur = durations.get(STATE_AWAY, 0.0)

    # Focus percentage
    if total_duration > 0:
        focus_pct = (focused_dur / total_duration) * 100.0
    else:
        focus_pct = 100.0

    # Longest focus period — check if current streak beats the record
    longest = session["longest_focus_period"]
    if session["current_state"] == STATE_FOCUSED:
        current_streak = now - session["current_focus_start"]
        if current_streak > longest:
            longest = current_streak

    return {
        "total_duration":       round(total_duration, 1),
        "focused_duration":     round(focused_dur, 1),
        "distracted_duration":  round(distracted_dur, 1),
        "away_duration":        round(away_dur, 1),
        "focus_percentage":     round(focus_pct, 1),
        "phone_events":         session["event_counts"].get(STATE_DISTRACTED_PHONE, 0),
        "look_away_events":     session["event_counts"].get(STATE_DISTRACTED_LOOK, 0),
        "away_events":          session["event_counts"].get(STATE_AWAY, 0),
        "longest_focus_period": round(longest, 1),
        "current_state":        session["current_state"],
    }


# ============================================================
# PUBLIC FUNCTION: load_all_sessions
# ============================================================
def load_all_sessions() -> pd.DataFrame:
    """
    Load all completed sessions from disk.

    Returns:
        pd.DataFrame with one row per session, or empty DataFrame.
    """
    if not os.path.exists(SESSIONS_FILE):
        return pd.DataFrame(columns=SESSION_COLUMNS)

    try:
        return pd.read_csv(SESSIONS_FILE, dtype={"session_id": str})
    except Exception:
        return pd.DataFrame(columns=SESSION_COLUMNS)


# ============================================================
# PUBLIC FUNCTION: load_session_events
# ============================================================
def load_session_events(session_id: str) -> pd.DataFrame:
    """
    Load all state-change events for a specific session.

    Args:
        session_id: The session's unique identifier.

    Returns:
        pd.DataFrame with one row per event, filtered to the
        given session. Empty DataFrame if no events found.
    """
    if not os.path.exists(EVENTS_FILE):
        return pd.DataFrame(columns=EVENT_COLUMNS)

    try:
        df = pd.read_csv(EVENTS_FILE, dtype={"session_id": str})
        return df[df["session_id"] == str(session_id)].reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=EVENT_COLUMNS)


# ============================================================
# PRIVATE: _compute_stats
# ============================================================
def _compute_stats(session: dict) -> dict:
    """
    Compute the final summary statistics for a session.

    Returns a dict whose keys exactly match SESSION_COLUMNS,
    so it can be directly appended to sessions.csv.
    """
    durations = session["durations"]

    total_duration = sum(durations.values())
    focused_dur    = durations.get(STATE_FOCUSED, 0.0)
    distracted_dur = (
        durations.get(STATE_DISTRACTED_PHONE, 0.0) +
        durations.get(STATE_DISTRACTED_LOOK, 0.0)
    )
    away_dur = durations.get(STATE_AWAY, 0.0)

    if total_duration > 0:
        focus_pct = (focused_dur / total_duration) * 100.0
    else:
        focus_pct = 100.0

    return {
        "session_id":           session["session_id"],
        "start_time":           session["start_time"],
        "end_time":             session.get("end_time", datetime.now().isoformat()),
        "total_duration":       round(total_duration, 1),
        "focused_duration":     round(focused_dur, 1),
        "distracted_duration":  round(distracted_dur, 1),
        "away_duration":        round(away_dur, 1),
        "focus_percentage":     round(focus_pct, 1),
        "phone_events":         session["event_counts"].get(STATE_DISTRACTED_PHONE, 0),
        "look_away_events":     session["event_counts"].get(STATE_DISTRACTED_LOOK, 0),
        "away_events":          session["event_counts"].get(STATE_AWAY, 0),
        "longest_focus_period": round(session["longest_focus_period"], 1),
    }


# ============================================================
# PRIVATE: _save_session
# ============================================================
def _save_session(stats: dict):
    """
    Append one session row to data/sessions.csv.

    Creates the data/ directory and CSV file if they don't exist.
    Uses pandas to ensure consistent column ordering.
    """
    # Create data directory if needed
    os.makedirs(DATA_DIR, exist_ok=True)

    # Create a one-row DataFrame from the stats dict
    new_row = pd.DataFrame([stats], columns=SESSION_COLUMNS)

    if os.path.exists(SESSIONS_FILE):
        # Append without writing the header again
        new_row.to_csv(SESSIONS_FILE, mode="a", header=False, index=False)
    else:
        # First session — write the header
        new_row.to_csv(SESSIONS_FILE, mode="w", header=True, index=False)

    print(f"[session_manager] Session {stats['session_id']} saved to {SESSIONS_FILE}")


# ============================================================
# PRIVATE: _save_events
# ============================================================
def _save_events(session_id: str, events: list):
    """
    Append all events from one session to data/events.csv.

    Args:
        session_id: The session's unique ID.
        events:     List of event dicts from the session.
    """
    if not events:
        return  # Nothing to save

    os.makedirs(DATA_DIR, exist_ok=True)

    # Build a list of rows with the session_id prepended
    rows = []
    for event in events:
        rows.append({
            "session_id":     session_id,
            "timestamp":      event["timestamp"],
            "previous_state": event["previous_state"],
            "new_state":      event["new_state"],
            "event_type":     event["event_type"],
        })

    events_df = pd.DataFrame(rows, columns=EVENT_COLUMNS)

    if os.path.exists(EVENTS_FILE):
        events_df.to_csv(EVENTS_FILE, mode="a", header=False, index=False)
    else:
        events_df.to_csv(EVENTS_FILE, mode="w", header=True, index=False)

    print(f"[session_manager] {len(events)} events saved for session {session_id}")


# ============================================================
# PRIVATE: _get_event_type
# ============================================================
def _get_event_type(old_state: str, new_state: str) -> str:
    """
    Generate a human-readable event type string.

    Examples:
        "focused → phone distraction"
        "phone distraction → focused"
        "focused → looking away"

    Args:
        old_state: The state before the transition.
        new_state: The state after the transition.

    Returns:
        A short description of what happened.
    """
    # Short labels for each state
    labels = {
        STATE_FOCUSED:          "focused",
        STATE_DISTRACTED_PHONE: "phone distraction",
        STATE_DISTRACTED_LOOK:  "looking away",
        STATE_AWAY:             "away from desk",
    }

    old_label = labels.get(old_state, old_state.lower())
    new_label = labels.get(new_state, new_state.lower())

    return f"{old_label} → {new_label}"
