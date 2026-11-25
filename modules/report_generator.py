# ============================================================
# modules/report_generator.py
# ============================================================
#
# PURPOSE:
#   Generate a personalized study report from session statistics.
#   Two paths exist:
#     1. AI report  — calls Google Gemini Flash and returns rich text.
#     2. Fallback   — pure Python template, works without any API key.
#
#   Reports are saved to data/reports.json so they are not
#   regenerated every time the page is refreshed.
#
# PUBLIC FUNCTIONS:
#   get_api_key()                       → str or None
#   generate_report(session_stats)      → str   (tries Gemini first)
#   generate_fallback_report(session_stats) → str  (no API, local only)
#   save_report(session_id, text, is_ai) → None
#   load_report(session_id)             → dict or None
#
# HOW IT INTERACTS WITH OTHER MODULES:
#   - app.py calls generate_report() when the user clicks the button.
#   - app.py calls save_report() after generation, then load_report()
#     on subsequent reruns so the report is not regenerated.
#   - session_manager.py provides the session_stats dict (same keys
#     as sessions.csv rows) that is passed here.
#
# WHAT IS AI vs WHAT IS TEMPLATE:
#   generate_report()          → AI (Gemini Flash)
#   generate_fallback_report() → pure Python string formatting
# ============================================================

import os
import json
from datetime import datetime

# ── Path to the reports storage file ──────────────────────────
# This constant must match the one in session_manager.py.
REPORTS_FILE = os.path.join("data", "reports.json")


# ============================================================
# HELPER: get_api_key
# ============================================================
def get_api_key() -> str:
    """
    Read the Gemini API key from Streamlit secrets or an environment variable.

    We try two sources in this order:
      1. st.secrets["GEMINI_API_KEY"]  — set in .streamlit/secrets.toml
      2. os.environ["GEMINI_API_KEY"]  — set in a .env file or shell

    Returns:
        The API key string if found and non-empty, or None otherwise.

    Why not hardcode the key?
        Hardcoding API keys is a security risk. If the code is ever
        uploaded to GitHub, the key would be exposed publicly and
        could be misused (or billed to your account).
    """
    # Try Streamlit secrets first (preferred when running via streamlit run)
    try:
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key and key.strip() and key.strip() != "your_gemini_api_key_here":
            return key.strip()
    except Exception:
        pass  # st.secrets is not available outside a Streamlit session

    # Fall back to environment variable (useful for testing from a terminal)
    env_key = os.environ.get("GEMINI_API_KEY", "")
    if env_key and env_key.strip() and env_key.strip() != "your_gemini_api_key_here":
        return env_key.strip()

    return None  # No key found


# ============================================================
# HELPER: _build_prompt
# ============================================================
def _build_prompt(stats: dict) -> str:
    """
    Build the text prompt that is sent to Gemini.

    We include every important statistic so Gemini has full context.
    The prompt asks for 7 specific sections — this keeps the output
    structured so app.py can display it section by section.

    Args:
        stats: The session stats dict from session_manager.end_session()
               or session_manager.get_session_stats().

    Returns:
        A multi-line string that is the complete Gemini prompt.
    """
    # ── Format durations as human-readable strings ─────────────
    total_min  = round(stats.get("total_duration", 0) / 60, 1)
    focus_min  = round(stats.get("focused_duration", 0) / 60, 1)
    dist_min   = round(stats.get("distracted_duration", 0) / 60, 1)
    away_min   = round(stats.get("away_duration", 0) / 60, 1)
    focus_pct  = stats.get("focus_percentage", 0)
    phone_evt  = stats.get("phone_events", 0)
    look_evt   = stats.get("look_away_events", 0)
    away_evt   = stats.get("away_events", 0)
    longest_fp = round(stats.get("longest_focus_period", 0) / 60, 1)

    prompt = f"""You are a friendly, supportive AI study coach. A student has just completed a study session and their attention monitoring data is shown below. Write a warm, honest, and actionable study report for them.

SESSION DATA:
- Total session duration: {total_min} minutes
- Time spent focused:     {focus_min} minutes ({focus_pct:.1f}% of session)
- Time distracted:        {dist_min} minutes
- Time away from desk:    {away_min} minutes
- Phone distractions:     {phone_evt} times
- Times looking away:     {look_evt} times
- Times away from desk:   {away_evt} times
- Longest uninterrupted focus period: {longest_fp} minutes

Write the report in EXACTLY 7 clearly labeled sections. Use this exact format for each section header:

## 1. Overall Session Assessment
## 2. Focus Performance
## 3. Major Distraction Patterns
## 4. Positive Study Habits
## 5. Areas Requiring Improvement
## 6. Personalized Recommendations
## 7. Suggested Strategy for the Next Study Session

Guidelines:
- Use plain conversational language. The student is in college.
- Be honest but encouraging. Avoid harsh criticism.
- Keep each section 2-4 sentences unless more detail is genuinely helpful.
- In section 6, give 3-5 specific, actionable recommendations based on the data.
- In section 7, describe one concrete study strategy for the next session.
- Do NOT repeat the raw numbers in every section — analyse them, don't just restate them.
- Do NOT use markdown bold (**text**) in the final output — use plain text only.
"""
    return prompt


# ============================================================
# PUBLIC: generate_report
# ============================================================
def generate_report(session_stats: dict) -> tuple:
    """
    Generate a personalized study report using Google Gemini Flash.

    If the API key is missing or the API call fails for any reason,
    automatically falls back to generate_fallback_report().

    Args:
        session_stats: Dict from session_manager.end_session() or
                       session_manager.get_session_stats().
                       Required keys: total_duration, focused_duration,
                       distracted_duration, away_duration, focus_percentage,
                       phone_events, look_away_events, away_events,
                       longest_focus_period.

    Returns:
        A tuple (report_text: str, is_ai: bool).
        is_ai is True if Gemini was used, False if the fallback was used.
    """
    api_key = get_api_key()

    if api_key is None:
        # No key → use the template fallback immediately
        print("[report_generator] No API key found. Using fallback report.")
        return generate_fallback_report(session_stats), False

    try:
        # ── Import the google-genai SDK ────────────────────────
        # The SDK is installed via: pip install google-genai
        # We import inside the function so the app still loads
        # even if google-genai is not installed (the fallback works).
        from google import genai

        # ── Create a Gemini client using the API key ───────────
        client = genai.Client(api_key=api_key)

        # ── Choose the model ───────────────────────────────────
        # Check Streamlit secrets, then environment variable, with
        # robust fallback to currently active models.
        configured_model = None
        try:
            import streamlit as st
            configured_model = st.secrets.get("GEMINI_MODEL")
        except Exception:
            pass
        if not configured_model:
            configured_model = os.environ.get("GEMINI_MODEL")

        # Models to try in order of preference
        models_to_try = [
            configured_model,
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash",
        ]
        # Remove duplicates while preserving order
        models_to_try = [m for m in dict.fromkeys(models_to_try) if m]

        # ── Build the prompt and call the API ─────────────────
        prompt = _build_prompt(session_stats)
        last_error = None

        for model_name in models_to_try:
            try:
                print(f"[report_generator] Calling Gemini model: {model_name}")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                report_text = response.text.strip() if response.text else ""
                if report_text:
                    print(f"[report_generator] Gemini report generated successfully with {model_name}.")
                    return report_text, True
            except Exception as model_err:
                print(f"[report_generator] Model {model_name} failed: {model_err}")
                last_error = model_err
                continue

        if last_error:
            raise last_error

        raise ValueError("Gemini returned an empty response.")

    except Exception as e:
        # Any error (network, quota, bad key, etc.) → fallback
        print(f"[report_generator] Gemini call failed: {e}. Using fallback.")
        fallback_text = generate_fallback_report(session_stats)
        # Prepend an honest note so the student knows what happened
        err_msg = str(e).split("\n")[0][:100]
        note = f"[Note: AI report generation failed ({type(e).__name__}: {err_msg}). This is a locally generated fallback report.]\n\n"
        return note + fallback_text, False


# ============================================================
# PUBLIC: generate_fallback_report
# ============================================================
def generate_fallback_report(session_stats: dict) -> str:
    """
    Generate a template-based study report without any external API.

    This function uses only Python string formatting. It reads the
    session stats and fills in pre-written sentences. The output
    is clearly labeled as a local fallback report.

    Args:
        session_stats: Same dict as generate_report() expects.

    Returns:
        A multi-section report string in the same format as the
        Gemini report, so app.py can display both identically.
    """
    # ── Extract stats ──────────────────────────────────────────
    total_sec  = stats.get("total_duration", 0)   if (stats := session_stats) else 0
    focus_sec  = stats.get("focused_duration", 0)
    dist_sec   = stats.get("distracted_duration", 0)
    away_sec   = stats.get("away_duration", 0)
    focus_pct  = stats.get("focus_percentage", 0)
    phone_evt  = int(stats.get("phone_events", 0))
    look_evt   = int(stats.get("look_away_events", 0))
    away_evt   = int(stats.get("away_events", 0))
    longest_fp = stats.get("longest_focus_period", 0)

    total_min  = round(total_sec / 60, 1)
    focus_min  = round(focus_sec / 60, 1)
    longest_min = round(longest_fp / 60, 1)
    total_distractions = phone_evt + look_evt + away_evt

    # ── Determine performance level ────────────────────────────
    # These thresholds are informal — they guide the tone of the report.
    if focus_pct >= 80:
        performance = "excellent"
        overall_tone = "You maintained strong focus throughout this session."
    elif focus_pct >= 60:
        performance = "good"
        overall_tone = "You had a solid session with some room for improvement."
    elif focus_pct >= 40:
        performance = "moderate"
        overall_tone = "Your focus was mixed — there were good stretches but also notable distractions."
    else:
        performance = "below average"
        overall_tone = "This was a challenging session. The data shows significant distraction — but awareness is the first step to improvement."

    # ── Phone distraction commentary ──────────────────────────
    if phone_evt == 0:
        phone_comment = "Excellent news: your phone did not appear in the camera frame during this session."
    elif phone_evt <= 2:
        phone_comment = f"Your phone was detected {phone_evt} time(s). This is relatively low — keep it up."
    else:
        phone_comment = f"Your phone was detected {phone_evt} times, which is the main source of distraction to address."

    # ── Gaze commentary ───────────────────────────────────────
    if look_evt == 0:
        gaze_comment = "Your gaze stayed toward the screen consistently."
    elif look_evt <= 5:
        gaze_comment = f"You looked away from the screen {look_evt} time(s), which is manageable."
    else:
        gaze_comment = f"You looked away from the screen {look_evt} times. Consider what might be drawing your attention elsewhere."

    # ── Recommendations ───────────────────────────────────────
    recommendations = []
    if phone_evt > 0:
        recommendations.append("Put your phone in another room or in a drawer before studying. Out of sight, out of mind.")
    if look_evt > 5:
        recommendations.append("Position your study materials directly in front of you so your natural gaze stays on them.")
    if away_evt > 2:
        recommendations.append("Use scheduled short breaks (e.g., 5 minutes every 25 minutes) so you don't feel the urge to leave unexpectedly.")
    if longest_min < 10:
        recommendations.append("Work on building longer uninterrupted focus periods. Start with 10-minute goals, then extend to 25 minutes.")
    if focus_pct >= 70:
        recommendations.append("Your focus is good. Try extending session length by 5 minutes each day to build study stamina.")
    if not recommendations:
        recommendations.append("Keep maintaining your current study habits — the data looks great!")

    rec_text = "\n".join(f"  - {r}" for r in recommendations)

    # ── Assemble the report ────────────────────────────────────
    report = f"""[This is a locally generated fallback report. It was produced using Python string templates, not an AI model. To get an AI-generated report, add your Gemini API key to .streamlit/secrets.toml or a .env file.]

## 1. Overall Session Assessment
You studied for {total_min} minutes with a focus score of {focus_pct:.1f}%. {overall_tone} Your performance this session is classified as {performance}.

## 2. Focus Performance
You were focused for {focus_min} out of {total_min} minutes. Your longest uninterrupted focus period was {longest_min} minutes. {'This shows you are capable of sustained concentration — aim to make this your typical period length.' if longest_min >= 10 else 'Short focus periods suggest frequent interruptions. Try minimizing external triggers before you start studying.'}

## 3. Major Distraction Patterns
You experienced {total_distractions} distraction event(s) in total. {phone_comment} {gaze_comment} {'You left your desk ' + str(away_evt) + ' time(s), which breaks momentum.' if away_evt > 0 else 'You stayed at your desk throughout the session, which is great.'}

## 4. Positive Study Habits
{'You maintained a good focus ratio, which shows discipline and effective study habits.' if focus_pct >= 60 else 'You committed to sitting down and studying, which is the most important first step.'} {'Your phone was not a major distraction this session.' if phone_evt <= 1 else ''} Completing the full session without giving up shows persistence.

## 5. Areas Requiring Improvement
{'Phone usage is the primary area to work on.' if phone_evt >= 3 else 'Focus duration could be extended with practice.'} {'Looking away from the screen frequently suggests external distractions in your study environment.' if look_evt > 5 else ''} {'Aim to reduce desk departures to planned breaks only.' if away_evt > 2 else ''}

## 6. Personalized Recommendations
Based on your session data, here are specific actions to try:
{rec_text}

## 7. Suggested Strategy for the Next Study Session
Try the Pomodoro Technique: study for 25 minutes without any distractions, then take a 5-minute break. Before starting, silence your phone and place it face-down (or in another room). Set a clear goal for what you want to accomplish in the session. After the session ends, review this report and compare your focus score to today's.
"""
    return report.strip()


# ============================================================
# PUBLIC: save_report
# ============================================================
def save_report(session_id: str, report_text: str, is_ai: bool):
    """
    Save a generated report to data/reports.json.

    The file is a JSON object (dict) where each key is a session_id
    and the value is a dict with the report text and metadata.

    Structure of reports.json:
    {
        "session-uuid-1": {
            "text": "...",
            "is_ai": true,
            "generated_at": "2024-01-15T14:32:00"
        },
        "session-uuid-2": { ... }
    }

    Args:
        session_id:  The unique ID of the session (from session_manager).
        report_text: The full report string to save.
        is_ai:       True if Gemini generated it, False if fallback.
    """
    os.makedirs("data", exist_ok=True)

    # Load existing reports (or start with an empty dict)
    existing_reports = {}
    if os.path.exists(REPORTS_FILE):
        try:
            with open(REPORTS_FILE, "r", encoding="utf-8") as f:
                existing_reports = json.load(f)
        except (json.JSONDecodeError, IOError):
            # Corrupted file — start fresh rather than crashing
            existing_reports = {}

    # Add or overwrite the report for this session
    existing_reports[session_id] = {
        "text":         report_text,
        "is_ai":        is_ai,
        "generated_at": datetime.now().isoformat(),
    }

    # Write the updated dict back to the file
    with open(REPORTS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_reports, f, indent=2, ensure_ascii=False)

    print(f"[report_generator] Report saved for session {session_id} (is_ai={is_ai})")


# ============================================================
# PUBLIC: load_report
# ============================================================
def load_report(session_id: str) -> dict:
    """
    Load a previously saved report from data/reports.json.

    Args:
        session_id: The session ID whose report we want to retrieve.

    Returns:
        A dict with keys: "text", "is_ai", "generated_at"
        — or None if no report has been saved for this session.
    """
    if not os.path.exists(REPORTS_FILE):
        return None

    try:
        with open(REPORTS_FILE, "r", encoding="utf-8") as f:
            all_reports = json.load(f)
        return all_reports.get(session_id, None)
    except (json.JSONDecodeError, IOError):
        return None
