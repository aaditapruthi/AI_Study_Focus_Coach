# ============================================================
# modules/analytics.py
# ============================================================
#
# PURPOSE:
#   Build the three Plotly charts shown on the Session Analytics page.
#
#   Three charts are created here:
#     1. Focus Donut Chart     — how time was split between states
#     2. Behavioral Timeline   — when each state occurred over time
#     3. Distraction Bar Chart — how many events of each type occurred
#
#   Every function returns a Plotly Figure object that app.py
#   displays using st.plotly_chart(fig, use_container_width=True).
#
# WHY PLOTLY?
#   Plotly produces interactive charts (hover tooltips, zoom, pan)
#   that look professional and work directly inside Streamlit with
#   a single function call. We do not need to write any JavaScript.
#
# PUBLIC FUNCTIONS:
#   make_focus_donut(session_stats)         → plotly Figure
#   make_state_timeline(events_df, stats)   → plotly Figure
#   make_distraction_bar(session_stats)     → plotly Figure
#   make_multi_session_trend(sessions_df)   → plotly Figure
#
# HOW IT INTERACTS WITH OTHER MODULES:
#   - app.py imports this module and passes stats / DataFrames to it.
#   - session_stats comes from session_manager._compute_stats() or
#     from a row of sessions.csv read as a dict.
#   - events_df comes from session_manager.load_session_events().
#
# COLOR PALETTE (must match the UI CSS):
#   Focused         → #426B54  (green)
#   Distracted-Phone→ #D87568  (coral/red)
#   Looking Away    → #D69B43  (amber)
#   Away from Desk  → #ABABAB  (grey)
#   Background      → #F7F7F2  (off-white)
#   Card background → #FFFFFF
# ============================================================

import pandas as pd
import plotly.graph_objects as go


# ============================================================
# SHARED CONSTANTS
# ============================================================

# State string constants — must match attention_classifier.py exactly.
# We define them here again (instead of importing) to avoid a circular
# dependency between analytics.py and attention_classifier.py.
STATE_FOCUSED          = "FOCUSED"
STATE_DISTRACTED_PHONE = "DISTRACTED - PHONE"
STATE_DISTRACTED_LOOK  = "DISTRACTED - LOOKING AWAY"
STATE_AWAY             = "AWAY FROM DESK"

# Color for each state — consistent across all three charts
STATE_COLORS = {
    STATE_FOCUSED:          "#426B54",   # green
    STATE_DISTRACTED_PHONE: "#D87568",   # coral / red
    STATE_DISTRACTED_LOOK:  "#D69B43",   # amber / orange
    STATE_AWAY:             "#ABABAB",   # grey
}

# Short display labels for use in chart legends and axes
STATE_LABELS = {
    STATE_FOCUSED:          "Focused",
    STATE_DISTRACTED_PHONE: "Phone Distraction",
    STATE_DISTRACTED_LOOK:  "Looking Away",
    STATE_AWAY:             "Away from Desk",
}

# Base Plotly layout settings shared across all charts
_LAYOUT_BASE = dict(
    paper_bgcolor="#FFFFFF",   # chart card background
    plot_bgcolor="#F7F7F2",    # inner plot area background
    font=dict(family="Inter, Manrope, sans-serif", color="#20251F", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    showlegend=True,
)


# ============================================================
# CHART 1: make_focus_donut
# ============================================================
def make_focus_donut(session_stats: dict) -> go.Figure:
    """
    Create a donut chart showing the time breakdown across all four
    behavioral states for a single study session.

    HOW A DONUT CHART WORKS:
        A donut chart is a pie chart with a hole in the centre.
        Each slice represents the percentage of total session time
        spent in one state. The bigger the slice, the more time
        was spent in that state.
        We use a donut (not a pie) because the hole lets us display
        the overall focus score as a large number in the centre.

    Args:
        session_stats: Dict with keys:
            focused_duration, distracted_duration, away_duration,
            total_duration, focus_percentage.
            (All durations are in seconds.)

    Returns:
        A Plotly Figure object ready for st.plotly_chart().
    """
    # ── Extract durations in MINUTES for display ───────────────
    # We convert seconds → minutes so the hover tooltip shows
    # human-readable values like "12.3 min" instead of "738 sec".
    focused_min  = round(session_stats.get("focused_duration", 0) / 60, 1)
    phone_min    = round(
        # phone distraction is part of distracted_duration, but we
        # need it separately. We reconstruct it from the event counts
        # and total distracted_duration by assuming it was split
        # proportionally. Simpler: session_stats may have phone_duration
        # or we use distracted_duration split is unavailable.
        # For simplicity, we work with what sessions.csv stores:
        # distracted_duration = phone + look_away combined.
        # We will split it using event counts as a weight.
        session_stats.get("distracted_duration", 0) / 60 * _phone_fraction(session_stats),
        1
    )
    look_min     = round(
        session_stats.get("distracted_duration", 0) / 60 * _look_fraction(session_stats),
        1
    )
    away_min     = round(session_stats.get("away_duration", 0) / 60, 1)
    focus_pct    = session_stats.get("focus_percentage", 0)

    # ── Build parallel lists of labels, values, colors ─────────
    # Plotly needs three matching lists: one label per slice,
    # one numeric value per slice, one color per slice.
    labels = [
        STATE_LABELS[STATE_FOCUSED],
        STATE_LABELS[STATE_DISTRACTED_PHONE],
        STATE_LABELS[STATE_DISTRACTED_LOOK],
        STATE_LABELS[STATE_AWAY],
    ]
    values = [focused_min, phone_min, look_min, away_min]
    colors = [
        STATE_COLORS[STATE_FOCUSED],
        STATE_COLORS[STATE_DISTRACTED_PHONE],
        STATE_COLORS[STATE_DISTRACTED_LOOK],
        STATE_COLORS[STATE_AWAY],
    ]

    # ── Create the Plotly Figure ───────────────────────────────
    fig = go.Figure()

    # go.Pie with hole > 0 creates a donut chart.
    # textinfo="none" hides slice labels (we show a legend instead).
    # hovertemplate controls what text appears when you hover over a slice.
    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        hole=0.6,              # 60% hole in the centre = donut shape
        marker=dict(
            colors=colors,
            line=dict(color="#FFFFFF", width=2),  # white gaps between slices
        ),
        textinfo="none",       # don't clutter slices with text
        hovertemplate="<b>%{label}</b><br>%{value} min<br>%{percent}<extra></extra>",
        direction="clockwise",
        sort=False,            # keep slices in the order we defined above
    ))

    # ── Annotation in the donut hole ───────────────────────────
    # We place two text annotations inside the hole:
    # the focus score (large) and the word "Focus Score" (small).
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="Time Breakdown by State", x=0.5, xanchor="center", font=dict(size=14, weight="bold")),
        legend=dict(
            orientation="h",        # horizontal legend below the chart
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        annotations=[
            dict(
                text=f"<b>{focus_pct:.0f}%</b>",
                x=0.5, y=0.55,
                font=dict(size=28, color="#426B54", family="Manrope, sans-serif"),
                showarrow=False,
            ),
            dict(
                text="Focus<br>Score",
                x=0.5, y=0.38,
                font=dict(size=12, color="#8A8F88"),
                showarrow=False,
            ),
        ],
    )

    return fig


# ============================================================
# HELPER FUNCTIONS for splitting distracted_duration
# ============================================================
def _phone_fraction(stats: dict) -> float:
    """
    Return the fraction of distracted_duration attributable to phone.

    sessions.csv stores distracted_duration as phone + look_away combined.
    We split it by using phone_events and look_away_events as weights.

    If there are no events at all, we split 50/50.
    """
    phone_evt = stats.get("phone_events", 0)
    look_evt  = stats.get("look_away_events", 0)
    total_evt = phone_evt + look_evt
    if total_evt == 0:
        return 0.5
    return phone_evt / total_evt


def _look_fraction(stats: dict) -> float:
    """Return the fraction of distracted_duration attributable to looking away."""
    return 1.0 - _phone_fraction(stats)


# ============================================================
# CHART 2: make_state_timeline
# ============================================================
def make_state_timeline(events_df: pd.DataFrame, session_stats: dict) -> go.Figure:
    """
    Create a horizontal Gantt-style timeline chart showing WHEN each
    behavioral state occurred during the study session.

    HOW IT WORKS:
        Each state occupies one horizontal "lane" (row).
        When the student was in a particular state, a colored bar is
        drawn in that lane. The X axis is time elapsed in minutes.

        We reconstruct the time intervals from the events log:
        - Each row in events_df is a STATE CHANGE.
        - Interval = [previous event's timestamp → this event's timestamp]
        - The state shown in that interval is the PREVIOUS state.

    Args:
        events_df:     DataFrame with columns: session_id, timestamp,
                       previous_state, new_state, event_type.
                       (May be empty if the state never changed.)
        session_stats: Dict with start_time (ISO string) and total_duration.

    Returns:
        A Plotly Figure with one horizontal bar trace per state.
    """
    fig = go.Figure()

    # ── Handle edge case: no events (student stayed in one state all session) ──
    # If events_df is empty it means the state never changed.
    # We draw one bar covering the whole session in STATE_FOCUSED.
    total_dur_min = session_stats.get("total_duration", 0) / 60

    if events_df is None or len(events_df) == 0:
        for state in [STATE_FOCUSED, STATE_DISTRACTED_PHONE, STATE_DISTRACTED_LOOK, STATE_AWAY]:
            fig.add_trace(go.Bar(
                name=STATE_LABELS[state],
                x=[total_dur_min if state == STATE_FOCUSED else 0],
                y=[STATE_LABELS[state]],
                orientation="h",
                marker_color=STATE_COLORS[state],
                hovertemplate=f"{STATE_LABELS[state]}: {total_dur_min:.1f} min<extra></extra>",
            ))
        _apply_timeline_layout(fig, total_dur_min)
        return fig

    # ── Parse event timestamps into floats (seconds since session start) ──
    # timestamps in events.csv are stored as ISO 8601 strings.
    # We need to convert them to "minutes elapsed since session start"
    # so the X axis shows meaningful time labels.
    try:
        import datetime as dt

        # Parse the session start time
        start_iso = session_stats.get("start_time", "")
        if start_iso:
            session_start = dt.datetime.fromisoformat(start_iso)
        else:
            # Fallback: treat first event as time=0
            session_start = dt.datetime.fromisoformat(events_df.iloc[0]["timestamp"])

        def to_elapsed_min(ts_str: str) -> float:
            """Convert an ISO timestamp string to minutes since session start."""
            ts = dt.datetime.fromisoformat(ts_str)
            return (ts - session_start).total_seconds() / 60.0

    except Exception:
        # If timestamp parsing fails entirely, use a simple index fallback
        def to_elapsed_min(ts_str: str) -> float:
            return 0.0

    # ── Build a list of time intervals with their state ────────
    # Each interval = (start_min, end_min, state)
    # The state is the state that was ACTIVE during that interval.
    intervals = []

    # Sort events by timestamp just to be safe
    sorted_events = events_df.sort_values("timestamp").reset_index(drop=True)

    for i, row in sorted_events.iterrows():
        # interval start = when this state began (= timestamp of this transition)
        # interval end   = when this state ended (= timestamp of next transition)
        interval_start = to_elapsed_min(row["timestamp"])

        if i + 1 < len(sorted_events):
            interval_end = to_elapsed_min(sorted_events.iloc[i + 1]["timestamp"])
        else:
            # Last event: the state ran until the session ended
            interval_end = total_dur_min

        active_state = row["new_state"]   # state that became active at this event
        duration = interval_end - interval_start

        if duration > 0:
            intervals.append((interval_start, interval_end, active_state, duration))

    # Also add the interval BEFORE the first event (the initial state)
    if len(sorted_events) > 0:
        first_event_min = to_elapsed_min(sorted_events.iloc[0]["timestamp"])
        if first_event_min > 0:
            # The session started in STATE_FOCUSED (guaranteed by start_session())
            intervals.insert(0, (0.0, first_event_min, STATE_FOCUSED, first_event_min))

    # ── Draw one scatter/bar trace per state type ───────────────
    # We use go.Bar with orientation="h" (horizontal bars).
    # Plotly stacked bars do not easily support Gantt positioning,
    # so instead we use a workaround: for each interval we add a
    # separate trace positioned at the right Y row using a base offset.

    # Group intervals by state
    state_intervals: dict = {s: [] for s in STATE_COLORS}
    for (start, end, state, dur) in intervals:
        if state in state_intervals:
            state_intervals[state].append((start, dur))

    # We need to draw segments, not stacked bars.
    # Strategy: use go.Bar with base= to position each segment correctly.
    # We clear all previous traces and draw one trace per segment.
    # Each bar's y-axis position is the state label, its x-width is duration,
    # and its base (left edge) is the start time.

    shown_states = set()  # track which states have already been added to legend

    for state, segs in state_intervals.items():
        for (start, dur) in segs:
            show_in_legend = state not in shown_states
            shown_states.add(state)

            fig.add_trace(go.Bar(
                name=STATE_LABELS[state],
                x=[dur],                       # bar width = duration in minutes
                y=[STATE_LABELS[state]],        # which row to draw in
                base=[start],                   # left edge of the bar
                orientation="h",
                marker_color=STATE_COLORS[state],
                marker_line=dict(color="#FFFFFF", width=1),
                showlegend=show_in_legend,
                legendgroup=state,              # group same-state traces in legend
                hovertemplate=(
                    f"<b>{STATE_LABELS[state]}</b><br>"
                    f"Start: {start:.1f} min<br>"
                    f"Duration: {dur:.1f} min<extra></extra>"
                ),
            ))

    _apply_timeline_layout(fig, total_dur_min)
    return fig


def _apply_timeline_layout(fig: go.Figure, total_dur_min: float):
    """
    Apply shared layout settings to the timeline chart.

    Extracted into its own function to avoid repeating it in
    both the normal and the edge-case code paths above.
    """
    # Create custom layout dict based on _LAYOUT_BASE but with
    # sufficient bottom margin to prevent legend/x-axis collision.
    layout_settings = dict(_LAYOUT_BASE)
    layout_settings["margin"] = dict(l=30, r=30, t=45, b=85)

    fig.update_layout(
        **layout_settings,
        title=dict(text="Behavioral Timeline", x=0.5, xanchor="center", font=dict(size=14, weight="bold")),
        barmode="overlay",        # bars are drawn on top of each other (not stacked)
        xaxis=dict(
            title=dict(text="Time Elapsed (minutes)", standoff=12),
            range=[0, max(total_dur_min, 0.5)],
            gridcolor="#E0E0D8",
        ),
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=[          # fix the row order (top → bottom)
                STATE_LABELS[STATE_FOCUSED],
                STATE_LABELS[STATE_DISTRACTED_PHONE],
                STATE_LABELS[STATE_DISTRACTED_LOOK],
                STATE_LABELS[STATE_AWAY],
            ],
            tickfont=dict(size=11),
            gridcolor="#E0E0D8",
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.32,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        height=320,
    )


# ============================================================
# CHART 3: make_distraction_bar
# ============================================================
def make_distraction_bar(session_stats: dict) -> go.Figure:
    """
    Create a vertical bar chart showing how many distraction events
    of each type occurred during the session.

    WHY THIS IS USEFUL:
        The donut chart shows HOW LONG distractions lasted.
        This chart shows HOW OFTEN they started.
        A student might have one long phone distraction (high duration,
        low count) or many quick glances away (low duration, high count).
        Both matter differently for improvement strategies.

    Args:
        session_stats: Dict with keys phone_events, look_away_events, away_events.

    Returns:
        A Plotly Figure with one bar per distraction type.
    """
    # ── Data ───────────────────────────────────────────────────
    categories = [
        STATE_LABELS[STATE_DISTRACTED_PHONE],
        STATE_LABELS[STATE_DISTRACTED_LOOK],
        STATE_LABELS[STATE_AWAY],
    ]
    counts = [
        int(session_stats.get("phone_events", 0)),
        int(session_stats.get("look_away_events", 0)),
        int(session_stats.get("away_events", 0)),
    ]
    colors = [
        STATE_COLORS[STATE_DISTRACTED_PHONE],
        STATE_COLORS[STATE_DISTRACTED_LOOK],
        STATE_COLORS[STATE_AWAY],
    ]

    # ── Create the bar chart ───────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=categories,
        y=counts,
        marker_color=colors,
        marker_line=dict(color="#FFFFFF", width=1),
        text=counts,                  # show the count number on top of each bar
        textposition="outside",       # place the number above the bar
        hovertemplate="<b>%{x}</b><br>Events: %{y}<extra></extra>",
        showlegend=False,
    ))

    # ── Max Y axis value ───────────────────────────────────────
    # We add a small buffer above the tallest bar so the count label
    # doesn't get cut off at the top of the chart.
    max_count = max(counts) if max(counts) > 0 else 1

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="Distraction Events by Type", x=0.5, xanchor="center", font=dict(size=14, weight="bold")),
        xaxis=dict(title="", tickfont=dict(size=11)),
        yaxis=dict(
            title="Number of Events",
            range=[0, max_count * 1.3],   # 30% buffer above tallest bar
            gridcolor="#E0E0D8",
            dtick=1,                       # only whole-number ticks (events are integers)
        ),
        height=300,
    )

    return fig


# ============================================================
# CHART 4: make_multi_session_trend
# ============================================================
def make_multi_session_trend(sessions_df: pd.DataFrame) -> go.Figure:
    """
    Create a line chart showing focus score trend across multiple
    historical sessions.

    This chart is shown at the BOTTOM of the analytics page and
    gives the student a long-term view of their progress.

    HOW IT WORKS:
        X axis = session date/time (chronological order)
        Y axis = focus_percentage (0–100%)
        A line connects all sessions so trends are visible.
        A shaded area below the line makes it easier to read.

    Args:
        sessions_df: The full sessions DataFrame from load_all_sessions().
                     Must have columns: start_time, focus_percentage.

    Returns:
        A Plotly Figure. If fewer than 2 sessions exist, returns a
        simple message figure (we need at least 2 points for a line).
    """
    # ── Edge case: not enough data for a meaningful trend ──────
    if sessions_df is None or len(sessions_df) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Complete at least 2 sessions to see your focus trend.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=13, color="#8A8F88"),
        )
        fig.update_layout(**_LAYOUT_BASE, height=220,
                          title=dict(text="Focus Score Trend (All Sessions)", x=0.5, xanchor="center", font=dict(size=14, weight="bold")))
        return fig

    # ── Prepare data ───────────────────────────────────────────
    # Sort sessions by start_time so the X axis is chronological.
    df = sessions_df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df = df.sort_values("start_time").reset_index(drop=True)
    df["focus_percentage"] = pd.to_numeric(df["focus_percentage"], errors="coerce").fillna(0)

    # Short X-axis labels: "Nov 25, 2:30 PM"
    x_labels = df["start_time"].dt.strftime("%b %d, %I:%M %p")
    y_values = df["focus_percentage"]

    # ── Build the figure ───────────────────────────────────────
    fig = go.Figure()

    # Shaded area below the line (go.Scatter with fill="tozeroy")
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=y_values,
        fill="tozeroy",
        fillcolor="rgba(66, 107, 84, 0.12)",   # very light green fill
        line=dict(color="#426B54", width=2.5),
        mode="lines+markers",
        marker=dict(size=7, color="#426B54", line=dict(color="#FFFFFF", width=2)),
        hovertemplate="<b>%{x}</b><br>Focus Score: %{y:.1f}%<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="Focus Score Trend (All Sessions)", x=0.5, xanchor="center", font=dict(size=14, weight="bold")),
        xaxis=dict(title="", tickangle=-30, tickfont=dict(size=10), gridcolor="#E0E0D8"),
        yaxis=dict(title="Focus Score (%)", range=[0, 105], gridcolor="#E0E0D8"),
        height=260,
    )

    return fig
