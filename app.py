# ============================================================
# app.py — Main entry point for the AI Study Focus Coach
# ============================================================
#
# PURPOSE:
#   The ONLY file Streamlit runs. Defines sidebar navigation
#   and renders one of four pages.
#
# HOW TO RUN:
#   streamlit run app.py
#
# REDESIGN (UI/UX overhaul):
#   - New deep-navy / teal / sage color palette
#   - Premium dashboard cards with icons
#   - Gradient headings, feature grids, How It Works section
#   - FAQ accordion, professional footer on Home page
#   - Polished Live Session monitoring dashboard
#   - Analytics page with premium data cards
#   - Report page with elegant section cards
#   - All existing Python logic preserved exactly
# ============================================================

import streamlit as st
import pandas as pd
from datetime import datetime
import os
import time

# ── Page configuration (MUST be first Streamlit call) ─────────
st.set_page_config(
    page_title="AI Study Focus Coach",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DESIGN SYSTEM — Complete CSS
# ============================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Manrope:wght@400;500;600;700;800;900&display=swap');

/* ── CSS Variables ───────────────────────────────────────── */
:root {
    --navy:        #172B3A;
    --teal:        #258F87;
    --teal-lt:     #34B5AC;
    --sage:        #DCEFE7;
    --lavender:    #EAE9FA;
    --bg:          #F6F8FA;
    --bg2:         #EDF3F5;
    --card:        #FFFFFF;
    --text:        #172B3A;
    --text2:       #64748B;
    --muted:       #94A3B8;
    --border:      #E2E8F0;
    --border-lt:   #F1F5F9;

    /* Attention state colors */
    --focused:     #22A477;
    --look-away:   #E5A23B;
    --phone:       #E46F69;
    --away:        #8995A5;

    /* Gradients */
    --grad-teal:   linear-gradient(135deg, #258F87 0%, #34B5AC 100%);
    --grad-navy:   linear-gradient(135deg, #172B3A 0%, #1E3A4F 100%);
    --grad-hero:   linear-gradient(135deg, #172B3A 0%, #1A3F52 50%, #1D4A5E 100%);

    /* Shadows */
    --shadow-sm:   0 1px 3px rgba(23,43,58,0.08), 0 1px 2px rgba(23,43,58,0.04);
    --shadow-md:   0 4px 12px rgba(23,43,58,0.10), 0 2px 6px rgba(23,43,58,0.06);
    --shadow-lg:   0 10px 30px rgba(23,43,58,0.12), 0 4px 12px rgba(23,43,58,0.08);

    --radius-sm:   8px;
    --radius-md:   12px;
    --radius-lg:   18px;
    --radius-xl:   24px;
}

/* ── Base ────────────────────────────────────────────────── */
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {
    font-family: 'Inter', 'Manrope', sans-serif !important;
    color: var(--text) !important;
}

.stApp {
    background-color: var(--bg) !important;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: none !important;
}

[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.85) !important;
}

[data-testid="stSidebar"] .stRadio label {
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.88rem !important;
    padding: 0.4rem 0 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: rgba(255,255,255,0.55) !important;
}

/* ── Sidebar brand ───────────────────────────────────────── */
.sidebar-brand {
    padding: 1.4rem 0 1.6rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 1.2rem;
}
.sidebar-brand .brand-icon {
    font-size: 1.8rem;
    margin-bottom: 0.5rem;
    display: block;
}
.sidebar-brand h2 {
    font-family: 'Manrope', sans-serif !important;
    font-weight: 800;
    font-size: 1.05rem;
    color: #FFFFFF !important;
    margin: 0 0 0.2rem 0;
    letter-spacing: -0.2px;
}
.sidebar-brand p {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.5) !important;
    margin: 0;
    letter-spacing: 0.02em;
}

.sidebar-nav-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: rgba(255,255,255,0.35) !important;
    margin: 0 0 0.6rem 0;
    padding-top: 0.4rem;
}

.sidebar-footer {
    font-size: 0.7rem;
    color: rgba(255,255,255,0.3) !important;
    line-height: 1.6;
    padding: 1rem 0;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin-top: 1rem;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button[kind="primary"],
div[data-testid="stButton"] > button {
    background: var(--grad-teal) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.5rem !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 8px rgba(37,143,135,0.3) !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
}
.stButton > button[kind="primary"]:hover,
div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(37,143,135,0.4) !important;
}

/* ── Metrics ─────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1.1rem 1.3rem !important;
    box-shadow: var(--shadow-sm) !important;
    transition: box-shadow 0.2s ease !important;
}
[data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-md) !important;
}
[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--muted) !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700 !important;
    color: var(--navy) !important;
}

/* ── Selectbox ───────────────────────────────────────────── */
[data-testid="stSelectbox"] label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--muted) !important;
}

/* ── Divider ─────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Images ──────────────────────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: var(--radius-md) !important;
}

/* ============================================================
   PAGE HERO SECTION
   ============================================================ */
.page-hero {
    padding: 2rem 0 1.5rem 0;
    margin-bottom: 0.5rem;
}
.page-hero h1 {
    font-family: 'Manrope', sans-serif !important;
    font-size: 2.2rem;
    font-weight: 900;
    color: var(--navy) !important;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.6px;
    line-height: 1.2;
}
.page-hero p {
    font-size: 1rem;
    color: var(--text2) !important;
    margin: 0;
    max-width: 560px;
    line-height: 1.65;
}

/* gradient word highlight */
.grad-text {
    background: var(--grad-teal);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ============================================================
   SECTION HEADING
   ============================================================ */
.section-heading {
    font-family: 'Manrope', sans-serif !important;
    font-size: 1.25rem;
    font-weight: 800;
    color: var(--navy) !important;
    margin: 0 0 1rem 0;
    letter-spacing: -0.3px;
}
.section-sub {
    font-size: 0.92rem;
    color: var(--text2);
    margin: -0.6rem 0 1.2rem 0;
    line-height: 1.6;
}

/* ============================================================
   CARD SYSTEM
   ============================================================ */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}

/* ── Metric card ─────────────────────────────────────────── */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.3rem 1.5rem;
    box-shadow: var(--shadow-sm);
    margin-bottom: 0.9rem;
    transition: box-shadow 0.2s ease;
}
.metric-card:hover { box-shadow: var(--shadow-md); }
.metric-card .metric-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted) !important;
    margin-bottom: 0.4rem;
}
.metric-card .metric-value {
    font-family: 'Manrope', sans-serif !important;
    font-size: 2.1rem;
    font-weight: 800;
    color: var(--navy) !important;
    line-height: 1.1;
}
.metric-card .metric-sub {
    font-size: 0.75rem;
    color: var(--muted) !important;
    margin-top: 0.3rem;
}

/* ── Feature card ────────────────────────────────────────── */
.feature-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.6rem;
    box-shadow: var(--shadow-sm);
    height: 100%;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.feature-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-3px);
}
.feature-icon {
    width: 48px;
    height: 48px;
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    margin-bottom: 1rem;
}
.feature-card h4 {
    font-family: 'Manrope', sans-serif !important;
    font-size: 1rem;
    font-weight: 700;
    color: var(--navy) !important;
    margin: 0 0 0.5rem 0;
}
.feature-card p {
    font-size: 0.85rem;
    color: var(--text2) !important;
    line-height: 1.65;
    margin: 0;
}

/* ── Stat card (home overview) ───────────────────────────── */
.stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.4rem 1.6rem;
    box-shadow: var(--shadow-sm);
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.stat-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}
.stat-icon-wrap {
    width: 50px;
    height: 50px;
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    flex-shrink: 0;
}
.stat-info .stat-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: var(--muted) !important;
    margin-bottom: 0.2rem;
}
.stat-info .stat-value {
    font-family: 'Manrope', sans-serif !important;
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--navy) !important;
    line-height: 1.1;
}
.stat-info .stat-sub {
    font-size: 0.72rem;
    color: var(--muted) !important;
    margin-top: 0.1rem;
}

/* ── Session history row ─────────────────────────────────── */
.session-row {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1rem 1.3rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.15s ease, transform 0.15s ease;
}
.session-row:hover {
    box-shadow: var(--shadow-md);
    transform: translateX(2px);
}
.session-row .sr-date {
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--navy) !important;
}
.session-row .sr-meta {
    font-size: 0.75rem;
    color: var(--muted) !important;
    margin-top: 0.1rem;
}

/* ============================================================
   STATE BADGES
   ============================================================ */
.state-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.03em;
}
.state-badge::before { content: "●"; font-size: 0.6rem; }

.state-focused   { background: rgba(34,164,119,0.12); color: #1A7D56; }
.state-phone     { background: rgba(228,111,105,0.12); color: #B83A35; }
.state-away-look { background: rgba(229,162,59,0.12);  color: #9A6A10; }
.state-away-desk { background: rgba(137,149,165,0.12); color: #546070; }

/* Large state indicator on live session page */
.state-indicator {
    padding: 1rem 1.5rem;
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    gap: 0.9rem;
    margin-bottom: 0.6rem;
}
.si-focused  { background: rgba(34,164,119,0.08);  border: 1.5px solid rgba(34,164,119,0.25); }
.si-phone    { background: rgba(228,111,105,0.08); border: 1.5px solid rgba(228,111,105,0.25); }
.si-look     { background: rgba(229,162,59,0.08);  border: 1.5px solid rgba(229,162,59,0.25); }
.si-away     { background: rgba(137,149,165,0.08); border: 1.5px solid rgba(137,149,165,0.25); }
.si-dot {
    width: 14px; height: 14px;
    border-radius: 50%;
    flex-shrink: 0;
}
.si-focused  .si-dot { background: var(--focused); box-shadow: 0 0 0 4px rgba(34,164,119,0.2); }
.si-phone    .si-dot { background: var(--phone);   box-shadow: 0 0 0 4px rgba(228,111,105,0.2); }
.si-look     .si-dot { background: var(--look-away);box-shadow: 0 0 0 4px rgba(229,162,59,0.2); }
.si-away     .si-dot { background: var(--away);    box-shadow: 0 0 0 4px rgba(137,149,165,0.2); }
.si-label {
    font-size: 1rem;
    font-weight: 700;
    font-family: 'Manrope', sans-serif !important;
}
.si-focused  .si-label { color: #1A7D56 !important; }
.si-phone    .si-label { color: #B83A35 !important; }
.si-look     .si-label { color: #9A6A10 !important; }
.si-away     .si-label { color: #546070 !important; }
.si-explain  {
    font-size: 0.78rem;
    color: var(--text2) !important;
    margin-top: 0.1rem;
}

/* ============================================================
   CALLOUTS / ALERTS
   ============================================================ */
.callout {
    padding: 0.9rem 1.1rem;
    border-radius: var(--radius-sm);
    font-size: 0.85rem;
    line-height: 1.65;
    margin: 0.6rem 0;
    display: flex;
    gap: 0.7rem;
}
.callout-info   { background: rgba(37,143,135,0.07); border-left: 3px solid var(--teal); }
.callout-warn   { background: rgba(229,162,59,0.08);  border-left: 3px solid var(--look-away); }
.callout-danger { background: rgba(228,111,105,0.08); border-left: 3px solid var(--phone); }
.callout-success{ background: rgba(34,164,119,0.08);  border-left: 3px solid var(--focused); }

/* ============================================================
   EMPTY STATES
   ============================================================ */
.empty-state {
    text-align: center;
    padding: 3.5rem 2rem;
    background: var(--card);
    border: 1.5px dashed var(--border);
    border-radius: var(--radius-xl);
    margin: 1rem 0;
}
.empty-state .empty-icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.6; }
.empty-state h3 {
    font-family: 'Manrope', sans-serif !important;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--navy) !important;
    margin: 0 0 0.5rem 0;
}
.empty-state p {
    font-size: 0.88rem;
    color: var(--muted) !important;
    max-width: 340px;
    margin: 0 auto;
    line-height: 1.65;
}

/* ============================================================
   WEBCAM FEED AREA
   ============================================================ */
.webcam-placeholder {
    background: linear-gradient(135deg, #0F1E28 0%, #172B3A 100%);
    border-radius: var(--radius-lg);
    height: 380px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 0.8rem;
    color: rgba(255,255,255,0.4);
    font-size: 0.85rem;
    border: 1px solid rgba(255,255,255,0.06);
}
.webcam-icon { font-size: 3rem; opacity: 0.3; margin-bottom: 0.4rem; }

/* ============================================================
   HOW IT WORKS STEPS
   ============================================================ */
.how-step {
    text-align: center;
    padding: 1.5rem 1rem;
}
.step-number-wrap {
    width: 60px;
    height: 60px;
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    margin: 0 auto 1.1rem auto;
    position: relative;
}
.step-num-badge {
    position: absolute;
    top: -8px;
    right: -8px;
    background: var(--teal);
    color: #fff !important;
    font-size: 0.65rem;
    font-weight: 800;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 2px solid var(--bg);
}
.how-step h4 {
    font-family: 'Manrope', sans-serif !important;
    font-size: 1rem;
    font-weight: 700;
    color: var(--navy) !important;
    margin: 0 0 0.4rem 0;
}
.how-step p {
    font-size: 0.84rem;
    color: var(--text2) !important;
    line-height: 1.65;
    margin: 0;
}

/* ============================================================
   FAQ ACCORDION
   ============================================================ */
.faq-item {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    margin-bottom: 0.5rem;
    overflow: hidden;
}
.faq-q {
    padding: 1rem 1.3rem;
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--navy) !important;
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
}
.faq-a {
    padding: 0 1.3rem 1rem 1.3rem;
    font-size: 0.85rem;
    color: var(--text2) !important;
    line-height: 1.65;
}

/* ============================================================
   REPORT SECTION CARDS
   ============================================================ */
.report-section {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem 1.8rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s ease;
}
.report-section:hover { box-shadow: var(--shadow-md); }
.report-section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.9rem;
}
.report-section-icon {
    width: 36px;
    height: 36px;
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.report-section h4 {
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--teal) !important;
    margin: 0;
}
.report-section p {
    font-size: 0.9rem;
    line-height: 1.75;
    color: var(--text) !important;
    margin: 0;
}

/* ============================================================
   DISTRACTION LOG EVENTS
   ============================================================ */
.event-row {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    padding: 0.6rem 0.9rem;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    margin-bottom: 0.4rem;
    font-size: 0.8rem;
    transition: background 0.15s ease;
}
.event-row:hover { background: var(--bg2); }
.event-time {
    color: var(--muted) !important;
    font-weight: 600;
    min-width: 72px;
    font-size: 0.75rem;
}
.event-desc { flex: 1; color: var(--text2) !important; }

/* ============================================================
   FOOTER
   ============================================================ */
.page-footer {
    background: var(--navy);
    border-radius: var(--radius-xl);
    padding: 2rem 2.5rem 1.5rem 2.5rem;
    margin-top: 2.5rem;
    color: rgba(255,255,255,0.65);
}
.footer-brand h3 {
    font-family: 'Manrope', sans-serif !important;
    font-weight: 800;
    font-size: 1.1rem;
    color: #FFFFFF !important;
    margin: 0 0 0.3rem 0;
}
.footer-brand p {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.45) !important;
    line-height: 1.6;
    margin: 0;
}
.footer-copy {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.3) !important;
    padding-top: 1.2rem;
    margin-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.08);
    text-align: center;
}

/* ============================================================
   HOME HERO SECTION
   ============================================================ */
.hero-section {
    background: var(--grad-hero);
    border-radius: var(--radius-xl);
    padding: 3.5rem 3rem 3rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-section::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(37,143,135,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 999px;
    padding: 0.3rem 0.9rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: rgba(255,255,255,0.8) !important;
    margin-bottom: 1.2rem;
    letter-spacing: 0.04em;
}
.hero-h1 {
    font-family: 'Manrope', sans-serif !important;
    font-size: 3rem;
    font-weight: 900;
    color: #FFFFFF !important;
    line-height: 1.15;
    letter-spacing: -1px;
    margin: 0 0 1rem 0;
}
.hero-h1 .teal-word {
    background: linear-gradient(135deg, #34D3CA 0%, #22A477 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.68) !important;
    line-height: 1.65;
    max-width: 480px;
    margin: 0 0 2rem 0;
}
.hero-stats {
    display: flex;
    gap: 2rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.1);
    margin-top: 1.5rem;
}
.hero-stat-item .hs-num {
    font-family: 'Manrope', sans-serif !important;
    font-size: 1.6rem;
    font-weight: 800;
    color: #FFFFFF !important;
    line-height: 1;
}
.hero-stat-item .hs-label {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.45) !important;
    margin-top: 0.2rem;
}

/* ============================================================
   TEAL SECTION LABEL (small caps above heading)
   ============================================================ */
.section-eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--teal) !important;
    margin-bottom: 0.5rem;
}
.section-title {
    font-family: 'Manrope', sans-serif !important;
    font-size: 1.9rem;
    font-weight: 900;
    color: var(--navy) !important;
    letter-spacing: -0.5px;
    line-height: 1.25;
    margin: 0 0 0.6rem 0;
}
.section-title .teal-word {
    background: var(--grad-teal);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.section-desc {
    font-size: 0.95rem;
    color: var(--text2) !important;
    line-height: 1.65;
    max-width: 540px;
}

/* ============================================================
   PROGRESS BAR (focus bar on live session)
   ============================================================ */
.focus-bar-wrap {
    background: var(--border);
    border-radius: 999px;
    height: 8px;
    margin-top: 0.5rem;
    overflow: hidden;
}
.focus-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: var(--grad-teal);
    transition: width 0.4s ease;
}

/* ============================================================
   CAMERA STATUS PILL
   ============================================================ */
.cam-status {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
}
.cam-active   { background: rgba(34,164,119,0.1); color: #1A7D56 !important; border: 1px solid rgba(34,164,119,0.3); }
.cam-inactive { background: rgba(137,149,165,0.1); color: #546070 !important; border: 1px solid rgba(137,149,165,0.3); }
.cam-error    { background: rgba(228,111,105,0.1); color: #B83A35 !important; border: 1px solid rgba(228,111,105,0.3); }
.cam-pulse { width: 7px; height: 7px; border-radius: 50%; animation: pulse 1.5s infinite; }
.cam-active .cam-pulse   { background: var(--focused); }
.cam-inactive .cam-pulse { background: var(--away); animation: none; }
.cam-error .cam-pulse    { background: var(--phone); animation: none; }
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.85); }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================
def load_sessions() -> pd.DataFrame:
    """Load completed sessions from data/sessions.csv."""
    sessions_path = "data/sessions.csv"
    if not os.path.exists(sessions_path):
        return pd.DataFrame(columns=[
            "session_id", "start_time", "end_time",
            "total_duration", "focused_duration", "distracted_duration",
            "away_duration", "focus_percentage",
            "phone_events", "look_away_events", "away_events",
            "longest_focus_period"
        ])
    try:
        return pd.read_csv(sessions_path)
    except Exception as e:
        st.warning(f"Could not read session history: {e}")
        return pd.DataFrame()


def format_duration(seconds: float) -> str:
    """Convert seconds to 'Xh Ym Zs' string."""
    if seconds is None or seconds < 0:
        return "0m 00s"
    total_seconds = int(seconds)
    hours   = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs    = total_seconds % 60
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def state_to_badge(state: str) -> tuple:
    """
    Map a behavioral state string to its CSS class and short label.
    Returns: (badge_css_class, display_label, explanation_text,
               indicator_class)
    """
    mapping = {
        "FOCUSED": (
            "state-focused", "Focused",
            "Your gaze is directed at the screen.", "si-focused"
        ),
        "DISTRACTED - PHONE": (
            "state-phone", "Phone Detected",
            "A mobile phone is visible in the camera frame.", "si-phone"
        ),
        "DISTRACTED - LOOKING AWAY": (
            "state-away-look", "Looking Away",
            "Your gaze appears directed away from the screen.", "si-look"
        ),
        "AWAY FROM DESK": (
            "state-away-desk", "Away from Desk",
            "No face detected for several seconds.", "si-away"
        ),
    }
    return mapping.get(state, (
        "state-away-desk", "Detecting…",
        "Analyzing — please hold still.", "si-away"
    ))


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="brand-icon">📚</span>
        <h2>Study Focus Coach</h2>
        <p>AI-Powered Attention Monitor</p>
    </div>
    <div class="sidebar-nav-label">Navigation</div>
    """, unsafe_allow_html=True)

    nav_choice = st.radio(
        label="Navigation",
        options=["Home", "Live Study Session", "Session Analytics", "AI Study Report"],
        index=["Home", "Live Study Session", "Session Analytics", "AI Study Report"]
             .index(st.session_state["current_page"]),
        label_visibility="collapsed",
    )
    st.session_state["current_page"] = nav_choice

    st.markdown("""
    <div class="sidebar-footer">
        <strong style="color:rgba(255,255,255,0.55)!important;">AI Study Focus Coach</strong><br>
        MediaPipe · YOLOv8n · Gemini Flash<br><br>
        <em style="color:rgba(255,255,255,0.3)!important;">
        For personal productivity use only.<br>
        Not a clinical or medical tool.</em>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 1: HOME / DASHBOARD
# ============================================================
def render_home_page():

    sessions_df  = load_sessions()
    has_sessions = len(sessions_df) > 0

    # ── Compute overview stats ─────────────────────────────────
    if has_sessions:
        total_sessions     = len(sessions_df)
        total_hours        = sessions_df["total_duration"].sum() / 3600
        avg_focus          = sessions_df["focus_percentage"].mean()
        last_session_focus = sessions_df.iloc[-1]["focus_percentage"]
        total_distractions = int(
            sessions_df["phone_events"].sum() +
            sessions_df["look_away_events"].sum() +
            sessions_df["away_events"].sum()
        )
    else:
        total_sessions = total_hours = avg_focus = last_session_focus = total_distractions = 0

    # ── HERO SECTION ───────────────────────────────────────────
    st.markdown(f"""
    <div class="hero-section">
        <div class="hero-eyebrow">🤖 AI-Powered · Real-Time · Personal</div>
        <div class="hero-h1">Study Smarter.<br><span class="teal-word">Stay Focused.</span><br>Achieve More.</div>
        <div class="hero-sub">
            Monitor your attention in real time using your webcam. Understand when and
            why you get distracted. Get personalized AI feedback after every session.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CTA buttons
    cta1, cta2, _ = st.columns([1, 1, 2])
    with cta1:
        if st.button("▶  Start Focus Session", type="primary", use_container_width=True):
            st.session_state["current_page"] = "Live Study Session"
            st.rerun()
    with cta2:
        if st.button("📊  View Analytics", use_container_width=True):
            st.session_state["current_page"] = "Session Analytics"
            st.rerun()

    # ── OVERVIEW STATS GRID ────────────────────────────────────
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-eyebrow">Your Progress</div>
    <div class="section-title">Focus <span class="teal-word">Overview</span></div>
    """, unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    focus_color = "#22A477" if avg_focus >= 70 else ("#E5A23B" if avg_focus >= 40 else "#E46F69")

    with s1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon-wrap" style="background:rgba(37,143,135,0.1);">📅</div>
            <div class="stat-info">
                <div class="stat-label">Sessions</div>
                <div class="stat-value">{total_sessions}</div>
                <div class="stat-sub">completed</div>
            </div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon-wrap" style="background:rgba(37,143,135,0.1);">⏱️</div>
            <div class="stat-info">
                <div class="stat-label">Study Hours</div>
                <div class="stat-value">{total_hours:.1f}h</div>
                <div class="stat-sub">total recorded</div>
            </div>
        </div>""", unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon-wrap" style="background:rgba(34,164,119,0.1);">🎯</div>
            <div class="stat-info">
                <div class="stat-label">Avg. Focus</div>
                <div class="stat-value" style="color:{focus_color}!important;">{avg_focus:.0f}%</div>
                <div class="stat-sub">across all sessions</div>
            </div>
        </div>""", unsafe_allow_html=True)
    with s4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon-wrap" style="background:rgba(228,111,105,0.1);">⚡</div>
            <div class="stat-info">
                <div class="stat-label">Distractions</div>
                <div class="stat-value">{total_distractions}</div>
                <div class="stat-sub">total logged</div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── RECENT SESSIONS ────────────────────────────────────────
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    if not has_sessions:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📖</div>
            <h3>No Study Sessions Yet</h3>
            <p>Start your first session to begin tracking your focus and
               receiving personalized AI-powered study reports.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="section-eyebrow">History</div>
        <div class="section-title">Recent <span class="teal-word">Sessions</span></div>
        """, unsafe_allow_html=True)

        recent = sessions_df.tail(8).iloc[::-1].reset_index(drop=True)
        for _, row in recent.iterrows():
            try:
                start_str = datetime.fromisoformat(row["start_time"]).strftime("%b %d, %Y  ·  %I:%M %p")
            except Exception:
                start_str = str(row.get("start_time", "Unknown"))
            duration_str = format_duration(row.get("total_duration", 0))
            focus_pct    = row.get("focus_percentage", 0)
            phone_ev     = int(row.get("phone_events", 0))
            look_ev      = int(row.get("look_away_events", 0))

            if focus_pct >= 70:
                badge_cls = "state-focused"
            elif focus_pct >= 40:
                badge_cls = "state-away-look"
            else:
                badge_cls = "state-phone"

            st.markdown(f"""
            <div class="session-row">
                <div style="flex:1;">
                    <div class="sr-date">{start_str}</div>
                    <div class="sr-meta">Duration: {duration_str} &nbsp;·&nbsp; Phone: {phone_ev} &nbsp;·&nbsp; Look-away: {look_ev}</div>
                </div>
                <span class="state-badge {badge_cls}">{focus_pct:.0f}% focused</span>
            </div>
            """, unsafe_allow_html=True)

    # ── FEATURES SECTION ──────────────────────────────────────
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; margin-bottom:1.5rem;">
        <div class="section-eyebrow" style="text-align:center;">What it does</div>
        <div class="section-title" style="text-align:center;">Everything You Need to <span class="teal-word">Stay Focused</span></div>
        <div class="section-desc" style="text-align:center; margin:0 auto;">
            Powered by computer vision and deep learning. One app to monitor, analyze,
            and understand your study habits.
        </div>
    </div>
    """, unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    features = [
        ("🧠", "rgba(37,143,135,0.1)", "Real-Time Attention Monitoring",
         "Your webcam continuously analyzes face presence and gaze direction using MediaPipe's 478-point facial landmark model."),
        ("📱", "rgba(228,111,105,0.1)", "Smart Distraction Detection",
         "YOLOv8n scans every frame for mobile phones. When one appears, the system immediately flags a phone distraction event."),
        ("📊", "rgba(37,143,135,0.1)", "Focus Performance Analytics",
         "Interactive Plotly charts show your focus percentage, distraction timeline, and long-term focus score trend."),
        ("🤖", "rgba(23,43,58,0.07)", "AI-Powered Study Reports",
         "Google Gemini Flash generates a personalized 7-section analysis of your session with concrete improvement strategies."),
        ("⏱️", "rgba(37,143,135,0.1)", "Detailed Session Tracking",
         "Every state change is logged with a timestamp. Session history is saved locally to CSV and JSON for later review."),
        ("🎯", "rgba(229,162,59,0.1)", "Behavioral State Classification",
         "A rule-based classifier with 5-frame temporal smoothing assigns one of four states: Focused, Looking Away, Phone, or Away."),
    ]

    all_cols = [fc1, fc2, fc3, fc1, fc2, fc3]
    for i, (icon, bg, title, desc) in enumerate(features):
        with all_cols[i]:
            st.markdown(f"""
            <div class="feature-card" style="margin-bottom:1rem;">
                <div class="feature-icon" style="background:{bg};">{icon}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>""", unsafe_allow_html=True)

    # ── HOW IT WORKS ───────────────────────────────────────────
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; margin-bottom:1.8rem;">
        <div class="section-eyebrow" style="text-align:center;">Process</div>
        <div class="section-title" style="text-align:center;">How It <span class="teal-word">Works</span></div>
        <div class="section-desc" style="text-align:center; margin:0 auto;">
            Three simple steps — no account needed, no cloud upload, no complexity.
        </div>
    </div>
    """, unsafe_allow_html=True)

    hw1, hw2, hw3 = st.columns(3)
    steps = [
        ("01", "🎥", "rgba(37,143,135,0.12)", "Start Your Session",
         "Sit in front of your laptop, click Start Session. The app activates your webcam and begins monitoring immediately."),
        ("02", "👁️", "rgba(23,43,58,0.08)", "Get Monitored in Real Time",
         "MediaPipe detects your face and gaze. YOLOv8 watches for phones. The classifier updates your attention state every few frames."),
        ("03", "📋", "rgba(229,162,59,0.10)", "Review & Improve",
         "After the session, explore analytics charts and generate your AI report for personalized feedback on how to study better."),
    ]
    for col, (num, icon, bg, title, desc) in zip([hw1, hw2, hw3], steps):
        with col:
            st.markdown(f"""
            <div class="how-step">
                <div class="step-number-wrap" style="background:{bg};">
                    {icon}
                    <span class="step-num-badge">{num}</span>
                </div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>""", unsafe_allow_html=True)

    # ── FAQ ────────────────────────────────────────────────────
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; margin-bottom:1.5rem;">
        <div class="section-eyebrow" style="text-align:center;">Help</div>
        <div class="section-title" style="text-align:center;">Frequently Asked <span class="teal-word">Questions</span></div>
    </div>
    """, unsafe_allow_html=True)

    faqs = [
        ("Does it store my webcam footage?",
         "No. Video frames are processed in memory in real time and immediately discarded. No images or video are ever saved to disk."),
        ("Do I need a GPU?",
         "No. The app is designed to run entirely on CPU. MediaPipe and YOLOv8n are both lightweight enough for a standard laptop."),
        ("How accurate is the gaze estimation?",
         "Gaze estimation uses the relative position of the iris within the eye boundaries — it is a simplified heuristic, not a clinically validated measurement. It works best in good lighting without glasses glare."),
        ("Is a Gemini API key required?",
         "No. If no API key is provided, the app generates a locally computed template report instead. The key is only needed for the AI-generated version."),
        ("What does the focus score mean?",
         "Focus score = (seconds with face detected and gaze centered) ÷ (total session seconds) × 100. It is an estimate of visual engagement, not a scientific measure of cognitive focus."),
        ("Can I use it on Windows / macOS / Linux?",
         "Yes. The app runs on any OS that supports Python 3.11, OpenCV, and Streamlit. Webcam access depends on your OS permissions."),
    ]

    fq1, fq2 = st.columns(2)
    for i, (q, a) in enumerate(faqs):
        col = fq1 if i % 2 == 0 else fq2
        with col:
            with st.expander(q):
                st.markdown(f'<p style="font-size:0.87rem; color:#64748B; line-height:1.65;">{a}</p>',
                            unsafe_allow_html=True)

    # ── FOOTER ─────────────────────────────────────────────────
    st.markdown("""
    <div class="page-footer">
        <div style="display:flex; gap:3rem; flex-wrap:wrap; align-items:flex-start;">
            <div class="footer-brand" style="flex:1; min-width:220px;">
                <h3>📚 AI Study Focus Coach</h3>
                <p>An AI-powered attention monitoring tool for students.<br>
                Built with MediaPipe, YOLOv8, Streamlit and Gemini Flash.</p>
            </div>
            <div style="display:flex; gap:3rem; flex-wrap:wrap;">
                <div>
                    <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase;
                                letter-spacing:0.1em; color:rgba(255,255,255,0.4)!important;
                                margin-bottom:0.7rem;">Pages</div>
                    <div style="font-size:0.82rem; line-height:2; color:rgba(255,255,255,0.55)!important;">
                        Home Dashboard<br>Live Study Session<br>Session Analytics<br>AI Study Report
                    </div>
                </div>
                <div>
                    <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase;
                                letter-spacing:0.1em; color:rgba(255,255,255,0.4)!important;
                                margin-bottom:0.7rem;">Technology</div>
                    <div style="font-size:0.82rem; line-height:2; color:rgba(255,255,255,0.55)!important;">
                        Python 3.11<br>MediaPipe FaceLandmarker<br>YOLOv8n (COCO)<br>Google Gemini Flash
                    </div>
                </div>
            </div>
        </div>
        <div class="footer-copy">
            © 2024 AI Study Focus Coach &nbsp;·&nbsp; For personal productivity use only.
            Not a medical or clinical assessment tool.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 2: LIVE STUDY SESSION
# ============================================================
def render_live_session_page():
    """
    Live Study Session page — wired to:
      - camera.py (webcam + CV pipeline)
      - attention_classifier.py (temporal smoothing)
      - session_manager.py (session lifecycle, stats, saving)
    """
    # ── Import modules ─────────────────────────────────────────
    try:
        from modules import camera as cam
        from modules import session_manager as sm
        from modules import attention_classifier as ac
        import cv2
        modules_ok = True
    except ImportError as e:
        modules_ok = False
        st.error(f"**Module unavailable:** `{e}`\n\nRun `pip install -r requirements.txt`")
        return

    # ── Initialize session state ───────────────────────────────
    if "session_running" not in st.session_state:
        st.session_state["session_running"] = False
    if "session_data" not in st.session_state:
        st.session_state["session_data"] = None
    if "session_ended_stats" not in st.session_state:
        st.session_state["session_ended_stats"] = None

    # ── Page hero ──────────────────────────────────────────────
    st.markdown("""
    <div class="page-hero">
        <h1>Live Study Session</h1>
        <p>Sit in front of your webcam and press <strong>Start Session</strong>.
           The app will monitor your attention state in real time.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Two-column layout ──────────────────────────────────────
    cam_col, stats_col = st.columns([3, 2], gap="large")

    with stats_col:
        # ── Session controls ───────────────────────────────────
        st.markdown('<div class="section-heading">Session Controls</div>', unsafe_allow_html=True)

        ctrl1, ctrl2 = st.columns(2)
        with ctrl1:
            start_btn = st.button(
                "▶  Start Session", type="primary",
                use_container_width=True,
                disabled=st.session_state["session_running"]
            )
        with ctrl2:
            stop_btn = st.button(
                "⏹  End Session",
                use_container_width=True,
                disabled=not st.session_state["session_running"]
            )

        # ── Handle Start ───────────────────────────────────────
        if start_btn and not st.session_state["session_running"]:
            with st.spinner("Opening camera..."):
                success = cam.start_camera(camera_index=0)
            if success:
                st.session_state["session_data"]        = sm.start_session()
                st.session_state["session_running"]     = True
                st.session_state["session_ended_stats"] = None
                st.rerun()
            else:
                cs = cam.get_camera_state()
                st.error(f"**Could not open camera:** {cs.get('error', 'Unknown error')}")

        # ── Handle Stop ────────────────────────────────────────
        if stop_btn and st.session_state["session_running"]:
            session = st.session_state["session_data"]
            if session is not None:
                stats = sm.end_session(session)
                st.session_state["session_ended_stats"] = stats
            cam.stop_camera()
            st.session_state["session_running"] = False
            st.session_state["session_data"]    = None
            st.rerun()

        # ── Read camera state ──────────────────────────────────
        camera_state   = cam.get_camera_state()
        is_cam_running = camera_state.get("running", False)
        classified     = camera_state.get("classified_state", "FOCUSED")
        current_fps    = camera_state.get("fps", 0.0)
        cam_error      = camera_state.get("error")

        # ── Feed classifier into session manager ───────────────
        session = st.session_state.get("session_data")
        if session is not None and session.get("is_active"):
            sm.update_session(session, classified)

        # ── Get live stats ─────────────────────────────────────
        if session is not None and session.get("is_active"):
            live_stats = sm.get_session_stats(session)
        else:
            live_stats = None

        # ── Current state indicator ────────────────────────────
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        if not is_cam_running and not st.session_state["session_running"]:
            badge_cls, label, explain, si_cls = "state-away-desk", "Not Started", "Press Start Session to begin.", "si-away"
        else:
            badge_cls, label, explain, si_cls = state_to_badge(classified)

        st.markdown(f"""
        <div class="state-indicator {si_cls}">
            <div class="si-dot"></div>
            <div>
                <div class="si-label">{label}</div>
                <div class="si-explain">{explain}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Session timer ──────────────────────────────────────
        timer_str = format_duration(live_stats["total_duration"]) if live_stats else "0m 00s"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⏱ Session Timer</div>
            <div class="metric-value">{timer_str}</div>
            <div class="metric-sub">wall-clock elapsed time</div>
        </div>""", unsafe_allow_html=True)

        # ── Focus percentage with bar ──────────────────────────
        focus_val = live_stats['focus_percentage'] if live_stats else 0
        focus_str = f"{focus_val:.0f}%" if live_stats else "—"
        focus_bar_w = max(0, min(100, focus_val))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🎯 Focus Score</div>
            <div class="metric-value">{focus_str}</div>
            <div class="focus-bar-wrap">
                <div class="focus-bar-fill" style="width:{focus_bar_w}%;"></div>
            </div>
            <div class="metric-sub" style="margin-top:0.5rem;">Focused time ÷ total time</div>
        </div>""", unsafe_allow_html=True)

        # ── Distraction counts ─────────────────────────────────
        if live_stats:
            phone_ev = live_stats["phone_events"]
            look_ev  = live_stats["look_away_events"]
            away_ev  = live_stats["away_events"]
            total_ev = phone_ev + look_ev + away_ev
        else:
            phone_ev = look_ev = away_ev = total_ev = 0

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚡ Distractions</div>
            <div class="metric-value">{total_ev}</div>
            <div style="display:flex; gap:0.9rem; margin-top:0.4rem; flex-wrap:wrap;">
                <span style="font-size:0.72rem; color:#E46F69;">📱 Phone: {phone_ev}</span>
                <span style="font-size:0.72rem; color:#E5A23B;">👀 Look: {look_ev}</span>
                <span style="font-size:0.72rem; color:#8995A5;">🚶 Away: {away_ev}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── FPS / processing speed ─────────────────────────────
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚙️ Processing Speed</div>
            <div class="metric-value">{current_fps:.0f} <span style="font-size:1rem;font-weight:400;">FPS</span></div>
        </div>""", unsafe_allow_html=True)

        # ── Camera status pill ─────────────────────────────────
        if cam_error and not is_cam_running:
            st.markdown(f"""
            <div class="cam-status cam-error">
                <span class="cam-pulse"></span> Camera Error — {cam_error[:60]}
            </div>""", unsafe_allow_html=True)
        elif is_cam_running:
            st.markdown("""
            <div class="cam-status cam-active">
                <span class="cam-pulse"></span>
                MediaPipe active &nbsp;·&nbsp; YOLOv8n scanning &nbsp;·&nbsp; 5-frame smoothing
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="cam-status cam-inactive">
                <span class="cam-pulse"></span> Camera inactive
            </div>""", unsafe_allow_html=True)

    # ── Camera feed column ─────────────────────────────────────
    with cam_col:
        st.markdown('<div class="section-heading">Camera Feed</div>', unsafe_allow_html=True)
        frame_placeholder = st.empty()

        annotated_frame = camera_state.get("annotated_frame")
        if is_cam_running and annotated_frame is not None:
            try:
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(
                    rgb_frame, channels="RGB",
                    use_container_width=True,
                    caption="Green dots: eye corners  |  Blue dots: iris centers  |  Red box: phone detected"
                )
            except Exception as e:
                frame_placeholder.error(f"Frame display error: {e}")
        else:
            frame_placeholder.markdown("""
            <div class="webcam-placeholder">
                <div class="webcam-icon">📷</div>
                <div style="color:rgba(255,255,255,0.4); font-weight:600;">Camera Feed</div>
                <div style="font-size:0.78rem; color:rgba(255,255,255,0.25); text-align:center; max-width:220px; margin-top:0.3rem;">
                    Press <strong style="color:rgba(255,255,255,0.5);">Start Session</strong> to activate webcam monitoring
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Session-ended summary ──────────────────────────────────
    ended_stats = st.session_state.get("session_ended_stats")
    if ended_stats and not st.session_state["session_running"]:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        total_dist = ended_stats['phone_events'] + ended_stats['look_away_events'] + ended_stats['away_events']
        st.markdown(f"""
        <div class="callout callout-success">
            <span style="font-size:1.3rem;">✅</span>
            <div>
                <strong>Session saved successfully!</strong><br>
                <span style="font-size:0.85rem;">
                Duration: {format_duration(ended_stats['total_duration'])} &nbsp;·&nbsp;
                Focus: {ended_stats['focus_percentage']:.0f}% &nbsp;·&nbsp;
                Distractions: {total_dist}<br>
                Go to <strong>Session Analytics</strong> for detailed charts, or
                <strong>AI Study Report</strong> for personalized feedback.
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Distraction event log ──────────────────────────────────
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Distraction Log</div>', unsafe_allow_html=True)

    events = []
    if session is not None and session.get("events"):
        events = session["events"]

    if not events:
        st.markdown("""
        <div class="empty-state" style="padding:1.8rem;">
            <div class="empty-icon" style="font-size:2rem;">📋</div>
            <h3>No Events Yet</h3>
            <p>State-change events will appear here as the classifier
               detects distractions during your session.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for event in reversed(events[-15:]):
            try:
                ts = datetime.fromisoformat(event["timestamp"]).strftime("%I:%M:%S %p")
            except Exception:
                ts = "—"
            desc = event.get("event_type", f"{event['previous_state']} → {event['new_state']}")
            new_state = event.get("new_state", "")
            badge_cls_evt, evt_badge, _, _ = state_to_badge(new_state)
            st.markdown(f"""
            <div class="event-row">
                <span class="event-time">{ts}</span>
                <span class="event-desc">{desc}</span>
                <span class="state-badge {badge_cls_evt}" style="font-size:0.68rem; padding:0.1rem 0.6rem;">{evt_badge}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Auto-refresh ───────────────────────────────────────────
    if st.session_state.get("session_running", False) and is_cam_running:
        time.sleep(0.1)
        st.rerun()


# ============================================================
# PAGE 3: SESSION ANALYTICS
# ============================================================
from modules import analytics
from modules import session_manager as sm


def render_analytics_page():
    st.markdown("""
    <div class="page-hero">
        <h1>Session Analytics</h1>
        <p>Explore detailed statistics and interactive visualizations for any completed study session.</p>
    </div>
    """, unsafe_allow_html=True)

    sessions_df = load_sessions()
    if len(sessions_df) == 0:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📊</div>
            <h3>No Sessions to Analyze</h3>
            <p>Complete at least one study session to unlock your analytics dashboard.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Session selector ───────────────────────────────────────
    session_labels = []
    for _, row in sessions_df.iterrows():
        try:
            label = f"{datetime.fromisoformat(row['start_time']).strftime('%b %d %Y, %I:%M %p')} — {format_duration(row['total_duration'])}"
        except Exception:
            label = str(row["session_id"])
        session_labels.append(label)

    sel_col, _ = st.columns([2, 1])
    with sel_col:
        selected_label   = st.selectbox("Select a session to analyze", session_labels, index=len(session_labels) - 1)
    selected_session = sessions_df.iloc[session_labels.index(selected_label)]
    session_id       = selected_session["session_id"]

    stats = {
        "session_id":           session_id,
        "start_time":           str(selected_session.get("start_time", "")),
        "total_duration":       float(selected_session.get("total_duration", 0)),
        "focused_duration":     float(selected_session.get("focused_duration", 0)),
        "distracted_duration":  float(selected_session.get("distracted_duration", 0)),
        "away_duration":        float(selected_session.get("away_duration", 0)),
        "focus_percentage":     float(selected_session.get("focus_percentage", 0)),
        "phone_events":         int(selected_session.get("phone_events", 0)),
        "look_away_events":     int(selected_session.get("look_away_events", 0)),
        "away_events":          int(selected_session.get("away_events", 0)),
        "longest_focus_period": float(selected_session.get("longest_focus_period", 0)),
    }

    # ── Summary metric strip ───────────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-eyebrow">Session Summary</div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Total Duration",    format_duration(stats["total_duration"]))
    with c2: st.metric("Focused",           format_duration(stats["focused_duration"]))
    with c3: st.metric("Focus Score",       f"{stats['focus_percentage']:.1f}%")
    with c4: st.metric("Longest Streak",    format_duration(stats["longest_focus_period"]))
    with c5: st.metric("Distractions",      str(stats["phone_events"] + stats["look_away_events"] + stats["away_events"]))

    # ── Chart row 1: Donut + Bar ───────────────────────────────
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Visualizations</div>', unsafe_allow_html=True)

    col_donut, col_bar = st.columns([1, 1], gap="large")
    with col_donut:
        donut_fig = analytics.make_focus_donut(stats)
        st.plotly_chart(donut_fig, use_container_width=True, config={"displayModeBar": False})
    with col_bar:
        bar_fig = analytics.make_distraction_bar(stats)
        st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

    # ── Behavioral timeline ────────────────────────────────────
    events_df    = sm.load_session_events(session_id)
    timeline_fig = analytics.make_state_timeline(events_df, stats)
    st.plotly_chart(timeline_fig, use_container_width=True, config={"displayModeBar": False})

    # ── Multi-session trend ────────────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Progress Over Time</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading" style="margin-top:0;">Focus Score Trend</div>', unsafe_allow_html=True)

    if len(sessions_df) >= 2:
        trend_fig = analytics.make_multi_session_trend(sessions_df)
        st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown("""
        <div class="callout callout-info">
            <span>💡</span>
            <span>Complete at least <strong>2 sessions</strong> to see your focus score trend over time.</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Detailed breakdown cards ───────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Detailed Breakdown</div>', unsafe_allow_html=True)

    total_min      = round(stats["total_duration"] / 60, 1)
    focused_min    = round(stats["focused_duration"] / 60, 1)
    away_min       = round(stats["away_duration"] / 60, 1)
    longest_min    = round(stats["longest_focus_period"] / 60, 1)

    st.markdown(f"""
    <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap: 0.9rem; margin-top:0.5rem;">
        <div class="metric-card">
            <div class="metric-label">🟢 Focused Time</div>
            <div class="metric-value" style="color:#1A7D56!important;">{focused_min} min</div>
            <div class="metric-sub">{stats['focus_percentage']:.1f}% of session</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">📱 Phone Distractions</div>
            <div class="metric-value" style="color:#B83A35!important;">{stats['phone_events']}</div>
            <div class="metric-sub">events detected</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">👀 Looked Away</div>
            <div class="metric-value" style="color:#9A6A10!important;">{stats['look_away_events']}</div>
            <div class="metric-sub">gaze-away events</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">🕐 Total Session</div>
            <div class="metric-value">{total_min} min</div>
            <div class="metric-sub">wall-clock time</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">⚡ Longest Focus Streak</div>
            <div class="metric-value" style="color:#1A7D56!important;">{longest_min} min</div>
            <div class="metric-sub">uninterrupted focus</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">🚶 Away from Desk</div>
            <div class="metric-value" style="color:#546070!important;">{stats['away_events']}</div>
            <div class="metric-sub">departures ({away_min} min total)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 4: AI STUDY REPORT
# ============================================================
from modules import report_generator as rg

# Icons for the 7 report sections
_SECTION_ICONS = {
    "1": ("📊", "rgba(37,143,135,0.1)"),
    "2": ("🎯", "rgba(34,164,119,0.1)"),
    "3": ("⚡", "rgba(228,111,105,0.1)"),
    "4": ("✅", "rgba(37,143,135,0.1)"),
    "5": ("🔧", "rgba(229,162,59,0.1)"),
    "6": ("💡", "rgba(37,143,135,0.1)"),
    "7": ("🚀", "rgba(23,43,58,0.08)"),
}


def _render_report_sections(report_text: str):
    """
    Parse a report string that uses '## N. Title' headings and render
    each section inside a styled .report-section card with an icon.
    """
    parts    = report_text.split("\n## ")
    preamble = parts[0].strip()
    sections = parts[1:]

    if preamble:
        is_fallback = preamble.startswith("[Note:") or preamble.startswith("[This is a locally")
        css_class   = "callout-warn" if is_fallback else "callout-info"
        icon        = "⚠️" if is_fallback else "💡"
        st.markdown(f"""
        <div class="callout {css_class}">
            <span>{icon}</span>
            <span>{preamble}</span>
        </div>""", unsafe_allow_html=True)

    for section in sections:
        lines = section.strip().split("\n", 1)
        title = lines[0].strip() if lines else "Section"
        body  = lines[1].strip() if len(lines) > 1 else ""

        body_html = (body
                     .replace("&", "&amp;")
                     .replace("<", "&lt;")
                     .replace(">", "&gt;")
                     .replace("\n", "<br>"))

        # Extract section number for icon lookup
        sec_num = title.split(".")[0].strip() if "." in title else "1"
        icon_emoji, icon_bg = _SECTION_ICONS.get(sec_num, ("📌", "rgba(37,143,135,0.1)"))

        st.markdown(f"""
        <div class="report-section">
            <div class="report-section-header">
                <div class="report-section-icon" style="background:{icon_bg};">{icon_emoji}</div>
                <h4>{title}</h4>
            </div>
            <p>{body_html}</p>
        </div>
        """, unsafe_allow_html=True)


def render_report_page():
    st.markdown("""
    <div class="page-hero">
        <h1>AI Study Report</h1>
        <p>Get a personalized 7-section analysis of your study session, powered by Google Gemini Flash.</p>
    </div>
    """, unsafe_allow_html=True)

    sessions_df = load_sessions()
    if len(sessions_df) == 0:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🤖</div>
            <h3>No Sessions Available</h3>
            <p>Complete a study session first. Then come here to generate your personalized AI report.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Session selector ───────────────────────────────────────
    session_labels = []
    for _, row in sessions_df.iterrows():
        try:
            label = f"{datetime.fromisoformat(row['start_time']).strftime('%b %d %Y, %I:%M %p')} — {format_duration(row['total_duration'])}"
        except Exception:
            label = str(row["session_id"])
        session_labels.append(label)

    sel_col2, _ = st.columns([2, 1])
    with sel_col2:
        selected_label   = st.selectbox("Select a session", session_labels, index=len(session_labels) - 1)
    selected_session = sessions_df.iloc[session_labels.index(selected_label)]
    session_id       = selected_session["session_id"]

    # ── Session at a glance ────────────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Session at a Glance</div>', unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    total_dist = (int(selected_session.get("phone_events", 0)) +
                  int(selected_session.get("look_away_events", 0)) +
                  int(selected_session.get("away_events", 0)))
    with r1: st.metric("Duration",        format_duration(selected_session.get("total_duration", 0)))
    with r2: st.metric("Focus Score",     f"{selected_session.get('focus_percentage', 0):.1f}%")
    with r3: st.metric("Distractions",    str(total_dist))
    with r4: st.metric("Longest Streak",  format_duration(selected_session.get("longest_focus_period", 0)))

    # ── API key status ─────────────────────────────────────────
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    api_key      = rg.get_api_key()
    gemini_ready = api_key is not None
    if gemini_ready:
        st.markdown("""
        <div class="callout callout-success">
            <span>✅</span>
            <span><strong>Gemini API key detected</strong> — AI report generation is available.</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="callout callout-warn">
            <span>⚠️</span>
            <span><strong>No Gemini API key found.</strong> A locally generated fallback report will be produced instead.
            Add your key to <code>.streamlit/secrets.toml</code> for the full AI report.</span>
        </div>""", unsafe_allow_html=True)

    # ── Report display / generate ──────────────────────────────
    existing_report = rg.load_report(session_id)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    if existing_report:
        # ── Show existing report ───────────────────────────────
        is_ai       = existing_report.get("is_ai", False)
        badge_label = "🤖 AI Generated" if is_ai else "📋 Local Fallback"
        badge_col   = "#22A477" if is_ai else "#E5A23B"
        gen_time    = existing_report.get("generated_at", "")
        try:
            gen_time = datetime.fromisoformat(gen_time).strftime("%b %d %Y, %I:%M %p")
        except Exception:
            pass

        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem;">
            <span style="background:{badge_col}18; color:{badge_col}; border:1px solid {badge_col}44;
                         padding:0.3rem 0.9rem; border-radius:999px; font-size:0.8rem; font-weight:700;">
                {badge_label}
            </span>
            <span style="font-size:0.78rem; color:#94A3B8;">Generated: {gen_time}</span>
        </div>
        <div class="section-eyebrow">Your Personalized Report</div>
        <div class="section-heading" style="margin-top:0;">AI Analysis Results</div>
        """, unsafe_allow_html=True)

        _render_report_sections(existing_report["text"])

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        if st.button("🔄  Regenerate Report", type="primary"):
            with st.spinner("Generating report…"):
                report_text, is_ai = rg.generate_report(dict(selected_session))
            rg.save_report(session_id, report_text, is_ai)
            st.rerun()

    else:
        # ── No report yet — show placeholder cards + generate button ──
        st.markdown("""
        <div class="section-eyebrow">Your Personalized Report</div>
        <div class="section-heading" style="margin-top:0;">Generate Your Analysis</div>
        """, unsafe_allow_html=True)

        sec_placeholders = [
            ("📊", "rgba(37,143,135,0.08)", "1. Overall Session Assessment"),
            ("🎯", "rgba(34,164,119,0.08)", "2. Focus Performance"),
            ("⚡", "rgba(228,111,105,0.08)", "3. Major Distraction Patterns"),
            ("✅", "rgba(37,143,135,0.08)", "4. Positive Study Habits"),
            ("🔧", "rgba(229,162,59,0.08)", "5. Areas Requiring Improvement"),
            ("💡", "rgba(37,143,135,0.08)", "6. Personalized Recommendations"),
            ("🚀", "rgba(23,43,58,0.05)",   "7. Suggested Strategy for Next Session"),
        ]
        for icon, bg, title in sec_placeholders:
            st.markdown(f"""
            <div class="report-section" style="opacity:0.4;">
                <div class="report-section-header">
                    <div class="report-section-icon" style="background:{bg};">{icon}</div>
                    <h4>{title}</h4>
                </div>
                <p style="color:#94A3B8;">Click "Generate Report" to see your personalized analysis here.</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        stats = {
            "total_duration":       float(selected_session.get("total_duration", 0)),
            "focused_duration":     float(selected_session.get("focused_duration", 0)),
            "distracted_duration":  float(selected_session.get("distracted_duration", 0)),
            "away_duration":        float(selected_session.get("away_duration", 0)),
            "focus_percentage":     float(selected_session.get("focus_percentage", 0)),
            "phone_events":         int(selected_session.get("phone_events", 0)),
            "look_away_events":     int(selected_session.get("look_away_events", 0)),
            "away_events":          int(selected_session.get("away_events", 0)),
            "longest_focus_period": float(selected_session.get("longest_focus_period", 0)),
        }

        if st.button("🤖  Generate Report", type="primary"):
            with st.spinner("Generating your study report… this may take a few seconds."):
                report_text, is_ai = rg.generate_report(stats)
            rg.save_report(session_id, report_text, is_ai)
            st.rerun()


# ============================================================
# MAIN ROUTER
# ============================================================
page = st.session_state["current_page"]

if page == "Home":
    render_home_page()
elif page == "Live Study Session":
    render_live_session_page()
elif page == "Session Analytics":
    render_analytics_page()
elif page == "AI Study Report":
    render_report_page()
