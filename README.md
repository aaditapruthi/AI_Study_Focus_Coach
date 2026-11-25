# 📚 AI Study Focus Coach

> **Real-Time Attention Monitoring and Distraction Analysis Using Computer Vision and Deep Learning**

A Python web application that uses your laptop's webcam to monitor your focus during a study session. It detects faces, estimates gaze direction, spots mobile phones, classifies your behavioral state in real time, and generates a personalized AI-powered study report at the end of each session.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Features](#features)
3. [Technology Stack](#technology-stack)
4. [System Architecture](#system-architecture)
5. [Folder Structure](#folder-structure)
6. [Installation](#installation)
7. [Running the App](#running-the-app)
8. [How to Use](#how-to-use)
9. [Testing](#testing)
10. [Limitations](#limitations)
11. [Future Enhancements](#future-enhancements)

---

## Problem Statement

Students often struggle to maintain focus during self-study. Distractions from phones, environmental stimuli, and fatigue go unnoticed and unquantified. Without objective data, it is hard to know when, how often, or for how long distractions occur.

This application provides:
- **Objective, real-time feedback** on attention state
- **Logged session history** for self-review
- **Personalized AI reports** that give actionable improvement suggestions

> **Important:** This application is intended for personal productivity and educational demonstration. It is not a medically validated attention assessment tool and should not be used for examination surveillance.

---

## Features

| Feature | Description |
|---|---|
| 📷 Live webcam feed | Processed frames shown in the dashboard at ~15 FPS |
| 👤 Face detection | MediaPipe FaceLandmarker detects face presence and 478 landmarks |
| 👁️ Gaze estimation | Iris position relative to eye width classifies gaze as centered or looking away |
| 📱 Phone detection | YOLOv8n (pretrained on COCO) detects mobile phones in the frame |
| 🧠 Behavioral classifier | Rule-based with 5-frame temporal smoothing; 4 states |
| ⏱️ Session logging | Timestamps, state durations, event counts saved to CSV |
| 📊 Analytics dashboard | Donut chart, Gantt timeline, distraction bar chart, multi-session trend |
| 🤖 AI study report | 7-section personalized report via Google Gemini Flash |
| 📋 Fallback report | Template-based report generated locally (no API key required) |
| 💾 Local storage | Sessions and reports saved to `data/` as CSV and JSON |

---

## Technology Stack

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.11 | Programming language |
| Streamlit | ≥ 1.35 | Web dashboard and UI |
| OpenCV | ≥ 4.9 | Webcam capture and image processing |
| MediaPipe | ≥ 0.10.14 | Face landmark detection (Tasks API) |
| Ultralytics YOLOv8n | ≥ 8.2 | Phone object detection |
| NumPy | ≥ 1.26 | Numerical array operations |
| Pandas | ≥ 2.2 | Session data management and CSV I/O |
| Plotly | ≥ 5.22 | Interactive charts |
| google-genai | ≥ 0.7 | Gemini Flash API integration |
| python-dotenv | ≥ 1.0 | Load API keys from `.env` |
| Pillow | ≥ 10.3 | Image format handling |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        app.py  (Streamlit)                  │
│  ┌──────────┐  ┌──────────────┐  ┌────────────┐  ┌──────┐  │
│  │   Home   │  │ Live Session │  │ Analytics  │  │Report│  │
│  └──────────┘  └──────┬───────┘  └─────┬──────┘  └──┬───┘  │
└─────────────────────────────────────────────────────────────┘
                         │                │              │
           ┌─────────────▼──────┐  ┌──────▼───────┐  ┌─▼──────────────┐
           │   modules/camera   │  │  analytics   │  │report_generator│
           │  (background thread│  │  (Plotly     │  │(Gemini Flash / │
           │   + frame queue)   │  │   charts)    │  │ local fallback)│
           └──────┬─────────────┘  └──────────────┘  └────────────────┘
                  │
       ┌──────────┼──────────────┐
       ▼          ▼              ▼
 ┌───────────┐ ┌──────────┐ ┌──────────────────┐
 │ face_     │ │ phone_   │ │ attention_       │
 │ detection │ │ detection│ │ classifier       │
 │(MediaPipe)│ │(YOLOv8n) │ │(rule-based +     │
 └───────────┘ └──────────┘ │ 5-frame smooth.) │
                             └──────────────────┘
                                      │
                             ┌────────▼────────┐
                             │ session_manager │
                             │  data/          │
                             │  ├─sessions.csv │
                             │  ├─events.csv   │
                             │  └─reports.json │
                             └─────────────────┘
```

**Data flow on each frame:**

```
Webcam frame
  → face_detection.analyze_frame()   → face_result dict
  → phone_detection.detect_phone()   → phone_result dict
  → attention_classifier.classify_frame()  → state string
  → session_manager.update_session() → event dict (if state changed)
  → annotated frame displayed in Streamlit
```

---

## Folder Structure

```
AI_Study_Focus_Coach/
│
├── app.py                      # Main Streamlit entry point (run this)
│
├── modules/                    # All CV and logic modules
│   ├── __init__.py             # Makes modules/ a Python package
│   ├── camera.py               # Webcam capture (background thread)
│   ├── face_detection.py       # MediaPipe FaceLandmarker + gaze estimation
│   ├── phone_detection.py      # YOLOv8n phone detection
│   ├── attention_classifier.py # Rule-based behavioral classifier
│   ├── session_manager.py      # Session lifecycle and CSV/JSON I/O
│   ├── analytics.py            # Plotly chart functions
│   └── report_generator.py     # Gemini Flash + fallback report
│
├── assets/
│   └── face_landmarker.task    # MediaPipe model file (download separately)
│
├── data/                       # Created automatically on first session save
│   ├── sessions.csv            # One row per completed session
│   ├── events.csv              # One row per state-change event
│   └── reports.json            # AI reports keyed by session_id
│
├── test_classifier.py          # Standalone unit tests (no webcam needed)
│
├── requirements.txt            # Python dependencies
├── .streamlit/
│   ├── config.toml             # Streamlit theme and server settings
│   └── secrets.toml.example   # Template for API key storage
├── .env.example                # Template for .env API key storage
└── .gitignore                  # Prevents secrets and data from being committed
```

---

## Installation

### Prerequisites

- Python **3.11** (MediaPipe requires 3.11; it may not work on 3.12+)
- A working webcam
- Git (for cloning)
- A free [Gemini API key](https://aistudio.google.com/app/apikey) *(optional — a local fallback report works without it)*

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI_Study_Focus_Coach.git
cd AI_Study_Focus_Coach
```

---

### Step 2 — Create and activate a Python 3.11 virtual environment

**macOS / Linux:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
py -3.11 -m venv venv
venv\Scripts\activate
```

> **If you don't have Python 3.11:** Install it via [pyenv](https://github.com/pyenv/pyenv):
> ```bash
> pyenv install 3.11.9
> pyenv local 3.11.9
> python -m venv venv && source venv/bin/activate
> ```

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs all packages: Streamlit, OpenCV, MediaPipe, Ultralytics, Plotly, google-genai, and others.

> **YOLOv8n model:** Downloaded automatically (~6 MB) on first run. Requires internet for the first launch.

---

### Step 4 — Download the MediaPipe face landmark model

The MediaPipe FaceLandmarker needs a `.task` model file that is not included in this repository (too large for GitHub).

Run this command once from the project root:

```bash
curl -o assets/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "assets\face_landmarker.task"
```

Confirm the file exists:
```bash
ls assets/face_landmarker.task
# Expected: a file ~26 MB in size
```

---

### Step 5 — Add your Gemini API key *(optional)*

**Option A — Streamlit secrets (recommended):**

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` and replace the placeholder:
```toml
GEMINI_API_KEY = "AIza..."
```

**Option B — `.env` file:**

```bash
cp .env.example .env
```

Edit `.env`:
```
GEMINI_API_KEY=AIza...
```

Then load it before running:
```bash
export $(cat .env | xargs)
```

> Get a free key at: https://aistudio.google.com/app/apikey
>
> If you skip this step, the app still works — a template-based fallback report is generated locally instead.

---

## Running the App

```bash
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

**To stop:** Press `Ctrl + C` in the terminal.

---

## How to Use

### Page 1 — Home
- Displays an overview of the application.
- Shows a list of your previously completed sessions.
- Use the sidebar to navigate between pages.

### Page 2 — Live Study Session
1. Click **Start Session** to begin monitoring.
2. Allow browser/system access to your webcam if prompted.
3. The webcam feed appears with colored overlays showing:
   - Face detection mesh
   - Gaze direction indicator
   - Phone detection bounding boxes (if detected)
4. The behavioral state badge updates in real time:
   - 🟢 **FOCUSED** — face visible, gaze centered
   - 🔴 **DISTRACTED - PHONE** — phone detected in frame
   - 🟡 **DISTRACTED - LOOKING AWAY** — gaze away from screen
   - ⚪ **AWAY FROM DESK** — no face for 5+ seconds
5. Live metrics (session time, focus %, distraction counts) update every 100 ms.
6. Click **End Session** to save and stop monitoring.

### Page 3 — Session Analytics
- Select any completed session from the dropdown.
- View four interactive charts:
  - **Donut chart** — time split across all four states
  - **Distraction bar chart** — event counts per distraction type
  - **Behavioral timeline** — Gantt chart of when each state was active
  - **Focus trend** — focus score across all sessions (needs ≥ 2 sessions)

### Page 4 — AI Study Report
- Select a session from the dropdown.
- Click **Generate Report**.
- If a Gemini API key is configured: a 7-section AI report is generated.
- If no key: a locally generated template report appears.
- Generated reports are saved and reloaded instantly on future visits.

---

## Testing

The file `test_classifier.py` contains standalone unit tests for the behavioral classifier and session manager. **No webcam is needed** — all tests run on simulated inputs.

```bash
python test_classifier.py
```

Expected output:
```
Running AI Study Focus Coach unit tests...
============================================================
Test 1 PASSED: Initial state is FOCUSED
Test 2 PASSED: Phone detection → DISTRACTED - PHONE (priority override)
Test 3 PASSED: Temporal smoothing — 4 frames not enough to change state
Test 4 PASSED: Temporal smoothing — 5 frames triggers state change
Test 5 PASSED: State priority — phone overrides looking away
Test 6 PASSED: Face returns → resets no_face timer
Test 7 PASSED: start_session() creates correct structure
Test 8 PASSED: update_session() accumulates durations correctly
Test 9 PASSED: focus_percentage calculation is correct
Test 10 PASSED: Longest focus period is tracked correctly
Test 11 PASSED: Session CSV round-trip save and load
============================================================
All 11 tests passed.
```

---

## Limitations

| Limitation | Explanation |
|---|---|
| Gaze estimation is approximate | Iris position relative to eye width is a simplified metric. It cannot determine what the student is reading or looking at precisely. |
| Lighting-dependent | Poor lighting, backlighting, or glare on glasses can reduce face detection accuracy. |
| Single-face only | The classifier assumes one person in the frame. Multiple faces may cause unexpected behavior. |
| No audio analysis | Sounds (music, talking, notifications) that distract the student are not detected. |
| Phone must be visible | If the phone is below the camera's field of view, it will not be detected. |
| CPU-only processing | On slower machines, the frame rate may drop to 5–10 FPS, reducing responsiveness. |
| Blink handling | Very long blinks or briefly looking down at a book may briefly trigger "looking away." The 5-frame temporal smoothing reduces this but does not eliminate it. |
| No scientific validation | The focus score is based on computer vision heuristics, not validated psychometric measurement. |

---

## Future Enhancements

- **Pomodoro timer integration** — built-in work/break timer with automatic session segmentation
- **Sound detection** — microphone-based distraction detection (notification sounds, music)
- **Posture estimation** — MediaPipe Pose to detect slouching or forward head posture
- **Multi-day streak tracking** — gamification with study streaks and goals
- **Export to PDF** — downloadable session reports with embedded charts
- **Custom distraction thresholds** — user-configurable gaze and phone confidence thresholds
- **Cloud sync** — optional sync of session history across devices
- **Mobile app** — React Native or Flutter frontend connected to a lightweight Python backend

---

## Project Information

| Field | Detail |
|---|---|
| Project name | AI Study Focus Coach |
| Course | Computer Vision / Deep Learning / Final Year Project |
| Language | Python 3.11 |
| Total Python lines | ~4,000 (across 9 files) |
| External models used | MediaPipe FaceLandmarker (float16), YOLOv8n (COCO pretrained) |
| Custom model training | None — only pretrained models |
| GPU required | No — CPU-only operation |

---

*Built for educational and personal productivity purposes. Not intended for surveillance or validated medical attention assessment.*
