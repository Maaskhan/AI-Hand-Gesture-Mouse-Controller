# AI Hand Gesture Mouse Controller

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0%2B-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-1.2%2B-orange.svg)](https://developers.google.com/mediapipe)
[![Pytest](https://img.shields.io/badge/Tested%20With-Pytest-yellow.svg)](https://pytest.org/)

A professional, modular, real-time Computer Vision system designed to track hands and recognize gestures using a standard laptop webcam.

> **Project Phase:** **Phase 1 — Real-Time Hand Detection & Tracking Engine**  
> *Phase 1 delivers the low-latency computer vision foundation without mouse automation or external hardware dependencies.*

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features (Phase 1)](#key-features-phase-1)
- [Technology Stack](#technology-stack)
- [Project Architecture](#project-architecture)
- [Installation Guide](#installation-guide)
- [Execution Instructions](#execution-instructions)
- [Configuration Guide](#configuration-guide)
- [Automated Testing](#automated-testing)
- [Troubleshooting](#troubleshooting)
- [Phase 1 Limitations](#phase-1-limitations)
- [Future Development Roadmap](#future-development-roadmap)
- [License](#license)

---

## Project Overview

The **AI Hand Gesture Mouse Controller** project aims to build a touchless computer interaction system that translates natural hand gestures into operating system mouse events (clicks, scrolls, navigation).

**Phase 1** focuses on building a robust, production-ready computer vision core:
- Real-time video capture via laptop webcam (device index 0).
- Hand localization and extraction of all 21 3D landmarks per hand.
- Left vs. Right hand identification.
- Pixel coordinate mapping and state tracking.
- Interactive Heads-Up Display (HUD) showing live FPS, detection status, and recent event history.
- Safe resource management and lifecycle cleanup.

---

## Key Features (Phase 1)

1. **Hardware Capture & Safety:**
   - Automatically connects to built-in laptop webcam (`index: 0`).
   - Verifies camera availability before starting.
   - Graceful hardware release upon exit using Python context managers.

2. **Hand Detection:**
   - Leverages Google MediaPipe Hands model.
   - Detects up to 2 hands concurrently.
   - Identifies all **21 hand landmarks** (Wrist, Thumb, Index, Middle, Ring, Pinky joints & tips).
   - Classifies **Left** and **Right** hands with confidence scores.

3. **Coordinate Tracking & Transformation:**
   - Translates normalized `[0.0, 1.0]` coordinates to exact image pixel space.
   - Clamps coordinates to prevent boundary overflow errors.
   - Computes dynamic bounding boxes around detected hands.

4. **Visualization & HUD:**
   - Renders skeletal connections and landmark joints with high-visibility color coding.
   - Highlights fingertips and wrists with distinct accent markers.
   - Semi-transparent top HUD showing **Application Title**, **Hand Count**, **Detection Status**, and **Smoothed FPS**.
   - Keyboard hint overlay (`Press 'Q' or 'ESC' to exit`).

5. **Detection Event History:**
   - Real-time state machine tracks hand transitions (`HAND_DETECTED` and `HAND_LOST`).
   - Maintains an in-memory chronological event buffer (configured to latest 5 events).
   - Renders an on-screen semi-transparent card displaying timestamped event logs.

6. **Keyboard Controls:**
   - `Q` / `q`: Safely close the application and destroy OpenCV windows.
   - `ESC`: Safely close the application and destroy OpenCV windows.

7. **Configurability:**
   - Fully externalized configuration via `configs/config.yaml`.
   - Modifiable camera resolution, target FPS, confidence thresholds, and UI options.

8. **Enterprise Logging:**
   - Timestamped file logging into `logs/app.log`.
   - Simultaneous console output with log levels (`INFO`, `WARNING`, `ERROR`).

---

## Technology Stack

| Technology | Purpose |
| :--- | :--- |
| **Python 3.11 / 3.12** | Core programming language |
| **OpenCV (`opencv-python`)** | Video capture, frame preprocessing, UI rendering, display loop |
| **MediaPipe (`mediapipe`)** | High-fidelity machine learning pipeline for hand landmark extraction |
| **NumPy (`numpy`)** | Efficient matrix operations and coordinate conversions |
| **PyYAML (`pyyaml`)** | Structured configuration loading and schema validation |
| **Pytest (`pytest`)** | Automated testing with mocked hardware & synthetic inputs |

---

## Project Architecture

```
AI-Hand-Gesture-Mouse-Controller/
│
├── app/                                # Application layer
│   ├── __init__.py
│   ├── main.py                         # Application runner & main CV loop
│   └── camera/
│       ├── __init__.py
│       └── webcam.py                   # Camera hardware wrapper & frame reader
│
├── configs/                            # Configuration files
│   └── config.yaml                     # System parameters & thresholds
│
├── docs/                               # Documentation & specs
│   └── development_plan.md             # Multi-phase roadmap
│
├── logs/                               # Runtime log outputs
│   ├── .gitkeep
│   └── app.log
│
├── screenshots/                        # Application demo captures
│   └── .gitkeep
│
├── src/                                # Core domain logic
│   ├── __init__.py
│   ├── detection/
│   │   ├── __init__.py
│   │   └── hand_detector.py            # MediaPipe Hands integration
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── landmark_tracker.py         # Coordinate mapping & event history
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config_loader.py            # YAML configuration loader
│   │   └── logger.py                   # Rotating / file logger
│   └── visualization/
│       ├── __init__.py
│       └── draw_landmarks.py           # HUD, skeleton & overlay renderer
│
├── tests/                              # Pytest test suite
│   ├── __init__.py
│   ├── test_hand_detector.py           # Detector mock unit tests
│   ├── test_landmark_tracker.py        # Coordinate & event tracking unit tests
│   └── test_webcam.py                  # Camera lifecycle mock tests
│
├── .gitignore                          # Git ignore rules
├── README.md                           # Documentation
├── requirements.txt                    # Project dependencies
└── run.py                              # Top-level executable script
```

---

## Installation Guide

### Prerequisites
- Python 3.11 or Python 3.12 installed on your system.
- Built-in laptop webcam or integrated camera.

### Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd AI-Hand-Gesture-Mouse-Controller
   ```

2. **Create and activate a virtual environment (recommended):**
   - **Windows:**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

---

## Execution Instructions

Launch the application directly using the top-level launcher:

```bash
python run.py
```

Or execute via the module entry point:

```bash
python -m app.main
```

### Controls:
- **`Q`** or **`ESC`**: Exit the application cleanly.

---

## Configuration Guide

Settings can be customized inside [`configs/config.yaml`](configs/config.yaml):

```yaml
camera:
  index: 0                    # Default 0 for laptop webcam
  width: 1280                 # Frame resolution width
  height: 720                 # Frame resolution height
  fps: 30                     # Target frame rate

detection:
  max_num_hands: 2            # Maximum hands to detect (1 or 2)
  min_detection_confidence: 0.7  # Initial hand detection threshold [0.0 - 1.0]
  min_tracking_confidence: 0.5   # Landmark tracking threshold [0.0 - 1.0]
  model_complexity: 1         # 0 = Light, 1 = Full

visualization:
  app_title: "AI Hand Gesture System (Phase 1: Detection & Tracking)"
  show_fps: true              # Display real-time FPS
  show_landmarks: true        # Draw 21 landmark points
  show_connections: true      # Draw skeletal connections
  show_hand_status: true      # Show hand count and status
  show_event_history: true    # Show recent event log card
  event_history_length: 5     # Number of events retained

logging:
  level: "INFO"               # DEBUG, INFO, WARNING, ERROR
  log_dir: "logs"
  log_file: "app.log"
  console_output: true
```

---

## Automated Testing

The project includes unit tests with mocked webcam hardware and synthetic landmark data. Tests can run in headless CI/CD environments without requiring a physical camera.

Run the entire test suite:
```bash
python -m pytest
```

Run with verbose output:
```bash
python -m pytest -v
```

---

## Troubleshooting

1. **"Webcam at index 0 is unavailable or in use":**
   - Check if another application (Zoom, Teams, Skype, Windows Camera) is using the webcam.
   - On Windows, verify camera privacy permissions under **Windows Settings > Privacy & Security > Camera** ("Let desktop apps access your camera").
   - If using a secondary webcam, change `camera.index` in `configs/config.yaml` to `1`.

2. **Low Frame Rate (FPS < 20):**
   - In `configs/config.yaml`, lower the capture resolution to `width: 640`, `height: 480`.
   - Change `model_complexity: 0` for reduced computational load.

3. **Hand Not Detected:**
   - Ensure adequate room lighting.
   - Keep your hand within 0.3m – 1.5m from the camera.
   - Lower `min_detection_confidence` to `0.5` in `configs/config.yaml`.

---

## Phase 1 Limitations

In accordance with project scope:
- **No Mouse Control:** Cursor movement and OS clicks are strictly reserved for Phase 3.
- **No Gesture Classification:** Pinch, click, and swipe gestures will be implemented in Phase 2.
- **No Face Recognition:** Face detection/identification is intentionally excluded.
- **Single Stream:** Only one camera device (index 0) is processed at a time.

---

## Future Development Roadmap

- **Phase 2: Gesture Recognition Engine**
  - Finger extension heuristics and distance metrics.
  - Multi-gesture catalog: Open Palm, Fist, Pinch, Peace, Thumbs Up.
  - Temporal debounce filtering to prevent accidental triggers.
- **Phase 3: Mouse Automation & Virtual Control**
  - Smooth cursor tracking via Kalman filtering / Exponential Moving Average.
  - Left click, Right click, Drag & Drop, and Two-Finger Scrolling.
  - Coordinate remapping with edge calibration.
- **Phase 4: UI/UX & Packaging**
  - Settings GUI and sensitivity sliders.
  - Multi-threaded asynchronous video capture.
  - One-click installer packaging via PyInstaller.

---

## License
MIT License. Created for AI Hand Gesture Mouse Controller.

