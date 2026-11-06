# AI Hand Gesture Mouse Controller - Development Plan

## Project Overview
The **AI Hand Gesture Mouse Controller** is a real-time computer vision system that enables touchless, gesture-driven human-computer interaction using a standard laptop webcam.

The development is divided into four distinct phases to ensure reliability, performance, modularity, and comprehensive test coverage.

---

## Architecture Pipeline

```text
[Webcam Stream (index 0)]
           │
           ▼
[Frame Preprocessing (Resize, Mirror Flip, RGB)]
           │
           ▼
[MediaPipe Hand Detection & Landmark Extraction]
           │
           ▼
[Landmark Tracker (Pixel Mapping, Handedness, Event Tracking)]
           │
           ▼
[Visualizer (Landmark Overlay, HUD, Status, FPS, Event Log)]
           │
           ▼
[OpenCV Display Window & Safe Event Loop]
```

---

## Phase Roadmap

### Phase 1: Real-Time Hand Detection & Tracking (Current Phase)
**Goal:** Deliver a rock-solid, low-latency hand detection and tracking pipeline with real-time visual feedback and state tracking.

- **Webcam Engine (`app/camera/webcam.py`):**
  - Robust hardware detection and graceful exception handling.
  - Automatic selection of camera index 0 (configurable via YAML).
  - Clean frame capture and safe release mechanics via context managers.
- **Hand Detection Pipeline (`src/detection/hand_detector.py`):**
  - MediaPipe Hands integration detecting up to 2 hands.
  - Identification of 21 3D landmarks per hand.
  - Multi-handedness detection (Left / Right hand classification).
- **Landmark Tracking & Normalization (`src/tracking/landmark_tracker.py`):**
  - Coordinate translation from normalized `[0.0, 1.0]` space to frame pixel coordinates.
  - Structured landmark data models with bounding box calculations.
  - Event history tracking (Hand Detected, Hand Lost) keeping latest 5 occurrences.
- **HUD & Visualization (`src/visualization/draw_landmarks.py`):**
  - Skeletal hand connection rendering and landmark points.
  - Real-time FPS counter, detected hand count, and tracking status.
  - On-screen event history overlay and application branding.
- **Configuration & Logging (`src/utils/`):**
  - Externalized configuration via `configs/config.yaml`.
  - Structured file logging in `logs/app.log` and console streaming.
- **Automated Testing Suite (`tests/`):**
  - Unit tests with mocked hardware dependencies for CI/CD compatibility.

---

### Phase 2: Gesture Recognition & State Engine (Future Phase)
**Goal:** Translate landmark positions and finger configurations into discrete gestures.
- Finger extension calculation (angles, Euclidean distances).
- Gesture classification (Index pinch, Palm open, Fist, Peace sign, Thumbs up).
- Gesture debounce filter and temporal smoothing to prevent false positives.
- Gesture state transitions.

---

### Phase 3: Mouse Automation & Virtual Control (Future Phase)
**Goal:** Map gestures to OS-level mouse and keyboard actions.
- Cursor movement smoothing (exponential moving average / Kalman filter).
- Left click (Index pinch / quick tap).
- Right click (Middle finger pinch).
- Drag & Drop (Holding pinch).
- Scroll control (Two-finger vertical translation).
- Volume / Media gesture control.
- Screen coordinate remapping with calibration margin.

---

### Phase 4: UI/UX, Performance Optimization & Packaging (Future Phase)
**Goal:** Polish the user experience and create distributable binaries.
- Settings GUI (PyQt / Tkinter / Web-based dashboard).
- Gesture calibration wizard.
- Threaded asynchronous video pipeline for 60+ FPS performance.
- Standalone executable packaging (PyInstaller).

