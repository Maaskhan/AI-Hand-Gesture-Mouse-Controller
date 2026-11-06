# AI Hand Gesture Mouse Controller

A real-time computer vision project built with Python, OpenCV, and MediaPipe to detect, track, and visualize hand landmarks through a laptop webcam.

> **Current Status:** Phase 1 — Real-Time Hand Detection & Tracking  
> Mouse automation and gesture classification are planned for future phases.

## Overview

AI Hand Gesture Mouse Controller is a modular computer vision application designed as the foundation for touchless computer interaction.

The current phase focuses on webcam-based hand detection, landmark extraction, tracking, and live visualization.

## Features

- Real-time webcam video capture
- Detection of up to two hands
- Extraction of 21 hand landmarks per hand
- Left and right hand identification
- Landmark coordinate mapping
- Dynamic hand bounding boxes
- Live FPS and detection status display
- Hand detection event history
- Configurable detection thresholds
- Safe webcam resource management
- Automated unit tests using mocked hardware

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.11 / 3.12 | Core development |
| OpenCV | Webcam capture and visualization |
| MediaPipe | Hand landmark detection |
| NumPy | Coordinate processing |
| PyYAML | Configuration management |
| Pytest | Automated testing |

## Project Architecture

```text
AI-Hand-Gesture-Mouse-Controller/
│
├── app/
│   ├── main.py
│   └── camera/
│       └── webcam.py
│
├── configs/
│   └── config.yaml
│
├── docs/
│   └── development_plan.md
│
├── src/
│   ├── detection/
│   │   └── hand_detector.py
│   ├── tracking/
│   │   └── landmark_tracker.py
│   ├── visualization/
│   │   └── draw_landmarks.py
│   └── utils/
│       ├── config_loader.py
│       └── logger.py
│
├── tests/
│   ├── test_hand_detector.py
│   ├── test_landmark_tracker.py
│   └── test_webcam.py
│
├── run.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

### Prerequisites

- Python 3.11 or 3.12
- Laptop webcam
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Maaskhan/AI-Hand-Gesture-Mouse-Controller.git
```

```bash
cd AI-Hand-Gesture-Mouse-Controller
```

### 2. Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
```

```powershell
python -m pip install -r requirements.txt
```

## Run the Application

Start the application:

```powershell
python run.py
```

The application will initialize the webcam and display the hand detection visualization.

### Controls

| Key | Action |
|---|---|
| Q | Exit application |
| ESC | Exit application |

## Configuration

Configuration settings are available in:

`configs/config.yaml`

You can customize:

- Webcam index
- Camera resolution
- Target FPS
- Maximum number of hands
- Detection confidence thresholds
- Visualization options
- Event history length
- Logging settings

## Testing

Run the test suite:

```powershell
python -m pytest
```

Run tests with detailed output:

```powershell
python -m pytest -v
```

The unit tests use mocked webcam hardware and synthetic landmark data.

## Development Roadmap

### Phase 1 — Hand Detection & Tracking
- Webcam integration
- Hand landmark detection
- Landmark visualization
- Coordinate tracking
- Detection event history

### Phase 2 — Gesture Recognition
- Open Palm
- Fist
- Peace Sign
- Thumbs Up
- Pinch Gesture
- Gesture classification and debounce filtering

### Phase 3 — Mouse Automation
- Cursor movement
- Left and right click
- Drag and drop
- Scrolling
- Cursor smoothing

### Phase 4 — GUI & Packaging
- Professional settings interface
- Sensitivity controls
- Application packaging
- Performance optimization

## Limitations

The current implementation is focused on Phase 1.

- Mouse control is not yet implemented.
- Gesture classification is not yet implemented.
- Face detection and face recognition are excluded.
- Only one webcam stream is processed at a time.

## Author

**Maas Khan**

GitHub: [@Maaskhan](https://github.com/Maaskhan)

## License

This project is released under the MIT License.
