"""
Main application entry point for AI Hand Gesture Mouse Controller (Phase 1).
Orchestrates webcam capture, hand detection, landmark tracking, HUD rendering,
and safe shutdown.
"""

import sys
from typing import Optional
import cv2

from app.camera.webcam import Webcam
from src.detection.hand_detector import HandDetector
from src.tracking.landmark_tracker import LandmarkTracker
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.visualization.draw_landmarks import HandVisualizer


def run_app(config_path: str = "configs/config.yaml") -> int:
    """
    Run the Phase 1 Real-time Hand Detection and Tracking application loop.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        int: Exit status code (0 for clean exit, non-zero for errors).
    """
    # 1. Load Configuration
    config = load_config(config_path)

    log_cfg = config.get("logging", {})
    logger = get_logger(
        name="HandGestureApp",
        log_dir=log_cfg.get("log_dir", "logs"),
        log_file=log_cfg.get("log_file", "app.log"),
        level=log_cfg.get("level", "INFO"),
        console_output=log_cfg.get("console_output", True),
    )

    logger.info("==================================================")
    logger.info("Starting AI Hand Gesture Controller - Phase 1")
    logger.info("==================================================")

    camera_cfg = config.get("camera", {})
    detect_cfg = config.get("detection", {})
    vis_cfg = config.get("visualization", {})

    window_name = vis_cfg.get(
        "app_title", "AI Hand Gesture System (Phase 1: Detection & Tracking)"
    )

    webcam: Optional[Webcam] = None
    detector: Optional[HandDetector] = None

    try:
        # 2. Initialize Camera
        webcam = Webcam(
            camera_index=camera_cfg.get("index", 0),
            width=camera_cfg.get("width", 1280),
            height=camera_cfg.get("height", 720),
            fps=camera_cfg.get("fps", 30),
        )
        webcam.start()

        # 3. Initialize MediaPipe Hand Detector
        detector = HandDetector(
            static_image_mode=False,
            max_num_hands=detect_cfg.get("max_num_hands", 2),
            min_detection_confidence=float(
                detect_cfg.get("min_detection_confidence", 0.7)
            ),
            min_tracking_confidence=float(
                detect_cfg.get("min_tracking_confidence", 0.5)
            ),
            model_complexity=int(detect_cfg.get("model_complexity", 1)),
        )

        # 4. Initialize Landmark Tracker & Visualizer
        tracker = LandmarkTracker(
            max_history=vis_cfg.get("event_history_length", 5)
        )
        visualizer = HandVisualizer(
            app_title=window_name,
            show_fps=vis_cfg.get("show_fps", True),
            show_landmarks=vis_cfg.get("show_landmarks", True),
            show_connections=vis_cfg.get("show_connections", True),
            show_hand_status=vis_cfg.get("show_hand_status", True),
            show_event_history=vis_cfg.get("show_event_history", True),
        )

        # 5. Setup Display Window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        act_w, act_h = webcam.get_actual_resolution()
        if act_w > 0 and act_h > 0:
            cv2.resizeWindow(window_name, act_w, act_h)

        logger.info("Entering live video processing loop. Press 'Q' or 'ESC' to exit.")

        # 6. Real-time Processing Loop
        while webcam.is_opened():
            success, frame = webcam.read_frame()
            if not success or frame is None:
                logger.warning("Empty frame received from webcam. Retrying...")
                continue

            # Mirror flip frame horizontally for natural selfie interaction
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Detect hand landmarks via MediaPipe
            results = detector.detect(frame)

            # Process structured hand data and update event transitions
            hands = tracker.process_landmarks(results, w, h)
            events = tracker.get_recent_events()

            # Render visual overlays, landmarks, and dashboard
            rendered_frame = visualizer.render(frame, hands, events)

            # Display window
            cv2.imshow(window_name, rendered_frame)

            # Keyboard controls: 'q', 'Q', or ESC (code 27)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                logger.info(f"Exit command received (key code {key}). Shutting down gracefully...")
                break

        return 0

    except KeyboardInterrupt:
        logger.info("Application interrupted by user (KeyboardInterrupt).")
        return 0
    except Exception as e:
        logger.error(f"Fatal application error encountered: {e}", exc_info=True)
        return 1
    finally:
        logger.info("Cleaning up resources...")
        if webcam is not None:
            webcam.release()
        if detector is not None:
            detector.close()
        cv2.destroyAllWindows()
        logger.info("Application terminated safely.")


if __name__ == "__main__":
    sys.exit(run_app())

