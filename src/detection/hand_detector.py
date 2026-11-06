"""
Hand detection module supporting both MediaPipe Solutions (legacy) and MediaPipe Tasks APIs.
Processes video frames to detect hands and extract 21 landmarks and handedness.
"""

import os
from typing import Any, Optional
import cv2
import mediapipe as mp
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("HandDetector")


class HandDetector:
    """
    Detects hands in video frames using MediaPipe.
    Automatically adapts to MediaPipe Solutions (legacy) or MediaPipe Tasks (v1.0+).
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        model_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the MediaPipe Hand detector.

        Args:
            static_image_mode: Whether to treat input as static images (True) or video stream (False).
            max_num_hands: Maximum number of hands to detect (default: 2).
            min_detection_confidence: Minimum confidence threshold for detection [0.0, 1.0].
            min_tracking_confidence: Minimum confidence threshold for tracking [0.0, 1.0].
            model_complexity: MediaPipe landmark model complexity (0 or 1).
            model_path: Optional path to hand_landmarker.task model file (for MediaPipe Tasks API).
        """
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity

        self.backend = "unknown"
        self.hands = None
        self.landmarker = None

        logger.info(
            f"Initializing HandDetector (max_hands={max_num_hands}, "
            f"min_detect_conf={min_detection_confidence}, min_track_conf={min_tracking_confidence})"
        )

        # 1. Check for legacy mp.solutions.hands
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.backend = "solutions"
            logger.info("Using MediaPipe Solutions API backend.")
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=self.static_image_mode,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                model_complexity=self.model_complexity,
            )
        # 2. Use MediaPipe Tasks API (MediaPipe 1.0+)
        else:
            self.backend = "tasks"
            logger.info("Using MediaPipe Tasks Vision API backend.")
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            if model_path is None:
                # Default path relative to this file
                base_dir = os.path.dirname(os.path.abspath(__file__))
                candidate = os.path.join(base_dir, "hand_landmarker.task")
                if os.path.exists(candidate):
                    model_path = candidate
                elif os.path.exists("src/detection/hand_landmarker.task"):
                    model_path = "src/detection/hand_landmarker.task"
                else:
                    model_path = "hand_landmarker.task"

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"MediaPipe task model file not found at: {model_path}. "
                    "Please ensure hand_landmarker.task is present."
                )

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray) -> Optional[Any]:
        """
        Process an image/frame and detect hand landmarks.

        Args:
            frame: BGR image from OpenCV (shape: HxWxC).

        Returns:
            Detection results object containing hand landmarks and handedness,
            or None if frame is invalid.
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty or invalid frame passed to HandDetector.detect()")
            return None

        # Convert BGR (OpenCV) to RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.backend == "solutions" and self.hands is not None:
            rgb_frame.flags.writeable = False
            results = self.hands.process(rgb_frame)
            return results
        elif self.backend == "tasks" and self.landmarker is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.landmarker.detect(mp_image)
            return results
        else:
            logger.error("No active detection backend initialized.")
            return None

    def close(self) -> None:
        """Safely release MediaPipe resources."""
        if self.hands is not None:
            logger.info("Closing MediaPipe Hands instance.")
            try:
                self.hands.close()
            except Exception as e:
                logger.error(f"Error while closing MediaPipe Hands: {e}")
            finally:
                self.hands = None

        if self.landmarker is not None:
            logger.info("Closing MediaPipe HandLandmarker instance.")
            try:
                self.landmarker.close()
            except Exception as e:
                logger.error(f"Error while closing HandLandmarker: {e}")
            finally:
                self.landmarker = None

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
