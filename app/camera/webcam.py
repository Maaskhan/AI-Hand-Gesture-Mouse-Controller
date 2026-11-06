"""
Webcam management module for capturing live video streams.
Provides safe initialization, frame extraction, resolution adjustment, and resource cleanup.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("Webcam")


class Webcam:
    """
    Manages OpenCV VideoCapture stream with error handling and resource cleanup.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        """
        Initialize the webcam settings.

        Args:
            camera_index: Index of the camera device (default: 0 for built-in laptop webcam).
            width: Desired capture frame width.
            height: Desired capture frame height.
            fps: Desired frame rate.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self) -> "Webcam":
        """
        Open the camera capture and apply resolution/fps properties.

        Returns:
            Webcam: The instance itself for fluent chaining.

        Raises:
            RuntimeError: If the webcam cannot be opened.
        """
        logger.info(f"Initializing webcam at index {self.camera_index}...")

        # Initialize VideoCapture (try standard first, CAP_DSHOW on Windows if available)
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            error_msg = (
                f"Webcam at index {self.camera_index} is unavailable or in use by another application. "
                "Please verify camera permissions and connection."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Apply settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        try:
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        except (TypeError, ValueError):
            actual_w = self.width
            actual_h = self.height
            actual_fps = float(self.fps)

        logger.info(
            f"Webcam started successfully: index={self.camera_index}, "
            f"resolution={actual_w}x{actual_h}, target_fps={actual_fps:.1f}"
        )
        return self

    def is_opened(self) -> bool:
        """Check if webcam capture stream is actively open."""
        return self.cap is not None and self.cap.isOpened()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the webcam.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (True, frame) if successful, (False, None) otherwise.
        """
        if not self.is_opened():
            logger.warning("Attempted to read frame from closed webcam.")
            return False, None

        success, frame = self.cap.read()
        if not success or frame is None:
            logger.warning("Failed to grab frame from webcam stream.")
            return False, None

        return True, frame

    def get_actual_resolution(self) -> Tuple[int, int]:
        """Return actual frame width and height."""
        if not self.is_opened():
            return 0, 0
        try:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        except (TypeError, ValueError):
            return self.width, self.height

    def release(self) -> None:
        """Release the video capture hardware safely."""
        if self.cap is not None:
            logger.info("Releasing webcam resources...")
            try:
                self.cap.release()
            except Exception as e:
                logger.error(f"Error while releasing webcam: {e}")
            finally:
                self.cap = None
            logger.info("Webcam released.")

    def __enter__(self) -> "Webcam":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
