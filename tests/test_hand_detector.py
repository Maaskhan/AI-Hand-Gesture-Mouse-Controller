"""
Unit tests for HandDetector module with mocked MediaPipe backends.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.detection.hand_detector import HandDetector


def test_hand_detector_detect_invalid_frame():
    """Verify detector handles None or empty frames gracefully without exceptions."""
    detector = HandDetector()
    assert detector.detect(None) is None
    assert detector.detect(np.array([])) is None
    detector.close()


@patch("mediapipe.tasks.python.vision.HandLandmarker.create_from_options")
def test_hand_detector_tasks_backend_detect(mock_create):
    """Verify detector converts frame to MediaPipe Image and invokes landmarker.detect."""
    mock_landmarker = MagicMock()
    mock_results = MagicMock()
    mock_landmarker.detect.return_value = mock_results
    mock_create.return_value = mock_landmarker

    detector = HandDetector()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    results = detector.detect(dummy_frame)
    assert results == mock_results
    mock_landmarker.detect.assert_called_once()

    detector.close()
    mock_landmarker.close.assert_called_once()


@patch("mediapipe.tasks.python.vision.HandLandmarker.create_from_options")
def test_hand_detector_context_manager(mock_create):
    """Verify context manager automatically closes the landmarker."""
    mock_landmarker = MagicMock()
    mock_create.return_value = mock_landmarker

    with HandDetector() as detector:
        assert detector is not None

    mock_landmarker.close.assert_called_once()


def test_hand_detector_missing_model_raises_error():
    """Verify that a non-existent model path triggers FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        HandDetector(model_path="definitely_non_existent_file.task")
