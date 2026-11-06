"""
Unit tests for webcam management module using mocked OpenCV VideoCapture.
"""

from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from app.camera.webcam import Webcam


def test_webcam_default_properties():
    """Verify default webcam configuration parameters."""
    cam = Webcam()
    assert cam.camera_index == 0
    assert cam.width == 1280
    assert cam.height == 720
    assert cam.fps == 30
    assert not cam.is_opened()


@patch("cv2.VideoCapture")
def test_webcam_start_success(mock_videocapture_cls):
    """Verify successful webcam initialization with resolution and fps configuration."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_WIDTH: 1280.0,
        cv2.CAP_PROP_FRAME_HEIGHT: 720.0,
        cv2.CAP_PROP_FPS: 30.0,
    }.get(prop, 0.0)

    mock_videocapture_cls.return_value = mock_cap

    cam = Webcam(camera_index=0, width=1280, height=720, fps=30)
    cam.start()

    assert cam.is_opened()
    mock_videocapture_cls.assert_called_once_with(0)
    mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    mock_cap.set.assert_any_call(cv2.CAP_PROP_FPS, 30)

    w, h = cam.get_actual_resolution()
    assert w == 1280
    assert h == 720


@patch("cv2.VideoCapture")
def test_webcam_start_failure_raises_runtime_error(mock_videocapture_cls):
    """Verify that unavailable camera triggers a descriptive RuntimeError."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_videocapture_cls.return_value = mock_cap

    cam = Webcam(camera_index=99)
    with pytest.raises(RuntimeError) as exc_info:
        cam.start()

    assert "unavailable or in use" in str(exc_info.value)
    assert not cam.is_opened()


@patch("cv2.VideoCapture")
def test_webcam_read_frame_success(mock_videocapture_cls):
    """Verify reading a valid frame from the active stream."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, dummy_frame)
    mock_videocapture_cls.return_value = mock_cap

    cam = Webcam()
    cam.start()

    success, frame = cam.read_frame()
    assert success is True
    assert frame is not None
    assert frame.shape == (720, 1280, 3)


def test_webcam_read_frame_when_closed():
    """Verify that reading from an uninitialized camera safely returns (False, None)."""
    cam = Webcam()
    success, frame = cam.read_frame()
    assert success is False
    assert frame is None


@patch("cv2.VideoCapture")
def test_webcam_release(mock_videocapture_cls):
    """Verify safe hardware release and cleanup."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_videocapture_cls.return_value = mock_cap

    cam = Webcam()
    cam.start()
    assert cam.is_opened()

    cam.release()
    mock_cap.release.assert_called_once()
    assert not cam.is_opened()


@patch("cv2.VideoCapture")
def test_webcam_context_manager(mock_videocapture_cls):
    """Verify context manager automatically starts and releases resources."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_videocapture_cls.return_value = mock_cap

    with Webcam() as cam:
        assert cam.is_opened()

    mock_cap.release.assert_called_once()
    assert not cam.is_opened()

