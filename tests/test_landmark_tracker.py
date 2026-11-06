"""
Unit tests for landmark tracking, coordinate conversion, and event history.
"""

from typing import List
import pytest
from src.tracking.landmark_tracker import LandmarkTracker, LANDMARK_NAMES
from src.utils.config_loader import load_config, DEFAULT_CONFIG


class MockLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class MockLandmarkList:
    def __init__(self, landmarks: List[MockLandmark]):
        self.landmark = landmarks


class MockClassification:
    def __init__(self, label: str, score: float):
        self.label = label
        self.score = score


class MockClassificationList:
    def __init__(self, label: str, score: float):
        self.classification = [MockClassification(label, score)]


class MockMediaPipeResults:
    def __init__(self, multi_landmarks=None, multi_handedness=None):
        self.multi_hand_landmarks = multi_landmarks or []
        self.multi_handedness = multi_handedness or []


def test_normalized_to_pixel_conversion_center():
    """Verify coordinate transformation maps normalized coordinates to pixel center."""
    tracker = LandmarkTracker()
    px, py = tracker.convert_normalized_to_pixel(0.5, 0.5, 1000, 800)
    assert px == 500
    assert py == 400


def test_normalized_to_pixel_conversion_boundaries():
    """Verify coordinate transformation clamps within valid frame boundary pixels."""
    tracker = LandmarkTracker()
    # Upper left corner
    px_min, py_min = tracker.convert_normalized_to_pixel(0.0, 0.0, 640, 480)
    assert px_min == 0
    assert py_min == 0

    # Lower right corner
    px_max, py_max = tracker.convert_normalized_to_pixel(1.0, 1.0, 640, 480)
    assert px_max == 639
    assert py_max == 479

    # Out of bounds clamping
    px_clamp, py_clamp = tracker.convert_normalized_to_pixel(-0.5, 1.5, 640, 480)
    assert px_clamp == 0
    assert py_clamp == 479


def test_process_landmarks_empty_results():
    """Verify that empty or None detection results yield no hands and no errors."""
    tracker = LandmarkTracker()
    hands = tracker.process_landmarks(None, 640, 480)
    assert hands == []
    assert len(tracker.get_recent_events()) == 0

    empty_results = MockMediaPipeResults([], [])
    hands2 = tracker.process_landmarks(empty_results, 640, 480)
    assert hands2 == []


def test_process_landmarks_structured_hand():
    """Verify extraction of 21 landmarks, handedness, bounding box, and landmark names."""
    tracker = LandmarkTracker()

    # Generate 21 mock landmarks
    mock_lms = [MockLandmark(x=i * 0.04, y=i * 0.04, z=-0.01 * i) for i in range(21)]
    landmarks_list = MockLandmarkList(mock_lms)
    handedness_list = MockClassificationList(label="Right", score=0.96)

    results = MockMediaPipeResults(
        multi_landmarks=[landmarks_list],
        multi_handedness=[handedness_list],
    )

    hands = tracker.process_landmarks(results, 1000, 1000)

    assert len(hands) == 1
    hand = hands[0]
    assert hand.hand_id == 0
    assert hand.handedness == "Right"
    assert pytest.approx(hand.confidence, 0.01) == 0.96
    assert len(hand.landmarks) == 21
    assert len(hand.pixel_landmarks) == 21

    # Check named landmark mappings
    assert hand.landmarks[0].name == "WRIST"
    assert hand.landmarks[4].name == "THUMB_TIP"
    assert hand.landmarks[8].name == "INDEX_FINGER_TIP"
    assert hand.landmarks[12].name == "MIDDLE_FINGER_TIP"
    assert hand.landmarks[16].name == "RING_FINGER_TIP"
    assert hand.landmarks[20].name == "PINKY_TIP"

    # Check bounding box validity
    bx, by, bw, bh = hand.bbox
    assert bw > 0
    assert bh > 0
    assert bx >= 0
    assert by >= 0


def test_event_history_transitions():
    """Verify that HAND_DETECTED and HAND_LOST events are accurately tracked."""
    tracker = LandmarkTracker(max_history=5)

    mock_lms = [MockLandmark(0.1, 0.1) for _ in range(21)]
    hand_right = MockLandmarkList(mock_lms)
    handedness_right = MockClassificationList("Right", 0.95)

    # Frame 1: Right hand detected
    res1 = MockMediaPipeResults([hand_right], [handedness_right])
    tracker.process_landmarks(res1, 640, 480)

    events1 = tracker.get_recent_events()
    assert len(events1) == 1
    assert events1[0].event_type == "HAND_DETECTED"
    assert "Right Hand Detected" in events1[0].message

    # Frame 2: Same hand continues -> no duplicate transition event
    tracker.process_landmarks(res1, 640, 480)
    events2 = tracker.get_recent_events()
    assert len(events2) == 1

    # Frame 3: Hand removed -> HAND_LOST event
    res3 = MockMediaPipeResults([], [])
    tracker.process_landmarks(res3, 640, 480)
    events3 = tracker.get_recent_events()
    assert len(events3) == 2
    assert events3[1].event_type == "HAND_LOST"
    assert "Right Hand Lost" in events3[1].message


def test_event_history_max_capacity():
    """Verify event history maintains fixed max length (5 events)."""
    tracker = LandmarkTracker(max_history=5)
    mock_lms = [MockLandmark(0.1, 0.1) for _ in range(21)]

    # Cycle hand 6 times to generate 12 events
    for _ in range(6):
        res_appear = MockMediaPipeResults(
            [MockLandmarkList(mock_lms)],
            [MockClassificationList("Left", 0.9)],
        )
        tracker.process_landmarks(res_appear, 640, 480)
        res_disappear = MockMediaPipeResults([], [])
        tracker.process_landmarks(res_disappear, 640, 480)

    events = tracker.get_recent_events()
    assert len(events) == 5


def test_config_loader():
    """Verify config loading and defaults."""
    config = load_config("configs/config.yaml")
    assert "camera" in config
    assert "detection" in config
    assert "visualization" in config
    assert "logging" in config
    assert config["camera"]["index"] == 0
    assert config["detection"]["max_num_hands"] == 2
    assert config["visualization"]["event_history_length"] == 5

    # Non-existent config path should cleanly return defaults
    fallback = load_config("configs/non_existent.yaml")
    assert fallback == DEFAULT_CONFIG

