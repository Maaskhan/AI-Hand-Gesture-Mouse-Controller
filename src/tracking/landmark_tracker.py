"""
Landmark tracking and processing module.
Converts normalized coordinates to pixel space, structures landmark data,
and maintains hand detection event history.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

# 21 standard MediaPipe Hand Landmarks in order
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
]

# Standard skeletal connections between landmark indices
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    # Index finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    # Middle finger
    (9, 10),
    (10, 11),
    (11, 12),
    # Ring finger
    (13, 14),
    (14, 15),
    (15, 16),
    # Pinky finger
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    # Knuckles / Palm base
    (5, 9),
    (9, 13),
    (13, 17),
]


@dataclass
class LandmarkPoint:
    """Individual hand landmark point with pixel and normalized coordinates."""

    id: int
    name: str
    x: int
    y: int
    z: float
    norm_x: float
    norm_y: float
    norm_z: float


@dataclass
class HandInfo:
    """Complete structured hand detection information."""

    hand_id: int
    handedness: str  # 'Left' or 'Right'
    confidence: float
    landmarks: List[LandmarkPoint]
    pixel_landmarks: List[Tuple[int, int]]
    bbox: Tuple[int, int, int, int]  # (min_x, min_y, width, height)


@dataclass
class HandEvent:
    """Detection lifecycle event (hand detected / lost)."""

    timestamp: str
    event_type: str  # 'HAND_DETECTED' or 'HAND_LOST'
    message: str


class LandmarkTracker:
    """
    Tracks hand landmarks, performs coordinate transformations,
    and maintains historical detection events.
    """

    def __init__(self, max_history: int = 5) -> None:
        """
        Initialize the tracker.

        Args:
            max_history: Maximum number of recent events to retain.
        """
        self.max_history = max_history
        self.event_history: Deque[HandEvent] = deque(maxlen=max_history)
        self.previous_hands: Set[str] = set()

    def convert_normalized_to_pixel(
        self, norm_x: float, norm_y: float, frame_width: int, frame_height: int
    ) -> Tuple[int, int]:
        """
        Convert normalized [0.0, 1.0] coordinates to clamped pixel coordinates.

        Args:
            norm_x: Normalized horizontal coordinate.
            norm_y: Normalized vertical coordinate.
            frame_width: Width of the frame in pixels.
            frame_height: Height of the frame in pixels.

        Returns:
            Tuple[int, int]: (pixel_x, pixel_y) clamped within frame boundaries.
        """
        px = int(round(norm_x * frame_width))
        py = int(round(norm_y * frame_height))
        clamped_x = max(0, min(frame_width - 1, px))
        clamped_y = max(0, min(frame_height - 1, py))
        return clamped_x, clamped_y

    def process_landmarks(
        self, detection_results: Optional[Any], frame_width: int, frame_height: int
    ) -> List[HandInfo]:
        """
        Extract structured hand information from MediaPipe detection results.
        Supports both Solutions API (multi_hand_landmarks) and Tasks API (hand_landmarks).

        Args:
            detection_results: MediaPipe results object.
            frame_width: Frame pixel width.
            frame_height: Frame pixel height.

        Returns:
            List[HandInfo]: List of detected hands with structured 21 landmarks.
        """
        hands: List[HandInfo] = []
        current_hands: Set[str] = set()

        multi_landmarks = None
        multi_handedness = None
        is_tasks_format = False

        if detection_results is not None:
            if (
                hasattr(detection_results, "multi_hand_landmarks")
                and detection_results.multi_hand_landmarks
            ):
                multi_landmarks = detection_results.multi_hand_landmarks
                multi_handedness = getattr(detection_results, "multi_handedness", None)
                is_tasks_format = False
            elif (
                hasattr(detection_results, "hand_landmarks")
                and detection_results.hand_landmarks
            ):
                multi_landmarks = detection_results.hand_landmarks
                multi_handedness = getattr(detection_results, "handedness", None)
                is_tasks_format = True

        if multi_landmarks:
            for idx, raw_hand in enumerate(multi_landmarks):
                # Extract handedness and confidence score
                handedness_label = "Unknown"
                confidence = 1.0

                if multi_handedness and idx < len(multi_handedness):
                    item = multi_handedness[idx]
                    if is_tasks_format:
                        category = item[0] if isinstance(item, list) and item else item
                        handedness_label = (
                            getattr(category, "category_name", None)
                            or getattr(category, "display_name", None)
                            or "Unknown"
                        )
                        confidence = float(getattr(category, "score", 1.0))
                    else:
                        classification = item.classification[0]
                        handedness_label = classification.label
                        confidence = float(classification.score)

                # Ensure unique key if two hands of same label detected
                hand_key = f"{handedness_label}_{idx}"
                current_hands.add(hand_key)

                landmarks_list: List[LandmarkPoint] = []
                pixel_coords: List[Tuple[int, int]] = []

                xs: List[int] = []
                ys: List[int] = []

                raw_points = (
                    raw_hand.landmark if hasattr(raw_hand, "landmark") else raw_hand
                )

                for l_idx, lm in enumerate(raw_points):
                    px, py = self.convert_normalized_to_pixel(
                        lm.x, lm.y, frame_width, frame_height
                    )
                    lm_name = (
                        LANDMARK_NAMES[l_idx]
                        if l_idx < len(LANDMARK_NAMES)
                        else f"LM_{l_idx}"
                    )

                    point = LandmarkPoint(
                        id=l_idx,
                        name=lm_name,
                        x=px,
                        y=py,
                        z=float(lm.z),
                        norm_x=float(lm.x),
                        norm_y=float(lm.y),
                        norm_z=float(lm.z),
                    )
                    landmarks_list.append(point)
                    pixel_coords.append((px, py))
                    xs.append(px)
                    ys.append(py)

                # Compute bounding box
                min_x = max(0, min(xs) - 10)
                min_y = max(0, min(ys) - 10)
                max_x = min(frame_width - 1, max(xs) + 10)
                max_y = min(frame_height - 1, max(ys) + 10)
                bbox = (min_x, min_y, max_x - min_x, max_y - min_y)

                hand_info = HandInfo(
                    hand_id=idx,
                    handedness=handedness_label,
                    confidence=confidence,
                    landmarks=landmarks_list,
                    pixel_landmarks=pixel_coords,
                    bbox=bbox,
                )
                hands.append(hand_info)

        # Update event history based on state changes
        self._update_events(current_hands)
        self.previous_hands = current_hands

        return hands

    def _update_events(self, current_hands: Set[str]) -> None:
        """Detect and log hand state transitions (detected vs lost)."""
        time_str = datetime.now().strftime("%H:%M:%S")

        # Hands newly detected
        appeared = current_hands - self.previous_hands
        for hand in sorted(appeared):
            label = hand.split("_")[0]
            self.event_history.append(
                HandEvent(
                    timestamp=time_str,
                    event_type="HAND_DETECTED",
                    message=f"{label} Hand Detected",
                )
            )

        # Hands lost
        lost = self.previous_hands - current_hands
        for hand in sorted(lost):
            label = hand.split("_")[0]
            self.event_history.append(
                HandEvent(
                    timestamp=time_str,
                    event_type="HAND_LOST",
                    message=f"{label} Hand Lost",
                )
            )

    def get_recent_events(self) -> List[HandEvent]:
        """Return a list of recent hand events in chronological order."""
        return list(self.event_history)
