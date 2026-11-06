"""
Visualization module for rendering hand landmarks, skeletal connections,
heads-up display (HUD), performance metrics, and detection event history.
"""

import time
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.tracking.landmark_tracker import (
    HAND_CONNECTIONS,
    HandEvent,
    HandInfo,
)


class HandVisualizer:
    """
    Renders hand landmarks, connections, overlays, and status dashboards on OpenCV frames.
    """

    # Color palette (BGR format)
    COLOR_BG_OVERLAY = (24, 24, 24)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_DIM = (160, 160, 160)
    COLOR_ACCENT = (255, 178, 50)  # Bright Cyan / Blue
    COLOR_SUCCESS = (80, 210, 80)  # Green
    COLOR_WARNING = (0, 165, 255)  # Orange
    COLOR_RIGHT_HAND = (255, 191, 0)  # Deep Sky Blue
    COLOR_LEFT_HAND = (0, 140, 255)  # Amber / Coral
    COLOR_JOINT = (240, 240, 240)
    COLOR_TIP = (0, 255, 255)  # Yellow for fingertips

    def __init__(
        self,
        app_title: str = "AI Hand Gesture System (Phase 1)",
        show_fps: bool = True,
        show_landmarks: bool = True,
        show_connections: bool = True,
        show_hand_status: bool = True,
        show_event_history: bool = True,
    ) -> None:
        """
        Initialize the visualizer.

        Args:
            app_title: Title displayed on HUD.
            show_fps: Whether to render FPS counter.
            show_landmarks: Whether to render landmark points.
            show_connections: Whether to render skeletal connections.
            show_hand_status: Whether to render hand count and status text.
            show_event_history: Whether to render recent event history overlay.
        """
        self.app_title = app_title
        self.show_fps = show_fps
        self.show_landmarks = show_landmarks
        self.show_connections = show_connections
        self.show_hand_status = show_hand_status
        self.show_event_history = show_event_history

        # FPS calculation state
        self._prev_time: float = 0.0
        self._fps: float = 0.0

    def calculate_fps(self) -> float:
        """Calculate smoothed Frames Per Second."""
        current_time = time.time()
        if self._prev_time == 0.0:
            self._prev_time = current_time
            return 0.0

        delta = current_time - self._prev_time
        self._prev_time = current_time

        if delta > 0:
            instant_fps = 1.0 / delta
            # Simple exponential smoothing
            self._fps = 0.85 * self._fps + 0.15 * instant_fps if self._fps > 0 else instant_fps

        return self._fps

    def draw_hand_skeleton(self, frame: np.ndarray, hand: HandInfo) -> None:
        """
        Draw landmarks, connections, and bounding box for a single hand.

        Args:
            frame: OpenCV image frame (modified in-place).
            hand: Structured HandInfo containing 21 landmark points.
        """
        conn_color = (
            self.COLOR_RIGHT_HAND if hand.handedness == "Right" else self.COLOR_LEFT_HAND
        )

        # 1. Draw skeletal connections
        if self.show_connections:
            for p1_idx, p2_idx in HAND_CONNECTIONS:
                if p1_idx < len(hand.pixel_landmarks) and p2_idx < len(hand.pixel_landmarks):
                    pt1 = hand.pixel_landmarks[p1_idx]
                    pt2 = hand.pixel_landmarks[p2_idx]
                    cv2.line(frame, pt1, pt2, conn_color, 2, cv2.LINE_AA)

        # 2. Draw landmark points
        if self.show_landmarks:
            tip_indices = {4, 8, 12, 16, 20}  # Fingertips
            for lm in hand.landmarks:
                pt = (lm.x, lm.y)
                if lm.id in tip_indices:
                    # Fingertip: larger circle with contrasting outline
                    cv2.circle(frame, pt, 6, self.COLOR_TIP, -1, cv2.LINE_AA)
                    cv2.circle(frame, pt, 7, (0, 0, 0), 1, cv2.LINE_AA)
                elif lm.id == 0:
                    # Wrist: distinctive marker
                    cv2.circle(frame, pt, 6, (0, 200, 255), -1, cv2.LINE_AA)
                else:
                    # Finger joints
                    cv2.circle(frame, pt, 4, self.COLOR_JOINT, -1, cv2.LINE_AA)
                    cv2.circle(frame, pt, 5, conn_color, 1, cv2.LINE_AA)

        # 3. Draw bounding box and handedness label
        bx, by, bw, bh = hand.bbox
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), conn_color, 1, cv2.LINE_AA)
        label_text = f"{hand.handedness} ({int(hand.confidence * 100)}%)"

        # Label background
        (tw, th), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_y = max(18, by - 5)
        cv2.rectangle(
            frame,
            (bx, label_y - th - 4),
            (bx + tw + 6, label_y + 2),
            conn_color,
            -1,
        )
        cv2.putText(
            frame,
            label_text,
            (bx + 3, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    def draw_hud(
        self,
        frame: np.ndarray,
        hand_count: int,
        fps: float,
        events: List[HandEvent],
    ) -> None:
        """
        Render dashboard overlays: top title bar, stats, and event history log.

        Args:
            frame: OpenCV image frame (modified in-place).
            hand_count: Number of hands currently detected.
            fps: Current frame rate.
            events: List of recent detection events.
        """
        h, w = frame.shape[:2]

        # --- Top HUD Header Bar ---
        header_h = 52
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), self.COLOR_BG_OVERLAY, -1)
        # Semi-transparent blending
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Bottom accent border for header
        cv2.line(frame, (0, header_h), (w, header_h), (60, 60, 60), 1)

        # Title
        cv2.putText(
            frame,
            self.app_title,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            self.COLOR_TEXT,
            2,
            cv2.LINE_AA,
        )

        # Status & Hands Count
        if self.show_hand_status:
            status_text = (
                f"Hands: {hand_count} detected"
                if hand_count > 0
                else "Hands: 0 (Searching...)"
            )
            status_color = self.COLOR_SUCCESS if hand_count > 0 else self.COLOR_WARNING
            cv2.putText(
                frame,
                status_text,
                (w - 380, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                status_color,
                1,
                cv2.LINE_AA,
            )

        # FPS Display
        if self.show_fps:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                frame,
                fps_text,
                (w - 120, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                self.COLOR_ACCENT,
                2,
                cv2.LINE_AA,
            )

        # --- Footer Help Bar ---
        footer_y = h - 16
        cv2.putText(
            frame,
            "Press 'Q' or 'ESC' to exit",
            (16, footer_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            self.COLOR_TEXT_DIM,
            1,
            cv2.LINE_AA,
        )

        # --- Event History Overlay (Bottom Right Card) ---
        if self.show_event_history and events:
            card_w = 260
            line_h = 20
            card_h = 28 + (len(events) * line_h)
            card_x = w - card_w - 16
            card_y = h - card_h - 20

            # Background box
            card_overlay = frame.copy()
            cv2.rectangle(
                card_overlay,
                (card_x, card_y),
                (card_x + card_w, card_y + card_h),
                self.COLOR_BG_OVERLAY,
                -1,
            )
            cv2.addWeighted(card_overlay, 0.70, frame, 0.30, 0, frame)
            cv2.rectangle(
                frame,
                (card_x, card_y),
                (card_x + card_w, card_y + card_h),
                (70, 70, 70),
                1,
            )

            # Card Header
            cv2.putText(
                frame,
                "Event History (Latest)",
                (card_x + 10, card_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                self.COLOR_ACCENT,
                1,
                cv2.LINE_AA,
            )

            # Event rows
            for i, ev in enumerate(events):
                row_y = card_y + 36 + (i * line_h)
                ev_color = (
                    self.COLOR_SUCCESS
                    if ev.event_type == "HAND_DETECTED"
                    else self.COLOR_WARNING
                )
                bullet_text = f"[{ev.timestamp}] {ev.message}"
                cv2.putText(
                    frame,
                    bullet_text,
                    (card_x + 10, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    ev_color,
                    1,
                    cv2.LINE_AA,
                )

    def render(
        self,
        frame: np.ndarray,
        hands: List[HandInfo],
        events: List[HandEvent],
    ) -> np.ndarray:
        """
        Apply complete visualization pipeline to the frame.

        Args:
            frame: OpenCV image frame.
            hands: Structured hand tracking data.
            events: Recent event history.

        Returns:
            np.ndarray: Rendered frame with all overlays.
        """
        fps = self.calculate_fps()

        # Render each detected hand
        for hand in hands:
            self.draw_hand_skeleton(frame, hand)

        # Render HUD, title, FPS, and event history
        self.draw_hud(frame, len(hands), fps, events)

        return frame

