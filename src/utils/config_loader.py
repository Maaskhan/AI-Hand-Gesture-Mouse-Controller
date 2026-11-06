"""
Configuration loader utility for AI Hand Gesture Mouse Controller.
Loads and validates settings from YAML configuration files.
"""

import os
from typing import Any, Dict
import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "camera": {
        "index": 0,
        "width": 1280,
        "height": 720,
        "fps": 30,
    },
    "detection": {
        "max_num_hands": 2,
        "min_detection_confidence": 0.7,
        "min_tracking_confidence": 0.5,
        "model_complexity": 1,
    },
    "visualization": {
        "app_title": "AI Hand Gesture System (Phase 1: Detection & Tracking)",
        "show_fps": True,
        "show_landmarks": True,
        "show_connections": True,
        "show_hand_status": True,
        "show_event_history": True,
        "event_history_length": 5,
    },
    "logging": {
        "level": "INFO",
        "log_dir": "logs",
        "log_file": "app.log",
        "console_output": True,
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries, overriding base with values in override.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Loads configuration from a YAML file.
    Falls back to default configurations if file does not exist or has missing keys.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        dict: Merged configuration dictionary.
    """
    if not os.path.exists(config_path):
        return DEFAULT_CONFIG.copy()

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
            if not isinstance(loaded, dict):
                return DEFAULT_CONFIG.copy()
            return deep_merge(DEFAULT_CONFIG, loaded)
    except Exception:
        return DEFAULT_CONFIG.copy()

